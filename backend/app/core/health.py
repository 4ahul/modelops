"""Provider health and rolling latency, shared across replicas.

Both live in Redis so every replica routes on the same picture: a provider that
failed for one instance is out of rotation for all of them, and measured latency
is the fleet's, not one process's.

Redis is optional. When it is unavailable the tracker falls back to per-process
state and keeps serving, because a monitoring dependency must never be able to
take down the request path. The degradation is reported at ``/health`` rather
than hidden.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

#: Rolling window of latency samples kept per model.
_WINDOW = 200

_KEY_FAILURES = "modelops:health:failures:{model_id}"
_KEY_DOWN = "modelops:health:down:{model_id}"
_KEY_LATENCY = "modelops:latency:{model_id}"


class HealthTracker:
    """Records provider outcomes and answers "is this model usable right now".

    Args:
        redis: An async Redis client, or ``None`` for in-process only.
        failure_threshold: Consecutive failures before a model is taken out.
        unhealthy_ttl: Seconds a model stays out before it is retried. Short by
            design — a recovered provider should return without an operator.
    """

    def __init__(
        self,
        redis: Any | None = None,
        *,
        failure_threshold: int = 3,
        unhealthy_ttl: int = 60,
    ) -> None:
        self.redis = redis
        self.failure_threshold = failure_threshold
        self.unhealthy_ttl = unhealthy_ttl
        self._local_failures: dict[str, int] = defaultdict(int)
        self._local_down: dict[str, float] = {}
        self._local_latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self.degraded = False

    # ------------------------------------------------------------ recording

    async def record_success(self, model_id: str, latency_ms: float) -> None:
        """Clear the failure count and add a latency sample."""
        self._local_failures[model_id] = 0
        self._local_down.pop(model_id, None)
        self._local_latency[model_id].append(latency_ms)
        if self.redis is None:
            return
        try:
            pipe = self.redis.pipeline()
            pipe.delete(_KEY_FAILURES.format(model_id=model_id))
            pipe.delete(_KEY_DOWN.format(model_id=model_id))
            pipe.lpush(_KEY_LATENCY.format(model_id=model_id), latency_ms)
            pipe.ltrim(_KEY_LATENCY.format(model_id=model_id), 0, _WINDOW - 1)
            # Latency data that stops updating is worse than none: it would let
            # a stale p95 keep excluding a model that is now fast.
            pipe.expire(_KEY_LATENCY.format(model_id=model_id), 3600)
            await pipe.execute()
        except Exception as exc:
            self._degrade("record_success", exc)

    async def record_failure(self, model_id: str) -> None:
        """Count a failure and mark the model down once the threshold is hit."""
        self._local_failures[model_id] += 1
        if self._local_failures[model_id] >= self.failure_threshold:
            self._local_down[model_id] = time.monotonic() + self.unhealthy_ttl

        if self.redis is None:
            return
        try:
            key = _KEY_FAILURES.format(model_id=model_id)
            count = await self.redis.incr(key)
            # The counter itself expires, so a slow trickle of unrelated
            # failures over hours never accumulates into a false outage.
            await self.redis.expire(key, self.unhealthy_ttl * 4)
            if int(count) >= self.failure_threshold:
                await self.redis.setex(_KEY_DOWN.format(model_id=model_id), self.unhealthy_ttl, "1")
                log.warning(
                    "provider_marked_down",
                    model_id=model_id,
                    failures=int(count),
                    ttl_s=self.unhealthy_ttl,
                )
        except Exception as exc:
            self._degrade("record_failure", exc)

    # ------------------------------------------------------------- querying

    async def is_healthy(self, model_id: str) -> bool:
        """Whether the model may be routed to right now."""
        if self.redis is not None:
            try:
                return not await self.redis.exists(_KEY_DOWN.format(model_id=model_id))
            except Exception as exc:
                self._degrade("is_healthy", exc)
        return self._locally_healthy(model_id)

    def _locally_healthy(self, model_id: str) -> bool:
        until = self._local_down.get(model_id)
        return until is None or time.monotonic() >= until

    async def healthy_models(self, model_ids: list[str]) -> list[str]:
        """Filter a list to the models currently usable."""
        status = await self.batch_status(model_ids)
        return [m for m in model_ids if status[m][0]]

    async def batch_status(self, model_ids: list[str]) -> dict[str, tuple[bool, float | None]]:
        """Health and p95 for many models in **one** Redis round trip.

        The router needs both facts for every candidate on every request. Asking
        per model costs two round trips each — fourteen for a seven-model
        deployment, which on a 2ms link is a quarter of the 100ms routing budget
        spent waiting on Redis. One pipeline makes it a single hop.

        Returns ``{model_id: (healthy, p95_latency_ms | None)}``.
        """
        if not model_ids:
            return {}

        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                for model_id in model_ids:
                    pipe.exists(_KEY_DOWN.format(model_id=model_id))
                    pipe.lrange(_KEY_LATENCY.format(model_id=model_id), 0, -1)
                replies = await pipe.execute()
                return {
                    model_id: (
                        not replies[index * 2],
                        self._percentile([float(v) for v in (replies[index * 2 + 1] or [])], 0.95),
                    )
                    for index, model_id in enumerate(model_ids)
                }
            except Exception as exc:
                self._degrade("batch_status", exc)

        return {
            model_id: (
                self._locally_healthy(model_id),
                self._percentile(list(self._local_latency[model_id]), 0.95),
            )
            for model_id in model_ids
        }

    async def p95_latency_ms(self, model_id: str) -> float | None:
        """Measured p95, or ``None`` when there is not enough data.

        ``None`` is meaningful: it tells the router to skip the latency
        constraint rather than assume a number. Defaulting to an optimistic
        value would let an unmeasured model pass a budget it has never met.
        """
        return self._percentile(await self._samples(model_id), 0.95)

    async def mean_latency_ms(self, model_id: str) -> float | None:
        samples = await self._samples(model_id)
        if not samples:
            return None
        return sum(samples) / len(samples)

    @staticmethod
    def _percentile(samples: list[float], quantile: float) -> float | None:
        """Nearest-rank percentile, or ``None`` below five samples.

        One definition, shared by the single-model and batched paths — two
        would eventually disagree, and the router would exclude a model that the
        dashboard shows as fast.

        Fewer than five samples is not a percentile; pretending otherwise would
        exclude models on noise.
        """
        if len(samples) < 5:
            return None
        ordered = sorted(samples)
        index = max(0, min(len(ordered) - 1, int(round(quantile * len(ordered) + 0.5)) - 1))
        return ordered[index]

    async def _samples(self, model_id: str) -> list[float]:
        if self.redis is not None:
            try:
                raw = await self.redis.lrange(_KEY_LATENCY.format(model_id=model_id), 0, -1)
                return [float(v) for v in raw]
            except Exception as exc:
                self._degrade("latency_samples", exc)
        return list(self._local_latency[model_id])

    async def snapshot(self, model_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Per-model health, for ``/health`` and the dashboard.

        Batched, so listing models is one round trip rather than two per model.
        """
        status = await self.batch_status(model_ids)
        counts = await self._sample_counts(model_ids)
        return {
            model_id: {
                "healthy": healthy,
                "samples": counts.get(model_id, 0),
                "p95_latency_ms": round(p95, 1) if p95 is not None else None,
            }
            for model_id, (healthy, p95) in status.items()
        }

    async def _sample_counts(self, model_ids: list[str]) -> dict[str, int]:
        """How many latency samples each model has, in one round trip."""
        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                for model_id in model_ids:
                    pipe.llen(_KEY_LATENCY.format(model_id=model_id))
                replies = await pipe.execute()
                return dict(zip(model_ids, (int(n or 0) for n in replies), strict=False))
            except Exception as exc:
                self._degrade("sample_counts", exc)
        return {m: len(self._local_latency[m]) for m in model_ids}

    def _degrade(self, operation: str, exc: Exception) -> None:
        if not self.degraded:
            log.warning("health_tracker_degraded", operation=operation, error=str(exc))
        self.degraded = True

    def reset(self) -> None:
        """Clear in-process state. For tests."""
        self._local_failures.clear()
        self._local_down.clear()
        self._local_latency.clear()
        self.degraded = False


__all__ = ["HealthTracker"]
