"""The router: pick the cheapest model that still meets the bar, then call it.

Three rules shape everything here.

**Hard constraints exclude, they do not penalise.** A model over the cost ceiling
is removed from consideration. Expressed as a score penalty it could be outvoted
by a high enough quality term, which is how a cost-aware router ends up
expensive.

**Never fail the customer's request to save money.** If the chosen model errors
or times out, the router escalates through the remaining candidates. A cost
optimiser that converts a saving into an availability incident is a bad trade at
any price.

**Quality comes from measurement or it is labelled as absent.** Scores come from
the last eval run for that task type. A model with no measurement is penalised
and marked ``unmeasured`` in the decision record, so nobody reads a guess as a
number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.health import HealthTracker
from app.core.logging import get_logger
from app.core.policy import RoutingPolicy, TaskPolicy
from app.providers.base import (
    CompletionResult,
    ProviderBadRequest,
    ProviderError,
)
from app.providers.registry import ProviderRegistry

log = get_logger(__name__)


class NoEligibleModel(Exception):
    """No model satisfied the policy's hard constraints.

    Carries the per-model reasons, because "no model available" with no
    explanation is the least actionable error a router can produce.
    """

    def __init__(self, task_type: str | None, reasons: dict[str, str]) -> None:
        self.task_type = task_type
        self.reasons = reasons
        detail = "; ".join(f"{model}: {why}" for model, why in reasons.items()) or "(no models)"
        super().__init__(
            f"No model satisfies the policy for task {task_type or 'default'!r}. {detail}"
        )


class AllProvidersFailed(Exception):
    """Every candidate was tried and every one failed."""

    def __init__(self, attempts: dict[str, str]) -> None:
        self.attempts = attempts
        detail = "; ".join(f"{model}: {why}" for model, why in attempts.items())
        super().__init__(f"All candidate models failed. {detail}")


@dataclass
class Candidate:
    """One model's standing for a particular request."""

    model_id: str
    provider: str
    estimated_cost: float
    quality: float | None
    p95_latency_ms: float | None
    score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    quality_score: float = 0.0
    unmeasured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "estimated_cost": round(self.estimated_cost, 8),
            "quality": self.quality,
            "p95_latency_ms": self.p95_latency_ms,
            "score": round(self.score, 4),
            "unmeasured": self.unmeasured,
        }


@dataclass
class RoutingDecision:
    """Why a model was chosen, and what it cost to decide.

    Persisted for every routed call. Once this table has real traffic in it the
    routing algorithm stops being a heuristic and becomes a fit to the actual
    workload — which is the only way the product's claim can be verified rather
    than asserted.
    """

    task_type: str | None
    chosen: str
    candidates: list[Candidate]
    excluded: dict[str, str]
    overhead_ms: float
    fallbacks: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def reason(self) -> str:
        winner = next((c for c in self.candidates if c.model_id == self.chosen), None)
        if winner is None:
            return f"chose {self.chosen}"
        bits = [f"score {winner.score:.3f}", f"est ${winner.estimated_cost:.6f}"]
        if winner.quality is not None:
            bits.append(f"quality {winner.quality:.3f}")
        else:
            bits.append("quality unmeasured")
        if winner.p95_latency_ms is not None:
            bits.append(f"p95 {winner.p95_latency_ms:.0f}ms")
        if self.fallbacks:
            bits.append(f"after {len(self.fallbacks)} failed provider(s)")
        return f"{self.chosen}: " + ", ".join(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "chosen": self.chosen,
            "reason": self.reason,
            "overhead_ms": round(self.overhead_ms, 2),
            "fallbacks": self.fallbacks,
            "candidates": [c.as_dict() for c in self.candidates],
            "excluded": self.excluded,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RoutedCompletion:
    """A completion plus the decision that produced it."""

    result: CompletionResult
    decision: RoutingDecision

    # Flat accessors, so callers do not reach through two objects for the basics.
    @property
    def content(self) -> str:
        return self.result.content

    @property
    def model_id(self) -> str:
        return self.result.model_id

    @property
    def provider(self) -> str:
        return self.result.provider

    @property
    def cost_usd(self) -> float:
        return self.result.cost_usd

    @property
    def latency_ms(self) -> float:
        return self.result.latency_ms

    @property
    def routing_overhead_ms(self) -> float:
        return self.decision.overhead_ms


class Router:
    """Scores models against a policy and executes with fallback.

    Args:
        registry: Models this deployment can call.
        policy: Task constraints and weights.
        health: Shared health and latency state.
        quality_scores: ``{task_type: {model_id: accuracy}}`` from the last eval
            run. Injected rather than fetched so the router stays synchronous and
            testable; the API refreshes it from the database.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        policy: RoutingPolicy | None = None,
        *,
        health: HealthTracker | None = None,
        quality_scores: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or RoutingPolicy()
        self.health = health or HealthTracker()
        self.quality_scores: dict[str, dict[str, float]] = quality_scores or {}

    # -------------------------------------------------------------- quality

    def set_quality_scores(self, scores: dict[str, dict[str, float]]) -> None:
        """Replace the measured-quality table, after an eval run."""
        self.quality_scores = scores

    def quality_for(self, task_type: str | None, model_id: str) -> float | None:
        """Measured accuracy for this model on this task, or ``None``.

        Scores are strictly per task type. A model measured on classification
        says nothing about its reasoning, and reusing the number across tasks is
        the fastest way to a confidently wrong routing decision.
        """
        return self.quality_scores.get(task_type or "default", {}).get(model_id)

    # -------------------------------------------------------------- scoring

    async def rank(
        self,
        prompt: str,
        task_type: str | None = None,
        *,
        expected_output_tokens: int = 512,
    ) -> tuple[list[Candidate], dict[str, str]]:
        """Score every model, returning survivors best-first plus exclusions."""
        task = self.policy.for_task(task_type)
        allowed = self.policy.models or self.registry.model_ids

        candidates: list[Candidate] = []
        excluded: dict[str, str] = {}

        # Split out models this deployment cannot call before touching Redis,
        # so the batch covers exactly the set worth asking about.
        configured = [m for m in allowed if m in self.registry]
        for model_id in allowed:
            if model_id not in self.registry:
                excluded[model_id] = "not configured in this deployment"

        # One round trip for health and measured p95 across every candidate.
        # Per-model lookups would be two each, which on a seven-model
        # deployment is most of the routing latency budget spent on Redis.
        status = await self.health.batch_status(configured)

        for model_id in configured:
            provider = self.registry.get(model_id)
            healthy, p95 = status.get(model_id, (True, None))

            if not healthy:
                excluded[model_id] = "provider marked unhealthy"
                continue

            estimated = provider.estimate_cost(prompt, expected_output_tokens)
            if task.cost_limit is not None and estimated > task.cost_limit:
                excluded[model_id] = (
                    f"estimated ${estimated:.6f} over cost_limit ${task.cost_limit:.6f}"
                )
                continue

            if task.latency_budget_ms is not None and p95 is not None:
                if p95 > task.latency_budget_ms:
                    excluded[model_id] = (
                        f"measured p95 {p95:.0f}ms over budget {task.latency_budget_ms:.0f}ms"
                    )
                    continue

            quality = self.quality_for(task_type, model_id)
            if quality is None and not task.allow_unmeasured:
                excluded[model_id] = (
                    f"no eval result for task {task_type or 'default'!r} and "
                    "allow_unmeasured is False"
                )
                continue
            if quality is not None and task.min_quality is not None:
                if quality < task.min_quality:
                    excluded[model_id] = (
                        f"measured quality {quality:.3f} below min_quality {task.min_quality:.3f}"
                    )
                    continue

            candidates.append(
                Candidate(
                    model_id=model_id,
                    provider=provider.name,
                    estimated_cost=estimated,
                    quality=quality,
                    p95_latency_ms=p95,
                    unmeasured=quality is None,
                )
            )

        self._score(candidates, task)
        candidates.sort(
            key=lambda c: (
                -c.score,
                task.prefer.index(c.model_id) if c.model_id in task.prefer else len(task.prefer),
            )
        )
        return candidates, excluded

    @staticmethod
    def _score(candidates: list[Candidate], task: TaskPolicy) -> None:
        """Assign each candidate a 0–1 score, normalised within this request.

        Normalising against the actual candidate set rather than against absolute
        limits keeps the comparison meaningful: with three models between
        $0.001 and $0.002, dividing by a $0.05 ceiling would make all three look
        identically cheap and hand the decision entirely to quality.
        """
        if not candidates:
            return

        costs = [c.estimated_cost for c in candidates]
        cost_lo, cost_hi = min(costs), max(costs)
        cost_span = cost_hi - cost_lo

        latencies = [c.p95_latency_ms for c in candidates if c.p95_latency_ms is not None]
        lat_lo = min(latencies) if latencies else 0.0
        lat_hi = max(latencies) if latencies else 0.0
        lat_span = lat_hi - lat_lo

        weights = task.weights
        for candidate in candidates:
            # Cheaper is better, so the scale is inverted.
            candidate.cost_score = (
                1.0 if cost_span == 0 else 1 - (candidate.estimated_cost - cost_lo) / cost_span
            )

            if candidate.p95_latency_ms is None or lat_span == 0:
                # An unmeasured latency scores neutral rather than best: it has
                # not earned the top slot and should not be punished for being
                # new either.
                candidate.latency_score = 0.5
            else:
                candidate.latency_score = 1 - (candidate.p95_latency_ms - lat_lo) / lat_span

            if candidate.quality is None:
                candidate.quality_score = max(0.0, 0.5 - task.unmeasured_penalty)
            else:
                candidate.quality_score = candidate.quality

            candidate.score = (
                weights.get("cost", 0.0) * candidate.cost_score
                + weights.get("latency", 0.0) * candidate.latency_score
                + weights.get("quality", 0.0) * candidate.quality_score
            )

    # ------------------------------------------------------------ execution

    async def route(
        self,
        prompt: str,
        task_type: str | None = None,
        *,
        expected_output_tokens: int = 512,
    ) -> tuple[list[Candidate], RoutingDecision]:
        """Decide without calling anything. Used by ``/route`` and by tests."""
        started = time.perf_counter()
        candidates, excluded = await self.rank(
            prompt, task_type, expected_output_tokens=expected_output_tokens
        )
        overhead = (time.perf_counter() - started) * 1000
        if not candidates:
            raise NoEligibleModel(task_type, excluded)
        decision = RoutingDecision(
            task_type=task_type,
            chosen=candidates[0].model_id,
            candidates=candidates,
            excluded=excluded,
            overhead_ms=overhead,
        )
        return candidates, decision

    async def complete(
        self,
        prompt: str,
        task_type: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> RoutedCompletion:
        """Route and execute, escalating on failure.

        Raises:
            NoEligibleModel: the policy excluded everything. The request is not
                attempted, because guessing past a cost ceiling is worse than a
                clear error.
            AllProvidersFailed: every candidate was tried and failed.
            ProviderBadRequest: the request itself is invalid. Not retried
                elsewhere — a prompt over the context window or a content-filter
                rejection fails identically at every vendor, and three attempts
                would mean three bills for one mistake.
        """
        task = self.policy.for_task(task_type)
        candidates, decision = await self.route(prompt, task_type)
        order = self._fallback_order(candidates, task)

        attempts: dict[str, str] = {}
        for index, model_id in enumerate(order):
            provider = self.registry.get(model_id)
            try:
                result = await provider.complete(
                    prompt,
                    max_tokens=max_tokens or task.max_tokens,
                    temperature=temperature,
                    system=system,
                )
            except ProviderBadRequest:
                # The caller's request is wrong, not the provider. Not counted
                # as a provider failure — doing so would mark a healthy vendor
                # down because a customer sent an oversized prompt.
                raise
            except ProviderError as exc:
                attempts[model_id] = str(exc)
                await self.health.record_failure(model_id)
                log.warning(
                    "provider_failed_falling_back",
                    model_id=model_id,
                    task_type=task_type,
                    remaining=len(order) - index - 1,
                    error=str(exc),
                )
                continue

            await self.health.record_success(model_id, result.latency_ms)
            decision.chosen = model_id
            decision.fallbacks = list(attempts)
            log.info(
                "routed",
                model_id=model_id,
                task_type=task_type,
                cost_usd=round(result.cost_usd, 6),
                latency_ms=round(result.latency_ms, 1),
                overhead_ms=round(decision.overhead_ms, 2),
                fallbacks=len(attempts),
            )
            return RoutedCompletion(result=result, decision=decision)

        raise AllProvidersFailed(attempts)

    @staticmethod
    def _fallback_order(candidates: list[Candidate], task: TaskPolicy) -> list[str]:
        """The order to try models in.

        An explicit ``fallback_chain`` is honoured first, then any remaining
        candidates by score. Models in the chain that were excluded by a hard
        constraint stay excluded — a fallback chain is an escalation preference,
        not a way around a cost ceiling.
        """
        eligible = [c.model_id for c in candidates]
        if not task.fallback_chain:
            return eligible
        ordered = [m for m in task.fallback_chain if m in eligible]
        ordered += [m for m in eligible if m not in ordered]
        return ordered


__all__ = [
    "AllProvidersFailed",
    "Candidate",
    "NoEligibleModel",
    "RoutedCompletion",
    "Router",
    "RoutingDecision",
]
