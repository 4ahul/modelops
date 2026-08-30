"""Alert detection and delivery.

Three regressions matter, and each is defined against a baseline rather than a
fixed number, because a fixed threshold is either noisy for a small customer or
useless for a large one:

**Cost spike** — spend in this window against the previous one of equal length.
**Latency regression** — p95 now against p95 then. p95, not mean: the mean hides
exactly the tail that causes an incident.
**Accuracy regression** — the newest eval result against the one before it.

Two properties keep the system trustworthy:

- Every alert carries a ``dedupe_key``, so one condition pages once. An alerting
  system that repeats gets muted, and a muted alert is worse than none.
- A window with too little traffic produces no alert. Comparing three requests
  to two is arithmetic, not signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.crud import create_alert
from app.db.models import EvalResultRow, RoutingDecisionRow

log = get_logger(__name__)


@dataclass
class AlertThresholds:
    """When a change counts as a regression.

    Args:
        cost_spike_pct: Relative increase in spend that alerts. 0.5 = +50%.
        latency_regression_pct: Relative p95 increase that alerts.
        accuracy_drop: Absolute accuracy points that alert. 0.05 = 5 points.
        min_requests: Requests needed in both windows before cost or latency is
            compared at all.
        min_examples: Examples needed in both eval runs before accuracy is
            compared.
    """

    cost_spike_pct: float = 0.5
    latency_regression_pct: float = 0.5
    accuracy_drop: float = 0.05
    min_requests: int = 20
    min_examples: int = 10


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _window_key(kind: str, subject: str, when: datetime) -> str:
    """A dedupe key stable within the hour.

    Hourly granularity means a persistent problem re-alerts once an hour —
    frequent enough to stay visible, rare enough not to be noise.
    """
    return f"{kind}:{subject}:{when.strftime('%Y-%m-%dT%H')}"


class AlertEngine:
    """Detects regressions and records them."""

    def __init__(self, thresholds: AlertThresholds | None = None) -> None:
        self.thresholds = thresholds or AlertThresholds()

    async def check_all(
        self, session: AsyncSession, *, window_hours: int = 1
    ) -> list[dict[str, Any]]:
        """Run every check. Returns the alerts that fired."""
        fired: list[dict[str, Any]] = []
        for check in (self.check_cost_spike, self.check_latency_regression):
            fired.extend(await check(session, window_hours=window_hours))
        fired.extend(await self.check_accuracy_regression(session))
        return fired

    # ----------------------------------------------------------------- cost

    async def check_cost_spike(
        self, session: AsyncSession, *, window_hours: int = 1
    ) -> list[dict[str, Any]]:
        now = _utcnow()
        current_start = now - timedelta(hours=window_hours)
        previous_start = current_start - timedelta(hours=window_hours)

        current = await self._spend(session, current_start, now)
        previous = await self._spend(session, previous_start, current_start)

        if (
            current["requests"] < self.thresholds.min_requests
            or previous["requests"] < self.thresholds.min_requests
            or previous["cost"] <= 0
        ):
            return []

        # Per-request cost, not total: a doubling of traffic is not a cost
        # regression, and alerting on it trains people to ignore the alert.
        current_rate = current["cost"] / current["requests"]
        previous_rate = previous["cost"] / previous["requests"]
        increase = (current_rate - previous_rate) / previous_rate
        if increase < self.thresholds.cost_spike_pct:
            return []

        alert = {
            "kind": "cost_spike",
            "severity": "critical" if increase >= self.thresholds.cost_spike_pct * 2 else "warning",
            "message": (
                f"Cost per request up {increase:.0%}: ${current_rate:.6f} vs "
                f"${previous_rate:.6f} over the previous {window_hours}h"
            ),
            "metric": "cost_per_request",
            "observed": round(current_rate, 8),
            "baseline": round(previous_rate, 8),
            "threshold": self.thresholds.cost_spike_pct,
            "details": {
                "window_hours": window_hours,
                "current_requests": current["requests"],
                "previous_requests": previous["requests"],
                "current_total_usd": round(current["cost"], 6),
            },
            "dedupe_key": _window_key("cost_spike", "all", now),
        }
        return await self._emit(session, alert)

    @staticmethod
    async def _spend(
        session: AsyncSession, start: datetime, end: datetime
    ) -> dict[str, float | int]:
        row = (
            await session.execute(
                select(
                    func.count(RoutingDecisionRow.id),
                    func.coalesce(func.sum(RoutingDecisionRow.cost_usd), 0.0),
                ).where(
                    RoutingDecisionRow.created_at >= start,
                    RoutingDecisionRow.created_at < end,
                )
            )
        ).one()
        return {"requests": int(row[0]), "cost": float(row[1])}

    # -------------------------------------------------------------- latency

    async def check_latency_regression(
        self, session: AsyncSession, *, window_hours: int = 1
    ) -> list[dict[str, Any]]:
        now = _utcnow()
        current_start = now - timedelta(hours=window_hours)
        previous_start = current_start - timedelta(hours=window_hours)

        fired: list[dict[str, Any]] = []
        models = (
            await session.execute(
                select(RoutingDecisionRow.chosen_model)
                .where(RoutingDecisionRow.created_at >= previous_start)
                .group_by(RoutingDecisionRow.chosen_model)
            )
        ).scalars()

        for model_id in models:
            current = await self._p95(session, model_id, current_start, now)
            previous = await self._p95(session, model_id, previous_start, current_start)
            if current is None or previous is None or previous <= 0:
                continue
            increase = (current - previous) / previous
            if increase < self.thresholds.latency_regression_pct:
                continue
            fired.extend(
                await self._emit(
                    session,
                    {
                        "kind": "latency_regression",
                        "severity": "warning",
                        "model_id": model_id,
                        "message": (
                            f"{model_id} p95 latency up {increase:.0%}: "
                            f"{current:.0f}ms vs {previous:.0f}ms"
                        ),
                        "metric": "p95_latency_ms",
                        "observed": round(current, 1),
                        "baseline": round(previous, 1),
                        "threshold": self.thresholds.latency_regression_pct,
                        "details": {"window_hours": window_hours},
                        "dedupe_key": _window_key("latency_regression", model_id, now),
                    },
                )
            )
        return fired

    async def _p95(
        self, session: AsyncSession, model_id: str, start: datetime, end: datetime
    ) -> float | None:
        values = list(
            (
                await session.execute(
                    select(RoutingDecisionRow.latency_ms)
                    .where(
                        RoutingDecisionRow.chosen_model == model_id,
                        RoutingDecisionRow.created_at >= start,
                        RoutingDecisionRow.created_at < end,
                        RoutingDecisionRow.succeeded.is_(True),
                    )
                    .order_by(RoutingDecisionRow.latency_ms)
                )
            ).scalars()
        )
        if len(values) < self.thresholds.min_requests:
            return None
        index = max(0, min(len(values) - 1, int(round(0.95 * len(values) + 0.5)) - 1))
        return values[index]

    # ------------------------------------------------------------- accuracy

    async def check_accuracy_regression(self, session: AsyncSession) -> list[dict[str, Any]]:
        """Compare the two most recent eval results per (task, model)."""
        rows = (
            await session.execute(
                select(
                    EvalResultRow.task_type,
                    EvalResultRow.model_id,
                    EvalResultRow.accuracy,
                    EvalResultRow.example_count,
                    EvalResultRow.evaluated_at,
                    EvalResultRow.eval_version,
                ).order_by(EvalResultRow.evaluated_at.desc())
            )
        ).all()

        history: dict[tuple[str, str], list[Any]] = {}
        for task_type, model_id, accuracy, count, when, version in rows:
            history.setdefault((task_type or "default", model_id), []).append(
                (accuracy, count, when, version)
            )

        fired: list[dict[str, Any]] = []
        for (task_type, model_id), entries in history.items():
            if len(entries) < 2:
                continue
            new_acc, new_count, when, new_version = entries[0]
            old_acc, old_count, _, old_version = entries[1]
            if new_count < self.thresholds.min_examples or old_count < self.thresholds.min_examples:
                continue
            drop = old_acc - new_acc
            if drop < self.thresholds.accuracy_drop:
                continue

            # A version bump means the test data changed, so the two numbers do
            # not measure the same thing. Flagged as informational rather than
            # reported as a regression that may not exist.
            data_changed = new_version != old_version
            fired.extend(
                await self._emit(
                    session,
                    {
                        "kind": "accuracy_regression",
                        "severity": "info" if data_changed else "critical",
                        "task_type": task_type,
                        "model_id": model_id,
                        "message": (
                            f"{model_id} accuracy on {task_type} fell "
                            f"{drop:.1%} ({old_acc:.1%} to {new_acc:.1%})"
                            + (
                                " — but the eval set changed version, so this may not be "
                                "a like-for-like comparison"
                                if data_changed
                                else ""
                            )
                        ),
                        "metric": "accuracy",
                        "observed": round(new_acc, 4),
                        "baseline": round(old_acc, 4),
                        "threshold": self.thresholds.accuracy_drop,
                        "details": {
                            "eval_version_now": new_version,
                            "eval_version_before": old_version,
                            "examples": new_count,
                        },
                        "dedupe_key": _window_key(
                            "accuracy_regression", f"{task_type}:{model_id}:{new_version}", when
                        ),
                    },
                )
            )
        return fired

    # -------------------------------------------------------------- writing

    @staticmethod
    async def _emit(session: AsyncSession, alert: dict[str, Any]) -> list[dict[str, Any]]:
        row = await create_alert(session, **alert)
        if row is None:
            return []  # suppressed as a duplicate
        log.warning("alert_fired", kind=alert["kind"], message=alert["message"])
        return [{**alert, "id": row.id}]


class SlackNotifier:
    """Posts alerts to a Slack webhook.

    Delivery failures are logged and swallowed: an unreachable webhook must not
    fail the request that triggered the check, and the alert is already durable
    in the database either way.
    """

    _COLOURS = {"critical": "#d32f2f", "warning": "#f9a825", "info": "#0288d1"}

    def __init__(self, webhook_url: str, *, timeout: float = 5.0) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send(self, alert: dict[str, Any]) -> bool:
        payload = {
            "attachments": [
                {
                    "color": self._COLOURS.get(alert.get("severity", "warning"), "#757575"),
                    "title": f"ModelOps: {alert['kind'].replace('_', ' ')}",
                    "text": alert["message"],
                    "fields": [
                        {"title": key, "value": str(alert[key]), "short": True}
                        for key in ("model_id", "task_type", "observed", "baseline")
                        if alert.get(key) is not None
                    ],
                }
            ]
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
            return True
        except Exception as exc:
            log.warning("slack_delivery_failed", error=str(exc), kind=alert["kind"])
            return False


__all__ = ["AlertEngine", "AlertThresholds", "SlackNotifier"]
