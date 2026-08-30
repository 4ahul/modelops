"""Database queries.

Aggregation lives here rather than in the route handlers, so a metrics endpoint
and the alert engine ask the same question the same way and cannot drift into
reporting two different numbers for the same window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Integer, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.router import RoutingDecision
from app.db.models import (
    AlertRow,
    EvalExampleRow,
    EvalResultRow,
    EvalSetRow,
    RoutingDecisionRow,
)
from app.eval.dataset import EvalSet
from app.eval.runner import EvalReport
from app.providers.base import CompletionResult


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------- routing


async def record_routing_decision(
    session: AsyncSession,
    *,
    decision: RoutingDecision,
    result: CompletionResult | None,
    api_key_hash: str | None = None,
    quality_score: float | None = None,
    error: str | None = None,
    prompt_preview: str | None = None,
) -> RoutingDecisionRow:
    """Persist one routed call, successful or not.

    Failures are recorded too: a table containing only successes cannot answer
    "how often does the cheap model fall over", which is the question that
    decides whether a saving was real.
    """
    row = RoutingDecisionRow(
        api_key_hash=api_key_hash,
        task_type=decision.task_type,
        chosen_model=decision.chosen,
        provider=result.provider if result else "",
        input_tokens=result.input_tokens if result else 0,
        output_tokens=result.output_tokens if result else 0,
        cost_usd=result.cost_usd if result else 0.0,
        latency_ms=result.latency_ms if result else 0.0,
        routing_overhead_ms=decision.overhead_ms,
        quality_score=quality_score,
        routing_reason=decision.reason,
        fallback_count=len(decision.fallbacks),
        succeeded=result is not None,
        error=error,
        prompt_preview=prompt_preview,
        extra={"excluded": decision.excluded} if decision.excluded else None,
    )
    session.add(row)
    await session.commit()
    return row


async def cost_summary(
    session: AsyncSession,
    *,
    hours: int = 24,
    task_type: str | None = None,
    api_key_hash: str | None = None,
) -> dict[str, Any]:
    """Spend, volume and latency for a window, plus a per-model breakdown.

    ``api_key_hash`` scopes the result to one caller. The API passes it unless
    the deployment opts into a shared view, so one tenant cannot read another
    tenant''s spend, model mix or volume.
    """
    since = _utcnow() - timedelta(hours=hours)
    filters = [RoutingDecisionRow.created_at >= since]
    if task_type:
        filters.append(RoutingDecisionRow.task_type == task_type)
    if api_key_hash:
        filters.append(RoutingDecisionRow.api_key_hash == api_key_hash)

    totals = (
        await session.execute(
            select(
                func.count(RoutingDecisionRow.id),
                func.coalesce(func.sum(RoutingDecisionRow.cost_usd), 0.0),
                func.coalesce(func.avg(RoutingDecisionRow.latency_ms), 0.0),
                func.coalesce(func.avg(RoutingDecisionRow.routing_overhead_ms), 0.0),
                func.coalesce(func.sum(RoutingDecisionRow.input_tokens), 0),
                func.coalesce(func.sum(RoutingDecisionRow.output_tokens), 0),
                func.coalesce(func.sum(RoutingDecisionRow.fallback_count), 0),
            ).where(*filters)
        )
    ).one()

    failures = (
        await session.execute(
            select(func.count(RoutingDecisionRow.id)).where(
                *filters, RoutingDecisionRow.succeeded.is_(False)
            )
        )
    ).scalar_one()

    by_model = (
        await session.execute(
            select(
                RoutingDecisionRow.chosen_model,
                RoutingDecisionRow.provider,
                func.count(RoutingDecisionRow.id),
                func.coalesce(func.sum(RoutingDecisionRow.cost_usd), 0.0),
                func.coalesce(func.avg(RoutingDecisionRow.latency_ms), 0.0),
            )
            .where(*filters)
            .group_by(RoutingDecisionRow.chosen_model, RoutingDecisionRow.provider)
            .order_by(func.sum(RoutingDecisionRow.cost_usd).desc())
        )
    ).all()

    count = int(totals[0])
    return {
        "window_hours": hours,
        "task_type": task_type,
        "requests": count,
        "failures": int(failures),
        "failure_rate": round(int(failures) / count, 4) if count else 0.0,
        "total_cost_usd": round(float(totals[1]), 6),
        "avg_cost_usd": round(float(totals[1]) / count, 8) if count else 0.0,
        "avg_latency_ms": round(float(totals[2]), 1),
        "avg_routing_overhead_ms": round(float(totals[3]), 2),
        "input_tokens": int(totals[4]),
        "output_tokens": int(totals[5]),
        "fallbacks": int(totals[6]),
        "by_model": [
            {
                "model_id": model,
                "provider": provider,
                "requests": int(n),
                "cost_usd": round(float(cost), 6),
                "avg_latency_ms": round(float(latency), 1),
            }
            for model, provider, n, cost, latency in by_model
        ],
    }


async def latency_percentiles(
    session: AsyncSession,
    *,
    hours: int = 24,
    model_id: str | None = None,
    api_key_hash: str | None = None,
) -> dict[str, float]:
    """p50/p95/p99 latency over a window.

    Each percentile is fetched by rank — ``ORDER BY latency_ms LIMIT 1 OFFSET k``
    — rather than by pulling the window into Python and sorting it. ``hours``
    allows up to 90 days, so at real traffic the naive version would load
    millions of rows into memory to compute three numbers. This is four small
    indexed queries with flat memory, and it stays exact.

    ``percentile_cont`` would be one query but is Postgres-only, and the test
    suite runs on SQLite; a percentile that is only exercised in production is
    not one to trust.
    """
    since = _utcnow() - timedelta(hours=hours)
    filters = [RoutingDecisionRow.created_at >= since, RoutingDecisionRow.succeeded.is_(True)]
    if model_id:
        filters.append(RoutingDecisionRow.chosen_model == model_id)
    if api_key_hash:
        filters.append(RoutingDecisionRow.api_key_hash == api_key_hash)

    total = int(
        (
            await session.execute(select(func.count(RoutingDecisionRow.id)).where(*filters))
        ).scalar_one()
    )
    if total == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "samples": 0}

    async def at(quantile: float) -> float:
        # Nearest rank, so the answer is always a latency that was observed
        # rather than an interpolation between two that were not.
        offset = max(0, min(total - 1, int(round(quantile * total + 0.5)) - 1))
        value = (
            await session.execute(
                select(RoutingDecisionRow.latency_ms)
                .where(*filters)
                .order_by(RoutingDecisionRow.latency_ms)
                .limit(1)
                .offset(offset)
            )
        ).scalar_one()
        return round(float(value), 1)

    return {
        "p50": await at(0.50),
        "p95": await at(0.95),
        "p99": await at(0.99),
        "samples": total,
    }


def _bucket_expression(dialect: str, since: datetime, width_seconds: int) -> Any:
    """A SQL expression giving the integer bucket index of each row.

    Dialect-specific because there is no portable epoch conversion: Postgres has
    ``EXTRACT(EPOCH ...)`` and SQLite has ``julianday``. Returns ``None`` for any
    other dialect so the caller can fall back rather than emit invalid SQL.
    """
    if dialect == "postgresql":
        return func.floor(
            func.extract("epoch", RoutingDecisionRow.created_at - since) / width_seconds
        )
    if dialect == "sqlite":
        return func.cast(
            (func.julianday(RoutingDecisionRow.created_at) - func.julianday(since))
            * 86400.0
            / width_seconds,
            Integer,
        )
    return None


async def cost_timeseries(
    session: AsyncSession,
    *,
    hours: int = 24,
    bucket_minutes: int = 60,
    api_key_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Cost and volume bucketed over time, for the dashboard chart.

    Aggregated in the database and grouped by a computed bucket index, so the
    result is one row per bucket instead of one per request. Bucketing in Python
    would mean transferring the whole window to draw a chart with 24 points.
    """
    since = _utcnow() - timedelta(hours=hours)
    width = max(1, bucket_minutes) * 60
    filters = [RoutingDecisionRow.created_at >= since]
    if api_key_hash:
        filters.append(RoutingDecisionRow.api_key_hash == api_key_hash)

    dialect = session.get_bind().dialect.name
    bucket = _bucket_expression(dialect, since, width)
    if bucket is None:  # pragma: no cover - only for an unrecognised dialect
        return await _cost_timeseries_in_python(session, filters, since, width)

    rows = (
        await session.execute(
            select(
                bucket.label("bucket"),
                func.coalesce(func.sum(RoutingDecisionRow.cost_usd), 0.0),
                func.count(RoutingDecisionRow.id),
                RoutingDecisionRow.chosen_model,
            )
            .where(*filters)
            .group_by("bucket", RoutingDecisionRow.chosen_model)
            .order_by("bucket")
        )
    ).all()

    buckets: dict[int, dict[str, Any]] = {}
    for index, cost, count, model in rows:
        key = int(index or 0)
        entry = buckets.setdefault(
            key,
            {
                "timestamp": (since + timedelta(seconds=key * width)).isoformat(),
                "cost_usd": 0.0,
                "requests": 0,
                "models": {},
            },
        )
        entry["cost_usd"] += float(cost)
        entry["requests"] += int(count)
        entry["models"][model] = entry["models"].get(model, 0) + int(count)

    for entry in buckets.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return [buckets[k] for k in sorted(buckets)]


async def _cost_timeseries_in_python(
    session: AsyncSession, filters: list[Any], since: datetime, width: int
) -> list[dict[str, Any]]:
    """Fallback bucketing for a dialect with no epoch expression."""
    rows = (
        await session.execute(
            select(
                RoutingDecisionRow.created_at,
                RoutingDecisionRow.cost_usd,
                RoutingDecisionRow.chosen_model,
            )
            .where(*filters)
            .order_by(RoutingDecisionRow.created_at)
        )
    ).all()

    buckets: dict[int, dict[str, Any]] = {}
    for created_at, cost, model in rows:
        # SQLite hands back naive datetimes even from a timezone-aware column,
        # and mixing the two raises on subtraction.
        when = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        key = int((when - since).total_seconds() // width)
        entry = buckets.setdefault(
            key,
            {
                "timestamp": (since + timedelta(seconds=key * width)).isoformat(),
                "cost_usd": 0.0,
                "requests": 0,
                "models": {},
            },
        )
        entry["cost_usd"] += float(cost)
        entry["requests"] += 1
        entry["models"][model] = entry["models"].get(model, 0) + 1

    for entry in buckets.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return [buckets[k] for k in sorted(buckets)]


async def purge_routing_decisions(session: AsyncSession, *, before: datetime) -> int:
    """Delete rows older than ``before``. Returns the count removed."""
    result = await session.execute(
        delete(RoutingDecisionRow).where(RoutingDecisionRow.created_at < before)
    )
    await session.commit()
    return int(getattr(result, "rowcount", 0) or 0)


# --------------------------------------------------------------------- evals


async def upsert_eval_set(session: AsyncSession, eval_set: EvalSet) -> EvalSetRow:
    """Store an eval set, bumping the version when the examples change.

    A changed set gets a new version rather than overwriting the old one, so a
    stored result always refers to the exact data it was measured against.
    """
    existing = (
        await session.execute(
            select(EvalSetRow)
            .where(EvalSetRow.name == eval_set.name)
            .order_by(EvalSetRow.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    version = eval_set.version
    if existing is not None:
        current = (
            await session.execute(
                select(EvalExampleRow.input, EvalExampleRow.expected).where(
                    EvalExampleRow.eval_set_id == existing.id
                )
            )
        ).all()
        incoming = [(e.input, None if e.expected is None else str(e.expected)) for e in eval_set]
        if sorted(current) == sorted(incoming):
            return existing
        version = existing.version + 1

    row = EvalSetRow(
        name=eval_set.name,
        version=version,
        task_type=eval_set.task_type,
        grader=eval_set.grader,
        description=eval_set.description,
    )
    session.add(row)
    await session.flush()
    for example in eval_set:
        session.add(
            EvalExampleRow(
                eval_set_id=row.id,
                external_id=example.id,
                input=example.input,
                expected=None if example.expected is None else str(example.expected),
                tags={"tags": list(example.tags)} if example.tags else None,
                grader=example.grader,
                weight=example.weight,
            )
        )
    await session.commit()
    return row


async def record_eval_report(
    session: AsyncSession, eval_set_row: EvalSetRow, report: EvalReport
) -> list[EvalResultRow]:
    """Store one row per model from an eval run."""
    rows = [
        EvalResultRow(
            eval_set_id=eval_set_row.id,
            eval_version=eval_set_row.version,
            task_type=report.task_type or eval_set_row.task_type,
            model_id=model.model_id,
            provider=model.provider,
            accuracy=model.accuracy,
            pass_rate=model.pass_rate,
            error_rate=model.error_rate,
            p50_latency_ms=model.p50_latency_ms,
            p95_latency_ms=model.p95_latency_ms,
            p99_latency_ms=model.p99_latency_ms,
            cost_per_query=model.cost_per_query,
            total_cost_usd=model.total_cost_usd,
            example_count=len(model.results),
            duration_s=report.duration_s,
        )
        for model in report.models.values()
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def latest_quality_scores(session: AsyncSession) -> dict[str, dict[str, float]]:
    """``{task_type: {model_id: accuracy}}`` from the most recent eval per pair.

    This is what the router multiplies its quality weight by. It is deliberately
    the *last measured* value â€” stale but honest â€” rather than an online
    estimate. The dashboard labels its age so nobody reads it as live.
    """
    rows = (
        await session.execute(
            select(
                EvalResultRow.task_type,
                EvalResultRow.model_id,
                EvalResultRow.accuracy,
                EvalResultRow.evaluated_at,
            ).order_by(EvalResultRow.evaluated_at.desc())
        )
    ).all()

    scores: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    for task_type, model_id, accuracy, _ in rows:
        key = (task_type or "default", model_id)
        if key in seen:
            continue
        seen.add(key)
        scores.setdefault(key[0], {})[model_id] = float(accuracy)
    return scores


async def eval_history(
    session: AsyncSession, *, name: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent eval results, newest first."""
    query = (
        select(EvalResultRow, EvalSetRow.name)
        .join(EvalSetRow, EvalResultRow.eval_set_id == EvalSetRow.id)
        .order_by(EvalResultRow.evaluated_at.desc())
        .limit(limit)
    )
    if name:
        query = query.where(EvalSetRow.name == name)

    return [
        {
            "eval_set": set_name,
            "eval_version": row.eval_version,
            "task_type": row.task_type,
            "model_id": row.model_id,
            "accuracy": round(row.accuracy, 4),
            "pass_rate": round(row.pass_rate, 4),
            "error_rate": round(row.error_rate, 4),
            "p95_latency_ms": round(row.p95_latency_ms, 1),
            "cost_per_query": round(row.cost_per_query, 6),
            "examples": row.example_count,
            "evaluated_at": row.evaluated_at.isoformat(),
        }
        for row, set_name in (await session.execute(query)).all()
    ]


async def get_eval_set(session: AsyncSession, name: str) -> EvalSetRow | None:
    return (
        await session.execute(
            select(EvalSetRow)
            .where(EvalSetRow.name == name)
            .order_by(EvalSetRow.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def load_eval_set(session: AsyncSession, name: str) -> EvalSet | None:
    """Rebuild a domain :class:`EvalSet` from stored rows."""
    row = await get_eval_set(session, name)
    if row is None:
        return None
    examples = (
        await session.execute(select(EvalExampleRow).where(EvalExampleRow.eval_set_id == row.id))
    ).scalars()
    records = [
        {
            "input": e.input,
            "expected": e.expected,
            "tags": (e.tags or {}).get("tags", []),
            "grader": e.grader,
            "weight": e.weight,
            "id": e.external_id or "",
        }
        for e in examples
    ]
    if not records:
        return None
    built = EvalSet.from_records(row.name, records, grader=row.grader)
    built.version = row.version
    built.task_type = row.task_type
    built.description = row.description
    return built


async def list_eval_sets(session: AsyncSession) -> list[dict[str, Any]]:
    rows = (
        await session.execute(
            select(
                EvalSetRow.name,
                EvalSetRow.version,
                EvalSetRow.task_type,
                EvalSetRow.grader,
                EvalSetRow.created_at,
                func.count(EvalExampleRow.id),
            )
            .outerjoin(EvalExampleRow, EvalExampleRow.eval_set_id == EvalSetRow.id)
            .group_by(
                EvalSetRow.id,
                EvalSetRow.name,
                EvalSetRow.version,
                EvalSetRow.task_type,
                EvalSetRow.grader,
                EvalSetRow.created_at,
            )
            .order_by(EvalSetRow.name, EvalSetRow.version.desc())
        )
    ).all()
    return [
        {
            "name": name,
            "version": version,
            "task_type": task_type,
            "grader": grader,
            "examples": int(count),
            "created_at": created_at.isoformat(),
        }
        for name, version, task_type, grader, created_at, count in rows
    ]


# -------------------------------------------------------------------- alerts


async def create_alert(session: AsyncSession, **fields: Any) -> AlertRow | None:
    """Insert an alert unless its ``dedupe_key`` already exists.

    Returns ``None`` when suppressed as a duplicate.
    """
    dedupe_key = fields.get("dedupe_key")
    if dedupe_key:
        existing = (
            await session.execute(select(AlertRow.id).where(AlertRow.dedupe_key == dedupe_key))
        ).scalar_one_or_none()
        if existing:
            return None
    row = AlertRow(**fields)
    session.add(row)
    await session.commit()
    return row


async def list_alerts(
    session: AsyncSession, *, limit: int = 50, unacknowledged_only: bool = False
) -> list[AlertRow]:
    query = select(AlertRow).order_by(AlertRow.created_at.desc()).limit(limit)
    if unacknowledged_only:
        query = query.where(AlertRow.acknowledged.is_(False))
    return list((await session.execute(query)).scalars())


async def acknowledge_alert(session: AsyncSession, alert_id: str) -> bool:
    row = (
        await session.execute(select(AlertRow).where(AlertRow.id == alert_id))
    ).scalar_one_or_none()
    if row is None:
        return False
    row.acknowledged = True
    await session.commit()
    return True


__all__ = [
    "acknowledge_alert",
    "cost_summary",
    "cost_timeseries",
    "create_alert",
    "eval_history",
    "get_eval_set",
    "latency_percentiles",
    "latest_quality_scores",
    "list_alerts",
    "list_eval_sets",
    "load_eval_set",
    "purge_routing_decisions",
    "record_eval_report",
    "record_routing_decision",
    "upsert_eval_set",
]
