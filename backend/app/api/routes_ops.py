"""Metrics, alerts, models and health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.api.auth import AuthContext, require_api_key
from app.api.deps import get_alert_engine, get_app_settings, get_registry
from app.api.schemas import AlertOut, HealthResponse
from app.core.alerts import AlertEngine, SlackNotifier
from app.core.config import Settings
from app.core.logging import get_logger
from app.db import ping as db_ping
from app.db.crud import (
    acknowledge_alert,
    cost_summary,
    cost_timeseries,
    latency_percentiles,
    list_alerts,
)
from app.db.session import connection_ok, get_session
from app.providers.pricing import (
    MODEL_CATALOG,
    PRICING_AS_OF,
    assert_fresh,
    pricing_age_days,
)
from app.providers.registry import ProviderRegistry

log = get_logger(__name__)

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and configuration")
async def health(
    request: Request, settings: Settings = Depends(get_app_settings)
) -> HealthResponse:
    """Unauthenticated, because a load balancer cannot hold a key.

    Reports only whether dependencies answer and which models are configured â€”
    never a credential, and never a hostname.
    """
    warnings: list[str] = []
    if (stale := assert_fresh()) is not None:
        warnings.append(stale)

    database = False
    try:
        database = await db_ping()
        if not database and await connection_ok():
            # The server is up but the schema is missing — a deploy that skipped
            # its migration. Distinguished from an unreachable database because
            # the fix is completely different.
            warnings.append(
                "Database is reachable but the schema is missing. Run: alembic upgrade head"
            )
    except RuntimeError:
        warnings.append("database engine not initialised")

    redis_ok = False
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        warnings.append("Redis not configured; health and rate limits are per-process")
    else:
        try:
            await redis.ping()
            redis_ok = True
        except Exception as exc:
            warnings.append(f"Redis unreachable: {type(exc).__name__}")

    registry: ProviderRegistry | None = getattr(request.app.state, "registry", None)
    models = sorted(registry.model_ids) if registry else []
    if not models:
        warnings.append("No provider API keys configured; /complete will fail")

    return HealthResponse(
        status="ok" if database and models and not warnings else "degraded",
        environment=settings.environment,
        version=__version__,
        database=database,
        redis=redis_ok,
        models=models,
        pricing_as_of=PRICING_AS_OF.isoformat(),
        pricing_age_days=pricing_age_days(),
        warnings=warnings,
    )


@router.get("/models", summary="Models available in this deployment")
async def models(
    request: Request,
    _: AuthContext = Depends(require_api_key),
    registry: ProviderRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Configured models with pricing and measured latency.

    The full catalog is included separately so a caller can see what adding a
    vendor key would unlock.
    """
    engine = getattr(request.app.state, "router", None)
    snapshot = await engine.health.snapshot(registry.model_ids) if engine else {}

    return {
        "pricing_as_of": PRICING_AS_OF.isoformat(),
        "available": [
            {
                "model_id": spec.id,
                "provider": spec.provider,
                "model": spec.model,
                "tier": spec.tier,
                "input_per_mtok": spec.pricing.input_per_mtok,
                "output_per_mtok": spec.pricing.output_per_mtok,
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                **snapshot.get(spec.id, {}),
            }
            for spec in sorted(registry.specs, key=lambda s: s.blended_cost_per_mtok)
        ],
        "catalog": sorted(MODEL_CATALOG),
    }


def _metrics_scope(auth: AuthContext, settings: Settings) -> str | None:
    """The ``api_key_hash`` to filter reads by, or ``None`` for everything.

    Scoped to the caller unless the deployment explicitly opts into a shared
    view. An unauthenticated deployment has no key to scope by and is
    single-tenant by definition.
    """
    if settings.metrics_scope == "deployment":
        return None
    return auth.key_hash


@router.get("/metrics", summary="Cost, volume and latency")
async def metrics(
    hours: int = Query(default=24, ge=1, le=24 * 90),
    task_type: str | None = Query(default=None, max_length=64),
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Spend and latency for a window.

    Percentiles are reported alongside the averages because an average latency
    hides the tail that actually causes incidents.
    """
    scope = _metrics_scope(auth, settings)
    summary = await cost_summary(session, hours=hours, task_type=task_type, api_key_hash=scope)
    summary["latency"] = await latency_percentiles(session, hours=hours, api_key_hash=scope)
    summary["scope"] = settings.metrics_scope
    return summary


@router.get("/metrics/timeseries", summary="Cost over time")
async def timeseries(
    hours: int = Query(default=24, ge=1, le=24 * 90),
    bucket_minutes: int = Query(default=60, ge=1, le=1440),
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> list[dict[str, Any]]:
    return await cost_timeseries(
        session,
        hours=hours,
        bucket_minutes=bucket_minutes,
        api_key_hash=_metrics_scope(auth, settings),
    )


@router.get("/alerts", response_model=list[AlertOut], summary="Recent alerts")
async def alerts(
    limit: int = Query(default=50, ge=1, le=500),
    unacknowledged_only: bool = Query(default=False),
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[AlertOut]:
    rows = await list_alerts(session, limit=limit, unacknowledged_only=unacknowledged_only)
    return [
        AlertOut(
            id=row.id,
            created_at=row.created_at.isoformat(),
            kind=row.kind,
            severity=row.severity,
            message=row.message,
            task_type=row.task_type,
            model_id=row.model_id,
            metric=row.metric,
            observed=row.observed,
            baseline=row.baseline,
            acknowledged=row.acknowledged,
        )
        for row in rows
    ]


@router.post("/alerts/check", summary="Run the regression checks now")
async def check_alerts(
    window_hours: int = Query(default=1, ge=1, le=168),
    _: AuthContext = Depends(require_api_key),
    engine: AlertEngine = Depends(get_alert_engine),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Detect regressions and deliver any that fired.

    Exposed as an endpoint so the schedule lives outside the service â€” a cron
    job or a scheduler calls this. Running it on an internal timer would mean
    every replica alerted independently.
    """
    fired = await engine.check_all(session, window_hours=window_hours)

    delivered = 0
    if fired and settings.slack_webhook_url:
        notifier = SlackNotifier(settings.slack_webhook_url)
        for alert in fired:
            if await notifier.send(alert):
                delivered += 1

    return {"fired": len(fired), "delivered": delivered, "alerts": fired}


@router.post("/alerts/{alert_id}/acknowledge", summary="Acknowledge an alert")
async def acknowledge(
    alert_id: str,
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    if not await acknowledge_alert(session, alert_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return {"status": "acknowledged", "id": alert_id}


__all__ = ["router"]
