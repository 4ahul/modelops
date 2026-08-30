"""FastAPI application factory.

Startup does four things, in order, and refuses to continue past the first that
proves the deployment is misconfigured:

1. Configure logging, so every later failure is visible in the right format.
2. Validate configuration for the environment. In production, an empty API-key
   list is a hard failure rather than a warning — a service that silently serves
   unauthenticated traffic is worse than one that does not start.
3. Connect to Postgres and Redis. Redis is optional and degrades; Postgres is not.
4. Build the provider registry, router and eval runner, and load the measured
   quality scores so routing is correct on the first request rather than after
   the first eval.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.auth import RateLimiter
from app.api.routes_complete import router as complete_router
from app.api.routes_evals import router as evals_router
from app.api.routes_ops import router as ops_router
from app.core.alerts import AlertEngine
from app.core.config import Settings, get_settings
from app.core.health import HealthTracker
from app.core.logging import configure_logging, get_logger
from app.core.policy import cost_optimised
from app.core.router import Router
from app.db.session import dispose_engine, get_sessionmaker, init_engine
from app.eval.runner import EvalRunner
from app.providers.pricing import assert_fresh
from app.providers.registry import ProviderRegistry

log = get_logger(__name__)


async def _connect_redis(url: str, *, connect_timeout_s: float = 2.0) -> Any | None:
    """Connect to Redis, or return ``None``.

    A Redis outage degrades health tracking and rate limiting to per-process
    state; it does not stop the service. A monitoring dependency that can take
    down the request path is a worse design than an approximate rate limit.

    The timeout is short and explicit: the default would let an unreachable
    Redis stall startup, which delays every replica in a rolling deploy for no
    benefit — the answer after waiting is the same as the answer now.
    """
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=connect_timeout_s,
            socket_timeout=connect_timeout_s,
        )
        async with asyncio.timeout(connect_timeout_s):
            await client.ping()
        log.info("redis_connected")
        return client
    except Exception as exc:
        log.warning(
            "redis_unavailable",
            error=str(exc),
            effect="health and rate limits fall back to per-process state",
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, settings.log_format)

    problems = settings.validate_for_production()
    if problems:
        for problem in problems:
            log.error("unsafe_configuration", problem=problem)
        raise RuntimeError(
            "Refusing to start in production with an unsafe configuration:\n  - "
            + "\n  - ".join(problems)
        )

    if (stale := assert_fresh()) is not None:
        log.warning("pricing_table_stale", detail=stale)

    init_engine(settings.database_url, echo=False)
    app.state.redis = await _connect_redis(settings.redis_url)
    app.state.rate_limiter = RateLimiter(app.state.redis, per_minute=settings.rate_limit_per_minute)

    health = HealthTracker(
        app.state.redis,
        failure_threshold=settings.provider_failure_threshold,
        unhealthy_ttl=settings.provider_unhealthy_ttl_seconds,
    )
    registry = ProviderRegistry.from_settings(settings)
    app.state.registry = registry
    app.state.router = Router(registry, cost_optimised(), health=health)
    app.state.eval_runner = EvalRunner(registry, concurrency=settings.eval_concurrency)
    app.state.alert_engine = AlertEngine()

    # Load measured quality so the first request routes on evidence rather than
    # on the unmeasured penalty.
    try:
        from app.db.crud import latest_quality_scores

        async with get_sessionmaker()() as session:
            app.state.router.set_quality_scores(await latest_quality_scores(session))
    except Exception as exc:
        log.warning(
            "quality_scores_unavailable",
            error=str(exc),
            effect="models route as unmeasured until an eval runs",
        )

    log.info(
        "startup_complete",
        environment=settings.environment,
        models=sorted(registry.model_ids),
        store_prompts=settings.store_prompts,
    )
    try:
        yield
    finally:
        if app.state.redis is not None:
            await app.state.redis.aclose()
        await dispose_engine()
        log.info("shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests pass their own settings."""
    resolved = settings or get_settings()

    app = FastAPI(
        title="ModelOps",
        version=__version__,
        description=(
            "Multi-model routing and evaluation. Route each query to the cheapest "
            "model that still meets your quality bar, and catch regressions before "
            "your users do."
        ),
        lifespan=lifespan,
        # Interactive docs are useful in development and an information leak in
        # production, where the OpenAPI schema enumerates every endpoint.
        docs_url=None if resolved.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if resolved.is_production else "/openapi.json",
    )
    app.state.settings = resolved

    origins = resolved.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    app.include_router(ops_router)
    app.include_router(complete_router)
    app.include_router(evals_router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return a generic message.

        A stack trace or exception string in a response body leaks internals —
        table names, file paths, occasionally a connection string.
        """
        log.exception("unhandled_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "kind": "internal"},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "modelops", "version": __version__, "health": "/health"}

    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
