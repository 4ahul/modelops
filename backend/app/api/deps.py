"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.core.alerts import AlertEngine
from app.core.config import Settings, get_settings
from app.core.router import Router
from app.eval.runner import EvalRunner
from app.providers.registry import ProviderRegistry

#: 422, spelled as an integer on purpose: Starlette renamed
#: ``HTTP_422_UNPROCESSABLE_ENTITY`` to ``..._UNPROCESSABLE_CONTENT`` and
#: deprecated the old name, so the number is the only spelling that works across
#: both versions without a try/except import.
UNPROCESSABLE = 422


def get_app_settings(request: Request) -> Settings:
    """The settings this application was built with.

    Deliberately *not* ``Depends(get_settings)``. That returns the process-wide
    cached object read from the environment, so an app constructed with explicit
    settings â€” a test, or any embedded deployment â€” would authenticate against a
    different configuration than the one it was given. With an empty key list in
    the environment, that means no authentication at all.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings or get_settings()


def get_router(request: Request) -> Router:
    """The process-wide router.

    Raises 503 rather than 500 when routing is unavailable: no provider key
    configured is an operational state the caller should retry against, not a
    bug in the request.
    """
    router: Router | None = getattr(request.app.state, "router", None)
    if router is None or len(router.registry) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No model providers are configured. Set at least one of "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY."
            ),
        )
    return router


def get_registry(request: Request) -> ProviderRegistry:
    registry: ProviderRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Provider registry unavailable"
        )
    return registry


def get_eval_runner(request: Request) -> EvalRunner:
    runner: EvalRunner | None = getattr(request.app.state, "eval_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation is unavailable because no providers are configured",
        )
    return runner


def get_alert_engine(request: Request) -> AlertEngine:
    engine: AlertEngine | None = getattr(request.app.state, "alert_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Alerts unavailable"
        )
    return engine


__all__ = [
    "UNPROCESSABLE",
    "get_alert_engine",
    "get_app_settings",
    "get_eval_runner",
    "get_registry",
    "get_router",
]
