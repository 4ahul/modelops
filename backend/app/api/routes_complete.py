"""Routing endpoints: ``/complete`` and ``/route``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import AuthContext, require_api_key
from app.api.deps import UNPROCESSABLE, get_app_settings, get_router
from app.api.schemas import (
    CandidateOut,
    CompletionRequest,
    CompletionResponse,
    RouteRequest,
    RouteResponse,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.router import (
    AllProvidersFailed,
    NoEligibleModel,
    RoutedCompletion,
    Router,
    RoutingDecision,
)
from app.db.crud import record_routing_decision
from app.db.session import get_session
from app.providers.base import ProviderBadRequest, ProviderError

log = get_logger(__name__)

router = APIRouter(tags=["routing"])

#: How much of a prompt is kept when STORE_PROMPTS is deliberately enabled.
_PREVIEW_CHARS = 200


@router.post(
    "/complete",
    response_model=CompletionResponse,
    summary="Route a prompt to the cheapest model meeting the policy, and run it",
)
async def complete(
    payload: CompletionRequest,
    auth: AuthContext = Depends(require_api_key),
    engine: Router = Depends(get_router),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> CompletionResponse:
    """Route and execute a completion.

    A failure is recorded before it is raised, so the decisions table shows the
    real reliability of each model rather than only its successes.
    """
    preview = payload.prompt[:_PREVIEW_CHARS] if settings.store_prompts else None

    if payload.model_id:
        return await _pinned(payload, auth, engine, session, preview)

    try:
        routed: RoutedCompletion = await engine.complete(
            payload.prompt,
            payload.task_type,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            system=payload.system,
        )
    except NoEligibleModel as exc:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail={
                "detail": str(exc),
                "kind": "no_eligible_model",
                # The per-model reasons are the whole value of this error: they
                # say which constraint to relax.
                "context": {"excluded": exc.reasons},
            },
        ) from exc
    except ProviderBadRequest as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": str(exc), "kind": "bad_request"},
        ) from exc
    except AllProvidersFailed as exc:
        await record_routing_decision(
            session,
            decision=RoutingDecision(
                task_type=payload.task_type,
                chosen=next(iter(exc.attempts), ""),
                candidates=[],
                excluded={},
                overhead_ms=0.0,
                fallbacks=list(exc.attempts),
            ),
            result=None,
            api_key_hash=auth.key_hash,
            error=str(exc),
            prompt_preview=preview,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "detail": str(exc),
                "kind": "all_providers_failed",
                "context": {"attempts": exc.attempts},
            },
        ) from exc

    await record_routing_decision(
        session,
        decision=routed.decision,
        result=routed.result,
        api_key_hash=auth.key_hash,
        quality_score=engine.quality_for(payload.task_type, routed.model_id),
        prompt_preview=preview,
    )

    return CompletionResponse(
        content=routed.content,
        model_id=routed.model_id,
        provider=routed.provider,
        input_tokens=routed.result.input_tokens,
        output_tokens=routed.result.output_tokens,
        cost_usd=routed.cost_usd,
        latency_ms=routed.latency_ms,
        routing_overhead_ms=routed.routing_overhead_ms,
        routing_reason=routed.decision.reason,
        fallbacks=routed.decision.fallbacks,
        task_type=payload.task_type,
    )


async def _pinned(
    payload: CompletionRequest,
    auth: AuthContext,
    engine: Router,
    session: AsyncSession,
    preview: str | None,
) -> CompletionResponse:
    """Run a specific model, bypassing routing.

    Still recorded, and still health-tracked: a pinned call is real traffic, and
    excluding it would make the cost dashboard disagree with the invoice.
    """
    model_id = payload.model_id or ""
    if model_id not in engine.registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Model {model_id!r} is not available. "
                f"Available: {sorted(engine.registry.model_ids)}"
            ),
        )
    provider = engine.registry.get(model_id)
    task = engine.policy.for_task(payload.task_type)
    decision = RoutingDecision(
        task_type=payload.task_type,
        chosen=model_id,
        candidates=[],
        excluded={},
        overhead_ms=0.0,
    )
    try:
        result = await provider.complete(
            payload.prompt,
            max_tokens=payload.max_tokens or task.max_tokens,
            temperature=payload.temperature,
            system=payload.system,
        )
    except ProviderBadRequest as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": str(exc), "kind": "bad_request"},
        ) from exc
    except ProviderError as exc:
        await engine.health.record_failure(model_id)
        await record_routing_decision(
            session,
            decision=decision,
            result=None,
            api_key_hash=auth.key_hash,
            error=str(exc),
            prompt_preview=preview,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"detail": str(exc), "kind": "provider_failed"},
        ) from exc

    await engine.health.record_success(model_id, result.latency_ms)
    await record_routing_decision(
        session,
        decision=decision,
        result=result,
        api_key_hash=auth.key_hash,
        quality_score=engine.quality_for(payload.task_type, model_id),
        prompt_preview=preview,
    )
    return CompletionResponse(
        content=result.content,
        model_id=result.model_id,
        provider=result.provider,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        routing_overhead_ms=0.0,
        routing_reason=f"pinned to {model_id} by request",
        task_type=payload.task_type,
    )


@router.post(
    "/route",
    response_model=RouteResponse,
    summary="Show which model would be chosen, and why, without running it",
)
async def route(
    payload: RouteRequest,
    _: AuthContext = Depends(require_api_key),
    engine: Router = Depends(get_router),
) -> RouteResponse:
    """Explain a routing decision.

    Free to call and free of side effects, so a policy can be inspected in
    review rather than discovered in production.
    """
    try:
        candidates, decision = await engine.route(
            payload.prompt,
            payload.task_type,
            expected_output_tokens=payload.expected_output_tokens,
        )
    except NoEligibleModel as exc:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail={
                "detail": str(exc),
                "kind": "no_eligible_model",
                "context": {"excluded": exc.reasons},
            },
        ) from exc

    return RouteResponse(
        chosen=decision.chosen,
        reason=decision.reason,
        overhead_ms=decision.overhead_ms,
        candidates=[CandidateOut(**c.as_dict()) for c in candidates],
        excluded=decision.excluded,
    )


__all__ = ["router"]
