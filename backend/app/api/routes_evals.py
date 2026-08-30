"""Eval endpoints: upload a set, run it, read history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import AuthContext, require_api_key
from app.api.deps import UNPROCESSABLE, get_eval_runner, get_router
from app.api.schemas import (
    EvalReportOut,
    EvalRunRequest,
    EvalSetIn,
    ModelReportOut,
)
from app.core.logging import get_logger
from app.core.router import Router
from app.db.crud import (
    eval_history,
    latest_quality_scores,
    list_eval_sets,
    load_eval_set,
    record_eval_report,
    upsert_eval_set,
)
from app.db.session import get_session
from app.eval.dataset import EvalSet
from app.eval.graders import GRADERS
from app.eval.runner import EvalRunner

log = get_logger(__name__)

router = APIRouter(prefix="/evals", tags=["evals"])


@router.get("", summary="List stored eval sets")
async def list_sets(
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    return await list_eval_sets(session)


@router.get("/graders", summary="List available graders")
async def graders(_: AuthContext = Depends(require_api_key)) -> dict[str, str]:
    return {
        name: (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        for name, fn in GRADERS.items()
    }


@router.post("", status_code=status.HTTP_201_CREATED, summary="Upload an eval set")
async def create_set(
    payload: EvalSetIn,
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Store an eval set, versioning it if the examples changed.

    An unknown grader is rejected here rather than at run time, so the mistake
    surfaces on upload instead of after paying for 300 model calls.
    """
    unknown = {e.grader for e in payload.examples if e.grader and e.grader not in GRADERS} | (
        {payload.grader}
        if payload.grader not in GRADERS and payload.grader != "llm_judge"
        else set()
    )
    if unknown:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=f"Unknown grader(s) {sorted(unknown)}. Available: {sorted(GRADERS)}, llm_judge",
        )

    try:
        eval_set = EvalSet.from_records(
            payload.name,
            [e.model_dump() for e in payload.examples],
            grader=payload.grader,
        )
    except ValueError as exc:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc
    eval_set.task_type = payload.task_type
    eval_set.description = payload.description

    row = await upsert_eval_set(session, eval_set)
    return {
        "name": row.name,
        "version": row.version,
        "task_type": row.task_type,
        "grader": row.grader,
        "examples": len(eval_set),
    }


@router.post("/run", response_model=EvalReportOut, summary="Run an eval set across models")
async def run_eval(
    payload: EvalRunRequest,
    _: AuthContext = Depends(require_api_key),
    runner: EvalRunner = Depends(get_eval_runner),
    engine: Router = Depends(get_router),
    session: AsyncSession = Depends(get_session),
) -> EvalReportOut:
    """Evaluate every requested model and store the results.

    Synchronous by design at this size: a 100-example run across three models
    finishes inside a request, and a job queue would add operational surface for
    no benefit until eval sets are much larger.

    Afterwards the router's quality table is refreshed from the database, so the
    run immediately affects routing rather than at the next restart.
    """
    stored = await load_eval_set(session, payload.eval_set)
    if stored is None:
        available = [s["name"] for s in await list_eval_sets(session)]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Eval set {payload.eval_set!r} not found. Available: {available}",
        )

    runner.max_tokens = payload.max_tokens
    try:
        report = await runner.run(
            stored,
            model_ids=payload.model_ids,
            system=payload.system,
            keep_outputs=payload.keep_outputs,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc

    from app.db.crud import get_eval_set

    row = await get_eval_set(session, stored.name)
    if row is not None:
        await record_eval_report(session, row, report)
        engine.set_quality_scores(await latest_quality_scores(session))

    data = report.as_dict()
    baseline = max(report.models.values(), key=lambda m: m.cost_per_query, default=None)
    recommendation = None
    if baseline is not None:
        task = engine.policy.for_task(report.task_type)
        # Recommend against the policy's own bar when it has one, so the advice
        # matches the constraint routing will actually apply.
        bar = task.min_quality if task.min_quality is not None else baseline.accuracy
        recommendation = report.savings_vs(baseline.model_id, bar)

    return EvalReportOut(
        eval_set=data["eval_set"],
        eval_version=data["eval_version"],
        task_type=data["task_type"],
        grader=data["grader"],
        examples=data["examples"],
        started_at=data["started_at"],
        duration_s=data["duration_s"],
        models=[ModelReportOut(**m) for m in data["models"]],
        recommendation=recommendation,
    )


@router.get("/history", summary="Recent eval results")
async def history(
    name: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=50, ge=1, le=500),
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    return await eval_history(session, name=name, limit=limit)


@router.get("/quality", summary="Quality scores currently used for routing")
async def quality(
    _: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The measured accuracy the router multiplies by, with its provenance.

    Labelled as last-measured rather than live, so nobody reads a stale number
    as a current one.
    """
    scores = await latest_quality_scores(session)
    recent = await eval_history(session, limit=1)
    return {
        "scores": scores,
        "source": "most recent eval result per (task_type, model)",
        "measured_at": recent[0]["evaluated_at"] if recent else None,
        "note": (
            "Scores are the last measurement, not a live estimate. "
            "Models with no score route with a penalty and are marked unmeasured."
        ),
    }


__all__ = ["router"]
