"""Request and response schemas.

Pydantic models rather than free dicts, so a malformed request is rejected at the
boundary with a field-level message instead of failing somewhere inside the
router.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompletionRequest(BaseModel):
    """A routed completion request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1_000_000)
    task_type: str | None = Field(default=None, max_length=64)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system: str | None = Field(default=None, max_length=100_000)
    #: Pin a model, skipping routing. For A/B comparisons and for reproducing a
    #: stored decision.
    model_id: str | None = Field(default=None, max_length=64)

    @field_validator("prompt", "system")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be only whitespace")
        return value


class CompletionResponse(BaseModel):
    """The completion, plus what it cost and why this model ran it."""

    content: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    routing_overhead_ms: float
    routing_reason: str
    fallbacks: list[str] = Field(default_factory=list)
    task_type: str | None = None


class RouteRequest(BaseModel):
    """Ask which model would be chosen, without paying for a completion."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1_000_000)
    task_type: str | None = Field(default=None, max_length=64)
    expected_output_tokens: int = Field(default=512, ge=1, le=200_000)


class CandidateOut(BaseModel):
    model_id: str
    provider: str
    estimated_cost: float
    quality: float | None
    p95_latency_ms: float | None
    score: float
    unmeasured: bool


class RouteResponse(BaseModel):
    chosen: str
    reason: str
    overhead_ms: float
    candidates: list[CandidateOut]
    excluded: dict[str, str]


class EvalExampleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    expected: Any = None
    tags: list[str] = Field(default_factory=list)
    grader: str | None = None
    weight: float = Field(default=1.0, gt=0)
    id: str = ""


class EvalSetIn(BaseModel):
    """Upload an eval set."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    examples: list[EvalExampleIn] = Field(min_length=1)
    grader: str = "exact_match"
    task_type: str | None = Field(default=None, max_length=64)
    description: str | None = None


class EvalRunRequest(BaseModel):
    """Run a stored eval set against a set of models."""

    model_config = ConfigDict(extra="forbid")

    eval_set: str = Field(min_length=1, max_length=128)
    model_ids: list[str] | None = None
    max_tokens: int = Field(default=1024, ge=1, le=200_000)
    system: str | None = None
    #: Off by default. An eval report is stored and shared, and model output can
    #: carry the customer's data.
    keep_outputs: bool = False


class ModelReportOut(BaseModel):
    model_id: str
    provider: str
    examples: int
    errors: int
    accuracy: float
    pass_rate: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cost_per_query: float
    total_cost_usd: float
    cost_per_correct_answer: float | None


class EvalReportOut(BaseModel):
    eval_set: str
    eval_version: int
    task_type: str | None
    grader: str
    examples: int
    started_at: str
    duration_s: float
    models: list[ModelReportOut]
    recommendation: dict[str, Any] | None = None


class AlertOut(BaseModel):
    id: str
    created_at: str
    kind: str
    severity: str
    message: str
    task_type: str | None = None
    model_id: str | None = None
    metric: str | None = None
    observed: float | None = None
    baseline: float | None = None
    acknowledged: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    environment: str
    version: str
    database: bool
    redis: bool
    models: list[str]
    pricing_as_of: str
    pricing_age_days: int
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    kind: str | None = None
    context: dict[str, Any] | None = None


__all__ = [
    "AlertOut",
    "CandidateOut",
    "CompletionRequest",
    "CompletionResponse",
    "ErrorResponse",
    "EvalExampleIn",
    "EvalReportOut",
    "EvalRunRequest",
    "EvalSetIn",
    "HealthResponse",
    "ModelReportOut",
    "RouteRequest",
    "RouteResponse",
]
