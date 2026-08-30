"""SQLAlchemy models.

Five tables carry the product. ``routing_decisions`` is the asset: once it holds
real traffic, the routing algorithm stops being a heuristic and becomes a fit to
the customer's actual workload.

One rule runs through the schema: **prompt bodies are not stored by default.**
Token counts, cost and latency are. Customers send production data through this
service, and logging it by default makes the product unadoptable at exactly the
companies that would pay for it. ``prompt_preview`` exists, is nullable, and is
only ever populated when ``STORE_PROMPTS=true`` is set deliberately.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base with a JSON type that works on Postgres and SQLite.

    The test suite runs on SQLite for speed; production is Postgres. Using
    portable types means the tests exercise the same models rather than a
    parallel schema that can drift.
    """

    type_annotation_map = {dict[str, Any]: JSON}


class RoutingDecisionRow(Base):
    """One routed call: what was chosen, what it cost, how long it took."""

    __tablename__ = "routing_decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    api_key_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    task_type: Mapped[str | None] = mapped_column(String(64), index=True)
    chosen_model: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    #: Time spent deciding, excluding the provider call. Published because a
    #: router that saves 30% and adds 200ms is a bad trade for interactive work,
    #: and the only honest way to make that judgement is to measure it.
    routing_overhead_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    #: Measured accuracy for the chosen model on this task at decision time.
    #: Null means the decision was made without a measurement.
    quality_score: Mapped[float | None] = mapped_column(Float)
    routing_reason: Mapped[str | None] = mapped_column(Text)

    #: How many providers failed before this one succeeded. Non-zero rows are
    #: the availability story the cost story has to be weighed against.
    fallback_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    #: Null unless STORE_PROMPTS is enabled. See the module docstring.
    prompt_preview: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    __table_args__ = (
        # The dashboard's main query is "cost over time for this task type",
        # and the alert engine's is "this window versus the one before it".
        Index("ix_routing_created_task", "created_at", "task_type"),
        Index("ix_routing_created_model", "created_at", "chosen_model"),
    )

    def __repr__(self) -> str:
        return f"<RoutingDecision {self.chosen_model} ${self.cost_usd:.6f}>"


class EvalSetRow(Base):
    """A versioned test set."""

    __tablename__ = "eval_sets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64), index=True)
    grader: Mapped[str] = mapped_column(String(32), default="exact_match", nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    examples: Mapped[list[EvalExampleRow]] = relationship(
        back_populates="eval_set", cascade="all, delete-orphan"
    )
    results: Mapped[list[EvalResultRow]] = relationship(
        back_populates="eval_set", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # A set and a version together identify the data a result was measured
        # against. Without this, an edited set silently invalidates every stored
        # result while the comparison keeps rendering.
        UniqueConstraint("name", "version", name="uq_eval_set_name_version"),
    )

    def __repr__(self) -> str:
        return f"<EvalSet {self.name} v{self.version}>"


class EvalExampleRow(Base):
    """One graded case inside an eval set."""

    __tablename__ = "eval_examples"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    eval_set_id: Mapped[str] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str | None] = mapped_column(String(128))
    input: Mapped[str] = mapped_column(Text, nullable=False)
    expected: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    grader: Mapped[str | None] = mapped_column(String(32))
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    eval_set: Mapped[EvalSetRow] = relationship(back_populates="examples")


class EvalResultRow(Base):
    """One model's performance on one eval set version."""

    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    eval_set_id: Mapped[str] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    eval_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)

    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p50_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p99_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_per_query: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    example_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    eval_set: Mapped[EvalSetRow] = relationship(back_populates="results")

    __table_args__ = (
        # The router reads "latest accuracy per (task_type, model)" on every
        # refresh; this is that query's index.
        Index("ix_eval_results_task_model_time", "task_type", "model_id", "evaluated_at"),
    )

    def __repr__(self) -> str:
        return f"<EvalResult {self.model_id} acc={self.accuracy:.3f}>"


class AlertRow(Base):
    """A cost spike, latency regression or accuracy regression."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="warning", nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64))
    model_id: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text, nullable=False)

    metric: Mapped[str | None] = mapped_column(String(64))
    observed: Mapped[float | None] = mapped_column(Float)
    baseline: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)

    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    #: Set to a stable string per (kind, subject, window) so the same condition
    #: cannot page twice. An alerting system that repeats itself gets muted, and
    #: a muted alert is worse than none.
    dedupe_key: Mapped[str | None] = mapped_column(String(160), index=True, unique=True)

    def __repr__(self) -> str:
        return f"<Alert {self.kind} {self.severity}>"


__all__ = [
    "AlertRow",
    "Base",
    "EvalExampleRow",
    "EvalResultRow",
    "EvalSetRow",
    "RoutingDecisionRow",
]
