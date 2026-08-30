"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-30

Creates the five tables the product runs on: routing_decisions, eval_sets,
eval_examples, eval_results and alerts.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "routing_decisions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("api_key_hash", sa.String(length=64), nullable=True),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("chosen_model", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("routing_overhead_ms", sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("routing_reason", sa.Text(), nullable=True),
        sa.Column("fallback_count", sa.Integer(), nullable=False),
        sa.Column("succeeded", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        # Null unless STORE_PROMPTS is enabled. Prompt bodies are not stored by
        # default; see backend/app/db/models.py.
        sa.Column("prompt_preview", sa.Text(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_routing_decisions_api_key_hash", "routing_decisions", ["api_key_hash"])
    op.create_index("ix_routing_decisions_task_type", "routing_decisions", ["task_type"])
    op.create_index("ix_routing_decisions_chosen_model", "routing_decisions", ["chosen_model"])
    op.create_index("ix_routing_decisions_provider", "routing_decisions", ["provider"])
    op.create_index("ix_routing_created_task", "routing_decisions", ["created_at", "task_type"])
    op.create_index("ix_routing_created_model", "routing_decisions", ["created_at", "chosen_model"])

    op.create_table(
        "eval_sets",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("grader", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_eval_set_name_version"),
    )
    op.create_index("ix_eval_sets_task_type", "eval_sets", ["task_type"])

    op.create_table(
        "eval_examples",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("eval_set_id", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("input", sa.Text(), nullable=False),
        sa.Column("expected", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("grader", sa.String(length=32), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["eval_set_id"], ["eval_sets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_examples_eval_set_id", "eval_examples", ["eval_set_id"])

    op.create_table(
        "eval_results",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("eval_set_id", sa.String(length=32), nullable=False),
        sa.Column("eval_version", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=False),
        sa.Column("error_rate", sa.Float(), nullable=False),
        sa.Column("p50_latency_ms", sa.Float(), nullable=False),
        sa.Column("p95_latency_ms", sa.Float(), nullable=False),
        sa.Column("p99_latency_ms", sa.Float(), nullable=False),
        sa.Column("cost_per_query", sa.Float(), nullable=False),
        sa.Column("total_cost_usd", sa.Float(), nullable=False),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["eval_set_id"], ["eval_sets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_results_eval_set_id", "eval_results", ["eval_set_id"])
    op.create_index("ix_eval_results_model_id", "eval_results", ["model_id"])
    op.create_index("ix_eval_results_task_type", "eval_results", ["task_type"])
    op.create_index("ix_eval_results_evaluated_at", "eval_results", ["evaluated_at"])
    op.create_index(
        "ix_eval_results_task_model_time",
        "eval_results",
        ["task_type", "model_id", "evaluated_at"],
    )

    op.create_table(
        "alerts",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=True),
        sa.Column("observed", sa.Float(), nullable=True),
        sa.Column("baseline", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), nullable=False),
        sa.Column("delivered", sa.Boolean(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("dedupe_key", sa.String(length=160), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedupe_key"),
    )
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])
    op.create_index("ix_alerts_kind", "alerts", ["kind"])
    op.create_index("ix_alerts_dedupe_key", "alerts", ["dedupe_key"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("eval_results")
    op.drop_table("eval_examples")
    op.drop_table("eval_sets")
    op.drop_table("routing_decisions")
