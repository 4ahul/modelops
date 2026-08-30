"""Routing policy: the constraints and weights that decide a model.

A policy is per task type, because a classification call and a drafting call
have nothing in common: one needs 200ms and costs a hundredth of a cent, the
other can take five seconds and is worth fifty times as much.

Hard constraints exclude; weights order what survives. Those are deliberately
different mechanisms — a cost ceiling expressed as a penalty can always be
outvoted by a high enough quality score, which is exactly the bug that makes a
"cost-aware" router quietly expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_TASK = "default"


@dataclass
class TaskPolicy:
    """Constraints and preferences for one class of work.

    Args:
        cost_limit: Maximum estimated dollars per call. A model whose estimate
            exceeds this is excluded, not penalised.
        latency_budget_ms: Maximum acceptable p95 latency, measured. Excludes.
        min_quality: Minimum measured accuracy from the eval set for this task.
            Excludes. A model with no measurement is handled by
            ``allow_unmeasured``.
        allow_unmeasured: Whether a model with no eval result may be chosen.
            Default ``True`` with a penalty, so a new deployment routes at all;
            set ``False`` once eval coverage exists and unmeasured routing
            should be an error rather than a guess.
        weights: Relative importance of cost, latency and quality in ordering
            the survivors.
        prefer: Model ids to try first when their scores tie.
        max_tokens: Output ceiling for calls on this task.
        fallback_chain: Explicit escalation order. When empty, the router
            escalates by score with failed providers removed.
    """

    cost_limit: float | None = None
    latency_budget_ms: float | None = None
    min_quality: float | None = None
    allow_unmeasured: bool = True
    weights: dict[str, float] = field(
        default_factory=lambda: {"cost": 0.4, "latency": 0.2, "quality": 0.4}
    )
    prefer: list[str] = field(default_factory=list)
    max_tokens: int = 1024
    fallback_chain: list[str] = field(default_factory=list)

    #: Score penalty applied to a model with no measured quality. Large enough
    #: that a measured model of similar cost always wins, small enough that an
    #: unmeasured model still beats nothing.
    unmeasured_penalty: float = 0.25

    def __post_init__(self) -> None:
        if self.cost_limit is not None and self.cost_limit <= 0:
            raise ValueError(f"cost_limit must be positive, got {self.cost_limit}")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError(f"latency_budget_ms must be positive, got {self.latency_budget_ms}")
        if self.min_quality is not None and not 0 <= self.min_quality <= 1:
            raise ValueError(f"min_quality must be in [0, 1], got {self.min_quality}")

        unknown = set(self.weights) - {"cost", "latency", "quality"}
        if unknown:
            raise ValueError(f"Unknown weight key(s) {sorted(unknown)}; use cost, latency, quality")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        # Normalised so a score is always comparable across policies, and so
        # {"cost": 4, "quality": 4} means the same as {"cost": .5, "quality": .5}.
        self.weights = {k: v / total for k, v in self.weights.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_limit": self.cost_limit,
            "latency_budget_ms": self.latency_budget_ms,
            "min_quality": self.min_quality,
            "allow_unmeasured": self.allow_unmeasured,
            "weights": self.weights,
            "prefer": self.prefer,
            "max_tokens": self.max_tokens,
            "fallback_chain": self.fallback_chain,
        }


@dataclass
class RoutingPolicy:
    """Task policies, plus the models the router may consider at all.

    Example::

        policy = RoutingPolicy(
            tasks={
                "classification": TaskPolicy(
                    cost_limit=0.001, latency_budget_ms=800, min_quality=0.92
                ),
                "drafting": TaskPolicy(
                    cost_limit=0.05, latency_budget_ms=5000, min_quality=0.85
                ),
            }
        )
    """

    tasks: dict[str, TaskPolicy] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    default: TaskPolicy = field(default_factory=TaskPolicy)

    def for_task(self, task_type: str | None) -> TaskPolicy:
        """The policy for a task, falling back to the default.

        An unknown task type resolves to the default rather than raising: a
        caller passing a new task name should get conservative routing, not a
        failed request.
        """
        if task_type is None:
            return self.default
        return self.tasks.get(task_type, self.tasks.get(DEFAULT_TASK, self.default))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingPolicy:
        tasks = {
            name: TaskPolicy(**config) if isinstance(config, dict) else config
            for name, config in (data.get("tasks") or {}).items()
        }
        default_config = data.get("default")
        return cls(
            tasks=tasks,
            models=list(data.get("models", [])),
            default=TaskPolicy(**default_config) if default_config else TaskPolicy(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": {name: policy.as_dict() for name, policy in self.tasks.items()},
            "models": self.models,
            "default": self.default.as_dict(),
        }


def cost_optimised() -> RoutingPolicy:
    """A starting policy: cheap for bulk work, capable for reasoning."""
    return RoutingPolicy(
        tasks={
            "classification": TaskPolicy(
                cost_limit=0.002,
                latency_budget_ms=1500,
                min_quality=0.90,
                weights={"cost": 0.5, "latency": 0.2, "quality": 0.3},
            ),
            "extraction": TaskPolicy(
                cost_limit=0.01,
                latency_budget_ms=3000,
                min_quality=0.90,
                weights={"cost": 0.4, "latency": 0.15, "quality": 0.45},
            ),
            "summarization": TaskPolicy(cost_limit=0.02, latency_budget_ms=5000, min_quality=0.85),
            "drafting": TaskPolicy(cost_limit=0.05, latency_budget_ms=8000, min_quality=0.85),
            "reasoning": TaskPolicy(
                cost_limit=0.20,
                latency_budget_ms=20_000,
                min_quality=0.95,
                weights={"cost": 0.15, "latency": 0.1, "quality": 0.75},
            ),
        }
    )


__all__ = ["DEFAULT_TASK", "RoutingPolicy", "TaskPolicy", "cost_optimised"]
