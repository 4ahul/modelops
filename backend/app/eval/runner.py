"""Parallel eval runner and report.

Target from the roadmap: 100 examples across 3 providers in under 5 minutes.
That is 300 calls, so the runner is concurrent — but bounded, because every
vendor rate-limits and an unbounded ``gather`` converts a 300-call run into a
wall of 429s and a meaningless result.

Two properties matter for the numbers to be usable:

**Percentiles, not averages.** A mean latency hides the tail that causes
incidents. The report carries p50/p95/p99.

**A failed call is not a zero-quality answer.** If a provider errors, that is an
availability fact, recorded separately. Averaging it into accuracy would let an
outage look like a quality regression and vice versa.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.eval.dataset import EvalExample, EvalSet
from app.eval.graders import GradeResult, get_grader, make_llm_judge
from app.providers.base import CompletionResult, ModelProvider, ProviderError
from app.providers.registry import ProviderRegistry

log = get_logger(__name__)


@dataclass
class ExampleResult:
    """The outcome of one example against one model."""

    example_id: str
    model_id: str
    score: float
    passed: bool
    latency_ms: float
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    explanation: str = ""
    error: str | None = None
    output: str | None = None

    @property
    def errored(self) -> bool:
        return self.error is not None


@dataclass
class ModelReport:
    """Aggregate performance of one model on one eval set."""

    model_id: str
    provider: str
    results: list[ExampleResult] = field(default_factory=list)

    # ---------------------------------------------------------- partitions

    @property
    def graded(self) -> list[ExampleResult]:
        """Results that produced an answer to grade."""
        return [r for r in self.results if not r.errored]

    @property
    def errors(self) -> list[ExampleResult]:
        return [r for r in self.results if r.errored]

    # ------------------------------------------------------------ quality

    @property
    def accuracy(self) -> float:
        """Mean score over answered examples.

        Errors are excluded deliberately: mixing them in would conflate "the
        model was wrong" with "the provider was down", and those two facts lead
        to opposite actions.
        """
        graded = self.graded
        if not graded:
            return 0.0
        return sum(r.score for r in graded) / len(graded)

    @property
    def pass_rate(self) -> float:
        graded = self.graded
        if not graded:
            return 0.0
        return sum(1 for r in graded if r.passed) / len(graded)

    @property
    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.errors) / len(self.results)

    # ------------------------------------------------------------ latency

    def _latencies(self) -> list[float]:
        return sorted(r.latency_ms for r in self.graded)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    @property
    def mean_latency_ms(self) -> float:
        values = self._latencies()
        return statistics.fmean(values) if values else 0.0

    def _percentile(self, pct: float) -> float:
        """Nearest-rank percentile.

        Interpolation is avoided so a reported p95 is always a latency that was
        actually observed, not an average of two that were not.
        """
        values = self._latencies()
        if not values:
            return 0.0
        index = max(0, min(len(values) - 1, int(round(pct / 100 * len(values) + 0.5)) - 1))
        return values[index]

    # --------------------------------------------------------------- cost

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def cost_per_query(self) -> float:
        graded = self.graded
        if not graded:
            return 0.0
        return sum(r.cost_usd for r in graded) / len(graded)

    @property
    def cost_per_correct_answer(self) -> float:
        """Cost divided by passes.

        The number that actually decides a downgrade: a model at half the price
        that gets a third fewer answers right is more expensive per correct
        answer, not less.
        """
        passes = sum(1 for r in self.graded if r.passed)
        if passes == 0:
            return float("inf")
        return self.total_cost_usd / passes

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "examples": len(self.results),
            "errors": len(self.errors),
            "accuracy": round(self.accuracy, 4),
            "pass_rate": round(self.pass_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "cost_per_query": round(self.cost_per_query, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_correct_answer": (
                None
                if self.cost_per_correct_answer == float("inf")
                else round(self.cost_per_correct_answer, 6)
            ),
        }

    def failures(self, limit: int = 10) -> list[ExampleResult]:
        """Lowest-scoring answered examples, for inspecting a regression."""
        return sorted(self.graded, key=lambda r: r.score)[:limit]


@dataclass
class EvalReport:
    """One eval set's results across every model tested."""

    eval_set: str
    eval_version: int
    models: dict[str, ModelReport] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_s: float = 0.0
    task_type: str | None = None
    grader: str = "exact_match"

    @property
    def example_count(self) -> int:
        return max((len(m.results) for m in self.models.values()), default=0)

    def best_by_accuracy(self) -> ModelReport | None:
        return max(self.models.values(), key=lambda m: m.accuracy, default=None)

    def cheapest_above(self, min_accuracy: float) -> ModelReport | None:
        """The least expensive model that clears an accuracy bar.

        This is the product's actual output: not "which model is best", but
        "which is the cheapest one I can defend switching to".
        """
        eligible = [m for m in self.models.values() if m.accuracy >= min_accuracy]
        if not eligible:
            return None
        return min(eligible, key=lambda m: m.cost_per_query)

    def savings_vs(self, baseline_model_id: str, min_accuracy: float) -> dict[str, Any] | None:
        """What switching from ``baseline_model_id`` would save, and at what cost
        in quality."""
        baseline = self.models.get(baseline_model_id)
        candidate = self.cheapest_above(min_accuracy)
        if baseline is None or candidate is None or candidate.model_id == baseline_model_id:
            return None
        if baseline.cost_per_query == 0:
            return None
        saving = 1 - candidate.cost_per_query / baseline.cost_per_query
        return {
            "from": baseline_model_id,
            "to": candidate.model_id,
            "cost_reduction_pct": round(saving * 100, 1),
            "accuracy_delta": round(candidate.accuracy - baseline.accuracy, 4),
            "p95_latency_delta_ms": round(candidate.p95_latency_ms - baseline.p95_latency_ms, 1),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_set": self.eval_set,
            "eval_version": self.eval_version,
            "task_type": self.task_type,
            "grader": self.grader,
            "examples": self.example_count,
            "started_at": self.started_at.isoformat(),
            "duration_s": round(self.duration_s, 2),
            "models": [m.as_dict() for m in self.models.values()],
        }

    def table(self) -> str:
        """A fixed-width comparison table.

        Ordered cheapest-first, because the decision this table exists to
        support is "how far down can I go".
        """
        header = (
            f"{'MODEL':<16} {'ACC':>6} {'PASS':>6} {'ERR':>6} "
            f"{'P50 ms':>9} {'P95 ms':>9} {'$/QUERY':>10} {'$/CORRECT':>11}"
        )
        lines = [
            f"{self.eval_set} v{self.eval_version} — {self.example_count} examples, "
            f"grader={self.grader}, {self.duration_s:.1f}s",
            header,
            "-" * len(header),
        ]
        for report in sorted(self.models.values(), key=lambda m: m.cost_per_query):
            per_correct = report.cost_per_correct_answer
            lines.append(
                f"{report.model_id:<16} "
                f"{report.accuracy:>6.1%} {report.pass_rate:>6.1%} {report.error_rate:>6.1%} "
                f"{report.p50_latency_ms:>9.0f} {report.p95_latency_ms:>9.0f} "
                f"{report.cost_per_query:>10.6f} "
                f"{'—' if per_correct == float('inf') else format(per_correct, '>11.6f')}"
            )
        return "\n".join(lines)

    def print_table(self) -> None:
        print(self.table())


class EvalRunner:
    """Runs an eval set across models, concurrently and bounded.

    Args:
        registry: The models available to test.
        concurrency: Maximum simultaneous provider calls, across all models.
        max_tokens: Output ceiling per call.
        judge_model_id: Model used when an example's grader is ``llm_judge``.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        concurrency: int = 8,
        max_tokens: int = 1024,
        judge_model_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.concurrency = max(1, concurrency)
        self.max_tokens = max_tokens
        self.judge_model_id = judge_model_id

    async def run(
        self,
        eval_set: EvalSet,
        *,
        model_ids: list[str] | None = None,
        system: str | None = None,
        keep_outputs: bool = False,
    ) -> EvalReport:
        """Evaluate every model on every example.

        ``keep_outputs`` is off by default: an eval report is stored and shared,
        and model output can carry the customer's data.
        """
        targets = model_ids or self.registry.model_ids
        if not targets:
            raise ValueError("No models to evaluate — the registry is empty")
        missing = [m for m in targets if m not in self.registry]
        if missing:
            raise KeyError(
                f"Models not available in this deployment: {missing}. "
                f"Available: {self.registry.model_ids}"
            )

        report = EvalReport(
            eval_set=eval_set.name,
            eval_version=eval_set.version,
            task_type=eval_set.task_type,
            grader=eval_set.grader,
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        started = time.perf_counter()

        async def one(model_id: str, example: EvalExample) -> tuple[str, ExampleResult]:
            async with semaphore:
                return model_id, await self._evaluate_one(
                    self.registry.get(model_id),
                    example,
                    default_grader=eval_set.grader,
                    system=system,
                    keep_outputs=keep_outputs,
                )

        tasks = [one(m, e) for m in targets for e in eval_set.examples]
        log.info(
            "eval_started",
            eval_set=eval_set.name,
            models=targets,
            examples=len(eval_set),
            calls=len(tasks),
            concurrency=self.concurrency,
        )

        for model_id in targets:
            spec = self.registry.get(model_id).spec
            report.models[model_id] = ModelReport(model_id=model_id, provider=spec.provider)

        for coro in asyncio.as_completed(tasks):
            model_id, result = await coro
            report.models[model_id].results.append(result)

        report.duration_s = time.perf_counter() - started
        log.info(
            "eval_finished",
            eval_set=eval_set.name,
            duration_s=round(report.duration_s, 2),
            models={m: round(r.accuracy, 3) for m, r in report.models.items()},
        )
        return report

    async def _evaluate_one(
        self,
        provider: ModelProvider,
        example: EvalExample,
        *,
        default_grader: str,
        system: str | None,
        keep_outputs: bool,
    ) -> ExampleResult:
        grader_name = example.grader or default_grader
        try:
            result: CompletionResult = await provider.complete(
                example.input, max_tokens=self.max_tokens, system=system
            )
        except ProviderError as exc:
            # Recorded as an availability failure, not a wrong answer.
            return ExampleResult(
                example_id=example.id,
                model_id=provider.model_id,
                score=0.0,
                passed=False,
                latency_ms=0.0,
                cost_usd=0.0,
                error=str(exc),
            )

        grade = await self._grade(grader_name, result.content, example)
        return ExampleResult(
            example_id=example.id,
            model_id=provider.model_id,
            score=grade.score,
            passed=grade.passed,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            explanation=grade.explanation,
            output=result.content if keep_outputs else None,
        )

    async def _grade(self, name: str, output: str, example: EvalExample) -> GradeResult:
        if name != "llm_judge":
            return get_grader(name)(output, example.expected)

        if self.judge_model_id is None:
            return GradeResult(0.0, False, "llm_judge requested but no judge_model_id configured")
        judge = self.registry.get(self.judge_model_id)

        async def judge_call(prompt: str) -> str:
            reply = await judge.complete(prompt, max_tokens=8)
            return reply.content

        grade_fn = make_llm_judge(judge_call)
        return await grade_fn(output, example.expected, example.input)


__all__ = ["EvalReport", "EvalRunner", "ExampleResult", "ModelReport"]
