"""Eval framework: datasets, graders, runner, report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.dataset import EvalExample, EvalSet
from app.eval.graders import (
    GRADERS,
    GradeResult,
    contains,
    exact_match,
    fuzzy,
    get_grader,
    json_match,
    json_schema,
    make_llm_judge,
    numeric,
    regex_match,
)
from app.eval.runner import EvalRunner, ExampleResult, ModelReport
from app.providers.base import ProviderUnavailable
from app.providers.registry import ProviderRegistry
from tests.conftest import EchoProvider, FakeProvider


class TestEvalSet:
    def test_empty_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="no examples"):
            EvalSet(name="empty", examples=[])

    def test_blank_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            EvalExample(input="")

    def test_non_positive_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            EvalExample(input="x", weight=0)

    def test_ids_are_filled_in(self) -> None:
        """A report must be able to name a failing case even when the source
        file supplied no id."""
        eval_set = EvalSet(name="s", examples=[EvalExample(input="a"), EvalExample(input="b")])
        assert [e.id for e in eval_set] == ["s-0", "s-1"]

    def test_explicit_ids_are_preserved(self) -> None:
        eval_set = EvalSet(name="s", examples=[EvalExample(input="a", id="keep-me")])
        assert eval_set.examples[0].id == "keep-me"

    def test_from_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "evals.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"input": "q1", "expected": "a1", "tags": ["short"]}),
                    "",
                    "// a comment line is skipped",
                    json.dumps({"input": "q2", "expected": "a2", "grader": "contains"}),
                ]
            ),
            encoding="utf-8",
        )
        eval_set = EvalSet.from_jsonl(path, task_type="classification")

        assert len(eval_set) == 2
        assert eval_set.name == "evals"
        assert eval_set.task_type == "classification"
        assert eval_set.examples[0].tags == ("short",)
        assert eval_set.examples[1].grader == "contains"

    def test_jsonl_error_names_the_line(self, tmp_path: Path) -> None:
        """A 500-example file with one typo is otherwise painful to debug."""
        path = tmp_path / "bad.jsonl"
        path.write_text('{"input": "ok"}\nnot json at all\n', encoding="utf-8")

        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            EvalSet.from_jsonl(path)

    def test_jsonl_missing_input_field_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"expected": "a"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field 'input'"):
            EvalSet.from_jsonl(path)

    def test_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EvalSet.from_jsonl(tmp_path / "nope.jsonl")

    def test_roundtrip_through_jsonl(self, tmp_path: Path, eval_set: EvalSet) -> None:
        path = tmp_path / "out.jsonl"
        assert eval_set.to_jsonl(path) == 4
        assert len(EvalSet.from_jsonl(path)) == 4

    def test_filter_by_tag(self) -> None:
        eval_set = EvalSet(
            name="s",
            examples=[
                EvalExample(input="a", tags=("keep",)),
                EvalExample(input="b", tags=("drop",)),
            ],
        )
        assert len(eval_set.filter(tag="keep")) == 1

    def test_filter_on_absent_tag_is_an_error_not_an_empty_set(self) -> None:
        """An empty eval set would silently report 0% accuracy for everything."""
        eval_set = EvalSet(name="s", examples=[EvalExample(input="a")])
        with pytest.raises(ValueError, match="carry tag"):
            eval_set.filter(tag="missing")

    def test_sample_is_deterministic(self, eval_set: EvalSet) -> None:
        """Two runs must compare like with like."""
        first = [e.id for e in eval_set.sample(2, seed=7)]
        second = [e.id for e in eval_set.sample(2, seed=7)]
        assert first == second

    def test_sample_larger_than_the_set_returns_it_whole(self, eval_set: EvalSet) -> None:
        assert len(eval_set.sample(99)) == len(eval_set)


class TestGraders:
    @pytest.mark.parametrize(
        "output,expected,score",
        [
            ("spam", "spam", 1.0),
            ("Spam.", "spam", 1.0),  # normalised: punctuation and case
            ("  spam  ", "spam", 1.0),
            ("ham", "spam", 0.0),
        ],
    )
    def test_exact_match(self, output: str, expected: str, score: float) -> None:
        assert exact_match(output, expected).score == score

    def test_exact_match_without_an_expectation_scores_zero_and_says_why(self) -> None:
        result = exact_match("anything", None)
        assert result.score == 0.0 and "no expected value" in result.explanation

    def test_contains(self) -> None:
        assert contains("The label is spam, definitely", "spam").passed
        assert not contains("clean", "spam").passed

    def test_fuzzy_reports_the_ratio(self) -> None:
        result = fuzzy("the quick brown fox", "the quick brown fix")
        assert 0.8 < result.score < 1.0

    def test_fuzzy_threshold(self) -> None:
        assert not fuzzy("completely different", "nothing alike").passed

    def test_regex(self) -> None:
        assert regex_match("order 12345 shipped", r"order \d+").passed
        assert not regex_match("no digits", r"order \d+").passed

    def test_regex_invalid_pattern_scores_zero_rather_than_crashing(self) -> None:
        """A malformed pattern in one example must not abort a 300-call run."""
        result = regex_match("x", "[unclosed")
        assert result.score == 0.0 and "invalid pattern" in result.explanation

    @pytest.mark.parametrize(
        "output,expected,passed",
        [
            ("42", 42, True),
            ("the answer is 42.0", 42, True),
            ("1,234", 1234, True),
            ("-5", -5, True),
            ("43", 42, False),
            ("no number here", 42, False),
        ],
    )
    def test_numeric(self, output: str, expected: float, passed: bool) -> None:
        assert numeric(output, expected).passed is passed

    def test_numeric_tolerance_scales_with_magnitude(self) -> None:
        """1_000_000 and 0.5 cannot be judged by the same absolute yardstick."""
        assert numeric("1000000.5", 1_000_000, tolerance=1e-3).passed
        assert not numeric("0.6", 0.5, tolerance=1e-3).passed

    def test_json_match_ignores_formatting(self) -> None:
        assert json_match('{"b": 2, "a": 1}', {"a": 1, "b": 2}).score == 1.0

    def test_json_match_strips_a_code_fence(self) -> None:
        """Models wrap JSON in fences even when told not to. Failing a correct
        extraction over that would measure instruction-following instead."""
        assert json_match('```json\n{"a": 1}\n```', {"a": 1}).score == 1.0

    def test_json_match_finds_json_inside_prose(self) -> None:
        assert json_match('Sure! Here it is: {"a": 1} — hope that helps', {"a": 1}).score == 1.0

    def test_json_match_gives_partial_credit(self) -> None:
        """Wrong on one field of four is a different failure from wrong on all."""
        result = json_match('{"a": 1, "b": 2, "c": 3, "d": 9}', {"a": 1, "b": 2, "c": 3, "d": 4})
        assert result.score == pytest.approx(0.75)
        assert not result.passed

    def test_json_match_reports_missing_and_wrong_keys(self) -> None:
        result = json_match('{"a": 1, "b": 99}', {"a": 1, "b": 2, "c": 3})
        assert "missing ['c']" in result.explanation
        assert "wrong ['b']" in result.explanation

    def test_json_match_on_unparseable_output(self) -> None:
        assert json_match("not json", {"a": 1}).score == 0.0

    def test_json_schema_checks_shape_not_values(self) -> None:
        schema = {"required": ["name", "age"], "types": {"age": "int"}}
        assert json_schema('{"name": "Ana", "age": 31}', schema).score == 1.0
        assert json_schema('{"name": "Bob", "age": 44}', schema).score == 1.0

    def test_json_schema_partial_credit_and_reasons(self) -> None:
        schema = {"required": ["name", "age"], "types": {"age": "int"}}
        result = json_schema('{"name": "Ana", "age": "thirty"}', schema)
        assert 0 < result.score < 1
        assert "'age' is not int" in result.explanation

    def test_json_schema_unknown_type_is_reported(self) -> None:
        result = json_schema('{"a": 1}', {"types": {"a": "widget"}})
        assert "unknown type" in result.explanation

    def test_grade_result_rejects_an_out_of_range_score(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            GradeResult(1.5, True)

    def test_get_grader_lists_the_valid_names(self) -> None:
        with pytest.raises(KeyError, match="exact_match"):
            get_grader("does_not_exist")

    def test_every_registered_grader_is_callable(self) -> None:
        for name in GRADERS:
            assert get_grader(name)("x", "x") is not None

    async def test_llm_judge_parses_an_ordinal_verdict(self) -> None:
        async def judge(_: str) -> str:
            return "4"

        result = await make_llm_judge(judge)("an answer", "the reference", "the question")
        assert result.score == pytest.approx(0.8)
        assert result.passed

    async def test_llm_judge_unparseable_reply_scores_zero_with_a_reason(self) -> None:
        async def judge(_: str) -> str:
            return "I would rather not say"

        result = await make_llm_judge(judge)("a", "b")
        assert result.score == 0.0 and "unparseable" in result.explanation

    async def test_judge_failure_is_not_reported_as_a_model_failure(self) -> None:
        """A broken judge would otherwise record a quality regression that never
        happened."""

        async def judge(_: str) -> str:
            raise ProviderUnavailable("judge is down")

        result = await make_llm_judge(judge)("a", "b")
        assert "judge unavailable" in result.explanation


class TestRunner:
    async def test_scores_every_model_on_every_example(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        report = await EvalRunner(echo_registry).run(eval_set)

        assert set(report.models) == {"gemini-flash", "claude-opus"}
        assert report.models["gemini-flash"].accuracy == 1.0
        assert report.example_count == 4

    async def test_provider_error_is_availability_not_wrongness(self, eval_set: EvalSet) -> None:
        """Mixing an outage into accuracy would make a down provider look like a
        quality regression, and those two facts lead to opposite actions."""
        registry = ProviderRegistry(
            {
                "gemini-flash": EchoProvider("gemini-flash"),
                "claude-opus": FakeProvider("claude-opus", fail_times=99),
            }
        )
        report = await EvalRunner(registry).run(eval_set)
        failed = report.models["claude-opus"]

        assert failed.error_rate == 1.0
        assert failed.accuracy == 0.0
        assert failed.graded == []
        assert report.models["gemini-flash"].error_rate == 0.0

    async def test_partial_failure_does_not_poison_accuracy(self, eval_set: EvalSet) -> None:
        registry = ProviderRegistry({"gemini-flash": EchoProvider("gemini-flash", fail_times=2)})
        report = await EvalRunner(registry, concurrency=1).run(eval_set)
        model = report.models["gemini-flash"]

        assert len(model.errors) == 2
        assert model.accuracy == 1.0, "the answered examples were all correct"
        assert model.error_rate == 0.5

    async def test_unknown_model_is_rejected_before_any_calls(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        with pytest.raises(KeyError, match="not available"):
            await EvalRunner(echo_registry).run(eval_set, model_ids=["gpt-5"])

    async def test_empty_registry_is_an_error(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="registry is empty"):
            await EvalRunner(ProviderRegistry()).run(eval_set)

    async def test_outputs_are_not_kept_by_default(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        """An eval report is stored and shared, and output can carry customer data."""
        report = await EvalRunner(echo_registry).run(eval_set)
        assert all(r.output is None for r in report.models["gemini-flash"].results)

    async def test_outputs_kept_when_asked(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        report = await EvalRunner(echo_registry).run(eval_set, keep_outputs=True)
        assert all(r.output for r in report.models["gemini-flash"].results)

    async def test_concurrency_is_bounded(self, eval_set: EvalSet) -> None:
        """Unbounded gather turns a 300-call run into a wall of 429s."""
        import asyncio

        live = 0
        peak = 0

        class Counting(EchoProvider):
            async def _complete(self, prompt: str, **kwargs: object) -> object:
                nonlocal live, peak
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.01)
                live -= 1
                return await super()._complete(prompt, **kwargs)  # type: ignore[arg-type]

        registry = ProviderRegistry({"gemini-flash": Counting("gemini-flash")})
        await EvalRunner(registry, concurrency=2).run(eval_set)
        assert peak <= 2

    async def test_per_example_grader_overrides_the_default(self) -> None:
        eval_set = EvalSet(
            name="mixed",
            grader="exact_match",
            examples=[
                EvalExample(input="q||the label is spam", expected="spam", grader="contains")
            ],
        )
        report = await EvalRunner(
            ProviderRegistry({"gemini-flash": EchoProvider("gemini-flash")})
        ).run(eval_set)
        assert report.models["gemini-flash"].accuracy == 1.0

    async def test_llm_judge_without_a_judge_model_is_reported_not_silent(self) -> None:
        eval_set = EvalSet(
            name="judged",
            grader="llm_judge",
            examples=[EvalExample(input="q||a", expected="a")],
        )
        report = await EvalRunner(
            ProviderRegistry({"gemini-flash": EchoProvider("gemini-flash")})
        ).run(eval_set)
        result = report.models["gemini-flash"].results[0]
        assert result.score == 0.0
        assert "no judge_model_id" in result.explanation


class TestReport:
    @staticmethod
    def _report(
        *scores: float, latencies: list[float] | None = None, cost: float = 0.001
    ) -> ModelReport:
        report = ModelReport(model_id="m", provider="p")
        lat = latencies or [100.0] * len(scores)
        for index, score in enumerate(scores):
            report.results.append(
                ExampleResult(
                    example_id=f"e{index}",
                    model_id="m",
                    score=score,
                    passed=score >= 0.5,
                    latency_ms=lat[index],
                    cost_usd=cost,
                )
            )
        return report

    def test_accuracy_and_pass_rate_differ(self) -> None:
        report = self._report(1.0, 0.6, 0.4, 0.0)
        assert report.accuracy == pytest.approx(0.5)
        assert report.pass_rate == pytest.approx(0.5)

    def test_percentiles_are_observed_values(self) -> None:
        report = self._report(*[1.0] * 10, latencies=[float(v) for v in range(10, 110, 10)])
        assert report.p50_latency_ms in {50.0, 60.0}
        assert report.p95_latency_ms in {90.0, 100.0}

    def test_percentiles_on_an_empty_report_are_zero_not_an_error(self) -> None:
        empty = ModelReport(model_id="m", provider="p")
        assert empty.p95_latency_ms == 0.0
        assert empty.accuracy == 0.0
        assert empty.cost_per_query == 0.0

    def test_cost_per_correct_answer_is_the_decisive_number(self) -> None:
        """A model at half the price that gets a third fewer right is more
        expensive per correct answer, not less."""
        cheap_but_wrong = self._report(1.0, 0.0, 0.0, 0.0, cost=0.001)
        pricey_but_right = self._report(1.0, 1.0, 1.0, 1.0, cost=0.003)

        assert cheap_but_wrong.cost_per_query < pricey_but_right.cost_per_query
        assert cheap_but_wrong.cost_per_correct_answer > pricey_but_right.cost_per_correct_answer

    def test_cost_per_correct_is_infinite_when_nothing_passes(self) -> None:
        assert self._report(0.0, 0.0).cost_per_correct_answer == float("inf")

    def test_as_dict_renders_infinity_as_null(self) -> None:
        """``float('inf')`` is not valid JSON."""
        assert self._report(0.0).as_dict()["cost_per_correct_answer"] is None

    async def test_cheapest_above_a_bar(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        report = await EvalRunner(echo_registry).run(eval_set)
        pick = report.cheapest_above(0.9)
        assert pick is not None and pick.model_id == "gemini-flash"

    async def test_cheapest_above_an_unreachable_bar_is_none(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        assert (await EvalRunner(echo_registry).run(eval_set)).cheapest_above(1.01) is None

    async def test_savings_calculation(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        report = await EvalRunner(echo_registry).run(eval_set)
        savings = report.savings_vs("claude-opus", 0.9)

        assert savings is not None
        assert savings["to"] == "gemini-flash"
        assert savings["cost_reduction_pct"] > 90
        assert savings["accuracy_delta"] == pytest.approx(0.0)

    async def test_savings_is_none_when_the_baseline_already_wins(self, eval_set: EvalSet) -> None:
        registry = ProviderRegistry({"gemini-flash": EchoProvider("gemini-flash")})
        report = await EvalRunner(registry).run(eval_set)
        assert report.savings_vs("gemini-flash", 0.9) is None

    async def test_table_is_ordered_cheapest_first(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        table = (await EvalRunner(echo_registry).run(eval_set)).table()
        assert table.index("gemini-flash") < table.index("claude-opus")

    async def test_failures_lists_the_worst_examples(self, eval_set: EvalSet) -> None:
        registry = ProviderRegistry({"gemini-flash": FakeProvider("gemini-flash", reply="wrong")})
        report = await EvalRunner(registry).run(eval_set)
        worst = report.models["gemini-flash"].failures(limit=2)
        assert len(worst) == 2 and all(r.score == 0.0 for r in worst)

    async def test_as_dict_is_json_serialisable(
        self, echo_registry: ProviderRegistry, eval_set: EvalSet
    ) -> None:
        report = await EvalRunner(echo_registry).run(eval_set)
        assert json.loads(json.dumps(report.as_dict()))["eval_set"] == "test_classification"
