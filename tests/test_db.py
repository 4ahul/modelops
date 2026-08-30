"""Persistence: recording, aggregation, versioning, retention."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.alerts import AlertEngine, AlertThresholds
from app.core.router import Candidate, RoutingDecision
from app.db.crud import (
    _utcnow,
    acknowledge_alert,
    cost_summary,
    cost_timeseries,
    create_alert,
    eval_history,
    latency_percentiles,
    latest_quality_scores,
    list_alerts,
    list_eval_sets,
    load_eval_set,
    purge_routing_decisions,
    record_eval_report,
    record_routing_decision,
    upsert_eval_set,
)
from app.db.models import EvalResultRow, RoutingDecisionRow
from app.eval.dataset import EvalExample, EvalSet
from app.eval.runner import EvalRunner
from app.providers.base import CompletionResult
from app.providers.registry import ProviderRegistry


def _decision(model: str = "gemini-flash", task: str | None = "classification") -> RoutingDecision:
    return RoutingDecision(
        task_type=task,
        chosen=model,
        candidates=[
            Candidate(
                model_id=model,
                provider="gemini",
                estimated_cost=0.0001,
                quality=0.93,
                p95_latency_ms=120.0,
                score=0.9,
            )
        ],
        excluded={},
        overhead_ms=3.2,
    )


def _result(
    model: str = "gemini-flash", cost: float = 0.0002, latency: float = 130.0
) -> CompletionResult:
    return CompletionResult(
        content="ok",
        provider="gemini",
        model="gemini-1.5-flash-002",
        model_id=model,
        input_tokens=100,
        output_tokens=50,
        latency_ms=latency,
        cost_usd=cost,
    )


class TestRoutingRecords:
    async def test_records_a_successful_call(self, session: AsyncSession) -> None:
        row = await record_routing_decision(
            session, decision=_decision(), result=_result(), quality_score=0.93
        )

        assert row.succeeded is True
        assert row.chosen_model == "gemini-flash"
        assert row.routing_overhead_ms == pytest.approx(3.2)
        assert row.quality_score == pytest.approx(0.93)

    async def test_records_a_failure_too(self, session: AsyncSession) -> None:
        """A table of only successes cannot answer "how often does the cheap
        model fall over", which is the question that decides if a saving is real."""
        row = await record_routing_decision(
            session, decision=_decision(), result=None, error="all providers failed"
        )

        assert row.succeeded is False
        assert row.cost_usd == 0.0
        assert row.error == "all providers failed"

    async def test_prompt_is_not_stored_by_default(self, session: AsyncSession) -> None:
        row = await record_routing_decision(session, decision=_decision(), result=_result())
        assert row.prompt_preview is None

    async def test_prompt_preview_only_when_passed_explicitly(self, session: AsyncSession) -> None:
        row = await record_routing_decision(
            session, decision=_decision(), result=_result(), prompt_preview="first 200 chars"
        )
        assert row.prompt_preview == "first 200 chars"

    async def test_excluded_models_are_kept_for_debugging(self, session: AsyncSession) -> None:
        decision = _decision()
        decision.excluded = {"claude-opus": "over cost_limit"}
        row = await record_routing_decision(session, decision=decision, result=_result())
        assert row.extra == {"excluded": {"claude-opus": "over cost_limit"}}


class TestAggregation:
    async def test_summary_totals(self, session: AsyncSession) -> None:
        for _ in range(3):
            await record_routing_decision(session, decision=_decision(), result=_result(cost=0.001))

        summary = await cost_summary(session, hours=24)
        assert summary["requests"] == 3
        assert summary["total_cost_usd"] == pytest.approx(0.003)
        assert summary["avg_cost_usd"] == pytest.approx(0.001)
        assert summary["failure_rate"] == 0.0

    async def test_summary_counts_failures_separately(self, session: AsyncSession) -> None:
        await record_routing_decision(session, decision=_decision(), result=_result())
        await record_routing_decision(session, decision=_decision(), result=None, error="down")

        summary = await cost_summary(session, hours=24)
        assert summary["requests"] == 2
        assert summary["failures"] == 1
        assert summary["failure_rate"] == pytest.approx(0.5)

    async def test_summary_filters_by_task_type(self, session: AsyncSession) -> None:
        await record_routing_decision(
            session, decision=_decision(task="classification"), result=_result()
        )
        await record_routing_decision(
            session, decision=_decision(task="drafting"), result=_result()
        )

        assert (await cost_summary(session, hours=24, task_type="drafting"))["requests"] == 1

    async def test_summary_breaks_down_by_model(self, session: AsyncSession) -> None:
        await record_routing_decision(
            session, decision=_decision("gemini-flash"), result=_result(cost=0.001)
        )
        await record_routing_decision(
            session, decision=_decision("claude-opus"), result=_result("claude-opus", cost=0.05)
        )

        by_model = {r["model_id"]: r for r in (await cost_summary(session, hours=24))["by_model"]}
        assert by_model["claude-opus"]["cost_usd"] == pytest.approx(0.05)
        # Ordered by spend, so the expensive one is first.
        assert (await cost_summary(session, hours=24))["by_model"][0]["model_id"] == "claude-opus"

    async def test_empty_window_returns_zeros_not_an_error(self, session: AsyncSession) -> None:
        summary = await cost_summary(session, hours=1)
        assert summary["requests"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["by_model"] == []

    async def test_percentiles_ignore_failed_calls(self, session: AsyncSession) -> None:
        for latency in (100.0, 200.0, 300.0, 400.0, 5000.0):
            await record_routing_decision(
                session, decision=_decision(), result=_result(latency=latency)
            )
        await record_routing_decision(session, decision=_decision(), result=None, error="down")

        percentiles = await latency_percentiles(session, hours=24)
        assert percentiles["samples"] == 5
        assert percentiles["p50"] in (200.0, 300.0)

    async def test_percentiles_on_empty_data(self, session: AsyncSession) -> None:
        assert (await latency_percentiles(session, hours=1))["samples"] == 0

    async def test_timeseries_buckets_by_time(self, session: AsyncSession) -> None:
        for _ in range(4):
            await record_routing_decision(session, decision=_decision(), result=_result(cost=0.001))

        series = await cost_timeseries(session, hours=24, bucket_minutes=60)
        assert len(series) >= 1
        assert sum(b["requests"] for b in series) == 4
        assert sum(b["cost_usd"] for b in series) == pytest.approx(0.004)

    async def test_purge_removes_old_rows_only(self, session: AsyncSession) -> None:
        old = RoutingDecisionRow(
            created_at=datetime.now(UTC) - timedelta(days=100),
            chosen_model="gemini-flash",
            provider="gemini",
        )
        session.add(old)
        await session.commit()
        await record_routing_decision(session, decision=_decision(), result=_result())

        removed = await purge_routing_decisions(
            session, before=datetime.now(UTC) - timedelta(days=30)
        )
        remaining = (await session.execute(select(RoutingDecisionRow))).scalars().all()

        assert removed == 1
        assert len(remaining) == 1


class TestAggregationIsPushedIntoSQL:
    """These queries were rewritten to aggregate in the database rather than
    pull a window into Python. ``hours`` allows 90 days, so the naive version
    loaded millions of rows to produce three numbers or a 24-point chart.

    A rewrite of percentile logic is exactly where an off-by-one hides, so both
    are checked against a straightforward reference implementation.
    """

    @staticmethod
    def _reference_percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(quantile * len(ordered) + 0.5)) - 1))
        return round(ordered[index], 1)

    async def test_percentiles_match_a_python_reference(self, session: AsyncSession) -> None:
        latencies = [float(v) for v in (91, 12, 47, 350, 8, 1200, 63, 25, 180, 74, 33, 900, 51)]
        for latency in latencies:
            await record_routing_decision(
                session, decision=_decision(), result=_result(latency=latency)
            )

        result = await latency_percentiles(session, hours=24)

        assert result["samples"] == len(latencies)
        assert result["p50"] == self._reference_percentile(latencies, 0.50)
        assert result["p95"] == self._reference_percentile(latencies, 0.95)
        assert result["p99"] == self._reference_percentile(latencies, 0.99)

    async def test_percentile_of_a_single_sample(self, session: AsyncSession) -> None:
        """The offset arithmetic must not walk off the end of a one-row window."""
        await record_routing_decision(session, decision=_decision(), result=_result(latency=42.0))
        result = await latency_percentiles(session, hours=24)
        assert result == {"p50": 42.0, "p95": 42.0, "p99": 42.0, "samples": 1}

    async def test_percentiles_are_observed_values(self, session: AsyncSession) -> None:
        """Nearest rank, so a reported p99 is a latency that actually happened
        and not an interpolation between two that did not."""
        latencies = [10.0, 20.0, 30.0, 40.0, 5000.0]
        for latency in latencies:
            await record_routing_decision(
                session, decision=_decision(), result=_result(latency=latency)
            )

        result = await latency_percentiles(session, hours=24)
        assert result["p99"] in latencies
        assert result["p50"] in latencies

    async def test_percentiles_respect_the_key_scope(self, session: AsyncSession) -> None:
        await record_routing_decision(
            session, decision=_decision(), result=_result(latency=10.0), api_key_hash="a"
        )
        await record_routing_decision(
            session, decision=_decision(), result=_result(latency=9000.0), api_key_hash="b"
        )

        scoped = await latency_percentiles(session, hours=24, api_key_hash="a")
        assert scoped["samples"] == 1
        assert scoped["p95"] == 10.0

    async def test_sql_bucketing_matches_the_python_fallback(self, session: AsyncSession) -> None:
        """The dialect-specific bucket expression is the fast path; the Python
        version is the fallback. They must not disagree about which bucket a row
        lands in."""
        from app.db.crud import _cost_timeseries_in_python

        now = datetime.now(UTC)
        for minutes, model in ((5, "gemini-flash"), (75, "gemini-flash"), (95, "claude-opus")):
            session.add(
                RoutingDecisionRow(
                    created_at=now - timedelta(minutes=minutes),
                    chosen_model=model,
                    provider="x",
                    cost_usd=0.001,
                    latency_ms=100.0,
                )
            )
        await session.commit()

        via_sql = await cost_timeseries(session, hours=4, bucket_minutes=60)

        since = _utcnow() - timedelta(hours=4)
        via_python = await _cost_timeseries_in_python(
            session, [RoutingDecisionRow.created_at >= since], since, 3600
        )

        assert [b["requests"] for b in via_sql] == [b["requests"] for b in via_python]
        assert [b["models"] for b in via_sql] == [b["models"] for b in via_python]
        assert sum(b["requests"] for b in via_sql) == 3

    async def test_timeseries_returns_one_row_per_bucket_not_per_request(
        self, session: AsyncSession
    ) -> None:
        now = datetime.now(UTC)
        for _ in range(50):
            session.add(
                RoutingDecisionRow(
                    created_at=now - timedelta(minutes=5),
                    chosen_model="gemini-flash",
                    provider="gemini",
                    cost_usd=0.001,
                    latency_ms=100.0,
                )
            )
        await session.commit()

        series = await cost_timeseries(session, hours=1, bucket_minutes=60)

        assert len(series) == 1
        assert series[0]["requests"] == 50
        assert series[0]["cost_usd"] == pytest.approx(0.05)

    async def test_timeseries_respects_the_key_scope(self, session: AsyncSession) -> None:
        await record_routing_decision(
            session, decision=_decision(), result=_result(), api_key_hash="a"
        )
        series = await cost_timeseries(session, hours=1, api_key_hash="b")
        assert sum(b["requests"] for b in series) == 0


class TestEvalPersistence:
    async def test_store_and_reload(self, session: AsyncSession, eval_set: EvalSet) -> None:
        row = await upsert_eval_set(session, eval_set)
        assert row.version == 1

        reloaded = await load_eval_set(session, eval_set.name)
        assert reloaded is not None
        assert len(reloaded) == 4
        assert reloaded.task_type == "classification"

    async def test_unchanged_set_is_not_re_versioned(
        self, session: AsyncSession, eval_set: EvalSet
    ) -> None:
        first = await upsert_eval_set(session, eval_set)
        second = await upsert_eval_set(session, eval_set)
        assert first.id == second.id
        assert second.version == 1

    async def test_changed_set_gets_a_new_version(
        self, session: AsyncSession, eval_set: EvalSet
    ) -> None:
        """A stored result must always refer to the exact data it measured."""
        await upsert_eval_set(session, eval_set)
        eval_set.examples.append(EvalExample(input="new||yes", expected="yes", id="e5"))

        assert (await upsert_eval_set(session, eval_set)).version == 2

    async def test_load_returns_the_newest_version(
        self, session: AsyncSession, eval_set: EvalSet
    ) -> None:
        await upsert_eval_set(session, eval_set)
        eval_set.examples.append(EvalExample(input="new||yes", expected="yes"))
        await upsert_eval_set(session, eval_set)

        loaded = await load_eval_set(session, eval_set.name)
        assert loaded is not None and len(loaded) == 5

    async def test_load_missing_set_returns_none(self, session: AsyncSession) -> None:
        assert await load_eval_set(session, "never-uploaded") is None

    async def test_list_sets(self, session: AsyncSession, eval_set: EvalSet) -> None:
        await upsert_eval_set(session, eval_set)
        listed = await list_eval_sets(session)
        assert listed[0]["name"] == eval_set.name
        assert listed[0]["examples"] == 4

    async def test_record_report_writes_one_row_per_model(
        self, session: AsyncSession, eval_set: EvalSet, echo_registry: ProviderRegistry
    ) -> None:
        row = await upsert_eval_set(session, eval_set)
        report = await EvalRunner(echo_registry).run(eval_set)
        written = await record_eval_report(session, row, report)

        assert len(written) == 2
        assert {r.model_id for r in written} == {"gemini-flash", "claude-opus"}
        assert all(r.eval_version == row.version for r in written)

    async def test_quality_scores_use_the_latest_result(self, session: AsyncSession) -> None:
        eval_row = await upsert_eval_set(
            session,
            EvalSet(name="s", task_type="classification", examples=[EvalExample(input="q")]),
        )
        older = EvalResultRow(
            eval_set_id=eval_row.id,
            task_type="classification",
            model_id="gemini-flash",
            provider="gemini",
            accuracy=0.70,
            evaluated_at=datetime.now(UTC) - timedelta(days=2),
        )
        newer = EvalResultRow(
            eval_set_id=eval_row.id,
            task_type="classification",
            model_id="gemini-flash",
            provider="gemini",
            accuracy=0.95,
            evaluated_at=datetime.now(UTC),
        )
        session.add_all([older, newer])
        await session.commit()

        scores = await latest_quality_scores(session)
        assert scores["classification"]["gemini-flash"] == pytest.approx(0.95)

    async def test_quality_scores_are_keyed_by_task(self, session: AsyncSession) -> None:
        eval_row = await upsert_eval_set(
            session, EvalSet(name="s", examples=[EvalExample(input="q")])
        )
        session.add_all(
            [
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    task_type="classification",
                    model_id="m",
                    provider="p",
                    accuracy=0.9,
                ),
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    task_type="reasoning",
                    model_id="m",
                    provider="p",
                    accuracy=0.4,
                ),
            ]
        )
        await session.commit()

        scores = await latest_quality_scores(session)
        assert scores["classification"]["m"] == pytest.approx(0.9)
        assert scores["reasoning"]["m"] == pytest.approx(0.4)

    async def test_history(
        self, session: AsyncSession, eval_set: EvalSet, echo_registry: ProviderRegistry
    ) -> None:
        row = await upsert_eval_set(session, eval_set)
        await record_eval_report(session, row, await EvalRunner(echo_registry).run(eval_set))

        history = await eval_history(session, name=eval_set.name)
        assert len(history) == 2
        assert history[0]["eval_set"] == eval_set.name


class TestAlerts:
    async def test_dedupe_suppresses_a_repeat(self, session: AsyncSession) -> None:
        """An alerting system that repeats gets muted, and a muted alert is
        worse than none."""
        first = await create_alert(
            session, kind="cost_spike", message="up 60%", dedupe_key="cost_spike:all:2026-08-30T12"
        )
        second = await create_alert(
            session, kind="cost_spike", message="up 61%", dedupe_key="cost_spike:all:2026-08-30T12"
        )

        assert first is not None
        assert second is None
        assert len(await list_alerts(session)) == 1

    async def test_acknowledge(self, session: AsyncSession) -> None:
        row = await create_alert(session, kind="cost_spike", message="x")
        assert row is not None
        assert await acknowledge_alert(session, row.id) is True
        assert await list_alerts(session, unacknowledged_only=True) == []

    async def test_acknowledge_unknown_id(self, session: AsyncSession) -> None:
        assert await acknowledge_alert(session, "nope") is False

    async def test_cost_spike_fires_on_per_request_cost(self, session: AsyncSession) -> None:
        """Rate, not total: a doubling of traffic is not a cost regression, and
        alerting on it trains people to ignore the alert."""
        now = datetime.now(UTC)
        for _ in range(25):
            session.add(
                RoutingDecisionRow(
                    created_at=now - timedelta(hours=1, minutes=30),
                    chosen_model="gemini-flash",
                    provider="gemini",
                    cost_usd=0.001,
                    latency_ms=100,
                )
            )
        for _ in range(25):
            session.add(
                RoutingDecisionRow(
                    created_at=now - timedelta(minutes=10),
                    chosen_model="claude-opus",
                    provider="anthropic",
                    cost_usd=0.010,
                    latency_ms=100,
                )
            )
        await session.commit()

        fired = await AlertEngine().check_cost_spike(session, window_hours=1)
        assert len(fired) == 1
        assert fired[0]["kind"] == "cost_spike"
        assert fired[0]["severity"] == "critical"

    async def test_traffic_growth_alone_does_not_fire(self, session: AsyncSession) -> None:
        now = datetime.now(UTC)
        for count, offset in ((25, timedelta(hours=1, minutes=30)), (100, timedelta(minutes=10))):
            for _ in range(count):
                session.add(
                    RoutingDecisionRow(
                        created_at=now - offset,
                        chosen_model="gemini-flash",
                        provider="gemini",
                        cost_usd=0.001,
                        latency_ms=100,
                    )
                )
        await session.commit()

        assert await AlertEngine().check_cost_spike(session, window_hours=1) == []

    async def test_thin_traffic_produces_no_alert(self, session: AsyncSession) -> None:
        """Comparing three requests to two is arithmetic, not signal."""
        now = datetime.now(UTC)
        session.add(
            RoutingDecisionRow(
                created_at=now - timedelta(hours=1, minutes=30),
                chosen_model="gemini-flash",
                provider="gemini",
                cost_usd=0.001,
                latency_ms=100,
            )
        )
        session.add(
            RoutingDecisionRow(
                created_at=now - timedelta(minutes=5),
                chosen_model="claude-opus",
                provider="anthropic",
                cost_usd=0.500,
                latency_ms=100,
            )
        )
        await session.commit()

        assert await AlertEngine().check_cost_spike(session, window_hours=1) == []

    async def test_latency_regression_fires_per_model(self, session: AsyncSession) -> None:
        now = datetime.now(UTC)
        for latency, offset in (
            (100.0, timedelta(hours=1, minutes=30)),
            (900.0, timedelta(minutes=5)),
        ):
            for _ in range(25):
                session.add(
                    RoutingDecisionRow(
                        created_at=now - offset,
                        chosen_model="gemini-flash",
                        provider="gemini",
                        cost_usd=0.001,
                        latency_ms=latency,
                        succeeded=True,
                    )
                )
        await session.commit()

        fired = await AlertEngine().check_latency_regression(session, window_hours=1)
        assert len(fired) == 1
        assert fired[0]["model_id"] == "gemini-flash"
        assert fired[0]["metric"] == "p95_latency_ms"

    async def test_accuracy_regression_fires(self, session: AsyncSession) -> None:
        eval_row = await upsert_eval_set(
            session,
            EvalSet(name="s", task_type="classification", examples=[EvalExample(input="q")]),
        )
        now = datetime.now(UTC)
        session.add_all(
            [
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    eval_version=1,
                    task_type="classification",
                    model_id="gemini-flash",
                    provider="gemini",
                    accuracy=0.95,
                    example_count=50,
                    evaluated_at=now - timedelta(days=1),
                ),
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    eval_version=1,
                    task_type="classification",
                    model_id="gemini-flash",
                    provider="gemini",
                    accuracy=0.70,
                    example_count=50,
                    evaluated_at=now,
                ),
            ]
        )
        await session.commit()

        fired = await AlertEngine().check_accuracy_regression(session)
        assert len(fired) == 1
        assert fired[0]["severity"] == "critical"

    async def test_eval_version_change_downgrades_to_informational(
        self, session: AsyncSession
    ) -> None:
        """Different test data means the two numbers do not measure the same
        thing, so it must not be reported as a confirmed regression."""
        eval_row = await upsert_eval_set(
            session,
            EvalSet(name="s", task_type="classification", examples=[EvalExample(input="q")]),
        )
        now = datetime.now(UTC)
        session.add_all(
            [
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    eval_version=1,
                    task_type="classification",
                    model_id="m",
                    provider="p",
                    accuracy=0.95,
                    example_count=50,
                    evaluated_at=now - timedelta(days=1),
                ),
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    eval_version=2,
                    task_type="classification",
                    model_id="m",
                    provider="p",
                    accuracy=0.70,
                    example_count=50,
                    evaluated_at=now,
                ),
            ]
        )
        await session.commit()

        fired = await AlertEngine().check_accuracy_regression(session)
        assert fired[0]["severity"] == "info"
        assert "changed version" in fired[0]["message"]

    async def test_small_eval_sets_do_not_alert(self, session: AsyncSession) -> None:
        eval_row = await upsert_eval_set(
            session, EvalSet(name="s", examples=[EvalExample(input="q")])
        )
        now = datetime.now(UTC)
        session.add_all(
            [
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    model_id="m",
                    provider="p",
                    accuracy=0.95,
                    example_count=3,
                    evaluated_at=now - timedelta(days=1),
                ),
                EvalResultRow(
                    eval_set_id=eval_row.id,
                    model_id="m",
                    provider="p",
                    accuracy=0.20,
                    example_count=3,
                    evaluated_at=now,
                ),
            ]
        )
        await session.commit()

        assert await AlertEngine().check_accuracy_regression(session) == []

    async def test_thresholds_are_configurable(self, session: AsyncSession) -> None:
        now = datetime.now(UTC)
        for cost, offset in (
            (0.001, timedelta(hours=1, minutes=30)),
            (0.0012, timedelta(minutes=5)),
        ):
            for _ in range(25):
                session.add(
                    RoutingDecisionRow(
                        created_at=now - offset,
                        chosen_model="gemini-flash",
                        provider="gemini",
                        cost_usd=cost,
                        latency_ms=100,
                    )
                )
        await session.commit()

        assert await AlertEngine().check_cost_spike(session, window_hours=1) == []
        strict = AlertEngine(AlertThresholds(cost_spike_pct=0.1))
        assert len(await strict.check_cost_spike(session, window_hours=1)) == 1
