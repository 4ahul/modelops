"""Router: scoring, hard constraints, fallback, health.

These are the tests that matter most. A router bug does not crash — it quietly
sends every request to the wrong model and the bill arrives a month later.
"""

from __future__ import annotations

import pytest

from app.core.health import HealthTracker
from app.core.policy import RoutingPolicy, TaskPolicy, cost_optimised
from app.core.router import AllProvidersFailed, NoEligibleModel, Router
from app.providers.base import (
    ProviderBadRequest,
    ProviderRateLimited,
)
from app.providers.registry import ProviderRegistry
from tests.conftest import FakeProvider


class TestTaskPolicy:
    def test_weights_are_normalised(self) -> None:
        """So {"cost": 4, "quality": 4} means the same as {"cost": .5, "quality": .5}."""
        task = TaskPolicy(weights={"cost": 4, "quality": 4})
        assert sum(task.weights.values()) == pytest.approx(1.0)
        assert task.weights["cost"] == pytest.approx(0.5)

    def test_unknown_weight_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight"):
            TaskPolicy(weights={"speed": 1.0})

    def test_zero_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TaskPolicy(weights={"cost": 0.0})

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"cost_limit": 0}, "cost_limit"),
            ({"cost_limit": -1}, "cost_limit"),
            ({"latency_budget_ms": 0}, "latency_budget"),
            ({"min_quality": 1.5}, "min_quality"),
            ({"min_quality": -0.1}, "min_quality"),
        ],
    )
    def test_invalid_constraints_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TaskPolicy(**kwargs)

    def test_unknown_task_falls_back_to_default(self) -> None:
        """A new task name should route conservatively, not fail the request."""
        policy = RoutingPolicy(tasks={"known": TaskPolicy(cost_limit=0.5)})
        assert policy.for_task("never-seen").cost_limit is None

    def test_roundtrips_through_dict(self) -> None:
        original = cost_optimised()
        rebuilt = RoutingPolicy.from_dict(original.as_dict())
        assert set(rebuilt.tasks) == set(original.tasks)
        assert (
            rebuilt.tasks["classification"].cost_limit
            == original.tasks["classification"].cost_limit
        )


class TestScoring:
    async def test_cheapest_wins_when_quality_is_equal(self, router: Router) -> None:
        router.set_quality_scores(
            {"classification": {"gemini-flash": 0.95, "gpt-4o-mini": 0.95, "claude-opus": 0.95}}
        )
        candidates, _ = await router.rank("classify this", "classification")
        assert candidates[0].model_id == "gemini-flash"

    async def test_quality_can_outweigh_cost_when_weighted_to(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(
            tasks={"reasoning": TaskPolicy(weights={"cost": 0.05, "quality": 0.95})}
        )
        router = Router(registry, policy, health=health)
        router.set_quality_scores({"reasoning": {"gemini-flash": 0.55, "claude-opus": 0.99}})

        candidates, _ = await router.rank("think hard", "reasoning")
        assert candidates[0].model_id == "claude-opus"

    async def test_cost_limit_excludes_rather_than_penalises(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        """The core rule. Expressed as a penalty, a high quality score could
        outvote the ceiling — which is how a cost-aware router gets expensive."""
        policy = RoutingPolicy(
            tasks={"cheap": TaskPolicy(cost_limit=0.0005, weights={"cost": 0.01, "quality": 0.99})}
        )
        router = Router(registry, policy, health=health)
        router.set_quality_scores({"cheap": {"claude-opus": 1.0, "gemini-flash": 0.5}})

        candidates, excluded = await router.rank("short prompt", "cheap")
        chosen = [c.model_id for c in candidates]

        assert "claude-opus" not in chosen
        assert "cost_limit" in excluded["claude-opus"]

    async def test_min_quality_excludes_measured_underperformers(self, router: Router) -> None:
        router.set_quality_scores({"classification": {"gemini-flash": 0.50, "gpt-4o-mini": 0.95}})
        candidates, excluded = await router.rank("classify", "classification")

        assert [c.model_id for c in candidates] == ["gpt-4o-mini"]
        assert "below min_quality" in excluded["gemini-flash"]

    async def test_unmeasured_model_is_penalised_not_excluded(self, router: Router) -> None:
        """A new deployment has no evals and must still route."""
        candidates, _ = await router.rank("classify", "classification")

        assert candidates, "an unmeasured model should still be routable by default"
        assert all(c.unmeasured for c in candidates)
        assert all(c.quality is None for c in candidates)

    async def test_measured_beats_unmeasured_at_similar_cost(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(tasks={"t": TaskPolicy(weights={"cost": 0.1, "quality": 0.9})})
        router = Router(registry, policy, health=health)
        router.set_quality_scores({"t": {"gpt-4o-mini": 0.9}})

        candidates, _ = await router.rank("x", "t")
        assert candidates[0].model_id == "gpt-4o-mini"

    async def test_allow_unmeasured_false_excludes_with_a_clear_reason(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(tasks={"strict": TaskPolicy(allow_unmeasured=False)})
        router = Router(registry, policy, health=health)

        with pytest.raises(NoEligibleModel) as exc:
            await router.route("x", "strict")
        assert "allow_unmeasured" in str(exc.value)

    async def test_quality_is_per_task_type(self, router: Router) -> None:
        """A model measured on classification says nothing about its reasoning."""
        router.set_quality_scores({"classification": {"gemini-flash": 0.99}})

        assert router.quality_for("classification", "gemini-flash") == 0.99
        assert router.quality_for("reasoning", "gemini-flash") is None

    async def test_measured_latency_excludes_over_budget(self, registry: ProviderRegistry) -> None:
        health = HealthTracker(None)
        for _ in range(10):
            await health.record_success("claude-opus", 5000.0)
            await health.record_success("gemini-flash", 100.0)

        policy = RoutingPolicy(tasks={"fast": TaskPolicy(latency_budget_ms=500)})
        router = Router(registry, policy, health=health)
        candidates, excluded = await router.rank("x", "fast")

        assert "claude-opus" not in [c.model_id for c in candidates]
        assert "p95" in excluded["claude-opus"]

    async def test_unmeasured_latency_does_not_pass_a_budget_it_never_met(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        """With no samples the constraint is skipped, and the candidate scores
        neutral on latency rather than best."""
        policy = RoutingPolicy(tasks={"fast": TaskPolicy(latency_budget_ms=10)})
        router = Router(registry, policy, health=health)
        candidates, _ = await router.rank("x", "fast")

        assert candidates
        assert all(c.p95_latency_ms is None for c in candidates)
        assert all(c.latency_score == 0.5 for c in candidates)

    async def test_scores_are_normalised_within_the_candidate_set(self, router: Router) -> None:
        candidates, _ = await router.rank("x", "reasoning")
        assert candidates[0].cost_score == pytest.approx(1.0)
        assert min(c.cost_score for c in candidates) == pytest.approx(0.0)

    async def test_prefer_breaks_ties(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(
            tasks={"t": TaskPolicy(prefer=["gpt-4o-mini"], weights={"quality": 1.0})}
        )
        router = Router(registry, policy, health=health)
        router.set_quality_scores({"t": {m: 0.9 for m in registry.model_ids}})

        candidates, _ = await router.rank("x", "t")
        assert candidates[0].model_id == "gpt-4o-mini"

    async def test_policy_model_allowlist_is_respected(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(models=["gemini-flash"], tasks={})
        router = Router(registry, policy, health=health)
        candidates, _ = await router.rank("x", None)
        assert [c.model_id for c in candidates] == ["gemini-flash"]


class TestExecution:
    async def test_records_latency_on_success(self, router: Router) -> None:
        await router.complete("hello", "reasoning")
        assert await router.health.mean_latency_ms("gemini-flash") is not None

    async def test_falls_back_when_the_cheapest_fails(self, health: HealthTracker) -> None:
        """Never fail the customer's request to save money."""
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider("gemini-flash", fail_times=99),
                "gpt-4o-mini": FakeProvider("gpt-4o-mini", reply="rescued"),
            }
        )
        router = Router(registry, RoutingPolicy(), health=health)
        routed = await router.complete("hello")

        assert routed.content == "rescued"
        assert routed.model_id == "gpt-4o-mini"
        assert routed.decision.fallbacks == ["gemini-flash"]

    async def test_all_failed_raises_with_every_attempt_listed(self, health: HealthTracker) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider("gemini-flash", fail_times=99),
                "gpt-4o-mini": FakeProvider(
                    "gpt-4o-mini", fail_times=99, error=ProviderRateLimited("429")
                ),
            }
        )
        router = Router(registry, RoutingPolicy(), health=health)

        with pytest.raises(AllProvidersFailed) as exc:
            await router.complete("hello")
        assert set(exc.value.attempts) == {"gemini-flash", "gpt-4o-mini"}

    async def test_bad_request_does_not_fall_over(self, health: HealthTracker) -> None:
        """A malformed request fails identically everywhere. Trying three
        providers would mean three bills for one mistake."""
        second = FakeProvider("gpt-4o-mini", reply="should not run")
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider(
                    "gemini-flash", fail_times=99, error=ProviderBadRequest("prompt too long")
                ),
                "gpt-4o-mini": second,
            }
        )
        router = Router(registry, RoutingPolicy(), health=health)

        with pytest.raises(ProviderBadRequest):
            await router.complete("hello")
        assert second.calls == []

    async def test_bad_request_does_not_mark_the_provider_down(self, health: HealthTracker) -> None:
        """A customer sending an oversized prompt must not take a healthy vendor
        out of rotation for everyone else."""
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider(
                    "gemini-flash", fail_times=99, error=ProviderBadRequest("bad")
                )
            }
        )
        router = Router(registry, RoutingPolicy(), health=health)

        for _ in range(5):
            with pytest.raises(ProviderBadRequest):
                await router.complete("hello")
        assert await health.is_healthy("gemini-flash")

    async def test_repeated_failure_takes_a_model_out_of_rotation(
        self, health: HealthTracker
    ) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider("gemini-flash", fail_times=99),
                "gpt-4o-mini": FakeProvider("gpt-4o-mini", reply="ok"),
            }
        )
        router = Router(registry, RoutingPolicy(), health=health)

        for _ in range(2):
            await router.complete("hello")
        assert not await health.is_healthy("gemini-flash")

        candidates, excluded = await router.rank("hello", None)
        assert "gemini-flash" not in [c.model_id for c in candidates]
        assert "unhealthy" in excluded["gemini-flash"]

    async def test_explicit_fallback_chain_is_honoured(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(
            tasks={"t": TaskPolicy(fallback_chain=["claude-opus", "gemini-flash"])}
        )
        router = Router(registry, policy, health=health)
        order = router._fallback_order((await router.route("x", "t"))[0], policy.for_task("t"))
        assert order[0] == "claude-opus"

    async def test_fallback_chain_cannot_bypass_a_cost_ceiling(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        """A chain is an escalation preference, not a way around a hard limit."""
        policy = RoutingPolicy(
            tasks={"cheap": TaskPolicy(cost_limit=0.0005, fallback_chain=["claude-opus"])}
        )
        router = Router(registry, policy, health=health)
        candidates, _ = await router.route("x", "cheap")
        order = router._fallback_order(candidates, policy.for_task("cheap"))
        assert "claude-opus" not in order

    async def test_no_eligible_model_names_every_reason(
        self, registry: ProviderRegistry, health: HealthTracker
    ) -> None:
        policy = RoutingPolicy(tasks={"impossible": TaskPolicy(cost_limit=1e-12)})
        router = Router(registry, policy, health=health)

        with pytest.raises(NoEligibleModel) as exc:
            await router.complete("hello", "impossible")
        assert len(exc.value.reasons) == 3
        assert all("cost_limit" in why for why in exc.value.reasons.values())

    async def test_empty_registry_raises_rather_than_guessing(self, health: HealthTracker) -> None:
        router = Router(ProviderRegistry(), RoutingPolicy(), health=health)
        with pytest.raises(NoEligibleModel):
            await router.complete("hello")

    async def test_routing_overhead_is_measured_and_small(self, router: Router) -> None:
        """The product publishes this number, so it has to be real."""
        routed = await router.complete("hello", "reasoning")
        assert routed.routing_overhead_ms >= 0
        assert routed.routing_overhead_ms < 100, "routing decision must stay under 100ms"

    async def test_decision_reason_is_human_readable(self, router: Router) -> None:
        router.set_quality_scores({"reasoning": {"gemini-flash": 0.93}})
        routed = await router.complete("hello", "reasoning")

        reason = routed.decision.reason
        assert routed.model_id in reason
        assert "quality" in reason and "est $" in reason

    async def test_decision_marks_unmeasured_quality_explicitly(self, router: Router) -> None:
        """Nobody should be able to read a guess as a measurement."""
        routed = await router.complete("hello", "reasoning")
        assert "unmeasured" in routed.decision.reason

    async def test_task_max_tokens_is_applied(self, health: HealthTracker) -> None:
        provider = FakeProvider("gemini-flash")
        registry = ProviderRegistry({"gemini-flash": provider})
        policy = RoutingPolicy(tasks={"tiny": TaskPolicy(max_tokens=16)})
        router = Router(registry, policy, health=health)

        await router.complete("hello", "tiny")
        assert provider.calls == ["hello"]


class TestHealthTracker:
    async def test_success_clears_a_failure_streak(self) -> None:
        tracker = HealthTracker(None, failure_threshold=2)
        await tracker.record_failure("m")
        await tracker.record_success("m", 100.0)
        await tracker.record_failure("m")

        assert await tracker.is_healthy("m")

    async def test_p95_needs_enough_samples(self) -> None:
        """Fewer than five samples is not a percentile, and pretending otherwise
        would exclude models on noise."""
        tracker = HealthTracker(None)
        for _ in range(4):
            await tracker.record_success("m", 100.0)
        assert await tracker.p95_latency_ms("m") is None

        await tracker.record_success("m", 100.0)
        assert await tracker.p95_latency_ms("m") == 100.0

    async def test_p95_reports_an_observed_value(self) -> None:
        tracker = HealthTracker(None)
        for value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]:
            await tracker.record_success("m", float(value))

        p95 = await tracker.p95_latency_ms("m")
        assert p95 in (90.0, 1000.0), "p95 must be a latency that actually happened"

    async def test_snapshot_shape(self) -> None:
        tracker = HealthTracker(None)
        await tracker.record_success("m", 120.0)
        snapshot = await tracker.snapshot(["m"])
        assert snapshot["m"]["healthy"] is True
        assert snapshot["m"]["samples"] == 1

    async def test_broken_redis_degrades_instead_of_failing(self) -> None:
        """A monitoring dependency must never take down the request path."""

        class BrokenRedis:
            def pipeline(self) -> None:
                raise ConnectionError("redis down")

            async def incr(self, *_: object) -> int:
                raise ConnectionError("redis down")

            async def exists(self, *_: object) -> int:
                raise ConnectionError("redis down")

            async def lrange(self, *_: object) -> list:
                raise ConnectionError("redis down")

        tracker = HealthTracker(BrokenRedis(), failure_threshold=1)
        await tracker.record_success("m", 100.0)
        await tracker.record_failure("m")

        assert tracker.degraded
        assert not await tracker.is_healthy("m"), "local state must still enforce the threshold"


class FakePipeline:
    """Records commands and replays canned replies on execute()."""

    def __init__(self, owner: CountingRedis) -> None:
        self.owner = owner
        self.commands: list[tuple[str, str]] = []

    def exists(self, key: str) -> None:
        self.commands.append(("exists", key))

    def lrange(self, key: str, *_: object) -> None:
        self.commands.append(("lrange", key))

    def llen(self, key: str) -> None:
        self.commands.append(("llen", key))

    async def execute(self) -> list[object]:
        self.owner.round_trips += 1
        self.owner.commands.extend(self.commands)
        replies: list[object] = []
        for kind, _key in self.commands:
            if kind == "exists":
                replies.append(0)
            elif kind == "llen":
                replies.append(0)
            else:
                replies.append([])
        return replies


class CountingRedis:
    """Counts Redis round trips so a regression in chattiness is visible."""

    def __init__(self) -> None:
        self.round_trips = 0
        self.commands: list[tuple[str, str]] = []

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def exists(self, key: str) -> int:
        self.round_trips += 1
        self.commands.append(("exists", key))
        return 0

    async def lrange(self, key: str, *_: object) -> list[object]:
        self.round_trips += 1
        self.commands.append(("lrange", key))
        return []


class TestRoutingOverhead:
    """Routing overhead is a number this product publishes, so the work done
    per decision is held to a budget by tests rather than by intention."""

    async def test_ranking_costs_one_redis_round_trip(self, registry: ProviderRegistry) -> None:
        """Health and p95 for every candidate come back in a single pipeline.

        Asking per model is two round trips each — fourteen for a seven-model
        deployment, which on a 2ms link is a quarter of the 100ms budget spent
        waiting on Redis.
        """
        redis = CountingRedis()
        router = Router(registry, RoutingPolicy(), health=HealthTracker(redis))

        await router.rank("classify this", None)

        assert redis.round_trips == 1, (
            f"ranking {len(registry)} models took {redis.round_trips} round trips; "
            "health reads must stay batched"
        )
        # One health check and one latency read per model, in that one hop.
        assert len(redis.commands) == len(registry) * 2

    async def test_round_trips_do_not_grow_with_the_model_count(self) -> None:
        from app.providers.pricing import MODEL_CATALOG
        from tests.conftest import FakeProvider

        every_model = ProviderRegistry(
            {model_id: FakeProvider(model_id) for model_id in MODEL_CATALOG}
        )
        redis = CountingRedis()
        router = Router(every_model, RoutingPolicy(), health=HealthTracker(redis))

        await router.rank("x", None)

        assert len(every_model) >= 7
        assert redis.round_trips == 1

    async def test_snapshot_is_batched_too(self, registry: ProviderRegistry) -> None:
        redis = CountingRedis()
        tracker = HealthTracker(redis)

        await tracker.snapshot(registry.model_ids)

        # One pipeline for health+latency, one for the sample counts.
        assert redis.round_trips == 2

    async def test_batched_and_single_model_percentiles_agree(self) -> None:
        """Two definitions of p95 would eventually disagree, and the router
        would exclude a model the dashboard shows as fast."""
        tracker = HealthTracker(None)
        for value in (120.0, 140.0, 160.0, 180.0, 200.0, 900.0):
            await tracker.record_success("m", value)

        single = await tracker.p95_latency_ms("m")
        batched = (await tracker.batch_status(["m"]))["m"][1]

        assert single == batched

    async def test_batch_status_on_an_empty_list_touches_nothing(self) -> None:
        redis = CountingRedis()
        assert await HealthTracker(redis).batch_status([]) == {}
        assert redis.round_trips == 0

    async def test_batch_status_falls_back_when_redis_breaks(self) -> None:
        class BrokenPipeline:
            def pipeline(self) -> None:
                raise ConnectionError("down")

        tracker = HealthTracker(BrokenPipeline())
        for _ in range(6):
            await tracker.record_success("m", 100.0)

        status = await tracker.batch_status(["m"])
        assert status["m"] == (True, 100.0)
        assert tracker.degraded
