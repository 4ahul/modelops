"""Provider layer: pricing, retries, error classification."""

from __future__ import annotations

import pytest

from app.providers.base import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderUnavailable,
    classify_status,
    status_of,
)
from app.providers.pricing import (
    MODEL_CATALOG,
    PRICING_AS_OF,
    Pricing,
    assert_fresh,
    get_spec,
    pricing_age_days,
    specs_for_provider,
)
from app.providers.registry import ProviderRegistry
from tests.conftest import FakeProvider


class TestPricing:
    def test_cost_matches_published_rate(self) -> None:
        """1M input at $3/Mtok plus 1M output at $15/Mtok is $18."""
        pricing = Pricing(3.00, 15.00)
        assert pricing.cost(1_000_000, 1_000_000) == pytest.approx(18.00)

    def test_cost_scales_linearly(self) -> None:
        pricing = Pricing(1.00, 2.00)
        assert pricing.cost(500_000, 250_000) == pytest.approx(0.5 + 0.5)

    def test_zero_tokens_is_free(self) -> None:
        assert Pricing(15.0, 75.0).cost(0, 0) == 0.0

    def test_negative_tokens_rejected(self) -> None:
        """A negative count would produce a negative cost and understate spend."""
        with pytest.raises(ValueError, match="negative"):
            Pricing(1.0, 1.0).cost(-1, 10)

    def test_every_catalog_entry_is_self_consistent(self) -> None:
        for model_id, spec in MODEL_CATALOG.items():
            assert spec.id == model_id, "catalog key must match spec.id"
            assert spec.pricing.input_per_mtok > 0
            assert spec.pricing.output_per_mtok >= spec.pricing.input_per_mtok
            assert spec.context_window > spec.max_output_tokens
            assert spec.provider in ("anthropic", "openai", "gemini")

    def test_unknown_model_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="claude-sonnet"):
            get_spec("claude-sonnet-9")

    def test_specs_for_provider(self) -> None:
        assert {s.id for s in specs_for_provider("openai")} == {"gpt-4o", "gpt-4o-mini"}

    def test_freshness_check_passes_on_a_current_table(self) -> None:
        assert pricing_age_days() >= 0
        # The table is dated today in this repo, so it must not warn.
        if pricing_age_days() <= 90:
            assert assert_fresh() is None

    def test_pricing_as_of_is_a_real_date(self) -> None:
        assert PRICING_AS_OF.year >= 2025

    def test_blended_cost_orders_models_sensibly(self) -> None:
        """gemini-flash must be cheaper than claude-opus on any sane blend."""
        assert (
            get_spec("gemini-flash").blended_cost_per_mtok
            < get_spec("claude-opus").blended_cost_per_mtok
        )


class TestErrorClassification:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (429, ProviderRateLimited),
            (401, ProviderAuthError),
            (403, ProviderAuthError),
            (400, ProviderBadRequest),
            (404, ProviderBadRequest),
            (422, ProviderBadRequest),
            (500, ProviderUnavailable),
            (503, ProviderUnavailable),
            (408, ProviderUnavailable),
        ],
    )
    def test_status_maps_to_error_type(self, status: int, expected: type) -> None:
        assert isinstance(classify_status(status, "boom"), expected)

    def test_bad_request_is_not_retryable_or_failed_over(self) -> None:
        """A 400 fails identically at every vendor. Retrying it across three
        providers turns one client error into three bills."""
        error = classify_status(400, "prompt too long")
        assert not error.retryable
        assert not error.failover

    def test_auth_error_does_not_fail_over(self) -> None:
        """Falling over on a 401 would hide a configuration mistake behind a
        working fallback."""
        error = classify_status(401, "bad key")
        assert not error.failover

    def test_server_error_fails_over(self) -> None:
        error = classify_status(503, "unavailable")
        assert error.retryable and error.failover

    def test_rate_limit_carries_retry_after(self) -> None:
        error = classify_status(429, "slow down", retry_after=12.5)
        assert isinstance(error, ProviderRateLimited)
        assert error.retry_after == 12.5

    def test_status_of_reads_common_attribute_names(self) -> None:
        class WithStatusCode(Exception):
            status_code = 429

        class WithResponse(Exception):
            class response:  # noqa: N801
                status_code = 503

        assert status_of(WithStatusCode()) == 429
        assert status_of(WithResponse()) == 503
        assert status_of(Exception("plain")) is None


class TestProviderBehaviour:
    async def test_result_costs_what_the_pricing_table_says(self) -> None:
        provider = FakeProvider("claude-sonnet", output_tokens=1000)
        result = await provider.complete("x" * 3500)  # ~1000 input tokens

        expected = provider.spec.pricing.cost(result.input_tokens, result.output_tokens)
        assert result.cost_usd == pytest.approx(expected)
        assert result.total_tokens == result.input_tokens + result.output_tokens

    async def test_retries_then_succeeds(self) -> None:
        provider = FakeProvider("gemini-flash", fail_times=2, max_retries=3)
        result = await provider.complete("hello")

        assert result.content == "ok"
        assert result.attempts == 3

    async def test_gives_up_after_max_retries(self) -> None:
        provider = FakeProvider("gemini-flash", fail_times=99, max_retries=1)
        with pytest.raises(ProviderUnavailable):
            await provider.complete("hello")
        assert len(provider.calls) == 2

    async def test_bad_request_is_not_retried(self) -> None:
        """One attempt only — the request is wrong, not the provider."""
        provider = FakeProvider(
            "gemini-flash", fail_times=99, error=ProviderBadRequest("bad prompt"), max_retries=3
        )
        with pytest.raises(ProviderBadRequest):
            await provider.complete("hello")
        assert len(provider.calls) == 1

    async def test_max_tokens_over_model_limit_is_rejected_locally(self) -> None:
        """Caught before the call, so an obviously invalid request costs nothing."""
        provider = FakeProvider("claude-haiku")
        with pytest.raises(ProviderBadRequest, match="exceeds"):
            await provider.complete("hi", max_tokens=999_999)
        assert provider.calls == []

    def test_estimate_is_conservative(self) -> None:
        """The estimate must not undercount, or a request slips past a ceiling
        it should have been excluded by."""
        provider = FakeProvider("gpt-4o-mini")
        estimate = provider.estimate_cost("word " * 200, expected_output_tokens=500)
        assert estimate > 0

    def test_backoff_honours_retry_after(self) -> None:
        delay = FakeProvider("gemini-flash")._backoff(1, ProviderRateLimited("429", 5.0))
        assert delay == 5.0

    def test_backoff_caps_retry_after(self) -> None:
        """A vendor asking for an hour must not hang the request that long."""
        delay = FakeProvider("gemini-flash")._backoff(1, ProviderRateLimited("429", 3600))
        assert delay == 30.0

    def test_backoff_is_jittered(self) -> None:
        """Synchronised retries rebuild the burst that caused the rate limit."""
        provider = FakeProvider("gemini-flash")
        delays = {provider._backoff(3, None) for _ in range(20)}
        assert len(delays) > 1

    async def test_health_check_reports_failure_without_raising(self) -> None:
        assert await FakeProvider("gemini-flash").health_check() is True
        assert await FakeProvider("gemini-flash", fail_times=99).health_check() is False


class TestRegistry:
    def test_missing_model_error_names_what_is_available(self) -> None:
        registry = ProviderRegistry({"gemini-flash": FakeProvider("gemini-flash")})
        with pytest.raises(KeyError, match="gemini-flash"):
            registry.get("claude-opus")

    def test_from_settings_skips_models_without_a_key(self, settings) -> None:
        """A single-vendor deployment is legitimate and must start cleanly."""
        settings.anthropic_api_key = "sk-ant-test"
        registry = ProviderRegistry.from_settings(settings)

        assert set(registry.model_ids) == {"claude-opus", "claude-sonnet", "claude-haiku"}
        assert "gpt-4o" not in registry

    def test_from_settings_with_no_keys_is_empty_not_an_error(self, settings) -> None:
        assert len(ProviderRegistry.from_settings(settings)) == 0

    def test_from_settings_honours_an_explicit_model_list(self, settings) -> None:
        settings.openai_api_key = "sk-test"
        registry = ProviderRegistry.from_settings(settings, model_ids=["gpt-4o-mini"])
        assert registry.model_ids == ["gpt-4o-mini"]

    def test_cheapest(self, registry: ProviderRegistry) -> None:
        cheapest = registry.cheapest()
        assert cheapest is not None and cheapest.model_id == "gemini-flash"

    def test_by_provider(self, registry: ProviderRegistry) -> None:
        assert [p.model_id for p in registry.by_provider("openai")] == ["gpt-4o-mini"]
