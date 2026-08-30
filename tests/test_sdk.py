"""SDK client: transport, typed errors, retries.

The client is tested against the real ASGI app rather than a mocked transport,
so a change to a response shape breaks these tests instead of a customer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import RateLimiter, hash_key
from app.core.alerts import AlertEngine
from app.core.config import Settings
from app.core.health import HealthTracker
from app.core.policy import RoutingPolicy, TaskPolicy
from app.core.router import Router
from app.db.models import Base
from app.eval.runner import EvalRunner
from app.main import create_app
from app.providers.registry import ProviderRegistry
from modelops import (
    ModelOps,
    ModelOpsError,
    ModelOpsSync,
    NoEligibleModelError,
    ProviderFailedError,
    RateLimitedError,
)
from modelops.client import AuthenticationError, EvalReport
from tests.conftest import EchoProvider, FakeProvider

KEY = "mo_sdk_test_key"


async def _client_over(
    registry: ProviderRegistry,
    *,
    policy: RoutingPolicy | None = None,
    rate_limit: int = 120,
) -> tuple[ModelOps, httpx.AsyncClient, object]:
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key_hashes=hash_key(KEY),
        rate_limit_per_minute=rate_limit,
        log_format="console",
    )
    app = create_app(settings)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from app.db import session as session_module

    session_module._engine = engine
    session_module._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    app.state.redis = None
    app.state.rate_limiter = RateLimiter(None, per_minute=rate_limit)
    app.state.registry = registry
    app.state.router = Router(registry, policy or RoutingPolicy(), health=HealthTracker(None))
    app.state.eval_runner = EvalRunner(registry, concurrency=4)
    app.state.alert_engine = AlertEngine()

    transport = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    return ModelOps("http://test", KEY, client=transport), transport, engine


@pytest_asyncio.fixture
async def sdk(registry: ProviderRegistry) -> AsyncIterator[ModelOps]:
    client, transport, engine = await _client_over(registry)
    yield client
    await transport.aclose()
    await engine.dispose()


class TestCompletion:
    async def test_returns_a_typed_result(self, sdk: ModelOps) -> None:
        result = await sdk.complete("classify this", "classification")

        assert result.content == "cheap"
        assert result.model_id == "gemini-flash"
        assert result.cost_usd > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens

    async def test_str_is_the_content(self, sdk: ModelOps) -> None:
        """So a caller can drop the result straight into a prompt or a log line."""
        assert str(await sdk.complete("hi")) == "cheap"

    async def test_pinning_a_model(self, sdk: ModelOps) -> None:
        result = await sdk.complete("hi", model_id="claude-opus")
        assert result.model_id == "claude-opus"

    async def test_route_costs_nothing(self, sdk: ModelOps, registry: ProviderRegistry) -> None:
        decision = await sdk.route("hi")
        assert decision.chosen == "gemini-flash"
        assert len(decision.candidates) == 3
        assert all(not p.calls for p in registry)


class TestErrors:
    async def test_bad_key_raises_authentication_error(self, registry: ProviderRegistry) -> None:
        client, transport, engine = await _client_over(registry)
        transport.headers["Authorization"] = "Bearer wrong"

        with pytest.raises(AuthenticationError):
            await client.complete("hi")

        await transport.aclose()
        await engine.dispose()

    async def test_no_eligible_model_carries_the_reasons(self, registry: ProviderRegistry) -> None:
        """The reasons are the whole value of the error: they say which
        constraint to relax."""
        policy = RoutingPolicy(tasks={"impossible": TaskPolicy(cost_limit=1e-12)})
        client, transport, engine = await _client_over(registry, policy=policy)

        with pytest.raises(NoEligibleModelError) as exc:
            await client.complete("hi", "impossible")
        assert len(exc.value.excluded) == 3
        assert all("cost_limit" in why for why in exc.value.excluded.values())

        await transport.aclose()
        await engine.dispose()

    async def test_provider_failure_lists_every_attempt(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider("gemini-flash", fail_times=99),
                "gpt-4o-mini": FakeProvider("gpt-4o-mini", fail_times=99),
            }
        )
        client, transport, engine = await _client_over(registry)

        with pytest.raises(ProviderFailedError) as exc:
            await client.complete("hi")
        assert set(exc.value.attempts) == {"gemini-flash", "gpt-4o-mini"}

        await transport.aclose()
        await engine.dispose()

    async def test_rate_limit_raises_with_retry_after(self, registry: ProviderRegistry) -> None:
        client, transport, engine = await _client_over(registry, rate_limit=1)
        client.max_retries = 0  # so the error surfaces instead of being retried

        await client.complete("first")
        with pytest.raises(RateLimitedError) as exc:
            await client.complete("second")
        assert exc.value.retry_after == 60.0

        await transport.aclose()
        await engine.dispose()

    async def test_unreachable_host_is_a_clear_error(self) -> None:
        client = ModelOps("http://127.0.0.1:1", "k", max_retries=0)
        with pytest.raises(ModelOpsError, match="Could not reach"):
            await client.health()
        await client.close()

    async def test_client_errors_are_not_retried(self, registry: ProviderRegistry) -> None:
        """A 4xx fails identically however many times it is sent."""
        client, transport, engine = await _client_over(registry)
        transport.headers["Authorization"] = "Bearer wrong"

        calls = 0
        original = transport.request

        async def counting(*args: object, **kwargs: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            return await original(*args, **kwargs)  # type: ignore[arg-type]

        transport.request = counting  # type: ignore[method-assign]
        with pytest.raises(AuthenticationError):
            await client.complete("hi")
        assert calls == 1

        await transport.aclose()
        await engine.dispose()


class TestEvalsThroughSDK:
    async def test_upload_run_and_pick_a_model(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": EchoProvider("gemini-flash", latency_ms=100),
                "claude-opus": EchoProvider("claude-opus", latency_ms=900),
            }
        )
        client, transport, engine = await _client_over(registry)

        created = await client.upload_eval_set(
            "classification",
            [
                {"input": "q1||spam", "expected": "spam"},
                {"input": "q2||ham", "expected": "ham"},
            ],
            task_type="classification",
        )
        assert created["version"] == 1

        report = await client.run_eval("classification")
        assert isinstance(report, EvalReport)
        assert len(report.models) == 2

        pick = report.cheapest_above(0.9)
        assert pick is not None and pick["model_id"] == "gemini-flash"
        assert "gemini-flash" in report.table()
        assert report.recommendation is not None

        await transport.aclose()
        await engine.dispose()

    async def test_best_by_accuracy_on_an_empty_report(self) -> None:
        assert EvalReport("s", 1, 0, 0.0, []).best_by_accuracy() is None


class TestOpsThroughSDK:
    async def test_metrics_reflect_traffic(self, sdk: ModelOps) -> None:
        await sdk.complete("hi")
        metrics = await sdk.metrics(hours=1)
        assert metrics["requests"] == 1

    async def test_models_and_health(self, sdk: ModelOps) -> None:
        assert len((await sdk.models())["available"]) == 3
        assert (await sdk.health())["database"] is True

    async def test_alerts_start_empty(self, sdk: ModelOps) -> None:
        assert await sdk.alerts() == []

    async def test_context_manager_closes_its_own_transport(self) -> None:
        async with ModelOps("http://127.0.0.1:1", "k", max_retries=0) as client:
            assert client.api_url == "http://127.0.0.1:1"


class TestSyncClient:
    def test_refuses_to_run_inside_a_loop(self) -> None:
        """Nesting event loops silently deadlocks; an explicit error is better."""
        import asyncio

        async def inner() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                ModelOpsSync("http://test", "k").health()

        asyncio.run(inner())

    def test_trailing_slash_is_normalised(self) -> None:
        assert ModelOps("http://test/", "k").api_url == "http://test"
