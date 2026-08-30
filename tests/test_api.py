"""HTTP API: auth, rate limiting, routing endpoints, evals, ops.

The app is built with an in-memory SQLite database and fake providers, so these
are real end-to-end request tests with no network and no Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import RateLimiter, hash_key, verify_key
from app.core.alerts import AlertEngine
from app.core.config import Settings
from app.core.health import HealthTracker
from app.core.policy import RoutingPolicy, TaskPolicy
from app.core.router import Router
from app.db.models import Base
from app.eval.runner import EvalRunner
from app.main import create_app
from app.providers.base import ProviderBadRequest
from app.providers.registry import ProviderRegistry
from tests.conftest import EchoProvider, FakeProvider

TEST_KEY = "mo_test_key_value"


async def _build_client(
    settings: Settings,
    registry: ProviderRegistry,
    policy: RoutingPolicy | None = None,
) -> tuple[AsyncClient, Any]:
    """Build an app with its state wired by hand.

    The real lifespan is bypassed: it would connect to Redis and Postgres, and
    what these tests exercise is the request path, not startup.
    """
    app = create_app(settings)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.db import session as session_module

    session_module._engine = engine
    session_module._sessionmaker = factory

    app.state.redis = None
    app.state.rate_limiter = RateLimiter(None, per_minute=settings.rate_limit_per_minute)
    app.state.registry = registry
    app.state.router = Router(registry, policy or RoutingPolicy(), health=HealthTracker(None))
    app.state.eval_runner = EvalRunner(registry, concurrency=4)
    app.state.alert_engine = AlertEngine()

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return client, engine


@pytest_asyncio.fixture
async def client(registry: ProviderRegistry) -> AsyncIterator[AsyncClient]:
    """An authenticated client with three fake models."""
    settings = Settings(
        environment="development",
        database_url="sqlite+aiosqlite:///:memory:",
        api_key_hashes=hash_key(TEST_KEY),
        log_format="console",
    )
    http, engine = await _build_client(settings, registry)
    http.headers["Authorization"] = f"Bearer {TEST_KEY}"
    yield http
    await http.aclose()
    await engine.dispose()


@pytest_asyncio.fixture
async def open_client(registry: ProviderRegistry) -> AsyncIterator[AsyncClient]:
    """A client against a deployment with no keys configured — local dev mode."""
    settings = Settings(
        environment="development",
        database_url="sqlite+aiosqlite:///:memory:",
        api_key_hashes="",
        log_format="console",
    )
    http, engine = await _build_client(settings, registry)
    yield http
    await http.aclose()
    await engine.dispose()


class TestKeyHashing:
    def test_hash_is_stable_and_not_the_key(self) -> None:
        digest = hash_key("secret")
        assert digest == hash_key("secret")
        assert "secret" not in digest
        assert len(digest) == 64

    def test_verify_matches_only_the_right_key(self) -> None:
        valid = frozenset({hash_key("right")})
        assert verify_key("right", valid) == hash_key("right")
        assert verify_key("wrong", valid) is None

    def test_verify_against_no_configured_keys(self) -> None:
        assert verify_key("anything", frozenset()) is None


class TestRateLimiter:
    async def test_allows_up_to_the_limit_then_blocks(self) -> None:
        limiter = RateLimiter(None, per_minute=3)
        results = [await limiter.check("k") for _ in range(4)]
        assert [allowed for allowed, _ in results] == [True, True, True, False]

    async def test_limits_are_per_key(self) -> None:
        limiter = RateLimiter(None, per_minute=1)
        assert (await limiter.check("a"))[0] is True
        assert (await limiter.check("b"))[0] is True
        assert (await limiter.check("a"))[0] is False

    async def test_zero_means_unlimited(self) -> None:
        limiter = RateLimiter(None, per_minute=0)
        for _ in range(50):
            allowed, remaining = await limiter.check("k")
            assert allowed and remaining == -1

    async def test_broken_redis_falls_back_rather_than_refusing_traffic(self) -> None:
        """A monitoring outage must not become a total outage."""

        class BrokenRedis:
            async def incr(self, *_: object) -> int:
                raise ConnectionError("down")

        limiter = RateLimiter(BrokenRedis(), per_minute=2)
        assert (await limiter.check("k"))[0] is True
        assert limiter.degraded


class TestAuth:
    async def test_missing_key_is_401_with_instructions(self, client: AsyncClient) -> None:
        response = await client.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": ""}
        )
        assert response.status_code == 401
        assert "X-API-Key" in response.json()["detail"]

    async def test_wrong_key_is_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        assert response.status_code == 401

    async def test_wrong_key_reveals_nothing_about_why(self, client: AsyncClient) -> None:
        """Distinguishing "unknown" from "revoked" tells an attacker which
        guesses were close."""
        response = await client.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        assert response.json()["detail"] == "Invalid API key"

    async def test_x_api_key_header_works(self, client: AsyncClient) -> None:
        response = await client.post(
            "/complete",
            json={"prompt": "hi"},
            headers={"Authorization": "", "X-API-Key": TEST_KEY},
        )
        assert response.status_code == 200

    async def test_no_keys_configured_runs_open_for_local_dev(
        self, open_client: AsyncClient
    ) -> None:
        assert (await open_client.post("/complete", json={"prompt": "hi"})).status_code == 200

    async def test_auth_uses_the_settings_the_app_was_built_with(
        self, registry: ProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the auth dependency must not read process-wide settings.

        ``Depends(get_settings)`` returns the cached object parsed from the
        environment. An app constructed with explicit settings would then check
        keys against a different configuration — and with no keys in the
        environment, that meant no authentication at all.
        """
        monkeypatch.delenv("API_KEY_HASHES", raising=False)
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key("only-this-key"),
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)

        assert (await http.post("/complete", json={"prompt": "hi"})).status_code == 401
        authorised = await http.post(
            "/complete",
            json={"prompt": "hi"},
            headers={"Authorization": "Bearer only-this-key"},
        )
        assert authorised.status_code == 200

        await http.aclose()
        await engine.dispose()

    async def test_production_refuses_to_start_without_keys(self) -> None:
        """A service that silently serves unauthenticated traffic is worse than
        one that does not start."""
        settings = Settings(
            environment="production",
            api_key_hashes="",
            database_url="postgresql+asyncpg://u:p@db:5432/x",
        )
        problems = settings.validate_for_production()
        assert any("API_KEY_HASHES" in p for p in problems)

    async def test_health_needs_no_key(self, client: AsyncClient) -> None:
        """A load balancer cannot hold a credential."""
        response = await client.get("/health", headers={"Authorization": ""})
        assert response.status_code == 200


class TestComplete:
    async def test_routes_to_the_cheapest_and_reports_cost(self, client: AsyncClient) -> None:
        response = await client.post(
            "/complete", json={"prompt": "classify this", "task_type": "classification"}
        )
        body = response.json()

        assert response.status_code == 200
        assert body["model_id"] == "gemini-flash"
        assert body["cost_usd"] > 0
        assert body["routing_overhead_ms"] >= 0
        assert "gemini-flash" in body["routing_reason"]

    async def test_records_the_decision(self, client: AsyncClient) -> None:
        await client.post("/complete", json={"prompt": "hello"})
        metrics = (await client.get("/metrics", params={"hours": 1})).json()
        assert metrics["requests"] == 1

    async def test_pinned_model_bypasses_routing(self, client: AsyncClient) -> None:
        response = await client.post("/complete", json={"prompt": "hi", "model_id": "claude-opus"})
        body = response.json()
        assert body["model_id"] == "claude-opus"
        assert "pinned" in body["routing_reason"]

    async def test_pinned_unknown_model_is_404_listing_what_exists(
        self, client: AsyncClient
    ) -> None:
        response = await client.post("/complete", json={"prompt": "hi", "model_id": "gpt-9"})
        assert response.status_code == 404
        assert "gemini-flash" in response.json()["detail"]

    async def test_pinned_calls_are_still_recorded(self, client: AsyncClient) -> None:
        """Excluding them would make the cost dashboard disagree with the invoice."""
        await client.post("/complete", json={"prompt": "hi", "model_id": "claude-opus"})
        metrics = (await client.get("/metrics", params={"hours": 1})).json()
        assert metrics["by_model"][0]["model_id"] == "claude-opus"

    async def test_impossible_policy_is_422_with_reasons(self, registry: ProviderRegistry) -> None:
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        policy = RoutingPolicy(tasks={"impossible": TaskPolicy(cost_limit=1e-12)})
        http, engine = await _build_client(settings, registry, policy)
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        response = await http.post("/complete", json={"prompt": "hi", "task_type": "impossible"})
        detail = response.json()["detail"]

        assert response.status_code == 422
        assert detail["kind"] == "no_eligible_model"
        # The per-model reasons say which constraint to relax.
        assert len(detail["context"]["excluded"]) == 3

        await http.aclose()
        await engine.dispose()

    async def test_all_providers_down_is_502_and_recorded(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider("gemini-flash", fail_times=99),
                "gpt-4o-mini": FakeProvider("gpt-4o-mini", fail_times=99),
            }
        )
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        response = await http.post("/complete", json={"prompt": "hi"})
        assert response.status_code == 502
        assert response.json()["detail"]["kind"] == "all_providers_failed"

        metrics = (await http.get("/metrics", params={"hours": 1})).json()
        assert metrics["failures"] == 1

        await http.aclose()
        await engine.dispose()

    async def test_bad_request_is_400_not_502(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": FakeProvider(
                    "gemini-flash", fail_times=99, error=ProviderBadRequest("prompt too long")
                )
            }
        )
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        response = await http.post("/complete", json={"prompt": "hi"})
        assert response.status_code == 400

        await http.aclose()
        await engine.dispose()

    async def test_no_providers_configured_is_503(self) -> None:
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        http, engine = await _build_client(settings, ProviderRegistry())
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        response = await http.post("/complete", json={"prompt": "hi"})
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

        await http.aclose()
        await engine.dispose()

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"prompt": ""},
            {"prompt": "   "},
            {"prompt": "hi", "temperature": 5},
            {"prompt": "hi", "max_tokens": 0},
            {"prompt": "hi", "unexpected_field": 1},
        ],
    )
    async def test_invalid_payloads_are_422(
        self, client: AsyncClient, payload: dict[str, Any]
    ) -> None:
        assert (await client.post("/complete", json=payload)).status_code == 422

    async def test_prompt_is_not_persisted_by_default(self, client: AsyncClient) -> None:
        from sqlalchemy import select

        from app.db.models import RoutingDecisionRow
        from app.db.session import get_sessionmaker

        await client.post("/complete", json={"prompt": "a very sensitive prompt"})
        async with get_sessionmaker()() as db:
            rows = (await db.execute(select(RoutingDecisionRow))).scalars().all()
        assert all(row.prompt_preview is None for row in rows)


class TestRoute:
    async def test_explains_the_decision_without_calling_a_model(
        self, client: AsyncClient, registry: ProviderRegistry
    ) -> None:
        response = await client.post("/route", json={"prompt": "hi"})
        body = response.json()

        assert response.status_code == 200
        assert body["chosen"] == "gemini-flash"
        assert len(body["candidates"]) == 3
        # Nothing ran.
        assert all(not p.calls for p in registry)

    async def test_candidates_carry_their_scores(self, client: AsyncClient) -> None:
        body = (await client.post("/route", json={"prompt": "hi"})).json()
        first = body["candidates"][0]
        assert set(first) >= {"model_id", "estimated_cost", "score", "unmeasured"}
        assert first["unmeasured"] is True


class TestEvals:
    async def test_upload_run_and_history(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": EchoProvider("gemini-flash", latency_ms=100),
                "claude-opus": EchoProvider("claude-opus", latency_ms=800),
            }
        )
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        created = await http.post(
            "/evals",
            json={
                "name": "classification",
                "task_type": "classification",
                "grader": "exact_match",
                "examples": [
                    {"input": "q1||spam", "expected": "spam"},
                    {"input": "q2||ham", "expected": "ham"},
                ],
            },
        )
        assert created.status_code == 201
        assert created.json()["version"] == 1

        run = await http.post("/evals/run", json={"eval_set": "classification"})
        report = run.json()
        assert run.status_code == 200
        assert len(report["models"]) == 2
        assert all(m["accuracy"] == 1.0 for m in report["models"])

        history = (await http.get("/evals/history")).json()
        assert len(history) == 2

        # The run must feed routing immediately, not at the next restart.
        quality = (await http.get("/evals/quality")).json()
        assert quality["scores"]["classification"]["gemini-flash"] == 1.0

        await http.aclose()
        await engine.dispose()

    async def test_run_recommends_a_cheaper_model(self) -> None:
        registry = ProviderRegistry(
            {
                "gemini-flash": EchoProvider("gemini-flash"),
                "claude-opus": EchoProvider("claude-opus"),
            }
        )
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=hash_key(TEST_KEY),
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)
        http.headers["Authorization"] = f"Bearer {TEST_KEY}"

        await http.post(
            "/evals",
            json={
                "name": "s",
                "examples": [{"input": f"q{i}||a{i}", "expected": f"a{i}"} for i in range(4)],
            },
        )
        report = (await http.post("/evals/run", json={"eval_set": "s"})).json()

        assert report["recommendation"] is not None
        assert report["recommendation"]["to"] == "gemini-flash"
        assert report["recommendation"]["cost_reduction_pct"] > 90

        await http.aclose()
        await engine.dispose()

    async def test_unknown_grader_rejected_at_upload(self, client: AsyncClient) -> None:
        """Better than after paying for 300 model calls."""
        response = await client.post(
            "/evals",
            json={"name": "s", "grader": "vibes", "examples": [{"input": "q", "expected": "a"}]},
        )
        assert response.status_code == 422
        assert "Unknown grader" in response.json()["detail"]

    async def test_running_an_unknown_set_is_404_listing_what_exists(
        self, client: AsyncClient
    ) -> None:
        response = await client.post("/evals/run", json={"eval_set": "never-uploaded"})
        assert response.status_code == 404
        assert "Available" in response.json()["detail"]

    async def test_empty_example_list_rejected(self, client: AsyncClient) -> None:
        response = await client.post("/evals", json={"name": "s", "examples": []})
        assert response.status_code == 422

    async def test_graders_are_discoverable(self, client: AsyncClient) -> None:
        graders = (await client.get("/evals/graders")).json()
        assert "exact_match" in graders and "json_schema" in graders


class TestOps:
    async def test_health_reports_configuration_not_secrets(self, client: AsyncClient) -> None:
        body = (await client.get("/health")).json()

        assert body["status"] in ("ok", "degraded")
        assert body["database"] is True
        assert body["redis"] is False
        assert sorted(body["models"]) == ["claude-opus", "gemini-flash", "gpt-4o-mini"]
        assert body["pricing_age_days"] >= 0
        assert "key" not in str(body).lower() or "api_key" not in str(body).lower()

    async def test_models_lists_pricing_cheapest_first(self, client: AsyncClient) -> None:
        body = (await client.get("/models")).json()
        ids = [m["model_id"] for m in body["available"]]
        assert ids[0] == "gemini-flash"
        assert body["available"][0]["input_per_mtok"] > 0
        assert "catalog" in body

    async def test_metrics_includes_percentiles(self, client: AsyncClient) -> None:
        await client.post("/complete", json={"prompt": "hi"})
        body = (await client.get("/metrics", params={"hours": 1})).json()
        assert set(body["latency"]) == {"p50", "p95", "p99", "samples"}

    async def test_metrics_are_scoped_to_the_calling_key(self, registry: ProviderRegistry) -> None:
        """One tenant must not read another's spend, volume or model mix.

        Without scoping, every key on a deployment sees every other key's
        numbers — harmless for one team, a cross-tenant data leak for anything
        hosted.
        """
        tenant_a, tenant_b = "mo_tenant_a", "mo_tenant_b"
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=f"{hash_key(tenant_a)},{hash_key(tenant_b)}",
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)

        for _ in range(3):
            await http.post(
                "/complete",
                json={"prompt": "hi"},
                headers={"Authorization": f"Bearer {tenant_a}"},
            )
        await http.post(
            "/complete",
            json={"prompt": "hi", "model_id": "claude-opus"},
            headers={"Authorization": f"Bearer {tenant_b}"},
        )

        seen_by_a = (
            await http.get(
                "/metrics",
                params={"hours": 1},
                headers={"Authorization": f"Bearer {tenant_a}"},
            )
        ).json()
        seen_by_b = (
            await http.get(
                "/metrics",
                params={"hours": 1},
                headers={"Authorization": f"Bearer {tenant_b}"},
            )
        ).json()

        assert seen_by_a["requests"] == 3
        assert seen_by_b["requests"] == 1
        assert [m["model_id"] for m in seen_by_b["by_model"]] == ["claude-opus"]
        # B's expensive call must not appear in A's spend.
        assert seen_by_a["total_cost_usd"] < seen_by_b["total_cost_usd"]

        await http.aclose()
        await engine.dispose()

    async def test_deployment_scope_shows_everything(self, registry: ProviderRegistry) -> None:
        """A single team sharing staging/prod/CI keys wants one picture, and
        opts in explicitly."""
        key_a, key_b = "mo_a", "mo_b"
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=f"{hash_key(key_a)},{hash_key(key_b)}",
            metrics_scope="deployment",
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)

        await http.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": f"Bearer {key_a}"}
        )
        await http.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": f"Bearer {key_b}"}
        )

        body = (
            await http.get(
                "/metrics", params={"hours": 1}, headers={"Authorization": f"Bearer {key_a}"}
            )
        ).json()
        assert body["requests"] == 2
        assert body["scope"] == "deployment"

        await http.aclose()
        await engine.dispose()

    async def test_timeseries_is_scoped_too(self, registry: ProviderRegistry) -> None:
        key_a, key_b = "mo_a", "mo_b"
        settings = Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            api_key_hashes=f"{hash_key(key_a)},{hash_key(key_b)}",
            log_format="console",
        )
        http, engine = await _build_client(settings, registry)

        await http.post(
            "/complete", json={"prompt": "hi"}, headers={"Authorization": f"Bearer {key_a}"}
        )
        series = (
            await http.get(
                "/metrics/timeseries",
                params={"hours": 1},
                headers={"Authorization": f"Bearer {key_b}"},
            )
        ).json()
        assert sum(bucket["requests"] for bucket in series) == 0

        await http.aclose()
        await engine.dispose()

    async def test_timeseries(self, client: AsyncClient) -> None:
        await client.post("/complete", json={"prompt": "hi"})
        series = (await client.get("/metrics/timeseries", params={"hours": 1})).json()
        assert series and series[0]["requests"] >= 1

    async def test_alerts_endpoints(self, client: AsyncClient) -> None:
        assert (await client.get("/alerts")).json() == []
        checked = (await client.post("/alerts/check")).json()
        assert checked["fired"] == 0

    async def test_acknowledging_an_unknown_alert_is_404(self, client: AsyncClient) -> None:
        assert (await client.post("/alerts/nope/acknowledge")).status_code == 404

    async def test_root_points_at_health(self, client: AsyncClient) -> None:
        assert (await client.get("/")).json()["health"] == "/health"

    async def test_docs_are_disabled_in_production(self) -> None:
        """The OpenAPI schema enumerates every endpoint."""
        app = create_app(
            Settings(
                environment="production",
                api_key_hashes=hash_key("k"),
                database_url="postgresql+asyncpg://u:p@db/x",
                cors_origins="https://app.example.com",
            )
        )
        assert app.docs_url is None
        assert app.openapi_url is None

    async def test_docs_are_available_in_development(self, client: AsyncClient) -> None:
        assert (await client.get("/docs")).status_code == 200
