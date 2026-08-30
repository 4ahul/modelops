"""Application startup.

The lifespan is the one code path that decides whether a misconfigured
deployment serves traffic or refuses to start, so it is tested directly rather
than bypassed the way the request tests do.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.db.session import create_all, dispose_engine
from app.main import create_app, lifespan


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "development",
        "database_url": "sqlite+aiosqlite:///./_startup_test.db",
        # Port 1 is closed, so the Redis path exercises its fallback rather than
        # depending on a server being up.
        "redis_url": "redis://127.0.0.1:1/0",
        "log_format": "console",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
async def _clean_engine() -> None:
    yield
    await dispose_engine()
    import pathlib

    pathlib.Path("_startup_test.db").unlink(missing_ok=True)


class TestLifespan:
    async def test_starts_with_no_redis_and_no_providers(self) -> None:
        """Neither is fatal in development: Redis degrades to per-process state,
        and a deployment with no vendor key should still boot so its /health can
        say why it is useless."""
        app = create_app(_settings())

        async with lifespan(app):
            assert app.state.redis is None
            assert len(app.state.registry) == 0
            assert app.state.rate_limiter is not None
            assert app.state.router is not None

    async def test_health_reports_degraded_rather_than_failing(self) -> None:
        """A rolling deploy needs to tell "starting" from "broken"; a crash on
        startup tells it neither."""
        app = create_app(_settings())

        async with lifespan(app):
            await create_all()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                body = (await client.get("/health")).json()

        assert body["status"] == "degraded"
        assert body["database"] is True
        assert body["redis"] is False
        assert body["models"] == []
        assert any("provider API key" in w for w in body["warnings"])

    async def test_health_distinguishes_a_missing_schema_from_a_dead_database(
        self,
    ) -> None:
        """A deploy that skipped its migration would otherwise report a healthy
        database and then fail every write."""
        app = create_app(_settings())

        async with lifespan(app):  # tables deliberately not created
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                body = (await client.get("/health")).json()

        assert body["database"] is False
        assert any("alembic upgrade head" in w for w in body["warnings"])

    async def test_providers_are_built_from_configured_keys(self) -> None:
        app = create_app(_settings(anthropic_api_key="sk-ant-test"))

        async with lifespan(app):
            assert sorted(app.state.registry.model_ids) == [
                "claude-haiku",
                "claude-opus",
                "claude-sonnet",
            ]

    async def test_production_refuses_an_unsafe_configuration(self) -> None:
        """The reason this is a hard failure: a service that silently serves
        unauthenticated traffic is worse than one that does not start."""
        app = create_app(
            _settings(
                environment="production",
                api_key_hashes="",
                database_url="postgresql+asyncpg://u:p@db.internal:5432/modelops",
            )
        )

        with pytest.raises(RuntimeError, match="unsafe configuration"):
            async with lifespan(app):
                pass  # pragma: no cover - startup must not get this far

    async def test_production_starts_when_correctly_configured(self) -> None:
        app = create_app(
            _settings(
                environment="production",
                api_key_hashes="a" * 64,
                anthropic_api_key="sk-ant-test",
                cors_origins="https://app.example.com",
            )
        )

        async with lifespan(app):
            assert len(app.state.registry) == 3

    async def test_quality_scores_load_from_the_database(self) -> None:
        """So the first request routes on evidence rather than the unmeasured
        penalty."""
        from app.db.crud import record_eval_report, upsert_eval_set
        from app.db.session import get_sessionmaker
        from app.eval.dataset import EvalExample, EvalSet
        from app.eval.runner import EvalRunner
        from app.providers.registry import ProviderRegistry
        from tests.conftest import EchoProvider

        app = create_app(_settings())
        eval_set = EvalSet(
            name="startup",
            task_type="classification",
            examples=[EvalExample(input="q||a", expected="a")],
        )

        async with lifespan(app):
            await create_all()
            async with get_sessionmaker()() as db:
                row = await upsert_eval_set(db, eval_set)
                registry = ProviderRegistry({"gemini-flash": EchoProvider("gemini-flash")})
                report = await EvalRunner(registry).run(eval_set)
                await record_eval_report(db, row, report)

        # A fresh app must pick the stored score up at startup.
        second = create_app(_settings())
        async with lifespan(second):
            scores = second.state.router.quality_scores

        assert scores["classification"]["gemini-flash"] == 1.0

    async def test_cors_is_only_added_when_origins_are_configured(self) -> None:
        without = create_app(_settings())
        with_origins = create_app(_settings(cors_origins="https://app.example.com"))

        def has_cors(app: object) -> bool:
            return any(
                "CORSMiddleware" in str(m.cls) for m in app.user_middleware  # type: ignore[attr-defined]
            )

        assert not has_cors(without)
        assert has_cors(with_origins)

    async def test_unhandled_errors_do_not_leak_internals(self) -> None:
        """A stack trace in a response body leaks table names, file paths and
        occasionally a connection string."""
        app = create_app(_settings())

        @app.get("/_boom")
        async def boom() -> None:
            raise RuntimeError("connection to postgres://user:hunter2@db failed")

        async with lifespan(app):
            async with AsyncClient(
                # raise_app_exceptions=False so the handler's response comes
                # back instead of httpx re-raising the server-side exception.
                transport=ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.get("/_boom")

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error", "kind": "internal"}
        assert "hunter2" not in response.text
