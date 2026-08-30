"""Shared fixtures.

Two decisions keep this suite fast and honest:

**No network.** Providers are fakes with scripted behaviour, so latency, cost and
failure are all controllable and the suite never touches a vendor.

**SQLite, same models.** The database tests run against the real SQLAlchemy
models on aiosqlite rather than a parallel schema, so a model change cannot pass
the tests and break Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.core.health import HealthTracker
from app.core.policy import RoutingPolicy, TaskPolicy
from app.core.router import Router
from app.db.models import Base
from app.eval.dataset import EvalExample, EvalSet
from app.providers.base import (
    CompletionResult,
    ModelProvider,
    ProviderUnavailable,
)
from app.providers.pricing import get_spec
from app.providers.registry import ProviderRegistry


class FakeProvider(ModelProvider):
    """A provider with scripted latency, output and failures.

    Args:
        model_id: Which catalog entry to impersonate, so pricing is real.
        reply: What to return.
        latency_ms: Reported latency.
        fail_times: Fail this many calls before succeeding — for exercising
            retry and fallback.
        error: The error to raise while failing.
        output_tokens: Reported output usage, which drives the cost.
    """

    def __init__(
        self,
        model_id: str,
        *,
        reply: str = "ok",
        latency_ms: float = 100.0,
        fail_times: int = 0,
        error: Exception | None = None,
        output_tokens: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(get_spec(model_id), "test-key", max_retries=kwargs.pop("max_retries", 0))
        self.reply = reply
        self.latency_ms = latency_ms
        self.fail_times = fail_times
        self.error = error or ProviderUnavailable(f"{model_id} is down")
        self.output_tokens = output_tokens
        self.calls: list[str] = []

    async def _complete(
        self, prompt: str, *, max_tokens: int, temperature: float, system: str | None
    ) -> CompletionResult:
        self.calls.append(prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return self.build_result(
            content=self.reply,
            input_tokens=self.count_tokens(prompt),
            output_tokens=self.output_tokens,
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )


class EchoProvider(FakeProvider):
    """Returns the prompt's expected answer, for eval tests.

    The prompt is ``"question||answer"``; everything after ``||`` is echoed. That
    makes a grader's behaviour testable without a model.
    """

    async def _complete(
        self, prompt: str, *, max_tokens: int, temperature: float, system: str | None
    ) -> CompletionResult:
        self.calls.append(prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        answer = prompt.split("||", 1)[1] if "||" in prompt else prompt
        return self.build_result(
            content=answer,
            input_tokens=self.count_tokens(prompt),
            output_tokens=self.output_tokens,
            latency_ms=self.latency_ms,
        )


@pytest.fixture
def settings() -> Settings:
    """Settings for a test run: no real keys, SQLite, no Redis."""
    return Settings(
        environment="development",
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        api_key_hashes="",
        log_format="console",
    )


@pytest.fixture
def health() -> HealthTracker:
    return HealthTracker(None, failure_threshold=2, unhealthy_ttl=60)


@pytest.fixture
def registry() -> ProviderRegistry:
    """Three models spanning two orders of magnitude in price."""
    return ProviderRegistry(
        {
            "gemini-flash": FakeProvider("gemini-flash", reply="cheap", latency_ms=150),
            "gpt-4o-mini": FakeProvider("gpt-4o-mini", reply="mid", latency_ms=300),
            "claude-opus": FakeProvider("claude-opus", reply="expensive", latency_ms=900),
        }
    )


@pytest.fixture
def policy() -> RoutingPolicy:
    return RoutingPolicy(
        tasks={
            "classification": TaskPolicy(cost_limit=0.01, latency_budget_ms=1000, min_quality=0.90),
            "reasoning": TaskPolicy(cost_limit=1.0, latency_budget_ms=30_000),
        }
    )


@pytest.fixture
def router(registry: ProviderRegistry, policy: RoutingPolicy, health: HealthTracker) -> Router:
    return Router(registry, policy, health=health)


@pytest.fixture
def eval_set() -> EvalSet:
    """Four examples whose expected answer is embedded in the prompt."""
    return EvalSet(
        name="test_classification",
        task_type="classification",
        grader="exact_match",
        examples=[
            EvalExample(input="is this spam?||spam", expected="spam", id="e1"),
            EvalExample(input="is this spam?||ham", expected="ham", id="e2"),
            EvalExample(input="urgent?||yes", expected="yes", id="e3"),
            EvalExample(input="urgent?||no", expected="no", id="e4"),
        ],
    )


@pytest.fixture
def echo_registry() -> ProviderRegistry:
    return ProviderRegistry(
        {
            "gemini-flash": EchoProvider("gemini-flash", latency_ms=120),
            "claude-opus": EchoProvider("claude-opus", latency_ms=800),
        }
    )


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A real database on SQLite, created and dropped per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        yield db
    await engine.dispose()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


__all__ = ["EchoProvider", "FakeProvider"]
