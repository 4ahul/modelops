"""Runtime configuration.

Everything is read from the environment, so the same image runs in dev and
production with no code change. :func:`get_settings` is cached, so importing
this module has no cost and a test can override the cache in one place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    database_url: str = "postgresql+asyncpg://modelops:modelops_dev@localhost:5432/modelops_dev"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    #: Comma-separated SHA-256 hashes of valid API keys. Plaintext keys are
    #: never stored, not even in the environment of the process that checks
    #: them, so a leaked env dump does not hand over customer credentials.
    api_key_hashes: str = ""

    #: Requests per minute per API key. Enforced in Redis, so the limit holds
    #: across replicas rather than per process.
    rate_limit_per_minute: int = 120

    #: Who sees which numbers on ``/metrics``.
    #:
    #: ``"key"`` (default) scopes every read to the calling API key. ``"deployment"``
    #: shows the whole deployment to every key — correct for a single team sharing
    #: staging/prod/CI keys, and a cross-tenant data leak for anything hosted.
    #: Defaulting to the restrictive option means a multi-tenant deployment is
    #: safe before anyone thinks about it.
    metrics_scope: Literal["key", "deployment"] = "key"

    #: Seconds a provider stays marked unhealthy after a failure. Short by
    #: design: a recovered provider should come back without an operator.
    provider_unhealthy_ttl_seconds: int = 60

    #: Consecutive failures before a provider is taken out of rotation.
    provider_failure_threshold: int = 3

    #: Wall-clock ceiling for one provider call.
    request_timeout_seconds: float = 60.0

    #: Retries per provider before the router falls through to the next one.
    max_retries: int = 2

    #: Concurrent provider calls during an eval run. The bound exists because
    #: every vendor rate-limits, and an unbounded gather turns a 100-example
    #: eval into a wall of 429s.
    eval_concurrency: int = 8

    #: Off by default. Prompt bodies are the one thing customers cannot
    #: un-send; storing them has to be a deliberate, per-deployment decision.
    store_prompts: bool = False

    cors_origins: str = ""

    slack_webhook_url: str | None = None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @property
    def api_key_hash_set(self) -> frozenset[str]:
        """Valid key hashes, parsed once."""
        return frozenset(h.strip() for h in self.api_key_hashes.split(",") if h.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def configured_providers(self) -> dict[str, str]:
        """Which vendor keys are actually present.

        The API reports this at ``/health`` so a misconfigured deployment is
        visible immediately, rather than at the first request that happens to
        route to the provider whose key is missing.
        """
        return {
            name: key
            for name, key in (
                ("anthropic", self.anthropic_api_key),
                ("openai", self.openai_api_key),
                ("gemini", self.google_api_key),
            )
            if key
        }

    def validate_for_production(self) -> list[str]:
        """Return the reasons this configuration is unsafe to run in production.

        Called at startup. Failing loudly beats discovering in an incident that
        the deployment has been serving unauthenticated traffic.
        """
        problems: list[str] = []
        if not self.is_production:
            return problems
        if not self.api_key_hash_set:
            problems.append(
                "API_KEY_HASHES is empty — every endpoint would be open. "
                "Generate one with: python -m app.cli hash-key <key>"
            )
        if not self.configured_providers():
            problems.append("No provider API key is set; every completion would fail")
        if "*" in self.cors_origin_list:
            problems.append("CORS_ORIGINS='*' in production; list the real origins")
        if "localhost" in self.database_url:
            problems.append("DATABASE_URL still points at localhost")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings object."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cache. For tests that patch the environment."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
