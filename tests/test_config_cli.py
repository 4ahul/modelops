"""Configuration, logging redaction and the CLI."""

from __future__ import annotations

import json

import pytest

from app.cli import main
from app.core.config import Settings, get_settings, reset_settings_cache
from app.core.logging import _redact, configure_logging, reset_logging


class TestSettings:
    def test_defaults_are_development_safe(self) -> None:
        settings = Settings()
        assert settings.environment == "development"
        assert settings.store_prompts is False, "prompt storage must be opt-in"
        assert settings.api_key_hash_set == frozenset()

    def test_log_level_is_validated_and_upper_cased(self) -> None:
        assert Settings(log_level="debug").log_level == "DEBUG"
        with pytest.raises(ValueError, match="log_level"):
            Settings(log_level="chatty")

    def test_key_hashes_are_parsed_from_a_list(self) -> None:
        settings = Settings(api_key_hashes=" abc , def ,, ghi ")
        assert settings.api_key_hash_set == frozenset({"abc", "def", "ghi"})

    def test_cors_origins_are_parsed(self) -> None:
        settings = Settings(cors_origins="https://a.com, https://b.com")
        assert settings.cors_origin_list == ["https://a.com", "https://b.com"]

    def test_configured_providers_reports_only_present_keys(self) -> None:
        settings = Settings(anthropic_api_key="sk-ant", openai_api_key=None)
        assert list(settings.configured_providers()) == ["anthropic"]

    def test_development_has_no_production_requirements(self) -> None:
        assert Settings(environment="development").validate_for_production() == []

    def test_production_rejects_an_unsafe_configuration(self) -> None:
        problems = Settings(
            environment="production",
            api_key_hashes="",
            cors_origins="*",
            database_url="postgresql+asyncpg://u:p@localhost/x",
        ).validate_for_production()

        joined = " ".join(problems)
        assert "API_KEY_HASHES" in joined
        assert "CORS_ORIGINS" in joined
        assert "localhost" in joined
        assert "provider API key" in joined

    def test_production_accepts_a_complete_configuration(self) -> None:
        assert (
            Settings(
                environment="production",
                api_key_hashes="a" * 64,
                anthropic_api_key="sk-ant",
                cors_origins="https://app.example.com",
                database_url="postgresql+asyncpg://u:p@db.internal:5432/modelops",
            ).validate_for_production()
            == []
        )

    def test_settings_are_cached_and_resettable(self) -> None:
        reset_settings_cache()
        assert get_settings() is get_settings()
        reset_settings_cache()


class TestLogRedaction:
    @pytest.mark.parametrize(
        "key", ["prompt", "api_key", "authorization", "password", "secret", "token", "content"]
    )
    def test_sensitive_keys_are_masked(self, key: str) -> None:
        """Enforced by a processor rather than by discipline at hundreds of call
        sites, because one of those eventually logs a prompt."""
        assert _redact(None, "", {key: "sensitive value"})[key] == "[redacted]"

    def test_case_is_ignored(self) -> None:
        assert _redact(None, "", {"API_KEY": "x"})["API_KEY"] == "[redacted]"

    def test_safe_fields_pass_through(self) -> None:
        event = _redact(None, "", {"model_id": "gemini-flash", "cost_usd": 0.001})
        assert event["model_id"] == "gemini-flash"
        assert event["cost_usd"] == 0.001

    def test_none_is_left_alone(self) -> None:
        """Masking a missing value would imply one was present."""
        assert _redact(None, "", {"prompt": None})["prompt"] is None

    def test_configure_is_idempotent(self) -> None:
        reset_logging()
        configure_logging("INFO", "console")
        configure_logging("DEBUG", "json")  # must not raise or double-handle
        reset_logging()


class TestCLI:
    def test_hash_key_prints_a_sha256(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["hash-key", "my-secret-key"]) == 0
        digest = capsys.readouterr().out.strip()
        assert len(digest) == 64
        assert "my-secret-key" not in digest

    def test_pricing_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["pricing"]) == 0
        output = capsys.readouterr().out
        assert "gemini-flash" in output
        assert "Pricing verified" in output

    def test_pricing_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["pricing", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["age_days"] >= 0
        assert "claude-opus" in data["models"]

    def test_check_config_passes_in_development(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["check-config"]) == 0
        assert "Configuration OK" in capsys.readouterr().out

    def test_check_config_fails_for_production(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("API_KEY_HASHES", raising=False)
        assert main(["check-config", "--production"]) == 1
        assert "Problems:" in capsys.readouterr().out

    def test_check_config_redacts_credentials(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connection string printed in full ends up in a terminal scrollback,
        a CI log, and a screenshot."""
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+asyncpg://admin:hunter2@db.internal:5432/modelops"
        )
        main(["check-config"])
        output = capsys.readouterr().out
        assert "hunter2" not in output
        assert "***@db.internal" in output

    def test_eval_without_providers_exits_nonzero(
        self, tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        path = tmp_path / "evals.jsonl"
        path.write_text(json.dumps({"input": "q", "expected": "a"}) + "\n", encoding="utf-8")

        assert main(["eval", str(path)]) == 1
        assert "No provider API keys" in capsys.readouterr().err

    def test_unknown_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            main(["nonsense"])

    def test_version_flag(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
