"""Vendor adapters, exercised against fake SDKs.

The three adapters are the part of the system that touches money: they translate
a vendor's response into the token counts every cost figure derives from. Left
untested they are the highest-risk code here, because a mistranslation does not
crash â€” it silently reports the wrong cost forever.

Rather than install three vendor SDKs and hit the network, each SDK is replaced
in ``sys.modules`` with a fake that reproduces its real response shape. That
exercises the translation logic, the usage extraction and the error mapping,
which is where the bugs actually live.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.providers.base import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderUnavailable,
)
from app.providers.pricing import get_spec


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Block:
    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        if text is not None:
            self.text = text


class _VendorError(Exception):
    def __init__(self, message: str, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = types.SimpleNamespace(
            status_code=status_code, headers={"retry-after": retry_after} if retry_after else {}
        )


# --------------------------------------------------------------------- Anthropic


def _install_anthropic(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> None:
    class AsyncAnthropic:
        def __init__(self, api_key: str, max_retries: int = 0, **kwargs: Any) -> None:
            self.api_key = api_key
            self.max_retries = max_retries
            self.messages = types.SimpleNamespace(create=behaviour)

    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = AsyncAnthropic  # type: ignore[attr-defined]
    module.BadRequestError = type("BadRequestError", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


class TestAnthropicAdapter:
    async def test_cost_comes_from_reported_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content=[_Block("text", "hello")],
                usage=_Usage(1_000_000, 1_000_000),
                stop_reason="end_turn",
            )

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        spec = get_spec("claude-sonnet")
        result = await AnthropicProvider(spec, "sk-test").complete("hi")

        # $3/Mtok in + $15/Mtok out on 1M each.
        assert result.cost_usd == pytest.approx(18.00)
        assert result.content == "hello"
        assert result.finish_reason == "end_turn"
        assert result.provider == "anthropic"

    async def test_multiple_text_blocks_are_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content=[_Block("text", "part one "), _Block("text", "part two")],
                usage=_Usage(10, 5),
            )

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        result = await AnthropicProvider(get_spec("claude-haiku"), "sk").complete("hi")
        assert result.content == "part one part two"

    async def test_non_text_blocks_do_not_crash_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``content[0].text`` breaks the first time a model emits a tool-use or
        thinking block. Filtering by type is the fix."""

        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content=[_Block("thinking"), _Block("text", "the answer")],
                usage=_Usage(10, 5),
            )

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        result = await AnthropicProvider(get_spec("claude-haiku"), "sk").complete("hi")
        assert result.content == "the answer"

    async def test_empty_content_is_an_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(content=[], usage=_Usage(10, 0))

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        assert (
            await AnthropicProvider(get_spec("claude-haiku"), "sk").complete("hi")
        ).content == ""

    @pytest.mark.parametrize(
        "status,expected",
        [
            (429, ProviderRateLimited),
            (401, ProviderAuthError),
            (400, ProviderBadRequest),
            (503, ProviderUnavailable),
        ],
    )
    async def test_errors_are_translated(
        self, monkeypatch: pytest.MonkeyPatch, status: int, expected: type
    ) -> None:
        async def create(**kwargs: Any) -> Any:
            raise _VendorError("vendor said no", status)

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(get_spec("claude-haiku"), "sk", max_retries=0)
        with pytest.raises(expected):
            await provider.complete("hi")

    async def test_retry_after_header_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            raise _VendorError("slow down", 429, retry_after="7")

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(get_spec("claude-haiku"), "sk", max_retries=0)
        with pytest.raises(ProviderRateLimited) as exc:
            await provider.complete("hi")
        assert exc.value.retry_after == 7.0

    async def test_system_prompt_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def create(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return types.SimpleNamespace(content=[_Block("text", "ok")], usage=_Usage(1, 1))

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        await AnthropicProvider(get_spec("claude-haiku"), "sk").complete("hi", system="be terse")
        assert seen["system"] == "be terse"
        assert seen["messages"] == [{"role": "user", "content": "hi"}]

    async def test_vendor_retries_are_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retries are handled once, uniformly, one level up. Leaving the SDK's
        own loop on would multiply them."""

        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(content=[_Block("text", "ok")], usage=_Usage(1, 1))

        _install_anthropic(monkeypatch, create)
        from app.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(get_spec("claude-haiku"), "sk")
        await provider.complete("hi")
        assert provider._get_client().max_retries == 0

    async def test_missing_sdk_is_an_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.providers.base import ProviderNotInstalled

        monkeypatch.setitem(sys.modules, "anthropic", None)
        from app.providers.anthropic_ import AnthropicProvider

        with pytest.raises(ProviderNotInstalled, match="pip install"):
            AnthropicProvider(get_spec("claude-haiku"), "sk")._get_client()


# ------------------------------------------------------------------------ OpenAI


def _install_openai(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> None:
    class AsyncOpenAI:
        def __init__(self, api_key: str, max_retries: int = 0, **kwargs: Any) -> None:
            self.max_retries = max_retries
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=behaviour))

    module = types.ModuleType("openai")
    module.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _choice(content: str | None, finish_reason: str = "stop") -> Any:
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=content), finish_reason=finish_reason
    )


class TestOpenAIAdapter:
    async def test_cost_comes_from_reported_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[_choice("hello")],
                usage=types.SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
            )

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        result = await OpenAIProvider(get_spec("gpt-4o"), "sk").complete("hi")

        # $2.50/Mtok in + $10/Mtok out.
        assert result.cost_usd == pytest.approx(12.50)
        assert result.content == "hello"

    async def test_missing_usage_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without usage there is no honest cost. A fabricated zero would
        silently understate spend on every affected call."""

        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(choices=[_choice("hello")], usage=None)

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        provider = OpenAIProvider(get_spec("gpt-4o-mini"), "sk", max_retries=0)
        with pytest.raises(ProviderUnavailable, match="usage"):
            await provider.complete("hi")

    async def test_null_content_becomes_an_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A filtered or tool-only reply has ``content=None``; concatenating that
        would raise."""

        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[_choice(None, "content_filter")],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=0),
            )

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        result = await OpenAIProvider(get_spec("gpt-4o-mini"), "sk").complete("hi")
        assert result.content == ""
        assert result.finish_reason == "content_filter"

    async def test_no_choices_is_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[], usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=0)
            )

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        assert (await OpenAIProvider(get_spec("gpt-4o-mini"), "sk").complete("hi")).content == ""

    async def test_system_message_is_prepended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def create(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return types.SimpleNamespace(
                choices=[_choice("ok")],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        await OpenAIProvider(get_spec("gpt-4o-mini"), "sk").complete("hi", system="be terse")
        assert seen["messages"][0] == {"role": "system", "content": "be terse"}
        assert seen["messages"][1] == {"role": "user", "content": "hi"}

    @pytest.mark.parametrize(
        "status,expected",
        [(429, ProviderRateLimited), (401, ProviderAuthError), (422, ProviderBadRequest)],
    )
    async def test_errors_are_translated(
        self, monkeypatch: pytest.MonkeyPatch, status: int, expected: type
    ) -> None:
        async def create(**kwargs: Any) -> Any:
            raise _VendorError("no", status)

        _install_openai(monkeypatch, create)
        from app.providers.openai_ import OpenAIProvider

        provider = OpenAIProvider(get_spec("gpt-4o-mini"), "sk", max_retries=0)
        with pytest.raises(expected):
            await provider.complete("hi")

    def test_token_count_falls_back_without_tiktoken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "tiktoken", None)
        from app.providers.openai_ import OpenAIProvider

        provider = OpenAIProvider(get_spec("gpt-4o-mini"), "sk")
        assert provider.count_tokens("hello world") > 0
        # The failure is recorded, so the import is not retried on every call.
        provider.count_tokens("again")
        assert provider.tokenizer_unavailable is True


# ------------------------------------------------------------------------ Gemini


def _install_gemini(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class GenerativeModel:
        def __init__(self, model: str, system_instruction: str | None = None) -> None:
            captured["model"] = model
            captured["system_instruction"] = system_instruction
            self.generate_content_async = behaviour

    module = types.ModuleType("google.generativeai")
    module.GenerativeModel = GenerativeModel  # type: ignore[attr-defined]
    module.configure = lambda **kwargs: captured.update(kwargs)  # type: ignore[attr-defined]

    google = types.ModuleType("google")
    google.generativeai = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", module)
    return captured


def _candidate(text: str, finish_reason: str = "STOP") -> Any:
    return types.SimpleNamespace(
        content=types.SimpleNamespace(parts=[types.SimpleNamespace(text=text)]),
        finish_reason=finish_reason,
    )


class TestGeminiAdapter:
    async def test_cost_comes_from_reported_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def generate(prompt: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                candidates=[_candidate("hello")],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=1_000_000, candidates_token_count=1_000_000
                ),
            )

        _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        result = await GeminiProvider(get_spec("gemini-flash"), "key").complete("hi")

        # $0.075/Mtok in + $0.30/Mtok out.
        assert result.cost_usd == pytest.approx(0.375)
        assert result.content == "hello"
        assert result.provider == "gemini"

    async def test_blocked_response_returns_empty_text_not_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``response.text`` raises when a safety filter left no candidate.
        Reading the parts directly gives the grader something to score instead of
        failing the whole eval run."""

        async def generate(prompt: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                candidates=[],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=0
                ),
            )

        _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        result = await GeminiProvider(get_spec("gemini-flash"), "key").complete("hi")
        assert result.content == ""
        assert result.finish_reason == "no_candidates"

    async def test_missing_usage_metadata_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def generate(prompt: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(candidates=[_candidate("hi")], usage_metadata=None)

        _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        provider = GeminiProvider(get_spec("gemini-flash"), "key", max_retries=0)
        with pytest.raises(ProviderUnavailable, match="usage"):
            await provider.complete("hi")

    async def test_system_instruction_is_passed_and_the_model_is_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def generate(prompt: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                candidates=[_candidate("ok")],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=1, candidates_token_count=1
                ),
            )

        captured = _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        provider = GeminiProvider(get_spec("gemini-flash"), "key")
        await provider.complete("hi", system="be terse")
        await provider.complete("again", system="be terse")

        assert captured["system_instruction"] == "be terse"
        assert captured["api_key"] == "key"
        # One cached model per distinct instruction, not one per call.
        assert list(provider._model_cache) == ["be terse"]

    async def test_generation_config_carries_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def generate(prompt: str, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return types.SimpleNamespace(
                candidates=[_candidate("ok")],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=1, candidates_token_count=1
                ),
            )

        _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        await GeminiProvider(get_spec("gemini-flash"), "key").complete(
            "hi", max_tokens=64, temperature=0.3
        )
        assert seen["generation_config"] == {"max_output_tokens": 64, "temperature": 0.3}

    @pytest.mark.parametrize(
        "error_name,expected",
        [
            ("InvalidArgument", ProviderBadRequest),
            ("PermissionDenied", ProviderBadRequest),
            ("InternalServerError", ProviderUnavailable),
        ],
    )
    async def test_named_errors_without_a_status_are_classified(
        self, monkeypatch: pytest.MonkeyPatch, error_name: str, expected: type
    ) -> None:
        """Google's client raises exceptions that carry no HTTP status, so the
        adapter has to fall back to the exception's own name."""

        async def generate(prompt: str, **kwargs: Any) -> Any:
            raise type(error_name, (Exception,), {})("vendor said no")

        _install_gemini(monkeypatch, generate)
        from app.providers.gemini_ import GeminiProvider

        provider = GeminiProvider(get_spec("gemini-flash"), "key", max_retries=0)
        with pytest.raises(expected):
            await provider.complete("hi")

    async def test_missing_sdk_is_an_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.providers.base import ProviderNotInstalled

        monkeypatch.setitem(sys.modules, "google.generativeai", None)
        from app.providers.gemini_ import GeminiProvider

        with pytest.raises(ProviderNotInstalled, match="pip install"):
            GeminiProvider(get_spec("gemini-flash"), "key")._get_model(None)


class TestCostAgreementAcrossVendors:
    async def test_the_same_token_split_costs_what_the_table_says_everywhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One prompt, three vendors, one arithmetic. If an adapter multiplies by
        the wrong field the whole product's numbers stop being trustworthy."""

        async def anthropic_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content=[_Block("text", "x")], usage=_Usage(500_000, 100_000)
            )

        async def openai_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[_choice("x")],
                usage=types.SimpleNamespace(prompt_tokens=500_000, completion_tokens=100_000),
            )

        async def gemini_generate(prompt: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                candidates=[_candidate("x")],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=500_000, candidates_token_count=100_000
                ),
            )

        _install_anthropic(monkeypatch, anthropic_create)
        _install_openai(monkeypatch, openai_create)
        _install_gemini(monkeypatch, gemini_generate)

        from app.providers.anthropic_ import AnthropicProvider
        from app.providers.gemini_ import GeminiProvider
        from app.providers.openai_ import OpenAIProvider

        for provider_cls, model_id in (
            (AnthropicProvider, "claude-sonnet"),
            (OpenAIProvider, "gpt-4o"),
            (GeminiProvider, "gemini-flash"),
        ):
            spec = get_spec(model_id)
            result = await provider_cls(spec, "key").complete("hi")  # type: ignore[abstract]
            assert result.cost_usd == pytest.approx(spec.pricing.cost(500_000, 100_000)), model_id
            assert result.input_tokens == 500_000
            assert result.output_tokens == 100_000
