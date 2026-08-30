"""Google Gemini adapter."""

from __future__ import annotations

import time
from typing import Any

from app.providers.base import (
    CompletionResult,
    ModelProvider,
    ProviderBadRequest,
    ProviderError,
    ProviderNotInstalled,
    ProviderUnavailable,
    classify_status,
    status_of,
)
from app.providers.pricing import ModelSpec


class GeminiProvider(ModelProvider):
    """Gemini models via ``google-generativeai``."""

    def __init__(self, spec: ModelSpec, api_key: str, **kwargs: Any) -> None:
        super().__init__(spec, api_key, **kwargs)
        self._model_cache: dict[str | None, Any] = {}

    def _get_model(self, system: str | None) -> Any:
        # The SDK bakes the system instruction into the model object, so one is
        # cached per distinct instruction rather than rebuilt on every call.
        if system not in self._model_cache:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ProviderNotInstalled("Gemini", "gemini") from None
            genai.configure(api_key=self.api_key)
            self._model_cache[system] = genai.GenerativeModel(
                self.spec.model, system_instruction=system
            )
        return self._model_cache[system]

    async def _complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> CompletionResult:
        model = self._get_model(system)

        started = time.perf_counter()
        try:
            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
        except Exception as exc:
            raise _translate(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            raise ProviderUnavailable("gemini returned no usage metadata; cannot compute cost")

        return self.build_result(
            content=_text_of(response),
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            latency_ms=latency_ms,
            finish_reason=_finish_reason(response),
        )


def _text_of(response: Any) -> str:
    """Extract text without tripping over a blocked response.

    ``response.text`` raises when the model returned no candidate — a safety
    block, or a stop before any token. Reading the parts directly turns that
    into an empty string, which the grader can score, instead of an exception
    that fails the whole eval run.
    """
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(getattr(part, "text", "") or "" for part in parts)
        if text:
            return text
    return ""


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "no_candidates"
    reason = getattr(candidates[0], "finish_reason", None)
    return str(reason) if reason is not None else None


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    status = status_of(exc)
    message = f"gemini: {type(exc).__name__}: {exc}"
    if status is not None:
        return classify_status(status, message)
    name = type(exc).__name__
    if name in ("InvalidArgument", "FailedPrecondition", "PermissionDenied"):
        return ProviderBadRequest(message)
    return ProviderUnavailable(message)


__all__ = ["GeminiProvider"]
