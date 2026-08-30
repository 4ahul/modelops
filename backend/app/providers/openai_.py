"""OpenAI adapter."""

from __future__ import annotations

import time
from typing import Any

from app.providers.base import (
    CompletionResult,
    ModelProvider,
    ProviderError,
    ProviderNotInstalled,
    ProviderUnavailable,
    classify_status,
    status_of,
)
from app.providers.pricing import ModelSpec


class OpenAIProvider(ModelProvider):
    """GPT models via the official ``openai`` SDK."""

    def __init__(self, spec: ModelSpec, api_key: str, **kwargs: Any) -> None:
        super().__init__(spec, api_key, **kwargs)
        self._client: Any | None = None
        self._encoder: Any | None = None
        #: Set once when tiktoken is absent, so the import is not retried on
        #: every call. A separate flag rather than a falsy sentinel in
        #: ``_encoder``, which would conflate "not tried" with "unavailable".
        self.tokenizer_unavailable = False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ProviderNotInstalled("OpenAI", "openai") from None
            self._client = AsyncOpenAI(api_key=self.api_key, max_retries=0)
        return self._client

    def count_tokens(self, text: str) -> int:
        """Exact count via ``tiktoken`` when it is installed.

        Falls back to the character heuristic otherwise, because an exact count
        is a nice-to-have for a pre-call estimate and not worth a hard
        dependency.
        """
        if self._encoder is None and not self.tokenizer_unavailable:
            try:
                import tiktoken

                self._encoder = tiktoken.encoding_for_model(self.spec.model)
            except Exception:
                self.tokenizer_unavailable = True
        if self._encoder is not None:
            return int(len(self._encoder.encode(text)))
        return super().count_tokens(text)

    async def _complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> CompletionResult:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.spec.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise _translate(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choice = response.choices[0] if response.choices else None
        usage = response.usage
        if usage is None:
            # Without usage there is no honest cost, and a fabricated zero would
            # silently understate spend on every affected call.
            raise ProviderUnavailable("openai returned no usage data; cannot compute cost")

        return self.build_result(
            content=(getattr(choice.message, "content", None) or "") if choice else "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        )


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    status = status_of(exc)
    message = f"openai: {type(exc).__name__}: {exc}"
    if status is not None:
        return classify_status(status, message, _retry_after(exc))
    return ProviderUnavailable(message)


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


__all__ = ["OpenAIProvider"]
