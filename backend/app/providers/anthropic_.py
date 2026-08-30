"""Anthropic adapter."""

from __future__ import annotations

import time
from typing import Any

from app.providers.base import (
    CompletionResult,
    ModelProvider,
    ProviderBadRequest,
    ProviderError,
    ProviderNotInstalled,
    classify_status,
    status_of,
)
from app.providers.pricing import ModelSpec


class AnthropicProvider(ModelProvider):
    """Claude models via the official ``anthropic`` SDK."""

    def __init__(self, spec: ModelSpec, api_key: str, **kwargs: Any) -> None:
        super().__init__(spec, api_key, **kwargs)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ProviderNotInstalled("Anthropic", "anthropic") from None
            # Retries are handled one level up, uniformly across providers, so
            # the SDK's own retry loop is disabled to avoid multiplying them.
            self._client = AsyncAnthropic(api_key=self.api_key, max_retries=0)
        return self._client

    async def _complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> CompletionResult:
        client = self._get_client()
        request: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system

        started = time.perf_counter()
        try:
            response = await client.messages.create(**request)
        except Exception as exc:
            raise _translate(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        return self.build_result(
            content=_text_of(response),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "stop_reason", None),
        )


def _text_of(response: Any) -> str:
    """Concatenate the text blocks of a message.

    A response may contain several blocks, and non-text blocks (tool use,
    thinking) have no ``.text`` — indexing ``content[0].text`` blindly is how
    this breaks in production the first time a model emits one.
    """
    parts = [
        block.text
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "".join(parts)


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    status = status_of(exc)
    message = f"anthropic: {type(exc).__name__}: {exc}"
    if status is not None:
        return classify_status(status, message, _retry_after(exc))
    # An unmapped SDK error with no status is treated as a client error only
    # when the SDK says so; otherwise the caller gets a failover-able error.
    if type(exc).__name__ in ("BadRequestError", "UnprocessableEntityError"):
        return ProviderBadRequest(message)
    from app.providers.base import ProviderUnavailable

    return ProviderUnavailable(message)


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


__all__ = ["AnthropicProvider"]
