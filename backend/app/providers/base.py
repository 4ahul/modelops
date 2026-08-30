"""Provider abstraction.

Every model vendor is adapted to one interface so the router can compare them on
the same three axes: cost, latency and measured quality.

Two decisions in here matter more than the rest:

**Errors are classified, not just raised.** The router's fallback chain needs to
know the difference between "this provider is broken, try another" and "this
request is malformed, trying another provider will fail identically". Retrying a
400 across three vendors turns one client error into three bills.

**Cost is computed from reported usage, never estimated.** Estimation is only
used to exclude a provider *before* calling it. Once a call returns, the vendor's
own token counts are authoritative — an estimate that drifts from the invoice
makes the whole product untrustworthy.
"""

from __future__ import annotations

import abc
import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.providers.pricing import ModelSpec

log = get_logger(__name__)

#: Characters per token, used only when a vendor offers no local tokenizer.
#: Deliberately conservative: under-counting tokens would let a request slip
#: past a cost ceiling it should have been excluded by.
_CHARS_PER_TOKEN = 3.5


class ProviderError(Exception):
    """Base class for provider failures."""

    #: Whether trying the same request again, here or elsewhere, could succeed.
    retryable: bool = False
    #: Whether the router should fall through to a different provider.
    failover: bool = False


class ProviderUnavailable(ProviderError):
    """Transport failure, 5xx, or timeout. Another provider may well work."""

    retryable = True
    failover = True


class ProviderRateLimited(ProviderError):
    """429. Retryable here after a delay, and a good reason to fail over now."""

    retryable = True
    failover = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderBadRequest(ProviderError):
    """4xx that is the caller's fault. Never retried, never failed over.

    A prompt over the context window, a malformed parameter, or a content
    filter rejection will fail the same way at every vendor.
    """


class ProviderAuthError(ProviderError):
    """Bad or missing credentials. Failing over hides a config error."""


class ProviderNotInstalled(ProviderError):
    """The vendor SDK is not installed."""

    def __init__(self, provider: str, extra: str) -> None:
        super().__init__(
            f"The {provider} provider needs its SDK. "
            f'Install it with: pip install "modelops-backend[{extra}]"'
        )


@dataclass
class CompletionResult:
    """One completion, with everything the router needs to learn from it."""

    content: str
    provider: str
    model: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str | None = None
    attempts: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_per_1k_output(self) -> float:
        if self.output_tokens == 0:
            return 0.0
        return self.cost_usd / self.output_tokens * 1000


class ModelProvider(abc.ABC):
    """One model behind one uniform interface.

    Args:
        spec: Which model this instance speaks to, and what it costs.
        api_key: Vendor credential.
        timeout: Per-call wall-clock ceiling.
        max_retries: Attempts on a retryable error before giving up. The router
            handles cross-provider fallback; this is only same-provider retry.
    """

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.spec = spec
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    # ---------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return self.spec.provider

    @property
    def model_id(self) -> str:
        return self.spec.id

    @property
    def model(self) -> str:
        return self.spec.model

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.spec.id!r})"

    # ------------------------------------------------------------- calls

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> CompletionResult:
        """Run a completion, retrying transient failures with jittered backoff.

        Temperature defaults to 0: this system's core claim is that a cheaper
        model gives comparable output, and that claim is only measurable if the
        output is reproducible.
        """
        if max_tokens > self.spec.max_output_tokens:
            raise ProviderBadRequest(
                f"max_tokens={max_tokens} exceeds {self.spec.id}'s limit of "
                f"{self.spec.max_output_tokens}"
            )

        last: ProviderError | None = None
        for attempt in range(1, self.max_retries + 2):
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self.timeout):
                    result = await self._complete(
                        prompt, max_tokens=max_tokens, temperature=temperature, system=system
                    )
                result.attempts = attempt
                return result
            except TimeoutError as exc:
                last = ProviderUnavailable(f"{self.spec.id} timed out after {self.timeout}s")
                last.__cause__ = exc
            except ProviderError as exc:
                last = exc
                if not exc.retryable:
                    raise
            except Exception as exc:  # a vendor SDK raising something unmapped
                last = ProviderUnavailable(f"{self.spec.id}: {type(exc).__name__}: {exc}")
                last.__cause__ = exc

            elapsed_ms = (time.perf_counter() - started) * 1000
            if attempt > self.max_retries:
                break
            delay = self._backoff(attempt, last)
            log.warning(
                "provider_retry",
                model_id=self.spec.id,
                attempt=attempt,
                elapsed_ms=round(elapsed_ms, 1),
                retry_in_s=round(delay, 2),
                error=str(last),
            )
            await asyncio.sleep(delay)

        assert last is not None
        raise last

    @staticmethod
    def _backoff(attempt: int, error: ProviderError | None) -> float:
        """Exponential backoff with full jitter, honouring ``Retry-After``.

        Jitter matters here: an eval run fires many calls at once, and
        synchronised retries would rebuild the same burst that caused the 429.
        """
        if isinstance(error, ProviderRateLimited) and error.retry_after:
            return min(error.retry_after, 30.0)
        return random.uniform(0, min(2.0**attempt * 0.5, 8.0))

    @abc.abstractmethod
    async def _complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> CompletionResult:
        """Vendor-specific call. Implementations map SDK errors onto
        :class:`ProviderError` subclasses and build a
        :class:`CompletionResult` from the vendor's reported usage."""

    # ------------------------------------------------------------- tokens

    def count_tokens(self, text: str) -> int:
        """Approximate token count for pre-call cost estimation.

        Deliberately local and cheap. A vendor's exact counting endpoint costs a
        network round trip, which would put the router's own latency budget out
        of reach; overrides that have a local tokenizer should use it.
        """
        return max(1, int(len(text) / _CHARS_PER_TOKEN))

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 512) -> float:
        """Pre-call cost estimate, so routing can exclude providers over budget
        without paying to find out."""
        return self.spec.pricing.cost(self.count_tokens(prompt), expected_output_tokens)

    def build_result(
        self,
        *,
        content: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        finish_reason: str | None = None,
    ) -> CompletionResult:
        """Assemble a result, computing cost from the vendor's own usage numbers."""
        return CompletionResult(
            content=content,
            provider=self.spec.provider,
            model=self.spec.model,
            model_id=self.spec.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=self.spec.pricing.cost(input_tokens, output_tokens),
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------- health

    async def health_check(self) -> bool:
        """Cheapest possible live call, for readiness probes."""
        try:
            await self.complete("ping", max_tokens=1)
            return True
        except ProviderError:
            return False


def classify_status(status: int, message: str, retry_after: float | None = None) -> ProviderError:
    """Map an HTTP status onto the right :class:`ProviderError`.

    Shared by the adapters so all three classify failures identically — the
    router's fallback behaviour depends on that consistency.
    """
    if status == 429:
        return ProviderRateLimited(message, retry_after)
    if status in (401, 403):
        return ProviderAuthError(message)
    if status >= 500 or status in (408, 409):
        return ProviderUnavailable(message)
    if 400 <= status < 500:
        return ProviderBadRequest(message)
    return ProviderUnavailable(message)


def status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status from a vendor SDK exception."""
    for attr in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response: Any = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


__all__ = [
    "CompletionResult",
    "ModelProvider",
    "ProviderAuthError",
    "ProviderBadRequest",
    "ProviderError",
    "ProviderNotInstalled",
    "ProviderRateLimited",
    "ProviderUnavailable",
    "classify_status",
    "status_of",
]
