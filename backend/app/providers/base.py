"""Provider abstraction.

Every model vendor is adapted to this interface so the router can compare them
on the same three axes: cost, latency and measured quality.

Status: interface only. Implementations land in Phase 1 (see ../../ROADMAP.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CompletionResult:
    """One completion, with everything the router needs to learn from it."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Pricing:
    """Per-million-token rates.

    Dated deliberately: vendor prices change, and a stale table silently
    mis-routes every request while looking perfectly healthy.
    """

    input_per_mtok: float
    output_per_mtok: float
    as_of: str

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


class ModelProvider(ABC):
    """One model behind one uniform interface."""

    name: str
    model: str
    pricing: Pricing

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: object) -> CompletionResult:
        """Run a completion and report what it cost and how long it took."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count for this model's tokenizer — used to estimate cost
        before committing to a provider."""

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 512) -> float:
        """Pre-call cost estimate, so routing can exclude providers over budget
        without paying to find out."""
        return self.pricing.cost(self.count_tokens(prompt), expected_output_tokens)
