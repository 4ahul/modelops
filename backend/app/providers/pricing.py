"""Model pricing, versioned and dated.

Every rate here is per million tokens, copied from the vendor's public pricing
page on :data:`PRICING_AS_OF`. This table is the single input to every cost
number the product reports, which makes a stale entry the most dangerous kind
of bug in the system: routing keeps working, the dashboard keeps rendering, and
every decision is quietly wrong.

Two things follow from that:

- :data:`PRICING_AS_OF` is exposed at ``/health`` and printed by
  ``modelops pricing``, so the age of the table is visible without reading
  source.
- :func:`assert_fresh` fails a startup check once the table is older than
  :data:`STALE_AFTER_DAYS`, rather than letting it rot silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

#: The date every rate below was verified against the vendor's pricing page.
PRICING_AS_OF = date(2026, 8, 30)

#: How long the table may go unverified before startup emits a warning.
STALE_AFTER_DAYS = 90


@dataclass(frozen=True)
class Pricing:
    """Per-million-token rates for one model."""

    input_per_mtok: float
    output_per_mtok: float
    as_of: date = PRICING_AS_OF

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Dollar cost of a call with this token split."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000

    @property
    def age_days(self) -> int:
        return (datetime.now(UTC).date() - self.as_of).days


@dataclass(frozen=True)
class ModelSpec:
    """Everything the router needs to know about a model before calling it."""

    id: str
    provider: str
    model: str
    pricing: Pricing
    context_window: int
    max_output_tokens: int
    #: Rough tier, used only to break ties and to explain a decision in words.
    tier: str = "standard"

    @property
    def blended_cost_per_mtok(self) -> float:
        """A single comparable number, assuming a 3:1 input:output split.

        Real workloads vary, so this is for ordering models in a UI — never for
        a routing decision, which uses the measured token counts.
        """
        return (self.pricing.input_per_mtok * 3 + self.pricing.output_per_mtok) / 4


#: The models the router can choose between.
#:
#: Keys are stable ids used in API requests and stored in ``routing_decisions``;
#: the vendor's own model string lives in ``model`` and can change under a
#: stable id when a vendor ships a dated snapshot.
MODEL_CATALOG: dict[str, ModelSpec] = {
    # ---------------------------------------------------------- Anthropic
    "claude-opus": ModelSpec(
        id="claude-opus",
        provider="anthropic",
        model="claude-opus-4-20250514",
        pricing=Pricing(15.00, 75.00),
        context_window=200_000,
        max_output_tokens=32_000,
        tier="frontier",
    ),
    "claude-sonnet": ModelSpec(
        id="claude-sonnet",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        pricing=Pricing(3.00, 15.00),
        context_window=200_000,
        max_output_tokens=64_000,
        tier="balanced",
    ),
    "claude-haiku": ModelSpec(
        id="claude-haiku",
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        pricing=Pricing(0.80, 4.00),
        context_window=200_000,
        max_output_tokens=8_192,
        tier="fast",
    ),
    # ------------------------------------------------------------- OpenAI
    "gpt-4o": ModelSpec(
        id="gpt-4o",
        provider="openai",
        model="gpt-4o-2024-11-20",
        pricing=Pricing(2.50, 10.00),
        context_window=128_000,
        max_output_tokens=16_384,
        tier="balanced",
    ),
    "gpt-4o-mini": ModelSpec(
        id="gpt-4o-mini",
        provider="openai",
        model="gpt-4o-mini-2024-07-18",
        pricing=Pricing(0.15, 0.60),
        context_window=128_000,
        max_output_tokens=16_384,
        tier="fast",
    ),
    # ------------------------------------------------------------- Google
    "gemini-pro": ModelSpec(
        id="gemini-pro",
        provider="gemini",
        model="gemini-1.5-pro-002",
        pricing=Pricing(1.25, 5.00),
        context_window=2_000_000,
        max_output_tokens=8_192,
        tier="balanced",
    ),
    "gemini-flash": ModelSpec(
        id="gemini-flash",
        provider="gemini",
        model="gemini-1.5-flash-002",
        pricing=Pricing(0.075, 0.30),
        context_window=1_000_000,
        max_output_tokens=8_192,
        tier="fast",
    ),
}


def get_spec(model_id: str) -> ModelSpec:
    """Look up a model by id.

    Raises:
        KeyError: naming the known ids, because a typo in a routing policy
            should not surface as a mysterious ``None``.
    """
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        known = ", ".join(sorted(MODEL_CATALOG))
        raise KeyError(f"Unknown model {model_id!r}. Known models: {known}") from None


def specs_for_provider(provider: str) -> list[ModelSpec]:
    return [s for s in MODEL_CATALOG.values() if s.provider == provider]


def pricing_age_days() -> int:
    """How long since the pricing table was verified."""
    return (datetime.now(UTC).date() - PRICING_AS_OF).days


def assert_fresh() -> str | None:
    """Return a warning if the pricing table is stale, else ``None``."""
    age = pricing_age_days()
    if age > STALE_AFTER_DAYS:
        return (
            f"Pricing table is {age} days old (verified {PRICING_AS_OF.isoformat()}). "
            f"Every cost and routing decision is computed from it. "
            f"Re-check vendor pricing and update backend/app/providers/pricing.py."
        )
    return None


__all__ = [
    "MODEL_CATALOG",
    "PRICING_AS_OF",
    "STALE_AFTER_DAYS",
    "ModelSpec",
    "Pricing",
    "assert_fresh",
    "get_spec",
    "pricing_age_days",
    "specs_for_provider",
]
