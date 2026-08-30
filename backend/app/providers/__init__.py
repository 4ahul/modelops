"""Model providers: one interface over Anthropic, OpenAI and Gemini.

Adapters are imported lazily by :class:`~app.providers.registry.ProviderRegistry`
so a deployment that uses one vendor does not need the other two SDKs installed.
"""

from __future__ import annotations

from app.providers.base import (
    CompletionResult,
    ModelProvider,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderNotInstalled,
    ProviderRateLimited,
    ProviderUnavailable,
)
from app.providers.pricing import (
    MODEL_CATALOG,
    PRICING_AS_OF,
    ModelSpec,
    Pricing,
    get_spec,
)
from app.providers.registry import ProviderRegistry

__all__ = [
    "MODEL_CATALOG",
    "PRICING_AS_OF",
    "CompletionResult",
    "ModelProvider",
    "ModelSpec",
    "Pricing",
    "ProviderAuthError",
    "ProviderBadRequest",
    "ProviderError",
    "ProviderNotInstalled",
    "ProviderRateLimited",
    "ProviderRegistry",
    "ProviderUnavailable",
    "get_spec",
]
