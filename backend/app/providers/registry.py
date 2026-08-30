"""Provider registry.

Turns configuration into live provider instances. Kept separate from the router
so the router can be tested with fakes, and so a deployment with one vendor key
starts cleanly instead of failing on the two it does not have.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.providers.base import ModelProvider
from app.providers.pricing import MODEL_CATALOG, ModelSpec, get_spec

log = get_logger(__name__)

#: Vendor name → (adapter import path, settings attribute holding the key).
_ADAPTERS: dict[str, tuple[str, str, str]] = {
    "anthropic": ("app.providers.anthropic_", "AnthropicProvider", "anthropic_api_key"),
    "openai": ("app.providers.openai_", "OpenAIProvider", "openai_api_key"),
    "gemini": ("app.providers.gemini_", "GeminiProvider", "google_api_key"),
}


def _load_adapter(provider: str) -> type[ModelProvider]:
    import importlib

    module_path, class_name, _ = _ADAPTERS[provider]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


class ProviderRegistry:
    """The set of models this deployment can actually call.

    Args:
        providers: ``{model_id: provider}``. Built by :meth:`from_settings` in
            normal use; passed directly in tests.
    """

    def __init__(self, providers: dict[str, ModelProvider] | None = None) -> None:
        self._providers: dict[str, ModelProvider] = dict(providers or {})

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model_ids: Iterable[str] | None = None,
    ) -> ProviderRegistry:
        """Instantiate every model whose vendor key is configured.

        Models whose key is missing are skipped with a log line rather than
        raising: a single-vendor deployment is a legitimate configuration, and
        the router only ever chooses among what it was given.
        """
        wanted = list(model_ids) if model_ids is not None else list(MODEL_CATALOG)
        registry = cls()
        skipped: list[str] = []

        for model_id in wanted:
            spec = get_spec(model_id)
            _, _, key_attr = _ADAPTERS[spec.provider]
            api_key = getattr(settings, key_attr, None)
            if not api_key:
                skipped.append(model_id)
                continue
            adapter = _load_adapter(spec.provider)
            registry.add(
                adapter(
                    spec,
                    api_key,
                    timeout=settings.request_timeout_seconds,
                    max_retries=settings.max_retries,
                )
            )

        if skipped:
            log.info("providers_skipped", models=skipped, reason="no api key configured")
        log.info("providers_ready", models=sorted(registry.model_ids))
        return registry

    # ---------------------------------------------------------- accessors

    def add(self, provider: ModelProvider) -> None:
        self._providers[provider.model_id] = provider

    def get(self, model_id: str) -> ModelProvider:
        try:
            return self._providers[model_id]
        except KeyError:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise KeyError(
                f"Model {model_id!r} is not available in this deployment. "
                f"Available: {available}. Check the vendor API key is set."
            ) from None

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Any:
        return iter(self._providers.values())

    @property
    def model_ids(self) -> list[str]:
        return list(self._providers)

    @property
    def specs(self) -> list[ModelSpec]:
        return [p.spec for p in self._providers.values()]

    def by_provider(self, provider: str) -> list[ModelProvider]:
        return [p for p in self._providers.values() if p.name == provider]

    def cheapest(self) -> ModelProvider | None:
        """Lowest blended cost per million tokens. For defaults and docs."""
        if not self._providers:
            return None
        return min(self._providers.values(), key=lambda p: p.spec.blended_cost_per_mtok)


__all__ = ["ProviderRegistry"]
