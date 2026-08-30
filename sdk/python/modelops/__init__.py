"""ModelOps Python SDK.

The client for a running ModelOps deployment::

    from modelops import ModelOps

    async with ModelOps(api_url="https://modelops.example.com", api_key="mo_...") as client:
        result = await client.complete("Classify this ticket: ...", task_type="classification")
        print(result.content, result.model_id, result.cost_usd)

Sync callers use :class:`ModelOpsSync`, which wraps the same transport::

    from modelops import ModelOpsSync

    client = ModelOpsSync(api_url=..., api_key=...)
    print(client.complete("...").content)
"""

from __future__ import annotations

from modelops.client import (
    CompletionResult,
    EvalReport,
    ModelOps,
    ModelOpsError,
    ModelOpsSync,
    NoEligibleModelError,
    ProviderFailedError,
    RateLimitedError,
    RouteDecision,
)

__version__ = "0.1.0"

__all__ = [
    "CompletionResult",
    "EvalReport",
    "ModelOps",
    "ModelOpsError",
    "ModelOpsSync",
    "NoEligibleModelError",
    "ProviderFailedError",
    "RateLimitedError",
    "RouteDecision",
    "__version__",
]
