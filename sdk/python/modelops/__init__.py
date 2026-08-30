"""ModelOps Python SDK.

Status: the public API is designed but not implemented. This module names the
surface the backend will serve; see ../../../ROADMAP.md for the build order.

    from modelops import Router, RoutingPolicy

    router = Router(providers=[...], policy=RoutingPolicy(tasks={...}))
    result = await router.complete(prompt="...", task_type="classification")
"""

__version__ = "0.0.0.dev0"
__all__ = ["Router", "RoutingPolicy", "EvalSet"]


def __getattr__(name: str) -> object:
    if name in __all__:
        raise NotImplementedError(
            f"modelops.{name} is not implemented yet. "
            "See ROADMAP.md — implementation begins in Week 5."
        )
    raise AttributeError(f"module 'modelops' has no attribute {name!r}")
