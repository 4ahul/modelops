"""Core: routing policy, scoring, execution and provider health."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.health import HealthTracker
from app.core.logging import configure_logging, get_logger
from app.core.policy import RoutingPolicy, TaskPolicy, cost_optimised
from app.core.router import (
    AllProvidersFailed,
    Candidate,
    NoEligibleModel,
    RoutedCompletion,
    Router,
    RoutingDecision,
)

__all__ = [
    "AllProvidersFailed",
    "Candidate",
    "HealthTracker",
    "NoEligibleModel",
    "RoutedCompletion",
    "Router",
    "RoutingDecision",
    "RoutingPolicy",
    "Settings",
    "TaskPolicy",
    "configure_logging",
    "cost_optimised",
    "get_logger",
    "get_settings",
]
