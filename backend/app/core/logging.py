"""Structured logging.

JSON in production so a log aggregator can filter on fields; coloured
key-values in development so a human can read them. One call to
:func:`configure_logging` at startup wires both stdlib ``logging`` and
``structlog`` through the same renderer, so a library that logs via stdlib
still lands in the same stream.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_configured = False

#: Keys whose values are never logged, whatever a caller passes. Prompt bodies
#: and API keys are the two things that must not end up in a log aggregator.
_REDACT_KEYS = frozenset(
    {"prompt", "api_key", "authorization", "password", "secret", "token", "content"}
)


def _redact(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Mask sensitive values rather than trusting every call site.

    A processor is the only place this can be enforced: relying on discipline at
    hundreds of log statements means one of them eventually logs a prompt.
    """
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS and event_dict[key] is not None:
            event_dict[key] = "[redacted]"
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    global _configured
    if _configured:
        return

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _redact,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[_redact, renderer], foreign_pre_chain=shared
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; route them through ours so one
    # deployment emits one log format.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A logger bound to ``name``."""
    return structlog.stdlib.get_logger(name)


def reset_logging() -> None:
    """Allow reconfiguration. For tests."""
    global _configured
    _configured = False


__all__ = ["configure_logging", "get_logger", "reset_logging"]
