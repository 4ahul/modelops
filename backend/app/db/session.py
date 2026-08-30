"""Async database session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.logging import get_logger
from app.db.models import Base

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build an engine with pooling suited to the driver.

    SQLite gets no pool arguments — it does not support them, and passing them
    is a startup error rather than a warning, which is why the test database
    would otherwise fail to open.
    """
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if not database_url.startswith("sqlite"):
        kwargs.update(
            pool_size=10,
            max_overflow=20,
            # Recycle below the typical managed-Postgres idle timeout, so a
            # connection is never handed out after the server has closed it.
            pool_recycle=1800,
            pool_pre_ping=True,
        )
    return create_async_engine(database_url, **kwargs)


def init_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Create the process-wide engine and session factory."""
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine
    _engine = create_engine(database_url, echo=echo)
    _sessionmaker = async_sessionmaker(
        _engine,
        expire_on_commit=False,  # so a response can read a committed row
        autoflush=False,
    )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised; call init_engine() at startup")
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("Database engine not initialised; call init_engine() at startup")
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that rolls back on error."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def create_all() -> None:
    """Create tables directly.

    For tests and local development only. Production schema changes go through
    Alembic, so that a column rename is a reviewed migration rather than a
    silent no-op against an existing database.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Close the pool at shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def ping() -> bool:
    """Whether the database answers *and* has the schema. For readiness probes.

    ``SELECT 1`` only proves the connection is up. A deployment where the
    migration never ran would pass that check, report itself healthy, and then
    fail every write — so the probe touches a real table instead.
    """
    from sqlalchemy import text

    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1 FROM routing_decisions LIMIT 1"))
        return True
    except Exception as exc:
        log.warning("database_ping_failed", error=str(exc))
        return False


async def connection_ok() -> bool:
    """Whether the server is reachable, ignoring the schema.

    Lets ``/health`` distinguish "database is down" from "migrations have not
    run", which are different incidents with different fixes.
    """
    from sqlalchemy import text

    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


__all__ = [
    "connection_ok",
    "create_all",
    "create_engine",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "init_engine",
    "ping",
]
