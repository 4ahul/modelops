"""Alembic environment.

The database URL comes from ``DATABASE_URL``, never from ``alembic.ini`` — a
checked-in URL is how a migration gets run against the wrong environment.

``asyncpg`` cannot be driven by Alembic's synchronous runner, so the async
engine is created and the migration executed inside ``connection.run_sync``.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Migrations refuse to guess a database; "
            "export it explicitly, e.g. "
            "DATABASE_URL=postgresql+asyncpg://modelops:...@localhost:5432/modelops"
        )
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it — for reviewing a migration."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Without these two, a changed column type or default is silently
        # ignored by autogenerate and the schema drifts from the models.
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
