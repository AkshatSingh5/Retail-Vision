"""Alembic environment for Retail Vision (SQLite dev + PostgreSQL prod).

- Resolves the database URL through ``backend.app.database.resolve_database_url``
  so relative SQLite paths anchor to the repo root (matching the application).
- When ``backend.app.database.init_db`` runs migrations at startup it passes the
  already-created engine via ``config.attributes["engine"]`` so in-memory SQLite
  tests share one connection pool.
- The pgvector ``embedding_vec`` column and its HNSW/IVFFlat indexes are managed
  by ``backend.app.services.vector_search.ensure_pgvector_schema`` (config aware),
  so they are excluded from autogenerate diffing.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

_ALEMBIC_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _ALEMBIC_DIR.parent
_ROOT_DIR = _BACKEND_DIR.parent
for _path in (str(_ROOT_DIR), str(_BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.app.database import Base, resolve_database_url  # noqa: E402
from backend.app import models  # noqa: E402, F401  (registers all tables on Base.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Managed out-of-band by services.vector_search.ensure_pgvector_schema (PostgreSQL only).
_PGVECTOR_COLUMNS = {("product_embeddings", "embedding_vec")}
_PGVECTOR_INDEXES = {"ix_product_embeddings_hnsw", "ix_product_embeddings_ivfflat"}


def include_object(obj, name: str, type_: str, reflected: bool, compare_to) -> bool:
    """Keep pgvector artifacts out of autogenerate diffs."""
    if type_ == "column" and (obj.table.name, name) in _PGVECTOR_COLUMNS:
        return False
    if type_ == "index" and name in _PGVECTOR_INDEXES:
        return False
    return True


def _database_url() -> str:
    """Resolve the URL Alembic operates against.

    Precedence:
      1. ``config.attributes["forced_url"]`` — set by
         ``backend.app.database.run_migrations`` so tests / init_db overrides win.
      2. ``DATABASE_URL`` environment variable / .env (CLI usage).
      3. The ``sqlalchemy.url`` value in ``alembic.ini`` (a plain default).
    """
    forced = config.attributes.get("forced_url")
    if forced:
        return resolve_database_url(forced)
    env_value = os.getenv("DATABASE_URL")
    configured = config.get_main_option("sqlalchemy.url")
    return resolve_database_url(env_value or configured or None)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL to stdout, no DB connection)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_database_url().startswith("sqlite"),
        include_object=include_object,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Prefer the engine provided by the application (same pool, in-memory safe);
    otherwise build a short-lived engine from the resolved URL.
    """
    engine = config.attributes.get("engine")
    if engine is not None:
        connectable = engine
    else:
        configuration = config.get_section(config.config_ini_section, {})
        configuration["sqlalchemy.url"] = _database_url()
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connectable.dialect.name == "sqlite",
            include_object=include_object,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()