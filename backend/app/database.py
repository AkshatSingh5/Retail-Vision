from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import DATABASE_URL, ON_VERCEL, ROOT_DIR


class Base(DeclarativeBase):
    """SQLAlchemy declarative base (SQLite and PostgreSQL)."""


_engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def resolve_database_url(url: str | None = None) -> str:
    """Make SQLite paths absolute; leave PostgreSQL URLs unchanged.

    On Vercel, localhost Postgres is unreachable, so fall back to /tmp SQLite.
    """
    raw = url or DATABASE_URL
    parsed = make_url(raw)
    if ON_VERCEL:
        host = (parsed.host or "").lower()
        if (not parsed.drivername.startswith("sqlite")) and host in {"", "localhost", "127.0.0.1", "::1"}:
            db_path = Path("/tmp/retail-vision/retail_vision.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path.as_posix()}"
    if not parsed.drivername.startswith("sqlite"):
        return raw
    database = parsed.database
    if not database or database == ":memory:":
        return raw
    db_path = Path(database)
    if ON_VERCEL:
        db_path = Path("/tmp/retail-vision") / db_path.name
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"
    if not db_path.is_absolute():
        db_path = (ROOT_DIR / db_path).resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _sqlite_connect_args(url: str) -> dict:
    if make_url(url).drivername.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def get_engine(url: str | None = None) -> Engine:
    global _engine, SessionLocal
    if url is not None:
        reset_engine()
    if _engine is None:
        resolved = resolve_database_url(url)
        _engine = create_engine(
            resolved,
            future=True,
            pool_pre_ping=True,
            connect_args=_sqlite_connect_args(resolved),
        )
        if make_url(resolved).drivername.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if SessionLocal is None:
        if ON_VERCEL:
            init_db()
        else:
            get_engine()
    assert SessionLocal is not None
    return SessionLocal


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(url: str | None = None) -> Engine:
    from backend.app import models as _models  # noqa: F401

    engine = get_engine(url)
    _repair_orphaned_sequences(engine)
    Base.metadata.create_all(bind=engine)
    _migrate_product_columns(engine)
    try:
        from backend.app.services.vector_search import ensure_pgvector_schema

        ensure_pgvector_schema(engine)
    except Exception as extra:
        import logging

        logging.getLogger(__name__).warning("pgvector setup skipped: %s", extra)
    try:
        from backend.app.services.seed import seed_products_from_registry

        if SessionLocal is None:
            return engine
        session = SessionLocal()
        try:
            seed_products_from_registry(session)
            session.commit()
        except Exception as extra:
            import logging

            session.rollback()
            logging.getLogger(__name__).warning("catalog seed skipped: %s", extra)
        finally:
            session.close()
    except Exception:
        pass
    return engine


def _repair_orphaned_sequences(engine: Engine) -> None:
    """Drop SERIAL sequences left behind by a failed CREATE TABLE (Postgres)."""
    if engine.dialect.name != "postgresql":
        return
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected = {
        "product_embeddings": "product_embeddings_id_seq",
        "scan_logs": "scan_logs_id_seq",
    }
    with engine.begin() as connection:
        for table, sequence in expected.items():
            if table in tables:
                continue
            connection.execute(text(f'DROP SEQUENCE IF EXISTS "{sequence}" CASCADE'))


def _migrate_product_columns(engine: Engine) -> None:
    """Add columns / tables introduced after the first create_all."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "products" not in tables:
        return
    existing = {column["name"] for column in inspector.get_columns("products")}
    statements = []
    if "barcode" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN barcode VARCHAR(64)")
    if "description" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN description VARCHAR(1024)")
    if "recognition_embedding" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN recognition_embedding TEXT")
    if "variant" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN variant VARCHAR(128)")
    if "weight" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN weight VARCHAR(64)")
    if "image_url" not in existing:
        statements.append("ALTER TABLE products ADD COLUMN image_url VARCHAR(512)")

    if "product_images" in tables:
        image_cols = {column["name"] for column in inspector.get_columns("product_images")}
        if "storage_key" not in image_cols:
            statements.append("ALTER TABLE product_images ADD COLUMN storage_key VARCHAR(512)")
        if "image_url" not in image_cols:
            statements.append("ALTER TABLE product_images ADD COLUMN image_url VARCHAR(512)")

    if "transaction_items" in tables:
        item_cols = {column["name"] for column in inspector.get_columns("transaction_items")}
        if "weight" not in item_cols:
            statements.append("ALTER TABLE transaction_items ADD COLUMN weight VARCHAR(64)")

    if not statements:
        return
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def reset_engine() -> None:
    global _engine, SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    SessionLocal = None
