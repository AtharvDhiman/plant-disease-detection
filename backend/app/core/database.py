"""SQLAlchemy engine, session factory and schema creation."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _engine_kwargs() -> dict:
    if settings.database_url.startswith("sqlite"):
        return {
            # FastAPI runs sync endpoints in a threadpool, so the connection can
            # legitimately be used from a thread other than the creating one.
            "connect_args": {"check_same_thread": False},
            "pool_pre_ping": True,
        }
    return {"pool_pre_ping": True}


engine: Engine = create_engine(settings.database_url, future=True, **_engine_kwargs())


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:
    """Enable WAL and foreign keys on SQLite.

    WAL lets the dashboard read history while a prediction is being written,
    instead of hitting 'database is locked'.
    """
    if not settings.database_url.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables for every registered model."""
    from app.models import prediction  # noqa: F401  (registers the mappings)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
