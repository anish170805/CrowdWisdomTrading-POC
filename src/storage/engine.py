"""Database engine construction and schema bootstrap."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.pool import ConnectionPoolEntry

from src.config import Settings
from src.storage.models import create_schema


def create_engine_from_settings(settings: Settings, *, echo: bool = False) -> Engine:
    """Create an engine tuned for the configured backend.

    SQLite gets a busy timeout, WAL journaling (for file databases), and foreign
    keys.  PostgreSQL uses pre-ping so stale pooled connections are replaced.
    """
    if settings.database_backend == "sqlite":
        sqlite_path = settings.sqlite_path
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            settings.database_url,
            echo=echo,
            future=True,
            connect_args={"timeout": settings.sqlite_busy_timeout_ms / 1000},
        )

        @event.listens_for(engine, "connect")
        def _configure_sqlite(
            dbapi_connection: object,
            connection_record: ConnectionPoolEntry,
        ) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                cursor.execute(f"PRAGMA busy_timeout = {settings.sqlite_busy_timeout_ms}")
                if sqlite_path is not None:
                    cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA foreign_keys = ON")
            finally:
                cursor.close()

        return engine

    return create_engine(settings.database_url, echo=echo, future=True, pool_pre_ping=True)


def initialize_database(engine: Engine) -> None:
    """Create canonical tables if they do not exist."""
    create_schema(engine)


__all__ = ["create_engine_from_settings", "initialize_database"]
