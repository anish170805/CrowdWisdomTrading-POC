"""SQLAlchemy table definitions for canonical storage.

The same metadata creates SQLite and PostgreSQL schemas.  Timestamps are stored
as timezone-aware UTC in PostgreSQL; SQLite has no native timezone type, so
:class:`UtcDateTime` stores ISO-8601 text with an explicit ``+00:00`` offset.
That keeps ordering correct and round-trips aware datetimes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from src.schemas.common import as_utc

SCHEMA_VERSION = 1
SQLITE_UTC_FORMAT = "%Y-%m-%d %H:%M:%S.%f+00:00"


class UtcDateTime(TypeDecorator[datetime]):
    """A datetime column that always round-trips as aware UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(Text())
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        value = as_utc(value)
        if dialect.name == "sqlite":
            return value.strftime(SQLITE_UTC_FORMAT)
        return value

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.strptime(value, SQLITE_UTC_FORMAT).replace(tzinfo=UTC)
        return as_utc(value)


metadata = MetaData()

_PK_TYPE = BigInteger().with_variant(Integer, "sqlite")

tweets = Table(
    "tweets",
    metadata,
    Column("id", _PK_TYPE, primary_key=True, autoincrement=True),
    Column("post_id", String(512), nullable=True),
    Column("post_fingerprint", String(64), nullable=False),
    Column("asset_symbol", String(64), nullable=True),
    Column("source", String(128), nullable=False),
    Column("source_format", String(32), nullable=True),
    Column("source_file", Text, nullable=True),
    Column("source_row", Integer, nullable=True),
    Column("source_record_id", String(512), nullable=True),
    Column("batch_id", String(128), nullable=False),
    Column("raw_payload_ref", Text, nullable=True),
    Column("ingested_at", UtcDateTime, nullable=False),
    Column("text", Text, nullable=False),
    Column("text_original", Text, nullable=False),
    Column("created_at", UtcDateTime, nullable=False),
    Column("created_at_original", Text, nullable=False),
    Column("source_timezone_offset", String(16), nullable=True),
    Column("timezone_assumed", Boolean, nullable=False, default=False),
    Column("author", Text, nullable=True),
    Column("author_handle", String(255), nullable=True),
    Column("likes", BigInteger().with_variant(Integer, "sqlite"), nullable=True),
    Column("reposts", BigInteger().with_variant(Integer, "sqlite"), nullable=True),
    Column("replies", BigInteger().with_variant(Integer, "sqlite"), nullable=True),
    Column("quotes", BigInteger().with_variant(Integer, "sqlite"), nullable=True),
    Column("views", BigInteger().with_variant(Integer, "sqlite"), nullable=True),
    Column("url", Text, nullable=True),
    Column("warnings", JSON, nullable=False, default=list),
    UniqueConstraint("post_id", name="uq_tweets_post_id"),
    UniqueConstraint("post_fingerprint", name="uq_tweets_post_fingerprint"),
)
Index("ix_tweets_created_at", tweets.c.created_at)
Index("ix_tweets_asset_created_at", tweets.c.asset_symbol, tweets.c.created_at)
Index("ix_tweets_batch_id", tweets.c.batch_id)

price_bars = Table(
    "price_bars",
    metadata,
    Column("id", _PK_TYPE, primary_key=True, autoincrement=True),
    Column("asset_symbol", String(64), nullable=False),
    Column("asset_type", String(32), nullable=True),
    Column("bar_date", Date, nullable=False),
    Column("timestamp", UtcDateTime, nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("adjusted_close", Float, nullable=True),
    Column("volume", Float, nullable=True),
    Column("provider", String(64), nullable=False),
    Column("batch_id", String(128), nullable=False),
    Column("source_record_id", String(512), nullable=True),
    Column("raw_payload_ref", Text, nullable=True),
    Column("retrieved_at", UtcDateTime, nullable=True),
    Column("ingested_at", UtcDateTime, nullable=False),
    Column("warnings", JSON, nullable=False, default=list),
    UniqueConstraint("asset_symbol", "timestamp", name="uq_price_bars_asset_timestamp"),
    UniqueConstraint("asset_symbol", "bar_date", name="uq_price_bars_asset_date"),
)
Index("ix_price_bars_timestamp", price_bars.c.timestamp)
Index("ix_price_bars_asset_date", price_bars.c.asset_symbol, price_bars.c.bar_date)
Index("ix_price_bars_batch_id", price_bars.c.batch_id)

quarantined_rows = Table(
    "quarantined_rows",
    metadata,
    Column("id", _PK_TYPE, primary_key=True, autoincrement=True),
    Column("entity", String(32), nullable=False),
    Column("record_key", String(512), nullable=False),
    Column("reason_code", String(64), nullable=False),
    Column("reason", Text, nullable=False),
    Column("reason_codes", JSON, nullable=False, default=list),
    Column("source", String(128), nullable=True),
    Column("source_record_id", String(512), nullable=True),
    Column("batch_id", String(128), nullable=False),
    Column("quarantined_at", UtcDateTime, nullable=False),
    Column("raw_payload_ref", Text, nullable=True),
    Column("raw_payload", JSON, nullable=False, default=dict),
    Column("warnings", JSON, nullable=False, default=list),
    UniqueConstraint(
        "entity",
        "batch_id",
        "record_key",
        "reason_code",
        name="uq_quarantined_rows_identity",
    ),
)
Index("ix_quarantined_rows_entity", quarantined_rows.c.entity)
Index("ix_quarantined_rows_quarantined_at", quarantined_rows.c.quarantined_at)
Index("ix_quarantined_rows_batch_id", quarantined_rows.c.batch_id)

ALL_TABLES = (tweets, price_bars, quarantined_rows)


def create_schema(engine: Any) -> None:
    """Create every canonical table and index (idempotent)."""
    metadata.create_all(engine, tables=list(ALL_TABLES), checkfirst=True)


__all__ = [
    "ALL_TABLES",
    "SCHEMA_VERSION",
    "SQLITE_UTC_FORMAT",
    "UtcDateTime",
    "create_schema",
    "metadata",
    "price_bars",
    "quarantined_rows",
    "tweets",
]
