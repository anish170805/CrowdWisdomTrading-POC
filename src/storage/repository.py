"""Repository for canonical tweets, price bars, and quarantine rows.

All writes are idempotent: repeated imports never create duplicate canonical
rows.  Reads return typed schema objects, so downstream modules never see raw
table dictionaries or provider payloads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import Engine, and_, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.schemas.common import as_utc
from src.schemas.prices import PriceBar
from src.schemas.quarantine import QuarantineRecord
from src.schemas.tweets import TweetRecord
from src.storage.models import create_schema, price_bars, quarantined_rows, tweets


@dataclass(frozen=True)
class UpsertStats:
    """Row counts from one idempotent write."""

    attempted: int
    inserted: int
    updated: int = 0
    duplicates: int = 0


def _tweet_values(record: TweetRecord) -> dict[str, Any]:
    return {
        "post_id": record.post_id,
        "post_fingerprint": record.post_fingerprint,
        "asset_symbol": record.asset_symbol,
        "source": record.source,
        "source_format": record.source_format,
        "source_file": record.source_file,
        "source_row": record.source_row,
        "source_record_id": record.source_record_id,
        "batch_id": record.batch_id,
        "raw_payload_ref": record.raw_payload_ref,
        "ingested_at": as_utc(record.ingested_at),
        "text": record.text,
        "text_original": record.text_original,
        "created_at": as_utc(record.created_at),
        "created_at_original": record.created_at_original,
        "source_timezone_offset": record.source_timezone_offset,
        "timezone_assumed": record.timezone_assumed,
        "author": record.author,
        "author_handle": record.author_handle,
        "likes": record.likes,
        "reposts": record.reposts,
        "replies": record.replies,
        "quotes": record.quotes,
        "views": record.views,
        "url": record.url,
        "warnings": list(record.warnings),
    }


def _price_values(bar: PriceBar) -> dict[str, Any]:
    return {
        "asset_symbol": bar.asset_symbol,
        "asset_type": bar.asset_type,
        "bar_date": bar.bar_date,
        "timestamp": as_utc(bar.timestamp),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "adjusted_close": bar.adjusted_close,
        "volume": bar.volume,
        "provider": bar.provider,
        "batch_id": bar.batch_id,
        "source_record_id": bar.source_record_id,
        "raw_payload_ref": bar.raw_payload_ref,
        "retrieved_at": as_utc(bar.retrieved_at) if bar.retrieved_at else None,
        "ingested_at": as_utc(bar.ingested_at),
        "warnings": list(bar.warnings),
    }


def _quarantine_values(row: QuarantineRecord) -> dict[str, Any]:
    return {
        "entity": row.entity.value,
        "record_key": row.record_key,
        "reason_code": row.reason_code,
        "reason": row.reason,
        "reason_codes": list(row.reason_codes),
        "source": row.source,
        "source_record_id": row.source_record_id,
        "batch_id": row.batch_id,
        "quarantined_at": as_utc(row.quarantined_at),
        "raw_payload_ref": row.raw_payload_ref,
        "raw_payload": row.raw_payload,
        "warnings": list(row.warnings),
    }


def _tweet_from_row(row: Any) -> TweetRecord:
    payload = dict(row)
    payload.pop("id", None)
    payload["warnings"] = tuple(payload.get("warnings") or ())
    return TweetRecord(**payload)


def _price_from_row(row: Any) -> PriceBar:
    payload = dict(row)
    payload.pop("id", None)
    payload["warnings"] = tuple(payload.get("warnings") or ())
    return PriceBar(**payload)


def _as_datetime(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    return datetime.combine(value, time.min, tzinfo=UTC)


class CanonicalRepository:
    """Stable read/write interface over the canonical tables."""

    def __init__(self, engine: Engine, *, ensure_schema: bool = True) -> None:
        self.engine = engine
        self._dialect = engine.dialect.name
        if self._dialect not in {"sqlite", "postgresql"}:
            raise ValueError(f"unsupported database backend: {self._dialect}")
        if ensure_schema:
            create_schema(engine)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _insert_statement(self, table: Any, values: list[dict[str, Any]]) -> Any:
        if self._dialect == "sqlite":
            return sqlite_insert(table).values(values)
        return postgresql_insert(table).values(values)

    def upsert_tweets(self, records: Sequence[TweetRecord]) -> UpsertStats:
        """Insert tweets, skipping any existing platform ID or fingerprint."""
        if not records:
            return UpsertStats(attempted=0, inserted=0)

        unique: dict[str, TweetRecord] = {}
        for record in records:
            unique.setdefault(record.post_fingerprint, record)
        keys = list(unique)
        values = [_tweet_values(unique[key]) for key in keys]

        with self.engine.begin() as connection:
            existing = set(
                connection.execute(
                    select(tweets.c.post_fingerprint).where(tweets.c.post_fingerprint.in_(keys))
                ).scalars()
            )
            statement = self._insert_statement(tweets, values).on_conflict_do_nothing()
            connection.execute(statement)

        inserted = len(keys) - len(existing)
        return UpsertStats(
            attempted=len(records),
            inserted=inserted,
            duplicates=len(records) - inserted,
        )

    def upsert_price_bars(self, bars: Sequence[PriceBar]) -> UpsertStats:
        """Insert price bars, updating the OHLCV values on an asset/date conflict."""
        if not bars:
            return UpsertStats(attempted=0, inserted=0)

        unique: dict[tuple[str, datetime], PriceBar] = {}
        for bar in bars:
            unique.setdefault((bar.asset_symbol, as_utc(bar.timestamp)), bar)
        values = [_price_values(bar) for bar in unique.values()]

        with self.engine.begin() as connection:
            assets = {bar.asset_symbol for bar in unique.values()}
            timestamps = {as_utc(bar.timestamp) for bar in unique.values()}
            existing_rows = connection.execute(
                select(price_bars.c.asset_symbol, price_bars.c.timestamp).where(
                    price_bars.c.asset_symbol.in_(assets),
                    price_bars.c.timestamp.in_(timestamps),
                )
            ).all()
            existing = {(row[0], as_utc(row[1])) for row in existing_rows}

            base = self._insert_statement(price_bars, values)
            statement = base.on_conflict_do_update(
                index_elements=["asset_symbol", "timestamp"],
                set_={
                    "asset_type": base.excluded.asset_type,
                    "bar_date": base.excluded.bar_date,
                    "open": base.excluded.open,
                    "high": base.excluded.high,
                    "low": base.excluded.low,
                    "close": base.excluded.close,
                    "adjusted_close": base.excluded.adjusted_close,
                    "volume": base.excluded.volume,
                    "provider": base.excluded.provider,
                    "batch_id": base.excluded.batch_id,
                    "source_record_id": base.excluded.source_record_id,
                    "raw_payload_ref": base.excluded.raw_payload_ref,
                    "retrieved_at": base.excluded.retrieved_at,
                    "ingested_at": base.excluded.ingested_at,
                    "warnings": base.excluded.warnings,
                },
            )
            connection.execute(statement)

        updated = len(existing & set(unique))
        inserted = len(unique) - updated
        return UpsertStats(
            attempted=len(bars),
            inserted=inserted,
            updated=updated,
            duplicates=len(bars) - len(unique),
        )

    def insert_quarantine(self, rows: Sequence[QuarantineRecord]) -> UpsertStats:
        """Persist quarantine rows; the same row/reason in a batch is written once."""
        if not rows:
            return UpsertStats(attempted=0, inserted=0)

        unique: dict[tuple[str, str, str, str], QuarantineRecord] = {}
        for row in rows:
            key = (row.entity.value, row.batch_id, row.record_key, row.reason_code)
            unique.setdefault(key, row)
        values = [_quarantine_values(row) for row in unique.values()]

        with self.engine.begin() as connection:
            statement = self._insert_statement(quarantined_rows, values).on_conflict_do_nothing(
                index_elements=["entity", "batch_id", "record_key", "reason_code"]
            )
            result = connection.execute(statement)
            inserted = result.rowcount if result.rowcount is not None else len(values)

        inserted = max(0, min(inserted, len(unique)))
        return UpsertStats(
            attempted=len(rows),
            inserted=inserted,
            duplicates=len(rows) - inserted,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_tweet(self, post_id: str) -> TweetRecord | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(tweets).where(tweets.c.post_id == post_id))
                .mappings()
                .first()
            )
        return _tweet_from_row(row) if row is not None else None

    def read_tweets(
        self,
        start: datetime | date | None = None,
        end: datetime | date | None = None,
        *,
        asset_symbol: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TweetRecord]:
        """Read posts whose ``created_at`` is in the half-open range ``[start, end)``."""
        conditions = []
        start_at = _as_datetime(start)
        end_at = _as_datetime(end)
        if start_at is not None:
            conditions.append(tweets.c.created_at >= start_at)
        if end_at is not None:
            conditions.append(tweets.c.created_at < end_at)
        if asset_symbol is not None:
            conditions.append(tweets.c.asset_symbol == asset_symbol)

        statement = select(tweets)
        if conditions:
            statement = statement.where(and_(*conditions))
        statement = statement.order_by(tweets.c.created_at, tweets.c.id)
        if offset:
            statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)

        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [_tweet_from_row(row) for row in rows]

    def read_price_bars(
        self,
        asset_symbol: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[PriceBar]:
        """Read daily bars in the inclusive ``[start_date, end_date]`` range."""
        conditions = [price_bars.c.asset_symbol == asset_symbol]
        if start_date is not None:
            start_value = _as_datetime(start_date)
            if start_value is not None:
                conditions.append(price_bars.c.bar_date >= start_value.date())
        if end_date is not None:
            end_value = _as_datetime(end_date)
            if end_value is not None:
                conditions.append(price_bars.c.bar_date <= end_value.date())

        statement = select(price_bars).where(and_(*conditions)).order_by(price_bars.c.timestamp)
        if offset:
            statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)

        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [_price_from_row(row) for row in rows]

    def get_price_bar(self, asset_symbol: str, bar_date: date) -> PriceBar | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(price_bars).where(
                        price_bars.c.asset_symbol == asset_symbol,
                        price_bars.c.bar_date == bar_date,
                    )
                )
                .mappings()
                .first()
            )
        return _price_from_row(row) if row is not None else None

    def count_tweets(self, *, asset_symbol: str | None = None) -> int:
        statement = select(func.count()).select_from(tweets)
        if asset_symbol is not None:
            statement = statement.where(tweets.c.asset_symbol == asset_symbol)
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def count_price_bars(self, *, asset_symbol: str | None = None) -> int:
        statement = select(func.count()).select_from(price_bars)
        if asset_symbol is not None:
            statement = statement.where(price_bars.c.asset_symbol == asset_symbol)
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def count_quarantined(
        self,
        *,
        entity: str | None = None,
        batch_id: str | None = None,
    ) -> int:
        statement = select(func.count()).select_from(quarantined_rows)
        if entity is not None:
            statement = statement.where(quarantined_rows.c.entity == entity)
        if batch_id is not None:
            statement = statement.where(quarantined_rows.c.batch_id == batch_id)
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def read_quarantined(
        self,
        *,
        entity: str | None = None,
        batch_id: str | None = None,
        limit: int | None = None,
    ) -> list[QuarantineRecord]:
        statement = select(quarantined_rows)
        conditions = []
        if entity is not None:
            conditions.append(quarantined_rows.c.entity == entity)
        if batch_id is not None:
            conditions.append(quarantined_rows.c.batch_id == batch_id)
        if conditions:
            statement = statement.where(and_(*conditions))
        statement = statement.order_by(quarantined_rows.c.id)
        if limit is not None:
            statement = statement.limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            QuarantineRecord(
                **{
                    **{key: value for key, value in dict(row).items() if key != "id"},
                    "reason_codes": tuple(row["reason_codes"] or ()),
                    "warnings": tuple(row["warnings"] or ()),
                }
            )
            for row in rows
        ]

    def tweet_date_range(
        self,
        *,
        asset_symbol: str | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        statement = select(func.min(tweets.c.created_at), func.max(tweets.c.created_at))
        if asset_symbol is not None:
            statement = statement.where(tweets.c.asset_symbol == asset_symbol)
        with self.engine.connect() as connection:
            start, end = connection.execute(statement).one()
        return (
            as_utc(start) if start is not None else None,
            as_utc(end) if end is not None else None,
        )

    def price_date_range(self, asset_symbol: str) -> tuple[date | None, date | None]:
        statement = select(func.min(price_bars.c.bar_date), func.max(price_bars.c.bar_date)).where(
            price_bars.c.asset_symbol == asset_symbol
        )
        with self.engine.connect() as connection:
            start, end = connection.execute(statement).one()
        return start, end

    def list_assets(self) -> list[str]:
        """Distinct asset symbols present in the price table."""
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(price_bars.c.asset_symbol).distinct().order_by(price_bars.c.asset_symbol)
            ).scalars()
        return list(rows)


def repository_from_settings(settings: Any, *, ensure_schema: bool = True) -> CanonicalRepository:
    """Open a repository for the configured database URL."""
    from src.storage.engine import create_engine_from_settings

    return CanonicalRepository(
        create_engine_from_settings(settings),
        ensure_schema=ensure_schema,
    )


__all__ = ["CanonicalRepository", "UpsertStats", "repository_from_settings"]
