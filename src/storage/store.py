"""High-level canonical store used by ingestion and downstream modules.

``CanonicalStore`` implements the Module 1/2 :class:`TweetSink` protocol: it
normalizes raw source records, quarantines invalid ones, and writes canonical
rows idempotently.  Module 4 uses the same store for price bars.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine

from src.config import Settings
from src.ingestion.records import RawTweetRecord
from src.logging_config import get_logger
from src.normalization.prices import normalize_price_bars
from src.normalization.tweets import normalize_tweets
from src.schemas.common import as_utc
from src.schemas.prices import PriceBar
from src.schemas.tweets import TweetRecord
from src.storage.engine import create_engine_from_settings
from src.storage.repository import CanonicalRepository, UpsertStats

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Summary of one canonical ingestion call."""

    entity: str
    batch_id: str
    received: int
    inserted: int
    updated: int = 0
    duplicates: int = 0
    quarantined: int = 0
    quarantine_reasons: Mapping[str, int] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def accepted(self) -> int:
        """Rows that became canonical (inserted or updated)."""
        return self.inserted + self.updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "batch_id": self.batch_id,
            "received": self.received,
            "inserted": self.inserted,
            "updated": self.updated,
            "duplicates": self.duplicates,
            "quarantined": self.quarantined,
            "quarantine_reasons": dict(self.quarantine_reasons),
            "ingested_at": as_utc(self.ingested_at).isoformat(),
        }


class CanonicalStore:
    """Normalize, validate, deduplicate, and persist canonical rows."""

    def __init__(self, repository: CanonicalRepository, *, owns_engine: bool = False) -> None:
        self.repository = repository
        self.engine: Engine = repository.engine
        self._owns_engine = owns_engine

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        ensure_schema: bool = True,
        echo: bool = False,
    ) -> CanonicalStore:
        engine = create_engine_from_settings(settings, echo=echo)
        repository = CanonicalRepository(engine, ensure_schema=ensure_schema)
        return cls(repository, owns_engine=True)

    def close(self) -> None:
        if self._owns_engine:
            self.engine.dispose()

    def __enter__(self) -> CanonicalStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Tweets (TweetSink protocol)
    # ------------------------------------------------------------------
    def ingest_tweets(
        self,
        records: Sequence[RawTweetRecord | TweetRecord | Mapping[str, Any]],
        *,
        batch_id: str | None = None,
        asset_symbol: str | None = None,
        raw_payload_ref: str | None = None,
        ingested_at: datetime | None = None,
    ) -> IngestResult:
        """Normalize and persist source posts; invalid rows are quarantined."""
        record_list = list(records)
        effective_batch = batch_id or _infer_batch(record_list)
        if not effective_batch:
            raise ValueError("batch_id is required to ingest tweets")
        now = as_utc(ingested_at or datetime.now(UTC))

        outcomes = normalize_tweets(
            record_list,
            batch_id=effective_batch,
            ingested_at=now,
            raw_payload_ref=raw_payload_ref,
            asset_symbol=asset_symbol,
        )
        canonical = [outcome.record for outcome in outcomes if outcome.record is not None]
        quarantine = [outcome.quarantine for outcome in outcomes if outcome.quarantine is not None]

        stats = self.repository.upsert_tweets(canonical)
        self.repository.insert_quarantine(quarantine)

        result = _build_result(
            entity="tweet",
            batch_id=effective_batch,
            received=len(record_list),
            stats=stats,
            quarantine=quarantine,
            ingested_at=now,
        )
        logger.info(
            "Canonical tweets ingested",
            extra={
                "event": "storage.tweets.ingested",
                "batch_id": effective_batch,
                **result.to_dict(),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Price bars
    # ------------------------------------------------------------------
    def ingest_price_bars(
        self,
        bars: Sequence[PriceBar | Mapping[str, Any]],
        *,
        batch_id: str | None = None,
        raw_payload_ref: str | None = None,
        ingested_at: datetime | None = None,
    ) -> IngestResult:
        """Normalize and upsert daily price bars."""
        bar_list = list(bars)
        effective_batch = batch_id or _infer_batch(bar_list)
        if not effective_batch:
            raise ValueError("batch_id is required to ingest price bars")
        now = as_utc(ingested_at or datetime.now(UTC))

        outcomes = normalize_price_bars(
            bar_list,
            batch_id=effective_batch,
            ingested_at=now,
            raw_payload_ref=raw_payload_ref,
        )
        canonical = [outcome.record for outcome in outcomes if outcome.record is not None]
        quarantine = [outcome.quarantine for outcome in outcomes if outcome.quarantine is not None]

        stats = self.repository.upsert_price_bars(canonical)
        self.repository.insert_quarantine(quarantine)

        result = _build_result(
            entity="price_bar",
            batch_id=effective_batch,
            received=len(bar_list),
            stats=stats,
            quarantine=quarantine,
            ingested_at=now,
        )
        logger.info(
            "Canonical price bars ingested",
            extra={
                "event": "storage.price_bars.ingested",
                "batch_id": effective_batch,
                **result.to_dict(),
            },
        )
        return result

    # Alias used by some callers for symmetry with ``ingest_tweets``.
    ingest_prices = ingest_price_bars


def _infer_batch(records: Sequence[Any]) -> str | None:
    for record in records:
        if isinstance(record, Mapping):
            value = record.get("batch_id") or record.get("import_batch_id")
        else:
            value = getattr(record, "batch_id", None) or getattr(record, "import_batch_id", None)
        if value:
            return str(value)
    return None


def _build_result(
    *,
    entity: str,
    batch_id: str,
    received: int,
    stats: UpsertStats,
    quarantine: Sequence[Any],
    ingested_at: datetime,
) -> IngestResult:
    reasons = Counter(row.reason_code for row in quarantine)
    return IngestResult(
        entity=entity,
        batch_id=batch_id,
        received=received,
        inserted=stats.inserted,
        updated=stats.updated,
        duplicates=stats.duplicates,
        quarantined=len(quarantine),
        quarantine_reasons=dict(sorted(reasons.items())),
        ingested_at=ingested_at,
    )


def open_store(settings: Settings, *, ensure_schema: bool = True) -> CanonicalStore:
    """Open a store for the configured database."""
    return CanonicalStore.from_settings(settings, ensure_schema=ensure_schema)


__all__ = ["CanonicalStore", "IngestResult", "open_store"]
