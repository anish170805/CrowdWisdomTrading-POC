"""Shared raw-record contracts for tweet ingestion sources.

Module 1 (historical exports) and Module 2 (live Apify collection) both produce
:class:`RawTweetRecord` objects.  Module 3 owns canonical normalization and
storage; it consumes these records through the :class:`TweetSink` protocol.

Keeping the contract here means the two ingestion adapters converge on one
shape without either module importing the other's implementation details.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RawTweetRecord(BaseModel):
    """A source post with provenance and the untouched source payload.

    ``created_at_utc`` is always timezone-aware UTC.  ``created_at_original``
    preserves the source representation (including any offset) so downstream
    code can audit the conversion.  ``raw`` retains every source field, even
    columns the importer does not understand.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(description="Logical source name, e.g. historical_export.")
    source_format: str = Field(description="csv, tsv, xlsx, json, or jsonl.")
    source_file: str = Field(description="Original export filename.")
    source_row: int | None = Field(
        default=None,
        description="1-based row/record number in the source, when available.",
    )
    import_batch_id: str
    imported_at: datetime

    post_id: str | None = Field(
        default=None,
        description="Stable platform post ID when present; None triggers a downstream fingerprint.",
    )
    text: str = Field(description="Original post text with only outer whitespace trimmed.")
    created_at_utc: datetime
    created_at_original: str = Field(description="Timestamp exactly as supplied by the source.")
    source_timezone_offset: str | None = Field(
        default=None,
        description="UTC offset used by the source timestamp, e.g. +05:30.",
    )
    timezone_assumed: bool = Field(
        default=False,
        description="True when a naive timestamp was interpreted in the configured timezone.",
    )

    author: str | None = None
    author_handle: str | None = None
    likes: int | None = None
    reposts: int | None = None
    replies: int | None = None
    quotes: int | None = None
    views: int | None = None
    url: str | None = None

    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class QuarantinedTweetRow(BaseModel):
    """A source row that could not be turned into a :class:`RawTweetRecord`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_row: int | None = None
    reason_code: str
    reason: str
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class TweetSink(Protocol):
    """Downstream canonical-storage interface implemented by Module 3.

    ``batch_id`` is the same provenance identifier recorded in every
    :class:`RawTweetRecord`, so the sink can make repeated imports idempotent.
    """

    def ingest_tweets(self, records: Sequence[RawTweetRecord], *, batch_id: str) -> Any:
        """Normalize, deduplicate, and persist ``records``; return any result."""


__all__ = ["QuarantinedTweetRow", "RawTweetRecord", "TweetSink"]
