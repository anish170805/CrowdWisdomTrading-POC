"""Quarantine schema for rows that fail canonical validation.

Invalid source rows are never dropped silently: each one is persisted with a
stable reason code, a human-readable message, and the raw payload reference so
an operator can inspect and repair it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from src.schemas.common import FrozenModel, ValidationIssue, as_utc


class QuarantineEntity(StrEnum):
    TWEET = "tweet"
    PRICE_BAR = "price_bar"


class QuarantineReason(StrEnum):
    """Stable reason codes recorded for quarantined rows."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_POST_ID = "invalid_post_id"
    EMPTY_TEXT = "empty_text"
    INVALID_TIMESTAMP = "invalid_timestamp"
    TIMESTAMP_OUT_OF_RANGE = "timestamp_out_of_range"
    INVALID_ENGAGEMENT = "invalid_engagement"
    INVALID_URL = "invalid_url"
    INVALID_ASSET_SYMBOL = "invalid_asset_symbol"
    INVALID_BAR_DATE = "invalid_bar_date"
    INVALID_PRICE = "invalid_price"
    INVALID_OHLC = "invalid_ohlc"
    INVALID_VOLUME = "invalid_volume"
    INVALID_PROVIDER = "invalid_provider"
    INVALID_RECORD = "invalid_record"


class QuarantineRecord(FrozenModel):
    """A persisted invalid row with provenance and reasons."""

    entity: QuarantineEntity
    record_key: str = Field(
        min_length=1,
        description=(
            "Best available identity for the row: post ID, asset/date, "
            "source record ID, or a hash of the raw payload."
        ),
    )
    reason_code: str = Field(min_length=1, description="Primary machine-readable reason.")
    reason: str = Field(min_length=1, description="Combined human-readable reasons.")
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)
    source: str | None = None
    source_record_id: str | None = None
    batch_id: str = Field(min_length=1)
    quarantined_at: datetime
    raw_payload_ref: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("quarantined_at", mode="before")
    @classmethod
    def _to_utc(cls, value: Any) -> Any:
        return as_utc(value)

    @classmethod
    def from_issues(
        cls,
        *,
        entity: QuarantineEntity,
        record_key: str,
        issues: list[ValidationIssue],
        batch_id: str,
        quarantined_at: datetime,
        source: str | None = None,
        source_record_id: str | None = None,
        raw_payload_ref: str | None = None,
        raw_payload: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> QuarantineRecord:
        """Build a record from one or more validation issues."""
        if not issues:
            raise ValueError("at least one validation issue is required")
        codes = tuple(dict.fromkeys(issue.code for issue in issues))
        return cls(
            entity=entity,
            record_key=record_key,
            reason_code=codes[0],
            reason="; ".join(f"{issue.code}: {issue.message}" for issue in issues),
            reason_codes=codes,
            source=source,
            source_record_id=source_record_id,
            batch_id=batch_id,
            quarantined_at=quarantined_at,
            raw_payload_ref=raw_payload_ref,
            raw_payload=raw_payload or {},
            warnings=warnings,
        )


__all__ = ["QuarantineEntity", "QuarantineReason", "QuarantineRecord"]
