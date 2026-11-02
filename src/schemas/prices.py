"""Canonical daily price-bar schema.

Module 4 fetches provider data; this schema is the single representation stored
by Module 3 and read by Modules 8-13.  Bars are daily and asset-scoped: one row
per ``(asset_symbol, timestamp)``.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.schemas.common import FrozenModel, as_utc

_PRICE_EPSILON = 1e-9


class PriceBar(FrozenModel):
    """A validated daily OHLCV bar for one asset.

    ``timestamp`` is the UTC instant the bar starts (midnight UTC for daily
    data).  ``bar_date`` is the corresponding UTC calendar date, which is the
    key Module 8 uses for joins.
    """

    asset_symbol: str = Field(min_length=1, description="Instrument symbol, e.g. AAPL or BTC-USD.")
    asset_type: str | None = Field(
        default=None,
        description="Asset class: equity, crypto, fx, commodity, or index.",
    )
    bar_date: date = Field(description="UTC calendar date of the bar.")
    timestamp: datetime = Field(description="UTC instant the bar starts.")
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float | None = Field(
        default=None,
        description="Provider-adjusted close when available.",
    )
    volume: float | None = Field(
        default=None,
        description="Traded volume; None when unavailable.",
    )
    provider: str = Field(min_length=1, description="Price data provider name.")
    batch_id: str = Field(
        min_length=1,
        description="Retrieval/import batch that delivered the bar.",
    )
    source_record_id: str | None = Field(
        default=None,
        description="Provider-specific record identifier when available.",
    )
    raw_payload_ref: str | None = Field(
        default=None,
        description="Path or URI of the immutable provider payload.",
    )
    retrieved_at: datetime | None = Field(
        default=None,
        description="When the provider returned the bar (UTC).",
    )
    ingested_at: datetime = Field(description="When this module persisted the row (UTC).")
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("asset_symbol", "provider", "batch_id", mode="before")
    @classmethod
    def _required_text(cls, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        if any(character.isspace() for character in text):
            raise ValueError("must not contain whitespace")
        return text

    @field_validator("asset_type", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @field_validator("timestamp", "retrieved_at", "ingested_at", mode="before")
    @classmethod
    def _to_utc(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_utc(value)

    @field_validator("open", "high", "low", "close", "adjusted_close", mode="before")
    @classmethod
    def _validate_price_number(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("price must be a number, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"price {value!r} is not a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"price {value!r} is not finite")
        if number <= 0:
            raise ValueError(f"price {value!r} must be positive")
        return number

    @field_validator("volume", mode="before")
    @classmethod
    def _validate_volume_number(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("volume must be a number, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"volume {value!r} is not a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"volume {value!r} is not finite")
        if number < 0:
            raise ValueError("volume must not be negative")
        return number

    @model_validator(mode="after")
    def _validate_bar(self) -> PriceBar:
        if self.bar_date != as_utc(self.timestamp).date():
            raise ValueError("bar_date must match the UTC date of timestamp")
        if self.high + _PRICE_EPSILON < self.low:
            raise ValueError("high must be greater than or equal to low")
        if self.high + _PRICE_EPSILON < max(self.open, self.close):
            raise ValueError("high must be greater than or equal to open and close")
        if self.low - _PRICE_EPSILON > min(self.open, self.close):
            raise ValueError("low must be less than or equal to open and close")
        return self


__all__ = ["PriceBar"]
