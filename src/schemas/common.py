"""Shared building blocks for canonical schemas.

Canonical timestamps are always timezone-aware UTC.  This module centralises the
conversion so every schema and normalizer applies the same rule: a naive value
is interpreted as UTC (the source adapters should already have resolved local
timezones), an aware value is converted to UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field


def as_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are treated as UTC rather than silently inheriting the
    machine's local timezone.  Callers that need to reject naive values should
    check ``value.tzinfo is None`` first.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"expected a datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def as_utc_date(value: datetime | date) -> date:
    """Return the UTC calendar date of ``value``."""
    if isinstance(value, datetime):
        return as_utc(value).date()
    if isinstance(value, date):
        return value
    raise TypeError(f"expected a date or datetime, got {type(value).__name__}")


class ValidationIssue(BaseModel):
    """A single reason a source row could not become a canonical row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(description="Stable machine-readable reason code.")
    message: str = Field(description="Human-readable explanation.")
    field: str | None = Field(
        default=None,
        description="Canonical field involved, when the issue is field-specific.",
    )


class FrozenModel(BaseModel):
    """Base class for immutable canonical records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


__all__ = ["FrozenModel", "ValidationIssue", "as_utc", "as_utc_date"]
