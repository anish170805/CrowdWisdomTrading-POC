"""Outcome types returned by normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from src.schemas.quarantine import QuarantineRecord

RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class NormalizationOutcome(Generic[RecordT]):
    """Either a canonical record or a quarantine record, never both."""

    record: RecordT | None = None
    quarantine: QuarantineRecord | None = None

    @property
    def ok(self) -> bool:
        return self.record is not None and self.quarantine is None

    def __post_init__(self) -> None:
        if (self.record is None) == (self.quarantine is None):
            raise ValueError("exactly one of record or quarantine must be set")


__all__ = ["NormalizationOutcome"]
