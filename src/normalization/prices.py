"""Turn provider price records into canonical :class:`PriceBar` rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from typing import Any

from pydantic import ValidationError

from src.ingestion.field_parsing import json_safe, parse_datetime_text
from src.normalization.outcomes import NormalizationOutcome
from src.normalization.text import raw_fingerprint
from src.schemas.common import ValidationIssue, as_utc
from src.schemas.prices import PriceBar
from src.schemas.quarantine import QuarantineEntity, QuarantineReason, QuarantineRecord

_PRICE_FIELDS = ("open", "high", "low", "close")
_ALIASES: dict[str, tuple[str, ...]] = {
    "asset_symbol": ("asset_symbol", "symbol", "ticker", "asset"),
    "asset_type": ("asset_type", "asset_class"),
    "bar_date": ("bar_date", "date", "day"),
    "timestamp": ("timestamp", "timestamp_utc", "ts", "datetime"),
    "adjusted_close": ("adjusted_close", "adj_close", "adjclose", "adjusted"),
    "volume": ("volume", "vol"),
    "provider": ("provider", "data_provider", "source"),
    "retrieved_at": ("retrieved_at", "fetched_at", "downloaded_at"),
    "source_record_id": ("source_record_id", "record_id", "id"),
    "raw_payload_ref": ("raw_payload_ref", "raw_ref", "payload_ref"),
}


def _field(raw: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        for alias in _ALIASES.get(name, (name,)):
            if alias in raw:
                return raw[alias]
        return default
    if name in _ALIASES:
        for alias in _ALIASES[name]:
            if hasattr(raw, alias):
                return getattr(raw, alias)
        return default
    return getattr(raw, name, default)


def _raw_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, PriceBar):
        payload = raw.model_dump(mode="json")
    elif isinstance(raw, Mapping):
        payload = json_safe(dict(raw))
    else:
        payload = {"repr": repr(raw)}
    return payload if isinstance(payload, dict) else {"value": payload}


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return as_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = parse_datetime_text(value)
        except ValueError:
            return None
        return as_utc(parsed)
    return None


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return as_utc(value).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            parsed = _coerce_datetime(text)
            return parsed.date() if parsed is not None else None
    return None


def _coerce_number(
    value: Any,
    *,
    field_name: str,
    issues: list[ValidationIssue],
    positive: bool,
) -> float | None:
    if value is None:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.MISSING_REQUIRED_FIELD,
                field=field_name,
                message=f"{field_name} is required",
            )
        )
        return None
    if isinstance(value, bool):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_PRICE,
                field=field_name,
                message=f"{field_name} must be a number, not a boolean",
            )
        )
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_PRICE,
                field=field_name,
                message=f"{field_name} {value!r} is not a number",
            )
        )
        return None
    if not math.isfinite(number):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_PRICE,
                field=field_name,
                message=f"{field_name} {value!r} is not finite",
            )
        )
        return None
    if positive and number <= 0:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_PRICE,
                field=field_name,
                message=f"{field_name} must be positive, got {number}",
            )
        )
        return None
    if not positive and number < 0:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_VOLUME,
                field=field_name,
                message=f"{field_name} must not be negative, got {number}",
            )
        )
        return None
    return number


def _issues_from_validation_error(exc: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for error in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in error.get("loc", ())) or None
        message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
        code = QuarantineReason.INVALID_RECORD
        if loc in _PRICE_FIELDS or loc == "adjusted_close":
            code = QuarantineReason.INVALID_PRICE
        elif loc == "volume":
            code = QuarantineReason.INVALID_VOLUME
        elif loc in {"high", "low"} or "high" in message or "low" in message:
            code = QuarantineReason.INVALID_OHLC
        elif loc == "bar_date":
            code = QuarantineReason.INVALID_BAR_DATE
        issues.append(ValidationIssue(code=code, field=loc, message=message))
    return issues or [
        ValidationIssue(
            code=QuarantineReason.INVALID_RECORD,
            message="price bar failed canonical validation",
        )
    ]


def normalize_price_bar(
    raw: PriceBar | Mapping[str, Any],
    *,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
    raw_payload_ref: str | None = None,
) -> NormalizationOutcome[PriceBar]:
    """Normalize and validate one provider bar or quarantine it."""
    if isinstance(raw, PriceBar):
        updates: dict[str, Any] = {}
        if batch_id:
            updates["batch_id"] = batch_id
        if raw_payload_ref and not raw.raw_payload_ref:
            updates["raw_payload_ref"] = raw_payload_ref
        bar = raw.model_copy(update=updates) if updates else raw
        return NormalizationOutcome(record=bar)

    now = as_utc(ingested_at or datetime.now(UTC))
    effective_batch = (
        batch_id
        or _field(raw, "batch_id")
        or _field(raw, "import_batch_id")
        or _field(raw, "retrieval_batch_id")
    )
    if not effective_batch:
        raise ValueError("batch_id is required to normalize a price bar")
    effective_batch = str(effective_batch)

    payload = _raw_payload(raw)
    warnings: list[str] = list(_field(raw, "warnings") or [])
    issues: list[ValidationIssue] = []

    asset_symbol = _field(raw, "asset_symbol")
    asset_text = str(asset_symbol).strip() if asset_symbol is not None else ""
    if not asset_text or any(character.isspace() for character in asset_text):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_ASSET_SYMBOL,
                field="asset_symbol",
                message="asset symbol is required and must not contain whitespace",
            )
        )

    timestamp = _coerce_datetime(_field(raw, "timestamp"))
    bar_date = _coerce_date(_field(raw, "bar_date"))
    if timestamp is None and bar_date is None:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_BAR_DATE,
                field="bar_date",
                message="bar_date or timestamp is required",
            )
        )
    elif timestamp is None:
        if bar_date is None:
            raise ValueError("bar_date or timestamp is required")
        timestamp = datetime.combine(bar_date, time.min, tzinfo=UTC)
    elif bar_date is None:
        bar_date = timestamp.date()
    elif timestamp.date() != bar_date:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_BAR_DATE,
                field="bar_date",
                message=(
                    f"bar_date {bar_date.isoformat()} does not match timestamp "
                    f"{timestamp.isoformat()}"
                ),
            )
        )

    provider = _field(raw, "provider")
    provider_text = str(provider).strip() if provider is not None else ""
    if not provider_text:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_PROVIDER,
                field="provider",
                message="provider is required",
            )
        )

    prices = {
        field_name: _coerce_number(
            _field(raw, field_name),
            field_name=field_name,
            issues=issues,
            positive=True,
        )
        for field_name in _PRICE_FIELDS
    }
    adjusted_close = _field(raw, "adjusted_close")
    adjusted_value = (
        _coerce_number(
            adjusted_close,
            field_name="adjusted_close",
            issues=issues,
            positive=True,
        )
        if adjusted_close is not None
        else None
    )
    volume = _field(raw, "volume")
    volume_value = (
        _coerce_number(volume, field_name="volume", issues=issues, positive=False)
        if volume is not None
        else None
    )

    if all(prices[field_name] is not None for field_name in _PRICE_FIELDS):
        high = prices["high"]
        low = prices["low"]
        open_price = prices["open"]
        close = prices["close"]
        if high is not None and low is not None and open_price is not None and close is not None:
            if high < low:
                issues.append(
                    ValidationIssue(
                        code=QuarantineReason.INVALID_OHLC,
                        field="high",
                        message=f"high {high} is below low {low}",
                    )
                )
            if high < max(open_price, close):
                issues.append(
                    ValidationIssue(
                        code=QuarantineReason.INVALID_OHLC,
                        field="high",
                        message="high is below open or close",
                    )
                )
            if low > min(open_price, close):
                issues.append(
                    ValidationIssue(
                        code=QuarantineReason.INVALID_OHLC,
                        field="low",
                        message="low is above open or close",
                    )
                )

    record_key = (
        f"{asset_text}:{bar_date.isoformat()}"
        if asset_text and bar_date is not None
        else (_field(raw, "source_record_id") or "price-" + raw_fingerprint(payload))
    )

    if issues:
        return NormalizationOutcome(
            quarantine=QuarantineRecord.from_issues(
                entity=QuarantineEntity.PRICE_BAR,
                record_key=str(record_key),
                issues=issues,
                batch_id=effective_batch,
                quarantined_at=now,
                source=provider_text or None,
                source_record_id=_field(raw, "source_record_id"),
                raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
                raw_payload=payload,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        )

    if bar_date is None or timestamp is None:
        raise ValueError("bar_date and timestamp must be resolved before building a PriceBar")
    try:
        bar = PriceBar(
            asset_symbol=asset_text,
            asset_type=_field(raw, "asset_type"),
            bar_date=bar_date,
            timestamp=timestamp,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            adjusted_close=adjusted_value,
            volume=volume_value,
            provider=provider_text,
            batch_id=effective_batch,
            source_record_id=_field(raw, "source_record_id"),
            raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
            retrieved_at=_coerce_datetime(_field(raw, "retrieved_at")),
            ingested_at=now,
            warnings=tuple(dict.fromkeys(warnings)),
        )
    except ValidationError as exc:
        return NormalizationOutcome(
            quarantine=QuarantineRecord.from_issues(
                entity=QuarantineEntity.PRICE_BAR,
                record_key=str(record_key),
                issues=_issues_from_validation_error(exc),
                batch_id=effective_batch,
                quarantined_at=now,
                source=provider_text or None,
                source_record_id=_field(raw, "source_record_id"),
                raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
                raw_payload=payload,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        )

    return NormalizationOutcome(record=bar)


def normalize_price_bars(
    bars: Sequence[PriceBar | Mapping[str, Any]],
    *,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
    raw_payload_ref: str | None = None,
) -> list[NormalizationOutcome[PriceBar]]:
    """Normalize a batch of provider bars in input order."""
    return [
        normalize_price_bar(
            bar,
            batch_id=batch_id,
            ingested_at=ingested_at,
            raw_payload_ref=raw_payload_ref,
        )
        for bar in bars
    ]


__all__ = ["normalize_price_bar", "normalize_price_bars"]
