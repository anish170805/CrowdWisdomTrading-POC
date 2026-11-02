"""Turn raw ingestion records into canonical :class:`TweetRecord` rows.

The normalizer is deliberately conservative: it only changes whitespace and
timezone representation, records every warning, and quarantines a row instead of
silently dropping or "fixing" questionable data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from src.ingestion.field_parsing import json_safe
from src.ingestion.records import RawTweetRecord
from src.normalization.outcomes import NormalizationOutcome
from src.normalization.text import (
    derive_source_record_id,
    normalize_whitespace,
    post_fingerprint,
    raw_fingerprint,
)
from src.schemas.common import ValidationIssue, as_utc
from src.schemas.quarantine import QuarantineEntity, QuarantineReason, QuarantineRecord
from src.schemas.tweets import MAX_POST_ID_LENGTH, TweetRecord

MIN_TWEET_CREATED_AT = datetime(2006, 1, 1, tzinfo=UTC)
ENGAGEMENT_FIELDS = ("likes", "reposts", "replies", "quotes", "views")
FALLBACK_RECORD_KEY_PREFIX = "tweet-"


def _field(raw: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _raw_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, (RawTweetRecord, TweetRecord)):
        payload = raw.model_dump(mode="json")
    elif isinstance(raw, Mapping):
        payload = json_safe(dict(raw))
    else:
        payload = {"repr": repr(raw)}
    return payload if isinstance(payload, dict) else {"value": payload}


def _record_key(
    raw: Any,
    *,
    post_id: str | None,
    source_record_id: str | None,
) -> str:
    if post_id:
        return f"post:{post_id}"
    if source_record_id:
        return f"source:{source_record_id}"
    return FALLBACK_RECORD_KEY_PREFIX + raw_fingerprint(_raw_payload(raw))


def _validate_post_id(value: Any, issues: list[ValidationIssue]) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_POST_ID,
                field="post_id",
                message="post ID is a boolean",
            )
        )
        return None
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_POST_ID,
                field="post_id",
                message=f"post ID has unsupported type {type(value).__name__}",
            )
        )
        return None
    if not text:
        return None
    if any(character.isspace() for character in text):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_POST_ID,
                field="post_id",
                message="post ID contains whitespace",
            )
        )
        return None
    if len(text) > MAX_POST_ID_LENGTH:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_POST_ID,
                field="post_id",
                message=f"post ID exceeds {MAX_POST_ID_LENGTH} characters",
            )
        )
        return None
    return text


def _validate_engagement(
    value: Any,
    *,
    field_name: str,
    issues: list[ValidationIssue],
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_ENGAGEMENT,
                field=field_name,
                message=f"{field_name} must be a whole number, got {value!r}",
            )
        )
        return None
    if value < 0:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_ENGAGEMENT,
                field=field_name,
                message=f"{field_name} must not be negative, got {value}",
            )
        )
        return None
    return value


def _validate_url(value: Any, issues: list[ValidationIssue]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_URL,
                field="url",
                message=f"URL {text!r} is not an absolute HTTP(S) URL",
            )
        )
        return None
    return text


def _validate_created_at(
    value: Any,
    *,
    ingested_at: datetime,
    issues: list[ValidationIssue],
    warnings: list[str],
) -> datetime | None:
    if not isinstance(value, datetime):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_TIMESTAMP,
                field="created_at",
                message=f"created_at is {type(value).__name__}, not a datetime",
            )
        )
        return None
    if value.tzinfo is None:
        warnings.append("created_at_naive_assumed_utc")
    created_at = as_utc(value)
    if created_at < MIN_TWEET_CREATED_AT:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.TIMESTAMP_OUT_OF_RANGE,
                field="created_at",
                message=f"created_at {created_at.isoformat()} predates X",
            )
        )
        return None
    if created_at > ingested_at + timedelta(days=1):
        issues.append(
            ValidationIssue(
                code=QuarantineReason.TIMESTAMP_OUT_OF_RANGE,
                field="created_at",
                message=f"created_at {created_at.isoformat()} is in the future",
            )
        )
        return None
    return created_at


def _issues_from_validation_error(exc: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for error in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in error.get("loc", ())) or None
        issues.append(
            ValidationIssue(
                code=QuarantineReason.INVALID_RECORD,
                field=loc,
                message=str(error.get("msg", "invalid value")).removeprefix("Value error, "),
            )
        )
    return issues or [
        ValidationIssue(
            code=QuarantineReason.INVALID_RECORD,
            message="record failed canonical validation",
        )
    ]


def normalize_tweet(
    raw: RawTweetRecord | TweetRecord | Mapping[str, Any],
    *,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
    raw_payload_ref: str | None = None,
    asset_symbol: str | None = None,
) -> NormalizationOutcome[TweetRecord]:
    """Normalize one source record or quarantine it with explicit reasons."""
    if isinstance(raw, TweetRecord):
        updates: dict[str, Any] = {}
        if batch_id:
            updates["batch_id"] = batch_id
        if asset_symbol and not raw.asset_symbol:
            updates["asset_symbol"] = asset_symbol
        if raw_payload_ref and not raw.raw_payload_ref:
            updates["raw_payload_ref"] = raw_payload_ref
        record = raw.model_copy(update=updates) if updates else raw
        return NormalizationOutcome(record=record)

    effective_batch = (
        batch_id
        or _field(raw, "batch_id")
        or _field(raw, "import_batch_id")
        or _field(raw, "collection_id")
    )
    if not effective_batch:
        raise ValueError("batch_id is required to normalize a tweet")
    effective_batch = str(effective_batch)

    now = as_utc(
        ingested_at or _field(raw, "imported_at") or _field(raw, "ingested_at") or datetime.now(UTC)
    )
    source = _field(raw, "source")
    source = str(source).strip() if source is not None else None
    source_record_id = _field(raw, "source_record_id") or derive_source_record_id(
        _field(raw, "source_file"),
        _field(raw, "source_row"),
    )
    payload = _raw_payload(raw)
    warnings: list[str] = list(_field(raw, "warnings") or [])
    issues: list[ValidationIssue] = []

    post_id = _validate_post_id(_field(raw, "post_id"), issues)
    original_text = _field(raw, "text")
    if not isinstance(original_text, str) or not original_text.strip():
        issues.append(
            ValidationIssue(
                code=QuarantineReason.EMPTY_TEXT,
                field="text",
                message="post text is empty",
            )
        )
        normalized_text = ""
    else:
        normalized_text = normalize_whitespace(original_text)
        if not normalized_text:
            issues.append(
                ValidationIssue(
                    code=QuarantineReason.EMPTY_TEXT,
                    field="text",
                    message="post text is only whitespace",
                )
            )

    created_at = _validate_created_at(
        _field(raw, "created_at_utc") or _field(raw, "created_at"),
        ingested_at=now,
        issues=issues,
        warnings=warnings,
    )

    engagement: dict[str, int | None] = {}
    for field_name in ENGAGEMENT_FIELDS:
        engagement[field_name] = _validate_engagement(
            _field(raw, field_name),
            field_name=field_name,
            issues=issues,
        )

    url = _validate_url(_field(raw, "url"), issues)

    if not source:
        issues.append(
            ValidationIssue(
                code=QuarantineReason.MISSING_REQUIRED_FIELD,
                field="source",
                message="source name is required",
            )
        )

    author = _field(raw, "author")
    author_handle = _field(raw, "author_handle")
    asset_value = _field(raw, "asset_symbol") or asset_symbol

    record_key = _record_key(raw, post_id=post_id, source_record_id=source_record_id)
    if issues:
        return NormalizationOutcome(
            quarantine=QuarantineRecord.from_issues(
                entity=QuarantineEntity.TWEET,
                record_key=record_key,
                issues=issues,
                batch_id=effective_batch,
                quarantined_at=now,
                source=source,
                source_record_id=source_record_id,
                raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
                raw_payload=payload,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        )

    if created_at is None:
        raise ValueError("created_at must be resolved before building a TweetRecord")
    fingerprint = post_fingerprint(
        post_id=post_id,
        source=source or "",
        author_handle=str(author_handle) if author_handle else None,
        created_at=created_at,
        text=normalized_text,
    )
    created_at_original = _field(raw, "created_at_original")
    if created_at_original is None:
        created_at_original = created_at.isoformat()
        warnings.append("created_at_original_missing")

    try:
        record = TweetRecord(
            post_id=post_id,
            post_fingerprint=fingerprint,
            asset_symbol=str(asset_value).strip() if asset_value else None,
            source=source or "",
            source_format=_field(raw, "source_format"),
            source_file=_field(raw, "source_file"),
            source_row=_field(raw, "source_row"),
            source_record_id=source_record_id,
            batch_id=effective_batch,
            raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
            ingested_at=now,
            text=normalized_text,
            text_original=original_text,
            created_at=created_at,
            created_at_original=str(created_at_original),
            source_timezone_offset=_field(raw, "source_timezone_offset"),
            timezone_assumed=bool(_field(raw, "timezone_assumed") or False),
            author=str(author) if author is not None else None,
            author_handle=str(author_handle) if author_handle is not None else None,
            **engagement,
            url=url,
            warnings=tuple(dict.fromkeys(warnings)),
        )
    except ValidationError as exc:
        return NormalizationOutcome(
            quarantine=QuarantineRecord.from_issues(
                entity=QuarantineEntity.TWEET,
                record_key=record_key,
                issues=_issues_from_validation_error(exc),
                batch_id=effective_batch,
                quarantined_at=now,
                source=source,
                source_record_id=source_record_id,
                raw_payload_ref=raw_payload_ref or _field(raw, "raw_payload_ref"),
                raw_payload=payload,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        )

    return NormalizationOutcome(record=record)


def normalize_tweets(
    records: Sequence[RawTweetRecord | TweetRecord | Mapping[str, Any]],
    *,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
    raw_payload_ref: str | None = None,
    asset_symbol: str | None = None,
) -> list[NormalizationOutcome[TweetRecord]]:
    """Normalize a batch of source records in input order."""
    return [
        normalize_tweet(
            record,
            batch_id=batch_id,
            ingested_at=ingested_at,
            raw_payload_ref=raw_payload_ref,
            asset_symbol=asset_symbol,
        )
        for record in records
    ]


__all__ = ["ENGAGEMENT_FIELDS", "MIN_TWEET_CREATED_AT", "normalize_tweet", "normalize_tweets"]
