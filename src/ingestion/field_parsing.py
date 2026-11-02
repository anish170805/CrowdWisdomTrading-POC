"""Shared value parsing and file helpers for tweet ingestion adapters.

Module 1 (historical exports) and Module 2 (live Apify collection) both need to
turn heterogeneous source values into the canonical raw-record fields.  Keeping
the helpers here means the two adapters converge on the same rules instead of
drifting apart.

Nothing in this module knows about a specific source, actor, or export format.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dateutil import parser as date_parser

EMPTY_VALUES = frozenset({"", "-", "--", "n/a", "na", "null", "none", "nan", "nil"})


@dataclass(frozen=True)
class ParsedTimestamp:
    """A source timestamp after timezone-aware normalization."""

    utc: datetime
    original: str
    offset: str | None
    assumed: bool


def normalise_key(value: Any) -> str:
    """Lower-case a header/field name and collapse non-alphanumerics to ``_``."""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def is_present(value: Any) -> bool:
    """Return True when a source value is not missing or blank."""
    if value is None:
        return False
    return not (isinstance(value, str) and not value.strip())


def as_text(value: Any) -> str:
    """Render a scalar or nested value as post text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "full_text", "content", "body", "value"):
            if key in value:
                candidate = as_text(value[key])
                if candidate.strip():
                    return candidate
        return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(item) for item in value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def clean_post_id(
    value: Any,
    *,
    warn_numeric_precision: bool = False,
) -> tuple[str | None, list[str]]:
    """Normalize a post ID, deriving it from a status URL when necessary."""
    if value is None:
        return None, []
    if isinstance(value, dict):
        for key in ("id_str", "id", "post_id", "tweet_id", "status_id", "rest_id"):
            if value.get(key) not in (None, ""):
                return clean_post_id(
                    value[key],
                    warn_numeric_precision=warn_numeric_precision,
                )
        raise ValueError("post ID is an object without a recognised ID field")
    if isinstance(value, list):
        raise ValueError("post ID is a list, not a scalar")
    if isinstance(value, bool):
        raise ValueError("post ID is a boolean")
    if isinstance(value, int):
        text = str(value)
        numeric_source = True
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"post ID {value!r} is not a whole number")
        text = str(int(value))
        numeric_source = True
    else:
        text = str(value).strip()
        numeric_source = re.fullmatch(r"\d+\.0+", text) is not None
        if numeric_source:
            text = text.split(".", 1)[0]

    if not text or text.lower() in EMPTY_VALUES:
        return None, []

    derived = post_id_from_url(text)
    if derived is not None:
        return derived, ["post_id_derived_from_url"]
    if text.lower().startswith(("http://", "https://")):
        raise ValueError(f"post ID {text!r} is a URL without a /status/<id> segment")
    if any(character.isspace() for character in text):
        raise ValueError(f"post ID {text!r} contains whitespace")
    if len(text) > 512:
        raise ValueError("post ID is unreasonably long")

    warnings_list: list[str] = []
    if warn_numeric_precision and numeric_source and len(text) > 15:
        warnings_list.append("post_id_precision_risk")
    return text, warnings_list


def post_id_from_url(value: Any) -> str | None:
    """Extract the numeric status ID from an X/Twitter URL."""
    if not isinstance(value, str):
        return None
    match = re.search(r"/status(?:es)?/(\d+)", value)
    return match.group(1) if match else None


def parse_count(value: Any, *, field_name: str) -> int | None:
    """Parse an engagement count, accepting separators and K/M/B suffixes."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} is a boolean, not a count")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} {value!r} is not a whole number")
        number = int(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in EMPTY_VALUES:
            return None
        text = re.sub(r"(?<=\d)[ \u00a0_](?=\d)", "", text)
        match = re.fullmatch(
            r"([+-]?\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*([kKmMbB])?",
            text,
        )
        if match is None:
            raise ValueError(f"{field_name} cannot parse {value!r} as a count")
        whole = match.group(1).replace(",", "")
        fraction = match.group(2) or ""
        suffix = (match.group(3) or "").lower()
        multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
        if fraction:
            try:
                decimal_value = Decimal(f"{whole}.{fraction}") * multiplier
            except InvalidOperation as exc:
                raise ValueError(f"{field_name} cannot parse {value!r} as a count") from exc
            if decimal_value != decimal_value.to_integral_value():
                raise ValueError(
                    f"{field_name} {value!r} is not a whole number after applying the suffix"
                )
            number = int(decimal_value)
        else:
            number = int(whole) * multiplier

    if number < 0:
        raise ValueError(f"{field_name} must not be negative")
    return number


def normalise_url(value: Any) -> tuple[str | None, str | None]:
    """Return a normalized HTTP(S) URL and an optional warning code."""
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text.lower() in EMPTY_VALUES:
        return None, None
    if text.startswith("//"):
        text = f"https:{text}"
    elif not re.match(r"^[a-z][a-z0-9+.-]*://", text, flags=re.IGNORECASE):
        host = text.split("/", 1)[0]
        if "." in host:
            text = f"https://{text}"
        else:
            return None, "not_a_url"

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, "not_a_url"
    return text, None


def normalise_author(value: Any) -> tuple[str | None, str | None]:
    """Return ``(display_name, handle)`` from an author scalar or object."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        for key in (
            "screen_name",
            "screenName",
            "username",
            "userName",
            "user_name",
            "handle",
            "name",
        ):
            if value.get(key):
                return normalise_author(value[key])
        return None, None
    text = str(value).strip()
    if not text or text.lower() in EMPTY_VALUES:
        return None, None
    if text.lower().startswith(("http://", "https://")):
        parsed = urlparse(text)
        candidate = parsed.path.strip("/").split("/", 1)[0]
        if candidate:
            text = candidate
    if text.startswith("@"):
        handle = text[1:].strip()
        return text, handle or None
    if " " not in text:
        return text, text
    return text, None


def parse_timestamp(
    value: Any,
    *,
    assume_timezone: tzinfo,
    allow_excel_serial: bool,
) -> ParsedTimestamp:
    """Parse a timestamp while preserving its original representation."""
    if isinstance(value, datetime):
        parsed = value
        original = value.isoformat()
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
        original = value.isoformat()
    elif isinstance(value, bool):
        raise ValueError("timestamp is a boolean")
    elif isinstance(value, (int, float)):
        parsed = timestamp_from_number(float(value), allow_excel_serial=allow_excel_serial)
        original = str(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("timestamp is empty")
        original = text
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
            number = float(text)
            integer_part = text.lstrip("+-").split(".", 1)[0]
            if allow_excel_serial and 20_000 <= number <= 80_000:
                parsed = timestamp_from_number(number, allow_excel_serial=True)
            elif len(integer_part) >= 10:
                parsed = timestamp_from_number(number, allow_excel_serial=False)
            else:
                raise ValueError(f"ambiguous numeric timestamp {text!r}")
        else:
            parsed = parse_datetime_text(text)

    assumed = parsed.tzinfo is None
    if assumed:
        parsed = parsed.replace(tzinfo=assume_timezone)
    return ParsedTimestamp(
        utc=parsed.astimezone(UTC),
        original=original,
        offset=format_offset(parsed),
        assumed=assumed,
    )


def parse_datetime_text(text: str) -> datetime:
    """Parse ISO 8601, RFC 2822, and X-style date strings."""
    normalised = text.strip()
    if normalised.upper().endswith(" UTC"):
        normalised = f"{normalised[:-4]}+00:00"
    try:
        return datetime.fromisoformat(normalised.replace("Z", "+00:00"))
    except ValueError:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return date_parser.parse(text, fuzzy=False)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"cannot parse timestamp {text!r}") from exc


def timestamp_from_number(value: float, *, allow_excel_serial: bool) -> datetime:
    """Interpret an epoch (seconds/millis/micros/nanos) or Excel serial number."""
    if allow_excel_serial and 20_000 <= value <= 80_000:
        # Excel's serial epoch is 1899-12-30 (it preserves Lotus 1-2-3's
        # fictitious 1900-02-29 leap day).
        return datetime(1899, 12, 30, tzinfo=UTC) + timedelta(days=value)

    magnitude = abs(value)
    if magnitude < 1_000_000_000:
        raise ValueError(f"ambiguous numeric timestamp {value!r}")
    if magnitude >= 1e17:
        seconds = value / 1e9
    elif magnitude >= 1e14:
        seconds = value / 1e6
    elif magnitude >= 1e11:
        seconds = value / 1e3
    else:
        seconds = value
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"timestamp {value!r} is out of range") from exc


def format_offset(value: datetime) -> str | None:
    """Format a datetime's UTC offset as ``+HH:MM``."""
    offset = value.utcoffset()
    if offset is None:
        return None
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def json_safe(value: Any) -> Any:
    """Recursively convert a value into JSON-serialisable primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def hash_file(path: Path) -> tuple[str, int]:
    """Return the SHA-256 hex digest and byte size of a file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def now_utc(clock: Callable[[], datetime] | None = None) -> datetime:
    """Return an aware UTC timestamp, optionally from an injected clock."""
    value = clock() if clock is not None else datetime.now(tz=UTC)
    if not isinstance(value, datetime):
        raise ValueError("clock must return a datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def safe_component(value: str) -> str:
    """Make a value safe for use as a single filesystem path component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe[:128] or "batch"


def safe_filename(value: str) -> str:
    """Make a value safe for use as a filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe[:180] or "source"


def write_jsonl(path: Path, items: Iterable[Any]) -> None:
    """Atomically write newline-delimited JSON."""
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(
                    json.dumps(
                        json_safe(item),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    """Atomically write an indented JSON document."""
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def date_range(records: Iterable[Any]) -> dict[str, str | None]:
    """Return the min/max ``created_at_utc`` across raw records."""
    timestamps = [record.created_at_utc for record in records]
    if not timestamps:
        return {"start": None, "end": None}
    return {
        "start": min(timestamps).isoformat(),
        "end": max(timestamps).isoformat(),
    }


__all__ = [
    "EMPTY_VALUES",
    "ParsedTimestamp",
    "as_text",
    "clean_post_id",
    "date_range",
    "format_offset",
    "hash_file",
    "is_present",
    "json_safe",
    "normalise_author",
    "normalise_key",
    "normalise_url",
    "now_utc",
    "parse_count",
    "parse_datetime_text",
    "parse_timestamp",
    "post_id_from_url",
    "safe_component",
    "safe_filename",
    "timestamp_from_number",
    "write_json",
    "write_jsonl",
]
