"""Whitespace normalization and conservative post fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

_WHITESPACE_RE = re.compile(r"\s+")
_WHITESPACE_RUNS = re.compile(r"[ \t\r\n\f\v\u00a0]+")


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and trim the ends.

    Only whitespace is touched; characters, casing, punctuation, emoji, and
    URLs are left byte-for-byte intact.  The caller keeps the untouched source
    text in ``text_original``.
    """
    return _WHITESPACE_RUNS.sub(" ", text).strip()


def post_fingerprint(
    *,
    post_id: str | None,
    source: str,
    author_handle: str | None,
    created_at: datetime,
    text: str,
) -> str:
    """Return a stable SHA-256 deduplication key.

    When a platform post ID exists the fingerprint depends only on it.  Without
    an ID the fallback combines the source, author handle, exact UTC timestamp,
    and normalized text.  Matching text alone therefore never merges two
    distinct posts.
    """
    if post_id:
        payload = ["platform_post_id", post_id]
    else:
        payload = [
            "content_fallback_v1",
            source,
            author_handle or "",
            created_at.isoformat(),
            text,
        ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_fingerprint(payload: object) -> str:
    """Return a stable hash for a raw payload, used for quarantine identity."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def derive_source_record_id(source_file: str | None, source_row: int | None) -> str | None:
    """Return a compact source record identity such as ``export.csv:42``."""
    if source_file and source_row is not None:
        return f"{source_file}:{source_row}"
    if source_file:
        return source_file
    if source_row is not None:
        return str(source_row)
    return None


__all__ = [
    "derive_source_record_id",
    "normalize_whitespace",
    "post_fingerprint",
    "raw_fingerprint",
]
