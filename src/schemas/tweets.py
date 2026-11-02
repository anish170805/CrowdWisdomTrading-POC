"""Canonical post schema shared by every tweet source.

A :class:`TweetRecord` is the *only* representation Modules 6-8 read.  Source
adapters keep their provider payloads; this schema keeps just enough provenance
to audit where a row came from and to make repeated imports idempotent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.schemas.common import FrozenModel, as_utc

MAX_POST_ID_LENGTH = 512


class TweetRecord(FrozenModel):
    """A normalized post with explicit provenance.

    ``text`` is whitespace-normalized while ``text_original`` preserves the
    source text exactly.  ``post_id`` is nullable because some exports omit the
    platform ID; in that case ``post_fingerprint`` is the conservative
    deduplication key.
    """

    # Identity and provenance ------------------------------------------------
    post_id: str | None = Field(
        default=None,
        description="Platform post ID. Unique when present.",
    )
    post_fingerprint: str = Field(
        min_length=16,
        max_length=64,
        description=(
            "Stable SHA-256 deduplication key. Derived from the post ID when "
            "available, otherwise from source + author + exact UTC timestamp + "
            "normalized text."
        ),
    )
    asset_symbol: str | None = Field(
        default=None,
        description="Asset this post was collected for, when the source provides it.",
    )
    source: str = Field(min_length=1, description="Logical source name.")
    source_format: str | None = Field(
        default=None,
        description="Source serialization: csv, tsv, xlsx, json, or jsonl.",
    )
    source_file: str | None = Field(default=None, description="Original export filename.")
    source_row: int | None = Field(
        default=None,
        ge=1,
        description="1-based row/record number in the source.",
    )
    source_record_id: str | None = Field(
        default=None,
        description="Source-specific record identifier (file:row when nothing better exists).",
    )
    batch_id: str = Field(
        min_length=1,
        description="Import or collection batch that first delivered this row.",
    )
    raw_payload_ref: str | None = Field(
        default=None,
        description="Path or URI of the immutable raw payload for this post.",
    )
    ingested_at: datetime = Field(description="When this module persisted the row (UTC).")

    # Content and source timestamps -----------------------------------------
    text: str = Field(min_length=1, description="Whitespace-normalized post text.")
    text_original: str = Field(description="Post text exactly as supplied by the source.")
    created_at: datetime = Field(description="Post creation time (UTC).")
    created_at_original: str = Field(description="Timestamp exactly as supplied by the source.")
    source_timezone_offset: str | None = Field(
        default=None,
        description="UTC offset of the source timestamp, e.g. +05:30.",
    )
    timezone_assumed: bool = Field(
        default=False,
        description="True when a naive source timestamp was interpreted in a configured zone.",
    )

    # Author and engagement --------------------------------------------------
    author: str | None = None
    author_handle: str | None = None
    likes: int | None = Field(default=None, ge=0)
    reposts: int | None = Field(default=None, ge=0)
    replies: int | None = Field(default=None, ge=0)
    quotes: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    url: str | None = None

    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal normalization warnings retained for audit.",
    )

    @field_validator("ingested_at", "created_at", mode="before")
    @classmethod
    def _to_utc(cls, value: Any) -> Any:
        if value is None:
            return value
        return as_utc(value)

    @field_validator("post_id", mode="before")
    @classmethod
    def _validate_post_id(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if any(character.isspace() for character in text):
            raise ValueError("post ID must not contain whitespace")
        if len(text) > MAX_POST_ID_LENGTH:
            raise ValueError(f"post ID must be at most {MAX_POST_ID_LENGTH} characters")
        return text

    @field_validator("asset_symbol", "source", "batch_id", mode="before")
    @classmethod
    def _required_text(cls, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_consistency(self) -> TweetRecord:
        if self.post_id is None and not self.post_fingerprint:
            raise ValueError("a post without a platform ID needs a fingerprint")
        if self.url is not None and not self.url.lower().startswith(("http://", "https://")):
            raise ValueError("url must be an absolute HTTP(S) URL")
        return self


__all__ = ["MAX_POST_ID_LENGTH", "TweetRecord"]
