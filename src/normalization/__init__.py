"""Canonical normalization for source posts and price bars.

Normalizers validate row-by-row, preserve source text/timestamps, and return a
:class:`NormalizationOutcome` that is either a canonical record or a quarantine
record with stable reason codes.
"""

from __future__ import annotations

from src.normalization.outcomes import NormalizationOutcome
from src.normalization.prices import normalize_price_bar, normalize_price_bars
from src.normalization.text import (
    derive_source_record_id,
    normalize_whitespace,
    post_fingerprint,
    raw_fingerprint,
)
from src.normalization.tweets import normalize_tweet, normalize_tweets

__all__ = [
    "NormalizationOutcome",
    "derive_source_record_id",
    "normalize_price_bar",
    "normalize_price_bars",
    "normalize_tweet",
    "normalize_tweets",
    "normalize_whitespace",
    "post_fingerprint",
    "raw_fingerprint",
]
