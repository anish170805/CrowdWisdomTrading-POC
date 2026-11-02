"""Daily sentiment aggregation (Module 7).

Converts post-level scores into one row of daily sentiment features per asset.

Each scored post needs: ``created_at_utc`` (tz-aware datetime; ``created_at`` is
accepted too), ``positive`` / ``negative`` / ``neutral`` probabilities, and
optionally ``asset_symbol`` and ``weight`` (engagement weight, default 1).

The output columns match ``DEFAULT_SENTIMENT_COLUMNS`` in ``src.features.dataset``.
Days with no posts get *no row* (never filled with neutral); the dataset builder
turns that into a null feature plus a ``sentiment_row_present_lag_*`` indicator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["OUTPUT_COLUMNS", "PREPARED_SENTIMENT_COLUMNS", "aggregate_daily_sentiment"]

OUTPUT_COLUMNS: tuple[str, ...] = (
    "asset_symbol",
    "bar_date",
    "post_count",
    "sentiment_mean",
    "sentiment_median",
    "positive_share",
    "negative_share",
    "neutral_share",
    "sentiment_std",
    "sentiment_weighted_mean",
    "missing_day_flag",
)


@dataclass(frozen=True)
class PREPARED_SENTIMENT_COLUMNS:  # noqa: N801 - public name kept for compatibility
    """The expected column names of the daily sentiment table."""

    post_count: str = "post_count"
    sentiment_mean: str = "sentiment_mean"
    sentiment_median: str = "sentiment_median"
    positive_share: str = "positive_share"
    negative_share: str = "negative_share"
    neutral_share: str = "neutral_share"
    sentiment_std: str = "sentiment_std"
    sentiment_weighted_mean: str = "sentiment_weighted_mean"
    missing_day_flag: str = "missing_day_flag"


def _bucket_date(created: Any, tzinfo: ZoneInfo) -> date | None:
    if isinstance(created, pd.Timestamp):
        created = created.to_pydatetime()
    if isinstance(created, datetime):
        if created.tzinfo is None:
            return created.date()
        return created.astimezone(tzinfo).date()
    if isinstance(created, date):
        return created
    return None


def aggregate_daily_sentiment(
    scored_posts: Sequence[Any],
    *,
    settings: Settings | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Aggregate post-level scores into daily features per asset.

    Posts without a usable timestamp are skipped (and counted in the log) rather
    than being dated "today", which would silently leak into the wrong day.
    """
    resolved = settings or get_settings()
    tzinfo = ZoneInfo(timezone or resolved.timezone)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for post in scored_posts:
        created = getattr(post, "created_at_utc", None) or getattr(post, "created_at", None)
        day = _bucket_date(created, tzinfo)
        if day is None:
            skipped += 1
            continue
        try:
            pos = float(getattr(post, "positive"))
            neg = float(getattr(post, "negative"))
            neu = float(getattr(post, "neutral"))
        except (AttributeError, TypeError, ValueError):
            skipped += 1
            continue
        weight = getattr(post, "weight", 1.0)
        rows.append(
            {
                "asset_symbol": str(getattr(post, "asset_symbol", None) or resolved.asset_symbol),
                "bar_date": day,
                "pos": pos,
                "neg": neg,
                "neu": neu,
                "weight": float(weight) if weight is not None and weight > 0 else 1.0,
            }
        )
    if skipped:
        logger.warning("Skipped %d scored post(s) without a usable timestamp or scores", skipped)

    if not rows:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    frame = pd.DataFrame(rows)
    frame["signed"] = frame["pos"] - frame["neg"]
    frame["weighted"] = frame["signed"] * frame["weight"]

    grouped = frame.groupby(["asset_symbol", "bar_date"], sort=True)
    daily = grouped.agg(
        post_count=("signed", "size"),
        sentiment_mean=("signed", "mean"),
        sentiment_median=("signed", "median"),
        positive_share=("pos", "mean"),
        negative_share=("neg", "mean"),
        neutral_share=("neu", "mean"),
        sentiment_std=("signed", lambda s: float(np.std(s.to_numpy(), ddof=0))),
        weighted_sum=("weighted", "sum"),
        weight_sum=("weight", "sum"),
    ).reset_index()
    daily["sentiment_weighted_mean"] = daily["weighted_sum"] / daily["weight_sum"]
    daily["missing_day_flag"] = 0
    daily = daily.drop(columns=["weighted_sum", "weight_sum"])

    numeric = [c for c in OUTPUT_COLUMNS if c not in {"asset_symbol", "bar_date", "post_count"}]
    daily[numeric] = daily[numeric].astype("float64").round(6)
    daily["post_count"] = daily["post_count"].astype("int64")
    daily["bar_date"] = pd.to_datetime(daily["bar_date"]).dt.date
    return daily.loc[:, list(OUTPUT_COLUMNS)].sort_values(
        ["asset_symbol", "bar_date"], kind="mergesort"
    ).reset_index(drop=True)
