"""Leakage-safe daily modelling table for Module 8.

Joins the Module 4 price bars with the Module 7 daily sentiment table into one
row per ``(asset_symbol, bar_date)``, adds ``n+1`` ... ``n+k`` targets, and emits
a leakage audit with the dataset.

Conventions
-----------
* ``bar_date`` is the ``n+0`` date: the latest *completed* observed close.  It is
  also the feature cutoff.  ``n+0`` is never a target.
* Horizon ``h`` means the ``h``-th *next available bar* (trading session) for the
  same asset.  Non-trading days are not filled; a target that would span a data
  gap wider than ``max_gap_days`` is left null and counted in the audit.
* Targets: ``target_return_n{h}`` (simple return from ``close_n0``),
  ``target_close_n{h}`` and the metadata date ``target_date_n{h}``.
* Sentiment (Module 7) is joined as *calendar-day lags*: the feature
  ``<col>_lag_L`` on row ``t`` is the sentiment bucket dated ``t - L`` days.
  Lag 0 (same-day bucket) is refused unless the caller asserts that Module 7's
  bucket cutoff precedes the prediction cutoff (``allow_same_day_sentiment``).
  Missing sentiment stays null (never filled with neutral) and a coverage
  indicator ``sentiment_row_present_lag_L`` distinguishes "no row" from a null
  value inside a present row.
* Raw targets and metadata live in separate columns from features; the feature
  matrix is projected through an explicit allowlist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.features.price_features import (
    PriceFeatureConfig,
    bars_to_frame,
    build_price_features,
    normalize_dates,
    price_feature_specs,
)
from src.validation.leakage import (
    CLOSE_N0,
    KEY_COLUMNS,
    SENTIMENT_FAMILY,
    TARGET_CLOSE_FMT,
    TARGET_DATE_FMT,
    TARGET_RETURN_FMT,
    FeatureSpec,
    LeakageError,
    assert_audit_passed,
    audit_to_json,
    build_leakage_audit,
    check_features_exclude_targets,
    check_unique_keys,
    find_news_columns,
    to_jsonable,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.schemas.prices import PriceBar

DATASET_SCHEMA_VERSION = 1
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Module 7 output contract (column names of the daily sentiment table).  Update
# this tuple, or pass ``DatasetConfig(sentiment_columns=...)``, if Module 7 names
# its columns differently.  Any column outside this tuple is rejected.
DEFAULT_SENTIMENT_COLUMNS: tuple[str, ...] = (
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


class DatasetBuildError(ValueError):
    """Raised for unusable inputs or configuration (not for leakage violations)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """All parameters that determine the modelling table."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    price: PriceFeatureConfig = field(default_factory=PriceFeatureConfig)
    sentiment_lags: tuple[int, ...] = (1, 2, 3)
    sentiment_columns: tuple[str, ...] = DEFAULT_SENTIMENT_COLUMNS
    allow_same_day_sentiment: bool = False
    market_timezone: str = "UTC"
    sentiment_bucket_timezone: str | None = None
    max_gap_days: int = 7
    drop_warmup_rows: bool = True
    verify_truncation: bool = True
    feature_allowlist: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        horizons = tuple(sorted({int(h) for h in self.horizons}))
        if not horizons:
            raise ValueError("at least one horizon is required")
        if horizons[0] < 1:
            raise ValueError(
                "horizons must be >= 1: n+0 is the latest completed close, not a forecast"
            )
        lags = tuple(sorted({int(lag) for lag in self.sentiment_lags}))
        if lags and lags[0] < 0:
            raise ValueError("sentiment_lags must be >= 0")
        if lags and lags[0] == 0 and not self.allow_same_day_sentiment:
            raise ValueError(
                "sentiment lag 0 uses the same-day bucket; set allow_same_day_sentiment=True "
                "only if Module 7's bucket cutoff is before the prediction cutoff"
            )
        if int(self.max_gap_days) < 1:
            raise ValueError("max_gap_days must be >= 1")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "sentiment_lags", lags)
        object.__setattr__(self, "sentiment_columns", tuple(self.sentiment_columns))
        object.__setattr__(self, "max_gap_days", int(self.max_gap_days))
        if self.feature_allowlist is not None:
            object.__setattr__(self, "feature_allowlist", tuple(self.feature_allowlist))

    @property
    def min_sentiment_offset(self) -> int:
        return 0 if self.allow_same_day_sentiment else 1

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> DatasetConfig:
        """Build from the Module 0 ``Settings`` (horizons and market timezone)."""
        params: dict[str, Any] = {
            "horizons": tuple(settings.horizons),
            "market_timezone": settings.timezone,
        }
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "horizons": self.horizons,
                "price": self.price.to_dict(),
                "sentiment_lags": self.sentiment_lags,
                "sentiment_columns": self.sentiment_columns,
                "allow_same_day_sentiment": self.allow_same_day_sentiment,
                "market_timezone": self.market_timezone,
                "sentiment_bucket_timezone": self.sentiment_bucket_timezone,
                "max_gap_days": self.max_gap_days,
                "drop_warmup_rows": self.drop_warmup_rows,
                "feature_allowlist": self.feature_allowlist,
            }
        )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def target_columns_for(horizons: Sequence[int]) -> tuple[str, ...]:
    """Raw target columns, ordered by horizon."""
    return tuple(
        name
        for h in horizons
        for name in (TARGET_RETURN_FMT.format(h=h), TARGET_CLOSE_FMT.format(h=h))
    )


def metadata_columns_for(horizons: Sequence[int]) -> tuple[str, ...]:
    """Reference close plus the target dates (timestamps, not features)."""
    return (CLOSE_N0, *(TARGET_DATE_FMT.format(h=h) for h in horizons))


def build_targets(
    prices: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    price_column: str = "close",
    max_gap_days: int = 7,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Construct ``n+h`` targets per asset from a sorted bar frame.

    Returns ``(targets, nulled_by_gap)``.  ``targets`` shares ``prices``' index and
    has ``close_n0``, ``target_date_n{h}``, ``target_close_n{h}`` and
    ``target_return_n{h}``.  Targets only look forward by ``h`` bars within the
    same asset, so the last ``h`` rows of each asset have a null ``n+h`` target.
    """
    if any(int(h) < 1 for h in horizons):
        raise ValueError("horizons must be >= 1: n+0 is not a forecast target")
    nulled = {int(h): 0 for h in horizons}
    parts: list[pd.DataFrame] = []
    for _, group in prices.groupby("asset_symbol", sort=False):
        price = group[price_column].astype("float64")
        dates = group["bar_date"]
        # gap (calendar days) between bar i and bar i+1
        step_gaps = dates.diff().dt.days.shift(-1).to_numpy(dtype="float64")
        data: dict[str, pd.Series] = {CLOSE_N0: price}
        for h in horizons:
            h = int(h)
            future_price = price.shift(-h)
            future_date = dates.shift(-h)
            widest = pd.Series(step_gaps[::-1]).rolling(h, min_periods=h).max().to_numpy()[::-1]
            too_wide = pd.Series(widest > max_gap_days, index=group.index)
            usable = future_price.notna() & ~too_wide
            nulled[h] += int((future_price.notna() & too_wide).sum())
            data[TARGET_DATE_FMT.format(h=h)] = future_date.where(usable)
            data[TARGET_CLOSE_FMT.format(h=h)] = future_price.where(usable)
            data[TARGET_RETURN_FMT.format(h=h)] = (future_price / price - 1.0).where(usable)
        parts.append(pd.DataFrame(data, index=group.index))
    if not parts:
        return pd.DataFrame(index=prices.index), nulled
    return pd.concat(parts).reindex(prices.index), nulled


# ---------------------------------------------------------------------------
# Sentiment (Module 7 table)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreparedSentiment:
    """Validated Module 7 table restricted to the declared columns."""

    frame: pd.DataFrame
    used_columns: tuple[str, ...]
    missing_declared: tuple[str, ...]
    rejected_columns: tuple[str, ...]


def prepare_sentiment_table(sentiment: pd.DataFrame, config: DatasetConfig) -> PreparedSentiment:
    """Validate the Module 7 daily sentiment table.

    Requires ``asset_symbol`` and a date column (``bar_date`` or ``date``).
    News-like columns raise :class:`LeakageError`; undeclared columns are dropped
    and reported; null values are preserved (never filled).
    """
    frame = sentiment.copy()
    if "bar_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "bar_date"})
    missing_keys = [key for key in KEY_COLUMNS if key not in frame.columns]
    if missing_keys:
        raise DatasetBuildError(f"sentiment table is missing key column(s): {missing_keys}")

    other = [c for c in frame.columns if c not in KEY_COLUMNS]
    news = find_news_columns(other)
    if news:
        raise LeakageError([f"sentiment table: news column(s) present: {news}"])

    used = tuple(c for c in config.sentiment_columns if c in other)
    missing_declared = tuple(c for c in config.sentiment_columns if c not in other)
    rejected = tuple(c for c in other if c not in config.sentiment_columns)
    if not used:
        raise DatasetBuildError(
            "sentiment table has none of the declared sentiment columns "
            f"{list(config.sentiment_columns)}; got {other}. Pass "
            "DatasetConfig(sentiment_columns=...) to match Module 7's output."
        )

    frame["asset_symbol"] = frame["asset_symbol"].astype(str)
    frame["bar_date"] = normalize_dates(frame["bar_date"])
    violations = check_unique_keys(frame, label="sentiment table")
    if violations:
        raise LeakageError(violations)

    for column in used:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise DatasetBuildError(f"sentiment column {column!r} is not numeric") from exc

    frame = frame.loc[:, [*KEY_COLUMNS, *used]].sort_values(list(KEY_COLUMNS), kind="mergesort")
    return PreparedSentiment(frame.reset_index(drop=True), used, missing_declared, rejected)


def sentiment_feature_names(used: Sequence[str], lags: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        name
        for lag in lags
        for name in (*(f"{col}_lag_{lag}" for col in used), f"sentiment_row_present_lag_{lag}")
    )


def lag_sentiment_features(
    keys: pd.DataFrame,
    sentiment: PreparedSentiment,
    lags: Sequence[int],
) -> pd.DataFrame:
    """Left-join sentiment buckets dated ``bar_date - lag`` days onto ``keys``.

    A sentiment bucket dated ``d`` only ever feeds rows dated ``d + lag`` with
    ``lag >= 0``, so no future bucket reaches an earlier row.
    """
    out = keys[list(KEY_COLUMNS)].copy()
    for lag in lags:
        shifted = sentiment.frame.copy()
        shifted["bar_date"] = shifted["bar_date"] + pd.Timedelta(days=int(lag))
        present = f"sentiment_row_present_lag_{lag}"
        shifted[present] = 1.0
        shifted = shifted.rename(columns={c: f"{c}_lag_{lag}" for c in sentiment.used_columns})
        out = out.merge(shifted, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        out[present] = out[present].fillna(0.0)
    out.index = keys.index
    return out


def sentiment_feature_specs(used: Sequence[str], lags: Sequence[int]) -> list[FeatureSpec]:
    specs = []
    for lag in lags:
        for col in used:
            specs.append(
                FeatureSpec(
                    f"{col}_lag_{lag}",
                    SENTIMENT_FAMILY,
                    int(lag),
                    "calendar_days",
                    f"Module 7 {col} bucket dated bar_date - {lag} day(s)",
                )
            )
        specs.append(
            FeatureSpec(
                f"sentiment_row_present_lag_{lag}",
                SENTIMENT_FAMILY,
                int(lag),
                "calendar_days",
                "1.0 if the sentiment table had a row for that bucket date, else 0.0",
            )
        )
    return specs


def read_sentiment_table(path: str | Path) -> pd.DataFrame:
    """Load a saved Module 7 table (``.parquet`` or ``.csv``)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise DatasetBuildError(f"unsupported sentiment table format: {path.suffix!r}")


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------
def _build_feature_frame(
    prices: pd.DataFrame,
    sentiment: PreparedSentiment | None,
    config: DatasetConfig,
    include_volume: bool,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Keys + all computed feature columns (price first, then sentiment)."""
    features = build_price_features(prices, config.price, include_volume=include_volume)
    specs = price_feature_specs(config.price, include_volume)
    if sentiment is not None and config.sentiment_lags:
        lagged = lag_sentiment_features(prices, sentiment, config.sentiment_lags)
        features = pd.concat([features, lagged.drop(columns=list(KEY_COLUMNS))], axis=1)
        specs += sentiment_feature_specs(sentiment.used_columns, config.sentiment_lags)
    return features, specs


def default_allowlist(specs: Iterable[FeatureSpec]) -> tuple[str, ...]:
    """The explicit allowlist implied by the configuration (ordered)."""
    return tuple(spec.name for spec in specs)


# ---------------------------------------------------------------------------
# Truncation violations
# ---------------------------------------------------------------------------
def _truncation_violations(
    prices: pd.DataFrame,
    sentiment: PreparedSentiment | None,
    config: DatasetConfig,
    include_volume: bool,
    full: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Rebuild features from data truncated at sample cutoffs and compare.

    If removing every input dated after the cutoff changes a feature value, that
    feature used future data.  Targets are excluded (they look forward by design).
    """
    violations: list[str] = []
    for asset, kept in full.groupby("asset_symbol", sort=True):
        dates = kept["bar_date"].reset_index(drop=True)
        n = len(dates)
        if n == 0:
            continue
        positions: set[int] = set()
        positions.add(min(n - 1, n // 4))
        positions.add(min(n - 1, n // 2))
        positions.add(min(n - 1, (3 * n) // 4))
        positions.add(n - 1)
        positions = sorted(positions)
        asset_prices = prices[prices["asset_symbol"] == asset]
        for position in positions:
            cutoff = dates.iloc[position]
            truncated = asset_prices[asset_prices["bar_date"] <= cutoff].reset_index(drop=True)
            truncated_sentiment = sentiment
            if sentiment is not None:
                visible = sentiment.frame[sentiment.frame["bar_date"] <= cutoff]
                truncated_sentiment = PreparedSentiment(
                    visible.reset_index(drop=True),
                    sentiment.used_columns,
                    sentiment.missing_declared,
                    sentiment.rejected_columns,
                )
            rebuilt, _ = _build_feature_frame(truncated, truncated_sentiment, config, include_volume)
            columns = [c for c in feature_columns if c in rebuilt.columns]
            new_row = rebuilt.loc[rebuilt["bar_date"] == cutoff, columns].to_numpy(dtype="float64")
            old_row = kept.loc[kept["bar_date"] == cutoff, columns].to_numpy(dtype="float64")
            if new_row.shape != old_row.shape or not np.allclose(
                new_row, old_row, rtol=1e-7, atol=1e-10, equal_nan=True
            ):
                changed = [
                    column
                    for i, column in enumerate(columns)
                    if new_row.shape != old_row.shape
                    or not np.allclose(new_row[:, i], old_row[:, i], rtol=1e-7, atol=1e-10, equal_nan=True)
                ]
                violations.append(
                    f"{asset} {pd.Timestamp(cutoff).date()}: feature(s) change when data after the "
                    f"cutoff is removed (look-ahead): {changed}"
                )
    return violations


# ---------------------------------------------------------------------------
# Dataset object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModellingDataset:
    """Versioned modelling table: features, targets, dates, and audit metadata."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    horizons: tuple[int, ...]
    config: dict[str, Any]
    audit: dict[str, Any]
    version: str
    key_columns: tuple[str, ...] = KEY_COLUMNS

    @property
    def features(self) -> pd.DataFrame:
        """Feature matrix (allowlisted columns only)."""
        return self.frame[list(self.feature_columns)]

    @property
    def targets(self) -> pd.DataFrame:
        """Raw target columns (never part of the feature matrix)."""
        return self.frame[list(self.target_columns)]

    @property
    def dates(self) -> pd.DataFrame:
        """Keys plus the reference close and target dates."""
        return self.frame[[*self.key_columns, *self.metadata_columns]]

    def labeled(self, horizon: int) -> pd.DataFrame:
        """Rows whose ``n+horizon`` target is known (training/validation rows)."""
        return self.frame[self.frame[TARGET_RETURN_FMT.format(h=horizon)].notna()]

    def latest_rows(self) -> pd.DataFrame:
        """Most recent row per asset (features available, targets typically null)."""
        return self.frame.groupby("asset_symbol", sort=True).tail(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _date_span(dates: pd.Series) -> dict[str, Any]:
    if dates.empty:
        return {"count": 0, "first": None, "last": None}
    return {
        "count": int(len(dates)),
        "first": pd.Timestamp(dates.min()).date().isoformat(),
        "last": pd.Timestamp(dates.max()).date().isoformat(),
    }


def _dataset_version(frame: pd.DataFrame, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return f"v{DATASET_SCHEMA_VERSION}-{digest.hexdigest()[:12]}"


def build_modelling_dataset(
    price_bars: Iterable[PriceBar] | pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    config: DatasetConfig | None = None,
    *,
    raise_on_failure: bool = True,
) -> ModellingDataset:
    """Build the leakage-safe modelling table.

    Parameters
    ----------
    price_bars:
        Module 4 ``PriceBar`` rows (or an equivalent frame with ``asset_symbol``,
        ``bar_date``, ``close`` and optionally ``volume``).
    sentiment:
        Module 7 daily table keyed by ``asset_symbol`` and ``bar_date`` (or
        ``date``).  ``None`` builds a price-only table; the audit records this.
    raise_on_failure:
        Raise :class:`LeakageError` when any audit check fails (default).  With
        ``False`` the failed audit is returned for inspection.
    """
    config = config or DatasetConfig()
    prices = bars_to_frame(price_bars)
    if prices.empty:
        raise DatasetBuildError("no price bars supplied")

    violations = check_unique_keys(prices, label="price bars")
    if violations:
        raise LeakageError(violations)

    price_column = config.price.price_column
    input_bars = int(len(prices))
    bad_price = ~np.isfinite(prices[price_column].to_numpy(dtype="float64"))
    missing_price_dates = prices.loc[bad_price, "bar_date"]
    prices = prices.loc[~bad_price].reset_index(drop=True)
    if prices.empty:
        raise DatasetBuildError(f"no bars have a usable {price_column!r}")
    include_volume = bool(prices["volume"].notna().any())

    prepared = prepare_sentiment_table(sentiment, config) if sentiment is not None else None

    features, specs = _build_feature_frame(prices, prepared, config, include_volume)
    feature_part = features.drop(columns=list(KEY_COLUMNS))
    computed = tuple(feature_part.columns)

    # ---- allowlist ------------------------------------------------------
    allowlist = config.feature_allowlist or default_allowlist(specs)
    unknown = sorted(set(allowlist) - set(computed))
    if unknown:
        raise LeakageError([f"allowlist names columns that are not computed: {unknown}"])
    dangerous = check_features_exclude_targets(allowlist, (), ())
    if dangerous:
        raise LeakageError(dangerous)
    feature_columns = tuple(c for c in computed if c in set(allowlist))
    deselected = tuple(c for c in computed if c not in set(allowlist))
    specs = [s for s in specs if s.name in set(feature_columns)]

    # ---- targets --------------------------------------------------------
    targets, nulled_by_gap = build_targets(
        prices, config.horizons, price_column=price_column, max_gap_days=config.max_gap_days
    )
    metadata_columns = metadata_columns_for(config.horizons)
    target_columns = target_columns_for(config.horizons)

    full = pd.concat(
        [
            prices[list(KEY_COLUMNS)],
            targets[list(metadata_columns)],
            feature_part[list(feature_columns)],
            targets[list(target_columns)],
        ],
        axis=1,
    )

    # ---- warm-up rows ---------------------------------------------------
    required = [c for c in config.price.required_feature_names() if c in full.columns]
    warmup = full[required].isna().any(axis=1) if required else pd.Series(False, index=full.index)
    warmup_by_asset = {
        str(asset): _date_span(group["bar_date"])
        for asset, group in full.loc[warmup].groupby("asset_symbol", sort=True)
    }
    if config.drop_warmup_rows:
        frame = full.loc[~warmup].reset_index(drop=True)
    else:
        frame = full.reset_index(drop=True)
    if frame.empty:
        raise DatasetBuildError(
            "no rows remain after dropping feature warm-up rows; supply more history "
            "(Module 4 should fetch warm-up bars before the modelling period)"
        )

    # ---- look-ahead probe ----------------------------------------------
    extra_checks: dict[str, list[str]] = {}
    if config.verify_truncation:
        extra_checks["features_invariant_to_future_data"] = _truncation_violations(
            prices, prepared, config, include_volume, frame, feature_columns
        )

    # ---- accounting / alignment ----------------------------------------
    sentiment_assets = set(prepared.frame["asset_symbol"]) if prepared is not None else set()
    assets_without_sentiment = (
        sorted(set(frame["asset_symbol"]) - sentiment_assets) if prepared is not None else []
    )
    warnings: list[str] = []
    if prepared is None:
        warnings.append("no sentiment table supplied: price-only modelling table")
    else:
        if prepared.missing_declared:
            warnings.append(f"declared sentiment columns absent from the table: {list(prepared.missing_declared)}")
        if prepared.rejected_columns:
            warnings.append(f"undeclared sentiment columns rejected: {list(prepared.rejected_columns)}")
        if assets_without_sentiment:
            warnings.append(f"assets with no sentiment rows (features stay null): {assets_without_sentiment}")
        bucket_tz = config.sentiment_bucket_timezone
        if bucket_tz is None:
            warnings.append(
                "sentiment_bucket_timezone not declared; Module 7's bucket timezone must equal "
                f"the market timezone ({config.market_timezone}) for the date join to be valid"
            )
        elif bucket_tz != config.market_timezone:
            warnings.append(
                f"sentiment bucket timezone {bucket_tz} differs from market timezone "
                f"{config.market_timezone}; verify Module 7 bucket dates are session dates"
            )

    matched = {}
    if prepared is not None:
        for lag in config.sentiment_lags:
            column = f"sentiment_row_present_lag_{lag}"
            if column in frame.columns:
                matched[f"lag_{lag}"] = int((frame[column] == 1.0).sum())

    row_accounting = {
        "input_price_bars": input_bars,
        "rows_per_asset": {
            str(a): int(n) for a, n in frame.groupby("asset_symbol", sort=True).size().items()
        },
        "dropped": {
            "missing_price": _date_span(missing_price_dates),
            "warmup": {
                "dropped": bool(config.drop_warmup_rows),
                "total": int(warmup.sum()),
                "by_asset": warmup_by_asset,
                "rule": "any of return_lag_*, volatility_*, ma_dist_* is null",
            },
        },
        "unlabeled_rows": {
            f"n{h}": {
                "kept_rows_without_target": int(frame[TARGET_RETURN_FMT.format(h=h)].isna().sum()),
                "nulled_by_gap_over_limit": int(nulled_by_gap[h]),
            }
            for h in config.horizons
        },
    }
    alignment = {
        "price_date_basis": "bar_date = UTC calendar date of the bar start; one row per available "
        "bar, no calendar fill, no forward-fill",
        "horizon_basis": "n+h is the h-th next available bar for the same asset",
        "market_timezone": config.market_timezone,
        "sentiment_bucket_timezone": config.sentiment_bucket_timezone,
        "sentiment_join": "left join on (asset_symbol, bar_date - lag calendar days); "
        "missing sentiment stays null",
        "sentiment_lags": list(config.sentiment_lags),
        "sentiment_present": prepared is not None,
        "sentiment_rows_matched": matched,
        "assets_without_sentiment": assets_without_sentiment,
        "price_column": price_column,
        "warnings": warnings,
    }

    config_dict = config.to_dict()
    version = _dataset_version(frame, config_dict)
    audit = build_leakage_audit(
        frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        metadata_columns=metadata_columns,
        horizons=config.horizons,
        specs=specs,
        allowlist=allowlist,
        min_sentiment_offset=config.min_sentiment_offset,
        extra_checks=extra_checks,
        row_accounting=row_accounting,
        alignment=alignment,
        rejected_input_columns=prepared.rejected_columns if prepared is not None else (),
        parameters={**config_dict, "allowlist_deselected": list(deselected)},
        dataset_version=version,
    )
    if raise_on_failure:
        assert_audit_passed(audit)

    return ModellingDataset(
        frame=frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        metadata_columns=metadata_columns,
        horizons=config.horizons,
        config=config_dict,
        audit=audit,
        version=version,
    )


# ---------------------------------------------------------------------------
# Persistence (versioned)
# ---------------------------------------------------------------------------
def save_dataset(dataset: ModellingDataset, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/<version>/{dataset.parquet,audit.json,manifest.json}``.

    The version is a content hash, so re-saving identical data rewrites the same
    directory.  ``<output_dir>/latest.json`` points at the newest save.
    """
    root = Path(output_dir)
    directory = root / dataset.version
    directory.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(directory / "dataset.parquet", index=False)
    (directory / "audit.json").write_text(audit_to_json(dataset.audit), encoding="utf-8")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "version": dataset.version,
        "key_columns": list(dataset.key_columns),
        "feature_columns": list(dataset.feature_columns),
        "target_columns": list(dataset.target_columns),
        "metadata_columns": list(dataset.metadata_columns),
        "horizons": list(dataset.horizons),
        "rows": int(len(dataset.frame)),
        "audit_passed": bool(dataset.audit.get("passed")),
        "config": dataset.config,
    }
    (directory / "manifest.json").write_text(
        json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "latest.json").write_text(
        json.dumps({"version": dataset.version, "path": str(directory)}, indent=2), encoding="utf-8"
    )
    return directory


def load_dataset(path: str | Path) -> ModellingDataset:
    """Load a saved dataset directory (or an output dir containing ``latest.json``)."""
    directory = Path(path)
    if (directory / "latest.json").exists() and not (directory / "manifest.json").exists():
        directory = directory / json.loads((directory / "latest.json").read_text())["version"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(directory / "dataset.parquet")
    return ModellingDataset(
        frame=frame,
        feature_columns=tuple(manifest["feature_columns"]),
        target_columns=tuple(manifest["target_columns"]),
        metadata_columns=tuple(manifest["metadata_columns"]),
        horizons=tuple(manifest["horizons"]),
        config=manifest["config"],
        audit=audit,
        version=manifest["version"],
        key_columns=tuple(manifest["key_columns"]),
    )


# ---------------------------------------------------------------------------
# Convenience wrappers around Module 0/3
# ---------------------------------------------------------------------------
def build_dataset_from_repository(
    repository: Any,
    asset_symbols: str | Sequence[str],
    sentiment: pd.DataFrame | None = None,
    config: DatasetConfig | None = None,
    *,
    start_date: Any = None,
    end_date: Any = None,
) -> ModellingDataset:
    """Read bars from the Module 3 repository and build the dataset."""
    symbols = [asset_symbols] if isinstance(asset_symbols, str) else list(asset_symbols)
    bars: list[Any] = []
    for symbol in symbols:
        bars.extend(repository.read_price_bars(symbol, start_date, end_date))
    if not bars:
        raise DatasetBuildError(f"no stored price bars for {symbols}; run Module 4 first")
    return build_modelling_dataset(bars, sentiment, config)


def build_dataset_from_settings(
    settings: Any,
    sentiment: pd.DataFrame | None = None,
    *,
    save: bool = True,
    **config_overrides: Any,
) -> tuple[ModellingDataset, Path | None]:
    """Build (and optionally save) the dataset for the configured asset."""
    from src.storage.repository import repository_from_settings

    config = DatasetConfig.from_settings(settings, **config_overrides)
    repository = repository_from_settings(settings)
    dataset = build_dataset_from_repository(repository, settings.asset_symbol, sentiment, config)
    path: Path | None = None
    if save:
        path = save_dataset(dataset, settings.processed_data_dir / "datasets" / settings.asset_slug)
    return dataset, path


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_HORIZONS",
    "DEFAULT_SENTIMENT_COLUMNS",
    "DatasetBuildError",
    "DatasetConfig",
    "ModellingDataset",
    "PreparedSentiment",
    "build_dataset_from_repository",
    "build_dataset_from_settings",
    "build_modelling_dataset",
    "build_targets",
    "default_allowlist",
    "lag_sentiment_features",
    "load_dataset",
    "metadata_columns_for",
    "prepare_sentiment_table",
    "read_sentiment_table",
    "save_dataset",
    "sentiment_feature_names",
    "target_columns_for",
]