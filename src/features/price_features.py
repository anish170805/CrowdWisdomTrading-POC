"""Price-only features for Module 8.

Every feature on the row for ``bar_date = t`` is computed from bars dated ``<= t``
(the ``n+0`` close is the latest *completed* observation, not a forecast), so a
feature row can be built at the prediction cutoff without any future data.

Features (long format: one row per ``asset_symbol`` / ``bar_date``)::

    return_lag_{k}     1-day simple return, lag 1 = the return ending on bar t,
                       lag k = the return ending k-1 bars earlier
    volatility_{w}     sample std-dev (ddof=1) of the last w 1-day returns
    ma_dist_{w}        close / SMA_w(close) - 1   (scale-free distance from the SMA)
    volume_change_{w}  volume_t / volume_{t-w} - 1   (only where volume exists)

Windows count *bars* (trading sessions), never calendar days: non-trading days
are not fabricated or forward-filled.  The ``close`` column is used by default
because ``adjusted_close`` is retroactively revised by later splits/dividends.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.validation.leakage import PRICE_FAMILY, FeatureSpec

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps this module pydantic-free
    from src.schemas.prices import PriceBar

BAR_FRAME_COLUMNS: tuple[str, ...] = (
    "asset_symbol",
    "bar_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "adjusted_close", "volume")
_ALLOWED_PRICE_COLUMNS = ("close", "adjusted_close")


class PriceDataError(ValueError):
    """Raised for malformed price input."""


@dataclass(frozen=True)
class PriceFeatureConfig:
    """Windows for the price feature family."""

    return_lags: tuple[int, ...] = (1, 2, 3, 5)
    volatility_windows: tuple[int, ...] = (5, 10, 20)
    ma_windows: tuple[int, ...] = (5, 10, 20, 50)
    volume_change_windows: tuple[int, ...] = (1, 5)
    price_column: str = "close"
    # New technical indicator windows
    rsi_windows: tuple[int, ...] = (14,)
    roc_windows: tuple[int, ...] = (5, 10, 20)
    bollinger_windows: tuple[int, ...] = (20,)
    atr_windows: tuple[int, ...] = (14,)

    def __post_init__(self) -> None:
        for name in ("return_lags", "volatility_windows", "ma_windows", "volume_change_windows", "rsi_windows", "roc_windows", "bollinger_windows", "atr_windows"):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers, got {values!r}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates, got {values!r}")
            object.__setattr__(self, name, tuple(int(v) for v in values))
        if any(w < 2 for w in self.volatility_windows):
            raise ValueError("volatility_windows must be >= 2 (a std-dev needs two returns)")
        if any(w < 2 for w in self.rsi_windows):
            raise ValueError("rsi_windows must be >= 2 (RSI needs at least 2 bars)")
        if any(w < 2 for w in self.bollinger_windows):
            raise ValueError("bollinger_windows must be >= 2 (needs mean and std)")
        if self.price_column not in _ALLOWED_PRICE_COLUMNS:
            raise ValueError(f"price_column must be one of {_ALLOWED_PRICE_COLUMNS}")

    def required_feature_names(self) -> tuple[str, ...]:
        """Features that gate warm-up (volume is optional and never gates rows)."""
        return (
            *(f"return_lag_{k}" for k in self.return_lags),
            *(f"volatility_{w}" for w in self.volatility_windows),
            *(f"ma_dist_{w}" for w in self.ma_windows),
            *(f"rsi_{w}" for w in self.rsi_windows),
            *(f"roc_{w}" for w in self.roc_windows),
        )

    def feature_names(self, include_volume: bool) -> tuple[str, ...]:
        names = self.required_feature_names()
        names += tuple(f"bb_position_{w}" for w in self.bollinger_windows)
        names += tuple(f"atr_{w}" for w in self.atr_windows)
        if include_volume:
            names += tuple(f"volume_change_{w}" for w in self.volume_change_windows)
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_lags": list(self.return_lags),
            "volatility_windows": list(self.volatility_windows),
            "ma_windows": list(self.ma_windows),
            "volume_change_windows": list(self.volume_change_windows),
            "price_column": self.price_column,
            "rsi_windows": list(self.rsi_windows),
            "roc_windows": list(self.roc_windows),
            "bollinger_windows": list(self.bollinger_windows),
            "atr_windows": list(self.atr_windows),
        }


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------
def normalize_dates(series: pd.Series) -> pd.Series:
    """Return naive ``datetime64[ns]`` calendar dates (tz-aware values become UTC dates)."""
    converted = pd.to_datetime(series)
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted.dt.normalize().astype("datetime64[ns]")


def bars_to_frame(bars: Iterable[PriceBar] | pd.DataFrame) -> pd.DataFrame:
    """Convert ``PriceBar`` objects (or a DataFrame) into a sorted bar frame.

    Duplicates are *not* removed here; the dataset builder rejects them loudly.
    """
    if isinstance(bars, pd.DataFrame):
        frame = bars.copy()
        missing = [c for c in ("asset_symbol", "bar_date", "close") if c not in frame.columns]
        if missing:
            raise PriceDataError(f"price frame is missing required column(s): {missing}")
        for column in BAR_FRAME_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame.loc[:, list(BAR_FRAME_COLUMNS)]
    else:
        rows: list[dict[str, Any]] = []
        for bar in bars:
            row: dict[str, Any] = {}
            for column in BAR_FRAME_COLUMNS:
                row[column] = getattr(bar, column, None)
            rows.append(row)
        frame = pd.DataFrame(rows, columns=list(BAR_FRAME_COLUMNS))

    if frame.empty:
        return frame.assign(bar_date=pd.Series(dtype="datetime64[ns]"))

    frame["asset_symbol"] = frame["asset_symbol"].astype(str)
    frame["bar_date"] = normalize_dates(frame["bar_date"])
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame.sort_values(["asset_symbol", "bar_date"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
# noqa: E501
def _asset_features(group: pd.DataFrame, config: PriceFeatureConfig, include_volume: bool) -> pd.DataFrame:
    price = group[config.price_column].astype("float64")
    one_day_return = price / price.shift(1) - 1.0
    high = group["high"].astype("float64")
    low = group["low"].astype("float64")

    columns: dict[str, pd.Series] = {}
    for lag in config.return_lags:
        columns[f"return_lag_{lag}"] = one_day_return.shift(lag - 1)
    for window in config.volatility_windows:
        rolling = one_day_return.rolling(window, min_periods=window)
        columns[f"volatility_{window}"] = rolling.std(ddof=1)
    for window in config.ma_windows:
        price_rolling = price.rolling(window, min_periods=window)
        columns[f"ma_dist_{window}"] = price / price_rolling.mean() - 1.0
    for window in config.rsi_windows:
        gains = one_day_return.where(one_day_return > 0, 0.0)
        losses = (-one_day_return).where(one_day_return < 0, 0.0)
        avg_gain = gains.rolling(window, min_periods=window).mean()
        avg_loss = losses.rolling(window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        columns[f"rsi_{window}"] = 100.0 - (100.0 / (1.0 + rs))
    for window in config.roc_windows:
        columns[f"roc_{window}"] = price / price.shift(window) - 1.0
    for window in config.bollinger_windows:
        price_rolling = price.rolling(window, min_periods=window)
        rolling_mean = price_rolling.mean()
        rolling_std = price_rolling.std(ddof=1)
        columns[f"bb_position_{window}"] = (price - rolling_mean) / (2.0 * rolling_std)
    for window in config.atr_windows:
        prev_close = price.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        columns[f"atr_{window}"] = true_range.rolling(window, min_periods=window).mean() / price
    if include_volume:
        volume = group["volume"].astype("float64")
        for window in config.volume_change_windows:
            base = volume.shift(window)
            columns[f"volume_change_{window}"] = (volume / base - 1.0).where(base > 0)
    return pd.DataFrame(columns, index=group.index)


def build_price_features(
    bars: Iterable[PriceBar] | pd.DataFrame,
    config: PriceFeatureConfig | None = None,
    *,
    include_volume: bool | None = None,
) -> pd.DataFrame:
    """Build the price feature table, one row per input bar.

    Returns ``asset_symbol``, ``bar_date`` and the feature columns.  Warm-up
    rows carry NaN (never zero-filled).  Features for an asset depend only on
    that asset's bars dated ``<=`` the row's ``bar_date``.

    ``include_volume`` defaults to "any bar has volume"; assets without volume
    (e.g. FX) then get NaN volume features rather than a different schema.
    """
    config = config or PriceFeatureConfig()
    frame = bars if _is_bar_frame(bars) else bars_to_frame(bars)
    if include_volume is None:
        include_volume = bool(frame["volume"].notna().any()) if len(frame) else False

    names = config.feature_names(include_volume)
    if frame.empty:
        empty = frame.loc[:, ["asset_symbol", "bar_date"]].copy()
        return empty.assign(**{name: pd.Series(dtype="float64") for name in names})

    parts = [
        _asset_features(group, config, include_volume)
        for _, group in frame.groupby("asset_symbol", sort=False)
    ]
    features = pd.concat(parts).reindex(frame.index)
    return pd.concat([frame[["asset_symbol", "bar_date"]], features], axis=1)


def _is_bar_frame(value: object) -> bool:
    """True for frames already normalised by :func:`bars_to_frame`."""
    return isinstance(value, pd.DataFrame) and tuple(value.columns) == BAR_FRAME_COLUMNS and (
        value.empty or str(value["bar_date"].dtype).startswith("datetime64")
    )


def price_feature_specs(config: PriceFeatureConfig, include_volume: bool) -> list[FeatureSpec]:
    """Declared timing: every price feature's newest input is the cutoff bar itself."""
    descriptions = {
        "return_lag_": "1-day simple return ending (k-1) bars before the cutoff",
        "volatility_": "rolling sample std of 1-day returns",
        "ma_dist_": "close / simple moving average - 1",
        "volume_change_": "volume / volume w bars earlier - 1",
        "rsi_": "relative strength index over w bars",
        "roc_": "rate of change over w bars",
        "bb_position_": "Bollinger Band position (close - SMA) / (2 * std)",
        "atr_": "average true range (normalized by close)",
    }
    specs = []
    for name in config.feature_names(include_volume):
        description = next(text for prefix, text in descriptions.items() if name.startswith(prefix))
        specs.append(FeatureSpec(name, PRICE_FAMILY, 0, "trading_rows", description))
    return specs


__all__ = [
    "BAR_FRAME_COLUMNS",
    "PriceDataError",
    "PriceFeatureConfig",
    "bars_to_frame",
    "build_price_features",
    "normalize_dates",
    "price_feature_specs",
]
