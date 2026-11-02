"""Prediction snapshot contracts for Module 11.

Defines the data structures that Module 11 produces and consumes:
* ``PredictionHorizon`` — a single n+horizon forecast entry.
* ``PredictionSnapshot`` — the complete six-slot prediction object
  (observed n+0 + five forecasts n+1 … n+5).
* ``PredictionInput`` — validated inputs required to generate a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionHorizon:
    """A single horizon forecast entry."""

    horizon: int
    """Horizon index: n+1, n+2, …, n+5."""

    return_prediction: float
    """Predicted simple return from n+0 close to n+horizon close."""

    close_prediction: float
    """Reconstructed predicted n+horizon close price."""

    forecast_date: str
    """ISO date string of the forecast bar date."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "return_prediction": self.return_prediction,
            "close_prediction": self.close_prediction,
            "forecast_date": self.forecast_date,
        }


@dataclass
class PredictionSnapshot:
    """The complete six-slot prediction object.

    Structure: observed n+0 close + five forecasts n+1 … n+5.
    All timestamps are UTC-aware ISO format strings.
    """

    asset_symbol: str
    """Asset symbol being forecast."""

    cutoff_timestamp: str
    """UTC ISO timestamp of the data cutoff (when features were built)."""

    input_freshness: str
    """Human-readable description of how recent the input data is."""

    n0_observed: float
    """The observed latest completed close (n+0)."""

    forecasts: tuple[PredictionHorizon, ...]
    """Six-slot tuple: n+0 is implicit (observed), n+1 … n+5 are forecast."""

    model_run_id: str
    """Identifier of the trained model bundle used (run_id from Module 9)."""

    model_version: str
    """Human-readable model version / library versions snapshot."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_symbol": self.asset_symbol,
            "cutoff_timestamp": self.cutoff_timestamp,
            "input_freshness": self.input_freshness,
            "n0_observed": self.n0_observed,
            "forecasts": [h.as_dict() for h in self.forecasts],
            "model_run_id": self.model_run_id,
            "model_version": self.model_version,
        }

    @property
    def forecast_horizons(self) -> tuple[int, ...]:
        """Return (1, 2, 3, 4, 5) for convenience."""
        return tuple(h.horizon for h in self.forecasts)


@dataclass
class PredictionInput:
    """Validated inputs required to generate a latest prediction."""

    asset_symbol: str
    """Asset symbol."""

    model_bundle: "ModelBundleProxy"
    """Lightweight proxy to the Module 9 ModelBundle (preprocessor + trained models)."""

    latest_frame: pd.DataFrame
    """Single-row DataFrame with the latest completed bar and all features.

    Must contain the column ``close_n0`` and all feature columns expected by
    the preprocessor, in the correct order.
    """

    cutoff_timestamp: datetime
    """UTC datetime of the data cutoff."""

    def validate(self) -> None:
        """Check that the input is consistent; raises ``ValueError`` on failure."""
        if not isinstance(self.latest_frame, pd.DataFrame):
            raise ValueError("latest_frame must be a pandas DataFrame")
        if len(self.latest_frame) != 1:
            raise ValueError("latest_frame must contain exactly one row")
        if "close_n0" not in self.latest_frame.columns:
            raise ValueError("latest_frame must contain column 'close_n0'")
        # Check that all required feature columns are present
        missing = [c for c in self.model_bundle.feature_columns if c not in self.latest_frame.columns]
        if missing:
            raise ValueError(f"latest_frame is missing required feature columns: {missing}")
        # Check column order if the preprocessor enforces strict order
        present_order = tuple(self.latest_frame.columns)
        expected_order = self.model_bundle.feature_columns
        if present_order != expected_order:
            raise ValueError(
                f"Feature column order mismatch:\n  expected: {expected_order}\n  got:      {present_order}"
            )


# ---------------------------------------------------------------------------
# Lightweight proxy that exposes only what Module 11 needs from a ModelBundle.
# ---------------------------------------------------------------------------


class ModelBundleProxy:
    """Minimal read-only view of a ModelBundle for Module 11 consumption.

    This avoids importing the full ModelBundle (which depends on xgboost and
    preprocessing fit-state) at the prediction-cutoff boundary.
    """

    __slots__ = ("_feature_columns", "_horizons", "_library_versions", "_asset_symbol")

    def __init__(self, feature_columns: tuple[str, ...], horizons: tuple[int, ...], library_versions: dict[str, str], asset_symbol: str) -> None:
        self._feature_columns = feature_columns
        self._horizons = horizons
        self._library_versions = library_versions
        self._asset_symbol = asset_symbol

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    @property
    def library_versions(self) -> dict[str, str]:
        return self._library_versions

    @property
    def asset_symbol(self) -> str:
        return self._asset_symbol


__all__ = [
    "PredictionHorizon",
    "PredictionSnapshot",
    "PredictionInput",
    "ModelBundleProxy",
]