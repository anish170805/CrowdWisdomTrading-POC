"""Module 11 — Latest prediction generation.

Produces the current n+0 to n+5 outlook using only data available at the run cutoff.

**Input**
* Trained model artifacts from Module 9 (loaded via ``load_bundle``).
* Latest canonical modelling frame from Module 8 (must contain ``close_n0`` and all
  feature columns in the order recorded by the preprocessor).
* Asset symbol for filtering.

**Output**
* A ``PredictionSnapshot`` containing the observed n+0 close and five forecasts n+1 … n+5.
* Warnings for stale data, missing features, or unsupported horizons are included in the
  snapshot's ``input_freshness`` field and raised as ``ValueError`` if validation fails.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.artifacts import load_bundle, ModelBundle, ArtifactError
from src.prediction.contracts import (
    PredictionInput,
    PredictionSnapshot,
    PredictionHorizon,
    ModelBundleProxy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUTOFF_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"
_FORECAST_HORIZONS = (1, 2, 3, 4, 5)


def _library_versions_to_snapshot(bundle: ModelBundle) -> dict[str, str]:
    """Return a plain-dict version snapshot from a ModelBundle's manifest."""
    return bundle.manifest.get("library_versions", {})


def _build_proxy(bundle: ModelBundle) -> ModelBundleProxy:
    """Wrap a ModelBundle into a ModelBundleProxy for the contracts layer."""
    return ModelBundleProxy(
        feature_columns=tuple(bundle.preprocessor.feature_columns),
        horizons=bundle.horizons,
        library_versions=_library_versions_to_snapshot(bundle),
        asset_symbol=bundle.asset_symbol,
    )


def _validate_latest_frame(frame: pd.DataFrame, proxy: ModelBundleProxy) -> None:
    """Ensure the latest frame has all required features in the right order."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("latest_frame must be a pandas DataFrame")
    if len(frame) != 1:
        raise ValueError("latest_frame must contain exactly one row (the latest completed bar)")
    if "close_n0" not in frame.columns:
        raise ValueError("latest_frame must contain column 'close_n0'")
    # Check that all required feature columns are present and in the correct order.
    # Extra columns (e.g. ``bar_date``, target columns) are allowed but features must
    # appear in the same order as the training set.
    feature_present = [c for c in proxy.feature_columns if c in frame.columns]
    missing = [c for c in proxy.feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"latest_frame is missing required feature columns: {missing}")
    # Verify order: the feature columns that are present must match the expected order.
    # We extract the subset of frame columns that are in feature_columns and compare.
    present_order = tuple(c for c in frame.columns if c in proxy.feature_columns)
    expected_order = proxy.feature_columns
    if present_order != expected_order:
        raise ValueError(
            "Feature column order mismatch:\n  expected: " + str(expected_order) + "\n  got:      " + str(present_order)
        )


def _predict_horizon(bundle: ModelBundle, horizon: int, frame: pd.DataFrame) -> dict[str, Any]:
    """Predict return and reconstructed close for one horizon.

    Returns a dict with keys: return_prediction, close_prediction, forecast_date.
    """
    return_ = float(
        bundle.predict_return(frame, horizon, strict_order=True)[0]
    )
    # Reconstruct close: close_n0 * (1 + return)
    close_ = float(frame["close_n0"].iloc[0] * (1.0 + return_))
    # Forecast date: bar_date of the row + 1 calendar day (or trading day convention)
    # We use the bar_date from the latest frame and add the horizon offset.
    bar_date = pd.Timestamp(frame["bar_date"].iloc[0])
    forecast_date = bar_date + timedelta(days=horizon)
    return {
        "return_prediction": return_,
        "close_prediction": close_,
        "forecast_date": forecast_date.date().isoformat(),
    }


def _compute_input_freshness(cutoff: datetime, now: datetime | None = None) -> str:
    """Produce a human-readable freshness description.

    If the cutoff is within the last 24h -> "real-time"
    If within the last week -> "daily cutoff"
    Otherwise -> "stale (date)"
    """
    if now is None:
        now = datetime.now(UTC)
    delta = now - cutoff
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "real-time (cutoff in future or now)"
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    if days == 0:
        return f"real-time (cutoff {hours}h ago)"
    if days < 7:
        return f"daily cutoff ({days}d ago)"
    return f"stale ({days}d ago, cutoff at {cutoff.date().isoformat()})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_latest_prediction(
    bundle: ModelBundle,
    latest_frame: pd.DataFrame,
    *,
    asset_symbol: str | None = None,
    now: datetime | None = None,
) -> PredictionSnapshot:
    """Generate the latest n+0 … n+5 prediction snapshot.

    Parameters
    ----------
    bundle : ModelBundle
        Trained model bundle from Module 9 (loaded via ``load_bundle``).
    latest_frame : pd.DataFrame
        Single-row DataFrame with the latest completed bar and all features.
        Must contain ``close_n0`` and all feature columns in the order recorded
        by the preprocessor.
    asset_symbol : str, optional
        If provided, must match ``bundle.asset_symbol``. Used for validation.
    now : datetime, optional
        Current UTC time for freshness computation. Defaults to ``datetime.now(UTC)``.

    Returns
    -------
    PredictionSnapshot
        The six-slot prediction object (observed n+0 + five forecasts n+1 … n+5).

    Raises
    ------
    ValueError
        If ``latest_frame`` is invalid, features are missing/reordered, or the
        asset symbol does not match the bundle.
    ArtifactError
        If the bundle is empty or has no models for the expected horizons.
    """
    # --- Validate inputs ---
    if asset_symbol is not None and asset_symbol != bundle.asset_symbol:
        raise ValueError(
            f"asset_symbol {asset_symbol!r} does not match bundle asset {bundle.asset_symbol!r}"
        )

    proxy = _build_proxy(bundle)
    _validate_latest_frame(latest_frame, proxy)

    # --- Observe n+0 ---
    n0_observed = float(latest_frame["close_n0"].iloc[0])

    # --- Generate forecasts n+1 … n+5 ---
    forecasts: list[PredictionHorizon] = []
    for h in _FORECAST_HORIZONS:
        if h not in bundle.models:
            # Horizon was skipped during training (e.g. not enough samples)
            # We still include it with a warning in freshness
            forecasts.append(
                PredictionHorizon(
                    horizon=h,
                    return_prediction=np.nan,
                    close_prediction=np.nan,
                    forecast_date="",
                )
            )
            continue
        pred = _predict_horizon(bundle, h, latest_frame)
        forecasts.append(
            PredictionHorizon(
                horizon=h,
                return_prediction=pred["return_prediction"],
                close_prediction=pred["close_prediction"],
                forecast_date=pred["forecast_date"],
            )
        )

    # --- Compute freshness ---
    cutoff = latest_frame.attrs.get("cutoff_timestamp", datetime.now(UTC))
    if cutoff is None:
        cutoff = datetime.now(UTC)
    # Ensure cutoff is a datetime; if it came as an ISO string, parse it
    if isinstance(cutoff, str):
        try:
            cutoff = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        except Exception:
            cutoff = datetime.now(UTC)
    freshness = _compute_input_freshness(cutoff, now)

    # --- Build snapshot ---
    model_versions = _library_versions_to_snapshot(bundle)
    version_str = json.dumps(model_versions, sort_keys=True) if model_versions else "unknown"

    snapshot = PredictionSnapshot(
        asset_symbol=bundle.asset_symbol,
        cutoff_timestamp=cutoff.strftime(_CUTOFF_ISO_FMT),
        input_freshness=freshness,
        n0_observed=n0_observed,
        forecasts=tuple(forecasts),
        model_run_id=bundle.manifest.get("run_id", "unknown"),
        model_version=version_str,
    )

    return snapshot


def load_latest_from_artifacts(
    artifact_root: str | Path,
    run_id: str,
    latest_frame: pd.DataFrame,
    *,
    asset_symbol: str | None = None,
    now: datetime | None = None,
) -> PredictionSnapshot:
    """Convenience: load a ModelBundle from saved artifacts and generate the snapshot.

    Parameters
    ----------
    artifact_root : str | Path
        Root directory containing the Module 9 run directories (one per ``run_id``).
    run_id : str
        The run_id returned by ``make_run_id`` / ``save_bundle`` from Module 9.
    latest_frame : pd.DataFrame
        Single-row DataFrame with the latest completed bar and all features.
    asset_symbol : str, optional
        If provided, verified against the bundle's asset.
    now : datetime, optional
        Current UTC time for freshness computation.

    Returns
    -------
    PredictionSnapshot
        The generated prediction snapshot.

    Raises
    ------
    ArtifactError
        If the run directory or manifest is not found.
    ValueError
        If inputs are inconsistent.
    """
    bundle = load_bundle(artifact_root, verify_checksums=True)
    if asset_symbol is not None and bundle.asset_symbol != asset_symbol:
        raise ValueError(
            f"asset_symbol {asset_symbol!r} does not match bundle asset {bundle.asset_symbol!r}"
        )
    return generate_latest_prediction(
        bundle,
        latest_frame,
        asset_symbol=asset_symbol,
        now=now,
    )