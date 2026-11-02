"""Forecast evaluation metrics for walk-forward validation (Module 10).

Provides MAE, RMSE, and directional accuracy per horizon and model, consistent
with the long-format prediction frame used by Modules 8‑9 (``PREDICTION_COLUMNS``
from ``src.models.baseline``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Sequence
from typing import Any

from src.models.baseline import PREDICTION_COLUMNS, TARGET_RETURN_FMT, TARGET_CLOSE_FMT
from src.validation.leakage import CLOSE_N0


def _directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    """Share of rows with the right sign.

    Rows with a zero prediction or outcome carry no direction and are excluded.
    Returns (accuracy, n), where accuracy is NaN if n == 0.
    """
    mask = (pred != 0) & (actual != 0) & np.isfinite(pred) & np.isfinite(actual)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.sign(pred[mask]) == np.sign(actual[mask])).mean()), n


def summarize_forecasts(
    predictions: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """MAE / RMSE (return and close) and directional accuracy per model and horizon.

    Parameters
    ----------
    predictions : pd.DataFrame
        Long-format prediction frame (``PREDICTION_COLUMNS``).
    models : sequence of str, optional
        Model names to include.  If ``None``, all models in the data are scored.
    horizons : sequence of int, optional
        Forecast horizons to include.  If ``None``, all horizons present are scored.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``horizon``, ``n_samples``, ``mae_close``,
        ``rmse_close``, ``mae_return``, ``rmse_return``,
        ``directional_accuracy``, ``n_directional``.
    """
    scored = predictions.dropna(subset=["pred_return", "actual_return"]).copy()

    if models is not None:
        scored = scored[scored["model"].isin(models)].copy()
    if horizons is not None:
        scored = scored[scored["horizon"].isin([int(h) for h in horizons])].copy()

    rows: list[dict[str, Any]] = []
    for (model, horizon), group in scored.groupby(["model", "horizon"], sort=True):
        err_close = (group["pred_close"] - group["actual_close"]).to_numpy(dtype="float64")
        err_ret = (group["pred_return"] - group["actual_return"]).to_numpy(dtype="float64")
        accuracy, n_dir = _directional_accuracy(
            group["pred_return"].to_numpy(dtype="float64"),
            group["actual_return"].to_numpy(dtype="float64"),
        )
        rows.append(
            {
                "model": str(model),
                "horizon": int(horizon),
                "n_samples": int(len(group)),
                "mae_close": float(np.mean(np.abs(err_close))),
                "rmse_close": float(np.sqrt(np.mean(err_close ** 2))),
                "mae_return": float(np.mean(np.abs(err_ret))),
                "rmse_return": float(np.sqrt(np.mean(err_ret ** 2))),
                "directional_accuracy": accuracy,
                "n_directional": n_dir,
            }
        )

    columns = [
        "model",
        "horizon",
        "n_samples",
        "mae_close",
        "rmse_close",
        "mae_return",
        "rmse_return",
        "directional_accuracy",
        "n_directional",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_by_horizon(
    summary: pd.DataFrame,
    *,
    metric: str = "mae_return",
) -> pd.Series:
    """Aggregate a single metric across models, returning a Series indexed by horizon.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of :func:`summarize_forecasts`.
    metric : str
        One of ``"mae_return"``, ``"rmse_return"``, ``"directional_accuracy"``,
        ``"n_samples"``.

    Returns
    -------
    pd.Series
        Index = horizon, Value = aggregated metric (mean across models, or as appropriate).
    """
    if summary.empty:
        return pd.Series(dtype=float)
    # Mean across models per horizon
    return summary.groupby("horizon")[metric].mean()


def compare_horizons(
    summary: pd.DataFrame,
    *,
    baseline: str = "persistence",
    min_samples: int = 30,
) -> pd.DataFrame:
    """Per-horizon comparison against a baseline with an evidence guard.

    ``verdict`` is ``"better"`` only when MAE *and* RMSE (on returns) are both lower
    than the baseline and at least ``min_samples`` out-of-sample rows back it;
    with fewer rows it is ``"insufficient_evidence"``.
    """
    from src.models.baseline import compare_to_baseline as _compare_to_baseline

    # Use the existing baseline comparison but ensure we pass the right summary
    base = summary[summary["model"] == baseline].set_index("horizon") if not summary.empty else pd.DataFrame()
    if base.empty:
        raise ValueError(f"baseline {baseline!r} not present in the summary")
    out = _compare_to_baseline(summary, baseline=baseline, min_samples=min_samples)
    return out