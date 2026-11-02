"""Mandatory forecasting baselines for Module 9.

Baselines depend only on ``close_n0`` (and, for drift, the training targets), so
they run on any frame, with any number of rows, with no xgboost installed.

* :class:`PersistenceBaseline` — the naive forecast: ``close_{n+h} = close_n0``
  for every horizon (predicted return ``0``).  It is the benchmark any model must beat.
* :class:`DriftBaseline` — a constant forecast equal to the *training-window* mean
  ``n+h`` return.  Secondary baseline: unlike persistence it has a direction, so
  directional accuracy is defined for it.

Shared helpers build the long-format prediction frame and the scoring summary
used for every model and baseline, so comparisons are like-for-like.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from src.validation.leakage import (
    CLOSE_N0,
    KEY_COLUMNS,
    TARGET_CLOSE_FMT,
    TARGET_DATE_FMT,
    TARGET_RETURN_FMT,
)

PERSISTENCE = "persistence"
DRIFT = "drift"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "model",
    "asset_symbol",
    "bar_date",
    "horizon",
    "target_date",
    "close_n0",
    "pred_return",
    "pred_close",
    "actual_return",
    "actual_close",
)


class Baseline(Protocol):
    name: str

    def fit(self, train_frame: pd.DataFrame, horizons: Sequence[int]) -> Baseline: ...

    def predict_return(self, frame: pd.DataFrame, horizon: int) -> np.ndarray: ...


class PersistenceBaseline:
    """Naive forecast: the future close equals the latest observed close."""

    name = PERSISTENCE

    def fit(self, train_frame: pd.DataFrame, horizons: Sequence[int]) -> PersistenceBaseline:
        return self  # nothing to learn

    def predict_return(self, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        _require_horizon(horizon)
        zeros = np.zeros(len(frame), dtype="float64")
        # unknown reference close -> unknown forecast (never a silent 0 price)
        return np.where(frame[CLOSE_N0].isna().to_numpy(), np.nan, zeros)


class DriftBaseline:
    """Constant forecast: mean realised ``n+h`` return in the training window."""

    name = DRIFT

    def __init__(self) -> None:
        self.mean_return: dict[int, float] = {}

    def fit(self, train_frame: pd.DataFrame, horizons: Sequence[int]) -> DriftBaseline:
        self.mean_return = {}
        for h in horizons:
            values = train_frame[TARGET_RETURN_FMT.format(h=int(h))].dropna()
            self.mean_return[int(h)] = float(values.mean()) if len(values) else 0.0
        return self

    def predict_return(self, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        _require_horizon(horizon)
        if int(horizon) not in self.mean_return:
            raise KeyError(f"drift baseline was not fitted for horizon {horizon}")
        value = self.mean_return[int(horizon)]
        out = np.full(len(frame), value, dtype="float64")
        return np.where(frame[CLOSE_N0].isna().to_numpy(), np.nan, out)


def _require_horizon(horizon: int) -> None:
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1: n+0 is the observed close, not a forecast")


def default_baselines() -> list[Baseline]:
    return [PersistenceBaseline(), DriftBaseline()]


# ---------------------------------------------------------------------------
# Long-format predictions
# ---------------------------------------------------------------------------
def build_prediction_frame(
    frame: pd.DataFrame,
    horizon: int,
    pred_return: np.ndarray,
    *,
    model: str,
    fold_id: int | None,
) -> pd.DataFrame:
    """One row per ``(bar_date, horizon)``: prediction beside the realised outcome."""
    _require_horizon(horizon)
    pred_return = np.asarray(pred_return, dtype="float64")
    if len(pred_return) != len(frame):
        raise ValueError("pred_return length does not match the frame")
    close0 = frame[CLOSE_N0].to_numpy(dtype="float64")
    out = pd.DataFrame(
        {
            "fold_id": fold_id,
            "model": model,
            "asset_symbol": frame[KEY_COLUMNS[0]].to_numpy(),
            "bar_date": frame[KEY_COLUMNS[1]].to_numpy(),
            "horizon": int(horizon),
            "target_date": frame[TARGET_DATE_FMT.format(h=horizon)].to_numpy(),
            "close_n0": close0,
            "pred_return": pred_return,
            "pred_close": close0 * (1.0 + pred_return),
            "actual_return": frame[TARGET_RETURN_FMT.format(h=horizon)].to_numpy(dtype="float64"),
            "actual_close": frame[TARGET_CLOSE_FMT.format(h=horizon)].to_numpy(dtype="float64"),
        }
    )
    return out.loc[:, list(PREDICTION_COLUMNS)]


def baseline_predictions(
    frame: pd.DataFrame,
    horizons: Iterable[int],
    baselines: Sequence[Baseline] | None = None,
    *,
    fold_id: int | None = None,
    train_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict every horizon with every baseline on the rows of ``frame``.

    Only rows whose ``n+h`` target is known are returned for that horizon, so the
    baselines and the model are scored on identical samples.  ``train_frame`` is
    used only to fit stateful baselines (drift); persistence ignores it.
    """
    horizons = tuple(int(h) for h in horizons)
    chosen = list(baselines) if baselines is not None else default_baselines()
    if not any(b.name == PERSISTENCE for b in chosen):
        raise ValueError("the persistence baseline is mandatory")
    parts: list[pd.DataFrame] = []
    for baseline in chosen:
        baseline.fit(train_frame if train_frame is not None else frame.iloc[0:0], horizons)
        for h in horizons:
            labeled = frame[frame[TARGET_RETURN_FMT.format(h=h)].notna()]
            parts.append(
                build_prediction_frame(
                    labeled,
                    h,
                    baseline.predict_return(labeled, h),
                    model=baseline.name,
                    fold_id=fold_id,
                )
            )
    if not parts:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Scoring and comparison (like-for-like; Module 10 adds the full report)
# ---------------------------------------------------------------------------
def _directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    """Share of rows with the right sign.  Rows with a zero prediction or outcome
    carry no direction and are excluded (so persistence yields ``(nan, 0)``)."""
    mask = (pred != 0) & (actual != 0) & np.isfinite(pred) & np.isfinite(actual)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.sign(pred[mask]) == np.sign(actual[mask])).mean()), n


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE (price and return) and directional accuracy per model and horizon."""
    rows: list[dict[str, float | int | str]] = []
    scored = predictions.dropna(subset=["pred_return", "actual_return"])
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
                "rmse_close": float(np.sqrt(np.mean(err_close**2))),
                "mae_return": float(np.mean(np.abs(err_ret))),
                "rmse_return": float(np.sqrt(np.mean(err_ret**2))),
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


def compare_to_baseline(
    summary: pd.DataFrame,
    *,
    baseline: str = PERSISTENCE,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Per horizon and model: improvement over the baseline, with an evidence guard.

    ``verdict`` is ``"better"`` only when MAE *and* RMSE (on returns) are both lower
    than the baseline and at least ``min_samples`` out-of-sample rows back it;
    with fewer rows it is ``"insufficient_evidence"``.  A lower error on a small
    sample is not evidence of skill, so no superiority is claimed there.
    """
    base = summary[summary["model"] == baseline].set_index("horizon")
    if base.empty:
        raise ValueError(f"baseline {baseline!r} not present in the summary")
    out: list[dict[str, float | int | str]] = []
    for row in summary[summary["model"] != baseline].itertuples(index=False):
        if row.horizon not in base.index:
            continue
        ref = base.loc[row.horizon]
        mae_gain = 1.0 - row.mae_return / ref["mae_return"] if ref["mae_return"] > 0 else np.nan
        rmse_gain = 1.0 - row.rmse_return / ref["rmse_return"] if ref["rmse_return"] > 0 else np.nan
        if row.n_samples < min_samples:
            verdict = "insufficient_evidence"
        elif row.mae_return < ref["mae_return"] and row.rmse_return < ref["rmse_return"]:
            verdict = "better"
        elif row.mae_return > ref["mae_return"] and row.rmse_return > ref["rmse_return"]:
            verdict = "worse"
        else:
            verdict = "mixed"
        out.append(
            {
                "model": row.model,
                "horizon": int(row.horizon),
                "n_samples": int(row.n_samples),
                "baseline": baseline,
                "mae_return_improvement": float(mae_gain),
                "rmse_return_improvement": float(rmse_gain),
                "directional_accuracy": row.directional_accuracy,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out)


__all__ = [
    "DRIFT",
    "PERSISTENCE",
    "PREDICTION_COLUMNS",
    "Baseline",
    "DriftBaseline",
    "PersistenceBaseline",
    "baseline_predictions",
    "build_prediction_frame",
    "compare_to_baseline",
    "default_baselines",
    "summarize_predictions",
]
