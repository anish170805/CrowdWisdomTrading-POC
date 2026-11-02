"""Per-horizon model training for Module 9.

Design
------
* One independent XGBoost regressor per horizon ``n+1 ... n+k``.  The target is the
  simple return from the observed ``n+0`` close (``target_return_n{h}``); prices are
  reconstructed as ``close_n0 * (1 + return)``.  ``n+0`` is never modelled.
* Splits are *supplied*, never drawn: callers pass :class:`FoldSpec` objects (Module 10
  owns the chronological procedure).  There is deliberately no random-split API, and
  every fold is checked to be strictly chronological.
* Targets that have not matured by the test start are *purged* from training: a
  training row whose ``n+h`` target date is on/after the first test ``bar_date`` is
  dropped for that horizon, so no training label overlaps the test period even if the
  embargo chosen by Module 10 is shorter than the horizon in trading days.
* Preprocessing (:class:`~src.models.preprocessing.FeaturePreprocessor`) is fitted on
  the training window of each fold only.
* Baselines (persistence, drift) are produced for every fold and horizon on exactly
  the same test rows as the model, so scores are comparable.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.artifacts import (
    ModelBundle,
    library_versions,
    save_bundle,
    save_bundle_pkl,
    target_definition,
)
from src.models.baseline import (
    PREDICTION_COLUMNS,
    Baseline,
    baseline_predictions,
    build_prediction_frame,
    compare_to_baseline,
    default_baselines,
    summarize_predictions,
)
from src.models.preprocessing import MISSING_STRATEGIES, FeaturePreprocessor
from src.validation.leakage import (
    CLOSE_N0,
    TARGET_DATE_FMT,
    TARGET_RETURN_FMT,
)

MODEL_TYPE = "xgboost"
DEFAULT_MIN_SAMPLES_FOR_CLAIM = 30


class TrainingError(ValueError):
    """Raised for unusable folds, datasets, or configuration."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Everything that determines a trained model (besides the data)."""

    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    random_seed: int = 42
    min_train_samples: int = 60
    missing_strategy: str = "native"

    def __post_init__(self) -> None:
        horizons = tuple(sorted({int(h) for h in self.horizons}))
        if not horizons:
            raise ValueError("at least one horizon is required")
        if horizons[0] < 1:
            raise ValueError("horizons must be >= 1: n+0 is the observed close, not a forecast")
        if self.missing_strategy not in MISSING_STRATEGIES:
            raise ValueError(f"missing_strategy must be one of {MISSING_STRATEGIES}")
        if self.min_train_samples < 2:
            raise ValueError("min_train_samples must be >= 2")
        object.__setattr__(self, "horizons", horizons)

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> ModelConfig:
        """Build from the Module 0 ``Settings``."""
        params: dict[str, Any] = {
            "horizons": tuple(settings.horizons),
            "n_estimators": settings.model_n_estimators,
            "max_depth": settings.model_max_depth,
            "learning_rate": settings.model_learning_rate,
            "random_seed": settings.model_random_seed,
            "min_train_samples": settings.min_train_samples,
        }
        params.update(overrides)
        return cls(**params)

    def xgb_params(self) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_seed,
            "tree_method": "hist",
            "n_jobs": 1,  # single thread keeps results bit-reproducible
            "verbosity": 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "horizons": list(self.horizons),
            "min_train_samples": self.min_train_samples,
            "missing_strategy": self.missing_strategy,
            "random_seed": self.random_seed,
            "xgb_params": self.xgb_params(),
        }


@dataclass(frozen=True)
class FoldSpec:
    """A chronological split supplied by Module 10.

    ``train_index`` / ``test_index`` are *positional* row indices into
    ``dataset.frame`` (``iloc``).
    """

    fold_id: int
    train_index: np.ndarray
    test_index: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_index", np.asarray(self.train_index, dtype="int64"))
        object.__setattr__(self, "test_index", np.asarray(self.test_index, dtype="int64"))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _single_asset(frame: pd.DataFrame, asset: str | None) -> str:
    assets = sorted(frame["asset_symbol"].astype(str).unique())
    if asset is None:
        if len(assets) != 1:
            raise TrainingError(
                f"dataset holds {len(assets)} assets {assets}; pass asset_symbol=... "
                "(models are trained per asset)"
            )
        return assets[0]
    if asset not in assets:
        raise TrainingError(f"asset {asset!r} not in dataset (found {assets})")
    return asset


def _check_dataset(dataset: Any, config: ModelConfig) -> None:
    missing_targets = [
        h for h in config.horizons if TARGET_RETURN_FMT.format(h=h) not in dataset.frame.columns
    ]
    if missing_targets:
        raise TrainingError(
            f"dataset has no target column for horizon(s) {missing_targets}; "
            f"dataset horizons are {list(dataset.horizons)}"
        )
    if CLOSE_N0 not in dataset.frame.columns:
        raise TrainingError(f"dataset is missing {CLOSE_N0!r}")


def validate_fold(
    frame: pd.DataFrame, fold: FoldSpec, asset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` rows after asserting a strictly chronological split."""
    n = len(frame)
    for label, index in (("train", fold.train_index), ("test", fold.test_index)):
        if len(index) == 0:
            raise TrainingError(f"fold {fold.fold_id}: empty {label} index")
        if index.min() < 0 or index.max() >= n:
            raise TrainingError(f"fold {fold.fold_id}: {label} index out of range")
        if len(np.unique(index)) != len(index):
            raise TrainingError(f"fold {fold.fold_id}: duplicate {label} rows")
    if np.intersect1d(fold.train_index, fold.test_index).size:
        raise TrainingError(f"fold {fold.fold_id}: train and test rows overlap")
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index]
    for label, part in (("train", train), ("test", test)):
        extra = sorted(set(part["asset_symbol"].astype(str)) - {asset})
        if extra:
            raise TrainingError(f"fold {fold.fold_id}: {label} contains other assets {extra}")
    train_end = pd.Timestamp(train["bar_date"].max())
    test_start = pd.Timestamp(test["bar_date"].min())
    if not train_end < test_start:
        raise TrainingError(
            f"fold {fold.fold_id}: train dates must precede test dates "
            f"(train ends {train_end.date()}, test starts {test_start.date()}); "
            "chronological splits only, never random"
        )
    return train, test


def mature_targets(
    train: pd.DataFrame, horizons: Sequence[int], before: pd.Timestamp
) -> pd.DataFrame:
    """Copy of ``train`` with targets nulled whose realisation date is ``>= before``.

    Nulling (rather than dropping rows) keeps the feature rows available for
    preprocessing statistics while removing every label that would overlap the test
    period.
    """
    out = train.copy()
    for h in horizons:
        ret = TARGET_RETURN_FMT.format(h=h)
        dates = pd.to_datetime(out[TARGET_DATE_FMT.format(h=h)])
        out.loc[~(dates < before), ret] = np.nan
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _window(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(frame)),
        "first_bar_date": pd.Timestamp(frame["bar_date"].min()).date().isoformat(),
        "last_bar_date": pd.Timestamp(frame["bar_date"].max()).date().isoformat(),
    }


def fit_bundle(
    dataset: Any,
    train_index: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    purge_before: pd.Timestamp | None = None,
) -> ModelBundle:
    """Fit one XGBoost model per horizon on the supplied training rows only.

    ``purge_before`` is the first date the model will be evaluated/used on; labels
    realised on/after it are excluded.  Horizons with fewer than
    ``config.min_train_samples`` usable rows are *skipped and recorded*, not trained.
    """
    from xgboost import XGBRegressor  # lazy: baselines must run without xgboost

    _check_dataset(dataset, config)
    frame: pd.DataFrame = dataset.frame
    index = np.asarray(train_index, dtype="int64")
    if len(index) == 0:
        raise TrainingError("empty training window")
    train = frame.iloc[index].sort_values("bar_date", kind="mergesort")
    asset = _single_asset(train, asset_symbol)
    other_assets = sorted(set(train["asset_symbol"].astype(str)) - {asset})
    if other_assets:
        raise TrainingError(f"training window contains other assets {other_assets}")
    if train["bar_date"].duplicated().any():
        raise TrainingError("training window has duplicate bar_date rows for one asset")
    if purge_before is not None:
        train = mature_targets(train, config.horizons, pd.Timestamp(purge_before))

    preprocessor = FeaturePreprocessor(dataset.feature_columns, config.missing_strategy).fit(train)

    models: dict[int, Any] = {}
    per_horizon: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    original = frame.iloc[index]
    for h in config.horizons:
        ret = TARGET_RETURN_FMT.format(h=h)
        before = int(original[ret].notna().sum())
        rows = train[train[ret].notna()]
        rows = rows[np.isfinite(rows[ret].to_numpy(dtype="float64"))]
        info = {
            "n_labeled_before_purge": before,
            "n_train": int(len(rows)),
            "n_purged": before - int(len(rows)),
            "target": target_definition(h),
        }
        if len(rows) < config.min_train_samples:
            skipped[str(h)] = (
                f"only {len(rows)} usable training rows (< min_train_samples="
                f"{config.min_train_samples})"
            )
            info["trained"] = False
        else:
            model = XGBRegressor(**config.xgb_params())
            model.fit(preprocessor.transform(rows), rows[ret].to_numpy(dtype="float64"))
            models[h] = model
            info["trained"] = True
            info["last_target_date"] = pd.Timestamp(
                rows[TARGET_DATE_FMT.format(h=h)].max()
            ).date().isoformat()
        per_horizon[str(h)] = info

    window = _window(train)
    config_dict = config.to_dict()
    manifest: dict[str, Any] = {
        "asset_symbol": asset,
        "model_type": MODEL_TYPE,
        "dataset_version": getattr(dataset, "version", None),
        "dataset_horizons": list(dataset.horizons),
        "config": config_dict,
        "random_seed": config.random_seed,
        "feature_columns": list(dataset.feature_columns),
        "feature_count": len(dataset.feature_columns),
        "training_window": {**window, "purge_before": (
            None if purge_before is None else pd.Timestamp(purge_before).date().isoformat()
        )},
        "per_horizon": per_horizon,
        "skipped_horizons": skipped,
        "n0_policy": "n+0 is the observed latest close; no model is trained for it",
        "library_versions": library_versions(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    return ModelBundle(manifest=_jsonable(manifest), preprocessor=preprocessor, models=models)


def _gridsearch_param_grid() -> dict[str, list[Any]]:
    """Hyperparameter grid searched per horizon."""
    return {
        "n_estimators": [200, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0.0],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0],
        "min_child_weight": [1],
    }


def fit_bundle_with_gridsearch(
    dataset: Any,
    train_index: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    purge_before: pd.Timestamp | None = None,
    cv_splits: int = 2,
) -> ModelBundle:
    """Fit one XGBoost model per horizon using GridSearchCV to select hyper-parameters.

    For each horizon, GridSearchCV with TimeSeriesSplit searches the param grid
    on the purged training rows.  The best estimator is then refit on the full
    training window for that horizon.

    Returns a ModelBundle with the best-performing model per horizon.
    """
    from xgboost import XGBRegressor

    _check_dataset(dataset, config)
    frame: pd.DataFrame = dataset.frame
    index = np.asarray(train_index, dtype="int64")
    if len(index) == 0:
        raise TrainingError("empty training window")
    train = frame.iloc[index].sort_values("bar_date", kind="mergesort")
    asset = _single_asset(train, asset_symbol)
    other_assets = sorted(set(train["asset_symbol"].astype(str)) - {asset})
    if other_assets:
        raise TrainingError(f"training window contains other assets {other_assets}")
    if train["bar_date"].duplicated().any():
        raise TrainingError("training window has duplicate bar_date rows for one asset")
    if purge_before is not None:
        train = mature_targets(train, config.horizons, pd.Timestamp(purge_before))

    preprocessor = FeaturePreprocessor(dataset.feature_columns, config.missing_strategy).fit(train)

    models: dict[int, Any] = {}
    per_horizon: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    original = frame.iloc[index]
    param_grid = _gridsearch_param_grid()
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for h in config.horizons:
        ret = TARGET_RETURN_FMT.format(h=h)
        before = int(original[ret].notna().sum())
        rows = train[train[ret].notna()]
        rows = rows[np.isfinite(rows[ret].to_numpy(dtype="float64"))]
        info = {
            "n_labeled_before_purge": before,
            "n_train": int(len(rows)),
            "n_purged": before - int(len(rows)),
            "target": target_definition(h),
            "gridsearch": True,
        }
        if len(rows) < config.min_train_samples:
            skipped[str(h)] = (
                f"only {len(rows)} usable training rows (< min_train_samples="
                f"{config.min_train_samples})"
            )
            info["trained"] = False
        else:
            X = preprocessor.transform(rows)
            y = rows[ret].to_numpy(dtype="float64")

            base = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=1,
                random_state=config.random_seed,
                verbosity=0,
            )
            search = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=tscv,
                n_jobs=1,
                verbose=0,
            )
            search.fit(X, y)
            best = search.best_estimator_

            models[h] = best
            info["trained"] = True
            info["best_params"] = search.best_params_
            info["best_cv_mae"] = float(-search.best_score_)
            info["last_target_date"] = pd.Timestamp(
                rows[TARGET_DATE_FMT.format(h=h)].max()
            ).date().isoformat()
        per_horizon[str(h)] = info

    window = _window(train)
    config_dict = config.to_dict()
    config_dict["gridsearch"] = True
    manifest: dict[str, Any] = {
        "asset_symbol": asset,
        "model_type": MODEL_TYPE,
        "dataset_version": getattr(dataset, "version", None),
        "dataset_horizons": list(dataset.horizons),
        "config": config_dict,
        "random_seed": config.random_seed,
        "feature_columns": list(dataset.feature_columns),
        "feature_count": len(dataset.feature_columns),
        "training_window": {**window, "purge_before": (
            None if purge_before is None else pd.Timestamp(purge_before).date().isoformat()
        )},
        "per_horizon": per_horizon,
        "skipped_horizons": skipped,
        "n0_policy": "n+0 is the observed latest close; no model is trained for it",
        "library_versions": library_versions(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    return ModelBundle(manifest=_jsonable(manifest), preprocessor=preprocessor, models=models)


def make_run_id(bundle: ModelBundle) -> str:
    """Deterministic id: asset, dataset version, window end and a config hash."""
    manifest = bundle.manifest
    payload = json.dumps(
        {
            "config": manifest["config"],
            "features": manifest["feature_columns"],
            "window": manifest["training_window"],
            "dataset": manifest["dataset_version"],
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    end = manifest["training_window"]["last_bar_date"].replace("-", "")
    asset = "".join(ch if ch.isalnum() else "_" for ch in manifest["asset_symbol"])
    return f"{asset}_{end}_{digest}"


def train_final_with_gridsearch(
    dataset: Any,
    train_index: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    models_dir: str | Path | None = None,
    cv_splits: int = 3,
) -> tuple[ModelBundle, Path | None]:
    """Train with GridSearchCV and save the bundle as a pickle file."""
    bundle = fit_bundle_with_gridsearch(
        dataset, train_index, config, asset_symbol=asset_symbol
    )
    path = save_bundle_pkl(bundle, models_dir, make_run_id(bundle)) if models_dir is not None else None
    return bundle, path
    """Deterministic id: asset, dataset version, window end and a config hash."""
    manifest = bundle.manifest
    payload = json.dumps(
        {
            "config": manifest["config"],
            "features": manifest["feature_columns"],
            "window": manifest["training_window"],
            "dataset": manifest["dataset_version"],
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    end = manifest["training_window"]["last_bar_date"].replace("-", "")
    asset = "".join(ch if ch.isalnum() else "_" for ch in manifest["asset_symbol"])
    return f"{asset}_{end}_{digest}"


def train_final(
    dataset: Any,
    train_index: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    models_dir: str | Path | None = None,
) -> tuple[ModelBundle, Path | None]:
    """Train the bundle Module 11 will load, on a window supplied by Module 10.

    ``train_index`` is required: this function never defaults to "all rows", so a
    final model can only be trained on a window that Module 10 has defined.
    """
    bundle = fit_bundle(dataset, train_index, config, asset_symbol=asset_symbol)
    path = save_bundle(bundle, models_dir, make_run_id(bundle)) if models_dir is not None else None
    return bundle, path


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    fold_id: int
    bundle: ModelBundle
    predictions: pd.DataFrame
    metadata: dict[str, Any]


def train_fold(
    dataset: Any,
    fold: FoldSpec,
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    baselines: Sequence[Baseline] | None = None,
) -> FoldResult:
    """Train on ``fold.train_index`` and predict ``fold.test_index`` (baselines + model)."""
    _check_dataset(dataset, config)
    asset = _single_asset(dataset.frame, asset_symbol)
    train, test = validate_fold(dataset.frame, fold, asset)
    test_start = pd.Timestamp(test["bar_date"].min())

    bundle = fit_bundle(
        dataset, fold.train_index, config, asset_symbol=asset, purge_before=test_start
    )
    mature_train = mature_targets(train, config.horizons, test_start)
    predictions = [
        baseline_predictions(
            test,
            config.horizons,
            baselines if baselines is not None else default_baselines(),
            fold_id=fold.fold_id,
            train_frame=mature_train,
        )
    ]
    for h in bundle.horizons:
        labeled = test[test[TARGET_RETURN_FMT.format(h=h)].notna()]
        if labeled.empty:
            continue
        predictions.append(
            build_prediction_frame(
                labeled,
                h,
                bundle.predict_return(labeled, h),
                model=MODEL_TYPE,
                fold_id=fold.fold_id,
            )
        )
    frame = pd.concat(predictions, ignore_index=True)

    metadata = {
        "fold_id": int(fold.fold_id),
        "asset_symbol": asset,
        "train": _window(train),
        "test": _window(test),
        "trained_horizons": list(bundle.horizons),
        "skipped_horizons": bundle.manifest["skipped_horizons"],
        "per_horizon": bundle.manifest["per_horizon"],
    }
    return FoldResult(int(fold.fold_id), bundle, frame, metadata)


@dataclass
class TrainingRun:
    """All folds of one evaluation plus pooled predictions and scores."""

    folds: list[FoldResult]
    predictions: pd.DataFrame
    summary: pd.DataFrame
    comparison: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)

    def fold_metadata(self) -> list[dict[str, Any]]:
        return [fold.metadata for fold in self.folds]


def run_folds(
    dataset: Any,
    folds: Sequence[FoldSpec],
    config: ModelConfig,
    *,
    asset_symbol: str | None = None,
    baselines: Sequence[Baseline] | None = None,
    min_samples_for_claim: int = DEFAULT_MIN_SAMPLES_FOR_CLAIM,
) -> TrainingRun:
    """Train and score every fold; compare the model with persistence per horizon."""
    if not folds:
        raise TrainingError("no folds supplied; Module 10 must provide the chronological folds")
    results = [
        train_fold(dataset, fold, config, asset_symbol=asset_symbol, baselines=baselines)
        for fold in sorted(folds, key=lambda f: f.fold_id)
    ]
    predictions = pd.concat([r.predictions for r in results], ignore_index=True)
    summary = summarize_predictions(predictions)
    comparison = (
        compare_to_baseline(summary, min_samples=min_samples_for_claim)
        if not summary.empty
        else pd.DataFrame()
    )
    return TrainingRun(
        results, predictions[list(PREDICTION_COLUMNS)], summary, comparison, config.to_dict()
    )


def save_run(run: TrainingRun, directory: str | Path) -> Path:
    """Persist pooled predictions, scores, comparison and fold metadata."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    run.predictions.to_parquet(path / "fold_predictions.parquet", index=False)
    run.summary.to_csv(path / "metrics_summary.csv", index=False)
    run.comparison.to_csv(path / "baseline_comparison.csv", index=False)
    (path / "fold_metadata.json").write_text(
        json.dumps(_jsonable({"config": run.config, "folds": run.fold_metadata()}), indent=2),
        encoding="utf-8",
    )
    return path


__all__ = [
    "DEFAULT_MIN_SAMPLES_FOR_CLAIM",
    "FoldResult",
    "FoldSpec",
    "ModelConfig",
    "TrainingError",
    "TrainingRun",
    "fit_bundle",
    "fit_bundle_with_gridsearch",
    "make_run_id",
    "mature_targets",
    "run_folds",
    "save_run",
    "train_final",
    "train_final_with_gridsearch",
    "train_fold",
    "validate_fold",
]
