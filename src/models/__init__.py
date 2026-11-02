"""Module 9 — baselines and model training."""

from __future__ import annotations

from src.models.artifacts import ModelBundle, load_bundle, save_bundle
from src.models.baseline import (
    DriftBaseline,
    PersistenceBaseline,
    baseline_predictions,
    compare_to_baseline,
    summarize_predictions,
)
from src.models.train import (
    FoldSpec,
    ModelConfig,
    fit_bundle,
    run_folds,
    train_final,
    train_fold,
)

__all__ = [
    "DriftBaseline",
    "FoldSpec",
    "ModelBundle",
    "ModelConfig",
    "PersistenceBaseline",
    "baseline_predictions",
    "compare_to_baseline",
    "fit_bundle",
    "load_bundle",
    "run_folds",
    "save_bundle",
    "summarize_predictions",
    "train_final",
    "train_fold",
]
