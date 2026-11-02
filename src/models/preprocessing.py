"""Fold-local feature preprocessing for Module 9.

The preprocessor is *fitted on a training window only* and then applied unchanged
to validation/test/live rows, so no statistic ever leaks backwards in time.

Missing features are handled explicitly:

* ``"native"`` (default): nulls are kept as ``NaN``.  XGBoost routes missing values
  natively, and Module 8 deliberately keeps absent sentiment null rather than
  filling it with a neutral value.
* ``"median"``: nulls are replaced by the per-feature median of the *training*
  window.  A feature that is entirely null in the training window has no median;
  it stays ``NaN`` and is listed in ``all_null_in_train``.

Infinite values and missing columns are never silently repaired: they raise
:class:`FeatureMatrixError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

MISSING_STRATEGIES: tuple[str, ...] = ("native", "median")


class FeatureMatrixError(ValueError):
    """Raised when a frame cannot be turned into the exact training feature matrix."""


class FeaturePreprocessor:
    """Select, order, validate and (optionally) impute the model features."""

    def __init__(self, feature_columns: Sequence[str], strategy: str = "native") -> None:
        columns = tuple(str(c) for c in feature_columns)
        if not columns:
            raise FeatureMatrixError("at least one feature column is required")
        if len(set(columns)) != len(columns):
            raise FeatureMatrixError("feature_columns contains duplicates")
        if strategy not in MISSING_STRATEGIES:
            raise FeatureMatrixError(
                f"unknown missing-value strategy {strategy!r}; use one of {MISSING_STRATEGIES}"
            )
        self.feature_columns: tuple[str, ...] = columns
        self.strategy = strategy
        self.fill_values: dict[str, float | None] | None = None
        self.train_null_rate: dict[str, float] | None = None
        self.all_null_in_train: tuple[str, ...] = ()
        self.n_fit_rows: int = 0

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.train_null_rate is not None

    def validate_columns(self, columns: Sequence[str], *, strict_order: bool = False) -> None:
        """Raise if required features are absent (and, if asked, reordered)."""
        present = list(columns)
        missing = [c for c in self.feature_columns if c not in present]
        if missing:
            raise FeatureMatrixError(f"required feature column(s) missing: {missing}")
        if strict_order:
            wanted = set(self.feature_columns)
            subset = [c for c in present if c in wanted]
            if tuple(subset) != self.feature_columns:
                raise FeatureMatrixError(
                    "feature columns are reordered relative to the training order: "
                    f"expected {list(self.feature_columns)}, got {subset}"
                )

    def _numeric(self, frame: pd.DataFrame, *, strict_order: bool = False) -> pd.DataFrame:
        self.validate_columns(frame.columns, strict_order=strict_order)
        try:
            matrix = frame.loc[:, list(self.feature_columns)].astype("float64")
        except (TypeError, ValueError) as exc:
            raise FeatureMatrixError(f"feature columns must be numeric: {exc}") from exc
        flags = np.isinf(matrix.to_numpy()).any(axis=0)
        if flags.any():
            bad = [c for c, flag in zip(matrix.columns, flags, strict=True) if flag]
            raise FeatureMatrixError(f"infinite values in feature column(s): {bad}")
        return matrix

    # ------------------------------------------------------------------
    def fit(self, train_frame: pd.DataFrame) -> FeaturePreprocessor:
        """Learn statistics from the training window only."""
        if len(train_frame) == 0:
            raise FeatureMatrixError("cannot fit the preprocessor on an empty training window")
        matrix = self._numeric(train_frame)
        self.n_fit_rows = int(len(matrix))
        self.train_null_rate = {c: float(matrix[c].isna().mean()) for c in matrix.columns}
        self.all_null_in_train = tuple(c for c in matrix.columns if matrix[c].isna().all())
        if self.strategy == "median":
            medians = matrix.median(skipna=True)
            self.fill_values = {
                c: (None if pd.isna(medians[c]) else float(medians[c])) for c in matrix.columns
            }
        else:
            self.fill_values = None
        return self

    def transform(self, frame: pd.DataFrame, *, strict_order: bool = False) -> np.ndarray:
        """Return the ``float64`` matrix in the exact training column order."""
        if not self.is_fitted:
            raise FeatureMatrixError("preprocessor is not fitted")
        matrix = self._numeric(frame, strict_order=strict_order)
        if self.strategy == "median" and self.fill_values is not None:
            fills = {c: v for c, v in self.fill_values.items() if v is not None}
            matrix = matrix.fillna(fills)
        return matrix.to_numpy(dtype="float64")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise FeatureMatrixError("preprocessor is not fitted")
        return {
            "feature_columns": list(self.feature_columns),
            "strategy": self.strategy,
            "fill_values": self.fill_values,
            "train_null_rate": self.train_null_rate,
            "all_null_in_train": list(self.all_null_in_train),
            "n_fit_rows": self.n_fit_rows,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeaturePreprocessor:
        pre = cls(payload["feature_columns"], payload["strategy"])
        pre.fill_values = payload.get("fill_values")
        pre.train_null_rate = dict(payload["train_null_rate"])
        pre.all_null_in_train = tuple(payload.get("all_null_in_train", ()))
        pre.n_fit_rows = int(payload.get("n_fit_rows", 0))
        return pre


__all__ = ["MISSING_STRATEGIES", "FeatureMatrixError", "FeaturePreprocessor"]
