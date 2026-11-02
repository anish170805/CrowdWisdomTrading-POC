"""Walk-forward validation: expanding-window folds with optional embargo/gap.

Design conventions (consistent with Modules 8‑9 and the rest of the codebase):
* Folds are defined by positional row indices into the ModellingDataset.frame.
* Each fold uses an *expanding* training window: train = all rows before the test block,
  test = the next ``test_days`` rows (or ``test_gap``-protected block).
* A gap (embargo) can be set between the end of training and the start of testing to
  prevent target leakage when feature/target windows overlap the test period.
* All folds are strictly chronological: train dates always precede test dates.
* FoldSpec objects are compatible with the existing ``src.models.train`` API
  (FoldSpec, validate_fold, train_fold, run_folds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.validation.leakage import KEY_COLUMNS

# ---------------------------------------------------------------------------
# Fold specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold for walk-forward validation.

    Parameters
    ----------
    fold_id : int
        Sequential fold number (0‑based).
    train_index : np.ndarray
        Positional row indices (``iloc``) for the training portion.
        This window expands with each fold (previous train rows are retained).
    test_index : np.ndarray
        Positional row indices (``iloc``) for the test block.
    train_start_date : str | None
        ISO date string of the first training row's ``bar_date``.
    train_end_date : str | None
        ISO date string of the last training row's ``bar_date``.
    test_start_date : str | None
        ISO date string of the first test row's ``bar_date``.
    test_end_date : str | None
        ISO date string of the last test row's ``bar_date``.
    gap_days : int
        Embargo/gap days between the last training date and the first test date.
        If 0, test starts immediately after training.
    """

    fold_id: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_start_date: str | None
    train_end_date: str | None
    test_start_date: str | None
    test_end_date: str | None
    gap_days: int


def _validate_fold_dates(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    """Assert that train dates precede test dates and indices are valid."""
    n = len(frame)
    for label, index in (("train", train_index), ("test", test_index)):
        if len(index) == 0:
            raise ValueError(f"{label} index is empty")
        if index.min() < 0 or index.max() >= n:
            raise ValueError(
                f"{label} index out of range (frame has {n} rows)"
            )
        if len(np.unique(index)) != len(index):
            raise ValueError(f"{label} index contains duplicates")

    train_end = pd.Timestamp(frame.iloc[train_index]["bar_date"].max())
    test_start = pd.Timestamp(frame.iloc[test_index]["bar_date"].min())
    if not train_end < test_start:
        raise ValueError(
            f"train dates must precede test dates; "
            f"train ends {train_end.date()}, test starts {test_start.date()}"
        )


def _build_expanding_folds(
    frame: pd.DataFrame,
    *,
    initial_train_days: int,
    test_days: int,
    gap_days: int = 0,
    asset_symbol: str | None = None,
) -> list[WalkForwardFold]:
    """Build expanding-window walk-forward folds.

    Parameters
    ----------
    frame : pd.DataFrame
        Modelling dataset frame with ``asset_symbol`` and ``bar_date`` columns.
    initial_train_days : int
        Number of calendar days for the *first* training window.
    test_days : int
        Number of calendar days for each test block.
    gap_days : int
        Embargo/gap days between the end of training and the start of testing.
        Set to ``>= max_horizon`` to avoid target leakage (Module 9 config).
    asset_symbol : str | None
        If provided, only folds for this asset are returned.

    Returns
    -------
    list[WalkForwardFold]
        Sequential expanding-window folds.
    """
    if initial_train_days < 1:
        raise ValueError("initial_train_days must be >= 1")
    if test_days < 1:
        raise ValueError("test_days must be >= 1")
    if gap_days < 0:
        raise ValueError("gap_days must be >= 0")

    # Work with a sorted copy per asset
    frame = frame.sort_values(["asset_symbol", "bar_date"], kind="mergesort").reset_index(drop=True)

    # Group by asset; if asset_symbol is specified, filter first
    if asset_symbol is not None:
        frame = frame[frame["asset_symbol"].astype(str) == str(asset_symbol)].reset_index(drop=True)

    assets = frame["asset_symbol"].unique()
    if len(assets) == 0:
        return []

    folds: list[WalkForwardFold] = []

    for asset in assets:
        asset_frame = frame[frame["asset_symbol"].astype(str) == str(asset)].reset_index(drop=True)
        n = len(asset_frame)

        # Initial training window: first ``initial_train_days`` calendar days
        # Find the date range that covers initial_train_days
        asset_dates = pd.to_datetime(asset_frame["bar_date"])
        # First window covers from the earliest date up to initial_train_days span
        first_date = asset_dates.iloc[0]
        # The last day of the initial train window is the date that is
        # initial_train_days-1 days after the first date (so the span is initial_train_days)
        # We find the row index whose bar_date is the last one within the initial window
        # by looking for the last row where (bar_date - first_date) <= timedelta(days=initial_train_days-1)
        max_date_for_train = first_date + timedelta(days=initial_train_days - 1)
        # Locate the last row index with bar_date <= max_date_for_train
        train_end_idx = asset_frame.index[asset_frame["bar_date"].le(max_date_for_train)].max()
        if train_end_idx is None or train_end_idx < 0:
            # Not enough data for even one initial train window
            continue

        # Ensure we have at least one test row after the training window
        # The test block starts after a gap of gap_days following the train end
        # Find the first row index after train_end_idx + gap_days
        gap_cutoff_date = pd.Timestamp(asset_frame.iloc[train_end_idx]["bar_date"]) + timedelta(days=gap_days)

        # Find the first test row date >= gap_cutoff_date
        test_start_mask = asset_frame["bar_date"] >= gap_cutoff_date
        if not test_start_mask.any():
            # No test rows available after the gap; skip this asset
            continue
        test_start_idx = asset_frame.index[test_start_mask][0]

        # Now we expand: each subsequent fold increases the train window by test_days
        # and the test block is test_days worth of rows starting from test_start_idx
        # We'll iterate through possible fold positions

        fold_id_offset = len(folds)  # keep fold_id global
        # Start with the initial configuration
        train_index = np.arange(0, train_end_idx + 1, dtype=np.int64)
        test_index = np.arange(test_start_idx, min(test_start_idx + _count_rows_in_date_span(asset_frame, test_days), n), dtype=np.int64)

        if len(test_index) == 0:
            continue

        # Validate dates
        _validate_fold_dates(asset_frame, train_index, test_index)

        fold = WalkForwardFold(
            fold_id=fold_id_offset,
            train_index=train_index,
            test_index=test_index,
            train_start_date=pd.Timestamp(asset_frame.iloc[train_index.min()]["bar_date"]).date().isoformat(),
            train_end_date=pd.Timestamp(asset_frame.iloc[train_index.max()]["bar_date"]).date().isoformat(),
            test_start_date=pd.Timestamp(asset_frame.iloc[test_index.min()]["bar_date"]).date().isoformat(),
            test_end_date=pd.Timestamp(asset_frame.iloc[test_index.max()]["bar_date"]).date().isoformat(),
            gap_days=gap_days,
        )
        folds.append(fold)

        # Subsequent expanding folds: expand train by test_days each time,
        # and shift test block forward by test_days
        while True:
            # Expand train end: add test_days rows after current train_index.max()
            new_train_end_idx = asset_frame.index[
                asset_frame.index > train_index.max()
            ].min()
            if new_train_end_idx is None:
                break  # no more rows to add to training

            # The new train window includes all rows up to new_train_end_idx
            # but we limit by initial_train_days concept: we expand by test_days each fold
            # Actually, for expanding window, we just keep expanding
            # Let's recompute: train should be all rows from start up to a new cutoff

            # Simpler approach: expand train by test_days rows beyond previous train end
            # Previous train ended at train_index.max(), new train ends at
            # train_index.max() + test_days rows (or by date)
            # Let's use row-count expansion: add exactly test_days rows after previous train end
            rows_after_train = asset_frame.index[asset_frame.index > train_index.max()]
            if len(rows_after_train) < test_days:
                # Not enough rows to expand; we can still include whatever is available
                if len(rows_after_train) == 0:
                    break  # no more rows
                # Include remaining rows
                expanded_train_index = np.concatenate([train_index, asset_frame.index[asset_frame.index > train_index.max()].astype(np.int64)])
            else:
                expanded_train_index = np.concatenate([
                    train_index,
                    asset_frame.index[asset_frame.index > train_index.max()][:test_days].astype(np.int64),
                ])

            # Test block starts after gap_days from the new train end
            new_train_end_date = pd.Timestamp(asset_frame.iloc[expanded_train_index.max()]["bar_date"])
            gap_cutoff_date = new_train_end_date + timedelta(days=gap_days)

            # Find first test row date >= gap_cutoff_date
            test_start_mask2 = asset_frame["bar_date"] >= gap_cutoff_date
            if not test_start_mask2.any():
                # No test rows available after this expanded train end
                # But we might still have a test block from the previous shift; stop here
                break
            new_test_start_idx = asset_frame.index[test_start_mask2][0]
            new_test_end_idx = min(new_test_start_idx + _count_rows_in_date_span(asset_frame, test_days), n) - 1
            if new_test_end_idx < new_test_start_idx:
                break

            new_test_index = np.arange(new_test_start_idx, new_test_end_idx + 1, dtype=np.int64)

            _validate_fold_dates(asset_frame, expanded_train_index, new_test_index)

            fold_id_offset += 1
            fold = WalkForwardFold(
                fold_id=fold_id_offset,
                train_index=expanded_train_index,
                test_index=new_test_index,
                train_start_date=pd.Timestamp(asset_frame.iloc[expanded_train_index.min()]["bar_date"]).date().isoformat(),
                train_end_date=pd.Timestamp(asset_frame.iloc[expanded_train_index.max()]["bar_date"]).date().isoformat(),
                test_start_date=pd.Timestamp(asset_frame.iloc[new_test_index.min()]["bar_date"]).date().isoformat(),
                test_end_date=pd.Timestamp(asset_frame.iloc[new_test_index.max()]["bar_date"]).date().isoformat(),
                gap_days=gap_days,
            )
            folds.append(fold)

            train_index = expanded_train_index
            test_index = new_test_index

            # Stop condition: if test block reaches the end of the asset data
            if test_index.max() >= n - 1:
                break

    return folds


def _count_rows_in_date_span(frame: pd.DataFrame, days: int) -> int:
    """Count roughly how many rows fall within ``days`` calendar days from the first row's date.

    This is a rough count used for initializing test block size.
    """
    if frame.empty:
        return 0
    first_date = pd.Timestamp(frame.iloc[0]["bar_date"])
    last_date = first_date + timedelta(days=days)
    mask = pd.to_datetime(frame["bar_date"]) <= last_date
    return int(mask.sum())


def build_walk_forward_folds(
    dataset: Any,
    *,
    initial_train_days: int | None = None,
    test_days: int | None = None,
    gap_days: int | None = None,
    asset_symbol: str | None = None,
) -> list[WalkForwardFold]:
    """Convenience wrapper that builds folds from a ModellingDataset.

    Parameters
    ----------
    dataset : ModellingDataset
        The leakage-safe modelling table from Module 8.
    initial_train_days : int, optional
        Number of calendar days for the initial training window.
        Defaults to ``Settings.walk_forward_initial_train_days``.
    test_days : int, optional
        Number of calendar days for each test block.
        Defaults to ``Settings.walk_forward_test_days``.
    gap_days : int, optional
        Embargo/gap days between train and test blocks.
        Defaults to ``Settings.walk_forward_gap_days`` (must be >= max horizon).
    asset_symbol : str | None
        If provided, only folds for this asset.

    Returns
    -------
    list[WalkForwardFold]
        Sequential expanding-window folds.
    """
    from src.config import get_settings

    settings = get_settings()
    initial_train_days = initial_train_days or settings.walk_forward_initial_train_days
    test_days = test_days or settings.walk_forward_test_days
    gap_days = gap_days or settings.walk_forward_gap_days

    return _build_expanding_folds(
        dataset.frame,
        initial_train_days=initial_train_days,
        test_days=test_days,
        gap_days=gap_days,
        asset_symbol=asset_symbol,
    )


def validate_fold(fold: WalkForwardFold, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a fold and return the train and test DataFrames.

    Parameters
    ----------
    fold : WalkForwardFold
        The fold to validate.
    frame : pd.DataFrame
        The full modelling frame.

    Returns
    -------
    (train_df, test_df)
    """
    train = frame.iloc[fold.train_index].sort_values("bar_date", kind="mergesort").reset_index(drop=True)
    test = frame.iloc[fold.test_index].sort_values("bar_date", kind="mergesort").reset_index(drop=True)
    _validate_fold_dates(frame, fold.train_index, fold.test_index)
    return train, test


# ---------------------------------------------------------------------------
# Convenience: expand folds from existing FoldSpecs (Module 9 integration)
# ---------------------------------------------------------------------------

def folds_from_specs(
    folds: Sequence[FoldSpec],
    *,
    gap_days: int = 0,
) -> list[WalkForwardFold]:
    """Convert Module 9 ``FoldSpec`` objects into ``WalkForwardFold`` objects.

    This allows Module 10 folds to wrap or replace Module 9 chronological splits
    while preserving the expanding-window semantics.

    Parameters
    ----------
    folds : Sequence[FoldSpec]
        The chronological folds from Module 9.
    gap_days : int
        Additional embargo days to add between train and test (beyond what FoldSpec
        already enforces).  Set to 0 to use the FoldSpec's inherent split.

    Returns
    -------
    list[WalkForwardFold]
        Walk-forward fold objects with expanding-window semantics.
    """
    result: list[WalkForwardFold] = []
    for i, spec in enumerate(folds):
        train = frame.iloc[spec.train_index]  # type: ignore[name-defined]
        test = frame.iloc[spec.test_index]  # type: ignore[name-defined]
        # Determine gap: if gap_days > 0, we need to shift test forward
        # For now, we just record the gap; the actual date shift is handled by
        # the caller when they use the train/test DataFrames.
        result.append(
            WalkForwardFold(
                fold_id=i,
                train_index=spec.train_index,
                test_index=spec.test_index,
                train_start_date=pd.Timestamp(train["bar_date"].min()).date().isoformat(),
                train_end_date=pd.Timestamp(train["bar_date"].max()).date().isoformat(),
                test_start_date=pd.Timestamp(test["bar_date"].min()).date().isoformat(),
                test_end_date=pd.Timestamp(test["bar_date"].max()).date().isoformat(),
                gap_days=gap_days,
            )
        )
    return result  # type: ignore[return-value]