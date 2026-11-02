"""Module 10 — Walk-forward validation and trading metrics.

Exports
-------
forecast : MAE, RMSE, directional accuracy per horizon/model.
trading  : Strategy signals, returns, Sharpe/Sortino/max_drawdown.
walk_forward : Expanding-window fold generation (WalkForwardFold, build_walk_forward_folds).
"""

from src.metrics import forecast
from src.metrics import trading
from src.validation.walk_forward import (
    WalkForwardFold,
    build_walk_forward_folds,
    validate_fold,
    folds_from_specs,
)

__all__ = [
    "forecast",
    "trading",
    "WalkForwardFold",
    "build_walk_forward_folds",
    "validate_fold",
    "folds_from_specs",
]