"""Feature engineering for Module 8 — Feature engineering and modelling dataset.

Price feature construction (lags, volatility, moving-average distances, volume
changes) and dataset assembly with leakage-safe target construction.
"""

from __future__ import annotations

from .price_features import build_price_features

__all__ = ["build_price_features"]