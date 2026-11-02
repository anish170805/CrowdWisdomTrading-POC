"""Reporting module for the X sentiment to asset price prediction pipeline.

Generates the one-page outlook and transparent evaluation report at the
end of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config import Settings
from src.prediction.latest import PredictionSnapshot
from src.features.dataset import ModellingDataset

__all__ = [
    "generate_outlook_report",
    "generate_evaluation_report",
]