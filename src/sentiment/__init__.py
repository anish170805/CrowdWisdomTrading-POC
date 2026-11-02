"""Sentiment analysis module for the X sentiment to asset price prediction pipeline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SentimentResult", "score_all_posts", "finite_check"]