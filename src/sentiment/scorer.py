"""Per-post sentiment scoring (Module 6).

Scores posts with a finance-pretrained transformer (FinBERT by default).

Design notes
------------
* Class probabilities are mapped through the model's own ``config.id2label``;
  nothing assumes a fixed label order (ProsusAI/finbert is
  ``0=positive, 1=negative, 2=neutral``).
* ``score_all_posts`` returns exactly one result per input post, in input order.
* If the transformer cannot be loaded the function **raises**
  :class:`SentimentModelError` by default.  A keyword fallback exists, but only
  when the caller passes ``allow_fallback=True``; its results are labelled
  ``model_name="rule-based-fallback"`` so they are never mistaken for FinBERT
  scores (the pipeline never caches them).
* Caching across runs is owned by ``src.pipeline`` (keyed by post fingerprint and
  model identity), so this module stays stateless.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.config import Settings, get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

FALLBACK_MODEL_NAME = "rule-based-fallback"


class SentimentModelError(RuntimeError):
    """The configured sentiment model could not be loaded or run."""


@dataclass(frozen=True)
class SentimentResult:
    """Sentiment score for one post."""

    post_id: str
    text: str
    positive: float
    negative: float
    neutral: float
    label: str
    signed_score: float  # positive - negative, range [-1, 1]
    model_name: str
    model_revision: str
    scored_at: datetime


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_all_posts(
    posts: Iterable[Any],
    *,
    settings: Settings | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    device: str | None = None,
    allow_fallback: bool = False,
) -> list[SentimentResult]:
    """Score every post; one result per input, same order.

    ``posts`` need ``post_id`` and ``text`` attributes.
    """
    resolved = settings or get_settings()
    post_list = list(posts)
    if not post_list:
        return []

    model_name = model_name or resolved.sentiment_model_name
    revision = resolved.sentiment_model_revision
    batch_size = batch_size or resolved.sentiment_batch_size
    max_length = max_length or resolved.sentiment_max_length
    device = device or resolved.sentiment_device

    try:
        return _score_with_transformers(
            post_list, model_name, revision, batch_size, max_length, device
        )
    except Exception as exc:  # noqa: BLE001 - surfaced or explicitly degraded below
        if not allow_fallback:
            raise SentimentModelError(
                f"Could not score with {model_name!r}: {type(exc).__name__}: {exc}. "
                "Check that torch/transformers are installed and the model can be "
                "downloaded, or pass allow_fallback=True for keyword scoring."
            ) from exc
        logger.warning("Transformer scoring failed (%s); using keyword fallback", exc)
        return _score_rule_based(post_list)


# ---------------------------------------------------------------------------
# Transformer scoring
# ---------------------------------------------------------------------------
def _resolve_device(torch: Any, requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _label_indices(id2label: dict[Any, str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for key, name in id2label.items():
        lowered = str(name).strip().lower()
        for canonical in ("positive", "negative", "neutral"):
            if lowered.startswith(canonical[:3]):
                indices[canonical] = int(key)
    missing = {"positive", "negative", "neutral"} - set(indices)
    if missing:
        raise SentimentModelError(
            f"model labels {dict(id2label)!r} do not include {sorted(missing)}; "
            "a positive/negative/neutral classifier is required"
        )
    return indices


def _score_with_transformers(
    posts: list[Any],
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[SentimentResult]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs = {} if revision in {"", "main"} else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    target = _resolve_device(torch, device)
    model.to(target)
    model.eval()
    indices = _label_indices(model.config.id2label)

    results: list[SentimentResult] = []
    for start in range(0, len(posts), batch_size):
        batch = posts[start : start + batch_size]
        texts = [str(getattr(p, "text", "") or "") for p in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(target)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for post, text, row in zip(batch, texts, probs, strict=True):
            pos = float(row[indices["positive"]])
            neg = float(row[indices["negative"]])
            neu = float(row[indices["neutral"]])
            label = max(
                (("positive", pos), ("negative", neg), ("neutral", neu)),
                key=lambda item: item[1],
            )[0]
            results.append(
                SentimentResult(
                    post_id=str(getattr(post, "post_id", "")),
                    text=text[:500],
                    positive=round(pos, 6),
                    negative=round(neg, 6),
                    neutral=round(neu, 6),
                    label=label,
                    signed_score=round(pos - neg, 6),
                    model_name=model_name,
                    model_revision=revision,
                    scored_at=datetime.now(UTC),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Explicit, opt-in keyword fallback
# ---------------------------------------------------------------------------
_POSITIVE = {"gain", "gains", "rise", "rises", "bull", "bullish", "growth", "profit", "buy", "beat"}
_NEGATIVE = {"drop", "drops", "fall", "falls", "bear", "bearish", "loss", "decline", "sell", "miss"}


def _score_rule_based(posts: list[Any]) -> list[SentimentResult]:
    results: list[SentimentResult] = []
    for post in posts:
        text = str(getattr(post, "text", "") or "")
        words = set(re.findall(r"[a-z]+", text.lower()))
        pos_n = len(words & _POSITIVE)
        neg_n = len(words & _NEGATIVE)
        total = pos_n + neg_n
        if total == 0:
            pos, neg, neu, label = 0.0, 0.0, 1.0, "neutral"
        else:
            pos, neg, neu = pos_n / total, neg_n / total, 0.0
            label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        results.append(
            SentimentResult(
                post_id=str(getattr(post, "post_id", "")),
                text=text[:500],
                positive=pos,
                negative=neg,
                neutral=neu,
                label=label,
                signed_score=pos - neg,
                model_name=FALLBACK_MODEL_NAME,
                model_revision="n/a",
                scored_at=datetime.now(UTC),
            )
        )
    return results


__all__ = [
    "FALLBACK_MODEL_NAME",
    "SentimentModelError",
    "SentimentResult",
    "score_all_posts",
]
