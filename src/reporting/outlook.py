"""One-page outlook report (Module 13)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings
from src.prediction.contracts import PredictionSnapshot


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _warnings(
    comparison: pd.DataFrame | None,
    sentiment_coverage: dict[str, Any] | None,
) -> list[str]:
    """Honest banners shown above the forecast table."""
    notes: list[str] = []

    if comparison is None or comparison.empty:
        notes.append(
            "**No out-of-sample evaluation is available for this model.** The forecasts below "
            "are unvalidated."
        )
    else:
        model_rows = comparison[comparison["model"] == "xgboost"]
        if model_rows.empty:
            model_rows = comparison[comparison["model"] != "drift"]
        verdicts = {int(r.horizon): str(r.verdict) for r in model_rows.itertuples(index=False)}
        better = sorted(h for h, v in verdicts.items() if v == "better")
        detail = ", ".join(f"n+{h}: {v}" for h, v in sorted(verdicts.items()))
        if not better:
            notes.append(
                "**The model has NOT shown that it beats the 'price stays the same' baseline "
                f"out-of-sample** ({detail}). Treat these forecasts as experimental, not as a "
                "signal."
            )
        elif len(better) < len(verdicts):
            notes.append(
                "The model beat the persistence baseline only at "
                + ", ".join(f"n+{h}" for h in better)
                + f" ({detail}). Treat the other horizons as unvalidated."
            )

    if sentiment_coverage is None or not sentiment_coverage.get("latest_row"):
        share = (sentiment_coverage or {}).get("share_rows", 0.0)
        notes.append(
            "**X sentiment is not informing this forecast.** No sentiment data falls in the "
            f"lookback window of the latest bar, and only {share:.0%} of training rows carry any "
            "sentiment. The forecast is effectively price-only."
        )
    elif sentiment_coverage.get("share_rows", 0.0) < 0.5:
        notes.append(
            f"Sentiment is sparse: only {sentiment_coverage['share_rows']:.0%} of training rows "
            "carry sentiment information, so its influence on the model is weak."
        )
    return notes


def generate_outlook_report(
    settings: Settings,
    prediction_snapshot: PredictionSnapshot,
    *,
    asset_symbol: str | None = None,
    sentiment_summary: dict[str, Any] | None = None,
    news_items: list[Any] | None = None,
    evaluation_comparison: pd.DataFrame | None = None,
    sentiment_coverage: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Write ``outlook.md`` and return its path.

    ``sentiment_summary`` (latest daily row) and ``news_items`` are contextual
    evidence only; neither is a model input here.
    """
    asset = asset_symbol or settings.asset_symbol
    snap = prediction_snapshot

    lines: list[str] = [
        "# X Sentiment Price Outlook",
        f"**Asset:** {asset}",
        f"**Data cutoff:** {snap.cutoff_timestamp or 'N/A'}",
        "",
    ]

    notes = _warnings(evaluation_comparison, sentiment_coverage)
    if notes:
        lines.append("## Read this first")
        lines += [f"> {note}" for note in notes]
        lines.append("")

    lines += [
        "## Observed n+0 (latest completed close)",
        f"- Close: {snap.n0_observed:,.2f}",
        "",
        "## Forecasts n+1 … n+5",
        "| Horizon | Predicted return | Predicted close |",
        "|---|---|---|",
    ]
    for forecast in snap.forecasts:
        if _finite(forecast.return_prediction) and _finite(forecast.close_prediction):
            lines.append(
                f"| n+{forecast.horizon} | {forecast.return_prediction:+.4%} "
                f"| {forecast.close_prediction:,.2f} |"
            )
        else:
            lines.append(f"| n+{forecast.horizon} | — (no model trained) | — |")
    lines += ["", "## Input freshness", f"- {snap.input_freshness}", f"- Model run: {snap.model_run_id}", ""]

    lines.append("## X sentiment context")
    if sentiment_summary:
        lines.append(
            f"- Latest bucket {sentiment_summary['bar_date']}: {sentiment_summary['post_count']} post(s), "
            f"mean signed sentiment {sentiment_summary['sentiment_mean']:+.3f}, "
            f"{sentiment_summary['positive_share']:.0%} positive / "
            f"{sentiment_summary['negative_share']:.0%} negative"
        )
    else:
        lines.append("- No scored X posts available.")
    lines.append("")

    lines.append("## Recent news (context only, not a model feature)")
    if news_items:
        for item in news_items[:5]:
            published = item.published_at.date().isoformat() if item.published_at else "n/a"
            lines.append(
                f"- [{item.title}]({item.url}) — {item.domain or item.source}, {published} "
                f"({item.relevance_flag})"
            )
    else:
        lines.append("- No news retrieved.")
    lines += [
        "",
        "## Caveats",
        "- Forecasts are model estimates, not guaranteed outcomes.",
        "- See evaluation.md for out-of-sample accuracy versus the persistence baseline.",
        "- Forecast dates are calendar offsets from the cutoff, not exchange-calendar dates.",
        "",
    ]

    target = Path(output_path) if output_path else Path(settings.reports_dir) / "outlook.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
