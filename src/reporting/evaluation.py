"""Evaluation report (Module 13).

Renders the *real* walk-forward results produced by ``src.models.train.run_folds``:
per-horizon error and directional accuracy for the model and each baseline, the
baseline comparison verdict, and fold/sample counts.  No figure in this report is
simulated: when results are unavailable the report says so.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings


def _fmt(value: Any, spec: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(number) else format(number, spec)


def generate_evaluation_report(
    settings: Settings,
    *,
    folds: list[Any] | None = None,
    dataset: Any = None,
    prediction_snapshot: Any = None,
    summary: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    fold_metadata: list[dict[str, Any]] | None = None,
    strategy_metrics: dict[str, Any] | None = None,
    min_samples_for_claim: int = 30,
    output_path: Path | str | None = None,
) -> Path:
    """Write ``evaluation.md`` and return its path."""
    lines: list[str] = [
        "# X Sentiment Price Evaluation Report",
        f"**Asset:** {settings.asset_symbol}",
        f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if dataset is not None:
        lines.append(f"**Dataset version:** {dataset.version}")
    lines.append("")

    has_results = summary is not None and not summary.empty

    lines.append("## Walk-Forward Design")
    n_folds = len(fold_metadata) if fold_metadata else (len(folds) if folds else 0)
    lines.append(f"- Expanding training window; {n_folds} fold(s)")
    lines.append(
        f"- Initial train {settings.walk_forward_initial_train_days}d, test block "
        f"{settings.walk_forward_test_days}d, embargo {settings.walk_forward_gap_days}d "
        f"(largest horizon {settings.max_horizon})"
    )
    lines.append("- Training labels that mature on/after a fold's first test date are purged")
    if fold_metadata:
        lines += ["", "| Fold | Train window | Test window | Train rows | Test rows |", "|---|---|---|---|---|"]
        for meta in fold_metadata:
            train, test = meta["train"], meta["test"]
            lines.append(
                f"| {meta['fold_id']} | {train['first_bar_date']} → {train['last_bar_date']} "
                f"| {test['first_bar_date']} → {test['last_bar_date']} "
                f"| {train['n_rows']} | {test['n_rows']} |"
            )
    lines.append("")

    if not has_results:
        lines += [
            "## Results",
            "- No walk-forward results are available (too little history for even one fold, "
            "or evaluation did not run).",
            "- No performance claim is made. Collect more history and re-run.",
            "",
        ]
    else:
        assert summary is not None
        lines += [
            "## Per-Horizon Errors (out-of-sample)",
            "| Model | Horizon | Samples | MAE (return) | RMSE (return) | MAE (close) | Directional acc. |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.model} | n+{row.horizon} | {row.n_samples} | {_fmt(row.mae_return, '.5f')} "
                f"| {_fmt(row.rmse_return, '.5f')} | {_fmt(row.mae_close, '.3f')} "
                f"| {_fmt(row.directional_accuracy, '.1%')} |"
            )
        lines.append("")

        if comparison is not None and not comparison.empty:
            lines += [
                "## Comparison With the Persistence Baseline",
                "| Model | Horizon | Samples | MAE improvement | RMSE improvement | Verdict |",
                "|---|---|---|---|---|---|",
            ]
            for row in comparison.itertuples(index=False):
                lines.append(
                    f"| {row.model} | n+{row.horizon} | {row.n_samples} "
                    f"| {_fmt(row.mae_return_improvement, '+.1%')} "
                    f"| {_fmt(row.rmse_return_improvement, '+.1%')} | {row.verdict} |"
                )
            lines += [
                "",
                f"`better` requires lower MAE and RMSE than persistence on at least "
                f"{min_samples_for_claim} out-of-sample rows; otherwise the verdict is "
                "`insufficient_evidence` or `mixed`.",
                "",
            ]

    # ---- Strategy risk metrics -------------------------------------------
    strategy_metrics = strategy_metrics or {}
    if strategy_metrics:
        lines += [
            "## Strategy Risk Metrics (out-of-sample)",
            "| Model | Horizon | Sharpe | Sortino | Max Drawdown | Ann. Return | Ann. Volatility | Total Trades |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key in sorted(strategy_metrics):
            m = strategy_metrics[key]
            # key format: "model_n+h"
            parts = key.rsplit("_n+", 1)
            model_name = parts[0]
            horizon = parts[1] if len(parts) > 1 else "?"
            dd = m.get("max_drawdown", {})
            dd_val = dd.get("max_drawdown", 0.0) if isinstance(dd, dict) else 0.0
            lines.append(
                f"| {model_name} | n+{horizon} | {_fmt(m.get('sharpe'), '.3f')} "
                f"| {_fmt(m.get('sortino'), '.3f')} | {_fmt(dd_val, '.2%')} "
                f"| {_fmt(m.get('annualized_return'), '+.2%')} "
                f"| {_fmt(m.get('annualized_volatility'), '.2%')} | {m.get('total_trades', '-')} |"
            )
        lines += [""]

    lines += [
        "## Caveats",
        "- All metrics are out-of-sample walk-forward results; nothing is tuned on the test blocks.",
        "- Small samples are noisy: do not read a lower error on few rows as skill.",
        "- News is context only and is not a model feature.",
        "",
    ]

    target = Path(output_path) if output_path else Path(settings.reports_dir) / "evaluation.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
