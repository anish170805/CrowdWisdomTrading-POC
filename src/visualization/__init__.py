"""Charts for the prediction snapshot (Module 12)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def generate_charts(
    *,
    prediction_snapshot: Any,
    asset_symbol: str,
    output_dir: str | Path,
    recent_closes: Any = None,
) -> list[Path]:
    """Write a forecast chart PNG and return the created paths.

    ``recent_closes`` is an optional ``pandas.Series`` of recent closes indexed
    by date, drawn as context before the forecast.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    horizons = [0]
    closes = [float(prediction_snapshot.n0_observed)]
    for forecast in prediction_snapshot.forecasts:
        if math.isfinite(float(forecast.close_prediction)):
            horizons.append(int(forecast.horizon))
            closes.append(float(forecast.close_prediction))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if recent_closes is not None and len(recent_closes) > 0:
        tail = recent_closes.tail(10)
        offsets = list(range(-len(tail) + 1, 1))
        ax.plot(offsets, tail.to_numpy(dtype="float64"), marker="o", color="#555555", label="Recent close")
    ax.plot(horizons, closes, marker="o", color="#1f77b4", linestyle="--", label="Model forecast")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Trading days relative to n+0")
    ax.set_ylabel("Close")
    ax.set_title(f"{asset_symbol}: observed close and n+1…n+5 forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    path = out / f"{asset_symbol}_forecast.png".replace("/", "_")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


__all__ = ["generate_charts"]
