"""Leakage checks and audit report for Module 8.

The checks in this module operate on the *assembled* modelling frame and on the
declared feature specifications.  They never modify data; they only report.

Rules enforced (Module 8, points 5-7 and the acceptance criteria):

* no duplicate ``(asset_symbol, bar_date)`` rows and strictly increasing dates;
* every target date is strictly after the feature cutoff (``bar_date`` is the
  ``n+0`` close; targets are ``n+1`` ... ``n+k``);
* every feature is built only from data available at the cutoff (declared
  source offsets are never negative; sentiment honours its minimum lag);
* the feature matrix equals the explicit allowlist and contains no target-like
  columns;
* no news columns anywhere in the modelling frame (news is report context only).

``build_leakage_audit`` assembles these checks plus feature timestamps, target
timestamps, null rates, and dropped-row accounting into one JSON-serialisable
report.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

AUDIT_VERSION = 1
KEY_COLUMNS: tuple[str, ...] = ("asset_symbol", "bar_date")

# Column-name conventions shared with ``src.features.dataset``.
CLOSE_N0 = "close_n0"
TARGET_RETURN_FMT = "target_return_n{h}"
TARGET_CLOSE_FMT = "target_close_n{h}"
TARGET_DATE_FMT = "target_date_n{h}"

PRICE_FAMILY = "price"
SENTIMENT_FAMILY = "sentiment"

_NEWS_TOKENS = frozenset(
    {"news", "headline", "headlines", "article", "articles", "tavily", "exa", "publisher"}
)
# Names that look like targets or forward-dated values must never be features.
_TARGET_LIKE = re.compile(r"^(target_|future_|fwd_|forward_)|(^|_)n\d+$")


class LeakageError(ValueError):
    """Raised when a leakage rule is violated.  ``violations`` lists every problem."""

    def __init__(self, violations: Iterable[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "Leakage audit failed:\n" + "\n".join(f"  - {item}" for item in self.violations)
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Declared timing of one feature column.

    ``source_offset`` is the age of the *newest* input value used by the feature,
    measured back from the row's ``bar_date`` (the ``n+0`` close).  ``0`` means
    the feature uses data up to and including the cutoff bar; a negative value
    would mean future data and is always a violation.
    """

    name: str
    family: str
    source_offset: int
    offset_unit: str  # "trading_rows" or "calendar_days"
    description: str = ""


# ---------------------------------------------------------------------------
# Name-based helpers
# ---------------------------------------------------------------------------
def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(name).lower()) if token}


def find_news_columns(columns: Iterable[str]) -> list[str]:
    """Return column names that look like Module 5 news content."""
    return sorted(str(c) for c in columns if _tokens(c) & _NEWS_TOKENS)


def is_target_like(name: str) -> bool:
    """True for names that look like targets or forward-dated values."""
    return bool(_TARGET_LIKE.search(str(name).lower()))


# ---------------------------------------------------------------------------
# Individual checks (each returns a list of violation strings)
# ---------------------------------------------------------------------------
def check_unique_keys(
    frame: pd.DataFrame,
    keys: Sequence[str] = KEY_COLUMNS,
    *,
    label: str = "frame",
) -> list[str]:
    """No duplicate ``(asset_symbol, bar_date)`` rows."""
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        return [f"{label}: missing key column(s) {missing}"]
    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if not duplicated.any():
        return []
    counts = frame.loc[duplicated].groupby(list(keys), sort=True).size()
    violations = [
        f"{label}: duplicate key {tuple(_plain(part) for part in _as_tuple(index))} "
        f"appears {int(count)} times"
        for index, count in list(counts.items())[:5]
    ]
    if len(counts) > 5:
        violations.append(f"{label}: {len(counts) - 5} more duplicated key(s)")
    return violations


def check_monotonic_dates(frame: pd.DataFrame, *, label: str = "frame") -> list[str]:
    """Dates must be strictly increasing within each asset."""
    if frame.empty or not set(KEY_COLUMNS) <= set(frame.columns):
        return []
    violations = []
    for asset, group in frame.groupby("asset_symbol", sort=True):
        dates = group["bar_date"]
        if not dates.is_monotonic_increasing:
            violations.append(f"{label}: dates for {asset} are not sorted ascending")
    return violations


def check_feature_allowlist(
    feature_columns: Iterable[str],
    allowlist: Iterable[str],
) -> list[str]:
    """The feature matrix must equal the allowlist exactly."""
    present, allowed = set(feature_columns), set(allowlist)
    violations = []
    unexpected = sorted(present - allowed)
    missing = sorted(allowed - present)
    if unexpected:
        violations.append(f"unexpected feature column(s) not in allowlist: {unexpected}")
    if missing:
        violations.append(f"allowlisted feature column(s) not produced: {missing}")
    return violations


def check_no_news_columns(columns: Iterable[str], *, label: str = "modelling frame") -> list[str]:
    """No news-derived columns (Module 5 output is report context only)."""
    found = find_news_columns(columns)
    return [f"{label}: news column(s) present: {found}"] if found else []


def check_features_exclude_targets(
    feature_columns: Iterable[str],
    target_columns: Iterable[str],
    metadata_columns: Iterable[str] = (),
) -> list[str]:
    """Raw target/metadata columns must stay out of the feature matrix."""
    features = list(feature_columns)
    violations = []
    overlap = sorted(set(features) & (set(target_columns) | set(metadata_columns)))
    if overlap:
        violations.append(f"target/metadata column(s) used as features: {overlap}")
    target_like = sorted(name for name in features if is_target_like(name))
    if target_like:
        violations.append(f"feature column(s) with target-like names: {target_like}")
    return violations


def check_feature_timing(
    specs: Iterable[FeatureSpec],
    *,
    min_sentiment_offset: int = 1,
) -> list[str]:
    """Every feature must be available at the cutoff.

    Offsets are never negative, and sentiment features respect the configured
    minimum lag (default one calendar day: same-day sentiment is only allowed
    when the caller has asserted Module 7's bucket closes before the cutoff).
    """
    violations = []
    for spec in specs:
        if spec.source_offset < 0:
            violations.append(
                f"{spec.name}: uses data {-spec.source_offset} {spec.offset_unit} after the cutoff"
            )
        elif spec.family == SENTIMENT_FAMILY and spec.source_offset < min_sentiment_offset:
            violations.append(
                f"{spec.name}: sentiment offset {spec.source_offset} is below the minimum "
                f"{min_sentiment_offset}"
            )
    return violations


def check_targets_after_cutoff(frame: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    """Target dates are strictly after the feature cutoff and increase with horizon."""
    violations = []
    previous: str | None = None
    for horizon in horizons:
        column = TARGET_DATE_FMT.format(h=horizon)
        if column not in frame.columns:
            violations.append(f"missing target date column {column}")
            continue
        valid = frame[column].notna()
        not_after = valid & (frame[column] <= frame["bar_date"])
        if not_after.any():
            count = int(not_after.sum())
            violations.append(
                f"{column}: {count} row(s) have a target date on or before the cutoff"
            )
        if previous is not None:
            both = valid & frame[previous].notna()
            not_increasing = both & (frame[column] <= frame[previous])
            if not_increasing.any():
                violations.append(
                    f"{column}: {int(not_increasing.sum())} row(s) are not later than {previous}"
                )
        previous = column

        for template in (TARGET_RETURN_FMT, TARGET_CLOSE_FMT):
            value_col = template.format(h=horizon)
            if value_col in frame.columns:
                orphan = frame[value_col].notna() & ~valid
                if orphan.any():
                    count = int(orphan.sum())
                    violations.append(f"{value_col}: {count} value(s) lack a target date")
    return violations


def check_target_values(frame: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    """``target_return`` must equal ``target_close / close_n0 - 1``."""
    if CLOSE_N0 not in frame.columns:
        return [f"missing reference close column {CLOSE_N0}"]
    violations = []
    for horizon in horizons:
        ret_col = TARGET_RETURN_FMT.format(h=horizon)
        close_col = TARGET_CLOSE_FMT.format(h=horizon)
        if ret_col not in frame.columns or close_col not in frame.columns:
            continue
        both = frame[ret_col].notna() & frame[close_col].notna()
        expected = frame.loc[both, close_col] / frame.loc[both, CLOSE_N0] - 1.0
        if not np.allclose(frame.loc[both, ret_col], expected, rtol=1e-9, atol=1e-12):
            violations.append(f"{ret_col} is inconsistent with {close_col} and {CLOSE_N0}")
        if (frame[ret_col].notna() != frame[close_col].notna()).any():
            violations.append(f"{ret_col} and {close_col} have different null patterns")
    return violations


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def null_rates(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    """Fraction of null values per column (empty frames report 0.0)."""
    rates: dict[str, float] = {}
    for column in columns:
        rates[column] = round(float(frame[column].isna().mean()), 6) if len(frame) else 0.0
    return rates


def build_leakage_audit(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    metadata_columns: Sequence[str],
    horizons: Sequence[int],
    specs: Sequence[FeatureSpec],
    allowlist: Sequence[str],
    min_sentiment_offset: int = 1,
    extra_checks: Mapping[str, Sequence[str]] | None = None,
    row_accounting: Mapping[str, Any] | None = None,
    alignment: Mapping[str, Any] | None = None,
    rejected_input_columns: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    """Run every check and assemble the audit report (JSON-serialisable)."""
    checks: dict[str, list[str]] = {
        "no_duplicate_asset_date": check_unique_keys(frame, label="modelling frame")
        + check_monotonic_dates(frame, label="modelling frame"),
        "feature_allowlist": check_feature_allowlist(feature_columns, allowlist),
        "no_news_columns": check_no_news_columns(frame.columns),
        "features_exclude_targets": check_features_exclude_targets(
            feature_columns, target_columns, metadata_columns
        ),
        "feature_timing": check_feature_timing(specs, min_sentiment_offset=min_sentiment_offset),
        "targets_strictly_after_cutoff": check_targets_after_cutoff(frame, horizons),
        "target_values_consistent": check_target_values(frame, horizons),
    }
    for name, violations in (extra_checks or {}).items():
        checks[name] = list(violations)

    check_report = {
        name: {"passed": not violations, "violations": list(violations)}
        for name, violations in checks.items()
    }

    spec_by_name = {spec.name: spec for spec in specs}
    families: dict[str, dict[str, Any]] = {}
    for family in dict.fromkeys(spec.family for spec in specs):
        names: list[str] = [
            c for c in feature_columns if c in spec_by_name and spec_by_name[c].family == family
        ]
        families[family] = {
            "feature_count": len(names),
            "cutoff_range": _date_range(frame["bar_date"]) if len(frame) else None,
            "features": {
                name: {
                    "newest_source_offset": spec_by_name[name].source_offset,
                    "offset_unit": spec_by_name[name].offset_unit,
                    "null_rate": null_rates(frame, [name])[name],
                }
                for name in names
            },
        }

    target_timestamps: dict[str, Any] = {}
    for horizon in horizons:
        date_col = TARGET_DATE_FMT.format(h=horizon)
        ret_col = TARGET_RETURN_FMT.format(h=horizon)
        if date_col not in frame.columns:
            continue
        valid = frame[date_col].notna()
        spans = (frame.loc[valid, date_col] - frame.loc[valid, "bar_date"]).dt.days
        target_timestamps[f"n{horizon}"] = {
            "non_null": int(valid.sum()),
            "null_rate": null_rates(frame, [ret_col])[ret_col] if ret_col in frame else None,
            "target_date_range": _date_range(frame.loc[valid, date_col]) if valid.any() else None,
            "calendar_span_days": (
                [int(spans.min()), int(spans.max())] if valid.any() else None
            ),
        }

    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "dataset_version": dataset_version,
        "passed": all(item["passed"] for item in check_report.values()),
        "checks": check_report,
        "rows": {"output_rows": int(len(frame)), **dict(row_accounting or {})},
        "feature_timestamps": families,
        "target_timestamps": target_timestamps,
        "null_rates": {
            "features": null_rates(frame, feature_columns),
            "targets": null_rates(frame, target_columns),
        },
        "alignment": dict(alignment or {}),
        "rejected_input_columns": sorted(rejected_input_columns),
        "parameters": dict(parameters or {}),
    }
    return to_jsonable(report)


def assert_audit_passed(report: Mapping[str, Any]) -> None:
    """Raise :class:`LeakageError` listing every violation if the audit failed."""
    if report.get("passed"):
        return
    violations = [
        f"[{name}] {message}"
        for name, item in report.get("checks", {}).items()
        for message in item.get("violations", [])
    ]
    raise LeakageError(violations or ["audit did not pass"])


def audit_to_json(report: Mapping[str, Any]) -> str:
    """Serialise an audit report deterministically."""
    return json.dumps(to_jsonable(report), indent=2, sort_keys=True, allow_nan=False)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _plain(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _date_range(series: pd.Series) -> list[str] | None:
    series = series.dropna()
    if series.empty:
        return None
    min_date = pd.Timestamp(series.min()).date().isoformat()
    max_date = pd.Timestamp(series.max()).date().isoformat()
    return [min_date, max_date]


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars so ``json.dumps`` works (NaN -> None)."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [to_jsonable(item) for item in items]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value is not pd.NaT else None
    if value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


__all__ = [
    "AUDIT_VERSION",
    "CLOSE_N0",
    "KEY_COLUMNS",
    "PRICE_FAMILY",
    "SENTIMENT_FAMILY",
    "TARGET_CLOSE_FMT",
    "TARGET_DATE_FMT",
    "TARGET_RETURN_FMT",
    "FeatureSpec",
    "LeakageError",
    "assert_audit_passed",
    "audit_to_json",
    "build_leakage_audit",
    "check_feature_allowlist",
    "check_feature_timing",
    "check_features_exclude_targets",
    "check_monotonic_dates",
    "check_no_news_columns",
    "check_target_values",
    "check_targets_after_cutoff",
    "check_unique_keys",
    "find_news_columns",
    "is_target_like",
    "null_rates",
    "to_jsonable",
]
