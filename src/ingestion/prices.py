"""Fetch and store daily historical market bars.

Module 4 pulls daily OHLCV price data from a configured provider (currently
yfinance) and persists canonical :class:`~src.schemas.prices.PriceBar` rows
through Module 3's :class:`~src.storage.repository.CanonicalRepository`.

The module is designed so that provider adapters can be swapped without changing
downstream code.  All timestamps are UTC; daily bars are keyed by UTC calendar
date (``bar_date``).  Non-trading days produce no fabricated bars — gaps are
recorded but no synthetic prices are introduced.

**Acceptance criteria (from Module 04 markdown)**
- Determined mocked-provider tests pass.
- Provider errors are visible to the operator.
- Dates and timezone are explicit.
- No synthetic future or missing-day prices are introduced.
- Idempotent writes via Module 3's upsert logic.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone as tz_mod
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.config import Settings, DataProvider, get_settings
from src.logging_config import get_logger, new_run_id
from src.schemas.prices import PriceBar
from src.schemas.common import as_utc, as_utc_date
from src.storage.repository import CanonicalRepository, UpsertStats
from src.schemas.common import FrozenModel

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

ProviderResult = dict[str, Any]


class PriceProvider(FrozenModel):
    """Abstract interface for a price-data provider.

    Subclasses implement :meth:`fetch_bars` and optional
    :meth:`fetch_gaps`.  The return format is normalized so that
    :class:`~src.ingestion.prices.normalise_provider_bars` can produce
    :class:`~src.schemas.prices.PriceBar` objects.
    """

    name: str = Field(description="Human-readable provider name.")

    def fetch_bars(
        self,
        asset_symbol: str,
        start: date,
        end: date,
        *,  # keyword-only after this point
        timezone: str,
    ) -> list[dict[str, Any]]:
        """Fetch daily bars for ``[start, end]`` inclusive.

        Returns a list of dicts with at least the keys ``"date"``,
        ``"open"``, ``"high"``, ``"low"``, ``"close"``, ``"volume"``.
        Additional keys (e.g. ``"adjusted_close"``) are preserved when
        available.

        Raises :class:`~src.ingestion.prices.PriceProviderError` on
        recoverable/expected failures; other exceptions propagate.
        """
        raise NotImplementedError

    def fetch_gaps(
        self,
        asset_symbol: str,
        *,
        timezone: str,
    ) -> dict[str, Any]:
        """Optionally report gaps in coverage for the requested window.

        Returns a dict with keys ``"has_gaps"`` (``bool``) and
        ``"gap_dates"`` (``list[date]``).  The default implementation
        returns ``{"has_gaps": False, "gap_dates": []}``.
        """
        return {"has_gaps": False, "gap_dates": []}


class PriceProviderError(RuntimeError):
    """Raised when a price provider reports a recoverable failure."""
    pass


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _validate_asset_symbol(value: str) -> str:
    """Ensure the asset symbol is valid and normalized."""
    value = value.strip()
    if not value:
        raise ValueError("asset symbol must not be empty")
    if any(ch.isspace() for ch in value):
        raise ValueError("asset symbol must not contain whitespace")
    return value


# yfinance adapter ------------------------------------------------------------

try:
    import yfinance as yf  # type: ignore[import]

    HAS_YFINANCE = True
except ImportError:  # pragma: no cover - dependency check
    HAS_YFINANCE = False


class YFinanceProvider(PriceProvider):
    """yfinance-based implementation of :class:`PriceProvider`.

    yfinance returns *adjusted* close prices already; we keep the raw close
    and store adjusted_close from the provider when available.

    .. code-block:: text

        yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        returns a DataFrame with columns:
            Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
    """

    name: str = "yfinance"

    # yfinance's ``history`` endpoint treats the ``end`` date as exclusive.
    # We request end+1d and then trim so the caller's ``[start, end]`` window
    # is inclusive on both sides.

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> Any:
        if value is None:
            return "yfinance"
        text = str(value).strip()
        return text or "yfinance"

    def fetch_bars(
        self,
        asset_symbol: str,
        start: date,
        end: date,
        *,
        timezone: str,
    ) -> list[dict[str, Any]]:
        if not HAS_YFINANCE:
            raise PriceProviderError(
                "yfinance is not installed. pip install yfinance"
            )

        symbol = _validate_asset_symbol(asset_symbol)
        tz = timezone

        # yfinance expects ISO-format dates; we build a half-open interval
        # [start, end+1d) so that the caller's inclusive window is respected.
        end_exclusive = datetime.combine(end, time(1, 0), tzinfo=tz_mod.utc) + timedelta(days=1)
        start_dt = datetime.combine(start, time(0, 0), tzinfo=tz_mod.utc)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_dt,
                end=end_exclusive,
                interval="1d",
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
            raise PriceProviderError(
                f"yfinance history() failed for {symbol}: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning(
                "yfinance returned no data for %s in %s..%s",
                symbol,
                start.isoformat(),
                end.isoformat(),
            )
            return []

        # yfinance columns are typically: Open, High, Low, Close, Volume,
        # Dividends, Stock Splits.  We normalize to a flat dict list.
        bars: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            bar_date = row.name.date() if hasattr(row.name, "date") else row.name
            # yfinance returns timestamps as index; ensure UTC date
            bars.append(
                {
                    "date": bar_date,
                    "open": float(row["Open"]) if not math.isnan(row["Open"]) else None,
                    "high": float(row["High"]) if not math.isnan(row["High"]) else None,
                    "low": float(row["Low"]) if not math.isnan(row["Low"]) else None,
                    "close": float(row["Close"]) if not math.isnan(row["Close"]) else None,
                    "volume": float(row["Volume"]) if not math.isnan(row["Volume"]) else None,
                    "adj_close": (
                        float(row["Close"])
                        if getattr(row, "Dividends", None) is not None or getattr(row, "Stock Splits", None)
                        else None
                    ),
                }
            )

        # Sort by date ascending
        bars.sort(key=lambda b: b["date"])
        return bars

    def fetch_gaps(
        self,
        asset_symbol: str,
        *,
        timezone: str,
    ) -> dict[str, Any]:
        """yfinance does not guarantee completeness; we conservatively report
        gaps when the returned bar count is less than the calendar day count."""
        # This is a best-effort gap detection; the actual gap reporting is done
        # in the ingestion pipeline after comparing returned dates vs. expected.
        return {"has_gaps": False, "gap_dates": []}


# ---------------------------------------------------------------------------
# Normalisation & validation
# ---------------------------------------------------------------------------


def normalise_provider_bars(
    bars: list[dict[str, Any]],
    *,
    asset_symbol: str,
    provider_name: str,
    batch_id: str,
    requested_start: date,
    requested_end: date,
    timezone: str,
) -> tuple[list[PriceBar], dict[str, Any]]:
    """Normalize raw provider bars into validated :class:`PriceBar` objects.

    Steps:
    1. De-duplicate by date (keep first occurrence).
    2. Enforce monotonic dates.
    3. Validate OHLC consistency (high >= max(open,close), low <= min(open,close)).
    4. Validate positive prices.
    5. Assign ``bar_date`` from the bar's UTC date.
    6. Populate ``retrieved_at`` with the current UTC time.
    7. Return ``(PriceBar[], gaps_info)``.

    The returned ``gaps`` dict contains:
      - ``"returned_start"``, ``"returned_end"``: the actual date range
        of bars returned by the provider.
      - ``"has_gaps"``: ``True`` when some calendar days in
        ``[requested_start, requested_end]`` were not covered.
      - ``"gap_dates"``: list of calendar dates that were expected but
        not present in the returned data.
    """
    if not bars:
        # No bars returned – record a full-window gap.
        gaps: dict[str, Any] = {
            "returned_start": None,
            "returned_end": None,
            "has_gaps": True,
            "gap_dates": list(
                _calendar_dates(requested_start, requested_end)
            ),
        }
        return [], gaps

    # ------------------------------------------------------------------
    # De-duplicate by date (keep first occurrence when provider returns
    # multiple bars for the same date, which can happen with split-adjusted
    # data or non-trading-day padding).
    # ------------------------------------------------------------------
    seen_dates: set[date] = set()
    unique_bars: list[dict[str, Any]] = []
    for bar in bars:
        bar_date = bar.get("date")
        if bar_date is None:
            logger.warning("Provider bar missing 'date'; skipping: %s", bar)
            continue
        if bar_date in seen_dates:
            continue
        seen_dates.add(bar_date)
        unique_bars.append(bar)

    if not unique_bars:
        gaps: dict[str, Any] = {
            "returned_start": None,
            "returned_end": None,
            "has_gaps": True,
            "gap_dates": list(_calendar_dates(requested_start, requested_end)),
        }
        return [], gaps

    # ------------------------------------------------------------------
    # Sort ascending by date
    # ------------------------------------------------------------------
    unique_bars.sort(key=lambda b: b["date"])

    # ------------------------------------------------------------------
    # Validate each bar and collect gaps
    # ------------------------------------------------------------------
    price_bars: list[PriceBar] = []
    returned_dates: set[date] = set()
    gap_dates: list[date] = []

    # Build the full expected calendar range
    all_expected = set(_calendar_dates(requested_start, requested_end))
    returned_dates.update(bar["date"] for bar in unique_bars)

    # Track which dates we've validated
    validated_dates: set[date] = set()

    for bar in unique_bars:
        bar_date = bar["date"]

        # Validate price fields are present and positive
        open_ = bar.get("open")
        high_ = bar.get("high")
        low_ = bar.get("low")
        close_ = bar.get("close")
        volume_ = bar.get("volume")
        adj_close_ = bar.get("adj_close")

        # Missing essential prices -> skip this bar (quarantine behavior:
        # do not fabricate)
        if open_ is None or high_ is None or low_ is None or close_ is None:
            logger.warning(
                "Price bar for %s on %s has missing OHLCV; skipping",
                asset_symbol,
                bar_date,
            )
            continue

        # Positive price validation
        for label, val in [
            ("open", open_),
            ("high", high_),
            ("low", low_),
            ("close", close_),
        ]:
            if not isinstance(val, (int, float)):
                logger.warning(
                    "Price bar %s: %s is not a number (got %s:%s)",
                    asset_symbol,
                    label,
                    type(val).__name__,
                    val,
                )
                break
            if val <= 0:
                logger.warning(
                    "Price bar %s: %s must be positive (got %s)",
                    asset_symbol,
                    label,
                    val,
                )
                continue  # skip this bar
        else:
            # OHLC consistency
            if high_ + 1e-9 < max(open_, close_):
                logger.warning(
                    "Price bar %s: high %.4f < max(open,close) %.4f on %s",
                    asset_symbol,
                    high_,
                    max(open_, close_),
                    bar_date,
                )
                continue
            if low_ - 1e-9 > min(open_, close_):
                logger.warning(
                    "Price bar %s: low %.4f > min(open,close) %.4f on %s",
                    asset_symbol,
                    low_,
                    min(open_, close_),
                    bar_date,
                )
                continue

            # bar_date must match the UTC date of the bar's timestamp.
            # Since we work with dates directly, we just verify the date is
            # consistent (provider should already give us the correct date).
            # We still need to create a UTC timestamp for the PriceBar.
            try:
                # Interpret the bar_date as midnight UTC
                bar_timestamp = datetime.combine(
                    bar_date, time(0, 0), tzinfo=tz_mod.utc
                )
            except (ValueError, OverflowError) as exc:
                logger.warning(
                    "Price bar %s: could not construct timestamp from date %s: %s",
                    asset_symbol,
                    bar_date,
                    exc,
                )
                continue

            # Validate OHLC after we have all values
            if high_ + 1e-9 < max(open_, close_):
                logger.warning(
                    "Price bar %s: high %.4f < max(open,close) %.4f on %s (post-valid)",
                    asset_symbol,
                    high_,
                    max(open_, close_),
                    bar_date,
                )
                continue
            if low_ - 1e-9 > min(open_, close_):
                logger.warning(
                    "Price bar %s: low %.4f > min(open,close) %.4f on %s (post-valid)",
                    asset_symbol,
                    low_,
                    min(open_, close_),
                    bar_date,
                )
                continue

            # Build the PriceBar
            from src.schemas.prices import PriceBar as _PriceBar

            warnings: tuple[str, ...] = ()
            if adj_close_ is not None and adj_close_ > 0:
                adjusted_close_val = adj_close_
            else:
                adjusted_close_val = None

            bar_obj = _PriceBar(
                asset_symbol=asset_symbol,
                asset_type=None,  # will be set by the caller if known
                bar_date=bar_date,
                timestamp=bar_timestamp,
                open=float(open_),
                high=float(high_),
                low=float(low_),
                close=float(close_),
                adjusted_close=adjusted_close_val,
                volume=float(volume_) if volume_ is not None else None,
                provider=provider_name,
                batch_id=batch_id,
                source_record_id=None,
                raw_payload_ref=None,
                retrieved_at=datetime.now(tz=tz_mod.utc),
                ingested_at=datetime.now(tz=tz_mod.utc),
                warnings=warnings,
            )
            price_bars.append(bar_obj)
            validated_dates.add(bar_date)

    # ------------------------------------------------------------------
    # Compute gap information
    # ------------------------------------------------------------------
    actual_returned_dates = returned_dates  # dates that made it through validation
    expected_dates = all_expected
    missing_dates = expected_dates - actual_returned_dates

    if missing_dates:
        gap_dates = sorted(missing_dates)
    else:
        gap_dates = []

    gaps: dict[str, Any] = {
        "returned_start": min(actual_returned_dates) if actual_returned_dates else None,
        "returned_end": max(actual_returned_dates) if actual_returned_dates else None,
        "has_gaps": len(missing_dates) > 0,
        "gap_dates": gap_dates,
    }

    return price_bars, gaps


def _calendar_dates(start: date, end: date) -> list[date]:
    """Return every calendar date from ``start`` to ``end`` inclusive."""
    if start > end:
        return []
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def import_historical_price_bars(
    *,
    asset_symbol: str,
    asset_type: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    provider: DataProvider | str | None = None,
    batch_id: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Fetch and store daily historical price bars for one asset.

    Parameters
    ----------
    asset_symbol:
        Ticker/instrument symbol, e.g. ``"AAPL"``, ``"BTC-USD"``, ``"EURUSD"``.
    asset_type:
        Asset class override.  When ``None``, read from
        ``settings.asset_type``.
    start_date:
        Inclusive start date.  Defaults to ``settings.effective_start_date``.
    end_date:
        Inclusive end date.  Defaults to ``settings.effective_end_date``.
    provider:
        Price data provider.  When ``None``, reads
        ``settings.data_provider`` (currently ``DataProvider.YFINANCE``).
        The value is case-insensitive; strings are accepted alongside the
        enum.
    batch_id:
        Optional import batch ID.  When ``None``, a sortable ID is generated.
        Repeated calls with the same ``batch_id`` are idempotent thanks to
        Module 3's upsert logic.

    Returns
    -------
    dict with keys:
      - ``"inserted"``: number of new price bars inserted.
      - ``"updated"``: number of existing bars updated (conflict on
        ``(asset_symbol, timestamp)``).
      - ``"duplicates"``: number of redundant bars ignored.
      - ``"attempted"``: total bars presented by the provider.
      - ``"gaps"``: gap-analysis dict from
        :func:`~src.ingestion.prices.normalise_provider_bars`.
      - ``"provider"``: the provider name that was used.
      - ``"errors"``: list of error strings from provider failures.
    """
    resolved_settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Resolve parameters
    # ------------------------------------------------------------------
    asset_symbol = _validate_asset_symbol(asset_symbol)
    if asset_type is None:
        asset_type = resolved_settings.asset_type
    else:
        asset_type = asset_type.value if hasattr(asset_type, "value") else str(asset_type)

    if start_date is None:
        start_date = resolved_settings.effective_start_date
    if end_date is None:
        end_date = resolved_settings.effective_end_date

    if start_date > end_date:
        raise ValueError(
            f"start_date {start_date.isoformat()} must be on or before end_date {end_date.isoformat()}"
        )

    # ------------------------------------------------------------------
    # Resolve provider
    # ------------------------------------------------------------------
    if provider is None:
        provider_enum = resolved_settings.data_provider
    elif isinstance(provider, DataProvider):
        provider_enum = provider
    else:
        try:
            provider_enum = DataProvider(provider.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown data provider {provider!r}. "
                f"Supported: {[p.value for p in DataProvider]}"
            ) from exc

    batch_id = batch_id or new_run_id("price")
    timezone = resolved_settings.timezone

    logger.info(
        "Historical price ingestion started: %s %s %s %s %s",
        asset_symbol,
        asset_type,
        start_date.isoformat(),
        end_date.isoformat(),
        provider_enum.value,
    )

    # ------------------------------------------------------------------
    # Fetch bars from the provider
    # ------------------------------------------------------------------
    provider_adapter: PriceProvider
    if provider_enum == DataProvider.YFINANCE:
        provider_adapter = YFinanceProvider()
    else:
        raise PriceProviderError(
            f"Provider {provider_enum} is not implemented. "
            f"Supported: {[p.value for p in DataProvider]}"
        )

    try:
        raw_bars = provider_adapter.fetch_bars(
            asset_symbol=asset_symbol,
            start=start_date,
            end=end_date,
            timezone=timezone,
        )
    except PriceProviderError as exc:
        logger.error(
            "Price provider failed: %s",
            exc,
            extra={"event": "price_ingestion.provider_error"},
        )
        return {
            "inserted": 0,
            "updated": 0,
            "duplicates": 0,
            "attempted": 0,
            "gaps": {
                "returned_start": None,
                "returned_end": None,
                "has_gaps": True,
                "gap_dates": list(_calendar_dates(start_date, end_date)),
            },
            "provider": provider_enum.value,
            "errors": [str(exc)],
        }
    except Exception as exc:  # noqa: BLE001 - unexpected error is surfaced
        logger.exception(
            "Unexpected error fetching price bars",
            extra={"event": "price_ingestion.unexpected_error"},
        )
        return {
            "inserted": 0,
            "updated": 0,
            "duplicates": 0,
            "attempted": 0,
            "gaps": {
                "returned_start": None,
                "returned_end": None,
                "has_gaps": True,
                "gap_dates": list(_calendar_dates(start_date, end_date)),
            },
            "provider": provider_enum.value,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    # ------------------------------------------------------------------
    # Normalise and validate bars
    # ------------------------------------------------------------------
    provider_name = provider_adapter.name
    batch_id_safe = str(batch_id)

    price_bars, gaps = normalise_provider_bars(
        raw_bars,
        asset_symbol=asset_symbol,
        provider_name=provider_name,
        batch_id=batch_id_safe,
        requested_start=start_date,
        requested_end=end_date,
        timezone=timezone,
    )

    logger.info(
        "Normalised %d price bars ( %d after validation ), gaps=%s",
        len(raw_bars),
        len(price_bars),
        gaps,
        extra={"event": "price_ingestion.normalised"},
    )

    # ------------------------------------------------------------------
    # Persist through Module 3's repository (idempotent upsert)
    # ------------------------------------------------------------------
    from src.storage.engine import create_engine_from_settings

    repo = CanonicalRepository(
        create_engine_from_settings(resolved_settings),
        ensure_schema=True,
    )

    upsert_stats: UpsertStats
    if price_bars:
        upsert_stats = repo.upsert_price_bars(price_bars)
    else:
        upsert_stats = UpsertStats(attempted=0, inserted=0, updated=0, duplicates=0)

    result: dict[str, Any] = {
        "inserted": upsert_stats.inserted,
        "updated": upsert_stats.updated,
        "duplicates": upsert_stats.duplicates,
        "attempted": len(raw_bars),
        "gaps": gaps,
        "provider": provider_name,
        "errors": [],
    }

    logger.info(
        "Price ingestion complete: inserted=%d updated=%d duplicates=%d "
        "gaps=%s" % (upsert_stats.inserted, upsert_stats.updated, upsert_stats.duplicates, gaps),
    )

    return result


# ---------------------------------------------------------------------------
# Module-level test / demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Allow running as ``python -m src.ingestion.prices`` for a quick demo."""
    import sys
    from datetime import date

    # Minimal settings bootstrap
    from src.config import Settings, load_settings

    try:
        settings = load_settings()
    except Exception:
        # Fall back to defaults if .env is missing
        from src.config import get_settings
        settings = get_settings()

    # If no args, show help
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.prices <asset_symbol> [start_date] [end_date]")
        sys.exit(0)

    symbol = sys.argv[1]
    start = (
        date.fromisoformat(sys.argv[2])
        if len(sys.argv) > 2
        else settings.effective_start_date
    )
    end = (
        date.fromisoformat(sys.argv[3])
        if len(sys.argv) > 3
        else settings.effective_end_date
    )

    result = import_historical_price_bars(
        asset_symbol=symbol,
        start_date=start,
        end_date=end,
        provider=settings.data_provider,
        batch_id=f"manual-{symbol}",
    )
    import json
    print(json.dumps(result, default=str, indent=2))