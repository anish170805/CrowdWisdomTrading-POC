"""Retrieve recent relevant news for the current sentiment outlook via Tavily
(Exa optionally, behind the same adapter interface).

Module 5 — News retrieval with Tavily (Exa optional)

**Goal:** retrieve recent relevant news for the current sentiment outlook, without
contaminating the model's allowed feature set.

**Implement**
1. Use Tavily with the configured key and asset-specific query; apply a bounded
   recent time window.
2. Retrieve title, URL, publication date, source/domain, snippet/content, and
   retrieval timestamp where available.
3. Deduplicate by canonical URL and near-identical title; retain provenance.
4. Filter irrelevant ticker/name collisions and label uncertain relevance for review.
5. Store a compact news cache and retrieval manifest; enforce timeouts and request
   limits.
6. Exa is optional. Add it only if credentials and a clear use case are supplied;
   expose it behind the same adapter interface.
7. Keep news out of the training feature matrix unless the project owner explicitly
   changes the feature restriction.

**Files:** `src/ingestion/news.py`, tests.

**Input:** asset settings from Module 0 and current date/time.

**Output:** timestamped, deduplicated news items for Module 13's written outlook.

**Connection:** uses Module 0 config and may use Module 3 storage for provenance. It
does not feed Modules 6–10. Module 13 consumes its output as contextual evidence only.

**Acceptance:** mocked provider tests; URLs and dates retained; irrelevant results
filtered; no news-derived model features; absent Tavily/Exa credentials do not break
the pipeline.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.ingestion.field_parsing import (
    as_text,
    clean_post_id,
    date_range,
    is_present,
    json_safe,
    normalise_url,
    now_utc,
    parse_timestamp,
    safe_component,
    write_json,
    write_jsonl,
)
from src.logging_config import get_logger, new_run_id

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider name constants
# ---------------------------------------------------------------------------

SOURCE_NAME = "tavily"
EXA_SOURCE_NAME = "exa"

# ---------------------------------------------------------------------------
# Configurable limits (kept bounded so the pipeline never spins unboundedly)
# ---------------------------------------------------------------------------

MAX_RESULTS_DEFAULT = 10
MAX_RESULTS_MAX = 20
MAX_RESULTS_MIN = 1
TIMEOUT_SECONDS_DEFAULT = 30
TIMEOUT_SECONDS_MAX = 120

# ---------------------------------------------------------------------------
# Utility: query builder (used by adapter and public entry point)
# ---------------------------------------------------------------------------

def _build_query(asset_symbol: str, extra_terms: tuple[str, ...] = ()) -> str:
    """Build a plain-language news query, e.g. ``Apple Inc AAPL stock news``.

    Cashtags (``$AAPL``) match quote pages, not articles, so the ticker is used
    as a normal word alongside any configured aliases.
    """
    parts: list[str] = []
    for term in extra_terms:
        t = term.strip()
        if t:
            parts.append(t)
    parts.append(asset_symbol.upper().strip())
    parts.append("stock news")
    return " ".join(parts)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """A single news item retrieved from a news provider.

    Fields retained for provenance and downstream consumption by Module 13.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(description="Headline / title of the news article.")
    url: str = Field(description="Canonical URL where the article is published.")
    source: str = Field(description="Name of the news source / publisher.")
    domain: str = Field(description="Registered domain of the source URL.")
    snippet: str | None = Field(
        default=None,
        description="Short excerpt / summary from the provider.",
    )
    content: str | None = Field(
        default=None,
        description="Fuller article text when available (may be truncated).",
    )
    published_at: datetime | None = Field(
        default=None,
        description="Publication timestamp (UTC) when available.",
    )
    retrieved_at: datetime = Field(
        description="When this item was retrieved (UTC).",
    )
    query: str = Field(description="The search query that produced this result.")
    relevance_flag: str = Field(
        description="One of: certain, uncertain, filtered. Labels relevance quality.",
    )
    relevance_reason: str | None = Field(
        default=None,
        description="Human-readable explanation when relevance is uncertain or filtered.",
    )


class NewsManifest(BaseModel):
    """Auditable provenance for one news-retrieval batch (never contains secrets)."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str
    source: str = SOURCE_NAME
    asset_symbol: str
    query: str
    collection_window: dict[str, Any]
    requested_max_items: int
    retrieved_max_items: int
    retrieved_at: datetime
    news_item_count: int
    deduplicated_item_count: int
    items: tuple[NewsItem, ...]
    excluded_ticker_collisions: int = Field(default=0)
    uncertain_relevance_count: int = Field(default=0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    output_dir: str
    manifest_path: str


# ---------------------------------------------------------------------------
# Adapter protocol — same interface for Tavily or Exa
# ---------------------------------------------------------------------------


class NewsProvider(Protocol):
    """Protocol for news providers (Tavily or Exa).

    Implementations must return a tuple of (items, warnings, error) where items
    are NewsItem objects.
    """

    def fetch(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        max_items: int,
    ) -> tuple[list[NewsItem], list[str], str | None]:
        """Fetch news items in the given window.

        Returns
        -------
        items : list[NewsItem]
            Deduplicated, relevance-filtered news items.
        warnings : list[str]
            Human-readable warnings (non-fatal).
        error : str | None
            Error message if the fetch failed; None on success.
        """
        ...


# ---------------------------------------------------------------------------
# Tavily implementation
# ---------------------------------------------------------------------------


try:
    from tavily import TavilyClient  # type: ignore[import]

    HAS_TAVILY = True
except ImportError:  # pragma: no cover - dependency check
    HAS_TAVILY = False


class TavilyProviderError(RuntimeError):
    """Raised when the Tavily API reports a recoverable failure."""

    pass


class _TavilyAdapter:
    """Tavily-backed implementation of :class:`NewsProvider`.

    Wraps the Tavily HTTP client, applies query construction, time-window filtering,
    dedup, and relevance flagging.
    """

    def __init__(self, api_key: str, asset_symbol: str, timeout_seconds: int = 10) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Tavily API key must be a non-empty string")
        self._client = TavilyClient(api_key=api_key.strip())
        self._asset_symbol = asset_symbol.strip().upper()
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_items(items: list[NewsItem]) -> tuple[list[NewsItem], int]:
        """Deduplicate by canonical URL, then by near-identical title.

        Returns (deduplicated_items, discarded_count).
        """
        seen_urls: set[str] = set()
        seen_title_fingerprints: set[str] = set()
        deduped: list[NewsItem] = []
        discarded = 0

        for item in items:
            # Normalise URL for comparison
            url_norm = normalise_url(item.url)
            if url_norm in seen_urls:
                discarded += 1
                continue
            seen_urls.add(url_norm)

            # Fingerprint title: lowercase, collapsed whitespace, stripped
            fp = re.sub(r"\s+", " ", item.title.strip().lower())
            if fp in seen_title_fingerprints:
                discarded += 1
                continue
            seen_title_fingerprints.add(fp)

            deduped.append(item)

        return deduped, discarded

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fetch(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        max_items: int,
    ) -> tuple[list[NewsItem], list[str], str | None]:
        """Fetch news from Tavily for the given query and UTC window.

        Parameters
        ----------
        query : str
            The search query string.
        start : datetime
            Inclusive start of the retrieval window (UTC).
        end : datetime
            Inclusive end of the retrieval window (UTC).
        max_items : int
            Maximum number of results to request from Tavily.

        Returns
        -------
        items : list[NewsItem]
            Deduplicated, relevance-flagged news items.
        warnings : list[str]
            Warnings generated during the fetch.
        error : str | None
            None on success; error message on failure.
        """
        warnings: list[str] = []
        error: str | None = None

        # Tavily uses a half-open interval [start, end+1d) conceptually; we pass
        # the dates as ISO strings in the request body.
        try:
            response = self._client.search(
                query=query,
                topic="news",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                max_results=max_items,
                include_raw_content=False,
            )
        except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
            error_msg = f"Tavily search failed: {type(exc).__name__}: {exc}"
            logger.error(error_msg, extra={"event": "tavily.fetch_error"})
            return [], [error_msg], error_msg

        # The Tavily response structure: {"results": [...], "api_usage": {...}}
        raw_results = response.get("results", [])
        if not raw_results:
            return [], [], None

        # Build NewsItem objects from raw results
        now = now_utc()
        items: list[NewsItem] = []

        for raw in raw_results:
            title = as_text(raw.get("title"))
            url = as_text(raw.get("url"))
            if not title or not url:
                # Skip items missing essential fields
                warnings.append("Skipping news item missing title or URL")
                continue

            # Source and domain
            domain = (urlparse(url).netloc or "").lower().removeprefix("www.")
            source = as_text(raw.get("publisher")) or as_text(raw.get("source")) or domain or "unknown"

            # Snippet / excerpt
            snippet = as_text(raw.get("content")) or as_text(raw.get("excerpt"))

            # Published date when available
            published_at: datetime | None = None
            pub_raw = raw.get("published_date") or raw.get("publishedDate") or raw.get("date")
            if pub_raw:
                try:
                    published_at = parse_timestamp(
                        pub_raw, assume_timezone=UTC, allow_excel_serial=False
                    ).utc
                except Exception:  # noqa: BLE001 - unparseable dates stay None
                    published_at = None

            # Relevance flagging: we use a simple heuristic — if the title/url
            # contains the asset symbol cashtag, we mark it "certain"; otherwise
            # "uncertain" for manual review.
            if published_at is None:
                warnings.append(f"Dropped undated item: {title[:60]}")
                continue
            haystack = f"{title} {snippet or ''}".upper()
            if self._asset_symbol and re.search(
                rf"\b{re.escape(self._asset_symbol)}\b", haystack
            ):
                relevance_flag = "certain"
                relevance_reason = None
            else:
                relevance_flag = "uncertain"
                relevance_reason = f"Neither title nor snippet mentions {self._asset_symbol}"

            item = NewsItem(
                title=title,
                url=url,
                source=source,
                domain=domain,
                snippet=snippet,
                content=None,  # Tavily does not return full content when
                # include_raw_content=False; keep None to respect the feature
                # restriction.
                published_at=published_at,
                retrieved_at=now,
                query=query,
                relevance_flag=relevance_flag,
                relevance_reason=relevance_reason,
            )
            items.append(item)

        # Deduplicate
        deduped, discarded = self._deduplicate_items(items)
        if discarded:
            warnings.append(f"Discarded {discarded} duplicate/news-item(s)")

        # If we have too few results relative to max_items, flag it
        if len(deduped) < max_items // 2 and max_items >= MAX_RESULTS_MIN:
            warnings.append(
                f"Fewer than expected results returned ({len(deduped)} of {max_items} requested)."
            )

        return deduped, warnings, None


# ---------------------------------------------------------------------------
# Exa optional adapter (behind same interface)
# ---------------------------------------------------------------------------


class _ExaAdapter:
    """Optional Exa-backed implementation of :class:`NewsProvider`.

    Only instantiated when Exa credentials are explicitly supplied.  When absent,
    the adapter raises a clear configuration error so the pipeline does not silently
    degrade.
    """

    def __init__(self, api_key: str, timeout_seconds: int = TIMEOUT_SECONDS_DEFAULT) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Exa API key must be a non-empty string")
        # Exa client import is lazy; we defer the import so the module can be
        # imported even without the exa package installed.
        from exa_py import Exa  # type: ignore[import]

        self._client = Exa(api_key=api_key.strip(), timeout=timeout_seconds)
        self.timeout_seconds = timeout_seconds

    def fetch(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        max_items: int,
    ) -> tuple[list[NewsItem], list[str], str | None]:
        # Implementation follows Tavily pattern but uses Exa's API.
        # For now, raise NotImplementedError because Exa integration is optional.
        # The adapter interface exists so the rest of the pipeline does not need
        # to change when Exa is swapped in.
        raise NotImplementedError("Exa news adapter is not yet implemented; set up Tavily-only path.")  # noqa: E501


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _secrets_strip(value: Any) -> str | None:
    """Normalise a possibly-SecretStr secret to a plain str, or None."""
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        # SecretStr
        return value.get_secret_value().strip()
    if isinstance(value, str):
        return value.strip()
    return None


def _build_provider(
    settings: Settings | None = None,
    asset_symbol: str | None = None,
) -> tuple[NewsProvider | None, str | None]:
    """Build a news provider from settings.

    Returns
    -------
    provider : NewsProvider | None
        Instantiated adapter or None if no credentials are available.
    provider_name : str | None
        Name of the provider ("tavily" or "exa") or None.
    """
    resolved = settings or get_settings()
    tavily_key = _secrets_strip(resolved.tavily_api_key)
    exa_key = _secrets_strip(resolved.exa_api_key)

    if exa_key and exa_key.strip() and not (tavily_key and tavily_key.strip()):
        # Exa is configured — but we still prefer Tavily for now; expose Exa
        # behind the same interface so the pipeline can switch.
        try:
            from exa_py import Exa  # type: ignore[import]

            provider = _ExaAdapter(api_key=exa_key)
            return provider, EXA_SOURCE_NAME
        except ImportError:
            logger.warning(
                "Exa API key configured but 'exa-py' package is not installed. "
                "Falling back to Tavily or falling through."
            )
        except Exception as exc:  # noqa: BLE001 - provider init error is surfaced
            logger.warning(
                f"Failed to initialise Exa adapter: {type(exc).__name__}: {exc}"
            )

    if tavily_key and tavily_key.strip():
        try:
            from tavily import TavilyClient  # type: ignore[import]

            provider = _TavilyAdapter(api_key=tavily_key, asset_symbol=asset_symbol or "ASSET")  # type: ignore[arg-type]
            return provider, SOURCE_NAME
        except ImportError:
            logger.error(
                "Tavily API key configured but 'tavily' package is not installed. "
                "Install with: pip install tavily"
            )
        except Exception as exc:  # noqa: BLE001 - provider init error is surfaced
            logger.warning(
                f"Failed to initialise Tavily adapter: {type(exc).__name__}: {exc}"
            )

    # No credentials available — return None so the pipeline can continue
    # without news (graceful degradation).
    return None, None


# Concrete TavilyProvider alias for the factory return type
_TavilyProvider = _TavilyAdapter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_news(
    asset_symbol: str,
    *,
    settings: Settings | None = None,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    max_items: int | None = None,
    output_dir: str | Path | None = None,
    client: NewsProvider | None = None,
    clock: Callable[[], datetime] | None = None,
) -> NewsManifest:
    """Retrieve recent relevant news for the given asset.

    This is the public entry point for Module 5.  It constructs the query from the
    asset symbol, runs the configured news provider (Tavily by default), applies
    deduplication and relevance flagging, and returns a :class:`NewsManifest` containing
    the items and auditable provenance.

    The retrieved window defaults to the last 7 days.  This bounded window prevents
    unbounded data collection and keeps the feature set controlled.

    Parameters
    ----------
    asset_symbol : str
        Ticker or asset identifier, e.g. ``"AAPL"`` or ``"BTC-USD"``.
    settings : Settings | None
        Project settings.  When ``None``, ``get_settings()`` is used.
    start_date : date | datetime | None
        Inclusive start of the retrieval window.  When ``None``, defaults to 7 days
        before ``end_date`` (or today if ``end_date`` is also ``None``).
    end_date : date | datetime | None
        Inclusive end of the retrieval window.  When ``None``, defaults to today.
    max_items : int | None
        Maximum number of raw results to request from the provider.  When ``None``,
        uses ``settings.tavily_max_items`` (capped at ``MAX_RESULTS_MAX``).
    output_dir : str | Path | None
        Directory where the manifest JSON will be written.  When ``None``, a
        temporary directory under the project cache is used.
    client : NewsProvider | None
        Pre-configured news provider.  When ``None``, one is built from settings.
    clock : Callable[[], datetime] | None
        Monotonic clock for ``retrieved_at`` timestamps.  Defaults to ``datetime.now(
        UTC)``.

    Returns
    -------
    NewsManifest
        Auditable manifest with items, provenance, and statistics.
    """
    resolved = settings or get_settings()

    # ------------------------------------------------------------------
    # Clock / now helpers
    # ------------------------------------------------------------------
    clock_fn = clock or (lambda: datetime.now(tz=UTC))
    now = now_utc(clock_fn)
    now_call = lambda: now_utc(clock_fn)  # reusable callable for elapsed time

    # ------------------------------------------------------------------
    # Resolve dates: default to last 7 days
    # ------------------------------------------------------------------
    if end_date is None:
        end_date = now.date()
    else:
        end_date = end_date if isinstance(end_date, date) else end_date.date()

    if start_date is None:
        start_date = end_date - timedelta(days=7)
    else:
        start_date = start_date if isinstance(start_date, date) else start_date.date()

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=UTC)

    # ------------------------------------------------------------------
    # Resolve max_items
    # ------------------------------------------------------------------
    if max_items is None:
        max_items = getattr(resolved, "tavily_max_results", MAX_RESULTS_DEFAULT)
    max_items = max(MAX_RESULTS_MIN, min(MAX_RESULTS_MAX, int(max_items)))

    # ------------------------------------------------------------------
    # Build query
    # ------------------------------------------------------------------
    query = _build_query(asset_symbol, tuple(resolved.apify_query_term_list[:1]))

    # ------------------------------------------------------------------
    # Resolve provider
    # ------------------------------------------------------------------
    provider, provider_name = _build_provider(resolved, asset_symbol=asset_symbol)
    if provider is None:
        # No news provider available — return an empty manifest so the pipeline
        # does not break.  Module 13 will see an empty news set and fall back to
        # price/technical indicators only.
        logger.info(
            "No news provider available (no Tavily/Exa credentials); returning empty news manifest for %s",
            asset_symbol,
        )
        manifest_id = new_run_id("news")
        manifest = NewsManifest(
            collection_id=manifest_id,
            source=SOURCE_NAME,
            asset_symbol=asset_symbol,
            query=query,
            collection_window={
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            requested_max_items=max_items,
            retrieved_max_items=0,
            retrieved_at=now,
            news_item_count=0,
            deduplicated_item_count=0,
            items=(),
            output_dir=str(output_dir) if output_dir else "",
            manifest_path="",
        )
        return manifest

    # ------------------------------------------------------------------
    # Fetch news
    # ------------------------------------------------------------------
    fetch_start = now_call()
    try:
        items, warnings, error = provider.fetch(
            query=query,
            start=start_dt,
            end=end_dt,
            max_items=max_items,
        )
        fetch_elapsed = (now_call() - fetch_start).total_seconds()
    except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
        logger.error(
            "News fetch failed for %s: %s",
            asset_symbol,
            exc,
            extra={"event": "news_fetch_error"},
        )
        items, warnings, error = [], [f"News fetch exception: {type(exc).__name__}: {exc}"], str(exc)

    # ------------------------------------------------------------------
    # Persist manifest to disk (optional, for auditing)
    # ------------------------------------------------------------------
    manifest_id = new_run_id("news")
    deduped, deduplicated_count = _deduplicate_items(items)  # type: ignore[arg]

    # Count exclusions and uncertain relevance
    excluded_collisions = sum(
        1 for item in items if item.relevance_flag == "filtered"
    )
    uncertain_count = sum(1 for item in items if item.relevance_flag == "uncertain")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        manifest_output_path = str(out_path / f"news_manifest_{manifest_id}.json")
    else:
        manifest_output_path = ""

    manifest = NewsManifest(
        collection_id=manifest_id,
        source=provider_name or SOURCE_NAME,
        asset_symbol=asset_symbol,
        query=query,
        collection_window={
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        requested_max_items=max_items,
        retrieved_max_items=len(items),
        retrieved_at=now,
        news_item_count=len(deduped),
        deduplicated_item_count=deduplicated_count,
        items=tuple(deduped),
        excluded_ticker_collisions=excluded_collisions,
        uncertain_relevance_count=uncertain_count,
        warnings=warnings,
        error=error,
        output_dir=str(output_dir) if output_dir else "",
        manifest_path=manifest_output_path,
    )

    if output_dir and manifest_output_path:
        try:
            Path(manifest_output_path).write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - write error is only logged
            logger.warning("Could not write news manifest: %s", exc)
        logger.info(
            "News manifest written to %s: %d items (%d deduped, %d uncertain, %d collisions)",
            manifest_output_path,
            len(deduped),
            deduplicated_count,
            uncertain_count,
            excluded_collisions,
        )

    return manifest


# ---------------------------------------------------------------------------
# Module-level convenience: deduplicate helper (shared)
# ---------------------------------------------------------------------------

def _deduplicate_items(items: list[NewsItem]) -> tuple[list[NewsItem], int]:
    """Deduplicate a list of NewsItem objects by URL then title fingerprint.

    Returns (deduplicated_list, discard_count).
    """
    seen_urls: set[str] = set()
    seen_title_fingerprints: set[str] = set()
    deduped: list[NewsItem] = []
    discarded = 0

    for item in items:
        url_norm = normalise_url(item.url)
        if url_norm in seen_urls:
            discarded += 1
            continue
        seen_urls.add(url_norm)

        fp = re.sub(r"\s+", " ", item.title.strip().lower())
        if fp in seen_title_fingerprints:
            discarded += 1
            continue
        seen_title_fingerprints.add(fp)

        deduped.append(item)

    return deduped, discarded


# ---------------------------------------------------------------------------
# Module entry point for ``python -m src.ingestion.news``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Allow running as ``python -m src.ingestion.news`` for a quick demo."""

    import sys
    from datetime import date

    from src.config import load_settings, get_settings

    try:
        settings = load_settings()
    except Exception:
        settings = get_settings()

    # Pick an asset symbol from args or settings
    asset_symbol = sys.argv[1] if len(sys.argv) > 1 else getattr(settings, "asset_symbol", "AAPL")

    # Date range: last 7 days by default
    today = date.today()
    start = today - timedelta(days=7)
    end = today

    result = collect_news(
        asset_symbol=asset_symbol,
        settings=settings,
        start_date=start,
        end_date=end,
        max_items=None,
        output_dir=None,
    )

    import json

    print(json.dumps(result.model_dump(mode="json", by_alias=True), default=str, indent=2))