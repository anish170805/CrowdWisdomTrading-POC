"""Live X ingestion through a configurable Apify actor (Module 2).

The module never guesses an actor's input field names.  The operator supplies a
verified JSON input template (inline or a file) whose placeholders are filled
from Module 0 settings and the generated X query.  The untouched dataset items
are persisted next to the normalized :class:`~src.ingestion.records.RawTweetRecord`
objects, so Module 3 receives the same downstream contract as Module 1.

Nothing here scores sentiment, deduplicates canonical records, or aggregates by
day.  Those steps belong to later modules.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.config import ApifyQueryMode, Settings, get_settings
from src.ingestion.field_parsing import (
    as_text,
    clean_post_id,
    date_range,
    is_present,
    json_safe,
    normalise_author,
    normalise_key,
    normalise_url,
    now_utc,
    parse_count,
    parse_timestamp,
    post_id_from_url,
    safe_component,
    write_json,
    write_jsonl,
)
from src.ingestion.records import QuarantinedTweetRow, RawTweetRecord, TweetSink
from src.logging_config import get_logger, new_run_id

logger = get_logger(__name__)

SOURCE_NAME = "apify_x"
SOURCE_FORMAT = "apify_dataset"

SUCCESS_RUN_STATUS = "SUCCEEDED"
TERMINAL_RUN_STATUSES = frozenset({"SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"})
FAILED_RUN_STATUSES = frozenset({"FAILED", "TIMED-OUT", "ABORTED"})

PLACEHOLDERS = frozenset(
    {
        "query",
        "query_terms",
        "max_items",
        "asset_symbol",
        "asset_type",
        "asset_slug",
        "start_date",
        "end_date",
        "since",
        "until",
        "language",
    }
)
QUERY_PLACEHOLDERS = frozenset({"query", "query_terms"})

# Alias order is a priority order.  Values are actor-neutral JSON paths; the
# configured APIFY_OUTPUT_FIELD_MAP overrides any entry when an actor uses a
# different shape.  Lookups are case-insensitive and tolerate nested objects.
APIFY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "post_id": (
        "id_str",
        "post_id",
        "tweet_id",
        "status_id",
        "rest_id",
        "id",
        "legacy.id_str",
        "tweet.id",
    ),
    "text": (
        "note_tweet.text",
        "noteTweet.text",
        "full_text",
        "fullText",
        "legacy.full_text",
        "tweet_text",
        "text",
        "content",
        "body",
        "message",
    ),
    "created_at": (
        "created_at",
        "createdAt",
        "created_at_utc",
        "timestamp",
        "posted_at",
        "postedAt",
        "date",
        "time",
        "tweet.created_at",
        "legacy.created_at",
        "published_at",
    ),
    "author": (
        "author.username",
        "author.userName",
        "author.screen_name",
        "author.handle",
        "user.screen_name",
        "user.username",
        "user.handle",
        "screen_name",
        "username",
        "user_name",
        "author",
        "user",
    ),
    "author_handle": (
        "author_handle",
        "authorHandle",
        "user.screen_name",
        "screen_name",
        "username",
        "handle",
    ),
    "likes": (
        "like_count",
        "likeCount",
        "likes",
        "favorite_count",
        "favoriteCount",
        "favorites",
        "public_metrics.like_count",
        "legacy.favorite_count",
    ),
    "reposts": (
        "retweet_count",
        "retweetCount",
        "repost_count",
        "reposts",
        "retweets",
        "public_metrics.retweet_count",
        "legacy.retweet_count",
    ),
    "replies": (
        "reply_count",
        "replyCount",
        "replies",
        "public_metrics.reply_count",
        "legacy.reply_count",
    ),
    "quotes": (
        "quote_count",
        "quoteCount",
        "quotes",
        "public_metrics.quote_count",
        "legacy.quote_count",
    ),
    "views": (
        "view_count",
        "viewCount",
        "views",
        "impression_count",
        "impressions",
        "public_metrics.impression_count",
        "legacy.impression_count",
    ),
    "url": (
        "url",
        "tweet_url",
        "tweetUrl",
        "post_url",
        "postUrl",
        "permalink",
        "link",
        "twitterUrl",
        "x_url",
        "legacy.url",
    ),
}
COUNT_FIELDS = ("likes", "reposts", "replies", "quotes", "views")
CANONICAL_FIELDS = frozenset(APIFY_FIELD_ALIASES)

_PLACEHOLDER_PATTERN = "|".join(sorted(PLACEHOLDERS, key=len, reverse=True))
_PLACEHOLDER_RE = re.compile(r"\{(" + _PLACEHOLDER_PATTERN + r")\}")
_EXACT_PLACEHOLDER_RE = re.compile(r"^\{\s*(" + _PLACEHOLDER_PATTERN + r")\s*\}$")
_SIMPLE_BRACE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|token|secret|password|passwd|"
    r"authorization|credential)",
    re.IGNORECASE,
)
_MISSING = object()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ApifyIngestionError(RuntimeError):
    """Base class for live-ingestion failures."""


class ApifyConfigurationError(ApifyIngestionError):
    """The module cannot build a safe actor request from configuration."""


class ApifyTemplateError(ApifyConfigurationError):
    """The configured actor input template is invalid or incomplete."""


class ApifyActorInputError(ApifyIngestionError):
    """The rendered actor input failed schema or remote validation."""


class ApifyRunError(ApifyIngestionError):
    """The actor run could not be started or tracked."""


class ApifyRunFailedError(ApifyRunError):
    """The actor run finished in a failed, timed-out, or aborted state."""


class ApifyRunTimeoutError(ApifyRunError):
    """Waiting for the actor run exceeded the configured timeout."""


class ApifyDatasetError(ApifyIngestionError):
    """The run's dataset could not be retrieved."""


# ---------------------------------------------------------------------------
# Apify client protocols (kept small so tests can inject fakes)
# ---------------------------------------------------------------------------
class ApifyActorClient(Protocol):
    def start(
        self,
        *,
        run_input: Any = None,
        max_items: int | None = None,
        timeout_secs: int | None = None,
    ) -> dict[str, Any]: ...

    def get(self) -> dict[str, Any] | None: ...

    def validate_input(
        self,
        run_input: Any = None,
        *,
        build_tag: str | None = None,
    ) -> bool: ...


class ApifyRunClient(Protocol):
    def get(self) -> dict[str, Any] | None: ...

    def abort(self, *, gracefully: bool | None = None) -> dict[str, Any]: ...


class ApifyDatasetClient(Protocol):
    def list_items(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
        clean: bool | None = None,
    ) -> Any: ...


class ApifyClientProtocol(Protocol):
    def actor(self, actor_id: str) -> ApifyActorClient: ...

    def run(self, run_id: str) -> ApifyRunClient: ...

    def dataset(self, dataset_id: str) -> ApifyDatasetClient: ...


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuerySpec:
    """The asset query, disambiguation terms, and collection window."""

    asset_symbol: str
    asset_type: str
    mode: ApifyQueryMode
    terms: tuple[str, ...]
    exclusions: tuple[str, ...]
    language: str | None
    start_date: date
    end_date: date
    timezone: str
    window_start_utc: datetime
    window_end_utc: datetime
    query: str

    @property
    def since(self) -> str:
        return self.window_start_utc.isoformat()

    @property
    def until(self) -> str:
        """Exclusive end of the collection window, in UTC."""
        return self.window_end_utc.isoformat()

    def template_values(self, max_items: int) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_terms": list(self.terms),
            "max_items": max_items,
            "asset_symbol": self.asset_symbol,
            "asset_type": self.asset_type,
            "asset_slug": safe_component(self.asset_symbol).upper(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "since": self.since,
            "until": self.until,
            "language": self.language or "",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_symbol": self.asset_symbol,
            "asset_type": self.asset_type,
            "query_mode": self.mode.value,
            "query": self.query,
            "query_terms": list(self.terms),
            "query_exclusions": list(self.exclusions),
            "language": self.language,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timezone": self.timezone,
            "window_start_utc": self.window_start_utc.isoformat(),
            "window_end_utc": self.window_end_utc.isoformat(),
        }


@dataclass(frozen=True)
class PreparedCollection:
    """A validated actor request that is ready to run or inspect."""

    query: QuerySpec
    run_input: dict[str, Any]
    max_items: int
    template_source: str
    used_placeholders: tuple[str, ...]
    input_schema_path: str | None = None
    input_schema_validated: bool = False

    def to_dict(self, *, include_run_input: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query.to_dict(),
            "max_items": self.max_items,
            "template_source": self.template_source,
            "used_placeholders": list(self.used_placeholders),
            "input_schema_path": self.input_schema_path,
            "input_schema_validated": self.input_schema_validated,
        }
        if include_run_input:
            payload["run_input"] = json_safe(self.run_input)
        return payload


class CollectionManifest(BaseModel):
    """Auditable provenance for one live collection batch (never contains secrets)."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str
    source: str = SOURCE_NAME
    actor_id: str
    actor_run_id: str | None = None
    dataset_id: str | None = None
    run_status: str
    run_status_message: str | None = None

    query: str
    query_spec: dict[str, Any]
    collection_window: dict[str, Any]
    requested_max_items: int
    truncated: bool = False

    input_template_source: str
    input_template_placeholders: list[str]
    input_schema_path: str | None = None
    input_schema_validated: bool = False
    output_field_map: dict[str, str] = Field(default_factory=dict)

    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    poll_attempts: int = 0
    api_retries: int = 0
    rate_limit_retries: int = 0

    dataset_item_count: int = 0
    valid_records: int = 0
    quarantined_records: int = 0
    duplicate_post_ids: int = 0
    missing_post_ids: int = 0
    outside_configured_window: int = 0
    date_window_applied: bool = False
    empty_result: bool = False
    partial: bool = False

    date_range_utc: dict[str, str | None] = Field(default_factory=dict)
    parse_error_counts: dict[str, int] = Field(default_factory=dict)
    parse_errors: list[str] = Field(default_factory=list)
    parse_errors_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)

    output_dir: str
    actor_run_path: str
    dataset_items_path: str
    records_path: str
    quarantine_path: str
    manifest_path: str

    sink: str | None = None
    sink_error: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CollectionResult:
    """Normalized records, quarantine rows, and the collection manifest."""

    records: tuple[RawTweetRecord, ...]
    quarantined: tuple[QuarantinedTweetRow, ...]
    manifest: CollectionManifest
    sink_exception: Exception | None = field(default=None, repr=False, compare=False)

    @property
    def empty_result(self) -> bool:
        return self.manifest.empty_result

    @property
    def partial(self) -> bool:
        return self.manifest.partial

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"manifest": self.manifest.model_dump(mode="json")}
        if include_records:
            payload["records"] = [record.model_dump(mode="json") for record in self.records]
            payload["quarantined"] = [row.model_dump(mode="json") for row in self.quarantined]
        return payload


@dataclass
class _RetryTracker:
    api_retries: int = 0
    rate_limit_retries: int = 0

    def record(self, attempt: int, exc: Exception, delay: float) -> None:
        self.api_retries += 1
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            self.rate_limit_retries += 1
        logger.warning(
            "Retrying Apify API call after a transient failure",
            extra={
                "event": "apify_x.retry",
                "attempt": attempt,
                "delay_seconds": delay,
                "status_code": status_code,
                "error_type": type(exc).__name__,
            },
        )


@dataclass(frozen=True)
class _DatasetFetch:
    items: list[Any]
    truncated: bool
    total: int | None


@dataclass(frozen=True)
class _NormalisationStats:
    duplicate_post_ids: int
    missing_post_ids: int
    outside_configured_window: int
    parse_error_counts: dict[str, int]
    parse_errors: list[str]
    parse_errors_truncated: bool
    warning_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Query preparation
# ---------------------------------------------------------------------------
def build_query_spec(
    settings: Settings | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    query_mode: ApifyQueryMode | str | None = None,
    extra_terms: Iterable[str] | None = None,
    exclude_terms: Iterable[str] | None = None,
    language: str | None = None,
) -> QuerySpec:
    """Build the X query and UTC collection window from settings/overrides."""
    resolved = settings or get_settings()

    start = _coerce_date(start_date or resolved.effective_start_date, "start_date")
    end = _coerce_date(end_date or resolved.effective_end_date, "end_date")
    if start > end:
        raise ApifyConfigurationError(
            f"start_date {start.isoformat()} must be on or before end_date {end.isoformat()}"
        )

    mode = _coerce_query_mode(query_mode or resolved.apify_query_mode)
    aliases = tuple(extra_terms) if extra_terms is not None else resolved.apify_query_term_list
    exclusions = (
        tuple(exclude_terms) if exclude_terms is not None else resolved.apify_query_exclusion_list
    )
    language_value = _normalise_language(
        language if language is not None else resolved.apify_query_language
    )

    terms = _build_terms(resolved.asset_symbol, mode, aliases)
    query = _build_query(terms, exclusions, language_value, start, end)

    timezone = resolved.tzinfo
    window_start = datetime.combine(start, dt_time.min, tzinfo=timezone)
    window_end = datetime.combine(end + timedelta(days=1), dt_time.min, tzinfo=timezone)

    return QuerySpec(
        asset_symbol=resolved.asset_symbol,
        asset_type=resolved.asset_type.value,
        mode=mode,
        terms=terms,
        exclusions=tuple(_exclusion_operator(term) for term in exclusions if term.strip()),
        language=language_value,
        start_date=start,
        end_date=end,
        timezone=timezone.key,
        window_start_utc=window_start.astimezone(UTC),
        window_end_utc=window_end.astimezone(UTC),
        query=query,
    )


def prepare_x_collection(
    settings: Settings | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    query_mode: ApifyQueryMode | str | None = None,
    extra_terms: Iterable[str] | None = None,
    exclude_terms: Iterable[str] | None = None,
    language: str | None = None,
    max_items: int | None = None,
    input_template: Mapping[str, Any] | None = None,
    output_field_map: Mapping[str, str] | None = None,
) -> PreparedCollection:
    """Resolve the query, render the actor input, and validate it locally."""
    resolved = settings or get_settings()
    resolved_max_items = _positive_int(
        max_items if max_items is not None else resolved.apify_max_items,
        "max_items",
    )
    query = build_query_spec(
        resolved,
        start_date=start_date,
        end_date=end_date,
        query_mode=query_mode,
        extra_terms=extra_terms,
        exclude_terms=exclude_terms,
        language=language,
    )
    _validate_field_map(
        output_field_map if output_field_map is not None else resolved.apify_output_field_map
    )

    template, template_source = _resolve_template(resolved, input_template)
    used_placeholders = _template_placeholders(template)
    unknown = used_placeholders - PLACEHOLDERS
    if unknown:
        raise ApifyTemplateError(
            "Actor input template contains unsupported placeholder(s): "
            f"{', '.join(sorted(unknown))}. Supported placeholders: "
            f"{', '.join(sorted(PLACEHOLDERS))}."
        )
    if not used_placeholders & QUERY_PLACEHOLDERS:
        raise ApifyTemplateError(
            "Actor input template must use {query} or {query_terms}; otherwise the "
            "generated X query is not sent to the actor."
        )

    run_input = _render_template(template, query.template_values(resolved_max_items))
    if not isinstance(run_input, dict):
        raise ApifyTemplateError("Rendered actor input must be a JSON object.")

    schema_path = resolved.apify_input_schema_path
    schema_validated = False
    if schema_path is not None:
        _validate_run_input_against_schema(run_input, schema_path)
        schema_validated = True

    return PreparedCollection(
        query=query,
        run_input=run_input,
        max_items=resolved_max_items,
        template_source=template_source,
        used_placeholders=tuple(sorted(used_placeholders)),
        input_schema_path=str(schema_path) if schema_path is not None else None,
        input_schema_validated=schema_validated,
    )


def _coerce_date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise ApifyConfigurationError(f"{label} must be a date, got {value!r}")


def _coerce_query_mode(value: Any) -> ApifyQueryMode:
    if isinstance(value, ApifyQueryMode):
        return value
    try:
        return ApifyQueryMode(str(value).strip().lower())
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in ApifyQueryMode)
        raise ApifyConfigurationError(
            f"Unknown Apify query mode {value!r}. Supported modes: {supported}."
        ) from exc


def _normalise_language(value: Any) -> str | None:
    if value is None:
        return None
    language = str(value).strip().lower()
    if not language or language in {"none", "null", "unset"}:
        return None
    if not re.fullmatch(r"[a-z]{2,3}", language):
        raise ApifyConfigurationError(
            f"Invalid language code {value!r}; use a two- or three-letter code such as en."
        )
    return language


def _build_terms(
    asset_symbol: str,
    mode: ApifyQueryMode,
    aliases: Iterable[str],
) -> tuple[str, ...]:
    symbol = asset_symbol.strip()
    if not symbol:
        raise ApifyConfigurationError("ASSET_SYMBOL must not be empty")

    terms: list[str] = []
    if mode in {ApifyQueryMode.CASHTAG_AND_SYMBOL, ApifyQueryMode.CASHTAG_ONLY}:
        terms.append(f"${_cashtag_symbol(symbol)}")
    if mode in {ApifyQueryMode.CASHTAG_AND_SYMBOL, ApifyQueryMode.SYMBOL_ONLY}:
        terms.append(_quote_phrase(symbol))

    for alias in aliases:
        term = _quote_alias(alias)
        if term and term not in terms:
            terms.append(term)

    if not terms:
        raise ApifyConfigurationError("The configured query mode produced no search terms")
    return tuple(terms)


def _cashtag_symbol(symbol: str) -> str:
    base = symbol.split("-", 1)[0].split("=", 1)[0].lstrip("^")
    base = base.split(".", 1)[0]
    return base.upper() or symbol.upper()


def _quote_phrase(value: str) -> str:
    return '"' + value.strip().replace('"', '\\"') + '"'


def _quote_alias(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.startswith(("$", "#", "@", '"', "-", "(")):
        return text
    if re.match(r"^[a-z_]+:", text, flags=re.IGNORECASE):
        return text
    if re.search(r"[\s:]", text) or not re.fullmatch(r"[A-Za-z0-9_.^=+-]+", text):
        return _quote_phrase(text)
    return text


def _exclusion_operator(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.startswith("-"):
        return text
    if re.match(r"^[a-z_]+:", text, flags=re.IGNORECASE):
        return f"-{text}"
    return f"-{_quote_alias(text)}"


def _build_query(
    terms: Sequence[str],
    exclusions: Iterable[str],
    language: str | None,
    start: date,
    end: date,
) -> str:
    parts: list[str] = []
    if terms:
        parts.append("(" + " OR ".join(terms) + ")")
    if language:
        parts.append(f"lang:{language}")
    parts.append(f"since:{start.isoformat()}")
    # X's until: operator is exclusive, so request the day after the inclusive end.
    parts.append(f"until:{(end + timedelta(days=1)).isoformat()}")
    for exclusion in exclusions:
        operator = _exclusion_operator(exclusion)
        if operator:
            parts.append(operator)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Actor input templates
# ---------------------------------------------------------------------------
def _resolve_template(
    settings: Settings,
    override: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if override is not None:
        template = override
        source = "argument"
    elif settings.apify_input_template is not None:
        template = settings.apify_input_template
        source = "inline"
    elif settings.apify_input_template_path is not None:
        path = settings.apify_input_template_path
        if not path.exists():
            raise ApifyConfigurationError(f"APIFY_INPUT_TEMPLATE_PATH does not exist: {path}")
        try:
            template = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApifyTemplateError(
                f"Could not read the actor input template {path}: {exc}"
            ) from exc
        source = f"file:{path}"
    else:
        raise ApifyConfigurationError(
            "No Apify actor input template is configured. Inspect the actor's "
            "INPUT_SCHEMA.json, then set APIFY_INPUT_TEMPLATE (inline JSON) or "
            "APIFY_INPUT_TEMPLATE_PATH. Do not guess actor-specific field names."
        )

    if not isinstance(template, dict):
        raise ApifyTemplateError("The actor input template must be a JSON object")
    return json_safe(template), source


def _template_placeholders(template: Any) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            found.update(_PLACEHOLDER_RE.findall(value))
            for name in _SIMPLE_BRACE_RE.findall(value):
                if name not in PLACEHOLDERS:
                    raise ApifyTemplateError(
                        f"Actor input template references unknown placeholder {{{name}}}. "
                        f"Supported placeholders: {', '.join(sorted(PLACEHOLDERS))}."
                    )
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(template)
    return found


def _format_placeholder(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _render_template(template: Any, values: Mapping[str, Any]) -> Any:
    def render(value: Any) -> Any:
        if isinstance(value, str):
            exact = _EXACT_PLACEHOLDER_RE.match(value.strip())
            if exact is not None:
                name = exact.group(1)
                if name not in values:
                    raise ApifyTemplateError(
                        f"Actor input template references unknown placeholder {{{name}}}"
                    )
                return values[name]
            return _PLACEHOLDER_RE.sub(
                lambda match: _format_placeholder(values[match.group(1)]),
                value,
            )
        if isinstance(value, Mapping):
            return {str(key): render(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [render(item) for item in value]
        return value

    return render(template)


# ---------------------------------------------------------------------------
# Local schema validation
# ---------------------------------------------------------------------------
def _validate_field_map(field_map: Mapping[str, str]) -> None:
    unknown = sorted(set(field_map) - CANONICAL_FIELDS)
    if unknown:
        raise ApifyConfigurationError(
            "APIFY_OUTPUT_FIELD_MAP contains unknown canonical field(s): "
            f"{', '.join(unknown)}. Supported fields: {', '.join(sorted(CANONICAL_FIELDS))}."
        )
    for field_name, path in field_map.items():
        if not str(path).strip():
            raise ApifyConfigurationError(
                f"APIFY_OUTPUT_FIELD_MAP path for {field_name!r} must not be empty"
            )


def _validate_run_input_against_schema(run_input: Any, schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApifyConfigurationError(
            f"Could not read the actor input schema {schema_path}: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise ApifyConfigurationError(
            f"Actor input schema {schema_path} must contain a JSON object"
        )
    errors = _validate_json_schema(run_input, schema, path="$")
    if errors:
        details = "; ".join(errors[:10])
        if len(errors) > 10:
            details += f"; ... {len(errors) - 10} more"
        raise ApifyActorInputError(f"Actor input does not match {schema_path}: {details}")


def _validate_json_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_json_type(value, str(item)) for item in types):
            got = _json_type(value)
            return [f"{path}: expected {', '.join(map(str, types))}, got {got}"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: must be one of {enum_values!r}")

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{path}: must contain at least {minimum_length} character(s)")
        maximum_length = schema.get("maxLength")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            errors.append(f"{path}: must contain at most {maximum_length} character(s)")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    errors.append(f"{path}: must match pattern {pattern!r}")
            except re.error:
                pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: must be >= {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: must be <= {maximum}")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            errors.append(f"{path}: must be > {exclusive_minimum}")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            errors.append(f"{path}: must be < {exclusive_maximum}")

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties")
        property_schemas = properties if isinstance(properties, dict) else {}
        for key, property_schema in property_schemas.items():
            if key in value and isinstance(property_schema, dict):
                errors.extend(
                    _validate_json_schema(value[key], property_schema, path=f"{path}.{key}")
                )
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value:
                if key not in property_schemas:
                    errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(additional, dict):
            for key, item in value.items():
                if key not in property_schemas:
                    errors.extend(_validate_json_schema(item, additional, path=f"{path}.{key}"))

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{path}: must contain at least {minimum_items} item(s)")
        maximum_items = schema.get("maxItems")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            errors.append(f"{path}: must contain at most {maximum_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_json_schema(item, item_schema, path=f"{path}[{index}]"))

    for key in ("allOf",):
        subschemas = schema.get(key)
        if isinstance(subschemas, list):
            for subschema in subschemas:
                if isinstance(subschema, dict):
                    errors.extend(_validate_json_schema(value, subschema, path=path))

    any_of = schema.get("anyOf")
    if (
        isinstance(any_of, list)
        and any_of
        and not any(
            not _validate_json_schema(value, item, path=path)
            for item in any_of
            if isinstance(item, dict)
        )
    ):
        errors.append(f"{path}: does not match any allowed schema")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(
            1
            for item in one_of
            if isinstance(item, dict) and not _validate_json_schema(value, item, path=path)
        )
        if matches != 1:
            errors.append(f"{path}: must match exactly one allowed schema")

    return errors


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Public collection entry point
# ---------------------------------------------------------------------------
def collect_x_posts(
    settings: Settings | None = None,
    *,
    prepared: PreparedCollection | None = None,
    client: ApifyClientProtocol | None = None,
    client_factory: Callable[[Settings], ApifyClientProtocol] | None = None,
    output_dir: str | Path | None = None,
    collection_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    query_mode: ApifyQueryMode | str | None = None,
    extra_terms: Iterable[str] | None = None,
    exclude_terms: Iterable[str] | None = None,
    language: str | None = None,
    max_items: int | None = None,
    input_template: Mapping[str, Any] | None = None,
    output_field_map: Mapping[str, str] | None = None,
    run_timeout_seconds: float | None = None,
    poll_interval_seconds: float | None = None,
    validate_remote: bool = False,
    allow_partial: bool = False,
    apply_date_window: bool = False,
    sink: TweetSink | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    max_parse_errors: int = 500,
) -> CollectionResult:
    """Run the configured actor and persist raw records plus provenance.

    A successful run with an empty dataset is not an error; the manifest records
    ``empty_result=True``.  Failed, aborted, or timed-out runs persist whatever
    items were available and raise unless ``allow_partial=True``.
    """
    resolved = settings or get_settings()
    prepared_collection = prepared or prepare_x_collection(
        resolved,
        start_date=start_date,
        end_date=end_date,
        query_mode=query_mode,
        extra_terms=extra_terms,
        exclude_terms=exclude_terms,
        language=language,
        max_items=max_items,
        input_template=input_template,
        output_field_map=output_field_map,
    )
    field_map = _resolve_field_map(resolved, output_field_map)

    timeout = _positive_float(
        run_timeout_seconds
        if run_timeout_seconds is not None
        else resolved.apify_run_timeout_seconds,
        "run_timeout_seconds",
    )
    poll_interval = _positive_float(
        poll_interval_seconds
        if poll_interval_seconds is not None
        else resolved.apify_poll_interval_seconds,
        "poll_interval_seconds",
    )
    if max_parse_errors < 0:
        raise ApifyConfigurationError("max_parse_errors must be zero or greater")

    clock_fn = clock or (lambda: datetime.now(tz=UTC))
    monotonic_fn = monotonic or time.monotonic
    sleep_fn = sleep or time.sleep

    started_at = now_utc(clock_fn)
    requested_id = collection_id or new_run_id("apify-x")
    resolved_collection_id = safe_component(requested_id)
    warnings: list[str] = []
    if resolved_collection_id != requested_id:
        warnings.append(
            f"collection_id {requested_id!r} was sanitized to "
            f"{resolved_collection_id!r} for use as a directory name"
        )

    destination = _resolve_output_directory(
        resolved,
        output_dir=output_dir,
        collection_id=resolved_collection_id,
    )
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ApifyIngestionError(
            f"Could not create collection directory {destination}: {exc.strerror or exc}"
        ) from exc

    apify_client = client
    if apify_client is None:
        apify_client = (
            client_factory(resolved) if client_factory is not None else _default_client(resolved)
        )

    tracker = _RetryTracker()
    retries = max(0, int(resolved.http_max_retries))
    actor = apify_client.actor(resolved.apify_actor_id)

    logger.info(
        "Apify X collection started",
        extra={
            "event": "apify_x.started",
            "collection_id": resolved_collection_id,
            "actor_id": resolved.apify_actor_id,
            "query": prepared_collection.query.query,
            "max_items": prepared_collection.max_items,
        },
    )

    if validate_remote:
        _validate_remote_input(
            actor,
            prepared_collection.run_input,
            retries=retries,
            tracker=tracker,
            sleep=sleep_fn,
        )

    try:
        start_response = _call_with_retries(
            lambda: actor.start(
                run_input=prepared_collection.run_input,
                max_items=prepared_collection.max_items,
                timeout_secs=int(timeout),
            ),
            max_retries=retries,
            tracker=tracker,
            sleep=sleep_fn,
        )
    except ApifyIngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
        raise ApifyRunError(
            f"Could not start Apify actor {resolved.apify_actor_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(start_response, dict):
        raise ApifyRunError("Apify actor.start() did not return a run object")

    run_id = _first_string(start_response, "id", "runId")
    if run_id is None:
        raise ApifyRunError(
            "Apify actor.start() response did not include a run ID; check the actor "
            "and API token permissions."
        )
    dataset_id = _first_string(start_response, "defaultDatasetId", "datasetId")
    run_client = apify_client.run(run_id)

    poll_attempts = 0
    final_run = start_response
    timed_out = False
    deadline = monotonic_fn() + timeout
    max_poll_attempts = max(1, int(timeout / poll_interval) + 5)

    while True:
        status = _run_status(final_run)
        if status in TERMINAL_RUN_STATUSES:
            break
        if monotonic_fn() >= deadline or poll_attempts >= max_poll_attempts:
            timed_out = True
            _abort_run(run_client)
            break

        sleep_fn(poll_interval)
        poll_attempts += 1
        try:
            next_run = _call_with_retries(
                run_client.get,
                max_retries=retries,
                tracker=tracker,
                sleep=sleep_fn,
            )
        except ApifyIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
            _abort_run(run_client)
            raise ApifyRunError(
                f"Could not poll Apify run {run_id}: {type(exc).__name__}: {exc}"
            ) from exc
        if next_run is None:
            raise ApifyRunError(f"Apify run {run_id} disappeared while polling")
        if not isinstance(next_run, dict):
            raise ApifyRunError(f"Apify run {run_id} returned an unexpected run object")
        final_run = next_run

    run_status = _run_status(final_run)
    if timed_out:
        run_status = "TIMED-OUT"
    dataset_id = _first_string(final_run, "defaultDatasetId", "datasetId") or dataset_id
    status_message = _first_string(final_run, "statusMessage", "message")

    dataset_fetch: _DatasetFetch | None = None
    dataset_error: str | None = None
    if dataset_id is not None:
        # Give dataset retrieval its own bounded window; a polling timeout must
        # still persist whatever items the actor already produced.
        dataset_deadline = monotonic_fn() + timeout
        try:
            dataset_fetch = _fetch_dataset_items(
                apify_client.dataset(dataset_id),
                max_items=prepared_collection.max_items,
                retries=retries,
                tracker=tracker,
                sleep=sleep_fn,
                deadline=dataset_deadline,
                monotonic=monotonic_fn,
            )
        except Exception as exc:  # noqa: BLE001 - persisted and surfaced below
            dataset_error = f"{type(exc).__name__}: {exc}"
            if run_status == SUCCESS_RUN_STATUS:
                raise ApifyDatasetError(
                    f"Run {run_id} succeeded but dataset {dataset_id} could not be "
                    f"retrieved: {dataset_error}"
                ) from exc
            warnings.append(
                f"Dataset {dataset_id} could not be retrieved for the {run_status} "
                f"run: {dataset_error}"
            )
    elif run_status == SUCCESS_RUN_STATUS:
        raise ApifyDatasetError(f"Run {run_id} succeeded but did not provide a default dataset ID")
    else:
        warnings.append("The run did not provide a dataset ID, so no partial items were available.")

    dataset_items = dataset_fetch.items if dataset_fetch is not None else []
    truncated = dataset_fetch.truncated if dataset_fetch is not None else False
    if truncated:
        warnings.append(
            f"Collection stopped at the configured APIFY_MAX_ITEMS="
            f"{prepared_collection.max_items}; the dataset may contain more items."
        )

    finished_at = now_utc(clock_fn)
    partial = run_status != SUCCESS_RUN_STATUS or dataset_error is not None
    if run_status != SUCCESS_RUN_STATUS:
        warnings.append(f"Apify run finished with status {run_status}; results are partial.")
    if timed_out:
        warnings.append(
            f"Waiting for the Apify run exceeded APIFY_RUN_TIMEOUT_SECONDS={timeout:g}."
        )

    result = _finalise_collection(
        settings=resolved,
        prepared=prepared_collection,
        collection_id=resolved_collection_id,
        output_dir=destination,
        started_at=started_at,
        finished_at=finished_at,
        run_status=run_status,
        status_message=status_message,
        run_id=run_id,
        dataset_id=dataset_id,
        start_response=start_response,
        final_run=final_run,
        dataset_items=dataset_items,
        truncated=truncated,
        partial=partial,
        error=dataset_error,
        warnings=warnings,
        poll_attempts=poll_attempts,
        tracker=tracker,
        field_map=field_map,
        apply_date_window=apply_date_window,
        max_parse_errors=max_parse_errors,
        sink=sink,
        call_sink=run_status == SUCCESS_RUN_STATUS or allow_partial,
    )

    logger.info(
        "Apify X collection completed",
        extra={
            "event": "apify_x.completed",
            "collection_id": resolved_collection_id,
            "actor_run_id": run_id,
            "dataset_id": dataset_id,
            "run_status": run_status,
            "dataset_items": len(dataset_items),
            "valid_records": result.manifest.valid_records,
            "quarantined_records": result.manifest.quarantined_records,
            "partial": result.manifest.partial,
            "manifest_path": result.manifest.manifest_path,
        },
    )

    if result.sink_exception is not None:
        raise ApifyIngestionError(
            f"Raw data was saved to {destination}, but the downstream sink failed: "
            f"{result.manifest.sink_error}"
        ) from result.sink_exception

    if timed_out and not allow_partial:
        raise ApifyRunTimeoutError(
            f"Apify run {run_id} did not finish within {timeout:g} seconds. "
            f"Partial results were saved to {destination}. Re-run with "
            "allow_partial=True (or --allow-partial) to accept them."
        )

    if run_status in FAILED_RUN_STATUSES and not allow_partial:
        raise ApifyRunFailedError(
            f"Apify run {run_id} finished with status {run_status}. "
            f"Partial results were saved to {destination}. Re-run with "
            "allow_partial=True (or --allow-partial) to accept them."
        )

    return result


def inspect_actor(
    settings: Settings | None = None,
    *,
    client: ApifyClientProtocol | None = None,
    actor_id: str | None = None,
    run_input: Any = None,
    validate_remote: bool = False,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Return actor metadata and, optionally, validate a rendered input payload.

    Apify's actor object does not include the input schema; use this helper to
    confirm the actor exists and to ask Apify to validate a payload built from
    the operator-supplied template.
    """
    resolved = settings or get_settings()
    resolved_actor_id = (actor_id or resolved.apify_actor_id).strip()
    if not resolved_actor_id:
        raise ApifyConfigurationError("APIFY_ACTOR_ID must not be empty")

    apify_client = client or _default_client(resolved)
    actor = apify_client.actor(resolved_actor_id)
    try:
        info = _call_with_retries(
            actor.get,
            max_retries=max(0, int(resolved.http_max_retries)),
            tracker=_RetryTracker(),
            sleep=sleep or time.sleep,
        )
    except ApifyIngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
        raise ApifyIngestionError(
            f"Could not inspect Apify actor {resolved_actor_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if info is None:
        raise ApifyIngestionError(
            f"Apify actor {resolved_actor_id!r} was not found or is not accessible"
        )

    payload: dict[str, Any] = {
        "actor_id": resolved_actor_id,
        "actor": json_safe(info),
        "remote_validation": None,
    }
    if validate_remote:
        if run_input is None:
            raise ApifyActorInputError("validate_remote=True requires a rendered run_input payload")
        payload["remote_validation"] = _validate_remote_input(
            actor,
            run_input,
            retries=max(0, int(resolved.http_max_retries)),
            tracker=_RetryTracker(),
            sleep=sleep or time.sleep,
        )
    return payload


def _default_client(settings: Settings) -> ApifyClientProtocol:
    try:
        from apify_client import ApifyClient
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ApifyIngestionError(
            "Live X ingestion requires the 'apify-client' package. Install it with: "
            "pip install apify-client"
        ) from exc

    return ApifyClient(
        settings.apify_api_token.get_secret_value(),
        max_retries=0,
        timeout_secs=settings.http_timeout_seconds,
    )


def _validate_remote_input(
    actor: ApifyActorClient,
    run_input: Any,
    *,
    retries: int,
    tracker: _RetryTracker,
    sleep: Callable[[float], None],
) -> bool:
    try:
        valid = _call_with_retries(
            lambda: actor.validate_input(run_input),
            max_retries=retries,
            tracker=tracker,
            sleep=sleep,
        )
    except ApifyIngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider rejection is surfaced with context
        raise ApifyActorInputError(
            "Apify rejected the rendered actor input. Compare the template with the "
            f"actor's INPUT_SCHEMA.json. Provider error: {type(exc).__name__}: {exc}"
        ) from exc

    if valid is False:
        raise ApifyActorInputError(
            "Apify actor input validation failed. Compare the template with the "
            "actor's INPUT_SCHEMA.json."
        )
    return True


# ---------------------------------------------------------------------------
# Polling, retries, and dataset retrieval
# ---------------------------------------------------------------------------
def _call_with_retries(
    operation: Callable[[], Any],
    *,
    max_retries: int,
    tracker: _RetryTracker,
    sleep: Callable[[float], None],
) -> Any:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_retries or not _is_transient_error(exc):
                raise
            delay = min(30.0, 2.0**attempt)
            attempt += 1
            tracker.record(attempt, exc, delay)
            sleep(delay)


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in (
            "ratelimit",
            "rate_limit",
            "timeout",
            "connection",
            "serviceunavailable",
            "temporar",
        )
    )


def _run_status(run: Mapping[str, Any]) -> str:
    value = run.get("status")
    return str(value).strip().upper() if value is not None else "UNKNOWN"


def _abort_run(run_client: ApifyRunClient) -> None:
    try:
        run_client.abort(gracefully=True)
    except Exception as exc:  # noqa: BLE001 - abort is best effort
        logger.warning(
            "Could not abort the timed-out Apify run",
            extra={"event": "apify_x.abort_failed", "error_type": type(exc).__name__},
        )


def _fetch_dataset_items(
    dataset: ApifyDatasetClient,
    *,
    max_items: int,
    retries: int,
    tracker: _RetryTracker,
    sleep: Callable[[float], None],
    deadline: float,
    monotonic: Callable[[], float],
) -> _DatasetFetch:
    items: list[Any] = []
    offset = 0
    limit = min(1000, max_items)
    total: int | None = None
    max_pages = max(1, (max_items + limit - 1) // limit + 2)

    for _ in range(max_pages):
        if len(items) >= max_items:
            break
        if monotonic() >= deadline:
            raise ApifyDatasetError(
                f"Timed out while retrieving the Apify dataset after {len(items)} item(s)."
            )

        page = _call_with_retries(
            lambda current_offset=offset: dataset.list_items(
                offset=current_offset, limit=limit, clean=False
            ),
            max_retries=retries,
            tracker=tracker,
            sleep=sleep,
        )
        page_items = _page_items(page)
        page_total = _page_total(page)
        if page_total is not None:
            total = page_total
        if not page_items:
            break

        remaining = max_items - len(items)
        items.extend(page_items[:remaining])
        if len(page_items) < limit:
            break
        offset += len(page_items)

    truncated = len(items) >= max_items
    if total is not None and total > len(items):
        truncated = True
    return _DatasetFetch(items=items, truncated=truncated, total=total)


def _page_items(page: Any) -> list[Any]:
    if page is None:
        return []
    if isinstance(page, (list, tuple)):
        return list(page)
    if isinstance(page, Mapping):
        value = page.get("items", [])
        return list(value) if isinstance(value, (list, tuple)) else []
    value = getattr(page, "items", None)
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ApifyDatasetError("Apify dataset list_items() returned an unsupported page object")


def _page_total(page: Any) -> int | None:
    value = page.get("total") if isinstance(page, Mapping) else getattr(page, "total", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _first_string(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value).strip()
            if text:
                return text
    return None


# ---------------------------------------------------------------------------
# Normalisation and persistence
# ---------------------------------------------------------------------------
def _resolve_field_map(
    settings: Settings,
    override: Mapping[str, str] | None,
) -> dict[str, str]:
    field_map = dict(settings.apify_output_field_map)
    if override is not None:
        field_map.update({str(key): str(value) for key, value in override.items()})
    _validate_field_map(field_map)
    return field_map


def _normalise_items(
    items: Sequence[Any],
    *,
    collection_id: str,
    dataset_id: str | None,
    imported_at: datetime,
    settings: Settings,
    query: QuerySpec,
    field_map: Mapping[str, str],
    apply_date_window: bool,
    max_parse_errors: int,
) -> tuple[list[RawTweetRecord], list[QuarantinedTweetRow], _NormalisationStats]:
    records: list[RawTweetRecord] = []
    quarantined: list[QuarantinedTweetRow] = []
    outside_window = 0
    warning_counts: Counter[str] = Counter()

    for source_row, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            quarantined.append(
                QuarantinedTweetRow(
                    source_row=source_row,
                    reason_code="invalid_item",
                    reason="dataset item is not a JSON object",
                    raw={"value": json_safe(item)},
                )
            )
            continue

        index = _index_item(item)
        row_warnings: list[str] = []

        text_value = _field_value(index, "text", field_map)
        text = as_text(text_value).strip()
        if not text:
            quarantined.append(
                _quarantine_item(
                    source_row,
                    "missing_text",
                    "no non-empty text field was found in the dataset item",
                    row_warnings,
                    item,
                )
            )
            continue

        created_value = _field_value(index, "created_at", field_map)
        if not is_present(created_value):
            quarantined.append(
                _quarantine_item(
                    source_row,
                    "missing_timestamp",
                    "no created-at timestamp was found in the dataset item",
                    row_warnings,
                    item,
                )
            )
            continue
        try:
            parsed_timestamp = parse_timestamp(
                created_value,
                assume_timezone=settings.tzinfo,
                allow_excel_serial=False,
            )
        except (ValueError, OverflowError) as exc:
            quarantined.append(
                _quarantine_item(source_row, "invalid_timestamp", str(exc), row_warnings, item)
            )
            continue
        if parsed_timestamp.assumed:
            row_warnings.append("timestamp_timezone_assumed")

        post_id_value = _field_value(index, "post_id", field_map)
        try:
            post_id, post_id_warnings = clean_post_id(post_id_value)
        except ValueError as exc:
            quarantined.append(
                _quarantine_item(source_row, "invalid_post_id", str(exc), row_warnings, item)
            )
            continue
        row_warnings.extend(post_id_warnings)

        url_value = _field_value(index, "url", field_map)
        url, url_warning = normalise_url(url_value)
        if url_warning is not None:
            row_warnings.append(f"url_{url_warning}")
        if post_id is None:
            derived = post_id_from_url(url_value)
            if derived is not None:
                post_id = derived
                row_warnings.append("post_id_derived_from_url")

        counts: dict[str, int | None] = {}
        engagement_error: str | None = None
        for field_name in COUNT_FIELDS:
            value = _field_value(index, field_name, field_map)
            try:
                counts[field_name] = parse_count(value, field_name=field_name)
            except ValueError as exc:
                engagement_error = str(exc)
                break
        if engagement_error is not None:
            quarantined.append(
                _quarantine_item(
                    source_row,
                    "invalid_engagement",
                    engagement_error,
                    row_warnings,
                    item,
                )
            )
            continue

        author_value = _field_value(index, "author", field_map)
        author, author_handle = normalise_author(author_value)
        explicit_handle = _field_value(index, "author_handle", field_map)
        if is_present(explicit_handle):
            handle = str(explicit_handle).strip().lstrip("@")
            if handle:
                author_handle = handle

        outside = not (query.window_start_utc <= parsed_timestamp.utc < query.window_end_utc)
        if outside:
            outside_window += 1
            if apply_date_window:
                quarantined.append(
                    _quarantine_item(
                        source_row,
                        "outside_date_window",
                        (
                            f"{parsed_timestamp.utc.isoformat()} is outside "
                            f"{query.window_start_utc.isoformat()} to "
                            f"{query.window_end_utc.isoformat()}"
                        ),
                        row_warnings,
                        item,
                    )
                )
                continue
            row_warnings.append("outside_date_window")

        record = RawTweetRecord(
            source=SOURCE_NAME,
            source_format=SOURCE_FORMAT,
            source_file=dataset_id or settings.apify_actor_id,
            source_row=source_row,
            import_batch_id=collection_id,
            imported_at=imported_at,
            post_id=post_id,
            text=text,
            created_at_utc=parsed_timestamp.utc,
            created_at_original=parsed_timestamp.original,
            source_timezone_offset=parsed_timestamp.offset,
            timezone_assumed=parsed_timestamp.assumed,
            author=author,
            author_handle=author_handle,
            likes=counts["likes"],
            reposts=counts["reposts"],
            replies=counts["replies"],
            quotes=counts["quotes"],
            views=counts["views"],
            url=url,
            warnings=list(dict.fromkeys(row_warnings)),
            raw=json_safe(item),
        )
        records.append(record)

    seen_ids: set[str] = set()
    duplicate_post_ids = 0
    missing_post_ids = 0
    for record in records:
        if record.post_id is None:
            missing_post_ids += 1
        elif record.post_id in seen_ids:
            duplicate_post_ids += 1
        else:
            seen_ids.add(record.post_id)

    for record in records:
        warning_counts.update(record.warnings)
    for row in quarantined:
        warning_counts.update(row.warnings)

    parse_error_counts = Counter(row.reason_code for row in quarantined)
    parse_errors = [
        f"item {row.source_row}: {row.reason_code}: {row.reason}" for row in quarantined
    ]
    parse_errors_truncated = len(parse_errors) > max_parse_errors
    if parse_errors_truncated:
        parse_errors = parse_errors[:max_parse_errors]

    return (
        records,
        quarantined,
        _NormalisationStats(
            duplicate_post_ids=duplicate_post_ids,
            missing_post_ids=missing_post_ids,
            outside_configured_window=outside_window,
            parse_error_counts=dict(sorted(parse_error_counts.items())),
            parse_errors=parse_errors,
            parse_errors_truncated=parse_errors_truncated,
            warning_counts=dict(sorted(warning_counts.items())),
        ),
    )


def _quarantine_item(
    source_row: int,
    reason_code: str,
    reason: str,
    warnings: Iterable[str],
    item: Mapping[str, Any],
) -> QuarantinedTweetRow:
    return QuarantinedTweetRow(
        source_row=source_row,
        reason_code=reason_code,
        reason=reason,
        warnings=list(dict.fromkeys(warnings)),
        raw=json_safe(item),
    )


def _index_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten an actor item into normalized path/leaf keys for alias lookup."""
    index: dict[str, Any] = {}

    def visit(value: Mapping[str, Any], prefix: str) -> None:
        for raw_key, child in value.items():
            key = normalise_key(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, Mapping):
                index.setdefault(path, child)
                index.setdefault(normalise_key(path), child)
                visit(child, path)
            else:
                index.setdefault(path, child)
                index.setdefault(normalise_key(path), child)
                index.setdefault(key, child)

    visit(item, "")
    return index


def _field_value(
    index: Mapping[str, Any],
    field_name: str,
    field_map: Mapping[str, str],
) -> Any:
    if field_name in field_map:
        value = _lookup_index(index, field_map[field_name])
        return None if value is _MISSING else value

    for alias in APIFY_FIELD_ALIASES[field_name]:
        value = _lookup_index(index, alias)
        if value is not _MISSING and is_present(value):
            return value
    return None


def _lookup_index(index: Mapping[str, Any], alias: str) -> Any:
    dot_path = ".".join(normalise_key(part) for part in alias.split("."))
    for key in (dot_path, normalise_key(alias)):
        if key in index:
            return index[key]
    return _MISSING


def _finalise_collection(
    *,
    settings: Settings,
    prepared: PreparedCollection,
    collection_id: str,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
    run_status: str,
    status_message: str | None,
    run_id: str,
    dataset_id: str | None,
    start_response: Mapping[str, Any],
    final_run: Mapping[str, Any],
    dataset_items: Sequence[Any],
    truncated: bool,
    partial: bool,
    error: str | None,
    warnings: Iterable[str],
    poll_attempts: int,
    tracker: _RetryTracker,
    field_map: Mapping[str, str],
    apply_date_window: bool,
    max_parse_errors: int,
    sink: TweetSink | None,
    call_sink: bool,
) -> CollectionResult:
    records, quarantined, stats = _normalise_items(
        dataset_items,
        collection_id=collection_id,
        dataset_id=dataset_id,
        imported_at=started_at,
        settings=settings,
        query=prepared.query,
        field_map=field_map,
        apply_date_window=apply_date_window,
        max_parse_errors=max_parse_errors,
    )

    warning_list = list(dict.fromkeys(warnings))
    for code, count in stats.warning_counts.items():
        warning_list.append(f"{code}: {count} item(s)")
    if stats.duplicate_post_ids:
        warning_list.append(
            f"{stats.duplicate_post_ids} item(s) share a post ID with an earlier item; "
            "Module 3 deduplicates by post ID."
        )
    if stats.missing_post_ids:
        warning_list.append(
            f"{stats.missing_post_ids} item(s) have no post ID; Module 3 uses its "
            "conservative fallback fingerprint."
        )
    if stats.outside_configured_window:
        if apply_date_window:
            warning_list.append(
                f"{stats.outside_configured_window} item(s) fall outside the configured "
                "date window and were quarantined."
            )
        else:
            warning_list.append(
                f"{stats.outside_configured_window} item(s) fall outside the configured "
                "date window and were kept. Use apply_date_window=True to quarantine them."
            )

    empty_result = run_status == SUCCESS_RUN_STATUS and len(dataset_items) == 0
    if empty_result:
        warning_list.append(
            "The actor run succeeded but the dataset was empty. Check the query, date "
            "window, and actor input."
        )

    actor_run_path = output_dir / "actor_run.json"
    dataset_items_path = output_dir / "dataset_items.json"
    records_path = output_dir / "records.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    manifest_path = output_dir / "manifest.json"

    try:
        write_json(
            actor_run_path,
            _redact_payload(
                {
                    "actor_id": settings.apify_actor_id,
                    "actor_run_id": run_id,
                    "dataset_id": dataset_id,
                    "status": run_status,
                    "status_message": status_message,
                    "run_input": json_safe(prepared.run_input),
                    "started_response": json_safe(start_response),
                    "final_run": json_safe(final_run),
                },
                settings,
            ),
        )
        write_json(dataset_items_path, list(dataset_items))
        write_jsonl(
            records_path,
            (record.model_dump(mode="json") for record in records),
        )
        write_jsonl(
            quarantine_path,
            (row.model_dump(mode="json") for row in quarantined),
        )
    except OSError as exc:
        raise ApifyIngestionError(
            f"Could not write collection files in {output_dir}: {exc.strerror or exc}"
        ) from exc

    sink_error: str | None = None
    sink_exception: Exception | None = None
    if sink is not None and call_sink:
        try:
            sink.ingest_tweets(records, batch_id=collection_id)
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced by the caller
            sink_error = f"{type(exc).__name__}: {exc}"
            sink_exception = exc
            logger.exception(
                "Apify X collection downstream sink failed",
                extra={
                    "event": "apify_x.sink_failed",
                    "collection_id": collection_id,
                    "sink": type(sink).__name__,
                },
            )

    manifest = CollectionManifest(
        collection_id=collection_id,
        actor_id=settings.apify_actor_id,
        actor_run_id=run_id,
        dataset_id=dataset_id,
        run_status=run_status,
        run_status_message=status_message,
        query=prepared.query.query,
        query_spec=prepared.query.to_dict(),
        collection_window={
            "start_date": prepared.query.start_date.isoformat(),
            "end_date": prepared.query.end_date.isoformat(),
            "timezone": prepared.query.timezone,
            "start_utc": prepared.query.window_start_utc.isoformat(),
            "end_utc": prepared.query.window_end_utc.isoformat(),
            "end_inclusive": True,
        },
        requested_max_items=prepared.max_items,
        truncated=truncated,
        input_template_source=prepared.template_source,
        input_template_placeholders=list(prepared.used_placeholders),
        input_schema_path=prepared.input_schema_path,
        input_schema_validated=prepared.input_schema_validated,
        output_field_map=dict(field_map),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        poll_attempts=poll_attempts,
        api_retries=tracker.api_retries,
        rate_limit_retries=tracker.rate_limit_retries,
        dataset_item_count=len(dataset_items),
        valid_records=len(records),
        quarantined_records=len(quarantined),
        duplicate_post_ids=stats.duplicate_post_ids,
        missing_post_ids=stats.missing_post_ids,
        outside_configured_window=stats.outside_configured_window,
        date_window_applied=apply_date_window,
        empty_result=empty_result,
        partial=partial,
        date_range_utc=date_range(records),
        parse_error_counts=stats.parse_error_counts,
        parse_errors=stats.parse_errors,
        parse_errors_truncated=stats.parse_errors_truncated,
        warnings=warning_list,
        output_dir=str(output_dir),
        actor_run_path=str(actor_run_path),
        dataset_items_path=str(dataset_items_path),
        records_path=str(records_path),
        quarantine_path=str(quarantine_path),
        manifest_path=str(manifest_path),
        sink=type(sink).__name__ if sink is not None and call_sink else None,
        sink_error=sink_error,
        error=error,
    )

    try:
        write_json(
            manifest_path,
            _redact_payload(manifest.model_dump(mode="json"), settings),
        )
    except OSError as exc:
        raise ApifyIngestionError(
            f"Could not write the collection manifest {manifest_path}: {exc.strerror or exc}"
        ) from exc

    return CollectionResult(
        records=tuple(records),
        quarantined=tuple(quarantined),
        manifest=manifest,
        sink_exception=sink_exception,
    )


def _redact_payload(value: Any, settings: Settings) -> Any:
    secrets = settings.secret_values()

    def redact(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): ("[REDACTED]" if _SENSITIVE_KEY_RE.search(str(key)) else redact(child))
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [redact(child) for child in item]
        if isinstance(item, str):
            text = item
            for secret in secrets:
                if secret:
                    text = text.replace(secret, "[REDACTED]")
            return text
        return item

    return redact(json_safe(value))


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ApifyConfigurationError(f"{label} must be an integer") from exc
    if number < 1:
        raise ApifyConfigurationError(f"{label} must be at least 1")
    return number


def _positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ApifyConfigurationError(f"{label} must be a number") from exc
    if number <= 0:
        raise ApifyConfigurationError(f"{label} must be greater than zero")
    return number


def _resolve_output_directory(
    settings: Settings,
    *,
    output_dir: str | Path | None,
    collection_id: str,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (settings.raw_data_dir / "live_x" / settings.asset_slug / collection_id).resolve()


__all__ = [
    "APIFY_FIELD_ALIASES",
    "ApifyActorInputError",
    "ApifyConfigurationError",
    "ApifyDatasetError",
    "ApifyIngestionError",
    "ApifyRunError",
    "ApifyRunFailedError",
    "ApifyRunTimeoutError",
    "ApifyTemplateError",
    "CANONICAL_FIELDS",
    "CollectionManifest",
    "CollectionResult",
    "PLACEHOLDERS",
    "PreparedCollection",
    "QuerySpec",
    "SOURCE_FORMAT",
    "SOURCE_NAME",
    "build_query_spec",
    "collect_x_posts",
    "inspect_actor",
    "prepare_x_collection",
]
