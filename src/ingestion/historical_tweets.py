"""Import historical X/Twitter posts from an explicit CSV, XLSX, or JSON export.

The module deliberately does not scrape a private document.  The operator
exports the source document (for example from Google Sheets) and passes the
file path here.  Every source row is preserved in ``raw_records.jsonl``;
malformed rows are written to ``quarantine.jsonl`` with a reason instead of
being dropped.  Valid rows are handed downstream through the
:class:`~src.ingestion.records.TweetSink` protocol owned by Module 3.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.ingestion.field_parsing import (
    as_text,
    clean_post_id,
    date_range,
    hash_file,
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
    safe_filename,
    write_json,
    write_jsonl,
)
from src.ingestion.records import QuarantinedTweetRow, RawTweetRecord, TweetSink
from src.logging_config import get_logger, new_run_id

logger = get_logger(__name__)

SOURCE_NAME = "historical_export"
SUPPORTED_FORMATS = frozenset({"csv", "tsv", "xlsx", "json", "jsonl"})
_FORMAT_ALIASES = {
    "excel": "xlsx",
    "xlsm": "xlsx",
    "ndjson": "jsonl",
    "jsonlines": "jsonl",
    "txt": "csv",
    "text": "csv",
}
_ROW_LIST_KEYS = ("data", "items", "tweets", "posts", "results", "rows", "records")
_COUNT_FIELDS = ("likes", "reposts", "replies", "quotes", "views")

# Alias order is a priority order: the first matching header wins.  Keys are
# compared after lower-casing and collapsing non-alphanumerics to underscores.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "post_id": (
        "id_str",
        "post_id",
        "tweet_id",
        "status_id",
        "postid",
        "tweetid",
        "statusid",
        "rest_id",
        "id",
    ),
    "text": (
        "note_tweet_text",
        "full_text",
        "full_text_note",
        "tweet_text",
        "post_text",
        "text",
        "tweet",
        "post",
        "content",
        "body",
        "message",
        "fulltext",
    ),
    "created_at": (
        "created_at",
        "createdat",
        "created",
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "posted_at",
        "postedat",
        "tweeted_at",
        "tweetedat",
        "post_created_at",
        "tweet_created_at",
        "created_at_utc",
        "timestamp_utc",
        "date_utc",
        "time_utc",
        "date_created",
        "created_date",
        "created_time",
        "post_date",
        "tweet_date",
        "publish_date",
        "published_at",
        "created_utc",
    ),
    "author": (
        "user_screen_name",
        "userscreenname",
        "user_username",
        "author_username",
        "authorusername",
        "screen_name",
        "screenname",
        "username",
        "user_name",
        "author",
        "handle",
        "account",
        "from_user",
        "user",
    ),
    "likes": (
        "like_count",
        "likecount",
        "likes",
        "likes_count",
        "favorite_count",
        "favoritecount",
        "favorites",
        "favourites",
        "num_likes",
    ),
    "reposts": (
        "retweet_count",
        "retweetcount",
        "reposts",
        "repost_count",
        "repostcount",
        "retweets",
        "shares",
        "share_count",
        "sharecount",
        "num_retweets",
    ),
    "replies": ("reply_count", "replycount", "replies", "replies_count", "num_replies"),
    "quotes": ("quote_count", "quotecount", "quotes", "quotes_count", "num_quotes"),
    "views": (
        "view_count",
        "viewcount",
        "views",
        "views_count",
        "impressions",
        "impression_count",
        "impressioncount",
        "num_views",
    ),
    "url": (
        "tweet_url",
        "tweeturl",
        "post_url",
        "posturl",
        "status_url",
        "statusurl",
        "web_url",
        "weburl",
        "permalink",
        "url",
        "link",
        "href",
    ),
}
_ALL_ALIASES = frozenset(alias for aliases in FIELD_ALIASES.values() for alias in aliases)


class HistoricalImportError(RuntimeError):
    """Raised when an export cannot be read or mapped at the file level."""


class ImportSummary(BaseModel):
    """Auditable record of one import batch."""

    model_config = ConfigDict(extra="forbid")

    import_batch_id: str
    source: str = SOURCE_NAME
    source_filename: str
    source_path: str
    source_format: str
    source_sha256: str
    source_bytes: int
    source_encoding: str | None = None
    source_delimiter: str | None = None
    source_sheet: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    imported_at: datetime
    started_at: datetime
    finished_at: datetime
    duration_seconds: float

    total_rows: int
    valid_rows: int
    quarantined_rows: int
    blank_rows_skipped: int
    duplicate_post_id_rows: int
    missing_post_id_rows: int
    outside_configured_window: int
    date_window_applied: bool
    configured_window: dict[str, str] | None = None
    date_range_utc: dict[str, str | None] = Field(default_factory=dict)

    columns: list[str]
    column_mapping: dict[str, str | None]
    unmapped_columns: list[str]
    missing_canonical_fields: list[str]
    parse_error_counts: dict[str, int]
    parse_errors: list[str]
    parse_errors_truncated: bool
    warnings: list[str]

    output_dir: str
    raw_records_path: str
    canonical_records_path: str
    quarantine_path: str
    manifest_path: str
    source_copy_path: str | None = None

    sink: str | None = None
    sink_error: str | None = None


@dataclass(frozen=True)
class ImportResult:
    """Imported records plus the batch summary and quarantine contents."""

    records: tuple[RawTweetRecord, ...]
    quarantined: tuple[QuarantinedTweetRow, ...]
    summary: ImportSummary

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"summary": self.summary.model_dump(mode="json")}
        if include_records:
            payload["records"] = [record.model_dump(mode="json") for record in self.records]
            payload["quarantined"] = [row.model_dump(mode="json") for row in self.quarantined]
        return payload


@dataclass
class _SourceRow:
    source_row: int | None
    raw: dict[str, Any]
    flat: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class _SourceTable:
    headers: list[str]
    rows: list[_SourceRow]
    source_format: str
    encoding: str | None = None
    delimiter: str | None = None
    sheet_name: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blank_rows_skipped: int = 0


@dataclass(frozen=True)
class _ValidationContext:
    mapping: dict[str, str | None]
    batch_id: str
    source_format: str
    source_file: str
    imported_at: datetime
    assume_timezone: tzinfo
    allow_excel_serial: bool
    window_start: datetime | None
    window_end: datetime | None
    apply_date_window: bool


@dataclass(frozen=True)
class _RowOutcome:
    record: RawTweetRecord | None = None
    quarantine: QuarantinedTweetRow | None = None
    outside_window: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def import_historical_tweets(
    source_path: str | Path,
    *,
    settings: Settings | None = None,
    output_dir: str | Path | None = None,
    source_format: str | None = None,
    column_map: Mapping[str, str] | None = None,
    assume_timezone: str | tzinfo | None = None,
    apply_date_window: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
    sheet_name: str | None = None,
    copy_source: bool = True,
    import_batch_id: str | None = None,
    sink: TweetSink | None = None,
    clock: Callable[[], datetime] | None = None,
    max_parse_errors: int = 500,
) -> ImportResult:
    """Import one explicit export file.

    Parameters
    ----------
    source_path:
        CSV, TSV, XLSX/XLSM, JSON, or JSON Lines export.  The file is never
        modified.
    settings:
        Module 0 settings.  Defaults to :func:`src.config.get_settings`.
    output_dir:
        Exact batch directory.  Defaults to
        ``<RAW_DATA_DIR>/historical_tweets/<ASSET_SLUG>/<batch_id>``.
    source_format:
        Override extension-based format detection.
    column_map:
        Canonical field to source column overrides, e.g. ``{"text": "Tweet"}``.
    assume_timezone:
        Timezone for naive timestamps.  Defaults to ``settings.timezone``.
    apply_date_window:
        When true, rows outside the configured/overridden date window are
        quarantined.  When false they are kept and counted.
    start_date / end_date:
        Optional inclusive date-window overrides (interpreted in
        ``settings.timezone``).
    sheet_name:
        Worksheet for XLSX input.  Defaults to the active sheet.
    copy_source:
        Copy the original export into the batch directory for provenance.
    import_batch_id:
        Optional caller-supplied batch ID; a sortable ID is generated otherwise.
    sink:
        Optional Module 3 implementation of ``ingest_tweets``.
    clock:
        Injectable UTC clock, used by tests.
    max_parse_errors:
        Maximum row errors listed in the manifest.  The quarantine file keeps
        every row regardless.
    """

    path = Path(source_path).expanduser()
    if not path.exists():
        raise HistoricalImportError(f"Source export does not exist: {path}")
    if not path.is_file():
        raise HistoricalImportError(f"Source export is not a file: {path}")
    path = path.resolve()

    if max_parse_errors < 0:
        raise HistoricalImportError("max_parse_errors must be zero or greater")

    resolved_settings = settings or get_settings()
    resolved_format = _detect_format(path, source_format)
    started_at = now_utc(clock)

    file_warnings: list[str] = []
    requested_batch_id = import_batch_id or new_run_id("historical")
    batch_id = safe_component(requested_batch_id)
    if batch_id != requested_batch_id:
        file_warnings.append(
            f"import_batch_id {requested_batch_id!r} was sanitized to {batch_id!r} "
            "for use as a directory name"
        )

    source_sha256, source_bytes = hash_file(path)
    logger.info(
        "Historical tweet import started",
        extra={
            "event": "historical_import.started",
            "import_batch_id": batch_id,
            "source_file": path.name,
            "source_format": resolved_format,
            "source_bytes": source_bytes,
        },
    )

    table = _read_source(
        path,
        source_format=resolved_format,
        sheet_name=sheet_name,
        warnings_list=file_warnings,
    )
    mapping, mapping_warnings = _build_mapping(table.headers, column_map or {})
    file_warnings.extend(mapping_warnings)
    _require_required_columns(mapping, table.headers)

    timezone = _coerce_timezone(assume_timezone, resolved_settings)
    window_start, window_end, configured_window = _resolve_window(
        resolved_settings,
        start_date=start_date,
        end_date=end_date,
    )

    context = _ValidationContext(
        mapping=mapping,
        batch_id=batch_id,
        source_format=table.source_format,
        source_file=path.name,
        imported_at=started_at,
        assume_timezone=timezone,
        allow_excel_serial=table.source_format == "xlsx",
        window_start=window_start,
        window_end=window_end,
        apply_date_window=apply_date_window,
    )

    records: list[RawTweetRecord] = []
    quarantined: list[QuarantinedTweetRow] = []
    outside_window = 0

    for row in table.rows:
        outcome = _validate_row(row, context)
        if outcome.outside_window:
            outside_window += 1
        if outcome.record is not None:
            records.append(outcome.record)
        elif outcome.quarantine is not None:
            quarantined.append(outcome.quarantine)

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

    parse_error_counts = Counter(row.reason_code for row in quarantined)
    parse_errors = [f"row {row.source_row}: {row.reason_code}: {row.reason}" for row in quarantined]
    parse_errors_truncated = len(parse_errors) > max_parse_errors
    if parse_errors_truncated:
        parse_errors = parse_errors[:max_parse_errors]

    row_warning_counts: Counter[str] = Counter()
    for record in records:
        row_warning_counts.update(record.warnings)
    for row in quarantined:
        row_warning_counts.update(row.warnings)

    warnings_list = list(dict.fromkeys(file_warnings))
    for code, count in sorted(row_warning_counts.items()):
        warnings_list.append(f"{code}: {count} row(s)")
    if duplicate_post_ids:
        warnings_list.append(
            f"{duplicate_post_ids} row(s) share a post ID with an earlier row; "
            "Module 3 deduplicates by post ID."
        )
    if missing_post_ids:
        warnings_list.append(
            f"{missing_post_ids} row(s) have no post ID; Module 3 uses its conservative "
            "fallback fingerprint."
        )
    if outside_window:
        if apply_date_window:
            warnings_list.append(
                f"{outside_window} row(s) fall outside the configured date window and "
                "were quarantined."
            )
        else:
            warnings_list.append(
                f"{outside_window} row(s) fall outside the configured date window and were "
                "kept. Use --apply-date-window to quarantine them."
            )

    date_range_utc = date_range(records)
    output_directory = _resolve_output_directory(
        resolved_settings, output_dir=output_dir, batch_id=batch_id
    )
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HistoricalImportError(
            f"Could not create import batch directory {output_directory}: {exc.strerror or exc}"
        ) from exc

    raw_records_path = output_directory / "raw_records.jsonl"
    canonical_records_path = output_directory / "canonical_records.jsonl"
    quarantine_path = output_directory / "quarantine.jsonl"
    manifest_path = output_directory / "manifest.json"

    try:
        write_jsonl(
            raw_records_path,
            (
                {
                    "source_row": row.source_row,
                    "raw": json_safe(row.raw),
                    "warnings": row.warnings,
                    "error_code": row.error_code,
                    "error_message": row.error_message,
                }
                for row in table.rows
            ),
        )
        write_jsonl(
            canonical_records_path,
            (record.model_dump(mode="json") for record in records),
        )
        write_jsonl(
            quarantine_path,
            (row.model_dump(mode="json") for row in quarantined),
        )
    except OSError as exc:
        raise HistoricalImportError(
            f"Could not write import batch files in {output_directory}: {exc.strerror or exc}"
        ) from exc

    source_copy_path: Path | None = None
    if copy_source:
        destination = output_directory / f"source_{safe_filename(path.name)}"
        if destination.resolve() != path:
            try:
                shutil.copy2(path, destination)
            except OSError as exc:
                raise HistoricalImportError(
                    f"Could not copy the source export to {destination}: {exc.strerror or exc}"
                ) from exc
        source_copy_path = destination

    sink_error: str | None = None
    sink_exception: Exception | None = None
    if sink is not None:
        try:
            sink.ingest_tweets(records, batch_id=batch_id)
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced below
            sink_error = f"{type(exc).__name__}: {exc}"
            sink_exception = exc
            logger.exception(
                "Historical import downstream sink failed",
                extra={
                    "event": "historical_import.sink_failed",
                    "import_batch_id": batch_id,
                    "sink": type(sink).__name__,
                },
            )

    finished_at = now_utc(clock)
    summary = ImportSummary(
        import_batch_id=batch_id,
        source_filename=path.name,
        source_path=str(path),
        source_format=table.source_format,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        source_encoding=table.encoding,
        source_delimiter=table.delimiter,
        source_sheet=table.sheet_name,
        source_metadata=table.source_metadata,
        imported_at=started_at,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        total_rows=len(table.rows),
        valid_rows=len(records),
        quarantined_rows=len(quarantined),
        blank_rows_skipped=table.blank_rows_skipped,
        duplicate_post_id_rows=duplicate_post_ids,
        missing_post_id_rows=missing_post_ids,
        outside_configured_window=outside_window,
        date_window_applied=apply_date_window,
        configured_window=configured_window,
        date_range_utc=date_range_utc,
        columns=list(table.headers),
        column_mapping=dict(mapping),
        unmapped_columns=[header for header in table.headers if header not in mapping.values()],
        missing_canonical_fields=[field for field, column in mapping.items() if column is None],
        parse_error_counts=dict(sorted(parse_error_counts.items())),
        parse_errors=parse_errors,
        parse_errors_truncated=parse_errors_truncated,
        warnings=list(dict.fromkeys(warnings_list)),
        output_dir=str(output_directory),
        raw_records_path=str(raw_records_path),
        canonical_records_path=str(canonical_records_path),
        quarantine_path=str(quarantine_path),
        manifest_path=str(manifest_path),
        source_copy_path=str(source_copy_path) if source_copy_path is not None else None,
        sink=type(sink).__name__ if sink is not None else None,
        sink_error=sink_error,
    )
    try:
        write_json(manifest_path, summary.model_dump(mode="json"))
    except OSError as exc:
        raise HistoricalImportError(
            f"Could not write the import manifest {manifest_path}: {exc.strerror or exc}"
        ) from exc

    logger.info(
        "Historical tweet import completed",
        extra={
            "event": "historical_import.completed",
            "import_batch_id": batch_id,
            "source_file": path.name,
            "total_rows": summary.total_rows,
            "valid_rows": summary.valid_rows,
            "quarantined_rows": summary.quarantined_rows,
            "manifest_path": str(manifest_path),
        },
    )

    if sink_error is not None:
        raise HistoricalImportError(
            f"Raw data was saved to {output_directory}, but the downstream sink failed: "
            f"{sink_error}"
        ) from sink_exception

    return ImportResult(
        records=tuple(records),
        quarantined=tuple(quarantined),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------
def _detect_format(path: Path, explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        value = explicit.strip().lower().lstrip(".")
        value = _FORMAT_ALIASES.get(value, value)
        if value not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise HistoricalImportError(
                f"Unsupported source format {explicit!r}. Supported formats: {supported}."
            )
        return value

    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise HistoricalImportError(
            f"Cannot infer a format for {path.name!r}; pass --format. Supported: {supported}."
        )
    value = _FORMAT_ALIASES.get(suffix, suffix)
    if value not in SUPPORTED_FORMATS:
        if suffix == "xls":
            raise HistoricalImportError(
                "Legacy .xls workbooks are not supported. Save the export as .xlsx, "
                "CSV, or JSON and try again."
            )
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise HistoricalImportError(
            f"Unsupported file extension {path.suffix!r}. Supported formats: {supported}."
        )
    return value


def _read_source(
    path: Path,
    *,
    source_format: str,
    sheet_name: str | None,
    warnings_list: list[str],
) -> _SourceTable:
    if source_format in {"csv", "tsv"}:
        return _read_delimited(path, source_format=source_format, warnings_list=warnings_list)
    if source_format == "xlsx":
        return _read_xlsx(path, sheet_name=sheet_name, warnings_list=warnings_list)
    if source_format in {"json", "jsonl"}:
        return _read_json(path, source_format=source_format, warnings_list=warnings_list)
    raise HistoricalImportError(f"Unsupported source format {source_format!r}")


def _decode_text(data: bytes) -> tuple[str, str, bool]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding, encoding not in {"utf-8-sig"}
        except UnicodeDecodeError:
            continue
    raise HistoricalImportError(
        "Could not decode the export as UTF-8, Windows-1252, or Latin-1. Re-export it as UTF-8."
    )


def _read_delimited(
    path: Path,
    *,
    source_format: str,
    warnings_list: list[str],
) -> _SourceTable:
    data = path.read_bytes()
    if not data.strip():
        raise HistoricalImportError(f"Source export is empty: {path.name}")

    text, encoding, encoding_warning = _decode_text(data)
    if encoding_warning:
        warnings_list.append(
            f"source was decoded as {encoding}; re-export as UTF-8 to avoid character loss"
        )

    delimiter = "\t" if source_format == "tsv" else _sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        raw_headers = _trim_trailing_blank(next(reader))
    except StopIteration as exc:
        raise HistoricalImportError(f"Source export has no header row: {path.name}") from exc

    headers, header_warnings = _normalise_headers(raw_headers)
    warnings_list.extend(header_warnings)

    rows: list[_SourceRow] = []
    blank_rows = 0
    for values in reader:
        values = _trim_trailing_blank(values)
        if not values:
            blank_rows += 1
            continue
        raw = _row_dict(headers, values)
        row_warnings = ["extra_columns_present"] if "__extra_values__" in raw else []
        rows.append(
            _SourceRow(
                source_row=reader.line_num,
                raw=raw,
                flat=raw,
                warnings=row_warnings,
            )
        )

    return _SourceTable(
        headers=headers,
        rows=rows,
        source_format=source_format,
        encoding=encoding,
        delimiter=delimiter,
        warnings=warnings_list,
        blank_rows_skipped=blank_rows,
    )


def _read_xlsx(
    path: Path,
    *,
    sheet_name: str | None,
    warnings_list: list[str],
) -> _SourceTable:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise HistoricalImportError(
            "XLSX support requires the 'openpyxl' package. Install it with: pip install openpyxl"
        ) from exc

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - provider error is surfaced with context
        raise HistoricalImportError(f"Could not read XLSX export {path.name!r}: {exc}") from exc

    try:
        if sheet_name is not None:
            if sheet_name not in workbook.sheetnames:
                available = ", ".join(workbook.sheetnames)
                raise HistoricalImportError(
                    f"Worksheet {sheet_name!r} was not found in {path.name!r}. "
                    f"Available sheets: {available}."
                )
            worksheet = workbook[sheet_name]
        else:
            worksheet = workbook.active
            if worksheet is None:
                raise HistoricalImportError(f"Workbook has no active worksheet: {path.name!r}")
            if len(workbook.sheetnames) > 1:
                warnings_list.append(
                    f"workbook has {len(workbook.sheetnames)} sheets; imported the active "
                    f"sheet {worksheet.title!r} (use --sheet to choose another)"
                )

        headers: list[str] | None = None
        rows: list[_SourceRow] = []
        blank_rows = 0

        for row_number, values in enumerate(worksheet.iter_rows(values_only=True), start=1):
            if _is_blank(values):
                if headers is not None:
                    blank_rows += 1
                continue
            values = _trim_trailing_blank(values)
            if headers is None:
                headers, header_warnings = _normalise_headers(values)
                warnings_list.extend(header_warnings)
                continue

            raw: dict[str, Any] = {}
            for index, header in enumerate(headers):
                raw[header] = values[index] if index < len(values) else None
            row_warnings: list[str] = []
            if len(values) > len(headers):
                raw["__extra_values__"] = list(values[len(headers) :])
                row_warnings.append("extra_columns_present")
            rows.append(
                _SourceRow(
                    source_row=row_number,
                    raw=json_safe(raw),
                    flat=raw,
                    warnings=row_warnings,
                )
            )

        if headers is None:
            raise HistoricalImportError(f"Worksheet {worksheet.title!r} is empty")

        return _SourceTable(
            headers=headers,
            rows=rows,
            source_format="xlsx",
            sheet_name=worksheet.title,
            warnings=warnings_list,
            blank_rows_skipped=blank_rows,
        )
    finally:
        workbook.close()


def _read_json(
    path: Path,
    *,
    source_format: str,
    warnings_list: list[str],
) -> _SourceTable:
    data = path.read_bytes()
    if not data.strip():
        raise HistoricalImportError(f"Source export is empty: {path.name}")

    text, encoding, encoding_warning = _decode_text(data)
    if encoding_warning:
        warnings_list.append(
            f"source was decoded as {encoding}; re-export as UTF-8 to avoid character loss"
        )

    rows: list[_SourceRow] = []
    blank_rows = 0
    metadata: dict[str, Any] = {}

    if source_format == "jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                blank_rows += 1
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    _SourceRow(
                        source_row=line_number,
                        raw={"line": line},
                        error_code="invalid_json",
                        error_message=f"line {line_number}: {exc.msg}",
                    )
                )
                continue
            rows.append(_json_object_row(payload, line_number))
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HistoricalImportError(
                f"Invalid JSON in {path.name!r} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        if isinstance(payload, list):
            objects: list[Any] = payload
        elif isinstance(payload, dict):
            list_key = next(
                (key for key in _ROW_LIST_KEYS if isinstance(payload.get(key), list)),
                None,
            )
            if list_key is not None:
                objects = payload[list_key]
                metadata = {
                    key: json_safe(value)
                    for key, value in payload.items()
                    if key != list_key and _is_scalar(value)
                }
            elif _looks_like_row(payload):
                objects = [payload]
            else:
                raise HistoricalImportError(
                    "JSON export must be a list of post objects or an object containing "
                    f"one of these lists: {', '.join(_ROW_LIST_KEYS)}. "
                    "Use --column to map a different shape."
                )
        else:
            raise HistoricalImportError(
                "JSON export must be a list of post objects or an object containing a list "
                "of post objects."
            )

        for index, obj in enumerate(objects, start=1):
            rows.append(_json_object_row(obj, index))

    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.flat or row.raw:
            if key not in seen:
                seen.add(key)
                headers.append(key)

    return _SourceTable(
        headers=headers,
        rows=rows,
        source_format=source_format,
        encoding=encoding,
        source_metadata=metadata,
        warnings=warnings_list,
        blank_rows_skipped=blank_rows,
    )


def _json_object_row(obj: Any, source_row: int) -> _SourceRow:
    if not isinstance(obj, dict):
        return _SourceRow(
            source_row=source_row,
            raw={"value": json_safe(obj)},
            error_code="invalid_row",
            error_message="JSON row is not an object",
        )
    return _SourceRow(
        source_row=source_row,
        raw=obj,
        flat=_flatten_json_row(obj),
    )


def _flatten_json_row(obj: dict[str, Any]) -> dict[str, Any]:
    """Expose common nested X API fields without changing the raw payload."""
    flat: dict[str, Any] = {}
    for nested_key in ("legacy", "public_metrics", "note_tweet"):
        nested = obj.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                flat[key] = value

    user = obj.get("user")
    if isinstance(user, dict):
        for key in ("screen_name", "username", "name"):
            if key in user:
                flat[f"user_{key}"] = user[key]

    for key, value in obj.items():
        if key in flat and not is_present(value) and is_present(flat[key]):
            continue
        flat[key] = value

    note_tweet = obj.get("note_tweet")
    if isinstance(note_tweet, dict) and isinstance(note_tweet.get("text"), str):
        note_text = note_tweet["text"].strip()
        if note_text:
            flat["note_tweet_text"] = note_text
            flat["text"] = note_text

    return flat


def _looks_like_row(obj: dict[str, Any]) -> bool:
    flat = _flatten_json_row(obj)
    return any(normalise_key(key) in _ALL_ALIASES for key in flat)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_blank(values: Iterable[Any]) -> bool:
    return all(value is None or (isinstance(value, str) and not value.strip()) for value in values)


def _trim_trailing_blank(values: Iterable[Any]) -> list[Any]:
    items = list(values)
    while items and (items[-1] is None or (isinstance(items[-1], str) and not items[-1].strip())):
        items.pop()
    return items


def _sniff_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _normalise_headers(raw_headers: Iterable[Any]) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    warnings_list: list[str] = []
    for index, value in enumerate(raw_headers):
        name = _header_text(value)
        if not name:
            name = f"column_{index + 1}"
            warnings_list.append(f"blank header at column {index + 1}; named {name!r}")
        base = name
        count = counts.get(base, 0) + 1
        counts[base] = count
        if count > 1:
            name = f"{base}__{count}"
            warnings_list.append(f"duplicate header {base!r}; renamed {name!r}")
        headers.append(name)
    return headers, warnings_list


def _header_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _row_dict(headers: list[str], values: list[str]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for index, header in enumerate(headers):
        raw[header] = values[index] if index < len(values) else None
    if len(values) > len(headers):
        raw["__extra_values__"] = list(values[len(headers) :])
    return raw


# ---------------------------------------------------------------------------
# Column mapping and row validation
# ---------------------------------------------------------------------------
def _build_mapping(
    headers: list[str],
    overrides: Mapping[str, str],
) -> tuple[dict[str, str | None], list[str]]:
    for field_name in overrides:
        if field_name not in FIELD_ALIASES:
            supported = ", ".join(FIELD_ALIASES)
            raise HistoricalImportError(
                f"Unknown canonical field {field_name!r} in column_map. "
                f"Supported fields: {supported}."
            )

    normalised = [(header, normalise_key(header)) for header in headers]
    mapping: dict[str, str | None] = {}
    warnings_list: list[str] = []
    used: set[str] = set()

    for field_name, aliases in FIELD_ALIASES.items():
        if field_name in overrides:
            column = _resolve_override(headers, overrides[field_name], field_name)
            mapping[field_name] = column
            used.add(column)
            continue

        selected: str | None = None
        for alias in aliases:
            matches = [
                header for header, normalised_header in normalised if normalised_header == alias
            ]
            if matches:
                selected = matches[0]
                if len(matches) > 1:
                    warnings_list.append(
                        f"multiple columns match {field_name!r} ({', '.join(matches)}); "
                        f"using {selected!r} (override with column_map)"
                    )
                break
        mapping[field_name] = selected
        if selected is not None:
            used.add(selected)

    return mapping, warnings_list


def _resolve_override(headers: list[str], requested: str, field_name: str) -> str:
    name = requested.strip()
    if not name:
        raise HistoricalImportError(f"column_map for {field_name!r} is empty")
    if name in headers:
        return name
    normalised_requested = normalise_key(name)
    matches = [header for header in headers if normalise_key(header) == normalised_requested]
    if matches:
        return matches[0]
    available = ", ".join(repr(header) for header in headers)
    raise HistoricalImportError(
        f"Column {requested!r} for {field_name!r} was not found. Available columns: {available}."
    )


def _require_required_columns(mapping: dict[str, str | None], headers: list[str]) -> None:
    missing = [field for field in ("text", "created_at") if mapping.get(field) is None]
    if not missing:
        return
    available = ", ".join(repr(header) for header in headers)
    details = "; ".join(
        f"{field} aliases include {', '.join(FIELD_ALIASES[field][:4])}" for field in missing
    )
    raise HistoricalImportError(
        f"Could not map required field(s) {', '.join(missing)}. Detected columns: {available}. "
        f"{details}. Rename the export columns or pass --column {missing[0]}=<column>."
    )


def _validate_row(row: _SourceRow, context: _ValidationContext) -> _RowOutcome:
    if row.error_code is not None:
        return _quarantine(
            row,
            row.error_code,
            row.error_message or row.error_code,
            row.warnings,
        )

    values = row.flat or row.raw
    warnings_list = list(row.warnings)
    mapping = context.mapping
    normalised_row = {normalise_key(key): key for key in values}

    text_column = _row_column(values, normalised_row, "text", mapping)
    text_value = _get_value(values, text_column)
    text = as_text(text_value)
    if not text.strip():
        return _quarantine(
            row,
            "missing_text",
            f"text column {text_column!r} is empty",
            warnings_list,
        )
    text = text.strip()

    timestamp_column = _row_column(values, normalised_row, "created_at", mapping)
    timestamp_value = _get_value(values, timestamp_column)
    if timestamp_value is None or (
        isinstance(timestamp_value, str) and not timestamp_value.strip()
    ):
        return _quarantine(
            row,
            "missing_timestamp",
            f"timestamp column {timestamp_column!r} is empty",
            warnings_list,
        )
    try:
        parsed_timestamp = parse_timestamp(
            timestamp_value,
            assume_timezone=context.assume_timezone,
            allow_excel_serial=context.allow_excel_serial,
        )
    except (ValueError, OverflowError) as exc:
        return _quarantine(row, "invalid_timestamp", str(exc), warnings_list)
    if parsed_timestamp.assumed:
        warnings_list.append("timestamp_timezone_assumed")

    post_id_column = _row_column(values, normalised_row, "post_id", mapping)
    post_id, post_id_warnings = clean_post_id(
        _get_value(values, post_id_column),
        warn_numeric_precision=context.allow_excel_serial,
    )
    warnings_list.extend(post_id_warnings)
    url_column = _row_column(values, normalised_row, "url", mapping)
    if post_id is None and url_column is not None:
        derived = post_id_from_url(_get_value(values, url_column))
        if derived is not None:
            post_id = derived
            warnings_list.append("post_id_derived_from_url")

    counts: dict[str, int | None] = {}
    for field_name in _COUNT_FIELDS:
        column = _row_column(values, normalised_row, field_name, mapping)
        if column is None:
            counts[field_name] = None
            continue
        try:
            counts[field_name] = parse_count(
                _get_value(values, column),
                field_name=field_name,
            )
        except ValueError as exc:
            return _quarantine(row, "invalid_engagement", str(exc), warnings_list)

    url: str | None = None
    if url_column is not None:
        url, url_warning = normalise_url(_get_value(values, url_column))
        if url_warning is not None:
            warnings_list.append(f"url_{url_warning}")

    author_column = _row_column(values, normalised_row, "author", mapping)
    author, author_handle = normalise_author(_get_value(values, author_column))

    outside_window = False
    if context.window_start is not None and context.window_end is not None:
        outside_window = not (context.window_start <= parsed_timestamp.utc < context.window_end)
        if outside_window:
            if context.apply_date_window:
                return _RowOutcome(
                    quarantine=QuarantinedTweetRow(
                        source_row=row.source_row,
                        reason_code="outside_date_window",
                        reason=(
                            f"{parsed_timestamp.utc.isoformat()} is outside "
                            f"{context.window_start.isoformat()} to "
                            f"{context.window_end.isoformat()}"
                        ),
                        warnings=list(dict.fromkeys(warnings_list)),
                        raw=json_safe(row.raw),
                    ),
                    outside_window=True,
                )
            warnings_list.append("outside_date_window")

    record = RawTweetRecord(
        source=SOURCE_NAME,
        source_format=context.source_format,
        source_file=context.source_file,
        source_row=row.source_row,
        import_batch_id=context.batch_id,
        imported_at=context.imported_at,
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
        warnings=list(dict.fromkeys(warnings_list)),
        raw=json_safe(row.raw),
    )
    return _RowOutcome(record=record, outside_window=outside_window)


def _quarantine(
    row: _SourceRow,
    reason_code: str,
    reason: str,
    warnings_list: Iterable[str],
) -> _RowOutcome:
    return _RowOutcome(
        quarantine=QuarantinedTweetRow(
            source_row=row.source_row,
            reason_code=reason_code,
            reason=reason,
            warnings=list(dict.fromkeys(warnings_list)),
            raw=json_safe(row.raw),
        )
    )


def _get_value(values: Mapping[str, Any], column: str | None) -> Any:
    if column is None:
        return None
    return values.get(column)


def _row_column(
    values: Mapping[str, Any],
    normalised_row: Mapping[str, str],
    field_name: str,
    mapping: Mapping[str, str | None],
) -> str | None:
    """Pick the mapped column, falling back to other aliases present in this row.

    JSON exports from different X endpoints often use different field names
    (``note_tweet_text`` on one row, ``full_text`` on another).  A single global
    mapping would otherwise quarantine otherwise-valid heterogeneous rows.
    """
    selected = mapping.get(field_name)
    if selected is not None and _has_value(values, selected):
        return selected
    for alias in FIELD_ALIASES[field_name]:
        candidate = normalised_row.get(alias)
        if candidate is not None and _has_value(values, candidate):
            return candidate
    return selected


def _has_value(values: Mapping[str, Any], column: str) -> bool:
    if column not in values:
        return False
    return is_present(values[column])


# ---------------------------------------------------------------------------
# Configuration helpers and batch writing
# ---------------------------------------------------------------------------
def _coerce_timezone(value: str | tzinfo | None, settings: Settings) -> tzinfo:
    if value is None:
        return settings.tzinfo
    if isinstance(value, str):
        try:
            return ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise HistoricalImportError(
                f"Unknown IANA timezone {value!r}; use a value such as UTC or America/New_York."
            ) from exc
    if isinstance(value, tzinfo):
        return value
    raise HistoricalImportError(f"Invalid timezone value {value!r}")


def _resolve_window(
    settings: Settings,
    *,
    start_date: date | None,
    end_date: date | None,
) -> tuple[datetime, datetime, dict[str, str]]:
    start = start_date or settings.effective_start_date
    end = end_date or settings.effective_end_date
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    if start > end:
        raise HistoricalImportError(
            f"start_date {start.isoformat()} must be on or before end_date {end.isoformat()}"
        )

    timezone = settings.tzinfo
    window_start = datetime.combine(start, time.min, tzinfo=timezone)
    window_end = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone)
    return (
        window_start,
        window_end,
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timezone": timezone.key,
        },
    )


def _resolve_output_directory(
    settings: Settings,
    *,
    output_dir: str | Path | None,
    batch_id: str,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (settings.raw_data_dir / "historical_tweets" / settings.asset_slug / batch_id).resolve()


__all__ = [
    "FIELD_ALIASES",
    "HistoricalImportError",
    "ImportResult",
    "ImportSummary",
    "SUPPORTED_FORMATS",
    "import_historical_tweets",
]
