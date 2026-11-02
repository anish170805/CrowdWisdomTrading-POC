"""Structured logging with run identifiers and secret redaction.

Console logs are human-readable by default; the rotating file log is always
newline-delimited JSON so runs can be audited later.  Every handler runs a
redaction filter seeded with the configured secrets, and extra fields whose
names look sensitive are replaced before formatting.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

if TYPE_CHECKING:
    from src.config import Settings

_RUN_ID: ContextVar[str | None] = ContextVar("crowdxbot_run_id", default=None)

_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|token|secret|password|passwd|"
    r"authorization|credential)",
    re.IGNORECASE,
)
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|token|secret|password|passwd|authorization|"
    r"credential)\b(\s*[:=]\s*)([^\s,;\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]+")
_URL_CREDENTIAL_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/@\s]+:)[^@\s]+(@)")

_RESERVED_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "asctime",
        "message",
    }
)

_NOISY_LOGGERS = (
    "urllib3",
    "httpx",
    "httpcore",
    "transformers",
    "filelock",
    "matplotlib",
    "asyncio",
)


def new_run_id(prefix: str = "run") -> str:
    """Create a sortable, unique run identifier."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def current_run_id() -> str:
    return _RUN_ID.get() or "-"


def set_run_id(run_id: str) -> None:
    _RUN_ID.set(run_id)


def reset_run_id() -> None:
    _RUN_ID.set(None)


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """Bind a run identifier for the duration of a block."""
    identifier = run_id or new_run_id()
    token = _RUN_ID.set(identifier)
    try:
        yield identifier
    finally:
        _RUN_ID.reset(token)


class RunContextFilter(logging.Filter):
    """Attach the current run identifier to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "run_id", None):
            record.run_id = current_run_id()
        return True


class SecretRedactionFilter(logging.Filter):
    """Remove configured secret values and common credential patterns."""

    def __init__(self, secrets: tuple[str, ...] | list[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(
            sorted({secret for secret in secrets if secret}, key=len, reverse=True)
        )

    def _redact_text(self, text: str) -> str:
        redacted = text
        for secret in self._secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        redacted = _KEY_VALUE_RE.sub(r"\1\2[REDACTED]", redacted)
        redacted = _BEARER_RE.sub("Bearer [REDACTED]", redacted)
        redacted = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]\2", redacted)
        return redacted

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, SecretStr):
            return "[REDACTED]"
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, tuple):
            return tuple(self._redact_value(item) for item in value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, set):
            return {self._redact_value(item) for item in value}
        if isinstance(value, dict):
            return {
                key: (
                    "[REDACTED]"
                    if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key)
                    else self._redact_value(item)
                )
                for key, item in value.items()
            }
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        # Render the message before redacting it.  Redacting the format string
        # and arguments independently can break ``%`` placeholders (for
        # example ``token=%s`` would become ``token=[REDACTED]`` while the
        # argument tuple still had one item).
        if isinstance(record.msg, SecretStr):
            record.msg = "[REDACTED]"
            record.args = ()
        elif isinstance(record.msg, str):
            if record.args:
                try:
                    message = record.getMessage()
                except (TypeError, ValueError):
                    message = str(record.msg)
                record.msg = self._redact_text(message)
                record.args = ()
            else:
                record.msg = self._redact_text(record.msg)
        else:
            record.msg = self._redact_value(record.msg)
            record.args = ()

        for key, value in list(record.__dict__.items()):
            if key in _RESERVED_RECORD_ATTRS or key.startswith("_"):
                continue
            if _SENSITIVE_KEY_RE.search(key):
                record.__dict__[key] = "[REDACTED]"
            else:
                record.__dict__[key] = self._redact_value(value)

        return True


class SecretAwareFormatter(logging.Formatter):
    """Base formatter that also redacts exception text."""

    def __init__(
        self, *args: Any, redaction: SecretRedactionFilter | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._redaction = redaction or SecretRedactionFilter()

    def formatException(self, ei) -> str:  # type: ignore[no-untyped-def]
        return self._redaction._redact_text(super().formatException(ei))

    def formatStack(self, stack_info: str) -> str:
        return self._redaction._redact_text(super().formatStack(stack_info))

    def format(self, record: logging.LogRecord) -> str:
        # Another handler may have populated exc_text before this formatter
        # ran; clear it so exception text is always re-rendered and redacted.
        record.exc_text = None
        return super().format(record)


class ConsoleFormatter(SecretAwareFormatter):
    """Readable UTC console format."""

    converter = staticmethod(time.gmtime)

    def __init__(self, redaction: SecretRedactionFilter | None = None) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s [%(run_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            defaults={"run_id": "-"},
            redaction=redaction,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return "[REDACTED]"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class JsonFormatter(SecretAwareFormatter):
    """Newline-delimited JSON formatter for auditing and machine parsing."""

    def __init__(
        self,
        service: str | None = None,
        redaction: SecretRedactionFilter | None = None,
    ) -> None:
        super().__init__(redaction=redaction)
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None) or current_run_id(),
            "service": self.service,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key.startswith("_") or key in payload:
                continue
            payload[key] = _json_safe(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    settings: Settings | None = None,
    *,
    level: str | int | None = None,
    log_format: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure console and rotating-file logging.

    Calling this repeatedly is safe; the existing managed handlers are reused
    unless ``force=True``.  ``settings`` defaults to :func:`src.config.get_settings`.
    """
    if settings is None:
        from src.config import get_settings

        settings = get_settings()

    root = logging.getLogger()
    tagged_handlers = [
        handler for handler in root.handlers if getattr(handler, "_crowdxbot_handler", False)
    ]

    resolved_level: str | int
    if isinstance(level, int):
        resolved_level = level
    else:
        resolved_level = str(level or settings.log_level.value).upper()

    if tagged_handlers and not force:
        root.setLevel(resolved_level)
        for handler in tagged_handlers:
            handler.setLevel(resolved_level)
        return logging.getLogger(settings.app_name)

    for handler in tagged_handlers:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(resolved_level)

    redaction = SecretRedactionFilter(settings.secret_values())
    context = RunContextFilter()

    format_value = getattr(
        log_format or settings.log_format, "value", log_format or settings.log_format
    )
    format_value = str(format_value).lower()

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(resolved_level)
    if format_value == "json":
        console_handler.setFormatter(JsonFormatter(service=settings.app_name, redaction=redaction))
    else:
        console_handler.setFormatter(ConsoleFormatter(redaction=redaction))
    console_handler.addFilter(context)
    console_handler.addFilter(redaction)
    console_handler._crowdxbot_handler = True  # type: ignore[attr-defined]
    root.addHandler(console_handler)

    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.log_file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "File logging disabled: could not open %s (%s)",
            settings.log_file_path,
            exc.strerror or exc,
        )
    else:
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(JsonFormatter(service=settings.app_name, redaction=redaction))
        file_handler.addFilter(context)
        file_handler.addFilter(redaction)
        file_handler._crowdxbot_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))

    logging.captureWarnings(True)
    root._crowdxbot_configured = True  # type: ignore[attr-defined]
    return logging.getLogger(settings.app_name)


def get_logger(name: str = "crowdxbot") -> logging.Logger:
    return logging.getLogger(name)


__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "SecretRedactionFilter",
    "configure_logging",
    "current_run_id",
    "get_logger",
    "new_run_id",
    "reset_run_id",
    "run_context",
    "set_run_id",
]
