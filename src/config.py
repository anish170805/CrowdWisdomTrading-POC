"""Typed, validated application configuration.

All modules must obtain configuration through :func:`get_settings` (or the
non-cached :func:`load_settings` in tests).  Nothing outside this module should
read environment variables directly.

Secret values are held in :class:`pydantic.SecretStr` fields (except the
database URL, whose password is never logged and is hidden by ``repr``).  The
public helpers in this module are designed to be safe to log.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
MAX_FORECAST_HORIZON = 30

_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|token|secret|password|passwd|credential)",
    re.IGNORECASE,
)


class ConfigurationError(RuntimeError):
    """Raised when settings are missing or invalid.

    ``issues`` contains actionable, secret-free messages suitable for display
    to an operator.
    """

    def __init__(self, issues: Iterable[str]) -> None:
        cleaned = tuple(dict.fromkeys(str(issue).strip() for issue in issues if str(issue).strip()))
        self.issues = cleaned or ("Unknown configuration error.",)
        super().__init__(
            "Invalid configuration:\n" + "\n".join(f"  - {issue}" for issue in self.issues)
        )


class AssetType(StrEnum):
    """Supported asset classes."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    INDEX = "index"


class DataProvider(StrEnum):
    """Price data providers.  Add adapters here as they are implemented."""

    YFINANCE = "yfinance"


class ApifyQueryMode(StrEnum):
    """How the configured asset is turned into X search terms."""

    CASHTAG_AND_SYMBOL = "cashtag_and_symbol"
    CASHTAG_ONLY = "cashtag_only"
    SYMBOL_ONLY = "symbol_only"


class ModelType(StrEnum):
    """Forecasting model families."""

    XGBOOST = "xgboost"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


def _lower_stripped(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _upper_stripped(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"[,;\n]+", value) if part.strip())


def _strip_required(value: Any, *, label: str) -> Any:
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


class Settings(BaseSettings):
    """Single source of truth for application configuration.

    Environment variable names are the upper-case field names, for example
    ``ASSET_SYMBOL``, ``APIFY_API_TOKEN`` and ``DATABASE_URL``.  Values are
    loaded from the process environment and then from ``.env`` (environment
    wins).  Empty environment values are treated as unset.
    """

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Application and asset scope
    # ------------------------------------------------------------------
    app_name: str = Field(
        default="crowdxbot",
        min_length=1,
        max_length=64,
        description="Application name used in logs and run identifiers.",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment: development, test, or production.",
    )
    asset_symbol: str = Field(
        ...,
        description="Ticker/instrument symbol to model, e.g. AAPL, BTC-USD, EURUSD.",
    )
    asset_type: AssetType = Field(
        ...,
        description="Asset class: equity, crypto, fx, commodity, or index.",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA market timezone used for daily bucketing, e.g. America/New_York.",
    )
    history_days: int = Field(
        default=180,
        ge=1,
        le=3650,
        description="Historical window in days when DATA_START_DATE is not set.",
    )
    data_start_date: date | None = Field(
        default=None,
        description="Optional inclusive start date (ISO YYYY-MM-DD). Overrides HISTORY_DAYS.",
    )
    data_end_date: date | None = Field(
        default=None,
        description="Optional inclusive end date (ISO YYYY-MM-DD). Defaults to today.",
    )
    data_provider: DataProvider = Field(
        ...,
        description="Market-data provider adapter.",
    )

    # ------------------------------------------------------------------
    # X / Apify ingestion
    # ------------------------------------------------------------------
    apify_actor_id: str = Field(
        ...,
        description="Verified Apify actor ID for X/Twitter scraping, e.g. user/actor-name.",
    )
    apify_api_token: SecretStr = Field(
        ...,
        description="Apify API token. Never logged or committed.",
    )
    apify_max_items: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Maximum number of posts to request per Apify run.",
    )
    apify_run_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Maximum time to wait for an Apify run.",
    )
    apify_poll_interval_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Polling interval while waiting for an Apify run.",
    )
    apify_input_template: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional inline JSON object template for the actor input. Placeholders: "
            "{query}, {query_terms}, {max_items}, {asset_symbol}, {start_date}, "
            "{end_date}, {language}."
        ),
    )
    apify_input_template_path: Path | None = Field(
        default=None,
        description="Path to a JSON file containing the actor input template.",
    )
    apify_input_schema_path: Path | None = Field(
        default=None,
        description="Path to the actor's verified INPUT_SCHEMA.json for local validation.",
    )
    apify_output_field_map: dict[str, str] = Field(
        default_factory=dict,
        description="Canonical record field to actor dataset JSON path overrides.",
    )
    apify_query_mode: ApifyQueryMode = Field(
        default=ApifyQueryMode.CASHTAG_AND_SYMBOL,
        description="Asset query strategy: cashtag_and_symbol, cashtag_only, or symbol_only.",
    )
    apify_query_terms: str = Field(
        default="",
        description="Comma-separated extra query terms (aliases) for disambiguation.",
    )
    apify_query_exclude_terms: str = Field(
        default="",
        description="Comma-separated terms excluded from the X query.",
    )
    apify_query_language: str | None = Field(
        default=None,
        description="Optional language code passed to the actor template as {language}.",
    )

    # ------------------------------------------------------------------
    # News retrieval
    # ------------------------------------------------------------------
    tavily_api_key: SecretStr = Field(
        ...,
        description="Tavily API key. Never logged or committed.",
    )
    tavily_max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum Tavily news results per query.",
    )
    tavily_search_depth: str = Field(
        default="basic",
        description="Tavily search depth: basic or advanced.",
    )
    exa_api_key: SecretStr | None = Field(
        default=None,
        description="Optional Exa API key. Leave unset to disable Exa.",
    )
    exa_max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum Exa news results per query when enabled.",
    )
    news_lookback_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Recent-news window in days for report context.",
    )
    http_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default timeout for external HTTP calls.",
    )
    http_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retries for transient external API failures.",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///data/processed/crowdxbot.db",
        description="SQLAlchemy database URL (SQLite or PostgreSQL).",
    )
    sqlite_busy_timeout_ms: int = Field(
        default=5000,
        ge=0,
        le=120_000,
        description="SQLite busy timeout in milliseconds.",
    )

    # ------------------------------------------------------------------
    # Model and validation settings
    # ------------------------------------------------------------------
    model_type: ModelType = Field(
        default=ModelType.XGBOOST,
        description="Model family used for per-horizon training.",
    )
    model_horizons: str = Field(
        default="1,2,3,4,5",
        description="Comma-separated forecast horizons in trading days (n+1 ... n+k).",
    )
    model_random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible model training.",
    )
    model_n_estimators: int = Field(
        default=60,
        ge=1,
        le=10_000,
        description="Number of boosting rounds.",
    )
    model_max_depth: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Maximum tree depth.",
    )
    model_learning_rate: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Boosting learning rate.",
    )
    walk_forward_initial_train_days: int = Field(
        default=120,
        ge=30,
        le=3650,
        description="Minimum initial training window for walk-forward validation.",
    )
    walk_forward_test_days: int = Field(
        default=5,
        ge=1,
        le=365,
        description="Number of days in each walk-forward test block.",
    )
    walk_forward_gap_days: int = Field(
        default=5,
        ge=0,
        le=365,
        description="Embargo between train and test blocks; at least the largest horizon.",
    )
    min_train_samples: int = Field(
        default=60,
        ge=10,
        description="Minimum rows required before a model is considered trainable.",
    )
    transaction_cost_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=10_000.0,
        description="Round-trip transaction cost in basis points for strategy metrics.",
    )
    risk_free_rate: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Annual risk-free rate used for Sharpe/Sortino calculations.",
    )

    # ------------------------------------------------------------------
    # Sentiment model
    # ------------------------------------------------------------------
    sentiment_model_name: str = Field(
        default="ProsusAI/finbert",
        min_length=1,
        description="Pinned Hugging Face sentiment model identifier.",
    )
    sentiment_model_revision: str = Field(
        default="main",
        min_length=1,
        description="Model revision/branch/commit recorded with scores.",
    )
    sentiment_batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for sentiment inference.",
    )
    sentiment_max_length: int = Field(
        default=128,
        ge=8,
        le=512,
        description="Maximum token length for sentiment inference.",
    )
    sentiment_device: str = Field(
        default="auto",
        description="Sentiment inference device: auto, cpu, cuda, or mps.",
    )

    # ------------------------------------------------------------------
    # Output paths (relative values are resolved against the project root)
    # ------------------------------------------------------------------
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Immutable raw source payloads and import batches.",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Canonical/processed data and local database files.",
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="Trained model artifacts.",
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Generated reports, charts, and evaluation outputs.",
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        description="Prediction snapshots, manifests, and run artifacts.",
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Rotating structured log files.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Console and file log level.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Console log format: console (human) or json (structured).",
    )
    log_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        description="Maximum size of a log file before rotation.",
    )
    log_backup_count: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of rotated log files to keep.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("asset_symbol", mode="before")
    @classmethod
    def _validate_asset_symbol(cls, value: Any) -> Any:
        value = _strip_required(value, label="ASSET_SYMBOL")
        if isinstance(value, str):
            if len(value) > 32:
                raise ValueError("must be at most 32 characters")
            if any(char.isspace() for char in value):
                raise ValueError("must not contain whitespace")
        return value

    @field_validator("apify_actor_id", mode="before")
    @classmethod
    def _validate_actor_id(cls, value: Any) -> Any:
        value = _strip_required(value, label="APIFY_ACTOR_ID")
        if isinstance(value, str) and any(char.isspace() for char in value):
            raise ValueError("must not contain whitespace")
        return value

    @field_validator("apify_api_token", "tavily_api_key", mode="before")
    @classmethod
    def _validate_required_secret(cls, value: Any) -> Any:
        return _strip_required(value, label="secret value")

    @field_validator("exa_api_key", mode="before")
    @classmethod
    def _normalise_optional_secret(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"none", "null", "unset"}:
                return None
        return value

    @field_validator("data_start_date", "data_end_date", mode="before")
    @classmethod
    def _blank_date_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("apify_input_template", mode="before")
    @classmethod
    def _validate_apify_input_template(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                return None
            raise ValueError(
                "must be a JSON object; set APIFY_INPUT_TEMPLATE to valid JSON or use "
                "APIFY_INPUT_TEMPLATE_PATH"
            )
        if not isinstance(value, dict):
            raise ValueError("must be a JSON object")
        return value

    @field_validator("apify_input_template_path", "apify_input_schema_path", mode="before")
    @classmethod
    def _resolve_optional_path(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            if not value.strip():
                return None
            path = Path(value.strip())
        else:
            return value
        path = path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @field_validator("apify_output_field_map", mode="before")
    @classmethod
    def _validate_output_field_map(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            if not value.strip():
                return {}
            raise ValueError(
                "must be a JSON object mapping canonical fields to actor dataset paths"
            )
        if not isinstance(value, dict):
            raise ValueError("must be a JSON object")
        cleaned: dict[str, str] = {}
        for raw_field, raw_path in value.items():
            field_name = str(raw_field).strip()
            path = str(raw_path).strip()
            if not field_name or not path:
                raise ValueError("keys and values must be non-empty strings")
            cleaned[field_name] = path
        return cleaned

    @field_validator("apify_query_language", mode="before")
    @classmethod
    def _normalise_query_language(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        language = value.strip().lower()
        if not language or language in {"none", "null", "unset"}:
            return None
        if not re.fullmatch(r"[a-z]{2,3}", language):
            raise ValueError("must be a two- or three-letter language code, e.g. en")
        return language

    @field_validator("timezone", mode="before")
    @classmethod
    def _validate_timezone(cls, value: Any) -> Any:
        value = _strip_required(value, label="TIMEZONE")
        if not isinstance(value, str):
            return value
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"unknown IANA timezone {value!r}; use a value such as UTC or "
                "America/New_York (Windows may require the tzdata package)"
            ) from exc
        return value

    @field_validator("sentiment_device", mode="before")
    @classmethod
    def _validate_sentiment_device(cls, value: Any) -> Any:
        value = _lower_stripped(value)
        if value not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("must be one of: auto, cpu, cuda, mps")
        return value

    @field_validator("tavily_search_depth", mode="before")
    @classmethod
    def _validate_tavily_depth(cls, value: Any) -> Any:
        value = _lower_stripped(value)
        if value not in {"basic", "advanced"}:
            raise ValueError("must be one of: basic, advanced")
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def _validate_database_url(cls, value: Any) -> Any:
        value = _strip_required(value, label="DATABASE_URL")
        if not isinstance(value, str):
            return value
        try:
            url = make_url(value)
        except ArgumentError as exc:
            raise ValueError("is not a valid SQLAlchemy database URL") from exc

        backend = url.get_backend_name()
        if backend not in {"sqlite", "postgresql"}:
            raise ValueError("must use a SQLite or PostgreSQL backend")

        # Resolve relative SQLite paths against the project root so commands
        # behave consistently regardless of the shell's working directory.
        if backend == "sqlite" and url.database and url.database != ":memory:":
            db_path = Path(url.database).expanduser()
            if not db_path.is_absolute():
                db_path = (PROJECT_ROOT / db_path).resolve()
            url = url.set(database=db_path.as_posix())
        return url.render_as_string(hide_password=False)

    @field_validator("model_horizons", mode="before")
    @classmethod
    def _validate_model_horizons(cls, value: Any) -> str:
        if isinstance(value, str):
            raw = value.strip()
            parts = [part for part in re.split(r"[,;\s]+", raw) if part]
        elif isinstance(value, (list, tuple, set)):
            parts = [str(part) for part in value]
        else:
            raise ValueError("must be a comma-separated list of positive integers")

        if not parts:
            raise ValueError("must contain at least one horizon")

        horizons: list[int] = []
        for part in parts:
            try:
                horizon = int(part)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid horizon {part!r}; use integers such as 1,2,3,4,5"
                ) from exc
            if horizon < 1 or horizon > MAX_FORECAST_HORIZON:
                raise ValueError(f"horizon {horizon} must be between 1 and {MAX_FORECAST_HORIZON}")
            if horizon in horizons:
                raise ValueError(f"duplicate horizon {horizon}")
            horizons.append(horizon)

        return ",".join(str(horizon) for horizon in sorted(horizons))

    @field_validator(
        "raw_data_dir",
        "processed_data_dir",
        "models_dir",
        "reports_dir",
        "artifacts_dir",
        "logs_dir",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        value = value.expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        return value.resolve()

    @field_validator(
        "environment",
        "asset_type",
        "data_provider",
        "model_type",
        "log_format",
        "apify_query_mode",
        mode="before",
    )
    @classmethod
    def _normalise_enum_values(cls, value: Any) -> Any:
        return _lower_stripped(value)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: Any) -> Any:
        return _upper_stripped(value)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> Settings:
        if self.apify_input_template is not None and self.apify_input_template_path is not None:
            raise ValueError("set only one of APIFY_INPUT_TEMPLATE and APIFY_INPUT_TEMPLATE_PATH")

        if (
            self.data_start_date is not None
            and self.data_end_date is not None
            and self.data_start_date > self.data_end_date
        ):
            raise ValueError("DATA_START_DATE must be on or before DATA_END_DATE")

        if self.effective_start_date > self.effective_end_date:
            raise ValueError(
                "DATA_START_DATE must be on or before DATA_END_DATE (or today when "
                "DATA_END_DATE is unset)"
            )

        largest_horizon = max(self.horizons)
        if self.walk_forward_gap_days < largest_horizon:
            raise ValueError(
                f"WALK_FORWARD_GAP_DAYS ({self.walk_forward_gap_days}) must be at least the "
                f"largest model horizon ({largest_horizon}) to prevent target leakage"
            )

        return self

    # ------------------------------------------------------------------
    # Derived, safe-to-use values
    # ------------------------------------------------------------------
    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.model_horizons.split(","))

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def effective_end_date(self) -> date:
        """Forecast/data window end, defaulting to today in the market timezone."""
        if self.data_end_date is not None:
            return self.data_end_date
        return datetime.now(tz=self.tzinfo).date()

    @property
    def effective_start_date(self) -> date:
        """Forecast/data window start."""
        if self.data_start_date is not None:
            return self.data_start_date
        return self.effective_end_date - timedelta(days=self.history_days)

    @property
    def date_range(self) -> tuple[date, date]:
        return self.effective_start_date, self.effective_end_date

    @property
    def asset_slug(self) -> str:
        """Filesystem-safe representation of the asset symbol."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.asset_symbol).strip("_")
        return slug.upper() or "ASSET"

    @property
    def apify_query_term_list(self) -> tuple[str, ...]:
        """Extra X query terms (aliases) configured for disambiguation."""
        return _split_csv(self.apify_query_terms)

    @property
    def apify_query_exclusion_list(self) -> tuple[str, ...]:
        """Terms excluded from the generated X query."""
        return _split_csv(self.apify_query_exclude_terms)

    @property
    def apify_input_template_source(self) -> str | None:
        """Where the actor input template comes from, if configured."""
        if self.apify_input_template is not None:
            return "inline"
        if self.apify_input_template_path is not None:
            return "file"
        return None

    @property
    def database_backend(self) -> str:
        return make_url(self.database_url).get_backend_name()

    @property
    def sqlite_path(self) -> Path | None:
        url = make_url(self.database_url)
        if url.get_backend_name() == "sqlite" and url.database not in {None, "", ":memory:"}:
            return Path(url.database)
        return None

    @property
    def log_file_path(self) -> Path:
        return self.logs_dir / "app.log"

    @property
    def output_directories(self) -> tuple[Path, ...]:
        paths = (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.reports_dir,
            self.artifacts_dir,
            self.logs_dir,
        )
        return tuple(dict.fromkeys(paths))

    @property
    def safe_database_url(self) -> str:
        """Database URL with any password removed, safe for display."""
        try:
            rendered = make_url(self.database_url).render_as_string(hide_password=True)
        except ArgumentError:
            return "<invalid database URL>"
        # SQLAlchemy percent-encodes the colon in Windows drive letters; make
        # the displayed URL readable without changing the stored value.
        return unquote(rendered)

    def secret_values(self) -> tuple[str, ...]:
        """Return configured secret strings for log redaction.

        Includes API keys/tokens, a database password, and a conservative
        environment scan.  Never log the return value.
        """
        values: list[str] = []
        for field_name in ("apify_api_token", "tavily_api_key", "exa_api_key"):
            secret = getattr(self, field_name)
            if isinstance(secret, SecretStr):
                value = secret.get_secret_value()
                if value:
                    values.append(value)

        try:
            password = make_url(self.database_url).password
        except ArgumentError:
            password = None
        if password:
            values.append(password)

        for name, value in os.environ.items():
            if value and len(value) >= 4 and _SENSITIVE_ENV_NAME_RE.search(name):
                values.append(value)

        return tuple(sorted(set(values), key=len, reverse=True))

    def ensure_directories(self) -> tuple[Path, ...]:
        """Create all configured output directories (idempotent)."""
        directories = list(self.output_directories)
        if self.sqlite_path is not None:
            directories.append(self.sqlite_path.parent)

        created: list[Path] = []
        for directory in dict.fromkeys(directories):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(
                    [
                        f"Could not create output directory {directory}: "
                        f"{exc.strerror or exc}. Check permissions and path settings."
                    ]
                ) from exc
            created.append(directory)
        return tuple(created)

    def safe_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary that never contains secrets."""
        start, end = self.date_range
        return {
            "app_name": self.app_name,
            "environment": self.environment.value,
            "asset_symbol": self.asset_symbol,
            "asset_type": self.asset_type.value,
            "timezone": self.timezone,
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "date_window_days": (end - start).days,
            "history_days": self.history_days,
            "data_provider": self.data_provider.value,
            "apify_actor_id": self.apify_actor_id,
            "apify": {
                "actor_id": self.apify_actor_id,
                "max_items": self.apify_max_items,
                "run_timeout_seconds": self.apify_run_timeout_seconds,
                "poll_interval_seconds": self.apify_poll_interval_seconds,
                "query_mode": self.apify_query_mode.value,
                "query_terms": list(self.apify_query_term_list),
                "query_exclude_terms": list(self.apify_query_exclusion_list),
                "query_language": self.apify_query_language,
                "input_template_source": self.apify_input_template_source,
                "input_template_path": (
                    str(self.apify_input_template_path)
                    if self.apify_input_template_path is not None
                    else None
                ),
                "input_schema_path": (
                    str(self.apify_input_schema_path)
                    if self.apify_input_schema_path is not None
                    else None
                ),
                "output_field_map": dict(self.apify_output_field_map),
            },
            "credentials": {
                "apify_api_token": "set" if self.apify_api_token else "not set",
                "tavily_api_key": "set" if self.tavily_api_key else "not set",
                "exa_api_key": "set" if self.exa_api_key else "not set",
            },
            "database_url": self.safe_database_url,
            "database_backend": self.database_backend,
            "model_type": self.model_type.value,
            "model_horizons": list(self.horizons),
            "model_random_seed": self.model_random_seed,
            "walk_forward": {
                "initial_train_days": self.walk_forward_initial_train_days,
                "test_days": self.walk_forward_test_days,
                "gap_days": self.walk_forward_gap_days,
            },
            "paths": {
                "raw_data": str(self.raw_data_dir),
                "processed_data": str(self.processed_data_dir),
                "models": str(self.models_dir),
                "reports": str(self.reports_dir),
                "artifacts": str(self.artifacts_dir),
                "logs": str(self.logs_dir),
                "log_file": str(self.log_file_path),
            },
            "logging": {
                "level": self.log_level.value,
                "format": self.log_format.value,
            },
        }

    def __repr_args__(self):  # type: ignore[override]
        """Keep secrets and the raw database URL out of repr/model displays."""
        for name, value in super().__repr_args__():
            if name == "database_url":
                yield name, self.safe_database_url
            elif isinstance(value, SecretStr):
                yield name, "**********"
            else:
                yield name, value


def _build_configuration_error(exc: ValidationError) -> ConfigurationError:
    issues: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        loc = tuple(str(part) for part in error.get("loc", ()) if part != "__root__")
        env_name = "_".join(loc).upper() if loc else "configuration"
        error_type = str(error.get("type", "value_error"))
        message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")

        if error_type == "missing":
            issues.append(f"{env_name} is required. Set it in .env or the environment.")
        else:
            issues.append(f"{env_name}: {message}")

    return ConfigurationError(issues)


def load_settings(**overrides: Any) -> Settings:
    """Build and validate settings without caching.

    Pass ``_env_file=None`` to ignore the project ``.env`` file (used by tests).
    Raises :class:`ConfigurationError` with actionable, secret-free messages.
    """
    if "_env_file" not in overrides:
        overrides["_env_file"] = os.environ.get("CROWDXBOT_ENV_FILE") or str(DEFAULT_ENV_FILE)
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise _build_configuration_error(exc) from None
    except SettingsError:
        raise ConfigurationError(
            ["Could not parse the .env file. Check it for invalid syntax or unsupported values."]
        ) from None
    except OSError:
        raise ConfigurationError(
            ["Could not read the .env file. Check that it exists and is readable."]
        ) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated, process-wide settings object.

    The result is cached; environment changes require ``reload_settings()``
    (normally only needed in tests).
    """
    return load_settings()


def reload_settings() -> Settings:
    """Clear the cache and reload settings from the environment."""
    get_settings.cache_clear()
    return get_settings()


__all__ = [
    "ApifyQueryMode",
    "AssetType",
    "ConfigurationError",
    "DataProvider",
    "Environment",
    "LogFormat",
    "LogLevel",
    "ModelType",
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "load_settings",
    "reload_settings",
]
