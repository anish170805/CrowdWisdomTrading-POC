# crowdxbot — X sentiment to asset price prediction

A configuration-driven Python project for turning X/Twitter sentiment, market
data, and optional news context into a reproducible asset-price outlook.

**Current status: Modules 0, 1, and 2 are implemented.** Module 0 provides the
validated configuration object; Module 1 imports explicit historical X exports;
Module 2 collects live X posts through a configurable Apify actor. Both ingestion
paths produce the same raw-record contract with provenance and row-level
quarantine. Later modules add storage, scoring, modelling, validation, charts,
and reporting behind the same settings object.

Nothing about the asset, provider, or credentials is hard-coded. The application
loads one validated `Settings` object from the environment and the project
`.env`, and all later modules must import that object rather than reading
environment variables directly.

## How it works

The pipeline runs 13 sequential stages. Each stage is idempotent: unchanged inputs
reuse cached outputs; changed inputs trigger a re-run.

```
Stage 0  Configuration directories            (create output dirs, validate settings)
Stage 1  Historical tweet import              (optional CSV/JSON export; skipped if none)
Stage 2  Apify X ingestion                    (live posts via Apify actor)
Stage 3  Canonical storage                    (deduplicate tweets into SQLite DB)
Stage 4  Price ingestion                      (daily OHLCV bars via yfinance)
Stage 5  News retrieval                       (context only; not a model feature)
Stage 6  Sentiment scoring                    (FinBERT per-post scores)
Stage 7  Daily sentiment aggregation          (per-day buckets with lags)
Stage 8  Feature engineering / dataset        (price features + sentiment lags + targets)
Stage 9  Model training                       (XGBoost per horizon; GridSearchCV tuned)
Stage 10 Walk-forward validation              (expanding window; Sharpe/Sortino/Max DD)
Stage 11 Latest prediction                    (next 1–5 day close forecasts)
Stage 12 Visualizations                       (price forecast chart)
Stage 13 Reports                              (evaluation.md + outlook.md)
```

## Project layout

```
.
├── src/
│   ├── config.py            # typed settings, validation, safe summaries
│   ├── logging_config.py    # structured logs, run IDs, secret redaction
│   ├── main.py              # single CLI entry point
│   ├── pipeline.py          # Pipeline orchestration (stages 0–13)
│   ├── ingestion/
│   │   ├── apify_x.py            # live Apify collection, query, normalisation
│   │   ├── field_parsing.py      # shared timestamp/ID/count/author helpers
│   │   ├── historical_tweets.py  # CSV/XLSX/JSON import, mapping, quarantine
│   │   └── records.py            # shared raw-record and sink contracts
│   ├── features/            # Price features + dataset builder
│   ├── models/              # XGBoost training, artifacts, preprocessing
│   ├── normalization/       # Tweet + price bar normalization
│   ├── prediction/          # Latest prediction generation
│   ├── reporting/           # evaluation.md + outlook.md reports
│   ├── schemas/             # Pydantic schemas (tweets, prices, quarantine)
│   ├── sentiment/           # FinBERT scorer
│   ├── storage/             # SQLite engine, repository, store
│   └── validation/          # Leakage checks, walk-forward folds
├── tests/                   # unit and contract tests
├── data/raw/                # immutable source payloads (git-ignored)
├── data/processed/          # canonical data and local database (git-ignored)
├── models/                  # trained artifacts (git-ignored)
├── reports/                 # charts and reports (git-ignored)
├── artifacts/               # prediction snapshots and manifests (git-ignored)
├── logs/                    # app log file (git-ignored)
├── .env.example             # documented environment template (no secrets)
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── explanation.md           # Design decisions, limits, deliverables
└── README.md                # This file
```

## Quick start (Windows PowerShell)

```powershell
# From the repository root
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .

Copy-Item .env.example .env
# Edit .env and fill in ASSET_SYMBOL, ASSET_TYPE, APIFY_ACTOR_ID,
# APIFY_API_TOKEN, and TAVILY_API_KEY.

.\venv\Scripts\python.exe -m src.main check-config
```

The `crowdxbot` console command is available after the editable install:

```powershell
crowdxbot check-config
crowdxbot check-config --json
crowdxbot --help
crowdxbot run-all --historical-file path/to/export.csv --reimport --force
```

On macOS/Linux the equivalent setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
python -m src.main check-config
```

## Configuration

Settings are loaded from environment variables first, then from `.env`
(environment wins). Empty environment values are treated as unset. The full,
documented list lives in [`.env.example`](.env.example); the groups are:

| Group | Variables | Notes |
| --- | --- | --- |
| Asset | `ASSET_SYMBOL`, `ASSET_TYPE`, `TIMEZONE` | Required symbol/type; IANA timezone for daily buckets. |
| Date window | `HISTORY_DAYS`, `DATA_START_DATE`, `DATA_END_DATE` | Default 180 days; explicit dates take precedence. |
| Prices | `DATA_PROVIDER` | `yfinance` today; adapters are replaceable. |
| X ingestion | `APIFY_ACTOR_ID`, `APIFY_API_TOKEN`, `APIFY_MAX_ITEMS`, `APIFY_INPUT_TEMPLATE`, ... | Actor ID and token are required; the input template is required for live runs. |
| News | `TAVILY_API_KEY`, `EXA_API_KEY`, `TAVILY_MAX_RESULTS`, ... | Exa is optional. |
| Storage | `DATABASE_URL`, `SQLITE_BUSY_TIMEOUT_MS` | SQLite by default; PostgreSQL supported. |
| Modelling | `MODEL_*`, `WALK_FORWARD_*`, `MIN_TRAIN_SAMPLES`, ... | Per-horizon models and walk-forward rules. |
| Sentiment | `SENTIMENT_MODEL_*` | Pinned model name/revision, batch size, device. |
| Paths | `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, ... | Relative paths resolve against the project root. |
| Logging | `LOG_LEVEL`, `LOG_FORMAT`, `LOG_MAX_BYTES`, ... | Console text or JSON; rotating file is always JSON. |

Key behaviours:

- Missing required settings fail with an actionable, secret-free message.
- `ASSET_SYMBOL`, `ASSET_TYPE`, `DATA_PROVIDER`, `APIFY_ACTOR_ID`,
  `APIFY_API_TOKEN`, and `TAVILY_API_KEY` are required at startup.
- `WALK_FORWARD_GAP_DAYS` must be at least the largest `MODEL_HORIZONS` value to
  prevent target leakage.
- `settings.date_range` returns the effective `(start, end)` dates.
- `settings.asset_slug` is a filesystem-safe symbol (for example `BTC-USD` →
  `BTC_USD`).
- `settings.ensure_directories()` creates all output directories and the SQLite
  parent directory idempotently.
- Live collection requires `APIFY_INPUT_TEMPLATE` or `APIFY_INPUT_TEMPLATE_PATH`;
  the module never guesses an actor's field names.

### Public contract

```python
from src.config import get_settings

settings = get_settings()
settings.asset_symbol          # configured symbol
settings.date_range            # (start_date, end_date)
settings.horizons              # (1, 2, 3, 4, 5)
settings.raw_data_dir          # absolute Path
settings.database_url          # SQLAlchemy URL
settings.safe_summary()        # redacted, display/JSON-safe mapping
```

`get_settings()` is cached for the process. Tests can call `load_settings()`
(uncached) or `reload_settings()` after changing the environment.

## Logging and secret handling

- `configure_logging(settings)` installs a console handler and a rotating JSON
  file handler (default `logs/app.log`).
- Every record carries a `run_id`; use `run_context()` or `new_run_id()`.
- Secrets (`APIFY_API_TOKEN`, `TAVILY_API_KEY`, optional `EXA_API_KEY`, and a
  database password) are redacted from messages, `extra` fields, and exception
  text. Field names such as `api_key`, `token`, `password`, and `authorization`
  are redacted automatically.
- `.env`, log files, raw data, models, reports, and artifacts are git-ignored.
- Do not put credentials in actor payloads, manifests, reports, or committed
  configuration. Use the environment or a secret manager.

Example structured event:

```python
from src.logging_config import get_logger, run_context

with run_context() as run_id:
    get_logger("src.pipeline").info(
        "stage complete",
        extra={"stage": "ingest_prices", "rows": 180},
    )
```

## Commands

| Command | Purpose |
| --- | --- |
| `python -m src.main check-config` | Validate settings and create output directories. |
| `python -m src.main check-config --json` | Emit the redacted summary as JSON for automation. |
| `python -m src.main check-config --no-paths` | Omit resolved paths from the summary. |
| `python -m src.main doctor` | Alias for `check-config`. |
| `python -m src.main import-historical-tweets <export>` | Import an explicit CSV/TSV/XLSX/JSON export. |
| `python -m src.main ingest-x` | Collect recent X posts through the configured Apify actor. |
| `python -m src.main ingest-x --dry-run` | Build and validate the query/actor input without starting a run. |
| `python -m src.main inspect-apify-actor` | Show Apify actor metadata without starting a run. |
| `python -m src.main version` | Print the application version. |
| `crowdxbot run-all` | Run the full pipeline end-to-end. |
| `crowdxbot run-all --historical-file <path>` | Run with a historical tweet export. |
| `crowdxbot run-all --reimport` | Re-run ingestion stages even if cached. |
| `crowdxbot run-all --retrain` | Retrain model even if cached. |
| `crowdxbot run-all --force` | Continue after failed stages (price-only run if no tweets). |

Pipeline commands (`ingest-prices`, `score`, `train`, `predict`, `report`,
`run-all`) are added by later modules.

### Flags for `run-all`

| Flag | Purpose |
| --- | --- |
| `--reimport` | Re-run ingestion stages even if cached. |
| `--retrain` | Retrain model even if cached. |
| `--force` | Continue after failed stages (price-only run if no tweets). |
| `--historical-file <path>` | Import a historical tweet export. |

### Inspect the Apify actor

```powershell
crowdxbot inspect-apify-actor
crowdxbot ingest-x --dry-run
```

## Historical tweet import

Module 1 never scrapes a private Google Doc. Export the document yourself
(**File → Download → CSV/XLSX**, or save/export JSON) and pass the file to the
importer:

```powershell
.\venv\Scripts\python.exe -m src.main import-historical-tweets .\exports\tweets.csv
```

Supported inputs: `.csv`, `.tsv`, `.xlsx`/`.xlsm`, `.json`, and `.jsonl`.
The importer auto-detects the format from the extension; use `--format` when the
extension is misleading. XLSX input reads the active sheet unless `--sheet` is
given. `openpyxl` is required for XLSX and is listed in `requirements.txt`.

### What it does

1. Reads the explicit export without modifying it.
2. Detects source columns and maps them to canonical candidates: post ID, text,
   created timestamp, author, likes, reposts, replies, quotes, views, and URL.
   Common aliases such as `Post Text`, `Tweet`, `Created At`, `Username`,
   `Like Count`, and `Tweet URL` are recognised.
3. Parses timestamps with timezone awareness. Offset-bearing timestamps keep
   their original text and offset; naive timestamps are interpreted in the
   configured `TIMEZONE` and flagged with `timestamp_timezone_assumed`.
4. Validates required fields (`text` and timestamp). Malformed rows are written
   to `quarantine.jsonl` with a reason code, never silently dropped. Missing
   post IDs are allowed and reported; Module 3 applies its conservative
   fallback fingerprint later.
5. Preserves every source column in `raw_records.jsonl` and copies the original
   export into the batch directory for provenance.
6. Records filename, SHA-256, import time, row counts, column mapping, parse
   errors, warnings, and output paths in `manifest.json`.

### Output layout

```text
data/raw/historical_tweets/<ASSET_SLUG>/<BATCH_ID>/
├── source_<original-name>    # untouched copy of the export
├── raw_records.jsonl         # every source row, including malformed rows
├── canonical_records.jsonl   # valid RawTweetRecord objects
├── quarantine.jsonl          # malformed rows with reason_code/reason
└── manifest.json             # import summary and provenance
```

`import_historical_tweets()` returns an `ImportResult` containing the valid
records, quarantined rows, and summary. An optional `sink` implementing
`ingest_tweets(records, *, batch_id=...)` is how Module 3's canonical storage
will receive the same records that Module 2 will later produce.

Useful options:

| Option | Purpose |
| --- | --- |
| `--column text=Tweet` | Override a canonical-to-source column mapping (repeatable). |
| `--apply-date-window` | Quarantine rows outside the configured date range instead of keeping and counting them. |
| `--start-date YYYY-MM-DD` / `--end-date YYYY-MM-DD` | Override the configured inclusive date window. |
| `--assume-timezone Asia/Kolkata` | Timezone for naive timestamps; defaults to `TIMEZONE`. |
| `--no-copy-source` | Skip copying the original export into the batch directory. |
| `--fail-on-quarantine` | Exit with status 2 when any row is quarantined. |
| `--json` | Emit the summary as machine-readable JSON. |

No sentiment scoring, deduplication, or date aggregation happens here; those
belong to Modules 6, 3, and 7.

## Live X ingestion through Apify

Module 2 collects recent posts through the Apify actor named by
`APIFY_ACTOR_ID`. It produces the same `RawTweetRecord` objects as Module 1, so
Module 3 can consume either source through the shared `TweetSink` contract.

### Actor input is schema-driven, never guessed

Apify actor objects do not expose their input schema. Inspect the actor's
`INPUT_SCHEMA.json` (or its documentation), then supply a JSON template using
only the actor's real field names. The template is rendered with these
placeholders:

| Placeholder | Value |
| --- | --- |
| `{query}` | Complete X search query, including date operators and exclusions. |
| `{query_terms}` | List of query terms (for actors that take an array). |
| `{max_items}` | `APIFY_MAX_ITEMS` or the per-run override. |
| `{asset_symbol}` / `{asset_type}` / `{asset_slug}` | Configured asset identity. |
| `{start_date}` / `{end_date}` | Inclusive ISO dates in the configured timezone. |
| `{since}` / `{until}` | UTC ISO datetimes; `{until}` is exclusive. |
| `{language}` | Optional language code, or an empty string. |

An exact placeholder such as `"maxItems": "{max_items}"` keeps the value's type
(integer in this case); a placeholder embedded in a larger string is formatted
as text. Use exactly one of `APIFY_INPUT_TEMPLATE` (inline JSON) or
`APIFY_INPUT_TEMPLATE_PATH` (a JSON file). Set `APIFY_INPUT_SCHEMA_PATH` to a
verified `INPUT_SCHEMA.json` for local validation, and pass
`--validate-remote` to ask Apify to validate the payload before starting.

```dotenv
APIFY_ACTOR_ID=apidojo/tweet-scraper
APIFY_INPUT_TEMPLATE={"searchTerms":["{query}"],"maxItems":"{max_items}","sort":"Latest"}
APIFY_INPUT_SCHEMA_PATH=config/INPUT_SCHEMA.json
```

The template above is the verified shape of `apidojo/tweet-scraper` (input
schema checked 2026-11-02): `searchTerms` is an array, `maxItems` is an integer,
and `sort` is a string. The query already contains `since:`/`until:` and
`lang:` operators, so the actor's `start`, `end`, and `tweetLanguage` fields are
not required.

### Query and disambiguation

The query is built from the configured asset:

- `cashtag_and_symbol` (default): `$AAPL` or the quoted symbol, plus aliases.
- `cashtag_only`: only the cashtag, which is the safest choice for short or
  ambiguous tickers.
- `symbol_only`: only the quoted symbol.

Add aliases with `APIFY_QUERY_TERMS` (for example `Apple Inc,Apple`) and exclude
colliding meanings with `APIFY_QUERY_EXCLUDE_TERMS` (for example
`giveaway,airdrop`). `APIFY_QUERY_LANGUAGE` adds an X `lang:` operator. The date
window always becomes inclusive `since:` and exclusive `until:` operators.

```powershell
# Inspect the actor, then build the request without spending a run:
.\venv\Scripts\python.exe -m src.main inspect-apify-actor
.\venv\Scripts\python.exe -m src.main ingest-x --dry-run --json
```

### Output layout

```text
data/raw/live_x/<ASSET_SLUG>/<COLLECTION_ID>/
├── actor_run.json        # run/dataset IDs, start response, final status (secrets redacted)
├── dataset_items.json    # untouched dataset items exactly as returned by Apify
├── records.jsonl         # valid RawTweetRecord objects
├── quarantine.jsonl      # malformed/out-of-window items with reason_code/reason
└── manifest.json         # query, window, counts, retries, paths, warnings
```

`collect_x_posts()` starts the actor, polls with a bounded timeout, retrieves the
dataset in pages up to `APIFY_MAX_ITEMS`, normalizes only the fields Module 3
needs, and persists everything before returning a `CollectionResult`. A
successful run with an empty dataset returns normally with `empty_result=True`.
Failed, aborted, or timed-out runs persist whatever items were available and
raise `ApifyRunFailedError`/`ApifyRunTimeoutError` unless `allow_partial=True`.
Rate-limit and transient 5xx responses are retried with backoff and counted in
the manifest.

Useful options:

| Option | Purpose |
| --- | --- |
| `--start-date YYYY-MM-DD` / `--end-date YYYY-MM-DD` | Override the configured inclusive date window. |
| `--query-mode cashtag_only` | Avoid bare-symbol collisions. |
| `--extra-term "Apple Inc"` | Add an alias (repeatable). |
| `--exclude-term giveaway` | Exclude a term (repeatable). |
| `--max-items 500` | Override the per-run item cap. |
| `--dry-run` | Render and validate the request without starting the actor. |
| `--validate-remote` | Ask Apify to validate the rendered input first. |
| `--allow-partial` | Accept records from a failed/aborted/timed-out run. |
| `--apply-date-window` | Quarantine items outside the configured window. |
| `--json` | Emit the manifest as machine-readable JSON. |

Secrets are never written to `actor_run.json` or `manifest.json`; the logging
redaction filter also covers Apify API errors. The untouched dataset payload is
preserved in `dataset_items.json` for audit and replay.

## Output

All outputs are written under the project root and are gitignored (regenerated on
each run):

| Path | Contents |
| --- | --- |
| `data/processed/crowdxbot.db` | SQLite DB: canonical tweets, price bars, quarantine rows |
| `data/processed/sentiment/` | Daily sentiment buckets + scores parquet |
| `data/processed/datasets/` | Versioned modelling dataset parquet + audit |
| `models/<run_id>.pkl` | Pickled ModelBundle (XGBoost models + preprocessor + manifest) |
| `reports/evaluation.md` | Walk-forward errors, baseline comparison, strategy risk metrics (Sharpe/Sortino/Max DD) |
| `reports/outlook.md` | Next 1–5 day forecasts + sentiment context + recent news |
| `reports/charts/AAPL_forecast.png` | Price forecast chart |
| `artifacts/pipeline/AAPL/state.json` | Per-stage fingerprints for cache/reuse |
| `logs/app.log` | Structured JSON log |

### Example output (reports/evaluation.md)

```
## Comparison With the Persistence Baseline
| Model   | Horizon | MAE improvement | RMSE improvement | Verdict         |
|---------|---------|-----------------|------------------|-----------------|
| drift   | n+1     | +0.1%           | +0.1%            | better          |
| xgboost | n+1     | -1.8%           | -3.3%            | worse           |

## Strategy Risk Metrics (out-of-sample)
| Model   | Horizon | Sharpe | Sortino | Max Drawdown | Ann. Return | Ann. Volatility | Total Trades |
|---------|---------|--------|---------|--------------|-------------|-----------------|--------------|
| xgboost | n+1     | 0.496  | 0.387   | 0.00%        | +7.64%      | 15.40%          | 56           |
```

### Example output (reports/outlook.md)

```
## Forecasts n+1 … n+5
| Horizon | Predicted return | Predicted close |
|---------|------------------|-----------------|
| n+1     | +0.2270%         | 342.02          |
| n+5     | +1.4203%         | 346.10          |
```

## Configuration notes

- **Date window:** Defaults to the last `HISTORY_DAYS` (180) days. Set
  `DATA_START_DATE` and/or `DATA_END_DATE` in `.env` to override.
- **Sentiment lag:** Same-day sentiment is refused by default (lag >= 1) to
  avoid look-ahead leakage.
- **Walk-forward:** Expanding training window, 120-day initial train, 5-day
  test blocks, 5-day embargo (>= largest horizon).
- **Hyperparameters:** GridSearchCV with TimeSeriesSplit searches
  `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`,
  `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight` per horizon.
- **Model cache:** Trained models are saved as `<run_id>.pkl` in `models/`.
  Subsequent runs reuse the cached bundle unless `--retrain` is passed.

## Tests and quality checks

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check src tests
.\venv\Scripts\python.exe -m ruff format --check src tests
```

Tests use isolated temporary directories, synthetic credentials, and mocked
external services. They point `CROWDXBOT_ENV_FILE` at a non-existent file so a
developer's real `.env` cannot affect results. Live API tests should be marked
`@pytest.mark.live` and are opt-in.

## Configuration troubleshooting

- **`ASSET_SYMBOL is required`** — copy `.env.example` to `.env` and fill it in.
- **Unknown timezone** — use an IANA name such as `America/New_York`; on Windows
  the `tzdata` package supplies the timezone database.
- **`WALK_FORWARD_GAP_DAYS` error** — set the gap to at least the largest
  configured horizon (default horizons 1–5 need a gap of 5).
- **Database path errors** — relative SQLite paths resolve against the project
  root; ensure the target directory is writable.
- **Secret accidentally printed** — rotate it immediately and remove it from
  any log/report artifact. Redaction is a safety net, not a substitute for
  careful handling.

## Limitations

- Daily stock returns have near-zero autocorrelation; persistence ("price stays
  same") is the near-optimal guess. XGBoost cannot consistently beat it.
- The free Apify plan caps results at ~10 tweets/day — insufficient for real
  sentiment coverage without a historical import.
- Strategy risk metrics (Sharpe/Sortino/Max Drawdown) can be near-zero or
  misleading when threshold-based signals are sparse (many zero-signal rows →
  flat equity curve).
- Forecasts are model estimates, not guaranteed outcomes. Not financial advice.