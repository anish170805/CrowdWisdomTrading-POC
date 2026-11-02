"""Pipeline orchestration: one command, only the stages whose inputs changed re-run.

``crowdxbot run-all`` calls :func:`run_pipeline`.  Every stage computes a
*fingerprint of its inputs* and compares it with the one recorded after its last
successful run (``artifacts/pipeline/<ASSET>/state.json``).  Unchanged inputs mean
the stage is skipped and its saved output reused.

Stage                  Re-runs when ...
---------------------  ------------------------------------------------------------
Historical import      the export file's SHA-256 changes (or ``--reimport``)
Apify X ingestion      asset / date window / query / max-items change, i.e. at most
                       once per day per config (or ``--reimport``)
Price ingestion        asset / date window / provider change, i.e. once per day
News (context only)    asset or calendar day changes
Sentiment scoring      only for posts with no cached score for the current model
Daily aggregation      always (cheap, in-memory)
Dataset build          price data, sentiment table, or feature config changed
Model training         dataset version or model hyper-parameters changed (or ``--retrain``)
Walk-forward eval      dataset version, model params, or fold settings changed
Latest prediction      always (cheap; uses the cached model)
Charts / reports       always (cheap)

Nothing here contains modelling logic; it only wires the modules together.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.markup import escape

from src.config import ConfigurationError, Settings, get_settings

console = Console()

STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _hash(payload: Any, length: int = 12) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _compute_run_id(settings: Settings) -> str:
    """Deterministic identifier for the configuration (display and provenance)."""
    pieces = {
        "asset": settings.asset_symbol,
        "type": settings.asset_type.value,
        "horizons": list(settings.horizons),
        "timezone": settings.timezone,
        "provider": settings.data_provider.value,
        "walk_forward": [
            settings.walk_forward_initial_train_days,
            settings.walk_forward_test_days,
            settings.walk_forward_gap_days,
        ],
        "model": [
            settings.model_type.value,
            settings.model_n_estimators,
            settings.model_max_depth,
            settings.model_learning_rate,
            settings.model_random_seed,
        ],
        "sentiment": [
            settings.sentiment_model_name,
            settings.sentiment_model_revision,
            settings.sentiment_max_length,
        ],
    }
    return f"{settings.asset_symbol}_{settings.timezone}_{_hash(pieces)}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


# ---------------------------------------------------------------------------
# Persistent stage state
# ---------------------------------------------------------------------------
class _State:
    """``stage -> {key, at, ...}`` recorded after each successful stage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if loaded.get("version") == STATE_VERSION:
                    self.data = loaded.get("stages", {})
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def fresh(self, stage: str, key: str) -> bool:
        return self.data.get(stage, {}).get("key") == key

    def get(self, stage: str) -> dict[str, Any]:
        return self.data.get(stage, {})

    def mark(self, stage: str, key: str, **info: Any) -> None:
        self.data[stage] = {"key": key, "at": datetime.now(UTC).isoformat(), **info}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": STATE_VERSION, "stages": _jsonable(self.data)}, indent=2),
            encoding="utf-8",
        )


class _AssetSink:
    """TweetSink that stamps the configured asset on every post it stores."""

    def __init__(self, store: Any, asset_symbol: str) -> None:
        self._store = store
        self._asset = asset_symbol

    def ingest_tweets(self, records: Any, *, batch_id: str) -> Any:
        return self._store.ingest_tweets(records, batch_id=batch_id, asset_symbol=self._asset)


@dataclass(frozen=True)
class _ScoredPost:
    post_id: str
    asset_symbol: str
    created_at_utc: datetime
    positive: float
    negative: float
    neutral: float
    label: str
    weight: float


@dataclass
class _Ctx:
    settings: Settings
    state: _State
    repo: Any
    store: Any
    reimport: bool
    retrain: bool
    historical_file: Path | None
    allow_rule_based: bool
    scored: list[_ScoredPost] = field(default_factory=list)
    sentiment_df: pd.DataFrame | None = None
    dataset: Any = None
    bundle: Any = None
    news_items: list[Any] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    snapshot: Any = None
    sentiment_coverage: dict[str, Any] | None = None
    model_cached: bool = False
    model_trained: bool = False

    @property
    def symbol(self) -> str:
        return self.settings.asset_symbol

    @property
    def slug(self) -> str:
        return self.settings.asset_slug


# ---------------------------------------------------------------------------
# Stages.  Each returns (status, detail, payload); status is one of
# "done" | "cached" | "skipped" | "warn".  Exceptions mean "failed".
# ---------------------------------------------------------------------------
def _stage_historical(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.ingestion.historical_tweets import import_historical_tweets

    if ctx.historical_file is None:
        return "skipped", "no --historical-file given", None
    path = ctx.historical_file
    if not path.is_file():
        raise FileNotFoundError(f"historical export not found: {path}")

    key = _hash({"sha256": _sha256_file(path), "asset": ctx.symbol})
    if not ctx.reimport and ctx.state.fresh("historical", key):
        return "cached", "this export was already imported", None

    result = import_historical_tweets(
        path, settings=ctx.settings, sink=_AssetSink(ctx.store, ctx.symbol)
    )
    summary = result.summary
    if summary.sink_error:
        raise RuntimeError(f"saved raw files but canonical storage failed: {summary.sink_error}")
    ctx.state.mark("historical", key, valid=summary.valid_rows, quarantined=summary.quarantined_rows)
    return "done", f"{summary.valid_rows} valid, {summary.quarantined_rows} quarantined", result


def _stage_apify(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.ingestion.apify_x import collect_x_posts, prepare_x_collection

    s = ctx.settings
    prepared = prepare_x_collection(s)  # validates the actor input template first
    key = _hash(
        {
            "asset": ctx.symbol,
            "start": s.effective_start_date,
            "end": s.effective_end_date,
            "max_items": prepared.max_items,
            "query": prepared.query.query,
            "actor": s.apify_actor_id,
        }
    )
    if not ctx.reimport and ctx.state.fresh("apify", key):
        return "cached", "already collected for this query/window today", None

    result = collect_x_posts(s, prepared=prepared, sink=_AssetSink(ctx.store, ctx.symbol))
    m = result.manifest
    ctx.state.mark("apify", key, valid=m.valid_records, quarantined=m.quarantined_records)
    if m.empty_result:
        return "warn", "actor run succeeded but returned no posts; check query/window", result
    window_days = (s.effective_end_date - s.effective_start_date).days + 1
    span = m.date_range_utc or {}
    span_days = None
    if span.get("start") and span.get("end"):
        span_days = (
            datetime.fromisoformat(span["end"]) - datetime.fromisoformat(span["start"])
        ).days + 1
    if (
        m.valid_records < prepared.max_items * 0.5
        and span_days is not None
        and span_days < window_days * 0.25
    ):
        return (
            "warn",
            f"only {m.valid_records} post(s) covering {span_days}d of a {window_days}d window "
            f"(requested {prepared.max_items}). The actor/plan is capping results (Apify free "
            "plans do); import history with --historical-file or use a paid plan",
            result,
        )
    return "done", f"{m.valid_records} valid, {m.quarantined_records} quarantined", result


def _stage_prices(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.ingestion.prices import import_historical_price_bars

    s = ctx.settings
    start, end = s.effective_start_date, s.effective_end_date
    key = _hash({"asset": ctx.symbol, "start": start, "end": end, "provider": s.data_provider.value})
    stored = ctx.repo.count_price_bars(asset_symbol=ctx.symbol)
    if not ctx.reimport and stored > 0 and ctx.state.fresh("prices", key):
        return "cached", f"{stored} stored bars up to date", None

    result = import_historical_price_bars(
        asset_symbol=ctx.symbol,
        asset_type=s.asset_type.value,
        start_date=start,
        end_date=end,
        provider=s.data_provider,
        batch_id=f"price-{ctx.slug}-{end.isoformat()}",
        settings=s,
    )
    if result["errors"]:
        message = "; ".join(result["errors"])
        if stored == 0:
            raise RuntimeError(f"price provider failed and no bars are stored: {message}")
        return "warn", f"provider failed ({message}); continuing with {stored} stored bars", result
    ctx.state.mark("prices", key, inserted=result["inserted"], updated=result["updated"])
    return "done", f"{result['inserted']} new, {result['updated']} updated bars", result


def _stage_news(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.ingestion.news import NewsManifest, collect_news

    s = ctx.settings
    out_dir = s.artifacts_dir / "news" / ctx.slug
    latest = out_dir / "latest.json"
    key = _hash({"asset": ctx.symbol, "day": datetime.now(UTC).date(), "v": 2})

    if not ctx.reimport and ctx.state.fresh("news", key) and latest.is_file():
        manifest = NewsManifest.model_validate_json(latest.read_text(encoding="utf-8"))
        ctx.news_items = list(manifest.items)
        return "cached", f"{len(manifest.items)} item(s) from today", manifest

    manifest = collect_news(ctx.symbol, settings=s, output_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    ctx.news_items = list(manifest.items)
    if manifest.error:
        return "warn", f"news provider error: {manifest.error}", manifest
    ctx.state.mark("news", key, items=len(manifest.items))
    return "done", f"{len(manifest.items)} item(s)", manifest


def _stage_sentiment(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.sentiment.scorer import FALLBACK_MODEL_NAME, score_all_posts

    s = ctx.settings
    tweets = ctx.repo.read_tweets(asset_symbol=ctx.symbol)
    if not tweets:
        return "skipped", "no canonical posts stored for this asset", None

    tag = _hash(
        {"model": s.sentiment_model_name, "rev": s.sentiment_model_revision, "len": s.sentiment_max_length}
    )
    cache_path = s.processed_data_dir / "sentiment" / ctx.slug / f"scores_{tag}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["post_key", "positive", "negative", "neutral", "label", "model_name"]
    cached = pd.read_parquet(cache_path) if cache_path.is_file() else pd.DataFrame(columns=columns)
    known = set(cached["post_key"])
    missing = [t for t in tweets if t.post_fingerprint not in known]

    fresh_rows: list[dict[str, Any]] = []
    if missing:
        batch = [SimpleNamespace(post_id=t.post_fingerprint, text=t.text) for t in missing]
        results = score_all_posts(batch, settings=s, allow_fallback=ctx.allow_rule_based)
        fresh_rows = [
            {
                "post_key": r.post_id,
                "positive": r.positive,
                "negative": r.negative,
                "neutral": r.neutral,
                "label": r.label,
                "model_name": r.model_name,
            }
            for r in results
        ]
        genuine = [row for row in fresh_rows if row["model_name"] != FALLBACK_MODEL_NAME]
        if genuine:  # keyword-fallback scores are never cached as if they were FinBERT's
            cached = pd.DataFrame(genuine) if cached.empty else pd.concat(
                [cached, pd.DataFrame(genuine)], ignore_index=True
            )
            cached.to_parquet(cache_path, index=False)

    by_key = {row["post_key"]: row for row in cached.to_dict("records")}
    by_key.update({row["post_key"]: row for row in fresh_rows})

    scored: list[_ScoredPost] = []
    for t in tweets:
        row = by_key.get(t.post_fingerprint)
        if row is None:
            continue
        engagement = sum(v or 0 for v in (t.likes, t.reposts, t.replies, t.quotes))
        scored.append(
            _ScoredPost(
                post_id=t.post_fingerprint,
                asset_symbol=ctx.symbol,
                created_at_utc=t.created_at,
                positive=float(row["positive"]),
                negative=float(row["negative"]),
                neutral=float(row["neutral"]),
                label=str(row["label"]),
                weight=1.0 + math.log1p(engagement),
            )
        )
    ctx.scored = scored
    status = "done" if missing else "cached"
    return status, f"{len(missing)} newly scored, {len(tweets) - len(missing)} from cache", scored


def _stage_aggregate(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.features.sentiment_daily import aggregate_daily_sentiment

    if not ctx.scored:
        ctx.sentiment_df = None
        return "skipped", "no scored posts", None
    table = aggregate_daily_sentiment(ctx.scored, settings=ctx.settings)
    if table.empty:
        ctx.sentiment_df = None
        return "skipped", "aggregation produced no rows", None
    out = ctx.settings.processed_data_dir / "sentiment" / ctx.slug / "daily.parquet"
    table.to_parquet(out, index=False)
    ctx.sentiment_df = table
    return "done", f"{len(table)} daily bucket(s)", table


def _stage_dataset_inner(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.features.dataset import (
        DatasetConfig,
        build_modelling_dataset,
        load_dataset,
        save_dataset,
    )
    from src.features.price_features import bars_to_frame

    s = ctx.settings
    bars = ctx.repo.read_price_bars(ctx.symbol, s.effective_start_date, s.effective_end_date)
    if not bars:
        raise RuntimeError("no stored price bars for the configured window")

    config = DatasetConfig.from_settings(s, sentiment_bucket_timezone=s.timezone)
    key = _hash(
        {
            "prices": _frame_digest(bars_to_frame(bars)),
            "sentiment": _frame_digest(ctx.sentiment_df) if ctx.sentiment_df is not None else None,
            "config": config.to_dict(),
        }
    )
    root = s.processed_data_dir / "datasets" / ctx.slug
    previous = ctx.state.get("dataset")
    if previous.get("key") == key and not ctx.reimport:
        try:
            ctx.dataset = load_dataset(root / previous["version"])
            return "cached", f"version {ctx.dataset.version}", ctx.dataset
        except Exception as exc:  # noqa: BLE001 - rebuild below
            console.print(f"[yellow]Saved dataset unreadable ({escape(str(exc))}); rebuilding[/yellow]")

    dataset = build_modelling_dataset(bars, sentiment=ctx.sentiment_df, config=config)
    save_dataset(dataset, root)
    ctx.state.mark("dataset", key, version=dataset.version, rows=len(dataset.frame))
    ctx.dataset = dataset
    kind = "with sentiment" if ctx.sentiment_df is not None else "PRICE-ONLY (no sentiment)"
    return "done", f"version {dataset.version}, {len(dataset.frame)} rows, {kind}", dataset


def _sentiment_coverage(dataset: Any) -> dict[str, Any]:
    """How much of the modelling table actually carries sentiment information."""
    frame = dataset.frame
    flags = [c for c in frame.columns if c.startswith('sentiment_row_present_lag_')]
    if not flags or frame.empty:
        return {'share_rows': 0.0, 'latest_row': False, 'rows': int(len(frame)), 'with_sentiment': 0}
    present = frame[flags].max(axis=1) == 1.0
    return {
        'share_rows': float(present.mean()),
        'latest_row': bool(present.iloc[-1]),
        'rows': int(len(frame)),
        'with_sentiment': int(present.sum()),
    }


def _stage_dataset(ctx: _Ctx) -> tuple[str, str, Any]:
    status, detail, payload = _stage_dataset_inner(ctx)
    cov = _sentiment_coverage(ctx.dataset)
    ctx.sentiment_coverage = cov
    detail += f'; sentiment present on {cov["with_sentiment"]}/{cov["rows"]} rows'
    if cov['share_rows'] < 0.5:
        detail += ' - too sparse for sentiment to inform the model (effectively price-only)'
        status = 'warn'
    return status, detail, payload


def _stage_model(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.models.artifacts import load_bundle_pkl, save_bundle_pkl
    from src.models.train import ModelConfig, train_final_with_gridsearch

    s = ctx.settings
    ds = ctx.dataset
    config = ModelConfig.from_settings(s)
    key = _hash({"dataset": ds.version, "model": config.to_dict(), "gridsearch": True})
    run_id = f"{ctx.slug}_{key}"
    bundle_path = s.models_dir / f"{run_id}.pkl"

    if bundle_path.is_file() and not ctx.retrain:
        try:
            ctx.bundle = load_bundle_pkl(bundle_path)
            ctx.model_cached = True
            return "cached", f"reused {bundle_path.name}", ctx.bundle
        except Exception as exc:
            console.print(f"[yellow]Cached model unusable ({escape(str(exc))}); retraining[/yellow]")

    bundle, _ = train_final_with_gridsearch(
        ds, np.arange(len(ds.frame)), config, asset_symbol=ctx.symbol
    )
    if not bundle.models:
        raise RuntimeError(
            "no horizon had enough labelled rows "
            f"(min_train_samples={s.min_train_samples}); collect more history. "
            f"Skipped: {bundle.manifest.get('skipped_horizons')}"
        )
    save_bundle_pkl(bundle, s.models_dir, run_id)
    ctx.bundle = bundle
    ctx.model_trained = True
    ctx.state.mark("model", key, run_id=run_id, horizons=list(bundle.horizons))
    return "done", f"trained horizons {list(bundle.horizons)} -> {run_id}.pkl", bundle


def _compute_strategy_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute Sharpe/Sortino/Max Drawdown per model and horizon from predictions."""
    from src.metrics.trading import (
        compute_metrics_from_signals,
        compute_strategy_returns,
        generate_signals,
    )

    metrics: dict[str, Any] = {}
    if predictions.empty:
        return metrics
    for model in predictions["model"].unique():
        model_preds = predictions[predictions["model"] == model]
        for horizon in sorted(model_preds["horizon"].unique()):
            h_preds = model_preds[model_preds["horizon"] == int(horizon)]
            signals = generate_signals(h_preds, horizon=int(horizon))
            strat_returns = compute_strategy_returns(signals)
            m = compute_metrics_from_signals(strat_returns)
            # Simplify max_drawdown to just the drawdown value for the table
            dd = m.get("max_drawdown", {})
            if isinstance(dd, dict):
                m["max_drawdown"] = dd.get("max_drawdown", 0.0)
            metrics[f"{model}_n+{int(horizon)}"] = m
    return metrics


def _stage_evaluation(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.models.train import ModelConfig, run_folds, save_run
    from src.validation.walk_forward import build_walk_forward_folds

    s = ctx.settings
    ds = ctx.dataset
    config = ModelConfig.from_settings(s)
    walk = {
        "initial": s.walk_forward_initial_train_days,
        "test": s.walk_forward_test_days,
        "gap": s.walk_forward_gap_days,
    }
    key = _hash({"dataset": ds.version, "model": config.to_dict(), "walk": walk})
    directory = s.artifacts_dir / "evaluation" / ctx.slug / key
    summary_path = directory / "metrics_summary.csv"

    def _read(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(directory / name)
        except (OSError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    def _load_cached_predictions() -> pd.DataFrame:
        try:
            return pd.read_parquet(directory / "fold_predictions.parquet")
        except Exception:
            return pd.DataFrame()

    if summary_path.is_file() and not ctx.retrain:
        meta = json.loads((directory / "fold_metadata.json").read_text(encoding="utf-8"))
        predictions_df = _load_cached_predictions()
        ctx.evaluation = {
            "summary": _read("metrics_summary.csv"),
            "comparison": _read("baseline_comparison.csv"),
            "fold_metadata": meta.get("folds", []),
            "strategy_metrics": _compute_strategy_metrics(predictions_df),
        }
        return "cached", f"{len(ctx.evaluation['fold_metadata'])} fold(s) reused", ctx.evaluation

    folds = build_walk_forward_folds(
        ds,
        initial_train_days=walk["initial"],
        test_days=walk["test"],
        gap_days=walk["gap"],
        asset_symbol=ctx.symbol,
    )
    if not folds:
        ctx.evaluation = None
        return (
            "skipped",
            "not enough history for one walk-forward fold (initial train + embargo + test)",
            None,
        )
    run = run_folds(ds, folds, config, asset_symbol=ctx.symbol)
    save_run(run, directory)
    ctx.evaluation = {
        "summary": run.summary,
        "comparison": run.comparison,
        "fold_metadata": run.fold_metadata(),
        "strategy_metrics": _compute_strategy_metrics(run.predictions),
    }
    ctx.state.mark("evaluation", key, folds=len(folds))
    return "done", f"{len(folds)} fold(s) evaluated", ctx.evaluation


def _stage_prediction(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.prediction.latest import generate_latest_prediction

    latest = ctx.dataset.latest_rows()
    latest = latest[latest["asset_symbol"].astype(str) == ctx.symbol]
    if latest.empty:
        raise RuntimeError("dataset has no latest row for the configured asset")
    row = latest.iloc[[-1]].copy()
    bar_date = pd.Timestamp(row["bar_date"].iloc[0]).date()
    row.attrs["cutoff_timestamp"] = datetime.combine(bar_date, time.min, tzinfo=UTC)

    snapshot = generate_latest_prediction(ctx.bundle, row, asset_symbol=ctx.symbol)
    ctx.snapshot = snapshot

    out_dir = ctx.settings.artifacts_dir / "predictions" / ctx.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(snapshot.as_dict()), indent=2)
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    (out_dir / f"{bar_date.isoformat()}.json").write_text(text, encoding="utf-8")
    return "done", f"n+0 close {snapshot.n0_observed:,.2f} as of {bar_date}", snapshot


def _stage_charts(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.visualization import generate_charts

    frame = ctx.dataset.frame
    closes = frame.set_index("bar_date")["close_n0"]
    paths = generate_charts(
        prediction_snapshot=ctx.snapshot,
        asset_symbol=ctx.symbol,
        output_dir=ctx.settings.reports_dir / "charts",
        recent_closes=closes,
    )
    return "done", f"{len(paths)} chart(s)", paths


def _stage_reports(ctx: _Ctx) -> tuple[str, str, Any]:
    from src.reporting.evaluation import generate_evaluation_report
    from src.reporting.outlook import generate_outlook_report

    s = ctx.settings
    sentiment_summary = None
    if ctx.sentiment_df is not None and not ctx.sentiment_df.empty:
        sentiment_summary = ctx.sentiment_df.iloc[-1].to_dict()

    ev = ctx.evaluation or {}
    outlook = generate_outlook_report(
        s,
        ctx.snapshot,
        sentiment_summary=sentiment_summary,
        news_items=ctx.news_items,
        evaluation_comparison=ev.get('comparison'),
        sentiment_coverage=ctx.sentiment_coverage,
        output_path=s.reports_dir / "outlook.md",
    )
    evaluation = generate_evaluation_report(
        s,
        dataset=ctx.dataset,
        prediction_snapshot=ctx.snapshot,
        summary=ev.get("summary"),
        comparison=ev.get("comparison"),
        fold_metadata=ev.get("fold_metadata"),
        strategy_metrics=ev.get("strategy_metrics"),
        output_path=s.reports_dir / "evaluation.md",
    )
    return "done", f"{outlook.name}, {evaluation.name}", {"outlook": str(outlook), "evaluation": str(evaluation)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    reimport: bool = False,
    retrain: bool = False,
    force: bool = False,
    historical_file: str | Path | None = None,
    allow_rule_based_sentiment: bool = False,
) -> dict[str, Any]:
    """Run the whole pipeline, re-running only stages whose inputs changed.

    Parameters
    ----------
    reimport : re-run ingestion stages and rebuild the dataset even if unchanged.
    retrain : retrain the model and re-run walk-forward evaluation.
    force : keep going after a failed stage instead of stopping.
    historical_file : optional historical tweet export to import (skipped if already imported).
    allow_rule_based_sentiment : permit keyword scoring if FinBERT cannot be loaded.
    """
    console.print("\n[bold blue]=== X Sentiment -> Asset Price Prediction Pipeline ===[/bold blue]\n")

    try:
        settings = get_settings()
    except ConfigurationError as error:
        console.print("[bold red]Configuration error[/bold red]")
        for issue in error.issues:
            console.print(f"  - {issue}")
        return {"error": str(error), "steps_completed": [], "steps_failed": [], "run_id": None}

    settings.ensure_directories()

    from src.storage.engine import create_engine_from_settings
    from src.storage.repository import CanonicalRepository
    from src.storage.store import CanonicalStore

    engine = create_engine_from_settings(settings)
    repo = CanonicalRepository(engine, ensure_schema=True)
    ctx = _Ctx(
        settings=settings,
        state=_State(settings.artifacts_dir / "pipeline" / settings.asset_slug / "state.json"),
        repo=repo,
        store=CanonicalStore(repo),
        reimport=reimport,
        retrain=retrain,
        historical_file=Path(historical_file).expanduser() if historical_file else None,
        allow_rule_based=allow_rule_based_sentiment,
    )

    results: dict[str, Any] = {
        "run_id": _compute_run_id(settings),
        "settings_asset": settings.asset_symbol,
        "steps_completed": ["config_directories"],
        "steps_cached": [],
        "steps_skipped": [],
        "steps_failed": [],
        "model_cached": False,
        "model_trained": False,
        "artifacts": {},
    }

    console.print(f"[dim]Asset: {settings.asset_symbol} | Timezone: {settings.timezone}[/dim]")
    console.print(f"[dim]Horizons: {settings.horizons} | Model: {settings.model_type.value}[/dim]\n")
    console.print("[green]Step 0: Configuration directories... done[/green]")

    colors = {"done": "green", "cached": "cyan", "skipped": "yellow", "warn": "yellow"}
    halted = False

    def run_stage(number: int, label: str, key: str, fn: Any, *, critical: bool = True) -> bool:
        """Run one stage; return False when the pipeline must stop."""
        title = f"Step {number}: {label}"
        try:
            status, detail, payload = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - reported, then decide whether to stop
            console.print(f"[red]{title}... failed: {escape(f'{type(exc).__name__}: {exc}')}[/red]")
            results["steps_failed"].append(key)
            return not (critical and not force)
        color = colors[status]
        label_text = {"warn": "done with warning"}.get(status, status)
        console.print(f"[{color}]{title}... {label_text}[/{color}] [dim]{escape(detail)}[/dim]")
        if status in {"done", "warn"}:
            results["steps_completed"].append(key)
        elif status == "cached":
            results["steps_completed"].append(key)
            results["steps_cached"].append(key)
        else:
            results["steps_skipped"].append(key)
        if payload is not None:
            results["artifacts"][key] = payload
        return True

    def tweets_available() -> int:
        return repo.count_tweets(asset_symbol=settings.asset_symbol)

    try:
        stages: list[tuple[int, str, str, Any, bool]] = [
            (1, "Historical tweet import", "historical_import", _stage_historical, False),
            (2, "Apify X ingestion", "apify_ingestion", _stage_apify, False),
            (4, "Price ingestion", "price_ingestion", _stage_prices, True),
            (5, "News retrieval (context only)", "news_retrieval", _stage_news, False),
        ]
        for number, label, key, fn, critical in stages:
            if not run_stage(number, label, key, fn, critical=critical):
                halted = True
                break

        if not halted:
            available = tweets_available()
            console.print(f"[green]Step 3: Canonical storage... {available} post(s) stored[/green]")
            results["steps_completed"].append("canonical_storage")
            if available == 0 and not force:
                console.print(
                    "[red]No X posts are stored, so there is no sentiment signal. Fix the Apify "
                    "step (see its error above), or run with --historical-file, or --force for a "
                    "price-only model.[/red]"
                )
                results["steps_failed"].append("no_tweets")
                halted = True

        later: list[tuple[int, str, str, Any, bool]] = [
            (6, "Sentiment scoring", "sentiment_scoring", _stage_sentiment, True),
            (7, "Sentiment aggregation", "sentiment_aggregation", _stage_aggregate, True),
            (8, "Feature engineering / dataset", "feature_engineering", _stage_dataset, True),
            (9, "Model training", "model_training", _stage_model, True),
            (10, "Walk-forward validation", "walk_forward_validation", _stage_evaluation, False),
            (11, "Latest prediction", "latest_prediction", _stage_prediction, True),
            (12, "Visualizations", "visualizations", _stage_charts, False),
            (13, "Reports", "reports", _stage_reports, True),
        ]
        for number, label, key, fn, critical in later:
            if halted:
                break
            if not run_stage(number, label, key, fn, critical=critical):
                halted = True
    finally:
        engine.dispose()

    results["model_cached"] = ctx.model_cached
    results["model_trained"] = ctx.model_trained
    results["last_step"] = results["steps_completed"][-1] if results["steps_completed"] else None

    console.print("\n" + "=" * 60)
    if results["steps_failed"]:
        console.print("[bold yellow]Pipeline finished with failures[/bold yellow]")
    else:
        console.print("[bold green]Pipeline Complete![/bold green]")
    console.print(f"Run ID: {results['run_id']}")
    console.print(
        f"Steps completed: {len(results['steps_completed'])} "
        f"({len(results['steps_cached'])} reused from cache, "
        f"{len(results['steps_skipped'])} skipped)"
    )
    if results["steps_failed"]:
        console.print("Failed steps:")
        for step in results["steps_failed"]:
            console.print(f"  - [red]{step}[/red]")
    if ctx.model_cached:
        console.print("[dim]Model reused from cache[/dim]")
    elif ctx.model_trained:
        console.print("[dim]New model trained[/dim]")
    console.print("=" * 60 + "\n")
    return results
