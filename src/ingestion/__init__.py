"""Ingestion adapters that produce raw records for canonical storage."""

from __future__ import annotations

from src.ingestion.apify_x import (
    ApifyActorInputError,
    ApifyConfigurationError,
    ApifyDatasetError,
    ApifyIngestionError,
    ApifyRunError,
    ApifyRunFailedError,
    ApifyRunTimeoutError,
    ApifyTemplateError,
    CollectionManifest,
    CollectionResult,
    PreparedCollection,
    QuerySpec,
    build_query_spec,
    collect_x_posts,
    inspect_actor,
    prepare_x_collection,
)
from src.ingestion.historical_tweets import (
    HistoricalImportError,
    ImportResult,
    ImportSummary,
    import_historical_tweets,
)
from src.ingestion.news import collect_news, NewsItem, NewsManifest
from src.ingestion.records import QuarantinedTweetRow, RawTweetRecord, TweetSink

__all__ = [
    "ApifyActorInputError",
    "ApifyConfigurationError",
    "ApifyDatasetError",
    "ApifyIngestionError",
    "ApifyRunError",
    "ApifyRunFailedError",
    "ApifyRunTimeoutError",
    "ApifyTemplateError",
    "CollectionManifest",
    "CollectionResult",
    "HistoricalImportError",
    "ImportResult",
    "ImportSummary",
    "PreparedCollection",
    "QuarantinedTweetRow",
    "QuerySpec",
    "RawTweetRecord",
    "TweetSink",
    "collect_news",
    "NewsItem",
    "NewsManifest",
    "build_query_spec",
    "collect_x_posts",
    "import_historical_tweets",
    "inspect_actor",
    "prepare_x_collection",
]
