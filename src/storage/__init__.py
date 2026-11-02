"""Canonical storage: schemas, validation, and the persistence repository.

Modules 1, 2, and 4 write through :class:`CanonicalStore`; Modules 6-8 read
through :class:`CanonicalRepository`.
"""

from __future__ import annotations

from src.storage.engine import create_engine_from_settings, initialize_database
from src.storage.models import SCHEMA_VERSION, create_schema, metadata
from src.storage.repository import CanonicalRepository, UpsertStats, repository_from_settings
from src.storage.store import CanonicalStore, IngestResult, open_store

__all__ = [
    "SCHEMA_VERSION",
    "CanonicalRepository",
    "CanonicalStore",
    "IngestResult",
    "UpsertStats",
    "create_engine_from_settings",
    "create_schema",
    "initialize_database",
    "metadata",
    "open_store",
    "repository_from_settings",
]
