"""Model artifacts for Module 9: save, reload and verify.

Layout (one directory per trained bundle)::

    <root>/<run_id>/
        manifest.json         asset, target definition, feature order, training window,
                              library versions, seed, hyper-parameters, file checksums
        preprocessor.json     fold-local preprocessing state (feature order, fill values)
        horizon_<h>.json      one XGBoost model per horizon (native JSON format)

The manifest carries everything Module 11 needs to refuse a mismatched feature row.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.preprocessing import FeatureMatrixError, FeaturePreprocessor
from src.validation.leakage import CLOSE_N0, TARGET_CLOSE_FMT, TARGET_RETURN_FMT

ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PREPROCESSOR_NAME = "preprocessor.json"
TRACKED_LIBRARIES: tuple[str, ...] = ("xgboost", "numpy", "pandas", "scikit-learn", "pyarrow")


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, altered, or inconsistent."""


class UnsupportedHorizonError(KeyError):
    """Raised when asked to forecast a horizon that has no trained model."""


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def target_definition(horizon: int) -> dict[str, str]:
    """Explicit, self-describing target for one horizon."""
    return {
        "target_column": TARGET_RETURN_FMT.format(h=horizon),
        "definition": (
            f"simple return from the n+0 close to the n+{horizon} close "
            f"(next-available-bar basis): close_n+{horizon} / {CLOSE_N0} - 1"
        ),
        "price_reconstruction": f"close_n+{horizon} = {CLOSE_N0} * (1 + predicted_return)",
        "close_column": TARGET_CLOSE_FMT.format(h=horizon),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file(horizon: int) -> str:
    return f"horizon_{int(horizon)}.json"


@dataclass
class ModelBundle:
    """Trained per-horizon models plus the metadata needed to use them safely."""

    manifest: dict[str, Any]
    preprocessor: FeaturePreprocessor
    models: dict[int, Any] = field(default_factory=dict)

    @property
    def horizons(self) -> tuple[int, ...]:
        """Horizons that actually have a trained model."""
        return tuple(sorted(self.models))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.preprocessor.feature_columns

    @property
    def asset_symbol(self) -> str:
        return str(self.manifest["asset_symbol"])

    def predict_return(
        self, frame: pd.DataFrame, horizon: int, *, strict_order: bool = False
    ) -> np.ndarray:
        """Predicted ``n+horizon`` simple return for each row of ``frame``."""
        horizon = int(horizon)
        if horizon not in self.models:
            skipped = self.manifest.get("skipped_horizons", {})
            reason = skipped.get(str(horizon), "no model was trained for this horizon")
            raise UnsupportedHorizonError(f"horizon n+{horizon} unsupported: {reason}")
        matrix = self.preprocessor.transform(frame, strict_order=strict_order)
        return np.asarray(self.models[horizon].predict(matrix), dtype="float64")

    def predict_close(
        self, frame: pd.DataFrame, horizon: int, *, strict_order: bool = False
    ) -> np.ndarray:
        """Predicted ``n+horizon`` close, anchored on the observed ``close_n0``."""
        if CLOSE_N0 not in frame.columns:
            raise FeatureMatrixError(f"{CLOSE_N0!r} is required to reconstruct a price forecast")
        returns = self.predict_return(frame, horizon, strict_order=strict_order)
        return frame[CLOSE_N0].to_numpy(dtype="float64") * (1.0 + returns)


def save_bundle(bundle: ModelBundle, root: str | Path, run_id: str) -> Path:
    """Write the bundle to ``<root>/<run_id>/`` and return the directory."""
    if not run_id or any(ch in run_id for ch in "/\\:*?\"<>|"):
        raise ArtifactError(f"run_id {run_id!r} is not filesystem-safe")
    directory = Path(root) / run_id
    directory.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    pre_path = directory / PREPROCESSOR_NAME
    pre_path.write_text(
        json.dumps(bundle.preprocessor.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    files[PREPROCESSOR_NAME] = _sha256(pre_path)
    for horizon, model in sorted(bundle.models.items()):
        path = directory / _model_file(horizon)
        model.save_model(str(path))
        files[path.name] = _sha256(path)

    manifest = {
        **bundle.manifest,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "trained_horizons": list(bundle.horizons),
        "files": files,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    bundle.manifest = manifest
    return directory


def save_bundle_pkl(bundle: ModelBundle, root: str | Path, run_id: str) -> Path:
    """Pickle the entire :class:`ModelBundle`` to ``<root>/<run_id>.pkl``."""
    if not run_id or any(ch in run_id for ch in "/\\:*?\"<>|"):
        raise ArtifactError(f"run_id {run_id!r} is not filesystem-safe")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.pkl"
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_bundle_pkl(path: str | Path) -> ModelBundle:
    """Load a :class:`ModelBundle`` from a pickle file written by ``save_bundle_pkl``."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"no pickle bundle at {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, ModelBundle):
        raise ArtifactError(f"pickle at {path} does not contain a ModelBundle")
    return bundle


def load_bundle(directory: str | Path, *, verify_checksums: bool = True) -> ModelBundle:
    """Reload a saved bundle, verifying checksums and feature-order consistency."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise ArtifactError(f"no {MANIFEST_NAME} in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"unsupported artifact schema {manifest.get('artifact_schema_version')!r}"
        )

    if verify_checksums:
        for name, expected in manifest["files"].items():
            path = directory / name
            if not path.exists():
                raise ArtifactError(f"artifact file missing: {name}")
            if _sha256(path) != expected:
                raise ArtifactError(f"checksum mismatch for {name}: artifact was modified")

    preprocessor = FeaturePreprocessor.from_dict(
        json.loads((directory / PREPROCESSOR_NAME).read_text(encoding="utf-8"))
    )
    if list(preprocessor.feature_columns) != list(manifest["feature_columns"]):
        raise ArtifactError("preprocessor feature order does not match the manifest")

    from xgboost import XGBRegressor  # lazy: baselines never need xgboost

    models: dict[int, Any] = {}
    for horizon in manifest["trained_horizons"]:
        model = XGBRegressor()
        model.load_model(str(directory / _model_file(horizon)))
        models[int(horizon)] = model
    return ModelBundle(manifest=manifest, preprocessor=preprocessor, models=models)


def version_mismatches(manifest: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Libraries whose installed version differs from the training environment."""
    current = library_versions()
    saved = manifest.get("library_versions", {})
    return {
        name: (saved[name], current.get(name, "unknown"))
        for name in saved
        if saved[name] != current.get(name)
    }


def python_environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform()}


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactError",
    "ModelBundle",
    "UnsupportedHorizonError",
    "library_versions",
    "load_bundle",
    "load_bundle_pkl",
    "save_bundle",
    "save_bundle_pkl",
    "target_definition",
    "version_mismatches",
]
