"""Versioned save/load for fitted Aurelius quantile forecasters.

Stores LightGBM model binaries (via joblib) alongside a JSON metadata manifest.
Each save creates a timestamped version so the full history is preserved.
A "latest" symlink always points to the most recently promoted model.

Directory layout:
    <store_root>/
        price/
            v_20240115T123456/
                model.joblib     ← serialized PriceQuantileForecaster
                metadata.json    ← training metadata + eval metrics
            active -> v_20240115T123456   ← symlink to current active version
        carbon/
            v_20240115T130000/
                model.joblib
                metadata.json
            active -> v_20240115T130000

Usage:
    store = ModelStore()

    # Save a newly trained model (not yet active)
    version_id = store.save(forecaster, model_type="price", metadata=meta_dict)

    # Promote to active after holdout validation confirms improvement
    store.promote(model_type="price", version_id=version_id)

    # Load the current active model
    forecaster = store.load_active(model_type="price", cls=PriceQuantileForecaster)

    # List all available versions
    versions = store.list_versions(model_type="price")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

_TIMESTAMP_FMT = "%Y%m%dT%H%M%S"


def _get_default_store_root() -> Path:
    package_dir = Path(__file__).parent.parent
    return package_dir / "data" / "model_store"


def _version_id_from_now() -> str:
    return "v_" + datetime.utcnow().strftime(_TIMESTAMP_FMT)


class ModelStore:
    """Versioned store for fitted forecaster objects.

    Thread-safety: not thread-safe; use external locking for concurrent writers.
    """

    SUPPORTED_TYPES = ("price", "carbon")

    def __init__(self, store_root: Optional[Path] = None):
        self.store_root = store_root or _get_default_store_root()
        self.store_root.mkdir(parents=True, exist_ok=True)
        for mt in self.SUPPORTED_TYPES:
            (self.store_root / mt).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        forecaster: Any,
        model_type: str,
        metadata: Optional[dict] = None,
        version_id: Optional[str] = None,
    ) -> str:
        """Serialize a fitted forecaster to a new versioned directory.

        The saved model is NOT automatically made active. Call promote() to
        designate this version as the active model.

        Args:
            forecaster: A fitted PriceQuantileForecaster or CarbonQuantileForecaster.
            model_type: "price" or "carbon".
            metadata: Optional dict of metadata (eval metrics, training info).
                      Will be merged with auto-generated fields.
            version_id: Optional version string (e.g. "v_20240115T123456").
                        Auto-generated from current UTC time if None.

        Returns:
            The version_id string that was created.

        Raises:
            ValueError: If model_type not in SUPPORTED_TYPES.
            RuntimeError: If the forecaster is not fitted.
        """
        self._validate_model_type(model_type)

        if not getattr(forecaster, "is_fitted", False):
            raise RuntimeError(
                "Forecaster is not fitted; cannot save unfitted model to store"
            )

        version_id = version_id or ("v_" + datetime.utcnow().strftime(_TIMESTAMP_FMT))
        version_dir = self.store_root / model_type / version_id
        version_dir.mkdir(parents=True, exist_ok=False)  # fail if version already exists

        # Serialize model
        model_path = version_dir / "model.joblib"
        try:
            import joblib
            joblib.dump(forecaster, model_path)
        except Exception as exc:
            shutil.rmtree(version_dir)
            raise RuntimeError(f"Failed to serialize forecaster: {exc}") from exc

        # Build metadata
        meta = {
            "version_id": version_id,
            "model_type": model_type,
            "saved_at_utc": datetime.utcnow().isoformat() + "Z",
            "forecaster_class": type(forecaster).__name__,
            "is_active": False,
        }
        if metadata:
            meta.update(metadata)

        # Attach ModelMetadata if available
        fitted_meta = getattr(forecaster, "metadata", None)
        if fitted_meta is not None and hasattr(fitted_meta, "to_dict"):
            meta["model_metadata"] = fitted_meta.to_dict()

        meta_path = version_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        logger.info(f"ModelStore: saved {model_type} model as {version_id}")
        return version_id

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(self, model_type: str, version_id: str) -> None:
        """Promote a version to active.

        Updates (or creates) an 'active' symlink pointing to *version_id*.
        Also marks the version's metadata.json with is_active=True and
        stamps the previous active version with promoted_from.

        Args:
            model_type: "price" or "carbon".
            version_id: The version to promote.

        Raises:
            FileNotFoundError: If version_id does not exist.
        """
        self._validate_model_type(model_type)
        version_dir = self.store_root / model_type / version_id
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version {version_id!r} not found in {model_type} store"
            )

        active_link = self.store_root / model_type / "active"

        # Remove old symlink or directory entry named 'active'
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()

        # Create new symlink (relative, for portability)
        try:
            os.symlink(version_id, active_link)
        except NotImplementedError:
            # Fallback for systems without symlink support (rare): write a plain file
            active_link.write_text(version_id, encoding="utf-8")

        # Update metadata
        meta_path = version_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["is_active"] = True
            meta["promoted_at_utc"] = datetime.utcnow().isoformat() + "Z"
            meta_path.write_text(
                json.dumps(meta, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

        logger.info(f"ModelStore: promoted {model_type}/{version_id} to active")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_active(
        self,
        model_type: str,
        cls: Optional[Type] = None,
    ) -> Any:
        """Load the currently active forecaster.

        Args:
            model_type: "price" or "carbon".
            cls: Optional class for type checking after load.

        Returns:
            The deserialized forecaster object.

        Raises:
            FileNotFoundError: If no active version is set.
            RuntimeError: If deserialization fails.
        """
        self._validate_model_type(model_type)
        version_id = self._resolve_active_version(model_type)
        return self.load_version(model_type, version_id, cls=cls)

    def load_version(
        self,
        model_type: str,
        version_id: str,
        cls: Optional[Type] = None,
    ) -> Any:
        """Load a specific version.

        Args:
            model_type: "price" or "carbon".
            version_id: The version directory name.
            cls: Optional class for type checking.

        Returns:
            The deserialized forecaster object.
        """
        self._validate_model_type(model_type)
        model_path = self.store_root / model_type / version_id / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            import joblib
            forecaster = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialize model: {exc}") from exc

        if cls is not None and not isinstance(forecaster, cls):
            raise TypeError(
                f"Loaded object is {type(forecaster).__name__!r}, expected {cls.__name__!r}"
            )

        logger.info(f"ModelStore: loaded {model_type}/{version_id}")
        return forecaster

    def load_metadata(self, model_type: str, version_id: str) -> dict:
        """Load the metadata dict for a specific version."""
        self._validate_model_type(model_type)
        meta_path = self.store_root / model_type / version_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load_active_metadata(self, model_type: str) -> dict:
        """Load metadata for the currently active version."""
        version_id = self._resolve_active_version(model_type)
        return self.load_metadata(model_type, version_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_versions(self, model_type: str) -> list[str]:
        """Return sorted list of version IDs (oldest first)."""
        self._validate_model_type(model_type)
        type_dir = self.store_root / model_type
        versions = [
            d.name
            for d in sorted(type_dir.iterdir())
            if d.is_dir() and d.name.startswith("v_")
        ]
        return versions

    def get_active_version(self, model_type: str) -> Optional[str]:
        """Return the active version ID, or None if no active version exists."""
        try:
            return self._resolve_active_version(model_type)
        except FileNotFoundError:
            return None

    def has_active(self, model_type: str) -> bool:
        """Return True if an active version exists."""
        return self.get_active_version(model_type) is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_model_type(self, model_type: str) -> None:
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_TYPES}, got {model_type!r}"
            )

    def _resolve_active_version(self, model_type: str) -> str:
        """Resolve the active version ID.

        Handles both symlink (unix) and plain-file fallback (windows).
        """
        active_link = self.store_root / model_type / "active"

        if not active_link.exists() and not active_link.is_symlink():
            raise FileNotFoundError(
                f"No active model set for model_type={model_type!r}. "
                "Call promote() after saving a validated model."
            )

        # Symlink: resolve target name (not full path)
        if active_link.is_symlink():
            target = os.readlink(active_link)
            # readlink may return a relative path; that's what we stored
            return Path(target).name

        # Fallback: plain file containing version_id
        return active_link.read_text(encoding="utf-8").strip()
