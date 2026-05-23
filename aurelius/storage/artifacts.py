"""Pluggable artifact storage for Aurelius model binaries and large reports.

Postgres stores *metadata only* (see aurelius.database.store model registry);
the actual model files / large reports live behind this generic interface so a
deployment can use local disk, any S3-compatible object store (AWS S3, Cloudflare
R2, MinIO, Supabase Storage), or be swapped without touching call sites.

Selection is driven by a single URI (env var ARTIFACT_STORE_URI):

    file:///abs/path/to/artifacts        local filesystem (default)
    file://./data/artifacts              local, relative to cwd
    s3://bucket/prefix                   S3-compatible (boto3 required)

S3-compatible providers that expose a custom endpoint (R2, MinIO, Supabase
Storage) are supported via the AWS_ENDPOINT_URL env var. No provider-specific
SDK is required — only boto3, and only when an s3:// URI is used.

The interface is intentionally tiny: put / get / exists / delete. Keys are
caller-chosen logical paths (e.g. "models/price/<model_id>/model.joblib"); the
returned artifact_uri is what gets persisted in the DB.
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_DIR = Path(__file__).parent.parent / "data" / "artifacts"


class ArtifactStore(ABC):
    """Abstract artifact store. Implementations must be deployment-portable."""

    @abstractmethod
    def put(self, key: str, local_path: str) -> str:
        """Upload local_path under logical key. Returns the artifact URI."""

    @abstractmethod
    def get(self, uri: str, local_path: str) -> str:
        """Download the artifact at uri to local_path. Returns local_path."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Return True if the artifact exists."""

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete the artifact (best-effort; no error if already absent)."""


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed artifact store. URIs are file://<absolute path>."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else _DEFAULT_LOCAL_DIR.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        # Defend against absolute keys / traversal escaping base_dir.
        rel = Path(key.lstrip("/"))
        dest = (self.base_dir / rel).resolve()
        if self.base_dir not in dest.parents and dest != self.base_dir:
            raise ValueError(f"artifact key escapes base dir: {key}")
        return dest

    @staticmethod
    def _path_from_uri(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in ("file", ""):
            raise ValueError(f"LocalArtifactStore cannot handle URI: {uri}")
        # file:///abs -> /abs ; bare path -> as-is
        return Path(parsed.path if parsed.scheme == "file" else uri)

    def put(self, key: str, local_path: str) -> str:
        dest = self._path_for_key(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"file://{dest}"

    def get(self, uri: str, local_path: str) -> str:
        src = self._path_from_uri(uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path

    def exists(self, uri: str) -> bool:
        return self._path_from_uri(uri).exists()

    def delete(self, uri: str) -> None:
        p = self._path_from_uri(uri)
        if p.exists():
            p.unlink()


class S3ArtifactStore(ArtifactStore):
    """S3-compatible artifact store (AWS S3, R2, MinIO, Supabase Storage).

    Requires boto3. Honors AWS_ENDPOINT_URL for non-AWS S3-compatible providers.
    URIs are s3://<bucket>/<key>.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        try:
            import boto3  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "S3ArtifactStore requires boto3 (pip install boto3). "
                "Use a file:// ARTIFACT_STORE_URI for local/dev."
            ) from exc
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
        self._client = boto3.client("s3", endpoint_url=endpoint)

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, local_path: str) -> str:
        full = self._full_key(key)
        self._client.upload_file(local_path, self.bucket, full)
        return f"s3://{self.bucket}/{full}"

    def get(self, uri: str, local_path: str) -> str:
        parsed = urlparse(uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(parsed.netloc, parsed.path.lstrip("/"), local_path)
        return local_path

    def exists(self, uri: str) -> bool:
        from botocore.exceptions import ClientError

        parsed = urlparse(uri)
        try:
            self._client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            return True
        except ClientError:
            return False

    def delete(self, uri: str) -> None:
        parsed = urlparse(uri)
        self._client.delete_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))


def get_artifact_store(uri: Optional[str] = None) -> ArtifactStore:
    """Return the configured artifact store.

    Args:
        uri: Override URI. Defaults to ARTIFACT_STORE_URI env var, then a local
             file store at <package>/data/artifacts.
    """
    uri = uri or os.environ.get("ARTIFACT_STORE_URI", "")
    if not uri:
        return LocalArtifactStore()

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3ArtifactStore(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    if parsed.scheme in ("file", ""):
        base = parsed.path if parsed.scheme == "file" else uri
        return LocalArtifactStore(base_dir=base or None)
    raise ValueError(f"Unsupported ARTIFACT_STORE_URI scheme: {parsed.scheme!r}")
