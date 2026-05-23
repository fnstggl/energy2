"""Tests for aurelius.storage.artifacts (local artifact store + factory)."""
from __future__ import annotations

from pathlib import Path

import pytest

from aurelius.storage import LocalArtifactStore, get_artifact_store
from aurelius.storage.artifacts import S3ArtifactStore


@pytest.fixture
def local_store(tmp_path):
    return LocalArtifactStore(base_dir=str(tmp_path / "artifacts"))


@pytest.fixture
def src_file(tmp_path):
    p = tmp_path / "src.bin"
    p.write_bytes(b"model-bytes-123")
    return str(p)


class TestLocalArtifactStore:
    def test_put_returns_file_uri(self, local_store, src_file):
        uri = local_store.put("models/price/m1/model.joblib", src_file)
        assert uri.startswith("file://")
        assert local_store.exists(uri)

    def test_get_roundtrip(self, local_store, src_file, tmp_path):
        uri = local_store.put("models/m1/model.joblib", src_file)
        out = str(tmp_path / "out.bin")
        local_store.get(uri, out)
        assert Path(out).read_bytes() == b"model-bytes-123"

    def test_exists_false_for_missing(self, local_store):
        assert local_store.exists("file:///nonexistent/path.joblib") is False

    def test_delete(self, local_store, src_file):
        uri = local_store.put("m/model.joblib", src_file)
        assert local_store.exists(uri)
        local_store.delete(uri)
        assert not local_store.exists(uri)

    def test_delete_missing_is_noop(self, local_store):
        local_store.delete("file:///nope/x.joblib")  # must not raise

    def test_key_traversal_rejected(self, local_store, src_file):
        with pytest.raises(ValueError):
            local_store.put("../../etc/passwd", src_file)

    def test_nested_keys_create_dirs(self, local_store, src_file):
        uri = local_store.put("a/b/c/d/model.joblib", src_file)
        assert local_store.exists(uri)


class TestGetArtifactStoreFactory:
    def test_default_is_local(self, monkeypatch):
        monkeypatch.delenv("ARTIFACT_STORE_URI", raising=False)
        store = get_artifact_store()
        assert isinstance(store, LocalArtifactStore)

    def test_file_uri(self, tmp_path):
        store = get_artifact_store(f"file://{tmp_path}/arts")
        assert isinstance(store, LocalArtifactStore)

    def test_env_var_used(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARTIFACT_STORE_URI", f"file://{tmp_path}/fromenv")
        store = get_artifact_store()
        assert isinstance(store, LocalArtifactStore)

    def test_s3_uri_constructs_s3_store_or_raises_without_boto3(self, monkeypatch):
        # boto3 is not installed in CI by default; constructing should raise a
        # clear RuntimeError rather than silently misbehave.
        try:
            import boto3  # noqa: F401
            has_boto3 = True
        except ImportError:
            has_boto3 = False
        if has_boto3:
            store = get_artifact_store("s3://bucket/prefix")
            assert isinstance(store, S3ArtifactStore)
        else:
            with pytest.raises(RuntimeError):
                get_artifact_store("s3://bucket/prefix")

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError):
            get_artifact_store("ftp://host/path")
