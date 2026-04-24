"""Artifact writing and loading for offline ML pipeline.

Artifacts are versioned, immutable JSON files that contain
ML-derived estimates for offline use only.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_default_artifact_dir() -> Path:
    """Get the default directory for ML artifacts."""
    package_dir = Path(__file__).parent.parent
    return package_dir / "data" / "ml_artifacts"


def canonical_json_dumps(data: dict[str, Any]) -> str:
    """Produce deterministic JSON string.

    Uses sort_keys=True, minimal separators, no ASCII escaping.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        indent=2,  # Human-readable for artifacts
    )


class ArtifactWriter:
    """Writer for versioned ML artifacts.

    Writes JSON artifacts to a directory with version suffixes.
    Supports overwrite protection (default: do not overwrite).
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
    ):
        """Initialize artifact writer.

        Args:
            output_dir: Directory to write artifacts (uses default if None)
            overwrite: If True, allow overwriting existing files
        """
        self.output_dir = output_dir or get_default_artifact_dir()
        self.overwrite = overwrite
        self._ensure_directory()
        self.written_files: list[str] = []

    def _ensure_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        filename: str,
        data: dict[str, Any],
    ) -> bool:
        """Write an artifact to disk.

        Args:
            filename: Artifact filename (e.g., "forecast_corrections_v1.json")
            data: Dictionary to serialize as JSON

        Returns:
            True if written successfully, False if skipped (exists and no overwrite)
        """
        path = self.output_dir / filename

        if path.exists() and not self.overwrite:
            logger.warning(f"Artifact exists and overwrite=False: {path}")
            return False

        try:
            content = canonical_json_dumps(data)
            path.write_text(content, encoding="utf-8")
            self.written_files.append(filename)
            return True
        except Exception as e:
            logger.error(f"Failed to write artifact {filename}: {e}")
            return False


def load_artifact(
    filename: str,
    artifact_dir: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Load an artifact from disk.

    Args:
        filename: Artifact filename
        artifact_dir: Directory containing artifacts (uses default if None)

    Returns:
        Parsed JSON dictionary, or None if not found/invalid
    """
    directory = artifact_dir or get_default_artifact_dir()
    path = directory / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load artifact {filename}: {e}")
        return None


def generate_timestamp_utc() -> str:
    """Generate ISO timestamp in UTC."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# Inline tests
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Artifacts Module Inline Tests")
    print("=" * 60)

    # Test 1: canonical_json_dumps is deterministic
    print("\n[Test 1] Canonical JSON determinism")
    data1 = {"z": 1, "a": 2, "m": {"y": 3, "x": 4}}
    data2 = {"a": 2, "m": {"x": 4, "y": 3}, "z": 1}
    json1 = canonical_json_dumps(data1)
    json2 = canonical_json_dumps(data2)
    assert json1 == json2, "Canonical JSON should be deterministic"
    print("  PASSED: Identical JSON for reordered dicts")

    # Test 2: ArtifactWriter creates directory
    print("\n[Test 2] Directory creation")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir"
        writer = ArtifactWriter(output_dir=path)
        assert path.exists()
        print("  PASSED: Directory created")

    # Test 3: Write and load artifact
    print("\n[Test 3] Write and load")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        writer = ArtifactWriter(output_dir=path)
        data = {"version": 1, "test": "value"}
        assert writer.write("test_v1.json", data) is True
        loaded = load_artifact("test_v1.json", artifact_dir=path)
        assert loaded == data
        print("  PASSED: Write/load roundtrip works")

    # Test 4: Overwrite protection
    print("\n[Test 4] Overwrite protection")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        writer = ArtifactWriter(output_dir=path, overwrite=False)
        writer.write("test_v1.json", {"version": 1})
        result = writer.write("test_v1.json", {"version": 2})
        assert result is False, "Should not overwrite"
        loaded = load_artifact("test_v1.json", artifact_dir=path)
        assert loaded["version"] == 1, "Original should be preserved"
        print("  PASSED: Overwrite protection works")

    # Test 5: Overwrite when enabled
    print("\n[Test 5] Overwrite when enabled")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        writer = ArtifactWriter(output_dir=path, overwrite=True)
        writer.write("test_v1.json", {"version": 1})
        result = writer.write("test_v1.json", {"version": 2})
        assert result is True
        loaded = load_artifact("test_v1.json", artifact_dir=path)
        assert loaded["version"] == 2
        print("  PASSED: Overwrite works when enabled")

    # Test 6: Load nonexistent artifact
    print("\n[Test 6] Load nonexistent artifact")
    result = load_artifact("nonexistent.json", artifact_dir=Path("/tmp"))
    assert result is None
    print("  PASSED: Returns None for missing artifact")

    # Test 7: written_files tracking
    print("\n[Test 7] Written files tracking")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        writer = ArtifactWriter(output_dir=path)
        writer.write("a_v1.json", {"a": 1})
        writer.write("b_v1.json", {"b": 2})
        assert len(writer.written_files) == 2
        assert "a_v1.json" in writer.written_files
        print(f"  PASSED: Tracked {len(writer.written_files)} files")

    print("\n" + "=" * 60)
    print("All 7 tests passed!")
    print("=" * 60)
