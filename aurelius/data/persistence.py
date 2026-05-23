"""Minimal JSONL persistence for Aurelius.

Provides append-only JSONL file writing with:
- No external dependencies
- No database
- No network calls
- Safe for air-gapped environments
- Failure-tolerant (exceptions swallowed, logged)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_default_post_execution_path() -> Path:
    """Get the default path for post-execution records.

    Returns:
        Path to aurelius/data/post_execution/post_execution_records.jsonl
    """
    # Get the aurelius package directory
    package_dir = Path(__file__).parent.parent
    return package_dir / "data" / "post_execution" / "post_execution_records.jsonl"


class JSONLWriter:
    """Append-only JSONL file writer.

    Writes JSON records to a JSONL file (one JSON object per line).
    Creates directories if they don't exist.
    Never overwrites existing data - append only.
    Failure-tolerant: exceptions are caught and logged.

    Usage:
        writer = JSONLWriter("/path/to/file.jsonl")
        writer.append({"key": "value"})
    """

    def __init__(self, path: str | Path):
        """Initialize the JSONL writer.

        Args:
            path: Path to the JSONL file
        """
        self.path = Path(path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create parent directories if they don't exist."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.debug(f"Failed to create directory {self.path.parent}: {e}")

    def append(self, record: dict[str, Any]) -> bool:
        """Append a record to the JSONL file.

        Args:
            record: Dictionary to write as JSON

        Returns:
            True if write succeeded, False otherwise

        Note:
            Exceptions are caught and logged - this method never raises.
            This ensures post-execution recording never affects execution.
        """
        try:
            line = json.dumps(record, default=str) + "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
            return True
        except Exception as e:
            logger.debug(f"Failed to write record to {self.path}: {e}")
            return False

    def read_all(self) -> list[dict[str, Any]]:
        """Read all records from the JSONL file.

        Returns:
            List of dictionaries (empty list if file doesn't exist or on error)
        """
        try:
            if not self.path.exists():
                return []
            records = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        except Exception as e:
            logger.debug(f"Failed to read records from {self.path}: {e}")
            return []


# Inline tests
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("JSONLWriter Inline Tests")
    print("=" * 60)

    # Test 1: Basic write and read
    print("\n[Test 1] Basic write and read")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        writer = JSONLWriter(path)

        record1 = {"id": 1, "name": "test1"}
        record2 = {"id": 2, "name": "test2"}

        assert writer.append(record1) is True
        assert writer.append(record2) is True

        records = writer.read_all()
        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2
        print("  PASSED: Basic write/read works")

    # Test 2: Nested directory creation
    print("\n[Test 2] Nested directory creation")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "a" / "b" / "c" / "test.jsonl"
        writer = JSONLWriter(path)

        assert writer.append({"nested": True}) is True
        records = writer.read_all()
        assert len(records) == 1
        print("  PASSED: Nested directories created")

    # Test 3: Append-only behavior
    print("\n[Test 3] Append-only behavior")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "append.jsonl"

        # First writer
        writer1 = JSONLWriter(path)
        writer1.append({"batch": 1})
        writer1.append({"batch": 1})

        # Second writer (simulating restart)
        writer2 = JSONLWriter(path)
        writer2.append({"batch": 2})

        records = writer2.read_all()
        assert len(records) == 3
        assert records[0]["batch"] == 1
        assert records[2]["batch"] == 2
        print("  PASSED: Append-only behavior works")

    # Test 4: Read non-existent file
    print("\n[Test 4] Read non-existent file")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.jsonl"
        writer = JSONLWriter(path)
        records = writer.read_all()
        assert records == []
        print("  PASSED: Non-existent file returns empty list")

    # Test 5: Default path function
    print("\n[Test 5] Default path function")
    default_path = get_default_post_execution_path()
    assert "post_execution" in str(default_path)
    assert str(default_path).endswith(".jsonl")
    print(f"  PASSED: Default path = {default_path}")

    # Test 6: Datetime serialization (via default=str)
    print("\n[Test 6] Datetime serialization")
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "datetime.jsonl"
        writer = JSONLWriter(path)

        now = datetime.utcnow()
        assert writer.append({"timestamp": now}) is True
        records = writer.read_all()
        assert len(records) == 1
        # Datetime should be serialized as string
        assert isinstance(records[0]["timestamp"], str)
        print("  PASSED: Datetime serialization works")

    print("\n" + "=" * 60)
    print("All 6 tests passed!")
    print("=" * 60)
