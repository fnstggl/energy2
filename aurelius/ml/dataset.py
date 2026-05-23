"""Dataset loading and extraction for offline ML training.

Reads PostExecutionRecord JSONL and extracts structured training data.
Handles missing fields gracefully without crashing.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_default_post_execution_path() -> Path:
    """Get the default path for post-execution records."""
    package_dir = Path(__file__).parent.parent
    return package_dir / "data" / "post_execution" / "post_execution_records.jsonl"


def compute_dataset_hash(path: Path) -> str:
    """Compute SHA256 hash of the dataset file.

    Args:
        path: Path to JSONL file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def load_post_execution_records(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load all PostExecutionRecord entries from JSONL.

    Args:
        path: Path to JSONL file (uses default if None)

    Returns:
        List of record dictionaries (empty if file missing/empty)
    """
    file_path = path or get_default_post_execution_path()

    if not file_path.exists():
        logger.warning(f"Post-execution records file not found: {file_path}")
        return []

    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping malformed line {line_num}: {e}")
    except Exception as e:
        logger.warning(f"Error reading post-execution records: {e}")
        return []

    return records


@dataclass
class TrainingRecord:
    """Structured training record extracted from PostExecutionRecord.

    All fields are optional to handle missing data gracefully.
    """
    job_id: Optional[str] = None
    region: Optional[str] = None
    hour_utc: Optional[int] = None
    baseline_start_time: Optional[str] = None
    optimized_start_time: Optional[str] = None
    realized_start_time: Optional[str] = None
    realized_energy_price: Optional[float] = None
    realized_carbon_intensity: Optional[float] = None
    forecast_energy_cost_p50: Optional[float] = None
    forecast_energy_cost_p90: Optional[float] = None
    forecast_energy_cost_baseline: Optional[float] = None
    forecast_carbon_p50: Optional[float] = None
    forecast_carbon_p90: Optional[float] = None
    forecast_carbon_baseline: Optional[float] = None
    energy_cost_p50_error: Optional[float] = None
    energy_cost_p90_covered: Optional[bool] = None
    carbon_p50_error: Optional[float] = None
    carbon_p90_covered: Optional[bool] = None
    realized_savings: Optional[float] = None
    decision_outcome_label: Optional[str] = None
    execution_mode: Optional[str] = None
    constraint_profile: Optional[str] = None


def _parse_hour_utc(timestamp_str: Optional[str]) -> Optional[int]:
    """Extract hour (UTC) from ISO timestamp string."""
    if not timestamp_str:
        return None
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.hour
    except (ValueError, AttributeError):
        return None


def extract_training_dataset(
    records: list[dict[str, Any]],
) -> list[TrainingRecord]:
    """Extract structured training records from raw JSONL records.

    Handles missing fields gracefully - no crashes on nulls.
    Sorts records for deterministic processing.

    Args:
        records: Raw records from load_post_execution_records()

    Returns:
        List of TrainingRecord objects, sorted by job_id
    """
    training_records = []

    for raw in records:
        # Extract hour from optimized_start_time for bucketing
        hour_utc = _parse_hour_utc(raw.get("optimized_start_time"))

        record = TrainingRecord(
            job_id=raw.get("job_id"),
            region=raw.get("region"),
            hour_utc=hour_utc,
            baseline_start_time=raw.get("baseline_start_time"),
            optimized_start_time=raw.get("optimized_start_time"),
            realized_start_time=raw.get("realized_start_time"),
            realized_energy_price=raw.get("realized_energy_price"),
            realized_carbon_intensity=raw.get("realized_carbon_intensity"),
            forecast_energy_cost_p50=raw.get("forecast_energy_cost_p50"),
            forecast_energy_cost_p90=raw.get("forecast_energy_cost_p90"),
            forecast_energy_cost_baseline=raw.get("forecast_energy_cost_baseline"),
            forecast_carbon_p50=raw.get("forecast_carbon_p50"),
            forecast_carbon_p90=raw.get("forecast_carbon_p90"),
            forecast_carbon_baseline=raw.get("forecast_carbon_baseline"),
            energy_cost_p50_error=raw.get("energy_cost_p50_error"),
            energy_cost_p90_covered=raw.get("energy_cost_p90_covered"),
            carbon_p50_error=raw.get("carbon_p50_error"),
            carbon_p90_covered=raw.get("carbon_p90_covered"),
            realized_savings=raw.get("realized_savings"),
            decision_outcome_label=raw.get("decision_outcome_label"),
            execution_mode=raw.get("execution_mode"),
            constraint_profile=raw.get("constraint_profile"),
        )
        training_records.append(record)

    # Sort for deterministic processing
    training_records.sort(key=lambda r: (r.job_id or "", r.optimized_start_time or ""))

    return training_records


# Inline tests
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Dataset Module Inline Tests")
    print("=" * 60)

    # Test 1: compute_dataset_hash is stable
    print("\n[Test 1] Dataset hash stability")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        path.write_text('{"job_id": "test"}\n{"job_id": "test2"}\n')
        hash1 = compute_dataset_hash(path)
        hash2 = compute_dataset_hash(path)
        assert hash1 == hash2, "Hash should be stable"
        assert len(hash1) == 64, "Should be SHA256 hex"
        print(f"  PASSED: hash={hash1[:16]}...")

    # Test 2: load_post_execution_records handles missing file
    print("\n[Test 2] Missing file handling")
    records = load_post_execution_records(Path("/nonexistent/path.jsonl"))
    assert records == [], "Should return empty list for missing file"
    print("  PASSED: Returns empty list for missing file")

    # Test 3: load_post_execution_records parses valid JSONL
    print("\n[Test 3] JSONL parsing")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        path.write_text(
            '{"job_id": "job-1", "region": "us-east"}\n'
            '{"job_id": "job-2", "region": "us-west"}\n'
        )
        records = load_post_execution_records(path)
        assert len(records) == 2
        assert records[0]["job_id"] == "job-1"
        print(f"  PASSED: Loaded {len(records)} records")

    # Test 4: extract_training_dataset handles nulls
    print("\n[Test 4] Null field handling")
    raw_records = [
        {"job_id": "job-1", "region": "us-east"},
        {"job_id": "job-2"},  # Missing region
        {},  # All missing
    ]
    training = extract_training_dataset(raw_records)
    assert len(training) == 3
    assert training[0].region is None or training[0].region == "us-east"
    print("  PASSED: Null fields handled gracefully")

    # Test 5: Deterministic sorting
    print("\n[Test 5] Deterministic sorting")
    raw_records = [
        {"job_id": "job-z", "optimized_start_time": "2024-01-01T10:00:00Z"},
        {"job_id": "job-a", "optimized_start_time": "2024-01-01T12:00:00Z"},
        {"job_id": "job-m", "optimized_start_time": "2024-01-01T08:00:00Z"},
    ]
    training1 = extract_training_dataset(raw_records)
    training2 = extract_training_dataset(raw_records)
    assert [r.job_id for r in training1] == [r.job_id for r in training2]
    assert training1[0].job_id == "job-a"  # Sorted by job_id
    print("  PASSED: Sorting is deterministic")

    # Test 6: Hour extraction
    print("\n[Test 6] Hour UTC extraction")
    raw_records = [
        {"job_id": "job-1", "optimized_start_time": "2024-01-15T14:30:00Z"},
    ]
    training = extract_training_dataset(raw_records)
    assert training[0].hour_utc == 14
    print(f"  PASSED: Extracted hour={training[0].hour_utc}")

    print("\n" + "=" * 60)
    print("All 6 tests passed!")
    print("=" * 60)
