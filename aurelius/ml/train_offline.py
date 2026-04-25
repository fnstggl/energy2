"""Offline ML training CLI for Aurelius.

Usage:
    python -m aurelius.ml.train_offline [options]

Options:
    --input PATH      Input JSONL file (default: aurelius/data/post_execution/post_execution_records.jsonl)
    --outdir PATH     Output directory for artifacts (default: aurelius/data/ml_artifacts/)
    --seed INT        Random seed for reproducibility (default: 1337)
    --overwrite       Allow overwriting existing artifact files
    --test            Run inline validation tests

This CLI:
- Loads PostExecutionRecord JSONL from disk
- Trains all ML estimation models
- Writes versioned artifacts to disk
- Produces a manifest for reproducibility

CRITICAL: This is OFFLINE ONLY.
ML outputs are advisory estimates.
They do NOT affect execution or grant permissions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .dataset import (
    load_post_execution_records,
    extract_training_dataset,
    compute_dataset_hash,
    get_default_post_execution_path,
)
from .artifacts import (
    ArtifactWriter,
    get_default_artifact_dir,
    generate_timestamp_utc,
)
from .trainers import (
    train_forecast_corrections,
    train_error_models,
    generate_uncertainty_rules,
    train_savings_model,
    train_risk_priors,
    train_savings_model_lgbm,
    train_risk_priors_lgbm,
    _MIN_LGBM_RECORDS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# MANIFEST GENERATION
# ============================================================================

def generate_manifest(
    dataset_path: str,
    dataset_hash: str,
    artifact_filenames: dict[str, str],
    seed: int,
) -> dict:
    """Generate manifest for reproducibility.

    Args:
        dataset_path: Path to input JSONL
        dataset_hash: SHA256 hash of input file
        artifact_filenames: Mapping of artifact type -> filename
        seed: Random seed used

    Returns:
        Manifest dictionary
    """
    return {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "dataset": {
            "path": dataset_path,
            "sha256": dataset_hash,
        },
        "artifacts": artifact_filenames,
        "seed": seed,
        "notes": "Offline advisory estimates only. No execution behavior changes.",
    }


# ============================================================================
# AUDIT LOGGING
# ============================================================================

def emit_audit_start(input_path: str, outdir: str, seed: int) -> None:
    """Emit start audit log."""
    audit = {
        "event": "offline_ml_training_started",
        "input_path": input_path,
        "outdir": outdir,
        "seed": seed,
    }
    logger.info(f"AUDIT: {json.dumps(audit)}")


def emit_audit_end(
    records_read: int,
    records_used: int,
    artifacts_written: list[str],
    manifest_path: str,
) -> None:
    """Emit end audit log."""
    audit = {
        "event": "offline_ml_training_completed",
        "records_read": records_read,
        "records_used": records_used,
        "artifacts_written": artifacts_written,
        "manifest_path": manifest_path,
    }
    logger.info(f"AUDIT: {json.dumps(audit)}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training(
    input_path: Path,
    output_dir: Path,
    seed: int,
    overwrite: bool,
    min_records: int = _MIN_LGBM_RECORDS,
    use_lgbm: bool = True,
) -> bool:
    """Run the complete offline training pipeline.

    Args:
        input_path: Path to PostExecutionRecord JSONL
        output_dir: Directory for output artifacts
        seed: Random seed (for determinism)
        overwrite: Whether to overwrite existing files
        min_records: Minimum records required for LightGBM training
        use_lgbm: If True, use LightGBM for savings and risk models

    Returns:
        True if successful, False otherwise
    """
    # Emit start audit
    emit_audit_start(str(input_path), str(output_dir), seed)

    # Check input file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Training cannot proceed without PostExecutionRecord data.")
        return False

    # Load records
    logger.info(f"Loading records from: {input_path}")
    raw_records = load_post_execution_records(input_path)

    if not raw_records:
        logger.error("No records found in input file. Training cannot proceed.")
        return False

    records_read = len(raw_records)
    logger.info(f"Loaded {records_read} raw records")

    # Extract training dataset
    training_records = extract_training_dataset(raw_records)
    records_used = len(training_records)
    logger.info(f"Extracted {records_used} training records")

    # Compute dataset hash for reproducibility
    dataset_hash = compute_dataset_hash(input_path)
    logger.info(f"Dataset hash: {dataset_hash[:16]}...")

    # Initialize artifact writer
    writer = ArtifactWriter(output_dir=output_dir, overwrite=overwrite)

    # Train all models
    logger.info("\n--- Training Models ---")

    # 1. Forecast corrections
    logger.info("Training forecast corrections...")
    forecast_corrections = train_forecast_corrections(training_records)
    writer.write("forecast_corrections_v1.json", forecast_corrections)
    logger.info(f"  → {len(forecast_corrections['buckets'])} buckets")

    # 2. Error models
    logger.info("Training error models...")
    error_models = train_error_models(training_records)
    writer.write("error_models_v1.json", error_models)
    logger.info(f"  → {len(error_models['buckets'])} buckets")

    # 3. Uncertainty rules
    logger.info("Generating uncertainty rules...")
    uncertainty_rules = generate_uncertainty_rules(error_models)
    writer.write("uncertainty_rules_v1.json", uncertainty_rules)
    logger.info(f"  → {len(uncertainty_rules['rules'])} rules")

    # 4. Savings model (LightGBM if available and sufficient data)
    if use_lgbm:
        logger.info("Training savings model (LightGBM)...")
        savings_model = train_savings_model_lgbm(
            training_records, seed=seed, min_records=min_records
        )
        method = savings_model.get("method", "unknown")
        logger.info(f"  → method={method}")
        if "metrics" in savings_model:
            m = savings_model["metrics"]
            logger.info(
                f"  → model_rmse={m.get('model_rmse_holdout', 'n/a')}, "
                f"naive_rmse={m.get('naive_mean_rmse_holdout', 'n/a')}, "
                f"beats_naive={m.get('beats_naive_baseline', 'n/a')}"
            )
        elif "buckets" in savings_model:
            logger.info(f"  → {len(savings_model['buckets'])} buckets (fallback)")
    else:
        logger.info("Training savings model (bucketed stats)...")
        savings_model = train_savings_model(training_records)
        logger.info(f"  → {len(savings_model['buckets'])} buckets")
    writer.write("savings_model_v1.json", savings_model)

    # 5. Risk priors (LightGBM if available and sufficient data)
    if use_lgbm:
        logger.info("Training risk priors (LightGBM)...")
        risk_priors = train_risk_priors_lgbm(
            training_records, error_models, seed=seed, min_records=min_records
        )
        method = risk_priors.get("method", "unknown")
        logger.info(f"  → method={method}")
        if "metrics" in risk_priors:
            m = risk_priors["metrics"]
            logger.info(
                f"  → model_logloss={m.get('model_logloss_holdout', 'n/a')}, "
                f"beats_naive={m.get('beats_naive_baseline', 'n/a')}"
            )
        elif "buckets" in risk_priors:
            logger.info(f"  → {len(risk_priors['buckets'])} buckets (fallback)")
    else:
        logger.info("Training risk priors (empirical)...")
        risk_priors = train_risk_priors(training_records, error_models)
        logger.info(f"  → {len(risk_priors['buckets'])} buckets")
    writer.write("risk_priors_v1.json", risk_priors)

    # 6. Manifest
    logger.info("Writing manifest...")
    artifact_filenames = {
        "forecast_corrections": "forecast_corrections_v1.json",
        "error_models": "error_models_v1.json",
        "uncertainty_rules": "uncertainty_rules_v1.json",
        "savings_model": "savings_model_v1.json",
        "risk_priors": "risk_priors_v1.json",
    }
    manifest = generate_manifest(
        dataset_path=str(input_path),
        dataset_hash=dataset_hash,
        artifact_filenames=artifact_filenames,
        seed=seed,
    )
    writer.write("manifest_v1.json", manifest)

    # Summary
    logger.info("\n--- Training Complete ---")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Artifacts written: {len(writer.written_files)}")
    for f in writer.written_files:
        logger.info(f"  - {f}")

    # Emit end audit
    emit_audit_end(
        records_read=records_read,
        records_used=records_used,
        artifacts_written=writer.written_files,
        manifest_path=str(output_dir / "manifest_v1.json"),
    )

    return True


# ============================================================================
# INLINE TESTS
# ============================================================================

def run_tests() -> bool:
    """Run inline validation tests."""
    import tempfile

    print("=" * 60)
    print("Offline ML Training Inline Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Dataset hash stability
    print("\n[Test 1] Dataset hash stability")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text('{"job_id": "test"}\n')
            hash1 = compute_dataset_hash(path)
            hash2 = compute_dataset_hash(path)
            assert hash1 == hash2, "Hash should be stable"
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 2: Empty input handling
    print("\n[Test 2] Empty input handling")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "empty.jsonl"
            input_path.touch()  # Create empty file
            output_dir = Path(tmpdir) / "artifacts"

            # Should handle empty gracefully (return False, no crash)
            result = run_training(input_path, output_dir, seed=1337, overwrite=True)
            assert result is False, "Should return False for empty input"
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 3: Missing input handling
    print("\n[Test 3] Missing input handling")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "nonexistent.jsonl"
            output_dir = Path(tmpdir) / "artifacts"

            result = run_training(input_path, output_dir, seed=1337, overwrite=True)
            assert result is False, "Should return False for missing input"
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 4: Full training pipeline
    print("\n[Test 4] Full training pipeline")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test input
            input_path = Path(tmpdir) / "records.jsonl"
            test_records = [
                {
                    "job_id": "job-1",
                    "region": "us-east",
                    "optimized_start_time": "2024-01-15T10:00:00Z",
                    "forecast_energy_cost_p50": 100.0,
                    "forecast_energy_cost_p90": 120.0,
                    "energy_cost_p50_error": 5.0,
                    "energy_cost_p90_covered": True,
                    "realized_savings": 10.0,
                    "decision_outcome_label": "good_decision",
                    "constraint_profile": "batch_optimized",
                },
                {
                    "job_id": "job-2",
                    "region": "us-east",
                    "optimized_start_time": "2024-01-15T11:00:00Z",
                    "forecast_energy_cost_p50": 100.0,
                    "energy_cost_p50_error": -3.0,
                    "energy_cost_p90_covered": True,
                    "realized_savings": 15.0,
                    "decision_outcome_label": "good_decision",
                    "constraint_profile": "batch_optimized",
                },
            ]
            with open(input_path, "w") as f:
                for record in test_records:
                    f.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "artifacts"

            result = run_training(input_path, output_dir, seed=1337, overwrite=True)
            assert result is True, "Training should succeed"

            # Check all artifacts exist
            expected_files = [
                "forecast_corrections_v1.json",
                "error_models_v1.json",
                "uncertainty_rules_v1.json",
                "savings_model_v1.json",
                "risk_priors_v1.json",
                "manifest_v1.json",
            ]
            for f in expected_files:
                assert (output_dir / f).exists(), f"Missing artifact: {f}"

            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 5: Overwrite protection
    print("\n[Test 5] Overwrite protection")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "records.jsonl"
            input_path.write_text('{"job_id": "test", "region": "us-east"}\n')
            output_dir = Path(tmpdir) / "artifacts"

            # First run
            run_training(input_path, output_dir, seed=1337, overwrite=True)

            # Create a marker to check if file was overwritten
            manifest_path = output_dir / "manifest_v1.json"
            original_content = manifest_path.read_text()

            # Second run without overwrite
            run_training(input_path, output_dir, seed=1337, overwrite=False)

            # Content should be unchanged
            assert manifest_path.read_text() == original_content
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 6: Null field handling
    print("\n[Test 6] Null field handling")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "records.jsonl"
            # Records with many null fields
            test_records = [
                {"job_id": "job-1", "region": "us-east"},
                {"job_id": "job-2"},  # No region
                {},  # Empty
            ]
            with open(input_path, "w") as f:
                for record in test_records:
                    f.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "artifacts"
            result = run_training(input_path, output_dir, seed=1337, overwrite=True)
            assert result is True, "Should handle null fields"
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 7: Determinism
    print("\n[Test 7] Determinism")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "records.jsonl"
            input_path.write_text(
                '{"job_id": "job-1", "region": "us-east", "energy_cost_p50_error": 5.0}\n'
                '{"job_id": "job-2", "region": "us-east", "energy_cost_p50_error": -3.0}\n'
            )

            output_dir1 = Path(tmpdir) / "artifacts1"
            output_dir2 = Path(tmpdir) / "artifacts2"

            run_training(input_path, output_dir1, seed=1337, overwrite=True)
            run_training(input_path, output_dir2, seed=1337, overwrite=True)

            # Compare forecast corrections (ignoring timestamp)
            with open(output_dir1 / "forecast_corrections_v1.json") as f:
                fc1 = json.load(f)
            with open(output_dir2 / "forecast_corrections_v1.json") as f:
                fc2 = json.load(f)

            assert fc1["buckets"] == fc2["buckets"], "Buckets should be identical"
            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    # Test 8: Artifact schema validation
    print("\n[Test 8] Artifact schema validation")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "records.jsonl"
            input_path.write_text(
                '{"job_id": "job-1", "region": "us-east", "realized_savings": 10.0, '
                '"constraint_profile": "batch_optimized"}\n'
            )
            output_dir = Path(tmpdir) / "artifacts"
            run_training(input_path, output_dir, seed=1337, overwrite=True)

            # Check savings model schema
            with open(output_dir / "savings_model_v1.json") as f:
                sm = json.load(f)
            assert "version" in sm
            assert sm["version"] == 1
            assert "method" in sm
            assert "buckets" in sm
            if sm["buckets"]:
                bucket = sm["buckets"][0]
                assert "region" in bucket
                assert "mean_savings" in bucket
                assert "n" in bucket

            # Check manifest schema
            with open(output_dir / "manifest_v1.json") as f:
                manifest = json.load(f)
            assert "version" in manifest
            assert "dataset" in manifest
            assert "sha256" in manifest["dataset"]
            assert "artifacts" in manifest
            assert "seed" in manifest

            print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All 8 tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)

    return all_passed


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Offline ML training for Aurelius",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility (default: 1337)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artifact files",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inline validation tests",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=_MIN_LGBM_RECORDS,
        help=f"Minimum labelled records for LightGBM training (default: {_MIN_LGBM_RECORDS}). "
             "Fewer records fall back to bucketed stats.",
    )
    parser.add_argument(
        "--no-lgbm",
        action="store_true",
        help="Disable LightGBM and use bucketed stats for savings/risk models",
    )

    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Default paths
    input_path = args.input or get_default_post_execution_path()
    output_dir = args.outdir or get_default_artifact_dir()

    logger.info("=" * 60)
    logger.info("Aurelius Offline ML Training")
    logger.info("=" * 60)
    logger.info(f"Input:       {input_path}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Seed:        {args.seed}")
    logger.info(f"Overwrite:   {args.overwrite}")
    logger.info(f"Min records: {args.min_records}")
    logger.info(f"Use LightGBM: {not args.no_lgbm}")
    logger.info("")

    success = run_training(
        input_path=input_path,
        output_dir=output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
        min_records=args.min_records,
        use_lgbm=not args.no_lgbm,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
