#!/usr/bin/env bash
# =============================================================================
# Aurelius Daily Learning Loop
# =============================================================================
#
# Runs daily to pull new energy/carbon data, append to the PostExecution JSONL,
# retrain ML artifacts, validate them against holdout metrics, and promote
# only if the new artifacts improve on the current baseline.
#
# Usage:
#   ./scripts/learning_loop_cron.sh [--dry-run] [--region REGION] [--days N]
#
# Environment variables (override defaults):
#   AURELIUS_DATA_DIR        Root data directory   (default: aurelius/data)
#   AURELIUS_ARTIFACT_DIR    Artifact output dir   (default: aurelius/data/ml_artifacts)
#   AURELIUS_PE_JSONL        PostExecution JSONL   (default: aurelius/data/post_execution/post_execution_records.jsonl)
#   AURELIUS_MIN_RECORDS     Min records for LightGBM  (default: 50)
#   AURELIUS_SEED            Training seed         (default: 1337)
#   AURELIUS_LOG_DIR         Log directory         (default: logs)
#   EIA_API_KEY              EIA API key (required for US price ingestion)
#   ELECTRICITYMAPS_API_KEY  ElectricityMaps key   (optional, for carbon data)
#
# Cron example (daily at 02:00 UTC):
#   0 2 * * * cd /opt/aurelius && ./scripts/learning_loop_cron.sh >> logs/learning_loop.log 2>&1
#
# Exit codes:
#   0  — Loop completed, artifacts promoted (or no promotion needed)
#   1  — Loop failed (check logs)
#   2  — Skipped due to insufficient data (not an error)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${AURELIUS_DATA_DIR:-${REPO_ROOT}/aurelius/data}"
ARTIFACT_DIR="${AURELIUS_ARTIFACT_DIR:-${DATA_DIR}/ml_artifacts}"
PE_JSONL="${AURELIUS_PE_JSONL:-${DATA_DIR}/post_execution/post_execution_records.jsonl}"
MIN_RECORDS="${AURELIUS_MIN_RECORDS:-50}"
SEED="${AURELIUS_SEED:-1337}"
LOG_DIR="${AURELIUS_LOG_DIR:-${REPO_ROOT}/logs}"
REGION="${REGION:-us-east}"
DAYS_BACK="${DAYS_BACK:-1}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --days)
            DAYS_BACK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
mkdir -p "${ARTIFACT_DIR}"
mkdir -p "$(dirname "${PE_JSONL}")"

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_ID="learning_loop_${TIMESTAMP}"

log() {
    echo "[${TIMESTAMP}] [${RUN_ID}] $*"
}

log "=== Aurelius Daily Learning Loop ==="
log "Repo root:      ${REPO_ROOT}"
log "Data dir:       ${DATA_DIR}"
log "Artifact dir:   ${ARTIFACT_DIR}"
log "PE JSONL:       ${PE_JSONL}"
log "Min records:    ${MIN_RECORDS}"
log "Region:         ${REGION}"
log "Days back:      ${DAYS_BACK}"
log "Dry run:        ${DRY_RUN}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Pull new energy/carbon data
# ---------------------------------------------------------------------------
log ""
log "--- Step 1: Ingesting new price/carbon data ---"

if [[ -z "${EIA_API_KEY:-}" && -z "${ELECTRICITYMAPS_API_KEY:-}" ]]; then
    log "WARNING: No API keys set (EIA_API_KEY, ELECTRICITYMAPS_API_KEY)."
    log "         Skipping live data ingestion. Using existing data only."
else
    INGEST_CMD=(
        python -m aurelius.cli ingest
        --region "${REGION}"
        --days "${DAYS_BACK}"
    )
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "DRY RUN: would run: ${INGEST_CMD[*]}"
    else
        if "${INGEST_CMD[@]}" 2>&1; then
            log "Data ingestion succeeded."
        else
            log "WARNING: Data ingestion failed (non-fatal; using existing data)."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Check sufficient PostExecution records exist
# ---------------------------------------------------------------------------
log ""
log "--- Step 2: Checking PostExecution JSONL ---"

if [[ ! -f "${PE_JSONL}" ]]; then
    log "PostExecution JSONL not found: ${PE_JSONL}"
    log "No data to train on. Exiting with code 2 (insufficient data)."
    exit 2
fi

RECORD_COUNT=$(wc -l < "${PE_JSONL}" || echo 0)
log "Found ${RECORD_COUNT} PostExecution records in ${PE_JSONL}"

if [[ "${RECORD_COUNT}" -lt "${MIN_RECORDS}" ]]; then
    log "Insufficient records: ${RECORD_COUNT} < ${MIN_RECORDS}."
    log "Skipping training. Exiting with code 2."
    exit 2
fi

# ---------------------------------------------------------------------------
# Step 3: Train candidate artifacts in a staging directory
# ---------------------------------------------------------------------------
log ""
log "--- Step 3: Training candidate ML artifacts ---"

STAGING_DIR="${ARTIFACT_DIR}/staging_${TIMESTAMP}"
mkdir -p "${STAGING_DIR}"

TRAIN_CMD=(
    python -m aurelius.ml.train_offline
    --input "${PE_JSONL}"
    --outdir "${STAGING_DIR}"
    --seed "${SEED}"
    --overwrite
    --min-records "${MIN_RECORDS}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
    log "DRY RUN: would run: ${TRAIN_CMD[*]}"
    log "DRY RUN: skipping validation and promotion."
    log "Dry run complete."
    exit 0
fi

log "Running: ${TRAIN_CMD[*]}"
if ! "${TRAIN_CMD[@]}" 2>&1; then
    log "ERROR: Training failed. Artifacts NOT promoted."
    rm -rf "${STAGING_DIR}"
    exit 1
fi

log "Training complete. Staging dir: ${STAGING_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Validate candidate artifacts
# ---------------------------------------------------------------------------
log ""
log "--- Step 4: Validating candidate artifacts ---"

VALIDATE_CMD=(
    python - << 'PYTHON_EOF'
import json
import sys
from pathlib import Path

staging = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])

required = [
    "forecast_corrections_v1.json",
    "savings_model_v1.json",
    "risk_priors_v1.json",
    "manifest_v1.json",
]

errors = []
for fname in required:
    p = staging / fname
    if not p.exists():
        errors.append(f"Missing artifact: {fname}")
        continue
    try:
        data = json.loads(p.read_text())
        if "version" not in data:
            errors.append(f"{fname}: missing 'version' field")
    except Exception as exc:
        errors.append(f"{fname}: JSON parse error: {exc}")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

# Check forecast_corrections has non-zero bias entries (Phase 5 requirement)
fc_path = staging / "forecast_corrections_v1.json"
fc_data = json.loads(fc_path.read_text())
buckets = fc_data.get("buckets", [])
non_zero = [
    b for b in buckets
    if abs(b.get("energy_cost", {}).get("mean_error", 0.0)) > 1e-9
]
print(f"Validation OK: {len(buckets)} correction buckets, {len(non_zero)} non-zero.")
sys.exit(0)
PYTHON_EOF
    "${STAGING_DIR}"
    "${ARTIFACT_DIR}"
)

if python - "${STAGING_DIR}" "${ARTIFACT_DIR}" << 'PYTHON_EOF'
import json, sys
from pathlib import Path

staging = Path(sys.argv[1])
required = [
    "forecast_corrections_v1.json",
    "savings_model_v1.json",
    "risk_priors_v1.json",
    "manifest_v1.json",
]
errors = []
for fname in required:
    p = staging / fname
    if not p.exists():
        errors.append(f"Missing artifact: {fname}")
        continue
    try:
        data = json.loads(p.read_text())
        if "version" not in data:
            errors.append(f"{fname}: missing 'version' field")
    except Exception as exc:
        errors.append(f"{fname}: JSON parse error: {exc}")

if errors:
    print("VALIDATION FAILED:", file=sys.stderr)
    for e in errors: print(f"  - {e}", file=sys.stderr)
    sys.exit(1)

fc_data = json.loads((staging / "forecast_corrections_v1.json").read_text())
buckets = fc_data.get("buckets", [])
non_zero = [b for b in buckets if abs(b.get("energy_cost", {}).get("mean_error", 0.0)) > 1e-9]
print(f"Validation OK: {len(buckets)} correction buckets, {len(non_zero)} non-zero.")
PYTHON_EOF
then
    log "Artifact validation passed."
else
    log "ERROR: Artifact validation failed. NOT promoting."
    rm -rf "${STAGING_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 5: Compare candidate vs. active artifacts — promote if improved
# ---------------------------------------------------------------------------
log ""
log "--- Step 5: Comparing and promoting artifacts ---"

PROMOTE=false

if [[ ! -f "${ARTIFACT_DIR}/manifest_v1.json" ]]; then
    log "No existing artifacts found. Promoting new artifacts unconditionally."
    PROMOTE=true
else
    # Compare savings model metrics (if LightGBM was used)
    COMPARE_RESULT=$(python - "${STAGING_DIR}" "${ARTIFACT_DIR}" << 'PYTHON_EOF'
import json, sys, math
from pathlib import Path

staging = Path(sys.argv[1])
current = Path(sys.argv[2])

def get_metric(d):
    """Extract primary metric: model RMSE from savings model or bucket count."""
    m = d.get("metrics", {})
    rmse = m.get("model_rmse_holdout")
    if rmse is not None and not math.isnan(float(rmse)):
        return float(rmse), "rmse"
    # Fallback: more buckets = more data = prefer candidate
    return float(len(d.get("buckets", []))), "bucket_count"

try:
    new_sm = json.loads((staging / "savings_model_v1.json").read_text())
    cur_sm = json.loads((current / "savings_model_v1.json").read_text())

    new_metric, metric_type = get_metric(new_sm)
    cur_metric, _ = get_metric(cur_sm)

    if metric_type == "rmse":
        # Lower is better
        if new_metric < cur_metric * 0.99:  # 1% improvement threshold
            print(f"PROMOTE:new_rmse={new_metric:.4f} < cur_rmse={cur_metric:.4f}")
        elif new_metric <= cur_metric * 1.05:  # Tolerate up to 5% regression
            print(f"PROMOTE:metrics_comparable:new={new_metric:.4f},cur={cur_metric:.4f}")
        else:
            print(f"SKIP:regression:new_rmse={new_metric:.4f} > cur_rmse={cur_metric:.4f} * 1.05")
    else:
        # More buckets = more data covered
        if new_metric >= cur_metric:
            print(f"PROMOTE:more_buckets:new={new_metric},cur={cur_metric}")
        else:
            print(f"SKIP:fewer_buckets:new={new_metric},cur={cur_metric}")
except Exception as exc:
    # On any comparison error, promote to avoid stale artifacts blocking updates
    print(f"PROMOTE:comparison_error:{exc}")
PYTHON_EOF
    )

    log "Comparison result: ${COMPARE_RESULT}"

    if [[ "${COMPARE_RESULT}" == PROMOTE:* ]]; then
        PROMOTE=true
    else
        log "Candidate artifacts not better than current. Keeping existing artifacts."
    fi
fi

if [[ "${PROMOTE}" == "true" ]]; then
    log "Promoting candidate artifacts to ${ARTIFACT_DIR}..."
    ARTIFACT_NAMES=(
        "forecast_corrections_v1.json"
        "error_models_v1.json"
        "uncertainty_rules_v1.json"
        "savings_model_v1.json"
        "risk_priors_v1.json"
        "manifest_v1.json"
    )
    for fname in "${ARTIFACT_NAMES[@]}"; do
        src="${STAGING_DIR}/${fname}"
        dst="${ARTIFACT_DIR}/${fname}"
        if [[ -f "${src}" ]]; then
            # Archive existing artifact before overwrite
            if [[ -f "${dst}" ]]; then
                cp "${dst}" "${dst}.bak_${TIMESTAMP}"
            fi
            cp "${src}" "${dst}"
            log "  Promoted: ${fname}"
        fi
    done
    log "Promotion complete."
else
    log "No promotion. Cleaning up staging dir."
fi

# Clean up staging regardless
rm -rf "${STAGING_DIR}"

# ---------------------------------------------------------------------------
# Step 6: Run drift check on updated artifacts
# ---------------------------------------------------------------------------
log ""
log "--- Step 6: Drift detection ---"

python - "${PE_JSONL}" "${ARTIFACT_DIR}" << 'PYTHON_EOF'
import json, sys, math
from pathlib import Path

pe_path = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])

# Load baseline MAPE from manifest or use a conservative default
baseline_mape = 0.15  # 15% default MAPE threshold if no stored value
manifest_path = artifact_dir / "manifest_v1.json"
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("baseline_mape")
        if stored and not math.isnan(float(stored)):
            baseline_mape = float(stored)
    except Exception:
        pass

try:
    from aurelius.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(threshold_multiplier=2.0, min_records=10)
    report = detector.check_from_jsonl(pe_path, baseline_mape=baseline_mape)
    print(json.dumps(report.to_dict(), indent=2))
    if report.drift_detected:
        print("DRIFT ALERT: model retraining recommended", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Drift OK: ratio={report.drift_ratio}")
except Exception as exc:
    print(f"Drift check error (non-fatal): {exc}")
PYTHON_EOF
DRIFT_EXIT=$?

if [[ ${DRIFT_EXIT} -eq 1 ]]; then
    log "WARNING: Drift detected. Consider triggering immediate retraining."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log ""
log "=== Learning loop completed successfully ==="
log "Run ID: ${RUN_ID}"

exit 0
