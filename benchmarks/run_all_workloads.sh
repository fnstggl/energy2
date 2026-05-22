#!/usr/bin/env bash
# =============================================================================
# Aurelius — Run benchmark across ALL workload types
# =============================================================================
#
# Runs the standardized benchmark for each of the 7 workload types using the
# CAISO+PJM DA-plan/RT-settle dataset (the most production-realistic setup).
#
# Outputs one JSON per workload to benchmarks/results/ and prints a summary
# table with savings vs current_price_only.
#
# Usage:
#   ./benchmarks/run_all_workloads.sh [--quick] [--oracle] [--output-dir DIR]
#
# Options:
#   --quick        Reduced run (fewer folds, smaller job count) for CI
#   --oracle       Run oracle diagnostics after main run (ceiling analysis)
#   --output-dir   Override output directory (default: benchmarks/results)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/benchmarks/results"
QUICK=""
ORACLE=""

for arg in "$@"; do
  case $arg in
    --quick) QUICK="--quick" ;;
    --oracle) ORACLE="--oracle" ;;
    --output-dir) OUTPUT_DIR="$2"; shift ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "============================================================"
echo "AURELIUS WORKLOAD BENCHMARK  (${TIMESTAMP})"
echo "Dataset: CAISO+PJM DA-plan / RT-settle"
echo "Primary baseline: current_price_only"
echo "============================================================"
echo ""

WORKLOAD_TYPES=(
  "training"
  "fine_tuning"
  "llm_batch_inference"
  "data_processing"
  "scheduled_batch"
  "background_maintenance"
  "realtime_inference"
)

PASS=0
FAIL=0

for WTYPE in "${WORKLOAD_TYPES[@]}"; do
  echo "--- Workload: ${WTYPE} ---"
  OUT_FILE="${OUTPUT_DIR}/workload_${WTYPE}_${TIMESTAMP}.json"
  python "${SCRIPT_DIR}/run_benchmark.py" \
    --workload "${WTYPE}" \
    --region-combo caiso_pjm_da_rt \
    --output-dir "${OUTPUT_DIR}" \
    ${QUICK} ${ORACLE} \
    2>&1 | tee "${OUTPUT_DIR}/workload_${WTYPE}_${TIMESTAMP}.log" \
    && PASS=$((PASS+1)) \
    || FAIL=$((FAIL+1))
  echo ""
done

echo "============================================================"
echo "WORKLOAD BENCHMARK COMPLETE"
echo "Passed: ${PASS}  Failed: ${FAIL}"
echo "Results: ${OUTPUT_DIR}"
echo "============================================================"

if [ "${FAIL}" -gt 0 ]; then
  echo "ERROR: ${FAIL} workload benchmark(s) failed" >&2
  exit 1
fi
exit 0
