#!/usr/bin/env bash
# =============================================================================
# Aurelius — Run benchmark across ALL region combinations
# =============================================================================
#
# Tests the optimizer against each configured region combination, using the
# "training" workload type (highest savings potential, longest runtime).
#
# Outputs JSON results and a summary table comparing savings vs current_price_only
# across regions. Use this to identify which regions deliver the best optimizer lift.
#
# Usage:
#   ./benchmarks/run_all_regions.sh [--quick] [--workload WTYPE] [--output-dir DIR]
#
# Options:
#   --quick          Reduced run for CI
#   --workload WTYPE Workload type to test (default: training)
#   --output-dir DIR Override output directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/benchmarks/results"
QUICK=""
WORKLOAD="training"

for arg in "$@"; do
  case $arg in
    --quick) QUICK="--quick" ;;
    --workload) WORKLOAD="$2"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "============================================================"
echo "AURELIUS REGION BENCHMARK  (${TIMESTAMP})"
echo "Workload: ${WORKLOAD}"
echo "Primary baseline: current_price_only"
echo "============================================================"
echo ""

REGION_COMBOS=(
  "us-west-only"
  "us-east-only"
  "caiso_pjm_da_rt"
)

PASS=0
FAIL=0

for COMBO in "${REGION_COMBOS[@]}"; do
  echo "--- Region combo: ${COMBO} ---"
  python "${SCRIPT_DIR}/run_benchmark.py" \
    --region-combo "${COMBO}" \
    --workload "${WORKLOAD}" \
    --output-dir "${OUTPUT_DIR}" \
    ${QUICK} \
    2>&1 | tee "${OUTPUT_DIR}/region_${COMBO}_${TIMESTAMP}.log" \
    && PASS=$((PASS+1)) \
    || FAIL=$((FAIL+1))
  echo ""
done

echo "============================================================"
echo "REGION BENCHMARK COMPLETE"
echo "Passed: ${PASS}  Failed: ${FAIL}"
echo "Results: ${OUTPUT_DIR}"
echo "============================================================"

if [ "${FAIL}" -gt 0 ]; then
  echo "ERROR: ${FAIL} region benchmark(s) failed" >&2
  exit 1
fi
exit 0
