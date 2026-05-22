#!/usr/bin/env bash
# =============================================================================
# Aurelius — Oracle Diagnostics (Savings Ceiling Analysis)
# =============================================================================
#
# Runs the optimizer with PERFECT FORESIGHT (oracle forecaster) to measure the
# maximum possible savings given the price data and workload flexibility.
#
# PURPOSE: Ceiling analysis only.
#
# If oracle savings >> seasonal_naive savings:
#   → Forecasting quality is the primary bottleneck. Invest in better ML.
#
# If oracle savings ≈ seasonal_naive savings:
#   → Price spread between hours/regions is the bottleneck. Better forecasting
#     won't help much — need more flexible workloads or wider regions.
#
# NEVER report oracle numbers as real savings. They are intentional leakage
# (the optimizer sees future prices). Label every oracle result as DIAGNOSTIC.
#
# Usage:
#   ./benchmarks/run_oracle_diagnostics.sh [--quick] [--output-dir DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/benchmarks/results"
QUICK=""

for arg in "$@"; do
  case $arg in
    --quick) QUICK="--quick" ;;
    --output-dir) OUTPUT_DIR="$2"; shift ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "============================================================"
echo "ORACLE DIAGNOSTIC RUN  (${TIMESTAMP})"
echo "⚠ CEILING ANALYSIS ONLY — NOT VALID SAVINGS CLAIMS"
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

echo "Running oracle diagnostics on CAISO+PJM DA-RT dataset..."
echo ""

python "${SCRIPT_DIR}/run_benchmark.py" \
  --region-combo caiso_pjm_da_rt \
  --oracle \
  --output-dir "${OUTPUT_DIR}" \
  ${QUICK} \
  2>&1 | tee "${OUTPUT_DIR}/oracle_diagnostics_${TIMESTAMP}.log"

ORACLE_FILE=$(ls -t "${OUTPUT_DIR}"/oracle_*.json 2>/dev/null | head -1)
SEASONAL_FILE=$(ls -t "${OUTPUT_DIR}"/benchmark_*.json 2>/dev/null | head -1)

echo ""
echo "============================================================"
echo "ORACLE DIAGNOSTIC COMPLETE"
echo ""
if [ -n "${ORACLE_FILE}" ] && [ -n "${SEASONAL_FILE}" ]; then
  echo "Compare oracle vs seasonal_naive savings to identify bottleneck:"
  echo "  Oracle results:         ${ORACLE_FILE}"
  echo "  Last benchmark results: ${SEASONAL_FILE}"
  echo ""
  echo "  If oracle >> seasonal_naive: forecasting is the bottleneck"
  echo "  If oracle ≈ seasonal_naive:  price spread is the bottleneck"
fi
echo ""
echo "⚠ DO NOT cite oracle numbers as Aurelius savings claims."
echo "⚠ These numbers are CEILING DIAGNOSTICS with intentional future leakage."
echo "============================================================"
