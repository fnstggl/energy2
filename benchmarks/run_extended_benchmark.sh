#!/usr/bin/env bash
# =============================================================================
# AURELIUS EXTENDED BENCHMARK — 90-Day Training Window Validation
# =============================================================================
#
# PURPOSE:
#   Validates the per-region forecaster (v4.0) vs joint model (v2.0) on a
#   287-day combined dataset (Jun 2025 – Mar 2026). With 90-day training
#   windows, each region gets ≥2160 records — matching joint-model data density.
#
# WHY:
#   Oracle diagnostics showed a 22-48pp forecasting gap for training/fine_tuning.
#   Root cause: 30-day windows give per-region only 720 records (vs 2160 for joint).
#   This benchmark tests whether 90-day windows close that gap.
#
# PREREQUISITES:
#   1. Fetch fall 2025 data (Sep-Dec 2025):
#      python scripts/fetch_caiso_pjm_prices.py \
#        --start 2025-09-01 --end 2026-01-01 --out-dir data/fall2025
#      (ERCOT requires ERCOT_PASSWORD in env; CAISO/PJM fetch automatically)
#
#   2. Build combined dataset (merges summer2025 + fall2025 + q12026):
#      python scripts/build_combined_dataset.py
#
# USAGE:
#   bash benchmarks/run_extended_benchmark.sh
#
# OUTPUTS:
#   benchmarks/results/benchmark_joint_90d_*.json
#   benchmarks/results/benchmark_perregion_90d_*.json
#   benchmarks/results/extended_benchmark_comparison_*.txt
#
# HONEST BENCHMARKS — requirements:
#   - Real CAISO/PJM/ERCOT price data (not synthetic)
#   - Leakage-free walk-forward backtesting
#   - Correct information constraints (no future data at decision time)
#   - Valid if: 0% missing price hours in the evaluation windows
#
# DO NOT CLAIM savings from these results without:
#   - Verifying 0% missing hours (check benchmark output)
#   - Comparing against current_price_only baseline (already done)
#   - Reproducing with the exact commands below
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COMBINED_DA="data/combined_2025_2026/3region_dam.csv"
COMBINED_RT="data/combined_2025_2026/3region_rt.csv"
TIMESTAMP=$(date -u +"%Y%m%d")
OUT_DIR="benchmarks/results"
JOINT_OUT="$OUT_DIR/benchmark_joint_90d_combined_${TIMESTAMP}"
PERREGION_OUT="$OUT_DIR/benchmark_perregion_90d_combined_${TIMESTAMP}"

echo "==============================================================="
echo "AURELIUS EXTENDED BENCHMARK — 90-Day Windows"
echo "Dataset: Jun 2025 – Mar 2026 (~287 days, 3 regions)"
echo "Comparing: joint ml_quantile v2.0 vs per-region v4.0"
echo "==============================================================="

# Check data exists
if [ ! -f "$COMBINED_DA" ]; then
  echo "ERROR: Combined dataset not found at $COMBINED_DA"
  echo "Run: python scripts/build_combined_dataset.py"
  exit 1
fi

echo ""
echo "--- Step 1: Joint model (ml_quantile v2.0) with 90-day windows ---"
python benchmarks/run_benchmark.py \
  --region-combo combined_2025_2026_3region \
  --forecaster ml_quantile \
  --train-days 90 \
  --eval-days 7 \
  --num-jobs 50 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$JOINT_OUT.log"
echo "[Done] Joint 90d benchmark complete."

echo ""
echo "--- Step 2: Per-region model (ml_quantile_perregion v4.0) with 90-day windows ---"
python benchmarks/run_benchmark.py \
  --region-combo combined_2025_2026_3region \
  --forecaster ml_quantile_perregion \
  --train-days 90 \
  --eval-days 7 \
  --num-jobs 50 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$PERREGION_OUT.log"
echo "[Done] Per-region 90d benchmark complete."

echo ""
echo "--- Step 3: Oracle diagnostics (ceiling analysis) ---"
python benchmarks/run_benchmark.py \
  --region-combo combined_2025_2026_3region \
  --forecaster seasonal_naive \
  --train-days 90 \
  --eval-days 7 \
  --num-jobs 50 \
  --oracle \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/oracle_90d_combined_${TIMESTAMP}.log"
echo "[Done] Oracle diagnostics complete."

echo ""
echo "--- Comparison summary ---"
echo "Results saved to $OUT_DIR/"
echo "Key files:"
ls -1t "$OUT_DIR"/benchmark_joint_90d_*.json 2>/dev/null | head -1
ls -1t "$OUT_DIR"/benchmark_perregion_90d_*.json 2>/dev/null | head -1
echo ""
echo "INTERPRETATION GUIDE:"
echo "  joint_90d vs caiso_pjm_ercot_da_rt (joint_30d):"
echo "    > current: 90d window helps joint model → more data is better"
echo "    < current: 90d window hurts → training on more diverse regimes confuses model"
echo "  perregion_90d vs joint_90d:"
echo "    > joint: per-region isolation benefit > cross-region calibration loss"
echo "    < joint: cross-region calibration still dominates with equal data"
echo ""
echo "HONEST NOTE:"
echo "  All results valid only if 0% missing price hours in evaluation windows."
echo "  Check 'missing_hours_pct' in benchmark JSON output."
echo "  60% savings is an aspirational target — NOT claimed from these results."
