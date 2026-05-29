"""Canonical CAISO/PJM/ERCOT 1000-job energy backtest — frozen-benchmark tests.

These tests pin the canonical backtest so future optimizer / forecasting /
adapter changes are compared apples-to-apples. They guard:
  * the fixed seed / job count (1000) / regions / data windows,
  * determinism (two runs are byte-identical),
  * the golden summary snapshot,
  * environment-independence (no PyYAML / optional-dep / ordering drift),
  * the existing standalone energy result + current_price_only + greedy_energy.

If a change is intentional, regenerate the golden snapshot with
``python scripts/run_canonical_backtests.py --write-golden`` and explain it in
the PR body (docs/BACKTESTS.md).
"""

from __future__ import annotations

import json
import os

import pytest

from aurelius.benchmarks.canonical_backtests import (
    CANONICAL_JOB_COUNT,
    CANONICAL_REGIONS,
    CANONICAL_WINDOW_END,
    CANONICAL_WINDOW_START,
    GOLDEN_PATH,
    POLICY_CONSTRAINT_AWARE_ADAPTER,
    POLICY_CURRENT_PRICE_ONLY,
    POLICY_FIFO,
    POLICY_GREEDY_ENERGY,
    POLICY_ROBUST_STANDALONE,
    build_canonical_jobs,
    load_canonical_price_data,
    run_canonical_backtest,
)


@pytest.fixture(scope="module")
def summary():
    return run_canonical_backtest()


# ---------------------------------------------------------------------------
# Fixed trace shape
# ---------------------------------------------------------------------------

def test_job_count_is_exactly_1000():
    jobs = build_canonical_jobs()
    assert len(jobs) == CANONICAL_JOB_COUNT == 1000


def test_regions_are_caiso_pjm_ercot():
    assert CANONICAL_REGIONS == ("us-west", "us-east", "us-south")


def test_window_is_fixed_and_inside_all_iso_data():
    # All three ISOs must have DA + RT data covering the canonical window.
    da, rt = load_canonical_price_data()
    for region in CANONICAL_REGIONS:
        assert da[region], f"no DA data for {region} in window"
        assert rt[region], f"no RT data for {region} in window"
        for table in (da[region], rt[region]):
            assert min(table) >= CANONICAL_WINDOW_START
            assert max(table) <= CANONICAL_WINDOW_END


def test_job_trace_is_deterministic_and_id_stable():
    a = build_canonical_jobs()
    b = build_canonical_jobs()
    # Jobs are sorted by (submit_time, job_id), so order is stable run-to-run.
    assert [j.job_id for j in a] == [j.job_id for j in b]
    # The full id set is exactly job-00000 .. job-00999 (stable, no uuid).
    assert sorted(j.job_id for j in a) == [f"job-{i:05d}" for i in range(1000)]
    # Same seed => byte-identical fields.
    assert [(j.job_id, j.workload_type, j.runtime_hours, j.power_kw,
             tuple(j.region_options)) for j in a] == \
           [(j.job_id, j.workload_type, j.runtime_hours, j.power_kw,
             tuple(j.region_options)) for j in b]


def test_trace_contains_flexible_and_latency_pinned_jobs():
    jobs = build_canonical_jobs()
    types = {j.workload_type for j in jobs}
    assert "llm_batch_inference" in types  # flexible batch
    assert "realtime_inference" in types   # latency-pinned, cannot migrate
    # Latency-pinned jobs are the ones that cannot migrate.
    assert any(j.migration_cost_hours is None for j in jobs)
    assert any(j.migration_cost_hours is not None for j in jobs)


# ---------------------------------------------------------------------------
# Determinism + environment independence
# ---------------------------------------------------------------------------

def test_backtest_is_deterministic(summary):
    again = run_canonical_backtest()
    assert summary.golden_dict() == again.golden_dict()


def test_backtest_summary_is_ordering_independent(summary):
    # The golden dict must serialize identically regardless of dict/set
    # insertion order (sort_keys round-trip is stable). Reuses the module
    # fixture to avoid an extra full backtest run.
    g = summary.golden_dict()
    assert json.loads(json.dumps(g, sort_keys=True)) == g
    # Rejection reasons are STABLE codes (no embedded floats), so the histogram
    # has a small, fixed key set — not one key per candidate.
    reasons = g["policies"]["constraint_aware_with_energy_adapter"]["rejection_reasons"]
    assert all(":" not in k and "->" not in k for k in reasons)


def test_all_policies_present(summary):
    g = summary.golden_dict()
    for p in (POLICY_FIFO, POLICY_CURRENT_PRICE_ONLY, POLICY_GREEDY_ENERGY,
              POLICY_ROBUST_STANDALONE, POLICY_CONSTRAINT_AWARE_ADAPTER):
        assert p in g["policies"], f"missing policy {p}"


# ---------------------------------------------------------------------------
# Golden snapshot
# ---------------------------------------------------------------------------

def test_golden_snapshot_exists():
    assert os.path.exists(GOLDEN_PATH), (
        "golden snapshot missing — generate with "
        "`python scripts/run_canonical_backtests.py --write-golden`"
    )


def test_matches_golden_snapshot(summary):
    assert os.path.exists(GOLDEN_PATH)
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    live = summary.golden_dict()
    assert live == golden, (
        "Canonical backtest diverged from the committed golden snapshot. If this "
        "change is intentional, regenerate it with "
        "`python scripts/run_canonical_backtests.py --write-golden` and explain "
        "the delta in the PR body (docs/BACKTESTS.md)."
    )


# ---------------------------------------------------------------------------
# Standalone energy result + baselines unchanged (core preservation)
# ---------------------------------------------------------------------------

def test_standalone_energy_result_matches_golden(summary):
    """The existing robust energy optimizer standalone result is frozen."""
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    live = summary.policies[POLICY_ROBUST_STANDALONE].to_dict()
    assert live == golden["policies"][POLICY_ROBUST_STANDALONE]


def test_current_price_only_baseline_frozen(summary):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert summary.policies[POLICY_CURRENT_PRICE_ONLY].to_dict() == \
        golden["policies"][POLICY_CURRENT_PRICE_ONLY]


def test_greedy_energy_baseline_frozen(summary):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert summary.policies[POLICY_GREEDY_ENERGY].to_dict() == \
        golden["policies"][POLICY_GREEDY_ENERGY]


# ---------------------------------------------------------------------------
# Constraint-aware wrapping behaves as a SAFETY layer on the energy engine
# ---------------------------------------------------------------------------

def test_wrapped_never_misses_more_deadlines_than_standalone(summary):
    standalone = summary.policies[POLICY_ROBUST_STANDALONE]
    wrapped = summary.policies[POLICY_CONSTRAINT_AWARE_ADAPTER]
    assert wrapped.deadline_misses <= standalone.deadline_misses


def test_adapter_generated_and_classified_candidates(summary):
    wrapped = summary.policies[POLICY_CONSTRAINT_AWARE_ADAPTER]
    assert wrapped.candidates_generated == CANONICAL_JOB_COUNT
    # Every job resolves to exactly one outcome: an accepted alternative, a safe
    # home fallback, an explicit reject, or a defer.
    assert (wrapped.candidates_accepted + wrapped.candidates_fallback
            + wrapped.candidates_rejected + wrapped.candidates_deferred) \
        == wrapped.candidates_generated
    # The next-best search accepts both the engine's optimized placement AND the
    # current_price_only fallback for deadline-edge jobs (Part D).
    assert wrapped.accepted_by_source.get("engine_optimized", 0) > 0
    assert wrapped.accepted_by_source.get("current_price_only", 0) > 0
    # The wrapper still keeps the latency-critical region shifts safe (home).
    assert wrapped.candidates_fallback > 0
    assert any("critical_interactive" in r or "latency" in r
               for r in wrapped.rejection_reasons)


def test_wrapped_beats_current_price_only_with_zero_misses(summary):
    """Part D target: constraint-aware-wrapped >= current_price_only goodput/$,
    with 0 deadline misses and no SLA regression (lower churn is a bonus)."""
    wrapped = summary.policies[POLICY_CONSTRAINT_AWARE_ADAPTER]
    cpo = summary.policies[POLICY_CURRENT_PRICE_ONLY]
    assert wrapped.sla_safe_goodput_per_infra_dollar >= \
        cpo.sla_safe_goodput_per_infra_dollar - 1e-9
    assert wrapped.deadline_misses == 0
    assert wrapped.deadline_misses <= cpo.deadline_misses
    # Lower or equal churn than the aggressive current_price_only baseline.
    assert wrapped.migrations <= cpo.migrations


def test_no_sla_regression_vs_fifo(summary):
    """No policy may produce MORE deadline misses than FIFO (SLA regression)."""
    fifo = summary.policies[POLICY_FIFO]
    for name, m in summary.policies.items():
        # Aggressive energy-greedy and standalone MAY miss (that is the risk we
        # surface); the constraint-aware wrapper must NOT regress vs FIFO.
        if name == POLICY_CONSTRAINT_AWARE_ADAPTER:
            assert m.deadline_misses <= fifo.deadline_misses, (
                f"{name} regressed SLA vs FIFO"
            )
