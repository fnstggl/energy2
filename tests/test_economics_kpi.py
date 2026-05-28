"""Tests for the canonical KPI: SLA-safe goodput per infrastructure dollar.

Covers every invariant the mission spec requires:
- SLA-violating tokens never count as compliant goodput.
- Raw throughput can rise while SLA-safe goodput falls.
- Lower energy cost can still lose (goodput collapse).
- Higher energy cost can still win (goodput rises enough).
- GPU infra cost is included and can dominate electricity.
- Network cost is 0 unless configured.
- Zero goodput → safe sentinel (no division by zero).
- No workload-value weights anywhere in the KPI.
- Benchmark reports surface primary + secondary KPIs.
- constraint_aware is compared against current_price_only / greedy_energy.
"""

import math

from aurelius.benchmarks.economics import (
    EconomicKPIResult,
    InfrastructureCostConfig,
    SLAFilterConfig,
    compute_cost_per_sla_compliant_token,
    compute_economic_kpi,
    compute_energy_cost,
    compute_gpu_infra_cost,
    compute_network_cost,
    compute_sla_compliant_goodput,
    compute_sla_safe_goodput_per_infra_dollar,
    compute_total_infrastructure_cost,
)

# ---------------------------------------------------------------------------
# Spec invariant 1: SLA-violating tokens never count
# ---------------------------------------------------------------------------

def test_sla_violating_tokens_excluded_from_goodput():
    # Tick 1: 100 tokens, 0% timeout → all count.
    # Tick 2: 100 tokens, 40% timeout → only 60 count.
    # Tick 3: 100 tokens, 100% timeout → 0 count (hard exclude).
    goodput = compute_sla_compliant_goodput([100, 100, 100], [0.0, 40.0, 100.0])
    assert goodput == 100 + 60 + 0


def test_hard_exclude_at_50_percent():
    # The simulator caps timeout_rate at 50; at the cap, ALL of that queue's
    # tokens this tick are excluded (the hard-exclude floor).
    goodput = compute_sla_compliant_goodput([100], [50.0])
    assert goodput == 0


# ---------------------------------------------------------------------------
# Spec invariant 2: Raw throughput can increase while SLA-safe goodput decreases
# ---------------------------------------------------------------------------

def test_raw_throughput_up_can_mean_sla_safe_goodput_down():
    # Policy A: 100 tokens fully SLA-compliant.
    # Policy B: 200 raw tokens, but the queue's timeout rate hit the hard-exclude
    # floor (≥ 50%) → 0 compliant. Raw goes up; SLA-safe goodput collapses.
    a = compute_sla_compliant_goodput([100], [0.0])
    b = compute_sla_compliant_goodput([200], [60.0])
    assert b == 0 and b < a, "B raw=200 > A raw=100, but A's SLA-safe goodput is higher"


# ---------------------------------------------------------------------------
# Spec invariants 3 & 4: cost vs goodput tradeoffs determine the winner
# ---------------------------------------------------------------------------

def test_lower_energy_cost_can_still_lose():
    # A spends LESS on energy but its compliant goodput collapses.
    # B spends MORE on energy but delivers SLA-compliant goodput.
    cfg = InfrastructureCostConfig(gpu_hour_prices={}, fallback_gpu_hour_price=0.0)
    # Zero GPU price for both → isolate the energy/goodput effect.
    a = compute_economic_kpi(
        tokens_per_tick=[1000], timeout_rate_pct_per_tick=[50.0],  # all excluded
        energy_cost_per_tick=[1.0],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA L4": 1.0}],
        migration_count=0, config=cfg,
    )
    b = compute_economic_kpi(
        tokens_per_tick=[1000], timeout_rate_pct_per_tick=[0.0],
        energy_cost_per_tick=[5.0],  # 5x energy
        active_gpu_hours_by_type_per_tick=[{"NVIDIA L4": 1.0}],
        migration_count=0, config=cfg,
    )
    assert a.sla_safe_goodput_per_infra_dollar == 0.0  # A wasted energy
    assert b.sla_safe_goodput_per_infra_dollar > 0
    assert b.sla_safe_goodput_per_infra_dollar > a.sla_safe_goodput_per_infra_dollar


def test_higher_energy_cost_can_still_win():
    # B has 4x more energy cost but 10x more compliant goodput → B wins.
    cfg = InfrastructureCostConfig(gpu_hour_prices={}, fallback_gpu_hour_price=0.0)
    a = compute_economic_kpi(
        tokens_per_tick=[100], timeout_rate_pct_per_tick=[0.0],
        energy_cost_per_tick=[1.0],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA L4": 1.0}],
        migration_count=0, config=cfg,
    )
    b = compute_economic_kpi(
        tokens_per_tick=[1000], timeout_rate_pct_per_tick=[0.0],
        energy_cost_per_tick=[4.0],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA L4": 1.0}],
        migration_count=0, config=cfg,
    )
    assert b.sla_safe_goodput_per_infra_dollar > a.sla_safe_goodput_per_infra_dollar


# ---------------------------------------------------------------------------
# Spec invariant 5: GPU infra cost is included and can dominate energy
# ---------------------------------------------------------------------------

def test_gpu_infra_cost_can_dominate_energy_cost():
    # Realistic mix: 4 H100 hours at $3/hr = $12; energy at $0.50.
    cfg = InfrastructureCostConfig()
    gpu = compute_gpu_infra_cost({"NVIDIA H100 SXM5 80GB": 4.0}, cfg)
    energy = compute_energy_cost([0.50])
    assert gpu == 12.0
    assert gpu > energy * 20
    total = compute_total_infrastructure_cost(gpu, energy, 0.0)
    assert total == 12.5
    # GPU's share of total cost dominates.
    assert gpu / total > 0.95


def test_gpu_infra_cost_per_type():
    cfg = InfrastructureCostConfig(
        gpu_hour_prices={"X": 10.0, "Y": 1.0}, fallback_gpu_hour_price=0.0,
    )
    assert compute_gpu_infra_cost({"X": 2.0, "Y": 5.0}, cfg) == 25.0


def test_unknown_gpu_type_uses_fallback():
    cfg = InfrastructureCostConfig(
        gpu_hour_prices={}, fallback_gpu_hour_price=2.50,
    )
    assert compute_gpu_infra_cost({"Brand New GPU 9000": 3.0}, cfg) == 7.5


# ---------------------------------------------------------------------------
# Spec invariant 6: Network cost is included ONLY when configured
# ---------------------------------------------------------------------------

def test_network_cost_default_is_zero():
    assert compute_network_cost(100) == 0.0
    assert compute_network_cost(0, egress_gb=1000.0) == 0.0


def test_network_cost_used_only_when_configured():
    cfg = InfrastructureCostConfig(
        network_cost_per_migration=0.25,
        network_cost_per_gb_egress=0.05,
    )
    assert compute_network_cost(10, config=cfg) == 2.5
    assert compute_network_cost(10, egress_gb=100.0, config=cfg) == 2.5 + 5.0


# ---------------------------------------------------------------------------
# Spec invariants 7 & 8: zero goodput is handled safely
# ---------------------------------------------------------------------------

def test_zero_goodput_with_zero_cost_returns_none():
    assert compute_sla_safe_goodput_per_infra_dollar(0, 0.0) is None
    assert compute_cost_per_sla_compliant_token(0.0, 0) is None


def test_zero_goodput_with_positive_cost_does_not_divide_by_zero():
    # Spent money, delivered nothing within SLA. Reported honestly.
    assert compute_sla_safe_goodput_per_infra_dollar(0, 100.0) == 0.0
    cpsct = compute_cost_per_sla_compliant_token(100.0, 0)
    assert cpsct == math.inf


def test_positive_goodput_with_zero_cost_returns_none():
    # No cost basis → undefined ratio (e.g. a config error). Reported as None,
    # not infinity (positive goodput / 0 cost is not a "great" result, it's
    # missing data).
    assert compute_sla_safe_goodput_per_infra_dollar(1000, 0.0) is None


# ---------------------------------------------------------------------------
# Spec invariant 9: NO workload-value weights anywhere
# ---------------------------------------------------------------------------

def test_no_workload_value_weights_in_kpi_signature():
    """The canonical KPI accepts only tokens, SLA outcomes, and resource costs.

    A 'workload value' / 'business priority multiplier' would have to live on
    one of these signatures — if any of them grow such a parameter, this test
    fails loudly.
    """
    import inspect
    forbidden = {"workload_value", "business_value", "priority_weight",
                 "value_per_token", "revenue", "sla_penalty_dollars"}
    for fn in (compute_sla_compliant_goodput, compute_gpu_infra_cost,
               compute_energy_cost, compute_network_cost,
               compute_total_infrastructure_cost,
               compute_sla_safe_goodput_per_infra_dollar,
               compute_cost_per_sla_compliant_token,
               compute_economic_kpi):
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden), (
            f"{fn.__name__} grew a forbidden value-weight parameter: "
            f"{params & forbidden}"
        )


def test_dataclasses_have_no_value_weight_fields():
    for cls in (InfrastructureCostConfig, SLAFilterConfig, EconomicKPIResult):
        fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
        forbidden = {"workload_value", "business_value", "priority_weight",
                     "value_per_token", "revenue", "sla_penalty_dollars"}
        assert not (fields & forbidden), f"{cls.__name__}: {fields & forbidden}"


# ---------------------------------------------------------------------------
# Spec invariants 10 & 11: benchmark report wiring
# ---------------------------------------------------------------------------

def test_benchmark_report_includes_primary_and_secondary_kpis():
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    res = ConstraintBenchmarkRunner().run_scenario(
        "thermal_hotspot_mixed_cluster", steps=8, seed=42,
    )
    for policy, kpi in res.report.aggregated.items():
        # Primary KPI present.
        assert kpi.sla_safe_goodput_per_infra_dollar is not None, policy
        assert kpi.cost_per_sla_compliant_token is not None, policy
        assert kpi.sla_compliant_goodput >= 0
        assert kpi.total_infrastructure_cost > 0
        assert kpi.gpu_infra_cost > 0
        # Secondary KPIs preserved.
        assert kpi.total_tokens >= 0
        assert kpi.total_energy_cost >= 0
        assert kpi.total_sla_violations >= 0
    # Report text surfaces both the primary KPI label and the diagnostics label.
    text = res.report.to_text()
    assert "SLA-safe goodput per infrastructure dollar" in text
    assert "Secondary KPIs" in text
    assert "GPU infra" in text


def test_constraint_aware_compared_against_current_price_only_and_greedy_energy():
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    res = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=8, seed=42,
    )
    agg = res.report.aggregated
    # All four reference baselines plus constraint_aware must appear in the
    # report so it's not "constraint_aware vs FIFO only."
    for policy in (
        "fifo", "current_price_only", "greedy_energy", "sla_aware", "constraint_aware"
    ):
        assert policy in agg, policy
        assert agg[policy].sla_safe_goodput_per_infra_dollar is not None


def test_aggregated_to_dict_lists_primary_kpi_first():
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    res = ConstraintBenchmarkRunner().run_scenario(
        "thermal_hotspot_mixed_cluster", steps=4, seed=42,
    )
    d = res.report.aggregated["constraint_aware"].to_dict()
    # Spec ordering: primary KPI fields up-front.
    keys = list(d)
    assert keys[0] == "policy"
    assert "sla_safe_goodput_per_infra_dollar" in keys[:5]


# ---------------------------------------------------------------------------
# A composed scenario: GPU-hour cost decides the winner, not energy
# ---------------------------------------------------------------------------

def test_more_active_gpu_hours_hurts_kpi_even_at_lower_energy_cost():
    cfg = InfrastructureCostConfig()
    # Policy A: 4 H100 GPUs × 1 hour = $12 GPU + $1 energy, all compliant.
    # Policy B: 8 H100 GPUs × 1 hour = $24 GPU + $0.50 energy, all compliant —
    # same goodput, double the bill. The KPI must reflect that.
    a = compute_economic_kpi(
        tokens_per_tick=[1000], timeout_rate_pct_per_tick=[0.0],
        energy_cost_per_tick=[1.0],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA H100 SXM5 80GB": 4.0}],
        migration_count=0, config=cfg,
    )
    b = compute_economic_kpi(
        tokens_per_tick=[1000], timeout_rate_pct_per_tick=[0.0],
        energy_cost_per_tick=[0.50],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA H100 SXM5 80GB": 8.0}],
        migration_count=0, config=cfg,
    )
    assert a.sla_compliant_goodput == b.sla_compliant_goodput == 1000
    assert a.sla_safe_goodput_per_infra_dollar > b.sla_safe_goodput_per_infra_dollar
    # B spent less on energy but more on GPU infra → loses on the canonical KPI.
    assert b.energy_cost < a.energy_cost
    assert b.gpu_infra_cost > a.gpu_infra_cost


def test_economic_kpi_result_serializes_inf_safely():
    cfg = InfrastructureCostConfig()
    r = compute_economic_kpi(
        tokens_per_tick=[100], timeout_rate_pct_per_tick=[50.0],   # all excluded
        energy_cost_per_tick=[1.0],
        active_gpu_hours_by_type_per_tick=[{"NVIDIA L4": 1.0}],
        migration_count=0, config=cfg,
    )
    d = r.to_dict()
    assert d["sla_compliant_goodput"] == 0
    assert d["sla_safe_goodput_per_infra_dollar"] == 0.0
    assert d["cost_per_sla_compliant_token"] == math.inf
