"""Validation tests for the energy / carbon / arbitrage realism layer.

Covers the audited gaps the energy upgrade targets:
- separate day-ahead / real-time settlement (DA and RT are NOT interchangeable);
- mean-reverting + heteroskedastic DA/RT basis with congestion jumps;
- LMP = energy + congestion + loss (nodal/regional arbitrage is not clean);
- heavy-tailed RT spike process (configurable, not hardcoded);
- regional carbon-intensity forecasts with uncertainty + provider disagreement;
- carbon optimization != price optimization;
- workload flexibility limits (nothing is infinitely deferrable);
- net-vs-gross savings with explicit penalties (KEEP if net <= 0);
- conservative forecast-error buffer; low telemetry biases to no-op;
- diminishing returns / churn from repeated shifting;
- energy-aware action vetoes (tiny arbitrage must not act);
- emergent: clean arbitrage captures safe net savings; basis blowout / migration
  trap / low telemetry force no-op; latency-critical is not shifted;
- default (basis OFF) pricing is deterministic and DA == RT;
- calibration metadata has no hidden constants.

Pure functions are deterministic; integration scenarios use a fixed seed.
"""

from __future__ import annotations

import random

from aurelius.simulation.cluster import energy as enrg
from aurelius.simulation.cluster.calibration import (
    ENERGY_FLEX_PROFILES,
    ENERGY_PARAMS,
    calibration_table,
    energy_flex_table,
    energy_value,
    resolve_energy_flex,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _run(name: str, steps: int = 16, seed: int | None = None):
    cfg = load_scenario(name).config
    sim = ClusterSimulator(cfg, seed=seed if seed is not None else cfg.seed)
    ms = sim.run_metrics_only(steps)
    return sim, ms


# ---------------------------------------------------------------------------
# Day-ahead / real-time settlement
# ---------------------------------------------------------------------------

class TestSettlement:
    def test_two_settlement_formula(self):
        # Scheduled 100 at DA 50; realized 110 at RT 200 → deviation pays RT.
        assert enrg.settlement_bill(100, 110, 50, 200) == 100 * 50 + 10 * 200

    def test_da_not_interchangeable_with_rt(self):
        # Same realized quantity, divergent RT → different bill.
        b_low = enrg.settlement_bill(100, 100, 50, 50)
        b_high = enrg.settlement_bill(100, 120, 50, 300)
        assert b_high > b_low

    def test_real_time_price_neutral_when_no_components(self):
        # No basis/congestion/loss/spike → RT == DA (deterministic base case).
        assert enrg.real_time_price(50.0, 0.0, 0.0, 0.0, 0.0) == 50.0


# ---------------------------------------------------------------------------
# DA/RT basis
# ---------------------------------------------------------------------------

class TestBasis:
    def test_mean_reverting(self):
        # From a large basis, with no congestion, it reverts toward 0 on average.
        rng = _rng(1)
        b = 100.0
        for _ in range(40):
            b, _ = enrg.basis_step(b, False, rng)
        assert abs(b) < 100.0

    def test_congestion_raises_volatility_and_jumps(self):
        # Congested basis paths reach larger magnitudes than calm ones.
        calm = _rng(2)
        cong = _rng(2)
        bc = bg = 0.0
        max_calm = max_cong = 0.0
        for _ in range(60):
            bc, _ = enrg.basis_step(bc, False, calm)
            bg, _ = enrg.basis_step(bg, True, cong)
            max_calm = max(max_calm, abs(bc))
            max_cong = max(max_cong, abs(bg))
        assert max_cong > max_calm

    def test_basis_seeded(self):
        a, _ = enrg.basis_step(0.0, True, _rng(5))
        b, _ = enrg.basis_step(0.0, True, _rng(5))
        assert a == b


# ---------------------------------------------------------------------------
# LMP + spike
# ---------------------------------------------------------------------------

class TestLMPandSpike:
    def test_lmp_components(self):
        cong, loss, total = enrg.lmp_total(100.0, False)
        assert loss > 0  # marginal loss component always present
        assert total == 100.0 + cong + loss

    def test_congestion_adds_component(self):
        _, _, total_calm = enrg.lmp_total(100.0, False)
        _, _, total_cong = enrg.lmp_total(100.0, True)
        assert total_cong > total_calm

    def test_spike_heavy_tail(self):
        # With a high spike probability, spikes occur and are large; stress raises
        # the rate further.
        cfg = {"spike_prob": 0.5, "spike_magnitude": 300.0}
        rng = _rng(3)
        spikes = [enrg.spike_increment(False, rng, cfg) for _ in range(200)]
        assert any(s > 100.0 for s in spikes)
        assert any(s == 0.0 for s in spikes)


# ---------------------------------------------------------------------------
# Carbon
# ---------------------------------------------------------------------------

class TestCarbon:
    def test_carbon_forecast_uncertain(self):
        f, err, dis = enrg.carbon_forecast(400.0, _rng(1))
        assert err > 0 and dis > 0          # forecast is NOT ground truth
        assert f >= 0

    def test_objective_weights_cost_and_carbon(self):
        price_only = enrg.objective(cost=10.0, carbon=800.0, alpha=1.0, beta=0.0)
        carbon_weighted = enrg.objective(cost=10.0, carbon=800.0, alpha=1.0, beta=0.01)
        assert carbon_weighted > price_only  # beta adds carbon term


# ---------------------------------------------------------------------------
# Net savings + forecast buffer + churn + flexibility
# ---------------------------------------------------------------------------

class TestNetSavings:
    def test_net_subtracts_penalties(self):
        net = enrg.net_savings(10.0, migration_cost=3.0, churn_penalty=1.0)
        assert net == 6.0

    def test_keep_when_net_non_positive(self):
        net = enrg.net_savings(1.0, migration_cost=3.0)
        assert net <= 0  # → KEEP / no-op

    def test_risk_adjusted_buffer(self):
        # Forecast error reduces the usable savings.
        assert enrg.risk_adjusted_savings(10.0, 4.0) < 10.0

    def test_required_margin_inflated_under_low_confidence(self):
        hi = enrg.required_margin(10.0, "high")
        med = enrg.required_margin(10.0, "medium")
        lo = enrg.required_margin(10.0, "low")
        assert lo > med > hi  # missing telemetry → higher bar → bias to no-op

    def test_action_requires_clearing_margin(self):
        # Tiny arbitrage (net just above 0) does NOT clear the margin.
        assert not enrg.energy_action_allowed(0.5, 0.5, 1.0)
        assert enrg.energy_action_allowed(5.0, 4.0, 1.0)

    def test_churn_super_linear(self):
        p1 = enrg.churn_penalty(1, 100.0)
        p2 = enrg.churn_penalty(2, 100.0)
        p3 = enrg.churn_penalty(3, 100.0)
        assert (p3 - p2) > (p2 - p1)  # increasing marginal churn cost

    def test_shift_window_never_infinite(self):
        assert enrg.shift_window_hours("low") == 0.0
        assert 0 < enrg.shift_window_hours("medium") < enrg.shift_window_hours("high")
        assert enrg.shift_window_hours("high") < 1e6  # finite

    def test_telemetry_confidence(self):
        assert enrg.energy_telemetry_confidence(True, True, 0) == "high"
        assert enrg.energy_telemetry_confidence(True, False, 0) == "medium"
        assert enrg.energy_telemetry_confidence(False, False, 6) == "low"


# ---------------------------------------------------------------------------
# Integration scenarios (emergent behaviour)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_default_scenario_da_equals_rt(self):
        # Basis OFF by default → RT == DA → deterministic, unchanged pricing.
        sim, _ = _run("energy_price_arbitrage_multiregion", steps=8)
        for r in sim._cluster.regions.values():
            assert abs(r.day_ahead_price - r.realtime_price) < 1e-9

    def test_determinism_under_fixed_seed(self):
        s1, _ = _run("da_rt_basis_blowout", steps=10, seed=3)
        s2, _ = _run("da_rt_basis_blowout", steps=10, seed=3)
        assert abs(
            s1._cluster.total_energy_cost - s2._cluster.total_energy_cost
        ) < 1e-9

    def test_clean_arbitrage_captures_net_savings(self):
        # Large stable spread, flexible batch → energy gate ALLOWS (net > 0).
        sim, _ = _run("clean_batch_shift_arbitrage", steps=12)
        wl = sim._cluster.workloads["batch-shift"]
        ns = sim._energy_net_savings(wl, "us-west")
        assert ns.gross_energy_savings > 0
        assert ns.net_savings > 0
        assert ns.action_allowed is True

    def test_basis_blowout_da_planner_wrong(self):
        # Congestion drives RT far above DA; the energy governor rejects the move.
        sim, ms = _run("da_rt_basis_blowout", steps=16)
        m = ms[-1]
        assert m.real_time_price_mean > m.day_ahead_price_mean
        assert sim.safe_migrate_workload("basis-batch", "us-west") is False

    def test_migration_trap_forces_noop(self):
        # Small positive gross spread; migration cost makes net <= 0 → no-op.
        sim, _ = _run("migration_trap_erased_savings", steps=12)
        wl = sim._cluster.workloads["trap-batch"]
        ns = sim._energy_net_savings(wl, "us-west")
        assert ns.gross_energy_savings > 0      # gross looks positive
        assert ns.net_savings <= 0              # but net is not
        assert ns.action_allowed is False
        # And the energy governor vetoes the move.
        assert sim._migration_veto(wl, "us-west", respect_governor=True) == (
            "energy_not_worth_it"
        )

    def test_low_telemetry_biases_to_noop(self):
        # Missing price/carbon telemetry inflates the margin → modest spread no-op.
        sim, ms = _run("low_confidence_energy_telemetry", steps=12)
        assert ms[-1].low_energy_telemetry_count >= 1
        wl = sim._cluster.workloads["blind-energy-batch"]
        ns = sim._energy_net_savings(wl, "us-west")
        assert ns.action_allowed is False
        # The inflated (low-confidence) margin is strictly above the high-conf one.
        src_cost = (
            sim._workload_tick_kwh(wl, sim._cluster)
            * sim._cluster.regions["us-east"].realtime_price / 1000.0
        )
        assert enrg.required_margin(src_cost, "low") > enrg.required_margin(
            src_cost, "high"
        )

    def test_carbon_cheap_not_price_cheap(self):
        # With carbon weighting, moving to the clean (pricey) region has positive
        # net value despite a NEGATIVE price-only gross — carbon != price.
        sim, _ = _run("carbon_cheap_price_expensive", steps=8)
        wl = sim._cluster.workloads["carbon-batch"]
        ns = sim._energy_net_savings(wl, "hydro")
        assert ns.gross_energy_savings < 0      # hydro is price-expensive
        assert ns.gross_carbon_value > 0        # but carbon-cheap
        assert ns.net_savings > 0               # carbon objective makes it worth it

    def test_latency_critical_not_shifted(self):
        # Latency-critical, low-flexibility inference is not migrated for energy.
        sim, _ = _run("latency_critical_no_energy_shift", steps=12)
        assert sim.safe_migrate_workload("lc-inf", "us-west") is False

    def test_repeated_shifting_grows_churn(self):
        # A flexible job migrated repeatedly accrues a growing churn penalty.
        sim, _ = _run("clean_batch_shift_arbitrage", steps=4)
        wl = sim._cluster.workloads["batch-shift"]
        # Force-disable other governors by using raw migrate_workload back and forth.
        sim.migrate_workload("batch-shift", "us-west")
        sim.migrate_workload("batch-shift", "us-east")
        sim.migrate_workload("batch-shift", "us-west")
        assert wl.energy.churn.recent_shifts >= 2
        p_more = enrg.churn_penalty(wl.energy.churn.recent_shifts, 100.0)
        p_one = enrg.churn_penalty(1, 100.0)
        assert p_more > p_one


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden constants
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_all_energy_params_have_provenance(self):
        for name, p in ENERGY_PARAMS.items():
            assert p.source, name
            assert p.source_type in (
                "measured", "benchmark_derived", "documented", "inferred", "heuristic"
            ), name
            assert p.confidence in ("high", "medium", "low"), name
            assert p.calibration_notes, name

    def test_energy_params_in_calibration_table(self):
        rows = calibration_table()
        groups = {r["group"] for r in rows}
        assert "energy" in groups
        energy_rows = [r for r in rows if r["group"] == "energy"]
        assert len(energy_rows) == len(ENERGY_PARAMS)

    def test_energy_flex_table_populated(self):
        assert len(energy_flex_table()) == len(ENERGY_FLEX_PROFILES)

    def test_config_override(self):
        assert energy_value("spike_prob") != 0.999
        assert energy_value("spike_prob", {"spike_prob": 0.999}) == 0.999

    def test_flex_resolution(self):
        assert resolve_energy_flex("high").max_shift_hours == 24.0
        assert resolve_energy_flex("low").max_shift_hours == 0.0
        assert resolve_energy_flex(None).name == "medium"
