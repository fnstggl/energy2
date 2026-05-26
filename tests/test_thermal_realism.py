"""Validation tests for the thermal / cooling / power realism layer.

Covers the audited gaps the thermal upgrade targets:
- saturating board-power curves (utilization alone does NOT predict heat);
- per-GPU-class thermal response (not one model for all GPUs);
- thermal inertia + delayed cooling recovery (lumped-capacitance ODE);
- rack-level kW density regimes + persistent hotspots;
- cooling regimes (air vs liquid vs weak airflow);
- CONTINUOUS thermal + power slowdown (not a binary flag);
- thermal telemetry confidence (missing ≠ safe) + thermal migration veto;
- emergent: dense air racks overheat/throttle while liquid stays safe;
- calibration metadata has no hidden constants.

Pure functions are deterministic; integration scenarios use a fixed seed.
"""

from __future__ import annotations

import random

from aurelius.simulation.cluster import thermal as therm
from aurelius.simulation.cluster.calibration import (
    COOLING_REGIMES,
    GPU_POWER_CLASSES,
    THERMAL_PARAMS,
    calibration_table,
    cooling_regime_table,
    power_class_for_model,
    power_class_table,
    resolve_cooling_regime,
    thermal_value,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Board power curve
# ---------------------------------------------------------------------------

class TestBoardPower:
    def test_saturates_with_utilization(self):
        cls = GPU_POWER_CLASSES["a100"]
        p_lo = therm.board_power_watts(0.2, cls)
        p_mid = therm.board_power_watts(0.6, cls)
        p_hi = therm.board_power_watts(1.0, cls)
        assert p_lo < p_mid < p_hi
        # Convex saturation: the first 0.4 of util adds more than the last 0.4.
        assert (p_mid - p_lo) > (p_hi - p_mid)

    def test_idle_floor(self):
        cls = GPU_POWER_CLASSES["a100"]
        idle = therm.board_power_watts(0.0, cls)
        assert abs(idle - thermal_value("power_idle_frac") * cls.p_max_w) < 1.0

    def test_training_draws_more_than_inference(self):
        cls = GPU_POWER_CLASSES["h100-sxm"]
        assert therm.board_power_watts(0.8, cls, "batch_training") > \
            therm.board_power_watts(0.8, cls, "inference")

    def test_classes_differ(self):
        # Same utilization, different board class → different power.
        h100 = therm.board_power_watts(1.0, GPU_POWER_CLASSES["h100-sxm"])
        l4 = therm.board_power_watts(1.0, GPU_POWER_CLASSES["l4"])
        assert h100 > l4 * 5

    def test_model_name_mapping(self):
        assert power_class_for_model("NVIDIA H100 SXM5 80GB").name == "h100-sxm"
        assert power_class_for_model("NVIDIA A100 SXM4 80GB").name == "a100"
        assert power_class_for_model("NVIDIA L4").name == "l4"


# ---------------------------------------------------------------------------
# Temperature evolution (inertia + recovery lag)
# ---------------------------------------------------------------------------

class TestTemperatureEvolution:
    def test_heats_toward_equilibrium(self):
        cls = GPU_POWER_CLASSES["a100"]
        regime = resolve_cooling_regime("air")
        t = 22.0
        p = therm.board_power_watts(1.0, cls)
        for _ in range(40):
            t = therm.temperature_step(t, p, 22.0, cls, regime, _rng(0))
        # Full-power A100 (air) settles roughly 45-60°C above 22°C inlet.
        assert 60.0 < t < 85.0

    def test_recovery_is_gradual(self):
        cls = GPU_POWER_CLASSES["a100"]
        regime = resolve_cooling_regime("air")
        # Heat up, then drop power to idle and watch it decay gradually.
        t = 80.0
        p_idle = therm.board_power_watts(0.0, cls)
        t1 = therm.temperature_step(t, p_idle, 22.0, cls, regime, _rng(0))
        # One tick does NOT snap to equilibrium (inertia / recovery lag).
        assert t1 < t and t1 > 40.0

    def test_liquid_recovers_faster(self):
        cls = GPU_POWER_CLASSES["h100-sxm"]
        air = resolve_cooling_regime("air")
        liquid = resolve_cooling_regime("liquid")
        p_idle = therm.board_power_watts(0.0, cls)
        t_air = therm.temperature_step(80.0, p_idle, 22.0, cls, air, _rng(0))
        t_liq = therm.temperature_step(80.0, p_idle, 22.0, cls, liquid, _rng(0))
        assert t_liq < t_air   # liquid cools faster

    def test_deterministic_given_rng(self):
        cls = GPU_POWER_CLASSES["a100"]
        regime = resolve_cooling_regime("air")
        a = therm.temperature_step(50.0, 400, 22.0, cls, regime, _rng(5))
        b = therm.temperature_step(50.0, 400, 22.0, cls, regime, _rng(5))
        assert a == b


# ---------------------------------------------------------------------------
# Rack density + hotspots
# ---------------------------------------------------------------------------

class TestRackDensityHotspots:
    def test_density_regimes(self):
        air = resolve_cooling_regime("air")
        assert therm.rack_density_regime(10.0, air) == therm.RackDensityRegime.NORMAL
        assert therm.rack_density_regime(25.0, air) == therm.RackDensityRegime.ELEVATED
        assert therm.rack_density_regime(35.0, air) == therm.RackDensityRegime.CRITICAL

    def test_liquid_tolerates_higher_density(self):
        liquid = resolve_cooling_regime("liquid")
        # 35 kW is CRITICAL for air but only normal/elevated for liquid.
        assert therm.rack_density_regime(35.0, liquid) != therm.RackDensityRegime.CRITICAL

    def test_hotspot_rises_with_density(self):
        air = resolve_cooling_regime("air")
        lo = therm.hotspot_risk(15.0, 1.0, 0.5, air)
        hi = therm.hotspot_risk(35.0, 0.5, 0.9, air)
        assert hi > lo

    def test_liquid_lowers_hotspot_but_not_zero(self):
        air = resolve_cooling_regime("air")
        liquid = resolve_cooling_regime("liquid")
        r_air = therm.hotspot_risk(35.0, 0.5, 0.9, air)
        r_liq = therm.hotspot_risk(35.0, 0.5, 0.9, liquid)
        assert 0.0 < r_liq < r_air

    def test_hotspot_persists(self):
        # An existing hotspot decays slowly even when instantaneous risk is 0.
        h = therm.hotspot_step(0.8, 0.0)
        assert 0.5 < h < 0.8   # persisted, not instantly cleared

    def test_inlet_includes_recirculation(self):
        air = resolve_cooling_regime("air")
        cold = therm.inlet_temperature(22.0, 0.0, air, _rng(0))
        hot = therm.inlet_temperature(22.0, 1.0, air, _rng(0))
        assert hot > cold


# ---------------------------------------------------------------------------
# Continuous slowdown
# ---------------------------------------------------------------------------

class TestSlowdown:
    def test_thermal_slowdown_continuous(self):
        cls = GPU_POWER_CLASSES["a100"]   # onset 83, max 90
        assert therm.thermal_slowdown_frac(70.0, cls) == 0.0      # below onset
        mid = therm.thermal_slowdown_frac(86.0, cls)
        hi = therm.thermal_slowdown_frac(90.0, cls)
        assert 0.0 < mid < hi                                     # continuous ramp
        assert hi <= thermal_value("thermal_slowdown_max") + 1e-9

    def test_power_slowdown(self):
        assert therm.power_slowdown_frac(100, 400) == 0.0         # well under cap
        assert therm.power_slowdown_frac(400, 400) > 0.0          # pinned at cap

    def test_throughput_factor(self):
        assert therm.throughput_factor(0.0, 0.0) == 1.0
        assert abs(therm.throughput_factor(0.3, 0.2) - 0.5) < 1e-9
        assert therm.throughput_factor(0.9, 0.9) == 0.05          # floored


# ---------------------------------------------------------------------------
# Telemetry confidence + thermal migration veto
# ---------------------------------------------------------------------------

class TestTelemetryAndVeto:
    def test_telemetry_tiers(self):
        assert therm.thermal_telemetry_confidence(True, True, 0) == "high"
        assert therm.thermal_telemetry_confidence(True, False, 2) == "medium"
        assert therm.thermal_telemetry_confidence(False, False, 9) == "low"

    def test_hot_destination_vetoed(self):
        assert therm.thermal_migration_blocked(85.0, 0.1, "high") is True
        assert therm.thermal_migration_blocked(50.0, 0.1, "high") is False

    def test_strong_hotspot_vetoed(self):
        assert therm.thermal_migration_blocked(50.0, 0.8, "high") is True

    def test_low_telemetry_more_conservative(self):
        # A borderline-warm destination is allowed under HIGH telemetry but the
        # threshold drops under LOW telemetry (missing ≠ safe).
        veto_c = thermal_value("thermal_migration_hot_veto_c")
        temp = veto_c - 2.0
        assert therm.thermal_migration_blocked(temp, 0.0, "high") is False
        assert therm.thermal_migration_blocked(temp, 0.0, "low") is True


# ---------------------------------------------------------------------------
# Integration: emergent simulator behaviour
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_gpus_run_realistic_temperatures(self):
        sim = ClusterSimulator(load_scenario("queue_surge_latency_sensitive").config, seed=42)
        sim.run_metrics_only(8)
        temps = [
            g.temperature_c
            for region in sim._cluster.regions.values()
            for node in region.nodes for g in node.gpus
            if g.assigned_workload_id is not None
        ]
        # Loaded GPUs sit well above ambient but below the hard limit.
        assert temps and max(temps) > 45.0 and max(temps) < 96.0

    def test_dense_air_rack_overheats_and_throttles(self):
        sim = ClusterSimulator(load_scenario("rack_density_overload_air").config, seed=42)
        ms = sim.run_metrics_only(12)
        m = ms[-1]
        assert m.rack_density_kw_max > 25.0          # dense rack
        assert m.hotspot_severity_max > 0.4          # persistent hotspot
        assert m.max_gpu_temp_c >= 82.0              # at/over throttle onset
        assert m.cooling_alarms > 0                  # critical-density alarms

    def test_liquid_cooling_beats_air_on_same_layout(self):
        air = ClusterSimulator(load_scenario("rack_density_overload_air").config, seed=42)
        liq = ClusterSimulator(load_scenario("rack_density_liquid_cooled").config, seed=42)
        air_m = air.run_metrics_only(12)[-1]
        liq_m = liq.run_metrics_only(12)[-1]
        assert liq_m.max_gpu_temp_c < air_m.max_gpu_temp_c - 10.0
        assert (liq_m.hotspot_severity_max or 0) < (air_m.hotspot_severity_max or 0)

    def test_thermal_throttle_reduces_throughput(self):
        # In the hot scenario the throttled workload serves fewer tokens than an
        # identical-load cool baseline (liquid).
        air = ClusterSimulator(load_scenario("rack_density_overload_air").config, seed=42)
        liq = ClusterSimulator(load_scenario("rack_density_liquid_cooled").config, seed=42)
        air_tokens = sum(m.total_tokens for m in air.run_metrics_only(12))
        liq_tokens = sum(m.total_tokens for m in liq.run_metrics_only(12))
        assert liq_tokens >= air_tokens

    def test_thermal_scenario_dominant_constraint(self):
        from aurelius.benchmarks.constraint_runner import (
            POLICY_CONSTRAINT_AWARE,
            POLICY_FIFO,
            ConstraintBenchmarkRunner,
        )
        r = ConstraintBenchmarkRunner(
            policies=[POLICY_FIFO, POLICY_CONSTRAINT_AWARE]
        ).run_scenario("thermal_hotspot_mixed_cluster", seed=42, steps=24)
        assert r.report.observed_dominant_constraint == "thermal"

    def test_thermal_migration_veto_into_hot_rack(self):
        # The hot, dense, high-utilization racks are flagged as unsafe migration
        # destinations by the thermal governor (migrating IN would land on heat).
        sim = ClusterSimulator(load_scenario("rack_density_overload_air").config, seed=42)
        sim.run_metrics_only(10)
        hot_racks = [
            n.rack_thermal for region in sim._cluster.regions.values()
            for n in region.nodes
            if n.rack_thermal is not None and n.rack_thermal.peak_gpu_temp_c >= 82.0
        ]
        assert hot_racks, "expected at least one hot rack in the dense scenario"
        assert any(
            therm.thermal_migration_blocked(
                rt.peak_gpu_temp_c, rt.hotspot.severity, rt.telemetry.tier
            )
            for rt in hot_racks
        )

    def test_deterministic_under_seed(self):
        def fp(seed):
            sim = ClusterSimulator(load_scenario("rack_density_overload_air").config, seed=seed)
            ms = sim.run_metrics_only(8)
            return [(round(m.max_gpu_temp_c or 0, 3), round(m.hotspot_severity_max or 0, 3),
                     m.thermal_throttle_events) for m in ms]
        assert fp(42) == fp(42)
        assert fp(42) != fp(7)


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden constants
# ---------------------------------------------------------------------------

class TestCalibrationMetadata:
    def test_thermal_params_have_full_provenance(self):
        assert THERMAL_PARAMS
        for name, p in THERMAL_PARAMS.items():
            assert p.source and p.source_type and p.calibration_notes, name
            assert p.confidence in ("high", "medium", "low"), name

    def test_combined_table_includes_thermal_group(self):
        assert "thermal" in {row.get("group") for row in calibration_table()}

    def test_cooling_regime_table(self):
        rows = cooling_regime_table()
        names = {r["name"] for r in rows}
        assert {"air", "liquid", "hybrid", "weak_airflow"} <= names
        liquid = next(r for r in rows if r["name"] == "liquid")
        assert liquid["beta_mult"] > 1.0 and liquid["density_mult"] > 1.0

    def test_power_class_table(self):
        rows = power_class_table()
        assert rows
        for r in rows:
            assert r["p_max_w"] > 0 and r["source"] and r["alpha"] > 0

    def test_thermal_params_overridable(self):
        assert thermal_value("thermal_beta_air", {"thermal_beta_air": 0.5}) == 0.5

    def test_all_regimes_resolvable(self):
        for name in COOLING_REGIMES:
            assert resolve_cooling_regime(name).beta_mult > 0
