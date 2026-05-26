"""Scenario loader for constraint-aware cluster simulator.

Scenarios are defined as YAML files under benchmarks/v1/.
They are frozen (version-locked) — changes require a version bump.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .model import SimulatorConfig

_BENCHMARKS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks"
_V1_DIR = _BENCHMARKS_DIR / "v1"


@dataclass
class ScenarioConfig:
    """Loaded and validated scenario configuration."""
    name: str
    version: str
    description: str
    config: SimulatorConfig
    scenario_hash: str
    expected_primary_constraint: Optional[str] = None
    expected_events: list[str] = field(default_factory=list)
    validation_rules: list[dict[str, Any]] = field(default_factory=list)


def load_scenario(
    name: str, version: str = "v1", seed_override: Optional[int] = None
) -> ScenarioConfig:
    """Load a scenario by name from benchmarks/{version}/{name}.yaml.

    Falls back to built-in scenario definitions if YAML loading is unavailable
    or the file does not exist.

    Args:
        name: Scenario name (e.g., "energy_price_arbitrage_multiregion")
        version: Scenario version directory (default: "v1")
        seed_override: Override scenario seed for reproducibility testing

    Returns:
        ScenarioConfig with loaded and validated configuration
    """
    scenario_dir = _BENCHMARKS_DIR / version

    # Try YAML file first
    yaml_path = scenario_dir / f"{name}.yaml"
    if yaml_path.exists():
        try:
            raw = _load_yaml_or_json(yaml_path)
        except Exception:
            raw = None
        if raw is not None:
            if seed_override is not None:
                raw["seed"] = seed_override
            config = SimulatorConfig.from_dict(raw)
            scenario_hash = _hash_file(yaml_path)
            return ScenarioConfig(
                name=name,
                version=version,
                description=raw.get("description", name),
                config=config,
                scenario_hash=scenario_hash,
                expected_primary_constraint=raw.get("expected_primary_constraint"),
                expected_events=raw.get("expected_events", []),
                validation_rules=raw.get("validation_rules", []),
            )

    # Fall back to built-in scenario
    builtin = _BUILTIN_SCENARIOS.get(name)
    if builtin is None:
        available = list(_BUILTIN_SCENARIOS.keys())
        raise ValueError(
            f"Scenario {name!r} not found in {yaml_path} or built-ins. "
            f"Available built-ins: {available}"
        )

    raw = dict(builtin)
    if seed_override is not None:
        raw["seed"] = seed_override

    config = SimulatorConfig.from_dict(raw)
    scenario_hash = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]
    return ScenarioConfig(
        name=name,
        version=version,
        description=raw.get("description", name),
        config=config,
        scenario_hash=scenario_hash,
        expected_primary_constraint=raw.get("expected_primary_constraint"),
        expected_events=raw.get("expected_events", []),
        validation_rules=raw.get("validation_rules", []),
    )


def list_scenarios(version: str = "v1") -> list[str]:
    """List available scenario names."""
    scenario_dir = _BENCHMARKS_DIR / version
    names: list[str] = []

    if scenario_dir.exists():
        for f in sorted(scenario_dir.glob("*.yaml")):
            if not f.name.startswith("."):
                names.append(f.stem)
        for f in sorted(scenario_dir.glob("*.json")):
            if not f.name.startswith(".") and f.stem not in names:
                names.append(f.stem)

    # Also include built-ins not covered by files
    for name in _BUILTIN_SCENARIOS:
        if name not in names:
            names.append(name)

    return sorted(set(names))


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    with open(path) as f:
        return json.load(f)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Built-in scenarios (YAML fallback — identical to benchmarks/v1/*.yaml)
# ---------------------------------------------------------------------------

_BUILTIN_SCENARIOS: dict[str, dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # 1. Energy price arbitrage — flexible batch jobs shift to cheaper region
    # -----------------------------------------------------------------------
    "energy_price_arbitrage_multiregion": {
        "scenario_name": "energy_price_arbitrage_multiregion",
        "description": (
            "Anti-correlated regional energy price traces; flexible batch mix. "
            "Expected: defer/route reduces cost vs current_price_only without SLA violations."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "energy_bound",
        "expected_events": ["energy_price_spike"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [
                    45, 48, 52, 80, 110, 95, 70, 55, 48, 45, 43, 42,  # hours 0-11
                    48, 55, 62, 75, 120, 140, 130, 100, 75, 60, 52, 48,  # hours 12-23
                ],
                "carbon_intensity_trace": [200, 210, 220, 250, 280, 260] * 4,
                "ambient_temp_c": 22.0,
                "network_latency_to": {"us-west": 70, "eu-west": 100},
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "us-east-node1",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1b",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "batch-llm-east",
                        "base_arrival_rate_per_sec": 0.5,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [
                    60, 55, 50, 48, 45, 42, 40, 42, 45, 50, 55, 60,  # anti-correlated
                    58, 52, 48, 45, 43, 42, 45, 50, 55, 60, 62, 62,
                ],
                "carbon_intensity_trace": [150, 140, 130, 120, 110, 120] * 4,
                "ambient_temp_c": 18.0,
                "network_latency_to": {"us-east": 70, "eu-west": 140},
                "nodes": [
                    {
                        "node_id": "us-west-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "batch-llm-west",
                        "base_arrival_rate_per_sec": 0.3,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "batch-wl-east",
                "service_id": "batch-llm-east",
                "workload_type": "batch_training",
                "priority_tier": "batch",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 75.0,
                "communication_intensity": "medium",
                "migration_allowed": True,
                "latency_sensitive": False,
            },
            {
                "workload_id": "inference-wl-east",
                "service_id": "inference-svc-east",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 60.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sensitive": True,
                "latency_sla_p99_ms": 3000.0,
            },
        ],
        "events": [
            {"tick": 16, "type": "energy_price_spike", "region_id": "us-east", "multiplier": 2.5},
            {"tick": 20, "type": "energy_price_spike_end", "region_id": "us-east"},
        ],
    },

    # -----------------------------------------------------------------------
    # 2. Thermal hotspot — mixed cluster with hot rack
    # -----------------------------------------------------------------------
    "thermal_hotspot_mixed_cluster": {
        "scenario_name": "thermal_hotspot_mixed_cluster",
        "description": "A subset of nodes runs hot/throttling under load. "
                       "Expected: spread/reroute reduces throttling and protects p99.",
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "thermal_bound",
        "expected_events": ["thermal_hotspot"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 28.0,  # hot environment
                "ambient_temp_trace": [28] * 6 + [32] * 6 + [35] * 6 + [30] * 6,  # gets hotter
                "nodes": [
                    {
                        "node_id": "hot-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "hot-rack",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "hot-node1",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "hot-rack",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "cool-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "cool-rack",
                        "zone": "us-east-1b",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "llm-inference",
                        "base_arrival_rate_per_sec": 1.5,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "hot-wl",
                "service_id": "llm-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 85.0,  # high utilization → heat
                "communication_intensity": "low",
                "migration_allowed": True,
                "latency_sla_p99_ms": 5000.0,
            }
        ],
        "events": [
            {"tick": 6, "type": "thermal_hotspot", "node_id": "hot-node0", "extra_heat_c": 20.0},
            {"tick": 6, "type": "thermal_hotspot", "node_id": "hot-node1", "extra_heat_c": 20.0},
            {"tick": 18, "type": "thermal_hotspot_end", "node_id": "hot-node0"},
            {"tick": 18, "type": "thermal_hotspot_end", "node_id": "hot-node1"},
        ],
    },

    # -----------------------------------------------------------------------
    # 3. Queue surge — latency-sensitive SLA
    # -----------------------------------------------------------------------
    "queue_surge_latency_sensitive": {
        "scenario_name": "queue_surge_latency_sensitive",
        "description": (
            "Arrival burst on one region with a latency-sensitive SLA. "
            "Expected: reroute/spread cuts queue wait and p95 without breaching capacity buffer."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "queue_bound",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [50.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "critical-inference",
                        "base_arrival_rate_per_sec": 2.0,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [48.0] * 24,
                "ambient_temp_c": 20.0,
                "nodes": [
                    {
                        "node_id": "us-west-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "critical-inference-west",
                        "base_arrival_rate_per_sec": 0.5,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "critical-wl",
                "service_id": "critical-inference",
                "workload_type": "inference",
                "priority_tier": "latency_sensitive",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 65.0,
                "communication_intensity": "low",
                "migration_allowed": False,   # critical — no migration
                "latency_sensitive": True,
                "latency_sla_p99_ms": 2000.0,
                "queue_sla_p95_ms": 500.0,
            },
            {
                "workload_id": "critical-wl-west",
                "service_id": "critical-inference-west",
                "workload_type": "inference",
                "priority_tier": "latency_sensitive",
                "region_id": "us-west",
                "gpu_count_required": 2,
                "target_util_pct": 40.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sensitive": True,
                "latency_sla_p99_ms": 2000.0,
            },
        ],
        "events": [
            {
                "tick": 8, "type": "queue_surge",
                "service_id": "critical-inference", "multiplier": 5.0,
            },
            {"tick": 16, "type": "queue_surge_end", "service_id": "critical-inference"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4. KV cache pressure — TTFT spike
    # -----------------------------------------------------------------------
    "latency_tail_kvcache_pressure": {
        "scenario_name": "latency_tail_kvcache_pressure",
        "description": (
            "Rising p99/TTFT with high KV-cache usage under steady arrivals. "
            "Expected: scale/spread reduces tail; migration NOT recommended for critical."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "memory_bound_indirect",
        "expected_events": ["kv_cache_pressure"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [50.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "llm-critical",
                        "base_arrival_rate_per_sec": 1.5,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "llm-critical-wl",
                "service_id": "llm-critical",
                "workload_type": "inference",
                "priority_tier": "critical",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 75.0,
                "memory_intensity": "high",
                "migration_allowed": False,   # critical — assert no migration recommended
                "latency_sensitive": True,
                "latency_sla_p99_ms": 2000.0,
                "kv_cache_usage_frac": 0.3,
                "prefix_cache_hit_rate_frac": 0.6,
            }
        ],
        "events": [
            {"tick": 6, "type": "kv_cache_pressure", "service_id": "llm-critical",
             "kv_cache_usage_frac": 0.92, "prefix_cache_hit_rate_frac": 0.08},
            {"tick": 18, "type": "kv_cache_pressure_end", "service_id": "llm-critical"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4b. Prefix-affinity energy arbitrage — cache-aware routing should win
    # -----------------------------------------------------------------------
    # A high-prefix-overlap inference workload with a long shared prefix, free to
    # migrate, in an anti-correlated energy market. Naive energy-greedy rerouting
    # chases cheap power across regions, loses the warm prefix cache on every hop
    # (cold_route_penalty), and pays it back in TTFT — so it should LOSE on
    # latency vs the affinity-preserving constraint-aware policy.
    "prefix_affinity_energy_arbitrage": {
        "scenario_name": "prefix_affinity_energy_arbitrage",
        "description": (
            "High-prefix-overlap inference, migration allowed, anti-correlated "
            "energy. Expected: cache-aware affinity preservation beats naive "
            "energy-greedy rerouting on TTFT/p99 (cold-route penalties dominate)."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "energy_bound",
        "expected_events": ["energy_price_spike"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [
                    45, 48, 52, 80, 110, 95, 70, 55, 48, 45, 43, 42,
                    48, 55, 62, 75, 120, 140, 130, 100, 75, 60, 52, 48,
                ],
                "ambient_temp_c": 22.0,
                "network_latency_to": {"us-west": 70},
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "chat-affinity",
                        "base_arrival_rate_per_sec": 2.0,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [
                    60, 55, 50, 48, 45, 42, 40, 42, 45, 50, 55, 60,
                    58, 52, 48, 45, 43, 42, 45, 50, 55, 60, 62, 62,
                ],
                "ambient_temp_c": 18.0,
                "network_latency_to": {"us-east": 70},
                "nodes": [
                    {
                        "node_id": "us-west-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "chat-affinity-west",
                        "base_arrival_rate_per_sec": 0.3,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "chat-affinity-wl",
                "service_id": "chat-affinity",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 65.0,
                "communication_intensity": "low",
                "migration_allowed": True,
                # Not hard latency-sensitive (so naive energy-greedy WILL reroute
                # it), but it still carries an SLA and a warm prefix cache to lose.
                "latency_sensitive": False,
                "latency_sla_p99_ms": 4000.0,
                # High overlap + long shared prefix → lots to lose on a cold hop.
                "model_kv_profile": "llama3-8b",
                "prefix_overlap": 0.85,
                "avg_seq_len_tokens": 4096,
            }
        ],
        "events": [
            {"tick": 10, "type": "energy_price_spike", "region_id": "us-east",
             "multiplier": 2.5},
            {"tick": 14, "type": "energy_price_spike_end", "region_id": "us-east"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4c. KV exhaustion / preemption storm — pressure → 1.0
    # -----------------------------------------------------------------------
    # Long contexts + a sustained arrival surge on a workload whose weights leave
    # little KV headroom drive KV pressure into the preemption region: preemption
    # count climbs, recompute penalties spike TTFT, and tails explode.
    "kv_exhaustion_preemption_storm": {
        "scenario_name": "kv_exhaustion_preemption_storm",
        "description": (
            "Long-context workload with thin KV headroom under an arrival surge. "
            "Expected: KV pressure enters the preemption region; preemptions and "
            "recompute spike TTFT/p99 (decode instability under memory pressure)."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "memory_bound_indirect",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "serving_config": {"enable_bursts": False},
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [50.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 2,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "longctx-inference",
                        "base_arrival_rate_per_sec": 3.0,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "longctx-wl",
                "service_id": "longctx-inference",
                "workload_type": "inference",
                "priority_tier": "critical",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 80.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sensitive": True,
                "latency_sla_p99_ms": 3000.0,
                # Classic MHA (32 KV heads) + very long context + large weights
                # leaving thin KV headroom → high KV bytes/token → pressure → 1.
                "model_kv_profile": "llama2-7b",
                "prefix_overlap": 0.2,
                "avg_seq_len_tokens": 8192,
                "memory_required_bytes": 64424509440,
            }
        ],
        "events": [
            {"tick": 6, "type": "queue_surge", "service_id": "longctx-inference",
             "multiplier": 4.0},
            {"tick": 18, "type": "queue_surge_end", "service_id": "longctx-inference"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4d. Startup-heavy migration — TensorRT-LLM cold-start storm
    # -----------------------------------------------------------------------
    # A compile-heavy TensorRT-LLM workload free to chase cheap energy. Each
    # migration triggers a multi-minute engine-build cold start; abrupt rerouting
    # should drown TTFT in startup + tail uplift, so naive arbitrage LOSES.
    "startup_heavy_migration_trtllm": {
        "scenario_name": "startup_heavy_migration_trtllm",
        "description": (
            "Compile-heavy TensorRT-LLM workload in an anti-correlated energy "
            "market. Expected: abrupt energy-greedy rerouting pays multi-minute "
            "cold starts + tail uplift and loses on TTFT/p99 vs migration restraint."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "energy_bound",
        "expected_events": ["energy_price_spike"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [
                    45, 48, 52, 80, 110, 95, 70, 55, 48, 45, 43, 42,
                    48, 55, 62, 75, 120, 140, 130, 100, 75, 60, 52, 48,
                ],
                "ambient_temp_c": 22.0,
                "network_latency_to": {"us-west": 70},
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "trt-inference",
                        "base_arrival_rate_per_sec": 1.5,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [
                    60, 55, 50, 48, 45, 42, 40, 42, 45, 50, 55, 60,
                    58, 52, 48, 45, 43, 42, 45, 50, 55, 60, 62, 62,
                ],
                "ambient_temp_c": 18.0,
                "network_latency_to": {"us-east": 70},
                "nodes": [
                    {
                        "node_id": "us-west-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "trt-inference-west",
                        "base_arrival_rate_per_sec": 0.3,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "trt-wl",
                "service_id": "trt-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 65.0,
                "communication_intensity": "low",
                "migration_allowed": True,
                "latency_sensitive": False,
                "latency_sla_p99_ms": 4000.0,
                "engine_runtime": "tensorrt-llm",   # compile-heavy cold start
                "model_kv_profile": "llama3-8b",
                "prefix_overlap": 0.7,
                "avg_seq_len_tokens": 2048,
            }
        ],
        "events": [
            {"tick": 10, "type": "energy_price_spike", "region_id": "us-east",
             "multiplier": 2.5},
            {"tick": 14, "type": "energy_price_spike_end", "region_id": "us-east"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4e. Proxy bottleneck — ingress saturation dominates scaling
    # -----------------------------------------------------------------------
    # High arrival rate against a small replica set: the proxy/ingress saturates,
    # so replica count alone does not deliver throughput — queue wait and TTFT
    # are dominated by proxy queue amplification.
    "proxy_bottleneck_ingress": {
        "scenario_name": "proxy_bottleneck_ingress",
        "description": (
            "High RPS against few replicas: proxy/ingress saturation dominates "
            "queue wait and TTFT (replica count alone does not set throughput)."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "queue_bound",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "serving_config": {"proxy_capacity_rps_per_replica": 20.0},
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [50.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "us-east-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "proxy-bound-inference",
                        "base_arrival_rate_per_sec": 30.0,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "proxy-wl",
                "service_id": "proxy-bound-inference",
                "workload_type": "inference",
                "priority_tier": "latency_sensitive",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 70.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sensitive": True,
                "latency_sla_p99_ms": 2000.0,
                "engine_runtime": "vllm",
                "prefix_overlap": 0.5,
                "avg_seq_len_tokens": 1024,
            }
        ],
        "events": [
            {"tick": 8, "type": "queue_surge", "service_id": "proxy-bound-inference",
             "multiplier": 3.0},
            {"tick": 16, "type": "queue_surge_end", "service_id": "proxy-bound-inference"},
        ],
    },

    # -----------------------------------------------------------------------
    # 4f. Rack-density overload (AIR) — dense H100 rack overheats + throttles
    # -----------------------------------------------------------------------
    # A single rack packed with many H100 nodes at high utilization pushes the
    # rack well past the air-cooling kW envelope → critical density, persistent
    # hotspots, sustained thermal throttling. The same layout under LIQUID
    # cooling (next scenario) stays safe — cooling regime matters.
    "rack_density_overload_air": {
        "scenario_name": "rack_density_overload_air",
        "description": (
            "Dense H100 rack (air-cooled) at high utilization exceeds the air kW "
            "envelope → critical density, hotspots, sustained throttling. "
            "Expected: thermal-dominant; spreading/consolidation must respect heat."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "thermal_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 26.0,
                "nodes": [
                    {
                        "node_id": f"dense-node{i}",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 8,
                        "topology_class": "nvswitch",
                        "rack_id": "dense-rack",   # all nodes share ONE rack
                        "zone": "us-east-1a",
                        "cooling_regime": "air",
                    }
                    for i in range(8)              # 8 nodes × 8 H100 × 700W ≈ 45 kW
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "dense-inference",
                        "base_arrival_rate_per_sec": 0.5,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "dense-wl",
                "service_id": "dense-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 32,          # pack the rack densely
                "target_util_pct": 90.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sla_p99_ms": 8000.0,
                "engine_runtime": "vllm",
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 4g. Rack-density LIQUID-cooled — same dense layout stays safe
    # -----------------------------------------------------------------------
    "rack_density_liquid_cooled": {
        "scenario_name": "rack_density_liquid_cooled",
        "description": (
            "Same dense H100 rack as rack_density_overload_air but LIQUID-cooled: "
            "higher density tolerance + faster recovery keep temps and throttling "
            "far lower. Expected: liquid materially beats air on thermal stability."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "thermal_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 26.0,
                "nodes": [
                    {
                        "node_id": f"dense-node{i}",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 8,
                        "topology_class": "nvswitch",
                        "rack_id": "dense-rack",
                        "zone": "us-east-1a",
                        "cooling_regime": "liquid",
                    }
                    for i in range(8)
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "dense-inference",
                        "base_arrival_rate_per_sec": 0.5,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "dense-wl",
                "service_id": "dense-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 32,
                "target_util_pct": 90.0,
                "communication_intensity": "low",
                "migration_allowed": False,
                "latency_sla_p99_ms": 8000.0,
                "engine_runtime": "vllm",
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 5. Topology fragmentation — H100 NVSwitch vs PCIe
    # -----------------------------------------------------------------------
    "topology_fragmentation_h100": {
        "scenario_name": "topology_fragmentation_h100",
        "description": "Multi-GPU collective workload on poor vs good interconnect. "
                       "Expected: placement recommendation moves to NVLink-connected GPUs.",
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "topology_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [50.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "nvswitch-node",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 8,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "pcie-node",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 8,
                        "topology_class": "pcie_multi_numa",
                        "rack_id": "us-east-rack1",
                        "zone": "us-east-1a",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "distributed-training",
                        "base_arrival_rate_per_sec": 0.1,
                        "diurnal_amplitude": 0.1,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "training-wl-bad-topo",
                "service_id": "distributed-training",
                "workload_type": "batch_training",
                "priority_tier": "batch",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 80.0,
                "communication_intensity": "high",  # critical: high communication
                "memory_intensity": "high",
                "migration_allowed": True,
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 6. Underutilization — stranded capacity
    # -----------------------------------------------------------------------
    "underutilization_stranded_capacity": {
        "scenario_name": "underutilization_stranded_capacity",
        "description": "Many half-idle allocated GPUs. "
                       "Expected: consolidation raises mean utilization and frees nodes.",
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": f"sparse-node{i}",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                    for i in range(4)
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "sparse-inference",
                        "base_arrival_rate_per_sec": 0.3,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": f"sparse-wl-{i}",
                "service_id": "sparse-inference",
                "workload_type": "inference",
                "priority_tier": "flexible",
                "region_id": "us-east",
                "gpu_count_required": 1,
                "target_util_pct": 20.0,  # very low — underutilized
                "communication_intensity": "low",
                "migration_allowed": True,
            }
            for i in range(4)
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 12. Tensor-parallel topology collapse — TP job split across racks
    # -----------------------------------------------------------------------
    # A tensor-parallel workload needs 4 GPUs but the region's GPUs are spread
    # across two racks (2+2). The all-reduce-per-layer collective is paced by
    # the worst (cross-rack) hop → throughput collapses and p95/p99 blow up.
    # Same layout with the GPUs co-located on one NVSwitch node would be safe.
    "tensor_parallel_topology_collapse": {
        "scenario_name": "tensor_parallel_topology_collapse",
        "description": (
            "Tensor-parallel job (all-reduce per layer) forced to span two racks. "
            "Expected: communication penalty collapses throughput and amplifies "
            "p95/p99; NVSwitch co-location would avoid it. Topology-dominant."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "topology_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "tp-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 2,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "tp-node1",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 2,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack1",
                        "zone": "us-east-1b",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "tp-inference",
                        "base_arrival_rate_per_sec": 3.0,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "tp-wl",
                "service_id": "tp-inference",
                "workload_type": "inference",
                "priority_tier": "latency_sensitive",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 70.0,
                "communication_intensity": "high",
                "comm_profile": "tensor_parallel",
                "comm_message_bytes": 8 * 1024 * 1024,
                "latency_sensitive": True,
                "latency_sla_p99_ms": 3000.0,
                "migration_allowed": True,
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 13. MoE all-to-all hotspot + NIC saturation under congestion
    # -----------------------------------------------------------------------
    # An expert-parallel (MoE) workload with all-to-all dispatch/combine traffic
    # co-located on PCIe nodes. As load rises the fabric oversubscribes and the
    # all-to-all hotspot amplifies communication → throughput drops, NIC saturates.
    "moe_hotspot_nic_saturation": {
        "scenario_name": "moe_hotspot_nic_saturation",
        "description": (
            "MoE / expert-parallel all-to-all traffic on PCIe nodes under a load "
            "surge. Expected: all-to-all hotspot amplification + NIC saturation + "
            "fabric congestion degrade throughput and tails. Topology-dominant."
        ),
        "seed": 7,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "topology_bound",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "moe-node0",
                        "gpu_type": "a100-pcie-80gb",
                        "gpu_count": 4,
                        "topology_class": "pcie_multi_numa",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    },
                    {
                        "node_id": "moe-node1",
                        "gpu_type": "a100-pcie-80gb",
                        "gpu_count": 4,
                        "topology_class": "pcie_multi_numa",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "moe-inference",
                        "base_arrival_rate_per_sec": 4.0,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "moe-wl",
                "service_id": "moe-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 8,
                "target_util_pct": 75.0,
                "communication_intensity": "high",
                "comm_profile": "moe_expert",
                "comm_message_bytes": 16 * 1024 * 1024,
                "migration_allowed": True,
            }
        ],
        "events": [
            {"tick": 8, "type": "queue_surge", "region_id": "us-east",
             "service_id": "moe-inference", "multiplier": 2.5},
            {"tick": 16, "type": "queue_surge_end", "region_id": "us-east",
             "service_id": "moe-inference"},
        ],
    },

    # -----------------------------------------------------------------------
    # 14. Degraded topology telemetry — missing NVLink/NIC visibility
    # -----------------------------------------------------------------------
    # Topology telemetry is partial (NVLink + NIC maps missing). The simulator
    # must NOT assume ideal proximity: confidence drops, the usable topology
    # score is discounted, and the topology-aware migration governor becomes more
    # conservative (cross-domain vetoes trigger at a lower distance threshold).
    "degraded_topology_telemetry": {
        "scenario_name": "degraded_topology_telemetry",
        "description": (
            "Partial topology telemetry (missing NVLink + NIC maps, stale topo). "
            "Expected: lowered placement confidence, discounted topology score, "
            "more conservative topology-aware migration vetoes. Missing != safe."
        ),
        "seed": 11,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "topology_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [80.0] * 24,
                "ambient_temp_c": 22.0,
                "network_latency_to": {"us-west": 70},
                "nodes": [
                    {
                        "node_id": "blind-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                        "nvlink_telemetry_visible": False,
                        "nic_telemetry_visible": False,
                        "topology_stale_ticks": 6,
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "blind-train",
                        "base_arrival_rate_per_sec": 1.0,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [35.0] * 24,
                "ambient_temp_c": 18.0,
                "network_latency_to": {"us-east": 70},
                "nodes": [
                    {
                        "node_id": "west-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "blind-train-west",
                        "base_arrival_rate_per_sec": 0.1,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "blind-train-wl",
                "service_id": "blind-train",
                "workload_type": "batch_training",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 70.0,
                "communication_intensity": "high",
                "comm_profile": "all_reduce_training",
                "migration_allowed": True,
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 15. DRAM-bound inference (utilization paradox)
    # -----------------------------------------------------------------------
    # A memory-bandwidth-bound inference service: moderate SM utilization yet
    # high DRAM_ACTIVE → effective throughput is pinned by memory bandwidth, not
    # compute. The GPU looks "underutilized" (low SM) but is NOT a safe packing
    # candidate (it is busy moving KV bytes). The utilization paradox.
    "dram_bound_inference": {
        "scenario_name": "dram_bound_inference",
        "description": (
            "Memory-bandwidth-bound inference: low SM utilization + high "
            "DRAM_ACTIVE. Expected: utilization paradox, memory-bound throughput "
            "cap, NOT a safe packing target. Utilization-dominant."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "dram-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "dram-inference",
                        "base_arrival_rate_per_sec": 12.0,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "dram-wl",
                "service_id": "dram-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 45.0,
                "communication_intensity": "low",
                "memory_intensity": "high",
                "workload_class": "memory_heavy",
                "migration_allowed": True,
            }
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 16. Scheduler-bound inference
    # -----------------------------------------------------------------------
    # Very high concurrency drives the active-sequence count past the scheduler
    # capacity: admission + scheduling overhead binds before compute, so the GPU
    # is scheduler-bound (low effective throughput at high offered load).
    "scheduler_bound_inference": {
        "scenario_name": "scheduler_bound_inference",
        "description": (
            "Very high concurrency exceeds scheduler capacity → scheduler-bound "
            "throughput cap. Expected: scheduler bottleneck dominates, effective "
            "utilization below compute. Utilization-dominant."
        ),
        "seed": 5,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "sched-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 2,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "sched-inference",
                        "base_arrival_rate_per_sec": 120.0,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "sched-wl",
                "service_id": "sched-inference",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 60.0,
                "communication_intensity": "low",
                "workload_class": "standard_inference",
                "migration_allowed": True,
            }
        ],
        "events": [
            {"tick": 6, "type": "queue_surge", "region_id": "us-east",
             "service_id": "sched-inference", "multiplier": 3.0},
            {"tick": 18, "type": "queue_surge_end", "region_id": "us-east",
             "service_id": "sched-inference"},
        ],
    },

    # -----------------------------------------------------------------------
    # 17. Fragmentation / stranded capacity
    # -----------------------------------------------------------------------
    # Free GPUs scattered as 1-GPU fragments across racks while a 4-GPU job needs
    # a contiguous topology-local block. The cluster looks underutilized yet
    # cannot place the large job → stranded capacity, high fragmentation.
    "fragmentation_stranded_capacity": {
        "scenario_name": "fragmentation_stranded_capacity",
        "description": (
            "Free GPUs scattered as single-GPU fragments across racks; a 4-GPU "
            "job cannot be placed. Expected: high fragmentation, stranded capacity "
            "despite free GPUs. Free GPUs are NOT universally schedulable."
        ),
        "seed": 42,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "big-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rackA",
                        "zone": "us-east-1a",
                    },
                ] + [
                    {
                        "node_id": f"frag-node{i}",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 2,
                        "topology_class": "nvswitch",
                        "rack_id": f"us-east-rack{chr(66 + i)}",
                        "zone": "us-east-1a",
                    }
                    for i in range(3)
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "frag-big",
                        "base_arrival_rate_per_sec": 0.5,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            }
        ],
        "workloads": [
            # Large 4-GPU job occupies the only contiguous block (defines demand).
            {
                "workload_id": "frag-big-wl",
                "service_id": "frag-big",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 4,
                "target_util_pct": 60.0,
                "communication_intensity": "low",
                "migration_allowed": True,
            },
        ] + [
            # Single-GPU jobs occupy one GPU on each 2-GPU fragment node, leaving
            # 1 free GPU per rack — none can host another 4-GPU job.
            {
                "workload_id": f"frag-small-{i}",
                "service_id": f"frag-small-{i}",
                "workload_type": "inference",
                "priority_tier": "flexible",
                "region_id": "us-east",
                "gpu_count_required": 1,
                "target_util_pct": 30.0,
                "communication_intensity": "low",
                "migration_allowed": True,
            }
            for i in range(3)
        ],
        "events": [],
    },

    # -----------------------------------------------------------------------
    # 18. Unsafe aggressive consolidation
    # -----------------------------------------------------------------------
    # A communication-heavy workload sharded across nodes under queue + thermal
    # pressure. Consolidation risk climbs into the unsafe regime; the packing
    # governor vetoes further consolidation / cross-region migration.
    "unsafe_aggressive_consolidation": {
        "scenario_name": "unsafe_aggressive_consolidation",
        "description": (
            "Communication-heavy workload sharded across nodes under queue + "
            "thermal pressure. Expected: consolidation risk enters the unsafe "
            "regime; packing governor vetoes risky migrations. Utilization-dominant."
        ),
        "seed": 9,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": ["queue_surge"],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [90.0] * 24,
                "ambient_temp_c": 30.0,
                "network_latency_to": {"us-west": 70},
                "nodes": [
                    {
                        "node_id": "cons-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                        "cooling_regime": "weak_airflow",
                    },
                    {
                        "node_id": "cons-node1",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack1",
                        "zone": "us-east-1b",
                        "cooling_regime": "weak_airflow",
                    },
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "cons-svc",
                        "base_arrival_rate_per_sec": 6.0,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            },
            {
                "region_id": "us-west",
                "energy_price_trace": [35.0] * 24,
                "ambient_temp_c": 18.0,
                "network_latency_to": {"us-east": 70},
                "nodes": [
                    {
                        "node_id": "west-node0",
                        "gpu_type": "h100-sxm5-80gb",
                        "gpu_count": 8,
                        "topology_class": "nvswitch",
                        "rack_id": "us-west-rack0",
                        "zone": "us-west-2a",
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-west-q0",
                        "service_id": "cons-svc-west",
                        "base_arrival_rate_per_sec": 0.1,
                        "diurnal_amplitude": 0.2,
                    }
                ],
            },
        ],
        "workloads": [
            {
                "workload_id": "cons-wl",
                "service_id": "cons-svc",
                "workload_type": "inference",
                "priority_tier": "latency_sensitive",
                "region_id": "us-east",
                "gpu_count_required": 8,
                "target_util_pct": 80.0,
                "communication_intensity": "high",
                "comm_profile": "tensor_parallel",
                "workload_class": "comm_heavy",
                "flexibility": "low",
                "migration_allowed": True,
            }
        ],
        "events": [
            {"tick": 6, "type": "queue_surge", "region_id": "us-east",
             "service_id": "cons-svc", "multiplier": 3.0},
        ],
    },

    # -----------------------------------------------------------------------
    # 19. Partial utilization telemetry
    # -----------------------------------------------------------------------
    # GPU_UTIL + DRAM telemetry is partial/stale. Packing confidence drops and the
    # packing governor becomes more conservative (missing != schedulable).
    "partial_utilization_telemetry": {
        "scenario_name": "partial_utilization_telemetry",
        "description": (
            "Partial / stale utilization + DRAM telemetry. Expected: lowered "
            "packing confidence, more conservative consolidation. Missing "
            "telemetry must NOT be read as schedulable."
        ),
        "seed": 11,
        "tick_duration_hours": 1.0,
        "expected_primary_constraint": "utilization_bound",
        "expected_events": [],
        "scenario_version": "v1",
        "simulator_version": "1.0.0",
        "regions": [
            {
                "region_id": "us-east",
                "energy_price_trace": [55.0] * 24,
                "ambient_temp_c": 22.0,
                "nodes": [
                    {
                        "node_id": "blind-util-node0",
                        "gpu_type": "a100-sxm4-80gb",
                        "gpu_count": 4,
                        "topology_class": "nvswitch",
                        "rack_id": "us-east-rack0",
                        "zone": "us-east-1a",
                        "nvlink_telemetry_visible": False,
                        "pcie_telemetry_visible": False,
                        "topology_stale_ticks": 6,
                    }
                ],
                "queues": [
                    {
                        "queue_id": "us-east-q0",
                        "service_id": "blind-util-svc",
                        "base_arrival_rate_per_sec": 2.0,
                        "diurnal_amplitude": 0.3,
                    }
                ],
            }
        ],
        "workloads": [
            {
                "workload_id": "blind-util-wl",
                "service_id": "blind-util-svc",
                "workload_type": "inference",
                "priority_tier": "standard",
                "region_id": "us-east",
                "gpu_count_required": 2,
                "target_util_pct": 55.0,
                "communication_intensity": "low",
                "migration_allowed": True,
            }
        ],
        "events": [],
    },
}
