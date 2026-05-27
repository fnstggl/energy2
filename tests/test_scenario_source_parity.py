"""Guard against YAML-vs-builtin scenario drift (benchmark determinism).

`load_scenario` prefers the on-disk YAML when PyYAML is installed and falls back
to `_BUILTIN_SCENARIOS` otherwise. If the two definitions drift, benchmark
results silently depend on whether PyYAML happens to be installed — which is
exactly the non-determinism that made the energy scenario produce different
greedy_energy behaviour under pytest (no PyYAML → builtin) vs a plain
interpreter (PyYAML → YAML).
"""

import glob
import os

import pytest

from aurelius.simulation.cluster.scenarios import _BUILTIN_SCENARIOS, load_scenario

_YAML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "v1")


def _signature(cfg: dict):
    workloads = tuple(sorted(w.get("workload_id") for w in cfg.get("workloads", [])))
    regions = tuple(sorted(r.get("region_id") for r in cfg.get("regions", [])))
    events = len(cfg.get("events", []))
    return workloads, regions, events


# --- yaml-free: builtins are the deterministic fallback and must be complete ---

def test_energy_arbitrage_builtin_has_flexible_west_workload():
    # batch-wl-west is what makes the scenario a real arbitrage. Its prior absence
    # in the builtin made benchmark results depend on PyYAML availability.
    cfg = _BUILTIN_SCENARIOS["energy_price_arbitrage_multiregion"]
    ids = {w["workload_id"] for w in cfg["workloads"]}
    assert {"batch-wl-east", "batch-wl-west", "inference-wl-east"} <= ids


def test_load_scenario_energy_is_deterministic_workload_count():
    # Whichever source loads (YAML or builtin), the scenario must have 3 workloads.
    sc = load_scenario("energy_price_arbitrage_multiregion", seed_override=42)
    assert len(sc.config.workloads) == 3


# --- where PyYAML exists: YAML and builtin must stay structurally identical ---

@pytest.mark.parametrize("yaml_path", sorted(glob.glob(os.path.join(_YAML_DIR, "*.yaml"))))
def test_builtin_matches_yaml_structure(yaml_path):
    yaml = pytest.importorskip("yaml", reason="PyYAML not installed in this venv")
    name = os.path.basename(yaml_path)[:-5]
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    builtin = _BUILTIN_SCENARIOS.get(name)
    if builtin is None:
        pytest.skip(f"{name} has no builtin (YAML-only scenario)")
    assert _signature(builtin) == _signature(y), (
        f"{name}: builtin drifted from YAML — benchmark results would depend on "
        "whether PyYAML is installed. Re-sync _BUILTIN_SCENARIOS with the YAML."
    )
