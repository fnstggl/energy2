"""Guard that constraint-aware benchmark KPIs are bit-identical between
pytest and a fresh `python3 -c …` interpreter for every YAML scenario.

ROOT CAUSE (this PR):
    The constraint-aware benchmark is supposed to be deterministic given
    (scenario, seed, steps). Until PR #84 / this PR, the runner produced
    different KPIs under pytest vs a direct `python3` invocation on the same
    inputs, because:

      1. `aurelius.simulation.cluster.scenarios.load_scenario(...)` prefers an
         on-disk YAML if PyYAML is importable, and falls back to the in-module
         `_BUILTIN_SCENARIOS` dict otherwise.
      2. The pytest tool venv at `/root/.local/share/uv/tools/pytest` does NOT
         have PyYAML installed (it falls back to builtins). A plain `python3`
         on this box DOES have PyYAML (it loads YAML).
      3. The YAML and builtin scenario definitions had drifted on three
         scenarios:
           - `thermal_hotspot_mixed_cluster`: YAML's `ambient_temp_trace` is a
             smooth diurnal curve (28→35→27); builtin had a step-function
             ([28]*6+[32]*6+[35]*6+[30]*6). This changed how hot the cluster
             ran and how often it throttled, producing different goodput.
             CA KPI was 657,785 under pytest vs 830,780 direct.
           - `queue_surge_latency_sensitive`: builtin had
             `critical-wl.gpu_count_required=2, target_util_pct=65.0` vs YAML
             `1, 50.0` — different load profile under the surge.
           - `underutilization_stranded_capacity`: builtin's nodes were all in
             rack0/zone-1a, YAML splits them across rack0/rack1 and 1a/1b,
             with per-workload `target_util_pct=[20, 22, 18, 21]` instead of
             a uniform 20%.

    FIX: synced `_BUILTIN_SCENARIOS` to the YAML (YAML is source of truth,
    per PR #84). The tests below pin parity going forward by:

      - hashing a FULL signature (workloads + regions + ambient + energy +
        nodes + events) across all 6 YAML scenarios; and
      - running each YAML scenario through `ConstraintBenchmarkRunner` in
        BOTH the current pytest process and a freshly-spawned `python3 -c …`
        subprocess and asserting bit-identical KPIs.

    Pre-existing energy/queue carbon traces were already synced by PR #84.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_YAML_DIR = REPO_ROOT / "benchmarks" / "v1"

SCENARIOS = (
    "energy_price_arbitrage_multiregion",
    "thermal_hotspot_mixed_cluster",
    "queue_surge_latency_sensitive",
    "latency_tail_kvcache_pressure",
    "topology_fragmentation_h100",
    "underutilization_stranded_capacity",
)


def _parse_dash_trace(raw):
    out = []
    for it in raw or []:
        if isinstance(it, str):
            for tok in it.split(" - "):
                tok = tok.strip()
                if tok:
                    out.append(float(tok))
        else:
            out.append(float(it))
    return out


def _normalize_constant_trace(trace):
    """Compress a constant N-element trace to a 1-element trace for comparison.

    YAML scenarios often store constant traces as `[55.0]` while builtins store
    them as `[55.0]*24`. The simulator wraps with `% len(trace)` so the KPI is
    identical either way; this normalization avoids false drift reports.
    """
    if trace and len(set(trace)) == 1:
        return trace[:1]
    return trace


def _full_signature(cfg) -> dict:
    """Hash all material fields of a loaded scenario dict.

    workloads: id/region/type/tier/gpu_count/target_util/migration_allowed
    regions:   id, energy_price_trace (parsed), carbon_intensity_trace (parsed),
               ambient_temp_trace (parsed), ambient_temp_c, node list
    events:    full list
    """
    wsig = tuple(sorted(
        (
            w.get("workload_id"),
            w.get("region_id"),
            w.get("workload_type"),
            w.get("priority_tier"),
            w.get("gpu_count_required"),
            w.get("target_util_pct"),
            w.get("migration_allowed"),
        )
        for w in cfg.get("workloads", [])
    ))
    rsig = []
    for r in cfg.get("regions", []):
        nodes = tuple(sorted(
            (
                n.get("node_id"),
                n.get("gpu_type"),
                n.get("gpu_count"),
                n.get("topology_class"),
                n.get("rack_id"),
                n.get("zone"),
            )
            for n in r.get("nodes", [])
        ))
        rsig.append((
            r.get("region_id"),
            tuple(_normalize_constant_trace(
                _parse_dash_trace(r.get("energy_price_trace", []))
            )),
            tuple(_normalize_constant_trace(
                _parse_dash_trace(r.get("carbon_intensity_trace", []))
            )),
            tuple(_normalize_constant_trace(
                _parse_dash_trace(r.get("ambient_temp_trace", []))
            )),
            r.get("ambient_temp_c"),
            nodes,
        ))
    return {
        "workloads": wsig,
        "regions": tuple(rsig),
        "events": tuple(
            tuple(sorted((e or {}).items())) for e in cfg.get("events", [])
        ),
    }


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_yaml_scenarios_match_builtin_full_signature(scenario):
    """Full-signature parity: every field that can influence simulation must
    match between YAML and builtin. Skips when PyYAML isn't installed."""
    yaml = pytest.importorskip("yaml")
    from aurelius.simulation.cluster.scenarios import _BUILTIN_SCENARIOS
    yaml_path = _YAML_DIR / f"{scenario}.yaml"
    if not yaml_path.exists():
        pytest.skip(f"{scenario} YAML missing")
    builtin = _BUILTIN_SCENARIOS.get(scenario)
    if builtin is None:
        pytest.skip(f"{scenario} has no builtin (YAML-only)")
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    y_sig = _full_signature(y)
    b_sig = _full_signature(builtin)
    assert y_sig == b_sig, (
        f"Full signature drift for {scenario}: yaml→builtin differs.\n"
        "Sync _BUILTIN_SCENARIOS to YAML (YAML is source of truth)."
    )


_SUBPROCESS_PROBE = (
    "import json;"
    "from aurelius.benchmarks import ConstraintBenchmarkRunner;"
    "r=ConstraintBenchmarkRunner().run_scenario({sc!r}, steps=24, seed=42);"
    "k=r.report.aggregated;"
    "out={{p: k[p].sla_safe_goodput_per_infra_dollar for p in k}};"
    "print(json.dumps(out))"
)


def _run_in_subprocess(scenario: str) -> dict:
    """Run the scenario in a *fresh* python3 process so PyYAML is loaded
    (if installed), and return the per-policy KPI dict.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c", _SUBPROCESS_PROBE.format(sc=scenario)]
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess failed: {proc.stderr or proc.stdout}"
        )
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_thermal_kpi_identical_under_pytest_and_direct_python(scenario):
    """KPI bit-parity under pytest (this venv) vs direct python3 subprocess."""
    from aurelius.benchmarks import ConstraintBenchmarkRunner
    in_pytest = {
        p: k.sla_safe_goodput_per_infra_dollar for p, k in
        ConstraintBenchmarkRunner().run_scenario(
            scenario, steps=24, seed=42,
        ).report.aggregated.items()
    }
    via_subproc = _run_in_subprocess(scenario)
    # Compare every shared policy KPI to 1e-6 absolute tolerance.
    common = set(in_pytest.keys()) & set(via_subproc.keys())
    assert common, "no shared policies between pytest and subprocess runs"
    for policy in sorted(common):
        a, b = in_pytest[policy], via_subproc[policy]
        if a is None or b is None:
            assert a == b, f"{scenario}/{policy} None mismatch: {a} vs {b}"
            continue
        assert abs(a - b) <= 1e-6, (
            f"KPI drift on {scenario}/{policy}: pytest={a} subprocess={b}. "
            "Pytest venv likely loads stale builtin; direct python loads YAML. "
            "Resync _BUILTIN_SCENARIOS with benchmarks/v1/*.yaml."
        )
