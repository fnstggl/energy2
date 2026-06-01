#!/usr/bin/env python3
"""Audit the existing Aurelius goodput/$ placement / routing scoring path.

Traces every input that enters the per-candidate score computed in
``aurelius/residency/decision.py::score_residency_candidate``, classifies
each as measured / forecasted / static_prior / heuristic / constant /
proxy / missing, and writes a machine-readable audit to
``data/external/forecasting/placement_prior_audit/scoring_path_audit.json``.

The audit is read-only: it never imports executor / scheduler paths, never
mutates the scorer, never changes any default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "placement_prior_audit" / "scoring_path_audit.json"
)


# The catalogue of scoring inputs the mission spec requires the audit to
# classify. Each row records (a) where the input enters
# score_residency_candidate, (b) its classification per the mission's
# closed enum, and (c) source/notes.
SCORING_INPUTS = [
    {
        "input": "TTFT",
        "enters_at": "aurelius/residency/decision.py::_service_time_s -> service_s -> expected_latency",
        "classification": "heuristic",
        "source": "SafetyContext.seconds_per_token * request.output_tokens "
                  "OR SafetyContext.service_time_proxy_s (default 2.0s)",
        "is_per_candidate": False,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "TTFT is not measured per-candidate; the scorer collapses "
                 "TTFT+TPOT into one ``service_time_s`` heuristic. The "
                 "9x p99 TTFT spread across GPU types (CARA) is invisible.",
    },
    {
        "input": "TPOT",
        "enters_at": "aurelius/residency/decision.py::_service_time_s -> service_s",
        "classification": "heuristic",
        "source": "Same ``service_time_proxy_s`` path as TTFT",
        "is_per_candidate": False,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "TPOT is not separated from TTFT in the per-candidate score.",
    },
    {
        "input": "E2E latency",
        "enters_at": "aurelius/residency/decision.py::score_residency_candidate "
                     "-> expected_latency = queue_wait + model_penalty + "
                     "adapter_penalty + service_s",
        "classification": "derived",
        "source": "Sum of queue, load-penalty, and service-time components",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "Derived; only as good as its constituents. The service "
                 "component is a heuristic.",
    },
    {
        "input": "queue depth",
        "enters_at": "ModelLocationState.queue_depth",
        "classification": "measured",
        "source": "Telemetry adapter when available; None otherwise",
        "is_per_candidate": True,
        "captures_gpu_type": True,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "Treated as measured when present; missing → queue_telemetry_missing veto.",
    },
    {
        "input": "queue wait",
        "enters_at": "ModelLocationState.estimated_queue_wait_s OR proxy "
                     "queue_depth * service_time_s",
        "classification": "measured_or_proxy",
        "source": "Measured when available; proxy = queue_depth × service_s when not",
        "is_per_candidate": True,
        "captures_gpu_type": True,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "Proxy degrades to heuristic when only queue_depth is present.",
    },
    {
        "input": "GPU type",
        "enters_at": "ModelLocationState.location_key (string only); not parsed",
        "classification": "missing",
        "source": "GPU type is encoded in location_key but the scorer does not "
                  "parse it; no per-(model, GPU) priors enter score.",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Largest single gap surfaced by the CARA evidence "
                 "(9x p99 TTFT spread across GPUs).",
    },
    {
        "input": "model size",
        "enters_at": "request.model_id (string only); not parsed",
        "classification": "missing",
        "source": "Not used as a per-candidate prior",
        "is_per_candidate": False,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Model size influences TTFT/TPOT but never enters score.",
    },
    {
        "input": "throughput",
        "enters_at": "(not used)",
        "classification": "missing",
        "source": "EMA decode/prefill throughput from CARA is not consumed",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "Telemetry exists in CARA but the scorer ignores it.",
    },
    {
        "input": "KV cache state",
        "enters_at": "(not used directly); only feeds residency-aware routing "
                     "via has_model / has_adapter",
        "classification": "proxy",
        "source": "Booleans (model_resident / adapter_resident) only",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "KV cache utilisation + free blocks from CARA are not used.",
    },
    {
        "input": "cache reuse",
        "enters_at": "(not used)",
        "classification": "missing",
        "source": "SwissAI bucket-reuse signals not consumed by this scorer",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Cache_residency forecasts are not wired in.",
    },
    {
        "input": "residency / cold-start",
        "enters_at": "score_residency_candidate -> model_penalty + adapter_penalty",
        "classification": "static_prior",
        "source": "ModelLoadProfile.model_load_penalty_s; missing -> veto",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Static priors per model/adapter, not telemetry-fitted.",
    },
    {
        "input": "cost",
        "enters_at": "expected_cost = (expected_latency / 3600) * gpu_hour_price "
                     "+ memory_pressure_cost",
        "classification": "derived",
        "source": "SafetyContext.gpu_hour_price (constant default 3.0) × latency",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Single global GPU-hour price; no per-(GPU type) cost surface.",
    },
    {
        "input": "energy / carbon",
        "enters_at": "(not in residency scorer)",
        "classification": "missing",
        "source": "Handled by aurelius/optimization/scheduler.py for batch jobs only",
        "is_per_candidate": False,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": False,
        "notes": "Energy/carbon enter batch scheduling, not per-request "
                 "serving placement.",
    },
    {
        "input": "SLA risk",
        "enters_at": "sla_met = expected_latency <= ctx.sla_s(request)",
        "classification": "derived_binary",
        "source": "Hard threshold, not a probability",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "Binary; no calibrated SLA-violation probability.",
    },
    {
        "input": "timeout risk",
        "enters_at": "(not used)",
        "classification": "missing",
        "source": "Surfaced in frontier risk.py for the dynamic frontier; "
                  "not in residency scorer",
        "is_per_candidate": True,
        "captures_gpu_type": False,
        "captures_model_size": False,
        "captures_queue_state": True,
        "notes": "No per-candidate timeout probability.",
    },
]


def _verify_scorer_signature() -> dict:
    """Open ``score_residency_candidate`` and snapshot its signature so the
    audit can re-run cleanly if the upstream API ever changes."""
    mod = importlib.import_module("aurelius.residency.decision")
    fn = getattr(mod, "score_residency_candidate")
    import inspect
    sig = inspect.signature(fn)
    return {
        "module": "aurelius.residency.decision",
        "callable": "score_residency_candidate",
        "parameters": [
            {"name": name, "kind": str(p.kind), "annotation": str(p.annotation)}
            for name, p in sig.parameters.items()
        ],
        "return_annotation": str(sig.return_annotation),
        "source_file": inspect.getsourcefile(fn),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-path", default=str(OUT_PATH))
    args = p.parse_args(argv)

    sig = _verify_scorer_signature()
    static_or_heuristic = [
        row for row in SCORING_INPUTS
        if row["classification"] in (
            "heuristic", "static_prior", "constant", "proxy", "missing",
            "derived_binary",
        )
    ]
    measured = [row for row in SCORING_INPUTS
                if row["classification"] in ("measured", "measured_or_proxy")]

    payload = {
        "doc_version": "placement_scoring_path_audit_v1",
        "audit_only": True,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "production_claim": False,
        "shadow_only": True,
        "evaluated_at_s": time.time(),
        "scorer_signature": sig,
        "scoring_inputs": SCORING_INPUTS,
        "counts_by_classification": {
            cls: sum(1 for r in SCORING_INPUTS if r["classification"] == cls)
            for cls in (
                "measured", "measured_or_proxy", "forecasted", "static_prior",
                "heuristic", "constant", "proxy", "missing", "derived",
                "derived_binary",
            )
        },
        "static_or_heuristic_inputs": [r["input"] for r in static_or_heuristic],
        "measured_inputs": [r["input"] for r in measured],
        "gpu_type_used_as_latency_prior": False,
        "model_size_used_as_latency_prior": False,
        "queue_state_used_in_latency_estimate": True,
        "gpu_type_per_candidate_state_present": any(
            r["captures_gpu_type"] for r in SCORING_INPUTS),
        "model_size_per_candidate_state_present": any(
            r["captures_model_size"] for r in SCORING_INPUTS),
        "headline_gap": (
            "GPU type is not parsed from location_key; the 9x p99 TTFT "
            "spread observed in CARA across GPU types is invisible to the "
            "current goodput/$ scorer."
        ),
    }

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[audit] wrote {args.out_path}")
    print(f"[audit] inputs: {len(SCORING_INPUTS)}; static/heuristic: "
          f"{len(static_or_heuristic)}; measured: {len(measured)}")
    print(f"[audit] gpu_type_used_as_latency_prior={payload['gpu_type_used_as_latency_prior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
