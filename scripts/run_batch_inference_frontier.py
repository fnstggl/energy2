#!/usr/bin/env python3
"""Run the Batch Inference Frontier v1 sweep over the committed Azure 2024
sample (treated as a synthetic batch-flex scenario, NOT as a native batch
trace).

Opt-in, shadow only, recommendation-only. No real cluster execution.

The script writes a small summary JSON under
``data/external/frontier/batch_inference_frontier_v1_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurelius.frontier import (  # noqa: E402
    batch_inference_estimator as bie,
    batch_inference_models as bim,
    batch_inference_safety as bis,
)
from aurelius.frontier.batch_inference_controller import (  # noqa: E402
    choose_batch_inference_frontier_target,
)
from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402
from aurelius.traces.schema import time_rescale  # noqa: E402

AZURE_FIXTURE = (REPO_ROOT / "tests" / "fixtures"
                 / "azure_llm_2024_sample.csv")
AZURE_FULL = (REPO_ROOT / "data" / "external" / "azure_llm_2024" / "raw"
              / "AzureLLMInferenceTrace_conv_1week.csv")
DEFAULT_OUTPUT = (REPO_ROOT / "data" / "external" / "frontier"
                  / "batch_inference_frontier_v1_summary.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run Batch Inference Frontier v1.")
    p.add_argument("--source", default=str(AZURE_FIXTURE),
                   help="Azure 2024 CSV (defaults to the committed fixture)")
    p.add_argument("--scale-rps", type=float, default=100.0,
                   help="time-rescale factor; mirrors the canonical audit's "
                        "10x/100x sweep (Azure 2024 raw rate is low)")
    p.add_argument("--tick-seconds", type=float, default=60.0)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args(argv)

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"[batch] source not found: {src_path}", file=sys.stderr)
        return 2
    print(f"[batch] source: {src_path}")

    reqs = azure_llm.load_csv(str(src_path))
    if args.scale_rps != 1.0:
        reqs = time_rescale(reqs, factor=args.scale_rps)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=args.tick_seconds)
    active = [t for t in ticks if t.request_count > 0]
    print(f"[batch] requests={len(reqs):,} ticks={len(ticks):,} "
          f"active_ticks={len(active):,}")

    profile = bim.BatchInferenceWorkloadProfile(
        workload_id="azure_2024_batch_inference_workload_v1",
        trace_source="azure_llm_2024",
        synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1",
        deadline_slack_seconds_baseline=600.0,
        deadline_miss_rate_sla_pct=2.0,
        queue_wait_sla_p99_ms=2000.0,
        telemetry_confidence="medium",
        source=str(src_path),
    )

    rho_grid = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    slack_grid_s = (0.0, 30.0, 60.0, 300.0, 900.0, 3600.0)
    candidates = [
        bim.BatchInferenceFrontierCandidate(
            target_rho=R, deadline_slack_seconds=s,
            source_policy=f"rho{R}_slack{s}s")
        for R in rho_grid for s in slack_grid_s
    ]

    points = bie.estimate_batch_inference_frontier(
        profile, ticks, candidates,
        estimator_config=bie.BatchInferenceEstimatorConfig(
            mode=bie.ANTICIPATORY, tick_seconds=args.tick_seconds),
        safety_config=bis.BatchInferenceSafetyConfig(
            max_deadline_miss_rate_pct=2.0,
            max_timeout_pct=10.0,
            max_queue_p99_ms=2000.0))

    decision = choose_batch_inference_frontier_target(profile, points)

    safety_dist: dict = {}
    for p_ in points:
        safety_dist[p_.safety_status] = (
            safety_dist.get(p_.safety_status, 0) + 1)

    # Per-slack max-safe goodput/$ summary (the "is the slope real?" answer).
    best_per_slack: dict = {}
    for p_ in points:
        if p_.is_safe:
            s = p_.candidate.deadline_slack_seconds
            g = p_.predicted_goodput_per_dollar or 0.0
            if s not in best_per_slack or g > best_per_slack[s][0]:
                best_per_slack[s] = (g, p_.candidate.target_rho)
    max_safe_per_slack = {
        str(s): {"goodput_per_dollar": g, "rho_target": R}
        for s, (g, R) in best_per_slack.items()
    }

    payload = {
        "doc_version": "batch_inference_frontier_v1_summary",
        "production_claim": False,
        "ml_training": False,
        "modifies_serving_rho_controller": False,
        "uses_oracle_as_headline": False,
        "executable_in_real_cluster": False,
        "source": {
            "trace": "azure_llm_2024",
            "path": str(src_path),
            "scale_rps": args.scale_rps,
            "tick_seconds": args.tick_seconds,
            "request_count": len(reqs),
            "tick_count": len(ticks),
            "active_tick_count": len(active),
        },
        "workload_profile": profile.to_dict(),
        "candidate_grid": {
            "rho": list(rho_grid),
            "deadline_slack_seconds": list(slack_grid_s),
            "total_candidates": len(candidates),
        },
        "safety_status_distribution": safety_dist,
        "max_safe_goodput_per_slack": max_safe_per_slack,
        "frontier_points": [p_.to_dict() for p_ in points],
        "recommendation": decision.to_dict(),
        "honesty_notes": [
            "simulator / public-trace evidence only — NOT production savings",
            "Azure LLM 2024 is a SERVING trace, NOT a native batch trace; "
            "the deadline-slack scenario is SYNTHETIC and labelled "
            f"({profile.synthetic_scenario_label})",
            "no oracle / clairvoyant baseline used as headline",
            "no serving rho controller default changed",
            "executable_in_real_cluster is False at construction",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[batch] action: {decision.action}")
    if decision.selected_candidate is not None:
        sc = decision.selected_candidate
        sp = decision.selected_point
        print(f"[batch] selected: rho={sc.target_rho} "
              f"slack_s={sc.deadline_slack_seconds}")
        if sp is not None:
            print(f"[batch]   predicted goodput/$ = "
                  f"{sp.predicted_goodput_per_dollar:,.2f}")
            print(f"[batch]   predicted deadline_miss% = "
                  f"{sp.predicted_deadline_miss_rate_pct:.4f}")
            print(f"[batch]   predicted queue_p99_ms = "
                  f"{sp.predicted_queue_p99_ms:.2f}")
    print(f"[batch] summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
