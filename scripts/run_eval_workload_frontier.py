#!/usr/bin/env python3
"""Run the Eval Workload Frontier v1 sweep over a committed eval-trace.

Opt-in, shadow only, recommendation-only (no real cluster execution).
Reads the bounded ShareGPT processed sample (when present) — falls back
to the committed fixture (``tests/fixtures/sharegpt_aiperf_sample/``) so
the script always produces a deterministic artifact in CI.

The script writes a small summary JSON under
``data/external/frontier/eval_workload_frontier_v1_summary.json`` recording:
  - the workload profile (synthetic-scenario label included),
  - the candidate grid swept,
  - every frontier point's (KPI, deadline-miss, completion-hours, safety
    verdict, vetoes),
  - the controller's recommendation,
  - the categorical action distribution,
  - the do-not-claim flags (NOT production savings, NO oracle headline,
    no controller default change).
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
    eval_workload_estimator as ewe,
    eval_workload_models as ewm,
    eval_workload_safety as ews,
)
from aurelius.frontier.eval_workload_controller import (  # noqa: E402
    choose_eval_workload_frontier_target,
)
from aurelius.traces import sharegpt_aiperf  # noqa: E402
from aurelius.traces.eval_schema import EvalWorkloadRequest  # noqa: E402


DEFAULT_BOUNDED = (REPO_ROOT / "data" / "external" / "sharegpt_aiperf"
                   / "raw" / "sg_52k_head.json")
DEFAULT_FIXTURE = (REPO_ROOT / "tests" / "fixtures"
                   / "sharegpt_aiperf_sample" / "sg_head_fixture.json")
DEFAULT_OUTPUT = (REPO_ROOT / "data" / "external" / "frontier"
                  / "eval_workload_frontier_v1_summary.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run Eval Workload Frontier v1.")
    p.add_argument("--source", choices=("auto", "bounded", "fixture"),
                   default="auto")
    p.add_argument("--max-records", type=int, default=500)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args(argv)

    src_label: str
    src_path: Path
    if args.source == "fixture":
        src_path, src_label = DEFAULT_FIXTURE, "fixture"
    elif args.source == "bounded":
        if not DEFAULT_BOUNDED.exists():
            print(f"[eval] bounded sample not found: {DEFAULT_BOUNDED}",
                  file=sys.stderr)
            return 2
        src_path, src_label = DEFAULT_BOUNDED, "bounded"
    else:  # auto
        if DEFAULT_BOUNDED.exists():
            src_path, src_label = DEFAULT_BOUNDED, "bounded"
        else:
            src_path, src_label = DEFAULT_FIXTURE, "fixture"

    print(f"[eval] eval source: {src_label} ({src_path})")
    recs = sharegpt_aiperf.load_json_path(
        str(src_path), max_records=args.max_records)
    print(f"[eval] loaded {len(recs)} eval requests")

    profile = ewm.EvalWorkloadProfile(
        workload_id="sharegpt_eval_workload_v1",
        trace_source="sharegpt_aiperf",
        synthetic_scenario_label="sharegpt_eval_overnight_v1",
        dedicated_fleet=True,
        deadline_slack_hours_baseline=4.0,
        deadline_miss_rate_sla_pct=1.0,
        eval_suite_completion_deadline_hours=24.0,
        telemetry_confidence="low",
        source=src_label,
    )

    rho_grid = (0.55, 0.65, 0.75, 0.85, 0.95)
    concurrency_grid = (1, 2, 4, 8)
    slack_grid_h = (0.5, 1.0, 4.0, 24.0)

    candidates = [
        ewm.EvalWorkloadFrontierCandidate(
            target_rho=R, concurrency=C, deadline_slack_hours=sh,
            dedicated_fleet=True, source_policy=f"rho{R}_c{C}_h{sh}")
        for R in rho_grid for C in concurrency_grid for sh in slack_grid_h
    ]

    points = ewe.estimate_eval_workload_frontier(
        profile, recs, candidates,
        estimator_config=ewe.EvalWorkloadEstimatorConfig(),
        safety_config=ews.EvalWorkloadSafetyConfig(
            max_deadline_miss_rate_pct=1.0,
            max_eval_suite_completion_hours=24.0))

    decision = choose_eval_workload_frontier_target(profile, points)

    action_dist: dict = {}
    for p_ in points:
        action_dist[p_.safety_status] = (
            action_dist.get(p_.safety_status, 0) + 1)

    payload = {
        "doc_version": "eval_workload_frontier_v1_summary",
        "production_claim": False,
        "ml_training": False,
        "modifies_serving_rho_controller": False,
        "uses_oracle_as_headline": False,
        "executable_in_real_cluster": False,
        "source": {
            "trace": "sharegpt_aiperf",
            "trace_kind": src_label,
            "path": str(src_path),
            "record_count": len(recs),
        },
        "workload_profile": profile.to_dict(),
        "candidate_grid": {
            "rho": list(rho_grid),
            "concurrency": list(concurrency_grid),
            "deadline_slack_hours": list(slack_grid_h),
            "total_candidates": len(candidates),
        },
        "safety_status_distribution": action_dist,
        "frontier_points": [p_.to_dict() for p_ in points],
        "recommendation": decision.to_dict(),
        "honesty_notes": [
            "simulator / public-trace evidence only — NOT production savings",
            "synthetic deadline-slack scenario; ShareGPT carries no real "
            "deadlines",
            "token counts are char/4 PROXY, not real tokenizer output",
            "no oracle / clairvoyant baseline used as headline",
            "no serving rho controller default changed",
            "executable_in_real_cluster is False at construction",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[eval] action: {decision.action}")
    if decision.selected_candidate is not None:
        sc = decision.selected_candidate
        sp = decision.selected_point
        print(f"[eval] selected: rho={sc.target_rho} conc={sc.concurrency} "
              f"slack_h={sc.deadline_slack_hours} "
              f"dedicated_fleet={sc.dedicated_fleet}")
        if sp is not None:
            print(f"[eval]   predicted goodput/$ = "
                  f"{sp.predicted_goodput_per_dollar:,.2f}")
            print(f"[eval]   predicted completion_h = "
                  f"{sp.predicted_eval_suite_completion_hours:.4f}")
            print(f"[eval]   predicted deadline_miss% = "
                  f"{sp.predicted_deadline_miss_rate_pct:.4f}")
    print(f"[eval] summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
