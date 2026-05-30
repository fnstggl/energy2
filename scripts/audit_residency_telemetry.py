#!/usr/bin/env python3
"""Read-only model-residency / cold-start telemetry audit + shadow report.

Ingests residency telemetry (events required; snapshots + per-request
observations optional), classifies cross-layer linkage without faking joins,
computes the honest §3 derived metrics, runs the **recommendation-only** shadow
recommender, and emits a JSON summary + a markdown report.

This is a **read-only** instrument: it never mutates a cluster, never loads or
evicts a real model, and never calls a Kubernetes write API. Every shadow
decision is logged with ``executed=False``.

Directional / pilot-telemetry diagnostics only — **not production savings**
(``docs/RESULTS.md`` §8). No production-savings number may be quoted until that
gate is met.

Examples
--------
    python scripts/audit_residency_telemetry.py \
        --events tests/fixtures/residency/events.jsonl \
        --snapshots tests/fixtures/residency/snapshots.jsonl \
        --requests tests/fixtures/residency/requests.jsonl \
        --output docs/RESIDENCY_TELEMETRY_AUDIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.residency import ingest, linkage, metrics, shadow  # noqa: E402

READINESS_LADDER = [
    "SPEC_ONLY", "SIMULATOR_APPROXIMATION", "TRACE_BACKTESTED_APPROXIMATION",
    "SHADOW_PILOT_READY_READ_ONLY", "PRODUCTION_READY",
]

# Fields whose absence blocks promotion past read-only pilot (contract §7).
# (event.duration_s is intentionally NOT required — only *_end events carry it.)
_PILOT_REQUIRED = {
    "observation": ("model_loaded_before_request", "container_id", "gpu_id"),
    "snapshot": ("gpu_id", "loaded_model_ids", "gpu_memory_used", "gpu_memory_total"),
    "event": ("request_id",),
}


def _mv(metric_dict, key):
    """Serialize a MetricValue from a metrics dict."""
    mv = metric_dict.get(key)
    return mv.to_dict() if mv is not None else None


def build_summary(events_path, *, snapshots_path=None, requests_path=None,
                  slo_s=None, gpu_hour_cost=None) -> dict:
    """Ingest + analyze residency telemetry into a JSON-serialisable summary."""
    ev = ingest.import_events(events_path)
    snaps = (ingest.import_snapshots(snapshots_path) if snapshots_path
             else ingest.IngestResult(record_type="snapshot"))
    reqs = (ingest.import_observations(requests_path) if requests_path
            else ingest.IngestResult(record_type="observation"))

    observations = reqs.records
    snapshots = snaps.records
    eventsl = ev.records

    # --- Linkage (no fake joins) ---
    link_obs_snap = linkage.build_linkage_report(observations, snapshots)
    link_obs_evt = linkage.build_linkage_report(observations, eventsl)
    # The attribution gate uses the request↔infra (snapshot) linkage.
    gate_quality = link_obs_snap.quality

    # --- Derived metrics (diagnostics only) ---
    mx = metrics.compute_all_metrics(
        observations, eventsl, snapshots,
        slo_s=slo_s, gpu_hour_cost=gpu_hour_cost, linkage_quality=gate_quality)
    miss = metrics.missingness(observations, eventsl, snapshots)

    # --- Shadow recommendations (recommendation-only) ---
    cfg = shadow.ShadowRecommenderConfig(slo_s=slo_s, gpu_hour_cost=gpu_hour_cost)
    log = shadow.ResidencyShadowLog()
    decisions = shadow.recommend_all(
        observations, config=cfg, linkage_quality=gate_quality, log=log)

    # --- Missing fields preventing pilot readiness ---
    missing_for_pilot = _missing_for_pilot(reqs, snaps, ev)

    summary = {
        "kpi_note": "residency metrics are diagnostics; NEVER folded into the "
                    "canonical KPI (docs/RESULTS.md §1-§2)",
        "directional_only_not_production_savings": True,
        "inputs": {
            "events_path": events_path,
            "snapshots_path": snapshots_path,
            "requests_path": requests_path,
            "slo_s": slo_s,
            "gpu_hour_cost": gpu_hour_cost,
        },
        "ingestion": {
            "events": ev.to_dict(),
            "snapshots": snaps.to_dict(),
            "observations": reqs.to_dict(),
        },
        "linkage": {
            "observations_to_snapshots": link_obs_snap.to_dict(),
            "observations_to_events": link_obs_evt.to_dict(),
            "attribution_gate_quality": gate_quality,
            "attributable": link_obs_snap.attributable,
        },
        "metrics": {k: _mv(mx, k) for k in mx},
        "missingness": miss,
        "shadow": {
            "posture": "recommendation_only",
            "cluster_mutation": False,
            "decisions": [d.to_dict() for d in decisions],
            "summary": log.summary(),
        },
        "missing_fields_preventing_pilot_readiness": missing_for_pilot,
        "readiness_ladder": READINESS_LADDER,
        "readiness_verdict": _verdict(reqs, snaps, ev, link_obs_snap, missing_for_pilot),
    }
    return summary


def _missing_for_pilot(reqs, snaps, ev) -> list:
    """List required fields that are wholly absent / sparsely present."""
    out = []
    streams = {"observation": reqs, "snapshot": snaps, "event": ev}
    for stream, result in streams.items():
        if not result.records:
            out.append(f"{stream}: stream not provided")
            continue
        cov = result.field_coverage
        for f in _PILOT_REQUIRED[stream]:
            c = cov.get(f, {}).get("coverage")
            if c is None:
                out.append(f"{stream}.{f}: field absent")
            elif c < 1.0:
                out.append(f"{stream}.{f}: present in only "
                           f"{round(c * 100)}% of records")
    return out


def _verdict(reqs, snaps, ev, link_obs_snap, missing_for_pilot) -> str:
    """Conservative verdict for the *telemetry substrate*.

    The substrate is SHADOW_PILOT_READY_READ_ONLY when it can ingest events +
    per-request residency, classify an attributable linkage, and log
    recommendation-only decisions. It is NOT production-ready and NOT
    autonomous-optimization-ready regardless of this value.
    """
    has_events = bool(ev.records)
    has_obs = bool(reqs.records)
    has_residency_flag = any(
        o.model_loaded_before_request is not None for o in reqs.records)
    if has_events and has_obs and has_residency_flag and link_obs_snap.attributable:
        return "SHADOW_PILOT_READY_READ_ONLY"
    if has_events or has_obs:
        return "TRACE_BACKTESTED_APPROXIMATION"
    return "SPEC_ONLY"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{round(v, nd)}"
    return str(v)


def render_markdown(summary: dict) -> str:
    L = []
    a = L.append
    a("# Model-Residency / Cold-Start Telemetry Audit\n")
    a("> **Read-only telemetry audit + shadow report.** Generated by "
      "`scripts/audit_residency_telemetry.py`. This instrument ingests "
      "residency telemetry, classifies cross-layer linkage, computes honest "
      "diagnostics, and logs **recommendation-only** shadow decisions. It "
      "**mutates no cluster**, loads/evicts **no real model**, and calls **no "
      "Kubernetes write API**.\n")
    a("> **Directional / pilot-telemetry diagnostics only — not production "
      "savings** (`docs/RESULTS.md` §8). Residency metrics are diagnostics and "
      "are **never** folded into the canonical KPI.\n")

    inp = summary["inputs"]
    a(f"\n**Verdict (telemetry substrate):** `{summary['readiness_verdict']}`  ")
    a(f"\n**Attribution linkage (request↔infra):** "
      f"`{summary['linkage']['attribution_gate_quality']}` "
      f"(attributable: {summary['linkage']['attributable']})\n")

    # 1. Telemetry coverage
    a("\n## 1. Telemetry coverage\n")
    a("| stream | rows | valid | errors | sources |")
    a("|---|---|---|---|---|")
    for stream in ("events", "snapshots", "observations"):
        s = summary["ingestion"][stream]
        srcs = ", ".join(f"{k}={v}" for k, v in s["sources"].items()) or "—"
        a(f"| {stream} | {s['n_rows']} | {s['n_valid']} | {s['n_errors']} | {srcs} |")

    a("\n### Per-field coverage (key fields)\n")
    for stream in ("observations", "snapshots", "events"):
        cov = summary["ingestion"][stream]["field_coverage"]
        keys = [f for f, c in cov.items() if c.get("is_key_field")]
        if not keys:
            continue
        a(f"\n**{stream}**\n")
        a("| field | present / total | coverage |")
        a("|---|---|---|")
        for f in keys:
            c = cov[f]
            a(f"| `{f}` | {c['present']} / {c['total']} | {round(c['coverage']*100)}% |")

    # 2. Linkage quality
    a("\n## 2. Linkage quality (no fake joins)\n")
    a("| pair | quality | attributable | joined / total | per-quality |")
    a("|---|---|---|---|---|")
    for label, key in (("obs → snapshots", "observations_to_snapshots"),
                       ("obs → events", "observations_to_events")):
        lk = summary["linkage"][key]
        pq = ", ".join(f"{k}={v}" for k, v in lk["per_quality"].items() if v)
        a(f"| {label} | `{lk['quality']}` | {lk['attributable']} | "
          f"{lk['n_joined']} / {lk['n_left']} | {pq or '—'} |")
    notes = summary["linkage"]["observations_to_snapshots"].get("notes", [])
    for n in notes:
        a(f"- {n}")

    # 3. Hit / miss + cold start
    a("\n## 3. Residency hit/miss + cold-start rates\n")
    a("| metric | value | numerator | denominator | note |")
    a("|---|---|---|---|---|")
    for key in ("model_residency_hit_rate", "adapter_residency_hit_rate",
                "cold_start_rate"):
        m = summary["metrics"].get(key) or {}
        a(f"| {key} | {_fmt(m.get('value'))} | {_fmt(m.get('numerator'))} | "
          f"{_fmt(m.get('denominator'))} | {m.get('note') or ''} |")

    # 4. Cold-start latency distribution
    a("\n## 4. Cold-start latency distribution (measured, cold requests only)\n")
    a("| metric | p50 | p95 | p99 | n |")
    a("|---|---|---|---|---|")
    for base in ("model_load_latency", "adapter_load_latency"):
        p50 = summary["metrics"].get(f"{base}_p50") or {}
        p95 = summary["metrics"].get(f"{base}_p95") or {}
        p99 = summary["metrics"].get(f"{base}_p99") or {}
        a(f"| {base} | {_fmt(p50.get('value'))} | {_fmt(p95.get('value'))} | "
          f"{_fmt(p99.get('value'))} | {_fmt(p50.get('denominator'))} |")
    csv_m = summary["metrics"].get("cold_start_attributed_sla_violations") or {}
    a(f"\n- **cold-start-attributed SLA violations:** {_fmt(csv_m.get('value'))} "
      f"(of {_fmt(csv_m.get('denominator'))} violations; note: "
      f"{csv_m.get('note') or 'OK'})")

    # 5. Warm-pool cost
    a("\n## 5. Warm-pool cost (if measurable)\n")
    wp = summary["metrics"].get("warm_pool_gpu_hours") or {}
    extra = wp.get("extra", {}) if isinstance(wp, dict) else {}
    a(f"- **warm-pool GPU-hours:** {_fmt(wp.get('value'), 6)} "
      f"(note: {wp.get('note') or 'measured from snapshots'})")
    if extra.get("warm_pool_cost") is not None:
        a(f"- **warm-pool cost:** {_fmt(extra.get('warm_pool_cost'))} "
          f"(at gpu_hour_cost={extra.get('gpu_hour_cost')})")
    if extra:
        a(f"- intervals={extra.get('resident_intervals')}, "
          f"dropped_stale={extra.get('dropped_stale_intervals')}, "
          f"gpus={extra.get('gpus')}")
    churn = summary["metrics"].get("residency_churn_score") or {}
    a(f"- **residency churn score:** {_fmt(churn.get('value'), 6)} "
      f"(load+evict per model per hour)")
    conf = summary["metrics"].get("telemetry_confidence") or {}
    a(f"- **telemetry confidence:** {_fmt(conf.get('value'))} "
      f"(blended; 1.0 = fully attributable + high provenance)")

    # 6. Shadow recommendations
    a("\n## 6. Shadow recommendations (recommendation-only — no mutation)\n")
    sh = summary["shadow"]["summary"]
    a(f"- posture: `{summary['shadow']['posture']}`, "
      f"cluster_mutation: {summary['shadow']['cluster_mutation']}, "
      f"all_recommendation_only: {sh.get('all_recommendation_only')}")
    a("\n| action | count |")
    a("|---|---|")
    for action, count in sh["action_counts"].items():
        a(f"| {action} | {count} |")
    a(f"\n- total expected cold-start saved (directional): "
      f"{_fmt(sh.get('total_expected_cold_start_saved_s'))}s")
    a("\n**Decision log (counterfactual: `executed=False` for all):**\n")
    a("| request_id | model | action | expected_saved_s | reason |")
    a("|---|---|---|---|---|")
    for d in summary["shadow"]["decisions"]:
        a(f"| {d.get('request_id') or '—'} | {d['model_id']} | "
          f"`{d['recommended_action']}` | {_fmt(d.get('expected_cold_start_saved_s'))} | "
          f"{d['reason'][:60]} |")

    # 7. Missing fields preventing pilot readiness
    a("\n## 7. Missing fields preventing pilot readiness\n")
    mfp = summary["missing_fields_preventing_pilot_readiness"]
    if not mfp:
        a("- none detected in this telemetry sample (substrate-conformant)")
    else:
        for m in mfp:
            a(f"- {m}")

    a("\n## 8. Claim discipline\n")
    a("- All numbers above are **directional pilot-telemetry diagnostics — not "
      "production savings**. The shadow log is **recommendation-only**; no "
      "cluster was mutated and no model was loaded/evicted.")
    a("- Per `docs/PILOT_TELEMETRY_CONTRACT.md` §4, residency metrics are "
      "attributable only at `exact_join` / `container_join`; weaker linkage is "
      "calibration-only.")
    a("- The `docs/RESULTS.md` §8 production-claim gate is **not** met; this "
      "audit does not promote any recommendation out of shadow mode.\n")
    return "\n".join(L) + "\n"


def _strip_ext(path: str) -> str:
    for ext in (".json", ".md", ".markdown"):
        if path.lower().endswith(ext):
            return path[: -len(ext)]
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="residency events JSONL/CSV/JSON")
    p.add_argument("--snapshots", default=None, help="residency snapshots (optional)")
    p.add_argument("--requests", default=None,
                   help="per-request residency observations (optional)")
    p.add_argument("--output", required=True,
                   help="output base path; writes <base>.json and <base>.md")
    p.add_argument("--slo-s", type=float, default=None,
                   help="SLO seconds for cold-start SLA attribution (optional)")
    p.add_argument("--gpu-hour-cost", type=float, default=None,
                   help="GPU-hour cost for warm-pool dollar metering (optional)")
    args = p.parse_args(argv)

    summary = build_summary(
        args.events, snapshots_path=args.snapshots, requests_path=args.requests,
        slo_s=args.slo_s, gpu_hour_cost=args.gpu_hour_cost)

    base = _strip_ext(args.output)
    out_dir = os.path.dirname(os.path.abspath(base))
    os.makedirs(out_dir, exist_ok=True)
    json_path = base + ".json"
    md_path = base + ".md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary))

    print(f"[residency-audit] verdict (substrate): {summary['readiness_verdict']}")
    print(f"[residency-audit] linkage (obs→infra): "
          f"{summary['linkage']['attribution_gate_quality']} "
          f"(attributable={summary['linkage']['attributable']})")
    mx = summary["metrics"]
    print(f"[residency-audit] model hit rate: "
          f"{_fmt((mx.get('model_residency_hit_rate') or {}).get('value'))}, "
          f"cold-start rate: "
          f"{_fmt((mx.get('cold_start_rate') or {}).get('value'))}")
    print(f"[residency-audit] shadow decisions: {summary['shadow']['summary']}")
    print(f"[residency-audit] JSON -> {json_path}")
    print(f"[residency-audit] MD   -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
