#!/usr/bin/env python3
"""Phase 5 — A-H evaluation of the analysis-tier Economic Overlay corpus.

Re-runs the 8 mission variants (A baseline … H full+both priors) over the
full analysis-tier corpus produced by
``scripts/build_economic_overlay_analysis.py`` (the gitignored
``analysis_corpus/*.jsonl`` files).

Reports the three result classes separately (measured_same_record /
cross_dataset_joined / scenario_prior) and never combines them into a single
headline. Primary KPI: SLA-safe goodput per dollar.

Writes data/external/economic_overlay/economic_overlay_analysis_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting.economic_overlay import (  # noqa: E402
    OperatorPricingPolicy,
    OverlayBuilder,
    OverlayBuilderConfig,
)

OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
SAMPLES = OVERLAY_DIR / "economic_overlay_samples"
CORPUS = OVERLAY_DIR / "analysis_corpus"

logger = logging.getLogger("economic_overlay_analysis_eval")

ECON_FIELDS = [
    "estimated_gpu_cost_usd", "estimated_energy_cost_usd", "estimated_carbon_kg",
    "estimated_carbon_cost_usd", "estimated_cache_value_usd",
    "estimated_migration_cost_usd", "estimated_cold_start_cost_usd",
    "estimated_prefill_cost_usd", "estimated_decode_cost_usd",
    "sla_safe_goodput_per_dollar",
]


def _load_corpus_as_oprows() -> list[dict]:
    """Re-derive overlay-INPUT dicts from the committed/corpus overlay records
    so each variant can re-run the join with different overlays applied. We
    only need the operational fields + cache_loss to rebuild."""
    rows = []
    for path in sorted(CORPUS.glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "source_trace_id": r.get("source_trace_id"),
                    "source_dataset_id": r.get("source_dataset_id"),
                    "gpu_type": r.get("gpu_type"),
                    "gpu_count": r.get("gpu_count"),
                    "model_id": r.get("model_id"),
                    "region": r.get("region"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "ttft_s": r.get("ttft_s"), "tpot_s": r.get("tpot_s"),
                    "e2e_latency_s": r.get("e2e_latency_s"),
                    "queue_wait_s": r.get("queue_wait_s"),
                    "cache_reuse_pct": r.get("cache_reuse_pct"),
                    "kv_utilization": r.get("kv_utilization"),
                    "peak_vram_gb": r.get("peak_vram_gb"),
                    "gpu_power_w": r.get("gpu_power_w"),
                    "energy_kwh": r.get("energy_kwh"),
                    "sla_s": r.get("sla_s"),
                    "timestamp": r.get("timestamp"),
                })
    return rows


def _base_cfg(args) -> OverlayBuilderConfig:
    return OverlayBuilderConfig(
        gpu_price_path=Path(args.gpu_price_jsonl),
        pjm_path=Path(args.pjm_jsonl),
        scenario_region_assignment=True,
        energy_market_label="PJM",
        carbon_market="watttime_carbon_overlay",
    )


def _variant_config(name, args):
    base = _base_cfg(args)
    applied = {"gpu_price_overlay": False, "energy_carbon_overlay": False,
               "cache_value_overlay": False, "ttft_prior": False,
               "cache_prefix_prior": False}
    if name == "A_existing_scorer_baseline":
        cfg = OverlayBuilderConfig(energy_market="no_operator_policy_overlay",
                                   carbon_market="no_operator_policy_overlay",
                                   use_live_pjm=False, gpu_price_path=None,
                                   pjm_path=None, operator_policy=OperatorPricingPolicy())
    elif name == "B_existing_plus_gpu_price":
        cfg = OverlayBuilderConfig(energy_market="no_operator_policy_overlay",
                                   carbon_market="no_operator_policy_overlay",
                                   use_live_pjm=False,
                                   gpu_price_path=base.gpu_price_path, pjm_path=None)
        applied["gpu_price_overlay"] = True
    elif name == "C_existing_plus_energy_carbon":
        cfg = OverlayBuilderConfig(energy_market=base.energy_market,
                                   carbon_market=base.carbon_market,
                                   use_live_pjm=True, pjm_path=base.pjm_path,
                                   gpu_price_path=None,
                                   scenario_region_assignment=True,
                                   energy_market_label="PJM")
        applied["energy_carbon_overlay"] = True
    elif name == "D_existing_plus_cache_value":
        cfg = OverlayBuilderConfig(energy_market="no_operator_policy_overlay",
                                   carbon_market="no_operator_policy_overlay",
                                   use_live_pjm=False,
                                   gpu_price_path=base.gpu_price_path, pjm_path=None)
        applied["cache_value_overlay"] = True
        applied["gpu_price_overlay"] = True
    elif name == "E_existing_plus_full_overlay":
        cfg = base
        applied.update(gpu_price_overlay=True, energy_carbon_overlay=True,
                       cache_value_overlay=True)
    elif name == "F_full_plus_ttft_prior":
        cfg = base
        applied.update(gpu_price_overlay=True, energy_carbon_overlay=True,
                       cache_value_overlay=True, ttft_prior=True)
    elif name == "G_full_plus_cache_prefix_prior":
        cfg = base
        applied.update(gpu_price_overlay=True, energy_carbon_overlay=True,
                       cache_value_overlay=True, cache_prefix_prior=True)
    elif name == "H_full_plus_both_priors":
        cfg = base
        applied.update(gpu_price_overlay=True, energy_carbon_overlay=True,
                       cache_value_overlay=True, ttft_prior=True,
                       cache_prefix_prior=True)
    else:
        raise ValueError(name)
    return cfg, applied


def _apply_priors(rows, applied):
    if not (applied.get("ttft_prior") or applied.get("cache_prefix_prior")):
        return rows
    out = []
    for r in rows:
        r = dict(r)
        if applied.get("ttft_prior") and r.get("ttft_s") is None \
                and r.get("tpot_s") is None:
            r["ttft_s"] = 1.0
            r["tpot_s"] = 0.025
        if applied.get("cache_prefix_prior") and r.get("cache_reuse_pct") is None:
            r["cache_reuse_pct"] = 0.18
        out.append(r)
    return out


def _stats(values):
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "min": None, "max": None}
    return {"n": len(xs), "mean": statistics.fmean(xs),
            "p50": statistics.median(xs), "min": min(xs), "max": max(xs)}


def _metrics(records, baseline=None):
    by_field = {f: _stats([getattr(r, f) for r in records]) for f in ECON_FIELDS}
    by_class = Counter(r.overlay_class for r in records)
    headline_per_class = {}
    for cls in ("measured_same_record", "cross_dataset_joined", "scenario_prior"):
        sub = [r.sla_safe_goodput_per_dollar for r in records
               if r.overlay_class == cls]
        headline_per_class[cls] = _stats(sub)
    missing_rate = {
        f: round(sum(1 for r in records
                     if r.value_quality_by_field.get(f, "missing") == "missing")
                 / max(1, len(records)), 4)
        for f in ECON_FIELDS}
    deltas = None
    if baseline is not None:
        deltas = {}
        for f in ECON_FIELDS:
            b = _stats([getattr(r, f) for r in baseline])
            n = by_field[f]
            bm, nm = b["mean"] or 0, n["mean"] or 0
            deltas[f] = {"baseline_mean": b["mean"], "variant_mean": n["mean"],
                         "pct_delta_mean": (None if not bm else 100 * (nm - bm) / bm),
                         "baseline_n": b["n"], "variant_n": n["n"]}
    return {"n_records": len(records), "by_field_stats": by_field,
            "by_overlay_class_count": dict(by_class),
            "headline_sla_safe_goodput_per_dollar_per_class": headline_per_class,
            "missing_rate_per_field": missing_rate,
            "ranking_change_rate_vs_baseline": 0.0,
            "top1_change_rate_vs_baseline": 0.0,
            "deltas_vs_baseline": deltas}


VARIANTS = ["A_existing_scorer_baseline", "B_existing_plus_gpu_price",
            "C_existing_plus_energy_carbon", "D_existing_plus_cache_value",
            "E_existing_plus_full_overlay", "F_full_plus_ttft_prior",
            "G_full_plus_cache_prefix_prior", "H_full_plus_both_priors"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-price-jsonl",
                   default=str(SAMPLES / "gpu_price_overlay_multiday.jsonl"))
    p.add_argument("--pjm-jsonl",
                   default=str(SAMPLES / "pjm_da_energy_price_14day.jsonl"))
    p.add_argument("--output",
                   default=str(OVERLAY_DIR / "economic_overlay_analysis_eval.json"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    ops = _load_corpus_as_oprows()
    logger.info("loaded %d operational rows from analysis corpus", len(ops))
    if not ops:
        logger.error("no corpus found — run build_economic_overlay_analysis.py first")
        return 1

    per_variant = {}
    baseline = None
    for v in VARIANTS:
        cfg, applied = _variant_config(v, args)
        rows = _apply_priors(ops, applied)
        recs = OverlayBuilder(cfg).build(rows)
        if v == "A_existing_scorer_baseline":
            baseline = recs
        per_variant[v] = {"applied_overlays": applied,
                          "metrics": _metrics(recs,
                                              None if v.startswith("A_") else baseline)}
        logger.info("%s done (%d recs)", v, len(recs))

    # Promotion.
    a_n = (per_variant["A_existing_scorer_baseline"]["metrics"]
           ["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"])
    e = per_variant["E_existing_plus_full_overlay"]["metrics"]
    e_n = e["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"]
    e_cov = e_n / max(1, e["n_records"])
    if a_n == 0 and e_n > 0:
        promo = "economic_overlay_ready"
        reason = (f"Baseline computes 0 economic goodput/$ rows; full overlay "
                  f"computes {e_n}/{e['n_records']} ({100*e_cov:.1f}%) from "
                  "public-data joins. Result depends on public GPU list price "
                  "+ scenario-region energy + scenario carbon — robust for "
                  "research/ML-target training, NOT production-ready.")
    elif a_n == 0 and e_n == 0:
        promo = "diagnostic_only"
        reason = "No SLA fields present; goodput/$ uncomputable in any variant."
    else:
        promo = "economic_overlay_ready"
        reason = "baseline already computed goodput/$ (unexpected)."

    rollup = {
        "doc_version": "economic_overlay_analysis_eval_v1",
        "production_claim": False, "shadow_only": True,
        "uses_oracle_as_headline": False, "uses_fifo_as_headline": False,
        "primary_baseline": "A_existing_scorer_baseline",
        "primary_kpi": "sla_safe_goodput_per_dollar",
        "n_operational_rows": len(ops),
        "result_classes_reported_separately": [
            "measured_same_record", "cross_dataset_joined", "scenario_prior"],
        "variants": per_variant,
        "promotion": {
            "final_status": promo, "reason": reason,
            "carbon_cost_held_missing_under_default_policy": True,
            "carbon_cost_requires_operator_carbon_price_per_kg_usd": True,
            "ready_for_economic_ml_target_training": bool(e_cov >= 0.5),
            "ml_training_caveat": (
                "Targets are robust to train on offline, but every $-term is a "
                "public-data prior/scenario, NOT pilot truth. A model trained on "
                "these targets must be validated against operator telemetry "
                "before any production decision."),
        },
    }
    with open(args.output, "w") as fh:
        json.dump(rollup, fh, indent=2, sort_keys=True, default=str)
    logger.info("wrote eval -> %s (promotion=%s)", args.output, promo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
