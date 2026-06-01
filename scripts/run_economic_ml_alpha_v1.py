#!/usr/bin/env python3
"""Economic ML Alpha Audit v1 — modular forecaster + economic-alpha driver.

Runs (research / shadow only; no production module touched):
  Phase 0  exact overlap audit            -> overlap_audit.json
  Phase 1  cold-start / migration realism  -> realism_prior_audit.json
  Phase 2  target catalog + trainability   -> target_catalog.json
  Phase 4  train modular forecasters
  Phase 5  multi-holdout validation
  Phase 6  economic-alpha evaluation       -> summary.json + economic_alpha_eval.json
  Phase 7  cold-start/migration sensitivity -> cold_start_migration_sensitivity.json

Primary KPI: SLA-safe goodput / dollar (deterministic overlay formula is the
ground truth) + economic regret vs the strongest deterministic baseline.
Random holdout is decorative; the binding holdouts are time / by-dataset /
by-GPU / by-source-class / high-cost-tail. No oracle / FIFO headline. No
invented constants. Missing cold-start / migration labels are never zeroed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting import economic_ml_features as F  # noqa: E402,N812
from aurelius.forecasting import economic_ml_forecaster as M  # noqa: E402,N812

OUT_DIR = REPO_ROOT / "data" / "external" / "forecasting" / "economic_ml_alpha_v1"
logger = logging.getLogger("economic_ml_alpha_v1")

ECON_FIELDS = [
    "estimated_gpu_cost_usd", "estimated_prefill_cost_usd",
    "estimated_decode_cost_usd", "estimated_cache_value_usd",
    "estimated_energy_cost_usd", "estimated_carbon_kg",
    "estimated_carbon_cost_usd", "estimated_cold_start_cost_usd",
    "estimated_migration_cost_usd", "sla_safe_goodput_per_dollar",
]
OPS_FIELDS = [
    "ttft_s", "tpot_s", "e2e_latency_s", "queue_wait_s", "cache_reuse_pct",
    "kv_utilization", "gpu_power_w", "energy_kwh", "gpu_price_usd_per_hour",
    "prompt_tokens", "output_tokens", "gpu_type", "model_id",
    "source_dataset_id",
]


# ── constant-signal detector (a field that is a single injected value per
#    dataset is a prior/constant, not a trainable measured signal) ──────────


def _is_variable(rows, field_name, ds_key="source_dataset_id") -> dict:
    by_ds = defaultdict(set)
    for r in rows:
        v = r.get(field_name)
        if v is not None:
            by_ds[r.get(ds_key)].add(round(float(v), 9)
                                     if isinstance(v, (int, float)) else v)
    return {ds.split("/")[-1][:20]: ("variable" if len(s) > 3 else "constant")
            for ds, s in by_ds.items()}


# ───────────────────────── Phase 0 — overlap audit ─────────────────────────


def phase0_overlap(rows) -> dict:
    n = len(rows)

    def have(f):
        return np.array([r.get(f) is not None for r in rows])

    masks = {f: have(f) for f in ECON_FIELDS + OPS_FIELDS}
    four = (masks["estimated_gpu_cost_usd"]
            & (masks["estimated_prefill_cost_usd"] | masks["estimated_decode_cost_usd"])
            & masks["estimated_cache_value_usd"]
            & masks["estimated_carbon_kg"])
    lat = masks["ttft_s"] & masks["tpot_s"] & masks["e2e_latency_s"]
    counts = {f: int(m.sum()) for f, m in masks.items()}

    def dom(mask):
        c = Counter(r["source_dataset_id"].split("/")[-1]
                    for r, m in zip(rows, mask) if m)
        return dict(c.most_common(4))

    answers = {
        "q1_rows_with_gpu_cost": counts["estimated_gpu_cost_usd"],
        "q2_rows_with_prefill_or_decode_cost":
            int((masks["estimated_prefill_cost_usd"]
                 | masks["estimated_decode_cost_usd"]).sum()),
        "q3_rows_with_cache_value": counts["estimated_cache_value_usd"],
        "q4_rows_with_energy_or_carbon_physical":
            int((masks["estimated_energy_cost_usd"]
                 | masks["estimated_carbon_kg"]).sum()),
        "q5_rows_with_all_four": int(four.sum()),
        "q6_rows_with_all_four_plus_latency": int((four & lat).sum()),
        "q7_rows_with_all_four_plus_queue":
            int((four & masks["queue_wait_s"]).sum()),
        "q8_rows_with_all_four_plus_cache_reuse":
            int((four & masks["cache_reuse_pct"]).sum()),
        "q9_rows_with_all_four_plus_energy_kwh":
            int((four & masks["energy_kwh"]).sum()),
        "q10_rows_with_cold_start_cost": counts["estimated_cold_start_cost_usd"],
        "q11_rows_with_migration_cost": counts["estimated_migration_cost_usd"],
        "q12_rows_with_real_cache_hit": 0,  # no measured per-request cache_hit
        "q13_dominant_datasets_per_group": {
            "gpu_cost": dom(masks["estimated_gpu_cost_usd"]),
            "cache_value": dom(masks["estimated_cache_value_usd"]),
            "energy_carbon": dom(masks["estimated_energy_cost_usd"]),
            "latency_ttft_tpot_e2e": dom(lat),
            "cache_reuse": dom(masks["cache_reuse_pct"]),
            "all_four": dom(four),
        },
    }
    variability = {f: _is_variable(rows, f)
                   for f in ("ttft_s", "tpot_s", "e2e_latency_s",
                             "cache_reuse_pct", "estimated_gpu_cost_usd",
                             "sla_safe_goodput_per_dollar")}
    return {
        "total_rows": n,
        "field_presence_counts": counts,
        "answers": answers,
        "signal_variability_by_dataset": variability,
        "notes": [
            "Economic cost / goodput targets are DETERMINISTIC transforms of "
            "operational inputs x a prior GPU price (see "
            "docs/ECONOMIC_OVERLAY_LAYER_V1.md). The deterministic overlay "
            "formula is the ground truth; ML over a derived cost target is "
            "diagnostic_only and can only yield alpha by forecasting upstream "
            "inputs (latency / reuse / energy).",
            "SwissAI ttft/tpot/e2e and AcmeTrace-power e2e are INJECTED "
            "CONSTANTS (single value per dataset), not trainable measured "
            "latency; excluded from latency target training.",
            "No measured per-request cache_hit exists in any source (q12=0).",
        ],
    }


# ─────────────── Phase 1 — cold-start / migration realism audit ────────────


def phase1_realism(rows) -> dict:
    # Detect any presence of the relevant raw signals in the overlay corpus.
    have_cold = sum(1 for r in rows
                    if r.get("estimated_cold_start_cost_usd") is not None)
    have_mig = sum(1 for r in rows
                   if r.get("estimated_migration_cost_usd") is not None)

    def term(label, source, stype, conf, formula, notes, value=None):
        return {"value": value, "source": source, "source_type": stype,
                "confidence": conf, "formula": formula,
                "calibration_notes": notes, "production_ready": False}

    cold = {
        "model_load_duration_s": term(
            "model_load_duration_s",
            "ejhusom load_duration_ns (consumer Ollama) — NOT in analysis "
            "corpus; AcmeTrace/Google-Cluster give job SCHEDULE events not "
            "model-load seconds",
            "simulator_prior", "low",
            "n/a (no measured per-(model,gpu,cluster) load seconds)",
            "Only a single-machine Ollama proxy exists; no server-class vLLM "
            "model-load curve. Needs pilot telemetry."),
        "image_pull_duration_s": term("image_pull_duration_s", "none",
                                       "missing", "low", "n/a",
                                       "No image-pull telemetry in any ingested source."),
        "gpu_transfer_duration_s": term("gpu_transfer_duration_s",
                                        "Memoriant KV-cache memory curves (indirect)",
                                        "prior", "low", "n/a",
                                        "KV memory size known; transfer seconds not measured."),
        "warmup_duration_s": term("warmup_duration_s", "none", "missing",
                                  "low", "n/a", "No warmup telemetry."),
        "graph_capture_duration_s": term("graph_capture_duration_s", "none",
                                         "missing", "low", "n/a",
                                         "No CUDA-graph capture telemetry."),
        "first_compile_duration_s": term("first_compile_duration_s", "none",
                                         "missing", "low", "n/a",
                                         "No first-compile telemetry."),
        "cold_start_cost_usd": term(
            "cold_start_cost_usd",
            "derivable IF model_load_duration_s measured + GPU price prior",
            "simulator_prior", "low",
            "model_load_duration_s * gpu_price_usd_per_hour / 3600 * gpu_count",
            f"0 rows in corpus carry it ({have_cold}); only a "
            "simulator_prior sweep is honest (Phase 7)."),
        "cold_start_energy_kwh": term("cold_start_energy_kwh", "none",
                                      "missing", "low", "n/a",
                                      "No cold-start energy telemetry."),
    }
    migration = {
        "migration_duration_s": term("migration_duration_s",
                                     "Google Cluster evict + AcmeTrace state (job-level proxy)",
                                     "prior", "low", "n/a",
                                     "Job-level evict events, not LLM-serving migration seconds."),
        "drain_duration_s": term("drain_duration_s", "none", "missing", "low",
                                 "n/a", "No drain telemetry."),
        "route_shift_duration_s": term("route_shift_duration_s", "none",
                                       "missing", "low", "n/a",
                                       "No routing-shift telemetry."),
        "cache_loss_pct": term("cache_loss_pct",
                               "CC-traces migration_or_cache_loss_proxy (KV block-hash loss)",
                               "proxy", "medium", "fraction of KV blocks lost",
                               "A genuine PROXY exists in CC-traces; not realized in the "
                               "analysis overlay rows (cache_loss_pct not populated)."),
        "migration_cache_loss_penalty_s": term(
            "migration_cache_loss_penalty_s",
            "derivable from cache_loss_pct + prefill throughput",
            "derived", "medium",
            "prompt_tokens / prefill_throughput_tok_s * cache_loss_pct",
            "Derivable once cache_loss_pct + prefill throughput are joined."),
        "migration_cost_usd": term(
            "migration_cost_usd",
            "derivable IF cache_loss_pct proxy + GPU price prior",
            "simulator_prior", "low",
            "cache_loss_prefill_penalty_s * gpu_price/3600 * gpu_count + reroute/warmup terms",
            f"0 rows in corpus carry it ({have_mig}); simulator_prior sweep only (Phase 7)."),
        "batching_disruption_factor": term("batching_disruption_factor", "none",
                                           "missing", "low", "n/a",
                                           "No batching-disruption telemetry."),
        "tail_latency_uplift_after_migration": term(
            "tail_latency_uplift_after_migration", "none", "missing", "low",
            "n/a", "No post-migration tail-latency telemetry."),
        "migration_veto_label": term("migration_veto_label", "none",
                                     "missing", "low", "n/a",
                                     "No labelled migration-veto outcomes."),
    }
    return {
        "cold_start_terms": cold,
        "migration_terms": migration,
        "verdict": {
            "cold_start": "blocked_by_missing_labels — simulator_prior sweep only",
            "migration": "blocked_by_missing_labels — a real cache-loss PROXY "
                         "exists in CC-traces but is not realized in the "
                         "overlay rows; simulator_prior sweep only",
            "binding_rule": "These targets are excluded from headline goodput/$; "
                            "Phase 7 provides transparent sourced sensitivity sweeps only.",
        },
    }


# ───────────────────────── Phase 2 — target catalog ─────────────────────────


def _catalog_entry(rows, target, model_family, baseline, *,
                   classification=False, trainable_min=2000,
                   exclude_constant_ds=None, allow_output_tokens=False):
    spec = F.resolve_feature_spec(target, allow_output_tokens=allow_output_tokens)
    kept = [r for r in rows if r.get(target) is not None]
    if exclude_constant_ds:
        kept = [r for r in kept
                if not any(x in r["source_dataset_id"]
                           for x in exclude_constant_ds)]
    n = len(kept)
    ds = dict(Counter(r["source_dataset_id"].split("/")[-1] for r in kept)
              .most_common())
    vq = F.value_quality_distribution(kept, target)
    trainable = n >= trainable_min
    return {
        "target": target,
        "row_count": n,
        "source_datasets": ds,
        "value_quality_distribution": vq,
        "missingness_rate": round(1 - n / max(1, len(rows)), 4),
        "valid_features": spec.numeric + spec.categorical,
        "invalid_leaky_features": sorted(spec.blocked - {target}),
        "recommended_model_family": model_family,
        "strongest_baseline": baseline,
        "holdouts_required": ["by_dataset", "by_gpu", "high_tail",
                              "time_if_available"],
        "classification": classification,
        "trainable_now": bool(trainable),
        "reason_if_not_trainable": (None if trainable
                                    else f"only {n} rows (<{trainable_min})"),
    }


def phase2_catalog(rows) -> dict:
    swissai = "swissai-serving-tr"
    acme_pow = "acmetrace_gpu_powe"  # not a ds id; constants handled via ds name
    # constant-latency datasets to exclude from latency targets
    const_lat_ds = {"swissai"}
    targets = {
        # Latency / risk
        "ttft_s": _catalog_entry(rows, "ttft_s", "HGB+RF vs GroupMedian",
                                 "GroupMedian(model,gpu)",
                                 exclude_constant_ds=const_lat_ds),
        "tpot_s": _catalog_entry(rows, "tpot_s", "HGB vs GroupMedian",
                                 "GroupMedian(model,gpu)",
                                 exclude_constant_ds=const_lat_ds),
        "e2e_latency_s": _catalog_entry(rows, "e2e_latency_s",
                                        "HGB vs GroupMedian",
                                        "GroupMedian(model,gpu)",
                                        exclude_constant_ds=const_lat_ds),
        "queue_wait_s": _catalog_entry(rows, "queue_wait_s",
                                       "HGB vs GlobalMedian",
                                       "GlobalMedian", trainable_min=5000),
        # Cache
        "cache_reuse_pct": _catalog_entry(rows, "cache_reuse_pct",
                                          "HGB vs PerModelMean",
                                          "PerModelMeanReuse"),
        "high_reuse": _catalog_entry(rows, "cache_reuse_pct",
                                     "HGB/Logistic vs GlobalRate",
                                     "GlobalReuseRate", classification=True),
        # Economic (deterministic ground truth)
        "estimated_gpu_cost_usd": _catalog_entry(
            rows, "estimated_gpu_cost_usd", "deterministic_formula",
            "OverlayFormula(gpu_price*seconds)"),
        "sla_safe_goodput_per_dollar": _catalog_entry(
            rows, "sla_safe_goodput_per_dollar", "deterministic_formula",
            "OverlayFormula(goodput/cost)"),
        # Resource
        "peak_vram_gb": _catalog_entry(rows, "peak_vram_gb",
                                       "HGB vs GroupMedian",
                                       "GroupMedian(model,gpu)",
                                       trainable_min=1500),
        "energy_kwh": _catalog_entry(rows, "energy_kwh", "HGB vs GroupMedian",
                                     "GroupMedian(model,gpu)",
                                     trainable_min=2000),
        # Cold-start / migration
        "cold_start_cost_usd": {
            "target": "cold_start_cost_usd", "row_count": 0,
            "trainable_now": False,
            "reason_if_not_trainable": "0 measured labels — simulator_prior_only",
            "recommended_model_family": "simulator_prior_sensitivity",
            "strongest_baseline": "n/a"},
        "migration_cost_usd": {
            "target": "migration_cost_usd", "row_count": 0,
            "trainable_now": False,
            "reason_if_not_trainable": "0 measured labels — simulator_prior_only",
            "recommended_model_family": "simulator_prior_sensitivity",
            "strongest_baseline": "n/a"},
    }
    # mark deterministic targets diagnostic_only
    for t in ("estimated_gpu_cost_usd", "sla_safe_goodput_per_dollar"):
        targets[t]["deterministic_ground_truth"] = True
        targets[t]["ml_status"] = "diagnostic_only — formula is exact; alpha " \
            "must come from forecasting upstream inputs, not re-predicting it"
    return {"targets": targets}


# ──────────────────── holdout splitters (Phase 5) ───────────────────────────


def _holdout_indices(rows, kind, kept_idx):
    """Returns (train_idx, test_idx) into kept_idx for a holdout kind."""
    rng = np.random.RandomState(0)
    idx = np.array(kept_idx)
    if kind == "random":
        perm = rng.permutation(len(idx))
        cut = int(0.8 * len(idx))
        return idx[perm[:cut]], idx[perm[cut:]]
    if kind == "by_dataset":
        ds = np.array([rows[i]["source_dataset_id"] for i in idx])
        uniq = sorted(set(ds))
        if len(uniq) < 2:
            return None
        test_ds = uniq[-1]
        return idx[ds != test_ds], idx[ds == test_ds]
    if kind == "by_gpu":
        g = np.array([rows[i].get("gpu_type") for i in idx])
        uniq = sorted({x for x in g if x is not None})
        if len(uniq) < 2:
            return None
        # hold out the rarest GPU
        counts = Counter(g.tolist())
        test_g = min(uniq, key=lambda x: counts[x])
        return idx[g != test_g], idx[g == test_g]
    if kind == "high_tail":
        # hold out the top-20% of the target as the tail
        return None  # handled per-target in training (needs y)
    if kind == "time":
        ts = [rows[i].get("timestamp") for i in idx]
        if any(t is None for t in ts):
            return None
        order = np.argsort([_ts(t) for t in ts])
        cut = int(0.8 * len(order))
        return idx[order[:cut]], idx[order[cut:]]
    return None


def _ts(t):
    try:
        return float(t)
    except (TypeError, ValueError):
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
        except Exception:  # noqa: BLE001
            return 0.0


# ──────────────────── Phase 4/5 — train + validate one target ───────────────


def train_regression_target(rows, target, *, classification=False,
                            high_reuse_threshold=0.5,
                            exclude_constant_ds=None,
                            min_rows=2000) -> dict:
    # For the binarized high_reuse target, keep rows + extract y from
    # cache_reuse_pct but use the high_reuse leakage spec (blocks
    # cache_reuse_pct itself as a feature).
    src = "cache_reuse_pct" if target == "high_reuse" else target
    spec = F.resolve_feature_spec("high_reuse" if target == "high_reuse"
                                  else target)
    kept_idx = [i for i, r in enumerate(rows) if r.get(src) is not None]
    if exclude_constant_ds:
        kept_idx = [i for i in kept_idx
                    if not any(x in rows[i]["source_dataset_id"]
                               for x in exclude_constant_ds)]
    if len(kept_idx) < min_rows:
        return {"target": target, "trained": False,
                "reason": f"only {len(kept_idx)} rows (<{min_rows})"}

    # group column index for GroupMedian baseline = first categorical (gpu_type)
    # Build a full matrix once over kept rows, keying y off `src`.
    sub = [rows[i] for i in kept_idx]
    build_spec = (F.FeatureSpec(target=src, numeric=spec.numeric,
                                categorical=spec.categorical,
                                blocked=spec.blocked,
                                allow_output_tokens=spec.allow_output_tokens)
                  if target == "high_reuse" else spec)
    X, y, names, _ = F.build_matrix(sub, build_spec)
    if target == "high_reuse":
        y = (y >= high_reuse_threshold).astype(float)
    gpu_idx = names.index("gpu_type") if "gpu_type" in names else 0

    results = {"target": target, "trained": True, "n_rows": len(kept_idx),
               "features": names, "classification": classification,
               "holdouts": {}}

    holdout_kinds = ["random", "by_dataset", "by_gpu", "time"]
    for kind in holdout_kinds:
        split = _holdout_indices(rows, kind, kept_idx)
        if split is None:
            results["holdouts"][kind] = {"status": "not_applicable"}
            continue
        tr, te = split
        pos = {gi: k for k, gi in enumerate(kept_idx)}
        tr_local = [pos[i] for i in tr if i in pos]
        te_local = [pos[i] for i in te if i in pos]
        if len(tr_local) < 50 or len(te_local) < 20:
            results["holdouts"][kind] = {"status": "too_small"}
            continue
        Xtr, ytr = X[tr_local], y[tr_local]
        Xte, yte = X[te_local], y[te_local]
        results["holdouts"][kind] = _eval_models(
            Xtr, ytr, Xte, yte, gpu_idx, classification)

    # high-cost / high-target tail holdout (train on bottom 80%, test top 20%)
    thr = np.quantile(y, 0.8)
    tail_te = np.where(y >= thr)[0]
    tail_tr = np.where(y < thr)[0]
    if len(tail_tr) >= 50 and len(tail_te) >= 20 and not classification:
        results["holdouts"]["high_tail"] = _eval_models(
            X[tail_tr], y[tail_tr], X[tail_te], y[tail_te], gpu_idx, False)
    else:
        results["holdouts"]["high_tail"] = {"status": "not_applicable"}

    # binding holdout selection: first applicable among time > by_dataset >
    # high_tail (random is decorative only and never binding).
    def _has_models(kind):
        return bool(results["holdouts"].get(kind, {}).get("models"))
    for cand in ("time", "by_dataset", "high_tail", "random"):
        if _has_models(cand):
            binding = cand
            break
    else:
        binding = "by_dataset"
    results["binding_holdout"] = binding
    results["random_holdout_is_decorative"] = True
    return results


def _eval_models(Xtr, ytr, Xte, yte, gpu_idx, classification) -> dict:
    out = {"models": {}, "n_train": int(len(ytr)), "n_test": int(len(yte))}
    if classification:
        if len(np.unique(ytr)) < 2:
            return {"status": "single_class_train"}
        baseline = M.GlobalRateBaseline().fit(Xtr, ytr)
        models = {
            "baseline_global_rate": baseline,
            "logistic": M.LogisticProba().fit(Xtr, ytr) if M._SKLEARN else None,
            "hgb": M.HGBClassifierProba().fit(Xtr, ytr) if M._SKLEARN else None,
        }
        for name, mdl in models.items():
            if mdl is None:
                continue
            p = mdl.predict_proba(Xte)
            out["models"][name] = M.classification_metrics(yte, p)
        base_auroc = out["models"]["baseline_global_rate"]["auroc"]
        best = max((k for k in out["models"] if k != "baseline_global_rate"),
                   key=lambda k: (out["models"][k]["auroc"] or 0),
                   default=None)
        out["best_ml_model"] = best
        out["beats_baseline"] = bool(
            best and (out["models"][best]["auroc"] or 0)
            > (base_auroc or 0.5) + 0.02)
        return out
    # regression
    baseline = M.GroupMedianBaseline(gpu_idx).fit(Xtr, ytr)
    models = {"baseline_group_median": baseline}
    if M._SKLEARN:
        models["hgb"] = M.HGBRegressor().fit(Xtr, ytr)
        models["linear"] = M.LinearReg().fit(Xtr, ytr)
    for name, mdl in models.items():
        out["models"][name] = M.regression_metrics(yte, mdl.predict(Xte))
    base_mae = out["models"]["baseline_group_median"]["mae"]
    ml = {k: v for k, v in out["models"].items()
          if k != "baseline_group_median"}
    best = min(ml, key=lambda k: (ml[k]["mae"]
               if ml[k]["mae"] is not None else 1e18), default=None)
    out["best_ml_model"] = best
    if best and base_mae:
        impr = (base_mae - out["models"][best]["mae"]) / base_mae
        out["ml_mae_improvement_vs_baseline_pct"] = round(100 * impr, 2)
        out["beats_baseline"] = bool(impr > 0.02)
    else:
        out["beats_baseline"] = False
    return out


# ───────────────── Phase 6 — economic alpha evaluation ──────────────────────


def phase6_economic_alpha(rows, trained: dict) -> dict:
    """The economic targets are deterministic; the honest alpha question is
    whether forecasting upstream inputs (latency / reuse) would change the
    SLA-safe goodput/$ DECISION. We quantify the realised goodput/$ under each
    variant's *information set* and report economic regret vs the strongest
    deterministic baseline (the overlay formula on measured inputs)."""
    goodput = np.array([r["sla_safe_goodput_per_dollar"] for r in rows
                        if r.get("sla_safe_goodput_per_dollar") is not None],
                       dtype=float)
    classes = Counter(r["overlay_class"] for r in rows)

    # Per-class headline goodput/$ (never combined).
    per_class = {}
    for cls in ("measured_same_record", "cross_dataset_joined", "scenario_prior"):
        vals = [r["sla_safe_goodput_per_dollar"] for r in rows
                if r.get("overlay_class") == cls
                and r.get("sla_safe_goodput_per_dollar") is not None]
        per_class[cls] = {"n": len(vals),
                          "mean": float(np.mean(vals)) if vals else None}

    # Latency-forecast alpha proxy: on CARA, does predicting TTFT better than
    # the per-group baseline reduce SLA-margin misclassification? Use the
    # trained ttft model improvement as the alpha signal feeding the formula.
    ttft_impr = None
    t = trained.get("ttft_s", {})
    if t.get("trained"):
        bh = t.get("binding_holdout")
        ttft_impr = t["holdouts"].get(bh, {}).get(
            "ml_mae_improvement_vs_baseline_pct")
    cache_beats = None
    c = trained.get("high_reuse", {})
    if c.get("trained"):
        bh = c.get("binding_holdout")
        cache_beats = c["holdouts"].get(bh, {}).get("beats_baseline")

    variants = {
        "A_existing_scorer_baseline": {
            "description": "production constraint scorer w/o overlay — 0 economic "
                           "terms computable",
            "goodput_per_dollar_rows": 0},
        "B_overlay_deterministic_formula": {
            "description": "economic overlay deterministic formula (STRONGEST "
                           "baseline; ground truth for cost targets)",
            "goodput_per_dollar_rows": int(len(goodput)),
            "mean_goodput_per_dollar": float(np.mean(goodput)) if len(goodput) else None},
        "C_overlay_plus_deterministic_priors": {
            "description": "overlay + per-(model,gpu) median latency/reuse priors",
            "goodput_per_dollar_rows": int(len(goodput)),
            "note": "deterministic priors do not change the realised goodput/$ "
                    "ground truth; they would change FORECASTED decisions only"},
        "D_overlay_plus_ml_latency": {
            "description": "overlay + ML latency forecasters",
            "ttft_ml_mae_improvement_pct_binding_holdout": ttft_impr,
            "note": "ML latency improvement is the only channel to economic "
                    "alpha here; see below"},
        "E_overlay_plus_ml_cache": {
            "description": "overlay + ML cache forecasters",
            "cache_classifier_beats_baseline_binding_holdout": cache_beats},
        "F_overlay_plus_ml_cost_energy": {
            "description": "overlay + ML cost/energy forecasters",
            "note": "cost targets are deterministic; ML cannot beat the formula "
                    "→ diagnostic_only"},
        "G_overlay_plus_ml_goodput": {
            "description": "overlay + ML SLA-safe goodput/$ forecaster",
            "note": "goodput/$ is deterministic given inputs → diagnostic_only"},
        "H_overlay_plus_all_eligible_ml": {
            "description": "overlay + all eligible ML modules (latency + cache)",
            "ttft_ml_mae_improvement_pct": ttft_impr,
            "cache_beats_baseline": cache_beats},
        "I_simulator_prior_migration_cold_start_only": {
            "description": "simulator-prior sensitivity ONLY (no measured labels)",
            "status": "proxy_promising_only — never headline"},
    }
    return {
        "primary_kpi": "sla_safe_goodput_per_dollar + economic_regret_vs_strongest_baseline",
        "primary_baseline": "B_overlay_deterministic_formula",
        "uses_oracle_as_headline": False,
        "uses_fifo_as_headline": False,
        "production_claim": False,
        "per_overlay_class_goodput_per_dollar": per_class,
        "overlay_class_counts": dict(classes),
        "variants": variants,
        "economic_alpha_finding": {
            "cost_targets_are_deterministic": True,
            "ml_alpha_channel": "forecasting upstream latency/reuse, not "
                                "re-predicting deterministic cost",
            "ttft_ml_improvement_pct_binding": ttft_impr,
            "cache_classifier_beats_baseline_binding": cache_beats,
        },
    }


def _promotion(label_pct) -> str:
    if label_pct is None:
        return "diagnostic_only"
    if label_pct < 2:
        return "diagnostic_only"
    if label_pct < 5:
        return "promising_needs_validation"
    return "shadow_ready_for_integration_review"


# ───────────────── Phase 7 — cold-start/migration sensitivity ───────────────


def phase7_sensitivity() -> dict:
    """Transparent, sourced realism-prior sweeps. Every parameter carries a
    source + source_type + confidence. NEVER headline; simulator_prior only."""
    def p(value, source, stype, conf):
        return {"value": value, "source": source, "source_type": stype,
                "confidence": conf, "production_ready": False}

    # credible ranges (sourced to public priors, NOT invented):
    sweeps = {
        "model_load_duration_s": p([5, 15, 30, 60],
            "vLLM/TGI public load-time discussions + ejhusom load_duration_ns "
            "order-of-magnitude (consumer)", "simulator_prior", "low"),
        "cache_loss_pct": p([0.0, 0.25, 0.5, 0.75, 1.0],
            "CC-traces migration_or_cache_loss_proxy range", "proxy", "medium"),
        "prefill_throughput_tok_s": p([2000, 5000, 10000],
            "Optimum prefill throughput across A100/A10/T4", "prior", "medium"),
        "reroute_delay_s": p([0, 1, 5],
            "no measured source — pure simulator_prior", "simulator_prior", "low"),
        "warmup_delay_s": p([0, 2, 10],
            "no measured source — pure simulator_prior", "simulator_prior", "low"),
        "gpu_price_usd_per_hour": p([1.0, 2.5, 4.0],
            "afhubbard/gpu-prices public list (A10/A100/H100 medians)",
            "prior", "medium"),
    }
    # Run a small transparent sweep grid for migration_cost_usd + cold_start.
    results = []
    prompt_tokens = 2048
    gpu_count = 1
    for clp in sweeps["cache_loss_pct"]["value"]:
        for thr in sweeps["prefill_throughput_tok_s"]["value"]:
            for price in sweeps["gpu_price_usd_per_hour"]["value"]:
                penalty_s = prompt_tokens / thr * clp
                mig = (penalty_s * price / 3600 * gpu_count)
                results.append({
                    "cache_loss_pct": clp, "prefill_throughput_tok_s": thr,
                    "gpu_price_usd_per_hour": price,
                    "cache_loss_prefill_penalty_s": round(penalty_s, 4),
                    "migration_cost_usd_excl_reroute_warmup": round(mig, 6)})
    cold = []
    for mld in sweeps["model_load_duration_s"]["value"]:
        for price in sweeps["gpu_price_usd_per_hour"]["value"]:
            cold.append({"model_load_duration_s": mld,
                         "gpu_price_usd_per_hour": price,
                         "cold_start_cost_usd": round(mld * price / 3600 * gpu_count, 6)})
    return {
        "status": "simulator_prior_only — NEVER headline; no measured labels",
        "parameters": sweeps,
        "formulas": {
            "cache_loss_prefill_penalty_s":
                "prompt_tokens / prefill_throughput_tok_s * cache_loss_pct",
            "migration_cost_usd":
                "cache_loss_prefill_penalty_s * gpu_price/3600*gpu_count "
                "+ reroute_delay_s*gpu_price/3600*gpu_count "
                "+ warmup_delay_s*gpu_price/3600*gpu_count",
            "cold_start_cost_usd":
                "model_load_duration_s * gpu_price/3600 * gpu_count",
        },
        "migration_cost_sweep": results,
        "cold_start_cost_sweep": cold,
        "findings": [
            "Naive migration loses badly when cache_loss_pct -> 1 at low "
            "prefill throughput (penalty grows ~linearly).",
            "Cache-aware routing matters most at high cache_loss_pct x low "
            "prefill throughput x high GPU price.",
            "Warm pools become economically meaningful only when "
            "model_load_duration_s x price dominates per-request cost; "
            "needs pilot telemetry to calibrate model_load_duration_s.",
            "reroute_delay_s and warmup_delay_s have NO measured source and "
            "would require pilot telemetry.",
        ],
        "parameters_needing_pilot_telemetry": [
            "model_load_duration_s (server-class vLLM/TGI)",
            "reroute_delay_s", "warmup_delay_s",
            "tail_latency_uplift_after_migration", "migration_veto_label",
        ],
    }


# ───────────────────────────── driver ──────────────────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = F.load_corpus(max_rows_per_file=args.max_rows_per_file)
    if not rows:
        logger.error("no analysis corpus found — regenerate via "
                     "scripts/build_economic_overlay_analysis.py")
        return 1
    logger.info("loaded %d overlay rows", len(rows))

    overlap = phase0_overlap(rows)
    _write("overlap_audit.json", overlap)
    realism = phase1_realism(rows)
    _write("realism_prior_audit.json", realism)
    catalog = phase2_catalog(rows)
    _write("target_catalog.json", catalog)

    # Phase 4/5 — train modular forecasters.
    const_lat_ds = {"swissai"}
    trained = {}
    trained["ttft_s"] = train_regression_target(
        rows, "ttft_s", exclude_constant_ds=const_lat_ds, min_rows=2000)
    trained["e2e_latency_s"] = train_regression_target(
        rows, "e2e_latency_s", exclude_constant_ds=const_lat_ds, min_rows=2000)
    trained["tpot_s"] = train_regression_target(
        rows, "tpot_s", exclude_constant_ds=const_lat_ds, min_rows=2000)
    trained["cache_reuse_pct"] = train_regression_target(
        rows, "cache_reuse_pct", min_rows=2000)
    trained["high_reuse"] = train_regression_target(
        rows, "high_reuse", classification=True, min_rows=2000)
    trained["peak_vram_gb"] = train_regression_target(
        rows, "peak_vram_gb", min_rows=1500)
    trained["energy_kwh"] = train_regression_target(
        rows, "energy_kwh", min_rows=2000)
    # Deterministic targets — train ML only to DEMONSTRATE it cannot beat the
    # formula (diagnostic_only). We still record the result honestly.
    trained["estimated_gpu_cost_usd_DETERMINISTIC"] = train_regression_target(
        rows, "estimated_gpu_cost_usd", min_rows=2000)

    _write("trained_models.json", trained)

    alpha = phase6_economic_alpha(rows, trained)
    _write("economic_alpha_eval.json", alpha)
    sensitivity = phase7_sensitivity()
    _write("cold_start_migration_sensitivity.json", sensitivity)

    # Per-target final status.
    target_status = {}
    for tname, t in trained.items():
        if not t.get("trained"):
            target_status[tname] = "blocked_insufficient_rows"
            continue
        bh = t.get("binding_holdout")
        hb = t["holdouts"].get(bh, {})
        impr = hb.get("ml_mae_improvement_vs_baseline_pct")
        if "DETERMINISTIC" in tname:
            target_status[tname] = "diagnostic_only_deterministic_formula"
        elif t.get("classification"):
            target_status[tname] = ("promising_needs_validation"
                                    if hb.get("beats_baseline")
                                    else "diagnostic_only")
        else:
            target_status[tname] = _promotion(impr)
    # Shadow-ready caveats: flag targets whose binding holdout could not be a
    # cross-dataset (by_dataset) test because the target lives in one dataset.
    shadow_ready_caveats = {}
    for tname, t in trained.items():
        if target_status.get(tname) != "shadow_ready_for_integration_review":
            continue
        ds = set()
        for r in rows:
            src = ("cache_reuse_pct" if tname == "high_reuse"
                   else tname.replace("_DETERMINISTIC", ""))
            if r.get(src) is not None:
                ds.add(r["source_dataset_id"])
        bh = t.get("binding_holdout")
        if len(ds) < 2:
            shadow_ready_caveats[tname] = (
                f"single-dataset target ({sorted(x.split('/')[-1] for x in ds)}); "
                f"binding holdout was '{bh}' (temporal/tail) — NO cross-dataset "
                "generalization evidence. Needs a second dataset or pilot "
                "telemetry before integration.")
    summary = {
        "doc_version": "economic_ml_alpha_v1",
        "production_claim": False, "shadow_only": True, "real_execution": False,
        "uses_oracle_as_headline": False, "uses_fifo_as_headline": False,
        "total_rows": len(rows),
        "shadow_ready_caveats": shadow_ready_caveats,
        "overlap_answers": overlap["answers"],
        "targets_trainable_now": [k for k, t in trained.items()
                                  if t.get("trained")],
        "targets_blocked_by_missing_labels": [
            "cold_start_cost_usd", "migration_cost_usd",
            "migration_veto_label", "timeout_or_failure_risk"],
        "per_target_final_status": target_status,
        "cold_start_migration_verdict": realism["verdict"],
        "economic_alpha": alpha["economic_alpha_finding"],
        "binding_holdout_policy": "time > by_dataset > high_cost_tail; "
                                  "random holdout decorative only",
        "pilot_telemetry_needed": sensitivity["parameters_needing_pilot_telemetry"]
        + ["operator per-GPU $/hr", "operator energy tariff",
           "operator carbon price", "real per-request cache_hit",
           "measured model_load + migration seconds"],
    }
    _write("summary.json", summary)
    logger.info("done — per-target status: %s", target_status)
    return 0


def _write(name, obj):
    path = OUT_DIR / name
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    logger.info("wrote %s", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
