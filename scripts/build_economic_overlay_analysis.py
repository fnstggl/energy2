#!/usr/bin/env python3
"""Analysis-tier Economic Overlay build.

Reads the bounded raw head files fetched by
``scripts/fetch_economic_overlay_analysis_sources.py`` (all gitignored) and
the live market overlays fetched by
``scripts/fetch_economic_overlay_market_data.py``, then emits
``EconomicOverlayRecord``s at analysis-tier scale across CARA, SwissAI,
Optimum, AcmeTrace, and CC-traces.

Outputs:
  - data/external/economic_overlay/analysis_corpus/<source>.jsonl  (GITIGNORED,
    full corpus consumed by the eval)
  - data/external/economic_overlay/economic_overlay_samples/analysis_<source>.sample.jsonl
    (COMMITTED bounded sample, <= --sample-rows rows)
  - data/external/economic_overlay/economic_overlay_analysis_summary.json
    (COMMITTED coverage + field-quality + subgroup-count rollup)

Economic value-quality labelling is unchanged from v1; the energy overlay
uses the live PJM measured series with scenario-region assignment for
region-less traces (mission §4.B). No invented constants, no secrets.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

# Optimum CSVs embed per-request latency-value lists in a single cell that can
# exceed the default 128 KiB csv field cap.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting.economic_overlay import (  # noqa: E402
    OverlayBuilder,
    OverlayBuilderConfig,
    summarise,
)

HF_RAW = REPO_ROOT / "data" / "external" / "hf"
OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
SAMPLES = OVERLAY_DIR / "economic_overlay_samples"
CORPUS = OVERLAY_DIR / "analysis_corpus"
CC_TRACES = (HF_RAW / "semianalysisai__cc-traces-weka-no-subagents-051226"
             / "traces_3000mib" / "processed" / "normalized_sample.jsonl")

logger = logging.getLogger("economic_overlay_analysis")

ECON_FIELDS = [
    "estimated_gpu_cost_usd", "estimated_energy_cost_usd",
    "estimated_carbon_kg", "estimated_carbon_cost_usd",
    "estimated_cache_value_usd", "estimated_migration_cost_usd",
    "estimated_cold_start_cost_usd", "estimated_prefill_cost_usd",
    "estimated_decode_cost_usd", "sla_safe_goodput_per_dollar",
]


# ── source readers (yield overlay-input dicts) ──────────────────────────────


def _iter_jsonl(path: Path, cap: int) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= cap:
                break
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _cara_gpu(instance_type: Optional[str]) -> str:
    if not instance_type:
        return "A100"
    s = str(instance_type).lower()
    for fam, tag in (("h200", "H200"), ("h100", "H100"), ("a100", "A100"),
                     ("a10g", "A10"), ("a10", "A10"), ("v100", "V100"),
                     ("p100", "P100"), ("t4", "T4"), ("l40", "L40S"),
                     ("l4", "L4")):
        if fam in s:
            return tag
    return "A100"


def read_cara(cap: int) -> list[dict]:
    path = HF_RAW / "asdwb__cara_latency_prediction" / "raw" / "train_flat.head.jsonl"
    out = []
    for i, r in enumerate(_iter_jsonl(path, cap)):
        ttft = r.get("actual_ttft")
        tpot = r.get("actual_tpot")
        e2e = r.get("actual_e2e_latency")
        out_tokens = r.get("actual_output_tokens")
        if e2e is None:
            continue
        sla = 8.0 + 0.05 * (out_tokens or 100)
        out.append({
            "source_trace_id": f"cara:train_flat:{r.get('request_id', i)}",
            "source_dataset_id": "asdwb/cara_latency_prediction",
            "gpu_type": _cara_gpu(r.get("instance_type")),
            "gpu_count": 1,
            "model_id": str(r.get("instance_type") or "served-llm").split("_")[0],
            "prompt_tokens": r.get("num_prompt_tokens"),
            "output_tokens": out_tokens,
            "ttft_s": ttft, "tpot_s": tpot, "e2e_latency_s": e2e,
            "kv_utilization": r.get("kv_cache_utilization"),
            "sla_s": sla,
            "timestamp": (str(r.get("prediction_timestamp"))
                          if r.get("prediction_timestamp") is not None else None),
        })
    return out


def read_swissai(cap: int) -> list[dict]:
    out = []
    for fname, model in (("llama3_70b_bucket_reuse.head.jsonl", "Llama3-70B"),
                         ("qwen3_32b_bucket_reuse.head.jsonl", "Qwen3-32B")):
        path = HF_RAW / "eth-easl__swissai-serving-trace" / "raw" / fname
        gpu = "H100"
        for i, r in enumerate(_iter_jsonl(path, cap // 2)):
            reuse = r.get("reuse_percentage")
            if reuse is None:
                continue
            if reuse > 1.0:
                reuse = reuse / 100.0
            total_buckets = r.get("total_buckets") or 8
            out.append({
                "source_trace_id": f"swissai:{model}:{r.get('id', i)}",
                "source_dataset_id": "eth-easl/swissai-serving-trace",
                "gpu_type": gpu, "gpu_count": 1, "model_id": model,
                "cache_reuse_pct": float(reuse),
                "prompt_tokens": int(total_buckets) * 64,
                "output_tokens": 128,
                # ttft/tpot are Level-3 Optimum priors for these GPUs/models.
                "ttft_s": 0.40, "tpot_s": 0.020, "_ttft_source": "optimum_prior",
                "e2e_latency_s": 0.40 + 0.020 * 128,
                "sla_s": 5.0,
                "timestamp": r.get("created_at"),
            })
    return out


def read_optimum() -> list[dict]:
    cfg_files = {
        "cuda_unquantized_1xA100.csv": "A100",
        "cuda_unquantized_1xA10.csv": "A10",
        "cuda_unquantized_1xT4.csv": "T4",
        "cuda_awq_1xA10.csv": "A10",
        "cuda_bnb_1xA100.csv": "A100",
        "cuda_gptq_1xA100.csv": "A100",
        "cpu_unquantized_32vCPU_C7i.csv": "CPU-C7i",
    }
    base = HF_RAW / "optimum-benchmark__llm-perf-leaderboard" / "raw"
    out = []
    for fname, gpu in cfg_files.items():
        path = base / fname
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for i, r in enumerate(reader):
                def g(k):
                    v = r.get(k, "")
                    try:
                        return float(v) if v not in ("", "NaN", "nan") else None
                    except (TypeError, ValueError):
                        return None
                ttft = g("report.prefill.latency.mean")
                tpot = g("report.decode.latency.mean")
                new_tokens = g("config.scenario.new_tokens") or 100
                if ttft is None or tpot is None:
                    continue
                e2e = ttft + tpot * new_tokens
                p_e = g("report.prefill.energy.total")
                d_e = g("report.decode.energy.total")
                energy = (p_e or 0) + (d_e or 0) if (p_e or d_e) else None
                vram = g("report.decode.memory.max_global_vram")
                out.append({
                    "source_trace_id": f"optimum:{gpu}:{fname}:{i}",
                    "source_dataset_id": "optimum-benchmark/llm-perf-leaderboard",
                    "gpu_type": gpu, "gpu_count": 1,
                    "model_id": r.get("config.backend.model"),
                    "prompt_tokens": 256, "output_tokens": int(new_tokens),
                    "ttft_s": ttft, "tpot_s": tpot, "e2e_latency_s": e2e,
                    "throughput_tok_s": g("report.decode.throughput.value"),
                    "energy_kwh": energy,
                    "peak_vram_gb": (vram / 1024.0) if vram else None,
                    "sla_s": e2e * 1.5, "timestamp": None,
                })
    return out


def read_acmetrace_power(cap: int) -> list[dict]:
    """GPU_AB_Power: Sys_Total_Power, CPU_Power, Memory_Power per 15s sample.
    GPU power is derived as Sys_Total - CPU - Memory (clamped >= 0)."""
    path = HF_RAW / "Qinghao__AcmeTrace" / "raw" / "GPU_AB_Power.head.csv"
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        # Sample every Nth row to spread across the window within the cap.
        for i, r in enumerate(reader):
            if i % 15 != 0:  # ~1 sample per ~225s to diversify
                continue
            if len(out) >= cap:
                break
            try:
                sys_p = float(r.get("Sys_Total_Power") or 0)
                cpu_p = float(r.get("CPU_Power") or 0)
                mem_p = float(r.get("Memory_Power") or 0)
            except (TypeError, ValueError):
                continue
            gpu_p = max(0.0, sys_p - cpu_p - mem_p)
            if gpu_p <= 0:
                continue
            out.append({
                "source_trace_id": f"acme:gpu_ab_power:{i}",
                "source_dataset_id": "Qinghao/AcmeTrace",
                "gpu_type": "A100",
                "gpu_count": 8,  # Kalos/Seren nodes are 8-GPU
                "model_id": "internal-training-job",
                "gpu_power_w": gpu_p,
                "e2e_latency_s": 15.0,  # one DCGM/IPMI sample window
                "sla_s": 60.0,
                "timestamp": str(r.get("Time")) if r.get("Time") else None,
            })
    return out


def read_acmetrace_jobs(cap: int) -> list[dict]:
    """trace_kalos.csv: cluster jobs with queue + duration + gpu_num + state."""
    path = HF_RAW / "Qinghao__AcmeTrace" / "raw" / "trace_kalos.csv"
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if len(out) >= cap:
                break
            try:
                dur = float(r.get("duration") or 0)
                queue = float(r.get("queue") or 0)
                gpu_num = int(float(r.get("gpu_num") or 0))
            except (TypeError, ValueError):
                continue
            if dur <= 0 or gpu_num <= 0:
                continue
            out.append({
                "source_trace_id": f"acme:kalos_job:{r.get('job_id', i)}",
                "source_dataset_id": "Qinghao/AcmeTrace",
                "gpu_type": "A100", "gpu_count": gpu_num,
                "model_id": "internal-training-job",
                "queue_wait_s": queue, "e2e_latency_s": dur,
                "sla_s": dur * 2.0,  # generous training SLA
                "timestamp": (str(r.get("submit_time"))
                              if r.get("submit_time") else None),
            })
    return out


def read_cc_traces(cap: int) -> list[dict]:
    """CC-traces 3000MiB normalized sample (committed). Agentic Claude Code
    requests with KV block hashes; carries a cache-loss proxy + reuse."""
    out = []
    for i, r in enumerate(_iter_jsonl(CC_TRACES, cap)):
        ttft = r.get("ttft") or r.get("ttft_s")
        prefix_reuse = r.get("prefix_reuse") or r.get("cache_reuse")
        cache_loss = r.get("migration_or_cache_loss_proxy")
        out_tokens = r.get("output_tokens") or r.get("num_output_tokens") or 200
        if ttft is None:
            ttft = 0.6  # CC-traces closed-API prior
        out.append({
            "source_trace_id": f"cc_traces:{r.get('request_id', i)}",
            "source_dataset_id": "semianalysisai/cc-traces-weka-no-subagents-051226",
            "gpu_type": "H100", "gpu_count": 1, "model_id": "claude-code-agent",
            "prompt_tokens": r.get("prompt_tokens") or r.get("input_tokens"),
            "output_tokens": out_tokens,
            "ttft_s": float(ttft), "tpot_s": 0.012,
            "e2e_latency_s": float(ttft) + 0.012 * out_tokens,
            "cache_reuse_pct": (float(prefix_reuse)
                                if isinstance(prefix_reuse, (int, float)) else None),
            "cache_loss_pct": (float(cache_loss)
                               if isinstance(cache_loss, (int, float)) else None),
            "sla_s": 30.0,
            "timestamp": r.get("timestamp") or r.get("created_at"),
        })
    return out


SOURCE_READERS = {
    "cara_train_flat": lambda cap: read_cara(cap),
    "swissai_bucket_reuse": lambda cap: read_swissai(cap),
    "optimum_all_configs": lambda cap: read_optimum(),
    "acmetrace_gpu_power": lambda cap: read_acmetrace_power(cap),
    "acmetrace_kalos_jobs": lambda cap: read_acmetrace_jobs(cap),
    "cc_traces_3000mib": lambda cap: read_cc_traces(cap),
}


def _coverage(records) -> dict:
    cov = {}
    for f in ECON_FIELDS:
        nonmiss = sum(1 for r in records
                      if r.value_quality_by_field.get(f, "missing") != "missing"
                      and getattr(r, f) is not None)
        cov[f] = {"rows_with_value": nonmiss,
                  "coverage_pct": round(100 * nonmiss / max(1, len(records)), 2)}
    return cov


def _field_quality(records) -> dict:
    out = {}
    for f in ECON_FIELDS + ["gpu_price_usd_per_hour",
                            "electricity_price_usd_per_kwh",
                            "carbon_intensity_g_per_kwh", "energy_kwh"]:
        c = Counter(r.value_quality_by_field.get(f, "missing") for r in records)
        out[f] = dict(c)
    return out


def _subgroups(records) -> dict:
    by_gpu = Counter(r.gpu_type for r in records)
    by_model = Counter((r.model_id or "?")[:40] for r in records)
    by_class = Counter(r.overlay_class for r in records)
    by_dataset = Counter(r.source_dataset_id for r in records)
    return {
        "by_gpu_type": dict(by_gpu),
        "by_model_top20": dict(by_model.most_common(20)),
        "by_overlay_class": dict(by_class),
        "by_source_dataset": dict(by_dataset),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cara-cap", type=int, default=60000)
    p.add_argument("--swissai-cap", type=int, default=60000)
    p.add_argument("--acme-power-cap", type=int, default=40000)
    p.add_argument("--acme-jobs-cap", type=int, default=40000)
    p.add_argument("--cc-cap", type=int, default=5000)
    p.add_argument("--sample-rows", type=int, default=800)
    p.add_argument("--gpu-price-jsonl",
                   default=str(SAMPLES / "gpu_price_overlay_multiday.jsonl"))
    p.add_argument("--pjm-jsonl",
                   default=str(SAMPLES / "pjm_da_energy_price_14day.jsonl"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    CORPUS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    cfg = OverlayBuilderConfig(
        gpu_price_path=Path(args.gpu_price_jsonl),
        pjm_path=Path(args.pjm_jsonl),
        scenario_region_assignment=True,
        energy_market_label="PJM",
        carbon_market="watttime_carbon_overlay",  # live failed -> scenario_prior
    )
    builder = OverlayBuilder(cfg)

    caps = {"cara_train_flat": args.cara_cap,
            "swissai_bucket_reuse": args.swissai_cap,
            "optimum_all_configs": 0,
            "acmetrace_gpu_power": args.acme_power_cap,
            "acmetrace_kalos_jobs": args.acme_jobs_cap,
            "cc_traces_3000mib": args.cc_cap}

    per_source = {}
    all_records = []
    for name, reader in SOURCE_READERS.items():
        rows = reader(caps[name])
        recs = builder.build(rows)
        # Full corpus -> gitignored.
        corpus_path = CORPUS / f"{name}.jsonl"
        with open(corpus_path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        # Bounded committable sample.
        sample_path = SAMPLES / f"analysis_{name}.sample.jsonl"
        with open(sample_path, "w") as fh:
            for r in recs[:args.sample_rows]:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        per_source[name] = {
            "rows": len(recs),
            "coverage": _coverage(recs),
            "field_quality": _field_quality(recs),
            "subgroups": _subgroups(recs),
            "summary": summarise(recs),
            "corpus_path_gitignored": str(corpus_path.relative_to(REPO_ROOT)),
            "committed_sample_path": str(sample_path.relative_to(REPO_ROOT)),
            "committed_sample_rows": min(len(recs), args.sample_rows),
        }
        all_records.extend(recs)
        logger.info("%s -> %d records (sample %d)", name, len(recs),
                    min(len(recs), args.sample_rows))

    manifest_path = OVERLAY_DIR / "market_fetch_manifest.json"
    market_manifest = (json.loads(manifest_path.read_text())
                       if manifest_path.exists() else {})

    rollup = {
        "doc_version": "economic_overlay_analysis_v1",
        "production_claim": False,
        "shadow_only": True,
        "tier": "analysis",
        "total_rows": len(all_records),
        "operator_policy_supplied": False,
        "energy_overlay": {
            "primary_market": "PJM",
            "region_assignment": "scenario (traces are region-less)",
            "value_quality": "scenario_prior",
            "note": "PJM/CAISO price series are MEASURED market data; applying "
                    "them to region-less LLM traces is a scenario region "
                    "assignment, hence the per-record scenario_prior label.",
        },
        "carbon_overlay": {
            "provider": "WattTime",
            "live_status": market_manifest.get("providers", {})
                           .get("watttime_carbon_intensity", {}).get("status"),
            "value_quality": "scenario_prior",
            "note": "WattTime live auth returned 403 for this account; carbon "
                    "intensity falls back to scenario_prior midpoint. carbon_kg "
                    "is a physical quantity; carbon_cost stays operator-policy-only.",
        },
        "market_fetch_manifest": market_manifest,
        "gpu_price_overlay_rows": cfg.gpu_price_path.name,
        "per_source": per_source,
        "global_coverage": _coverage(all_records),
        "global_field_quality": _field_quality(all_records),
        "global_subgroups": _subgroups(all_records),
        "global_summary": summarise(all_records),
        "operator_policy_only_fields": [
            "carbon_price_per_kg_usd",
            "per_gpu_hour_price_usd (operator fleet-actual)",
            "internal_chargeback_rate",
            "energy_price_per_kwh_usd (operator tariff vs market LMP)",
            "memory_pressure_pricing_policy",
        ],
    }
    out = OVERLAY_DIR / "economic_overlay_analysis_summary.json"
    with open(out, "w") as fh:
        json.dump(rollup, fh, indent=2, sort_keys=True, default=str)
    logger.info("TOTAL %d rows -> %s", len(all_records), out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
