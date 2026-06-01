#!/usr/bin/env python3
"""Frontier Discovery Audit v1 — discover NEW operational signals/datasets that
could unlock economic-alpha forecasting beyond the already-validated
cache_reuse / TTFT / peak_VRAM.

Metadata-only: queries the public HF dataset API (card + README + siblings +
features). Downloads NO data, ingests nothing, trains nothing. HF_TOKEN is read
from the environment and never written to any artefact.

Emits (under data/external/hf_discovery/frontier_v1/):
  frontier_dataset_registry.json          — per-candidate card + signal map
  frontier_field_matrix.json              — dataset x signal-category matrix
  economic_frontier_priority_ranking.json — ranked signals + datasets + top-N

Signal categories audited (mission Phase 1): cold_start, migration, queueing,
memory_pressure, serving_stability, autoscaling, plus the baseline ops/econ
signals (ttft/tpot/e2e/throughput/cache/energy/cost/gpu).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "external" / "hf_discovery" / "frontier_v1"
HF_API = "https://huggingface.co/api"
UA = "aurelius-frontier-discovery/1.0"

# ── search probes (single / two-word; HF text search is AND-strict) ─────────
PROBES = [
    "vllm", "sglang", "tensorrt", "serving", "kv-cache", "kvcache",
    "cold-start", "coldstart", "autoscaling", "kserve", "ray-serve",
    "inference-benchmark", "llm-benchmark", "gpu-trace", "cluster-trace",
    "scheduler", "prometheus", "dcgm", "triton", "latency-benchmark",
    "llmperf", "genai-perf", "moe-inference", "speculative-decoding",
    "prefill", "decode", "vllm benchmark", "sglang benchmark",
    "serving trace", "gpu telemetry", "kubernetes metrics", "queue wait",
    "model load", "warmup", "migration trace", "preemption",
    "gpu utilization", "power telemetry", "ipmi", "nvml", "carbon intensity",
    "spot instance", "tail latency", "p99 latency", "throughput benchmark",
    "ttft", "tpot", "disaggregated serving", "prefix cache", "eviction",
    "request trace", "workload trace", "datacenter trace", "h100 benchmark",
    "h200 benchmark", "a100 benchmark", "llm inference latency",
    "ray cluster", "node failure", "job trace",
]

# ── signal-category regex (matched against card text + features + files) ────
SIGNAL_PATTERNS = {
    "ttft": [r"\bttft\b", r"time[_ -]?to[_ -]?first[_ -]?token", r"prefill[_ ]?latency"],
    "tpot_itl": [r"\btpot\b", r"\bitl\b", r"inter[_ -]?token", r"decode[_ ]?latency",
                 r"\btbt\b"],
    "e2e_latency": [r"\be2e\b", r"end[_ -]?to[_ -]?end[_ ]?latency", r"\brequest[_ ]?latency\b"],
    "throughput": [r"\bthroughput\b", r"tokens?[_ ]?per[_ ]?sec", r"\btok/s\b",
                   r"\btps\b", r"req(uests)?[_ ]?per[_ ]?sec"],
    "queueing": [r"\bqueue[_ ]?wait\b", r"\bqueue[_ ]?depth\b", r"\bqueue[_ ]?time\b",
                 r"\bwaiting[_ ]?requests?\b", r"admission[_ ]?delay",
                 r"scheduler[_ ]?delay", r"\bnum[_ ]?waiting\b", r"pending[_ ]?requests?"],
    "cold_start": [r"cold[_ -]?start", r"model[_ ]?load(ing|_duration)?",
                   r"weight[_ ]?(load|transfer)", r"warm[_ ]?up", r"warmup",
                   r"graph[_ ]?capture", r"image[_ ]?pull", r"compile[_ ]?(time|latency)",
                   r"startup[_ ]?latency", r"scale[_ -]?from[_ -]?zero",
                   r"load_duration"],
    "migration": [r"\bmigrat", r"re[_ -]?rout", r"\bpreempt", r"\bevict",
                  r"drain", r"cache[_ ]?loss", r"locality", r"traffic[_ ]?shift",
                  r"\bveto\b"],
    "memory_pressure": [r"\boom\b", r"out[_ -]?of[_ -]?memory", r"kv[_ ]?evict",
                        r"\bvram\b", r"gpu[_ ]?memory", r"memory[_ ]?pressure",
                        r"fragmentation", r"peak[_ ]?mem", r"kv[_ ]?cache[_ ]?util",
                        r"max_global_vram", r"kv_free_blocks"],
    "serving_stability": [r"\bp95\b", r"\bp99\b", r"\btimeout", r"\bretry\b",
                          r"overload", r"failure[_ ]?rate", r"error[_ ]?rate",
                          r"\bsla\b", r"tail[_ ]?latency"],
    "autoscaling": [r"auto[_ -]?scal", r"scale[_ -]?up", r"scale[_ -]?down",
                    r"\breplica", r"warm[_ ]?pool", r"oscillat", r"\bhpa\b",
                    r"scaling[_ ]?event"],
    "gpu_telemetry": [r"\bdcgm\b", r"\bnvml\b", r"\bipmi\b", r"gpu[_ ]?util",
                      r"gpu[_ ]?power", r"power[_ ]?draw", r"\bwatt"],
    "energy": [r"\bkwh\b", r"\bjoule", r"energy[_ ]?(consum|kwh|per)", r"codecarbon",
               r"\brapl\b"],
    "carbon": [r"carbon[_ ]?intensity", r"\bco2\b", r"gco2", r"electricitymaps",
               r"watttime"],
    "cost_price": [r"\$/hr", r"price[_ ]?per[_ ]?hour", r"gpu[_ ]?price", r"\bspot[_ ]?price",
                   r"on[_ -]?demand", r"cost[_ ]?per[_ ]?(token|request|hour)",
                   r"billing", r"chargeback"],
}

FRONTIER_CATEGORIES = ["cold_start", "migration", "queueing", "memory_pressure",
                       "serving_stability", "autoscaling"]

# Compressed / serialized tabular extensions also count as data files.
DATA_EXT = (".csv", ".json", ".jsonl", ".parquet", ".tsv", ".csv.xz",
            ".csv.gz", ".parquet.gz", ".pkl", ".feather", ".arrow")

# Curated high-value serving-systems seeds to force-inspect (real infra
# traces/benchmarks found via org enumeration; many are license=None or
# compressed, which the scoring records honestly).
SEED_IDS = [
    "project-vajra/dev-staging-h100-dgx",
    "project-vajra/dev-staging-a100-dgx",
    "project-vajra/dev-staging-meta-llama-llama-3-70b-h100",
    "project-vajra/dev-staging-meta-llama-llama-3-8b-h100",
    "project-vajra/dev-staging-h100-pairwise-nvlink",
    "project-vajra/prefill-decode-meta-llama-meta-llama-3-8b-instruct-h200-nvl",
    "project-vajra/sep-prefill-meta-llama-meta-llama-3-8b-instruct-h200-nvl",
    "intellistream/sage-control-plane-benchmark",
    "intellistream/sage-control-plane-workloads",
    "intellistream/sage-control-plane-llm-workloads",
    "intellistream/sagellm-benchmark-results",
    "intellistream/vllm-hust-benchmark-results",
    "Isabella5/sglang-seglen-benchmark",
    "BBuf/ltx-fp8-sglang-benchmark-results",
    "crozai/vllm-benchmark-coding",
    "mbicanic/vllm-benchmark-coding",
    "metrum-ai/llm-perfdata",
    "hlarcher/inference-benchmarker",
    "rbgo/llm-inference-benchmark",
    "nishant-k/speculative-decoding-benchmark-results",
    "kshitijthakkar/moe-inference-benchmark",
    "Qinghao/AcmeTrace",
    "DistServe/2025-05-06T14-automatic-profiling",
]

# Strict infra-relevance regex on the dataset id (excludes NLP-eval noise).
INFRA_ID = re.compile(
    r"vllm|sglang|tensorrt|vajra|sarathi|distserv|kv[-_]?cache|kvcache|"
    r"serving[-_]?trace|gpu[-_]?(trace|telemetry|util|power)|cluster[-_]?trace|"
    r"acmetrace|supercloud|prefill|decode|disaggreg|inference[-_]?bench|"
    r"llm[-_]?perf|perfdata|benchmark[-_]?results|control[-_]?plane|"
    r"autoscal|coldstart|cold[-_]?start|preempt|migrat|"
    r"(h100|h200|a100|a40|l40|v100|p100)[-_]?(dgx|nvlink|pcie|bench)",
    re.I)


def _hf_headers():
    tok = os.environ.get("HF_TOKEN")
    return {"User-Agent": UA, **({"Authorization": f"Bearer {tok}"} if tok else {})}


def _get(url, *, raw=False, cap=200_000):
    req = urllib.request.Request(url, headers=_hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read(cap if raw else None)
            return body.decode("utf-8", "replace") if raw else json.loads(body)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            json.JSONDecodeError):
        return None


def search(q, limit=25):
    url = f"{HF_API}/datasets?" + urllib.parse.urlencode({"search": q, "limit": limit})
    r = _get(url)
    return [x["id"] for x in r if isinstance(x, dict) and "id" in x] if isinstance(r, list) else []


def card(ds):
    return _get(f"{HF_API}/datasets/{urllib.parse.quote(ds, safe='/')}")


def readme(ds):
    txt = _get(f"https://huggingface.co/datasets/"
               f"{urllib.parse.quote(ds, safe='/')}/raw/main/README.md", raw=True)
    return txt or ""


def _load_already_evaluated() -> set:
    """Dataset ids already inspected in prior discovery rounds (skip them as
    'previously_evaluated' but still record the pointer)."""
    seen = set()
    reg = REPO_ROOT / "docs" / "HF_DATASET_REGISTRY.md"
    if reg.exists():
        for m in re.findall(r"`([\w./-]+/[\w.-]+)`", reg.read_text()):
            if "/" in m and not m.endswith((".py", ".md", ".json", ".parquet")):
                seen.add(m)
    cand = OUT_DIR.parent / "hf_dataset_candidates.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text())
            for c in d.get("candidates", []):
                if c.get("dataset_id"):
                    seen.add(c["dataset_id"])
        except json.JSONDecodeError:
            pass
    return seen


def detect_signals(text: str) -> dict:
    low = text.lower()
    out = {}
    for sig, pats in SIGNAL_PATTERNS.items():
        hits = [p for p in pats if re.search(p, low)]
        if hits:
            out[sig] = len(hits)
    return out


def _extract(meta: dict) -> dict:
    card_data = meta.get("cardData") or {}
    lic = card_data.get("license")
    sibs = [s.get("rfilename", "") for s in meta.get("siblings", [])]
    data_files = [s for s in sibs if s.endswith(DATA_EXT)]
    feats = []
    di = card_data.get("dataset_info")
    infos = di if isinstance(di, list) else ([di] if di else [])
    for info in infos:
        if isinstance(info, dict):
            for f in (info.get("features") or []):
                if isinstance(f, dict) and "name" in f:
                    feats.append(f["name"])
    rows = None
    for info in infos:
        if isinstance(info, dict):
            for sp in (info.get("splits") or []):
                if isinstance(sp, dict) and isinstance(sp.get("num_examples"), int):
                    rows = (rows or 0) + sp["num_examples"]
    return {"license": lic, "data_files": data_files[:25], "features": feats[:60],
            "rows": rows, "downloads": meta.get("downloads"),
            "gated": meta.get("gated"), "tags": meta.get("tags", [])}


def forecastability_score(rec: dict) -> tuple[int, str]:
    """0-100. Rewards real data files + variable telemetry features + rows;
    penalises gated / no-data / request-shape-only / no-license."""
    s = 0
    reasons = []
    if rec["extract"]["gated"] not in (False, None):
        return 0, "gated_blocked"
    nfiles = len(rec["extract"]["data_files"])
    if nfiles == 0:
        return 5, "no_data_files (card/README only)"
    fb = min(20, nfiles * 4)
    s += fb
    reasons.append(f"+{fb} data files")
    rows = rec["extract"]["rows"]
    if rows:
        rb = 25 if rows >= 50000 else 18 if rows >= 5000 else 10 if rows >= 500 else 4
        s += rb
        reasons.append(f"+{rb} rows~{rows}")
    else:
        s += 6
        reasons.append("+6 rows unknown (parquet/csv present)")
    # telemetry richness: frontier categories present
    fc = [c for c in FRONTIER_CATEGORIES if c in rec["signals"]]
    fcb = min(30, len(fc) * 8)
    s += fcb
    reasons.append(f"+{fcb} frontier cats {fc}")
    # measured ops signals
    ops = [c for c in ("ttft", "tpot_itl", "e2e_latency", "throughput",
                       "gpu_telemetry", "energy") if c in rec["signals"]]
    ob = min(20, len(ops) * 5)
    s += ob
    reasons.append(f"+{ob} ops {ops}")
    if rec["extract"]["license"] is None or \
            (rec["extract"]["license"] or "").lower() in ("", "none"):
        s -= 8
        reasons.append("-8 no license (redistribution-blocked)")
    return max(0, min(100, s)), "; ".join(reasons)


def economic_relevance(rec: dict) -> str:
    sig = rec["signals"]
    fc = set(FRONTIER_CATEGORIES) & set(sig)
    has_cost = "cost_price" in sig or "energy" in sig
    has_latency = bool({"ttft", "tpot_itl", "e2e_latency", "throughput"} & set(sig))
    if {"cold_start", "migration"} & fc and (has_latency or "gpu_telemetry" in sig):
        return "Very High"   # blocked-but-needed terms with timing to derive penalty
    if {"queueing", "memory_pressure", "serving_stability"} & fc and has_latency:
        return "High"
    if has_latency and ("gpu_telemetry" in sig or has_cost):
        return "Medium"
    if fc or has_latency or has_cost:
        return "Low"
    return "Reject"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-inspect", type=int, default=120)
    p.add_argument("--limit-per-probe", type=int, default=25)
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    already = _load_already_evaluated()
    print(f"already-evaluated pointer set: {len(already)} ids", file=sys.stderr)

    # discover
    discovered: dict[str, list] = {}
    for q in PROBES:
        for ds in search(q, args.limit_per_probe):
            discovered.setdefault(ds, []).append(q)
        time.sleep(0.2)
    # force-inject curated seeds (org-enumerated real infra traces).
    for s in SEED_IDS:
        discovered.setdefault(s, []).append("seed::curated_infra")
    print(f"discovered {len(discovered)} unique ids across {len(PROBES)} probes "
          f"(+{len(SEED_IDS)} seeds)", file=sys.stderr)

    # Inspection priority: strict infra-id match first (excludes NLP noise),
    # then seed membership, then probe-hit count.
    candidates = sorted(
        discovered,
        key=lambda d: (bool(INFRA_ID.search(d)),
                       "seed::curated_infra" in discovered[d],
                       len(discovered[d])),
        reverse=True,
    )

    registry = []
    inspected = 0
    for ds in candidates:
        if inspected >= args.max_inspect:
            break
        meta = card(ds)
        if not meta or "id" not in meta:
            continue
        ex = _extract(meta)
        rd = readme(ds) if ex["gated"] in (False, None) else ""
        text = " ".join([ds, " ".join(ex["features"]), " ".join(ex["data_files"]),
                         " ".join(str(t) for t in ex["tags"]), rd[:40000]])
        signals = detect_signals(text)
        rec = {"dataset_id": ds, "matched_probes": discovered[ds][:8],
               "previously_evaluated": ds in already, "extract": ex,
               "signals": signals}
        rec["forecastability_score"], rec["forecastability_reason"] = \
            forecastability_score(rec)
        rec["economic_relevance"] = economic_relevance(rec)
        rec["url"] = f"https://huggingface.co/datasets/{ds}"
        registry.append(rec)
        inspected += 1
        time.sleep(0.1)

    # field matrix
    matrix = {}
    for r in registry:
        matrix[r["dataset_id"]] = {
            cat: (cat in r["signals"]) for cat in
            (list(SIGNAL_PATTERNS.keys()))
        }

    # priority ranking
    def prio(r):
        econ = {"Very High": 40, "High": 28, "Medium": 16, "Low": 6, "Reject": 0}
        return econ[r["economic_relevance"]] + r["forecastability_score"] * 0.6 \
            + (10 if not r["previously_evaluated"] else 0)
    ranked = sorted(registry, key=prio, reverse=True)

    def top_by(cat, n=10):
        return [r["dataset_id"] for r in ranked
                if cat in r["signals"] and r["economic_relevance"] != "Reject"][:n]

    ranking = {
        "doc_version": "frontier_discovery_v1",
        "production_claim": False, "no_training": True, "no_ingestion": True,
        "total_probes": len(PROBES),
        "total_unique_discovered": len(discovered),
        "total_inspected": len(registry),
        "new_not_previously_evaluated": sum(1 for r in registry
                                            if not r["previously_evaluated"]),
        "top_10_datasets_by_priority": [
            {"dataset_id": r["dataset_id"], "score": round(prio(r), 1),
             "economic_relevance": r["economic_relevance"],
             "forecastability": r["forecastability_score"],
             "frontier_signals": [c for c in FRONTIER_CATEGORIES
                                  if c in r["signals"]],
             "new": not r["previously_evaluated"]}
            for r in ranked[:10]],
        "top_signals_by_category": {cat: top_by(cat, 8)
                                    for cat in FRONTIER_CATEGORIES},
        "datasets_with_cold_start": top_by("cold_start"),
        "datasets_with_migration": top_by("migration"),
        "datasets_with_queueing": top_by("queueing"),
        "datasets_with_memory_pressure": top_by("memory_pressure"),
        "datasets_with_serving_stability": top_by("serving_stability"),
        "datasets_with_autoscaling": top_by("autoscaling"),
    }

    _w("frontier_dataset_registry.json",
       {"doc_version": "frontier_discovery_v1", "production_claim": False,
        "candidates": registry})
    _w("frontier_field_matrix.json",
       {"doc_version": "frontier_discovery_v1", "matrix": matrix,
        "signal_categories": list(SIGNAL_PATTERNS.keys())})
    _w("economic_frontier_priority_ranking.json", ranking)
    # token-leak guard
    tok = os.environ.get("HF_TOKEN") or ""
    if tok:
        for f in OUT_DIR.glob("*.json"):
            assert tok not in f.read_text(), f"HF_TOKEN leaked into {f}"
    print(f"inspected {len(registry)} | new {ranking['new_not_previously_evaluated']}",
          file=sys.stderr)
    return 0


def _w(name, obj):
    with open(OUT_DIR / name, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    print(f"wrote {(OUT_DIR / name).relative_to(REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
