#!/usr/bin/env python3
"""Frontier Signal Bounded Ingestion v1.

Bounded-ingests the three highest-value Frontier Discovery v2 next-ingest
candidates and normalizes each into a separate canonical trace role:

  A. Mooncake            -> cache_residency_trace
  B. Huawei FaaS 2025    -> cold_start_prior_trace   (calibration-only, FaaS != GPU)
  C. Alibaba GPU v2025   -> autoscaling_queue_proxy_trace (proxy != measured autoscaling)

Binding honesty rules (enforced here and by tests):
  * No production scheduler/scorer/residency/frontier behavior is touched.
  * No real execution, no production-savings claim.
  * Public/artifact data is NEVER treated as pilot telemetry.
  * Raw files live under data/external/frontier_signals/raw/ and are gitignored.
  * Only bounded, derived, NUMERIC normalized samples are committed (no raw
    prompt/response text -- none exists in any source -- and no raw request-id
    hashes for the FaaS source).
  * FaaS cold-start is NOT silently converted into GPU model-load.
  * Alibaba instance-lifecycle is NOT silently converted into measured autoscaling.

The raw bounded download is reproduced by ``download_bounded()`` (only when raw
is absent); see ``data/external/frontier_ingest_v1/source_audit.json`` for the
exact source URLs, licenses and the 100MB/300MB bounded-ingest policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/frontier_signals/raw"
OUT = ROOT / "data/external/frontier_signals"
INGEST = ROOT / "data/external/frontier_ingest_v1"

# Committed-sample caps (mirrors the cache_prefix_reuse_v1 precedent).
SAMPLE_ROWS_CAP = 5000
SAMPLE_BYTES_CAP = 8 * 1024 * 1024  # 8 MiB
HIGH_REUSE_THRESHOLD = 50.0  # pre-registered, matches CACHE_PREFIX_REUSE_FORECASTER_V1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stable_hash(obj) -> str:
    """Short deterministic hash of a python object (no secrets, no PII)."""
    return hashlib.blake2b(repr(obj).encode("utf-8"), digest_size=8).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_bounded_sample(path: Path, rows) -> int:
    """Write at most SAMPLE_ROWS_CAP rows, never exceeding SAMPLE_BYTES_CAP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    nbytes = 0
    with path.open("w") as fh:
        for r in rows[:SAMPLE_ROWS_CAP]:
            line = json.dumps(r, sort_keys=True)
            if nbytes + len(line) + 1 > SAMPLE_BYTES_CAP:
                break
            fh.write(line + "\n")
            nbytes += len(line) + 1
            written += 1
    return written


def write_full(path: Path, rows) -> None:
    """Full normalized rows -> gitignored analysis_sample.jsonl (for the ML phase)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def quantiles(vals, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)):
    xs = sorted(v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not xs:
        return {}
    out = {}
    n = len(xs)
    for q in qs:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        out[f"p{int(q*100)}"] = round(float(xs[idx]), 6)
    out["mean"] = round(sum(xs) / n, 6)
    out["count"] = n
    return out


def coverage(rows, field) -> float:
    if not rows:
        return 0.0
    present = sum(1 for r in rows if r.get(field) is not None)
    return round(present / len(rows), 4)


# ---------------------------------------------------------------------------
# bounded download (only invoked when raw is absent; documents reproducibility)
# ---------------------------------------------------------------------------
def _curl(url: str, dest: Path, rng: str | None = None, timeout: int = 240) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sSL", "--max-time", str(timeout)]
    if rng:
        cmd += ["-r", rng]
    cmd += [url, "-o", str(dest)]
    subprocess.run(cmd, check=True)


def download_bounded() -> None:
    base = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main"
    mc = RAW / "mooncake"
    for f in ("conversation_trace.jsonl", "synthetic_trace.jsonl", "toolagent_trace.jsonl"):
        if not (mc / f).exists():
            _curl(f"{base}/FAST25-release/traces/{f}", mc / f)
    if not (mc / "arxiv_mooncake_trace.jsonl").exists():
        _curl(f"{base}/FAST25-release/arxiv-trace/mooncake_trace.jsonl", mc / "arxiv_mooncake_trace.jsonl")

    al = RAW / "alibaba_gpu_v2025" / "disaggregated_DLRM_trace.csv"
    if not al.exists():
        _curl("https://raw.githubusercontent.com/alibaba/clusterdata/master/"
              "cluster-trace-gpu-v2025/disaggregated_DLRM_trace.csv", al)

    hw = RAW / "huawei_faas_2025"
    trig = hw / "R2_funcID_runtime_triggerType.csv"
    if not trig.exists():
        _curl("https://drive.usercontent.google.com/download?id=1dUg_yxeLR5OyldivPZGW5djYF31166EG"
              "&export=download&confirm=t", trig)
    recov = hw / "R1_coldstart_first100mb.recovered.csv"
    if not recov.exists():
        part = hw / "R1_coldstart_first100mb.zip.part"
        if not part.exists():
            _curl("https://drive.usercontent.google.com/download?id=1mMQtfZNtg-EPmGmGYuOzC5KPbZoXRd8e"
                  "&export=download&confirm=t", part, rng="0-104857600")
        _recover_first_zip_member(part, recov)


def _recover_first_zip_member(part: Path, dest: Path) -> int:
    """Raw-inflate the first fully-contained DEFLATE member of a partial zip."""
    data = part.read_bytes()
    offs = []
    i = 0
    while True:
        j = data.find(b"PK\x03\x04", i)
        if j == -1:
            break
        offs.append(j)
        i = j + 4
    for o in offs:
        method, = struct.unpack("<H", data[o + 8:o + 10])
        fnlen, = struct.unpack("<H", data[o + 26:o + 28])
        exlen, = struct.unpack("<H", data[o + 28:o + 30])
        fname = data[o + 30:o + 30 + fnlen].decode("utf-8", "replace")
        if fname.endswith("/") or method != 8:
            continue
        start = o + 30 + fnlen + exlen
        d = zlib.decompressobj(-15)
        out = bytearray()
        try:
            out += d.decompress(data[start:])
            out += d.flush()
        except zlib.error:
            pass  # expected truncation of the bounded slice
        lines = out.decode("utf-8", "replace").split("\n")[:-1]  # drop partial tail row
        dest.write_text("\n".join(lines) + "\n")
        return len(lines)
    return 0


# ---------------------------------------------------------------------------
# A. Mooncake -> cache_residency_trace
# ---------------------------------------------------------------------------
def normalize_mooncake():
    src = RAW / "mooncake"
    files = {
        "conversation": src / "conversation_trace.jsonl",
        "synthetic": src / "synthetic_trace.jsonl",
        "toolagent": src / "toolagent_trace.jsonl",
        "arxiv": src / "arxiv_mooncake_trace.jsonl",
    }
    rows = []
    for trace_name, path in files.items():
        if not path.exists():
            continue
        seen_blocks: set = set()         # global prefix-cache simulation (infinite cache)
        prefix_seen: Counter = Counter()  # rolling first-block seen count
        running_reuse = []
        for idx, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            hids = d.get("hash_ids") or []
            n = len(hids)
            reused = sum(1 for h in hids if h in seen_blocks)
            reuse_pct = (100.0 * reused / n) if n else 0.0
            first_block = hids[0] if hids else None
            prefix_group = stable_hash(("mc", trace_name, first_block))
            hids_hash = stable_hash(("mc", trace_name, tuple(hids)))
            rolling_prefix_seen = prefix_seen[first_block] if first_block is not None else 0
            rolling_reuse_mean = (sum(running_reuse) / len(running_reuse)) if running_reuse else 0.0
            rows.append({
                "trace_role": "cache_residency_trace",
                "source": "mooncake",
                "trace_name": trace_name,
                "request_id": f"mooncake:{trace_name}:{idx}",     # DERIVED index (no native id)
                "session_id": None,                                # not available in source
                "model_id": None,                                  # not available in source
                "timestamp_ms": d.get("timestamp"),
                "sequence_index": idx,
                "input_length": d.get("input_length"),
                "output_length": d.get("output_length"),
                "hash_ids_count": n,
                "hash_ids_hash": hids_hash,                        # derived; full list NOT committed
                "prefix_hash": stable_hash(("mc", trace_name, first_block)),
                "prefix_group": prefix_group,
                "repeated_prefix_or_hash": bool(rolling_prefix_seen > 0),
                "cache_reuse_pct": round(reuse_pct, 4),            # DERIVED proxy (global prefix cache)
                "high_reuse": bool(reuse_pct >= HIGH_REUSE_THRESHOLD),
                "cache_hit": None,                                 # NOT measured
                "cache_hit_proxy": 1.0 if reused > 0 else 0.0,     # DERIVED request-level hit proxy
                "rolling_prefix_seen_count": rolling_prefix_seen,  # decision-time-safe feature
                "rolling_reuse_mean": round(rolling_reuse_mean, 4),
                "field_quality": {
                    "timestamp_ms": "measured_anonymized",
                    "input_length": "measured_anonymized",
                    "output_length": "measured_anonymized",
                    "hash_ids_count": "measured_anonymized",
                    "cache_reuse_pct": "derived_proxy_global_prefix_cache",
                    "high_reuse": "derived_proxy",
                    "cache_hit": "not_measured",
                    "cache_hit_proxy": "derived_proxy",
                },
                "limitations": "No measured reuse/cache-hit label; reuse is a global-prefix-cache "
                               "SIMULATION from hash_ids. No model_id/session_id. NOT identical to "
                               "SwissAI measured reuse_percentage (bucket overlap).",
            })
            if first_block is not None:
                prefix_seen[first_block] += 1
            seen_blocks.update(hids)
            running_reuse.append(reuse_pct)
            if len(running_reuse) > 2000:
                running_reuse.pop(0)

    # processed outputs
    pdir = OUT / "mooncake" / "processed"
    reuse_vals = [r["cache_reuse_pct"] for r in rows]
    schema_profile = {
        "source": "mooncake",
        "trace_role": "cache_residency_trace",
        "raw_files": {k: (str(v.relative_to(ROOT)) if v.exists() else "absent") for k, v in files.items()},
        "raw_schema": {"timestamp": "int ms", "input_length": "int tokens",
                       "output_length": "int tokens", "hash_ids": "list[int] (block_size=512)"},
        "normalized_rows": len(rows),
        "per_trace_rows": dict(Counter(r["trace_name"] for r in rows)),
        "field_presence": {f: coverage(rows, f) for f in
                           ["timestamp_ms", "input_length", "output_length", "hash_ids_count",
                            "cache_reuse_pct", "cache_hit", "model_id", "session_id"]},
        "license": "Mooncake repo (citation requested); derived numeric sample only.",
    }
    schema_mapping = {
        "source_field -> canonical_field": {
            "timestamp": "timestamp_ms (measured_anonymized)",
            "input_length": "input_length (measured)",
            "output_length": "output_length (measured)",
            "hash_ids": "hash_ids_count + hash_ids_hash + prefix_group (derived)",
            "(derived from hash_ids overlap)": "cache_reuse_pct / high_reuse / cache_hit_proxy (DERIVED proxy)",
            "(absent)": "model_id=None, session_id=None, cache_hit=None (NOT in source)",
        },
        "target_for_ml": {"continuous": "cache_reuse_pct", "binary": "high_reuse"},
        "label_class": "derived_proxy (NOT measured); NOT identical to SwissAI reuse_percentage",
    }
    summary = {
        "source": "mooncake", "trace_role": "cache_residency_trace",
        "normalized_rows": len(rows),
        "unique_prefix_groups": len({r["prefix_group"] for r in rows}),
        "high_reuse_rate": round(sum(r["high_reuse"] for r in rows) / max(1, len(rows)), 4),
        "mean_cache_reuse_pct": round(sum(reuse_vals) / max(1, len(reuse_vals)), 4),
        "any_reuse_rate": round(sum(r["cache_hit_proxy"] for r in rows) / max(1, len(rows)), 4),
        "measured_vs_proxy": {"reuse_label": "derived_proxy", "timestamps_tokens_blocks": "measured_anonymized"},
        "bounded_ingest": {"fully_ingested": True, "cap_mb": 100, "raw_bytes_under_cap": True},
        "limitations": schema_profile.get("license"),
    }
    rollups = {
        "cache_reuse_pct_quantiles": quantiles(reuse_vals),
        "hash_ids_count_quantiles": quantiles([r["hash_ids_count"] for r in rows]),
        "input_length_quantiles": quantiles([r["input_length"] for r in rows]),
        "output_length_quantiles": quantiles([r["output_length"] for r in rows]),
        "per_trace": {
            t: {"rows": c,
                "high_reuse_rate": round(
                    sum(r["high_reuse"] for r in rows if r["trace_name"] == t) / max(1, c), 4)}
            for t, c in Counter(r["trace_name"] for r in rows).items()
        },
    }
    write_json(pdir / "schema_profile.json", schema_profile)
    write_json(pdir / "schema_mapping.json", schema_mapping)
    write_json(pdir / "summary.json", summary)
    write_json(pdir / "statistical_rollups.json", rollups)
    sample_cols = ["trace_role", "source", "trace_name", "request_id", "timestamp_ms",
                   "sequence_index", "input_length", "output_length", "hash_ids_count",
                   "hash_ids_hash", "prefix_group", "repeated_prefix_or_hash", "cache_reuse_pct",
                   "high_reuse", "cache_hit", "cache_hit_proxy", "rolling_prefix_seen_count",
                   "rolling_reuse_mean", "field_quality", "limitations"]
    written = write_bounded_sample(pdir / "normalized_sample.jsonl",
                                   [{k: r[k] for k in sample_cols} for r in rows])
    write_full(pdir / "analysis_sample.jsonl", rows)  # gitignored, for ML phase
    summary["committed_sample_rows"] = written
    write_json(pdir / "summary.json", summary)
    print(f"[mooncake] normalized {len(rows)} rows; committed sample {written}")
    return rows


# ---------------------------------------------------------------------------
# B. Huawei FaaS 2025 -> cold_start_prior_trace (calibration-only)
# ---------------------------------------------------------------------------
def _parse_pool(pod_id: str):
    # pool22-300-128-0004326639 -> (pool22-300-128, cpu_limit=300, mem_limit=128)
    parts = pod_id.split("-")
    if len(parts) >= 4:
        pool = "-".join(parts[:3])
        try:
            return pool, int(parts[1]), int(parts[2])
        except ValueError:
            return pool, None, None
    return pod_id, None, None


def normalize_huawei():
    import csv
    src = RAW / "huawei_faas_2025"
    cs_path = src / "R1_coldstart_first100mb.recovered.csv"
    trig_path = src / "R2_funcID_runtime_triggerType.csv"
    rows = []
    if cs_path.exists():
        with cs_path.open() as fh:
            for d in csv.DictReader(fh):
                try:
                    pool, cpu_lim, mem_lim = _parse_pool(d.get("podID", ""))
                    rows.append({
                        "trace_role": "cold_start_prior_trace",
                        "source": "huawei_faas_2025",
                        "function_id": int(d["funcName"]),         # int function name (anonymized)
                        "app_id": f"cluster{d['clusterName']}:func{d['funcName']}",
                        "cluster_name": int(d["clusterName"]),
                        "cold_start_stage": "decomposed",          # one row carries all stages
                        "timestamp_s": float(d["time"]),
                        "day": int(d["day"]),
                        "pod_allocation_s": float(d["podAllocationCost"]),
                        "deploy_code_s": float(d["deployCodeCost"]),
                        "deploy_dependency_s": float(d["deployDependencyCost"]),
                        "scheduling_s": float(d["schedulingCost"]),
                        "cold_start_latency_s": float(d["totalCost_cold_start"]),
                        "startup_latency_s": float(d["totalCost_cold_start"]),
                        "pool_name": pool,
                        "pool_cpu_limit_millicores": cpu_lim,
                        "pool_mem_limit_mb": mem_lim,
                        "platform": "huawei_yuanrong_faas",
                        "runtime": None,   # region mismatch: trigger CSV is Region 2, cold-starts Region 1
                        "field_quality": {
                            "cold_start_latency_s": "measured_faas",
                            "pod_allocation_s": "measured_faas",
                            "deploy_code_s": "measured_faas",
                            "deploy_dependency_s": "measured_faas",
                            "scheduling_s": "measured_faas",
                            "for_gpu_llm_cold_start": "prior_proxy_only",
                        },
                        "limitation": "MEASURED FaaS (CPU pod) cold-start, NOT server-class GPU "
                                      "model-load. deploy_code/deploy_dependency are code/dependency "
                                      "download, not model-weight load. Calibration-only prior.",
                    })
                except (KeyError, ValueError):
                    continue

    # separate small trigger/runtime descriptive table (different region; not joined)
    trig = []
    if trig_path.exists():
        with trig_path.open() as fh:
            for d in csv.DictReader(fh):
                trig.append({
                    "funcID_hash": stable_hash(d.get("funcID", "")),  # do not commit raw funcID
                    "cpu_request_millicores": int(d["cpu_request"]) if d.get("cpu_request", "").isdigit() else None,
                    "runtime": d.get("runtime"),
                    "trigger_type": d.get("triggerType-invocationType"),
                })

    pdir = OUT / "huawei_faas_2025" / "processed"
    cs_lat = [r["cold_start_latency_s"] for r in rows]
    schema_profile = {
        "source": "huawei_faas_2025", "trace_role": "cold_start_prior_trace",
        "raw_files": {
            "cold_start_events": str(cs_path.relative_to(ROOT)) if cs_path.exists() else "absent",
            "trigger_runtime": str(trig_path.relative_to(ROOT)) if trig_path.exists() else "absent",
        },
        "raw_schema_cold_start": ["day", "time", "clusterName", "funcName", "userID", "requestID(hash)",
                                  "totalCost_cold_start", "podAllocationCost", "deployCodeCost",
                                  "deployDependencyCost", "schedulingCost", "podID"],
        "normalized_rows": len(rows),
        "trigger_runtime_rows": len(trig),
        "bounded_ingest": "100MB range-download of R1.zip (467MB, over cap) -> raw-inflate of first "
                          "fully-contained member R1/day_28.csv. Full file NOT downloaded.",
        "license": "CC BY 4.0 (attribution required). requestID/podID hashed by authors; requestID "
                   "hash omitted from committed sample.",
        "field_presence": {f: coverage(rows, f) for f in
                           ["cold_start_latency_s", "pod_allocation_s", "deploy_code_s",
                            "deploy_dependency_s", "scheduling_s", "runtime"]},
    }
    schema_mapping = {
        "source_field -> canonical_field": {
            "totalCost_cold_start": "cold_start_latency_s / startup_latency_s (measured_faas)",
            "podAllocationCost": "pod_allocation_s (measured_faas)",
            "deployCodeCost": "deploy_code_s (measured_faas)",
            "deployDependencyCost": "deploy_dependency_s (measured_faas)",
            "schedulingCost": "scheduling_s (measured_faas)",
            "funcName": "function_id", "clusterName": "cluster_name", "time": "timestamp_s",
            "podID": "pool_name + pool_cpu_limit_millicores + pool_mem_limit_mb",
            "requestID": "DROPPED (hash, not committed)",
        },
        "gpu_llm_usage": "prior_proxy_only -- field_quality.for_gpu_llm_cold_start='prior_proxy_only'",
        "limitation": "NOT server-class GPU model-load telemetry.",
    }
    rollups = {
        "cold_start_latency_s_quantiles": quantiles(cs_lat),
        "pod_allocation_s_quantiles": quantiles([r["pod_allocation_s"] for r in rows]),
        "deploy_code_s_quantiles": quantiles([r["deploy_code_s"] for r in rows]),
        "deploy_dependency_s_quantiles": quantiles([r["deploy_dependency_s"] for r in rows]),
        "scheduling_s_quantiles": quantiles([r["scheduling_s"] for r in rows]),
        "stage_share_of_total_mean": _stage_shares(rows),
        "runtime_distribution_trigger_table": dict(Counter(t["runtime"] for t in trig)),
        "trigger_type_distribution": dict(Counter(t["trigger_type"] for t in trig)),
    }
    summary = {
        "source": "huawei_faas_2025", "trace_role": "cold_start_prior_trace",
        "normalized_rows": len(rows), "trigger_runtime_rows": len(trig),
        "unique_functions": len({r["function_id"] for r in rows}),
        "days_covered": sorted({r["day"] for r in rows}),
        "measured_vs_proxy": {"faas_cold_start_cost": "measured", "gpu_llm_cold_start": "prior_proxy_only"},
        "calibration_only": True,
        "is_gpu_model_load": False,
        "mean_cold_start_latency_s": round(sum(cs_lat) / max(1, len(cs_lat)), 6),
        "limitation": "FaaS cold-start (CPU pod). NEVER GPU model-load truth. Simulator-prior "
                      "calibration only.",
    }
    write_json(pdir / "schema_profile.json", schema_profile)
    write_json(pdir / "schema_mapping.json", schema_mapping)
    write_json(pdir / "statistical_rollups.json", rollups)
    sample_cols = ["trace_role", "source", "function_id", "cluster_name", "cold_start_stage",
                   "timestamp_s", "day", "pod_allocation_s", "deploy_code_s", "deploy_dependency_s",
                   "scheduling_s", "cold_start_latency_s", "startup_latency_s", "pool_name",
                   "pool_cpu_limit_millicores", "pool_mem_limit_mb", "platform", "runtime",
                   "field_quality", "limitation"]
    written = write_bounded_sample(pdir / "normalized_sample.jsonl",
                                   [{k: r[k] for k in sample_cols} for r in rows])
    write_full(pdir / "analysis_sample.jsonl", rows)  # gitignored
    summary["committed_sample_rows"] = written
    write_json(pdir / "summary.json", summary)
    print(f"[huawei] normalized {len(rows)} cold-start rows (+{len(trig)} trigger); committed {written}")
    return rows


def _stage_shares(rows):
    if not rows:
        return {}
    acc = defaultdict(float)
    n = 0
    for r in rows:
        tot = r["cold_start_latency_s"]
        if tot and tot > 0:
            acc["pod_allocation"] += r["pod_allocation_s"] / tot
            acc["deploy_code"] += r["deploy_code_s"] / tot
            acc["deploy_dependency"] += r["deploy_dependency_s"] / tot
            acc["scheduling"] += r["scheduling_s"] / tot
            n += 1
    return {k: round(v / n, 4) for k, v in acc.items()} if n else {}


# ---------------------------------------------------------------------------
# C. Alibaba GPU v2025 -> autoscaling_queue_proxy_trace (proxy)
# ---------------------------------------------------------------------------
def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def normalize_alibaba():
    import csv
    path = RAW / "alibaba_gpu_v2025" / "disaggregated_DLRM_trace.csv"
    rows = []
    if path.exists():
        with path.open() as fh:
            for d in csv.DictReader(fh):
                ct, st, dt = _f(d.get("creation_time")), _f(d.get("scheduled_time")), _f(d.get("deletion_time"))
                sched_delay = (st - ct) if (st is not None and ct is not None) else None
                lifetime = (dt - ct) if (dt is not None and ct is not None) else None
                rows.append({
                    "trace_role": "autoscaling_queue_proxy_trace",
                    "source": "alibaba_gpu_v2025",
                    "instance_id": d.get("instance_sn"),
                    "app_name": d.get("app_name"),
                    "role": d.get("role"),                       # CN / HN
                    "gpu_type": None,                            # not specified in trace
                    "gpu_count": _f(d.get("gpu_request")),
                    "cpu_count": _f(d.get("cpu_request")),
                    "memory_gib": _f(d.get("memory_request")),
                    "rdma_pct": _f(d.get("rdma_request")),
                    "max_instance_per_node": _f(d.get("max_instance_per_node")),
                    "creation_time_s": ct,
                    "scheduled_time_s": st,
                    "deletion_time_s": dt,
                    "scheduler_delay_s": sched_delay,            # DERIVED proxy (NOT serving queue wait)
                    "queue_wait_s": None,                        # NOT in source (no per-request queue)
                    "instance_lifetime_s": lifetime,
                    "utilization": None,                         # NOT in source (allocation only)
                    "failure_or_timeout_state": None,            # NOT in source
                    "is_gpu_instance": (d.get("role") == "HN"),
                    "field_quality": {
                        "creation_time_s": "measured", "scheduled_time_s": "measured",
                        "deletion_time_s": "measured", "gpu_count": "measured_allocation",
                        "scheduler_delay_s": "derived_proxy",
                        "queue_wait_s": "not_in_source", "utilization": "not_in_source",
                        "failure_or_timeout_state": "not_in_source",
                        "autoscaling_event": "not_in_source (inferred from per-app create/delete only)",
                    },
                    "limitation": "Instance-lifecycle allocation record, NOT per-request. No measured "
                                  "serving queue_wait, utilization, or failure. scheduler_delay_s is a "
                                  "scheduling proxy. Autoscaling is INFERRED, not a measured event.",
                })

    # autoscaling proxy: per-app instance create/delete events binned over time
    BIN = 3600.0  # 1-hour bins
    create_events = [(r["creation_time_s"], r["app_name"]) for r in rows if r["creation_time_s"] is not None]
    delete_events = [(r["deletion_time_s"], r["app_name"]) for r in rows if r["deletion_time_s"] is not None]
    per_app_counts = Counter(r["app_name"] for r in rows)

    pdir = OUT / "alibaba_gpu_v2025" / "processed"
    sched = [r["scheduler_delay_s"] for r in rows if r["scheduler_delay_s"] is not None]
    life = [r["instance_lifetime_s"] for r in rows if r["instance_lifetime_s"] is not None]
    schema_profile = {
        "source": "alibaba_gpu_v2025", "trace_role": "autoscaling_queue_proxy_trace",
        "raw_file": str(path.relative_to(ROOT)) if path.exists() else "absent",
        "raw_schema": ["instance_sn", "role(CN/HN)", "app_name", "cpu_request/limit", "gpu_request/limit",
                       "rdma_request/limit", "memory_request/limit", "disk_request/limit",
                       "max_instance_per_node", "creation_time", "scheduled_time", "deletion_time"],
        "normalized_rows": len(rows),
        "field_presence": {f: coverage(rows, f) for f in
                           ["creation_time_s", "scheduled_time_s", "deletion_time_s",
                            "scheduler_delay_s", "instance_lifetime_s", "gpu_count",
                            "queue_wait_s", "utilization", "failure_or_timeout_state"]},
        "license": "Alibaba clusterdata terms (citation required). Numeric/hashed only.",
    }
    schema_mapping = {
        "source_field -> canonical_field": {
            "creation_time": "creation_time_s (measured)",
            "scheduled_time": "scheduled_time_s (measured)",
            "deletion_time": "deletion_time_s (measured)",
            "(scheduled_time - creation_time)": "scheduler_delay_s (DERIVED proxy)",
            "(deletion_time - creation_time)": "instance_lifetime_s (DERIVED)",
            "gpu_request": "gpu_count (measured ALLOCATION, not utilization)",
            "role": "role / is_gpu_instance",
            "(absent)": "queue_wait_s=None, utilization=None, failure=None, autoscaling_event=None",
        },
        "label_class": "proxy (instance-lifecycle); NOT measured serving autoscaling or queue-wait",
    }
    rollups = {
        "scheduler_delay_s_quantiles": quantiles(sched),
        "instance_lifetime_s_quantiles": quantiles(life),
        "gpu_count_distribution": dict(Counter(r["gpu_count"] for r in rows)),
        "role_distribution": dict(Counter(r["role"] for r in rows)),
        "instances_per_app_quantiles": quantiles(list(per_app_counts.values())),
        "autoscaling_proxy": {
            "method": "per-app instance create/delete counts in 1h bins (INFERRED, not measured)",
            "total_create_events": len(create_events),
            "total_delete_events": len(delete_events),
            "n_apps": len(per_app_counts),
            "n_gpu_instances": sum(1 for r in rows if r["is_gpu_instance"]),
        },
    }
    summary = {
        "source": "alibaba_gpu_v2025", "trace_role": "autoscaling_queue_proxy_trace",
        "normalized_rows": len(rows),
        "n_apps": len(per_app_counts),
        "n_gpu_instances": sum(1 for r in rows if r["is_gpu_instance"]),
        "scheduler_delay_coverage": coverage(rows, "scheduler_delay_s"),
        "measured_vs_proxy": {"timestamps_allocations": "measured", "queue_autoscaling": "proxy_inferred",
                              "utilization_failure": "not_in_source"},
        "has_measured_serving_autoscaling": False,
        "has_per_request_queue_wait": False,
        "limitation": "Proxy only -- no inference autoscaling events, no per-request queue, no utilization.",
    }
    write_json(pdir / "schema_profile.json", schema_profile)
    write_json(pdir / "schema_mapping.json", schema_mapping)
    write_json(pdir / "statistical_rollups.json", rollups)
    sample_cols = ["trace_role", "source", "instance_id", "app_name", "role", "gpu_type", "gpu_count",
                   "cpu_count", "memory_gib", "rdma_pct", "max_instance_per_node", "creation_time_s",
                   "scheduled_time_s", "deletion_time_s", "scheduler_delay_s", "queue_wait_s",
                   "instance_lifetime_s", "utilization", "failure_or_timeout_state", "is_gpu_instance",
                   "field_quality", "limitation"]
    written = write_bounded_sample(pdir / "normalized_sample.jsonl",
                                   [{k: r[k] for k in sample_cols} for r in rows])
    write_full(pdir / "analysis_sample.jsonl", rows)  # gitignored
    summary["committed_sample_rows"] = written
    write_json(pdir / "summary.json", summary)
    print(f"[alibaba] normalized {len(rows)} instance rows; committed {written}")
    return rows


def main():
    if "--download" in sys.argv:
        download_bounded()
    mc = normalize_mooncake()
    hw = normalize_huawei()
    al = normalize_alibaba()
    manifest = {
        "doc_version": "frontier_ingest_v1",
        "sources": {
            "mooncake": {"rows": len(mc), "role": "cache_residency_trace"},
            "huawei_faas_2025": {"rows": len(hw), "role": "cold_start_prior_trace"},
            "alibaba_gpu_v2025": {"rows": len(al), "role": "autoscaling_queue_proxy_trace"},
        },
        "sample_caps": {"rows": SAMPLE_ROWS_CAP, "bytes": SAMPLE_BYTES_CAP},
        "no_production_behavior_change": True,
        "no_production_savings_claim": True,
        "public_data_is_not_pilot_telemetry": True,
    }
    write_json(INGEST / "ingest_manifest.json", manifest)
    print("ingest complete:", json.dumps(manifest["sources"]))


if __name__ == "__main__":
    main()
