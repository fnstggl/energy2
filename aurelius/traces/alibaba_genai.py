"""Alibaba cluster-trace-v2026-GenAI (GenTD26) multi-layer serving ingester.

A top-down GenAI (stable-diffusion) serving trace across three layers:
application (user requests + e2e latency), middleware (gateway queues), and
infrastructure (container/GPU utilisation + memory). This module ingests **all
primary telemetry layers** and is scrupulous about CROSS-LAYER LINKAGE: it never
fakes a join. Linkage quality is classified per layer-pair from the actual data:

  * ``exact_join``     — a shared request/batch key
  * ``container_join`` — a shared, non-empty ``container_ip`` + overlapping time
  * ``time_join``      — timestamps align within tolerance (same time base)
  * ``no_join``        — usable only independently

Discovered reality (verified): the application layer
(``lora_request_trace.csv``) is in a 2024 wall-clock base with **no
container_ip**, while every metric layer is in a 2022 anonymized epoch with
``container_ip``. So **application ↔ metric layers = no_join** — the request
replay and the metric layers are treated as SEPARATE replay/calibration layers.
The metric layers join to each other by ``container_ip`` (``container_join``).

What that means for the backtest:
  1. a request-level serving replay is built ONLY from the application layer;
  2. the pipeline cold-start latency layers (basemodel/LoRA/ControlNet load) are
     used as a **calibration** prior for the replay's cold-start model — a
     distribution-level calibration, NOT a per-request causal join;
  3. middleware/infra layers are summarised + container-joined for calibration;
  4. no end-to-end request→GPU causality is claimed.

Alibaba public data is a public dataset, **not customer telemetry**.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Optional

from .schema import (
    NormalizedGatewayQueueSample,
    NormalizedGenAIRequest,
    NormalizedInfraSample,
    NormalizedSchedulerPipelineEvent,
    TraceSchemaError,
    percentile,
)

DATASET_NAME = "alibaba_genai"
_BASE_URL = ("https://raw.githubusercontent.com/alibaba/clusterdata/master/"
             "cluster-trace-v2026-GenAI")
REQUEST_FILE = "lora_request_trace.csv"

# Classification of every known file in the GenAI dir. kind drives normalization.
# classification ∈ {primary, derived, metadata, documentation}; layer ∈
# {application, middleware, infrastructure, scheduler, mixed, n/a}.
FILE_REGISTRY: dict[str, dict] = {
    "lora_request_trace.csv": dict(layer="application", classification="primary",
                                   kind="request"),
    "qps.csv": dict(layer="middleware", classification="primary", kind="gateway",
                    field="arrival_rate", unit="rps"),
    "queue_size_raw_anon.csv": dict(layer="middleware", classification="primary",
                                    kind="gateway", field="queue_depth", unit="tasks"),
    "queue_rt_raw_anon.csv": dict(layer="middleware", classification="primary",
                                  kind="gateway", field="waiting_time_s", unit="ms"),
    "pipeline_inference_data_anon.csv": dict(layer="scheduler", classification="primary",
                                             kind="pipeline", stage="pipeline_inference",
                                             unit="ms"),
    "pipeline_update_latency_anon.csv": dict(layer="scheduler", classification="primary",
                                             kind="pipeline", stage="pipeline_update",
                                             unit="ms"),
    "model_predict_data_anon.csv": dict(layer="scheduler", classification="primary",
                                        kind="pipeline", stage="model_predict", unit="ms"),
    "basemodel_update_latency_anon.csv": dict(layer="scheduler", classification="primary",
                                              kind="pipeline", stage="basemodel_load",
                                              unit="ms"),
    "controlnet_latency_data_anon.csv": dict(layer="scheduler", classification="primary",
                                             kind="pipeline", stage="controlnet_load",
                                             unit="ms"),
    "lora_update_latency_anon.csv": dict(layer="scheduler", classification="primary",
                                         kind="pipeline", stage="lora_load", unit="ms"),
    "pod_gpu_duty_cycle_anon.csv": dict(layer="infrastructure", classification="primary",
                                        kind="infra", field="gpu_utilization", unit="pct"),
    "pod_gpu_memory_used_bytes_anon.csv": dict(layer="infrastructure",
                                               classification="primary", kind="infra",
                                               field="gpu_memory_used", unit="bytes"),
    "pod_memory_util_anon.csv": dict(layer="infrastructure", classification="primary",
                                     kind="infra", field="memory_used", unit="frac"),
    # non-telemetry
    "data_trace_processed.csv": dict(layer="mixed", classification="derived", kind="skip"),
    "README.md": dict(layer="n/a", classification="documentation", kind="skip"),
    "MLoRA-Pipeline.png": dict(layer="n/a", classification="documentation", kind="skip"),
    "lora_request_processing.ipynb": dict(layer="n/a", classification="documentation",
                                          kind="skip"),
}

_NULLS = ("", "NULL", "None", "null", "nan", "NaN")


def _f(v) -> Optional[float]:
    if v is None or str(v).strip() in _NULLS:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_request_time(raw: str) -> Optional[float]:
    """Parse the application ``gmt_create`` ``YYYY-MM-DD HH:MM:SS`` → UTC secs."""
    if not raw or str(raw).strip() in _NULLS:
        return None
    try:
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Application layer
# ---------------------------------------------------------------------------

def load_requests(
    path: str,
    *,
    sample_size: Optional[int] = None,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    include_failures: bool = False,
    seed: int = 0,
) -> list[NormalizedGenAIRequest]:
    """Load + normalize ``lora_request_trace.csv`` → NormalizedGenAIRequest."""
    import random

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        for req in ("gmt_create", "exec_time_seconds", "predict_status"):
            if req not in cols:
                raise TraceSchemaError(
                    f"{DATASET_NAME}: request file missing column '{req}'; "
                    f"found {sorted(cols)}")
        reqs = []
        for i, row in enumerate(reader):
            status = (row.get("predict_status") or "").strip() or None
            reqs.append(NormalizedGenAIRequest(
                request_id=f"genai-{i}",
                timestamp_s=parse_request_time(row.get("gmt_create")),
                service_id=(row.get("checkpoint_model_version_id") or None),
                prompt_or_input_size=_f(row.get("prompt_length")),
                output_size=_f(row.get("num_images_per_prompt")),
                e2e_latency_s=_f(row.get("exec_time_seconds")),
                status=status,
                request_type=(row.get("predict_type") or None),
                is_failed=status not in ("SUCCEED", None),
                num_lora=int(_f(row.get("num_lora")) or 0),
                num_inference_steps=_f(row.get("num_inference_steps")),
                group_id=(row.get("groupId") or None),
            ))

    reqs = [r for r in reqs if r.timestamp_s is not None]
    reqs.sort(key=lambda r: (r.timestamp_s, r.request_id))

    if start_s is not None or duration_s is not None:
        base = reqs[0].timestamp_s if reqs else 0.0
        lo = base + start_s if start_s is not None else float("-inf")
        hi = base + (start_s or 0.0) + duration_s if duration_s is not None else float("inf")
        reqs = [r for r in reqs if lo <= r.timestamp_s < hi]
    if not include_failures:
        reqs = [r for r in reqs if not r.is_failed]
    if sample_size is not None and 0 <= sample_size < len(reqs):
        rng = random.Random(seed)
        reqs = rng.sample(reqs, sample_size)
        reqs.sort(key=lambda r: (r.timestamp_s, r.request_id))
    return reqs


# ---------------------------------------------------------------------------
# Metric layers (middleware / scheduler / infra) — tolerant to column order
# ---------------------------------------------------------------------------

def _read_metric_rows(path: str):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        if "timestamp_anon" not in cols or "value" not in cols:
            raise TraceSchemaError(
                f"{DATASET_NAME}: metric file {os.path.basename(path)} missing "
                f"timestamp_anon/value; found {sorted(cols)}")
        for row in reader:
            ts = _f(row.get("timestamp_anon"))
            val = _f(row.get("value"))
            if ts is None:
                continue
            cip = (row.get("container_ip") or "").strip()
            yield ts, val, (cip if cip not in _NULLS else None), row.get("request_type")


def load_gateway(path: str, field: str) -> list[NormalizedGatewayQueueSample]:
    out = []
    for ts, val, cip, rtype in _read_metric_rows(path):
        kw = {field: (val / 1000.0 if field == "waiting_time_s" and val is not None
                      else val)}
        out.append(NormalizedGatewayQueueSample(
            timestamp_s=ts, gateway_id=cip, service_id=rtype, **kw))
    return out


def load_pipeline(path: str, stage: str) -> list[NormalizedSchedulerPipelineEvent]:
    out = []
    for ts, val, cip, _ in _read_metric_rows(path):
        out.append(NormalizedSchedulerPipelineEvent(
            timestamp_s=ts, request_id=None, service_id=None, stage=stage,
            container_id=cip, duration_s=(val / 1000.0 if val is not None else None)))
    return out


def load_infra(path: str, field: str) -> list[NormalizedInfraSample]:
    out = []
    for ts, val, cip, _ in _read_metric_rows(path):
        kw = {field: val}
        out.append(NormalizedInfraSample(
            timestamp_s=ts, node_id=None, container_id=cip, gpu_id=None, **kw))
    return out


# ---------------------------------------------------------------------------
# File discovery + linkage classification
# ---------------------------------------------------------------------------

def discover(source_dir: str) -> dict:
    """Classify every file in ``source_dir`` (present/empty/missing + layer)."""
    report = {"files": {}, "layers_present": set(), "primary_present": [],
              "skipped": [], "missing": [], "empty": []}
    for name, meta in FILE_REGISTRY.items():
        path = os.path.join(source_dir, name)
        entry = dict(meta)
        if not os.path.exists(path):
            entry["status"] = "missing"
            report["missing"].append(name)
        else:
            try:
                with open(path, newline="") as fh:
                    rows = sum(1 for _ in fh) - 1
            except OSError:
                rows = -1
            entry["rows"] = max(0, rows)
            if rows <= 0:
                entry["status"] = "empty"
                report["empty"].append(name)
            else:
                entry["status"] = "present"
                if meta["classification"] == "primary":
                    report["primary_present"].append(name)
                    report["layers_present"].add(meta["layer"])
                else:
                    report["skipped"].append(f"{name} ({meta['classification']})")
        report["files"][name] = entry
    report["layers_present"] = sorted(report["layers_present"])
    return report


def _ts_range(samples) -> tuple:
    ts = [getattr(s, "timestamp_s", None) for s in samples]
    ts = [t for t in ts if t is not None]
    return (min(ts), max(ts)) if ts else (None, None)


def _container_ids(samples) -> set:
    out = set()
    for s in samples:
        cid = getattr(s, "container_id", None) or getattr(s, "gateway_id", None)
        if cid:
            out.add(cid)
    return out


def classify_linkage(layer_a: str, samples_a, layer_b: str, samples_b,
                     *, app_request_layer: bool) -> str:
    """Classify the join quality between two ingested layers FROM THE DATA."""
    if app_request_layer:
        # application layer has no container_ip and a different time base
        return "no_join"
    a0, a1 = _ts_range(samples_a)
    b0, b1 = _ts_range(samples_b)
    time_overlap = (a0 is not None and b0 is not None
                    and min(a1, b1) >= max(a0, b0))
    cids = _container_ids(samples_a) & _container_ids(samples_b)
    if cids and time_overlap:
        return "container_join"
    if time_overlap:
        return "time_join"
    return "no_join"


# ---------------------------------------------------------------------------
# Cold-start calibration from the pipeline layer (distribution medians)
# ---------------------------------------------------------------------------

def calibrate_cold_start(pipeline_by_stage: dict) -> dict:
    """Median load latencies (s) per stage — a DISTRIBUTION calibration for the
    request replay's cold-start model, NOT a per-request join."""
    out = {}
    for stage, events in pipeline_by_stage.items():
        durs = [e.duration_s for e in events if e.duration_s and e.duration_s > 0]
        out[stage] = percentile(durs, 50) if durs else 0.0
    return out


def _pct(vals, q):
    vals = [v for v in vals if v is not None]
    return percentile(vals, q) if vals else None


def summarize(requests, gateway, pipeline, infra) -> dict:
    """Per-layer descriptive stats for the ingest summary."""
    out = {"application": {}, "middleware": {}, "scheduler": {}, "infrastructure": {}}
    if requests:
        ts = [r.timestamp_s for r in requests if r.timestamp_s is not None]
        e2e = [r.e2e_latency_s for r in requests if r.e2e_latency_s is not None]
        models = {}
        types = {}
        for r in requests:
            models[r.service_id or "?"] = models.get(r.service_id or "?", 0) + 1
            types[r.request_type or "?"] = types.get(r.request_type or "?", 0) + 1
        out["application"] = {
            "request_count": len(requests),
            "time_start_s": min(ts) if ts else None,
            "time_end_s": max(ts) if ts else None,
            "duration_s": (max(ts) - min(ts)) if ts else 0.0,
            "distinct_models": len(models),
            "model_top": dict(sorted(models.items(), key=lambda x: -x[1])[:5]),
            "request_type_distribution": dict(sorted(types.items())),
            "failed": sum(1 for r in requests if r.is_failed),
            "lora_request_frac": round(
                sum(1 for r in requests if (r.num_lora or 0) > 0) / len(requests), 4),
            "e2e_latency_s_p50": _pct(e2e, 50), "e2e_latency_s_p95": _pct(e2e, 95),
            "e2e_latency_s_p99": _pct(e2e, 99),
        }
    if gateway:
        out["middleware"] = {
            "samples": len(gateway),
            "queue_depth_p50": _pct([g.queue_depth for g in gateway], 50),
            "queue_depth_p95": _pct([g.queue_depth for g in gateway], 95),
            "waiting_time_s_p50": _pct([g.waiting_time_s for g in gateway], 50),
            "waiting_time_s_p95": _pct([g.waiting_time_s for g in gateway], 95),
            "waiting_time_s_p99": _pct([g.waiting_time_s for g in gateway], 99),
            "arrival_rate_p95": _pct([g.arrival_rate for g in gateway], 95),
        }
    if pipeline:
        by_stage = {}
        for e in pipeline:
            by_stage.setdefault(e.stage, []).append(e.duration_s)
        out["scheduler"] = {
            "events": len(pipeline),
            "stage_duration_s_p50": {k: _pct(v, 50) for k, v in by_stage.items()},
            "stage_duration_s_p95": {k: _pct(v, 95) for k, v in by_stage.items()},
        }
    if infra:
        out["infrastructure"] = {
            "samples": len(infra),
            "gpu_util_pct_p50": _pct([s.gpu_utilization for s in infra], 50),
            "gpu_util_pct_p95": _pct([s.gpu_utilization for s in infra], 95),
            "gpu_mem_used_bytes_p50": _pct([s.gpu_memory_used for s in infra], 50),
            "gpu_mem_used_bytes_p95": _pct([s.gpu_memory_used for s in infra], 95),
            "container_mem_frac_p95": _pct([s.memory_used for s in infra], 95),
            "distinct_containers": len(_container_ids(infra)),
        }
    return out


def load_all_layers(source_dir: str, *, request_kwargs=None) -> dict:
    """Load every present primary-telemetry layer from ``source_dir``."""
    request_kwargs = request_kwargs or {}
    disc = discover(source_dir)
    requests = []
    gateway = []
    pipeline = []
    infra = []
    req_path = os.path.join(source_dir, REQUEST_FILE)
    if os.path.exists(req_path) and disc["files"][REQUEST_FILE]["status"] == "present":
        requests = load_requests(req_path, **request_kwargs)
    for name, meta in FILE_REGISTRY.items():
        if meta["classification"] != "primary" or meta["kind"] == "request":
            continue
        path = os.path.join(source_dir, name)
        if not os.path.exists(path) or disc["files"][name]["status"] != "present":
            continue
        try:
            if meta["kind"] == "gateway":
                gateway += load_gateway(path, meta["field"])
            elif meta["kind"] == "pipeline":
                pipeline += load_pipeline(path, meta["stage"])
            elif meta["kind"] == "infra":
                infra += load_infra(path, meta["field"])
        except TraceSchemaError:
            disc["files"][name]["status"] = "unreadable"
    return {"discovery": disc, "requests": requests, "gateway": gateway,
            "pipeline": pipeline, "infra": infra}
