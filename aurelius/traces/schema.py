"""Shared normalized schema for public LLM-serving / cluster trace ingestion.

This module defines the dataset-agnostic interface that every public-trace
ingester in Aurelius normalizes into. Only **BurstGPT** is implemented today
(``aurelius/traces/burstgpt.py``); the schema is intentionally shaped so the
future datasets named in ``docs/PUBLIC_TRACE_BACKTESTS.md`` (Azure LLM/LMM,
Alibaba GPU, Philly, MIT Supercloud) can normalize into the **same**
``NormalizedLLMRequest`` without changing downstream replay / backtest code.

Design rules (consistent with ``docs/RESULTS.md`` and the energy backtest):

- Pure, deterministic, stdlib-only (``csv`` / ``statistics`` / ``math``). No
  pandas / numpy dependency, no network here (download lives in the ingestion
  script), no global state.
- The normalized record is the contract. Ingesters map their raw columns onto
  it; the replay / backtest layers only ever see ``NormalizedLLMRequest``.
- Nothing in this module is a production claim. A trace is replayed serving
  traffic, **not** customer telemetry, and a derived ``cache_affinity_key`` is
  an honest *proxy* for prefix/session locality — it is **not** a measured KV
  cache hit rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

# Canonical log-type labels (BurstGPT uses exactly these two strings).
LOG_TYPE_CONVERSATION = "Conversation log"
LOG_TYPE_API = "API log"


class TraceSchemaError(ValueError):
    """Raised when a raw trace is missing required columns or has bad values."""


@dataclass(frozen=True)
class NormalizedLLMRequest:
    """One normalized LLM-serving request — the cross-dataset contract.

    Fields map 1:1 to the mission spec. Optional fields (``session_id``,
    ``elapsed_s``) are ``None`` when the source dataset does not provide them
    (e.g. the published ``BurstGPT_1.csv`` carries neither a Session ID nor an
    Elapsed-time column — see ``aurelius/traces/burstgpt.py``).

    ``elapsed_s`` — when present — is the source's *end-to-end* final response
    time. It is **not** TTFT and must never be reported as a measured TTFT.
    """

    request_id: str
    timestamp_s: float
    session_id: Optional[str]
    model: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_s: Optional[float]
    log_type: str
    is_failure: bool
    # Proxy for prefix/session locality (NOT a measured KV hit rate). ``None``
    # when the source has no session/prefix/logical-stream signal at all (e.g.
    # the Azure LLM inference trace), in which case the replay applies NO cache
    # affinity benefit — see ``aurelius/traces/replay.py``.
    cache_affinity_key: Optional[str]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "timestamp_s": self.timestamp_s,
            "session_id": self.session_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_s": self.elapsed_s,
            "log_type": self.log_type,
            "is_failure": self.is_failure,
            "cache_affinity_key": self.cache_affinity_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedLLMRequest":
        return cls(
            request_id=str(d["request_id"]),
            timestamp_s=float(d["timestamp_s"]),
            session_id=(None if d.get("session_id") in (None, "") else str(d["session_id"])),
            model=str(d["model"]),
            prompt_tokens=int(d["prompt_tokens"]),
            output_tokens=int(d["output_tokens"]),
            total_tokens=int(d["total_tokens"]),
            elapsed_s=(None if d.get("elapsed_s") in (None, "") else float(d["elapsed_s"])),
            log_type=str(d["log_type"]),
            is_failure=bool(d["is_failure"]),
            cache_affinity_key=(
                None if d.get("cache_affinity_key") in (None, "")
                else str(d["cache_affinity_key"])
            ),
        )


@runtime_checkable
class TraceSource(Protocol):
    """Interface every dataset ingester implements.

    Only ``BurstGPTSource`` implements this today. Future datasets
    (Azure LLM/LMM, Alibaba GPU, Philly, MIT Supercloud) plug in by
    implementing the same three members so the replay / backtest layers stay
    dataset-agnostic.
    """

    name: str
    required_columns: Sequence[str]
    default_source_url: str

    def normalize(self, rows: Iterable[dict]) -> list[NormalizedLLMRequest]:
        """Map raw CSV ``DictReader`` rows onto ``NormalizedLLMRequest``."""
        ...


# ---------------------------------------------------------------------------
# Validation helpers (shared by all ingesters)
# ---------------------------------------------------------------------------

def validate_columns(
    header: Optional[Sequence[str]],
    required: Sequence[str],
    dataset_name: str,
) -> None:
    """Raise ``TraceSchemaError`` unless every required column is present.

    Column matching is exact (BurstGPT headers are well defined). This is the
    guard tests rely on to catch a malformed / wrong-dataset CSV early.
    """
    if not header:
        raise TraceSchemaError(f"{dataset_name}: empty/missing CSV header row")
    present = set(header)
    missing = [c for c in required if c not in present]
    if missing:
        raise TraceSchemaError(
            f"{dataset_name}: missing required column(s) {missing}; "
            f"found header {list(header)}"
        )


# ---------------------------------------------------------------------------
# Percentile + summary helpers (stdlib only, deterministic)
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation surprises).

    ``pct`` in [0, 100]. Empty input returns 0.0.
    """
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    # Nearest-rank: rank = ceil(pct/100 * n), 1-indexed.
    rank = math.ceil((pct / 100.0) * len(ordered))
    rank = max(1, min(len(ordered), rank))
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class TraceSummary:
    """Dataset-agnostic descriptive stats over a normalized request list."""

    dataset: str
    row_count: int
    included_count: int
    time_start_s: float
    time_end_s: float
    duration_s: float
    model_distribution: dict
    log_type_distribution: dict
    failure_rate_pct: float
    prompt_tokens_p50: float
    prompt_tokens_p95: float
    prompt_tokens_p99: float
    output_tokens_p50: float
    output_tokens_p95: float
    output_tokens_p99: float
    total_tokens_p50: float
    total_tokens_p95: float
    total_tokens_p99: float
    rps_mean_per_min: float
    rps_p95_per_min: float
    rps_max_per_min: float
    distinct_cache_keys: int
    cache_key_reuse_rate_pct: float
    mean_requests_per_cache_key: float
    has_session_ids: bool
    has_elapsed: bool
    has_cache_affinity: bool = False

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "row_count": self.row_count,
            "included_count": self.included_count,
            "time_start_s": self.time_start_s,
            "time_end_s": self.time_end_s,
            "duration_s": self.duration_s,
            "model_distribution": self.model_distribution,
            "log_type_distribution": self.log_type_distribution,
            "failure_rate_pct": round(self.failure_rate_pct, 4),
            "prompt_tokens_p50": self.prompt_tokens_p50,
            "prompt_tokens_p95": self.prompt_tokens_p95,
            "prompt_tokens_p99": self.prompt_tokens_p99,
            "output_tokens_p50": self.output_tokens_p50,
            "output_tokens_p95": self.output_tokens_p95,
            "output_tokens_p99": self.output_tokens_p99,
            "total_tokens_p50": self.total_tokens_p50,
            "total_tokens_p95": self.total_tokens_p95,
            "total_tokens_p99": self.total_tokens_p99,
            "rps_mean_per_min": round(self.rps_mean_per_min, 6),
            "rps_p95_per_min": round(self.rps_p95_per_min, 6),
            "rps_max_per_min": round(self.rps_max_per_min, 6),
            "distinct_cache_keys": self.distinct_cache_keys,
            "cache_key_reuse_rate_pct": round(self.cache_key_reuse_rate_pct, 4),
            "mean_requests_per_cache_key": round(self.mean_requests_per_cache_key, 4),
            "has_session_ids": self.has_session_ids,
            "has_elapsed": self.has_elapsed,
            "has_cache_affinity": self.has_cache_affinity,
        }


def time_rescale(
    requests: Sequence["NormalizedLLMRequest"], factor: float
) -> list["NormalizedLLMRequest"]:
    """Compress/dilate arrival timestamps about the first request by ``factor``.

    ``factor > 1`` makes arrivals denser (busier serving tier); ``< 1`` sparser.
    Token counts, models, sessions and failure flags are preserved exactly —
    only the inter-arrival spacing scales. Used by the backtest's documented
    load-regime sensitivity sweep so the same real burst SHAPE is replayed at
    several load levels.
    """
    if not requests or factor <= 0 or factor == 1.0:
        return list(requests)
    ordered = sorted(requests, key=lambda r: (r.timestamp_s, r.request_id))
    t0 = ordered[0].timestamp_s
    out = []
    for r in ordered:
        new_ts = t0 + (r.timestamp_s - t0) / factor
        out.append(
            NormalizedLLMRequest(
                request_id=r.request_id, timestamp_s=new_ts,
                session_id=r.session_id, model=r.model,
                prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens,
                total_tokens=r.total_tokens, elapsed_s=r.elapsed_s,
                log_type=r.log_type, is_failure=r.is_failure,
                cache_affinity_key=r.cache_affinity_key,
            )
        )
    return out


def summarize_trace(
    requests: Sequence[NormalizedLLMRequest],
    *,
    dataset: str,
    bin_seconds: float = 60.0,
) -> TraceSummary:
    """Compute the descriptive stats the ingestion script prints.

    RPS-by-minute uses ``bin_seconds`` fixed bins over the trace's own time
    range, so the numbers are reproducible and independent of wall clock. The
    cache-affinity stats are an honest *proxy* for prefix/session locality.
    """
    if not requests:
        return TraceSummary(
            dataset=dataset, row_count=0, included_count=0, time_start_s=0.0,
            time_end_s=0.0, duration_s=0.0, model_distribution={},
            log_type_distribution={}, failure_rate_pct=0.0,
            prompt_tokens_p50=0.0, prompt_tokens_p95=0.0, prompt_tokens_p99=0.0,
            output_tokens_p50=0.0, output_tokens_p95=0.0, output_tokens_p99=0.0,
            total_tokens_p50=0.0, total_tokens_p95=0.0, total_tokens_p99=0.0,
            rps_mean_per_min=0.0, rps_p95_per_min=0.0, rps_max_per_min=0.0,
            distinct_cache_keys=0, cache_key_reuse_rate_pct=0.0,
            mean_requests_per_cache_key=0.0, has_session_ids=False,
            has_elapsed=False,
        )

    times = [r.timestamp_s for r in requests]
    t0, t1 = min(times), max(times)
    duration = max(0.0, t1 - t0)

    model_dist: dict = {}
    log_dist: dict = {}
    for r in requests:
        model_dist[r.model] = model_dist.get(r.model, 0) + 1
        log_dist[r.log_type] = log_dist.get(r.log_type, 0) + 1

    failures = sum(1 for r in requests if r.is_failure)
    failure_rate = 100.0 * failures / len(requests)

    prompt = [r.prompt_tokens for r in requests]
    output = [r.output_tokens for r in requests]
    total = [r.total_tokens for r in requests]

    # RPS per fixed bin over the trace time range.
    n_bins = max(1, int(math.ceil((duration + 1e-9) / bin_seconds))) if duration > 0 else 1
    bin_counts = [0] * n_bins
    for r in requests:
        idx = 0 if duration <= 0 else min(n_bins - 1, int((r.timestamp_s - t0) / bin_seconds))
        bin_counts[idx] += 1
    rps_per_bin = [c / bin_seconds for c in bin_counts]

    # Cache-affinity proxy: how often a cache_affinity_key recurs. Requests with
    # NO affinity key (None) are excluded — a trace without any session/prefix/
    # logical-stream signal has no honest cache-affinity proxy at all.
    key_counts: dict = {}
    for r in requests:
        if r.cache_affinity_key is None:
            continue
        key_counts[r.cache_affinity_key] = key_counts.get(r.cache_affinity_key, 0) + 1
    distinct_keys = len(key_counts)
    keyed_requests = sum(key_counts.values())
    reused_requests = sum(c - 1 for c in key_counts.values() if c > 1)
    reuse_rate = 100.0 * reused_requests / keyed_requests if keyed_requests else 0.0
    mean_per_key = keyed_requests / distinct_keys if distinct_keys else 0.0

    return TraceSummary(
        dataset=dataset,
        row_count=len(requests),
        included_count=len(requests),
        time_start_s=t0,
        time_end_s=t1,
        duration_s=duration,
        model_distribution=dict(sorted(model_dist.items())),
        log_type_distribution=dict(sorted(log_dist.items())),
        failure_rate_pct=failure_rate,
        prompt_tokens_p50=percentile(prompt, 50),
        prompt_tokens_p95=percentile(prompt, 95),
        prompt_tokens_p99=percentile(prompt, 99),
        output_tokens_p50=percentile(output, 50),
        output_tokens_p95=percentile(output, 95),
        output_tokens_p99=percentile(output, 99),
        total_tokens_p50=percentile(total, 50),
        total_tokens_p95=percentile(total, 95),
        total_tokens_p99=percentile(total, 99),
        rps_mean_per_min=sum(rps_per_bin) / len(rps_per_bin),
        rps_p95_per_min=percentile(rps_per_bin, 95),
        rps_max_per_min=max(rps_per_bin),
        distinct_cache_keys=distinct_keys,
        cache_key_reuse_rate_pct=reuse_rate,
        mean_requests_per_cache_key=mean_per_key,
        has_session_ids=any(r.session_id is not None for r in requests),
        has_elapsed=any(r.elapsed_s is not None for r in requests),
        has_cache_affinity=distinct_keys > 0,
    )


# ===========================================================================
# GPU cluster / job-scheduling trace schema (fragmentation / packing backtests)
# ===========================================================================
# A SEPARATE normalized contract from NormalizedLLMRequest: GPU cluster traces
# (e.g. Alibaba cluster-trace-gpu-v2023) describe *jobs requesting GPUs on a
# heterogeneous fleet*, not token-level serving requests. The replay/backtest
# for these is a bin-packing / placement problem, not a serving-physics replay.


@dataclass(frozen=True)
class NormalizedGPUJob:
    """One normalized GPU-cluster job — the cross-dataset job contract.

    Fields map 1:1 to the mission spec. Optional fields are ``None`` when the
    source dataset does not provide them. Resource-request fields
    (``cpu_milli`` / ``memory_mib`` / ``gpu_milli``) are kept because real GPU
    fragmentation/packing needs the full request vector — ``gpu_milli`` is the
    thousandths-of-a-GPU sharing request (Alibaba v2023). ``token_equivalent_work``
    is a GPU-work proxy (effective_gpu × duration), labelled honestly as
    ``completed_gpu_job_work`` (token_equivalent) in reports — NOT inference
    output tokens.
    """

    job_id: str
    submit_time_s: Optional[float]
    start_time_s: Optional[float]
    end_time_s: Optional[float]
    duration_s: Optional[float]
    gpu_count: int
    gpu_type: Optional[str]
    gpu_memory_gb: Optional[float]
    status: Optional[str]
    user_or_group: Optional[str]
    workload_type: Optional[str]
    priority: Optional[str]
    placement_nodes: Optional[str]
    placement_gpus: Optional[str]
    is_failed: bool
    deadline_s: Optional[float] = None
    token_equivalent_work: Optional[float] = None
    # Extra request vector for packing (None when the source lacks them).
    cpu_milli: Optional[int] = None
    memory_mib: Optional[int] = None
    gpu_milli: Optional[int] = None
    # Scheduling-queue delay (start − submit) when both exist; e.g. Philly.
    queue_wait_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "submit_time_s": self.submit_time_s,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "status": self.status,
            "user_or_group": self.user_or_group,
            "workload_type": self.workload_type,
            "priority": self.priority,
            "placement_nodes": self.placement_nodes,
            "placement_gpus": self.placement_gpus,
            "is_failed": self.is_failed,
            "deadline_s": self.deadline_s,
            "token_equivalent_work": self.token_equivalent_work,
            "cpu_milli": self.cpu_milli,
            "memory_mib": self.memory_mib,
            "gpu_milli": self.gpu_milli,
            "queue_wait_s": self.queue_wait_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedGPUJob":
        def _f(k):
            v = d.get(k)
            return None if v in (None, "") else float(v)

        def _i(k):
            v = d.get(k)
            return None if v in (None, "") else int(float(v))

        return cls(
            job_id=str(d["job_id"]),
            submit_time_s=_f("submit_time_s"),
            start_time_s=_f("start_time_s"),
            end_time_s=_f("end_time_s"),
            duration_s=_f("duration_s"),
            gpu_count=int(d.get("gpu_count") or 0),
            gpu_type=(None if d.get("gpu_type") in (None, "") else str(d["gpu_type"])),
            gpu_memory_gb=_f("gpu_memory_gb"),
            status=(None if d.get("status") in (None, "") else str(d["status"])),
            user_or_group=(None if d.get("user_or_group") in (None, "")
                           else str(d["user_or_group"])),
            workload_type=(None if d.get("workload_type") in (None, "")
                           else str(d["workload_type"])),
            priority=(None if d.get("priority") in (None, "") else str(d["priority"])),
            placement_nodes=(None if d.get("placement_nodes") in (None, "")
                             else str(d["placement_nodes"])),
            placement_gpus=(None if d.get("placement_gpus") in (None, "")
                            else str(d["placement_gpus"])),
            is_failed=bool(d.get("is_failed")),
            deadline_s=_f("deadline_s"),
            token_equivalent_work=_f("token_equivalent_work"),
            cpu_milli=_i("cpu_milli"),
            memory_mib=_i("memory_mib"),
            gpu_milli=_i("gpu_milli"),
            queue_wait_s=_f("queue_wait_s"),
        )


@dataclass(frozen=True)
class NormalizedGPUUtilizationSample:
    """One normalized GPU utilization sample (time-series).

    Defined for the shared schema / future datasets (e.g. MIT Supercloud). The
    Alibaba v2023 trace has **no** utilization time-series, so its ingester
    returns an empty list — documented, not invented.
    """

    timestamp_s: float
    node_id: str
    gpu_id: str
    gpu_type: Optional[str]
    gpu_utilization: Optional[float]
    memory_utilization: Optional[float]
    power_w: Optional[float] = None
    temperature_c: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s, "node_id": self.node_id,
            "gpu_id": self.gpu_id, "gpu_type": self.gpu_type,
            "gpu_utilization": self.gpu_utilization,
            "memory_utilization": self.memory_utilization,
            "power_w": self.power_w, "temperature_c": self.temperature_c,
        }

# ===========================================================================
# Multi-layer GenAI serving trace schema (Alibaba cluster-trace-v2026-GenAI)
# ===========================================================================
# A top-down GenAI (stable-diffusion) serving trace spans three layers:
# application (user requests + e2e latency), middleware (gateway queues), and
# infrastructure (container/GPU utilisation + memory). These are SEPARATE
# normalized records; the layers are NOT necessarily joinable (the application
# layer and the metric layers use incompatible anonymized time bases and the
# request layer has no container_ip), so linkage quality is classified per pair
# rather than assumed — see ``aurelius/traces/alibaba_genai.py``.


@dataclass(frozen=True)
class NormalizedGenAIRequest:
    """Application-layer GenAI serving request (e.g. stable-diffusion)."""

    request_id: str
    timestamp_s: Optional[float]
    service_id: Optional[str]          # model / checkpoint id
    prompt_or_input_size: Optional[float]
    output_size: Optional[float]       # e.g. images-per-prompt; None if absent
    e2e_latency_s: Optional[float]     # measured end-to-end execution time
    status: Optional[str]
    request_type: Optional[str]
    is_failed: bool
    # extra GenAI fields kept because they drive cold-start modelling
    num_lora: Optional[int] = None
    num_inference_steps: Optional[float] = None
    group_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "timestamp_s": self.timestamp_s,
            "service_id": self.service_id,
            "prompt_or_input_size": self.prompt_or_input_size,
            "output_size": self.output_size, "e2e_latency_s": self.e2e_latency_s,
            "status": self.status, "request_type": self.request_type,
            "is_failed": self.is_failed, "num_lora": self.num_lora,
            "num_inference_steps": self.num_inference_steps,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedGenAIRequest":
        def _f(k):
            v = d.get(k)
            return None if v in (None, "") else float(v)
        return cls(
            request_id=str(d["request_id"]),
            timestamp_s=_f("timestamp_s"), service_id=d.get("service_id"),
            prompt_or_input_size=_f("prompt_or_input_size"),
            output_size=_f("output_size"), e2e_latency_s=_f("e2e_latency_s"),
            status=d.get("status"), request_type=d.get("request_type"),
            is_failed=bool(d.get("is_failed")),
            num_lora=(None if d.get("num_lora") in (None, "") else int(float(d["num_lora"]))),
            num_inference_steps=_f("num_inference_steps"),
            group_id=d.get("group_id"),
        )


@dataclass(frozen=True)
class NormalizedGatewayQueueSample:
    """Middleware-layer gateway/queue telemetry sample."""

    timestamp_s: float
    gateway_id: Optional[str]
    service_id: Optional[str]
    queue_depth: Optional[float] = None
    waiting_time_s: Optional[float] = None
    arrival_rate: Optional[float] = None
    dispatch_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s, "gateway_id": self.gateway_id,
            "service_id": self.service_id, "queue_depth": self.queue_depth,
            "waiting_time_s": self.waiting_time_s, "arrival_rate": self.arrival_rate,
            "dispatch_rate": self.dispatch_rate,
        }


@dataclass(frozen=True)
class NormalizedSchedulerPipelineEvent:
    """Middleware/scheduler-layer pipeline-stage timing event."""

    timestamp_s: float
    request_id: Optional[str]
    service_id: Optional[str]
    stage: Optional[str]
    container_id: Optional[str]
    gpu_id: Optional[str] = None
    node_id: Optional[str] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    duration_s: Optional[float] = None
    status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s, "request_id": self.request_id,
            "service_id": self.service_id, "stage": self.stage,
            "container_id": self.container_id, "gpu_id": self.gpu_id,
            "node_id": self.node_id, "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s, "duration_s": self.duration_s,
            "status": self.status,
        }


@dataclass(frozen=True)
class NormalizedInfraSample:
    """Infrastructure-layer container/GPU resource sample."""

    timestamp_s: float
    node_id: Optional[str]
    container_id: Optional[str]
    gpu_id: Optional[str]
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_utilization: Optional[float] = None
    memory_used: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s, "node_id": self.node_id,
            "container_id": self.container_id, "gpu_id": self.gpu_id,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "cpu_utilization": self.cpu_utilization, "memory_used": self.memory_used,
        }
