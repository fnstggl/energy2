"""Canonical record schemas for the federated HF benchmark corpus.

Every record carries:

- ``source_dataset_id`` — HF dataset id (``namespace/name``).
- ``trace_type`` — one of ``CANONICAL_TRACE_TYPES``.
- ``provenance`` — free-form label identifying the source variant +
  ingestion timestamp. Examples: ``"agent-perf-bench/AgentPerfBench@trace_replay#summary_v1"``.
- ``field_quality`` — mapping of field name -> one of ``FIELD_QUALITY_VALUES``.
- ``limitations`` — explicit list of what the source does NOT measure.

There is no "merged super-record". Federated means: datasets stay separate;
records remain typed by their canonical trace type; cross-dataset queries
must explicitly select compatible trace types + signals — see
``docs/HF_DATASET_REGISTRY.md``.

Honesty rules (inherited from ``aurelius/traces/eval_schema.py``):

- A field that came from a measurement is ``real``.
- A field computed from another field is ``derived``.
- A field substituted from a non-measured source (e.g. character-count
  token estimate) is ``proxy``.
- A field generated from a distribution is ``synthetic``.
- A field absent from the source is ``missing``.
- ``unknown`` is reserved for fields where the source did not document
  provenance.

Trust hierarchy mirrors the mission spec — the highest-trust class is
``telemetry_trace`` (real Prometheus / DCGM / vLLM / Triton / Ray Serve
/ Kubernetes exports). Benchmark traces are NEVER promoted to "production
telemetry" status — they are calibration priors at best.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

CANONICAL_TRACE_TYPES = frozenset({
    "request_shape_trace",
    "latency_benchmark_trace",
    "kernel_profile_trace",
    "cluster_scheduler_trace",
    "cache_residency_trace",
    "telemetry_trace",
    "mixed_or_unknown_trace",
})


FIELD_QUALITY_VALUES = frozenset({
    "real",
    "derived",
    "proxy",
    "synthetic",
    "missing",
    "unknown",
})


TRUST_TIERS = {
    "tier_1_real_pilot_telemetry": 1,
    "tier_2_public_telemetry_traces": 2,
    "tier_3_cluster_scheduler_traces": 3,
    "tier_4_latency_benchmark_traces": 4,
    "tier_5_request_shape_traces": 5,
    "tier_6_synthetic_benchmark_data": 6,
}


CANONICAL_TRACE_TYPE_TO_TRUST_TIER = {
    "telemetry_trace": "tier_2_public_telemetry_traces",
    "cluster_scheduler_trace": "tier_3_cluster_scheduler_traces",
    "latency_benchmark_trace": "tier_4_latency_benchmark_traces",
    "kernel_profile_trace": "tier_4_latency_benchmark_traces",
    "cache_residency_trace": "tier_4_latency_benchmark_traces",
    "request_shape_trace": "tier_5_request_shape_traces",
    "mixed_or_unknown_trace": "tier_6_synthetic_benchmark_data",
}


class HFCorpusSchemaError(ValueError):
    """Raised when a federated HF corpus record is malformed."""


def _check_field_quality_map(quality: dict, allowed_keys: set, where: str) -> None:
    for k, v in quality.items():
        if k not in allowed_keys:
            raise HFCorpusSchemaError(
                f"{where}: field_quality has unknown key '{k}'; allowed={sorted(allowed_keys)}"
            )
        if v not in FIELD_QUALITY_VALUES:
            raise HFCorpusSchemaError(
                f"{where}: field_quality[{k}]='{v}' not in {sorted(FIELD_QUALITY_VALUES)}"
            )


@dataclass(frozen=True)
class CanonicalCorpusRecord:
    """Base shape every canonical record carries.

    Subclasses add their type-specific measured / derived / proxy fields, but
    the cross-type contract (id, type, provenance, field_quality, limitations)
    stays identical so the evaluation harness can route records uniformly.
    """

    source_dataset_id: str
    trace_type: str
    provenance: str
    field_quality: dict
    limitations: tuple = field(default_factory=tuple)

    def _validate_base(self, payload_keys: set) -> None:
        if self.trace_type not in CANONICAL_TRACE_TYPES:
            raise HFCorpusSchemaError(
                f"trace_type='{self.trace_type}' not in {sorted(CANONICAL_TRACE_TYPES)}"
            )
        _check_field_quality_map(
            self.field_quality, payload_keys,
            where=f"{type(self).__name__}({self.source_dataset_id})",
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["limitations"] = list(self.limitations)
        return d


# ---------------------------------------------------------------------------
# Per-trace-type records
# ---------------------------------------------------------------------------


# Field-name groups for field_quality validation. The frontier harness only
# evaluates records when the relevant fields are labelled ``real`` (or
# ``derived`` from a measured base) — never when they are ``proxy``.


REQUEST_SHAPE_PAYLOAD_FIELDS = {
    "request_id",
    "timestamp_s",
    "created_at_iso",
    "finished_at_iso",
    "session_id",
    "turn_count",
    "prompt_tokens",
    "output_tokens",
    "model_id",
    "status",
    "model_parameters_json",
    "temperature",
    "max_tokens_param",
    "top_p",
    "seed",
}


LATENCY_BENCHMARK_PAYLOAD_FIELDS = {
    "model",
    "model_family",
    "gpu",
    "engine",
    "profile",
    "run_id",
    "tensor_parallelism",
    "concurrency",
    "num_requests",
    "duration_s",
    "request_throughput",
    "input_token_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
}


KERNEL_PROFILE_PAYLOAD_FIELDS = {
    "source",
    "gpu",
    "model",
    "kernel_name",
    "kernel_family",
    "op_type",
    "dtype",
    "m",
    "n",
    "k",
    "batch_size",
    "sequence_length",
    "duration_ms",
    "duration_us",
    "dram_bytes",
    "n_heads",
    "head_dim",
    "kv_heads",
    "numel",
    "held_out",
    "launch_block_size",
    "launch_grid_size",
    "launch_registers_per_thread",
}


CLUSTER_SCHEDULER_PAYLOAD_FIELDS = {
    "job_id",
    "submit_time_s",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "queue_wait_s",
    "gpu_count",
    "gpu_type",
    "status",
    "is_failed",
    "user_or_group",
}


CACHE_RESIDENCY_PAYLOAD_FIELDS = {
    "model_id",
    "prefix_id",
    "request_id",
    "timestamp_s",
    "created_at_iso",
    "finished_at_iso",
    "cache_hit",
    "cold_start",
    "residency_state",
    "bucket_count",
    "reused_bucket_count",
    "reuse_percentage",
    "token_count",
    "bucket_ids_hash",
    "bucket_ids_sample",
    "status",
    "prompt_tokens",
    "output_tokens",
    "model_parameters_json",
    "temperature",
    "max_tokens_param",
    "top_p",
    "seed",
}


TELEMETRY_PAYLOAD_FIELDS = {
    "timestamp_s",
    "created_at_iso",
    "service_id",
    "request_id",
    "instance_id",
    "instance_type",
    "model_id",
    "gpu",
    "engine",
    "engine_version",
    # Queue + scheduler state.
    "queue_depth",
    "queue_wait_s",
    "num_running",
    "num_waiting",
    "num_active_decode_seqs",
    "num_preempted",
    "running_requests_count",
    "waiting_requests_count",
    "pending_prefill_tokens",
    "pending_decode_tokens",
    "decode_ctx_p50",
    "decode_ctx_p95",
    "decode_ctx_max",
    # KV cache + residency.
    "kv_cache_utilization",
    "kv_free_blocks",
    "kv_evictions_per_s",
    "token_budget_per_iter",
    "prefill_chunk_size",
    "max_num_seqs",
    # Throughput EMAs.
    "ema_decode_tok_per_s",
    "ema_prefill_tok_per_s",
    "ema_decode_iter_ms",
    # Per-request measured latency.
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "ttft_p99_ms",
    "tpot_p99_ms",
    "actual_e2e_latency_s",
    "actual_ttft_s",
    "actual_tpot_s",
    "completion_timestamp_s",
    "prediction_timestamp_s",
    "prediction_latency_ms",
    "probe_latency_ms",
    "num_prompt_tokens",
    "num_predicted_output_tokens",
    "actual_output_tokens",
    # Aggregate rates.
    "throughput_rps",
    "concurrency",
    "replica_count",
    "gpu_utilization",
    "gpu_memory_pct",
    "timeout_rate_pct",
    "sla_violation_rate_pct",
    "status",
    "is_failed",
}


@dataclass(frozen=True)
class RequestShapeRecord(CanonicalCorpusRecord):
    request_id: Optional[str] = None
    timestamp_s: Optional[float] = None
    created_at_iso: Optional[str] = None
    finished_at_iso: Optional[str] = None
    session_id: Optional[str] = None
    turn_count: Optional[int] = None
    prompt_tokens: Optional[float] = None
    output_tokens: Optional[float] = None
    model_id: Optional[str] = None
    status: Optional[str] = None
    model_parameters_json: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens_param: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self._validate_base(REQUEST_SHAPE_PAYLOAD_FIELDS)
        if self.trace_type != "request_shape_trace":
            raise HFCorpusSchemaError(
                f"RequestShapeRecord trace_type must be 'request_shape_trace', "
                f"got '{self.trace_type}'"
            )


@dataclass(frozen=True)
class BenchmarkLatencyRecord(CanonicalCorpusRecord):
    model: Optional[str] = None
    model_family: Optional[str] = None
    gpu: Optional[str] = None
    engine: Optional[str] = None
    profile: Optional[str] = None
    run_id: Optional[str] = None
    tensor_parallelism: Optional[int] = None
    concurrency: Optional[int] = None
    num_requests: Optional[int] = None
    duration_s: Optional[float] = None
    request_throughput: Optional[float] = None
    input_token_throughput: Optional[float] = None
    output_token_throughput: Optional[float] = None
    total_token_throughput: Optional[float] = None
    mean_ttft_ms: Optional[float] = None
    p50_ttft_ms: Optional[float] = None
    p90_ttft_ms: Optional[float] = None
    p99_ttft_ms: Optional[float] = None
    mean_tpot_ms: Optional[float] = None
    p50_tpot_ms: Optional[float] = None
    p90_tpot_ms: Optional[float] = None
    p99_tpot_ms: Optional[float] = None
    mean_itl_ms: Optional[float] = None
    p50_itl_ms: Optional[float] = None
    p90_itl_ms: Optional[float] = None
    p99_itl_ms: Optional[float] = None
    mean_e2el_ms: Optional[float] = None
    p50_e2el_ms: Optional[float] = None
    p90_e2el_ms: Optional[float] = None
    p99_e2el_ms: Optional[float] = None

    def __post_init__(self):
        self._validate_base(LATENCY_BENCHMARK_PAYLOAD_FIELDS)
        if self.trace_type != "latency_benchmark_trace":
            raise HFCorpusSchemaError(
                f"BenchmarkLatencyRecord trace_type must be "
                f"'latency_benchmark_trace', got '{self.trace_type}'"
            )


@dataclass(frozen=True)
class KernelProfileRecord(CanonicalCorpusRecord):
    source: Optional[str] = None
    gpu: Optional[str] = None
    model: Optional[str] = None
    kernel_name: Optional[str] = None
    kernel_family: Optional[str] = None
    op_type: Optional[str] = None
    dtype: Optional[str] = None
    m: Optional[float] = None
    n: Optional[float] = None
    k: Optional[float] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    duration_ms: Optional[float] = None
    duration_us: Optional[float] = None
    dram_bytes: Optional[float] = None
    n_heads: Optional[float] = None
    head_dim: Optional[float] = None
    kv_heads: Optional[float] = None
    numel: Optional[float] = None
    held_out: Optional[bool] = None
    launch_block_size: Optional[float] = None
    launch_grid_size: Optional[float] = None
    launch_registers_per_thread: Optional[float] = None

    def __post_init__(self):
        self._validate_base(KERNEL_PROFILE_PAYLOAD_FIELDS)
        if self.trace_type != "kernel_profile_trace":
            raise HFCorpusSchemaError(
                f"KernelProfileRecord trace_type must be "
                f"'kernel_profile_trace', got '{self.trace_type}'"
            )


@dataclass(frozen=True)
class ClusterSchedulerRecord(CanonicalCorpusRecord):
    job_id: Optional[str] = None
    submit_time_s: Optional[float] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    duration_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    status: Optional[str] = None
    is_failed: Optional[bool] = None
    user_or_group: Optional[str] = None

    def __post_init__(self):
        self._validate_base(CLUSTER_SCHEDULER_PAYLOAD_FIELDS)
        if self.trace_type != "cluster_scheduler_trace":
            raise HFCorpusSchemaError(
                f"ClusterSchedulerRecord trace_type must be "
                f"'cluster_scheduler_trace', got '{self.trace_type}'"
            )


@dataclass(frozen=True)
class CacheResidencyRecord(CanonicalCorpusRecord):
    model_id: Optional[str] = None
    prefix_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp_s: Optional[float] = None
    created_at_iso: Optional[str] = None
    finished_at_iso: Optional[str] = None
    cache_hit: Optional[bool] = None
    cold_start: Optional[bool] = None
    residency_state: Optional[str] = None
    bucket_count: Optional[int] = None
    reused_bucket_count: Optional[int] = None
    reuse_percentage: Optional[float] = None
    token_count: Optional[int] = None
    bucket_ids_hash: Optional[str] = None
    bucket_ids_sample: Optional[str] = None
    status: Optional[str] = None
    prompt_tokens: Optional[float] = None
    output_tokens: Optional[float] = None
    model_parameters_json: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens_param: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self._validate_base(CACHE_RESIDENCY_PAYLOAD_FIELDS)
        if self.trace_type != "cache_residency_trace":
            raise HFCorpusSchemaError(
                f"CacheResidencyRecord trace_type must be "
                f"'cache_residency_trace', got '{self.trace_type}'"
            )


@dataclass(frozen=True)
class TelemetryRecord(CanonicalCorpusRecord):
    timestamp_s: Optional[float] = None
    created_at_iso: Optional[str] = None
    service_id: Optional[str] = None
    request_id: Optional[str] = None
    instance_id: Optional[str] = None
    instance_type: Optional[str] = None
    model_id: Optional[str] = None
    gpu: Optional[str] = None
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    queue_depth: Optional[float] = None
    queue_wait_s: Optional[float] = None
    num_running: Optional[int] = None
    num_waiting: Optional[int] = None
    num_active_decode_seqs: Optional[int] = None
    num_preempted: Optional[int] = None
    running_requests_count: Optional[int] = None
    waiting_requests_count: Optional[int] = None
    pending_prefill_tokens: Optional[int] = None
    pending_decode_tokens: Optional[int] = None
    decode_ctx_p50: Optional[float] = None
    decode_ctx_p95: Optional[float] = None
    decode_ctx_max: Optional[float] = None
    kv_cache_utilization: Optional[float] = None
    kv_free_blocks: Optional[int] = None
    kv_evictions_per_s: Optional[float] = None
    token_budget_per_iter: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    max_num_seqs: Optional[int] = None
    ema_decode_tok_per_s: Optional[float] = None
    ema_prefill_tok_per_s: Optional[float] = None
    ema_decode_iter_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None
    actual_e2e_latency_s: Optional[float] = None
    actual_ttft_s: Optional[float] = None
    actual_tpot_s: Optional[float] = None
    completion_timestamp_s: Optional[float] = None
    prediction_timestamp_s: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    probe_latency_ms: Optional[float] = None
    num_prompt_tokens: Optional[int] = None
    num_predicted_output_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    throughput_rps: Optional[float] = None
    concurrency: Optional[int] = None
    replica_count: Optional[int] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_pct: Optional[float] = None
    timeout_rate_pct: Optional[float] = None
    sla_violation_rate_pct: Optional[float] = None
    status: Optional[str] = None
    is_failed: Optional[bool] = None

    def __post_init__(self):
        self._validate_base(TELEMETRY_PAYLOAD_FIELDS)
        if self.trace_type != "telemetry_trace":
            raise HFCorpusSchemaError(
                f"TelemetryRecord trace_type must be 'telemetry_trace', "
                f"got '{self.trace_type}'"
            )


TRACE_TYPE_TO_RECORD_CLASS = {
    "request_shape_trace": RequestShapeRecord,
    "latency_benchmark_trace": BenchmarkLatencyRecord,
    "kernel_profile_trace": KernelProfileRecord,
    "cluster_scheduler_trace": ClusterSchedulerRecord,
    "cache_residency_trace": CacheResidencyRecord,
    "telemetry_trace": TelemetryRecord,
}


TRACE_TYPE_TO_PAYLOAD_FIELDS = {
    "request_shape_trace": REQUEST_SHAPE_PAYLOAD_FIELDS,
    "latency_benchmark_trace": LATENCY_BENCHMARK_PAYLOAD_FIELDS,
    "kernel_profile_trace": KERNEL_PROFILE_PAYLOAD_FIELDS,
    "cluster_scheduler_trace": CLUSTER_SCHEDULER_PAYLOAD_FIELDS,
    "cache_residency_trace": CACHE_RESIDENCY_PAYLOAD_FIELDS,
    "telemetry_trace": TELEMETRY_PAYLOAD_FIELDS,
}
