"""Calibration metadata for the inference-serving realism layer.

Every serving-realism parameter is wrapped in a ``CalibratedParam`` carrying its
value, source, source-type, confidence, and a calibration note, so that NO
constant is a hidden magic number. The audit explicitly required this: simulator
realism claims must be inspectable and honestly graded.

Confidence ladder (most → least trustworthy):
    MEASURED          — measured on real hardware/telemetry in THIS repo
    BENCHMARK_DERIVED  — taken from a public benchmark/paper number
    DOCUMENTED         — stated in vendor/system documentation
    INFERRED           — reasoned from a documented mechanism, not a number
    HEURISTIC          — engineering guess; MUST be calibrated before claims

IMPORTANT: the great majority of these are HEURISTIC or INFERRED. None are
measured against a live cluster. They exist to make the simulator's *dynamics*
qualitatively believable (convex saturation, exploding tails, autoscaling lag),
NOT to assert quantitative production accuracy. Treat every value as a tunable
prior, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Source-type / confidence vocabularies (strings kept simple for serialization).
MEASURED = "measured"
BENCHMARK_DERIVED = "benchmark_derived"
DOCUMENTED = "documented"
INFERRED = "inferred"
HEURISTIC = "heuristic"

_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class CalibratedParam:
    """A simulator parameter with explicit provenance and confidence.

    value:            the numeric value used by the simulator
    source:           short citation / origin (URL, paper, "engineering guess")
    source_type:      one of MEASURED/BENCHMARK_DERIVED/DOCUMENTED/INFERRED/HEURISTIC
    confidence:       "high" | "medium" | "low"
    calibration_notes: what would be needed to replace this with a real number
    """

    value: float
    source: str
    source_type: str
    confidence: str
    calibration_notes: str

    def __post_init__(self) -> None:
        if self.confidence not in _CONFIDENCE:
            raise ValueError(f"confidence must be one of {_CONFIDENCE}, got {self.confidence!r}")

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "calibration_notes": self.calibration_notes,
        }


def _h(value, notes, *, source_type=HEURISTIC, confidence="low", source="engineering guess"):
    return CalibratedParam(value=value, source=source, source_type=source_type,
                           confidence=confidence, calibration_notes=notes)


# ---------------------------------------------------------------------------
# Serving-realism parameter registry
# ---------------------------------------------------------------------------
# Grouped by subsystem. Every value is inspectable via SERVING_PARAMS.

SERVING_PARAMS: dict[str, CalibratedParam] = {
    # --- Arrivals ---------------------------------------------------------
    "burst_state_prob": _h(
        0.12,
        "Per-tick probability a queue enters a burst state (Markov-modulated "
        "arrivals). Calibrate from real arrival traces (autocorrelation of RPS).",
        source_type=INFERRED, source="MMPP arrival modelling (Fischer & Meier-Hellstern 1993)",
    ),
    "burst_exit_prob": _h(
        0.45,
        "Per-tick probability a burst ends. Burst mean length ≈ 1/exit. Calibrate "
        "from real spike durations.",
        source_type=INFERRED, source="MMPP arrival modelling",
    ),
    "burst_multiplier": _h(
        2.5,
        "Arrival-rate multiplier while in a burst state. Calibrate from p99/p50 "
        "of real RPS.",
    ),

    # --- Queueing / saturation -------------------------------------------
    "safe_utilization": _h(
        0.70,
        "Upper bound of the safe operating band. Above this, waiting time grows "
        "convexly (Erlang-C / Kingman). Documented rule-of-thumb for latency-"
        "sensitive serving.",
        source_type=INFERRED, source="Erlang-C / Kingman's formula (heavy-traffic)",
        confidence="medium",
    ),
    "overload_utilization": _h(
        0.92,
        "Start of the overload-collapse region: backlog accumulates faster than "
        "it drains; tails run away. Calibrate from real saturation tests.",
        source_type=INFERRED, source="Kingman's heavy-traffic approximation",
        confidence="medium",
    ),
    "saturation_convexity": _h(
        2.0,
        "Exponent on 1/(1-rho) tail amplification. Kingman gives ~1/(1-rho); we "
        "raise it so p95/p99 explode faster than the mean. Calibrate from real "
        "p99-vs-utilization curves.",
    ),

    # --- Latency tails ----------------------------------------------------
    "tail_p95_base": _h(
        1.5,
        "p95/p50 ratio at LOW load. At high load it grows toward tail_p95_max. "
        "Real LLM serving p95/p50 ~1.3-1.8 at low load.",
        source_type=INFERRED, source="queueing tail behaviour", confidence="medium",
    ),
    "tail_p95_max": _h(
        6.0,
        "p95/p50 ratio near saturation. Calibrate from real tail curves.",
    ),
    "tail_p99_base": _h(
        2.0,
        "p99/p50 ratio at low load (mild). Grows convexly toward tail_p99_max "
        "near saturation.", source_type=INFERRED, confidence="medium",
    ),
    "tail_p99_max": _h(
        15.0,
        "p99/p50 ratio near saturation — tails explode super-linearly. Calibrate "
        "from real p99 runaway tests.",
    ),

    # --- TTFT decomposition (ms per unit) ---------------------------------
    "ttft_per_prompt_token_ms": _h(
        0.25,
        "Prefill TTFT contribution per prompt token (alpha). Order-of-magnitude "
        "for ~7B on A100/H100; calibrate per model/GPU from real prefill timings.",
        source_type=BENCHMARK_DERIVED,
        source="vLLM/Sarathi-Serve prefill throughput (public benchmarks)",
        confidence="low",
    ),
    "ttft_per_active_seq_ms": _h(
        1.5,
        "TTFT contribution per concurrent active sequence (beta) — scheduler "
        "contention. Calibrate from real batching interference tests.",
    ),
    "ttft_kv_pressure_ms": _h(
        400.0,
        "Max TTFT inflation (gamma) at full KV pressure (allocation stalls / "
        "preemption). Calibrate from real KV-pressure tests.",
    ),

    # --- TPOT / batching --------------------------------------------------
    "tpot_per_active_token_ms": _h(
        0.02,
        "Decode TPOT contribution per active token in the batch (decode "
        "contention). Calibrate from real continuous-batching ITL curves.",
        source_type=INFERRED, source="continuous batching (vLLM) decode contention",
    ),
    "batch_efficiency_knee": _h(
        32.0,
        "Active sequences at which per-replica batching efficiency is ~maximal. "
        "Spreading the same load over more replicas pushes each below the knee, "
        "lowering throughput/GPU. Calibrate from real throughput-vs-batch curves.",
        source_type=INFERRED, source="continuous batching throughput curve",
    ),
    "batch_efficiency_floor": _h(
        0.5,
        "Minimum per-replica throughput fraction at very low concurrency. A "
        "single-stream request still gets ~half of full BATCH throughput (it is "
        "not throughput-bound). Calibrate from real low-QPS vs saturated tput.",
        source_type=INFERRED, source="continuous batching throughput curve",
        confidence="medium",
    ),

    # --- Autoscaling lag (ticks; 1 tick = scenario tick_duration_hours) ----
    "scale_detect_ticks": _h(
        1.0,
        "Polling/detection delay before a scale decision. Real HPA/KEDA poll "
        "windows are tens of seconds; here expressed in ticks.",
        source_type=DOCUMENTED, source="K8s HPA default sync period (15s) / KEDA",
        confidence="medium",
    ),
    "replica_warmup_ticks": _h(
        2.0,
        "Provision + container start + model load + readiness before a new "
        "replica serves at full throughput. GPU node provisioning + large-model "
        "load is minutes; calibrate per model/runtime.",
        source_type=INFERRED, source="GPU node provisioning + model load latency",
        confidence="medium",
    ),
    "scale_cooldown_ticks": _h(
        3.0,
        "Anti-flapping stabilization window between scaling actions for one "
        "workload. Real autoscalers use stabilization windows (e.g. HPA 300s).",
        source_type=DOCUMENTED, source="K8s HPA scale-down stabilization (default 300s)",
        confidence="medium",
    ),

    # --- Migration cost ---------------------------------------------------
    "migration_queue_disruption": _h(
        0.30,
        "Fraction of one tick's arrivals added to the destination backlog as "
        "migration disruption (drained in-flight + rebalancing). Migrations are "
        "NOT free. Calibrate from real drain/rebalance behaviour.",
    ),
}


# ---------------------------------------------------------------------------
# KV-cache / prefix-affinity / memory-pressure parameter registry
# ---------------------------------------------------------------------------
# Added for the KV-cache realism upgrade. These drive aurelius/simulation/
# cluster/kv_cache.py. As with SERVING_PARAMS, the great majority are HEURISTIC
# or INFERRED priors chosen to make the simulator's KV/cache/locality dynamics
# qualitatively believable — NOT measured against a live serving cluster. The
# operational pressure thresholds are inferred from publicly documented vLLM
# PagedAttention / preemption behaviour, NOT asserted as universal constants.

KV_CACHE_PARAMS: dict[str, CalibratedParam] = {
    # --- KV memory scaling ------------------------------------------------
    "kv_bytes_per_elem": _h(
        2.0,
        "Bytes per KV element. 2 = FP16/BF16 (default), 1 = FP8, 0.5 ≈ NVFP4-like "
        "4-bit KV. Overriding this is the KV-quantization lever: lower precision → "
        "smaller KV footprint → larger effective context / lower pressure.",
        source_type=DOCUMENTED, source="FP16/FP8/FP4 element sizes; vLLM kv_cache_dtype",
        confidence="high",
    ),
    "kv_reserved_budget_frac": _h(
        0.80,
        "Fraction of a GPU's memory reserved for the KV cache after weights + "
        "activations (vLLM gpu_memory_utilization minus model weights). Pressure "
        "is kv_allocated / (this × free-after-weights). Calibrate per model/GPU.",
        source_type=INFERRED, source="vLLM gpu_memory_utilization default 0.9 minus weights",
        confidence="medium",
    ),
    "kv_block_size_tokens": _h(
        16.0,
        "PagedAttention KV block size in tokens. vLLM default block size is 16. "
        "Fragmentation = internal slack of partially filled blocks, NOT heap "
        "fragmentation.",
        source_type=DOCUMENTED, source="vLLM PagedAttention default block_size=16",
        confidence="high",
    ),

    # --- Pressure regions (operational heuristics, NOT universal) ---------
    "kv_pressure_elevated": _h(
        0.70,
        "KV_pressure at which batching efficiency starts dropping and tail latency "
        "rises (ELEVATED region). Operational heuristic inferred from vLLM "
        "behaviour; NOT a universal threshold.",
        source_type=INFERRED, source="vLLM scheduler/PagedAttention behaviour (inferred)",
        confidence="low",
    ),
    "kv_pressure_throttling": _h(
        0.90,
        "KV_pressure at which scheduling hesitates, admission delay and p95/p99 "
        "climb (THROTTLING_RISK region). Operational heuristic; NOT universal.",
        source_type=INFERRED, source="vLLM scheduler behaviour near KV exhaustion (inferred)",
        confidence="low",
    ),
    "kv_pressure_preemption": _h(
        0.97,
        "KV_pressure approaching 1.0 where preemption/recompute and cache eviction "
        "occur (PREEMPTION region). Operational heuristic; NOT universal.",
        source_type=INFERRED, source="vLLM preemption-by-recompute trigger (inferred)",
        confidence="low",
    ),
    "kv_pressure_ttft_max_mult": _h(
        4.0,
        "Max multiplier on the active-sequence TTFT component as pressure → 1.0 "
        "(allocation stalls). Calibrate from real KV-pressure TTFT curves.",
    ),
    "kv_pressure_batch_floor": _h(
        0.35,
        "Floor on batching efficiency under maximal KV pressure (scheduler runs "
        "thin batches to fit KV). Calibrate from real saturated-KV throughput.",
        source_type=INFERRED, source="continuous-batching degradation under KV pressure",
    ),

    # --- Preemption / recompute -------------------------------------------
    "preemption_prob_max": _h(
        0.6,
        "Per-tick probability that at least one sequence is preempted at the top "
        "of the preemption region. Ramps from 0 at kv_pressure_throttling to this "
        "at pressure 1.0. Calibrate from real preemption-count telemetry.",
    ),
    "recompute_ms_per_token": _h(
        0.30,
        "Recompute cost per lost context token when a preempted sequence is "
        "recomputed (prefill-like cost). Order-of-magnitude tied to prefill; "
        "calibrate per model/GPU.",
        source_type=INFERRED, source="preemption-by-recompute ≈ re-prefill cost",
    ),

    # --- Prefix-cache hit-rate curve --------------------------------------
    "prefix_hit_sigmoid_a": _h(
        8.0,
        "Steepness (a) of the prefix-cache hit-rate sigmoid in overlap: "
        "hit = sigmoid(a·(overlap−b))·locality. Higher = sharper transition. "
        "Calibrate from real overlap-vs-hit-rate scatter per workload family.",
    ),
    "prefix_hit_sigmoid_b": _h(
        0.45,
        "Midpoint (b) of the prefix-cache hit-rate sigmoid: overlap at which hit "
        "rate is half of its locality-limited maximum. Calibrate per workload.",
    ),
    "prefix_max_prefill_savings_frac": _h(
        0.85,
        "Maximum fraction of prefill TTFT that a full prefix-cache hit can remove. "
        "Not 1.0: the first uncached suffix block + scheduling still cost. "
        "Calibrate from real cached-vs-uncold prefill timings.",
        source_type=INFERRED, source="prefix reuse removes prefill, not decode/scheduling",
        confidence="medium",
    ),

    # --- Routing affinity / locality confidence ---------------------------
    "locality_confidence_growth": _h(
        0.35,
        "Logistic growth rate of locality/cache confidence per tick of sustained "
        "affinity (repeated shared prefixes on the same route). Warmup is "
        "reuse-driven, NOT purely time-driven. Calibrate from real warm-up curves.",
        source_type=INFERRED, source="prefix-cache warm-up is reuse-driven",
    ),
    "locality_confidence_decay": _h(
        0.15,
        "Per-tick decay of locality confidence when affinity is broken / cache is "
        "idle (stale locality maps, LRU pressure). Calibrate from real eviction.",
    ),
    "locality_confidence_init": _h(
        0.5,
        "Initial locality confidence for a freshly placed (not-yet-warm) workload. "
        "Neutral prior. Calibrate from cold-start hit-rate ramps.",
        source_type=HEURISTIC, confidence="low",
    ),
    "cold_route_confidence": _h(
        0.05,
        "Locality confidence immediately after a cold reroute/migration (cache "
        "lost, must rewarm). Near-zero by construction. Calibrate from post-"
        "migration hit-rate recovery.",
        source_type=INFERRED, source="cold reroute loses prefix cache",
        confidence="medium",
    ),

    # --- Cold reroute penalty ---------------------------------------------
    "prefill_cost_per_token_ms": _h(
        0.25,
        "Prefill cost per token used to price lost-prefix reuse on a cold reroute "
        "(cold_route_penalty = lost_prefill_tokens × this). Tied to the serving "
        "ttft_per_prompt_token_ms prior; calibrate per model/GPU.",
        source_type=BENCHMARK_DERIVED,
        source="vLLM/Sarathi-Serve prefill throughput (public benchmarks)",
        confidence="low",
    ),

    # --- Telemetry confidence ---------------------------------------------
    "telemetry_missing_routing_damp": _h(
        0.5,
        "Multiplier applied to routing aggressiveness when KV/cache telemetry is "
        "missing or stale (LOW/MEDIUM confidence tiers). Missing telemetry LOWERS "
        "confidence; it must NOT be read as 'no pressure'. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
}


# ---------------------------------------------------------------------------
# Model KV-architecture profiles (for the KV scaling law)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelKVProfile:
    """KV-cache architecture of a served model, for the KV scaling law.

    KV_bytes = batch · seq_len · layers · kv_heads · head_dim · 2 · bytes_per_elem

    attention_type is informational: classic (kv_heads == n_heads), GQA
    (1 < kv_heads < n_heads), or MQA (kv_heads == 1). Reduced-KV-head
    architectures are captured directly via kv_heads, NOT via hidden_size.
    """
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    attention_type: str          # "classic" | "gqa" | "mqa"
    bytes_per_elem: float = 2.0  # FP16 default; override for FP8 / FP4 KV quant
    source: str = "model config card"
    source_type: str = DOCUMENTED
    confidence: str = "medium"

    def kv_bytes_per_token(self) -> float:
        """Bytes of KV cache per token (both K and V → factor of 2)."""
        return float(self.layers * self.kv_heads * self.head_dim * 2 * self.bytes_per_elem)


# Public, documented architectures (layer/kv_head/head_dim from model cards).
# These are config-card values (DOCUMENTED), not measured serving numbers.
MODEL_KV_PROFILES: dict[str, ModelKVProfile] = {
    # Llama-3 8B: 32 layers, 8 KV heads (GQA), head_dim 128.
    "llama3-8b": ModelKVProfile(
        name="llama3-8b", layers=32, kv_heads=8, head_dim=128, attention_type="gqa",
        source="Llama-3 8B config (GQA, 32 KV heads → 8 groups)",
    ),
    # Llama-2 7B: classic MHA, 32 layers, 32 KV heads, head_dim 128.
    "llama2-7b": ModelKVProfile(
        name="llama2-7b", layers=32, kv_heads=32, head_dim=128, attention_type="classic",
        source="Llama-2 7B config (MHA)",
    ),
    # Llama-3 70B: 80 layers, 8 KV heads (GQA), head_dim 128.
    "llama3-70b": ModelKVProfile(
        name="llama3-70b", layers=80, kv_heads=8, head_dim=128, attention_type="gqa",
        source="Llama-3 70B config (GQA)",
    ),
    # Mistral-7B: 32 layers, 8 KV heads (GQA), head_dim 128.
    "mistral-7b": ModelKVProfile(
        name="mistral-7b", layers=32, kv_heads=8, head_dim=128, attention_type="gqa",
        source="Mistral-7B config (GQA)",
    ),
    # A reduced-KV MQA proxy (single KV head) for sensitivity studies.
    "mqa-7b": ModelKVProfile(
        name="mqa-7b", layers=32, kv_heads=1, head_dim=128, attention_type="mqa",
        source="MQA architecture (single KV head); illustrative",
        source_type=INFERRED, confidence="low",
    ),
}

DEFAULT_MODEL_KV_PROFILE = "llama3-8b"


# ---------------------------------------------------------------------------
# Migration / rerouting / drain / cold-start parameter registry
# ---------------------------------------------------------------------------
# Added for the migration-realism upgrade (driven by migration.py). These price
# the operational cost of moving a workload: Kubernetes-style drain, cold-start
# decomposition, request rerouting + proxy bottlenecks, batching disruption,
# tail-latency uplift, and phased-rollout / governor behaviour. As elsewhere,
# most are HEURISTIC/INFERRED priors anchored to documented defaults (e.g. the
# K8s 30s termination grace period); none are MEASURED on a live cluster. They
# make migration *expensive and risky* so naive arbitrage can lose.

MIGRATION_PARAMS: dict[str, CalibratedParam] = {
    # --- Kubernetes drain (seconds) --------------------------------------
    "drain_evict_seconds": _h(
        5.0,
        "Eviction API delay before a pod begins terminating (cordon + evict "
        "admission). Calibrate from real kubectl drain traces.",
        source_type=INFERRED, source="K8s eviction API / drain behaviour",
    ),
    "drain_grace_seconds": _h(
        30.0,
        "Graceful termination window (terminationGracePeriodSeconds). K8s default "
        "is 30s; actual shutdown may finish earlier (modelled as a truncated "
        "right-skew, NOT a fixed downtime).",
        source_type=DOCUMENTED, source="K8s default terminationGracePeriodSeconds=30s",
        confidence="medium",
    ),
    "drain_grace_skew": _h(
        0.5,
        "Lognormal sigma for the graceful-termination window (right-skew). Higher "
        "= heavier tail of slow shutdowns. Calibrate from real shutdown timings.",
    ),
    "drain_rebind_seconds": _h(
        10.0,
        "Scheduling + rebinding delay before the rescheduled pod is admitted on a "
        "new node. Calibrate from real scheduler latency.",
        source_type=INFERRED, source="K8s scheduler rebind latency",
    ),

    # --- Request rerouting / proxy (seconds / rps) -----------------------
    "reroute_network_rtt_ms": _h(
        50.0,
        "Default cross-route network RTT added on reroute when no per-region "
        "latency is configured. Scenario network_latency_to overrides this.",
        source_type=INFERRED, source="inter-region RTT (varies widely)",
    ),
    "reroute_replica_accept_ms": _h(
        20.0,
        "Time for a destination replica to accept a rerouted request (connection "
        "+ admission). Calibrate from real ingress accept latency.",
    ),
    "proxy_capacity_rps_per_replica": _h(
        80.0,
        "Per-replica proxy/ingress request capacity before queueing. Replica "
        "count alone does NOT determine throughput — the proxy can bottleneck. "
        "Calibrate from real ingress/router saturation tests.",
        source_type=INFERRED, source="ingress/proxy concurrency limits",
    ),
    "proxy_saturation_convexity": _h(
        2.0,
        "Convexity of proxy queue amplification past capacity (1/(1-load))^k. "
        "Calibrate from real proxy latency-vs-load curves.",
    ),

    # --- Cold-start distribution shape -----------------------------------
    "coldstart_lognormal_sigma": _h(
        0.6,
        "Lognormal sigma applied to each cold-start stage so startup is "
        "heavy-tailed (NOT a single Gaussian). Higher = heavier tail.",
        source_type=INFERRED, source="serverless/GPU cold-start latency is heavy-tailed",
    ),
    "coldstart_firstcompile_prob": _h(
        0.15,
        "Probability a cold start hits the first-compile path (kernel/graph "
        "compilation not cached) → bimodal startup. Calibrate per engine/runtime.",
    ),
    "coldstart_firstcompile_mult": _h(
        4.0,
        "Multiplier on the warmup/compile stage when the first-compile path is "
        "hit. TensorRT-style engines can be far worse. Calibrate per engine.",
    ),
    "scale_from_zero_ttft_mult": _h(
        3.0,
        "Extra TTFT amplification when scaling FROM ZERO (no warm replica to "
        "absorb the queue while the first replica starts). Calibrate from real "
        "scale-from-zero incidents.",
    ),

    # --- Batching disruption under churn ---------------------------------
    "batch_churn_floor": _h(
        0.4,
        "Floor on batching efficiency η_batch under maximal reroute churn (decode "
        "cohorts fragmented, batch occupancy collapses). Calibrate from real "
        "churn-vs-throughput tests.",
        source_type=INFERRED, source="continuous-batching cohort fragmentation",
    ),
    "batch_churn_sensitivity": _h(
        0.5,
        "How fast η_batch falls toward the floor as churn rises (per recent "
        "migration). Calibrate from real reroute-churn throughput data.",
    ),

    # --- Migration tail uplift -------------------------------------------
    "tail_uplift_base": _h(
        1.2,
        "Baseline p95/p99 uplift multiplier for a single clean migration. "
        "Migration is NOT p50-only degradation. Calibrate from rollout tail data.",
        source_type=INFERRED, source="rollout p99 instability",
    ),
    "tail_uplift_max": _h(
        8.0,
        "Maximum p95/p99 uplift under combined rollout instability + queue "
        "pressure + churn + cache loss. Calibrate from real rollout incidents.",
    ),

    # --- Autoscaling scale-up (seconds) ----------------------------------
    "scaleup_scheduling_seconds": _h(
        8.0,
        "Scheduling delay for a scale-up pod (queue + bind). Adds to image-pull + "
        "model-load + warmup. Calibrate from real HPA/KEDA scale-up latency.",
        source_type=INFERRED, source="K8s scheduling + HPA reaction latency",
    ),

    # --- Phased rollout / governor ---------------------------------------
    "rollout_hold_ticks": _h(
        1.0,
        "Stabilization hold (ticks) at each phased-rollout step before advancing "
        "traffic fraction. Calibrate from real canary hold windows.",
        source_type=INFERRED, source="canary/blue-green stabilization windows",
        confidence="medium",
    ),
    "rollback_p99_budget_mult": _h(
        2.0,
        "Rollback trigger: if p99 exceeds this multiple of the SLA budget during "
        "a rollout phase, roll back. Calibrate from real rollback policies.",
    ),
    "governor_queue_pressure_qdepth": _h(
        2000.0,
        "Queue-depth threshold above which the migration governor vetoes a "
        "non-essential migration (do-nothing is safer under queue pressure). "
        "Calibrate from real overload thresholds.",
    ),

    # --- Warm pools -------------------------------------------------------
    "warm_pool_idle_power_frac": _h(
        0.35,
        "Idle power draw of a warm-pool replica as a fraction of full TDP (kept "
        "loaded/ready). Warm pools trade energy for startup safety. Calibrate "
        "from real idle-but-loaded GPU power.",
        source_type=INFERRED, source="loaded-idle GPU power draw",
        confidence="medium",
    ),
}


# ---------------------------------------------------------------------------
# Engine-specific cold-start profiles (seconds per stage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineStartupProfile:
    """Cold-start decomposition for a serving engine (mean seconds per stage).

    T_cold = T_node + T_pull + T_load + T_gpu_transfer + T_warmup
    compile_heavy engines (TensorRT-LLM) carry a large T_warmup and a higher
    first-compile penalty (graph build / kernel compilation can be multi-minute).
    Warm runtimes (vLLM, SGLang) skip the compile stage. Stage means are
    order-of-magnitude operational anchors, NOT measured per-cluster numbers.
    """
    name: str
    t_node: float          # node provisioning (0 if pre-provisioned pool)
    t_pull: float          # container image pull
    t_load: float          # weight load / deserialization
    t_gpu_transfer: float  # host→GPU weight transfer + allocation
    t_warmup: float        # graph capture / kernel compile / runtime warmup
    compile_heavy: bool = False
    source: str = "engine docs / public startup reports"
    source_type: str = INFERRED
    confidence: str = "low"

    def total_mean_seconds(self) -> float:
        return self.t_node + self.t_pull + self.t_load + self.t_gpu_transfer + self.t_warmup


ENGINE_STARTUP_PROFILES: dict[str, EngineStartupProfile] = {
    # vLLM: fast warm start, CUDA-graph capture but no AOT compile.
    "vllm": EngineStartupProfile(
        name="vllm", t_node=0.0, t_pull=15.0, t_load=25.0, t_gpu_transfer=10.0,
        t_warmup=15.0, compile_heavy=False,
        source="vLLM startup (image pull + weight load + CUDA graph capture)",
    ),
    # TensorRT-LLM: compilation/engine-build heavy → multi-minute cold path.
    "tensorrt-llm": EngineStartupProfile(
        name="tensorrt-llm", t_node=0.0, t_pull=20.0, t_load=30.0, t_gpu_transfer=15.0,
        t_warmup=180.0, compile_heavy=True,
        source="TensorRT-LLM engine build / graph compilation (compile-heavy)",
    ),
    # SGLang: warm runtime, RadixAttention; moderate warmup.
    "sglang": EngineStartupProfile(
        name="sglang", t_node=0.0, t_pull=15.0, t_load=25.0, t_gpu_transfer=10.0,
        t_warmup=20.0, compile_heavy=False,
        source="SGLang startup (warm runtime)",
    ),
    # Triton: model-repo load; warmup configurable.
    "triton": EngineStartupProfile(
        name="triton", t_node=0.0, t_pull=18.0, t_load=28.0, t_gpu_transfer=12.0,
        t_warmup=25.0, compile_heavy=False,
        source="Triton Inference Server model load + warmup",
    ),
    # Ray Serve: actor scheduling + replica init on top of the engine.
    "ray_serve": EngineStartupProfile(
        name="ray_serve", t_node=0.0, t_pull=15.0, t_load=25.0, t_gpu_transfer=10.0,
        t_warmup=30.0, compile_heavy=False,
        source="Ray Serve replica actor init + model load",
    ),
}

DEFAULT_ENGINE_PROFILE = "vllm"


def serving_value(name: str, config: dict | None = None) -> float:
    """Return a serving parameter's value, allowing per-run config override.

    Any uncertain assumption is therefore configurable (audit requirement):
    ``config={'saturation_convexity': 1.5}`` overrides the registry default.
    """
    if config and name in config:
        return float(config[name])
    return float(SERVING_PARAMS[name].value)


def kv_value(name: str, config: dict | None = None) -> float:
    """Return a KV-cache parameter's value, allowing per-run config override.

    Mirrors ``serving_value`` for the KV_CACHE_PARAMS registry so every KV/cache
    assumption is configurable: ``config={'kv_pressure_throttling': 0.85}``.
    """
    if config and name in config:
        return float(config[name])
    return float(KV_CACHE_PARAMS[name].value)


def resolve_kv_profile(name: str | None, config: dict | None = None) -> ModelKVProfile:
    """Resolve a model KV profile by name, applying a KV-quant override.

    ``config={'kv_bytes_per_elem': 1.0}`` re-prices the profile at FP8 without
    needing a separate profile entry (the quantization lever).
    """
    prof = MODEL_KV_PROFILES.get(name or DEFAULT_MODEL_KV_PROFILE,
                                 MODEL_KV_PROFILES[DEFAULT_MODEL_KV_PROFILE])
    if config and "kv_bytes_per_elem" in config:
        from dataclasses import replace
        prof = replace(prof, bytes_per_elem=float(config["kv_bytes_per_elem"]))
    return prof


def migration_value(name: str, config: dict | None = None) -> float:
    """Return a migration parameter's value, allowing per-run config override.

    Mirrors ``serving_value`` for the MIGRATION_PARAMS registry so every
    migration/drain/cold-start assumption is configurable.
    """
    if config and name in config:
        return float(config[name])
    return float(MIGRATION_PARAMS[name].value)


def resolve_engine_profile(name: str | None) -> EngineStartupProfile:
    """Resolve an engine cold-start profile by name (default vLLM)."""
    return ENGINE_STARTUP_PROFILES.get(
        (name or DEFAULT_ENGINE_PROFILE).lower(),
        ENGINE_STARTUP_PROFILES[DEFAULT_ENGINE_PROFILE],
    )


# Combined registry: every tunable constant is inspectable in one place.
ALL_PARAMS: dict[str, CalibratedParam] = {**SERVING_PARAMS, **KV_CACHE_PARAMS, **MIGRATION_PARAMS}


def calibration_table() -> list[dict[str, Any]]:
    """Inspectable list of ALL serving + KV-cache + migration parameters."""
    rows: list[dict[str, Any]] = []
    for group, registry in (
        ("serving", SERVING_PARAMS),
        ("kv_cache", KV_CACHE_PARAMS),
        ("migration", MIGRATION_PARAMS),
    ):
        for k, v in sorted(registry.items()):
            rows.append({"name": k, "group": group, **v.to_dict()})
    return rows


def model_profile_table() -> list[dict[str, Any]]:
    """Inspectable list of model KV-architecture profiles with provenance."""
    return [
        {
            "name": p.name,
            "layers": p.layers,
            "kv_heads": p.kv_heads,
            "head_dim": p.head_dim,
            "attention_type": p.attention_type,
            "bytes_per_elem": p.bytes_per_elem,
            "kv_bytes_per_token": p.kv_bytes_per_token(),
            "source": p.source,
            "source_type": p.source_type,
            "confidence": p.confidence,
        }
        for p in sorted(MODEL_KV_PROFILES.values(), key=lambda x: x.name)
    ]


def engine_profile_table() -> list[dict[str, Any]]:
    """Inspectable list of engine cold-start profiles with provenance."""
    return [
        {
            "name": p.name,
            "t_node": p.t_node,
            "t_pull": p.t_pull,
            "t_load": p.t_load,
            "t_gpu_transfer": p.t_gpu_transfer,
            "t_warmup": p.t_warmup,
            "compile_heavy": p.compile_heavy,
            "total_mean_seconds": p.total_mean_seconds(),
            "source": p.source,
            "source_type": p.source_type,
            "confidence": p.confidence,
        }
        for p in sorted(ENGINE_STARTUP_PROFILES.values(), key=lambda x: x.name)
    ]
