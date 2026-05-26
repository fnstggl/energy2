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


# ---------------------------------------------------------------------------
# Thermal / cooling / power parameter registry
# ---------------------------------------------------------------------------
# Added for the thermal-realism upgrade (driven by thermal.py). These model GPU
# board power saturation, thermal inertia, rack-level heat accumulation, hotspot
# formation, cooling regimes, and continuous thermal/power slowdown. As before,
# most are HEURISTIC/INFERRED priors anchored to documented behaviour (e.g. the
# ~30-40 kW/rack air-cooling envelope, H100 ~83°C throttle onset); none are
# MEASURED on a live cluster. They make dense placement thermally risky and
# cooling regimes matter.

THERMAL_PARAMS: dict[str, CalibratedParam] = {
    # --- Board power curve ------------------------------------------------
    "power_curve_k": _h(
        4.0,
        "Saturation rate k in P(u)=P_idle+(P_max-P_idle)(1-exp(-k·u)). Higher = "
        "power saturates earlier in utilization. Calibrate from real power-vs-util "
        "curves per GPU.",
        source_type=INFERRED, source="GPU board power saturates with utilization",
    ),
    "power_idle_frac": _h(
        0.30,
        "Idle board power as a fraction of TDP (P_idle = frac·P_max). Calibrate "
        "from real idle draw per GPU class.",
        source_type=INFERRED, source="GPU idle power ≈ 25-35% of TDP",
        confidence="medium",
    ),
    # workload power multipliers (relative draw at equal utilization)
    "power_mult_inference": _h(
        1.0, "Power multiplier for inference workloads (reference).",
        source_type=INFERRED, source="workload-dependent board power",
    ),
    "power_mult_training": _h(
        1.15, "Training draws more sustained board power than inference at equal "
        "util (dense matmul + comm). Calibrate from real job power.",
        source_type=INFERRED, source="training is power-denser than inference",
    ),
    "power_mult_memory_bound": _h(
        0.85, "Memory-bound workloads draw less compute power at equal util. "
        "Calibrate from real memory-bound job power.",
        source_type=INFERRED, source="memory-bound jobs are less power-dense",
    ),

    # --- Thermal inertia (temperature evolution) -------------------------
    "thermal_alpha": _h(
        0.039,
        "Heat-accumulation coefficient a in T_{t+1}=T_t+a·P−b·(T−T_amb)+ε (°C per "
        "watt per tick). Calibrated so a full-power A100 (400W, air) settles ~50°C "
        "above inlet (≈72°C @ 22°C inlet) with b below. Per-GPU-class alpha in "
        "GPU_POWER_CLASSES overrides this. Calibrate from real heat-up curves.",
        source_type=INFERRED, source="lumped-capacitance thermal model",
    ),
    "thermal_beta_air": _h(
        0.30,
        "Cooling coefficient b for AIR cooling (fraction of (T−T_amb) removed per "
        "tick). b≈0.3 gives a ~3-4 tick thermal time constant (inertia / recovery "
        "lag). Lower b = slower recovery. Calibrate from real cool-down curves.",
        source_type=INFERRED, source="Newton's law of cooling (air)",
    ),
    "thermal_noise_c": _h(
        0.4,
        "Std-dev of per-tick thermal noise ε (°C). Board-to-board variation. "
        "Calibrate from real per-GPU temperature variance.",
    ),

    # --- Rack density / hotspots -----------------------------------------
    "rack_density_elevated_kw": _h(
        20.0,
        "Per-rack kW above which hotspot probability and airflow instability start "
        "rising (air cooling). Operational heuristic (~20 kW), NOT universal.",
        source_type=INFERRED, source="air-cooled rack envelope (~15-25 kW)",
        confidence="medium",
    ),
    "rack_density_critical_kw": _h(
        30.0,
        "Per-rack kW above which hotspot/throttle risk rises sharply (air). "
        "Operational heuristic (~30 kW), NOT universal.",
        source_type=INFERRED, source="air-cooled rack limit (~30-40 kW)",
        confidence="medium",
    ),
    "hotspot_persistence": _h(
        0.85,
        "Per-tick persistence of an existing hotspot (EMA retention). High = "
        "hotspots linger after load drops (recovery lag). Calibrate from real "
        "hotspot decay.",
        source_type=INFERRED, source="thermal recirculation persistence",
    ),
    "hotspot_recirc_penalty_c": _h(
        8.0,
        "Max extra inlet °C from recirculation in a saturated/critical-density "
        "rack. Calibrate from real hot-aisle recirculation.",
    ),
    "airflow_penalty_c": _h(
        4.0,
        "Max extra °C from degraded airflow at full density. Calibrate from real "
        "airflow-vs-temperature data.",
    ),

    # --- Throttling (continuous slowdown) --------------------------------
    "thermal_slowdown_max": _h(
        0.3,
        "Max thermal throughput slowdown fraction s_thermal as temperature goes "
        "from throttle-onset to max. NOT a binary flag. Real GPUs typically lose "
        "~10-30% throughput to clock throttling before hard limits. Calibrate "
        "from real clock-throttle-vs-temp curves.",
        source_type=INFERRED, source="GPU clock throttling above thermal limit",
        confidence="medium",
    ),
    "power_slowdown_max": _h(
        0.3,
        "Max power-cap throughput slowdown fraction s_power when board power is "
        "pinned at the cap. Calibrate from real power-capped throughput.",
        source_type=INFERRED, source="power-cap clock reduction",
    ),
    "inlet_variance_c": _h(
        1.5,
        "Std-dev of local inlet temperature variation across a rack (°C). "
        "Calibrate from real inlet sensor spread.",
    ),

    # --- Thermal telemetry / migration risk ------------------------------
    "thermal_telemetry_missing_risk": _h(
        0.5,
        "Risk inflation when thermal telemetry is missing/stale (missing ≠ safe). "
        "Heuristic policy lever; raises migration conservatism.",
        source_type=HEURISTIC, confidence="low",
    ),
    "thermal_migration_hot_veto_c": _h(
        78.0,
        "Destination rack inlet/GPU °C above which migrating INTO the zone is "
        "vetoed by the thermal governor. Below the throttle onset to leave "
        "headroom. Calibrate from real safe-inlet targets.",
        source_type=INFERRED, source="leave thermal headroom below throttle onset",
        confidence="medium",
    ),
}


# ---------------------------------------------------------------------------
# GPU power/thermal classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUPowerClass:
    """Per-GPU-class power + thermal response (distinct, NOT one-size-fits-all)."""
    name: str
    p_max_w: float
    throttle_onset_c: float
    max_temp_c: float
    alpha: float            # heat-accumulation coefficient (overrides default)
    source: str = "vendor datasheet (TDP / throttle temp)"
    source_type: str = DOCUMENTED
    confidence: str = "medium"


# TDP / throttle temps are datasheet values; alpha (thermal response) is inferred
# and calibrated so each class settles ~50-55°C above inlet at full power (with
# the air beta), leaving headroom to throttle when inlet/hotspots push it up.
GPU_POWER_CLASSES: dict[str, GPUPowerClass] = {
    "h100-sxm": GPUPowerClass("h100-sxm", 700.0, 83.0, 90.0, 0.0223,
                              source="NVIDIA H100 SXM5 700W TDP, ~83-87°C throttle"),
    "h100-pcie": GPUPowerClass("h100-pcie", 350.0, 83.0, 90.0, 0.0446,
                               source="NVIDIA H100 PCIe 350W TDP"),
    "a100": GPUPowerClass("a100", 400.0, 83.0, 90.0, 0.039,
                          source="NVIDIA A100 SXM 400W TDP"),
    "l40s": GPUPowerClass("l40s", 350.0, 87.0, 92.0, 0.0446,
                          source="NVIDIA L40S 350W TDP"),
    "l4": GPUPowerClass("l4", 72.0, 80.0, 90.0, 0.217,
                        source="NVIDIA L4 72W TDP"),
}

DEFAULT_POWER_CLASS = "a100"


# ---------------------------------------------------------------------------
# Cooling-regime profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoolingRegime:
    """Cooling regime: alters recovery rate, hotspot variance, density tolerance.

    beta_mult scales the cooling coefficient (higher = faster recovery, more
    headroom); hotspot_mult scales hotspot probability/variance; density_mult
    scales the kW thresholds at which a rack enters elevated/critical regimes.
    Liquid cooling improves all three but does NOT eliminate thermal risk.
    """
    name: str
    beta_mult: float
    hotspot_mult: float
    density_mult: float
    inlet_variance_mult: float
    source: str = "cooling engineering (air vs liquid vs hybrid)"
    source_type: str = INFERRED
    confidence: str = "low"


COOLING_REGIMES: dict[str, CoolingRegime] = {
    "air": CoolingRegime("air", 1.0, 1.0, 1.0, 1.0,
                         source="baseline air cooling", confidence="medium"),
    "liquid": CoolingRegime("liquid", 2.2, 0.35, 2.5, 0.4,
                            source="direct-to-chip liquid cooling (higher heat "
                            "transfer, higher density tolerance, residual risk)"),
    "hybrid": CoolingRegime("hybrid", 1.5, 0.6, 1.6, 0.7,
                            source="rear-door / hybrid air+liquid"),
    "hot_aisle_containment": CoolingRegime("hot_aisle_containment", 1.2, 0.8, 1.3, 0.85,
                                           source="hot-aisle containment (better "
                                           "air management)"),
    "weak_airflow": CoolingRegime("weak_airflow", 0.6, 1.8, 0.6, 1.6,
                                  source="degraded/weak airflow environment"),
}

DEFAULT_COOLING_REGIME = "air"


def thermal_value(name: str, config: dict | None = None) -> float:
    """Return a thermal parameter's value, allowing per-run config override."""
    if config and name in config:
        return float(config[name])
    return float(THERMAL_PARAMS[name].value)


def resolve_power_class(name: str | None) -> GPUPowerClass:
    """Resolve a GPU power class by name (default A100)."""
    return GPU_POWER_CLASSES.get(
        (name or DEFAULT_POWER_CLASS).lower(), GPU_POWER_CLASSES[DEFAULT_POWER_CLASS]
    )


def power_class_for_model(model_name: str) -> GPUPowerClass:
    """Map a GPUProfile.model_name to a power class (substring heuristic)."""
    m = (model_name or "").lower()
    if "h100" in m and "pcie" in m:
        return GPU_POWER_CLASSES["h100-pcie"]
    if "h100" in m:
        return GPU_POWER_CLASSES["h100-sxm"]
    if "l40" in m:
        return GPU_POWER_CLASSES["l40s"]
    if "l4" in m:
        return GPU_POWER_CLASSES["l4"]
    return GPU_POWER_CLASSES["a100"]


def resolve_cooling_regime(name: str | None) -> CoolingRegime:
    """Resolve a cooling regime by name (default air)."""
    return COOLING_REGIMES.get(
        (name or DEFAULT_COOLING_REGIME).lower(), COOLING_REGIMES[DEFAULT_COOLING_REGIME]
    )


# ---------------------------------------------------------------------------
# Topology / communication / placement parameter registry
# ---------------------------------------------------------------------------
# Added for the topology-realism upgrade (driven by topology.py). These price
# the communication cost of a placement: latency-bandwidth message cost
# (T = alpha + m/B_eff), ring/tree collective amplification, small-message
# latency amplification, fabric contention/congestion, synchronization
# penalties, communication-induced tail latency, topology telemetry confidence,
# and a topology-aware migration veto. As elsewhere, the great majority are
# HEURISTIC/INFERRED priors anchored to documented mechanisms (e.g. NVLink vs
# PCIe vs InfiniBand bandwidth/latency regimes, ring-all-reduce scaling); NONE
# are MEASURED on a live cluster. They make topology-aware placement materially
# matter so bad placement can collapse throughput and synchronization-heavy jobs
# become hard to move.

TOPOLOGY_PARAMS: dict[str, CalibratedParam] = {
    # --- Small-message latency amplification -----------------------------
    "small_message_bytes": _h(
        262144.0,
        "Payload size (bytes) below which fixed per-message latency alpha "
        "dominates over m/B_eff (small-message regime). ~256 KiB is the rough "
        "NCCL latency/bandwidth crossover; calibrate from real message-size "
        "sweeps per fabric.",
        source_type=INFERRED, source="NCCL latency-bound vs bandwidth-bound crossover",
        confidence="low",
    ),
    "small_message_amp_max": _h(
        3.0,
        "Max latency amplification for tiny messages where alpha and protocol "
        "overhead dominate (latency-bound collectives pay this). Calibrate from "
        "real small-message all-reduce latency.",
        source_type=INFERRED, source="latency-bound small-message collectives",
    ),

    # --- Fabric contention / congestion ----------------------------------
    "congestion_onset": _h(
        0.60,
        "Link utilization fraction above which effective bandwidth starts "
        "degrading and queueing delay rises (oversubscription onset). "
        "Operational heuristic, NOT a universal limit. Calibrate from real "
        "fabric utilization-vs-latency curves.",
        source_type=INFERRED, source="fabric oversubscription / queueing onset",
        confidence="low",
    ),
    "congestion_convexity": _h(
        2.0,
        "Convexity k of congestion amplification (1/(1-load))^k past the onset. "
        "Higher = bandwidth collapses faster under saturation. Calibrate from "
        "real congestion-collapse tests.",
        source_type=INFERRED, source="queueing amplification under saturation",
    ),
    "congestion_bw_floor": _h(
        0.20,
        "Floor on effective-bandwidth fraction under full congestion collapse "
        "(a saturated fabric still moves SOME data). Calibrate from real "
        "saturated-fabric goodput.",
        source_type=INFERRED, source="goodput floor under congestion collapse",
    ),
    "nic_congestion_onset": _h(
        0.55,
        "Per-NIC throughput fraction above which NIC queueing / incast amplifies "
        "cross-node latency. Cross-node collectives bottleneck on the NIC before "
        "the intra-node fabric. Calibrate from real NIC saturation tests.",
        source_type=INFERRED, source="NIC incast / queueing onset",
        confidence="low",
    ),

    # --- Collective amplification ----------------------------------------
    "collective_amp_max": _h(
        6.0,
        "Max collective-communication amplification of point-to-point cost at "
        "large N under poor placement (ring all-reduce across many "
        "cross-rack hops). Calibrate from real collective scaling sweeps.",
        source_type=INFERRED, source="ring/tree all-reduce scaling with N and hops",
    ),
    "allreduce_alpha_term_weight": _h(
        1.0,
        "Weight on the latency (alpha) term in the ring-all-reduce approximation "
        "T_ring = 2(N-1)/N*(m/B_eff) + w*2(N-1)*alpha. Latency-bound collectives "
        "(small m, many ranks) are dominated by this. Calibrate per fabric.",
        source_type=INFERRED, source="ring all-reduce cost model (Rabenseifner/NCCL)",
        confidence="medium",
    ),

    # --- Synchronization penalty -----------------------------------------
    "sync_penalty_max": _h(
        0.50,
        "Max throughput slowdown fraction for a synchronization-heavy workload "
        "under worst-case topology (stragglers stall the whole collective; the "
        "slowest rank sets the pace). Calibrate from real straggler/sync-stall "
        "telemetry.",
        source_type=INFERRED, source="bulk-synchronous straggler stalls",
    ),
    "sync_straggler_jitter": _h(
        0.15,
        "Std-dev of per-rank straggler jitter (fraction) injected into "
        "synchronization-heavy collectives. Bulk-synchronous steps wait on the "
        "slowest rank, so jitter amplifies tail. Calibrate from real per-rank "
        "step-time variance.",
    ),

    # --- Communication-induced tail latency ------------------------------
    "comm_tail_p95_base": _h(
        1.3,
        "p95/p50 communication-latency tail multiplier at GOOD topology / low "
        "congestion. Grows toward comm_tail_max as topology degrades. Calibrate "
        "from real collective-latency histograms.",
        source_type=INFERRED, source="collective-latency tail behaviour",
        confidence="low",
    ),
    "comm_tail_p99_base": _h(
        1.8,
        "p99/p50 communication-latency tail multiplier at good topology. Grows "
        "convexly toward comm_tail_max under congestion/poor placement.",
        source_type=INFERRED, source="collective-latency tail behaviour",
        confidence="low",
    ),
    "comm_tail_max": _h(
        10.0,
        "Max communication-latency tail multiplier under congestion collapse / "
        "cross-rack synchronization instability (p99 blows up faster than the "
        "mean). Calibrate from real degraded-topology tail incidents.",
        source_type=INFERRED, source="topology-induced tail blow-up",
    ),

    # --- Topology score / penalty normalization --------------------------
    "topology_penalty_mu": _h(
        1.0,
        "Normalization mu in the communication penalty P = sum_k lambda_k*(m_k/B_k"
        "+alpha_k)/(mu*S). Scales the absolute penalty so it is comparable to "
        "energy savings; tune so comm penalties can OUTWEIGH arbitrage gains for "
        "sync-heavy jobs. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "topology_throughput_penalty_max": _h(
        0.85,
        "Max throughput slowdown fraction from a catastrophic placement of a "
        "highly communication-bound workload (e.g. tensor-parallel split across "
        "racks). Bad placement can collapse throughput, not just nudge it. "
        "Calibrate from real cross-domain TP throughput cliffs.",
        source_type=INFERRED, source="tensor-parallel cross-domain throughput collapse",
    ),
    "tp_instability_score": _h(
        0.45,
        "Topology-quality score (0-1) below which a tensor-parallel / sync-heavy "
        "workload is considered UNSTABLE (collective instability risk, runaway "
        "tails). Operational heuristic, NOT universal.",
        source_type=INFERRED, source="TP collectives unstable off-NVSwitch (inferred)",
        confidence="low",
    ),
    "moe_hotspot_amp": _h(
        2.5,
        "All-to-all hotspot amplification for MoE / expert-parallel traffic under "
        "congestion (expert popularity skew creates incast hotspots). Calibrate "
        "from real MoE dispatch/combine traffic.",
        source_type=INFERRED, source="MoE all-to-all expert-popularity hotspots",
    ),

    # --- Topology telemetry confidence -----------------------------------
    "topology_telemetry_missing_risk": _h(
        0.5,
        "Risk inflation when topology telemetry is missing/stale (missing "
        "topology must NOT be read as ideal proximity). Heuristic policy lever; "
        "raises placement conservatism and migration vetoes.",
        source_type=HEURISTIC, confidence="low",
    ),
    "topology_confidence_min_score": _h(
        0.6,
        "Floor applied to a placement's usable topology score under LOW telemetry "
        "confidence (we assume it MIGHT be a poor placement, so we discount the "
        "optimistic reading). Heuristic conservatism lever.",
        source_type=HEURISTIC, confidence="low",
    ),

    # --- Topology-aware migration veto -----------------------------------
    "migration_veto_distance": _h(
        4.0,
        "Topology-distance ladder rung (same-rack=4, cross-rack=5, "
        "cross-region=6) at/above which migrating a communication-sensitive or "
        "synchronization-heavy workload is vetoed (the move would break fabric "
        "locality). Calibrate from real placement-locality policies.",
        source_type=INFERRED, source="topology-locality preservation policy",
        confidence="low",
    ),
    "migration_veto_comm_weight": _h(
        0.5,
        "Communication-weight (lambda) threshold above which the distance-based "
        "topology migration veto applies. Low-comm jobs migrate freely; high-comm "
        "/ sync-heavy jobs are pinned. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),

    # --- Stochastic variation --------------------------------------------
    "collective_jitter_frac": _h(
        0.08,
        "Std-dev of multiplicative collective-latency jitter per tick (routing "
        "variation, NIC contention, scheduling). Topology behaviour is NOT one "
        "deterministic curve. Calibrate from real collective-latency variance.",
    ),
    "routing_variation_frac": _h(
        0.05,
        "Std-dev of multiplicative effective-bandwidth jitter from adaptive "
        "routing / path variation per tick. Calibrate from real fabric path "
        "variance.",
    ),
}


# ---------------------------------------------------------------------------
# Topology distance ladder (tunable engineering heuristic, NOT universal truth)
# ---------------------------------------------------------------------------
# d(i,j) in the topology score S = sum(w_ij * d(i,j)). Lower = closer = better.
# These rungs are deliberately ordinal: the *gaps* are tuned so that crossing a
# domain boundary (e.g. node->rack, rack->region) is materially more expensive
# than the one before, NOT measured hop counts.

TOPOLOGY_DISTANCE_LADDER: dict[str, int] = {
    "intra_gpu": 0,        # same GPU
    "nvswitch": 1,         # same NVSwitch domain (fully connected)
    "nvlink": 1,           # same NVLink domain (point-to-point links)
    "pcie_root": 2,        # same PCIe root complex
    "socket": 3,           # same NUMA socket (cross PCIe root, same socket)
    "node": 3,             # same node (cross-socket within a box)
    "rack": 4,             # same rack, different node
    "cross_rack": 5,       # different rack, same region
    "cross_region": 6,     # different region
}


@dataclass(frozen=True)
class FabricRegime:
    """A communication regime: distance rung + effective bandwidth + latency.

    b_eff_gbps is one-directional effective payload bandwidth (GB/s) AFTER
    protocol overhead; latency_us is the fixed per-message latency term alpha
    (microseconds). congestion_sensitivity scales how fast b_eff degrades under
    contention. These are regime PRIORS spanning NVSwitch -> NVLink -> PCIe ->
    same-rack IB -> cross-rack -> cross-region, NOT measured per-fabric numbers;
    the point is that the regimes differ MATERIALLY (orders of magnitude in
    latency, large bandwidth steps), so links are NOT interchangeable pipes.
    """
    name: str
    distance: int
    b_eff_gbps: float
    latency_us: float
    congestion_sensitivity: float = 1.0
    source: str = "fabric bandwidth/latency regime (vendor docs + inference)"
    source_type: str = INFERRED
    confidence: str = "low"


# Regime priors. NVSwitch/NVLink bandwidths are SXM-class one-way figures; PCIe
# Gen4/Gen5 ~ 32/64 GB/s effective; same-rack InfiniBand HDR/NDR ~ 25-50 GB/s
# with ~1-2 us latency; cross-rack adds switch hops; cross-region is WAN-class.
FABRIC_REGIMES: dict[str, FabricRegime] = {
    "intra_gpu": FabricRegime(
        "intra_gpu", 0, 2000.0, 0.1, 0.5,
        source="on-device HBM (no inter-GPU transfer)", confidence="medium"),
    "nvswitch": FabricRegime(
        "nvswitch", 1, 450.0, 0.8, 0.6,
        source="NVSwitch all-to-all (H100 SXM ~450 GB/s/dir uni)",
        source_type=DOCUMENTED, confidence="medium"),
    "nvlink": FabricRegime(
        "nvlink", 1, 300.0, 1.0, 0.8,
        source="point-to-point NVLink (A100/H100 partial mesh)",
        source_type=DOCUMENTED, confidence="medium"),
    "pcie_root": FabricRegime(
        "pcie_root", 2, 50.0, 3.0, 1.2,
        source="PCIe Gen5 x16 ~64 GB/s peak, ~50 effective",
        source_type=DOCUMENTED, confidence="medium"),
    "socket": FabricRegime(
        "socket", 3, 32.0, 5.0, 1.4,
        source="cross-PCIe-root within socket (Gen4-class ~32 GB/s)",
        source_type=INFERRED),
    "node": FabricRegime(
        "node", 3, 20.0, 8.0, 1.6,
        source="cross-socket QPI/UPI staging + PCIe (NUMA crossing)",
        source_type=INFERRED),
    "rack": FabricRegime(
        "rack", 4, 25.0, 2.0, 1.8,
        source="same-rack InfiniBand NDR/HDR (~25 GB/s, ~2 us)",
        source_type=DOCUMENTED, confidence="low"),
    "cross_rack": FabricRegime(
        "cross_rack", 5, 12.5, 5.0, 2.5,
        source="cross-rack via spine switches (oversubscribed, extra hops)",
        source_type=INFERRED),
    "cross_region": FabricRegime(
        "cross_region", 6, 1.25, 10000.0, 3.0,
        source="cross-region WAN (ms-class latency, limited bandwidth)",
        source_type=INFERRED),
}

DEFAULT_FABRIC_REGIME = "nvswitch"


@dataclass(frozen=True)
class NVLinkGeneration:
    """NVLink/NVSwitch generation regime (per-GPU aggregate, both directions).

    bidir_gbps is the marketed aggregate NVLink bandwidth per GPU; intra-domain
    point-to-point effective bandwidth is a fraction of this. Distinct
    generations exist precisely so the simulator does NOT treat all NVLink as
    one bandwidth (A100 ~600, H100 ~900, future ~1800 GB/s class).
    """
    name: str
    bidir_gbps: float
    latency_us: float
    source: str = "NVIDIA NVLink generation datasheet"
    source_type: str = DOCUMENTED
    confidence: str = "medium"


NVLINK_GENERATIONS: dict[str, NVLinkGeneration] = {
    "a100": NVLinkGeneration("a100", 600.0, 1.2,
                             source="NVIDIA A100 NVLink3 ~600 GB/s aggregate"),
    "h100": NVLinkGeneration("h100", 900.0, 0.9,
                             source="NVIDIA H100 NVLink4 ~900 GB/s aggregate"),
    "future": NVLinkGeneration("future", 1800.0, 0.7,
                               source="future-generation NVLink (~1.8 TB/s class)",
                               source_type=INFERRED, confidence="low"),
}


@dataclass(frozen=True)
class WorkloadCommProfile:
    """Topology sensitivity of a workload family.

    comm_weight (lambda) is the relative communication intensity used in the
    penalty P; collective names the dominant collective; small_msg_sensitivity
    and tail_sensitivity scale small-message amplification and how hard topology
    degradation hits p95/p99; sync_heavy marks bulk-synchronous workloads whose
    slowest rank sets the pace; nvlink_affinity is how strongly the workload
    benefits from NVLink/NVSwitch locality. These distinguish tensor-parallel
    (strongly prefers NVLink, unstable off it) from batch inference (low average
    sensitivity, still tails under poor topology). Priors, NOT measured.
    """
    name: str
    comm_weight: float
    collective: str               # all_reduce | all_to_all | p2p | tree | none
    small_msg_sensitivity: float
    tail_sensitivity: float
    sync_heavy: bool
    nvlink_affinity: float
    source: str = "workload communication character (inferred)"
    source_type: str = INFERRED
    confidence: str = "low"


WORKLOAD_COMM_PROFILES: dict[str, WorkloadCommProfile] = {
    "tensor_parallel": WorkloadCommProfile(
        "tensor_parallel", 1.0, "all_reduce", 0.9, 0.9, True, 1.0,
        source="TP all-reduce per layer; latency-bound, NVLink-critical"),
    "pipeline_parallel": WorkloadCommProfile(
        "pipeline_parallel", 0.4, "p2p", 0.5, 0.6, True, 0.6,
        source="PP point-to-point activations; bubble-sensitive"),
    "all_reduce_training": WorkloadCommProfile(
        "all_reduce_training", 0.85, "all_reduce", 0.7, 0.85, True, 0.8,
        source="data-parallel gradient all-reduce; bandwidth+latency bound"),
    "moe_expert": WorkloadCommProfile(
        "moe_expert", 0.7, "all_to_all", 0.6, 0.8, True, 0.7,
        source="MoE dispatch/combine all-to-all; hotspot-prone"),
    "embedding": WorkloadCommProfile(
        "embedding", 0.3, "p2p", 0.4, 0.4, False, 0.4,
        source="embedding lookup/exchange; moderate, sharded"),
    "retrieval": WorkloadCommProfile(
        "retrieval", 0.25, "p2p", 0.4, 0.4, False, 0.3,
        source="retrieval/RAG fan-out; moderate small-message"),
    "batch_inference": WorkloadCommProfile(
        "batch_inference", 0.15, "none", 0.2, 0.4, False, 0.2,
        source="batch inference; low average comm, still tails under bad topo"),
    "comm_light_inference": WorkloadCommProfile(
        "comm_light_inference", 0.05, "none", 0.1, 0.3, False, 0.1,
        source="single-GPU / comm-light inference; topology-insensitive"),
}

DEFAULT_COMM_PROFILE = "comm_light_inference"


def topology_value(name: str, config: dict | None = None) -> float:
    """Return a topology parameter's value, allowing per-run config override.

    Mirrors ``thermal_value`` for the TOPOLOGY_PARAMS registry so every topology
    / communication assumption is configurable: ``config={'sync_penalty_max':
    0.4}`` overrides the registry default.
    """
    if config and name in config:
        return float(config[name])
    return float(TOPOLOGY_PARAMS[name].value)


def resolve_fabric_regime(name: str | None) -> FabricRegime:
    """Resolve a fabric regime by name (default NVSwitch)."""
    return FABRIC_REGIMES.get(
        (name or DEFAULT_FABRIC_REGIME).lower(), FABRIC_REGIMES[DEFAULT_FABRIC_REGIME]
    )


def resolve_nvlink_generation(name: str | None) -> NVLinkGeneration:
    """Resolve an NVLink generation by name (default A100)."""
    return NVLINK_GENERATIONS.get((name or "a100").lower(), NVLINK_GENERATIONS["a100"])


def nvlink_generation_for_model(model_name: str) -> NVLinkGeneration:
    """Map a GPUProfile.model_name to an NVLink generation (substring heuristic)."""
    m = (model_name or "").lower()
    if "h100" in m or "h200" in m:
        return NVLINK_GENERATIONS["h100"]
    if "b100" in m or "b200" in m or "gb200" in m:
        return NVLINK_GENERATIONS["future"]
    return NVLINK_GENERATIONS["a100"]


def resolve_comm_profile(
    name: str | None,
    communication_intensity: str | None = None,
    workload_type: str | None = None,
) -> WorkloadCommProfile:
    """Resolve a workload communication profile.

    Explicit ``name`` wins. Otherwise infer from workload_type (embedding/
    training/etc.) then fall back to the coarse communication_intensity
    (low/medium/high) so existing scenarios keep working without a comm_profile.
    """
    if name and name in WORKLOAD_COMM_PROFILES:
        return WORKLOAD_COMM_PROFILES[name]
    wt = (workload_type or "").lower()
    if wt in ("batch_training", "fine_tuning", "training"):
        return WORKLOAD_COMM_PROFILES["all_reduce_training"]
    if wt == "embedding":
        return WORKLOAD_COMM_PROFILES["embedding"]
    ci = (communication_intensity or "low").lower()
    if ci == "high":
        return WORKLOAD_COMM_PROFILES["tensor_parallel"]
    if ci == "medium":
        return WORKLOAD_COMM_PROFILES["all_reduce_training"]
    return WORKLOAD_COMM_PROFILES["comm_light_inference"]


# ---------------------------------------------------------------------------
# Utilization / fragmentation / bin-packing parameter registry
# ---------------------------------------------------------------------------
# Added for the utilization-realism upgrade (driven by utilization.py). These
# model multi-dimensional GPU utilization (SM / DRAM-bandwidth / scheduler /
# PCIe / KV), a roofline-style token-throughput ceiling, continuous-batching
# gains with diminishing returns, KV/VRAM headroom, multidimensional +
# topology-aware fragmentation, stranded capacity, saturating consolidation
# benefit with nonlinear risk, queue amplification under packing, GPU-sharing
# interference, and utilization telemetry confidence. As elsewhere, the great
# majority are HEURISTIC/INFERRED priors anchored to documented behaviour (e.g.
# the ~5% VRAM headroom rule, inference vs training utilization regimes); NONE
# are MEASURED on a live cluster. They make "free GPUs" often unusable, packing
# nonlinearly risky, and utilization a multidimensional systems problem.

UTILIZATION_PARAMS: dict[str, CalibratedParam] = {
    # --- Utilization regimes (triangular priors) --------------------------
    "inference_util_min": _h(
        0.40,
        "Lower vertex of the inference compute-utilization triangular prior. "
        "Inference often runs at moderate SM utilization (memory/queue bound). "
        "Calibrate from real DCGM GPU_UTIL distributions per workload.",
        source_type=INFERRED, source="inference GPU utilization is moderate (memory-bound)",
        confidence="low",
    ),
    "inference_util_mode": _h(
        0.55,
        "Mode of the inference compute-utilization triangular prior. NOT a "
        "universal target. Calibrate from real serving telemetry.",
        source_type=INFERRED, source="inference GPU utilization mode",
        confidence="low",
    ),
    "inference_util_max": _h(
        0.70,
        "Upper vertex of the inference compute-utilization triangular prior. "
        "Calibrate from real saturated-serving telemetry.",
        source_type=INFERRED, source="inference GPU utilization ceiling",
        confidence="low",
    ),
    "training_util_min": _h(
        0.85,
        "Lower vertex of the training active-utilization triangular prior. "
        "Training targets very high sustained SM utilization. Calibrate from "
        "real training-job DCGM telemetry.",
        source_type=INFERRED, source="training targets high sustained utilization",
        confidence="medium",
    ),
    "training_util_mode": _h(
        0.90,
        "Mode of the training active-utilization triangular prior. Calibrate "
        "from real training telemetry.",
        source_type=INFERRED, source="training utilization mode", confidence="medium",
    ),
    "training_util_max": _h(
        0.95,
        "Upper vertex of the training active-utilization triangular prior. "
        "Sustained >95% is rare (comm/sync bubbles). Calibrate per job.",
        source_type=INFERRED, source="training utilization ceiling", confidence="medium",
    ),

    # --- Roofline ceilings (dimensionless utilization caps) ---------------
    "mem_bw_saturation_onset": _h(
        0.75,
        "Memory-bandwidth utilization fraction above which effective token "
        "throughput is bandwidth-bound (decode is memory-bound). Operational "
        "heuristic; calibrate from real DRAM_ACTIVE-vs-throughput curves.",
        source_type=INFERRED, source="LLM decode is memory-bandwidth-bound (roofline)",
        confidence="low",
    ),
    "scheduler_capacity_seqs": _h(
        256.0,
        "Active-sequence count at which the scheduler/service limit S_sched "
        "begins to bind (admission + scheduling overhead). Calibrate from real "
        "scheduler saturation tests per engine.",
        source_type=INFERRED, source="continuous-batching scheduler admission limit",
        confidence="low",
    ),
    "pcie_pressure_onset": _h(
        0.70,
        "PCIe transfer-pressure fraction above which host<->device staging "
        "suppresses effective occupancy (weight/activation/KV paging). "
        "Calibrate from real PCIe counters.",
        source_type=INFERRED, source="PCIe staging suppresses occupancy",
        confidence="low",
    ),

    # --- Continuous batching gain -----------------------------------------
    "batching_gain_cv_coeff": _h(
        1.5,
        "Coefficient a in gain = 1 + a*CV(output_len). Higher output-length "
        "variance → more continuous-batching headroom. Calibrate from real "
        "output-length distributions + throughput.",
        source_type=INFERRED, source="continuous batching benefits from length variance",
        confidence="low",
    ),
    "batching_gain_common_max": _h(
        8.0,
        "Cap on the COMMON continuous-batching throughput-gain regime (~1.5-8x). "
        "Beyond this requires highly favorable conditions. Calibrate from real "
        "vs naive-static-batching throughput.",
        source_type=BENCHMARK_DERIVED,
        source="vLLM/continuous-batching common 1.5-8x over static batching",
        confidence="low",
    ),
    "batching_gain_vendor_max": _h(
        23.0,
        "Long-tail OPTIMISTIC batching gain achievable ONLY under highly "
        "favorable vendor-benchmark conditions. NOT a typical operating point. "
        "Treat as an upper bound, not a target.",
        source_type=BENCHMARK_DERIVED,
        source="vendor continuous-batching benchmark (up to ~23x, favorable)",
        confidence="low",
    ),

    # --- KV / VRAM headroom -----------------------------------------------
    "vram_headroom_frac": _h(
        0.05,
        "Fraction of VRAM kept as reserve headroom (~5%). Aggressive packing "
        "near 100% occupancy becomes unstable (allocator stalls, preemption). "
        "gpu_memory_utilization = 1.0 is NOT safe. Calibrate per engine.",
        source_type=INFERRED, source="~5% VRAM reserve headroom (vLLM operational rule)",
        confidence="medium",
    ),
    "safe_occupancy_max": _h(
        0.95,
        "Safe upper bound on VRAM/KV occupancy before admission suppression + "
        "preemption risk rises. Calibrate from real OOM/preemption incidents.",
        source_type=INFERRED, source="safe KV/VRAM occupancy ceiling",
        confidence="medium",
    ),

    # --- Fragmentation / stranded capacity --------------------------------
    "fragmentation_elevated": _h(
        0.30,
        "Fragmentation score (1 - schedulable/free) above which scheduling "
        "starts failing for large/topology-constrained jobs. Operational "
        "heuristic. Calibrate from real bin-packing failure rates.",
        source_type=INFERRED, source="cluster fragmentation degrades schedulability",
        confidence="low",
    ),
    "fragmentation_critical": _h(
        0.60,
        "Fragmentation score above which large multi-GPU / cross-domain jobs are "
        "effectively unschedulable despite free capacity (stranded islands). "
        "Calibrate from real scheduler reject rates.",
        source_type=INFERRED, source="severe fragmentation strands capacity",
        confidence="low",
    ),

    # --- Consolidation benefit + risk -------------------------------------
    "consolidation_benefit_max": _h(
        0.6,
        "Max fractional idle-capacity benefit B_max from consolidation (saturating "
        "curve benefit = B_max*(1-exp(-k*frac))). Returns diminish. Calibrate "
        "from real consolidation savings.",
        source_type=INFERRED, source="diminishing returns from consolidation",
        confidence="low",
    ),
    "consolidation_benefit_k": _h(
        3.0,
        "Saturation rate k in the consolidation benefit curve. Higher = benefit "
        "saturates earlier in consolidation fraction. Calibrate from real curves.",
        source_type=INFERRED, source="consolidation benefit saturation",
        confidence="low",
    ),
    "consolidation_risk_cross_domain": _h(
        0.30,
        "Risk weight r1 on cross-domain (cross-node/rack) traffic in the "
        "consolidation-risk sum. Cross-node sharding is the dominant packing "
        "risk. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "consolidation_risk_queue": _h(
        0.25,
        "Risk weight r2 on p95 queue pressure in the consolidation-risk sum. "
        "Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "consolidation_risk_thermal": _h(
        0.15,
        "Risk weight r3 on inverse thermal margin in the consolidation-risk sum. "
        "Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "consolidation_risk_kv": _h(
        0.15,
        "Risk weight r4 on KV pressure in the consolidation-risk sum. Heuristic "
        "policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "consolidation_risk_scheduler": _h(
        0.15,
        "Risk weight r5 on scheduler pressure in the consolidation-risk sum. "
        "Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "packing_unsafe_risk": _h(
        0.55,
        "Consolidation-risk threshold above which a packing/consolidation "
        "migration is vetoed as unsafe. Operational heuristic; calibrate from "
        "real post-consolidation incident rates.",
        source_type=HEURISTIC, confidence="low",
    ),

    # --- Queue amplification under packing --------------------------------
    "queue_amp_onset": _h(
        0.70,
        "Packing-density fraction above which queue waiting time amplifies "
        "superlinearly (less slack to absorb bursts). Calibrate from real "
        "density-vs-p95 curves.",
        source_type=INFERRED, source="reduced slack amplifies queueing",
        confidence="low",
    ),
    "queue_amp_convexity": _h(
        2.0,
        "Convexity of queue amplification past the packing-density onset "
        "(1/(1-density))^k. Calibrate from real saturation curves.",
        source_type=INFERRED, source="queueing amplification convexity",
        confidence="low",
    ),

    # --- Underutilization / utilization paradox ---------------------------
    "underutilization_sm_threshold": _h(
        0.50,
        "Compute (SM) utilization below which a GPU is flagged underutilized "
        "(packing opportunity — but NOT a guarantee of safe consolidation). "
        "Calibrate from real idle-detection policies.",
        source_type=INFERRED, source="sustained low SM utilization = packing candidate",
        confidence="low",
    ),
    "paradox_dram_high": _h(
        0.70,
        "DRAM_ACTIVE fraction that, combined with LOW SM utilization, signals the "
        "utilization paradox (high resource use, low throughput — memory-bound). "
        "Calibrate from real DRAM_ACTIVE-vs-GPU_UTIL telemetry.",
        source_type=INFERRED, source="DRAM-bound: high DRAM_ACTIVE + low SM",
        confidence="low",
    ),

    # --- GPU sharing interference -----------------------------------------
    "gpu_sharing_interference": _h(
        0.20,
        "Throughput/latency interference fraction when GPUs are shared "
        "(MIG/time-slice/fractional). Sharing is NOT free — it adds variance + "
        "scheduler complexity. Calibrate from real co-located interference.",
        source_type=INFERRED, source="MIG/time-slice co-location interference",
        confidence="low",
    ),

    # --- Telemetry confidence + stochastic variation ----------------------
    "util_telemetry_missing_risk": _h(
        0.5,
        "Conservatism inflation when utilization/DRAM/scheduler telemetry is "
        "missing or stale (missing != schedulable). Heuristic policy lever; "
        "suppresses risky packing.",
        source_type=HEURISTIC, confidence="low",
    ),
    "util_noise_frac": _h(
        0.05,
        "Std-dev of multiplicative per-tick noise on the utilization dimensions "
        "(DRAM/scheduler/PCIe) so utilization is NOT one deterministic curve. "
        "Calibrate from real per-tick utilization variance.",
        source_type=HEURISTIC, confidence="low",
    ),
}


# ---------------------------------------------------------------------------
# Resource placement domains (admissible placement scopes)
# ---------------------------------------------------------------------------
# Ordered finest → coarsest. A job declares admissible domains; fragmentation is
# computed per domain (free-but-unschedulable capacity is stranded). These are
# operational placement scopes, NOT a measured hardware enumeration.

RESOURCE_DOMAINS: list[str] = [
    "mig_slice",
    "shared_pool",
    "numa",
    "socket",
    "nvlink",
    "nvswitch",
    "node",
    "rack",
    "cluster_zone",
]


@dataclass(frozen=True)
class WorkloadClassProfile:
    """Utilization / topology / batching / flexibility character of a workload class.

    util_target is the SM-utilization prior; mem_bytes_per_token scales the
    DRAM-bandwidth dimension (memory-heavy classes bottleneck on bandwidth);
    topology_sensitivity / batching_sensitivity / flexibility / sla_class drive
    packing feasibility and consolidation safety. Priors, NOT measured.
    """
    name: str
    util_target: float
    mem_bytes_per_token: float       # relative DRAM-bandwidth demand per token
    topology_sensitivity: float      # 0-1, higher = needs locality
    batching_sensitivity: float      # 0-1, higher = batching helps more
    flexibility: str                 # low | medium | high
    sla_class: str                   # latency_critical | standard | batch
    source: str = "workload-class character (inferred)"
    source_type: str = INFERRED
    confidence: str = "low"


WORKLOAD_CLASS_PROFILES: dict[str, WorkloadClassProfile] = {
    "latency_critical_inference": WorkloadClassProfile(
        "latency_critical_inference", 0.50, 1.0, 0.6, 0.5, "low", "latency_critical",
        source="latency-critical serving: moderate util, queue-sensitive"),
    "standard_inference": WorkloadClassProfile(
        "standard_inference", 0.55, 1.0, 0.4, 0.7, "medium", "standard",
        source="standard serving: batching-friendly, medium flexibility"),
    "batch_inference": WorkloadClassProfile(
        "batch_inference", 0.65, 1.1, 0.2, 0.9, "high", "batch",
        source="batch/offline inference: throughput-oriented, flexible"),
    "embeddings": WorkloadClassProfile(
        "embeddings", 0.55, 1.3, 0.3, 0.6, "medium", "standard",
        source="embedding service: memory-traffic-heavy, medium flexibility"),
    "fine_tuning": WorkloadClassProfile(
        "fine_tuning", 0.80, 1.2, 0.7, 0.5, "medium", "standard",
        source="fine-tuning: high util, topology-sensitive, medium flexibility"),
    "training": WorkloadClassProfile(
        "training", 0.90, 1.2, 0.9, 0.4, "low", "batch",
        source="training: very high util, topology+comm sensitive, low flexibility"),
    "comm_heavy": WorkloadClassProfile(
        "comm_heavy", 0.70, 1.0, 0.95, 0.4, "low", "standard",
        source="communication-heavy: locality-critical, low flexibility"),
    "memory_heavy": WorkloadClassProfile(
        "memory_heavy", 0.45, 2.2, 0.4, 0.5, "medium", "standard",
        source="memory-bandwidth-bound: low SM, high DRAM (utilization paradox)"),
}

DEFAULT_WORKLOAD_CLASS = "standard_inference"


# Workload-flexibility class → consolidation/migration freedom multiplier.
# low = pinned (latency-critical / comm-heavy / TP / training); high = freely
# movable (batch / async). Tunable heuristic, NOT a measured policy.
FLEXIBILITY_CLASSES: dict[str, float] = {
    "low": 0.2,
    "medium": 0.6,
    "high": 1.0,
}


def utilization_value(name: str, config: dict | None = None) -> float:
    """Return a utilization parameter's value, allowing per-run config override.

    Mirrors ``thermal_value`` for the UTILIZATION_PARAMS registry so every
    utilization / fragmentation / packing assumption is configurable:
    ``config={'vram_headroom_frac': 0.10}``.
    """
    if config and name in config:
        return float(config[name])
    return float(UTILIZATION_PARAMS[name].value)


def resolve_workload_class(
    name: str | None, workload_type: str | None = None,
    communication_intensity: str | None = None, memory_intensity: str | None = None,
) -> WorkloadClassProfile:
    """Resolve a workload class profile.

    Explicit ``name`` wins; otherwise inferred from workload_type +
    communication/memory intensity so existing scenarios keep working without a
    workload_class.
    """
    if name and name in WORKLOAD_CLASS_PROFILES:
        return WORKLOAD_CLASS_PROFILES[name]
    wt = (workload_type or "").lower()
    if (memory_intensity or "").lower() == "high":
        return WORKLOAD_CLASS_PROFILES["memory_heavy"]
    if (communication_intensity or "").lower() == "high":
        return WORKLOAD_CLASS_PROFILES["comm_heavy"]
    if wt in ("batch_training", "training"):
        return WORKLOAD_CLASS_PROFILES["training"]
    if wt == "fine_tuning":
        return WORKLOAD_CLASS_PROFILES["fine_tuning"]
    if wt == "embedding":
        return WORKLOAD_CLASS_PROFILES["embeddings"]
    return WORKLOAD_CLASS_PROFILES[DEFAULT_WORKLOAD_CLASS]


def flexibility_multiplier(flexibility: str | None) -> float:
    """Return the consolidation/migration freedom multiplier for a flexibility class."""
    return FLEXIBILITY_CLASSES.get((flexibility or "medium").lower(), 0.6)


# ---------------------------------------------------------------------------
# Energy / carbon / arbitrage parameter registry
# ---------------------------------------------------------------------------
# Added for the energy-realism upgrade (driven by energy.py). These model
# separate day-ahead / real-time settlement, mean-reverting + heteroskedastic
# DA/RT basis with congestion jumps, an LMP component decomposition (energy +
# congestion + loss), a heavy-tailed RT spike process, regional carbon-intensity
# forecasts with uncertainty + provider disagreement, conservative forecast-error
# buffers, net-vs-gross savings accounting with explicit penalties, and
# diminishing returns / churn from repeated shifting. As elsewhere, the great
# majority are HEURISTIC/INFERRED priors anchored to documented market behaviour
# (ISO DA/RT settlement, LMP = energy+congestion+loss); NONE are calibrated to a
# specific market. They make energy arbitrage a constrained problem where DA
# planning can be wrong under RT, gross savings != net savings, and tiny spreads
# do not justify action.
#
# IMPORTANT: basis and the spike process are OFF by default (enable_basis /
# enable_spikes in the scenario's serving_config) so the canonical detection
# scenarios keep deterministic, DA==RT pricing; energy-realism scenarios opt in.

ENERGY_PARAMS: dict[str, CalibratedParam] = {
    # --- DA/RT basis (mean-reverting, heteroskedastic) -------------------
    "basis_mean_reversion": _h(
        0.4,
        "Mean-reversion rate (theta) of the DA/RT basis OU process per tick. "
        "Higher = basis snaps back to ~0 faster. RT-DA basis is mean-reverting "
        "but spiky. Calibrate from real RT-DA spread autocorrelation.",
        source_type=INFERRED, source="RT-DA basis is mean-reverting (ISO markets)",
        confidence="low",
    ),
    "basis_vol": _h(
        8.0,
        "Base volatility ($/MWh) of the DA/RT basis innovations. RT settles can "
        "diverge materially from DA. Calibrate from real RT-DA spread std.",
        source_type=INFERRED, source="RT-DA spread volatility", confidence="low",
    ),
    "basis_vol_congestion_mult": _h(
        4.0,
        "Volatility multiplier on the basis during congestion regimes "
        "(heteroskedastic — congestion makes RT far more volatile). Calibrate "
        "from real congested-interval spread variance.",
        source_type=INFERRED, source="basis volatility clusters under congestion",
    ),
    "basis_jump_prob": _h(
        0.05,
        "Per-tick probability of a congestion-driven basis jump (RT spikes far "
        "above DA). Do NOT hardcode a universal rate — override per scenario. "
        "Calibrate from real congestion-event frequency.",
        source_type=INFERRED, source="congestion-driven RT basis jumps",
    ),
    "basis_jump_magnitude": _h(
        120.0,
        "Mean magnitude ($/MWh) of a congestion-driven basis jump. RT can exceed "
        "DA by 100s of $/MWh during constrained intervals. Calibrate per node.",
        source_type=INFERRED, source="constrained-interface RT price jumps",
    ),

    # --- LMP component decomposition -------------------------------------
    "lmp_congestion_base": _h(
        0.0,
        "Baseline congestion component ($/MWh) added to the energy component "
        "(LMP = energy + congestion + loss). 0 in the unconstrained base case; "
        "raised by congestion events. Calibrate from real nodal LMP decomposition.",
        source_type=DOCUMENTED, source="LMP = energy + congestion + loss (ISO)",
        confidence="medium",
    ),
    "lmp_congestion_event_adder": _h(
        80.0,
        "Congestion-component adder ($/MWh) at a node during a constrained-"
        "interface event. Nodal arbitrage is NOT clean — destination congestion "
        "can erase the spread. Calibrate from real congestion LMP.",
        source_type=INFERRED, source="nodal congestion component under constraints",
    ),
    "lmp_loss_frac": _h(
        0.03,
        "Loss component as a fraction of the energy component (~2-4% marginal "
        "losses). Location-specific loss adder. Calibrate from real loss factors.",
        source_type=INFERRED, source="marginal loss component ~2-4% (ISO)",
        confidence="medium",
    ),
    "lmp_region_correlation": _h(
        0.7,
        "Correlation of the system-wide energy component across regions "
        "(regions move together but imperfectly). Regional arbitrage relies on "
        "the UNcorrelated part. Calibrate from real inter-region price correlation.",
        source_type=INFERRED, source="regional prices correlated but imperfect",
        confidence="low",
    ),

    # --- RT spike process (heavy tails) ----------------------------------
    "spike_prob": _h(
        0.03,
        "Per-tick probability of an RT price spike (heavy right tail). RT has "
        "heavier tails than DA. Do NOT hardcode universally — override per "
        "scenario/region. Calibrate from real RT spike frequency.",
        source_type=INFERRED, source="RT price spikes (scarcity/stress)",
    ),
    "spike_magnitude": _h(
        300.0,
        "Mean magnitude ($/MWh) of an RT price spike. Scarcity pricing can reach "
        "many 100s-1000s $/MWh. Calibrate from real scarcity events.",
        source_type=INFERRED, source="RT scarcity-pricing spike magnitude",
    ),
    "spike_stress_mult": _h(
        2.5,
        "Spike-probability multiplier in a grid-stress regime (weather/load "
        "stress). Calibrate from real stress-correlated spike rates.",
        source_type=INFERRED, source="stress-regime spike clustering",
    ),

    # --- Carbon intensity + forecast uncertainty -------------------------
    "carbon_diurnal_amplitude": _h(
        0.35,
        "Diurnal swing of carbon intensity as a fraction of the baseline "
        "(solar/wind cycle). Calibrate from real regional CI curves.",
        source_type=INFERRED, source="diurnal carbon-intensity cycle",
        confidence="low",
    ),
    "carbon_forecast_error_std": _h(
        0.15,
        "Std-dev of carbon-intensity forecast error as a fraction of the value. "
        "Forecast carbon is NOT ground truth. Calibrate from real CI forecast "
        "vs actual.",
        source_type=INFERRED, source="carbon-intensity forecast error",
        confidence="low",
    ),
    "carbon_provider_disagreement": _h(
        0.12,
        "Fractional disagreement between carbon providers for the same region/"
        "time. Providers disagree. Calibrate from real multi-provider CI spread.",
        source_type=INFERRED, source="carbon providers disagree (ElectricityMaps/WattTime)",
        confidence="low",
    ),
    "carbon_price_correlation": _h(
        0.2,
        "Correlation between carbon-intensity and price minima. Low/variable: "
        "carbon-cheap windows are often NOT price-cheap windows. Calibrate from "
        "real CI-vs-price joint distributions.",
        source_type=INFERRED, source="carbon-cheap != price-cheap",
        confidence="low",
    ),

    # --- Net-savings accounting + forecast buffer ------------------------
    "forecast_error_buffer_k": _h(
        1.0,
        "Conservatism multiplier k on the forecast error std in "
        "risk_adjusted_savings = expected - k*error_std. Higher = more "
        "conservative (more no-ops). Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "required_margin_frac": _h(
        0.10,
        "Required net-savings margin as a fraction of the workload's tick energy "
        "cost before an energy action is allowed. Tiny arbitrage must NOT trigger "
        "action. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "missing_forecast_margin_mult": _h(
        2.0,
        "Multiplier on the required margin when price/carbon forecast confidence "
        "is missing/low (bias toward no-op). Missing telemetry must NOT be read "
        "as a safe opportunity. Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),

    # --- Diminishing returns / churn -------------------------------------
    "churn_penalty_base": _h(
        0.02,
        "Base churn penalty as a fraction of tick energy cost per recent shift. "
        "Repeated shifting is increasingly costly (cache loss, instability). "
        "Calibrate from real reshuffling overhead.",
        source_type=HEURISTIC, confidence="low",
    ),
    "churn_penalty_growth": _h(
        1.6,
        "Super-linear growth exponent of the churn penalty in the recent-shift "
        "count. Repeated shifting does NOT keep producing linear savings. "
        "Heuristic policy lever.",
        source_type=HEURISTIC, confidence="low",
    ),
    "churn_window_ticks": _h(
        6.0,
        "Window (ticks) over which recent shifts count toward churn. Calibrate "
        "from real anti-flap windows.",
        source_type=INFERRED, source="anti-flap / churn accounting window",
        confidence="low",
    ),
    "low_cost_window_capacity_frac": _h(
        0.3,
        "Fraction of a cheap region's spare capacity usable for shifted load "
        "before its price/queue rises (limited low-cost windows). Calibrate from "
        "real spare-capacity-vs-price elasticity.",
        source_type=INFERRED, source="limited regional spare capacity",
        confidence="low",
    ),
}


# ---------------------------------------------------------------------------
# Workload energy-flexibility profiles (shift windows by class)
# ---------------------------------------------------------------------------
# Default temporal-shift windows (hours) by flexibility class. A workload is
# deferrable only within its window; nothing is infinitely deferrable. Tunable
# operational priors, NOT universal truths.

@dataclass(frozen=True)
class EnergyFlexProfile:
    """Energy-shift flexibility of a workload class."""
    name: str
    max_shift_hours: float        # max temporal deferral window
    spatial_shift: bool           # may migrate region for energy
    requires_locality: bool       # only shift if cache/topology preserved
    source: str = "workload energy-flexibility (inferred)"
    source_type: str = INFERRED
    confidence: str = "low"


ENERGY_FLEX_PROFILES: dict[str, EnergyFlexProfile] = {
    "high": EnergyFlexProfile(
        "high", 24.0, True, False,
        source="batch/offline/embeddings/non-urgent: deferrable 1-24h, movable"),
    "medium": EnergyFlexProfile(
        "medium", 2.0, True, True,
        source="standard inference/retryable/async: short SLA-safe windows only"),
    "low": EnergyFlexProfile(
        "low", 0.0, False, True,
        source="latency-critical/cache/topology/training: usually no temporal shift"),
}


def energy_value(name: str, config: dict | None = None) -> float:
    """Return an energy parameter's value, allowing per-run config override.

    Mirrors ``thermal_value`` for the ENERGY_PARAMS registry so every energy /
    carbon / arbitrage assumption is configurable: ``config={'spike_prob':0.1}``.
    """
    if config and name in config:
        return float(config[name])
    return float(ENERGY_PARAMS[name].value)


def resolve_energy_flex(flexibility: str | None) -> EnergyFlexProfile:
    """Resolve an energy-flexibility profile by class (default medium)."""
    return ENERGY_FLEX_PROFILES.get(
        (flexibility or "medium").lower(), ENERGY_FLEX_PROFILES["medium"]
    )


# Combined registry: every tunable constant is inspectable in one place.
ALL_PARAMS: dict[str, CalibratedParam] = {
    **SERVING_PARAMS, **KV_CACHE_PARAMS, **MIGRATION_PARAMS, **THERMAL_PARAMS,
    **TOPOLOGY_PARAMS, **UTILIZATION_PARAMS, **ENERGY_PARAMS,
}


def calibration_table() -> list[dict[str, Any]]:
    """Inspectable list of ALL serving + KV-cache + migration + thermal +
    topology + utilization + energy params."""
    rows: list[dict[str, Any]] = []
    for group, registry in (
        ("serving", SERVING_PARAMS),
        ("kv_cache", KV_CACHE_PARAMS),
        ("migration", MIGRATION_PARAMS),
        ("thermal", THERMAL_PARAMS),
        ("topology", TOPOLOGY_PARAMS),
        ("utilization", UTILIZATION_PARAMS),
        ("energy", ENERGY_PARAMS),
    ):
        for k, v in sorted(registry.items()):
            rows.append({"name": k, "group": group, **v.to_dict()})
    return rows


def cooling_regime_table() -> list[dict[str, Any]]:
    """Inspectable cooling-regime comparison table."""
    return [
        {
            "name": r.name,
            "beta_mult": r.beta_mult,
            "hotspot_mult": r.hotspot_mult,
            "density_mult": r.density_mult,
            "inlet_variance_mult": r.inlet_variance_mult,
            "source": r.source,
            "source_type": r.source_type,
            "confidence": r.confidence,
        }
        for r in sorted(COOLING_REGIMES.values(), key=lambda x: x.name)
    ]


def power_class_table() -> list[dict[str, Any]]:
    """Inspectable GPU power-class table with provenance."""
    return [
        {
            "name": c.name,
            "p_max_w": c.p_max_w,
            "throttle_onset_c": c.throttle_onset_c,
            "max_temp_c": c.max_temp_c,
            "alpha": c.alpha,
            "source": c.source,
            "source_type": c.source_type,
            "confidence": c.confidence,
        }
        for c in sorted(GPU_POWER_CLASSES.values(), key=lambda x: x.name)
    ]


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


def fabric_regime_table() -> list[dict[str, Any]]:
    """Inspectable fabric-regime comparison table (NVSwitch -> cross-region)."""
    return [
        {
            "name": r.name,
            "distance": r.distance,
            "b_eff_gbps": r.b_eff_gbps,
            "latency_us": r.latency_us,
            "congestion_sensitivity": r.congestion_sensitivity,
            "source": r.source,
            "source_type": r.source_type,
            "confidence": r.confidence,
        }
        for r in sorted(FABRIC_REGIMES.values(), key=lambda x: x.distance)
    ]


def nvlink_generation_table() -> list[dict[str, Any]]:
    """Inspectable NVLink-generation comparison table with provenance."""
    return [
        {
            "name": g.name,
            "bidir_gbps": g.bidir_gbps,
            "latency_us": g.latency_us,
            "source": g.source,
            "source_type": g.source_type,
            "confidence": g.confidence,
        }
        for g in sorted(NVLINK_GENERATIONS.values(), key=lambda x: x.bidir_gbps)
    ]


def workload_comm_profile_table() -> list[dict[str, Any]]:
    """Inspectable workload communication-sensitivity table with provenance."""
    return [
        {
            "name": p.name,
            "comm_weight": p.comm_weight,
            "collective": p.collective,
            "small_msg_sensitivity": p.small_msg_sensitivity,
            "tail_sensitivity": p.tail_sensitivity,
            "sync_heavy": p.sync_heavy,
            "nvlink_affinity": p.nvlink_affinity,
            "source": p.source,
            "source_type": p.source_type,
            "confidence": p.confidence,
        }
        for p in sorted(WORKLOAD_COMM_PROFILES.values(), key=lambda x: -x.comm_weight)
    ]


def workload_class_table() -> list[dict[str, Any]]:
    """Inspectable workload-class profile table with provenance."""
    return [
        {
            "name": p.name,
            "util_target": p.util_target,
            "mem_bytes_per_token": p.mem_bytes_per_token,
            "topology_sensitivity": p.topology_sensitivity,
            "batching_sensitivity": p.batching_sensitivity,
            "flexibility": p.flexibility,
            "sla_class": p.sla_class,
            "source": p.source,
            "source_type": p.source_type,
            "confidence": p.confidence,
        }
        for p in sorted(WORKLOAD_CLASS_PROFILES.values(), key=lambda x: -x.util_target)
    ]


def energy_flex_table() -> list[dict[str, Any]]:
    """Inspectable energy-flexibility (shift-window) table with provenance."""
    return [
        {
            "name": p.name,
            "max_shift_hours": p.max_shift_hours,
            "spatial_shift": p.spatial_shift,
            "requires_locality": p.requires_locality,
            "source": p.source,
            "source_type": p.source_type,
            "confidence": p.confidence,
        }
        for p in sorted(ENERGY_FLEX_PROFILES.values(), key=lambda x: -x.max_shift_hours)
    ]
