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


def serving_value(name: str, config: dict | None = None) -> float:
    """Return a serving parameter's value, allowing per-run config override.

    Any uncertain assumption is therefore configurable (audit requirement):
    ``config={'saturation_convexity': 1.5}`` overrides the registry default.
    """
    if config and name in config:
        return float(config[name])
    return float(SERVING_PARAMS[name].value)


def calibration_table() -> list[dict[str, Any]]:
    """Inspectable list of all serving parameters with provenance."""
    return [{"name": k, **v.to_dict()} for k, v in sorted(SERVING_PARAMS.items())]
