"""Canonical normalized state models for constraint-aware GPU orchestration.

Design rules enforced throughout:
- timestamp fields must be UTC-aware (tzinfo != None)
- percentage fields are validated to [0, 100]
- rate/bytes/duration fields are validated to >= 0
- missing/unknown data → None, never coerced to 0
- frozen dataclasses match the repo's no-pydantic-in-core convention
- every model carries a Provenance so the classifier knows staleness+confidence
- JSON round-trip via to_dict() / classmethod from_dict()

These models live in aurelius/state/ and do NOT rename existing models in
aurelius/models.py or aurelius/sla/. Normalization adapters in normalize.py
bridge from the existing shapes into these canonical forms.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Validation helpers (also exposed via aurelius/state/normalize.py)
# ---------------------------------------------------------------------------

def _require_utc(dt: datetime, field_name: str) -> None:
    if dt.tzinfo is None:
        raise ValueError(
            f"{field_name} must be UTC-aware (got naive datetime). "
            "Use datetime.now(tz=timezone.utc) or datetime.fromisoformat(...) "
            "with a '+00:00' suffix."
        )


def _require_pct(value: Optional[float], field_name: str) -> None:
    if value is not None and not (0.0 <= value <= 100.0):
        raise ValueError(f"{field_name} must be in [0, 100], got {value}")


def _require_non_negative(value: Optional[float], field_name: str) -> None:
    if value is not None and value < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")


# ---------------------------------------------------------------------------
# Provenance — attached to every model snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Origin, freshness, and trustworthiness of a state snapshot.

    Attached to every model so the constraint classifier can reason about
    staleness and confidence before acting on a signal.

    Attributes:
        source:        connector/provider name, e.g. "dcgm-exporter", "simulator"
        fetched_at:    UTC timestamp when this value was collected
        confidence:    "high" | "medium" | "low"
        is_sandbox:    True if from simulator/sandbox — rejected from claims
        sample_age_s:  seconds between observation and fetched_at; None if unknown
    """
    source: str
    fetched_at: datetime
    confidence: str  # "high" | "medium" | "low"
    is_sandbox: bool = False
    sample_age_s: Optional[float] = None

    _VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

    def __post_init__(self) -> None:
        _require_utc(self.fetched_at, "Provenance.fetched_at")
        if self.confidence not in self._VALID_CONFIDENCE:
            raise ValueError(
                f"Provenance.confidence must be 'high', 'medium', or 'low'; "
                f"got {self.confidence!r}"
            )
        if self.sample_age_s is not None and self.sample_age_s < 0:
            raise ValueError(
                f"Provenance.sample_age_s must be >= 0, got {self.sample_age_s}"
            )

    @property
    def confidence_weight(self) -> float:
        return {"high": 1.0, "medium": 0.7, "low": 0.4}[self.confidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "confidence": self.confidence,
            "is_sandbox": self.is_sandbox,
            "sample_age_s": self.sample_age_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        fetched_at = datetime.fromisoformat(d["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return cls(
            source=d["source"],
            fetched_at=fetched_at,
            confidence=d["confidence"],
            is_sandbox=d.get("is_sandbox", False),
            sample_age_s=d.get("sample_age_s"),
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConstraintType(str, Enum):
    """Constraint families the classifier can detect and score."""
    ENERGY = "energy"
    THERMAL = "thermal"
    QUEUE = "queue"
    LATENCY = "latency"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    TOPOLOGY = "topology"
    UTILIZATION = "utilization"
    NONE = "none"


class TopologyLinkType(str, Enum):
    """GPU interconnect link types, ordered loosely by bandwidth (best first)."""
    NVSWITCH = "nvswitch"    # full NVSwitch fabric (DGX/HGX)
    NV4 = "nv4"              # 4 NVLinks
    NV3 = "nv3"              # 3 NVLinks
    NV2 = "nv2"              # 2 NVLinks
    NV1 = "nv1"              # 1 NVLink
    PIX = "pix"              # single PCIe switch
    PXB = "pxb"              # multiple PCIe switches
    PHB = "phb"              # PCIe host bridge
    NODE = "node"            # same NUMA, cross PCIe
    SYS = "sys"              # cross-NUMA / SMP
    RACK = "rack"            # cross-node, same rack
    REGION = "region"        # cross-region


# ---------------------------------------------------------------------------
# GPU state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUState:
    """Point-in-time state of a single GPU, normalized from DCGM/Prometheus.

    Adapts the existing GPUMetrics + GPUHealthScore into the canonical form.
    Every field that may be absent (not observed or metric disabled) is
    Optional with default None. The classifier infers missing signals from the
    missing_signals list, not from fabricated zeroes.

    Unit notes:
    - mem_* fields: MB
    - power_*: W
    - temp_*: °C
    - nvlink_tx/rx_bytes_per_s: bytes/s
    - pcie_tx/rx_bytes_per_s: bytes/s
    - thermal_violation_ns / power_violation_ns: nanoseconds
      (DCGM POWER_VIOLATION / THERMAL_VIOLATION are nanoseconds, not µs)
    - clocks_event_reasons: DCGM bitmask (supports both CLOCK_THROTTLE_REASONS
      and CLOCKS_EVENT_REASONS names at normalization time)
    """
    gpu_uuid: str
    node_id: str
    region: str
    timestamp: datetime
    provenance: Provenance

    # Identity
    gpu_index: Optional[int] = None
    gpu_type: Optional[str] = None        # e.g. "H100", "A100"
    mig_instance_id: Optional[str] = None

    # Utilization
    util_pct: Optional[float] = None                # DCGM_FI_DEV_GPU_UTIL
    sm_active_ratio: Optional[float] = None          # DCGM_FI_PROF_GR_ENGINE_ACTIVE (0-1)
    tensor_active_ratio: Optional[float] = None      # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (0-1)
    dram_active_ratio: Optional[float] = None        # DCGM_FI_PROF_DRAM_ACTIVE (0-1)

    # Memory (MB)
    mem_used_mb: Optional[float] = None              # DCGM_FI_DEV_FB_USED
    mem_free_mb: Optional[float] = None              # DCGM_FI_DEV_FB_FREE
    mem_reserved_mb: Optional[float] = None          # DCGM_FI_DEV_FB_RESERVED
    mem_total_mb: Optional[float] = None             # derived: used+free+reserved (NOT FB_TOTAL by default)

    # Power (W)
    power_w: Optional[float] = None                  # DCGM_FI_DEV_POWER_USAGE
    power_limit_w: Optional[float] = None            # DCGM_FI_DEV_POWER_MGMT_LIMIT

    # Thermal (°C)
    temp_c: Optional[float] = None                   # DCGM_FI_DEV_GPU_TEMP
    mem_temp_c: Optional[float] = None               # DCGM_FI_DEV_MEMORY_TEMP

    # Clock / throttle
    sm_clock_mhz: Optional[float] = None             # DCGM_FI_DEV_SM_CLOCK
    clocks_event_reasons: Optional[int] = None       # DCGM_FI_DEV_CLOCKS_EVENT_REASONS bitmask
    power_violation_ns: Optional[int] = None         # DCGM_FI_DEV_POWER_VIOLATION (ns, disabled by default)
    thermal_violation_ns: Optional[int] = None       # DCGM_FI_DEV_THERMAL_VIOLATION (ns, disabled by default)

    # ECC (disabled by default in dcgm-exporter)
    ecc_sbe_total: Optional[int] = None              # DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
    ecc_dbe_total: Optional[int] = None              # DCGM_FI_DEV_ECC_DBE_VOL_TOTAL

    # XID
    xid_last: Optional[int] = None                   # DCGM_FI_DEV_XID_ERRORS (last value, enabled)
    xid_count_window: Optional[int] = None           # DCGM_EXP_XID_ERRORS_COUNT (disabled by default)

    # PCIe traffic (bytes/s, enabled by default via PROF metrics)
    pcie_tx_bytes_per_s: Optional[float] = None      # DCGM_FI_PROF_PCIE_TX_BYTES
    pcie_rx_bytes_per_s: Optional[float] = None      # DCGM_FI_PROF_PCIE_RX_BYTES

    # NVLink traffic (bytes/s, disabled by default)
    nvlink_tx_bytes_per_s: Optional[float] = None    # DCGM_FI_PROF_NVLINK_TX_BYTES
    nvlink_rx_bytes_per_s: Optional[float] = None    # DCGM_FI_PROF_NVLINK_RX_BYTES

    # Derived health (from DCGMProvider.score_gpu_health)
    health_penalty: Optional[float] = None           # 0.0 (healthy) .. 1.0 (severely degraded)
    is_schedulable: Optional[bool] = None            # derived

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "GPUState.timestamp")
        _require_pct(self.util_pct, "GPUState.util_pct")
        if self.sm_active_ratio is not None and not (0.0 <= self.sm_active_ratio <= 1.0):
            raise ValueError(f"GPUState.sm_active_ratio must be in [0, 1], got {self.sm_active_ratio}")
        if self.tensor_active_ratio is not None and not (0.0 <= self.tensor_active_ratio <= 1.0):
            raise ValueError(f"GPUState.tensor_active_ratio must be in [0, 1], got {self.tensor_active_ratio}")
        if self.dram_active_ratio is not None and not (0.0 <= self.dram_active_ratio <= 1.0):
            raise ValueError(f"GPUState.dram_active_ratio must be in [0, 1], got {self.dram_active_ratio}")
        _require_non_negative(self.mem_used_mb, "GPUState.mem_used_mb")
        _require_non_negative(self.mem_free_mb, "GPUState.mem_free_mb")
        _require_non_negative(self.mem_reserved_mb, "GPUState.mem_reserved_mb")
        _require_non_negative(self.mem_total_mb, "GPUState.mem_total_mb")
        _require_non_negative(self.power_w, "GPUState.power_w")
        _require_non_negative(self.power_limit_w, "GPUState.power_limit_w")
        _require_non_negative(self.sm_clock_mhz, "GPUState.sm_clock_mhz")
        if self.health_penalty is not None and not (0.0 <= self.health_penalty <= 1.0):
            raise ValueError(f"GPUState.health_penalty must be in [0, 1], got {self.health_penalty}")
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError(f"GPUState.gpu_index must be >= 0, got {self.gpu_index}")
        for field_name, value in [
            ("pcie_tx_bytes_per_s", self.pcie_tx_bytes_per_s),
            ("pcie_rx_bytes_per_s", self.pcie_rx_bytes_per_s),
            ("nvlink_tx_bytes_per_s", self.nvlink_tx_bytes_per_s),
            ("nvlink_rx_bytes_per_s", self.nvlink_rx_bytes_per_s),
        ]:
            _require_non_negative(value, f"GPUState.{field_name}")
        for int_field, int_value in [
            ("ecc_sbe_total", self.ecc_sbe_total),
            ("ecc_dbe_total", self.ecc_dbe_total),
            ("xid_count_window", self.xid_count_window),
            ("power_violation_ns", self.power_violation_ns),
            ("thermal_violation_ns", self.thermal_violation_ns),
        ]:
            if int_value is not None and int_value < 0:
                raise ValueError(f"GPUState.{int_field} must be >= 0, got {int_value}")

    @property
    def mem_util_pct(self) -> Optional[float]:
        if self.mem_used_mb is not None and self.mem_total_mb is not None and self.mem_total_mb > 0:
            return min(100.0, 100.0 * self.mem_used_mb / self.mem_total_mb)
        return None

    @property
    def thermal_throttling(self) -> Optional[bool]:
        if self.clocks_event_reasons is None:
            return None
        SW_THERMAL = 0x20
        HW_THERMAL = 0x40
        HW_POWER_BRAKE = 0x80
        SW_POWER_CAP = 0x04
        HW_SLOWDOWN = 0x08
        thermal_mask = SW_THERMAL | HW_THERMAL | HW_POWER_BRAKE | SW_POWER_CAP | HW_SLOWDOWN
        return bool(self.clocks_event_reasons & thermal_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_uuid": self.gpu_uuid,
            "node_id": self.node_id,
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "gpu_index": self.gpu_index,
            "gpu_type": self.gpu_type,
            "mig_instance_id": self.mig_instance_id,
            "util_pct": self.util_pct,
            "sm_active_ratio": self.sm_active_ratio,
            "tensor_active_ratio": self.tensor_active_ratio,
            "dram_active_ratio": self.dram_active_ratio,
            "mem_used_mb": self.mem_used_mb,
            "mem_free_mb": self.mem_free_mb,
            "mem_reserved_mb": self.mem_reserved_mb,
            "mem_total_mb": self.mem_total_mb,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
            "temp_c": self.temp_c,
            "mem_temp_c": self.mem_temp_c,
            "sm_clock_mhz": self.sm_clock_mhz,
            "clocks_event_reasons": self.clocks_event_reasons,
            "power_violation_ns": self.power_violation_ns,
            "thermal_violation_ns": self.thermal_violation_ns,
            "ecc_sbe_total": self.ecc_sbe_total,
            "ecc_dbe_total": self.ecc_dbe_total,
            "xid_last": self.xid_last,
            "xid_count_window": self.xid_count_window,
            "pcie_tx_bytes_per_s": self.pcie_tx_bytes_per_s,
            "pcie_rx_bytes_per_s": self.pcie_rx_bytes_per_s,
            "nvlink_tx_bytes_per_s": self.nvlink_tx_bytes_per_s,
            "nvlink_rx_bytes_per_s": self.nvlink_rx_bytes_per_s,
            "health_penalty": self.health_penalty,
            "is_schedulable": self.is_schedulable,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GPUState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            gpu_uuid=d["gpu_uuid"],
            node_id=d["node_id"],
            region=d["region"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            gpu_index=d.get("gpu_index"),
            gpu_type=d.get("gpu_type"),
            mig_instance_id=d.get("mig_instance_id"),
            util_pct=d.get("util_pct"),
            sm_active_ratio=d.get("sm_active_ratio"),
            tensor_active_ratio=d.get("tensor_active_ratio"),
            dram_active_ratio=d.get("dram_active_ratio"),
            mem_used_mb=d.get("mem_used_mb"),
            mem_free_mb=d.get("mem_free_mb"),
            mem_reserved_mb=d.get("mem_reserved_mb"),
            mem_total_mb=d.get("mem_total_mb"),
            power_w=d.get("power_w"),
            power_limit_w=d.get("power_limit_w"),
            temp_c=d.get("temp_c"),
            mem_temp_c=d.get("mem_temp_c"),
            sm_clock_mhz=d.get("sm_clock_mhz"),
            clocks_event_reasons=d.get("clocks_event_reasons"),
            power_violation_ns=d.get("power_violation_ns"),
            thermal_violation_ns=d.get("thermal_violation_ns"),
            ecc_sbe_total=d.get("ecc_sbe_total"),
            ecc_dbe_total=d.get("ecc_dbe_total"),
            xid_last=d.get("xid_last"),
            xid_count_window=d.get("xid_count_window"),
            pcie_tx_bytes_per_s=d.get("pcie_tx_bytes_per_s"),
            pcie_rx_bytes_per_s=d.get("pcie_rx_bytes_per_s"),
            nvlink_tx_bytes_per_s=d.get("nvlink_tx_bytes_per_s"),
            nvlink_rx_bytes_per_s=d.get("nvlink_rx_bytes_per_s"),
            health_penalty=d.get("health_penalty"),
            is_schedulable=d.get("is_schedulable"),
        )


# ---------------------------------------------------------------------------
# Inference service state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceServiceState:
    """Point-in-time state of an inference service (vLLM/Triton/Ray/custom).

    Latency histograms are normalized to milliseconds regardless of source
    (Triton reports µs; vLLM reports seconds). The normalization layer handles
    the unit conversion before constructing this model.

    kv_cache_usage and prefix_cache_hit_rate are in [0, 1] (not percent).
    """
    service_id: str
    engine: str               # "vllm" | "triton" | "ray_serve" | "unknown"
    timestamp: datetime
    provenance: Provenance

    region: Optional[str] = None
    node_id: Optional[str] = None

    # Request counts
    requests_running: Optional[float] = None       # active/in-flight
    requests_waiting: Optional[float] = None       # queue depth

    # Latency (ms)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None

    # Time to first token (ms)
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None

    # Queue wait (ms)
    queue_time_p50_ms: Optional[float] = None
    queue_time_p95_ms: Optional[float] = None
    queue_time_p99_ms: Optional[float] = None

    # KV / cache (0–1 fractions)
    kv_cache_usage: Optional[float] = None         # vllm:kv_cache_usage_perc (V1)
    prefix_cache_hit_rate: Optional[float] = None  # vllm:gpu_prefix_cache_hit_rate (V0) or computed

    # Preemptions (vLLM — rising = KV pressure)
    preemptions_total: Optional[float] = None

    # Efficiency
    tokens_per_s: Optional[float] = None

    # Reliability
    error_rate_pct: Optional[float] = None         # 0–100

    # Replica count (Ray autoscaling / K8s)
    replicas: Optional[int] = None

    _VALID_ENGINES = frozenset({"vllm", "triton", "ray_serve", "unknown"})

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "InferenceServiceState.timestamp")
        if self.engine not in self._VALID_ENGINES:
            raise ValueError(
                f"InferenceServiceState.engine must be one of {sorted(self._VALID_ENGINES)}, "
                f"got {self.engine!r}"
            )
        for name, val in [
            ("p50_latency_ms", self.p50_latency_ms),
            ("p95_latency_ms", self.p95_latency_ms),
            ("p99_latency_ms", self.p99_latency_ms),
            ("ttft_p50_ms", self.ttft_p50_ms),
            ("ttft_p95_ms", self.ttft_p95_ms),
            ("ttft_p99_ms", self.ttft_p99_ms),
            ("queue_time_p50_ms", self.queue_time_p50_ms),
            ("queue_time_p95_ms", self.queue_time_p95_ms),
            ("queue_time_p99_ms", self.queue_time_p99_ms),
            ("requests_running", self.requests_running),
            ("requests_waiting", self.requests_waiting),
            ("tokens_per_s", self.tokens_per_s),
            ("preemptions_total", self.preemptions_total),
        ]:
            _require_non_negative(val, f"InferenceServiceState.{name}")
        if self.kv_cache_usage is not None and not (0.0 <= self.kv_cache_usage <= 1.0):
            raise ValueError(f"InferenceServiceState.kv_cache_usage must be in [0, 1], got {self.kv_cache_usage}")
        if self.prefix_cache_hit_rate is not None and not (0.0 <= self.prefix_cache_hit_rate <= 1.0):
            raise ValueError(
                f"InferenceServiceState.prefix_cache_hit_rate must be in [0, 1], "
                f"got {self.prefix_cache_hit_rate}"
            )
        _require_pct(self.error_rate_pct, "InferenceServiceState.error_rate_pct")
        if self.replicas is not None and self.replicas < 0:
            raise ValueError(f"InferenceServiceState.replicas must be >= 0, got {self.replicas}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "engine": self.engine,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "region": self.region,
            "node_id": self.node_id,
            "requests_running": self.requests_running,
            "requests_waiting": self.requests_waiting,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "ttft_p50_ms": self.ttft_p50_ms,
            "ttft_p95_ms": self.ttft_p95_ms,
            "ttft_p99_ms": self.ttft_p99_ms,
            "queue_time_p50_ms": self.queue_time_p50_ms,
            "queue_time_p95_ms": self.queue_time_p95_ms,
            "queue_time_p99_ms": self.queue_time_p99_ms,
            "kv_cache_usage": self.kv_cache_usage,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
            "preemptions_total": self.preemptions_total,
            "tokens_per_s": self.tokens_per_s,
            "error_rate_pct": self.error_rate_pct,
            "replicas": self.replicas,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceServiceState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            service_id=d["service_id"],
            engine=d["engine"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            region=d.get("region"),
            node_id=d.get("node_id"),
            requests_running=d.get("requests_running"),
            requests_waiting=d.get("requests_waiting"),
            p50_latency_ms=d.get("p50_latency_ms"),
            p95_latency_ms=d.get("p95_latency_ms"),
            p99_latency_ms=d.get("p99_latency_ms"),
            ttft_p50_ms=d.get("ttft_p50_ms"),
            ttft_p95_ms=d.get("ttft_p95_ms"),
            ttft_p99_ms=d.get("ttft_p99_ms"),
            queue_time_p50_ms=d.get("queue_time_p50_ms"),
            queue_time_p95_ms=d.get("queue_time_p95_ms"),
            queue_time_p99_ms=d.get("queue_time_p99_ms"),
            kv_cache_usage=d.get("kv_cache_usage"),
            prefix_cache_hit_rate=d.get("prefix_cache_hit_rate"),
            preemptions_total=d.get("preemptions_total"),
            tokens_per_s=d.get("tokens_per_s"),
            error_rate_pct=d.get("error_rate_pct"),
            replicas=d.get("replicas"),
        )


# ---------------------------------------------------------------------------
# Topology state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyState:
    """Intra-node (and inter-node) GPU interconnect topology.

    pair_levels: maps (gpu_uuid_a, gpu_uuid_b) → TopologyLinkType.
    The key is always ordered (min(a,b), max(a,b)) for determinism.

    interconnect_class summarizes the node's best available link.
    For sandbox/cloud-only deployments this may be entirely None — the
    classifier degrades gracefully when topology is absent.
    """
    node_id: str
    timestamp: datetime
    provenance: Provenance
    gpu_uuids: tuple[str, ...]                                  # immutable
    pair_levels: dict[tuple[str, str], TopologyLinkType]        # normalized key order
    numa_affinity: dict[str, int]                               # gpu_uuid → NUMA node id
    nvlink_present: Optional[bool] = None
    interconnect_class: Optional[str] = None   # "nvlink_full"|"nvlink_partial"|"pcie"|"cross_numa"|"unknown"

    _VALID_INTERCONNECT = frozenset({
        "nvlink_full", "nvlink_partial", "pcie", "cross_numa", "unknown", None
    })

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "TopologyState.timestamp")
        if self.interconnect_class not in self._VALID_INTERCONNECT:
            raise ValueError(
                f"TopologyState.interconnect_class must be one of "
                f"{sorted(v for v in self._VALID_INTERCONNECT if v)}, got {self.interconnect_class!r}"
            )
        # Validate pair key ordering
        for (a, b) in self.pair_levels:
            if a > b:
                raise ValueError(
                    f"TopologyState.pair_levels keys must be ordered (min, max); "
                    f"found ({a!r}, {b!r}) — use make_topology_pair_key() to normalize."
                )

    @staticmethod
    def make_pair_key(uuid_a: str, uuid_b: str) -> tuple[str, str]:
        return (min(uuid_a, uuid_b), max(uuid_a, uuid_b))

    def link_between(self, uuid_a: str, uuid_b: str) -> Optional[TopologyLinkType]:
        key = self.make_pair_key(uuid_a, uuid_b)
        return self.pair_levels.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "pair_levels": {
                f"{a}::{b}": link.value for (a, b), link in self.pair_levels.items()
            },
            "numa_affinity": self.numa_affinity,
            "nvlink_present": self.nvlink_present,
            "interconnect_class": self.interconnect_class,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopologyState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        pair_levels = {}
        for key_str, link_val in d.get("pair_levels", {}).items():
            a, b = key_str.split("::", 1)
            pair_levels[(a, b)] = TopologyLinkType(link_val)
        return cls(
            node_id=d["node_id"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            gpu_uuids=tuple(d.get("gpu_uuids", [])),
            pair_levels=pair_levels,
            numa_affinity=d.get("numa_affinity", {}),
            nvlink_present=d.get("nvlink_present"),
            interconnect_class=d.get("interconnect_class"),
        )


# ---------------------------------------------------------------------------
# Energy state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyState:
    """Per-region energy market + power draw snapshot.

    price_percentile: where current price sits vs the region's recent history
    (0 = cheapest ever seen, 100 = most expensive). Computed by the
    normalization layer from existing forecasting history. None if no history.
    """
    region: str
    timestamp: datetime
    provenance: Provenance

    price_per_mwh: Optional[float] = None        # $/MWh, from ISO connectors
    price_percentile: Optional[float] = None      # 0–100 vs region history
    day_ahead_price_per_mwh: Optional[float] = None
    real_time_price_per_mwh: Optional[float] = None
    carbon_gco2_per_kwh: Optional[float] = None  # gCO2/kWh
    pue: Optional[float] = None                   # >= 1.0
    power_cap_kw: Optional[float] = None          # configured cap
    power_draw_kw: Optional[float] = None         # sum of GPU power_w in region

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "EnergyState.timestamp")
        _require_non_negative(self.price_per_mwh, "EnergyState.price_per_mwh")
        _require_pct(self.price_percentile, "EnergyState.price_percentile")
        _require_non_negative(self.day_ahead_price_per_mwh, "EnergyState.day_ahead_price_per_mwh")
        _require_non_negative(self.real_time_price_per_mwh, "EnergyState.real_time_price_per_mwh")
        _require_non_negative(self.carbon_gco2_per_kwh, "EnergyState.carbon_gco2_per_kwh")
        if self.pue is not None and self.pue < 1.0:
            raise ValueError(f"EnergyState.pue must be >= 1.0, got {self.pue}")
        _require_non_negative(self.power_cap_kw, "EnergyState.power_cap_kw")
        _require_non_negative(self.power_draw_kw, "EnergyState.power_draw_kw")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "price_per_mwh": self.price_per_mwh,
            "price_percentile": self.price_percentile,
            "day_ahead_price_per_mwh": self.day_ahead_price_per_mwh,
            "real_time_price_per_mwh": self.real_time_price_per_mwh,
            "carbon_gco2_per_kwh": self.carbon_gco2_per_kwh,
            "pue": self.pue,
            "power_cap_kw": self.power_cap_kw,
            "power_draw_kw": self.power_draw_kw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnergyState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            region=d["region"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            price_per_mwh=d.get("price_per_mwh"),
            price_percentile=d.get("price_percentile"),
            day_ahead_price_per_mwh=d.get("day_ahead_price_per_mwh"),
            real_time_price_per_mwh=d.get("real_time_price_per_mwh"),
            carbon_gco2_per_kwh=d.get("carbon_gco2_per_kwh"),
            pue=d.get("pue"),
            power_cap_kw=d.get("power_cap_kw"),
            power_draw_kw=d.get("power_draw_kw"),
        )


# ---------------------------------------------------------------------------
# Thermal state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalState:
    """Per-region or per-node thermal snapshot.

    Honesty note: Aurelius has no direct facility/DCIM telemetry (CRAC, PDU,
    chilled water). cooling_headroom_pct is a proxy derived from ambient
    weather + GPU temps + PUE. It must be treated as low-confidence and never
    presented as measured cooling capacity.
    """
    region: str
    timestamp: datetime
    provenance: Provenance

    node_id: Optional[str] = None

    max_gpu_temp_c: Optional[float] = None         # max across all GPUs in scope
    mean_gpu_temp_c: Optional[float] = None        # mean across all GPUs
    throttling_gpu_count: Optional[int] = None     # GPUs with thermal bits set
    total_gpu_count: Optional[int] = None          # denominator for throttling fraction
    ambient_temp_c: Optional[float] = None         # from weather connector
    cooling_headroom_pct: Optional[float] = None   # 0–100 proxy (low confidence)

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "ThermalState.timestamp")
        _require_pct(self.cooling_headroom_pct, "ThermalState.cooling_headroom_pct")
        if self.throttling_gpu_count is not None and self.throttling_gpu_count < 0:
            raise ValueError(f"ThermalState.throttling_gpu_count must be >= 0, got {self.throttling_gpu_count}")
        if self.total_gpu_count is not None and self.total_gpu_count < 0:
            raise ValueError(f"ThermalState.total_gpu_count must be >= 0, got {self.total_gpu_count}")

    @property
    def throttling_fraction(self) -> Optional[float]:
        if self.throttling_gpu_count is not None and self.total_gpu_count:
            return self.throttling_gpu_count / self.total_gpu_count
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "node_id": self.node_id,
            "max_gpu_temp_c": self.max_gpu_temp_c,
            "mean_gpu_temp_c": self.mean_gpu_temp_c,
            "throttling_gpu_count": self.throttling_gpu_count,
            "total_gpu_count": self.total_gpu_count,
            "ambient_temp_c": self.ambient_temp_c,
            "cooling_headroom_pct": self.cooling_headroom_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ThermalState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            region=d["region"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            node_id=d.get("node_id"),
            max_gpu_temp_c=d.get("max_gpu_temp_c"),
            mean_gpu_temp_c=d.get("mean_gpu_temp_c"),
            throttling_gpu_count=d.get("throttling_gpu_count"),
            total_gpu_count=d.get("total_gpu_count"),
            ambient_temp_c=d.get("ambient_temp_c"),
            cooling_headroom_pct=d.get("cooling_headroom_pct"),
        )


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeState:
    """Per-node inventory and capacity snapshot from Kubernetes + DCGM.

    gpu_capacity: total GPU slots on the node (from K8s node.status.capacity)
    gpu_allocatable: allocatable GPU slots (may be < capacity if reserved)
    gpu_allocated: GPU slots currently claimed by running pods
    """
    node_id: str
    region: str
    timestamp: datetime
    provenance: Provenance

    zone: Optional[str] = None
    rack_id: Optional[str] = None
    instance_type: Optional[str] = None

    gpu_capacity: Optional[int] = None
    gpu_allocatable: Optional[int] = None
    gpu_allocated: Optional[int] = None

    # K8s labels / taints (read-only snapshot)
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, Any]] = field(default_factory=list)
    schedulable: Optional[bool] = None

    # GPU-level details keyed by gpu_uuid
    gpus: dict[str, GPUState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "NodeState.timestamp")
        for name, val in [
            ("gpu_capacity", self.gpu_capacity),
            ("gpu_allocatable", self.gpu_allocatable),
            ("gpu_allocated", self.gpu_allocated),
        ]:
            if val is not None and val < 0:
                raise ValueError(f"NodeState.{name} must be >= 0, got {val}")

    @property
    def gpu_spare(self) -> Optional[int]:
        if self.gpu_allocatable is not None and self.gpu_allocated is not None:
            return max(0, self.gpu_allocatable - self.gpu_allocated)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "zone": self.zone,
            "rack_id": self.rack_id,
            "instance_type": self.instance_type,
            "gpu_capacity": self.gpu_capacity,
            "gpu_allocatable": self.gpu_allocatable,
            "gpu_allocated": self.gpu_allocated,
            "labels": self.labels,
            "taints": self.taints,
            "schedulable": self.schedulable,
            "gpus": {k: v.to_dict() for k, v in self.gpus.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            node_id=d["node_id"],
            region=d["region"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            zone=d.get("zone"),
            rack_id=d.get("rack_id"),
            instance_type=d.get("instance_type"),
            gpu_capacity=d.get("gpu_capacity"),
            gpu_allocatable=d.get("gpu_allocatable"),
            gpu_allocated=d.get("gpu_allocated"),
            labels=d.get("labels", {}),
            taints=d.get("taints", []),
            schedulable=d.get("schedulable"),
            gpus={k: GPUState.from_dict(v) for k, v in d.get("gpus", {}).items()},
        )


# ---------------------------------------------------------------------------
# Region state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionState:
    """Per-region cluster snapshot.

    Aggregates node, service, queue, energy, thermal, and topology state for
    one canonical region (matching region_registry.py canonical IDs).
    """
    region: str
    timestamp: datetime
    provenance: Provenance

    nodes: dict[str, NodeState] = field(default_factory=dict)
    services: dict[str, InferenceServiceState] = field(default_factory=dict)
    energy: Optional[EnergyState] = None
    thermal: Optional[ThermalState] = None
    topology: Optional[TopologyState] = None   # per-node; may be None if unavailable

    spare_capacity_pct: Optional[float] = None  # 0–100, derived from K8s allocatable

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "RegionState.timestamp")
        _require_pct(self.spare_capacity_pct, "RegionState.spare_capacity_pct")

    @property
    def total_gpu_count(self) -> int:
        return sum(n.gpu_capacity or 0 for n in self.nodes.values())

    @property
    def allocated_gpu_count(self) -> int:
        return sum(n.gpu_allocated or 0 for n in self.nodes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "energy": self.energy.to_dict() if self.energy else None,
            "thermal": self.thermal.to_dict() if self.thermal else None,
            "topology": self.topology.to_dict() if self.topology else None,
            "spare_capacity_pct": self.spare_capacity_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegionState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            region=d["region"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            nodes={k: NodeState.from_dict(v) for k, v in d.get("nodes", {}).items()},
            services={
                k: InferenceServiceState.from_dict(v)
                for k, v in d.get("services", {}).items()
            },
            energy=EnergyState.from_dict(d["energy"]) if d.get("energy") else None,
            thermal=ThermalState.from_dict(d["thermal"]) if d.get("thermal") else None,
            topology=TopologyState.from_dict(d["topology"]) if d.get("topology") else None,
            spare_capacity_pct=d.get("spare_capacity_pct"),
        )


# ---------------------------------------------------------------------------
# Cluster state (root snapshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterState:
    """Root snapshot consumed by the constraint classifier.

    This is the single canonical object produced by the normalization layer
    and consumed by the classifier, cost model, and recommendation engine.
    Connectors never appear past this layer — only ClusterState does.

    is_partial: True if any connector failed or returned stale data
    missing_sources: connector names that failed (populated by normalization layer)
    config_hash: sha256 of the active Aurelius config at snapshot time
    """
    timestamp: datetime
    provenance: Provenance
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    regions: dict[str, RegionState] = field(default_factory=dict)
    is_partial: bool = False
    missing_sources: list[str] = field(default_factory=list)
    config_hash: Optional[str] = None

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "ClusterState.timestamp")

    @property
    def all_gpus(self) -> dict[str, GPUState]:
        result = {}
        for region in self.regions.values():
            for node in region.nodes.values():
                result.update(node.gpus)
        return result

    @property
    def all_services(self) -> dict[str, InferenceServiceState]:
        result = {}
        for region in self.regions.values():
            result.update(region.services)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "snapshot_id": self.snapshot_id,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "is_partial": self.is_partial,
            "missing_sources": list(self.missing_sources),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClusterState":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            regions={k: RegionState.from_dict(v) for k, v in d.get("regions", {}).items()},
            is_partial=d.get("is_partial", False),
            missing_sources=list(d.get("missing_sources", [])),
            config_hash=d.get("config_hash"),
        )


# ---------------------------------------------------------------------------
# Migration history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationEvent:
    """A single recorded workload migration."""
    workload_id: str
    from_region: str
    to_region: str
    timestamp: datetime
    reason: str
    cost_hours: Optional[float] = None  # actual disruption window in hours

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "MigrationEvent.timestamp")
        _require_non_negative(self.cost_hours, "MigrationEvent.cost_hours")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "from_region": self.from_region,
            "to_region": self.to_region,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "cost_hours": self.cost_hours,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MigrationEvent":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            workload_id=d["workload_id"],
            from_region=d["from_region"],
            to_region=d["to_region"],
            timestamp=ts,
            reason=d["reason"],
            cost_hours=d.get("cost_hours"),
        )


@dataclass(frozen=True)
class MigrationHistory:
    """Recent migration history for a workload.

    Used by migration-governance SLA enforcement and churn detection.
    """
    workload_id: str
    events: tuple[MigrationEvent, ...]
    provenance: Provenance

    @property
    def count_last_hour(self) -> int:
        from datetime import timedelta
        cutoff = self.provenance.fetched_at - timedelta(hours=1)
        return sum(1 for e in self.events if e.timestamp >= cutoff)

    @property
    def count_last_24h(self) -> int:
        from datetime import timedelta
        cutoff = self.provenance.fetched_at - timedelta(hours=24)
        return sum(1 for e in self.events if e.timestamp >= cutoff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "events": [e.to_dict() for e in self.events],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MigrationHistory":
        return cls(
            workload_id=d["workload_id"],
            events=tuple(MigrationEvent.from_dict(e) for e in d.get("events", [])),
            provenance=Provenance.from_dict(d["provenance"]),
        )


# ---------------------------------------------------------------------------
# Constraint assessment (classifier output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintAssessment:
    """Output of the constraint classifier.

    scores: per-ConstraintType score in [0, 1], only for constraints where
            required signals were present. Absent constraints not listed.
    binding_constraint: None if no constraint exceeds the confidence floor or
                        if required signals are all absent.
    confidence: overall classifier confidence in [0, 1].
    missing_signals: telemetry fields that were absent and needed.
    """
    timestamp: datetime
    provenance: Provenance
    region: Optional[str]

    scores: dict[ConstraintType, float]
    binding_constraint: Optional[ConstraintType]
    confidence: float
    missing_signals: list[str]
    rationale: str

    # Safe/disallowed action types for the current binding constraint
    # (populated by classifier; reuses ActionType from aurelius/sla/actions.py)
    safe_action_types: list[str] = field(default_factory=list)
    disallowed_action_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "ConstraintAssessment.timestamp")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ConstraintAssessment.confidence must be in [0, 1], got {self.confidence}")
        for ct, score in self.scores.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"ConstraintAssessment.scores[{ct}] must be in [0, 1], got {score}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "region": self.region,
            "scores": {ct.value: score for ct, score in self.scores.items()},
            "binding_constraint": self.binding_constraint.value if self.binding_constraint else None,
            "confidence": self.confidence,
            "missing_signals": list(self.missing_signals),
            "rationale": self.rationale,
            "safe_action_types": list(self.safe_action_types),
            "disallowed_action_types": list(self.disallowed_action_types),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConstraintAssessment":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        scores = {ConstraintType(k): v for k, v in d.get("scores", {}).items()}
        bc_raw = d.get("binding_constraint")
        binding = ConstraintType(bc_raw) if bc_raw else None
        return cls(
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            region=d.get("region"),
            scores=scores,
            binding_constraint=binding,
            confidence=d["confidence"],
            missing_signals=list(d.get("missing_signals", [])),
            rationale=d.get("rationale", ""),
            safe_action_types=list(d.get("safe_action_types", [])),
            disallowed_action_types=list(d.get("disallowed_action_types", [])),
        )


# ---------------------------------------------------------------------------
# Recommendation (product output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    """A ranked, explained optimization recommendation.

    Output of the recommendation engine. Recommendation-only by default;
    execution is the existing opt-in, gated path.

    net_benefit: expected_effect minus estimated penalties (signed; positive = better)
    is_noop: True when the engine chose KEEP (fail-safe)
    """
    recommendation_id: str
    workload_id: str
    action_type: str                    # ActionType.value from aurelius/sla/actions.py
    timestamp: datetime
    provenance: Provenance

    binding_constraint: Optional[ConstraintType] = None
    expected_effect: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    sla_status: str = "unknown"         # "satisfied"|"corrected"|"blocked"|"unknown"
    sla_evaluation: Optional[dict[str, Any]] = None
    migration_penalty: Optional[float] = None
    net_benefit: Optional[float] = None
    rationale: str = ""
    is_noop: bool = False
    implementation_mode: str = "recommendation_only"  # "recommendation_only"|"dry_run"|"executable"

    _VALID_SLA_STATUS = frozenset({"satisfied", "corrected", "blocked", "unknown"})
    _VALID_IMPL_MODES = frozenset({"recommendation_only", "dry_run", "executable"})

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "Recommendation.timestamp")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Recommendation.confidence must be in [0, 1], got {self.confidence}")
        if self.sla_status not in self._VALID_SLA_STATUS:
            raise ValueError(
                f"Recommendation.sla_status must be one of {sorted(self._VALID_SLA_STATUS)}, "
                f"got {self.sla_status!r}"
            )
        if self.implementation_mode not in self._VALID_IMPL_MODES:
            raise ValueError(
                f"Recommendation.implementation_mode must be one of "
                f"{sorted(self._VALID_IMPL_MODES)}, got {self.implementation_mode!r}"
            )
        _require_non_negative(self.migration_penalty, "Recommendation.migration_penalty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "workload_id": self.workload_id,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance.to_dict(),
            "binding_constraint": self.binding_constraint.value if self.binding_constraint else None,
            "expected_effect": dict(self.expected_effect),
            "confidence": self.confidence,
            "sla_status": self.sla_status,
            "sla_evaluation": self.sla_evaluation,
            "migration_penalty": self.migration_penalty,
            "net_benefit": self.net_benefit,
            "rationale": self.rationale,
            "is_noop": self.is_noop,
            "implementation_mode": self.implementation_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recommendation":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bc_raw = d.get("binding_constraint")
        binding = ConstraintType(bc_raw) if bc_raw else None
        return cls(
            recommendation_id=d["recommendation_id"],
            workload_id=d["workload_id"],
            action_type=d["action_type"],
            timestamp=ts,
            provenance=Provenance.from_dict(d["provenance"]),
            binding_constraint=binding,
            expected_effect=dict(d.get("expected_effect", {})),
            confidence=d.get("confidence", 0.0),
            sla_status=d.get("sla_status", "unknown"),
            sla_evaluation=d.get("sla_evaluation"),
            migration_penalty=d.get("migration_penalty"),
            net_benefit=d.get("net_benefit"),
            rationale=d.get("rationale", ""),
            is_noop=d.get("is_noop", False),
            implementation_mode=d.get("implementation_mode", "recommendation_only"),
        )
