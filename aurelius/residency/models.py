"""Read-only data models for model-residency / cold-start telemetry.

These dataclasses are the *contract* every residency ingestion adapter
normalizes into (``aurelius/residency/ingest.py``). They are pure, deterministic,
stdlib-only, and JSON-serialisable — consistent with the public-trace schema
(``aurelius/traces/schema.py``) and the shadow ``DecisionRecord``
(``aurelius/shadow/models.py``).

Design rules (binding, from ``docs/PILOT_TELEMETRY_CONTRACT.md`` §1):

- **Conservative optional fields.** Anything a source may not emit is
  ``Optional`` and defaults to ``None`` (*unknown*). A ``None`` boolean
  (e.g. ``model_loaded_before_request``) means "not observed" and MUST NOT be
  read as ``False``. Metrics exclude it from their denominator rather than
  counting it as a miss (``aurelius/residency/metrics.py``).
- **Missing != zero.** No optional numeric defaults to ``0.0``.
- **Honest provenance.** Every record carries a ``source`` (where it came from)
  and a categorical ``confidence`` (how much to trust it).

Nothing here is a production claim. Telemetry is *observed*, never acted upon by
this package (``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


class ResidencySchemaError(ValueError):
    """Raised when a raw residency record is missing a required field or has a
    structurally invalid value (e.g. an unknown ``event_type``)."""


# Engine-level residency lifecycle events. A pilot/runtime emits these only if it
# actually observes them — we never *invent* a load event a runtime cannot emit
# (``docs/PILOT_TELEMETRY_CONTRACT.md`` §2; vLLM, e.g., does not expose them).
EVENT_TYPES = frozenset({
    "model_load_start", "model_load_end",
    "adapter_load_start", "adapter_load_end",
    "request_start", "request_end",
    "model_evict", "adapter_evict",
})

# Provenance-style trust label. Categorical (not a fabricated probability).
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low", "unknown"})

_NULLS = ("", "NULL", "None", "null", "nan", "NaN", "NaT")

# Heuristic epoch disambiguation: a 2026 timestamp is ~1.78e9 s but ~1.78e12 ms.
# Anything >= 1e11 is treated as milliseconds. Documented per contract §1.
_EPOCH_MS_THRESHOLD = 1e11


def _is_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in _NULLS)


def parse_timestamp(value, *, unit: str = "auto") -> Optional[float]:
    """Parse a timestamp into epoch **seconds** (UTC float), or ``None``.

    ``unit`` ∈ {``"auto"``, ``"s"``, ``"ms"``, ``"iso"``}. ``"auto"`` accepts
    RFC-3339 / ISO-8601 strings and numeric epochs, disambiguating seconds vs
    milliseconds with a magnitude threshold (``>= 1e11`` → ms). Missing /
    null-marker input returns ``None`` (never ``0.0``).
    """
    if _is_null(value):
        return None
    if unit == "iso" or (unit == "auto" and isinstance(value, str)
                         and not _looks_numeric(value)):
        return _parse_iso(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        # last resort: maybe an ISO string mislabelled
        return _parse_iso(value) if isinstance(value, str) else None
    if unit == "ms":
        return num / 1000.0
    if unit == "s":
        return num
    # auto: disambiguate by magnitude
    return num / 1000.0 if abs(num) >= _EPOCH_MS_THRESHOLD else num


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _parse_iso(value) -> Optional[float]:
    if not isinstance(value, str) or _is_null(value):
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value.strip(), fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _opt_float(v) -> Optional[float]:
    if _is_null(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v) -> Optional[str]:
    if _is_null(v):
        return None
    return str(v).strip()


def _opt_bool(v) -> Optional[bool]:
    """Parse an optional boolean. ``None`` (unknown) is preserved — a missing
    residency flag is NOT a miss (``False``)."""
    if _is_null(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None


def _norm_confidence(v) -> str:
    s = (str(v).strip().lower() if not _is_null(v) else "unknown")
    return s if s in CONFIDENCE_LEVELS else "unknown"


def _opt_str_tuple(v) -> Optional[tuple[str, ...]]:
    """Normalise a list-valued field. ``None`` means *unknown* (the snapshot did
    not report residency); an explicit empty list means *known-empty* (nothing
    resident) and is preserved as ``()``."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s in _NULLS:
            return None
        # tolerate "a|b", "a;b", "a,b" CSV encodings
        for sep in ("|", ";", ","):
            if sep in s:
                return tuple(p.strip() for p in s.split(sep) if p.strip())
        return (s,) if s else ()
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if not _is_null(x))
    return None


@dataclass(frozen=True)
class ModelResidencyEvent:
    """One engine-level model/adapter residency lifecycle event.

    Required: ``timestamp``, ``model_id``, ``event_type``, ``source``.
    Everything else is optional (``None`` = not observed). ``duration_s`` is the
    measured load/evict wall-clock when the source provides it (e.g. a
    ``model_load_end`` carrying its own load latency).
    """

    timestamp: float
    model_id: str
    event_type: str
    source: str
    status: Optional[str] = None
    confidence: str = "unknown"
    request_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workload_id: Optional[str] = None
    adapter_id: Optional[str] = None
    region: Optional[str] = None
    node_id: Optional[str] = None
    gpu_id: Optional[str] = None
    container_id: Optional[str] = None
    duration_s: Optional[float] = None

    def __post_init__(self):
        if self.event_type not in EVENT_TYPES:
            raise ResidencySchemaError(
                f"unknown event_type {self.event_type!r}; "
                f"expected one of {sorted(EVENT_TYPES)}")

    @property
    def is_load_event(self) -> bool:
        return self.event_type in ("model_load_start", "model_load_end",
                                   "adapter_load_start", "adapter_load_end")

    @property
    def is_evict_event(self) -> bool:
        return self.event_type in ("model_evict", "adapter_evict")

    @property
    def is_adapter_event(self) -> bool:
        return self.event_type in ("adapter_load_start", "adapter_load_end",
                                   "adapter_evict")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, *, source: Optional[str] = None,
                  timestamp_unit: str = "auto") -> "ModelResidencyEvent":
        ts = parse_timestamp(d.get("timestamp"), unit=timestamp_unit)
        if ts is None:
            raise ResidencySchemaError("event missing required 'timestamp'")
        model_id = _opt_str(d.get("model_id"))
        if model_id is None:
            raise ResidencySchemaError("event missing required 'model_id'")
        event_type = _opt_str(d.get("event_type"))
        if event_type is None:
            raise ResidencySchemaError("event missing required 'event_type'")
        src = _opt_str(d.get("source")) or source
        if src is None:
            raise ResidencySchemaError("event missing required 'source'")
        return cls(
            timestamp=ts,
            model_id=model_id,
            event_type=event_type,
            source=src,
            status=_opt_str(d.get("status")),
            confidence=_norm_confidence(d.get("confidence")),
            request_id=_opt_str(d.get("request_id")),
            tenant_id=_opt_str(d.get("tenant_id")),
            workload_id=_opt_str(d.get("workload_id")),
            adapter_id=_opt_str(d.get("adapter_id") if d.get("adapter_id") is not None
                                 else d.get("lora_id")),
            region=_opt_str(d.get("region")),
            node_id=_opt_str(d.get("node_id")),
            gpu_id=_opt_str(d.get("gpu_id")),
            container_id=_opt_str(d.get("container_id")),
            duration_s=_opt_float(d.get("duration_s")),
        )


@dataclass(frozen=True)
class ModelResidencySnapshot:
    """A point-in-time view of what is resident on one GPU/container/node.

    ``loaded_model_ids`` / ``loaded_adapter_ids`` are ``None`` when the source
    did not report residency (unknown) and an empty tuple ``()`` when it
    reported *nothing* resident (known-empty). ``gpu_memory_used`` / ``_total``
    are ``None`` when unknown — never ``0.0``.
    """

    timestamp: float
    source: str
    region: Optional[str] = None
    node_id: Optional[str] = None
    gpu_id: Optional[str] = None
    container_id: Optional[str] = None
    loaded_model_ids: Optional[tuple[str, ...]] = None
    loaded_adapter_ids: Optional[tuple[str, ...]] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    confidence: str = "unknown"

    @property
    def has_residency(self) -> bool:
        """True if this snapshot reports at least one resident model/adapter."""
        return bool(self.loaded_model_ids) or bool(self.loaded_adapter_ids)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("loaded_model_ids", "loaded_adapter_ids"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict, *, source: Optional[str] = None,
                  timestamp_unit: str = "auto") -> "ModelResidencySnapshot":
        ts = parse_timestamp(d.get("timestamp"), unit=timestamp_unit)
        if ts is None:
            raise ResidencySchemaError("snapshot missing required 'timestamp'")
        src = _opt_str(d.get("source")) or source
        if src is None:
            raise ResidencySchemaError("snapshot missing required 'source'")
        return cls(
            timestamp=ts,
            source=src,
            region=_opt_str(d.get("region")),
            node_id=_opt_str(d.get("node_id")),
            gpu_id=_opt_str(d.get("gpu_id")),
            container_id=_opt_str(d.get("container_id")),
            loaded_model_ids=_opt_str_tuple(d.get("loaded_model_ids")),
            loaded_adapter_ids=_opt_str_tuple(d.get("loaded_adapter_ids")),
            gpu_memory_used=_opt_float(d.get("gpu_memory_used")),
            gpu_memory_total=_opt_float(d.get("gpu_memory_total")),
            confidence=_norm_confidence(d.get("confidence")),
        )


@dataclass(frozen=True)
class RequestResidencyObservation:
    """Per-request residency observation.

    Core fields are the ``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §2 request
    record. The cross-layer **join keys** (``node_id``/``gpu_id``/
    ``container_id`` + tenancy/region) are optional additions mandated by
    ``docs/PILOT_TELEMETRY_CONTRACT.md`` §4 so residency can be *attributed*
    rather than proxied; ``None`` when the request stream does not carry them
    (then linkage is ``no_join`` — see ``aurelius/residency/linkage.py``).

    Booleans are ``Optional``: ``model_loaded_before_request=None`` means the
    runtime did not report residency for this request and the request is
    excluded from the hit-rate denominator (NOT counted as a miss).
    """

    request_id: str
    timestamp: float
    model_id: str
    source: str
    adapter_id: Optional[str] = None
    model_loaded_before_request: Optional[bool] = None
    adapter_loaded_before_request: Optional[bool] = None
    model_load_latency_s: Optional[float] = None
    adapter_load_latency_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None
    e2e_latency_s: Optional[float] = None
    status: Optional[str] = None
    confidence: str = "unknown"
    # Optional contract §2/§4 join keys + tenancy (additive; not in the minimal
    # Task-1 field set, required by the linkage helper when present).
    tenant_id: Optional[str] = None
    workload_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    region: Optional[str] = None
    node_id: Optional[str] = None
    gpu_id: Optional[str] = None
    container_id: Optional[str] = None

    @property
    def is_failed(self) -> bool:
        """True only when status is explicitly a non-OK value. Unknown status is
        not treated as failure."""
        if _is_null(self.status):
            return False
        return str(self.status).strip().upper() not in ("OK", "SUCCEED", "SUCCESS",
                                                         "200", "COMPLETED")

    @property
    def total_load_latency_s(self) -> Optional[float]:
        """Sum of model + adapter load latency, treating an *unknown* component
        as 0 *only* if the other is known. Returns ``None`` if both unknown."""
        m, a = self.model_load_latency_s, self.adapter_load_latency_s
        if m is None and a is None:
            return None
        return (m or 0.0) + (a or 0.0)

    @property
    def is_cold_start(self) -> Optional[bool]:
        """True if a base-model and/or adapter load was required.

        Returns ``None`` (unknown) when neither residency flag was reported —
        the request cannot be classified and is excluded from cold-start rate.
        """
        m, a = self.model_loaded_before_request, self.adapter_loaded_before_request
        known = [x for x in (m, a) if x is not None]
        if not known:
            return None
        # cold if any reported component was NOT resident before the request
        return any(x is False for x in known)

    @property
    def has_join_keys(self) -> bool:
        return bool(self.container_id) or bool(self.gpu_id) or bool(self.node_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, *, source: Optional[str] = None,
                  timestamp_unit: str = "auto") -> "RequestResidencyObservation":
        request_id = _opt_str(d.get("request_id"))
        if request_id is None:
            raise ResidencySchemaError("observation missing required 'request_id'")
        ts = parse_timestamp(d.get("timestamp"), unit=timestamp_unit)
        if ts is None:
            raise ResidencySchemaError("observation missing required 'timestamp'")
        model_id = _opt_str(d.get("model_id"))
        if model_id is None:
            raise ResidencySchemaError("observation missing required 'model_id'")
        src = _opt_str(d.get("source")) or source
        if src is None:
            raise ResidencySchemaError("observation missing required 'source'")

        # Accept both the PILOT_TELEMETRY_CONTRACT §2 field names and the model
        # field names. Derive load latency from start/end timestamps if given.
        model_lat = _opt_float(d.get("model_load_latency_s"))
        if model_lat is None:
            model_lat = _latency_from_span(d.get("model_load_start"),
                                           d.get("model_load_end"), timestamp_unit)
        adapter_lat = _opt_float(d.get("adapter_load_latency_s"))
        if adapter_lat is None:
            adapter_lat = _latency_from_span(d.get("adapter_load_start"),
                                             d.get("adapter_load_end"), timestamp_unit)

        return cls(
            request_id=request_id,
            timestamp=ts,
            model_id=model_id,
            source=src,
            adapter_id=_opt_str(d.get("adapter_id") if d.get("adapter_id") is not None
                                else d.get("lora_id")),
            model_loaded_before_request=_opt_bool(d.get("model_loaded_before_request")),
            adapter_loaded_before_request=_opt_bool(
                d.get("adapter_loaded_before_request")),
            model_load_latency_s=model_lat,
            adapter_load_latency_s=adapter_lat,
            queue_wait_s=_opt_float(d.get("queue_wait_s") if d.get("queue_wait_s")
                                    is not None else d.get("queue_wait")),
            ttft_s=_opt_float(_first(d, "ttft_s", "TTFT", "ttft")),
            tpot_s=_opt_float(_first(d, "tpot_s", "TPOT", "tpot")),
            e2e_latency_s=_opt_float(_first(d, "e2e_latency_s", "e2e_latency",
                                            "e2e")),
            status=_opt_str(d.get("status") if d.get("status") is not None
                            else d.get("error")),
            confidence=_norm_confidence(d.get("confidence")),
            tenant_id=_opt_str(d.get("tenant_id")),
            workload_id=_opt_str(d.get("workload_id")),
            endpoint_id=_opt_str(d.get("endpoint_id")),
            region=_opt_str(d.get("region")),
            node_id=_opt_str(d.get("node_id")),
            gpu_id=_opt_str(d.get("gpu_id")),
            container_id=_opt_str(d.get("container_id")),
        )


def _first(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return None


def _latency_from_span(start, end, timestamp_unit: str) -> Optional[float]:
    s = parse_timestamp(start, unit=timestamp_unit)
    e = parse_timestamp(end, unit=timestamp_unit)
    if s is None or e is None or e < s:
        return None
    return e - s


# ===========================================================================
# Decision-engine models (Model Residency Decision Engine v1)
# ===========================================================================
#
# These models drive ``aurelius/residency/decision.py``. The decision engine is
# RECOMMENDATION-ONLY in real/customer mode; only the simulator
# (``aurelius/residency/sim.py``) may mutate (simulated) ``ModelLocationState``.
# A decision NEVER changes which model/adapter the user requested — only
# placement / routing / prewarm / evict recommendations.


class ResidencyAction:
    """Canonical residency decision actions (string enum)."""

    ROUTE_TO_RESIDENT_MODEL = "ROUTE_TO_RESIDENT_MODEL"
    PRESERVE_AFFINITY = "PRESERVE_AFFINITY"
    PREWARM_MODEL = "PREWARM_MODEL"
    PREWARM_ADAPTER = "PREWARM_ADAPTER"
    KEEP_CURRENT_ROUTE = "KEEP_CURRENT_ROUTE"
    REJECT_UNSAFE_ROUTE = "REJECT_UNSAFE_ROUTE"
    EVICT_CANDIDATE = "EVICT_CANDIDATE"
    INSUFFICIENT_TELEMETRY = "INSUFFICIENT_TELEMETRY"


RESIDENCY_ACTIONS = frozenset({
    ResidencyAction.ROUTE_TO_RESIDENT_MODEL, ResidencyAction.PRESERVE_AFFINITY,
    ResidencyAction.PREWARM_MODEL, ResidencyAction.PREWARM_ADAPTER,
    ResidencyAction.KEEP_CURRENT_ROUTE, ResidencyAction.REJECT_UNSAFE_ROUTE,
    ResidencyAction.EVICT_CANDIDATE, ResidencyAction.INSUFFICIENT_TELEMETRY,
})

PRIORITY_CLASSES = frozenset({"critical", "standard", "best_effort", "batch"})


@dataclass(frozen=True)
class ModelResidencyRequest:
    """A request for which the engine recommends a placement/residency action.

    The requested ``(model_id, adapter_id)`` is the contract — it is NEVER
    substituted. ``current_route`` is the location key the request is currently
    assigned to (if any), so the engine can recommend KEEP/PRESERVE vs reroute.
    """

    request_id: str
    timestamp: float
    workload_id: str
    model_id: str
    priority_class: str = "standard"
    tenant_id: Optional[str] = None
    adapter_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_sla_ms: Optional[float] = None
    deadline_s: Optional[float] = None
    region: Optional[str] = None
    current_route: Optional[str] = None
    allowed_regions: Optional[tuple[str, ...]] = None

    @property
    def is_safety_critical(self) -> bool:
        return self.priority_class in ("critical",)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("allowed_regions") is not None:
            d["allowed_regions"] = list(d["allowed_regions"])
        return d


@dataclass
class ModelLocationState:
    """A serving location (node/gpu/container) and what is resident on it.

    MUTABLE: the simulator (``sim.py``) mutates ``loaded_model_ids`` /
    ``loaded_adapter_ids`` / ``gpu_memory_used`` / ``queue_depth`` in simulator
    mode only. The decision engine treats it as READ-ONLY. Memory fields are in
    bytes (consistent with :class:`ModelResidencySnapshot` / the telemetry
    contract); :class:`ModelLoadProfile.memory_required_gb` is converted on use.
    """

    region: str
    node_id: str
    gpu_id: str
    container_id: str
    loaded_model_ids: list = field(default_factory=list)
    loaded_adapter_ids: list = field(default_factory=list)
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    queue_depth: Optional[int] = None
    estimated_queue_wait_s: Optional[float] = None
    thermal_risk: Optional[float] = None
    topology_score: Optional[float] = None
    telemetry_confidence: str = "unknown"
    last_updated_s: Optional[float] = None

    @property
    def location_key(self) -> str:
        return f"{self.region}/{self.node_id}/{self.gpu_id}/{self.container_id}"

    def has_model(self, model_id: str) -> bool:
        return model_id in self.loaded_model_ids

    def has_adapter(self, adapter_id: Optional[str]) -> bool:
        return adapter_id is not None and adapter_id in self.loaded_adapter_ids

    @property
    def memory_free_gb(self) -> Optional[float]:
        """Free GPU memory in GB, or ``None`` if memory telemetry is missing
        (NOT treated as zero / unlimited)."""
        if self.gpu_memory_total is None or self.gpu_memory_used is None:
            return None
        return max(0.0, (self.gpu_memory_total - self.gpu_memory_used)) / 1e9

    def to_dict(self) -> dict:
        d = asdict(self)
        d["location_key"] = self.location_key
        return d


@dataclass(frozen=True)
class ModelLoadProfile:
    """Calibrated cold-load characteristics for a ``(model_id[, adapter_id])``.

    Latencies are seconds. ``None`` means *unknown* and MUST NOT be read as 0 —
    a request that needs a load whose latency is unknown cannot be assumed free.
    """

    model_id: str
    cold_load_p50_s: Optional[float] = None
    cold_load_p95_s: Optional[float] = None
    adapter_id: Optional[str] = None
    adapter_load_p50_s: Optional[float] = None
    adapter_load_p95_s: Optional[float] = None
    memory_required_gb: Optional[float] = None
    gpu_type_requirements: Optional[tuple[str, ...]] = None
    source: str = "unknown"
    confidence: str = "unknown"

    def model_load_penalty_s(self, *, safety_critical: bool) -> Optional[float]:
        """p95 for safety-critical, else p50. ``None`` (unknown) stays ``None``."""
        return (self.cold_load_p95_s if safety_critical else self.cold_load_p50_s)

    def adapter_load_penalty_s(self, *, safety_critical: bool) -> Optional[float]:
        return (self.adapter_load_p95_s if safety_critical else self.adapter_load_p50_s)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("gpu_type_requirements") is not None:
            d["gpu_type_requirements"] = list(d["gpu_type_requirements"])
        return d


@dataclass(frozen=True)
class ResidencyDecision:
    """A recommendation-only residency decision.

    ``executable_in_real_cluster`` is ``False`` by default and is NEVER set true
    by the engine — real/customer mode is recommendation-only
    (``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §5). ``expected_*`` fields are
    directional estimates, not measured outcomes.
    """

    request_id: str
    action: str
    reason: str
    target_location: Optional[str] = None
    current_location: Optional[str] = None
    expected_cold_start_saved_s: Optional[float] = None
    expected_queue_delta_s: Optional[float] = None
    expected_latency_delta_s: Optional[float] = None
    expected_cost_delta: Optional[float] = None
    expected_goodput_per_dollar_delta: Optional[float] = None
    safety_vetoes: tuple = ()
    confidence: str = "unknown"
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False

    def __post_init__(self):
        if self.action not in RESIDENCY_ACTIONS:
            raise ResidencySchemaError(
                f"unknown residency action {self.action!r}; "
                f"expected one of {sorted(RESIDENCY_ACTIONS)}")
        if self.executable_in_real_cluster:
            raise ResidencySchemaError(
                "residency decisions are recommendation-only in real/customer "
                "mode; executable_in_real_cluster must be False")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safety_vetoes"] = list(self.safety_vetoes)
        return d

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())
