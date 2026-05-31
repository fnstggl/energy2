"""Per-field telemetry provenance for the Dynamic Frontier pilot bridge.

Every field that flows from real connectors into a
:class:`aurelius.frontier.dynamic_models.ServingTelemetryTick` carries a
small categorical provenance record:

- ``REAL`` — primary counter / gauge / histogram the adapter scraped.
- ``DERIVED`` — computed deterministically from one or more REAL fields
  (e.g. ``error_rate = failures / (failures + successes)``,
  ``sla_violation = latency_p99 > sla_p99``, K8s pod-set delta).
- ``PROXY`` — substituted from a related-but-not-equivalent metric
  (e.g. vLLM TTFT used as a queue p99 proxy; vLLM ``num_requests_running``
  used as an RPS proxy). The estimator + shadow log MUST be able to see
  that the field is a proxy.
- ``SIMULATED`` — value came from the simulator / a public-trace replay,
  not real telemetry.
- ``MISSING`` — adapter could not emit this field; downstream sees
  ``None``.

Shadow reports + calibration records carry a ``TickProvenance`` mapping
so the audit story stays honest: a calibration captured at 91 %
oracle-alpha capture on PROXY queue p99 is not the same evidence as the
same number on REAL queue p99.

Hard rules:
- No production-savings claim is permitted that relies on PROXY or
  DERIVED safety signals (see ``docs/RESULTS.md`` §8).
- Pure stdlib. JSON round-trippable.
- Adding a new origin (e.g. ``LEARNED``) is intentionally a model change
  so the gate cannot be relaxed accidentally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Optional


class FieldOrigin(str, Enum):
    """How a single ServingTelemetryTick field was sourced."""

    REAL = "real"
    DERIVED = "derived"
    PROXY = "proxy"
    SIMULATED = "simulated"
    MISSING = "missing"


# Categorical fallback levels for ``timeout_pct``. Letters preserve the
# order in the spec — A is the most trustworthy; D is the last-resort
# proxy. Each carries its own provenance so the gate cannot be confused.
class TimeoutFallback(str, Enum):
    """Which level of the timeout fallback chain produced ``timeout_pct``."""

    A_TIMEOUT_COUNTER = "A_explicit_timeout_counter"
    B_DEADLINE_COUNTER = "B_explicit_deadline_exceeded_counter"
    C_ERROR_RATE = "C_error_rate_proxy"
    D_SLA_RISK_PROXY = "D_latency_p99_over_sla_proxy"
    NONE = "none"


@dataclass(frozen=True)
class FieldProvenance:
    """Per-field provenance record carried alongside a tick.

    Attributes:
        field: the canonical ServingTelemetryTick / outcome field name.
        origin: REAL / DERIVED / PROXY / SIMULATED / MISSING.
        source: the raw metric / file / adapter it came from.
        confidence: connector-level confidence (low / medium / high /
            unknown).
        notes: short explanation when the choice was non-trivial (e.g.
            "TTFT used as queue p99 proxy", "K8s pod-set delta").
        fallback_level: when the field is in a fallback chain (currently
            only ``timeout_pct``), record which level produced it.
    """

    field: str
    origin: FieldOrigin
    source: str = ""
    confidence: str = "unknown"
    notes: str = ""
    fallback_level: Optional[TimeoutFallback] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"] = self.origin.value
        d["fallback_level"] = (self.fallback_level.value
                               if self.fallback_level is not None else None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FieldProvenance":
        origin = d.get("origin")
        if isinstance(origin, str):
            origin = FieldOrigin(origin)
        fb = d.get("fallback_level")
        if isinstance(fb, str):
            fb = TimeoutFallback(fb)
        return cls(
            field=str(d.get("field", "")),
            origin=origin,
            source=str(d.get("source", "")),
            confidence=str(d.get("confidence", "unknown")),
            notes=str(d.get("notes", "")),
            fallback_level=fb,
        )


@dataclass(frozen=True)
class TickProvenance:
    """Per-tick provenance — one ``FieldProvenance`` per field.

    Stored as a tuple so the dataclass stays hashable and JSON
    round-trippable. Use :meth:`get` for keyed lookup.
    """

    entries: tuple = ()

    def get(self, field_name: str) -> Optional[FieldProvenance]:
        for e in self.entries:
            if e.field == field_name:
                return e
        return None

    def fields_by_origin(self, origin: FieldOrigin) -> tuple:
        return tuple(e.field for e in self.entries if e.origin == origin)

    @property
    def real_fields(self) -> tuple:
        return self.fields_by_origin(FieldOrigin.REAL)

    @property
    def derived_fields(self) -> tuple:
        return self.fields_by_origin(FieldOrigin.DERIVED)

    @property
    def proxy_fields(self) -> tuple:
        return self.fields_by_origin(FieldOrigin.PROXY)

    @property
    def simulated_fields(self) -> tuple:
        return self.fields_by_origin(FieldOrigin.SIMULATED)

    @property
    def missing_fields(self) -> tuple:
        return self.fields_by_origin(FieldOrigin.MISSING)

    @property
    def has_any_proxy(self) -> bool:
        return any(e.origin == FieldOrigin.PROXY for e in self.entries)

    @property
    def has_any_simulated(self) -> bool:
        return any(e.origin == FieldOrigin.SIMULATED for e in self.entries)

    @property
    def has_any_real_safety_field(self) -> bool:
        """True iff at least one of the safety-critical fields
        (``timeout_pct``, ``queue_p99_ms``, ``latency_p99_ms``) is
        sourced from a REAL adapter metric. Used by the audit to refuse
        promoting a fully-proxy run to a production claim."""
        safety = {"timeout_pct", "queue_p99_ms", "latency_p99_ms"}
        for e in self.entries:
            if e.field in safety and e.origin == FieldOrigin.REAL:
                return True
        return False

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> "TickProvenance":
        entries = tuple(FieldProvenance.from_dict(x)
                         for x in (d or {}).get("entries", ()))
        return cls(entries=entries)


# ---------------------------------------------------------------------------
# Builders — keep these tiny + deterministic.
# ---------------------------------------------------------------------------

def make_real(field_name: str, source: str, *,
              confidence: str = "medium",
              notes: str = "") -> FieldProvenance:
    return FieldProvenance(field=field_name, origin=FieldOrigin.REAL,
                            source=source, confidence=confidence,
                            notes=notes)


def make_derived(field_name: str, source: str, *,
                 confidence: str = "medium",
                 notes: str = "") -> FieldProvenance:
    return FieldProvenance(field=field_name, origin=FieldOrigin.DERIVED,
                            source=source, confidence=confidence,
                            notes=notes)


def make_proxy(field_name: str, source: str, *,
               confidence: str = "low",
               notes: str = "") -> FieldProvenance:
    return FieldProvenance(field=field_name, origin=FieldOrigin.PROXY,
                            source=source, confidence=confidence,
                            notes=notes)


def make_simulated(field_name: str, source: str, *,
                   confidence: str = "low",
                   notes: str = "") -> FieldProvenance:
    return FieldProvenance(field=field_name, origin=FieldOrigin.SIMULATED,
                            source=source, confidence=confidence,
                            notes=notes)


def make_missing(field_name: str, *, notes: str = "") -> FieldProvenance:
    return FieldProvenance(field=field_name, origin=FieldOrigin.MISSING,
                            source="", confidence="unknown", notes=notes)


# ---------------------------------------------------------------------------
# Derived-metric helpers — pure, deterministic, no future leakage.
# ---------------------------------------------------------------------------

def derive_sla_violation_pct(
    *,
    latency_p99_ms: Optional[float],
    latency_sla_p99_ms: Optional[float],
) -> Optional[float]:
    """Per-tick SLA-violation share derived from ``latency_p99_ms`` vs
    the workload's SLA budget. Returns:

    - ``None`` when either input is missing,
    - ``100.0`` when ``latency_p99_ms > latency_sla_p99_ms``,
    - ``0.0`` otherwise.

    The result is intentionally binary per-tick — aggregating it over a
    window yields a real share. This is a DERIVED signal (per the
    contract); callers MUST flag the field provenance as
    :class:`FieldOrigin.DERIVED`.
    """
    if latency_p99_ms is None or latency_sla_p99_ms is None:
        return None
    if latency_sla_p99_ms <= 0:
        return None
    return 100.0 if latency_p99_ms > latency_sla_p99_ms else 0.0


@dataclass(frozen=True)
class TimeoutFallbackResult:
    """Outcome of running the timeout-percent fallback chain.

    ``value`` is the resolved ``timeout_pct`` (or ``None`` if every
    level was unavailable). ``level`` records which level produced the
    value so the bridge can attach the right :class:`FieldProvenance`.
    """

    value: Optional[float]
    level: TimeoutFallback
    source: str
    notes: str = ""


def resolve_timeout_pct(
    *,
    explicit_timeout_counter_rate: Optional[float] = None,
    deadline_exceeded_rate: Optional[float] = None,
    total_request_rate: Optional[float] = None,
    error_rate_pct: Optional[float] = None,
    latency_p99_ms: Optional[float] = None,
    latency_sla_p99_ms: Optional[float] = None,
) -> TimeoutFallbackResult:
    """Run the documented timeout fallback chain (A → B → C → D).

    - **A_TIMEOUT_COUNTER**: ``explicit_timeout_counter_rate`` / max(1,
      ``total_request_rate``) × 100. Most trustworthy when present —
      none of vLLM / Ray Serve / Triton expose this today.
    - **B_DEADLINE_COUNTER**: ``deadline_exceeded_rate`` / max(1,
      ``total_request_rate``) × 100. Same status as A — not exposed by
      mainstream serving runtimes today.
    - **C_ERROR_RATE**: ``error_rate_pct`` (Ray Serve HTTP-5xx ratio or
      Triton ``failure / (success + failure)``). REAL but errors are
      not the same as SLA timeouts; flagged accordingly.
    - **D_SLA_RISK_PROXY**: 100.0 when ``latency_p99_ms >
      latency_sla_p99_ms`` else 0.0. Last-resort PROXY; not a true
      timeout share.

    Returns the first available level. ``value=None`` only when every
    level was unavailable.
    """
    if (explicit_timeout_counter_rate is not None
            and total_request_rate is not None
            and total_request_rate > 0):
        pct = 100.0 * explicit_timeout_counter_rate / total_request_rate
        return TimeoutFallbackResult(
            value=max(0.0, min(100.0, pct)),
            level=TimeoutFallback.A_TIMEOUT_COUNTER,
            source="explicit_timeout_counter_rate",
            notes="primary path: per-second timeout counter")

    if (deadline_exceeded_rate is not None
            and total_request_rate is not None
            and total_request_rate > 0):
        pct = 100.0 * deadline_exceeded_rate / total_request_rate
        return TimeoutFallbackResult(
            value=max(0.0, min(100.0, pct)),
            level=TimeoutFallback.B_DEADLINE_COUNTER,
            source="deadline_exceeded_rate",
            notes="explicit deadline_exceeded / cancelled counter")

    if error_rate_pct is not None:
        return TimeoutFallbackResult(
            value=max(0.0, min(100.0, float(error_rate_pct))),
            level=TimeoutFallback.C_ERROR_RATE,
            source="error_rate_pct",
            notes=("error_rate is not a SLA-timeout counter — used as "
                   "fallback only"))

    sla = derive_sla_violation_pct(
        latency_p99_ms=latency_p99_ms,
        latency_sla_p99_ms=latency_sla_p99_ms)
    if sla is not None:
        return TimeoutFallbackResult(
            value=sla,
            level=TimeoutFallback.D_SLA_RISK_PROXY,
            source="latency_p99_ms vs latency_sla_p99_ms",
            notes=("last-resort proxy: per-tick binary; aggregate over a "
                   "window for a meaningful share"))

    return TimeoutFallbackResult(
        value=None, level=TimeoutFallback.NONE,
        source="", notes="no timeout signal available")
