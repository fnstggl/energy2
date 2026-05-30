"""Derived residency / cold-start metrics — honest about missing data.

Computes the ``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §3 diagnostics from the
ingested models. Every metric:

  * is reported as a :class:`MetricValue` with ``{value, numerator, denominator,
    window_s, linkage_quality, note}`` (contract §5),
  * **excludes unknown (``None``) inputs from its denominator** — a missing
    residency flag is never counted as a miss, a missing latency is never
    counted as ``0.0``,
  * returns ``value=None`` with ``note="insufficient_telemetry"`` when the
    denominator is empty, rather than a misleading ``0.0``.

These are **diagnostics only**. They are NEVER folded into the canonical KPI
(``docs/RESULTS.md`` §1–§2). No production-savings claim is implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: Optional[float]
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    window_s: Optional[float] = None
    linkage_quality: Optional[str] = None
    note: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "window_s": self.window_s,
            "linkage_quality": self.linkage_quality,
            "note": self.note,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


_INSUFFICIENT = "insufficient_telemetry"


def percentile(values, pct: float) -> Optional[float]:
    """Nearest-rank percentile over non-null values; ``None`` if no data."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if pct <= 0:
        return float(vals[0])
    if pct >= 100:
        return float(vals[-1])
    rank = max(1, min(len(vals), math.ceil((pct / 100.0) * len(vals))))
    return float(vals[rank - 1])


def _window_s(records) -> Optional[float]:
    ts = [getattr(r, "timestamp", None) for r in records]
    ts = [t for t in ts if t is not None]
    return (max(ts) - min(ts)) if len(ts) >= 2 else None


# ---------------------------------------------------------------------------
# Hit / miss rates
# ---------------------------------------------------------------------------

def model_residency_hit_rate(observations, *, linkage_quality=None) -> MetricValue:
    """Fraction of requests whose base model was resident, over requests with a
    **known** residency flag (unknowns excluded from the denominator)."""
    known = [o for o in observations if o.model_loaded_before_request is not None]
    hits = sum(1 for o in known if o.model_loaded_before_request is True)
    return MetricValue(
        name="model_residency_hit_rate",
        value=(hits / len(known)) if known else None,
        numerator=hits if known else None,
        denominator=len(known) if known else None,
        window_s=_window_s(observations),
        linkage_quality=linkage_quality,
        note=None if known else _INSUFFICIENT,
        extra={"unknown_excluded": sum(
            1 for o in observations if o.model_loaded_before_request is None)},
    )


def adapter_residency_hit_rate(observations, *, linkage_quality=None) -> MetricValue:
    """Fraction of adapter-using requests whose adapter was resident, over
    adapter requests with a **known** adapter residency flag."""
    adapter_reqs = [o for o in observations if o.adapter_id]
    known = [o for o in adapter_reqs
             if o.adapter_loaded_before_request is not None]
    hits = sum(1 for o in known if o.adapter_loaded_before_request is True)
    return MetricValue(
        name="adapter_residency_hit_rate",
        value=(hits / len(known)) if known else None,
        numerator=hits if known else None,
        denominator=len(known) if known else None,
        window_s=_window_s(observations),
        linkage_quality=linkage_quality,
        note=None if known else _INSUFFICIENT,
        extra={
            "adapter_requests": len(adapter_reqs),
            "adapter_flag_unknown": sum(
                1 for o in adapter_reqs
                if o.adapter_loaded_before_request is None),
        },
    )


def cold_start_rate(observations, *, linkage_quality=None) -> MetricValue:
    """Fraction of requests that triggered a model and/or adapter load, over
    requests that could be classified (``is_cold_start`` is not ``None``)."""
    classifiable = [o for o in observations if o.is_cold_start is not None]
    cold = sum(1 for o in classifiable if o.is_cold_start is True)
    return MetricValue(
        name="cold_start_rate",
        value=(cold / len(classifiable)) if classifiable else None,
        numerator=cold if classifiable else None,
        denominator=len(classifiable) if classifiable else None,
        window_s=_window_s(observations),
        linkage_quality=linkage_quality,
        note=None if classifiable else _INSUFFICIENT,
        extra={"unclassifiable": len(observations) - len(classifiable)},
    )


# ---------------------------------------------------------------------------
# Load-latency distributions
# ---------------------------------------------------------------------------

def _latency_percentiles(name: str, latencies, observations,
                         linkage_quality) -> dict:
    vals = [v for v in latencies if v is not None]
    out = {}
    for p in (50, 95, 99):
        out[f"{name}_p{p}"] = MetricValue(
            name=f"{name}_p{p}",
            value=percentile(vals, p) if vals else None,
            denominator=len(vals) if vals else None,
            window_s=_window_s(observations),
            linkage_quality=linkage_quality,
            note=None if vals else _INSUFFICIENT,
        )
    return out


def model_load_latency_percentiles(observations, *, linkage_quality=None) -> dict:
    """p50/p95/p99 of measured base-model load latency (cold requests only)."""
    return _latency_percentiles(
        "model_load_latency",
        [o.model_load_latency_s for o in observations],
        observations, linkage_quality)


def adapter_load_latency_percentiles(observations, *, linkage_quality=None) -> dict:
    """p50/p95/p99 of measured adapter load latency."""
    return _latency_percentiles(
        "adapter_load_latency",
        [o.adapter_load_latency_s for o in observations],
        observations, linkage_quality)


# ---------------------------------------------------------------------------
# SLA attribution
# ---------------------------------------------------------------------------

def cold_start_attributed_sla_violations(observations, *, slo_s: Optional[float],
                                         linkage_quality=None) -> MetricValue:
    """Count of SLA-violating requests that **would have met SLO** without the
    cold-start load (``e2e_latency - load_latency <= slo``).

    Requires an ``slo_s`` and a measured load latency + e2e on the offending
    requests; otherwise returns ``None`` (insufficient telemetry), never ``0``.
    """
    if slo_s is None:
        return MetricValue(
            name="cold_start_attributed_sla_violations", value=None,
            window_s=_window_s(observations), linkage_quality=linkage_quality,
            note="no_slo_provided")
    evaluable = [o for o in observations
                 if o.e2e_latency_s is not None and o.total_load_latency_s is not None]
    attributed = 0
    violations = 0
    for o in evaluable:
        if o.e2e_latency_s > slo_s:
            violations += 1
            if (o.e2e_latency_s - o.total_load_latency_s) <= slo_s:
                attributed += 1
    return MetricValue(
        name="cold_start_attributed_sla_violations",
        value=float(attributed) if evaluable else None,
        numerator=attributed if evaluable else None,
        denominator=violations if evaluable else None,
        window_s=_window_s(observations),
        linkage_quality=linkage_quality,
        note=None if evaluable else _INSUFFICIENT,
        extra={"slo_s": slo_s, "evaluable_requests": len(evaluable),
               "total_violations": violations},
    )


# ---------------------------------------------------------------------------
# Warm-pool cost (snapshot-based)
# ---------------------------------------------------------------------------

def warm_pool_gpu_hours(snapshots, *, gpu_hour_cost: Optional[float] = None,
                        max_gap_s: float = 3600.0,
                        linkage_quality=None) -> MetricValue:
    """GPU-hours of capacity held *resident* across the snapshot stream.

    Integrates, per ``gpu_id`` (falling back to ``container_id``), the time a GPU
    is observed holding ≥1 resident model. Each interval between consecutive
    snapshots for the same GPU contributes ``dt`` GPU-hours when the earlier
    snapshot reported residency; intervals longer than ``max_gap_s`` are dropped
    (stale — we do not extrapolate residency across a gap). Returns ``None`` when
    snapshots lack the timestamps/keys/residency needed — never ``0`` by
    assumption.

    If ``gpu_hour_cost`` is given, the dollar cost is reported in ``extra`` (the
    denominator of every prewarm decision); the headline value stays GPU-hours.
    """
    by_gpu: dict = {}
    for s in snapshots:
        key = s.gpu_id or s.container_id
        if key is None or s.timestamp is None or s.loaded_model_ids is None:
            continue  # cannot attribute residency-time without key + residency report
        by_gpu.setdefault(key, []).append(s)

    if not by_gpu:
        return MetricValue(
            name="warm_pool_gpu_hours", value=None,
            linkage_quality=linkage_quality, note=_INSUFFICIENT,
            extra={"reason": "no snapshots with gpu/container key + residency"})

    total_h = 0.0
    intervals = 0
    dropped_stale = 0
    for key, snaps in by_gpu.items():
        snaps = sorted(snaps, key=lambda x: x.timestamp)
        for a, b in zip(snaps, snaps[1:]):
            dt = b.timestamp - a.timestamp
            if dt <= 0:
                continue
            if dt > max_gap_s:
                dropped_stale += 1
                continue
            if a.has_residency:
                total_h += dt / 3600.0
                intervals += 1

    note = None if intervals else _INSUFFICIENT
    extra = {"gpus": len(by_gpu), "resident_intervals": intervals,
             "dropped_stale_intervals": dropped_stale}
    if gpu_hour_cost is not None:
        extra["warm_pool_cost"] = round(total_h * gpu_hour_cost, 4)
        extra["gpu_hour_cost"] = gpu_hour_cost
    return MetricValue(
        name="warm_pool_gpu_hours",
        value=round(total_h, 6) if intervals else None,
        window_s=_window_s(snapshots),
        linkage_quality=linkage_quality,
        note=note,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------

def residency_churn_score(events, *, linkage_quality=None) -> MetricValue:
    """Load+evict events per model per hour — a thrash / cost diagnostic.

    Counts ``model_load_end`` / ``adapter_load_end`` / ``*_evict`` events
    (completed residency transitions). Returns ``None`` when there are no events
    or no usable time window.
    """
    transitions = [e for e in events
                   if e.event_type in ("model_load_end", "adapter_load_end",
                                       "model_evict", "adapter_evict")]
    if not transitions:
        return MetricValue(name="residency_churn_score", value=None,
                           linkage_quality=linkage_quality, note=_INSUFFICIENT)
    models = {e.model_id for e in transitions if e.model_id}
    window = _window_s(transitions)
    hours = (window / 3600.0) if window else None
    n_models = max(1, len(models))
    if not hours:
        return MetricValue(
            name="residency_churn_score", value=None,
            numerator=len(transitions), denominator=n_models,
            linkage_quality=linkage_quality, note=_INSUFFICIENT,
            extra={"transitions": len(transitions), "distinct_models": len(models),
                   "reason": "insufficient time window"})
    score = len(transitions) / n_models / hours
    return MetricValue(
        name="residency_churn_score",
        value=round(score, 6),
        numerator=len(transitions),
        denominator=n_models,
        window_s=window,
        linkage_quality=linkage_quality,
        extra={"transitions": len(transitions), "distinct_models": len(models),
               "window_hours": round(hours, 4)},
    )


# ---------------------------------------------------------------------------
# Missingness / telemetry confidence
# ---------------------------------------------------------------------------

_CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3, "unknown": 0.0}


def telemetry_confidence(observations, events, snapshots, *,
                         linkage_quality=None) -> MetricValue:
    """A blended, honest confidence in the residency telemetry, in ``[0, 1]``.

    Combines (a) the fraction of observations carrying a **known** model
    residency flag, (b) the fraction carrying a join key, (c) the mean
    provenance-confidence weight of all records, and (d) the linkage quality.
    Low when the substrate cannot attribute or classify residency.
    """
    obs = list(observations)
    n = len(obs)
    known_flag = (sum(1 for o in obs if o.model_loaded_before_request is not None) / n
                  if n else 0.0)
    keyed = (sum(1 for o in obs if o.has_join_keys) / n) if n else 0.0

    all_records = obs + list(events) + list(snapshots)
    if all_records:
        prov = sum(_CONFIDENCE_WEIGHT.get(getattr(r, "confidence", "unknown"), 0.0)
                   for r in all_records) / len(all_records)
    else:
        prov = 0.0

    link_w = {"exact_join": 1.0, "container_join": 0.8, "time_join": 0.4,
              "no_join": 0.1}.get(linkage_quality or "no_join", 0.1)

    if not all_records:
        return MetricValue(name="telemetry_confidence", value=None,
                           note=_INSUFFICIENT, linkage_quality=linkage_quality)

    score = round(0.30 * known_flag + 0.25 * keyed + 0.25 * prov + 0.20 * link_w, 4)
    return MetricValue(
        name="telemetry_confidence",
        value=score,
        linkage_quality=linkage_quality,
        extra={
            "known_residency_flag_frac": round(known_flag, 4),
            "join_key_frac": round(keyed, 4),
            "mean_provenance_weight": round(prov, 4),
            "linkage_weight": link_w,
            "n_observations": n,
            "n_events": len(events),
            "n_snapshots": len(snapshots),
        },
    )


def missingness(observations, events, snapshots) -> dict:
    """Fraction of *missing* (null) values per required field, per stream.

    A field that is structurally absent is reported as missing — never silently
    zero-filled (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1).
    """
    def _miss(records, fields):
        if not records:
            return {f: {"missing": 0, "total": 0, "missing_frac": None}
                    for f in fields}
        out = {}
        total = len(records)
        for f in fields:
            miss = 0
            for r in records:
                v = getattr(r, f, None)
                if v is None or (isinstance(v, str) and not v.strip()):
                    miss += 1
            out[f] = {"missing": miss, "total": total,
                      "missing_frac": round(miss / total, 4)}
        return out

    return {
        "observations": _miss(observations, (
            "model_loaded_before_request", "adapter_loaded_before_request",
            "model_load_latency_s", "e2e_latency_s", "ttft_s",
            "container_id", "gpu_id", "node_id")),
        "events": _miss(events, (
            "duration_s", "request_id", "container_id", "gpu_id", "node_id")),
        "snapshots": _miss(snapshots, (
            "loaded_model_ids", "gpu_memory_used", "gpu_memory_total",
            "gpu_id", "container_id")),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(observations, events=None, snapshots=None, *,
                        slo_s: Optional[float] = None,
                        gpu_hour_cost: Optional[float] = None,
                        linkage_quality: Optional[str] = None) -> dict:
    """Compute every §3 metric. Returns a name → :class:`MetricValue` dict.

    All metrics are diagnostics; none is folded into the canonical KPI
    (``docs/RESULTS.md`` §1).
    """
    observations = list(observations or [])
    events = list(events or [])
    snapshots = list(snapshots or [])

    out: dict = {}
    out["model_residency_hit_rate"] = model_residency_hit_rate(
        observations, linkage_quality=linkage_quality)
    out["adapter_residency_hit_rate"] = adapter_residency_hit_rate(
        observations, linkage_quality=linkage_quality)
    out["cold_start_rate"] = cold_start_rate(
        observations, linkage_quality=linkage_quality)
    out.update(model_load_latency_percentiles(
        observations, linkage_quality=linkage_quality))
    out.update(adapter_load_latency_percentiles(
        observations, linkage_quality=linkage_quality))
    out["cold_start_attributed_sla_violations"] = cold_start_attributed_sla_violations(
        observations, slo_s=slo_s, linkage_quality=linkage_quality)
    out["warm_pool_gpu_hours"] = warm_pool_gpu_hours(
        snapshots, gpu_hour_cost=gpu_hour_cost, linkage_quality=linkage_quality)
    out["residency_churn_score"] = residency_churn_score(
        events, linkage_quality=linkage_quality)
    out["telemetry_confidence"] = telemetry_confidence(
        observations, events, snapshots, linkage_quality=linkage_quality)
    return out
