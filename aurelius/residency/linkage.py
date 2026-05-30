"""Cross-layer linkage helper for residency telemetry — no fake joins.

Residency/cold-start attribution requires linking a per-request observation to
the container/GPU/node that served it (``docs/PILOT_TELEMETRY_CONTRACT.md`` §4).
This module classifies the join quality *from the data*, exactly as the GenAI
ingester does (``aurelius/traces/alibaba_genai.py::classify_linkage``), and
never fabricates a join key the data does not contain:

  * ``exact_join``     — a shared ``request_id`` across the two streams.
  * ``container_join`` — shared ``container_id`` (+ ``gpu_id`` when present) and
                         overlapping time.
  * ``time_join``      — only timestamps align (same clock base, overlapping).
  * ``no_join``        — usable only independently.

**Honesty gate** (binding, contract §4): if the request↔infra linkage is
``no_join`` or ``time_join`` only, residency metrics are *calibration-only /
unattributed* and per-request request→GPU causality MUST NOT be claimed. The
``LinkageReport.attributable`` flag exposes this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

LINKAGE_QUALITIES = ("exact_join", "container_join", "time_join", "no_join")
# rank: higher is a stronger join
_RANK = {q: i for i, q in enumerate(reversed(LINKAGE_QUALITIES))}

# A join is good enough to *attribute* residency (not merely calibrate).
_ATTRIBUTABLE = frozenset({"exact_join", "container_join"})


def _ts(rec) -> Optional[float]:
    return getattr(rec, "timestamp", None)


def _ts_range(records) -> tuple[Optional[float], Optional[float]]:
    ts = [_ts(r) for r in records]
    ts = [t for t in ts if t is not None]
    return (min(ts), max(ts)) if ts else (None, None)


def _container_keys(records) -> set:
    out = set()
    for r in records:
        cid = getattr(r, "container_id", None)
        if cid:
            out.add(cid)
    return out


def _gpu_keys(records) -> set:
    out = set()
    for r in records:
        gid = getattr(r, "gpu_id", None)
        if gid:
            out.add(gid)
    return out


def _request_ids(records) -> set:
    out = set()
    for r in records:
        rid = getattr(r, "request_id", None)
        if rid:
            out.add(rid)
    return out


def classify_linkage(left, right, *, time_tolerance_s: float = 0.0) -> str:
    """Classify the join quality between two record streams FROM THE DATA.

    ``time_tolerance_s`` widens the time-overlap test (e.g. snapshots sampled on
    a coarse cadence). Returns one of :data:`LINKAGE_QUALITIES`.
    """
    left = list(left)
    right = list(right)
    if not left or not right:
        return "no_join"

    shared_rids = _request_ids(left) & _request_ids(right)
    if shared_rids:
        return "exact_join"

    a0, a1 = _ts_range(left)
    b0, b1 = _ts_range(right)
    time_overlap = (
        a0 is not None and b0 is not None
        and (min(a1, b1) + time_tolerance_s) >= (max(a0, b0) - time_tolerance_s)
    )

    shared_containers = _container_keys(left) & _container_keys(right)
    if shared_containers and time_overlap:
        return "container_join"
    if time_overlap:
        return "time_join"
    return "no_join"


@dataclass(frozen=True)
class LinkageReport:
    """Linkage classification of a request stream against an infra stream."""

    quality: str                      # overall (dominant achievable) linkage
    n_left: int                       # request observations
    n_right: int                      # infra records (snapshots/events)
    n_joined: int                     # observations with an attributable match
    per_quality: dict                 # per-observation quality histogram
    has_request_id_key: bool
    has_container_key: bool
    has_gpu_key: bool
    notes: list = field(default_factory=list)

    @property
    def attributable(self) -> bool:
        """Whether per-request request→GPU causality may be claimed (contract §4
        honesty gate). False for ``time_join`` / ``no_join``."""
        return self.quality in _ATTRIBUTABLE

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "attributable": self.attributable,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_joined": self.n_joined,
            "per_quality": dict(self.per_quality),
            "has_request_id_key": self.has_request_id_key,
            "has_container_key": self.has_container_key,
            "has_gpu_key": self.has_gpu_key,
            "notes": list(self.notes),
        }


def _pair_quality(obs, infra_rec, *, time_tolerance_s: float) -> str:
    """Best join quality between one observation and one infra record."""
    o_rid = getattr(obs, "request_id", None)
    if o_rid and getattr(infra_rec, "request_id", None) == o_rid:
        return "exact_join"
    o_cid = getattr(obs, "container_id", None)
    i_cid = getattr(infra_rec, "container_id", None)
    o_ts, i_ts = _ts(obs), _ts(infra_rec)
    time_ok = (o_ts is not None and i_ts is not None
               and abs(o_ts - i_ts) <= time_tolerance_s)
    if o_cid and i_cid and o_cid == i_cid:
        # same container but a different GPU → inconsistent, refuse the join
        o_gid = getattr(obs, "gpu_id", None)
        i_gid = getattr(infra_rec, "gpu_id", None)
        if o_gid and i_gid and o_gid != i_gid:
            return "no_join"
        # container key matches; require time proximity when both carry a clock
        if o_ts is None or i_ts is None or time_ok:
            return "container_join"
        return "no_join"
    if time_ok:
        return "time_join"
    return "no_join"


def best_match(obs, infra_records, *, time_tolerance_s: float = 300.0):
    """Find the strongest-joining infra record for one observation.

    Returns ``(record, quality)`` or ``(None, "no_join")``. Never fabricates a
    join — a record is only returned when a real key (request_id / container_id)
    or an in-tolerance timestamp links them. Among equal-quality candidates the
    nearest-in-time one wins.
    """
    best_rec = None
    best_q = "no_join"
    best_dt = None
    for rec in infra_records:
        q = _pair_quality(obs, rec, time_tolerance_s=time_tolerance_s)
        if q == "no_join":
            continue
        o_ts, i_ts = _ts(obs), _ts(rec)
        dt = abs(o_ts - i_ts) if (o_ts is not None and i_ts is not None) else float("inf")
        if (_RANK[q] > _RANK[best_q]) or (_RANK[q] == _RANK[best_q]
                                          and (best_dt is None or dt < best_dt)):
            best_rec, best_q, best_dt = rec, q, dt
    return best_rec, best_q


def build_linkage_report(observations, infra_records, *,
                         time_tolerance_s: float = 300.0) -> LinkageReport:
    """Classify how a request stream links to an infra stream (snapshots/events).

    Computes the overall (dominant) join quality plus a per-observation
    histogram and the join-key availability, so the audit can apply the contract
    §4 honesty gate.
    """
    observations = list(observations)
    infra_records = list(infra_records)

    per_quality = {q: 0 for q in LINKAGE_QUALITIES}
    n_joined = 0
    for obs in observations:
        _, q = best_match(obs, infra_records, time_tolerance_s=time_tolerance_s)
        per_quality[q] += 1
        if q in _ATTRIBUTABLE:
            n_joined += 1

    overall = classify_linkage(observations, infra_records,
                               time_tolerance_s=time_tolerance_s)

    notes = []
    if not observations or not infra_records:
        notes.append("one or both streams empty → no_join")
    if overall not in _ATTRIBUTABLE:
        notes.append("linkage is calibration-only / unattributed: per-request "
                     "request→GPU causality MUST NOT be claimed (contract §4)")
    n_no_keys = sum(1 for o in observations if not getattr(o, "has_join_keys", False))
    if n_no_keys:
        notes.append(f"{n_no_keys}/{len(observations)} observations carry no "
                     "node/gpu/container join key")

    return LinkageReport(
        quality=overall,
        n_left=len(observations),
        n_right=len(infra_records),
        n_joined=n_joined,
        per_quality=per_quality,
        has_request_id_key=bool(_request_ids(observations) & _request_ids(infra_records)),
        has_container_key=bool(_container_keys(observations)
                               & _container_keys(infra_records)),
        has_gpu_key=bool(_gpu_keys(observations) & _gpu_keys(infra_records)),
        notes=notes,
    )
