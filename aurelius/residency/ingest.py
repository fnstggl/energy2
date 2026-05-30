"""Read-only ingestion adapters for model-residency / cold-start telemetry.

Three adapters, all **read-only** and all honest about what a source can and
cannot emit (``docs/PILOT_TELEMETRY_CONTRACT.md``):

1. :func:`import_events` / :func:`import_snapshots` / :func:`import_observations`
   — a **generic JSONL/CSV importer** for pilot/exported logs. It validates the
   schema (raising :class:`ResidencySchemaError` on a missing required field
   rather than zero-filling), normalises into the §Task-1 models, and reports
   per-field coverage so missingness is visible.

2. :func:`adapt_vllm` — a **vLLM observation adapter**. vLLM ``/metrics`` exposes
   ``prefix_cache_hit_rate``, queue and latency/token aggregates — but **no
   model-load events** and **no** ``model_loaded_before_request``. The adapter
   therefore forwards only what vLLM really exposes, sets every residency field
   to ``None`` (unknown), and emits an **empty** model-load-event list. It never
   invents a load event.

3. The Kubernetes/Prometheus/DCGM join helper lives in
   ``aurelius/residency/linkage.py`` (imported + re-exported here as
   :func:`join_request_to_infra`) and classifies linkage quality without
   fabricating joins.

Stdlib-only, deterministic, no network, no global state.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .linkage import build_linkage_report
from .models import (
    ModelResidencyEvent,
    ModelResidencySnapshot,
    RequestResidencyObservation,
    ResidencySchemaError,
)

# Re-export the join helper so callers can ingest + link from one module.
join_request_to_infra = build_linkage_report

_RECORD_BUILDERS = {
    "event": ModelResidencyEvent,
    "snapshot": ModelResidencySnapshot,
    "observation": RequestResidencyObservation,
}

# Fields that, across the model set, are the cross-layer join keys / residency
# flags whose absence blocks pilot readiness. Used by the coverage report.
_KEY_FIELDS = {
    "event": ("request_id", "container_id", "gpu_id", "node_id", "duration_s"),
    "snapshot": ("gpu_id", "container_id", "node_id", "loaded_model_ids",
                 "gpu_memory_used", "gpu_memory_total"),
    "observation": ("model_loaded_before_request", "adapter_loaded_before_request",
                    "model_load_latency_s", "container_id", "gpu_id", "node_id"),
}


@dataclass
class IngestResult:
    """Outcome of a generic import: the normalised records plus honest coverage.

    ``errors`` collects per-row validation failures (the row is skipped, never
    zero-filled). ``field_coverage`` maps every field to its non-null count so
    the audit can surface missingness (``docs/PILOT_TELEMETRY_CONTRACT.md`` §1).
    """

    record_type: str
    records: list = field(default_factory=list)
    n_rows: int = 0
    n_valid: int = 0
    errors: list = field(default_factory=list)
    field_coverage: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict:
        return {
            "record_type": self.record_type,
            "n_rows": self.n_rows,
            "n_valid": self.n_valid,
            "n_errors": self.n_errors,
            "errors": list(self.errors[:50]),  # cap for report readability
            "field_coverage": self.field_coverage,
            "sources": dict(self.sources),
        }


# ---------------------------------------------------------------------------
# Raw row readers (format auto-detection)
# ---------------------------------------------------------------------------

def _detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext in (".csv", ".tsv"):
        return "csv"
    # sniff: first non-blank char
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            return "jsonl" if s[0] in "{[" else "csv"
    return "jsonl"


def read_rows(path: str) -> list[dict]:
    """Read raw dict rows from a JSONL / JSON-array / CSV file (read-only)."""
    if not os.path.exists(path):
        raise ResidencySchemaError(f"telemetry file not found: {path}")
    fmt = _detect_format(path)
    rows: list[dict] = []
    if fmt == "csv":
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter=delim):
                rows.append(dict(row))
    elif fmt == "json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("records") or data.get("data") or [data]
        rows = [dict(r) for r in data]
    else:  # jsonl
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ResidencySchemaError(
                        f"{os.path.basename(path)} line {i + 1}: bad JSON: {exc}")
    return rows


# ---------------------------------------------------------------------------
# Generic importer
# ---------------------------------------------------------------------------

def _import(record_type: str, path: str, *, source: Optional[str],
            timestamp_unit: str) -> IngestResult:
    builder = _RECORD_BUILDERS[record_type]
    default_source = source or f"{os.path.splitext(os.path.basename(path))[0]}"
    rows = read_rows(path)
    result = IngestResult(record_type=record_type, n_rows=len(rows))
    for i, row in enumerate(rows):
        try:
            rec = builder.from_dict(row, source=default_source,
                                    timestamp_unit=timestamp_unit)
        except ResidencySchemaError as exc:
            result.errors.append(f"row {i}: {exc}")
            continue
        result.records.append(rec)
        result.sources[rec.source] = result.sources.get(rec.source, 0) + 1
    result.n_valid = len(result.records)
    result.field_coverage = _coverage(record_type, result.records)
    return result


def import_events(path: str, *, source: Optional[str] = None,
                  timestamp_unit: str = "auto") -> IngestResult:
    """Import :class:`ModelResidencyEvent` records from JSONL/CSV/JSON."""
    return _import("event", path, source=source, timestamp_unit=timestamp_unit)


def import_snapshots(path: str, *, source: Optional[str] = None,
                     timestamp_unit: str = "auto") -> IngestResult:
    """Import :class:`ModelResidencySnapshot` records from JSONL/CSV/JSON."""
    return _import("snapshot", path, source=source, timestamp_unit=timestamp_unit)


def import_observations(path: str, *, source: Optional[str] = None,
                        timestamp_unit: str = "auto") -> IngestResult:
    """Import :class:`RequestResidencyObservation` records from JSONL/CSV/JSON.

    Accepts both the ``docs/PILOT_TELEMETRY_CONTRACT.md`` §2 field names
    (``TTFT``, ``e2e_latency``, ``queue_wait``, ``lora_id`` …) and the model
    field names.
    """
    return _import("observation", path, source=source, timestamp_unit=timestamp_unit)


def _coverage(record_type: str, records: list) -> dict:
    """Per-field non-null coverage over parsed records (honest missingness)."""
    if not records:
        return {}
    fields = list(records[0].to_dict().keys())
    total = len(records)
    cov = {}
    for f in fields:
        present = 0
        for r in records:
            v = getattr(r, f, None)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            present += 1
        cov[f] = {
            "present": present,
            "total": total,
            "coverage": round(present / total, 4) if total else 0.0,
            "is_key_field": f in _KEY_FIELDS.get(record_type, ()),
        }
    return cov


# ---------------------------------------------------------------------------
# vLLM observation adapter (read-only; emits NO invented load events)
# ---------------------------------------------------------------------------

VLLM_RESIDENCY_OBSERVABLE = False  # vLLM /metrics exposes no model-load events


@dataclass(frozen=True)
class VLLMResidencyAdapterResult:
    """What vLLM can honestly contribute to residency observation.

    ``model_load_events`` is **always empty** — vLLM does not expose model-load
    timestamps. ``observations`` carry the real latency/token aggregates with
    *all residency flags ``None``* (unknown). ``prefix_cache_hit_rate`` is the
    one real residency-adjacent signal vLLM exposes (cache affinity, not model
    residency — see the readiness audit).
    """

    observations: list
    model_load_events: list  # invariant: always []
    prefix_cache_hit_rate: Optional[float]
    residency_observable: bool
    notes: list


def _vllm_metric(src, *names):
    """Fetch the first present attribute/key from an InferenceServiceState or
    a raw vLLM metrics dict."""
    for n in names:
        if isinstance(src, dict):
            if src.get(n) is not None:
                return src.get(n)
        else:
            v = getattr(src, n, None)
            if v is not None:
                return v
    return None


def _ms_to_s(v):
    return None if v is None else float(v) / 1000.0


def adapt_vllm(service_state, *, service_id: Optional[str] = None,
               timestamp: Optional[float] = None,
               source: str = "vllm") -> VLLMResidencyAdapterResult:
    """Build a residency view from a vLLM ``InferenceServiceState`` or metrics dict.

    HONESTY (``docs/PILOT_TELEMETRY_CONTRACT.md`` §2, readiness audit §1a):
    vLLM does not expose model/adapter load events or
    ``model_loaded_before_request``. This adapter therefore:

      * sets ``model_loaded_before_request`` / ``adapter_loaded_before_request``
        / ``*_load_latency_s`` to ``None`` (unknown),
      * emits **no** :class:`ModelResidencyEvent` (``model_load_events == []``),
      * forwards only the aggregate latency/token signals vLLM exposes, at
        ``confidence="low"`` (these are service-level aggregates, not per-request
        measurements),
      * surfaces ``prefix_cache_hit_rate`` separately as the one real
        residency-adjacent signal.
    """
    sid = service_id or _vllm_metric(service_state, "service_id") or "vllm-service"
    prefix = _vllm_metric(service_state, "prefix_cache_hit_rate")
    # vLLM exposes percentiles in ms on the InferenceServiceState; treat p50 as a
    # service-level aggregate proxy, NOT a per-request measurement.
    ttft = _ms_to_s(_vllm_metric(service_state, "ttft_p50_ms"))
    e2e = _ms_to_s(_vllm_metric(service_state, "p50_latency_ms", "e2e_p50_ms"))
    tpot = _ms_to_s(_vllm_metric(service_state, "tpot_p50_ms"))

    obs = RequestResidencyObservation(
        request_id=f"vllm-agg/{sid}",
        timestamp=float(timestamp) if timestamp is not None else 0.0,
        model_id=str(sid),
        source=source,
        model_loaded_before_request=None,   # UNKNOWN — vLLM cannot report this
        adapter_loaded_before_request=None,
        model_load_latency_s=None,
        adapter_load_latency_s=None,
        ttft_s=ttft,
        tpot_s=tpot,
        e2e_latency_s=e2e,
        status="OK",
        confidence="low",                   # service-level aggregate
    )
    notes = [
        "vLLM /metrics exposes no model-load events or "
        "model_loaded_before_request; all residency fields are None (unknown).",
        "observation values are service-level aggregates (p50), not per-request.",
    ]
    if prefix is not None:
        notes.append("prefix_cache_hit_rate forwarded as the one real "
                     "residency-adjacent signal (cache affinity, not residency).")
    return VLLMResidencyAdapterResult(
        observations=[obs],
        model_load_events=[],               # INVARIANT: never invent load events
        prefix_cache_hit_rate=(float(prefix) if prefix is not None else None),
        residency_observable=VLLM_RESIDENCY_OBSERVABLE,
        notes=notes,
    )
