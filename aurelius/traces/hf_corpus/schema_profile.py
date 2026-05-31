"""Schema profiler for raw HF dataset records.

Produces the audit artefacts required by the mission spec PHASE 2:

- ``schema_profile.json`` — every observed raw column / nested key, its
  dtype, presence rate, missing rate, and a short example value per key.
- ``schema_mapping.json`` — the manual mapping from raw column ->
  normalised field, with field_quality + aurelius_signal_category +
  usable_for labels per raw column.

Heterogeneous JSON records are handled by inspecting every row in the
inspected sample, tracking presence rates per key. Nested ``dict`` keys
are flattened one level (``schedule_state.num_running`` etc); nested
``list`` columns are recorded with their length distribution + a
representative element key list.

The profiler is intentionally honest about missing data:

- ``presence_rate`` = fraction of rows where the key was present.
- ``non_null_rate`` = fraction where the key was present AND not ``None``
  AND not the sentinel ``-1`` (SwissAI uses ``-1`` for "unavailable").
- ``missing_rate`` = ``1 - non_null_rate``.

Nothing here downloads data; the caller supplies ``rows: list[dict]``.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from typing import Iterable, Optional


# Sentinel values that should be treated as missing for numeric counts.
# SwissAI's trace.jsonl uses ``-1`` for unavailable token counts; CARA uses
# ``0`` for some never-used scheduler fields (token_budget_per_iter etc).
NUMERIC_MISSING_SENTINELS = {-1}


def _dtype_of(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int64"
    if isinstance(v, float):
        return "float64"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _truncate_example(v, max_len: int = 80):
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= max_len else v[:max_len] + "...(truncated)"
    if isinstance(v, list):
        if not v:
            return []
        if len(v) <= 5:
            return [_truncate_example(x, max_len) for x in v]
        return [
            _truncate_example(v[0], max_len),
            _truncate_example(v[1], max_len),
            "...(+%d more)" % (len(v) - 2),
        ]
    if isinstance(v, dict):
        return {k: _truncate_example(val, max_len) for k, val in list(v.items())[:5]}
    return str(v)[:max_len]


def _flatten_one_level(row: dict) -> dict:
    """Flatten 1-level nested dicts. Lists are recorded as length+sample."""
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                out[f"{k}.{nk}"] = nv
            # Also keep an opaque sentinel for the parent key so it is
            # recorded as observed.
            out[k] = "<dict>"
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = v
    return out


def profile_rows(
    rows: list[dict],
    *,
    dataset_id: str,
    config_name: Optional[str],
    split: Optional[str],
    source_files_inspected: list,
    file_size_bytes: Optional[int] = None,
) -> dict:
    """Build the schema_profile.json payload from inspected ``rows``."""

    inspected = len(rows)
    if inspected == 0:
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "inspected_row_count": 0,
            "raw_columns": [],
            "nested_keys": [],
            "dtypes": {},
            "presence_rates": {},
            "missing_rates": {},
            "example_values": {},
            "list_length_summaries": {},
            "source_files_inspected": list(source_files_inspected),
            "file_size_bytes": file_size_bytes,
        }

    # Track per-flat-key.
    key_counts: dict = {}
    key_non_null_counts: dict = {}
    key_dtypes: dict = {}
    key_examples: dict = {}
    list_lengths: dict = {}

    top_level_keys: set = set()
    nested_keys: set = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in row.keys():
            top_level_keys.add(k)
        flat = _flatten_one_level(row)
        for fk, fv in flat.items():
            if "." in fk:
                nested_keys.add(fk)
            key_counts[fk] = key_counts.get(fk, 0) + 1
            non_null = fv is not None
            if isinstance(fv, (int, float)) and not isinstance(fv, bool):
                if fv in NUMERIC_MISSING_SENTINELS:
                    non_null = False
            if isinstance(fv, str) and fv.lower() == "null":
                non_null = False
            if non_null:
                key_non_null_counts[fk] = key_non_null_counts.get(fk, 0) + 1
            # First non-null value as the example; otherwise first observed.
            if fk not in key_examples or (
                non_null and key_examples[fk] in (None, "<dict>")
            ):
                key_examples[fk] = _truncate_example(fv)
            # dtype: pick most-common non-null dtype seen so far.
            dt = _dtype_of(fv)
            key_dtypes.setdefault(fk, dt)
            if isinstance(fv, list):
                list_lengths.setdefault(fk, []).append(len(fv))

    presence_rates = {k: round(v / inspected, 6) for k, v in key_counts.items()}
    missing_rates = {
        k: round(1.0 - (key_non_null_counts.get(k, 0) / inspected), 6)
        for k in key_counts.keys()
    }

    list_summaries: dict = {}
    for k, lens in list_lengths.items():
        list_summaries[k] = {
            "samples": len(lens),
            "min_len": min(lens),
            "max_len": max(lens),
            "mean_len": round(statistics.fmean(lens), 4),
        }

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "inspected_row_count": inspected,
        "raw_columns": sorted(top_level_keys),
        "nested_keys": sorted(nested_keys),
        "dtypes": dict(sorted(key_dtypes.items())),
        "presence_rates": dict(sorted(presence_rates.items())),
        "missing_rates": dict(sorted(missing_rates.items())),
        "example_values": dict(sorted(key_examples.items())),
        "list_length_summaries": dict(sorted(list_summaries.items())),
        "source_files_inspected": list(source_files_inspected),
        "file_size_bytes": file_size_bytes,
    }


def write_schema_profile(profile: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(profile, fh, indent=2, sort_keys=True)


def build_schema_mapping(
    profile: dict,
    column_mapping: dict,
    *,
    dataset_id: str,
    config_name: Optional[str],
) -> dict:
    """Compose the per-column mapping audit artefact.

    ``column_mapping`` is a dict keyed by raw_column_name (or flattened
    nested key) -> dict with the per-column metadata documented in the
    PHASE 2 mission spec (normalized_field, field_quality, units,
    aurelius_signal_category, usable_for, notes).

    Columns observed in ``profile`` but absent from ``column_mapping`` are
    recorded as ``rejected_columns`` (status='unmapped') so promotion gates
    catch them.
    """

    observed = set(profile.get("raw_columns") or []) | set(
        profile.get("nested_keys") or []
    )
    mapped_keys = set(column_mapping.keys())

    accepted = sorted(observed & mapped_keys)
    rejected = sorted(observed - mapped_keys)
    extra_mapped = sorted(mapped_keys - observed)  # listed but not observed

    out_columns = []
    for col in sorted(observed):
        m = column_mapping.get(col) or {
            "normalized_field": None,
            "field_quality": "unknown",
            "units": None,
            "aurelius_signal_category": "metadata_only",
            "usable_for": ["not_usable"],
            "notes": "raw column not mapped in column_mapping; rejected",
        }
        entry = {
            "raw_column_name": col,
            "normalized_field": m.get("normalized_field"),
            "field_quality": m.get("field_quality", "unknown"),
            "units": m.get("units"),
            "aurelius_signal_category": m.get(
                "aurelius_signal_category", "metadata_only"),
            "usable_for": list(m.get("usable_for") or []),
            "notes": m.get("notes"),
            "presence_rate": profile.get("presence_rates", {}).get(col),
            "missing_rate": profile.get("missing_rates", {}).get(col),
        }
        out_columns.append(entry)

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "accepted_columns": accepted,
        "rejected_columns": rejected,
        "extra_mapped_not_observed": extra_mapped,
        "columns": out_columns,
    }


def write_schema_mapping(mapping: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Stratified sampling + summary statistics
# ---------------------------------------------------------------------------


def stratify_indices(
    rows: list[dict], stratification_keys: list, per_stratum_cap: int,
) -> tuple[list, dict]:
    """Return (kept_indices, subgroup_counts).

    Stratifies by the tuple of values for ``stratification_keys`` (missing
    keys yield ``None`` in the tuple). Each stratum is capped at
    ``per_stratum_cap`` rows (first-come). ``subgroup_counts`` is a dict
    ``{stratum_str: kept_count}``.
    """
    if not stratification_keys:
        return list(range(min(len(rows), per_stratum_cap))), {
            "__all__": min(len(rows), per_stratum_cap)
        }
    bins: dict = {}
    kept: list = []
    counts: dict = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        key = tuple(str(r.get(k)) for k in stratification_keys)
        key_str = "|".join(key)
        bins.setdefault(key_str, 0)
        if bins[key_str] < per_stratum_cap:
            bins[key_str] += 1
            kept.append(i)
            counts[key_str] = counts.get(key_str, 0) + 1
    return kept, counts


def compute_numeric_summary(
    rows: list[dict],
    *,
    field: str,
    sentinel_missing=NUMERIC_MISSING_SENTINELS,
) -> dict:
    """p50/p90/p95/p99 + min/max/mean over a numeric field."""
    vals: list[float] = []
    for r in rows:
        v = r.get(field) if isinstance(r, dict) else None
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            if v in sentinel_missing:
                continue
            vals.append(float(v))
    if not vals:
        return {"count": 0, "missing": len(rows)}
    vs = sorted(vals)

    def _p(p):
        idx = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
        return float(vs[idx])

    return {
        "count": len(vals),
        "missing": len(rows) - len(vals),
        "min": vs[0],
        "max": vs[-1],
        "mean": float(statistics.fmean(vs)),
        "median": float(statistics.median(vs)),
        "p90": _p(90),
        "p95": _p(95),
        "p99": _p(99),
    }


# Subgroup-size threshold below which p99 is INSUFFICIENT_SAMPLE.
MIN_ROWS_FOR_P99 = 100
MIN_ROWS_FOR_P95 = 50


def per_subgroup_latency_summary(
    rows: list[dict],
    *,
    field: str,
    stratification_keys: list,
) -> dict:
    """Per-subgroup numeric summary with insufficient-sample flagging."""
    subgroups: dict = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = "|".join(str(r.get(k)) for k in stratification_keys)
        subgroups.setdefault(key, []).append(r)

    out: dict = {}
    insufficient: list = []
    for key, group_rows in subgroups.items():
        summary = compute_numeric_summary(group_rows, field=field)
        n = summary.get("count", 0)
        flags = []
        if n < MIN_ROWS_FOR_P95:
            flags.append("INSUFFICIENT_SAMPLE_P95")
            summary["p95"] = None
            summary["p99"] = None
            insufficient.append(key)
        elif n < MIN_ROWS_FOR_P99:
            flags.append("INSUFFICIENT_SAMPLE_P99")
            summary["p99"] = None
        summary["flags"] = flags
        out[key] = summary
    return {
        "subgroups": out,
        "insufficient_sample_groups": insufficient,
        "stratification_keys": list(stratification_keys),
    }


def hash_bucket_ids(bucket_ids) -> str:
    if not bucket_ids:
        return ""
    s = ",".join(str(x) for x in bucket_ids)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def sample_bucket_ids(bucket_ids, max_keep: int = 5) -> str:
    if not bucket_ids:
        return ""
    head = bucket_ids[:max_keep]
    if len(bucket_ids) > max_keep:
        return ",".join(str(x) for x in head) + f",...(+{len(bucket_ids) - max_keep})"
    return ",".join(str(x) for x in head)
