"""Feature engineering for the Cache / Prefix-Reuse Forecaster v1.

Pure feature pipeline. Does NOT train models, NOT call any scheduler,
NOT mutate state.

Honesty rules (binding):

- ``LEAKAGE_TARGET_FIELDS`` lists every column that depends on cache
  state at or after the request runs. None of them are ever used as a
  feature in any model variant.
- ``reuse_percentage`` / ``reused_buckets`` are *labels*, never features.
- ``cache_hit`` is a post-decision observation, never a feature.
- Bucket-level rolling features must use ONLY rows whose timestamp
  precedes the current request's timestamp. Per-row build is
  chronological inside ``add_rolling_features``.
- ``bucket_count`` / ``bucket_ids_hash`` describe the bucket *set the
  request is sending* — both are observable BEFORE the request runs,
  so they are allowed as features.

Inputs are list-of-dict rows from SwissAI bucket-reuse files, CC-traces
flattened-request rows, LMCache agentic rows, or PrefixBench rows.
Outputs are numpy arrays + matching feature names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Leakage rules + canonical targets
# ---------------------------------------------------------------------------


# Fields that are only observable AT OR AFTER cache lookup. They MUST NOT
# appear in any model's feature set.
LEAKAGE_TARGET_FIELDS = frozenset({
    # SwissAI cache-residency targets / post-decision fields
    "reuse_percentage",
    "reused_buckets",
    "reused_bucket_count",
    # Post-completion outcome fields (latency etc. are leakage at
    # decision time)
    "actual_e2e_latency_s",
    "actual_ttft_s",
    "ttft_s",
    "api_time_s",
    "tpot_s",
    "actual_tpot_s",
    "actual_output_tokens",
    "output_tokens",  # post-completion in SwissAI/CC-traces format
    "cache_hit",
    "prefix_hit",
    "completion_timestamp_s",
    "completion_timestamp",
})


# Canonical targets the forecaster can train on. Names match
# ``data/external/hf/.../processed/summary.json::field_quality`` keys.
TARGETS = (
    "reuse_percentage",       # continuous 0-100 (SwissAI)
    "high_reuse",             # derived binary (reuse_percentage >= threshold)
    "intra_session_reuse",    # derived binary (CC-traces in-session reuse proxy)
)


HIGH_REUSE_THRESHOLD = 50.0  # >= 50% of buckets reused = "high reuse"


# Predict-time numeric features that are ALWAYS observable before the
# request is run.
PREDICT_TIME_NUMERIC_FEATURES = (
    "bucket_count",                          # SwissAI total_buckets
    "input_tokens",                          # SwissAI/CC-traces prompt size
    "predicted_output_tokens",               # forecast input only
    "turn_index",                            # CC-traces turn within session
    "session_turns_so_far",                  # rolling: how many turns precede this
    "session_requests_total",                # session size (CC-traces requests_count)
    "request_arrival_delta_s",               # CC-traces per-request arrival delta
    "think_time_s",                          # CC-traces inter-turn gap
    "pre_gap_s",                             # LMCache inter-turn gap
    # Rolling priors (computed in chronological order)
    "rolling_per_model_reuse_pct",
    "rolling_per_hash_seen_count",
    "rolling_per_session_mean_block_count",
    "rolling_session_last_block_count",
)


PREDICT_TIME_CATEGORICAL_FEATURES = (
    "model_id",
    "request_type",
    "bucket_size_bin",
    "input_token_bin",
    "hour_of_day",
)


# Pre-registered bin boundaries (never fitted from data).
BUCKET_SIZE_BINS = [(0, 2), (2, 16), (16, 64), (64, 256), (256, 1024),
                    (1024, 10_000_000)]
INPUT_TOKEN_BINS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384),
                    (16384, 65536), (65536, 10_000_000)]


def _bin_label(v, bins) -> str:
    if v is None:
        return "missing"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in bins:
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f">={bins[-1][1]}"


def bin_bucket_size(n) -> str:
    return _bin_label(n, BUCKET_SIZE_BINS)


def bin_input_tokens(n) -> str:
    return _bin_label(n, INPUT_TOKEN_BINS)


def _parse_swissai_iso(s) -> Optional[float]:
    """Parse SwissAI ``created_at_iso`` into unix seconds. Accepts both
    ``"2025-05-23 15:05:19.910"`` (qwen3_32b / llama3 / apertus) and
    ``"2025-10-10T16:17:11.338Z"`` (qwen380b configs). Returns ``None``
    on failure.
    """
    if s is None:
        return None
    s_str = str(s).rstrip("Z").replace("T", " ")
    fmts = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            dt = datetime.strptime(s_str[: len(fmt) + 6], fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except (TypeError, ValueError):
            continue
    return None


def hour_of_day_label(ts) -> str:
    if ts is None:
        return "missing"
    try:
        h = int(datetime.fromtimestamp(float(ts), tz=timezone.utc).hour)
        return f"hour={h}"
    except (TypeError, ValueError, OSError):
        return "missing"


# ---------------------------------------------------------------------------
# Source-aware row enrichment
# ---------------------------------------------------------------------------


def _coerce_numeric(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def enrich_row(row: dict, *, source: str) -> dict:
    """Add derived fields. Pure: does not mutate ``row``."""
    out = dict(row)
    # Canonical timestamp.
    ts = row.get("decision_timestamp_s")
    if ts is None:
        ts = _parse_swissai_iso(row.get("created_at_iso"))
    out["__decision_timestamp_s"] = ts

    # Canonical predicted_output_tokens — never the real output (leakage).
    # If a source carries a predicted/budgeted output count we use it.
    out["predicted_output_tokens"] = row.get("max_tokens")

    out["bucket_size_bin"] = bin_bucket_size(row.get("bucket_count"))
    out["input_token_bin"] = bin_input_tokens(row.get("input_tokens"))
    out["hour_of_day"] = hour_of_day_label(ts)
    out["__source"] = source
    return out


# ---------------------------------------------------------------------------
# Rolling features (chronological, decision-time only)
# ---------------------------------------------------------------------------


def add_rolling_features(rows: list[dict], *, source: str) -> list[dict]:
    """Return a new list of rows with rolling per-model / per-hash /
    per-session features. Rows are processed in ascending timestamp order
    so feature_i uses ONLY rows 0..i-1.

    Mutates a copy of each row. Does not change ordering of the input
    list (callers can re-sort).
    """
    enriched = [enrich_row(r, source=source) for r in rows]

    # Stable sort by timestamp (NaN at the end so they get the global
    # rolling prior, not an inflated one).
    def _ts(r):
        t = r.get("__decision_timestamp_s")
        return t if isinstance(t, (int, float)) and not _is_nan(t) else float("inf")

    order = sorted(range(len(enriched)), key=lambda i: _ts(enriched[i]))

    per_model_sum = {}
    per_model_n = {}
    per_hash_count = {}
    per_session_block_sum = {}
    per_session_block_n = {}
    per_session_last_block = {}
    per_session_turn = {}

    for idx in order:
        r = enriched[idx]
        m = r.get("model_id")
        h = r.get("bucket_ids_hash") or r.get("block_hashes_hash")
        sid = r.get("session_id")

        n = per_model_n.get(m, 0)
        if n > 0:
            r["rolling_per_model_reuse_pct"] = per_model_sum[m] / n
        else:
            r["rolling_per_model_reuse_pct"] = float("nan")
        r["rolling_per_hash_seen_count"] = float(per_hash_count.get(h, 0))
        n_s = per_session_block_n.get(sid, 0)
        if n_s > 0:
            r["rolling_per_session_mean_block_count"] = (
                per_session_block_sum[sid] / n_s)
        else:
            r["rolling_per_session_mean_block_count"] = float("nan")
        r["rolling_session_last_block_count"] = (
            per_session_last_block.get(sid, float("nan")))
        r["session_turns_so_far"] = float(per_session_turn.get(sid, 0))

        # Update rolling state with the OUTCOME of this row.
        rp = r.get("reuse_percentage")
        if isinstance(rp, (int, float)) and not _is_nan(rp):
            per_model_sum[m] = per_model_sum.get(m, 0.0) + float(rp)
            per_model_n[m] = n + 1
        per_hash_count[h] = per_hash_count.get(h, 0) + 1
        bc = r.get("bucket_count") or r.get("block_hashes_count")
        if isinstance(bc, (int, float)) and not _is_nan(bc):
            per_session_block_sum[sid] = per_session_block_sum.get(sid, 0.0) + float(bc)
            per_session_block_n[sid] = n_s + 1
            per_session_last_block[sid] = float(bc)
        per_session_turn[sid] = per_session_turn.get(sid, 0) + 1

    return enriched


def _is_nan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


class LeakageError(ValueError):
    """Raised when a feature flagged as leakage would be emitted."""


def assert_no_leakage(feature_names: Iterable[str]) -> None:
    bad = [f for f in feature_names if f in LEAKAGE_TARGET_FIELDS]
    if bad:
        raise LeakageError(
            f"feature pipeline would emit leakage columns {bad}; "
            "refusing to build feature matrix"
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Frozen feature specification — fixed across train + holdout."""

    categorical_levels: dict
    numeric_columns: tuple
    categorical_columns: tuple


def fit_categorical_levels(rows: Iterable[dict],
                           categorical_columns: tuple) -> dict:
    levels: dict = {c: set() for c in categorical_columns}
    for r in rows:
        for c in categorical_columns:
            v = r.get(c)
            if v is None:
                v = "missing"
            levels[c].add(str(v))
    return {c: sorted(v) for c, v in levels.items()}


def build_feature_spec(rows: list) -> FeatureSpec:
    numeric = tuple(PREDICT_TIME_NUMERIC_FEATURES)
    categorical = tuple(PREDICT_TIME_CATEGORICAL_FEATURES)
    assert_no_leakage(numeric)
    levels = fit_categorical_levels(rows, categorical)
    return FeatureSpec(
        categorical_levels=levels,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )


def build_feature_matrix(rows: list, spec: FeatureSpec):
    """Build ``(X, feature_names, group_keys)`` from ``rows`` under
    ``spec``. ``rows`` must already carry the rolling features (the
    caller should run ``add_rolling_features`` first).
    """
    assert_no_leakage(spec.numeric_columns)

    n = len(rows)
    numeric_cols = spec.numeric_columns
    cat_cols = spec.categorical_columns
    cat_levels = spec.categorical_levels

    feature_names: list[str] = list(numeric_cols)
    cat_index: list[tuple] = []
    for c in cat_cols:
        for lvl in cat_levels[c]:
            feature_names.append(f"{c}={lvl}")
        feature_names.append(f"{c}=__UNSEEN__")
        cat_index.append((c, cat_levels[c]))

    X = np.zeros((n, len(feature_names)), dtype=np.float64)
    group_arrays: dict[str, list] = {c: [] for c in (
        "model_id", "session_id", "bucket_size_bin", "input_token_bin",
        "request_type", "hour_of_day",
    )}

    for i, r in enumerate(rows):
        for j, col in enumerate(numeric_cols):
            X[i, j] = _coerce_numeric(r.get(col))
        col_offset = len(numeric_cols)
        for c, levels in cat_index:
            v = r.get(c)
            v = "missing" if v is None else str(v)
            try:
                idx = levels.index(v)
                X[i, col_offset + idx] = 1.0
            except ValueError:
                X[i, col_offset + len(levels)] = 1.0
            col_offset += len(levels) + 1
        for c in group_arrays.keys():
            group_arrays[c].append(r.get(c) if c != "session_id" else
                                   r.get("session_id"))

    group_keys = {c: np.array(v, dtype=object) for c, v in group_arrays.items()}
    return X, feature_names, group_keys


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def extract_reuse_percentage(rows: list) -> np.ndarray:
    out = np.fromiter(
        (_coerce_numeric(r.get("reuse_percentage")) for r in rows),
        dtype=np.float64, count=len(rows),
    )
    return out


def derive_high_reuse(reuse_pct: np.ndarray, *,
                      threshold: float = HIGH_REUSE_THRESHOLD) -> np.ndarray:
    """Binary label: 1 if reuse_percentage >= threshold else 0.

    NaN reuse stays NaN; callers must mask before training.
    """
    out = np.where(np.isnan(reuse_pct), np.nan,
                   (reuse_pct >= threshold).astype(np.float64))
    return out


def derive_intra_session_reuse_from_cc_traces(rows: list) -> np.ndarray:
    """Per-row binary label: 1 if the row reuses the prior turn's block
    hash within the same session (block_hashes_hash matches a previous
    turn's hash OR block count grew without resetting), 0 otherwise.

    Decision-time-safe: this label uses ONLY information at or before
    the row's turn_index. Caller passes the same rows the forecaster
    will train on.
    """
    by_session: dict[str, list[tuple]] = {}
    for i, r in enumerate(rows):
        sid = r.get("session_id")
        t = r.get("turn_index")
        by_session.setdefault(sid, []).append((i, t, r))
    out = np.zeros(len(rows), dtype=np.float64)
    for sid, items in by_session.items():
        items.sort(key=lambda x: (x[1] if x[1] is not None else 0))
        prev_hash = None
        prev_count = None
        for (i, t, r) in items:
            hh = r.get("block_hashes_hash") or r.get("bucket_ids_hash")
            bc = r.get("block_hashes_count") or r.get("bucket_count")
            reused = 0.0
            if prev_hash is not None and hh is not None:
                if hh == prev_hash:
                    reused = 1.0
                elif (prev_count is not None and bc is not None
                      and bc >= prev_count and prev_count >= 1):
                    reused = 1.0
            out[i] = reused
            prev_hash = hh
            prev_count = bc
    return out


# ---------------------------------------------------------------------------
# Holdout split helpers (deterministic)
# ---------------------------------------------------------------------------


def random_holdout(n: int, *, holdout_frac: float = 0.2,
                   seed: int = 19690720) -> tuple:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    return np.sort(perm[n_holdout:]), np.sort(perm[:n_holdout])


def holdout_by_group(group_values: np.ndarray, hold_groups: tuple) -> tuple:
    hold_mask = np.zeros(len(group_values), dtype=bool)
    for g in hold_groups:
        hold_mask |= (group_values == g)
    train_idx = np.where(~hold_mask)[0]
    holdout_idx = np.where(hold_mask)[0]
    return train_idx, holdout_idx


def time_holdout(timestamps: np.ndarray, *, holdout_frac: float = 0.2) -> tuple:
    n = len(timestamps)
    # NaN timestamps go to the front (train side) so they don't dominate
    # the holdout.
    ts = np.where(np.isnan(timestamps), -np.inf, timestamps)
    order = np.argsort(ts, kind="stable")
    n_holdout = int(round(n * holdout_frac))
    holdout_idx = np.sort(order[-n_holdout:])
    train_idx = np.sort(order[:-n_holdout])
    return train_idx, holdout_idx


def holdout_by_session(session_ids: np.ndarray, *,
                       holdout_frac: float = 0.2,
                       seed: int = 11242025) -> tuple:
    sids = np.asarray(
        [("__none__" if s is None else str(s)) for s in session_ids.tolist()],
        dtype=object)
    uniq = np.unique(sids)
    if uniq.size <= 1:
        return np.arange(len(sids), dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_hold = max(1, int(round(len(uniq) * holdout_frac)))
    hold_sessions = set(perm[:n_hold].tolist())
    hold_mask = np.array([s in hold_sessions for s in sids.tolist()], dtype=bool)
    return np.where(~hold_mask)[0], np.where(hold_mask)[0]
