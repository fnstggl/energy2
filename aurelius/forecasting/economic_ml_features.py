"""Economic ML Alpha v1 — strict, leakage-safe feature contract.

Builds modular feature matrices from the analysis-tier Economic Overlay
corpus (``data/external/economic_overlay/analysis_corpus/*.jsonl``). Every
feature preserves the source row's ``value_quality`` provenance, and a hard
``LeakageError`` is raised if any blocked (post-decision / target-derived)
field is requested as a feature for a given target.

Binding rules (mission Phase 3):
  - The target itself, and any post-decision actual (actual cost / latency /
    reuse / goodput / SLA outcome), is NEVER a feature.
  - ``output_tokens`` is decision-time-UNSAFE for latency/cost targets that it
    drives (E2E / TPOT / decode / goodput). It is allowed only for targets
    where it is genuinely a known request attribute (e.g. predicting prompt-
    side reuse) — and even then the catalog marks it. Use
    ``predicted_output_tokens`` in production; here we simply block it.
  - No invented constants. Categorical features are one-hot/ordinal encoded
    from observed values only.

This module does NOT touch any production scheduler / scorer / residency /
frontier module. It is import-only research code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = (REPO_ROOT / "data" / "external" / "economic_overlay"
              / "analysis_corpus")


class LeakageError(ValueError):
    """Raised when a blocked (post-decision / target-derived) field is
    requested as a feature for a given target."""


# ---------------------------------------------------------------------------
# Leakage blocklist — per-target post-decision / target-derived fields.
# ---------------------------------------------------------------------------

# Fields that are ALWAYS post-decision actuals (never decision-time features).
ALWAYS_BLOCKED = frozenset({
    "estimated_gpu_cost_usd", "estimated_prefill_cost_usd",
    "estimated_decode_cost_usd", "estimated_energy_cost_usd",
    "estimated_carbon_kg", "estimated_carbon_cost_usd",
    "estimated_cache_value_usd", "estimated_migration_cost_usd",
    "estimated_cold_start_cost_usd", "estimated_memory_pressure_cost_usd",
    "estimated_gpu_seconds", "estimated_prefill_seconds",
    "estimated_decode_seconds",
    "sla_safe_goodput", "sla_safe_goodput_per_dollar", "sla_met",
    # provenance dicts are never features
    "value_quality_by_field", "formula_by_field", "limitations",
    "overlay_class", "source_trace_id",
})

# Per-target extra blocks: predicting X must not see X's direct actuals.
TARGET_EXTRA_BLOCK = {
    "ttft_s": {"ttft_s", "e2e_latency_s", "tpot_s", "throughput_tok_s",
               "output_tokens"},
    "tpot_s": {"tpot_s", "e2e_latency_s", "ttft_s", "throughput_tok_s",
               "output_tokens"},
    "e2e_latency_s": {"e2e_latency_s", "ttft_s", "tpot_s",
                      "throughput_tok_s", "output_tokens"},
    "queue_wait_s": {"queue_wait_s", "e2e_latency_s"},
    "cache_reuse_pct": {"cache_reuse_pct"},
    "high_reuse": {"cache_reuse_pct", "high_reuse"},
    "energy_kwh": {"energy_kwh", "gpu_power_w"},
    "gpu_power_w": {"gpu_power_w", "energy_kwh"},
    "peak_vram_gb": {"peak_vram_gb"},
}

# Candidate feature columns (decision-time-plausible numerics).
NUMERIC_FEATURES = [
    "prompt_tokens", "output_tokens", "gpu_count", "kv_utilization",
    "gpu_price_usd_per_hour",
]
CATEGORICAL_FEATURES = ["gpu_type", "model_id", "source_dataset_id"]


@dataclass
class FeatureSpec:
    """Resolved feature plan for one target."""
    target: str
    numeric: list[str]
    categorical: list[str]
    blocked: frozenset
    allow_output_tokens: bool = False


def resolve_feature_spec(target: str, *,
                         allow_output_tokens: bool = False,
                         extra_numeric: Optional[list[str]] = None,
                         ) -> FeatureSpec:
    blocked = set(ALWAYS_BLOCKED) | set(TARGET_EXTRA_BLOCK.get(target, set()))
    blocked.add(target)
    if allow_output_tokens:
        blocked.discard("output_tokens")
    numeric = [c for c in (NUMERIC_FEATURES + (extra_numeric or []))
               if c not in blocked]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in blocked]
    return FeatureSpec(target=target, numeric=numeric, categorical=categorical,
                       blocked=frozenset(blocked),
                       allow_output_tokens=allow_output_tokens)


# ---------------------------------------------------------------------------
# Corpus loading.
# ---------------------------------------------------------------------------


def load_corpus(corpus_dir: Path = CORPUS_DIR, *,
                datasets: Optional[Iterable[str]] = None,
                max_rows_per_file: Optional[int] = None) -> list[dict]:
    """Load overlay records (dicts) from the gitignored analysis corpus.

    ``datasets`` filters by the corpus filename stem (e.g. "cara_train_flat").
    """
    rows: list[dict] = []
    files = sorted(corpus_dir.glob("*.jsonl"))
    keep = set(datasets) if datasets else None
    for f in files:
        if keep is not None and f.stem not in keep:
            continue
        with open(f) as fh:
            for i, line in enumerate(fh):
                if max_rows_per_file is not None and i >= max_rows_per_file:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Matrix builder.
# ---------------------------------------------------------------------------


def _encode_categoricals(rows: list[dict], cats: list[str]
                         ) -> tuple[np.ndarray, dict]:
    """Deterministic ordinal encoding from observed values only (no invented
    categories). Unknown-at-transform values map to -1."""
    vocab: dict[str, dict] = {}
    for c in cats:
        seen = {}
        for r in rows:
            v = r.get(c)
            key = "" if v is None else str(v)
            if key not in seen:
                seen[key] = len(seen)
        vocab[c] = seen
    cols = []
    for c in cats:
        seen = vocab[c]
        col = np.array([seen.get("" if r.get(c) is None else str(r.get(c)), -1)
                        for r in rows], dtype=float)
        cols.append(col.reshape(-1, 1))
    mat = (np.hstack(cols) if cols
           else np.empty((len(rows), 0), dtype=float))
    return mat, vocab


def build_matrix(rows: list[dict], spec: FeatureSpec
                 ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Returns (X, y, feature_names, encode_vocab).

    Rows where the target is None are dropped. A hard ``LeakageError`` fires
    if any requested feature is in the blocklist (defensive: the spec already
    filters, this catches manual misuse)."""
    for f in spec.numeric + spec.categorical:
        if f in spec.blocked:
            raise LeakageError(
                f"feature {f!r} is blocked for target {spec.target!r}")

    kept = [r for r in rows if r.get(spec.target) is not None]
    if not kept:
        return (np.empty((0, 0)), np.empty((0,)), [], {})

    y = np.array([float(r[spec.target]) for r in kept], dtype=float)

    num_cols = []
    for c in spec.numeric:
        col = np.array([np.nan if r.get(c) is None else float(r.get(c))
                        for r in kept], dtype=float)
        num_cols.append(col.reshape(-1, 1))
    Xnum = (np.hstack(num_cols) if num_cols
            else np.empty((len(kept), 0), dtype=float))
    Xcat, vocab = _encode_categoricals(kept, spec.categorical)
    X = np.hstack([Xnum, Xcat]) if (Xnum.size or Xcat.size) else \
        np.empty((len(kept), 0))
    names = list(spec.numeric) + list(spec.categorical)
    return X, y, names, vocab


def value_quality_distribution(rows: list[dict], field_name: str) -> dict:
    """Per-field value_quality histogram across rows (preserves provenance)."""
    out: dict[str, int] = {}
    for r in rows:
        q = (r.get("value_quality_by_field", {}) or {}).get(field_name,
                                                            "missing")
        out[q] = out.get(q, 0) + 1
    return out
