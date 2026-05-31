"""Promotion gates for the federated corpus.

A dataset moves from ``candidate`` to one of the promoted states only after
the gates documented in the mission spec all pass. The gates are explicit
and refuse to promote on the first failure.

Promotion target states:

- ``promoted_for_backtest`` — eligible for the existing public-trace
  backtest harness when a compatible trace_type / signals exist.
- ``promoted_for_training_priors`` — eligible as a prior source for the
  training utilisation frontier or the eval / batch frontier.
- ``promoted_for_constraint_aware_evaluation`` — eligible for the
  constraint-aware scheduler smoke evaluator (placement, batching,
  scaling, deferral routing).
- ``promoted_for_dynamic_calibration`` — eligible for the dynamic safe
  utilisation frontier calibration harness. Requires ``telemetry_trace``.
- ``promoted_for_performance_priors`` — eligible as latency/throughput
  prior source (latency_benchmark_trace and kernel_profile_trace).
- ``promoted_for_cache_residency_evaluation`` — eligible for the
  cache/residency/routing evaluator.

Refusal states:

- ``candidate`` — pre-promotion default.
- ``rejected`` — failed at least one gate.
- ``gated_blocked`` — dataset is HF-gated and no usable token / access.

NOTE: promotion to training-priors does NOT mean production truth. The
trust hierarchy in ``docs/HF_DATASET_REGISTRY.md`` is binding — pilot
telemetry remains the only Tier 1 calibration source.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .schemas import (
    CANONICAL_TRACE_TYPES,
    CANONICAL_TRACE_TYPE_TO_TRUST_TIER,
)

logger = logging.getLogger(__name__)


PROMOTION_STATES = frozenset({
    "candidate",
    "validated_bounded",
    "promoted_for_backtest",
    "promoted_for_training_priors",
    "promoted_for_constraint_aware_evaluation",
    "promoted_for_dynamic_calibration",
    "promoted_for_performance_priors",
    "promoted_for_cache_residency_evaluation",
    "promoted_for_schema_only",
    "rejected",
    "gated_blocked",
    "auth_blocked",
    "deferred_bounded_ingest",
})


# Promotion-tag -> minimum statistical_sample_strength required. Fixture-only
# samples may pass schema/promotion-for-schema-only gates but MUST NOT claim
# performance / dynamic / backtest / cache-residency evidence value.
PROMOTION_TAG_MIN_SAMPLE_STRENGTH = {
    "promoted_for_schema_only": "fixture_only",
    "promoted_for_training_priors": "weak",
    "promoted_for_performance_priors": "moderate",
    "promoted_for_cache_residency_evaluation": "moderate",
    "promoted_for_constraint_aware_evaluation": "moderate",
    "promoted_for_dynamic_calibration": "strong",
    "promoted_for_backtest": "moderate",
}

_SAMPLE_STRENGTH_ORDER = {
    "fixture_only": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


def _sample_strength_satisfies(actual: str, required: str) -> bool:
    a = _SAMPLE_STRENGTH_ORDER.get(actual, -1)
    r = _SAMPLE_STRENGTH_ORDER.get(required, 99)
    return a >= r


# Which promotion states are valid for each canonical trace type. A
# dataset may carry MULTIPLE promotion tags (e.g. AgentPerfBench is both
# performance priors AND constraint-aware eval) — the registry stores a
# tuple of states, not a single state.
TRACE_TYPE_TO_ALLOWED_PROMOTIONS = {
    "telemetry_trace": [
        "promoted_for_dynamic_calibration",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_backtest",
    ],
    "cluster_scheduler_trace": [
        "promoted_for_backtest",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    ],
    "latency_benchmark_trace": [
        "promoted_for_performance_priors",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    ],
    "kernel_profile_trace": [
        "promoted_for_performance_priors",
        "promoted_for_training_priors",
    ],
    "cache_residency_trace": [
        "promoted_for_cache_residency_evaluation",
        "promoted_for_training_priors",
    ],
    "request_shape_trace": [
        "promoted_for_training_priors",
    ],
    "mixed_or_unknown_trace": [],
}


# Bounded-size guard: a committed sample is never allowed to exceed this.
MAX_COMMITTED_SAMPLE_BYTES = 16 * 1024 * 1024  # 16 MiB


class PromotionGateError(ValueError):
    """Raised when a promotion attempt fails a gate."""


def gates(summary: dict) -> list[dict]:
    """Run all promotion gates against the dataset summary.

    Returns the gate-evaluation log: a list of
    ``{"gate": ..., "passed": bool, "detail": ...}`` dicts. A dataset is
    promotable iff every gate has ``passed=True``.
    """

    out: list[dict] = []

    # Gate 1: schema test (raw_schema + normalized_schema non-empty +
    # no unknown columns).
    raw_schema = summary.get("raw_schema") or []
    norm_schema = summary.get("normalized_schema") or []
    unknown = summary.get("unknown_columns") or []
    schema_ok = bool(raw_schema) and bool(norm_schema) and not unknown
    out.append({
        "gate": "schema_test",
        "passed": schema_ok,
        "detail": {
            "raw_schema_size": len(raw_schema),
            "normalized_schema_size": len(norm_schema),
            "unknown_columns": list(unknown),
        },
    })

    # Gate 2: fixture test (sample_rows > 0 AND a sample sha256 exists).
    rows = int(summary.get("committed_sample_rows") or 0)
    sha = summary.get("sample_sha256")
    fixture_ok = rows > 0 and bool(sha)
    out.append({
        "gate": "fixture_test",
        "passed": fixture_ok,
        "detail": {"committed_sample_rows": rows, "sha256_present": bool(sha)},
    })

    # Gate 3: bounded-size guard.
    sample_bytes = int(summary.get("committed_sample_bytes") or 0)
    size_ok = 0 < sample_bytes <= MAX_COMMITTED_SAMPLE_BYTES
    out.append({
        "gate": "bounded_size_guard",
        "passed": size_ok,
        "detail": {
            "committed_sample_bytes": sample_bytes,
            "max_committed_sample_bytes": MAX_COMMITTED_SAMPLE_BYTES,
        },
    })

    # Gate 4: license / gating status recorded.
    license_recorded = "license" in summary
    gated_recorded = "gated" in summary
    lic_ok = license_recorded and gated_recorded
    out.append({
        "gate": "license_and_gating_recorded",
        "passed": lic_ok,
        "detail": {
            "license": summary.get("license"),
            "gated": summary.get("gated"),
        },
    })

    # Gate 5: canonical trace_type assigned (not mixed_or_unknown).
    tt = summary.get("canonical_trace_type")
    tt_ok = tt in CANONICAL_TRACE_TYPES and tt != "mixed_or_unknown_trace"
    out.append({
        "gate": "canonical_trace_type_assigned",
        "passed": tt_ok,
        "detail": {"canonical_trace_type": tt},
    })

    # Gate 6: available + missing signals explicit (both lists present).
    avail = summary.get("available_signals")
    miss = summary.get("missing_signals")
    sig_ok = isinstance(avail, list) and isinstance(miss, list) and bool(avail)
    out.append({
        "gate": "signals_explicit",
        "passed": sig_ok,
        "detail": {
            "available_signal_count": len(avail or []),
            "missing_signal_count": len(miss or []),
        },
    })

    # Gate 7: limitations recorded.
    lims = summary.get("limitations")
    lim_ok = isinstance(lims, list) and bool(lims)
    out.append({
        "gate": "limitations_recorded",
        "passed": lim_ok,
        "detail": {"limitations_count": len(lims or [])},
    })

    # Gate 8: at least one valid Aurelius use case (i.e. trace_type maps
    # to at least one allowed promotion).
    valid_promos = TRACE_TYPE_TO_ALLOWED_PROMOTIONS.get(tt or "", [])
    use_ok = bool(valid_promos)
    out.append({
        "gate": "at_least_one_aurelius_use_case",
        "passed": use_ok,
        "detail": {"allowed_promotions": valid_promos},
    })

    # Gate 9: analysis-sample policy recorded. Either an explicit
    # statistical_sample_strength field is present, or the summary records
    # both ``fixture_sample_rows`` and ``analysis_sample_rows`` (which
    # ``scripts/ingest_cara_swissai.py`` always emits).
    strength = summary.get("statistical_sample_strength")
    has_split = (
        "fixture_sample_rows" in summary and "analysis_sample_rows" in summary
    )
    sample_policy_ok = (
        strength in {"fixture_only", "weak", "moderate", "strong"} or has_split
    )
    out.append({
        "gate": "analysis_sample_policy_recorded",
        "passed": sample_policy_ok,
        "detail": {
            "statistical_sample_strength": strength,
            "fixture_sample_rows": summary.get("fixture_sample_rows"),
            "analysis_sample_rows": summary.get("analysis_sample_rows"),
        },
    })

    return out


def _filter_promotions_by_sample_strength(
    allowed: list, strength: Optional[str],
) -> list:
    if not strength:
        # Fail-closed: no strength label means we can only promote_for_schema_only.
        return [t for t in allowed if t == "promoted_for_schema_only"]
    return [
        t for t in allowed
        if _sample_strength_satisfies(
            strength,
            PROMOTION_TAG_MIN_SAMPLE_STRENGTH.get(t, "moderate"),
        )
    ]


def evaluate_promotion(summary: dict) -> dict:
    """Apply gates + return ``{state, promotion_tags, gate_log, reasons}``.

    ``state`` is one of ``PROMOTION_STATES``:
    - ``gated_blocked`` if ``summary["gated"] is True``.
    - ``rejected`` if any non-gating gate fails.
    - ``validated_bounded`` if all gates pass but trace_type allows no
      promotions (e.g. ``mixed_or_unknown_trace`` made it through somehow).
    - The first allowed promotion tag from the trace_type's allowed
      list (others are added to ``promotion_tags``).
    """

    if summary.get("gated") is True:
        return {
            "state": "gated_blocked",
            "promotion_tags": [],
            "gate_log": [],
            "reasons": ["dataset is HF-gated; supply HF_TOKEN with access"],
            "evaluated_at_s": time.time(),
        }

    # Auth failure short-circuit: caller explicitly recorded auth_blocked.
    if summary.get("auth_status") == "auth_blocked":
        return {
            "state": "auth_blocked",
            "promotion_tags": [],
            "gate_log": [],
            "reasons": ["HF auth failed despite token supplied; access denied"],
            "evaluated_at_s": time.time(),
        }

    gate_log = gates(summary)
    failed = [g for g in gate_log if not g["passed"]]
    if failed:
        return {
            "state": "rejected",
            "promotion_tags": [],
            "gate_log": gate_log,
            "reasons": [f"gate '{g['gate']}' failed" for g in failed],
            "evaluated_at_s": time.time(),
        }

    tt = summary.get("canonical_trace_type")
    allowed = TRACE_TYPE_TO_ALLOWED_PROMOTIONS.get(tt or "", [])
    if not allowed:
        return {
            "state": "validated_bounded",
            "promotion_tags": [],
            "gate_log": gate_log,
            "reasons": ["passed gates but trace_type has no allowed promotion"],
            "evaluated_at_s": time.time(),
        }

    strength = summary.get("statistical_sample_strength")
    qualifying = _filter_promotions_by_sample_strength(allowed, strength)

    if not qualifying:
        return {
            "state": "promoted_for_schema_only",
            "promotion_tags": ["promoted_for_schema_only"],
            "gate_log": gate_log,
            "reasons": [
                f"statistical_sample_strength='{strength}' is insufficient for "
                f"any of {allowed}; promoted_for_schema_only only"
            ],
            "evaluated_at_s": time.time(),
        }

    reasons = []
    if len(qualifying) < len(allowed):
        dropped = [t for t in allowed if t not in qualifying]
        reasons.append(
            f"statistical_sample_strength='{strength}' insufficient for "
            f"{dropped}; downgraded to {qualifying}"
        )

    return {
        "state": qualifying[0],
        "promotion_tags": list(qualifying),
        "gate_log": gate_log,
        "reasons": reasons,
        "evaluated_at_s": time.time(),
    }


# ---------------------------------------------------------------------------
# Registry writer
# ---------------------------------------------------------------------------


def build_registry_entry(summary: dict, decision: dict) -> dict:
    """Compose a single registry row from a summary + promotion decision."""
    return {
        "dataset_id": summary["dataset_id"],
        "config_name": summary.get("config_name"),
        "source_url": summary.get("source_url"),
        "canonical_trace_type": summary.get("canonical_trace_type"),
        "statistical_sample_strength": summary.get("statistical_sample_strength"),
        "fixture_sample_rows": summary.get("fixture_sample_rows"),
        "analysis_sample_rows": summary.get("analysis_sample_rows"),
        "fixture_sample_bytes": summary.get("fixture_sample_bytes"),
        "analysis_sample_bytes": summary.get("analysis_sample_bytes"),
        "sampling_method": summary.get("sampling_method"),
        "stratification_keys": summary.get("stratification_keys"),
        "schema_profile_path": summary.get("schema_profile_path"),
        "schema_mapping_path": summary.get("schema_mapping_path"),
        "trust_tier": CANONICAL_TRACE_TYPE_TO_TRUST_TIER.get(
            summary.get("canonical_trace_type") or "",
            "tier_6_synthetic_benchmark_data"),
        "license": summary.get("license"),
        "gated": summary.get("gated"),
        "promotion_state": decision["state"],
        "promotion_tags": decision.get("promotion_tags") or [],
        "promotion_reasons": decision.get("reasons") or [],
        "available_signals": list(summary.get("available_signals") or []),
        "missing_signals": list(summary.get("missing_signals") or []),
        "derived_fields": list(summary.get("derived_fields") or []),
        "proxy_fields": list(summary.get("proxy_fields") or []),
        "synthetic_fields": list(summary.get("synthetic_fields") or []),
        "limitations": list(summary.get("limitations") or []),
        "ingestion_timestamp_s": summary.get("ingestion_timestamp_s"),
        "promotion_evaluated_at_s": decision.get("evaluated_at_s"),
        "committed_sample_rows": summary.get("committed_sample_rows"),
        "committed_sample_bytes": summary.get("committed_sample_bytes"),
        "sample_sha256": summary.get("sample_sha256"),
        "provenance": summary.get("provenance"),
        "summary_path_relative": summary.get("summary_path_relative"),
    }


def write_canonical_registry(entries: list[dict], path: str) -> dict:
    """Write the canonical corpus registry. Returns the wrapped payload."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "doc_version": "hf_corpus_canonical_registry_v1",
        "stage": "federated_benchmark_corpus_v1",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "trust_hierarchy_note": (
            "Tier 1 (real pilot telemetry) remains the only production "
            "calibration source. Promotion here is research-class only."
        ),
        "written_at_s": time.time(),
        "entry_count": len(entries),
        "entries": entries,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def load_canonical_registry(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
