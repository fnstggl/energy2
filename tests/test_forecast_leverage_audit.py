"""Schema + invariant tests for the Forecast Leverage Audit.

The audit at `docs/FORECAST_LEVERAGE_AUDIT.md` is binding research-class
guidance: it ranks which forecasting engines Aurelius should build next.
These tests enforce the structural invariants the doc promises:

- Every decision in the inventory maps to a forecast + recommended priority.
- Every `build_now` forecast satisfies the 4-clause gate documented in
  §1: (a) controls a real decision, (b) has at least one dataset, (c)
  names a strongest-realistic baseline, (d) declares a success metric.
- Every `build_after_data_expansion` forecast names the blocking gate.
- `not_enough_data` forecasts cannot be promoted.
- `already_sufficient` forecasts must not duplicate `build_now` work.
- No production-claim, no oracle-as-headline, no ML training.
- Cross-document consistency: the markdown registry tables and the JSON
  registry agree on the engine IDs.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "forecast_leverage_audit.json",
)
AUDIT_MD = os.path.join(REPO_ROOT, "docs", "FORECAST_LEVERAGE_AUDIT.md")


# ---------- 1. Required artefacts exist ----------------------------------


def test_audit_artefacts_exist():
    assert os.path.exists(AUDIT_JSON), (
        "missing data/external/frontier/forecast_leverage_audit.json"
    )
    assert os.path.exists(AUDIT_MD), "missing docs/FORECAST_LEVERAGE_AUDIT.md"


@pytest.fixture(scope="module")
def audit():
    with open(AUDIT_JSON) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def md_text():
    with open(AUDIT_MD) as fh:
        return fh.read()


# ---------- 2. Top-level invariants ---------------------------------------


def test_doc_version_and_stage(audit):
    assert audit["doc_version"] == "forecast_leverage_audit_v1"
    assert audit["stage"] == "audit_discovery_only"


def test_production_claim_and_training_flags_false(audit):
    for flag in (
        "production_claim",
        "modifies_robust_energy_engine",
        "modifies_controllers_or_defaults",
        "trains_ml_models",
        "uses_oracle_as_headline",
    ):
        assert audit[flag] is False, f"audit asserts {flag}=true; non-goal violation"


# ---------- 3. Decision inventory completeness ---------------------------


REQUIRED_DECISIONS = {
    "placement_region",
    "placement_heterogeneous_gpu",
    "routing_request_level",
    "queue_admission",
    "cache_prefix_routing",
    "model_residency",
    "autoscaling_replica",
    "batching",
    "deferral_training_backfill",
    "migration_veto",
    "sla_risk_gating",
    "timeout_risk_gating",
    "gpu_packing_training",
    "energy_shifting",
    "carbon_shifting",
    "thermal_resource_pressure",
}


def test_every_canonical_decision_present(audit):
    present = {d["decision_id"] for d in audit["decision_inventory"]}
    missing = REQUIRED_DECISIONS - present
    assert not missing, f"decision_inventory missing canonical decisions {missing}"


REQUIRED_DECISION_FIELDS = {
    "decision_id", "name", "current_module", "current_signals_used",
    "forecast_needed", "forecast_horizon", "target_variable",
    "available_datasets", "missing_data", "decision_frequency",
    "economic_impact", "safety_impact", "expected_alpha_label",
    "implementation_complexity", "confidence",
    "value_x_frequency_score", "recommended_build_priority",
    "primary_baseline",
}


def test_every_decision_carries_required_fields(audit):
    for d in audit["decision_inventory"]:
        missing = REQUIRED_DECISION_FIELDS - set(d.keys())
        assert not missing, (
            f"decision {d.get('decision_id')} missing fields {missing}"
        )


def test_value_x_frequency_score_in_valid_range(audit):
    for d in audit["decision_inventory"]:
        s = d["value_x_frequency_score"]
        assert isinstance(s, (int, float)) and 0.0 <= s <= 10.0, (
            f"{d['decision_id']} value_x_frequency_score={s} out of [0,10]"
        )


# ---------- 4. Recommended build priority — closed enum -----------------


VALID_PRIORITIES = {
    "build_now",
    "build_after_data_expansion",
    "diagnostic_only",
    "not_enough_data",
    "already_sufficient",
}


def test_every_priority_is_in_closed_enum(audit):
    for d in audit["decision_inventory"]:
        p = d["recommended_build_priority"]
        assert p in VALID_PRIORITIES, (
            f"{d['decision_id']} priority='{p}' not in {sorted(VALID_PRIORITIES)}"
        )
    for e in audit["forecast_engine_rankings"]:
        b = e["build_status"]
        assert b in VALID_PRIORITIES, (
            f"engine {e['engine_id']} build_status='{b}' invalid"
        )


# ---------- 5. Engine ranking integrity ----------------------------------


REQUIRED_ENGINE_FIELDS = {
    "rank", "engine_id", "name", "controls_decisions", "target_variable",
    "horizon", "datasets_supporting", "primary_baseline", "success_metric",
    "expected_alpha_label", "expected_alpha_rationale", "build_status",
    "blocking_gates",
}


def test_engine_ranking_has_required_fields(audit):
    for e in audit["forecast_engine_rankings"]:
        missing = REQUIRED_ENGINE_FIELDS - set(e.keys())
        assert not missing, f"engine {e.get('engine_id')} missing {missing}"


def test_engine_ranks_unique_and_sequential(audit):
    ranks = [e["rank"] for e in audit["forecast_engine_rankings"]]
    assert ranks == sorted(ranks), "engine ranks must be ascending"
    assert len(set(ranks)) == len(ranks), "engine ranks must be unique"
    assert ranks[0] == 1, "first rank must be 1"


def test_engine_ids_unique(audit):
    ids = [e["engine_id"] for e in audit["forecast_engine_rankings"]]
    assert len(set(ids)) == len(ids), "engine_ids must be unique"


# ---------- 6. build_now four-clause gate (binding) ----------------------


def test_build_now_forecasts_meet_gate(audit):
    """A `build_now` engine must:
      (a) name at least one decision it controls,
      (b) cite at least one supporting in-repo or HF dataset,
      (c) name a strongest-realistic primary_baseline,
      (d) declare a success_metric.
    """
    decision_ids = {d["decision_id"] for d in audit["decision_inventory"]}
    for e in audit["forecast_engine_rankings"]:
        if e["build_status"] != "build_now":
            continue
        # (a)
        controls = e["controls_decisions"]
        assert controls, f"{e['engine_id']} build_now but controls_decisions empty"
        for c in controls:
            assert c in decision_ids, (
                f"{e['engine_id']} controls unknown decision '{c}'"
            )
        # (b)
        ds = e["datasets_supporting"]
        assert ds, f"{e['engine_id']} build_now but no supporting dataset"
        # (c)
        assert e["primary_baseline"] and e["primary_baseline"] != "oracle", (
            f"{e['engine_id']} primary_baseline must be a realistic baseline, "
            f"not oracle"
        )
        # (d)
        sm = e.get("success_metric") or ""
        assert sm.strip(), f"{e['engine_id']} build_now but no success_metric"


# ---------- 7. build_after_data_expansion must name blocking gates ------


def test_deferred_engines_name_blocking_gates(audit):
    for e in audit["forecast_engine_rankings"]:
        if e["build_status"] != "build_after_data_expansion":
            continue
        gates = e["blocking_gates"]
        assert gates, (
            f"{e['engine_id']} deferred but no blocking_gates listed"
        )


# ---------- 8. not_enough_data engines cannot be promoted ----------------


def test_not_enough_data_engines_have_no_alpha_label(audit):
    for e in audit["forecast_engine_rankings"]:
        if e["build_status"] != "not_enough_data":
            continue
        assert e["expected_alpha_label"] in ("unknown", "low"), (
            f"{e['engine_id']} not_enough_data but claims alpha "
            f"'{e['expected_alpha_label']}'"
        )
        # No datasets supporting + no decisions blocked elsewhere should
        # mean an explicit blocking gate.
        assert e["blocking_gates"], (
            f"{e['engine_id']} not_enough_data but no blocking_gates"
        )


# ---------- 9. already_sufficient must not appear in build_now list -----


def test_already_sufficient_disjoint_from_build_now(audit):
    bn = set(audit["build_now_list"])
    asuf = set(audit["already_sufficient_list"])
    overlap = bn & asuf
    assert not overlap, (
        f"engines marked both build_now and already_sufficient: {overlap}"
    )


# ---------- 10. List/ranking consistency --------------------------------


def test_engine_lists_match_rankings(audit):
    ranking_status = {
        e["engine_id"]: e["build_status"]
        for e in audit["forecast_engine_rankings"]
    }
    for status_key, list_key in (
        ("build_now", "build_now_list"),
        ("build_after_data_expansion", "build_after_data_expansion_list"),
        ("diagnostic_only", "diagnostic_only_list"),
        ("not_enough_data", "not_enough_data_list"),
        ("already_sufficient", "already_sufficient_list"),
    ):
        listed = set(audit[list_key])
        from_ranking = {eid for eid, s in ranking_status.items() if s == status_key}
        assert listed == from_ranking, (
            f"{list_key}={sorted(listed)} != ranking-derived "
            f"{sorted(from_ranking)} for status='{status_key}'"
        )


# ---------- 11. top_10_priority_list is the first 10 by rank -----------


def test_top_10_priority_list_matches_ranking_head(audit):
    by_rank = sorted(
        audit["forecast_engine_rankings"], key=lambda e: e["rank"]
    )[:10]
    expected = [{"rank": e["rank"], "engine_id": e["engine_id"]}
                for e in by_rank]
    assert audit["top_10_priority_list"] == expected, (
        "top_10_priority_list must be the rank-1..10 head of the ranking"
    )


# ---------- 12. Every controls_decisions entry is a real decision_id ---


def test_engine_controls_decisions_are_real(audit):
    decision_ids = {d["decision_id"] for d in audit["decision_inventory"]}
    for e in audit["forecast_engine_rankings"]:
        for c in e["controls_decisions"]:
            assert c in decision_ids, (
                f"engine {e['engine_id']} controls unknown decision '{c}'"
            )


# ---------- 13. Reading dependencies are real files --------------------


def test_reading_dependencies_exist(audit):
    for path in audit["reading_dependencies"]:
        full = os.path.join(REPO_ROOT, path)
        assert os.path.exists(full), (
            f"reading_dependencies references missing file '{path}'"
        )


# ---------- 14. CARA + SwissAI are cited where the spec requires -------


def test_cara_swissai_cited_for_telemetry_forecasts(audit):
    """The mission spec requires CARA + SwissAI to be specifically used
    in the audit. Check the cite shows up for the relevant engines."""
    by_id = {e["engine_id"]: e for e in audit["forecast_engine_rankings"]}

    def _any_dataset_mentions(engine_id, needle):
        ds = by_id[engine_id]["datasets_supporting"]
        return any(needle.lower() in d.lower() for d in ds)

    assert _any_dataset_mentions("ttft_forecast", "CARA")
    assert _any_dataset_mentions("queue_wait_forecast", "CARA")
    assert _any_dataset_mentions("e2e_latency_forecast", "CARA")
    assert _any_dataset_mentions("cache_prefix_reuse_forecast", "SwissAI")


# ---------- 15. Markdown ↔ JSON consistency (engine IDs) ---------------


def test_markdown_mentions_every_build_now_engine(audit, md_text):
    md_lower = md_text.lower()
    for eid in audit["build_now_list"]:
        engine = next(
            e for e in audit["forecast_engine_rankings"]
            if e["engine_id"] == eid
        )
        # Match on the first 2 words of the engine name (e.g. "TTFT forecast")
        # so qualifying parentheticals in the JSON name don't bind the doc.
        words = re.split(r"\s+", engine["name"].strip())
        head = " ".join(words[:2]).lower().rstrip(":,—-")
        assert head in md_lower, (
            f"docs/FORECAST_LEVERAGE_AUDIT.md does not mention build_now "
            f"engine '{engine['name']}' (head='{head}')"
        )


# ---------- 16. No production-claim phrases in the markdown ------------


BANNED_PHRASES = (
    "production savings",
    "guaranteed savings",
    "production-proven",
    "hyperscaler-validated economics",
    "enterprise-ready autonomous optimization",
)


def test_no_banned_production_claim_phrases(md_text):
    low = md_text.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in low, (
            f"docs/FORECAST_LEVERAGE_AUDIT.md contains banned phrase '{phrase}'"
        )


# ---------- 17. value_x_frequency_score monotone-aligns with rank ------


def test_value_score_aligns_with_ranking_at_top(audit):
    """The mission spec ties ranking to ``value × decision frequency``.
    Check the top-5 engines all map back to a decision whose
    value_x_frequency_score is in the top 50% of the inventory."""
    decisions_by_id = {
        d["decision_id"]: d for d in audit["decision_inventory"]
    }
    scores = sorted(
        (d["value_x_frequency_score"] for d in audit["decision_inventory"]),
        reverse=True,
    )
    median = scores[len(scores) // 2]
    for e in audit["forecast_engine_rankings"][:5]:
        controlled_scores = [
            decisions_by_id[c]["value_x_frequency_score"]
            for c in e["controls_decisions"]
            if c in decisions_by_id
        ]
        assert controlled_scores, (
            f"top-5 engine {e['engine_id']} controls no inventoried decision"
        )
        assert max(controlled_scores) >= median, (
            f"top-5 engine {e['engine_id']} controls only below-median-value "
            f"decisions; ranking inconsistent"
        )


# ---------- 18. Schema-only safety — no executor code referenced ------


def test_audit_does_not_reference_any_executor_module(audit, md_text):
    """The audit is research-class. It must not reference any executor
    or controller-execution path (``execute_*``, ``apply_*``,
    ``set_replicas``, etc.) — that would imply an unintended controller
    modification."""
    banned_referents = ("execute_frontier_decision", "apply_replica_scale")
    payload = json.dumps(audit).lower() + md_text.lower()
    for b in banned_referents:
        assert b not in payload, (
            f"audit references executor/controller path '{b}'; the audit "
            "must remain recommendation-only"
        )


# ---------- 19. Every engine ranks names a non-empty horizon -----------


def test_engine_horizons_non_empty(audit):
    for e in audit["forecast_engine_rankings"]:
        h = e["horizon"]
        assert isinstance(h, str) and h.strip(), (
            f"{e['engine_id']} horizon empty"
        )
