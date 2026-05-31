"""Tests for the Frontier Discovery Research Audit artifacts.

This is a discovery-stage audit (`docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md`),
NOT a benchmark and NOT a new controller. The tests pin the *shape and honesty*
of the committed audit artifacts so future edits cannot silently drop a
workload class, silently flip a recommendation, or quietly introduce a
production-savings claim or oracle headline.

Invariants:

1.  The audit JSON is valid and lives at the documented path.
2.  All eight workload classes from the audit charter are present with the
    expected names, in order.
3.  Each class carries the required descriptive fields.
4.  Feasibility / expected-alpha / complexity scores are integers in [1, 5].
5.  Each class carries a recommendation drawn from the closed enum.
6.  The ranked recommendation buckets cover every class exactly once.
7.  The JSON declares it is NOT a production claim, NOT an ML training phase,
    does NOT mutate the robust energy engine, does NOT default any new
    controller, does NOT download new datasets, and does NOT use an oracle
    as headline (the audit charter "DO NOT" list).
8.  The audit markdown exists and references the required prior-art docs
    (the "Read first" list from the audit charter).
9.  The audit markdown contains no unhedged production-savings claims, the
    same scan `tests/test_per_workload_reporting.py` runs against benchmark
    reports (`docs/RESULTS.md` §8).
10. The build-now bucket is small (≤ 2) — discovery audits should not
    recommend building everything at once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "FRONTIER_DISCOVERY_RESEARCH_AUDIT.md"
JSON_PATH = (
    REPO_ROOT
    / "data"
    / "external"
    / "frontier"
    / "frontier_discovery_research_summary.json"
)

EXPECTED_CLASS_NAMES = (
    "batch_inference",
    "embedding_generation",
    "data_processing_etl_feature_engineering",
    "vector_indexing_rag_indexing",
    "synthetic_data_generation",
    "evaluation_workloads_eval_harnesses",
    "rlhf_pipelines",
    "agent_swarms",
)

REQUIRED_PER_CLASS_FIELDS = (
    "id",
    "name",
    "best_public_trace",
    "sample_size",
    "signals_available",
    "missing_signals",
    "candidate_frontier_variable",
    "safety_constraints",
    "economic_lever",
    "feasibility_score",
    "expected_alpha_score",
    "implementation_complexity_score",
    "recommendation",
)

VALID_RECOMMENDATIONS = {
    "build_now",
    "investigate_later",
    "not_enough_data",
    "low_expected_alpha",
}

REQUIRED_PRIOR_ART_DOCS = (
    "docs/RESULTS.md",
    "docs/PUBLIC_TRACE_BACKTESTS.md",
    "docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md",
    "docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md",
    "docs/TRAINING_SAFE_UTILIZATION_FRONTIER.md",
    "docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md",
    "docs/AZURE_LLM_2024_BACKTEST_RESULTS.md",
    "docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md",
)

# Same banned substrings as `tests/test_per_workload_reporting.py` enforces
# against benchmark report markdown (`docs/RESULTS.md` §8).
BANNED_PHRASES = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)


@pytest.fixture(scope="module")
def audit_json():
    assert JSON_PATH.exists(), f"missing audit JSON at {JSON_PATH}"
    with JSON_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def audit_markdown():
    assert DOC_PATH.exists(), f"missing audit doc at {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def test_audit_json_loads(audit_json):
    assert audit_json["doc_version"] == "frontier_discovery_research_audit_v1"
    assert audit_json["stage"] == "discovery_signal_scan"


def test_eight_workload_classes_in_expected_order(audit_json):
    names = tuple(c["name"] for c in audit_json["workload_classes"])
    assert names == EXPECTED_CLASS_NAMES, (
        f"expected {EXPECTED_CLASS_NAMES} got {names}"
    )


def test_each_class_has_required_fields(audit_json):
    for cls in audit_json["workload_classes"]:
        missing = [f for f in REQUIRED_PER_CLASS_FIELDS if f not in cls]
        assert not missing, f"{cls.get('name')} missing fields: {missing}"


def test_scores_are_integers_one_to_five(audit_json):
    score_fields = (
        "feasibility_score",
        "expected_alpha_score",
        "implementation_complexity_score",
    )
    for cls in audit_json["workload_classes"]:
        for f in score_fields:
            v = cls[f]
            assert isinstance(v, int), f"{cls['name']}.{f} not int: {v!r}"
            assert 1 <= v <= 5, f"{cls['name']}.{f} out of [1,5]: {v}"


def test_recommendation_is_from_enum(audit_json):
    for cls in audit_json["workload_classes"]:
        assert cls["recommendation"] in VALID_RECOMMENDATIONS, (
            f"{cls['name']} bad recommendation: {cls['recommendation']!r}"
        )


def test_ranked_recommendation_covers_every_class_exactly_once(audit_json):
    ranked = audit_json["ranked_recommendation"]
    seen_ids = []
    for bucket in ("build_now", "investigate_later", "not_enough_data", "low_expected_alpha"):
        assert bucket in ranked, f"missing bucket {bucket}"
        for entry in ranked[bucket]:
            assert "id" in entry and "name" in entry and "rationale" in entry
            seen_ids.append(entry["id"])
    expected_ids = [c["id"] for c in audit_json["workload_classes"]]
    assert sorted(seen_ids) == sorted(expected_ids), (
        f"ranked recommendation does not cover every class exactly once: "
        f"ranked={sorted(seen_ids)} classes={sorted(expected_ids)}"
    )


def test_do_not_flags_are_false_per_audit_charter(audit_json):
    assert audit_json["production_claim"] is False
    assert audit_json["ml_training"] is False
    assert audit_json["modifies_robust_energy_engine"] is False
    assert audit_json["modifies_controllers_or_defaults"] is False
    assert audit_json["downloaded_new_datasets"] is False
    assert audit_json["uses_oracle_as_headline"] is False


def test_audit_doc_references_all_required_prior_art(audit_markdown):
    missing = [d for d in REQUIRED_PRIOR_ART_DOCS if d not in audit_markdown]
    assert not missing, f"audit doc missing prior-art references: {missing}"


def test_audit_doc_has_no_unhedged_production_savings_claims(audit_markdown):
    lowered = audit_markdown.lower()
    for phrase in BANNED_PHRASES:
        # Allow a banned phrase only if it appears inside a `NOT` / `Do not` /
        # `not yet` / `never` hedge on the same line (same rule used by the
        # canonical report scanner per `docs/RESULTS.md` §8).
        for line in lowered.splitlines():
            if phrase not in line:
                continue
            if any(
                hedge in line
                for hedge in (
                    "not ",
                    "no ",
                    "never",
                    "do not",
                    "must not",
                    "n't",
                )
            ):
                continue
            pytest.fail(
                f"unhedged banned phrase {phrase!r} in audit doc line: {line!r}"
            )


def test_build_now_bucket_is_focused(audit_json):
    # Discovery audits should not recommend building everything at once.
    # Two is the documented choice (eval workloads + batch inference). The
    # cap is here to flag silent expansion in future edits.
    build_now = audit_json["ranked_recommendation"]["build_now"]
    assert 1 <= len(build_now) <= 2, (
        f"build_now bucket size out of [1, 2]: {len(build_now)}"
    )


def test_workload_class_ids_match_audit_doc(audit_json):
    # Ids 1..8 must match the audit charter ordering.
    for expected_id, cls in enumerate(audit_json["workload_classes"], start=1):
        assert cls["id"] == expected_id, (
            f"workload class {cls['name']} id={cls['id']} expected {expected_id}"
        )


def test_each_class_lists_at_least_one_missing_signal(audit_json):
    # Honesty rule: every class has missing signals (none of the public
    # traces measure everything). A silent empty list is the failure mode
    # this test is here to catch.
    for cls in audit_json["workload_classes"]:
        assert cls["missing_signals"], (
            f"{cls['name']} declares no missing signals — public trace would "
            f"have to be perfect for this to be true"
        )


def test_next_datasets_to_ingest_bounded_only_well_formed(audit_json):
    for entry in audit_json["next_datasets_to_ingest_bounded_only"]:
        assert "dataset" in entry
        assert "url" in entry
        assert "proposed_bound" in entry
        assert "purpose" in entry
        # Every proposed bound must mention a size cap (bounded ingest rule
        # from `docs/MIT_SUPERCLOUD_BOUNDED_REAL_SAMPLE_RESULTS.md`).
        assert re.search(r"(mb|gb|sample|head|bounded)", entry["proposed_bound"], re.I), (
            f"{entry['dataset']} bound not bounded: {entry['proposed_bound']!r}"
        )
