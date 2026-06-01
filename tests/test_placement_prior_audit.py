"""Tests for the placement-prior scoring-path audit."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


AUDIT_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "placement_prior_audit", "scoring_path_audit.json",
)


@pytest.fixture(scope="module")
def audit():
    if not os.path.exists(AUDIT_PATH):
        pytest.skip("scoring_path_audit.json not generated; run "
                    "scripts/audit_placement_prior_scoring_path.py first")
    with open(AUDIT_PATH) as fh:
        return json.load(fh)


# ---------- 1. Audit invariants ------------------------------------------


def test_audit_doc_version_and_flags(audit):
    assert audit["doc_version"] == "placement_scoring_path_audit_v1"
    assert audit["audit_only"] is True
    assert audit["modifies_controllers_or_defaults"] is False
    assert audit["modifies_robust_energy_engine"] is False
    assert audit["uses_oracle_as_headline"] is False
    assert audit["production_claim"] is False
    assert audit["shadow_only"] is True


def test_scorer_signature_recorded(audit):
    sig = audit["scorer_signature"]
    assert sig["callable"] == "score_residency_candidate"
    assert sig["module"] == "aurelius.residency.decision"
    param_names = [p["name"] for p in sig["parameters"]]
    for required in ("request", "candidate_location", "safety_context"):
        assert required in param_names


# ---------- 2. All major goodput/$ inputs are catalogued ------------------


REQUIRED_INPUTS = (
    "TTFT", "TPOT", "E2E latency", "queue depth", "queue wait",
    "GPU type", "model size", "throughput", "KV cache state",
    "cache reuse", "residency / cold-start", "cost", "energy / carbon",
    "SLA risk", "timeout risk",
)


def test_audit_catalogues_all_required_inputs(audit):
    catalogued = {row["input"] for row in audit["scoring_inputs"]}
    missing = set(REQUIRED_INPUTS) - catalogued
    assert not missing, f"audit missing inputs: {missing}"


VALID_CLASSIFICATIONS = frozenset({
    "measured", "measured_or_proxy", "forecasted", "static_prior",
    "heuristic", "constant", "proxy", "missing", "derived",
    "derived_binary",
})


def test_every_input_has_valid_classification(audit):
    for row in audit["scoring_inputs"]:
        assert row["classification"] in VALID_CLASSIFICATIONS, (
            f"{row['input']} has invalid classification "
            f"{row['classification']}"
        )


# ---------- 3. Static/heuristic inputs explicitly identified -------------


def test_static_or_heuristic_list_includes_known_gaps(audit):
    static_or_heuristic = set(audit["static_or_heuristic_inputs"])
    # The TTFT/TPOT inputs are heuristic in the current scorer.
    assert "TTFT" in static_or_heuristic
    assert "TPOT" in static_or_heuristic
    # GPU type, throughput, cache reuse, energy/carbon, timeout risk are
    # all surfaced by the upstream telemetry but not used by the scorer.
    for gap in ("GPU type", "throughput", "cache reuse",
                "energy / carbon", "timeout risk"):
        assert gap in static_or_heuristic


def test_gpu_type_not_used_as_latency_prior(audit):
    assert audit["gpu_type_used_as_latency_prior"] is False


def test_headline_gap_documented(audit):
    assert "GPU type" in audit["headline_gap"]
    assert "CARA" in audit["headline_gap"]


# ---------- 4. Counts add up to the catalogue size -----------------------


def test_classification_counts_consistent(audit):
    total = sum(audit["counts_by_classification"].values())
    assert total == len(audit["scoring_inputs"])


# ---------- 5. Audit does not import executors --------------------------


def test_audit_script_has_no_executor_imports():
    path = os.path.join(REPO_ROOT, "scripts",
                        "audit_placement_prior_scoring_path.py")
    with open(path) as fh:
        src = fh.read()
    for forbidden in (
        "execute_frontier_decision",
        "apply_replica_scale",
        "set_replicas",
        "RUN_FOR_REAL",
    ):
        assert forbidden not in src, (
            f"audit script must not reference executor token '{forbidden}'"
        )


def test_audit_script_no_banned_production_phrase():
    path = os.path.join(REPO_ROOT, "scripts",
                        "audit_placement_prior_scoring_path.py")
    with open(path) as fh:
        src = fh.read().lower()
    for phrase in ("production savings", "guaranteed savings",
                   "production-proven"):
        assert phrase not in src
