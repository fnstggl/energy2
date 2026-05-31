"""Summary-JSON + dataset-registry tests for the Eval + Batch v1 build.

Pins:

1.  The committed summary JSONs (when present) carry the do-not-claim flags
    asserted false.
2.  The PUBLIC_TRACE_BACKTESTS.md dataset registry now includes ShareGPT
    + LMSYS rows AND explicitly marks Azure Functions 2019 as
    DEFERRED_BOUNDED_INGEST.
3.  The new docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md is present and
    references both v1 frontiers + the Azure 2024 sanity gate +
    explicitly states no production savings + no oracle headline.
4.  No banned production-savings phrase appears unhedged in the new
    docs (same scan rule the canonical report renderer uses).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SUMMARY = (REPO_ROOT / "data" / "external" / "frontier"
                / "eval_workload_frontier_v1_summary.json")
BATCH_SUMMARY = (REPO_ROOT / "data" / "external" / "frontier"
                 / "batch_inference_frontier_v1_summary.json")
REGISTRY_DOC = REPO_ROOT / "docs" / "PUBLIC_TRACE_BACKTESTS.md"
RESULTS_DOC = REPO_ROOT / "docs" / "EVAL_AND_BATCH_FRONTIER_RESULTS.md"

REQUIRED_DO_NOT_CLAIM_FLAGS = (
    "production_claim",
    "ml_training",
    "modifies_serving_rho_controller",
    "uses_oracle_as_headline",
    "executable_in_real_cluster",
)

BANNED_PHRASES = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)


def _assert_summary_shape(payload: dict, *, expected_doc_version: str):
    assert payload["doc_version"] == expected_doc_version
    for k in REQUIRED_DO_NOT_CLAIM_FLAGS:
        assert k in payload, f"missing do-not-claim flag {k!r}"
        assert payload[k] is False, (
            f"{k} must be false in summary; got {payload[k]!r}")
    assert "source" in payload
    assert "workload_profile" in payload
    assert "candidate_grid" in payload
    assert "frontier_points" in payload
    assert "recommendation" in payload
    assert "honesty_notes" in payload
    # honesty_notes must call out the "NOT production savings" rule.
    joined = " ".join(payload["honesty_notes"]).lower()
    assert ("not production savings" in joined
            or "no production savings" in joined), (
        "honesty_notes must explicitly state NOT production savings")


@pytest.mark.skipif(not EVAL_SUMMARY.exists(),
                    reason="eval frontier summary not committed yet")
def test_eval_summary_shape():
    with open(EVAL_SUMMARY) as fh:
        payload = json.load(fh)
    _assert_summary_shape(
        payload, expected_doc_version="eval_workload_frontier_v1_summary")
    # Eval profile must carry the synthetic-scenario label.
    label = payload["workload_profile"].get("synthetic_scenario_label", "")
    assert label, "eval profile is missing the synthetic_scenario_label"
    # The eval recommendation must NOT enable real-cluster execution.
    rec = payload["recommendation"]
    assert rec.get("executable_in_real_cluster") is False


@pytest.mark.skipif(not BATCH_SUMMARY.exists(),
                    reason="batch frontier summary not committed yet")
def test_batch_summary_shape():
    with open(BATCH_SUMMARY) as fh:
        payload = json.load(fh)
    _assert_summary_shape(
        payload,
        expected_doc_version="batch_inference_frontier_v1_summary")
    label = payload["workload_profile"].get("synthetic_scenario_label", "")
    assert label, "batch profile is missing the synthetic_scenario_label"
    rec = payload["recommendation"]
    assert rec.get("executable_in_real_cluster") is False
    # The sanity-sweep diagnostic must be present.
    assert "max_safe_goodput_per_slack" in payload


def test_registry_doc_includes_sharegpt_and_lmsys_and_functions():
    assert REGISTRY_DOC.exists(), f"missing {REGISTRY_DOC}"
    text = REGISTRY_DOC.read_text(encoding="utf-8")
    assert "ShareGPT (RyokoAI/ShareGPT52K)" in text, (
        "PUBLIC_TRACE_BACKTESTS.md must register ShareGPT")
    assert "LMSYS Chatbot Arena conversations" in text, (
        "PUBLIC_TRACE_BACKTESTS.md must register LMSYS Chatbot Arena")
    assert "Azure Functions 2019" in text and "DEFERRED_BOUNDED_INGEST" in text


def test_results_doc_exists_and_covers_both_frontiers():
    assert RESULTS_DOC.exists(), f"missing {RESULTS_DOC}"
    text = RESULTS_DOC.read_text(encoding="utf-8")
    assert "Eval Workload Frontier v1" in text
    assert "Batch Inference Frontier v1" in text
    # "NOT production savings" — match across possible blockquote markers
    # and line breaks (markdown formatting can split the phrase).
    flattened = re.sub(r"[\s>]+", " ", text)
    assert "NOT production savings" in flattened
    # Sanity gate is documented.
    assert "deadline-slack-vs-rho slope" in text
    # No oracle as headline.
    assert "oracle" in text.lower()
    assert "no oracle" in text.lower() or "not used as a headline" in text.lower()


def test_results_doc_no_unhedged_banned_phrases():
    text = RESULTS_DOC.read_text(encoding="utf-8").lower()
    for phrase in BANNED_PHRASES:
        for line in text.splitlines():
            if phrase not in line:
                continue
            if any(hedge in line for hedge in (
                "not ", "no ", "never", "do not", "must not", "n't",
            )):
                continue
            pytest.fail(
                f"unhedged banned phrase {phrase!r} in EVAL_AND_BATCH "
                f"results doc line: {line!r}")
