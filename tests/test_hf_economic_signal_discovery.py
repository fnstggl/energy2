"""Tests for the HF economic-signal discovery audit.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.
Enforce the safety contract from the mission spec:

- audit JSON schema present + per-candidate scores recorded;
- no `HF_TOKEN` literal committed under data/external/hf_discovery/;
- no raw data committed for any candidate (gitignore intact);
- no synthetic-economics candidate is promoted to ingest_now;
- every ingest_now candidate has both license + gated status recorded;
- every join_overlay_candidate has explicit join keys;
- the markdown doc names every operator-policy-only coefficient.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_JSON = (REPO_ROOT / "data" / "external" / "hf_discovery"
              / "economic_signal_discovery_audit.json")
AUDIT_MD = REPO_ROOT / "docs" / "HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md"
DISCOVERY_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
SCRIPT = REPO_ROOT / "scripts" / "discover_hf_economic_signals.py"

# HF_TOKEN values are 35+ chars starting with hf_; this regex catches any
# leak regardless of which specific token was used at discovery time.
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")


@pytest.fixture(scope="module")
def audit() -> dict:
    with open(AUDIT_JSON) as fh:
        return json.load(fh)


# ───────────────────────── 1. Schema present ─────────────────────────


def test_audit_json_exists():
    assert AUDIT_JSON.exists(), f"missing audit JSON: {AUDIT_JSON}"
    assert AUDIT_JSON.stat().st_size > 1000, "audit JSON suspiciously small"


def test_audit_top_level_keys(audit):
    required = {
        "doc_version", "stage", "production_claim", "audit_only",
        "hf_token_committed", "raw_data_committed", "client",
        "search_terms", "search_summary", "buckets",
        "top_20_economic_candidates", "special_audit", "candidates",
        "operator_policy_only_fields",
    }
    missing = required - set(audit.keys())
    assert not missing, f"audit missing keys: {missing}"


def test_audit_safety_flags_pinned(audit):
    assert audit["production_claim"] is False
    assert audit["audit_only"] is True
    assert audit["hf_token_committed"] is False
    assert audit["raw_data_committed"] is False


def test_audit_search_summary_consistent(audit):
    s = audit["search_summary"]
    assert s["total_terms_searched"] >= 50, (
        f"search ran too few terms: {s['total_terms_searched']}")
    assert s["total_unique_datasets_discovered"] >= len(audit["candidates"])
    assert s["total_inspected"] == len(audit["candidates"])


def test_buckets_match_classification_field(audit):
    buckets = audit["buckets"]
    expected_classes = {
        "A_PLUS": "A_PLUS_paired_ops_and_pricing",
        "A": "A_ops_plus_energy_only",
        "B": "B_economics_only_joinable",
        "C": "C_ops_only",
        "D": "D_synthetic_economics",
        "F": "F_irrelevant",
    }
    for cls, bucket_name in expected_classes.items():
        bucket_ids = set(buckets[bucket_name])
        actual_ids = {c["dataset_id"] for c in audit["candidates"]
                      if c["classification"] == cls}
        assert bucket_ids == actual_ids, (
            f"bucket {bucket_name} mismatch with classification={cls}: "
            f"only in bucket={bucket_ids - actual_ids}, "
            f"only in candidates={actual_ids - bucket_ids}")


# ───────────────────────── 2. Per-candidate schema ─────────────────────────


@pytest.fixture(scope="module")
def candidates(audit) -> list[dict]:
    return audit["candidates"]


def test_every_candidate_has_required_keys(candidates):
    required = {
        "dataset_id", "url", "license", "gated_status", "row_count",
        "storage_size_bytes", "files_inspected", "schema_fields",
        "matched_search_terms", "operational_signals_present",
        "economic_signals_present", "join_keys_available",
        "timestamp_available", "region_available", "gpu_type_available",
        "price_or_cost_available", "energy_available", "carbon_available",
        "field_quality", "trust_tier", "can_pair_with_existing_traces",
        "recommended_action", "scores", "classification",
    }
    for c in candidates:
        missing = required - set(c.keys())
        assert not missing, (
            f"candidate {c.get('dataset_id')} missing keys: {missing}")


def test_every_candidate_has_economic_signal_quality(candidates):
    """Mission spec test: every candidate must have an
    economic_signal_quality score."""
    for c in candidates:
        scores = c["scores"]
        assert "economic_signal_quality" in scores, (
            f"{c['dataset_id']} missing scores.economic_signal_quality")
        v = scores["economic_signal_quality"]
        assert isinstance(v, (int, float)), (
            f"{c['dataset_id']} economic_signal_quality not numeric: {v!r}")
        assert 0 <= v <= 10, (
            f"{c['dataset_id']} economic_signal_quality {v} out of [0,10]")


def test_every_candidate_has_all_seven_scores(candidates):
    expected_score_keys = {
        "economic_signal_quality", "operational_signal_quality",
        "joinability", "license_safety", "production_similarity",
        "uniqueness", "expected_scorer_value",
    }
    for c in candidates:
        missing = expected_score_keys - set(c["scores"].keys())
        assert not missing, (
            f"{c['dataset_id']} missing scores: {missing}")
        for k, v in c["scores"].items():
            assert isinstance(v, (int, float)), (
                f"{c['dataset_id']} {k} not numeric: {v!r}")


def test_classification_is_known(candidates):
    allowed = {"A_PLUS", "A", "B", "C", "D", "F"}
    for c in candidates:
        assert c["classification"] in allowed, (
            f"{c['dataset_id']} bad classification: {c['classification']}")


def test_field_quality_is_known(candidates):
    allowed = {"measured", "derived", "proxy", "synthetic", "missing"}
    for c in candidates:
        assert c["field_quality"] in allowed, (
            f"{c['dataset_id']} bad field_quality: {c['field_quality']}")


def test_recommended_action_is_known(candidates):
    allowed = {
        "ingest_now", "join_overlay_candidate", "metadata_only",
        "reject_synthetic", "reject_no_economics", "reject_no_ops",
        "license_blocked", "gated_blocked",
    }
    for c in candidates:
        assert c["recommended_action"] in allowed, (
            f"{c['dataset_id']} bad recommended_action: "
            f"{c['recommended_action']}")


def test_can_pair_with_existing_traces_has_all_traces(candidates):
    expected = {"CARA", "AcmeTrace", "Optimum", "BurstGPT",
                "Google Cluster", "SwissAI", "AgentPerfBench"}
    for c in candidates:
        assert set(c["can_pair_with_existing_traces"].keys()) == expected, (
            f"{c['dataset_id']} pairing-trace set mismatch")


# ───────────────────────── 3. Safety: no token / raw leak ─────────────────


def test_no_hf_token_in_discovery_dir():
    """No `hf_*` token literal anywhere under data/external/hf_discovery/."""
    leaks = []
    for p in DISCOVERY_DIR.rglob("*"):
        if not p.is_file():
            continue
        try:
            body = p.read_text(errors="ignore")
        except OSError:
            continue
        if HF_TOKEN_PATTERN.search(body):
            leaks.append(str(p.relative_to(REPO_ROOT)))
    assert not leaks, f"HF_TOKEN leaked into committed files: {leaks}"


def test_no_hf_token_in_script_or_docs():
    """Script + doc must not embed a token literal."""
    for p in (SCRIPT, AUDIT_MD):
        body = p.read_text()
        assert not HF_TOKEN_PATTERN.search(body), (
            f"HF_TOKEN literal embedded in {p.relative_to(REPO_ROOT)}")


def test_no_raw_files_tracked_by_git():
    """No raw downloads from the new datasets sit under data/external/hf."""
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    assert not raw_committed, (
        f"raw downloads committed (gitignore broken): {raw_committed}")


def test_no_new_processed_data_committed_for_economic_datasets():
    """The two NEW B-class economic candidates must not have any
    processed/normalized files committed yet — this PR is metadata
    only."""
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    for ds in ("afhubbard__gpu-prices", "labofsahil__aws-pricing-dataset"):
        committed = [p for p in out if f"/hf/{ds}/" in p]
        assert not committed, (
            f"this audit must not commit processed data for {ds}: "
            f"{committed}")


# ───────────────────────── 4. Action ladder safety ─────────────────────────


def test_no_synthetic_dataset_promoted_to_ingest_now(candidates):
    for c in candidates:
        if c["recommended_action"] == "ingest_now":
            assert c["field_quality"] != "synthetic", (
                f"{c['dataset_id']} synthetic but action=ingest_now")
            assert c["classification"] != "D", (
                f"{c['dataset_id']} D-class but action=ingest_now")


def test_ingest_now_has_license_and_gating(candidates):
    """Every ingest_now candidate must record both license + gated
    status so a downstream redistribution policy decision is possible."""
    for c in candidates:
        if c["recommended_action"] != "ingest_now":
            continue
        assert c.get("license") is not None and c["license"] != "", (
            f"{c['dataset_id']} action=ingest_now but license missing")
        assert c.get("gated_status") in ("public", "gated", "private"), (
            f"{c['dataset_id']} action=ingest_now but gated_status="
            f"{c.get('gated_status')!r}")


def test_join_overlay_candidate_has_explicit_join_keys(candidates):
    """Every join_overlay_candidate must record at least one join key."""
    found_at_least_one = False
    for c in candidates:
        if c["recommended_action"] != "join_overlay_candidate":
            continue
        found_at_least_one = True
        assert c["join_keys_available"], (
            f"{c['dataset_id']} action=join_overlay_candidate but "
            "join_keys_available is empty")
    # The audit headline relies on afhubbard/gpu-prices being a join
    # overlay candidate. If that drops out, the headline is wrong.
    assert found_at_least_one, (
        "expected at least one join_overlay_candidate in the audit")


def test_afhubbard_gpu_prices_is_join_overlay(candidates):
    """The single strongest new economic finding is afhubbard/gpu-prices.
    If this changes class or action, the doc + scorer-roadmap §10 of the
    markdown report must be revisited."""
    target = next((c for c in candidates
                   if c["dataset_id"] == "afhubbard/gpu-prices"), None)
    assert target is not None, (
        "afhubbard/gpu-prices missing — discovery script regressed")
    assert target["classification"] == "B", (
        f"afhubbard/gpu-prices class changed: {target['classification']}")
    assert target["recommended_action"] == "join_overlay_candidate", (
        f"afhubbard/gpu-prices action changed: "
        f"{target['recommended_action']}")
    assert target["price_or_cost_available"] is True
    assert target["gpu_type_available"] is True


def test_no_a_plus_paired_dataset_found(audit):
    """Headline finding: no public dataset combines same-record ops +
    pricing. If this ever flips to non-empty, §3.Q1 + §5 + §10 of the
    markdown report must be rewritten — the test forces that review."""
    assert audit["buckets"]["A_PLUS_paired_ops_and_pricing"] == [], (
        "A_PLUS bucket non-empty — markdown report needs to be updated: "
        f"{audit['buckets']['A_PLUS_paired_ops_and_pricing']}")


# ───────────────────────── 5. Doc consistency ─────────────────────────


def test_doc_exists_and_nontrivial():
    assert AUDIT_MD.exists()
    body = AUDIT_MD.read_text()
    assert len(body) > 3000, "doc suspiciously short"


def test_doc_mentions_every_operator_policy_only_field(audit):
    """Mission spec: docs mention remaining operator-policy-only fields.
    Every name in the audit's `operator_policy_only_fields` must appear
    in the markdown."""
    body = AUDIT_MD.read_text().lower()
    for field in audit["operator_policy_only_fields"]:
        # match the keyword token, not the full descriptive parenthetical
        token = re.split(r"\s*\(", field)[0].strip().lower()
        assert token in body, (
            f"operator-policy-only field not named in doc: {field!r}")


def test_doc_mentions_special_audit_answers(audit):
    """The §3 special audit must surface the five mission questions.
    Markers are short fragments that survive markdown line wrapping."""
    # Collapse whitespace runs so multi-line markdown still matches.
    body = re.sub(r"\s+", " ", AUDIT_MD.read_text().lower())
    for marker in (
        "real gpu-hour pricing plus",
        "cloud billing/chargeback plus workload",
        "energy per request plus",
        "calibrate the scorer without invented constants",
        "operator-policy-only",
    ):
        assert marker in body, (
            f"doc missing special-audit marker: {marker!r}")


def test_doc_lists_afhubbard_as_strongest_candidate():
    body = AUDIT_MD.read_text()
    assert "afhubbard/gpu-prices" in body
    assert "Strongest candidate for scorer calibration" in body


# ───────────────────────── 6. Search-term completeness ─────────────────────


def test_search_terms_cover_mission_spec(audit):
    """The mission spec lists ~57 verbatim search terms. The audit must
    have run all of them across its primary + combination buckets."""
    expected = {
        "gpu cost", "gpu pricing", "cloud gpu pricing", "spot gpu pricing",
        "cloud cost telemetry", "cloud billing", "cloud invoice",
        "chargeback", "cost per token", "cost per request",
        "llm cost telemetry", "serving cost", "inference cost",
        "gpu energy", "energy per request", "kwh inference",
        "power draw gpu", "gpu power telemetry", "ipmi power gpu",
        "dcgm power", "carbon intensity datacenter", "datacenter energy",
        "ttft cost", "tpot cost", "latency cost", "queue cost",
        "energy aware scheduling gpu", "carbon aware scheduling gpu",
        "spot instance trace", "preemptible instance trace",
    }
    terms = (set(audit["search_terms"]["primary"])
             | set(audit["search_terms"]["combinations"]))
    missing = expected - terms
    assert not missing, f"audit missed mission-spec search terms: {missing}"
