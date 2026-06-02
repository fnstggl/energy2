"""Tests for the Round-8 HF discovery audit.

Covers:

* Round-8 audit summary exists and records all 11 discovery-only
  candidates with the expected buckets / reasons.
* Candidate registry contains a round-8 audit block + the 11 round-8-
  tagged candidates.
* No new HF data committed (only JSON audits + a docs update).
* No HF_TOKEN leaked in any committed audit JSON.
* No production_claim / uses_oracle_as_headline anywhere in the
  Round-8 artefacts.
* The Round-8 audit carries the 4th-consecutive-round negative-result
  finding on economic signals.
* license_blocked_followup_candidates is a distinct, non-empty list
  reflecting the new failure mode this round.

Audit-only — tests read committed artefacts; they do NOT hit the HF API.
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

DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
ROUND8_AUDIT = DISC_DIR / "round8_broadened_discovery_audit_summary.json"
CANDIDATES = DISC_DIR / "hf_dataset_candidates.json"


ROUND8_DISCOVERY_ONLY_IDS = [
    "sasha/co2_models",
    "ohdoking/energy_consumption_by_model_and_gpu",
    "dadadada1/Inference-Performance-Dataset",
    "anon-betterbench/betterbench-inference-logs",
    "sairamn/gcp-cloud-billing-cost",
    "ClarusC64/datacenter-power-load-coherence-risk-v0.1",
    "deepanjalimishra99/datacenter-traces",
    "programasweights/paw-inference-logs",
    "minhkhoi1026/opencl-llmperf",
    "ICOS-AI/scaphandre_power_consumption",
    "ICOS-AI/scaphandre_cpu_usage",
]


LICENSE_BLOCKED_IDS = [
    "sasha/co2_models",
    "ohdoking/energy_consumption_by_model_and_gpu",
    "dadadada1/Inference-Performance-Dataset",
    "anon-betterbench/betterbench-inference-logs",
]


# ---------------------------------------------------------------------------
# Round-8 audit summary
# ---------------------------------------------------------------------------


def test_round8_audit_summary_exists() -> None:
    assert ROUND8_AUDIT.exists()


def test_round8_audit_summary_has_correct_doc_version() -> None:
    d = json.loads(ROUND8_AUDIT.read_text())
    assert d.get("doc_version") == "round8_broadened_discovery_audit_summary_v1"


def test_round8_audit_summary_ingested_is_empty() -> None:
    """Round 8 is a no-new-ingest audit round."""
    d = json.loads(ROUND8_AUDIT.read_text())
    assert d["ingested"] == []
    assert d["failed"] == []


def test_round8_audit_summary_no_production_claim_or_oracle() -> None:
    d = json.loads(ROUND8_AUDIT.read_text())
    assert d["production_claim"] is False
    assert d["uses_oracle_as_headline"] is False
    assert d["modifies_robust_energy_engine"] is False
    assert d["modifies_controllers_or_defaults"] is False


def test_round8_audit_summary_records_all_eleven_discovery_only() -> None:
    d = json.loads(ROUND8_AUDIT.read_text())
    recs = d["discovery_only_records"]
    assert len(recs) == len(ROUND8_DISCOVERY_ONLY_IDS)
    ids = {r["dataset_id"] for r in recs}
    assert ids == set(ROUND8_DISCOVERY_ONLY_IDS)


def test_round8_audit_records_all_carry_bucket_and_reason() -> None:
    d = json.loads(ROUND8_AUDIT.read_text())
    for r in d["discovery_only_records"]:
        assert r["bucket"].startswith(("D_", "F_", "I_"))
        assert isinstance(r["reason"], str) and len(r["reason"]) > 80
        assert r["kind"].startswith(("reject_", "inspect_"))


def test_round8_license_blocked_followup_is_explicit() -> None:
    """The new 'license=None' failure category must be a distinct list."""
    d = json.loads(ROUND8_AUDIT.read_text())
    fl = d["license_blocked_followup_candidates"]
    assert isinstance(fl, list)
    ids = {r["dataset_id"] for r in fl}
    assert ids == set(LICENSE_BLOCKED_IDS)
    for r in fl:
        assert "reason_to_revisit" in r
        assert "required_action_to_unblock" in r
        assert "license" in r["required_action_to_unblock"].lower() or \
               "operator-policy" in r["required_action_to_unblock"].lower()


def test_round8_audit_economic_priority_summary_shape() -> None:
    d = json.loads(ROUND8_AUDIT.read_text())
    eps = d["economic_priority_summary"]
    # No round-8 dataset closed the operational+economic join gap.
    assert eps["datasets_with_operational_and_economic_signals"] == []
    assert eps["datasets_with_economic_only_signals"] == []
    assert eps["scorer_coefficients_calibratable_from_round8"] == []
    # Operator-policy-only list MUST still mention the 6 known gaps.
    op = set(eps["scorer_coefficients_operator_policy_only_after_round8"])
    assert {
        "gpu_hour_price_usd", "kwh_per_request", "carbon_g_per_kwh",
        "spot_interruption_probability", "egress_cost_per_gb",
        "regional_price_usd_per_mwh",
    } <= op
    # Negative-result finding must reference the 4-consecutive-round count.
    nrf = eps["negative_result_finding"]
    assert "FOURTH CONSECUTIVE ROUND" in nrf or "fourth consecutive round" in nrf.lower()
    assert "license" in nrf.lower()


def test_round8_kind_distribution_matches_finding() -> None:
    """4 license_blocked + 2 synthetic + 3 irrelevant/out_of_scope + 2 low_value."""
    d = json.loads(ROUND8_AUDIT.read_text())
    kinds = [r["kind"] for r in d["discovery_only_records"]]
    assert kinds.count("inspect_manually_license_blocked") == 4
    assert kinds.count("reject_synthetic_economics") == 1
    assert kinds.count("reject_synthetic_estimates") == 1
    assert kinds.count("reject_irrelevant_domain") == 2
    assert kinds.count("reject_out_of_scope") == 1
    assert kinds.count("reject_low_value_no_workload_context") == 2


# ---------------------------------------------------------------------------
# Candidate registry updates
# ---------------------------------------------------------------------------


def test_candidate_registry_has_round8_audit_block() -> None:
    d = json.loads(CANDIDATES.read_text())
    audit = d.get("focused_audit_2026_06_02_round8")
    assert audit is not None
    assert audit["doc_version"] == "round8_broadened_discovery_audit_v1"
    assert "Round-8 broadened HF discovery" in audit["scope"]


def test_candidate_registry_records_all_eleven_round8_candidates() -> None:
    d = json.loads(CANDIDATES.read_text())
    cands = d["candidates"]
    by_id = {c["dataset_id"]: c for c in cands}
    for ds_id in ROUND8_DISCOVERY_ONLY_IDS:
        assert ds_id in by_id, f"missing {ds_id}"
        c = by_id[ds_id]
        assert "round8_audit_bucket" in c
        assert "round8_audit_reason" in c
        kw = c.get("matched_keywords") or []
        assert any(k.startswith("round8::") for k in kw)


def test_candidate_registry_license_blocked_have_inspect_recommendation() -> None:
    d = json.loads(CANDIDATES.read_text())
    by_id = {c["dataset_id"]: c for c in d["candidates"]}
    for ds_id in LICENSE_BLOCKED_IDS:
        c = by_id[ds_id]
        assert c["recommended_action"] == "inspect_manually_license_blocked"


def test_candidate_registry_synthetic_economics_have_reject_recommendation() -> None:
    d = json.loads(CANDIDATES.read_text())
    by_id = {c["dataset_id"]: c for c in d["candidates"]}
    assert by_id["sairamn/gcp-cloud-billing-cost"]["recommended_action"] == \
        "reject_synthetic_economics"
    assert by_id[
        "ClarusC64/datacenter-power-load-coherence-risk-v0.1"
    ]["recommended_action"] == "reject_synthetic_estimates"


def test_candidate_registry_scaphandre_have_low_value_recommendation() -> None:
    d = json.loads(CANDIDATES.read_text())
    by_id = {c["dataset_id"]: c for c in d["candidates"]}
    for ds_id in ("ICOS-AI/scaphandre_power_consumption",
                  "ICOS-AI/scaphandre_cpu_usage"):
        c = by_id[ds_id]
        assert c["recommended_action"] == "reject_low_value_no_workload_context"
        # Real Apache-2.0 telemetry, license is recorded
        assert c.get("license") == "apache-2.0"


# ---------------------------------------------------------------------------
# No-secret + no-raw-data guards
# ---------------------------------------------------------------------------


HF_TOKEN_RX = re.compile(r"hf_[A-Za-z0-9]{32,}")


def test_round8_audit_no_hf_token_leak() -> None:
    """No committed JSON may contain a real HF_TOKEN string."""
    for p in (ROUND8_AUDIT, CANDIDATES):
        body = p.read_text()
        assert not HF_TOKEN_RX.search(body), f"HF_TOKEN leak in {p}"


def test_round8_audit_no_raw_data_committed() -> None:
    """Round 8 must NOT commit raw HF data — only JSON audits."""
    for ds_id in ROUND8_DISCOVERY_ONLY_IDS:
        safe = ds_id.replace("/", "__")
        raw = REPO_ROOT / "data" / "external" / "hf" / safe / "raw"
        # No raw download directory may be committed for a discovery-only
        # dataset.
        if raw.exists():
            assert not any(
                p.is_file() and p.stat().st_size > 0
                for p in raw.rglob("*")
            ), f"raw data committed for discovery-only {ds_id}"


# ---------------------------------------------------------------------------
# Consistency with the registry doc
# ---------------------------------------------------------------------------


def test_registry_doc_mentions_round8() -> None:
    p = REPO_ROOT / "docs" / "HF_DATASET_REGISTRY.md"
    assert p.exists()
    body = p.read_text()
    assert "Round-8" in body or "Round 8" in body
    # Documents the new license-blocked failure category
    assert "license=None" in body or "license_blocked" in body or "license-blocked" in body
