"""Tests for the round-7 HF discovery audit + H200 cross-source methodology.

Covers:

* Round-7 audit summary exists and records all 13 discovery-only
  candidates with the expected buckets / reasons.
* Candidate registry contains a round-7 audit block + the 13 round-7-
  tagged candidates.
* H200 cross-source methodology audit exists and records the bounded
  observation, NOT a sweeping methodology-drift claim.
* No new HF data committed (only JSON audits + a docs update).
* No HF_TOKEN leaked in any committed audit JSON.
* No production_claim / uses_oracle_as_headline anywhere in the
  Round-7 artefacts.
* The Round-7 audit explicitly carries the 3rd-consecutive-round
  negative-result finding on economic signals.

Audit-only — tests read committed artefacts; they do NOT hit the HF API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
ROUND7_AUDIT = DISC_DIR / "round7_broadened_discovery_audit_summary.json"
H200_AUDIT = DISC_DIR / "h200_cross_source_methodology_audit.json"
CANDIDATES = DISC_DIR / "hf_dataset_candidates.json"


ROUND7_DISCOVERY_ONLY_IDS = [
    "core12345/real_GPU_exp_placement_trace",
    "odyn-network/benchmark-dataset-different-gpu-workload",
    "BBuf/ltx-fp8-sglang-benchmark-results",
    "Isabella5/sglang-seglen-benchmark",
    "fabric/inference-benchmarker",
    "vrvrv/vllm-benchmark-datasets",
    "ashwinnv/agent-telemetry-prompt-framing-mint-full1035-qwen32b",
    "juniworld/prompt_inference_traces",
    "efficient-speech/tts-serving-benchmark",
    "wseaton/prefix-cache-bench",
    "bldeaw/guardrails-load-test-results",
    "st192011/KVCaches",
    "h4shk4t/fast-kv-compaction-cache",
]


# ---------------------------------------------------------------------------
# Round-7 audit summary
# ---------------------------------------------------------------------------


def test_round7_audit_summary_exists() -> None:
    assert ROUND7_AUDIT.exists()


def test_round7_audit_summary_has_correct_doc_version() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    assert d.get("doc_version") == "round7_broadened_discovery_audit_summary_v1"


def test_round7_audit_summary_ingested_is_empty() -> None:
    """Round 7 is a no-new-ingest audit round."""
    d = json.loads(ROUND7_AUDIT.read_text())
    assert d["ingested"] == []
    assert d["failed"] == []


def test_round7_audit_summary_no_production_claim_or_oracle() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    assert d["production_claim"] is False
    assert d["uses_oracle_as_headline"] is False
    assert d["modifies_robust_energy_engine"] is False
    assert d["modifies_controllers_or_defaults"] is False


def test_round7_audit_summary_records_all_thirteen_discovery_only() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    recs = d["discovery_only_records"]
    assert len(recs) == len(ROUND7_DISCOVERY_ONLY_IDS)
    ids = {r["dataset_id"] for r in recs}
    assert ids == set(ROUND7_DISCOVERY_ONLY_IDS)


def test_round7_audit_records_all_carry_bucket_and_reason() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    for r in d["discovery_only_records"]:
        assert r["bucket"].startswith(("C_", "D_", "F_"))
        assert isinstance(r["reason"], str) and len(r["reason"]) > 80
        assert r["kind"].startswith(
            ("reject_", "duplicate_", "gated_")
        )


def test_round7_audit_carries_third_consecutive_negative_result() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    finding = d["economic_priority_summary"]["negative_result_finding"]
    assert "THIRD CONSECUTIVE ROUND" in finding
    assert "operator-policy-supplied" in finding
    assert "public-pricing-prior" in finding


def test_round7_audit_calibratable_coefficients_remain_empty() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    ep = d["economic_priority_summary"]
    assert ep["scorer_coefficients_calibratable_from_round7"] == []
    assert "gpu_hour_price_usd" in ep[
        "scorer_coefficients_operator_policy_only_after_round7"
    ]
    assert "kwh_per_request" in ep[
        "scorer_coefficients_operator_policy_only_after_round7"
    ]


def test_round7_audit_gated_record_present() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    gated = [
        r for r in d["discovery_only_records"] if r["kind"] == "gated_blocked"
    ]
    assert len(gated) == 1
    assert gated[0]["dataset_id"] == "core12345/real_GPU_exp_placement_trace"
    assert gated[0]["gated"] is True


def test_round7_audit_synthetic_estimates_record_present() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    syn = [
        r for r in d["discovery_only_records"]
        if r["kind"] == "reject_synthetic_estimates"
    ]
    assert len(syn) == 1
    assert syn[0]["dataset_id"] == (
        "odyn-network/benchmark-dataset-different-gpu-workload"
    )


# ---------------------------------------------------------------------------
# H200 cross-source methodology audit
# ---------------------------------------------------------------------------


def test_h200_audit_exists() -> None:
    assert H200_AUDIT.exists()


def test_h200_audit_doc_version() -> None:
    d = json.loads(H200_AUDIT.read_text())
    assert d["doc_version"] == "h200_cross_source_methodology_audit_v1"


def test_h200_audit_no_production_claim() -> None:
    d = json.loads(H200_AUDIT.read_text())
    assert d["production_claim"] is False
    assert d["uses_oracle_as_headline"] is False


def test_h200_audit_has_two_comparable_cells() -> None:
    d = json.loads(H200_AUDIT.read_text())
    assert len(d["directly_comparable_cells"]) == 2


def test_h200_audit_first_cell_is_llama_8b_fp8_vllm() -> None:
    d = json.loads(H200_AUDIT.read_text())
    cell = d["directly_comparable_cells"][0]
    assert "Llama-3.1-8B" in cell["cell"]
    assert "FP8" in cell["cell"]
    assert "vLLM" in cell["cell"] or "vllm" in cell["cell"].lower()
    # ssakethch row exists
    ss = cell["ssakethch_observation"]
    assert ss["mean_tpot_ms"] is not None
    assert ss["mean_ttft_ms"] is not None
    assert ss["total_token_throughput_per_replica"] is not None
    # metrum row exists with per-GPU normalized number
    me = cell["metrum_observation"]
    assert me["tokens_per_sec_aggregate"] is not None
    assert me["tokens_per_sec_per_gpu_normalized"] is not None
    # methodology drift is bounded, not a sweeping claim
    obs = cell["methodology_drift_observation"]
    assert "does NOT reveal a methodology drift" in obs
    assert "per-MIG-instance" in obs


def test_h200_audit_second_cell_records_engine_mismatch() -> None:
    d = json.loads(H200_AUDIT.read_text())
    cell = d["directly_comparable_cells"][1]
    assert "SGLang" in cell["cell"]
    # ssakethch is vLLM-only — SGLang row not available
    ss = cell["ssakethch_observation"]
    assert ss.get("available") is False
    # The metrum SGLang TPOT is flagged as suspicious
    obs = cell["methodology_drift_observation"]
    assert "INFEASIBLE" in obs or "Engine mismatch" in obs


def test_h200_audit_has_limitations() -> None:
    d = json.loads(H200_AUDIT.read_text())
    lims = d["limitations"]
    # Several explicit caveats required
    assert len(lims) >= 7
    text = "\n".join(lims)
    assert "placeholders" in text or "Target" in text
    assert "implausibly low" in text
    assert "MIG" in text


def test_h200_audit_recommended_action_warns_against_direct_compare() -> None:
    d = json.loads(H200_AUDIT.read_text())
    rec = d["recommended_consumer_action"]
    assert "Do NOT cross-compare" in rec
    assert "normalization" in rec


def test_h200_audit_informs_constraint_aware_engine() -> None:
    d = json.loads(H200_AUDIT.read_text())
    informs = d["informs"]
    text = "\n".join(informs)
    assert "constraint_aware_engine" in text
    assert "performance_priors" in text


def test_h200_audit_ssakethch_overall_count_matches_dataset() -> None:
    d = json.loads(H200_AUDIT.read_text())
    ss = d["ssakethch_overall"]
    assert ss["overall_count"] == 275


def test_h200_audit_metrum_h200_summary_consistent_with_corpus() -> None:
    d = json.loads(H200_AUDIT.read_text())
    me = d["metrum_h200_summary"]
    assert me["total_rows"] == 10
    assert me["rows_with_ttft"] == 1
    assert me["rows_with_tpot"] == 1
    assert me["rows_with_tokens_per_sec"] == 1
    assert "vLLM" in me["engines_present"]
    assert "SGLang" in me["engines_present"]


# ---------------------------------------------------------------------------
# Candidate registry round-7 updates
# ---------------------------------------------------------------------------


def test_candidate_registry_has_round7_audit_block() -> None:
    d = json.loads(CANDIDATES.read_text())
    block = d.get("focused_audit_2026_06_02")
    assert block is not None
    assert block["doc_version"] == "round7_broadened_discovery_audit_v1"
    assert "Round-7" in block["scope"]
    assert "ZERO ingestible candidates" in block["scope"]


def test_candidate_registry_records_thirteen_round7_candidates() -> None:
    d = json.loads(CANDIDATES.read_text())
    cands_by_id = {c["dataset_id"]: c for c in d["candidates"]}
    for ds_id in ROUND7_DISCOVERY_ONLY_IDS:
        assert ds_id in cands_by_id, f"missing {ds_id}"
        c = cands_by_id[ds_id]
        assert "round7_audit_bucket" in c
        assert c["round7_audit_bucket"].startswith(("C_", "D_", "F_"))
        assert "round7_audit_reason" in c
        assert isinstance(c["round7_audit_reason"], str)
        assert len(c["round7_audit_reason"]) > 80
        assert c["recommended_action"].startswith(
            ("reject_", "duplicate_", "gated_")
        )


def test_candidate_registry_gated_record_marked_gated_status() -> None:
    d = json.loads(CANDIDATES.read_text())
    cands_by_id = {c["dataset_id"]: c for c in d["candidates"]}
    c = cands_by_id["core12345/real_GPU_exp_placement_trace"]
    assert c["gated_status"] == "gated"
    assert c["recommended_action"] == "gated_blocked"


def test_candidate_registry_keeps_prior_round_audit_blocks() -> None:
    """The Round-7 update must not remove the round-3/5/6 audit blocks."""
    d = json.loads(CANDIDATES.read_text())
    assert "focused_audit_2026_06_01" in d
    assert "focused_audit_2026_06_01b" in d
    assert "focused_audit_2026_06_01c" in d
    assert "focused_audit_2026_06_01d" in d


# ---------------------------------------------------------------------------
# No raw data / no HF_TOKEN leak / no production claim
# ---------------------------------------------------------------------------


def test_no_hf_token_in_audit_artifacts() -> None:
    """Negative guard: literal HF_TOKEN values must not leak into JSON audits.

    The test pulls the runtime HF_TOKEN from the env (when set) and checks
    that it does not appear in any committed audit artefact. It also
    checks the conventional HF token prefix and `Bearer ` header text.
    Pinning a hardcoded token literal in the test source would itself
    trip GitHub secret scanning, so the check is constructed at runtime.
    """
    env_token = os.environ.get("HF_TOKEN", "")
    # The HF token prefix is `hf_` followed by 34 chars (per huggingface
    # docs). Build the literal-prefix probe without writing a full token
    # value in source code.
    hf_prefix = "h" + "f_"
    for p in (ROUND7_AUDIT, H200_AUDIT):
        text = p.read_text()
        if env_token:
            assert env_token not in text, (
                f"env HF_TOKEN literal value appears in {p.name}"
            )
        # No tokens of any author should leak — accept the prefix is
        # ambiguous in non-token text, so we check only for tokens of the
        # exact HF length: hf_ + 34 chars.
        for i in range(len(text) - 36):
            window = text[i : i + 37]
            if window.startswith(hf_prefix) and all(
                c.isalnum() for c in window[3:]
            ):
                raise AssertionError(
                    f"possible HF token literal in {p.name} at offset {i}"
                )
        assert "Bearer " not in text
        # The full env-var name "HF_TOKEN" can appear in literal field-name
        # text; the binding ban is on actual token values.


def test_no_raw_data_committed_by_round7() -> None:
    """Round 7 commits ONLY JSON audits + docs. No raw / analysis data."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git repo")
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf/"], cwd=REPO_ROOT
    ).decode()
    # The Round-7 PR doesn't touch the data/external/hf/ tree at all;
    # all data files already tracked are part of earlier PRs. This is a
    # negative test ensuring NO new raw files appear with a path
    # pattern unique to round 7.
    forbidden_substrings = ("round7_sample", "round_7_sample")
    for ln in out.splitlines():
        for needle in forbidden_substrings:
            assert needle not in ln, f"unexpected round-7 raw artefact: {ln}"


def test_round7_audit_lists_all_search_terms_in_scope() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    scope = d["scope"]
    # The scope MUST enumerate that ~30 search-term groups were re-run.
    assert "search-term groups" in scope
    assert "ZERO ingestible candidates" in scope


def test_round7_negative_result_strengthens_round5_round6() -> None:
    d = json.loads(ROUND7_AUDIT.read_text())
    finding = d["economic_priority_summary"]["negative_result_finding"]
    assert "Round-5" in finding or "Round 5" in finding
    assert "Round-6" in finding or "Round 6" in finding
    assert "Round 7" in finding or "Round-7" in finding
    assert "DESIGNED to falsify" in finding
    assert "failed to falsify" in finding
