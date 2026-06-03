"""Tests pinning that ``scripts/ingest_hf_llmperf_bedrock.py`` is wired
through the canonical :func:`decide_redistribution` gate.

Tenth consumer of the gate (after
``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``,
``scripts/ingest_hf_llm_energy_consumption.py``,
``scripts/ingest_hf_latency_benchmarks.py``,
``scripts/ingest_hf_optimum_benchmark.py``,
``scripts/ingest_hf_acmetrace.py``, and
``scripts/ingest_hf_lightcap_runtime_telemetry.py``).

The pre-wiring shape carried a hard-coded module-level ``LICENSE =
"apache-2.0"`` constant and wrote it directly into ``summary.json``
with no gate consultation. The refactor adds explicit ``LICENSE_TAG``
/ ``LICENSE_SOURCE`` / ``GATE_SCOPE`` constants, routes the verdict
through the canonical gate, and writes the gate-derived fields
additively onto the per-config ``summary.json`` + the top-level
``round3_broadened_discovery_audit_summary.json``.

This file pins that the script now:

* declares ``LICENSE_TAG`` / ``LICENSE_SOURCE`` / ``GATE_SCOPE`` at
  module level (so a future HF tag change is a one-line edit);
* imports ``decide_redistribution`` from the canonical gate and does
  NOT redeclare the closed permissive allow-list;
* derives ``license_redistribution_status`` from the gate;
* records the gate verdict on the per-config ``summary.json`` and on
  the ``ingested`` row of
  ``round3_broadened_discovery_audit_summary.json``;
* refreshes the audit summary to ``v2`` with the top-level
  ``redistribution_gate_*`` triple (scope / policy default / grant
  count);
* keeps the on-disk fixture bytes and the committed normalised
  sample bytes byte-for-byte unchanged.

This is the tenth-consumer milestone; it is also the **second
apache-2.0-tagged ingestion script** to wire through the gate
(the latency_benchmarks script was the first apache-2.0 consumer).
The apache-2.0 path of the gate's permissive allow-list is therefore
exercised end-to-end through two independent ingesters.

Audit-only — every test reads committed artefacts or runs pure-Python
decision functions. No HF API, no HF_TOKEN read, no data download.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_hf_llmperf_bedrock.py"
REFRESH_PATH = (
    REPO_ROOT / "scripts" / "refresh_hf_llmperf_bedrock_gate_metadata.py"
)
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = (
    DISC_DIR / "round3_broadened_discovery_audit_summary.json"
)
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "ssong1/llmperf-bedrock"
SAFE_NAME = "ssong1__llmperf-bedrock"
CONFIG = "bedrock_claude_instant_v1"


def _summary_path() -> Path:
    return HF_DIR / SAFE_NAME / CONFIG / "processed" / "summary.json"


def _committed_sample_path() -> Path:
    return (
        HF_DIR / SAFE_NAME / CONFIG / "processed"
        / "committed_normalized_sample.jsonl"
    )


def _fixture_path() -> Path:
    return FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"


def _load_module_directly(rel_path: str, name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, rel_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script_module():
    return _load_module_directly(
        "scripts/ingest_hf_llmperf_bedrock.py",
        "ingest_hf_llmperf_bedrock_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_llmperf_bedrock_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_llmperf_bedrock_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


# ---------------------------------------------------------------------------
# 1. Module-level license constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inline inside ``ingest`` / the summary writer.

    ssong1/llmperf-bedrock ships exactly one config sharing one
    license (apache-2.0), so a single ``LICENSE_TAG`` constant
    suffices (unlike the multi-license latency_benchmarks script
    which carries three tag constants).
    """

    assert script_module.DATASET_ID == DATASET_ID
    assert script_module.LICENSE_TAG == "apache-2.0"
    assert script_module.LICENSE == "apache-2.0", (
        "the pre-existing LICENSE module constant must remain unchanged "
        "so the rest of the script (raw_committed comment, etc.) is "
        "untouched; the new LICENSE_TAG mirrors it"
    )
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter license: apache-2.0 "
        "(ssong1 / Ray LLMPerf token_benchmark_ray.py output against "
        "AWS Bedrock anthropic.claude-instant-v1)"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"


def test_license_constant_matches_license_tag(script_module):
    """The pre-existing ``LICENSE`` module constant (used in summary["license"])
    must declare the same tag as the new module-level ``LICENSE_TAG``. A
    drift here would mean the gate is fed a different license than the
    summary records.
    """

    assert script_module.LICENSE == script_module.LICENSE_TAG, (
        f"LICENSE={script_module.LICENSE!r} != "
        f"LICENSE_TAG={script_module.LICENSE_TAG!r}"
    )


# ---------------------------------------------------------------------------
# 2. Script imports the canonical gate (no duplicated classifier)
# ---------------------------------------------------------------------------


def test_script_imports_decide_redistribution(script_source: str):
    """A future maintainer who silently re-introduces a hard-coded
    classifier inside the script must trip this test.
    """

    assert (
        "from aurelius.ingestion.redistribution_gate import"
        in script_source
    ), "script must import decide_redistribution from the canonical gate"
    assert "decide_redistribution" in script_source
    assert "OperatorPolicyLedger" in script_source, (
        "script must load the operator policy ledger"
    )


def test_script_does_not_redeclare_permissive_set(script_source: str):
    """Confidence rail: no second copy of the closed permissive
    allow-list. The gate is the single source of truth.
    """

    forbidden = [
        '"permissive_apache_2_0":',
        '"permissive_cc_by_4_0":',
        '"permissive_cdla_2":',
        '"permissive_mit":',
        '"permissive_cc_by_sa_4_0":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


def test_script_does_not_hardcode_status_code_in_code(script_source: str):
    """The canonical status string ``"permissive_apache_2_0"`` must not
    appear inline in the script's executable code — the gate produces
    it. Docstring mentions are allowed.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "permissive_apache_2_0"
        ):
            offending.append((node.lineno, node.value))
    # Filter docstring occurrences.
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            body = getattr(node, "body", []) or []
            doc = (
                body[0]
                if body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                else None
            )
            if doc and ("permissive_apache_2_0" in doc.value.value):
                for (ln, v) in list(offending):
                    if (
                        ln == doc.lineno
                        and v == "permissive_apache_2_0"
                    ):
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes 'permissive_apache_2_0' at lines "
        f"{[ln for (ln, _) in offending]!r}; the gate produces it"
    )


# ---------------------------------------------------------------------------
# 3. evaluate_redistribution — pure function returns the gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_redistribution_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_redistribution`` returns the gate's
    ``RedistributionGateDecision`` dataclass."""

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert type(decision).__name__ == "RedistributionGateDecision"
    for field in (
        "permitted", "reason_code", "reason_detail",
        "license_status", "license_observed", "scope",
        "operator_grant_dataset_id",
    ):
        assert hasattr(decision, field), (
            f"gate decision missing field {field!r}"
        )


def test_evaluate_redistribution_default_permits_under_empty_ledger(
    script_module, policy_module,
):
    """The default ``license=apache-2.0`` permits under any ledger —
    the gate short-circuits the ledger because the license is on the
    closed permissive allow-list.

    The decision proves the apache-2.0 path of the gate is exercised
    here through a second independent consumer (after the
    latency_benchmarks first apache-2.0 wiring).
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True
    assert decision.license_status == "permissive_apache_2_0"
    assert decision.reason_code == "permitted_declared_permissive_license"
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"
    assert decision.license_observed == "apache-2.0"


def test_evaluate_redistribution_swap_to_none_denies(
    script_module, policy_module,
):
    """Swap the license tag to ``None`` under the same empty ledger
    → the gate flips to DENY. Proves the wiring actually consults the
    license tag — it is not hard-coded to permit.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag=None,
    )
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"


def test_evaluate_redistribution_swap_to_restrictive_denies(
    script_module, policy_module, gate_module,
):
    """Swap the license tag to ``cc-by-nc-4.0`` (declared NON-permissive)
    → the gate denies even though apache-2.0 is on the permissive list.
    The closed permissive allow-list is conservative — variant tags
    are NOT auto-promoted.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag="cc-by-nc-4.0",
    )
    assert decision.permitted is False
    assert decision.license_status == "declared_non_permissive"
    assert decision.reason_code == (
        gate_module.REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE
    )


def test_evaluate_redistribution_operator_grant_irrelevant_for_permissive(
    script_module, policy_module, gate_module,
):
    """An operator grant cannot REVOKE redistribution for an upstream
    permissive license — the gate short-circuits the ledger for
    permissive tags. Pin this here so the tenth consumer cannot
    accidentally re-introduce a ledger check that overrides the
    permissive verdict.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=False,  # operator says "do not redistribute"
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-03T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes="operator opt-out has no effect on declared permissive licenses",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True, (
        "permissive license short-circuits the ledger; an operator "
        "'opt-out' grant must NOT flip the verdict to deny"
    )
    assert decision.reason_code == (
        gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
    )
    assert decision.operator_grant_dataset_id is None


def test_evaluate_redistribution_signature_exposes_keyword_args(script_module):
    """The script exposes ``license_tag`` / ``dataset_id`` / ``ledger``
    / ``scope`` as keyword args so each test override path threads
    cleanly. Pin that the signature still has these keyword args.
    """

    sig = inspect.signature(script_module.evaluate_redistribution)
    assert "license_tag" in sig.parameters
    assert "dataset_id" in sig.parameters
    assert "ledger" in sig.parameters
    assert "scope" in sig.parameters


# ---------------------------------------------------------------------------
# 4. Per-config summary.json carries the new gate-derived fields
# ---------------------------------------------------------------------------


def test_summary_carries_redistribution_gate_metadata():
    s = json.loads(_summary_path().read_text())
    required = {
        "license_redistribution_status",
        "license_redistribution_source",
        "redistribution_gate_reason_code",
        "redistribution_gate_reason_detail",
        "redistribution_gate_permitted",
        "redistribution_gate_operator_grant_dataset_id",
        "redistribution_gate_scope",
    }
    missing = required - s.keys()
    assert not missing, (
        f"committed summary.json missing gate-derived fields: "
        f"{sorted(missing)!r}"
    )
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail


def test_summary_records_permit_verdict_for_apache_2_0():
    """The HF card declares apache-2.0, so the gate permits regardless
    of the ledger contents.
    """

    s = json.loads(_summary_path().read_text())
    assert s["license"] == "apache-2.0"
    assert s["redistribution_gate_permitted"] is True
    assert s["redistribution_gate_reason_code"] == (
        "permitted_declared_permissive_license"
    )
    assert s["license_redistribution_status"] == "permissive_apache_2_0"
    assert s["license_redistribution_source"] == (
        "HF card frontmatter license: apache-2.0 "
        "(ssong1 / Ray LLMPerf token_benchmark_ray.py output against "
        "AWS Bedrock anthropic.claude-instant-v1)"
    )


def test_status_matches_gate_classification(gate_module):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed summary.
    """

    s = json.loads(_summary_path().read_text())
    expected = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected!r} of license tag "
        f"{s['license']!r}"
    )


# ---------------------------------------------------------------------------
# 5. Audit summary carries v2 doc_version + gate-derived fields
# ---------------------------------------------------------------------------


def test_audit_summary_doc_version_is_v2():
    """The Round-3 audit summary moves to v2 here. The v1 schema is a
    strict subset of v2 (every v1 key is preserved); v2 adds the
    top-level ``redistribution_gate_*`` triple and the per-row gate
    fields.
    """

    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == "round3_broadened_discovery_audit_summary_v2"


def test_audit_summary_top_level_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0


def test_audit_summary_preserves_v1_invariants():
    """v2 must NOT drop any v1 invariant the existing audit test asserts.
    The v1 fields (modifies_robust_energy_engine / modifies_controllers /
    production_claim / git_sha / ingested / failed / discovery_only)
    all remain.
    """

    a = json.loads(AUDIT_PATH.read_text())
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False
    assert a["production_claim"] is False
    assert a["uses_oracle_as_headline"] is False
    assert "git_sha" in a
    assert "audited_at_s" in a
    assert "ingested" in a
    assert "failed" in a
    assert "discovery_only_records" in a
    assert "scope" in a, "v1 scope description must be preserved"


def test_audit_summary_llmperf_bedrock_row_has_gate_fields():
    """The tenth consumer extends gate coverage to the llmperf-bedrock
    row — the single ingested entry must carry the four per-row gate
    fields.
    """

    a = json.loads(AUDIT_PATH.read_text())
    seen = 0
    for entry in a["ingested"]:
        if entry["dataset_id"] != DATASET_ID:
            continue
        seen += 1
        for key in (
            "license_redistribution_status",
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
            "redistribution_gate_operator_grant_dataset_id",
        ):
            assert key in entry, (
                f"audit entry {entry['dataset_id']}/"
                f"{entry.get('config_name')} missing {key!r}"
            )
        assert entry["license"] == "apache-2.0"
        assert entry["redistribution_gate_permitted"] is True
        assert entry["redistribution_gate_reason_code"] == (
            "permitted_declared_permissive_license"
        )
        assert entry["license_redistribution_status"] == (
            "permissive_apache_2_0"
        )
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None
    assert seen == 1, (
        f"expected exactly 1 llmperf-bedrock ingested row "
        f"(bedrock_claude_instant_v1), got {seen}"
    )


def test_audit_summary_discovery_only_records_preserved():
    """The eight discovery-only records (DistServe profiling, DynamoRIO,
    intellistream sage-control-plane, Nathan-Maine, hlarcher,
    kshitijthakkar MoE benchmarks) are NOT ingested and do NOT flow
    through the gate — they must still appear with their existing v1
    metadata.
    """

    a = json.loads(AUDIT_PATH.read_text())
    ids = {r["dataset_id"] for r in a["discovery_only_records"]}
    assert {
        "DistServe/2025-05-06T14-automatic-profiling",
        "DistServe/test-amd-ci-profiler",
        "DistServe/test-sample",
        "deepanjalimishra99/datacenter-traces",
        "intellistream/sage-control-plane-llm-workloads",
        "Nathan-Maine/dgx-spark-kv-cache-benchmark",
        "hlarcher/inference-benchmarker",
        "kshitijthakkar/moe-inference-benchmark",
    } <= ids


# ---------------------------------------------------------------------------
# 6. Function signatures accept ledger as keyword arg
# ---------------------------------------------------------------------------


def test_ingest_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module.ingest)
    assert "ledger" in sig.parameters
    p = sig.parameters["ledger"]
    assert p.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert p.default is None


def test_write_round3_audit_summary_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module.write_round3_audit_summary)
    assert "ledger" in sig.parameters
    p = sig.parameters["ledger"]
    assert p.default is None


def test_load_ledger_returns_empty_when_policy_path_missing(
    script_module, tmp_path,
):
    """``_load_ledger`` falls back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — fresh-checkout self-sufficiency.
    """

    nonexistent = tmp_path / "no_such_file.json"
    assert not nonexistent.exists()
    ledger = script_module._load_ledger(nonexistent)
    assert ledger.policy_default == "deny_all"
    assert ledger.grants == ()


def test_load_ledger_reads_committed_default(script_module):
    """``_load_ledger`` (no arg) loads the canonical committed
    operator policy ledger from
    ``data/external/hf_discovery/operator_redistribution_policy.json``
    when present, and that committed default is the deny_all + zero
    grants baseline.
    """

    ledger = script_module._load_ledger()
    assert ledger.policy_default == "deny_all"
    # The committed default ships zero grants — no operator override
    # in effect yet.
    assert ledger.grants == ()


# ---------------------------------------------------------------------------
# 7. Safety — no HF_TOKEN literal in the refactored script
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", script_source)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
    assert not suspicious, (
        f"script contains an HF-token-shaped literal: {suspicious!r}"
    )
    bad_assignment = re.search(
        r'HF_TOKEN\s*=\s*["\']hf_', script_source,
    )
    assert bad_assignment is None, (
        "HF_TOKEN appears to be assigned a literal hf_ value"
    )


def test_no_hf_token_literal_in_refresh_script():
    src = REFRESH_PATH.read_text()
    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", src)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
    assert not suspicious, (
        f"refresh helper contains an HF-token-shaped literal: "
        f"{suspicious!r}"
    )


# ---------------------------------------------------------------------------
# 8. Fixture + committed-sample bytes are byte-for-byte unchanged
# ---------------------------------------------------------------------------


def test_fixture_sha256_matches_summary():
    """Wiring the gate must not change the on-disk fixture bytes — the
    sha256 the summary records must match what's on disk.
    """

    s = json.loads(_summary_path().read_text())
    p = REPO_ROOT / s["fixture_sample_path"]
    assert p.exists(), "missing fixture file"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    assert h.hexdigest() == s["sample_sha256"], (
        "fixture bytes have drifted from recorded sha256"
    )
    assert p.stat().st_size == s["fixture_sample_bytes"]


def test_committed_normalized_sample_sha256_matches_summary():
    """The committed normalised sample (350 rows, ~380 KB) must also be
    byte-for-byte unchanged after the gate wiring.

    This is a stronger invariant than for the AcmeTrace eighth-consumer
    wiring (AcmeTrace ships fixtures only); for llmperf-bedrock every
    committed normalised sample byte is pinned, similar to the ninth
    consumer (Lightcap) pattern.
    """

    s = json.loads(_summary_path().read_text())
    p = REPO_ROOT / s["committed_normalized_sample_path"]
    assert p.exists(), "missing committed normalised sample file"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    assert h.hexdigest() == s["committed_normalized_sample_sha256"], (
        "committed normalised sample bytes have drifted from recorded "
        "sha256"
    )
    assert p.stat().st_size == s["committed_normalized_sample_bytes"]


# ---------------------------------------------------------------------------
# 9. Refresh helper preserves existing v1 fields
# ---------------------------------------------------------------------------


def test_refresh_helper_does_not_add_extra_ingest_rows():
    """The refresh helper must only update the single declared
    llmperf-bedrock row — no silent expansion of the audit summary."""

    a = json.loads(AUDIT_PATH.read_text())
    llmperf_rows = [
        e for e in a["ingested"] if e["dataset_id"] == DATASET_ID
    ]
    configs = {e["config_name"] for e in llmperf_rows}
    assert configs == {CONFIG}, (
        f"unexpected configs in audit summary: {configs}"
    )
    assert len(llmperf_rows) == 1, (
        f"expected exactly one llmperf-bedrock row, got {len(llmperf_rows)}"
    )


def test_refresh_helper_preserves_v1_row_fields():
    """The v2 refresh must NOT drop pre-existing v1 row keys like
    ``promotion_state`` / ``promotion_tags`` / ``promotion_reasons``.
    """

    a = json.loads(AUDIT_PATH.read_text())
    for entry in a["ingested"]:
        if entry["dataset_id"] != DATASET_ID:
            continue
        # v1 keys carried forward
        for key in (
            "canonical_trace_type", "available_signals",
            "missing_signals", "analysis_sample_rows",
            "statistical_sample_strength", "promotion_state",
            "promotion_tags", "limitations",
            "fixture_sample_rows", "committed_normalized_sample_rows",
            "committed_normalized_sample_bytes",
        ):
            assert key in entry, (
                f"audit row {entry.get('config_name')} lost v1 key "
                f"{key!r} during the v2 refresh"
            )


# ---------------------------------------------------------------------------
# 10. Apache-2.0 path coverage — second consumer cross-check
# ---------------------------------------------------------------------------


def test_apache_2_0_path_exercised_via_independent_consumer(
    script_module, policy_module, gate_module,
):
    """The latency_benchmarks ingester was the first apache-2.0
    consumer of the gate; ssong1/llmperf-bedrock is the second.

    Pin that the same canonical gate decision (permissive_apache_2_0
    → permit) is reached through this independent path. If a future
    refactor accidentally introduced a divergent classifier here, the
    two consumers would disagree — this test would trip.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    direct = gate_module.decide_redistribution(
        dataset_id=DATASET_ID,
        license_str="apache-2.0",
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert decision.permitted == direct.permitted
    assert decision.license_status == direct.license_status
    assert decision.reason_code == direct.reason_code
    assert decision.operator_grant_dataset_id == (
        direct.operator_grant_dataset_id
    )
    assert decision.scope == direct.scope
