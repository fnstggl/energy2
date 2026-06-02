"""Tests pinning that ``scripts/ingest_hf_optimum_benchmark.py`` is
wired through the canonical :func:`decide_redistribution` gate.

Seventh consumer of the gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``,
``scripts/ingest_hf_llm_energy_consumption.py``, and
``scripts/ingest_hf_latency_benchmarks.py``).

The pre-wiring shape carried a hard-coded ``LICENSE = None`` module
constant and an inline skip-reason string
``"license_unspecified_no_redistribution_promise"`` baked into
``_finalize_config``. The refactor lifts the raw HF license tag and
its human-curated provenance to module-level constants
(``LICENSE_TAG`` / ``LICENSE_SOURCE``) and routes the commit decision
through the canonical gate; the skip-reason string is preserved
verbatim on the committed ``committed_normalized_sample_reason_skipped``
field for downstream back-compat (pinned by
``test_license_recorded_as_none`` in
``tests/test_hf_optimum_benchmark_ingest.py``).

This file pins that the script now:

* records the raw HF license tag in ``LICENSE_TAG`` and the human
  provenance string in ``LICENSE_SOURCE`` (so a future HF tag change
  is a single-line edit);
* derives ``"license_redistribution_status"`` from the gate;
* preserves the v1 skip-reason string on
  ``committed_normalized_sample_reason_skipped`` for back-compat;
* writes the gate-derived metadata into both the per-config
  summary.json and the broadened-discovery audit summary;
* keeps the on-disk fixture samples byte-for-byte unchanged;
* never re-introduces a classifier or hard-coded permit/deny verdict.

Audit-only — every test reads committed artefacts or runs pure-Python
decision functions. No HF API, no HF_TOKEN read, no data download.
"""
from __future__ import annotations

import ast
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

SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_hf_optimum_benchmark.py"
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = DISC_DIR / "broadened_discovery_audit_summary.json"

DATASET_ID = "optimum-benchmark/llm-perf-leaderboard"
SAFE_NAME = "optimum-benchmark__llm-perf-leaderboard"

# The 9 working configs (openvino is intentionally omitted at the
# script level because every row in that CSV is an isolated-process
# crash with zero measured latency columns).
COMMITTED_CONFIGS = (
    "pytorch_cuda_unquantized_1xA100",
    "pytorch_cuda_unquantized_1xA10",
    "pytorch_cuda_unquantized_1xT4",
    "pytorch_cuda_bnb_1xA100",
    "pytorch_cuda_gptq_1xA100",
    "pytorch_cuda_awq_1xA10",
    "pytorch_cuda_bnb_1xT4",
    "pytorch_cuda_torchao_1xA10",
    "pytorch_cpu_unquantized_32vCPU_C7i",
)


def _summary_path(config: str) -> Path:
    return HF_DIR / SAFE_NAME / config / "processed" / "summary.json"


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
        "scripts/ingest_hf_optimum_benchmark.py",
        "ingest_hf_optimum_benchmark_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_optimum_benchmark_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_optimum_benchmark_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


# ---------------------------------------------------------------------------
# 1. Module-level license constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inline inside ``_finalize_config``.

    The previous shape carried ``LICENSE = None`` only; the refactor
    splits the raw HF tag (``LICENSE_TAG``) from the human-curated
    provenance string (``LICENSE_SOURCE``) so a future HF tag change
    (e.g. the owner declaring ``license: apache-2.0``) is a one-line
    edit.
    """

    assert script_module.DATASET_ID == DATASET_ID
    assert script_module.LICENSE_TAG is None
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter has no `license:` field; "
        "recorded as unspecified"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"
    # The pre-existing skip-reason string preserved verbatim — pinned by
    # ``test_license_recorded_as_none`` in the ingest test file.
    assert script_module.COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON == (
        "license_unspecified_no_redistribution_promise"
    )


def test_script_back_compat_license_alias(script_module):
    """The pre-wiring shape exported ``LICENSE = None``; downstream
    callers may still reference it. The alias must remain and must
    track ``LICENSE_TAG``.
    """

    assert script_module.LICENSE == script_module.LICENSE_TAG
    assert script_module.LICENSE is None


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
        '"permissive_cdla_2":',
        '"permissive_mit":',
        '"permissive_cc_by_sa_4_0":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


def test_script_does_not_hardcode_unspecified_code_in_code(script_source: str):
    """The canonical status string ``"unspecified_no_committed_sample"``
    must not appear inline in the script's executable code — the gate
    produces it. Docstring mentions are allowed.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "unspecified_no_committed_sample"
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
            if doc and (
                "unspecified_no_committed_sample" in doc.value.value
            ):
                for (ln, v) in list(offending):
                    if (
                        ln == doc.lineno
                        and v == "unspecified_no_committed_sample"
                    ):
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes 'unspecified_no_committed_sample' at lines "
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


def test_evaluate_redistribution_default_denies_under_empty_ledger(
    script_module, policy_module,
):
    """The default ``license=None`` + empty ledger combo denies — the
    same v1 behaviour the script has always had for this dataset.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"
    assert decision.license_observed is None


def test_evaluate_redistribution_swap_to_apache_now_permits(
    script_module, policy_module,
):
    """Swap the optimum tag to ``"apache-2.0"`` under the same ledger
    → the gate flips to PERMIT. Proves the wiring actually consults
    the license tag — it is not hard-coded to deny.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag="apache-2.0",
    )
    assert decision.permitted is True
    assert decision.license_status == "permissive_apache_2_0"
    assert decision.reason_code == "permitted_declared_permissive_license"


def test_evaluate_redistribution_operator_grant_for_none_permits(
    script_module, policy_module, gate_module,
):
    """An in-memory operator grant for the optimum-benchmark dataset
    under the ``committed_normalized_sample`` scope flips the gate to
    PERMIT with ``permitted_operator_grant``. The grant is never
    written to disk — this only verifies that the script's
    ``evaluate_redistribution`` reads the ledger.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "in-memory test grant — never written to disk; verifies "
            "the seventh gate consumer actually reads the ledger"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True, (
        "wiring is broken: an operator grant for license=None must "
        "flip the gate's verdict to permit."
    )
    assert decision.reason_code == gate_module.REASON_PERMITTED_OPERATOR_GRANT
    assert decision.operator_grant_dataset_id == DATASET_ID


def test_evaluate_redistribution_declared_restrictive_denies_even_with_grant(
    script_module, policy_module, gate_module,
):
    """An operator grant cannot override a declared restrictive license.
    Pin this here so the seventh consumer cannot accidentally
    short-circuit the gate's safety invariant.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes="grants do NOT override declared restrictive licenses",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag="cc-by-nc-4.0",  # declared NON-permissive
    )
    assert decision.permitted is False
    assert decision.license_status == "declared_non_permissive"
    assert decision.reason_code == (
        gate_module.REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE
    )


# ---------------------------------------------------------------------------
# 4. Per-config summary.json carries the new gate-derived fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_summary_carries_redistribution_gate_metadata(config: str):
    s = json.loads(_summary_path(config).read_text())
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
        f"committed summary.json for {config} missing gate-derived "
        f"fields: {sorted(missing)!r}"
    )
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_summary_records_deny_verdict_for_license_none(config: str):
    """The HF card has no declared license, so the gate denies under
    the default empty ledger.

    Critically, the pre-wiring skip-reason string
    ``"license_unspecified_no_redistribution_promise"`` is preserved
    verbatim — ``test_license_recorded_as_none`` in
    ``tests/test_hf_optimum_benchmark_ingest.py`` pins it.
    """

    s = json.loads(_summary_path(config).read_text())
    assert s["license"] is None
    assert s["redistribution_gate_permitted"] is False
    assert s["redistribution_gate_reason_code"] == "no_grant_recorded"
    assert s["license_redistribution_status"] == (
        "unspecified_no_committed_sample"
    )
    assert s["license_redistribution_source"] == (
        "HF card frontmatter has no `license:` field; "
        "recorded as unspecified"
    )
    # v1 skip behaviour preserved byte-for-byte.
    assert s["committed_normalized_sample_rows"] == 0
    assert s["committed_normalized_sample_bytes"] == 0
    assert s["committed_normalized_sample_path"] is None
    assert s["committed_normalized_sample_sha256"] is None
    assert s["committed_normalized_sample_reason_skipped"] == (
        "license_unspecified_no_redistribution_promise"
    )


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_status_matches_gate_classification(config: str, gate_module):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed fixtures.
    """

    s = json.loads(_summary_path(config).read_text())
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
    """The audit summary stays at v2 (introduced by PR #158); the
    seventh consumer extends v2 coverage to optimum-benchmark rows.
    """

    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == "broadened_discovery_audit_summary_v2"


def test_audit_summary_top_level_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0


def test_audit_summary_optimum_rows_have_gate_fields():
    """The seventh consumer extends the v2 gate-field coverage to
    optimum-benchmark rows. Pre-existing latency-benchmark / agent-
    llm-traces / energy-consumption rows keep their gate fields too.
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
        assert entry["license"] is None
        assert entry["redistribution_gate_permitted"] is False
        assert entry["redistribution_gate_reason_code"] == "no_grant_recorded"
        assert entry["license_redistribution_status"] == (
            "unspecified_no_committed_sample"
        )
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None
    assert seen >= 9, (
        f"expected ≥9 optimum-benchmark ingested rows (the 9 working "
        f"configs plus optionally the openvino failure record), got {seen}"
    )


def test_audit_summary_preserves_other_scripts_rows():
    """The seventh consumer must NOT silently rewrite or drop rows
    owned by the other gate-wired ingest scripts (odyn / memoriant /
    intellistream / agent-llm-traces / energy / h200). The merge is
    keyed on ``(dataset_id, config_name)``.
    """

    a = json.loads(AUDIT_PATH.read_text())
    foreign_dids = {
        "odyn-network/odyn-benchmarks",
        "memoriant/dgx-spark-kv-cache-benchmark",
        "intellistream/vllm-hust-benchmark-results",
    }
    foreign = [
        e for e in a["ingested"] if e["dataset_id"] in foreign_dids
    ]
    assert foreign, "expected foreign rows from earlier gate consumers"
    for e in foreign:
        # Every foreign row owned by a gate-wired script must already
        # carry the gate fields the sixth consumer (PR #158) attached.
        for key in (
            "license_redistribution_status",
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
        ):
            assert key in e, (
                f"foreign row {e['dataset_id']}/{e.get('config_name')} "
                f"lost {key!r} during seventh-consumer merge"
            )


# ---------------------------------------------------------------------------
# 6. Function signatures accept ledger + gate_decision as keyword args
# ---------------------------------------------------------------------------


def test_ingest_one_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module._ingest_one)
    assert "ledger" in sig.parameters
    p = sig.parameters["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_write_audit_summary_accepts_ledger_keyword_arg(script_module):
    """``_write_audit_summary`` must accept ``ledger`` keyword-only so
    ``main()`` can load it once and pass it through to both the
    ingest functions and the audit writer (single source of truth for
    ``redistribution_gate_policy_default`` /
    ``redistribution_gate_policy_grant_count``).
    """

    sig = inspect.signature(script_module._write_audit_summary)
    assert "ledger" in sig.parameters
    p = sig.parameters["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
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


def test_finalize_config_accepts_gate_decision(script_module):
    """``_finalize_config`` must accept a ``gate_decision`` keyword arg
    so the per-config body cannot silently re-implement permit/deny
    logic.
    """

    sig = inspect.signature(script_module._finalize_config)
    assert "gate_decision" in sig.parameters, (
        "_finalize_config missing keyword arg 'gate_decision'"
    )


def test_finalize_config_does_not_take_commit_normalized_flag(script_module):
    """The pre-wiring shape never carried ``commit_normalized: bool``
    in this script (the deny verdict was hard-coded). Confirm the
    refactor does not introduce one — the gate's verdict alone now
    drives the commit decision.
    """

    sig = inspect.signature(script_module._finalize_config)
    assert "commit_normalized" not in sig.parameters, (
        "_finalize_config carries a regression-risk commit_normalized "
        "flag; the gate's verdict is the only commit decision"
    )


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


# ---------------------------------------------------------------------------
# 8. The committed fixtures are byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_fixture_sha256_matches_summary(config: str):
    """Wiring the gate must not change the on-disk fixture bytes — the
    sha256 the summary records must match what's on disk.
    """

    import hashlib
    s = json.loads(_summary_path(config).read_text())
    p = REPO_ROOT / s["fixture_sample_path"]
    assert p.exists()
    h = hashlib.sha256()
    h.update(p.read_bytes())
    assert h.hexdigest() == s["sample_sha256"], (
        f"fixture for {config} bytes have drifted from recorded sha256"
    )
    assert p.stat().st_size == s["fixture_sample_bytes"]


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_no_committed_normalized_sample_path_on_disk(config: str):
    """The gate denies for license=None, so no committed normalised
    sample file may exist on disk for any optimum config.
    """

    proc_dir = HF_DIR / SAFE_NAME / config / "processed"
    forbidden = proc_dir / "committed_normalized_sample.jsonl"
    assert not forbidden.exists(), (
        f"unexpected committed normalised sample at {forbidden} — "
        f"the gate denies redistribution for license=None"
    )
