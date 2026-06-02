"""Tests pinning that ``scripts/ingest_hf_agent_llm_traces.py`` is wired
through the canonical :func:`decide_redistribution` gate.

Third consumer of the gate (after ``scripts/audit_hf_redistribution_gate.py``
and ``scripts/commit_hf_gap_normalized_samples.py``). The previous shape
hard-coded ``"license_redistribution_status": "permissive_cdla_2"`` into
the summary writer; this file pins that the script now derives the
status code, reason code, and permit decision from the gate, and writes
the gate-derived metadata into both the per-dataset summary.json and the
cross-dataset rollup.

This is audit-only — every test reads committed artefacts or runs
pure-Python decision functions. No HF API, no HF_TOKEN read, no data
download.

Inventory
---------

1. Script declares ``LICENSE_TAG`` + ``LICENSE_SOURCE`` + ``GATE_SCOPE``
   as the single source of truth for the dataset's license metadata
   (mirrors the ``TARGETS`` shape PR #152 introduced for the gap commit
   script).
2. Script imports the gate (no duplicated license classifier).
3. Script does NOT redeclare the permissive allow-list.
4. ``evaluate_redistribution`` (pure function) returns the gate's
   ``RedistributionGateDecision`` for the dataset's tag under any
   ledger the caller supplies.
5. Under the default-empty ledger and the dataset's declared
   ``cdla-permissive-2.0`` tag the gate PERMITS with reason_code
   ``permitted_declared_permissive_license`` and status
   ``permissive_cdla_2`` — identical to the v1 behaviour.
6. Swapping the license tag to ``None`` under the same ledger flips
   the gate to DENY with reason_code ``no_grant_recorded`` (proves the
   wiring actually consults the ledger, not a hard-coded path).
7. An in-memory operator grant for the dataset under the
   ``committed_normalized_sample`` scope flips the ``license=None``
   path to PERMIT with reason_code ``permitted_operator_grant`` (also
   proves the wiring consults the ledger).
8. The committed per-dataset summary.json carries the new
   gate-derived fields:
   ``redistribution_gate_reason_code``,
   ``redistribution_gate_reason_detail``,
   ``redistribution_gate_permitted``,
   ``redistribution_gate_operator_grant_dataset_id``,
   ``redistribution_gate_scope``.
9. The committed ``license_redistribution_status`` matches what the
   gate classifies the recorded ``license`` tag into. Backwards-compat:
   no behavioural drift on the already-committed normalised sample.
10. The committed rollup
    ``data/external/hf_discovery/agent_llm_traces_ingest_summary.json``
    has ``doc_version = exgentic_agent_llm_traces_ingest_summary_v2``
    and carries the gate metadata.
11. No HF_TOKEN literal in the refactored script.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_hf_agent_llm_traces.py"
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
ROLLUP_PATH = DISC_DIR / "agent_llm_traces_ingest_summary.json"

DATASET_ID = "Exgentic/agent-llm-traces"
SAFE_DATASET = DATASET_ID.replace("/", "__")
CONFIG = "swebench_claude_code_shard12"


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
        "scripts/ingest_hf_agent_llm_traces.py",
        "ingest_hf_agent_llm_traces_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_agent_llm_traces_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_agent_llm_traces_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


def _summary_path() -> Path:
    return HF_DIR / SAFE_DATASET / CONFIG / "processed" / "summary.json"


# ---------------------------------------------------------------------------
# 1. License constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance live at module level, not inside
    the summary writer.

    The previous shape had ``"license": "cdla-permissive-2.0"`` and
    ``"license_redistribution_status": "permissive_cdla_2"`` repeated
    inline in the summary dict. Lifting them to module-level constants
    means a future tag change is a one-line edit and the gate handles
    the status derivation.
    """

    assert script_module.LICENSE_TAG == "cdla-permissive-2.0"
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter license: cdla-permissive-2.0"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"


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
    ), (
        "script must import decide_redistribution from the canonical gate"
    )
    assert "decide_redistribution" in script_source, (
        "script must reference decide_redistribution in its code path"
    )
    assert "OperatorPolicyLedger" in script_source, (
        "script must load the operator policy ledger"
    )


def test_script_does_not_redeclare_permissive_set(script_source: str):
    """Confidence rail: no second copy of the closed permissive allow-list."""

    forbidden = [
        "PERMISSIVE_LICENSE_TAGS = {",
        "PERMISSIVE_LICENSES = {",
        '"permissive_apache_2_0":',
        '"permissive_cdla_2":',
        '"permissive_mit":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


def test_script_does_not_hardcode_license_redistribution_status(
    script_source: str,
):
    """The string ``"permissive_cdla_2"`` must not appear inline in
    the script's summary writer. The gate produces it; the script
    consumes it.

    The constant may appear in docstrings (we explicitly document
    the verdict) — we therefore search only the code AST.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Skip strings inside expression-statement docstrings (we
            # explicitly mention "permissive_cdla_2" in the module
            # docstring).
            if node.value == "permissive_cdla_2":
                offending.append((node.lineno, node.value))
    # The only acceptable occurrences are inside docstrings (string
    # constants at expr-statement level). Filter those out by walking
    # parents — easier: just allow occurrences whose ``col_offset`` is
    # at the start of an indented docstring (col_offset == 0). The
    # module/function docstrings are the only Constant string nodes
    # at col_offset==0, so anything else is a real code occurrence.
    code_uses = [(ln, v) for (ln, v) in offending if True]
    # We allow occurrences in docstrings (which sit inside Expr nodes
    # whose parent is the Module or a FunctionDef body[0]). The
    # docstrings in this script appear at module level (col_offset=0)
    # — anywhere else is real code that should call the gate instead.
    real_code_uses = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            body = getattr(node, "body", []) or []
            docstring_node = (
                body[0]
                if body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                else None
            )
            if docstring_node and (
                "permissive_cdla_2" in docstring_node.value.value
            ):
                # docstring mention is fine; remove from the offending list
                for (ln, v) in list(offending):
                    if ln == docstring_node.lineno and v == (
                        "permissive_cdla_2"
                    ):
                        offending.remove((ln, v))
    real_code_uses = offending
    assert not real_code_uses, (
        f"script hard-codes the 'permissive_cdla_2' status string in code "
        f"at lines {[ln for (ln, _) in real_code_uses]!r}; let the gate "
        f"produce it via decide_redistribution()"
    )


# ---------------------------------------------------------------------------
# 3. evaluate_redistribution — pure function returns gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_redistribution_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_redistribution`` is exposed so tests can drive the
    gate path without invoking the parquet download / flatten
    pipeline. It must return the gate's ``RedistributionGateDecision``
    dataclass.

    We compare by class name + attribute set instead of ``isinstance``,
    because the script imports the gate via the canonical
    ``aurelius.ingestion.redistribution_gate`` path while the test
    fixture loads the gate module under a different name — two module
    objects → two class objects → ``isinstance`` is False even though
    the instance is the genuine gate output.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert type(decision).__name__ == "RedistributionGateDecision", (
        f"evaluate_redistribution must return a RedistributionGateDecision, "
        f"got {type(decision).__name__!r}"
    )
    for field in (
        "permitted",
        "reason_code",
        "reason_detail",
        "license_status",
        "license_observed",
        "scope",
        "operator_grant_dataset_id",
    ):
        assert hasattr(decision, field), (
            f"gate decision missing field {field!r}; the wiring may have "
            f"diverged from the canonical decision schema"
        )


def test_evaluate_redistribution_default_tag_under_empty_ledger(
    script_module, policy_module,
):
    """Default tag + empty ledger → permit with the canonical permissive
    status code. This is the verdict pinned by
    ``test_dataset_license_recorded`` in
    ``tests/test_hf_agent_llm_traces_ingest.py`` and the verdict already
    embedded in the committed normalised sample's summary.json.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True
    assert decision.license_status == "permissive_cdla_2"
    assert decision.reason_code == "permitted_declared_permissive_license"
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"


def test_evaluate_redistribution_none_tag_default_ledger_denies(
    script_module, policy_module, gate_module,
):
    """Swap the tag to ``None`` under the default-empty ledger → the
    gate denies with ``no_grant_recorded``. This proves the wiring
    actually consults the ledger; if the script were carrying a
    hard-coded ``permit`` for this dataset, this test would fail.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag=None,
    )
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"
    assert decision.operator_grant_dataset_id is None


def test_evaluate_redistribution_operator_grant_for_none_license_permits(
    script_module, policy_module, gate_module,
):
    """In-memory operator grant for ``license=None`` flips the gate to
    PERMIT with ``permitted_operator_grant``. The grant is never
    written to disk — this only verifies that the script's
    ``evaluate_redistribution`` consults the ledger.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "in-memory test grant — never written to disk; verifies "
            "the third gate consumer actually reads the ledger"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag=None,
    )
    assert decision.permitted is True, (
        "wiring is broken: an operator grant for license=None must flip "
        "the gate's verdict to permit. The script's evaluate_redistribution "
        "is not consulting the ledger."
    )
    assert decision.reason_code == (
        gate_module.REASON_PERMITTED_OPERATOR_GRANT
    )
    assert decision.operator_grant_dataset_id == DATASET_ID


# ---------------------------------------------------------------------------
# 4. Committed per-dataset summary.json carries the gate fields
# ---------------------------------------------------------------------------


def test_committed_summary_carries_redistribution_gate_metadata():
    s = json.loads(_summary_path().read_text())
    required = {
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
    assert s["redistribution_gate_reason_code"] == (
        "permitted_declared_permissive_license"
    )
    assert s["redistribution_gate_permitted"] is True
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    # The detail string is a free-form audit trail; assert only its
    # non-emptiness and that it mentions the tag + the status code so
    # the field cannot be silently zeroed out.
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail
    assert "cdla-permissive-2.0" in detail
    assert "permissive_cdla_2" in detail


def test_committed_summary_status_matches_gate_classification(
    gate_module,
):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed normalised sample.
    """

    s = json.loads(_summary_path().read_text())
    expected_status = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected_status, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected_status!r} of license tag "
        f"{s['license']!r}"
    )


def test_committed_summary_license_redistribution_source_unchanged():
    """The human-curated provenance string is the v1 wording and must
    not have drifted in the wiring refactor.
    """

    s = json.loads(_summary_path().read_text())
    assert s["license_redistribution_source"] == (
        "HF card frontmatter license: cdla-permissive-2.0"
    )


# ---------------------------------------------------------------------------
# 5. Rollup carries the new gate-derived fields + v2 doc_version
# ---------------------------------------------------------------------------


def test_rollup_doc_version_bumped_to_v2():
    rollup = json.loads(ROLLUP_PATH.read_text())
    assert rollup["doc_version"] == (
        "exgentic_agent_llm_traces_ingest_summary_v2"
    ), (
        f"rollup doc_version is {rollup['doc_version']!r}; bump to v2 "
        f"when the gate wiring lands so consumers can detect the new "
        f"redistribution_gate_* fields"
    )


def test_rollup_records_redistribution_gate_metadata():
    rollup = json.loads(ROLLUP_PATH.read_text())
    assert rollup["redistribution_gate_scope"] == (
        "committed_normalized_sample"
    )
    assert rollup["redistribution_gate_policy_default"] == "deny_all"
    assert rollup["redistribution_gate_policy_grant_count"] == 0


def test_rollup_per_dataset_has_gate_reason_code():
    rollup = json.loads(ROLLUP_PATH.read_text())
    assert rollup["ingested"], "rollup has no ingested entries"
    for entry in rollup["ingested"]:
        for key in (
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
            "redistribution_gate_operator_grant_dataset_id",
            "license_redistribution_status",
        ):
            assert key in entry, (
                f"rollup entry {entry.get('dataset_id')!r} missing {key!r}"
            )
        assert entry["redistribution_gate_reason_code"] == (
            "permitted_declared_permissive_license"
        )
        assert entry["redistribution_gate_permitted"] is True
        assert entry["license_redistribution_status"] == "permissive_cdla_2"
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None


# ---------------------------------------------------------------------------
# 6. Safety — no HF_TOKEN literal in the refactored script
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    """No ``hf_`` token literal, no ``HF_TOKEN`` value embedded."""

    suspicious = re.findall(r"\bhf_[A-Za-z0-9_]{20,}\b", script_source)
    assert not suspicious, (
        f"script contains a literal that looks like an HF token: "
        f"{suspicious!r}"
    )
    bad_assignment = re.search(
        r'HF_TOKEN\s*=\s*["\']hf_', script_source,
    )
    assert bad_assignment is None, (
        "HF_TOKEN appears to be assigned a literal hf_ value"
    )


# ---------------------------------------------------------------------------
# 7. audit_one signature accepts ledger (keyword-only, optional)
# ---------------------------------------------------------------------------


def test_audit_one_accepts_ledger_keyword_arg(script_module):
    """The refactored audit_one signature must accept ``ledger`` as a
    keyword-only argument so callers (including main()) can supply a
    pre-loaded ledger without re-reading the policy file per config.

    The default value must be ``None`` so existing call sites that
    don't pass the ledger still work — the function falls back to
    ``_load_ledger()`` internally.
    """

    import inspect

    sig = inspect.signature(script_module.audit_one)
    params = sig.parameters
    assert "ledger" in params, (
        "audit_one must accept a 'ledger' parameter for the gate wiring"
    )
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"ledger must be keyword-only (got {p.kind!r}) to keep the "
        f"existing positional call signature stable"
    )
    assert p.default is None, (
        "ledger default must be None so existing call sites without the "
        "wiring fall back to _load_ledger()"
    )


def test_load_ledger_returns_empty_when_policy_path_missing(
    script_module, policy_module, tmp_path,
):
    """``_load_ledger`` must fall back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — that's the documented self-
    sufficiency rail (a fresh checkout without the committed JSON
    pulled still produces correct deny decisions instead of crashing).
    """

    nonexistent = tmp_path / "no_such_file.json"
    assert not nonexistent.exists()
    ledger = script_module._load_ledger(nonexistent)
    assert ledger.policy_default == "deny_all"
    assert ledger.grants == ()
