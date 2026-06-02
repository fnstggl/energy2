"""Tests pinning that ``scripts/ingest_hf_llm_energy_consumption.py`` is
wired through the canonical :func:`decide_redistribution` gate.

Fifth consumer of the gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``, and
``scripts/ingest_hf_h200_quantization.py``). The previous shape
hard-coded a free-form prose string into
``"license_redistribution_status"``
(``"cc-by-sa-4.0 — attribution + share-alike required when …"``).
This file pins that the script now:

* records the raw HF tag in ``LICENSE_TAG`` (``"cc-by-sa-4.0"``);
* derives ``"license_redistribution_status"`` from the gate (the
  canonical code ``"permissive_cc_by_sa_4_0"``);
* preserves the free-form attribution + share-alike + arxiv-citation
  prose verbatim in a new additive
  ``"license_redistribution_attribution_notes"`` field; and
* writes the gate-derived metadata into both the per-config
  summary.json and the round-4 audit summary.

The pre-existing test ``test_summary_records_redistribution_attribution``
in ``tests/test_hf_llm_energy_consumption_ingest.py`` continues to
pin the prose verbatim — only the field name changed.

This is audit-only — every test reads committed artefacts or runs
pure-Python decision functions. No HF API, no HF_TOKEN read, no data
download.

Inventory
---------

1. Script declares ``LICENSE_TAG`` + ``LICENSE_SOURCE`` + ``GATE_SCOPE``
   + ``LICENSE_REDISTRIBUTION_ATTRIBUTION_NOTES`` as the single source
   of truth for the dataset's license metadata.
2. Script imports the canonical gate (no duplicated license classifier).
3. Script does NOT redeclare the permissive allow-list.
4. ``evaluate_redistribution`` (pure function) returns the gate's
   ``RedistributionGateDecision`` for the dataset's tag under any
   ledger the caller supplies.
5. Under the default-empty ledger and the dataset's declared
   ``cc-by-sa-4.0`` tag the gate PERMITS with reason_code
   ``permitted_declared_permissive_license`` and status
   ``permissive_cc_by_sa_4_0`` — identical to the v1 behaviour
   (which committed ~400 KB of normalised samples across 4 configs).
6. Swapping the license tag to ``None`` under the same ledger flips
   the gate to DENY with reason_code ``no_grant_recorded`` (proves
   the wiring actually consults the ledger).
7. An in-memory operator grant for the dataset under the
   ``committed_normalized_sample`` scope flips the ``license=None``
   path to PERMIT with reason_code ``permitted_operator_grant``
   (also proves the wiring consults the ledger).
8. The committed per-config summary.json (all 4 configs) carries the
   new gate-derived fields:
   ``redistribution_gate_reason_code``,
   ``redistribution_gate_reason_detail``,
   ``redistribution_gate_permitted``,
   ``redistribution_gate_operator_grant_dataset_id``,
   ``redistribution_gate_scope``,
   ``license_redistribution_status``,
   ``license_redistribution_source``,
   ``license_redistribution_attribution_notes``.
9. The committed ``license_redistribution_status`` matches what the
   gate classifies the recorded ``license`` tag into. Backwards-compat:
   the bounded normalised samples themselves are unchanged byte-for-byte
   on disk (the new fields are additive metadata on the summary).
10. The committed round-4 audit summary
    ``data/external/hf_discovery/round4_broadened_discovery_audit_summary.json``
    has ``doc_version =
    round4_broadened_discovery_audit_summary_v2`` and carries the gate
    metadata both top-level and per-dataset.
11. No HF_TOKEN literal in the refactored script.
12. ``ingest`` and ``ingest_config`` and ``write_round4_audit_summary``
    accept the ledger as a keyword-only ``Optional`` argument so the
    round-4 ``main()`` can load it once and pass it through.
13. ``_load_ledger`` falls back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — fresh-checkout self-sufficiency.
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

SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "ingest_hf_llm_energy_consumption.py"
)
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = (
    DISC_DIR / "round4_broadened_discovery_audit_summary.json"
)

DATASET_ID = "ejhusom/llm-inference-energy-consumption"
SAFE_DATASET = "ejhusom__llm-inference-energy-consumption"
CONFIGS = [
    "alpaca_gemma_7b_laptop2",
    "alpaca_gemma_7b_workstation",
    "codefeedback_codellama_7b_workstation",
    "codefeedback_codellama_70b_workstation",
]


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
        "scripts/ingest_hf_llm_energy_consumption.py",
        "ingest_hf_llm_energy_consumption_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_llm_energy_consumption_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_llm_energy_consumption_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


def _summary_path(config: str) -> Path:
    return HF_DIR / SAFE_DATASET / config / "processed" / "summary.json"


# ---------------------------------------------------------------------------
# 1. License constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inside the summary writer.

    The previous shape had ``LICENSE = "cc-by-sa-4.0"`` and a
    free-form prose string baked into the summary writer's
    ``"license_redistribution_status"`` slot. Lifting the canonical
    tag + provenance + scope + attribution-notes prose to module-level
    constants means a future tag change (e.g. the dataset owner
    bumping to ``"cc-by-sa-5.0"``) is a one-line edit and the gate
    handles the status derivation.
    """

    assert script_module.LICENSE_TAG == "cc-by-sa-4.0"
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter license: cc-by-sa-4.0"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"
    # The attribution + share-alike + arxiv-citation prose constant.
    notes = script_module.LICENSE_REDISTRIBUTION_ATTRIBUTION_NOTES
    assert isinstance(notes, str) and notes
    assert "cc-by-sa-4.0" in notes.lower()
    assert "attribution" in notes.lower()
    assert "share-alike" in notes.lower()
    assert "2407.16893" in notes


def test_script_license_back_compat_alias(script_module):
    """The back-compat ``LICENSE`` alias still resolves to ``LICENSE_TAG``.

    A small number of out-of-tree audits read the script's
    module-level ``LICENSE`` attribute. The refactor renames the
    canonical constant to ``LICENSE_TAG`` (mirrors agent-llm-traces +
    h200) but keeps ``LICENSE`` as an alias.
    """

    assert script_module.LICENSE == script_module.LICENSE_TAG
    assert script_module.LICENSE == "cc-by-sa-4.0"


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
    """Confidence rail: no second copy of the closed permissive allow-list."""

    forbidden = [
        "PERMISSIVE_LICENSE_TAGS = {",
        "PERMISSIVE_LICENSES = {",
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


def test_script_does_not_hardcode_license_redistribution_status_code(
    script_source: str,
):
    """The canonical status string ``"permissive_cc_by_sa_4_0"`` must
    not appear inline in the script's summary writer — the gate
    produces it; the script consumes it.

    Docstring mentions are allowed (the module docstring documents
    the verdict). We therefore filter out occurrences that sit
    inside Module / FunctionDef / AsyncFunctionDef docstrings.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "permissive_cc_by_sa_4_0"
        ):
            offending.append((node.lineno, node.value))
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
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
                "permissive_cc_by_sa_4_0" in docstring_node.value.value
            ):
                for (ln, v) in list(offending):
                    if ln == docstring_node.lineno and v == (
                        "permissive_cc_by_sa_4_0"
                    ):
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes the 'permissive_cc_by_sa_4_0' status string "
        f"in code at lines {[ln for (ln, _) in offending]!r}; let the "
        f"gate produce it via decide_redistribution()"
    )


# ---------------------------------------------------------------------------
# 3. evaluate_redistribution — pure function returns gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_redistribution_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_redistribution`` is exposed so tests can drive the
    gate path without invoking the CSV download / normalisation
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
    assert type(decision).__name__ == "RedistributionGateDecision"
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
            f"gate decision missing field {field!r}"
        )


def test_evaluate_redistribution_default_tag_under_empty_ledger_permits(
    script_module, policy_module,
):
    """Default tag (``"cc-by-sa-4.0"``) + empty ledger → PERMIT with
    the canonical permissive status code ``permissive_cc_by_sa_4_0``.

    This is the verdict that preserves the v1 behaviour: the
    already-committed ~400 KB of normalised samples across 4 configs
    must remain permitted under the gate. If the gate denied this
    tag, the wiring would be regressing redistribution behaviour.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True
    assert decision.license_status == "permissive_cc_by_sa_4_0"
    assert decision.reason_code == (
        "permitted_declared_permissive_license"
    )
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"


def test_evaluate_redistribution_none_tag_default_ledger_denies(
    script_module, policy_module,
):
    """Swap the tag to ``None`` under the default-empty ledger → the
    gate denies with ``no_grant_recorded``. This proves the wiring
    actually consults the ledger; if the script were carrying a
    hard-coded ``permit`` for this dataset (the v1 behaviour), this
    test would fail.
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
    """In-memory operator grant for ``license=None`` flips the gate
    to PERMIT with ``permitted_operator_grant``. The grant is never
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
            "the fifth gate consumer actually reads the ledger"
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
        "the gate's verdict to permit. The script's "
        "evaluate_redistribution is not consulting the ledger."
    )
    assert decision.reason_code == (
        gate_module.REASON_PERMITTED_OPERATOR_GRANT
    )
    assert decision.operator_grant_dataset_id == DATASET_ID


# ---------------------------------------------------------------------------
# 4. Committed per-config summary.json carries the gate fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_summary_carries_redistribution_gate_metadata(
    config: str,
):
    s = json.loads(_summary_path(config).read_text())
    required = {
        "license_redistribution_status",
        "license_redistribution_source",
        "license_redistribution_attribution_notes",
        "redistribution_gate_reason_code",
        "redistribution_gate_reason_detail",
        "redistribution_gate_permitted",
        "redistribution_gate_operator_grant_dataset_id",
        "redistribution_gate_scope",
    }
    missing = required - s.keys()
    assert not missing, (
        f"committed summary.json for {config!r} missing gate-derived "
        f"fields: {sorted(missing)!r}"
    )
    assert s["redistribution_gate_reason_code"] == (
        "permitted_declared_permissive_license"
    )
    assert s["redistribution_gate_permitted"] is True
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["license_redistribution_status"] == (
        "permissive_cc_by_sa_4_0"
    )
    # The detail string is a free-form audit trail; assert only its
    # non-emptiness and that it mentions the tag + the status code so
    # the field cannot be silently zeroed.
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail
    assert "cc-by-sa-4.0" in detail
    assert "permissive_cc_by_sa_4_0" in detail


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_summary_status_matches_gate_classification(
    config: str, gate_module,
):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed normalised samples.
    """

    s = json.loads(_summary_path(config).read_text())
    expected_status = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected_status, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected_status!r} of license tag "
        f"{s['license']!r}"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_summary_attribution_notes_preserved_verbatim(
    config: str,
):
    """The free-form attribution + share-alike + arxiv-citation prose
    moved from ``license_redistribution_status`` (v1) to
    ``license_redistribution_attribution_notes`` (v2). The prose
    itself must be preserved verbatim so downstream audits and the
    CC-BY-SA attribution chain don't regress.
    """

    s = json.loads(_summary_path(config).read_text())
    notes = s["license_redistribution_attribution_notes"]
    assert notes == (
        "cc-by-sa-4.0 — attribution + share-alike required when "
        "redistributing committed normalised sample. "
        "Citation: Husom et al. 2024, 'The Price of Prompting: "
        "Profiling Energy Use in Large Language Models Inference', "
        "arxiv:2407.16893."
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_summary_license_redistribution_source_recorded(
    config: str,
):
    """The human-curated provenance string lives in the committed
    summary so a future maintainer can audit *why* the license tag is
    recorded as ``"cc-by-sa-4.0"`` (vs ``None`` or a different
    permissive variant).
    """

    s = json.loads(_summary_path(config).read_text())
    assert s["license_redistribution_source"] == (
        "HF card frontmatter license: cc-by-sa-4.0"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_summary_preserves_v1_commit_behaviour(config: str):
    """The pre-wiring behaviour was to commit a bounded normalised
    sample (cc-by-sa-4.0 permits redistribution). The gate-wired
    summary must preserve this: ``committed_normalized_sample_*``
    fields stay non-zero on every config.
    """

    s = json.loads(_summary_path(config).read_text())
    assert s["committed_normalized_sample_rows"] > 0
    assert s["committed_normalized_sample_bytes"] > 0
    assert s["committed_normalized_sample_path"] is not None
    assert s["committed_normalized_sample_sha256"] is not None
    assert s["committed_normalized_sample_reason_skipped"] is None
    # And the path actually points to a committed file on disk.
    p = REPO_ROOT / s["committed_normalized_sample_path"]
    assert p.exists(), (
        f"committed normalised sample for {config!r} missing on disk: {p}"
    )
    # Size on disk must match the summary.
    assert p.stat().st_size == s["committed_normalized_sample_bytes"]


# ---------------------------------------------------------------------------
# 5. Round-4 audit summary carries the new gate-derived fields + v2 doc
# ---------------------------------------------------------------------------


def test_audit_summary_doc_version_bumped_to_v2():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == (
        "round4_broadened_discovery_audit_summary_v2"
    ), (
        f"audit summary doc_version is {a['doc_version']!r}; bump to "
        f"v2 when the gate wiring lands so consumers can detect the "
        f"new redistribution_gate_* fields"
    )


def test_audit_summary_records_redistribution_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0


def test_audit_summary_per_dataset_has_gate_reason_code():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["ingested"], "audit summary has no ingested entries"
    for entry in a["ingested"]:
        for key in (
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
            "redistribution_gate_operator_grant_dataset_id",
            "license_redistribution_status",
        ):
            assert key in entry, (
                f"audit entry {entry.get('dataset_id')!r}/"
                f"{entry.get('config_name')!r} missing {key!r}"
            )
        assert entry["redistribution_gate_reason_code"] == (
            "permitted_declared_permissive_license"
        )
        assert entry["redistribution_gate_permitted"] is True
        assert entry["license_redistribution_status"] == (
            "permissive_cc_by_sa_4_0"
        )
        assert entry[
            "redistribution_gate_operator_grant_dataset_id"
        ] is None


# ---------------------------------------------------------------------------
# 6. Safety — no HF_TOKEN literal in the refactored script
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    """No ``hf_`` token literal, no ``HF_TOKEN`` value embedded.

    Real HF tokens are mixed-case base62 strings; the script's logger
    name and module references are lowercase + digits + underscores
    only, so the suspicious-token pattern requires both an uppercase
    and a lowercase char in the suffix.
    """

    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", script_source)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
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
# 7. ingest / ingest_config / write_round4_audit_summary accept ledger
# ---------------------------------------------------------------------------


def test_ingest_accepts_ledger_keyword_arg(script_module):
    """The refactored ``ingest`` signature must accept ``ledger`` as a
    keyword-only argument so callers (including ``main()``) can
    supply a pre-loaded ledger without re-reading the policy file.
    """

    import inspect

    sig = inspect.signature(script_module.ingest)
    params = sig.parameters
    assert "ledger" in params
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_ingest_config_accepts_ledger_keyword_arg(script_module):
    """``ingest_config`` must accept ``ledger`` keyword-only so each
    per-config call sees the same ledger the run was launched with.
    """

    import inspect

    sig = inspect.signature(script_module.ingest_config)
    params = sig.parameters
    assert "ledger" in params
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_write_round4_audit_summary_accepts_ledger_keyword_arg(
    script_module,
):
    """``write_round4_audit_summary`` must accept ``ledger`` so the
    round-4 ``main()`` can load the ledger once and pass it to both
    ``ingest`` and the audit summary writer (single source of truth
    for ``redistribution_gate_policy_default`` /
    ``redistribution_gate_policy_grant_count``).
    """

    import inspect

    sig = inspect.signature(script_module.write_round4_audit_summary)
    params = sig.parameters
    assert "ledger" in params
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_load_ledger_returns_empty_when_policy_path_missing(
    script_module, policy_module, tmp_path,
):
    """``_load_ledger`` must fall back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — that's the documented
    self-sufficiency rail (a fresh checkout without the committed
    JSON pulled still produces correct decisions instead of crashing).
    """

    nonexistent = tmp_path / "no_such_file.json"
    assert not nonexistent.exists()
    ledger = script_module._load_ledger(nonexistent)
    assert ledger.policy_default == "deny_all"
    assert ledger.grants == ()
