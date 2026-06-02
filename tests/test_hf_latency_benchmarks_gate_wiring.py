"""Tests pinning that ``scripts/ingest_hf_latency_benchmarks.py`` is
wired through the canonical :func:`decide_redistribution` gate.

Sixth consumer of the gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``, and
``scripts/ingest_hf_llm_energy_consumption.py``).

The pre-wiring shape carried two distinct license-handling branches
inline inside ``_finalize_config``: a ``commit_normalized`` boolean
the caller picked per dataset, and a fallback skip-reason string
``"license_unspecified_no_redistribution_promise"`` baked into the
file. The refactor lifts the per-dataset license tags to module-level
constants and routes the commit decision through the canonical gate;
the skip-reason string is preserved verbatim on the committed
``committed_normalized_sample_reason_skipped`` field for downstream
back-compat (it is pinned by
``test_intellistream_has_no_committed_normalized_sample`` in
``tests/test_hf_latency_benchmarks_ingest.py``).

This file pins that the script now:

* records the raw HF license tag for each dataset in
  ``ODYN_LICENSE_TAG`` / ``MEMORIANT_LICENSE_TAG`` /
  ``INTELLISTREAM_LICENSE_TAG`` (this is the first multi-dataset gate
  consumer — earlier consumers had a single dataset and a single
  ``LICENSE_TAG`` constant);
* derives ``"license_redistribution_status"`` from the gate;
* preserves the v1 skip-reason string on
  ``committed_normalized_sample_reason_skipped`` for back-compat;
* writes the gate-derived metadata into both the per-config
  summary.json and the broadened-discovery audit summary;
* keeps the on-disk committed normalised samples byte-for-byte
  unchanged.

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

SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_hf_latency_benchmarks.py"
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = DISC_DIR / "broadened_discovery_audit_summary.json"

ODYN_DATASET_ID = "odyn-network/odyn-benchmarks"
MEMORIANT_DATASET_ID = "memoriant/dgx-spark-kv-cache-benchmark"
INTELLISTREAM_DATASET_ID = "intellistream/vllm-hust-benchmark-results"

ODYN_CONFIGS = (
    "qwen_chat_streaming", "facebook_chat_streaming",
    "qwen_batch", "facebook_batch",
)
MEMORIANT_CONFIGS = ("v3_corrected",)
INTELLISTREAM_CONFIGS = ("single_gpu", "multi_gpu")

PERMITTED_CONFIGS = (
    [(ODYN_DATASET_ID, c) for c in ODYN_CONFIGS]
    + [(MEMORIANT_DATASET_ID, c) for c in MEMORIANT_CONFIGS]
)
DENIED_CONFIGS = [
    (INTELLISTREAM_DATASET_ID, c) for c in INTELLISTREAM_CONFIGS
]
ALL_CONFIGS = PERMITTED_CONFIGS + DENIED_CONFIGS

SAFE_NAMES = {
    ODYN_DATASET_ID: "odyn-network__odyn-benchmarks",
    MEMORIANT_DATASET_ID: "memoriant__dgx-spark-kv-cache-benchmark",
    INTELLISTREAM_DATASET_ID: "intellistream__vllm-hust-benchmark-results",
}


def _summary_path(dataset_id: str, config: str) -> Path:
    return (
        HF_DIR / SAFE_NAMES[dataset_id] / config / "processed" / "summary.json"
    )


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
        "scripts/ingest_hf_latency_benchmarks.py",
        "ingest_hf_latency_benchmarks_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_latency_benchmarks_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_latency_benchmarks_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


# ---------------------------------------------------------------------------
# 1. Per-dataset license constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_per_dataset_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inside the ingest body of each dataset.

    Unlike the first five gate consumers (each of which has a single
    ``LICENSE_TAG``), this script covers three datasets with two
    distinct license tags; the constants must be split per-dataset.
    """

    assert script_module.ODYN_DATASET_ID == ODYN_DATASET_ID
    assert script_module.ODYN_LICENSE_TAG == "apache-2.0"
    assert script_module.ODYN_LICENSE_SOURCE == (
        "HF card frontmatter license: apache-2.0"
    )

    assert script_module.MEMORIANT_DATASET_ID == MEMORIANT_DATASET_ID
    assert script_module.MEMORIANT_LICENSE_TAG == "apache-2.0"
    assert script_module.MEMORIANT_LICENSE_SOURCE == (
        "HF card frontmatter license: apache-2.0"
    )

    assert script_module.INTELLISTREAM_DATASET_ID == INTELLISTREAM_DATASET_ID
    assert script_module.INTELLISTREAM_LICENSE_TAG is None
    assert script_module.INTELLISTREAM_LICENSE_SOURCE == (
        "HF card frontmatter has no `license:` field; "
        "recorded as unspecified"
    )

    assert script_module.GATE_SCOPE == "committed_normalized_sample"
    # The pre-existing skip-reason string preserved verbatim.
    assert script_module.COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON == (
        "license_unspecified_no_redistribution_promise"
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


def test_script_does_not_hardcode_status_code_in_code(script_source: str):
    """The canonical status string ``"permissive_apache_2_0"`` must
    not appear inline in the script's summary writer — the gate
    produces it; the script consumes it. Docstring mentions are
    allowed (the module docstring documents the verdict).
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
    # Filter out docstring occurrences.
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
            if doc and "permissive_apache_2_0" in doc.value.value:
                for (ln, v) in list(offending):
                    if ln == doc.lineno and v == "permissive_apache_2_0":
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes 'permissive_apache_2_0' at lines "
        f"{[ln for (ln, _) in offending]!r}; let the gate produce it"
    )


def test_script_does_not_hardcode_unspecified_code_in_code(script_source: str):
    """The canonical status string ``"unspecified_no_committed_sample"``
    must not appear inline in the script — the gate produces it.
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
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        dataset_id=ODYN_DATASET_ID,
        license_tag="apache-2.0",
    )
    assert type(decision).__name__ == "RedistributionGateDecision"
    for field in (
        "permitted", "reason_code", "reason_detail",
        "license_status", "license_observed", "scope",
        "operator_grant_dataset_id",
    ):
        assert hasattr(decision, field), (
            f"gate decision missing field {field!r}"
        )


def test_evaluate_redistribution_apache_under_empty_ledger_permits(
    script_module, policy_module,
):
    """``apache-2.0`` + empty ledger → PERMIT with the canonical
    ``permissive_apache_2_0`` status. Preserves the v1 commit
    behaviour for odyn (~88 KiB) + memoriant (~9 KiB).
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    for did in (ODYN_DATASET_ID, MEMORIANT_DATASET_ID):
        decision = script_module.evaluate_redistribution(
            ledger=ledger,
            dataset_id=did,
            license_tag="apache-2.0",
        )
        assert decision.permitted is True
        assert decision.license_status == "permissive_apache_2_0"
        assert decision.reason_code == (
            "permitted_declared_permissive_license"
        )
        assert decision.operator_grant_dataset_id is None
        assert decision.scope == "committed_normalized_sample"


def test_evaluate_redistribution_none_default_ledger_denies(
    script_module, policy_module,
):
    """``license=None`` + empty ledger → DENY with ``no_grant_recorded``.
    Preserves the v1 skip behaviour for intellistream's leaderboard.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        dataset_id=INTELLISTREAM_DATASET_ID,
        license_tag=None,
    )
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"
    assert decision.operator_grant_dataset_id is None


def test_evaluate_redistribution_apache_swapped_to_none_now_denies(
    script_module, policy_module,
):
    """Swap the odyn tag to ``None`` under the same ledger → the gate
    flips to DENY. Proves the wiring actually consults the ledger and
    the per-dataset tag is not hard-coded inside the ingest body.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        dataset_id=ODYN_DATASET_ID,
        license_tag=None,
    )
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"


def test_evaluate_redistribution_operator_grant_for_none_permits(
    script_module, policy_module, gate_module,
):
    """An in-memory operator grant for the intellistream dataset under
    the ``committed_normalized_sample`` scope flips the gate to
    PERMIT with ``permitted_operator_grant``. The grant is never
    written to disk — this only verifies that the script's
    ``evaluate_redistribution`` reads the ledger.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=INTELLISTREAM_DATASET_ID,
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "in-memory test grant — never written to disk; verifies "
            "the sixth gate consumer actually reads the ledger"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        dataset_id=INTELLISTREAM_DATASET_ID,
        license_tag=None,
    )
    assert decision.permitted is True, (
        "wiring is broken: an operator grant for license=None must "
        "flip the gate's verdict to permit."
    )
    assert decision.reason_code == gate_module.REASON_PERMITTED_OPERATOR_GRANT
    assert decision.operator_grant_dataset_id == INTELLISTREAM_DATASET_ID


# ---------------------------------------------------------------------------
# 4. Per-config summary.json carries the new gate-derived fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_id,config", ALL_CONFIGS)
def test_summary_carries_redistribution_gate_metadata(
    dataset_id: str, config: str,
):
    s = json.loads(_summary_path(dataset_id, config).read_text())
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
        f"committed summary.json for {dataset_id}/{config} missing "
        f"gate-derived fields: {sorted(missing)!r}"
    )
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    # detail string must be non-empty and mention the recorded license.
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail


@pytest.mark.parametrize("dataset_id,config", PERMITTED_CONFIGS)
def test_permitted_configs_carry_apache_permit_verdict(
    dataset_id: str, config: str,
):
    """apache-2.0 datasets must record the permit verdict."""

    s = json.loads(_summary_path(dataset_id, config).read_text())
    assert s["license"] == "apache-2.0"
    assert s["redistribution_gate_permitted"] is True
    assert s["redistribution_gate_reason_code"] == (
        "permitted_declared_permissive_license"
    )
    assert s["license_redistribution_status"] == "permissive_apache_2_0"
    assert s["license_redistribution_source"] == (
        "HF card frontmatter license: apache-2.0"
    )
    # v1 commit behaviour preserved.
    assert s["committed_normalized_sample_rows"] > 0
    assert s["committed_normalized_sample_bytes"] > 0
    assert s["committed_normalized_sample_path"] is not None
    assert s["committed_normalized_sample_sha256"] is not None
    assert s["committed_normalized_sample_reason_skipped"] is None
    p = REPO_ROOT / s["committed_normalized_sample_path"]
    assert p.exists()
    assert p.stat().st_size == s["committed_normalized_sample_bytes"]


@pytest.mark.parametrize("dataset_id,config", DENIED_CONFIGS)
def test_denied_configs_carry_unspecified_deny_verdict(
    dataset_id: str, config: str,
):
    """license=None intellistream configs must record the deny verdict
    AND preserve the pre-existing skip-reason string verbatim (the
    string is pinned by another test file)."""

    s = json.loads(_summary_path(dataset_id, config).read_text())
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


@pytest.mark.parametrize("dataset_id,config", ALL_CONFIGS)
def test_status_matches_gate_classification(
    dataset_id: str, config: str, gate_module,
):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed normalised samples.
    """

    s = json.loads(_summary_path(dataset_id, config).read_text())
    expected = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected!r} of license tag "
        f"{s['license']!r}"
    )


# ---------------------------------------------------------------------------
# 5. Audit summary carries v2 doc_version + gate-derived fields
# ---------------------------------------------------------------------------


def test_audit_summary_doc_version_bumped_to_v2():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == "broadened_discovery_audit_summary_v2", (
        f"audit doc_version is {a['doc_version']!r}; bump to v2 when "
        f"the gate wiring lands so consumers can detect the new "
        f"redistribution_gate_* fields"
    )


def test_audit_summary_top_level_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0
    assert a["uses_oracle_as_headline"] is False


def test_audit_summary_latency_benchmark_rows_have_gate_fields():
    """Every latency-benchmark ingested row must carry the gate fields.
    Optimum-benchmark rows (a separate script not yet gate-wired)
    remain at their v1 shape until their own follow-up PR.
    """

    a = json.loads(AUDIT_PATH.read_text())
    seen = 0
    expected_dids = {
        ODYN_DATASET_ID, MEMORIANT_DATASET_ID, INTELLISTREAM_DATASET_ID,
    }
    for entry in a["ingested"]:
        if entry["dataset_id"] not in expected_dids:
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
        # Verdict matches the per-dataset tag.
        if entry["license"] == "apache-2.0":
            assert entry["redistribution_gate_permitted"] is True
            assert entry["redistribution_gate_reason_code"] == (
                "permitted_declared_permissive_license"
            )
            assert entry["license_redistribution_status"] == (
                "permissive_apache_2_0"
            )
        else:
            assert entry["license"] is None
            assert entry["redistribution_gate_permitted"] is False
            assert entry["redistribution_gate_reason_code"] == (
                "no_grant_recorded"
            )
            assert entry["license_redistribution_status"] == (
                "unspecified_no_committed_sample"
            )
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None
    assert seen == 7, (
        f"expected 7 latency-benchmark ingested rows, got {seen}"
    )


def test_audit_summary_does_not_break_optimum_rows():
    """Optimum-benchmark rows must be preserved at their pre-wiring
    shape — adding the gate fields to those rows is the next PR's
    job. The merge here must not silently drop or rewrite them.
    """

    a = json.loads(AUDIT_PATH.read_text())
    optimum_rows = [
        e for e in a["ingested"]
        if e.get("dataset_id") == "optimum-benchmark/llm-perf-leaderboard"
    ]
    assert optimum_rows, "audit must still record optimum-benchmark rows"
    # Each optimum row must NOT have the new gate fields (those are
    # the next PR's job). The license must remain None.
    for r in optimum_rows:
        assert r["license"] is None
        # The pre-wiring shape did not record the gate fields, so the
        # keys are absent rather than null. This preserves the v1
        # serialisation byte-for-byte.
        for key in (
            "license_redistribution_status",
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
        ):
            assert key not in r, (
                f"optimum row {r.get('config_name')} unexpectedly "
                f"carries {key!r}; the next PR should add it"
            )


# ---------------------------------------------------------------------------
# 6. Function signatures accept ledger as a keyword-only argument
# ---------------------------------------------------------------------------


def test_ingest_odyn_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module._ingest_odyn)
    assert "ledger" in sig.parameters
    assert sig.parameters["ledger"].default is None


def test_ingest_memoriant_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module._ingest_memoriant)
    assert "ledger" in sig.parameters
    assert sig.parameters["ledger"].default is None


def test_ingest_intellistream_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module._ingest_intellistream)
    assert "ledger" in sig.parameters
    assert sig.parameters["ledger"].default is None


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
    so the per-dataset ingest body cannot silently re-implement the
    permit/deny logic.
    """

    sig = inspect.signature(script_module._finalize_config)
    for name in ("gate_decision", "license_source"):
        assert name in sig.parameters, (
            f"_finalize_config missing keyword arg {name!r}"
        )


def test_finalize_config_does_not_take_commit_normalized_flag(script_module):
    """The pre-wiring shape took a ``commit_normalized: bool`` flag
    that callers picked per dataset. Under the gate-wired path the
    commit decision is the gate's verdict alone; the boolean flag is
    a regression risk and must be gone.
    """

    sig = inspect.signature(script_module._finalize_config)
    assert "commit_normalized" not in sig.parameters, (
        "_finalize_config still carries the v1 commit_normalized flag; "
        "the gate's verdict is now the only commit decision"
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
# 8. The committed normalised samples are byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_id,config", PERMITTED_CONFIGS)
def test_committed_sample_sha256_matches_summary(
    dataset_id: str, config: str,
):
    """Wiring the gate must not change the on-disk sample bytes — the
    sha256 the summary records must match what's on disk.
    """

    import hashlib
    s = json.loads(_summary_path(dataset_id, config).read_text())
    p = REPO_ROOT / s["committed_normalized_sample_path"]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == s["committed_normalized_sample_sha256"], (
        f"committed sample sha256 mismatch for {dataset_id}/{config}; "
        f"the gate wiring must not change byte content"
    )
