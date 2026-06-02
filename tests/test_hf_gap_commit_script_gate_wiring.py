"""Tests pinning that ``scripts/commit_hf_gap_normalized_samples.py``
is wired through the canonical :func:`decide_redistribution` gate.

The previous PR (#152) added the gate module + first audit consumer.
The pre-existing
``test_classify_license_agrees_with_commit_script_targets`` test
already pins that the gate's verdicts match the four already-committed
normalised samples' licenses. This file pins the *wiring* side: the
script must actually call the gate, must derive the
``license_redistribution_status`` from the gate (not from a duplicated
hard-coded string), and must produce identical commit decisions under
the committed default policy as the pre-existing v1 rollup did.

This is audit-only — every test reads committed artifacts or runs
pure-Python decision functions. No HF API, no HF_TOKEN read, no data
download.

Inventory
---------

1. Script TARGETS schema: each entry has ``dataset_id``,
   ``config_name``, ``license_tag``, ``license_source`` and NOTHING
   that pre-classifies the redistribution decision (no
   ``license_redistribution_status`` field, no ``commit_sample`` flag).
2. Script imports the gate module (no duplicated license classifier).
3. ``evaluate_target`` (pure function) returns the gate's decision for
   each TARGETS entry under the default-empty ledger.
4. Under the default policy, the four permissive-licensed TARGETS are
   PERMITTED with the canonical ``permissive_*`` status, and prefixbench
   is DENIED with ``no_grant_recorded``.
5. The committed rollup (``telemetry_gap_normalized_sample_commit_summary.json``)
   carries the new gate-derived metadata fields:
   ``redistribution_gate_reason_code``, ``redistribution_gate_permitted``,
   ``redistribution_gate_policy_default``, ``redistribution_gate_scope``.
6. Per-dataset committed summary.json files carry
   ``redistribution_gate_reason_code`` =
   ``permitted_declared_permissive_license`` (matches the gate output).
7. Backwards compatibility: every committed dataset's
   ``license_redistribution_status`` matches the value the gate
   classifies its ``license_tag`` into. The skipped (prefixbench)
   dataset stays at ``unspecified_no_committed_sample``.
8. If an operator grant is added for prefixbench (in-memory only,
   nothing written to disk), the gate would PERMIT the commit. This
   pins that the wiring actually consults the ledger — not that today's
   default policy denies it.
9. No HF_TOKEN literal in the refactored script.
10. The script does NOT import ``huggingface_hub`` or any HF SDK.
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

SCRIPT_PATH = REPO_ROOT / "scripts" / "commit_hf_gap_normalized_samples.py"
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
ROLLUP_PATH = DISC_DIR / "telemetry_gap_normalized_sample_commit_summary.json"


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
        "scripts/commit_hf_gap_normalized_samples.py",
        "commit_hf_gap_normalized_samples_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


def _summary_path(dataset_id: str, config: str) -> Path:
    return (
        HF_DIR
        / dataset_id.replace("/", "__")
        / config
        / "processed"
        / "summary.json"
    )


# ---------------------------------------------------------------------------
# 1. TARGETS schema — no duplicated classifier in the script
# ---------------------------------------------------------------------------


def test_targets_schema_is_minimal_no_pre_classified_status(script_module):
    """Each entry holds only the raw license tag + provenance.

    The previous shape had ``license_redistribution_status`` and
    ``commit_sample`` keys — those have been removed because they
    duplicated the gate's logic. If they come back, the gate is no
    longer the single source of truth.
    """

    required_keys = {"dataset_id", "config_name", "license_tag", "license_source"}
    forbidden_keys = {"license_redistribution_status", "commit_sample"}
    for t in script_module.TARGETS:
        assert isinstance(t, dict), f"TARGETS entry not a dict: {t!r}"
        missing = required_keys - t.keys()
        assert not missing, (
            f"TARGETS entry for {t.get('dataset_id')!r} missing required "
            f"keys {sorted(missing)!r}"
        )
        present_forbidden = forbidden_keys & t.keys()
        assert not present_forbidden, (
            f"TARGETS entry for {t.get('dataset_id')!r} carries forbidden "
            f"pre-classified keys {sorted(present_forbidden)!r} — the gate "
            f"must classify, not the TARGETS table"
        )


def test_targets_count_matches_known_datasets(script_module):
    """The TARGETS table covers exactly the 5 datasets from PR #129."""

    expected = {
        ("lzzmm/BurstGPT", "burstgpt_1_full"),
        (
            "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
            "instance_events_shard0",
        ),
        ("sammshen/lmcache-agentic-traces", "train_shard4"),
        ("semianalysisai/cc-traces-weka-no-subagents-051226", "traces_head"),
        ("jaytonde05/prefixbench", "prefixbench_all"),
    }
    actual = {(t["dataset_id"], t["config_name"]) for t in script_module.TARGETS}
    assert actual == expected, (
        f"TARGETS coverage drifted: expected {sorted(expected)!r}, "
        f"got {sorted(actual)!r}"
    )


# ---------------------------------------------------------------------------
# 2. Script imports the gate (no duplicated classifier)
# ---------------------------------------------------------------------------


def test_script_imports_decide_redistribution(script_source: str):
    """The script must call the canonical gate.

    A future maintainer who silently re-introduces a hard-coded
    classifier inside the script must trip this test.
    """

    assert "from aurelius.ingestion.redistribution_gate import" in script_source, (
        "script must import decide_redistribution from the canonical gate"
    )
    assert "decide_redistribution" in script_source, (
        "script must reference decide_redistribution in its code path"
    )
    assert "OperatorPolicyLedger" in script_source, (
        "script must load the operator policy ledger"
    )


def test_script_does_not_redeclare_permissive_set(script_source: str):
    """Confidence rail: no second copy of the closed permissive allow-list.

    If the script declares its own ``PERMISSIVE_*`` dict or list, the
    gate has competition. Use the gate.
    """

    forbidden = [
        "PERMISSIVE_LICENSE_TAGS = {",
        "PERMISSIVE_LICENSES = {",
        '"permissive_apache_2_0":',
        '"permissive_mit":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


# ---------------------------------------------------------------------------
# 3. evaluate_target — pure function returns gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_target_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_target`` is exposed so tests can drive the gate path
    without touching the filesystem. It must return the gate's
    ``RedistributionGateDecision`` dataclass.

    We compare by class name + the closed attribute set instead of
    ``isinstance``, because the script imports the gate via the
    canonical ``aurelius.ingestion.redistribution_gate`` path while
    the test fixture loads the gate module under a different name —
    two module objects → two class objects → ``isinstance`` is False
    even though the instance is the genuine gate output.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    target = script_module.TARGETS[0]
    decision = script_module.evaluate_target(target, ledger=ledger)
    assert type(decision).__name__ == "RedistributionGateDecision", (
        f"evaluate_target must return a RedistributionGateDecision, "
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


# ---------------------------------------------------------------------------
# 4. Default-policy outcomes — must match the v1 hardcoded verdicts
# ---------------------------------------------------------------------------


EXPECTED_VERDICTS = {
    ("lzzmm/BurstGPT", "burstgpt_1_full"): {
        "permitted": True,
        "license_status": "permissive_cc_by_4_0",
        "reason_code": "permitted_declared_permissive_license",
    },
    (
        "lsliwko/google-cluster-data-2019-sorted-by-timestamp",
        "instance_events_shard0",
    ): {
        "permitted": True,
        "license_status": "permissive_cc_by_4_0",
        "reason_code": "permitted_declared_permissive_license",
    },
    ("sammshen/lmcache-agentic-traces", "train_shard4"): {
        "permitted": True,
        "license_status": "permissive_mit",
        "reason_code": "permitted_declared_permissive_license",
    },
    (
        "semianalysisai/cc-traces-weka-no-subagents-051226",
        "traces_head",
    ): {
        "permitted": True,
        "license_status": "permissive_apache_2_0",
        "reason_code": "permitted_declared_permissive_license",
    },
    ("jaytonde05/prefixbench", "prefixbench_all"): {
        "permitted": False,
        "license_status": "unspecified_no_committed_sample",
        "reason_code": "no_grant_recorded",
    },
}


@pytest.mark.parametrize(
    "key,expected",
    sorted(EXPECTED_VERDICTS.items()),
    ids=lambda kv: kv[0] if isinstance(kv, tuple) else None,
)
def test_default_policy_per_target_decision(
    script_module, policy_module, key, expected,
):
    """Pin every TARGETS verdict under the default deny_all/zero-grants
    policy. If any of these flip, the four already-committed normalised
    samples' redistribution decision has changed silently and the wiring
    needs explicit operator review.
    """

    dataset_id, config_name = key
    target = next(
        t
        for t in script_module.TARGETS
        if t["dataset_id"] == dataset_id and t["config_name"] == config_name
    )
    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_target(target, ledger=ledger)
    assert decision.permitted is expected["permitted"], (
        f"{dataset_id}@{config_name}: permitted "
        f"{decision.permitted!r} != expected {expected['permitted']!r}"
    )
    assert decision.license_status == expected["license_status"], (
        f"{dataset_id}@{config_name}: license_status "
        f"{decision.license_status!r} != {expected['license_status']!r}"
    )
    assert decision.reason_code == expected["reason_code"], (
        f"{dataset_id}@{config_name}: reason_code "
        f"{decision.reason_code!r} != {expected['reason_code']!r}"
    )


# ---------------------------------------------------------------------------
# 5. Rollup carries the new gate-derived fields
# ---------------------------------------------------------------------------


def test_rollup_doc_version_bumped_to_v2():
    rollup = json.loads(ROLLUP_PATH.read_text())
    assert rollup["doc_version"] == "telemetry_gap_normalized_sample_commit_v2", (
        f"rollup doc_version is {rollup['doc_version']!r}; bump to v2 "
        f"when the gate wiring lands so consumers can detect the new "
        f"redistribution_gate_* fields"
    )


def test_rollup_records_redistribution_gate_metadata():
    rollup = json.loads(ROLLUP_PATH.read_text())
    assert rollup["redistribution_gate_scope"] == "committed_normalized_sample"
    assert rollup["redistribution_gate_policy_default"] == "deny_all"
    assert rollup["redistribution_gate_policy_grant_count"] == 0


def test_rollup_per_dataset_has_gate_reason_code():
    rollup = json.loads(ROLLUP_PATH.read_text())
    for entry in rollup["datasets"]:
        # ``audit_status: summary_missing`` rows do not include the gate
        # fields because materialize returns early. Skip them.
        if entry.get("audit_status") == "summary_missing":
            continue
        assert "redistribution_gate_reason_code" in entry, (
            f"rollup entry {entry.get('dataset_id')!r} missing "
            f"redistribution_gate_reason_code"
        )
        assert "redistribution_gate_permitted" in entry, (
            f"rollup entry {entry.get('dataset_id')!r} missing "
            f"redistribution_gate_permitted"
        )


def test_rollup_decisions_match_expected_default_policy_outcomes():
    rollup = json.loads(ROLLUP_PATH.read_text())
    by_key = {(e["dataset_id"], e["config_name"]): e for e in rollup["datasets"]}
    for key, expected in EXPECTED_VERDICTS.items():
        entry = by_key.get(key)
        assert entry is not None, f"rollup missing entry for {key!r}"
        assert entry["redistribution_gate_permitted"] is expected["permitted"], (
            f"{key}: rollup permitted "
            f"{entry['redistribution_gate_permitted']!r} != "
            f"{expected['permitted']!r}"
        )
        assert entry["redistribution_gate_reason_code"] == expected["reason_code"], (
            f"{key}: rollup reason_code "
            f"{entry['redistribution_gate_reason_code']!r} != "
            f"{expected['reason_code']!r}"
        )


# ---------------------------------------------------------------------------
# 6. Per-dataset summary.json files carry the gate fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_id,config",
    [k for k, v in EXPECTED_VERDICTS.items() if v["permitted"]],
)
def test_permitted_summary_records_gate_reason_code(
    dataset_id: str, config: str,
):
    s = json.loads(_summary_path(dataset_id, config).read_text())
    assert s.get("redistribution_gate_reason_code") == (
        "permitted_declared_permissive_license"
    ), (
        f"{dataset_id}@{config}: summary must record the gate reason "
        f"code; saw {s.get('redistribution_gate_reason_code')!r}"
    )
    assert s.get("redistribution_gate_permitted") is True


def test_denied_summary_records_no_grant_recorded():
    s = json.loads(
        _summary_path("jaytonde05/prefixbench", "prefixbench_all").read_text()
    )
    assert s["redistribution_gate_reason_code"] == "no_grant_recorded"
    assert s["redistribution_gate_permitted"] is False
    # The license_redistribution_status label that downstream tests
    # already check for must NOT have changed.
    assert s["license_redistribution_status"] == "unspecified_no_committed_sample"


# ---------------------------------------------------------------------------
# 7. Backwards compatibility — gate verdict matches committed status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_id,config",
    [k for k, v in EXPECTED_VERDICTS.items() if v["permitted"]],
)
def test_committed_summary_status_matches_gate_classification(
    dataset_id: str, config: str, script_module, gate_module, policy_module,
):
    """The status label in summary.json equals what the gate
    classifies the TARGETS-recorded license_tag into. Pinning this
    means the rollout has zero behavioural drift on the four already-
    committed normalised samples.
    """

    target = next(
        t
        for t in script_module.TARGETS
        if t["dataset_id"] == dataset_id and t["config_name"] == config
    )
    expected_status = gate_module.classify_license(target["license_tag"])
    s = json.loads(_summary_path(dataset_id, config).read_text())
    assert s["license_redistribution_status"] == expected_status, (
        f"{dataset_id}@{config}: summary status "
        f"{s['license_redistribution_status']!r} != gate classification "
        f"{expected_status!r} of license_tag "
        f"{target['license_tag']!r}"
    )


# ---------------------------------------------------------------------------
# 8. Operator-grant smoke test — wiring actually consults the ledger
# ---------------------------------------------------------------------------


def test_operator_grant_for_prefixbench_would_permit(
    script_module, policy_module, gate_module,
):
    """If an operator records a grant for prefixbench, the gate flips.

    This is in-memory only — nothing is written to
    operator_redistribution_policy.json, nothing is committed. The
    test pins that the wiring is real: the script's evaluate_target
    must consult the ledger, not a hard-coded ``unspecified → deny``
    table.
    """

    grant = policy_module.OperatorGrant(
        dataset_id="jaytonde05/prefixbench",
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "in-memory test grant — never written to disk; verifies "
            "the gate consumer actually reads the ledger"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    target = next(
        t
        for t in script_module.TARGETS
        if t["dataset_id"] == "jaytonde05/prefixbench"
    )
    decision = script_module.evaluate_target(target, ledger=ledger)
    assert decision.permitted is True, (
        "wiring is broken: an operator grant for license=None must flip "
        "the gate's verdict to permit. The script's evaluate_target is "
        "not consulting the ledger."
    )
    assert decision.reason_code == gate_module.REASON_PERMITTED_OPERATOR_GRANT
    assert decision.operator_grant_dataset_id == "jaytonde05/prefixbench"


# ---------------------------------------------------------------------------
# 9. Safety — no HF_TOKEN literal, no HF SDK import
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    """No ``hf_`` token literal, no ``HF_TOKEN`` value embedded."""

    suspicious = re.findall(r"\bhf_[A-Za-z0-9_]{20,}\b", script_source)
    assert not suspicious, (
        f"script contains a literal that looks like an HF token: {suspicious!r}"
    )
    # The string ``HF_TOKEN`` may appear in commentary; the value
    # must not.
    bad_assignment = re.search(
        r'HF_TOKEN\s*=\s*["\']hf_', script_source,
    )
    assert bad_assignment is None, (
        "HF_TOKEN appears to be assigned a literal hf_ value"
    )


def test_script_does_not_import_huggingface_hub(script_source: str):
    """Sample-commit decisions must not call the HF API.

    Re-running the script reads gitignored ``analysis_sample.jsonl``
    from disk (left over from a previous ingest run) and copies it
    to the committed path. There is no HF API call. Pinning this
    prevents a future "convenience" refactor that introduces a
    network dependency.
    """

    tree = ast.parse(script_source)
    bad_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("huggingface_hub", "datasets")):
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(
                ("huggingface_hub", "datasets")
            ):
                bad_imports.append(node.module)
    assert not bad_imports, (
        f"script imports HF SDK modules: {bad_imports!r}. The script "
        f"must never call the HF API."
    )


# ---------------------------------------------------------------------------
# 10. Materialise integration — idempotent rerun does not corrupt summaries
# ---------------------------------------------------------------------------


def test_idempotent_rerun_keeps_committed_samples_intact(
    script_module, policy_module, tmp_path,
):
    """Calling ``materialize`` for a permissive target whose committed
    sample already exists on disk and whose ``analysis_sample.jsonl``
    is missing (the gitignored sibling) must idempotently reuse the
    existing committed sample.

    Pinning this means re-running the script after a clean checkout
    does NOT zero out the committed_normalized_sample_* metadata.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    target = next(
        t
        for t in script_module.TARGETS
        if t["dataset_id"] == "lzzmm/BurstGPT"
    )
    # Read the on-disk summary before, so we can confirm bytes/sha match.
    before = json.loads(
        _summary_path(target["dataset_id"], target["config_name"]).read_text()
    )
    expected_bytes = int(before["committed_normalized_sample_bytes"])
    expected_sha = before["committed_normalized_sample_sha256"]
    result, new_total = script_module.materialize(
        target, total_committed_so_far=0, ledger=ledger,
    )
    assert result["commit_decision"] == "COMMITTED", (
        f"idempotent rerun did not COMMIT (got {result['commit_decision']!r}); "
        f"existing committed_normalized_sample.jsonl was not reused"
    )
    assert result["committed_bytes"] == expected_bytes
    assert result["committed_sha256"] == expected_sha
    assert new_total == expected_bytes
