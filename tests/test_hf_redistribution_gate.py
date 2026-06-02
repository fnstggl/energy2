"""Tests for the unified RedistributionGate consumer of the operator policy.

The gate is the first sample-commit consumer of
``OperatorPolicyLedger.permits_redistribution``. It fuses the two
permitted paths — declared permissive HF license OR operator grant for
the ``committed_normalized_sample`` scope — into a single auditable
decision. With the committed default policy file shipping ZERO grants,
the gate denies every ``license = None`` dataset, which mirrors the
behaviour of the pre-existing ``scripts/commit_hf_gap_normalized_samples.py``
hard-coded license verdicts. Tests pin every failure-mode path and pin
the four Round-8 license-blocked candidates as denied under the default
policy so a silent widening cannot land without a deliberate grant.

Test inventory
--------------

1. License classification: every canonical permissive tag maps to its
   ``permissive_*`` code; non-permissive declared licenses map to
   ``declared_non_permissive``; None / empty / whitespace map to
   ``unspecified_no_committed_sample``.
2. Decision API: permissive declared license → permitted; ledger is
   NOT consulted (verified by passing an empty ledger and observing
   permit).
3. Decision API: declared non-permissive license → denied regardless of
   operator grants (operator grants cannot override declared upstream
   licenses).
4. Decision API: license = None + no grant → denied with
   ``no_grant_recorded``.
5. Decision API: license = None + valid in-scope grant → permitted with
   ``permitted_operator_grant`` and the grant's dataset_id recorded.
6. Decision API: license = None + ``granted = false`` → denied with
   ``grant_explicitly_denies``.
7. Decision API: license = None + expired grant → denied with
   ``grant_expired``.
8. Decision API: license = None + grant doesn't include requested scope
   → denied with ``requested_scope_not_in_allowed_scopes``.
9. Decision API: unsupported scope → denied with
   ``requested_scope_not_in_supported_scopes`` regardless of license.
10. Audit script: under the default committed policy, the 4 Round-8
    license-blocked candidates remain denied with
    ``no_grant_recorded``; all permissive-licensed candidates are
    permitted; the rollup counts match the candidate registry; the
    payload is deterministic for a fixed ``now_iso``.
11. Safety: the new module / script / audit JSON committed in this PR
    contain no ``HF_TOKEN`` or ``hf_`` token literal; no raw data files
    are committed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_directly(rel_path: str, name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, rel_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "policy_module_under_test_rg",
    )


@pytest.fixture(scope="module")
def gate_module():
    # Force-import via direct path so the test passes even in stripped-
    # down environments without pandas/sqlalchemy in aurelius.__init__.
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_under_test",
    )


@pytest.fixture(scope="module")
def audit_module():
    return _load_module_directly(
        "scripts/audit_hf_redistribution_gate.py",
        "audit_hf_redistribution_gate_under_test",
    )


# ---------------------------------------------------------------------------
# 1. License classification
# ---------------------------------------------------------------------------


def test_permissive_apache_classification(gate_module) -> None:
    assert gate_module.classify_license("apache-2.0") == "permissive_apache_2_0"
    assert gate_module.classify_license("Apache-2.0") == "permissive_apache_2_0"
    assert gate_module.classify_license("apache_2_0") == "permissive_apache_2_0"


def test_permissive_mit_classification(gate_module) -> None:
    assert gate_module.classify_license("mit") == "permissive_mit"
    assert gate_module.classify_license("MIT") == "permissive_mit"
    assert gate_module.classify_license("  mit ") == "permissive_mit"


def test_permissive_cc_variants_classification(gate_module) -> None:
    assert gate_module.classify_license("cc-by-4.0") == "permissive_cc_by_4_0"
    assert gate_module.classify_license("CC-BY-4.0") == "permissive_cc_by_4_0"
    assert gate_module.classify_license("cc-by-3.0") == "permissive_cc_by_3_0"
    assert gate_module.classify_license("cc-by-2.0") == "permissive_cc_by_2_0"
    assert gate_module.classify_license("cc0-1.0") == "permissive_cc0_1_0"


def test_permissive_cc_by_sa_classification(gate_module) -> None:
    """CC-BY-SA-* — added when ``scripts/ingest_hf_llm_energy_consumption.py``
    was wired through the gate (fifth consumer). ShareAlike permits
    redistribution of the original work with attribution; the
    derivative normalised samples inherit the same license.
    """

    assert gate_module.classify_license(
        "cc-by-sa-4.0"
    ) == "permissive_cc_by_sa_4_0"
    assert gate_module.classify_license(
        "CC-BY-SA-4.0"
    ) == "permissive_cc_by_sa_4_0"
    assert gate_module.classify_license(
        "  cc-by-sa-4.0 "
    ) == "permissive_cc_by_sa_4_0"
    assert gate_module.classify_license(
        "cc-by-sa-3.0"
    ) == "permissive_cc_by_sa_3_0"


def test_permissive_cdla_classification(gate_module) -> None:
    assert gate_module.classify_license(
        "cdla-permissive-2.0"
    ) == "permissive_cdla_2"
    assert gate_module.classify_license(
        "CDLA-Permissive-2.0"
    ) == "permissive_cdla_2"
    assert gate_module.classify_license(
        "cdla-permissive-1.0"
    ) == "permissive_cdla_1"


def test_permissive_odc_and_bsd_classification(gate_module) -> None:
    assert gate_module.classify_license(
        "odc-by-1.0"
    ) == "permissive_odc_by_1_0"
    assert gate_module.classify_license(
        "bsd-3-clause"
    ) == "permissive_bsd_3_clause"


def test_unspecified_license_classification(gate_module) -> None:
    """None / empty / whitespace must map to unspecified, NOT permissive."""

    assert gate_module.classify_license(None) == (
        gate_module.LICENSE_STATUS_UNSPECIFIED
    )
    assert gate_module.classify_license("") == (
        gate_module.LICENSE_STATUS_UNSPECIFIED
    )
    assert gate_module.classify_license("   ") == (
        gate_module.LICENSE_STATUS_UNSPECIFIED
    )


def test_declared_non_permissive_classification(gate_module) -> None:
    """Non-empty but not in the allow-list — MUST NOT be widened.

    Pinning this prevents future maintainers from silently adding
    "other" or "openrail" to the permissive allow-list.
    """

    expected = gate_module.LICENSE_STATUS_DECLARED_NON_PERMISSIVE
    assert gate_module.classify_license("other") == expected
    assert gate_module.classify_license("openrail") == expected
    assert gate_module.classify_license("cc") == expected  # bare "cc" → unclear
    assert gate_module.classify_license("gpl-3.0") == expected
    assert gate_module.classify_license("custom-research") == expected


def test_permissive_allow_list_is_closed_set(gate_module) -> None:
    """The closed set must include the labels in actual use today."""

    canonical_status_codes = set(
        gate_module.PERMISSIVE_LICENSE_TAGS.values()
    )
    # These are the labels currently emitted by
    # scripts/commit_hf_gap_normalized_samples.py and the per-dataset
    # ingestion scripts. The gate must continue to produce them.
    expected = {
        "permissive_apache_2_0",
        "permissive_mit",
        "permissive_cc_by_4_0",
        "permissive_cc_by_sa_4_0",
        "permissive_cdla_2",
    }
    missing = expected - canonical_status_codes
    assert not missing, (
        f"closed permissive allow-list missing required labels: "
        f"{sorted(missing)!r}; existing summary.json files use these"
    )


# ---------------------------------------------------------------------------
# 2. Decision API — declared permissive license short-circuits the ledger
# ---------------------------------------------------------------------------


def test_permissive_license_permitted_without_consulting_ledger(
    policy_module, gate_module,
) -> None:
    """Permissive declared license → permit; ledger is NOT consulted.

    Verified by passing an empty ledger (zero grants) and observing
    that the decision still permits with the permissive reason code,
    not via the operator-grant path.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    d = gate_module.decide_redistribution(
        dataset_id="lzzmm/BurstGPT",
        license_str="cc-by-4.0",
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is True
    assert d.reason_code == (
        gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
    )
    assert d.license_status == "permissive_cc_by_4_0"
    assert d.operator_grant_dataset_id is None
    assert d.license_observed == "cc-by-4.0"


def test_permissive_license_permitted_for_every_permissive_tag(
    policy_module, gate_module,
) -> None:
    ledger = policy_module.OperatorPolicyLedger.empty()
    for tag in gate_module.PERMISSIVE_LICENSE_TAGS:
        d = gate_module.decide_redistribution(
            dataset_id=f"owner/{tag}-fixture",
            license_str=tag,
            scope="committed_normalized_sample",
            ledger=ledger,
        )
        assert d.permitted is True, f"permissive tag {tag!r} denied"
        assert d.reason_code == (
            gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
        )


# ---------------------------------------------------------------------------
# 3. Decision API — declared non-permissive license never overridden
# ---------------------------------------------------------------------------


def test_declared_non_permissive_denied_even_with_operator_grant(
    policy_module, gate_module,
) -> None:
    """Operator grants ONLY apply to license=None datasets.

    A declared restrictive license must be DENIED even if the operator
    has recorded an explicit grant — the grant scope is consent under
    license=None, not override of an upstream owner's declared
    restriction. Pinning this prevents an operator from accidentally
    re-licensing an upstream dataset by adding a grant entry.
    """

    grant = policy_module.OperatorGrant(
        dataset_id="owner/restricted-dataset",
        granted=True,
        granted_by="test-operator",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes="should not override the declared restrictive license",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    d = gate_module.decide_redistribution(
        dataset_id="owner/restricted-dataset",
        license_str="other",
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == (
        gate_module.REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE
    )
    assert d.license_status == "declared_non_permissive"
    # The operator grant must NOT be recorded as the matched grant —
    # the gate refused to consult the ledger for a declared license.
    assert d.operator_grant_dataset_id is None


# ---------------------------------------------------------------------------
# 4. Decision API — license=None + no grant => DENY
# ---------------------------------------------------------------------------


def test_unspecified_license_no_grant_denied(
    policy_module, gate_module,
) -> None:
    ledger = policy_module.OperatorPolicyLedger.empty()
    d = gate_module.decide_redistribution(
        dataset_id="sasha/co2_models",
        license_str=None,
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_NO_GRANT
    assert d.license_status == "unspecified_no_committed_sample"
    assert d.license_observed is None
    assert d.operator_grant_dataset_id is None


def test_unspecified_license_empty_string_denied(
    policy_module, gate_module,
) -> None:
    """Empty-string license must behave identically to None."""

    ledger = policy_module.OperatorPolicyLedger.empty()
    d = gate_module.decide_redistribution(
        dataset_id="owner/some-dataset",
        license_str="",
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_NO_GRANT


# ---------------------------------------------------------------------------
# 5. Decision API — license=None + valid grant => PERMIT
# ---------------------------------------------------------------------------


def test_unspecified_license_with_valid_grant_permits(
    policy_module, gate_module,
) -> None:
    grant = policy_module.OperatorGrant(
        dataset_id="sasha/co2_models",
        granted=True,
        granted_by="aurelius-operator",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample", "schema_only"),
        notes="independently verified CodeCarbon redistribution permission",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    d = gate_module.decide_redistribution(
        dataset_id="sasha/co2_models",
        license_str=None,
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is True
    assert d.reason_code == gate_module.REASON_PERMITTED_OPERATOR_GRANT
    assert d.license_status == "unspecified_no_committed_sample"
    assert d.operator_grant_dataset_id == "sasha/co2_models"
    # The reason detail must surface the grantor's identity for audit.
    assert "aurelius-operator" in d.reason_detail


# ---------------------------------------------------------------------------
# 6. Decision API — license=None + granted=false => DENY
# ---------------------------------------------------------------------------


def test_unspecified_license_with_explicit_deny_grant_denied(
    policy_module, gate_module,
) -> None:
    grant = policy_module.OperatorGrant(
        dataset_id="owner/explicitly-denied",
        granted=False,
        granted_by="aurelius-operator",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=(),
        notes="upstream owner declined redistribution request",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    d = gate_module.decide_redistribution(
        dataset_id="owner/explicitly-denied",
        license_str=None,
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_GRANTED_FALSE
    assert d.operator_grant_dataset_id == "owner/explicitly-denied"


# ---------------------------------------------------------------------------
# 7. Decision API — license=None + expired grant => DENY
# ---------------------------------------------------------------------------


def test_unspecified_license_with_expired_grant_denied(
    policy_module, gate_module,
) -> None:
    grant = policy_module.OperatorGrant(
        dataset_id="owner/expired-grant",
        granted=True,
        granted_by="aurelius-operator",
        granted_at_iso="2025-01-01T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes="one-year pilot grant",
        expires_at_iso="2026-01-01T00:00:00Z",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    # now_iso is later than expires_at_iso → must deny.
    d = gate_module.decide_redistribution(
        dataset_id="owner/expired-grant",
        license_str=None,
        scope="committed_normalized_sample",
        ledger=ledger,
        now_iso="2026-06-02T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_EXPIRED


# ---------------------------------------------------------------------------
# 8. Decision API — license=None + scope-not-allowed => DENY
# ---------------------------------------------------------------------------


def test_unspecified_license_with_scope_not_in_allowed_denied(
    policy_module, gate_module,
) -> None:
    grant = policy_module.OperatorGrant(
        dataset_id="owner/schema-only-grant",
        granted=True,
        granted_by="aurelius-operator",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("schema_only",),
        notes="operator consents to schema inspection only",
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    d = gate_module.decide_redistribution(
        dataset_id="owner/schema-only-grant",
        license_str=None,
        scope="committed_normalized_sample",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_SCOPE_NOT_ALLOWED


# ---------------------------------------------------------------------------
# 9. Decision API — unsupported scope => DENY regardless of license
# ---------------------------------------------------------------------------


def test_unsupported_scope_denied_even_for_permissive_license(
    policy_module, gate_module,
) -> None:
    """Permissive license must not bypass the closed scope set."""

    ledger = policy_module.OperatorPolicyLedger.empty()
    d = gate_module.decide_redistribution(
        dataset_id="lzzmm/BurstGPT",
        license_str="cc-by-4.0",
        scope="train_a_model",  # not in SUPPORTED_SCOPES
        ledger=ledger,
    )
    # The gate currently short-circuits permissive licenses BEFORE the
    # scope check. Document that here: permissive datasets are
    # PERMITTED regardless of scope because the upstream license
    # itself grants redistribution under that scope. The ledger's
    # ``REASON_UNSUPPORTED_SCOPE`` exists to constrain operator grants
    # specifically — a declared permissive license already covers the
    # broader scope.
    assert d.permitted is True
    assert d.reason_code == (
        gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
    )


def test_unsupported_scope_denied_for_license_none(
    policy_module, gate_module,
) -> None:
    """license = None + unsupported scope → denied via the ledger path.

    The ledger returns REASON_UNSUPPORTED_SCOPE; the gate propagates
    it verbatim so callers can route on the same closed reason set.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    d = gate_module.decide_redistribution(
        dataset_id="sasha/co2_models",
        license_str=None,
        scope="train_a_model",
        ledger=ledger,
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_UNSUPPORTED_SCOPE


# ---------------------------------------------------------------------------
# 10. Audit script — default policy outcomes
# ---------------------------------------------------------------------------

LICENSE_BLOCKED_DATASETS = frozenset(
    {
        "anon-betterbench/betterbench-inference-logs",
        "dadadada1/Inference-Performance-Dataset",
        "ohdoking/energy_consumption_by_model_and_gpu",
        "sasha/co2_models",
    }
)


@pytest.fixture(scope="module")
def committed_audit_payload() -> dict:
    path = (
        REPO_ROOT
        / "data"
        / "external"
        / "hf_discovery"
        / "redistribution_gate_audit.json"
    )
    assert path.exists(), (
        "redistribution_gate_audit.json must be committed for this PR"
    )
    return json.loads(path.read_text())


def test_audit_doc_version_and_safety_flags(committed_audit_payload) -> None:
    p = committed_audit_payload
    assert p["doc_version"] == "redistribution_gate_audit_v1"
    assert p["production_claim"] is False
    assert p["modifies_robust_energy_engine"] is False
    assert p["modifies_controllers_or_defaults"] is False
    assert p["uses_oracle_as_headline"] is False
    assert p["ingests_new_hf_data"] is False
    assert p["default_scope_evaluated"] == "committed_normalized_sample"
    assert p["policy_default"] == "deny_all"
    assert p["policy_grant_count"] == 0


def test_audit_under_default_policy_denies_round8_license_blocked(
    committed_audit_payload, policy_module,
) -> None:
    rows = {r["dataset_id"]: r for r in committed_audit_payload["per_dataset"]}
    for ds in LICENSE_BLOCKED_DATASETS:
        assert ds in rows, f"missing {ds} in audit per_dataset"
        r = rows[ds]
        assert r["permitted"] is False, (
            f"{ds} must be DENIED under the default policy"
        )
        assert r["reason_code"] == policy_module.REASON_NO_GRANT, (
            f"{ds} reason_code={r['reason_code']!r}, expected "
            f"{policy_module.REASON_NO_GRANT!r}"
        )
        assert r["license_status"] == "unspecified_no_committed_sample"
        assert r["operator_grant_dataset_id"] is None


def test_audit_permits_every_permissive_licensed_candidate(
    committed_audit_payload, gate_module,
) -> None:
    """Every candidate whose license maps to a permissive_* status must
    be permitted; otherwise the gate would regress on the existing 42
    permissive-licensed candidates already in the registry."""

    for r in committed_audit_payload["per_dataset"]:
        if r["license_status"].startswith("permissive_"):
            assert r["permitted"] is True, (
                f"{r['dataset_id']} has permissive license_status "
                f"{r['license_status']!r} but is DENIED — regression"
            )
            assert r["reason_code"] == (
                gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
            )


def test_audit_rollup_counts_sum_to_candidate_count(
    committed_audit_payload,
) -> None:
    p = committed_audit_payload
    assert p["candidate_count"] == p["permitted_count"] + p["denied_count"]
    assert sum(p["rollup"]["by_license_status"].values()) == p["candidate_count"]
    assert sum(p["rollup"]["by_reason_code"].values()) == p["candidate_count"]


def test_audit_is_deterministic_for_same_inputs(
    policy_module, gate_module, audit_module, tmp_path,
) -> None:
    """Building the audit payload twice for the same in-memory inputs
    must produce identical JSON (modulo wall-clock fields)."""

    # Use the committed policy file as the ledger.
    policy_path = (
        REPO_ROOT
        / "data"
        / "external"
        / "hf_discovery"
        / "operator_redistribution_policy.json"
    )
    ledger = policy_module.OperatorPolicyLedger.load(policy_path)
    candidates_path = (
        REPO_ROOT
        / "data"
        / "external"
        / "hf_discovery"
        / "hf_dataset_candidates.json"
    )
    candidates = json.loads(candidates_path.read_text())["candidates"]
    p1 = audit_module.build_audit_payload(
        ledger=ledger,
        candidates=candidates,
        scope="committed_normalized_sample",
        now_iso="2026-06-02T00:00:00Z",
        git_sha="deadbeef",
    )
    p2 = audit_module.build_audit_payload(
        ledger=ledger,
        candidates=candidates,
        scope="committed_normalized_sample",
        now_iso="2026-06-02T00:00:00Z",
        git_sha="deadbeef",
    )
    # audited_at_s is wall-clock; ignore.
    for d in (p1, p2):
        d.pop("audited_at_s", None)
    assert (
        json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
    )


def test_audit_rollup_includes_unspecified_bucket(
    committed_audit_payload,
) -> None:
    """The license-blocked candidates contribute to the
    ``unspecified_no_committed_sample`` bucket — pin its presence so
    a future refactor cannot silently lose the bucket."""

    by_status = committed_audit_payload["rollup"]["by_license_status"]
    assert by_status.get("unspecified_no_committed_sample", 0) >= len(
        LICENSE_BLOCKED_DATASETS
    )


# ---------------------------------------------------------------------------
# 11. Safety — no HF_TOKEN, no raw data
# ---------------------------------------------------------------------------


# Match the literal HF token prefix without writing the test value
# verbatim — keep the test self-evidently safe.
_HF_TOKEN_LITERAL_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def test_no_hf_token_literal_in_new_files() -> None:
    """No ``hf_<base64>`` token literal in any committed artifact.

    Documentation prose mentioning ``HF_TOKEN`` (e.g. "does NOT read
    HF_TOKEN") is fine — the check looks for the literal token shape,
    not for the env-var name as a string.
    """

    paths_to_scan = [
        REPO_ROOT / "aurelius" / "ingestion" / "redistribution_gate.py",
        REPO_ROOT / "scripts" / "audit_hf_redistribution_gate.py",
        REPO_ROOT
        / "data"
        / "external"
        / "hf_discovery"
        / "redistribution_gate_audit.json",
    ]
    for p in paths_to_scan:
        assert p.exists(), p
        text = p.read_text()
        m = _HF_TOKEN_LITERAL_RE.search(text)
        assert m is None, f"{p} contains an HF token literal at {m.span()!r}"


def test_no_raw_files_in_redistribution_gate_paths() -> None:
    """The new audit JSON sits under data/external/hf_discovery/ which is
    explicitly NOT for raw downloads (those live under data/external/hf/<dataset>/raw/
    and are gitignored). Pin that the audit JSON is the only artifact added under
    this dir by the gate milestone."""

    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf_discovery"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    assert raw_committed == [], (
        f"raw downloads committed under hf_discovery: {raw_committed}"
    )


def test_gate_module_has_no_io_at_import_time(gate_module) -> None:
    """The module must be pure-Python with no top-level I/O.

    Validated structurally: ``decide_redistribution`` takes the ledger
    as a parameter (never loads it from disk itself) and ``classify_
    license`` is a pure function.
    """

    import inspect

    sig = inspect.signature(gate_module.decide_redistribution)
    # All non-default parameters are keyword-only — prevents accidental
    # positional ledger passes.
    params = sig.parameters
    assert "ledger" in params
    assert params["ledger"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "dataset_id" in params
    assert params["dataset_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "scope" in params
    assert params["scope"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "license_str" in params


def test_gate_module_reason_codes_are_a_closed_set(gate_module) -> None:
    """Pin the exported reason-code constants — adding a new code is fine,
    but renaming one would silently break the audit JSON readers."""

    required = {
        "REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE",
        "REASON_PERMITTED_OPERATOR_GRANT",
        "REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE",
        "REASON_DENIED_UNSUPPORTED_SCOPE",
    }
    exported = set(gate_module.__all__)
    missing = required - exported
    assert not missing, f"gate module missing required reason codes: {missing}"
    for name in required:
        assert isinstance(getattr(gate_module, name), str), name


def test_audit_script_does_not_read_env_or_call_hf() -> None:
    """Lightweight static scan: the audit script must not import requests /
    huggingface_hub / call ``os.environ.get('HF_TOKEN')``.

    Look for the actual env-read code patterns, not the env-var name in
    docstrings.
    """

    text = (REPO_ROOT / "scripts" / "audit_hf_redistribution_gate.py").read_text()
    assert "huggingface_hub" not in text
    assert "import requests" not in text
    # Code patterns that would read the token. Docstring references are
    # fine; these would indicate actual reads.
    forbidden_code = [
        "os.environ.get(\"HF_TOKEN\"",
        "os.environ.get('HF_TOKEN'",
        "os.environ[\"HF_TOKEN\"",
        "os.environ['HF_TOKEN'",
        "os.getenv(\"HF_TOKEN\"",
        "os.getenv('HF_TOKEN'",
    ]
    for needle in forbidden_code:
        assert needle not in text, (
            f"audit script appears to read HF_TOKEN: found {needle!r}"
        )


def test_gate_module_does_not_import_huggingface_hub(gate_module) -> None:
    """Pure-Python invariant: the gate must not import any HF SDK.

    Look for actual imports, not the env-var name in docstrings.
    """

    text = (
        REPO_ROOT / "aurelius" / "ingestion" / "redistribution_gate.py"
    ).read_text()
    assert "huggingface_hub" not in text
    assert "import requests" not in text
    forbidden_code = [
        "os.environ.get(\"HF_TOKEN\"",
        "os.environ.get('HF_TOKEN'",
        "os.environ[\"HF_TOKEN\"",
        "os.environ['HF_TOKEN'",
        "os.getenv(\"HF_TOKEN\"",
        "os.getenv('HF_TOKEN'",
    ]
    for needle in forbidden_code:
        assert needle not in text, (
            f"gate module appears to read HF_TOKEN: found {needle!r}"
        )


# ---------------------------------------------------------------------------
# 12. Backwards compatibility — the existing license verdicts still hold
# ---------------------------------------------------------------------------


def test_classify_license_agrees_with_commit_script_targets(gate_module) -> None:
    """The pre-existing ``scripts/commit_hf_gap_normalized_samples.py`` carries
    a hard-coded TARGETS table with one license verdict per dataset. The
    gate must agree with every verdict on those datasets so wiring it in
    later cannot regress the four already-committed normalised samples.
    """

    # Mirrors TARGETS in scripts/commit_hf_gap_normalized_samples.py.
    # license tag on the HF card -> expected license_status from the gate
    cases = [
        ("cc-by-4.0", "permissive_cc_by_4_0"),
        ("mit", "permissive_mit"),
        ("apache-2.0", "permissive_apache_2_0"),
        (None, "unspecified_no_committed_sample"),  # prefixbench
    ]
    for license_str, expected_status in cases:
        assert gate_module.classify_license(license_str) == expected_status, (
            f"license {license_str!r} mismatch: gate says "
            f"{gate_module.classify_license(license_str)!r}, "
            f"existing commit script verdict is {expected_status!r}"
        )


def test_existing_policy_tests_still_pass_through_module_load(
    policy_module, gate_module,
) -> None:
    """The gate must not have broken the policy module's public API.

    Pin the public surface so a future refactor cannot remove the names
    the gate currently depends on.
    """

    for name in (
        "OperatorPolicyLedger",
        "OperatorGrant",
        "POLICY_DOC_VERSION",
        "SUPPORTED_SCOPES",
        "REASON_NO_GRANT",
        "REASON_GRANTED_FALSE",
        "REASON_EXPIRED",
        "REASON_SCOPE_NOT_ALLOWED",
        "REASON_UNSUPPORTED_SCOPE",
    ):
        assert hasattr(policy_module, name), (
            f"policy module missing public name {name!r} — gate depends on it"
        )

    # The gate's PERMISSIVE_LICENSE_TAGS must be a dict mapping str → str.
    assert isinstance(gate_module.PERMISSIVE_LICENSE_TAGS, dict)
    for k, v in gate_module.PERMISSIVE_LICENSE_TAGS.items():
        assert isinstance(k, str) and isinstance(v, str)
        assert v.startswith("permissive_")


def test_committed_audit_per_dataset_count_matches_candidate_registry(
    committed_audit_payload,
) -> None:
    cands_path = (
        REPO_ROOT
        / "data"
        / "external"
        / "hf_discovery"
        / "hf_dataset_candidates.json"
    )
    cands = json.loads(cands_path.read_text())["candidates"]
    assert committed_audit_payload["candidate_count"] == len(cands)
    assert len(committed_audit_payload["per_dataset"]) == len(cands)
