"""Unified redistribution gate for HF normalized sample commits.

Discovery / data-engine infrastructure. No scheduler change, no
controller change, no robust-energy-engine change, no production
claim, no Tier 1 promotion, no HF ingestion, no ``HF_TOKEN`` read, no
raw download. Pure-Python; no third-party deps.

Why this module exists
----------------------

The Round-8 audit + the operator redistribution policy framework
(PR #151) established a deny-by-default consent record for
``license = None`` HF datasets, but they left the *consumer side*
unwritten: the ledger has an API
(``OperatorPolicyLedger.permits_redistribution(...)``) and a
committed policy file with zero grants, yet **no code in the actual
sample-commit path consults it**. So today the existing
``scripts/commit_hf_gap_normalized_samples.py`` and the per-dataset
ingestion scripts each carry their *own* hard-coded license verdict
(``permissive_apache_2_0`` / ``unspecified_no_committed_sample`` /
…) and an operator who recorded a grant in
``operator_redistribution_policy.json`` would see no behavioural
effect.

This module IS that consumer. It defines one canonical decision
function — :func:`decide_redistribution` — that fuses the two
permitted paths into a single auditable result:

1. **Declared permissive license** on the HF card (Apache-2.0, MIT,
   CC-BY-4.0, CC-BY-3.0, CC-BY-2.0, CC-BY-SA-4.0, CC-BY-SA-3.0,
   CC0-1.0, CDLA-Permissive-2.0, ODC-BY-1.0). Skip the ledger.
   Permit. ShareAlike clauses constrain *derivative works* — they do
   not block redistribution of the original; the derivative
   normalised samples this corpus commits inherit the same license,
   so SA tags qualify as permissive at the redistribution layer.
2. **License declared but not on the permissive allow-list** (e.g.
   ``"other"``, ``"openrail"``, GPL, custom research licenses). Deny
   — these require per-dataset manual review even if an operator
   grant exists, because the operator grant scope is specifically
   for ``license = None`` consent, not for re-interpreting a
   declared restrictive license.
3. **License is None / empty / unspecified**. Consult the operator
   ledger. If a valid, in-scope, non-expired grant exists, permit
   with the grant id + provenance recorded on the decision. Else
   deny.

Design rules
------------

* **Default DENY** for everything outside the explicit permissive
  allow-list and not covered by an operator grant. With the
  committed policy file shipping zero grants, this gate denies every
  ``license = None`` dataset — identical to today's behaviour.
* **No silent widening**. The closed permissive allow-list is in
  :data:`PERMISSIVE_LICENSE_TAGS`. Any tag outside that frozenset
  (including ``"other"`` and ``"openrail"``) is treated as
  non-permissive.
* **License normalisation is conservative**. ``classify_license``
  lowercases, strips, and collapses whitespace, but does NOT try to
  pattern-match "Apache 2.0 with addendum" → permissive_apache_2_0,
  because such a match would silently widen redistribution. Only
  the exact canonical HF tag set is recognised.
* **Operator grant path is opt-in**. The ledger is consulted *only*
  when the declared license is None / empty / unspecified. An
  operator cannot grant redistribution for a declared restrictive
  license through this module — that would conflict with the upstream
  license owner.
* **Reason codes are a closed set**. Tests assert on the code, not
  the free-form detail.
* **No I/O**. The function takes the ledger as a parameter; callers
  load the ledger once and reuse it. The function never reads files
  or environment variables and never calls the HF API.
* **Backwards compatible**. The existing
  ``scripts/commit_hf_gap_normalized_samples.py`` is unchanged in
  this PR. Future PRs may wire the gate in to replace that script's
  hard-coded TARGETS table; until they do, both paths agree on
  outcomes for the permissive cases and the gate adds the only new
  behaviour (consulting the ledger for ``license = None``) under
  zero default grants → identical outcomes.

Recognised permissive license tags
----------------------------------

The closed set is derived from the HF tags actually present in the
99-candidate registry today (apache-2.0, mit, cc-by-4.0, cc-by-2.0,
cc-by-sa-4.0, cc0-1.0, cdla-permissive-2.0) plus a conservative
extension to near-equivalent CC variants (cc-by-3.0, cc-by-sa-3.0)
and the public-domain ODC-BY-1.0 used by some scheduler traces.
Each tag maps to a single canonical ``license_status`` so downstream
summary writers continue to use the existing string labels
(``permissive_apache_2_0`` etc.).

Non-permissive declared licenses (e.g. ``"other"``, ``"openrail"``,
``"cc"`` without a clause suffix, custom research licenses) are
classified as ``"declared_non_permissive"`` and DENIED by the gate
regardless of operator grants. That preserves the invariant that
operator grants exist only to record *consent under license = None*,
not to override an upstream owner's declared restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aurelius.ingestion.operator_redistribution_policy import (
    REASON_NO_GRANT,
    OperatorPolicyLedger,
)

# ---------------------------------------------------------------------------
# License classification — closed set of canonical permissive tags
# ---------------------------------------------------------------------------

# HF license tag (lowercased, stripped) -> canonical status code used in
# summary.json and the audit JSON. The status code mirrors the labels
# already used by ``scripts/commit_hf_gap_normalized_samples.py`` so the
# two paths produce equivalent outputs on the permissive cases.
PERMISSIVE_LICENSE_TAGS: dict[str, str] = {
    "apache-2.0": "permissive_apache_2_0",
    "apache_2_0": "permissive_apache_2_0",
    "mit": "permissive_mit",
    "cc-by-4.0": "permissive_cc_by_4_0",
    "cc-by-3.0": "permissive_cc_by_3_0",
    "cc-by-2.0": "permissive_cc_by_2_0",
    # CC-BY-SA-* permits redistribution of the original work with
    # attribution; the ShareAlike clause requires derivative works to be
    # released under the same license. The bounded normalised sample we
    # commit IS a derivative (we transform the raw CSV into JSONL with
    # additional provenance), and the committed federated corpus is
    # itself released under the same CC-BY-SA license inheritance, so
    # redistribution is permitted. Attribution is preserved in the
    # per-dataset summary.json's ``license_redistribution_attribution_notes``
    # field. This is the gate's first ShareAlike entry; added when
    # ``scripts/ingest_hf_llm_energy_consumption.py`` was wired through
    # the gate (fifth consumer).
    "cc-by-sa-4.0": "permissive_cc_by_sa_4_0",
    "cc-by-sa-3.0": "permissive_cc_by_sa_3_0",
    "cc0-1.0": "permissive_cc0_1_0",
    "cdla-permissive-2.0": "permissive_cdla_2",
    "cdla-permissive-1.0": "permissive_cdla_1",
    "odc-by-1.0": "permissive_odc_by_1_0",
    "bsd-3-clause": "permissive_bsd_3_clause",
    "bsd-2-clause": "permissive_bsd_2_clause",
}

LICENSE_STATUS_DECLARED_NON_PERMISSIVE = "declared_non_permissive"
LICENSE_STATUS_UNSPECIFIED = "unspecified_no_committed_sample"

# Reason codes — closed set so tests can assert without string-matching.
REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE = (
    "permitted_declared_permissive_license"
)
REASON_PERMITTED_OPERATOR_GRANT = "permitted_operator_grant"
REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE = (
    "denied_declared_non_permissive_license"
)
REASON_DENIED_UNSUPPORTED_SCOPE = "denied_unsupported_scope"
# All ledger denial codes propagate verbatim — caller may re-export from
# ``operator_redistribution_policy``. The most common is REASON_NO_GRANT.


@dataclass(frozen=True)
class RedistributionGateDecision:
    """Result of :func:`decide_redistribution`.

    ``permitted`` is the single-bit gate outcome. ``reason_code`` is
    a closed-set token (see module docstring). ``license_status`` is
    the canonical status code the summary writer should record.
    ``operator_grant_dataset_id`` is non-None only when the decision
    was made via the ledger path.
    """

    permitted: bool
    reason_code: str
    reason_detail: str
    license_status: str
    license_observed: Optional[str]
    scope: str
    operator_grant_dataset_id: Optional[str]


def _normalise_license(license_str: Optional[str]) -> Optional[str]:
    """Lowercase / strip; return None for empty or whitespace-only inputs.

    Conservative — does not attempt pattern matching. Only exact HF
    canonical tags map to the permissive set.
    """

    if license_str is None:
        return None
    s = license_str.strip().lower()
    if not s:
        return None
    # Collapse internal whitespace runs to a single space so e.g.
    # ``"Apache 2.0"`` and ``"Apache  2.0"`` normalise the same way.
    s = " ".join(s.split())
    return s


def classify_license(license_str: Optional[str]) -> str:
    """Map an HF license tag to a canonical ``license_status`` code.

    Returns one of:

    - one of the ``permissive_*`` values listed in
      :data:`PERMISSIVE_LICENSE_TAGS`,
    - :data:`LICENSE_STATUS_DECLARED_NON_PERMISSIVE` when the field
      is non-empty but not in the closed permissive allow-list,
    - :data:`LICENSE_STATUS_UNSPECIFIED` when the field is None /
      empty / whitespace-only.

    No state, no I/O, no ledger consultation. Cheap to call.
    """

    norm = _normalise_license(license_str)
    if norm is None:
        return LICENSE_STATUS_UNSPECIFIED
    if norm in PERMISSIVE_LICENSE_TAGS:
        return PERMISSIVE_LICENSE_TAGS[norm]
    return LICENSE_STATUS_DECLARED_NON_PERMISSIVE


def decide_redistribution(
    *,
    dataset_id: str,
    license_str: Optional[str],
    scope: str,
    ledger: OperatorPolicyLedger,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Canonical decision: may we commit a normalised sample of this dataset?

    The decision fuses the two permitted paths into one auditable
    result. ``scope`` is one of the operator-policy scopes (typically
    ``"committed_normalized_sample"``).

    Behaviour:

    1. Permissive declared license → permit, ledger NOT consulted.
       ``license_status`` is the matching ``permissive_*`` code.
    2. Declared license but not in the permissive allow-list →
       deny. The operator ledger does not unblock declared restrictive
       licenses; reach out to the dataset owner instead.
    3. License is None / empty / unspecified → consult the ledger.
       Propagate the ledger's decision (permit when a valid, in-scope,
       non-expired ``granted=true`` grant exists; deny otherwise).

    ``now_iso`` is forwarded to the ledger for test reproducibility;
    production callers should leave it unset.
    """

    license_status = classify_license(license_str)

    if license_status.startswith("permissive_"):
        return RedistributionGateDecision(
            permitted=True,
            reason_code=REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE,
            reason_detail=(
                f"declared HF license {license_str!r} maps to "
                f"{license_status!r}; ledger not consulted"
            ),
            license_status=license_status,
            license_observed=license_str,
            scope=scope,
            operator_grant_dataset_id=None,
        )

    if license_status == LICENSE_STATUS_DECLARED_NON_PERMISSIVE:
        return RedistributionGateDecision(
            permitted=False,
            reason_code=REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE,
            reason_detail=(
                f"declared HF license {license_str!r} is not in the "
                f"permissive allow-list "
                f"{sorted(set(PERMISSIVE_LICENSE_TAGS.values()))!r}; "
                f"operator grants do not override declared upstream "
                f"licenses — contact the dataset owner to clarify"
            ),
            license_status=license_status,
            license_observed=license_str,
            scope=scope,
            operator_grant_dataset_id=None,
        )

    # license_status == LICENSE_STATUS_UNSPECIFIED → ledger path.
    policy_decision = ledger.permits_redistribution(
        dataset_id=dataset_id,
        scope=scope,
        now_iso=now_iso,
    )
    if policy_decision.permitted:
        return RedistributionGateDecision(
            permitted=True,
            reason_code=REASON_PERMITTED_OPERATOR_GRANT,
            reason_detail=(
                f"license=None for {dataset_id!r}; operator grant "
                f"permits scope {scope!r}: {policy_decision.reason_detail}"
            ),
            license_status=license_status,
            license_observed=license_str,
            scope=scope,
            operator_grant_dataset_id=policy_decision.matched_grant_dataset_id,
        )

    # Propagate the ledger's exact denial reason code (no_grant_recorded,
    # grant_explicitly_denies, grant_expired, requested_scope_not_in_
    # allowed_scopes, requested_scope_not_in_supported_scopes, …) so
    # downstream tooling can pivot on the same closed set the ledger uses.
    if policy_decision.reason_code == REASON_NO_GRANT:
        detail = (
            f"license=None for {dataset_id!r}; no operator grant "
            f"recorded — default policy denies. "
            f"{policy_decision.reason_detail}"
        )
    else:
        detail = (
            f"license=None for {dataset_id!r}; operator ledger "
            f"denied scope {scope!r}: {policy_decision.reason_detail}"
        )

    return RedistributionGateDecision(
        permitted=False,
        reason_code=policy_decision.reason_code,
        reason_detail=detail,
        license_status=license_status,
        license_observed=license_str,
        scope=scope,
        operator_grant_dataset_id=policy_decision.matched_grant_dataset_id,
    )


__all__ = [
    "PERMISSIVE_LICENSE_TAGS",
    "LICENSE_STATUS_DECLARED_NON_PERMISSIVE",
    "LICENSE_STATUS_UNSPECIFIED",
    "REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE",
    "REASON_PERMITTED_OPERATOR_GRANT",
    "REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE",
    "REASON_DENIED_UNSUPPORTED_SCOPE",
    "RedistributionGateDecision",
    "classify_license",
    "decide_redistribution",
]
