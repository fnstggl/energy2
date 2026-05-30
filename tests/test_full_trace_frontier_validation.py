"""Tests for the Full-Trace Safe Utilization Frontier Validation Audit.

Proves:

- the committed Azure 2024 audit / backtest / controller / integration
  JSON are **read-only** under this audit;
- the audit JSON / markdown are generated and contain the expected
  structure;
- every applicable trace gets a recognised verdict in
  {FRONTIER_WIN, SAFE_TIE, REGRESSION};
- the audit's BurstGPT and Azure 2023 verdicts include an explicit
  root-cause code (the audit's central claim is that the previous
  fixture-bound tie *persists* on the full raw traces, with an explicit
  reason — never silently);
- the Azure 2024 verdict reproduces the committed controller's +12.98 %
  uplift within tolerance;
- no regression on any applicable trace (the safety invariant of
  ``frontier_controller_v1`` — see
  ``docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md``);
- the committed audit JSON byte content is unchanged after running the
  audit (the audit must NEVER overwrite committed artifacts);
- docs contain no unhedged production-savings claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "full_trace_frontier_validation_summary.json")
DOC_MD = os.path.join(REPO_ROOT, "docs", "FULL_TRACE_FRONTIER_VALIDATION.md")

COMMITTED_ARTIFACTS = (
    os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
                 "azure_2024_safe_utilization_frontier.json"),
    os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
                 "azure_2024_frontier_controller_summary.json"),
    os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
                 "azure_llm_2024_backtest_summary.json"),
    os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
                 "azure_2024_constraint_frontier_integration_summary.json"),
)

EXPECTED_APPLICABLE = {"burstgpt", "azure_llm_2023_conv",
                       "azure_llm_2023_code", "azure_llm_2024_week"}
EXPECTED_VERDICTS = {"FRONTIER_WIN", "SAFE_TIE", "REGRESSION"}

ROOT_CAUSE_CODES = {
    "A_insufficient_telemetry",
    "B_constraint_aware_already_on_frontier",
    "C_workload_saturation",
    "D_safety_limits_reached",
    "E_trace_limitations",
    "UNCLASSIFIED",
}

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


# ---------------------------------------------------------------------------
# 1 — committed audit summary exists and partitions correctly
# ---------------------------------------------------------------------------

def test_summary_json_exists():
    assert os.path.exists(SUMMARY_JSON), f"missing {SUMMARY_JSON}"
    d = json.load(open(SUMMARY_JSON))
    assert "config" in d and "per_trace" in d and "synthesis" in d


def test_doc_md_exists():
    assert os.path.exists(DOC_MD), f"missing {DOC_MD}"


def test_every_expected_trace_present():
    d = json.load(open(SUMMARY_JSON))
    traces = {t["trace"] for t in d["per_trace"]}
    assert EXPECTED_APPLICABLE <= traces, \
        f"missing traces: {EXPECTED_APPLICABLE - traces}"


def test_every_applicable_trace_has_recognised_verdict():
    d = json.load(open(SUMMARY_JSON))
    for t in d["per_trace"]:
        if not t["applicable"]:
            continue
        v = t["comparison"]["verdict"]
        assert v in EXPECTED_VERDICTS, f"{t['trace']}: unknown verdict {v!r}"


# ---------------------------------------------------------------------------
# 2 — central claim: BurstGPT and Azure 2023 SAFE_TIE has an explicit root cause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trace_name", ["burstgpt", "azure_llm_2023_conv",
                                         "azure_llm_2023_code"])
def test_safe_tie_has_explicit_root_cause(trace_name):
    """The audit's central claim is that the previous SAFE_TIE *persists* on
    the full raw trace with an explicit, evidence-backed reason — never
    silently."""
    d = json.load(open(SUMMARY_JSON))
    t = next((t for t in d["per_trace"] if t["trace"] == trace_name), None)
    assert t is not None and t["applicable"]
    if t["comparison"]["verdict"] != "SAFE_TIE":
        # win/regression — no root cause required
        return
    rc = t["root_cause"]
    assert rc is not None, f"{trace_name}: SAFE_TIE without root cause"
    assert rc["code"] in ROOT_CAUSE_CODES, f"unknown root cause {rc['code']!r}"
    assert rc.get("evidence"), f"{trace_name}: root cause missing evidence"


# ---------------------------------------------------------------------------
# 3 — Azure 2024 reproduces the committed +12.98 % uplift within tolerance
# ---------------------------------------------------------------------------

def test_azure_2024_reproduces_committed_uplift():
    d = json.load(open(SUMMARY_JSON))
    t = next(t for t in d["per_trace"] if t["trace"] == "azure_llm_2024_week")
    delta = t["comparison"]["delta_vs_constraint_aware_pct"]
    fc_path = os.path.join(
        REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
        "azure_2024_frontier_controller_summary.json")
    fc = json.load(open(fc_path))
    committed_delta = fc["deltas"]["frontier_vs_constraint_aware_pct"]
    assert abs(delta - committed_delta) < 0.5, \
        f"audit delta {delta} vs committed {committed_delta}"


# ---------------------------------------------------------------------------
# 4 — no regression across applicable LLM serving traces
# ---------------------------------------------------------------------------

def test_no_regression_across_applicable_traces():
    d = json.load(open(SUMMARY_JSON))
    vc = d["synthesis"]["verdict_counts"]
    assert vc["REGRESSION"] == 0, \
        f"unexpected regression on applicable traces: {vc}"


# ---------------------------------------------------------------------------
# 5 — Azure 2023 fixture-vs-full claim: full trace produces non-trivial n_ticks
# ---------------------------------------------------------------------------

def test_full_trace_n_ticks_exceeds_fixture_threshold():
    """The previous audit was fixture-bound with n_ticks ≈ 1 on Azure 2023
    and only ~50 row fixture on BurstGPT. The full-trace audit must show
    statistically meaningful tick counts on both."""
    d = json.load(open(SUMMARY_JSON))
    for trace_id, min_n_ticks in (
        ("burstgpt", 1000),                # 1.4 M reqs / ~60 days → >>1000
        ("azure_llm_2023_conv", 30),       # ~58 min trace at 60s ticks
        ("azure_llm_2023_code", 30),
    ):
        t = next((t for t in d["per_trace"] if t["trace"] == trace_id), None)
        assert t and t["applicable"], f"{trace_id} not applicable"
        n = t["n_ticks"]
        assert n >= min_n_ticks, \
            f"{trace_id}: n_ticks={n} below required {min_n_ticks}"


def test_full_trace_n_requests_exceeds_fixture_threshold():
    d = json.load(open(SUMMARY_JSON))
    for trace_id, min_n_reqs in (
        ("burstgpt", 100_000),
        ("azure_llm_2023_conv", 5_000),
        ("azure_llm_2023_code", 2_000),
    ):
        t = next((t for t in d["per_trace"] if t["trace"] == trace_id), None)
        assert t and t["applicable"]
        n = t["n_requests"]
        assert n is not None and n >= min_n_reqs, \
            f"{trace_id}: n_requests={n} below {min_n_reqs}"


# ---------------------------------------------------------------------------
# 6 — committed artifacts unchanged by re-running the audit
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_committed_artifacts_unchanged_by_audit(tmp_path):
    """Re-running the audit must NOT modify any committed Azure 2024
    artifact (defends against the previous loop-variable bug that wrote
    markdown into ``azure_2024_safe_utilization_frontier.json``)."""
    fingerprints_before = {p: _sha256(p) for p in COMMITTED_ARTIFACTS
                           if os.path.exists(p)}
    assert fingerprints_before, "no committed artifacts found to fingerprint"
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from scripts import run_full_trace_frontier_validation as fr  # noqa: E402
    out_json = str(tmp_path / "x.json")
    out_md = str(tmp_path / "x.md")
    fr.main(["--out-json", out_json, "--out-md", out_md])
    for p, fp in fingerprints_before.items():
        assert _sha256(p) == fp, \
            f"audit modified committed artifact {p}"


# ---------------------------------------------------------------------------
# 7 — answer to the headline question is one of the declared options
# ---------------------------------------------------------------------------

def test_answer_is_one_of_the_three_declared_options():
    d = json.load(open(SUMMARY_JSON))
    answer = d["synthesis"]["answer"]
    declared = [
        "FIXTURE LIMITATIONS",
        "SAFE_TIE PERSISTS on the full raw trace",
        "Mixed",
        "undetermined",
    ]
    assert any(decl in answer for decl in declared), \
        f"answer {answer!r} matches none of {declared}"


def test_generalization_verdict_recognised():
    d = json.load(open(SUMMARY_JSON))
    v = d["synthesis"]["generalization_verdict"]
    assert v.startswith(("YES", "NO", "PARTIAL")), v


def test_azure_2024_uniqueness_recognised():
    d = json.load(open(SUMMARY_JSON))
    v = d["synthesis"]["azure_2024_uniqueness"]
    assert v.startswith(("YES", "NO", "INSUFFICIENT_DATA")), v


# ---------------------------------------------------------------------------
# 8 — docs contain no unhedged production-savings claims
# ---------------------------------------------------------------------------

def test_doc_no_unhedged_banned_phrases():
    text = open(DOC_MD, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in
                       ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(DOC_MD)}"
            i = pos + len(phrase)


def test_doc_states_required_caveats():
    low = " ".join(open(DOC_MD, encoding="utf-8").read().lower().split())
    for phrase in ("simulator", "shadow", "pilot telemetry",
                   "real-cluster execution",
                   "constraint_aware"):
        assert phrase in low, f"doc missing required caveat: {phrase!r}"


# ---------------------------------------------------------------------------
# 9 — frontier sweep covers the configured rho grid for applicable traces
# ---------------------------------------------------------------------------

def test_anticipatory_sweep_covers_full_rho_grid():
    d = json.load(open(SUMMARY_JSON))
    grid = set(round(r, 4) for r in d["config"]["rhos"])
    for t in d["per_trace"]:
        if not t["applicable"] or not t.get("frontier_anticipatory"):
            continue
        rhos = {round(p["rho_target"], 4)
                for p in t["frontier_anticipatory"]}
        assert grid <= rhos, \
            f"{t['trace']}: anticipatory sweep missing {grid - rhos}"
