"""Tests for the residency telemetry audit CLI / report.

Validates that ``scripts/audit_residency_telemetry.py`` builds an honest summary
from the fixtures, that the markdown is reproducible (the committed
``docs/RESIDENCY_TELEMETRY_AUDIT.md`` regenerates byte-identically), and that no
unhedged production-savings claim leaks into the generated report.
"""

from __future__ import annotations

import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "audit_residency_telemetry.py")
FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "residency")
EVENTS = os.path.join(FIX, "events.jsonl")
SNAPSHOTS = os.path.join(FIX, "snapshots.jsonl")
REQUESTS = os.path.join(FIX, "requests.jsonl")
COMMITTED_MD = os.path.join(REPO_ROOT, "docs", "RESIDENCY_TELEMETRY_AUDIT.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_residency_telemetry", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_summary_from_fixtures():
    mod = _load_module()
    s = mod.build_summary(EVENTS, snapshots_path=SNAPSHOTS, requests_path=REQUESTS,
                          slo_s=5.0, gpu_hour_cost=2.5)
    assert s["readiness_verdict"] == "SHADOW_PILOT_READY_READ_ONLY"
    assert s["linkage"]["attribution_gate_quality"] == "container_join"
    assert s["linkage"]["attributable"] is True
    assert s["shadow"]["cluster_mutation"] is False
    assert s["shadow"]["summary"]["all_recommendation_only"] is True
    assert s["directional_only_not_production_savings"] is True
    # diagnostics never folded into KPI
    assert "NEVER folded" in s["kpi_note"]


def test_summary_handles_events_only():
    mod = _load_module()
    s = mod.build_summary(EVENTS)
    # without observations the substrate cannot reach shadow-pilot-ready
    assert s["readiness_verdict"] == "TRACE_BACKTESTED_APPROXIMATION"
    assert s["shadow"]["summary"]["n_decisions"] == 0


def test_markdown_is_reproducible():
    mod = _load_module()
    s = mod.build_summary("tests/fixtures/residency/events.jsonl",
                          snapshots_path="tests/fixtures/residency/snapshots.jsonl",
                          requests_path="tests/fixtures/residency/requests.jsonl",
                          slo_s=5.0, gpu_hour_cost=2.5)
    md = mod.render_markdown(s)
    with open(COMMITTED_MD, encoding="utf-8") as fh:
        committed = fh.read()
    assert md == committed, "regenerate with scripts/audit_residency_telemetry.py"


def test_report_has_required_sections():
    text = open(COMMITTED_MD, encoding="utf-8").read().lower()
    for section in ("telemetry coverage", "linkage quality",
                    "residency hit/miss", "cold-start latency distribution",
                    "warm-pool cost", "shadow recommendations",
                    "missing fields preventing pilot readiness"):
        assert section in text, f"report missing section: {section}"


def test_report_no_unhedged_production_savings_claims():
    text = open(COMMITTED_MD, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in ("not ", "no ", "never ", "n't ",
                                          "without ")), \
                f"unhedged '{phrase}' in residency audit report"
            i = pos + len(phrase)
