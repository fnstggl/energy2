"""Validation tests for the Model Residency / Cold-Start spec + telemetry contract.

These are DOCS-ONLY validation tests (no optimizer/simulator code is exercised).
They assert the spec docs exist, define every required concept/field/metric/rule,
state the shadow-mode + no-substitution + claim-gate constraints, list the
integration surfaces, reference the grounding docs, and contain no unhedged
production-savings claims.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(REPO_ROOT, "docs", "MODEL_RESIDENCY_COLD_START_SPEC.md")
CONTRACT = os.path.join(REPO_ROOT, "docs", "PILOT_TELEMETRY_CONTRACT.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _read(path: str) -> str:
    assert os.path.exists(path), f"missing doc: {path}"
    return open(path).read()


def test_docs_exist():
    assert os.path.exists(SPEC)
    assert os.path.exists(CONTRACT)


def test_core_concepts_defined():
    text = _read(SPEC).lower()
    for concept in ("model residency", "adapter / lora residency", "warm pool",
                    "cold start", "model load latency", "adapter load latency",
                    "cache affinity", "prewarm action", "preserve-affinity action",
                    "cold-start risk"):
        assert concept in text, f"spec missing core concept: {concept}"


def test_required_telemetry_fields_present():
    text = _read(CONTRACT)
    for field in ("request_id", "timestamp", "tenant_id", "workload_id", "model_id",
                  "adapter_id", "lora_id", "endpoint_id", "region", "node_id",
                  "gpu_id", "container_id", "model_loaded_before_request",
                  "adapter_loaded_before_request", "model_load_start",
                  "model_load_end", "adapter_load_start", "adapter_load_end",
                  "queue_wait", "TTFT", "TPOT", "e2e_latency", "status",
                  "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
                  "power", "prefix_hit"):
        assert field in text, f"telemetry contract missing field: {field}"


def test_derived_metrics_present():
    text = _read(SPEC).lower()
    for metric in ("model residency hit rate", "adapter residency hit rate",
                   "cold-start rate", "cold-start latency p50/p95/p99",
                   "warm-pool cost", "cold-start avoided latency",
                   "sla violations attributable to cold start",
                   "goodput/$ with and without prewarm",
                   "model popularity half-life", "residency churn score"):
        assert metric in text, f"spec missing derived metric: {metric}"


def test_decision_rules_present_and_no_substitution():
    text = _read(SPEC).lower()
    assert "prewarm rule" in text
    assert "preserve-affinity rule" in text
    assert "evict rule" in text
    # the binding safety rule: never substitute a different model
    assert "no-substitution rule" in text
    assert "never" in text and "different model" in text


def test_spec_is_spec_only_not_implementation():
    text = _read(SPEC).lower()
    assert "spec only" in text or "specification only" in text \
        or "not implemented here" in text
    assert "changes no optimizer behavior" in text \
        or "no optimizer behavior" in text


def test_shadow_mode_requirements():
    for path in (SPEC, CONTRACT):
        # whitespace-normalize so markdown line breaks don't split key phrases
        text = " ".join(_read(path).lower().split())
        assert "recommendation-only" in text
        assert ("no real cluster mutation" in text
                or "no production cluster mutation" in text)
        assert "counterfactual" in text


def test_integration_points_listed():
    text = _read(SPEC)
    for surface in ("vLLM", "Triton", "SGLang", "Ray Serve", "Kubernetes",
                    "DCGM", "Prometheus"):
        assert surface in text, f"spec missing integration surface: {surface}"


def test_benchmark_standard_separate_attribution():
    text = _read(SPEC).lower()
    # benchmarks must report affinity/prewarm separately from queue/util/energy
    assert "separately" in text
    assert "shapley" in text
    assert "model-affinity/prewarm contribution separately" in text \
        or "affinity/prewarm contribution separately" in text


def test_references_grounding_docs():
    text = _read(SPEC)
    for ref in ("docs/RESULTS.md", "docs/ALIBABA_GENAI_ABLATION_RESULTS.md",
                "docs/ALIBABA_GENAI_BACKTEST_RESULTS.md",
                "docs/PUBLIC_TRACE_BACKTESTS.md"):
        assert ref in text, f"spec must reference {ref}"
    assert "62" in text  # cites the ~62% affinity attribution


def test_claim_gate_and_no_unhedged_banned_claims():
    for path in (SPEC, CONTRACT):
        text = _read(path)
        assert "§8" in text or "production-claim gate" in text.lower()
        low = text.lower()
        for phrase in BANNED:
            i = 0
            while True:
                pos = low.find(phrase, i)
                if pos == -1:
                    break
                # whitespace-normalize the preceding window (markdown line breaks
                # can split a negation like "not\n> production savings")
                pre = " ".join(low[max(0, pos - 30):pos].split()) + " "
                assert any(n in pre for n in ("not ", "no ", "never ", "n't ",
                                              "without ")), \
                    f"unhedged '{phrase}' in {os.path.basename(path)}: " \
                    f"...{text[max(0,pos-30):pos+len(phrase)+8]}..."
                i = pos + len(phrase)


def test_no_optimizer_or_engine_source_changed_by_this_feature():
    # Guard: this is a docs/spec-only change. The spec must not ship alongside
    # edits to the optimizer/engine/simulator. We assert the spec asserts this;
    # the PR diff is the actual evidence (docs + this test only).
    text = _read(SPEC).lower()
    assert "robust energy engine" in text and "no robust-energy-engine" in text \
        or "no robust-energy-engine or simulator-constant change" in text
