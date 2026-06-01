#!/usr/bin/env python3
"""Audit the existing Aurelius scorer paths + the upgraded shadow
scorer term surface. Writes:

- ``data/external/forecasting/constraint_scorer_upgrade/
  scorer_path_audit.json`` (per-input classification of every
  goodput/cost/latency/SLA path),
- ``data/external/forecasting/constraint_scorer_upgrade/
  term_coverage_matrix.json`` (which scorer variant expresses which term,
  and at which signal level).

Audit-only. Touches no production module. Records every
$-denominated coefficient's calibration source so reviewers can verify
no invented constants were introduced.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting.constraint_scorer_features import (  # noqa: E402
    SCORER_TERM_CATALOG,
    signal_level_for,
    term_coverage_for_scorer,
)
from aurelius.forecasting.constraint_shadow_scorer import (  # noqa: E402
    ALL_VARIANTS,
    SHADOW_SCORER_VERSION,
)
from aurelius.residency.decision import (  # noqa: E402
    score_residency_candidate,
)

OUT_DIR = REPO_ROOT / "data" / "external" / "forecasting" / "constraint_scorer_upgrade"


SCORER_PATHS_INVENTORY = [
    {
        "input": "TTFT",
        "enters_at": (
            "aurelius/residency/decision.py::_service_time_s -> service_s -> "
            "expected_latency"),
        "production_classification": "heuristic",
        "production_source": (
            "SafetyContext.seconds_per_token * request.output_tokens "
            "OR SafetyContext.service_time_proxy_s (default 2.0s)"),
        "upgraded_classification": "derived_from_optimum_benchmark",
        "upgraded_source": (
            "Optimum-benchmark median p50_ttft_ms per (model_family, gpu) "
            "or CARA TTFT p50 shadow prior when the cell hits; "
            "max(production_value, refined) clamp is the safety floor"),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "TPOT",
        "enters_at": (
            "aurelius/residency/decision.py::_service_time_s -> service_s"),
        "production_classification": "heuristic",
        "production_source": "Collapsed into the same service_time_proxy_s",
        "upgraded_classification": "derived_from_optimum_benchmark",
        "upgraded_source": (
            "decode_throughput_tok_s × request.output_tokens or "
            "tpot_ms_p50 × output_tokens; both from Optimum cell."),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "E2E latency",
        "enters_at": (
            "score_residency_candidate -> expected_latency = "
            "queue_wait + model_penalty + adapter_penalty + service_s"),
        "production_classification": "derived",
        "production_source": "Sum of measured queue_wait + heuristic service_s",
        "upgraded_classification": "derived_from_per_gpu_prior",
        "upgraded_source": (
            "Same sum; ``service_s`` refined from Optimum/CARA priors, "
            "``migration_cache_loss_penalty_s`` added when current_route "
            "differs and Optimum prefill_throughput_tok_s is available."),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "queue wait",
        "enters_at": "ModelLocationState.estimated_queue_wait_s",
        "production_classification": "measured_or_proxy",
        "production_source": (
            "Measured when available; proxy = queue_depth × service_s "
            "when not"),
        "upgraded_classification": "measured_or_proxy",
        "upgraded_source": "Same; queue wait is Level-1 telemetry",
        "level_3_prior_used": None,
    },
    {
        "input": "queue depth",
        "enters_at": "ModelLocationState.queue_depth",
        "production_classification": "measured",
        "production_source": (
            "Telemetry adapter when available; None otherwise (veto)"),
        "upgraded_classification": "measured",
        "upgraded_source": "Same",
        "level_3_prior_used": None,
    },
    {
        "input": "prefill cost",
        "enters_at": "(not used in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_from_per_gpu_prior",
        "upgraded_source": (
            "prompt_tokens / prefill_throughput_tok_s (from Optimum)"
            " × per_gpu_hour_price / 3600  → $; "
            "USD form requires operator_per_gpu_hour_price; otherwise "
            "reported uncalibrated."),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "decode cost",
        "enters_at": "(not used as a separate term)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_from_per_gpu_prior",
        "upgraded_source": (
            "output_tokens / decode_throughput_tok_s × per_gpu_hour_price"
            "; subset of service_time_s computation."),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "cache-hit value",
        "enters_at": "(not used in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "forecasted_cache_prefix_reuse_v1_proxy",
        "upgraded_source": (
            "predicted_reuse_pct × prefill_time_s × per_gpu_hour_price "
            "→ savings_usd; USD form requires operator_per_gpu_hour_price; "
            "otherwise reported uncalibrated."),
        "level_3_prior_used": "cache_prefix_reuse_v1 forecaster",
    },
    {
        "input": "prefix reuse",
        "enters_at": "(not used in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "forecasted_cache_prefix_reuse_v1_proxy",
        "upgraded_source": (
            "SwissAI bucket-reuse forecaster output, fed into "
            "cache-hit value above. Currently classified diagnostic_only "
            "by docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md."),
        "level_3_prior_used": "cache_prefix_reuse_v1 forecaster",
    },
    {
        "input": "cold-start penalty",
        "enters_at": "score_residency_candidate -> model_penalty + adapter_penalty",
        "production_classification": "operator_supplied_via_ModelLoadProfile",
        "production_source": (
            "ModelLoadProfile.model_load_penalty_s — operator-supplied "
            "calibrated profile; missing → veto."),
        "upgraded_classification": "operator_supplied_via_ModelLoadProfile",
        "upgraded_source": (
            "Same; the shadow scorer reuses the existing profile and "
            "never invents a cold-start coefficient."),
        "level_3_prior_used": None,
    },
    {
        "input": "migration cache-loss penalty",
        "enters_at": "(not used in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_from_per_gpu_prior",
        "upgraded_source": (
            "When request.current_route differs and a recompute is needed: "
            "prompt_tokens / prefill_throughput_tok_s (from Optimum). "
            "Reported in SECONDS; USD form requires operator policy."),
        "level_3_prior_used": "optimum-benchmark/llm-perf-leaderboard",
    },
    {
        "input": "model load / unload",
        "enters_at": "ModelLoadProfile",
        "production_classification": "operator_supplied_via_ModelLoadProfile",
        "production_source": "ModelLoadProfile.model_load_penalty_s",
        "upgraded_classification": "operator_supplied_via_ModelLoadProfile",
        "upgraded_source": "Same",
        "level_3_prior_used": None,
    },
    {
        "input": "GPU type",
        "enters_at": "location_key (string only); not parsed",
        "production_classification": "missing",
        "production_source": (
            "Encoded in location_key but never parsed by the scorer."),
        "upgraded_classification": "derived",
        "upgraded_source": (
            "constraint_scorer_features.derive_gpu_type parses the "
            "trailing token of the location_key; used to look up "
            "Optimum / per-GPU operator price cells."),
        "level_3_prior_used": None,
    },
    {
        "input": "model size",
        "enters_at": "request.model_id (string only)",
        "production_classification": "missing",
        "production_source": "Not parsed",
        "upgraded_classification": "derived",
        "upgraded_source": (
            "constraint_scorer_features.derive_model_size_b regex-parses "
            "size tokens (``-7b``, ``-70b``, ``8x7b``); used for fallback"
            " family lookup."),
        "level_3_prior_used": None,
    },
    {
        "input": "VRAM / memory pressure",
        "enters_at": "score_residency_candidate -> memory_pressure_cost",
        "production_classification": "derived",
        "production_source": (
            "max(0, used_after / total - 0.9) × incremental_gpu_cost"),
        "upgraded_classification": "derived",
        "upgraded_source": "Same; carried through from production scorer",
        "level_3_prior_used": None,
    },
    {
        "input": "GPU utilization",
        "enters_at": "ModelLocationState.gpu_utilization",
        "production_classification": "measured",
        "production_source": "Telemetry adapter; not used in score formula",
        "upgraded_classification": "measured",
        "upgraded_source": "Reported via AcmeTrace prior bundle; not yet folded into score",
        "level_3_prior_used": "Qinghao/AcmeTrace",
    },
    {
        "input": "energy per request",
        "enters_at": "(not in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_from_optimum_or_acmetrace",
        "upgraded_source": (
            "prefill_energy_total_kwh + decode_energy_total_kwh × "
            "(output_tokens / 64)  [Optimum]; fallback to "
            "AcmeTrace mean_w × service_s when Optimum cell is missing."),
        "level_3_prior_used": (
            "optimum-benchmark/llm-perf-leaderboard, Qinghao/AcmeTrace"),
    },
    {
        "input": "power draw",
        "enters_at": "(not in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "level_3_prior",
        "upgraded_source": (
            "AcmeTrace IPMI per-GPU mean_w / p95_w (Tier-2 cluster "
            "telemetry treated as prior at integration layer)"),
        "level_3_prior_used": "Qinghao/AcmeTrace",
    },
    {
        "input": "cloud cost",
        "enters_at": "score_residency_candidate -> incremental_gpu_cost",
        "production_classification": "static_global_default",
        "production_source": (
            "SafetyContext.gpu_hour_price (default 3.0); single global "
            "constant per scorer call. Operator-configurable."),
        "upgraded_classification": "operator_policy_or_operator_global_default",
        "upgraded_source": (
            "OperatorPricingPolicy.gpu_hour_price_per_type per GPU type; "
            "falls back to ctx.gpu_hour_price when the policy is empty. "
            "No invented $/hr values are stored anywhere in this module."),
        "level_3_prior_used": None,
    },
    {
        "input": "energy cost ($)",
        "enters_at": "(not in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_iff_operator_energy_price_supplied",
        "upgraded_source": (
            "energy_kwh_per_request × operator_policy.energy_price_per_kwh_usd. "
            "If energy_price is None the term is reported uncalibrated and "
            "excluded from the headline cost."),
        "level_3_prior_used": (
            "optimum-benchmark/llm-perf-leaderboard, Qinghao/AcmeTrace"),
    },
    {
        "input": "carbon cost ($)",
        "enters_at": "(not in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "derived_iff_operator_carbon_price_supplied",
        "upgraded_source": (
            "energy_kwh × carbon_intensity_g_per_kwh × "
            "operator_policy.carbon_price_per_kg_usd. Uncalibrated when "
            "carbon_price is None."),
        "level_3_prior_used": "operator + grid intensity feed",
    },
    {
        "input": "SLA risk",
        "enters_at": "score_residency_candidate -> sla_met (binary)",
        "production_classification": "derived_binary",
        "production_source": "expected_latency_s <= ctx.sla_s(request)",
        "upgraded_classification": "derived_latency_margin_indicator",
        "upgraded_source": (
            "Latency MARGIN in seconds (sla_s - expected_latency_s); a "
            "quantitative indicator without a fitted sigmoid coefficient. "
            "Binary sla_met still drives goodput/$ to preserve the "
            "production safety contract."),
        "level_3_prior_used": None,
    },
    {
        "input": "timeout risk",
        "enters_at": "(not in production residency scorer)",
        "production_classification": "missing",
        "production_source": None,
        "upgraded_classification": "missing",
        "upgraded_source": (
            "Pilot timeout labels required; deferred to a future PR per "
            "FORECAST_LEVERAGE_AUDIT engine #12."),
        "level_3_prior_used": None,
    },
]


def write_scorer_path_audit():
    sig = inspect.signature(score_residency_candidate)
    parameters = [
        {"name": p.name,
         "kind": str(p.kind),
         "annotation": (str(p.annotation)
                        if p.annotation is not inspect.Parameter.empty else None)}
        for p in sig.parameters.values()
    ]
    inventory = []
    counts_existing: dict[str, int] = {}
    counts_upgraded: dict[str, int] = {}
    for entry in SCORER_PATHS_INVENTORY:
        e = dict(entry)
        e["production_signal_level"] = signal_level_for(
            entry["production_classification"])
        e["upgraded_signal_level"] = signal_level_for(
            entry["upgraded_classification"])
        inventory.append(e)
        counts_existing[entry["production_classification"]] = (
            counts_existing.get(entry["production_classification"], 0) + 1)
        counts_upgraded[entry["upgraded_classification"]] = (
            counts_upgraded.get(entry["upgraded_classification"], 0) + 1)

    payload = {
        "audit_only": True,
        "doc_version": "constraint_scorer_upgrade_audit_v1",
        "shadow_only": True,
        "production_claim": False,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "shadow_scorer_version": SHADOW_SCORER_VERSION,
        "evaluated_at_s": time.time(),
        "scorer_signature_existing": {
            "callable": "score_residency_candidate",
            "module": "aurelius.residency.decision",
            "parameters": parameters,
        },
        "signal_hierarchy": {
            "level_1_measured": (
                "Pilot telemetry / hardware measurement / per-request "
                "observation / explicit operator policy. Source + units "
                "recorded; no fitted coefficient allowed."),
            "level_2_derived": (
                "Transparent formula on Level-1 and Level-3 inputs; no "
                "learned utility coefficient."),
            "level_3_prior": (
                "Bounded benchmark / public-trace / forecasted values; "
                "tagged ``value_quality = prior``; never silently treated "
                "as production truth."),
            "level_4_prohibited_or_uncalibrated": (
                "Invented utility coefficient / reward / penalty / weight"
                "; or a $-denominated term whose operator coefficient is "
                "missing. Excluded from headline."),
        },
        "scoring_inputs_existing_vs_upgraded": inventory,
        "counts_by_classification_existing": counts_existing,
        "counts_by_classification_upgraded": counts_upgraded,
        "level_counts_existing": _level_counts(counts_existing),
        "level_counts_upgraded": _level_counts(counts_upgraded),
        "dollar_coefficient_calibration_sources": [
            {
                "coefficient": "per_gpu_hour_price_usd",
                "source": (
                    "OperatorPricingPolicy.gpu_hour_price_per_type or "
                    "SafetyContext.gpu_hour_price (existing operator "
                    "global default)"),
                "value_quality_when_supplied": "level_1_operator_policy",
                "value_quality_when_missing": (
                    "operator_global_default (Level-1 operator policy)"),
                "invented_constants_introduced": False,
            },
            {
                "coefficient": "energy_price_per_kwh_usd",
                "source": (
                    "OperatorPricingPolicy.energy_price_per_kwh_usd "
                    "(operator supplied; Aurelius already ingests real "
                    "CAISO/PJM/ERCOT price data via "
                    "aurelius.forecasting.price_model)"),
                "value_quality_when_supplied": "level_1_operator_policy",
                "value_quality_when_missing": (
                    "term reported uncalibrated; excluded from "
                    "headline SLA-safe goodput/$"),
                "invented_constants_introduced": False,
            },
            {
                "coefficient": "carbon_price_per_kg_usd",
                "source": (
                    "OperatorPricingPolicy.carbon_price_per_kg_usd"),
                "value_quality_when_supplied": "level_1_operator_policy",
                "value_quality_when_missing": (
                    "term reported uncalibrated; excluded from "
                    "headline SLA-safe goodput/$"),
                "invented_constants_introduced": False,
            },
            {
                "coefficient": "cache_value_weight",
                "source": "DOES NOT EXIST IN THIS MODULE",
                "value_quality_when_supplied": "n/a",
                "value_quality_when_missing": "n/a",
                "invented_constants_introduced": False,
                "note": (
                    "No CACHE_VALUE / CACHE_WEIGHT / MIGRATION_PENALTY "
                    "scalars exist anywhere in this PR. Cache value is "
                    "DERIVED at scoring time from "
                    "predicted_reuse × prefill_throughput × operator_$/hr."
                ),
            },
            {
                "coefficient": "utility_weights_in_composite",
                "source": "DOES NOT EXIST IN THIS MODULE",
                "value_quality_when_supplied": "n/a",
                "value_quality_when_missing": "n/a",
                "invented_constants_introduced": False,
                "note": (
                    "The shadow scorer's primary KPI is SLA-safe "
                    "goodput/$ — a single quotient, not a weighted "
                    "composite. No 0.4*latency + 0.3*cache + ... form "
                    "anywhere."),
            },
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "scorer_path_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path, payload


def _level_counts(counts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {
        "level_1_measured": 0,
        "level_1_operator_policy": 0,
        "level_2_derived": 0,
        "level_3_prior": 0,
        "level_4_prohibited_or_uncalibrated": 0,
        "missing": 0,
    }
    for cls, n in counts.items():
        lvl = signal_level_for(cls)
        if lvl is None:
            out["missing"] += n
        else:
            out[lvl] = out.get(lvl, 0) + n
    return out


def write_term_coverage_matrix():
    """Write the per-variant × per-term coverage matrix."""
    payload = {
        "doc_version": "constraint_scorer_term_coverage_matrix_v1",
        "audit_only": True,
        "shadow_only": True,
        "production_claim": False,
        "term_catalog": list(SCORER_TERM_CATALOG),
        "variants": {},
    }
    for variant in ALL_VARIANTS:
        # Map each variant to its term-coverage classification.
        if variant.name == "A_existing":
            kind = "existing"
        elif variant.name == "B_shadow_default_priors":
            kind = "shadow_default_priors"
        elif variant.name == "C_shadow_with_ttft_p50_prior":
            kind = "shadow_ttft_prior"
        elif variant.name == "D_shadow_with_cache_prefill":
            kind = "shadow_cache_prior"
        elif variant.name == "E_shadow_full":
            kind = "shadow_full"
        else:
            kind = "existing"
        coverage = term_coverage_for_scorer(kind)
        leveled = {term: {
            "classification": classification,
            "signal_level": signal_level_for(classification),
        } for term, classification in coverage.items()}
        payload["variants"][variant.name] = {
            "config": _config_to_dict(variant.config),
            "term_coverage": coverage,
            "term_signal_levels": leveled,
        }
    path = OUT_DIR / "term_coverage_matrix.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _config_to_dict(config) -> dict:
    return {
        "use_optimum_service_time": config.use_optimum_service_time,
        "use_ttft_p50_prior": config.use_ttft_p50_prior,
        "use_cache_prefill_savings": config.use_cache_prefill_savings,
        "use_per_gpu_hour_price": config.use_per_gpu_hour_price,
        "use_energy_term": config.use_energy_term,
        "use_migration_cache_loss": config.use_migration_cache_loss,
        "executable_in_real_cluster": config.executable_in_real_cluster,
    }


def main(argv=None) -> int:
    audit_path, _ = write_scorer_path_audit()
    matrix_path = write_term_coverage_matrix()
    print(f"[audit] wrote {audit_path}")
    print(f"[audit] wrote {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
