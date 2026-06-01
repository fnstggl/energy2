"""Economic Overlay Layer v1 — joins operational traces with public-data
calibrated economic signals.

Design contract (binding):

  Every numeric field on an `EconomicOverlayRecord` carries a `value_quality`
  label drawn from {measured, derived, prior, scenario_prior, missing}.
  Every derived field carries a `formula` string. No utility-weight composite
  exists anywhere in this module. No invented economic constant exists
  anywhere in this module. The audit
  `data/external/economic_overlay/source_coverage_matrix.json` is the
  human-readable proof of those properties.

  This is a SHADOW / OVERLAY layer. It does NOT modify production scheduler,
  residency, frontier, or scorer behaviour. The existing constraint scorer
  (see `aurelius.forecasting.constraint_shadow_scorer`) remains authoritative.

Public-data sources (Phase 1 inventory):

  Operational (Tier-2..5 HF traces): CARA / Optimum / AcmeTrace / SwissAI /
  CC-traces / Ejhusom.  Economic overlays: afhubbard/gpu-prices (B-class
  join_overlay_candidate, CC-BY-4.0, per PR #140), PJM Data Miner (real LIVE
  DA LMP, hourly, $/MWh — config via PJM_API_KEY), ERCOT / CAISO /
  ElectricityMaps / WattTime as scenario_prior tables when credentials are
  absent (see SCENARIO_OVERLAYS below).

This module does NOT call any external API itself. It consumes already-fetched
overlay tables (committed under data/external/economic_overlay/...). See
`scripts/build_economic_overlay_v1.py` for the bounded fetch driver.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
DEFAULT_SAMPLES_DIR = DEFAULT_OVERLAY_DIR / "economic_overlay_samples"

# ---------------------------------------------------------------------------
# Value-quality labels — binding vocabulary.
# ---------------------------------------------------------------------------

VALUE_QUALITY_LABELS = frozenset({
    "measured",        # Level 1 — direct observation in dataset/API/operator policy.
    "derived",         # Level 2 — transparent formula over level-1 / level-3 inputs.
    "prior",           # Level 3 — public benchmark or market prior (research only).
    "scenario_prior",  # Level 3 + scenario tag — region/market assumed, not joined.
    "missing",         # Level 4 — uncalibrated; do NOT fill with constants.
})


# ---------------------------------------------------------------------------
# Scenario overlays — used when a trace has no region / no live market data.
# These are EXPLICIT SCENARIO PRIORS, NOT MEASURED. Tests enforce labelling.
#
# Values are taken from the published market-level statistics in the related
# public documents (EIA, ISO/RTO reports). They are RECORDED as fixed
# scenario constants here only so the overlay can demonstrate end-to-end joins
# offline; production must replace them with a live feed.
# ---------------------------------------------------------------------------

SCENARIO_OVERLAYS = {
    "pjm_energy_overlay": {
        "market": "PJM",
        "region": "us-east",
        "price_per_kwh_usd_p50_scenario": 0.045,
        "value_quality": "scenario_prior",
        "source_note": (
            "PJM zonal LMP scenario midpoint; replace with live PJM Data "
            "Miner fetch via aurelius.ingestion.grid_apis.pjm in production."),
    },
    "ercot_energy_overlay": {
        "market": "ERCOT",
        "region": "us-south",
        "price_per_kwh_usd_p50_scenario": 0.038,
        "value_quality": "scenario_prior",
        "source_note": (
            "ERCOT SPP scenario midpoint; replace with live ERCOT API fetch "
            "in production (requires ERCOT_API_KEY + ERCOT_USERNAME + "
            "ERCOT_PASSWORD or pre-fetched ERCOT_ID_TOKEN)."),
    },
    "caiso_energy_overlay": {
        "market": "CAISO",
        "region": "us-west",
        "price_per_kwh_usd_p50_scenario": 0.072,
        "value_quality": "scenario_prior",
        "source_note": (
            "CAISO OASIS LMP scenario midpoint; replace with live CAISO "
            "fetch via aurelius.ingestion.grid_apis.caiso in production."),
    },
    "watttime_carbon_overlay": {
        "market": "WattTime",
        "region": "us-east",
        "carbon_intensity_g_per_kwh_scenario": 410.0,
        "value_quality": "scenario_prior",
        "source_note": (
            "WattTime MOER scenario midpoint for us-east; replace with live "
            "WattTime fetch via aurelius.ingestion.grid_apis.watttime in "
            "production (requires WATTTIME_USERNAME + WATTTIME_PASSWORD)."),
    },
    "no_operator_policy_overlay": {
        "market": None,
        "region": None,
        "value_quality": "missing",
        "source_note": (
            "Operator pricing policy not supplied — gpu_price_usd_per_hour, "
            "energy_price_per_kwh_usd, and carbon_price_per_kg_usd left "
            "missing; carbon_cost not computed."),
    },
}

# Public-list-price gpu overlay is loaded from disk (see load_gpu_price_overlay).
MARKET_PRICE_PUBLIC_GPU_OVERLAY_KEY = "market_price_public_gpu_overlay"


# ---------------------------------------------------------------------------
# Canonical record.
# ---------------------------------------------------------------------------


@dataclass
class EconomicOverlayRecord:
    """Per-(operational-trace-row) economic overlay.

    Every numeric field carries a per-field value_quality label in
    `value_quality_by_field` and a per-derived-field formula string in
    `formula_by_field`. Missing fields are stored as None (NOT 0.0)."""

    source_trace_id: str
    source_dataset_id: str
    model_id: Optional[str]
    gpu_type: Optional[str]
    gpu_count: Optional[int]
    region: Optional[str]
    zone: Optional[str]
    timestamp: Optional[str]

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None
    e2e_latency_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    throughput_tok_s: Optional[float] = None
    cache_reuse_pct: Optional[float] = None
    kv_utilization: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    gpu_power_w: Optional[float] = None
    energy_kwh: Optional[float] = None

    # Joined / overlaid economic inputs.
    electricity_price_usd_per_kwh: Optional[float] = None
    carbon_intensity_g_per_kwh: Optional[float] = None
    gpu_price_usd_per_hour: Optional[float] = None

    # Derived seconds / kWh.
    estimated_gpu_seconds: Optional[float] = None
    estimated_prefill_seconds: Optional[float] = None
    estimated_decode_seconds: Optional[float] = None

    # Derived costs.
    estimated_gpu_cost_usd: Optional[float] = None
    estimated_energy_cost_usd: Optional[float] = None
    estimated_carbon_kg: Optional[float] = None
    estimated_carbon_cost_usd: Optional[float] = None
    estimated_prefill_cost_usd: Optional[float] = None
    estimated_decode_cost_usd: Optional[float] = None
    estimated_cache_value_usd: Optional[float] = None
    estimated_migration_cost_usd: Optional[float] = None
    estimated_cold_start_cost_usd: Optional[float] = None
    estimated_memory_pressure_cost_usd: Optional[float] = None

    # SLA.
    sla_s: Optional[float] = None
    sla_met: Optional[bool] = None
    sla_safe_goodput: Optional[float] = None
    sla_safe_goodput_per_dollar: Optional[float] = None

    # Auditability.
    value_quality_by_field: dict[str, str] = field(default_factory=dict)
    formula_by_field: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    overlay_class: str = "cross_dataset_joined"  # or measured_same_record / scenario_prior

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Operator pricing policy (Level-1 slot; default-empty mirrors PR #139).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorPricingPolicy:
    """Operator-supplied $-denominated coefficients. ALL fields default to
    None — the overlay reports `missing` for any term that needs an absent
    field. This dataclass is FROZEN so a default-empty policy cannot be
    mutated to inject invented constants."""

    energy_price_per_kwh_usd: Optional[float] = None
    carbon_price_per_kg_usd: Optional[float] = None
    gpu_hour_price_per_type: Optional[dict[str, float]] = None
    operator_region: Optional[str] = None

    def gpu_price(self, gpu_type: Optional[str]) -> Optional[float]:
        if not gpu_type or not self.gpu_hour_price_per_type:
            return None
        return self.gpu_hour_price_per_type.get(gpu_type.lower())


# ---------------------------------------------------------------------------
# GPU price overlay loader.
# ---------------------------------------------------------------------------


def _gpu_family(gpu_type: Optional[str]) -> Optional[str]:
    if not gpu_type:
        return None
    t = str(gpu_type).strip().lower()
    for prefix in ("h200", "h100", "b200", "a100", "a10g", "a10",
                   "v100", "p100", "t4", "l40s", "l40", "l4", "gb10",
                   "rtxpro6000", "rtx6000"):
        if prefix in t:
            return prefix
    return t


@dataclass
class GPUPriceOverlay:
    """Public-list GPU rental price index. Built from
    `afhubbard/gpu-prices` (CC-BY-4.0). value_quality = `prior` for
    family-exact match, `prior_fuzzy_match` for nearest-family fallback,
    `missing` when no public listing exists."""

    rows: list[dict]
    snapshot_timestamp: Optional[str] = None
    source_dataset_id: str = "afhubbard/gpu-prices"
    source_license: str = "cc-by-4.0"

    @classmethod
    def load(cls, path: Path) -> "GPUPriceOverlay":
        rows = []
        snapshot = None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                snapshot = r.get("snapshot_timestamp") or snapshot
        return cls(rows=rows, snapshot_timestamp=snapshot)

    def lookup(self, *, gpu_type: Optional[str],
               region: Optional[str] = None,
               is_spot: Optional[bool] = None,
               provider: Optional[str] = None) -> dict[str, Any]:
        """Returns dict with `price_per_gpu_hour_usd` (or None), the matched
        provider/region/spot tuple, and `value_quality`.
        Selection priority:
          1. exact family + provider + region + is_spot
          2. exact family + region (any provider, on-demand)
          3. exact family on-demand (any region, any provider) → median
          4. nearest family fallback → median
          5. missing

        Memoized on the (gpu_type, region, is_spot, provider) key so an
        analysis-tier sweep of N records collapses to one computation per
        distinct key (the return contract is unchanged)."""
        cache = self.__dict__.setdefault("_lookup_cache", {})
        ck = (gpu_type, region, is_spot, provider)
        if ck in cache:
            return cache[ck]
        res = self._lookup_uncached(gpu_type=gpu_type, region=region,
                                    is_spot=is_spot, provider=provider)
        cache[ck] = res
        return res

    def _lookup_uncached(self, *, gpu_type: Optional[str],
                         region: Optional[str] = None,
                         is_spot: Optional[bool] = None,
                         provider: Optional[str] = None) -> dict[str, Any]:
        target_fam = _gpu_family(gpu_type)
        if target_fam is None:
            return {"price_per_gpu_hour_usd": None,
                    "value_quality": "missing",
                    "match_kind": "none",
                    "formula": "no_gpu_type"}

        def _match(rows, *, fam=None, prov=None, reg=None, spot=None):
            out = []
            for r in rows:
                if fam is not None and _gpu_family(r.get("gpu_type")) != fam:
                    continue
                if prov is not None and r.get("provider") != prov:
                    continue
                if reg is not None and r.get("region") != reg:
                    continue
                if spot is not None and bool(r.get("is_spot")) != bool(spot):
                    continue
                out.append(r)
            return out

        # Step 1 — most-specific.
        if provider and region and is_spot is not None:
            m = _match(self.rows, fam=target_fam, prov=provider,
                       reg=region, spot=is_spot)
            if m:
                return {"price_per_gpu_hour_usd": float(m[0]["price_per_gpu_hour_usd"]),
                        "value_quality": "prior",
                        "match_kind": "exact_family_provider_region_spot",
                        "matched_provider": m[0].get("provider"),
                        "matched_region": m[0].get("region"),
                        "formula": "afhubbard_gpu_prices_exact"}
        # Step 2 — family + region, on-demand.
        if region:
            m = _match(self.rows, fam=target_fam, reg=region, spot=False)
            if m:
                med = _median([r["price_per_gpu_hour_usd"] for r in m])
                return {"price_per_gpu_hour_usd": med,
                        "value_quality": "prior",
                        "match_kind": "exact_family_region_ondemand_median",
                        "matched_n": len(m),
                        "formula": "median(afhubbard_gpu_prices[family==X, region==R, spot==False])"}
        # Step 3 — family-only on-demand median (any region).
        m = _match(self.rows, fam=target_fam, spot=False)
        if m:
            med = _median([r["price_per_gpu_hour_usd"] for r in m])
            return {"price_per_gpu_hour_usd": med,
                    "value_quality": "prior",
                    "match_kind": "exact_family_global_ondemand_median",
                    "matched_n": len(m),
                    "formula": "median(afhubbard_gpu_prices[family==X, spot==False])"}
        # Step 4 — nearest-family fuzzy.
        # Order by descending family memory size (proxy for capability).
        rank = {"h200": 6, "h100": 5, "a100": 4, "a10g": 3, "a10": 3,
                "l40s": 3, "v100": 2, "p100": 2, "l40": 2, "l4": 1,
                "t4": 1, "gb10": 5, "rtxpro6000": 3, "b200": 6}
        target_rank = rank.get(target_fam, 2)
        ranked = sorted(
            [(abs(rank.get(_gpu_family(r.get("gpu_type")) or "", 2)
                  - target_rank), r) for r in self.rows
             if not bool(r.get("is_spot"))],
            key=lambda x: x[0],
        )
        if ranked:
            best_dist = ranked[0][0]
            same_dist = [r for d, r in ranked if d == best_dist]
            med = _median([r["price_per_gpu_hour_usd"] for r in same_dist])
            return {"price_per_gpu_hour_usd": med,
                    "value_quality": "prior_fuzzy_match",
                    "match_kind": "nearest_family",
                    "fuzzy_distance": best_dist,
                    "matched_n": len(same_dist),
                    "formula": "median(afhubbard_gpu_prices[family in nearest_capability_tier])"}
        return {"price_per_gpu_hour_usd": None,
                "value_quality": "missing",
                "match_kind": "none",
                "formula": "no_public_listing"}


def _median(xs: list[float]) -> float:
    s = sorted(float(x) for x in xs if x is not None)
    if not s:
        return 0.0
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------
# Energy / carbon overlays.
# ---------------------------------------------------------------------------


@dataclass
class EnergyPriceOverlay:
    """Time-indexed $/kWh by market. Loaded from PJM live samples
    (measured) and/or scenario tables (scenario_prior)."""

    market: str
    region: str
    rows: list[dict]  # each: {timestamp, price_per_mwh, source}
    value_quality: str  # measured | scenario_prior

    @classmethod
    def load_pjm(cls, path: Path, *, market: str = "PJM",
                 region: str = "us-east") -> "EnergyPriceOverlay":
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return cls(market=market, region=region, rows=rows,
                   value_quality="measured")

    @classmethod
    def scenario(cls, key: str) -> "EnergyPriceOverlay":
        sp = SCENARIO_OVERLAYS[key]
        if "price_per_kwh_usd_p50_scenario" not in sp:
            # `no_operator_policy_overlay` and any other non-price scenario
            # legitimately have no electricity price — return an empty
            # overlay so the lookup reports `missing`.
            return cls(market=sp.get("market") or key,
                       region=sp.get("region") or "",
                       rows=[], value_quality="missing")
        rows = [{
            "timestamp": None,
            "price_per_kwh": sp["price_per_kwh_usd_p50_scenario"],
            "source": f"scenario:{sp['market']}",
        }]
        return cls(market=sp["market"], region=sp["region"], rows=rows,
                   value_quality="scenario_prior")

    def lookup(self, *, timestamp: Optional[str] = None) -> dict[str, Any]:
        if not self.rows:
            return {"price_per_kwh_usd": None,
                    "value_quality": "missing",
                    "formula": "no_rows"}
        if self.value_quality == "scenario_prior":
            return {"price_per_kwh_usd": self.rows[0]["price_per_kwh"],
                    "value_quality": "scenario_prior",
                    "formula": f"{self.market}_scenario_midpoint"}
        # Measured PJM rows are hourly LMP in $/MWh — pick nearest by
        # timestamp (or median if no timestamp on the trace).
        if not timestamp:
            mwh = _median([r["price_per_mwh"] for r in self.rows])
            return {"price_per_kwh_usd": mwh / 1000.0,
                    "value_quality": "measured",
                    "formula": "median(pjm_da_lmp) / 1000",
                    "n_rows": len(self.rows)}
        # nearest-prior lookup. Precompute the price-series ordinals once
        # (cached on the instance) so an N-record sweep parses each series
        # timestamp once rather than once-per-record.
        idx = self.__dict__.get("_ord_index")
        if idx is None:
            idx = [(_iso_to_ord(r.get("timestamp")), r) for r in self.rows]
            self.__dict__["_ord_index"] = idx
        target_ord = _iso_to_ord(timestamp)
        nearest = min(idx, key=lambda pr: abs(pr[0] - target_ord))[1]
        return {"price_per_kwh_usd": float(nearest["price_per_mwh"]) / 1000.0,
                "value_quality": "measured",
                "formula": f"pjm_da_lmp[t≈{nearest['timestamp']}] / 1000",
                "matched_timestamp": nearest.get("timestamp")}


def _iso_to_ord(ts) -> float:
    if ts is None:
        return 0.0
    try:
        from datetime import datetime
        if isinstance(ts, (int, float)):
            return float(ts)
        if hasattr(ts, "timestamp"):
            return ts.timestamp()
        s = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return 0.0


@dataclass
class CarbonOverlay:
    """Carbon intensity (g CO2 / kWh) overlay. WattTime/ElectricityMaps
    scenario_prior in this PR; replace with live fetch in production."""

    market: str
    region: str
    carbon_intensity_g_per_kwh: Optional[float]
    value_quality: str

    @classmethod
    def scenario(cls, key: str = "watttime_carbon_overlay") -> "CarbonOverlay":
        sp = SCENARIO_OVERLAYS[key]
        ci = sp.get("carbon_intensity_g_per_kwh_scenario")
        # Non-carbon scenarios (e.g. no_operator_policy_overlay) legitimately
        # have no carbon intensity → report missing.
        vq = sp.get("value_quality", "missing") if ci is not None else "missing"
        return cls(market=sp.get("market") or key,
                   region=sp.get("region") or "",
                   carbon_intensity_g_per_kwh=ci,
                   value_quality=vq)

    def lookup(self) -> dict[str, Any]:
        return {"carbon_intensity_g_per_kwh": self.carbon_intensity_g_per_kwh,
                "value_quality": self.value_quality,
                "formula": f"{self.market}_scenario_midpoint"}


# ---------------------------------------------------------------------------
# Builder.
# ---------------------------------------------------------------------------


@dataclass
class OverlayBuilderConfig:
    energy_market: str = "pjm_energy_overlay"
    carbon_market: str = "watttime_carbon_overlay"
    use_live_pjm: bool = True
    pjm_path: Optional[Path] = None
    gpu_price_path: Optional[Path] = None
    operator_policy: OperatorPricingPolicy = field(
        default_factory=OperatorPricingPolicy)
    # When True, a measured market price (e.g. live PJM/CAISO LMP) applied to
    # a region-LESS operational trace is relabelled `scenario_prior`: the
    # *price series* is real measured market data, but the *region assignment*
    # is an assumption (mission §4.B fallback). Default False preserves the
    # original v1 semantics (the smoke-test builder + its 66 tests).
    scenario_region_assignment: bool = False
    energy_market_label: Optional[str] = None  # e.g. "PJM" / "CAISO"


# Aurelius region -> energy market, used to decide when a trace's region
# makes a measured market join legitimate (vs a scenario assignment).
REGION_TO_MARKET = {
    "us-east": "PJM",
    "us-east-1": "PJM",
    "us-west": "CAISO",
    "us-west-1": "CAISO",
    "us-west-2": "CAISO",
    "us-south": "ERCOT",
}


class OverlayBuilder:
    """Joins operational trace rows to economic overlay tables and emits
    `EconomicOverlayRecord`s.

    Inputs:
      operational rows — dicts with at least
        `{source_trace_id, source_dataset_id, gpu_type}` and any of the
        operational signals listed on `EconomicOverlayRecord`.
      `OverlayBuilderConfig` — paths + policy + scenario selection.
    """

    def __init__(self, cfg: OverlayBuilderConfig):
        self.cfg = cfg
        self.gpu = (GPUPriceOverlay.load(cfg.gpu_price_path)
                    if cfg.gpu_price_path and Path(cfg.gpu_price_path).exists()
                    else GPUPriceOverlay(rows=[]))
        if cfg.use_live_pjm and cfg.pjm_path and Path(cfg.pjm_path).exists():
            self.energy = EnergyPriceOverlay.load_pjm(cfg.pjm_path)
        else:
            self.energy = EnergyPriceOverlay.scenario(cfg.energy_market)
        self.carbon = CarbonOverlay.scenario(cfg.carbon_market)

    # ── Per-record builder ───────────────────────────────────────────────

    def build_record(self, op_row: dict) -> EconomicOverlayRecord:
        rec = EconomicOverlayRecord(
            source_trace_id=str(op_row.get("source_trace_id",
                                           op_row.get("request_id", ""))),
            source_dataset_id=str(op_row.get("source_dataset_id",
                                             "unknown")),
            model_id=op_row.get("model_id"),
            gpu_type=op_row.get("gpu_type"),
            gpu_count=op_row.get("gpu_count"),
            region=op_row.get("region"),
            zone=op_row.get("zone"),
            timestamp=op_row.get("timestamp"),
            prompt_tokens=op_row.get("prompt_tokens"),
            output_tokens=op_row.get("output_tokens"),
            ttft_s=op_row.get("ttft_s"),
            tpot_s=op_row.get("tpot_s"),
            e2e_latency_s=op_row.get("e2e_latency_s"),
            queue_wait_s=op_row.get("queue_wait_s"),
            throughput_tok_s=op_row.get("throughput_tok_s"),
            cache_reuse_pct=op_row.get("cache_reuse_pct"),
            kv_utilization=op_row.get("kv_utilization"),
            peak_vram_gb=op_row.get("peak_vram_gb"),
            gpu_power_w=op_row.get("gpu_power_w"),
            energy_kwh=op_row.get("energy_kwh"),
            sla_s=op_row.get("sla_s"),
        )
        vq = rec.value_quality_by_field
        fm = rec.formula_by_field

        for k in ("ttft_s", "tpot_s", "e2e_latency_s", "queue_wait_s",
                  "throughput_tok_s", "cache_reuse_pct", "kv_utilization",
                  "peak_vram_gb", "gpu_power_w", "energy_kwh",
                  "prompt_tokens", "output_tokens", "sla_s"):
            if getattr(rec, k) is not None:
                vq[k] = "measured"

        # ── A. GPU price ────────────────────────────────────────────────
        operator_gpu = self.cfg.operator_policy.gpu_price(rec.gpu_type)
        if operator_gpu is not None:
            rec.gpu_price_usd_per_hour = float(operator_gpu)
            vq["gpu_price_usd_per_hour"] = "measured"
            fm["gpu_price_usd_per_hour"] = "operator_policy.gpu_hour_price_per_type"
        else:
            res = self.gpu.lookup(gpu_type=rec.gpu_type, region=rec.region,
                                  is_spot=op_row.get("is_spot"),
                                  provider=op_row.get("provider"))
            rec.gpu_price_usd_per_hour = res["price_per_gpu_hour_usd"]
            vq["gpu_price_usd_per_hour"] = res["value_quality"]
            fm["gpu_price_usd_per_hour"] = res["formula"]

        # ── B. Energy price ─────────────────────────────────────────────
        if self.cfg.operator_policy.energy_price_per_kwh_usd is not None:
            rec.electricity_price_usd_per_kwh = float(
                self.cfg.operator_policy.energy_price_per_kwh_usd)
            vq["electricity_price_usd_per_kwh"] = "measured"
            fm["electricity_price_usd_per_kwh"] = "operator_policy.energy_price_per_kwh_usd"
        else:
            res = self.energy.lookup(timestamp=rec.timestamp)
            rec.electricity_price_usd_per_kwh = res.get("price_per_kwh_usd")
            energy_vq = res.get("value_quality", "missing")
            energy_fm = res.get("formula", "")
            # Scenario-region downgrade: measured market price applied to a
            # region-less trace is honestly a scenario_prior (real price,
            # assumed region). Only downgrades `measured`; never upgrades.
            if (self.cfg.scenario_region_assignment
                    and energy_vq == "measured"
                    and not rec.region):
                mkt = self.cfg.energy_market_label or self.energy.market
                energy_vq = "scenario_prior"
                energy_fm = (f"{mkt} measured market price applied to "
                             "region-less trace (scenario region assignment)")
            vq["electricity_price_usd_per_kwh"] = energy_vq
            fm["electricity_price_usd_per_kwh"] = energy_fm

        # ── C. Carbon intensity ─────────────────────────────────────────
        cres = self.carbon.lookup()
        rec.carbon_intensity_g_per_kwh = cres["carbon_intensity_g_per_kwh"]
        vq["carbon_intensity_g_per_kwh"] = cres["value_quality"]
        fm["carbon_intensity_g_per_kwh"] = cres["formula"]

        # ── seconds estimates ───────────────────────────────────────────
        if rec.e2e_latency_s is not None:
            rec.estimated_gpu_seconds = float(rec.e2e_latency_s)
            vq["estimated_gpu_seconds"] = "measured"
            fm["estimated_gpu_seconds"] = "e2e_latency_s"
        elif rec.ttft_s is not None and rec.tpot_s is not None \
                and rec.output_tokens:
            rec.estimated_gpu_seconds = (
                float(rec.ttft_s) + float(rec.tpot_s) * float(rec.output_tokens))
            vq["estimated_gpu_seconds"] = "derived"
            fm["estimated_gpu_seconds"] = "ttft_s + tpot_s * output_tokens"

        if rec.ttft_s is not None:
            rec.estimated_prefill_seconds = float(rec.ttft_s)
            vq["estimated_prefill_seconds"] = "measured"
            fm["estimated_prefill_seconds"] = "ttft_s"
        if rec.tpot_s is not None and rec.output_tokens:
            rec.estimated_decode_seconds = float(rec.tpot_s) * float(
                rec.output_tokens)
            vq["estimated_decode_seconds"] = "derived"
            fm["estimated_decode_seconds"] = "tpot_s * output_tokens"

        # ── D. Energy per request ───────────────────────────────────────
        if rec.energy_kwh is None and rec.gpu_power_w is not None \
                and rec.estimated_gpu_seconds is not None:
            rec.energy_kwh = (float(rec.gpu_power_w)
                              * float(rec.estimated_gpu_seconds) / 3_600_000.0)
            vq["energy_kwh"] = "derived_from_power_prior"
            fm["energy_kwh"] = "gpu_power_w * estimated_gpu_seconds / 3_600_000"

        # ── E. GPU cost ─────────────────────────────────────────────────
        if rec.estimated_gpu_seconds is not None \
                and rec.gpu_price_usd_per_hour is not None:
            n_gpus = float(rec.gpu_count or 1)
            rec.estimated_gpu_cost_usd = (
                float(rec.gpu_price_usd_per_hour) * n_gpus
                * float(rec.estimated_gpu_seconds) / 3600.0)
            vq["estimated_gpu_cost_usd"] = (
                "derived" if vq.get("gpu_price_usd_per_hour") != "measured"
                else "measured_input_derived_formula")
            fm["estimated_gpu_cost_usd"] = (
                "gpu_price_usd_per_hour * gpu_count "
                "* estimated_gpu_seconds / 3600")

        # ── Prefill / decode cost ───────────────────────────────────────
        gpu_psec = (float(rec.gpu_price_usd_per_hour) / 3600.0
                    if rec.gpu_price_usd_per_hour else None)
        if gpu_psec is not None and rec.estimated_prefill_seconds is not None:
            rec.estimated_prefill_cost_usd = (
                gpu_psec * float(rec.estimated_prefill_seconds)
                * float(rec.gpu_count or 1))
            vq["estimated_prefill_cost_usd"] = "derived"
            fm["estimated_prefill_cost_usd"] = (
                "gpu_price_usd_per_hour / 3600 * estimated_prefill_seconds "
                "* gpu_count")
        if gpu_psec is not None and rec.estimated_decode_seconds is not None:
            rec.estimated_decode_cost_usd = (
                gpu_psec * float(rec.estimated_decode_seconds)
                * float(rec.gpu_count or 1))
            vq["estimated_decode_cost_usd"] = "derived"
            fm["estimated_decode_cost_usd"] = (
                "gpu_price_usd_per_hour / 3600 * estimated_decode_seconds "
                "* gpu_count")

        # ── F. Energy cost ──────────────────────────────────────────────
        if rec.energy_kwh is not None \
                and rec.electricity_price_usd_per_kwh is not None:
            rec.estimated_energy_cost_usd = (
                float(rec.energy_kwh)
                * float(rec.electricity_price_usd_per_kwh))
            vq["estimated_energy_cost_usd"] = (
                "scenario_prior"
                if vq.get("electricity_price_usd_per_kwh") == "scenario_prior"
                else "derived")
            fm["estimated_energy_cost_usd"] = (
                "energy_kwh * electricity_price_usd_per_kwh")

        # ── G. Carbon kg + carbon cost ──────────────────────────────────
        if rec.energy_kwh is not None and rec.carbon_intensity_g_per_kwh:
            rec.estimated_carbon_kg = (
                float(rec.energy_kwh)
                * float(rec.carbon_intensity_g_per_kwh) / 1000.0)
            vq["estimated_carbon_kg"] = (
                "scenario_prior"
                if vq.get("carbon_intensity_g_per_kwh") == "scenario_prior"
                else "derived")
            fm["estimated_carbon_kg"] = (
                "energy_kwh * carbon_intensity_g_per_kwh / 1000")
        if rec.estimated_carbon_kg is not None \
                and self.cfg.operator_policy.carbon_price_per_kg_usd is not None:
            rec.estimated_carbon_cost_usd = (
                float(rec.estimated_carbon_kg)
                * float(self.cfg.operator_policy.carbon_price_per_kg_usd))
            vq["estimated_carbon_cost_usd"] = "derived"
            fm["estimated_carbon_cost_usd"] = (
                "estimated_carbon_kg * operator_policy.carbon_price_per_kg_usd")
        else:
            vq["estimated_carbon_cost_usd"] = "missing"

        # ── H. Cache value ──────────────────────────────────────────────
        if rec.cache_reuse_pct is not None \
                and rec.estimated_prefill_seconds is not None \
                and gpu_psec is not None:
            rec.estimated_cache_value_usd = (
                float(rec.cache_reuse_pct)
                * float(rec.estimated_prefill_seconds) * gpu_psec
                * float(rec.gpu_count or 1))
            vq["estimated_cache_value_usd"] = "derived"
            fm["estimated_cache_value_usd"] = (
                "cache_reuse_pct * estimated_prefill_seconds "
                "* gpu_price_usd_per_hour / 3600 * gpu_count")
        else:
            vq["estimated_cache_value_usd"] = "missing"

        # ── I. Migration cost ───────────────────────────────────────────
        cache_loss_pct = op_row.get("cache_loss_pct")
        if cache_loss_pct is not None \
                and rec.estimated_prefill_seconds is not None \
                and gpu_psec is not None:
            rec.estimated_migration_cost_usd = (
                float(cache_loss_pct)
                * float(rec.estimated_prefill_seconds) * gpu_psec
                * float(rec.gpu_count or 1))
            vq["estimated_migration_cost_usd"] = "derived"
            fm["estimated_migration_cost_usd"] = (
                "cache_loss_pct * rebuild_prefill_seconds "
                "* gpu_price_usd_per_hour / 3600 * gpu_count")
        else:
            vq["estimated_migration_cost_usd"] = "missing"

        # ── J. Cold-start cost ──────────────────────────────────────────
        model_load_s = op_row.get("model_load_duration_s")
        if model_load_s is not None and gpu_psec is not None:
            rec.estimated_cold_start_cost_usd = (
                float(model_load_s) * gpu_psec * float(rec.gpu_count or 1))
            vq["estimated_cold_start_cost_usd"] = (
                "derived" if op_row.get("model_load_source") == "measured"
                else "proxy")
            fm["estimated_cold_start_cost_usd"] = (
                "model_load_duration_s * gpu_price_usd_per_hour / 3600 "
                "* gpu_count")
        else:
            vq["estimated_cold_start_cost_usd"] = "missing"

        # ── SLA-safe goodput / $ ────────────────────────────────────────
        if rec.sla_s is not None and rec.e2e_latency_s is not None:
            rec.sla_met = bool(rec.e2e_latency_s <= rec.sla_s)
            vq["sla_met"] = "derived"
            fm["sla_met"] = "e2e_latency_s <= sla_s"
            useful = float(rec.output_tokens) if rec.output_tokens else 1.0
            rec.sla_safe_goodput = useful if rec.sla_met else 0.0
            vq["sla_safe_goodput"] = "derived"
            fm["sla_safe_goodput"] = (
                "output_tokens if sla_met else 0 (1 if output_tokens missing)")

            cost = 0.0
            cost_parts = []
            for term, label in (
                (rec.estimated_gpu_cost_usd, "gpu_cost"),
                (rec.estimated_energy_cost_usd, "energy_cost"),
                (rec.estimated_migration_cost_usd, "migration_cost"),
                (rec.estimated_cold_start_cost_usd, "cold_start_cost"),
            ):
                if term is not None:
                    cost += float(term)
                    cost_parts.append(label)
            if rec.estimated_cache_value_usd is not None:
                cost -= float(rec.estimated_cache_value_usd)
                cost_parts.append("-cache_value")
            if cost > 0:
                rec.sla_safe_goodput_per_dollar = rec.sla_safe_goodput / cost
                vq["sla_safe_goodput_per_dollar"] = "derived"
                fm["sla_safe_goodput_per_dollar"] = (
                    "sla_safe_goodput / (" + " + ".join(cost_parts) + ")")
            else:
                vq["sla_safe_goodput_per_dollar"] = "missing"

        # ── overlay class ───────────────────────────────────────────────
        rec.overlay_class = _classify_overlay(rec)

        # ── limitations ─────────────────────────────────────────────────
        for k, q in vq.items():
            if q == "scenario_prior":
                rec.limitations.append(
                    f"{k} comes from a scenario_prior; do NOT treat as "
                    "operator truth.")
            if q == "missing" and k.startswith("estimated_"):
                rec.limitations.append(
                    f"{k} not computed — required inputs missing.")
            if q == "prior_fuzzy_match":
                rec.limitations.append(
                    f"{k} matched on nearest GPU family, not exact family.")
        return rec

    def build(self, op_rows: Iterable[dict]) -> list[EconomicOverlayRecord]:
        return [self.build_record(r) for r in op_rows]


def _classify_overlay(rec: EconomicOverlayRecord) -> str:
    """Three result classes per mission §6 — never mixed in one headline.

    Classification is by the value-quality of the cost terms ACTUALLY
    realized in this row's `sla_safe_goodput_per_dollar` denominator
    (gpu_cost, energy_cost, cache_value, migration_cost, cold_start_cost) —
    not by overlays that were merely looked up. A scenario energy price that
    never enters the cost (because `energy_kwh` is missing) does NOT make the
    row scenario_prior. `carbon_cost` is excluded because it is missing
    whenever the operator carbon price is missing.

    - measured_same_record: every realized cost input is `measured`
      (operator-supplied / same-record measured signal).
    - scenario_prior: at least one realized cost term is `scenario_prior`.
    - cross_dataset_joined: otherwise (prior / derived public-data joins,
      no scenario term realized).
    """
    vq = rec.value_quality_by_field
    realized = []
    if rec.estimated_gpu_cost_usd is not None:
        realized.append(vq.get("gpu_price_usd_per_hour"))
    if rec.estimated_energy_cost_usd is not None:
        realized.append(vq.get("electricity_price_usd_per_kwh"))
        realized.append(vq.get("energy_kwh"))
    if rec.estimated_cache_value_usd is not None:
        realized.append(vq.get("estimated_cache_value_usd"))
    if rec.estimated_migration_cost_usd is not None:
        realized.append(vq.get("estimated_migration_cost_usd"))
    if rec.estimated_cold_start_cost_usd is not None:
        realized.append(vq.get("estimated_cold_start_cost_usd"))

    if realized and all(q == "measured" for q in realized):
        return "measured_same_record"
    if any(q == "scenario_prior" for q in realized):
        return "scenario_prior"
    return "cross_dataset_joined"


# ---------------------------------------------------------------------------
# Aggregator for evaluation reports.
# ---------------------------------------------------------------------------


def summarise(records: list[EconomicOverlayRecord]) -> dict:
    if not records:
        return {"n": 0}
    by_class = {"measured_same_record": [], "cross_dataset_joined": [],
                "scenario_prior": []}
    for r in records:
        by_class.setdefault(r.overlay_class, []).append(r)
    out = {"n": len(records), "by_overlay_class":
        {k: len(v) for k, v in by_class.items()}}
    fields = [
        "estimated_gpu_cost_usd", "estimated_energy_cost_usd",
        "estimated_carbon_kg", "estimated_carbon_cost_usd",
        "estimated_cache_value_usd", "estimated_migration_cost_usd",
        "estimated_cold_start_cost_usd", "estimated_prefill_cost_usd",
        "estimated_decode_cost_usd", "sla_safe_goodput_per_dollar",
    ]
    field_q = {}
    for k in fields:
        qs = [r.value_quality_by_field.get(k, "missing") for r in records]
        field_q[k] = {
            "n_total": len(qs),
            "n_missing": sum(1 for q in qs if q == "missing"),
            "n_measured": sum(1 for q in qs if q == "measured"),
            "n_derived": sum(1 for q in qs
                             if q in ("derived",
                                      "measured_input_derived_formula")),
            "n_prior": sum(1 for q in qs
                           if q in ("prior", "prior_fuzzy_match")),
            "n_scenario_prior": sum(1 for q in qs if q == "scenario_prior"),
        }
    out["field_quality_breakdown"] = field_q
    return out
