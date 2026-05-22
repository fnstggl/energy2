"""Canonical region registry for Aurelius.

Single place that maps a canonical Aurelius region (e.g. ``us-west``) to:

* the ISO/TSO market/operator and its source-region code (source of truth),
* the Electricity Maps zone key (optional aggregator/fallback),
* the carbon-provider zone keys (Electricity Maps / WattTime),
* cloud-provider region aliases (so a workload pinned to ``aws:us-east-1``
  can be resolved to a grid region),
* a confidence level for each mapping.

Design rules
------------
* Source-of-truth first: ``iso`` / ``source_region`` point at the operator that
  publishes settled prices. Electricity Maps is recorded as an *aggregator*
  zone, never as the price authority for US ISOs (it does not publish US
  wholesale prices — see docs/ELECTRICITYMAPS_CONTRIB_AUDIT.md).
* No silent guessing. Every mapping carries ``confidence``; uncertain ones are
  ``Confidence.LOW`` and explained in ``notes``.
* The cloud-region aliases below are a small, hand-verified subset adapted from
  the public Electricity Maps contrib ``config/data_centers/data_centers.json``
  (which is factual cloud-region geography). It is NOT copied wholesale and is
  not a runtime dependency.

Electricity Maps zone keys follow their published naming (US-CAL-CISO etc.);
these are factual identifiers, not copyrightable code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class Confidence:
    HIGH = "high"      # operator-confirmed, validated against real data
    MEDIUM = "medium"  # plausible 1:1 mapping, not yet validated end-to-end
    LOW = "low"        # approximate / partial coverage — verify before use

    ALL = frozenset({HIGH, MEDIUM, LOW})


@dataclass(frozen=True)
class RegionMapping:
    """Everything Aurelius knows about how to source data for one region."""

    canonical_region: str
    iso: str                       # market/operator short name (CAISO, PJM, ...)
    operator: str                  # full operator name
    source_region: str             # the ISO's own node/hub/zone code
    electricitymaps_zone: Optional[str]  # EM aggregator zone key, if any
    carbon_zones: dict[str, str]   # {carbon_provider: zone_key}
    cloud_aliases: dict[str, tuple[str, ...]]  # {provider: (region, ...)}
    timezone: str
    price_is_nodal_lmp: bool       # True nodal LMP vs zonal/hub/country price
    confidence: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.confidence not in Confidence.ALL:
            raise ValueError(f"Unknown confidence: {self.confidence!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# US ISO/TSO regions. Electricity Maps zone keys are taken from the public
# contrib config/zones/ filenames. Cloud aliases are the verified subset from
# config/data_centers/data_centers.json restricted to these grid zones.

REGION_REGISTRY: dict[str, RegionMapping] = {
    "us-west": RegionMapping(
        canonical_region="us-west",
        iso="CAISO",
        operator="California ISO",
        source_region="TH_NP15_GEN-APND",
        electricitymaps_zone="US-CAL-CISO",
        carbon_zones={"electricitymaps": "US-CAL-CISO", "watttime": "CAISO_NP15"},
        cloud_aliases={
            "aws": ("us-west-1",),
            "gcp": ("us-west2",),
            "azure": ("westus",),
        },
        timezone="America/Los_Angeles",
        price_is_nodal_lmp=True,
        confidence=Confidence.HIGH,
        notes=(
            "CAISO NP15 trading-hub LMP via OASIS (no auth). EM zone covers the "
            "whole CISO balancing area for carbon; CAISO has many other zones "
            "(SP15, ZP26) not represented by a single canonical region."
        ),
    ),
    "us-east": RegionMapping(
        canonical_region="us-east",
        iso="PJM",
        operator="PJM Interconnection",
        source_region="PJM-RTO (pnode_id=1)",
        electricitymaps_zone="US-MIDA-PJM",
        carbon_zones={"electricitymaps": "US-MIDA-PJM", "watttime": "PJM_DOM"},
        cloud_aliases={
            "aws": ("us-east-1", "us-east-2"),
            "gcp": ("us-east4", "us-east5"),
            "azure": ("eastus", "eastus2", "northcentralus"),
        },
        timezone="US/Eastern",
        price_is_nodal_lmp=True,
        confidence=Confidence.HIGH,
        notes=(
            "PJM-RTO system-aggregate LMP via PJM Data Miner (PJM_API_KEY). PJM "
            "spans 13 states; aws us-east-1/us-east-2 and azure eastus all fall "
            "inside the PJM footprint per EM data_centers mapping."
        ),
    ),
    "us-south": RegionMapping(
        canonical_region="us-south",
        iso="ERCOT",
        operator="Electric Reliability Council of Texas",
        source_region="HB_HOUSTON",
        electricitymaps_zone="US-TEX-ERCO",
        carbon_zones={"electricitymaps": "US-TEX-ERCO", "watttime": "ERCOT_HOUSTON"},
        cloud_aliases={
            "gcp": ("us-south1",),
            "azure": ("southcentralus",),
        },
        timezone="US/Central",
        price_is_nodal_lmp=False,
        confidence=Confidence.HIGH,
        notes=(
            "ERCOT publishes Settlement Point Prices (SPP) at HB_HOUSTON, not "
            "LMP — price_is_nodal_lmp=False. EM does not publish ERCOT prices. "
            "No AWS region maps to ERCOT in EM data_centers (aws us-south not "
            "present); azure southcentralus / gcp us-south1 are in Texas."
        ),
    ),
    "us-central": RegionMapping(
        canonical_region="us-central",
        iso="SPP",
        operator="Southwest Power Pool",
        source_region="SPP-system",
        electricitymaps_zone="US-CENT-SWPP",
        carbon_zones={"electricitymaps": "US-CENT-SWPP"},
        cloud_aliases={"gcp": ("us-central1",)},
        timezone="US/Central",
        price_is_nodal_lmp=True,
        confidence=Confidence.LOW,
        notes=(
            "SPP Integrated Marketplace LMP is NOT yet implemented in Aurelius "
            "(requires SPP Marketplace registration). Carbon via EM US-CENT-SWPP "
            "only. gcp us-central1 maps to SWPP per EM data_centers. Mark LOW "
            "until a real SPP price source is wired in."
        ),
    ),
    "us-north": RegionMapping(
        canonical_region="us-north",
        iso="MISO",
        operator="Midcontinent ISO",
        source_region="MISO-system",
        electricitymaps_zone="US-MIDW-MISO",
        carbon_zones={"electricitymaps": "US-MIDW-MISO", "watttime": "MISO_INDIANAPOLIS"},
        cloud_aliases={"azure": ("centralus",)},
        timezone="US/Central",
        price_is_nodal_lmp=True,
        confidence=Confidence.LOW,
        notes=(
            "MISO LMP is NOT yet implemented (no public unauthenticated API; "
            "registration at misoenergy.org required). Carbon via EM "
            "US-MIDW-MISO only. azure centralus maps to MISO per EM data_centers."
        ),
    ),
    "us-newengland": RegionMapping(
        canonical_region="us-newengland",
        iso="ISO-NE",
        operator="ISO New England",
        source_region="ISONE-system",
        electricitymaps_zone="US-NE-ISNE",
        carbon_zones={"electricitymaps": "US-NE-ISNE"},
        cloud_aliases={},
        timezone="US/Eastern",
        price_is_nodal_lmp=True,
        confidence=Confidence.LOW,
        notes=(
            "ISO-NE LMP not yet implemented. No cloud region maps to ISO-NE in "
            "EM data_centers. Carbon via EM US-NE-ISNE only."
        ),
    ),
    "us-nyiso": RegionMapping(
        canonical_region="us-nyiso",
        iso="NYISO",
        operator="New York ISO",
        source_region="NYISO-zones",
        electricitymaps_zone="US-NY-NYIS",
        carbon_zones={"electricitymaps": "US-NY-NYIS"},
        cloud_aliases={},
        timezone="US/Eastern",
        price_is_nodal_lmp=True,
        confidence=Confidence.LOW,
        notes=(
            "NYISO LMP not yet implemented. No cloud region maps to NYISO in EM "
            "data_centers. Carbon via EM US-NY-NYIS only."
        ),
    ),
    # ----- ENTSO-E (Europe). Source of truth = ENTSO-E Transparency Platform,
    # which is also where Electricity Maps' EU price parser reads from.
    "eu-west": RegionMapping(
        canonical_region="eu-west",
        iso="ENTSO-E",
        operator="ENTSO-E / EnBW TSO (DE)",
        source_region="10YDE-ENBW-----N",
        electricitymaps_zone="DE",
        carbon_zones={"electricitymaps": "DE"},
        cloud_aliases={"gcp": ("europe-west3",), "azure": ("germanywestcentral",)},
        timezone="Europe/Berlin",
        price_is_nodal_lmp=False,
        confidence=Confidence.MEDIUM,
        notes=(
            "Germany bidding-zone day-ahead price (EUR/MWh) via ENTSO-E "
            "(ENTSOE_API_KEY). Bidding-zone price, not nodal LMP. EM zone 'DE' "
            "matches. Cloud aliases approximate (verify before routing)."
        ),
    ),
    "eu-north": RegionMapping(
        canonical_region="eu-north",
        iso="ENTSO-E",
        operator="ENTSO-E / Statnett TSO (NO)",
        source_region="10YNO-1--------2",
        electricitymaps_zone="NO-NO1",
        carbon_zones={"electricitymaps": "NO-NO1"},
        cloud_aliases={"azure": ("norwayeast",)},
        timezone="Europe/Oslo",
        price_is_nodal_lmp=False,
        confidence=Confidence.MEDIUM,
        notes=(
            "Norway NO1 bidding-zone day-ahead price (EUR/MWh) via ENTSO-E. "
            "Bidding-zone price, not nodal LMP."
        ),
    ),
    "eu-central": RegionMapping(
        canonical_region="eu-central",
        iso="ENTSO-E",
        operator="ENTSO-E / RTE TSO (FR)",
        source_region="10YFR-RTE------C",
        electricitymaps_zone="FR",
        carbon_zones={"electricitymaps": "FR"},
        cloud_aliases={"gcp": ("europe-west9",), "azure": ("francecentral",)},
        timezone="Europe/Paris",
        price_is_nodal_lmp=False,
        confidence=Confidence.MEDIUM,
        notes=(
            "France bidding-zone day-ahead price (EUR/MWh) via ENTSO-E. "
            "Bidding-zone price, not nodal LMP."
        ),
    ),
}


class UnknownRegionError(KeyError):
    """Raised when a canonical region is not in the registry."""


def get_region_mapping(region: str) -> RegionMapping:
    """Return the RegionMapping for a canonical region, or raise."""
    try:
        return REGION_REGISTRY[region]
    except KeyError:
        raise UnknownRegionError(
            f"Region '{region}' is not in the region registry. "
            f"Known regions: {sorted(REGION_REGISTRY)}"
        )


def get_electricitymaps_zone(region: str) -> Optional[str]:
    """Return the Electricity Maps zone key for a region (None if unmapped)."""
    return get_region_mapping(region).electricitymaps_zone


def get_carbon_zone(region: str, provider: str) -> Optional[str]:
    """Return the carbon-provider zone key for a region/provider."""
    return get_region_mapping(region).carbon_zones.get(provider)


def resolve_cloud_region(provider: str, cloud_region: str) -> Optional[str]:
    """Resolve a cloud provider region (e.g. aws/us-east-1) to a canonical region.

    Returns None if no mapping is known (never guesses silently).
    """
    provider = provider.lower()
    for mapping in REGION_REGISTRY.values():
        if cloud_region in mapping.cloud_aliases.get(provider, ()):
            return mapping.canonical_region
    return None


def list_regions(min_confidence: Optional[str] = None) -> list[str]:
    """List canonical regions, optionally filtering to a minimum confidence."""
    if min_confidence is None:
        return list(REGION_REGISTRY)
    order = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    floor = order[min_confidence]
    return [r for r, m in REGION_REGISTRY.items() if order[m.confidence] >= floor]


def electricitymaps_zone_map() -> dict[str, str]:
    """Build the {canonical_region: EM zone} map for the EM provider.

    This replaces hard-coded zone maps elsewhere with the single registry.
    """
    return {
        r: m.electricitymaps_zone
        for r, m in REGION_REGISTRY.items()
        if m.electricitymaps_zone
    }


__all__ = [
    "Confidence",
    "RegionMapping",
    "REGION_REGISTRY",
    "UnknownRegionError",
    "get_region_mapping",
    "get_electricitymaps_zone",
    "get_carbon_zone",
    "resolve_cloud_region",
    "list_regions",
    "electricitymaps_zone_map",
]
