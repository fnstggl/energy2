"""ONE authoritative Aurelius-region -> WattTime balancing-authority mapping.

Every carbon path (ingestion, scheduler, optimizer, evaluator, reports, API,
CLI, tests) must resolve a region's WattTime BA through this module so there is
exactly one source of truth. The mapping is derived from
``aurelius.ingestion.region_registry`` (the canonical region registry) — it is
NOT a second hand-maintained table.

Rules enforced here
-------------------
* No default fallback BA. A region either has a real WattTime BA or is
  ``carbon_unavailable``.
* No silent fallback emissions value lives here — missing coverage is surfaced,
  never papered over.
* :func:`assert_optimizer_evaluator_consistency` guarantees the decision path and
  the scoring/replay path use the SAME mapping (the historical bug was the
  WattTime client carrying its own divergent BA map).
"""

from __future__ import annotations

from typing import Optional

from ..ingestion.region_registry import REGION_REGISTRY, get_region_mapping

_WATTTIME = "watttime"

# Status strings (kept as plain strings for easy JSON/CLI surfacing).
CARBON_AVAILABLE = "available"
CARBON_UNAVAILABLE = "carbon_unavailable"


def watttime_ba(region: str) -> Optional[str]:
    """Return the WattTime balancing-authority code for a region, or None.

    None means "this region has no WattTime coverage" — callers must treat that
    as ``carbon_unavailable`` and must NOT substitute another region's BA.
    """
    mapping = get_region_mapping(region)  # raises UnknownRegionError on bad region
    return mapping.carbon_zones.get(_WATTTIME)


def carbon_region_status(region: str) -> str:
    """``CARBON_AVAILABLE`` if the region has a WattTime BA, else CARBON_UNAVAILABLE."""
    try:
        return CARBON_AVAILABLE if watttime_ba(region) else CARBON_UNAVAILABLE
    except KeyError:
        return CARBON_UNAVAILABLE


def watttime_ba_map() -> dict[str, str]:
    """The full ``{canonical_region: WattTime BA}`` map (only mapped regions).

    This is THE map the WattTime client and every evaluator must use.
    """
    out: dict[str, str] = {}
    for region, mapping in REGION_REGISTRY.items():
        ba = mapping.carbon_zones.get(_WATTTIME)
        if ba:
            out[region] = ba
    return out


def carbon_unavailable_regions() -> list[str]:
    """Canonical regions that are schedulable but have NO WattTime coverage."""
    return [r for r in REGION_REGISTRY if not REGION_REGISTRY[r].carbon_zones.get(_WATTTIME)]


def validate_carbon_region_mapping(schedulable_regions: Optional[list[str]] = None) -> dict:
    """Validate that every schedulable region is either mapped or flagged.

    Returns a structured report. Raises ``ValueError`` only on an internal
    inconsistency (a region that claims a WattTime BA but is not in the
    registry). A region with no BA is NOT an error — it is reported as
    ``carbon_unavailable`` so the caller can disable carbon for it explicitly
    rather than silently defaulting it.
    """
    regions = schedulable_regions if schedulable_regions is not None else list(REGION_REGISTRY)
    available: dict[str, str] = {}
    unavailable: list[str] = []
    unknown: list[str] = []

    for r in regions:
        try:
            ba = watttime_ba(r)
        except KeyError:
            unknown.append(r)
            continue
        if ba:
            available[r] = ba
        else:
            unavailable.append(r)

    if unknown:
        raise ValueError(
            f"Regions not in the canonical region registry (cannot resolve carbon): {unknown}. "
            "Add them to aurelius.ingestion.region_registry or remove them from the schedulable set."
        )

    return {
        "available": available,
        "carbon_unavailable": unavailable,
        "all_mapped_or_flagged": True,  # unknowns would have raised above
        "n_available": len(available),
        "n_unavailable": len(unavailable),
    }


def assert_optimizer_evaluator_consistency(
    optimizer_map: dict[str, str],
    evaluator_map: dict[str, str],
) -> None:
    """Fail loudly if the optimizer and evaluator disagree on any region's BA.

    The historical bug: the WattTime client carried a private ``_DEFAULT_BA_MAP``
    that disagreed with the registry (e.g. us-east PJM vs PJM_DOM). If the
    scheduler plans against one BA's MOER and the evaluator scores against
    another's, every carbon saving is meaningless. This guard makes that
    impossible to ship.
    """
    mismatches = {}
    for region in set(optimizer_map) & set(evaluator_map):
        if optimizer_map[region] != evaluator_map[region]:
            mismatches[region] = (optimizer_map[region], evaluator_map[region])
    if mismatches:
        raise ValueError(
            "Optimizer/evaluator WattTime BA mismatch (carbon savings would be invalid): "
            + ", ".join(f"{r}: optimizer={o} evaluator={e}" for r, (o, e) in mismatches.items())
        )


__all__ = [
    "CARBON_AVAILABLE",
    "CARBON_UNAVAILABLE",
    "watttime_ba",
    "carbon_region_status",
    "watttime_ba_map",
    "carbon_unavailable_regions",
    "validate_carbon_region_mapping",
    "assert_optimizer_evaluator_consistency",
]
