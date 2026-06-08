"""Authoritative carbon accounting for Aurelius.

This module is the SINGLE source of truth for converting a workload's energy use
plus a marginal-operating-emissions-rate (MOER) signal into kilograms of CO2.
Every carbon path in Aurelius (scheduler, optimizer, evaluator, reports, API,
CLI) must compute emissions through :func:`emissions_kgco2` so the units and the
provenance are consistent and testable.

Canonical formula (unit-checked, see tests/test_carbon_accounting.py)
---------------------------------------------------------------------
    emissions_kgco2 =
        power_kw                # kW  (IT power draw at full speed)
        * utilization_fraction  # dimensionless in [0, 1]
        * duration_hours        # h
        * pue                   # dimensionless >= 1.0 (facility overhead)
        * moer_gco2_per_kwh     # gCO2 / kWh  (MARGINAL emissions rate)
        / 1000.0                # g -> kg

Dimensional analysis:
    kW * h            = kWh                 (energy at the IT load)
    kWh * pue         = kWh                 (facility energy incl. cooling/overhead)
    kWh * gCO2/kWh    = gCO2                 (emissions in grams)
    gCO2 / 1000       = kgCO2               (emissions in kilograms)

Units discipline
----------------
* Power is ALWAYS kW here. If you have MW, multiply by 1000 before calling.
* MOER is ALWAYS gCO2/kWh here. WattTime returns lbs/MWh; convert with
  :data:`aurelius.ingestion.grid_apis.watttime._LBS_PER_MWH_TO_GCO2_PER_KWH`
  (= 453.592 / 1000) BEFORE it reaches this module.
* This module uses MARGINAL emissions (MOER), never average grid intensity
  (AOER). Mixing the two is a correctness bug — see :class:`CarbonSignalType`.

Provenance discipline
---------------------
A bare number is not enough. :class:`WorkloadCarbonRecord` carries the data
source, the signal type, whether the MOER was real vs synthetic, forecast vs
historical, and the coverage of the underlying interval data, so downstream
reporting can never present a synthetic/forecast/low-coverage number as a
realized real-world saving.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

# Bump when the emissions math or record schema changes in a way that makes old
# records non-comparable. Stored on every WorkloadCarbonRecord.
CARBON_CALCULATION_VERSION = "carbon-accounting-v1.0"


class CarbonSignalType(str, enum.Enum):
    """Which physical quantity the gCO2/kWh number represents.

    MARGINAL_MOER is the correct signal for operational load-shifting: it is the
    emissions of the *marginal* generator that responds to an incremental kWh of
    flexible load. AVERAGE_AOER is the average grid intensity and must NOT be
    used for shifting decisions (it answers a different question).
    """

    MARGINAL_MOER = "marginal_moer"      # WattTime co2_moer — correct for shifting
    AVERAGE_AOER = "average_aoer"        # average grid intensity — reporting only
    UNKNOWN = "unknown"


class CarbonDataKind(str, enum.Enum):
    """Temporal provenance of the MOER value used."""

    FORECAST = "forecast"        # forward-looking MOER (decision-time)
    HISTORICAL = "historical"    # settled/measured MOER (realized replay)
    SYNTHETIC = "synthetic"      # generated/scenario data — NEVER a real saving
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def energy_kwh(
    power_kw: float,
    utilization_fraction: float,
    duration_hours: float,
    pue: float = 1.0,
) -> float:
    """Facility energy in kWh, including PUE overhead.

    energy_kwh = power_kw * utilization_fraction * duration_hours * pue
    """
    return float(power_kw) * float(utilization_fraction) * float(duration_hours) * float(pue)


def emissions_kgco2(
    power_kw: float,
    utilization_fraction: float,
    duration_hours: float,
    pue: float,
    moer_gco2_per_kwh: float,
) -> float:
    """Marginal CO2 emissions in kilograms for one (region, interval) leg.

    See the module docstring for the unit-checked derivation. ``moer_gco2_per_kwh``
    MUST be a marginal emissions rate already normalized to gCO2/kWh.
    """
    grams = (
        float(power_kw)
        * float(utilization_fraction)
        * float(duration_hours)
        * float(pue)
        * float(moer_gco2_per_kwh)
    )
    return grams / 1000.0


def emissions_intensity_gco2_per_kwh(emissions_kg: float, energy: float) -> float:
    """Effective emissions intensity (gCO2/kWh) over an energy total.

    intensity = emissions_kg * 1000 (g) / energy_kwh
    Returns 0.0 when energy is zero (no emissions to attribute).
    """
    if energy <= 0:
        return 0.0
    return (emissions_kg * 1000.0) / energy


# ---------------------------------------------------------------------------
# Authoritative per-workload carbon record
# ---------------------------------------------------------------------------

@dataclass
class WorkloadCarbonRecord:
    """Everything Aurelius knows about one workload's carbon footprint.

    One record describes a single (job, scheduler, baseline, region) evaluation.
    A migrated job produces one record per segment; callers may also aggregate
    segments into a single record by summing energy/emissions and energy-weighting
    the MOER (see :func:`aggregate_records`).

    Provenance fields exist so a downstream report can NEVER present a
    synthetic/forecast/low-coverage figure as a realized real-world saving:
    ``carbon_data_is_real`` is True only for real (non-synthetic) data, and the
    realized-vs-forecast distinction is explicit.
    """

    job_id: str
    scheduler_name: str
    baseline_name: str
    region: str
    start_time_utc: datetime
    end_time_utc: datetime
    duration_hours: float
    power_kw: float
    utilization_fraction: float
    pue: float
    energy_kwh: float
    moer_gco2_per_kwh: float
    emissions_kgco2: float

    # Provenance / trust
    carbon_data_source: str                 # e.g. "watttime_co2_moer", "synthetic"
    carbon_signal_type: CarbonSignalType
    carbon_data_is_real: bool
    carbon_data_is_forecast: bool
    carbon_data_is_historical: bool
    carbon_data_coverage_pct: float         # [0, 100]
    carbon_missing_intervals: int
    carbon_calculation_version: str = CARBON_CALCULATION_VERSION

    def __post_init__(self) -> None:
        # Coerce enum from string for ergonomics / round-tripping.
        if isinstance(self.carbon_signal_type, str):
            self.carbon_signal_type = CarbonSignalType(self.carbon_signal_type)

    @property
    def emissions_intensity_gco2_per_kwh(self) -> float:
        return emissions_intensity_gco2_per_kwh(self.emissions_kgco2, self.energy_kwh)

    @property
    def carbon_data_is_complete(self) -> bool:
        """True when every interval had real MOER coverage."""
        return self.carbon_missing_intervals == 0 and self.carbon_data_coverage_pct >= 100.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["carbon_signal_type"] = self.carbon_signal_type.value
        d["start_time_utc"] = self.start_time_utc.isoformat()
        d["end_time_utc"] = self.end_time_utc.isoformat()
        d["emissions_intensity_gco2_per_kwh"] = round(self.emissions_intensity_gco2_per_kwh, 4)
        # Round the heavy numerics for stable JSON, keep full precision in-memory.
        for k in ("duration_hours", "energy_kwh", "moer_gco2_per_kwh", "emissions_kgco2",
                  "carbon_data_coverage_pct"):
            d[k] = round(d[k], 6)
        return d


@dataclass
class MoerInterval:
    """One marginal-emissions reading. ``is_real`` is False for synthetic data."""

    value_gco2_per_kwh: Optional[float]   # None == missing (no silent default!)
    is_real: bool = True


def build_workload_carbon_record(
    *,
    job_id: str,
    scheduler_name: str,
    baseline_name: str,
    region: str,
    start_time_utc: datetime,
    end_time_utc: datetime,
    power_kw: float,
    utilization_fraction: float,
    pue: float,
    moer_intervals: list[MoerInterval],
    interval_hours: float,
    carbon_data_source: str,
    carbon_signal_type: CarbonSignalType,
    carbon_data_kind: CarbonDataKind,
) -> WorkloadCarbonRecord:
    """Compute a :class:`WorkloadCarbonRecord` from per-interval MOER readings.

    Missing intervals (``value_gco2_per_kwh is None``) are NEVER filled with a
    default emissions rate. They are counted in ``carbon_missing_intervals`` and
    excluded from both energy and emissions, and they lower the coverage pct.
    Emissions/energy are therefore computed over COVERED intervals only, so a
    partially-covered record under-reports rather than fabricates.

    ``utilization_fraction`` and ``pue`` apply uniformly across intervals.
    Emissions per interval use the authoritative :func:`emissions_kgco2`.
    """
    total_intervals = len(moer_intervals)
    covered = [iv for iv in moer_intervals if iv.value_gco2_per_kwh is not None]
    missing = total_intervals - len(covered)
    coverage_pct = (100.0 * len(covered) / total_intervals) if total_intervals else 0.0

    total_emissions_kg = 0.0
    total_energy = 0.0
    weighted_moer_num = 0.0  # energy-weighted MOER numerator
    is_real = bool(covered) and all(iv.is_real for iv in covered)

    for iv in covered:
        e_kwh = energy_kwh(power_kw, utilization_fraction, interval_hours, pue)
        emis = emissions_kgco2(
            power_kw, utilization_fraction, interval_hours, pue, iv.value_gco2_per_kwh
        )
        total_energy += e_kwh
        total_emissions_kg += emis
        weighted_moer_num += iv.value_gco2_per_kwh * e_kwh

    eff_moer = (weighted_moer_num / total_energy) if total_energy > 0 else 0.0
    duration_hours = total_intervals * interval_hours

    # Synthetic data is never "real" regardless of coverage.
    if carbon_data_kind is CarbonDataKind.SYNTHETIC:
        is_real = False

    return WorkloadCarbonRecord(
        job_id=job_id,
        scheduler_name=scheduler_name,
        baseline_name=baseline_name,
        region=region,
        start_time_utc=start_time_utc,
        end_time_utc=end_time_utc,
        duration_hours=duration_hours,
        power_kw=power_kw,
        utilization_fraction=utilization_fraction,
        pue=pue,
        energy_kwh=total_energy,
        moer_gco2_per_kwh=eff_moer,
        emissions_kgco2=total_emissions_kg,
        carbon_data_source=carbon_data_source,
        carbon_signal_type=carbon_signal_type,
        carbon_data_is_real=is_real,
        carbon_data_is_forecast=(carbon_data_kind is CarbonDataKind.FORECAST),
        carbon_data_is_historical=(carbon_data_kind is CarbonDataKind.HISTORICAL),
        carbon_data_coverage_pct=coverage_pct,
        carbon_missing_intervals=missing,
    )


def aggregate_records(records: list[WorkloadCarbonRecord]) -> dict:
    """Aggregate a set of records (e.g. a whole schedule) into totals.

    Coverage is interval-weighted; ``carbon_data_is_real`` is True only if EVERY
    record is real (one synthetic/forecast leg taints the aggregate's realness).
    """
    if not records:
        return {
            "total_emissions_kgco2": 0.0,
            "total_energy_kwh": 0.0,
            "emissions_intensity_gco2_per_kwh": 0.0,
            "carbon_data_coverage_pct": 0.0,
            "carbon_missing_intervals": 0,
            "carbon_data_is_real": False,
            "n_records": 0,
        }
    total_emissions = sum(r.emissions_kgco2 for r in records)
    total_energy = sum(r.energy_kwh for r in records)
    total_missing = sum(r.carbon_missing_intervals for r in records)
    # Interval-weighted coverage: covered intervals / total intervals.
    total_intervals = 0
    covered_intervals = 0
    for r in records:
        n = r.duration_hours  # intervals already folded into duration; use as weight
        total_intervals += n
        covered_intervals += n * (r.carbon_data_coverage_pct / 100.0)
    coverage = (100.0 * covered_intervals / total_intervals) if total_intervals else 0.0
    return {
        "total_emissions_kgco2": total_emissions,
        "total_energy_kwh": total_energy,
        "emissions_intensity_gco2_per_kwh": emissions_intensity_gco2_per_kwh(
            total_emissions, total_energy
        ),
        "carbon_data_coverage_pct": coverage,
        "carbon_missing_intervals": total_missing,
        "carbon_data_is_real": all(r.carbon_data_is_real for r in records),
        "n_records": len(records),
    }
