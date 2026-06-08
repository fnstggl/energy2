"""Carbon-aware migration evaluation (Phase 7).

The legacy scheduler decides migrations on PRICE ALONE. That can move a job to a
cheaper-but-dirtier region and silently increase emissions. This module makes
migration carbon-accountable:

A migration is only carbon-saving when

    emissions_saved_by_destination > emissions_added_by_migration

where ``emissions_added_by_migration`` includes the runtime emissions at the
destination AND the migration overhead (warmup/data-movement/extended-runtime
energy) if that overhead can be modeled.

Overhead honesty
----------------
If the migration overhead cannot be quantified yet, the assessment carries
``overhead_mode = unknown`` and ``carbon_savings_is_real = False`` — the caller
MUST NOT report a real carbon saving in that case. Modes:

* ``unknown``   — overhead not modeled; net savings are NOT trustworthy.
* ``estimated`` — overhead estimated from a model (warmup hours x dest MOER, etc.)
* ``measured``  — overhead from measured telemetry.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from .accounting import emissions_kgco2


class MigrationCarbonOverheadMode(str, enum.Enum):
    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    MEASURED = "measured"


@dataclass
class MigrationCarbonAssessment:
    """Result of evaluating a single source->destination migration for carbon."""

    source_region: str
    dest_region: str
    source_runtime_emissions_kgco2: float
    dest_runtime_emissions_kgco2: float
    migration_overhead_emissions_kgco2: float
    overhead_mode: MigrationCarbonOverheadMode

    @property
    def emissions_saved_by_destination(self) -> float:
        """Runtime emissions avoided by running at the destination vs the source."""
        return self.source_runtime_emissions_kgco2 - self.dest_runtime_emissions_kgco2

    @property
    def emissions_added_by_migration(self) -> float:
        """Extra emissions the migration itself costs (overhead only).

        The destination runtime is the new baseline of work, not an "addition";
        the addition the move incurs is the migration overhead.
        """
        return self.migration_overhead_emissions_kgco2

    @property
    def net_emissions_saved_kgco2(self) -> float:
        """Net carbon benefit of migrating (positive => migrating reduces CO2)."""
        return self.emissions_saved_by_destination - self.emissions_added_by_migration

    @property
    def is_carbon_saving(self) -> bool:
        """True only if the move strictly reduces net emissions."""
        return self.net_emissions_saved_kgco2 > 0.0

    @property
    def carbon_savings_is_real(self) -> bool:
        """Savings are only trustworthy when overhead is actually modeled."""
        return self.overhead_mode is not MigrationCarbonOverheadMode.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "source_region": self.source_region,
            "dest_region": self.dest_region,
            "source_runtime_emissions_kgco2": round(self.source_runtime_emissions_kgco2, 6),
            "dest_runtime_emissions_kgco2": round(self.dest_runtime_emissions_kgco2, 6),
            "migration_overhead_emissions_kgco2": round(self.migration_overhead_emissions_kgco2, 6),
            "emissions_saved_by_destination": round(self.emissions_saved_by_destination, 6),
            "emissions_added_by_migration": round(self.emissions_added_by_migration, 6),
            "net_emissions_saved_kgco2": round(self.net_emissions_saved_kgco2, 6),
            "is_carbon_saving": self.is_carbon_saving,
            "migration_carbon_overhead_mode": self.overhead_mode.value,
            "carbon_savings_is_real": self.carbon_savings_is_real,
        }


def evaluate_migration_carbon(
    *,
    source_region: str,
    dest_region: str,
    power_kw: float,
    utilization_fraction: float,
    runtime_hours: float,
    pue: float,
    source_moer_gco2_per_kwh: float,
    dest_moer_gco2_per_kwh: float,
    migration_overhead_hours: Optional[float] = None,
    data_transfer_kwh: float = 0.0,
    extended_runtime_hours: float = 0.0,
    overhead_mode: MigrationCarbonOverheadMode = MigrationCarbonOverheadMode.UNKNOWN,
) -> MigrationCarbonAssessment:
    """Assess whether migrating from source to destination saves carbon.

    Source/destination runtime emissions use the authoritative formula at each
    region's MOER. The migration overhead emissions (charged at the DESTINATION
    MOER, since warmup/transfer happen there) include:

      * warmup energy:   power * utilization * migration_overhead_hours * pue
      * data-movement:   data_transfer_kwh (already in kWh; e.g. network energy)
      * extended runtime: power * utilization * extended_runtime_hours * pue

    If ``overhead_mode`` is UNKNOWN, the overhead emissions are still computed
    from whatever was supplied, but the assessment flags
    ``carbon_savings_is_real = False`` so callers cannot report it as a real
    saving. Pass ESTIMATED/MEASURED only when the overhead inputs are trustworthy.
    """
    source_runtime = emissions_kgco2(
        power_kw, utilization_fraction, runtime_hours, pue, source_moer_gco2_per_kwh
    )
    dest_runtime = emissions_kgco2(
        power_kw, utilization_fraction, runtime_hours, pue, dest_moer_gco2_per_kwh
    )

    overhead_kg = 0.0
    if migration_overhead_hours:
        overhead_kg += emissions_kgco2(
            power_kw, utilization_fraction, migration_overhead_hours, pue, dest_moer_gco2_per_kwh
        )
    if extended_runtime_hours:
        overhead_kg += emissions_kgco2(
            power_kw, utilization_fraction, extended_runtime_hours, pue, dest_moer_gco2_per_kwh
        )
    if data_transfer_kwh:
        # data_transfer_kwh is energy already; emissions = kWh * MOER / 1000.
        overhead_kg += data_transfer_kwh * dest_moer_gco2_per_kwh / 1000.0

    return MigrationCarbonAssessment(
        source_region=source_region,
        dest_region=dest_region,
        source_runtime_emissions_kgco2=source_runtime,
        dest_runtime_emissions_kgco2=dest_runtime,
        migration_overhead_emissions_kgco2=overhead_kg,
        overhead_mode=overhead_mode,
    )


__all__ = [
    "MigrationCarbonOverheadMode",
    "MigrationCarbonAssessment",
    "evaluate_migration_carbon",
]
