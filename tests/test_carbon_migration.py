"""Phase 7/11.8 — carbon-aware migration (cheaper-but-dirtier is not carbon-saving)."""

import pytest

from aurelius.carbon.migration import (
    MigrationCarbonOverheadMode,
    evaluate_migration_carbon,
)


def _assess(src_moer, dst_moer, overhead_hours, mode):
    return evaluate_migration_carbon(
        source_region="src", dest_region="dst",
        power_kw=100.0, utilization_fraction=1.0, runtime_hours=2.0, pue=1.0,
        source_moer_gco2_per_kwh=src_moer, dest_moer_gco2_per_kwh=dst_moer,
        migration_overhead_hours=overhead_hours, overhead_mode=mode,
    )


class TestCarbonSavingDirection:
    def test_cleaner_destination_saves_carbon(self):
        # src 600 -> dst 100, modest warmup. Net should be a real saving.
        a = _assess(600.0, 100.0, 0.25, MigrationCarbonOverheadMode.ESTIMATED)
        assert a.emissions_saved_by_destination == pytest.approx(100.0)  # (120-20)
        assert a.is_carbon_saving is True
        assert a.net_emissions_saved_kgco2 > 0
        assert a.carbon_savings_is_real is True

    def test_dirtier_destination_is_not_carbon_saving(self):
        # Cheaper-but-dirtier move: dst MOER higher than src. Must NOT be carbon-saving.
        a = _assess(100.0, 600.0, 0.25, MigrationCarbonOverheadMode.ESTIMATED)
        assert a.emissions_saved_by_destination < 0
        assert a.is_carbon_saving is False

    def test_overhead_can_erase_marginal_saving(self):
        # Slightly cleaner destination, but a large warmup overhead at the dirty-ish
        # destination wipes out the runtime saving -> not carbon-saving.
        a = _assess(300.0, 250.0, 5.0, MigrationCarbonOverheadMode.ESTIMATED)
        # runtime saving = (60 - 50) = 10 kg; overhead = 100kW*5h*250/1000 = 125 kg
        assert a.emissions_saved_by_destination == pytest.approx(10.0)
        assert a.emissions_added_by_migration == pytest.approx(125.0)
        assert a.is_carbon_saving is False


class TestOverheadHonesty:
    def test_unknown_overhead_is_not_real(self):
        a = _assess(600.0, 100.0, None, MigrationCarbonOverheadMode.UNKNOWN)
        # It may look carbon-saving on runtime alone...
        assert a.is_carbon_saving is True
        # ...but with unknown overhead the saving must NOT be reported as real.
        assert a.carbon_savings_is_real is False
        assert a.to_dict()["migration_carbon_overhead_mode"] == "unknown"

    def test_measured_overhead_is_real(self):
        a = _assess(600.0, 100.0, 0.25, MigrationCarbonOverheadMode.MEASURED)
        assert a.carbon_savings_is_real is True

    def test_data_transfer_and_extended_runtime_count(self):
        a = evaluate_migration_carbon(
            source_region="src", dest_region="dst",
            power_kw=100.0, utilization_fraction=1.0, runtime_hours=2.0, pue=1.0,
            source_moer_gco2_per_kwh=400.0, dest_moer_gco2_per_kwh=300.0,
            migration_overhead_hours=0.5, extended_runtime_hours=0.5,
            data_transfer_kwh=10.0,
            overhead_mode=MigrationCarbonOverheadMode.ESTIMATED,
        )
        # overhead = warmup(0.5h*100*300/1000=15) + extended(15) + transfer(10*300/1000=3) = 33
        assert a.migration_overhead_emissions_kgco2 == pytest.approx(33.0)
