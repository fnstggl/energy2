"""One authoritative region mapping — consistency & validation (Phase 4)."""

import pytest

from aurelius.carbon.regions import (
    CARBON_AVAILABLE,
    CARBON_UNAVAILABLE,
    assert_optimizer_evaluator_consistency,
    carbon_region_status,
    carbon_unavailable_regions,
    validate_carbon_region_mapping,
    watttime_ba,
    watttime_ba_map,
)
from aurelius.ingestion.grid_apis.watttime import _default_ba_map


class TestSingleSourceOfTruth:
    def test_watttime_client_uses_registry_map(self):
        # The WattTime client's default BA map MUST equal the authoritative map.
        assert _default_ba_map() == watttime_ba_map()

    def test_known_bas(self):
        assert watttime_ba("us-west") == "CAISO_NP15"
        assert watttime_ba("us-east") == "PJM_DOM"      # was "PJM" in the old divergent map
        assert watttime_ba("us-south") == "ERCOT_HOUSTON"
        assert watttime_ba("us-north") == "MISO_INDIANAPOLIS"

    def test_unmapped_region_is_carbon_unavailable(self):
        assert watttime_ba("eu-west") is None
        assert carbon_region_status("eu-west") == CARBON_UNAVAILABLE
        assert carbon_region_status("us-west") == CARBON_AVAILABLE

    def test_no_default_fallback_ba(self):
        # An unmapped region must not silently inherit another region's BA.
        assert "eu-west" not in watttime_ba_map()
        assert "eu-west" in carbon_unavailable_regions()


class TestValidation:
    def test_validate_flags_all_regions(self):
        rep = validate_carbon_region_mapping()
        assert rep["all_mapped_or_flagged"] is True
        assert set(rep["available"]) == {"us-west", "us-east", "us-south", "us-north"}
        assert "eu-west" in rep["carbon_unavailable"]

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="not in the canonical region registry"):
            validate_carbon_region_mapping(["mars-west"])


class TestConsistencyGuard:
    def test_matching_maps_pass(self):
        m = watttime_ba_map()
        assert_optimizer_evaluator_consistency(m, dict(m)) is None

    def test_mismatch_raises(self):
        opt = {"us-east": "PJM_DOM"}
        evalmap = {"us-east": "PJM"}  # the historical divergence
        with pytest.raises(ValueError, match="BA mismatch"):
            assert_optimizer_evaluator_consistency(opt, evalmap)
