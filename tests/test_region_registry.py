"""Tests for the canonical region registry."""

import pytest

from aurelius.ingestion.region_registry import (
    Confidence,
    REGION_REGISTRY,
    UnknownRegionError,
    electricitymaps_zone_map,
    get_carbon_zone,
    get_electricitymaps_zone,
    get_region_mapping,
    list_regions,
    resolve_cloud_region,
)


class TestCoreMappings:
    def test_us_west_is_caiso_np15(self):
        m = get_region_mapping("us-west")
        assert m.iso == "CAISO"
        assert m.source_region == "TH_NP15_GEN-APND"
        assert m.electricitymaps_zone == "US-CAL-CISO"
        assert m.price_is_nodal_lmp is True
        assert m.confidence == Confidence.HIGH

    def test_us_south_ercot_is_spp_not_lmp(self):
        """ERCOT publishes Settlement Point Prices, not nodal LMP."""
        m = get_region_mapping("us-south")
        assert m.iso == "ERCOT"
        assert m.price_is_nodal_lmp is False

    def test_unimplemented_iso_markets_are_low_confidence(self):
        for region in ("us-central", "us-north", "us-newengland", "us-nyiso"):
            assert get_region_mapping(region).confidence == Confidence.LOW

    def test_unknown_region_raises(self):
        with pytest.raises(UnknownRegionError):
            get_region_mapping("mars-west")


class TestCloudResolution:
    def test_aws_us_east_resolves_to_pjm_region(self):
        assert resolve_cloud_region("aws", "us-east-1") == "us-east"
        assert resolve_cloud_region("aws", "us-east-2") == "us-east"

    def test_aws_us_west_resolves_to_caiso_region(self):
        assert resolve_cloud_region("aws", "us-west-1") == "us-west"

    def test_unknown_cloud_region_returns_none_not_guess(self):
        assert resolve_cloud_region("aws", "ap-south-1") is None
        assert resolve_cloud_region("gcp", "made-up-region") is None


class TestZoneMaps:
    def test_em_zone_map_has_no_none_values(self):
        zmap = electricitymaps_zone_map()
        assert all(v for v in zmap.values())
        assert zmap["us-west"] == "US-CAL-CISO"

    def test_carbon_zone_lookup(self):
        assert get_carbon_zone("us-west", "electricitymaps") == "US-CAL-CISO"
        assert get_carbon_zone("us-west", "watttime") == "CAISO_NP15"
        assert get_carbon_zone("us-west", "nonexistent") is None

    def test_get_em_zone_helper(self):
        assert get_electricitymaps_zone("eu-west") == "DE"


class TestConfidenceFiltering:
    def test_high_confidence_subset(self):
        high = list_regions(min_confidence=Confidence.HIGH)
        assert set(high) == {"us-west", "us-east", "us-south"}

    def test_all_regions_when_no_floor(self):
        assert set(list_regions()) == set(REGION_REGISTRY)
