"""WattTime forecast MOER + authoritative-map consumption (Phase 3/4)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import CARBON_COLUMNS, ProviderConfigError
from aurelius.ingestion.grid_apis.watttime import (
    _LBS_PER_MWH_TO_GCO2_PER_KWH,
    WattTimeCarbonProvider,
    _default_ba_map,
)


def _resp(json_data, status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.raise_for_status = MagicMock()
    return r


def _login():
    return _resp({"token": "tok"})


def _forecast_payload(n=12, value=800.0):
    base = pd.Timestamp("2026-01-01T00:00:00+00:00")
    data = [
        {"point_time": (base + pd.Timedelta(minutes=5 * i)).isoformat(), "value": value}
        for i in range(n)
    ]
    return _resp({"forecast": data, "meta": {}})


class TestForecastEndpoint:
    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_forecast_returns_canonical_schema(self, mock_requests):
        mock_requests.get.side_effect = [_login(), _forecast_payload()]
        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_forecast_carbon("us-west", horizon_hours=1)
        assert list(df.columns) == CARBON_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-west").all()

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_forecast_source_is_labeled_forecast(self, mock_requests):
        mock_requests.get.side_effect = [_login(), _forecast_payload()]
        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_forecast_carbon("us-west", horizon_hours=1)
        # A forecast value must never be mistaken for settled historical data.
        assert "forecast" in df["source"].iloc[0].lower()
        assert "moer" in df["source"].iloc[0].lower()

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_forecast_unit_conversion(self, mock_requests):
        mock_requests.get.side_effect = [_login(), _forecast_payload(value=800.0)]
        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_forecast_carbon("us-west", horizon_hours=1)
        assert df["gco2_per_kwh"].mean() == pytest.approx(800.0 * _LBS_PER_MWH_TO_GCO2_PER_KWH, abs=1.0)

    def test_forecast_missing_creds_raises(self, monkeypatch):
        monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
        monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
        provider = WattTimeCarbonProvider(username="", password="")
        with pytest.raises(ProviderConfigError, match="WATTTIME_USERNAME"):
            provider.fetch_forecast_carbon("us-west")

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_forecast_unknown_region_returns_empty(self, mock_requests):
        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_forecast_carbon("xx-unknown")
        assert df.empty
        assert list(df.columns) == CARBON_COLUMNS


class TestAuthoritativeMap:
    def test_default_map_is_registry_derived(self):
        from aurelius.carbon.regions import watttime_ba_map
        assert _default_ba_map() == watttime_ba_map()

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_us_east_uses_pjm_dom_not_pjm(self, mock_requests):
        # Regression guard for the old divergent BA map (us-east -> "PJM").
        mock_requests.get.side_effect = [_login(), _forecast_payload()]
        provider = WattTimeCarbonProvider(username="u", password="p")
        provider.fetch_forecast_carbon("us-east", horizon_hours=1)
        # Second call is the forecast request; inspect its region param.
        forecast_call = mock_requests.get.call_args_list[1]
        assert forecast_call.kwargs["params"]["region"] == "PJM_DOM"
