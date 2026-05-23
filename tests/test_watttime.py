"""Tests for WattTimeCarbonProvider (unit tests with mocked HTTP)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import (
    CARBON_COLUMNS,
    ProviderConfigError,
)
from aurelius.ingestion.grid_apis.watttime import (
    _LBS_PER_MWH_TO_GCO2_PER_KWH,
    WattTimeCarbonProvider,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2024, 3, 1, 2, 0, tzinfo=UTC)  # 2-hour window


def _make_mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _watttime_login_response():
    return _make_mock_response({"token": "test-bearer-token"})


def _watttime_hist_response(num_5min_points=24):
    """Build a plausible 5-min MOER payload (default: 2h of 5-min data)."""
    data = []
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    for i in range(num_5min_points):
        ts = base + pd.Timedelta(minutes=5 * i)
        data.append({
            "point_time": ts.isoformat(),
            "value": 800.0,   # 800 lbs CO2/MWh
            "version": "3.0",
        })
    return _make_mock_response({"data": data, "meta": {}})


# ---------------------------------------------------------------------------
# Credential guard
# ---------------------------------------------------------------------------

class TestWattTimeCredentialGuard:
    def test_missing_username_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("WATTTIME_USERNAME", raising=False)
        monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
        provider = WattTimeCarbonProvider(username="", password="")
        with pytest.raises(ProviderConfigError, match="WATTTIME_USERNAME"):
            provider.fetch_carbon("us-west", T0, T1)

    def test_missing_password_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("WATTTIME_PASSWORD", raising=False)
        provider = WattTimeCarbonProvider(username="user", password="")
        with pytest.raises(ProviderConfigError, match="WATTTIME_PASSWORD"):
            provider.fetch_carbon("us-west", T0, T1)

    def test_env_vars_used_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("WATTTIME_USERNAME", "env-user")
        monkeypatch.setenv("WATTTIME_PASSWORD", "env-pass")
        provider = WattTimeCarbonProvider()
        assert provider._username == "env-user"
        assert provider._password == "env-pass"


# ---------------------------------------------------------------------------
# source_name must contain "moer" to make signal type visible
# ---------------------------------------------------------------------------

class TestWattTimeSourceName:
    def test_source_name_contains_moer_by_default(self):
        provider = WattTimeCarbonProvider(username="u", password="p")
        assert "moer" in provider.source_name.lower()

    def test_source_name_reflects_signal_type(self):
        provider = WattTimeCarbonProvider(
            username="u", password="p", signal_type="co2_aoer"
        )
        assert "aoer" in provider.source_name.lower()

    def test_source_name_not_generic_carbon(self):
        """source_name must NOT be a generic label like 'carbon' (masks signal type)."""
        provider = WattTimeCarbonProvider(username="u", password="p")
        assert provider.source_name != "carbon"
        assert provider.source_name != "watttime"


# ---------------------------------------------------------------------------
# Fixture response: canonical carbon schema
# ---------------------------------------------------------------------------

class TestWattTimeFixtureResponse:
    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_fetch_carbon_canonical_schema(self, mock_requests):
        mock_requests.get.side_effect = [
            _watttime_login_response(),
            _watttime_hist_response(num_5min_points=24),  # 2h of 5-min data
        ]

        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("us-west", T0, T1)

        assert list(df.columns) == CARBON_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-west").all()
        assert df["timestamp"].dtype.tz is not None  # UTC-aware

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_unit_conversion_lbs_to_gco2_per_kwh(self, mock_requests):
        """800 lbs/MWh should convert to 800 * 0.453592 gCO2/kWh ≈ 362.87."""
        mock_requests.get.side_effect = [
            _watttime_login_response(),
            _watttime_hist_response(num_5min_points=24),
        ]

        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("us-west", T0, T1)

        expected = 800.0 * _LBS_PER_MWH_TO_GCO2_PER_KWH
        assert abs(df["gco2_per_kwh"].mean() - expected) < 1.0

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_five_min_resampled_to_hourly(self, mock_requests):
        """12 five-minute points per hour should resample to 1 hourly row."""
        mock_requests.get.side_effect = [
            _watttime_login_response(),
            _watttime_hist_response(num_5min_points=12),  # exactly 1 hour
        ]

        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("us-west", T0, T1)

        # Should get 1 hourly row (or possibly 2 if the window straddles an hour boundary)
        assert 1 <= len(df) <= 2

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_source_column_identifies_moer(self, mock_requests):
        mock_requests.get.side_effect = [
            _watttime_login_response(),
            _watttime_hist_response(num_5min_points=24),
        ]

        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("us-west", T0, T1)

        assert "moer" in df["source"].iloc[0].lower()

    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_gco2_per_kwh_is_never_negative(self, mock_requests):
        mock_requests.get.side_effect = [
            _watttime_login_response(),
            _watttime_hist_response(num_5min_points=24),
        ]

        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("us-west", T0, T1)

        assert (df["gco2_per_kwh"] >= 0).all()


# ---------------------------------------------------------------------------
# Unknown region → empty (not an error)
# ---------------------------------------------------------------------------

class TestWattTimeUnknownRegion:
    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_unknown_region_returns_empty(self, mock_requests):
        # No login request expected for unknown region
        provider = WattTimeCarbonProvider(username="u", password="p")
        df = provider.fetch_carbon("xx-unknown", T0, T1)
        assert df.empty
        assert list(df.columns) == CARBON_COLUMNS


# ---------------------------------------------------------------------------
# Auth failure → ProviderConfigError
# ---------------------------------------------------------------------------

class TestWattTimeAuthFailure:
    @patch("aurelius.ingestion.grid_apis.watttime.requests")
    def test_login_401_raises_config_error(self, mock_requests):
        bad_resp = _make_mock_response({}, status_code=401)
        bad_resp.raise_for_status = MagicMock(side_effect=Exception("401"))
        mock_requests.get.return_value = bad_resp

        provider = WattTimeCarbonProvider(username="wrong", password="wrong")
        # Clear cached token so login is attempted
        provider._token = None

        with pytest.raises(ProviderConfigError):
            provider.fetch_carbon("us-west", T0, T1)
