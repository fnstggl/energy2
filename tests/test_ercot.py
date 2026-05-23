"""Tests for ERCOTPriceProvider and ERCOTRealtimePriceProvider.

All HTTP calls (OAuth token + data) are mocked; no network access required.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import PRICE_COLUMNS, ProviderConfigError
from aurelius.ingestion.grid_apis.ercot import (
    _TOKEN_CACHE,
    ERCOTPriceProvider,
    ERCOTRealtimePriceProvider,
    _localize_central_to_utc,
    _parse_hour_ending,
)

UTC = timezone.utc
# Winter window (no DST transition) so Central->UTC is a clean +6h (CST).
T0 = datetime(2024, 2, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2024, 2, 10, 0, 0, tzinfo=UTC)

_HUB = "HB_HOUSTON"

_DAM_FIELDS = [
    "deliveryDate", "hourEnding", "settlementPoint", "settlementPointPrice", "DSTFlag",
]
_RT_FIELDS = [
    "deliveryDate", "deliveryHour", "deliveryInterval",
    "settlementPointName", "settlementPointType", "settlementPointPrice", "DSTFlag",
]


def _resp(fields, data, total_pages=1, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {
        "fields": [{"name": n} for n in fields],
        "data": data,
        "_meta": {"totalPages": total_pages},
    }
    r.text = ""
    return r


def _dam_data(num_hours=24, point=_HUB, date="2024-02-05"):
    return [[date, str(he), point, 20.0 + he, "N"] for he in range(1, num_hours + 1)]


def _rt_data(num_hours=2, point=_HUB, date="2024-02-05"):
    rows = []
    for h in range(1, num_hours + 1):
        for iv in range(1, 5):  # four 15-min intervals per hour
            rows.append([date, h, iv, point, "HU", 30.0 + h + iv, "N"])
    return rows


@pytest.fixture(autouse=True)
def _clear_token_cache(monkeypatch):
    _TOKEN_CACHE.clear()
    # Bypass the OAuth ROPC POST in fetch tests by supplying a token directly.
    monkeypatch.setenv("ERCOT_ID_TOKEN", "test-id-token")
    yield
    _TOKEN_CACHE.clear()


# ---------------------------------------------------------------------------
# Credential guard
# ---------------------------------------------------------------------------

class TestERCOTCredentialGuard:
    def test_missing_subscription_key_raises(self, monkeypatch):
        monkeypatch.delenv("ERCOT_API_KEY", raising=False)
        provider = ERCOTPriceProvider(api_key="", username="u", password="p")
        with pytest.raises(ProviderConfigError, match="ERCOT"):
            provider.fetch_prices("us-south", T0, T1)

    def test_missing_user_pass_and_token_raises(self, monkeypatch):
        monkeypatch.delenv("ERCOT_ID_TOKEN", raising=False)
        provider = ERCOTPriceProvider(api_key="key", username="", password="")
        with pytest.raises(ProviderConfigError, match="ERCOT"):
            provider.fetch_prices("us-south", T0, T1)

    def test_id_token_satisfies_creds(self):
        # ERCOT_ID_TOKEN is set by the fixture; api_key present → creds OK.
        provider = ERCOTPriceProvider(api_key="key", username="", password="")
        provider._check_creds()  # must not raise


# ---------------------------------------------------------------------------
# Day-ahead fetch
# ---------------------------------------------------------------------------

class TestERCOTDayAheadFetch:
    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_canonical_schema(self, mock_requests):
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(24))
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-south").all()
        assert df["timestamp"].dtype.tz is not None

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_source_currency_granularity(self, mock_requests):
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(3))
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)

        assert (df["source"] == "ercot_dam_spp").all()
        assert (df["currency"] == "USD").all()
        assert (df["source_granularity"] == "hourly").all()

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_spp_mapped_to_price(self, mock_requests):
        # hourEnding=1 → 20.0+1 = 21.0
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(1))
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)
        assert df["price_per_mwh"].iloc[0] == pytest.approx(21.0)

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_hour_ending_to_hour_beginning_utc(self, mock_requests):
        # hourEnding=1 on 2024-02-05 Central = hour beginning 00:00 CST = 06:00 UTC.
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(1))
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)
        ts = df["timestamp"].iloc[0]
        assert ts == pd.Timestamp("2024-02-05T06:00:00Z")

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_settlement_point_query_param(self, mock_requests):
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(2))
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        provider.fetch_prices("us-south", T0, T1)
        params = mock_requests.get.call_args[1]["params"]
        assert params["settlementPoint"] == "HB_HOUSTON"
        assert "deliveryDateFrom" in params and "deliveryDateTo" in params

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_uses_dam_endpoint_and_subscription_header(self, mock_requests):
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(2))
        provider = ERCOTPriceProvider(api_key="sub-key-123", username="u", password="p")
        provider.fetch_prices("us-south", T0, T1)
        url = mock_requests.get.call_args[0][0]
        headers = mock_requests.get.call_args[1]["headers"]
        assert url.endswith("/np4-190-cd/dam_stlmnt_pnt_prices")
        assert headers["Ocp-Apim-Subscription-Key"] == "sub-key-123"
        assert headers["Authorization"] == "Bearer test-id-token"

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_unknown_region_returns_empty_no_call(self, mock_requests):
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-west", T0, T1)
        assert df.empty
        assert list(df.columns) == PRICE_COLUMNS
        mock_requests.get.assert_not_called()

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_foreign_settlement_point_filtered_out(self, mock_requests):
        # A row for a different point must be dropped even if the API returns it.
        data = _dam_data(2, point="HB_HOUSTON") + _dam_data(2, point="HB_NORTH")
        mock_requests.get.return_value = _resp(_DAM_FIELDS, data)
        provider = ERCOTPriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Real-time fetch
# ---------------------------------------------------------------------------

class TestERCOTRealtimeFetch:
    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_canonical_schema_and_granularity(self, mock_requests):
        mock_requests.get.return_value = _resp(_RT_FIELDS, _rt_data(2))
        provider = ERCOTRealtimePriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["source"] == "ercot_rt_spp").all()
        assert (df["source_granularity"] == "15min").all()

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_uses_rt_endpoint(self, mock_requests):
        mock_requests.get.return_value = _resp(_RT_FIELDS, _rt_data(1))
        provider = ERCOTRealtimePriceProvider(api_key="key", username="u", password="p")
        provider.fetch_prices("us-south", T0, T1)
        url = mock_requests.get.call_args[0][0]
        assert url.endswith("/np6-905-cd/spp_node_zone_hub")

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_15min_intervals_preserved(self, mock_requests):
        mock_requests.get.return_value = _resp(_RT_FIELDS, _rt_data(2))
        provider = ERCOTRealtimePriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)
        minutes = {ts.minute for ts in df["timestamp"]}
        assert minutes == {0, 15, 30, 45}

    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_interval_to_timestamp_utc(self, mock_requests):
        # deliveryHour=1, deliveryInterval=1 → 00:00 CST = 06:00 UTC.
        mock_requests.get.return_value = _resp(_RT_FIELDS, [["2024-02-05", 1, 1, _HUB, "HU", 33.0, "N"]])
        provider = ERCOTRealtimePriceProvider(api_key="key", username="u", password="p")
        df = provider.fetch_prices("us-south", T0, T1)
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05T06:00:00Z")
        # deliveryHour=1, deliveryInterval=3 → 00:30 CST = 06:30 UTC.
        mock_requests.get.return_value = _resp(_RT_FIELDS, [["2024-02-05", 1, 3, _HUB, "HU", 33.0, "N"]])
        df = provider.fetch_prices("us-south", T0, T1)
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-02-05T06:30:00Z")


# ---------------------------------------------------------------------------
# OAuth ROPC token flow
# ---------------------------------------------------------------------------

class TestERCOTTokenFlow:
    @patch("aurelius.ingestion.grid_apis.ercot.requests")
    def test_ropc_post_when_no_id_token(self, mock_requests, monkeypatch):
        monkeypatch.delenv("ERCOT_ID_TOKEN", raising=False)
        _TOKEN_CACHE.clear()
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.json.return_value = {"id_token": "fresh-token"}
        mock_requests.post.return_value = token_resp
        mock_requests.get.return_value = _resp(_DAM_FIELDS, _dam_data(1))

        provider = ERCOTPriceProvider(api_key="key", username="me@x.com", password="pw")
        provider.fetch_prices("us-south", T0, T1)

        post_kwargs = mock_requests.post.call_args
        body = post_kwargs[1]["data"]
        assert body["grant_type"] == "password"
        assert body["client_id"] == "fec253ea-0d06-4272-a5e6-b478baeecd70"
        assert body["response_type"] == "id_token"
        assert body["username"] == "me@x.com"
        # The fetched token is used as the bearer on the data request.
        assert mock_requests.get.call_args[1]["headers"]["Authorization"] == "Bearer fresh-token"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestERCOTHelpers:
    def test_parse_hour_ending_variants(self):
        assert _parse_hour_ending("1") == 1
        assert _parse_hour_ending("13") == 13
        assert _parse_hour_ending("24") == 24
        assert _parse_hour_ending("01:00") == 1
        assert _parse_hour_ending("24:00") == 24

    def test_localize_spring_forward_no_error(self):
        # 2024-03-10 02:30 Central does not exist (spring-forward) → shift_forward.
        out = _localize_central_to_utc([pd.Timestamp("2024-03-10 02:30")], ["N"])
        assert out[0].tzinfo is not None
        assert str(out.tz) == "UTC"

    def test_localize_fall_back_ambiguous_uses_dst_flag(self):
        # 2024-11-03 01:30 occurs twice. DSTFlag "N" → first/DST (CDT, UTC-5);
        # "Y" → second/standard (CST, UTC-6). They must differ by one hour.
        first = _localize_central_to_utc([pd.Timestamp("2024-11-03 01:30")], ["N"])[0]
        second = _localize_central_to_utc([pd.Timestamp("2024-11-03 01:30")], ["Y"])[0]
        assert (second - first) == pd.Timedelta(hours=1)

    def test_localize_accepts_boolean_dst_flag(self):
        # The live ERCOT API returns DSTFlag as a JSON boolean, not "Y"/"N".
        # bool False = first/DST occurrence; bool True = repeated standard hour.
        first = _localize_central_to_utc([pd.Timestamp("2024-11-03 01:30")], [False])[0]
        second = _localize_central_to_utc([pd.Timestamp("2024-11-03 01:30")], [True])[0]
        assert (second - first) == pd.Timedelta(hours=1)
