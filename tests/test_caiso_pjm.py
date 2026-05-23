"""Tests for CAISOPriceProvider, CAISORealtimePriceProvider, and PJMPriceProvider.

All HTTP calls are mocked; no network access required.
"""

import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import (
    PRICE_COLUMNS,
    ProviderConfigError,
)
from aurelius.ingestion.grid_apis.caiso import (
    CAISOPriceProvider,
    CAISORealtimePriceProvider,
    _extract_lmp_rows,
    _parse_zip_response,
)
from aurelius.ingestion.grid_apis.market_registry import assert_price_type_not_demand
from aurelius.ingestion.grid_apis.pjm import (
    PJMPriceProvider,
    PJMRealtimePriceProvider,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2024, 3, 2, 0, 0, tzinfo=UTC)  # 24-hour window

# Canonical CAISO NP15 trading hub used in all tests
_NP15_NODE = "TH_NP15_GEN-APND"


# ---------------------------------------------------------------------------
# Helpers to build fake CAISO ZIP/CSV responses
# ---------------------------------------------------------------------------

def _make_caiso_csv(num_hours=24, include_non_lmp=True) -> str:
    """Build a minimal CAISO OASIS PRC_LMP CSV string (hourly day-ahead)."""
    rows = ["INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE"]
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    for i in range(num_hours):
        ts = (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        rows.append(f"{ts},LMP,{45.0 + i},{_NP15_NODE}")
        if include_non_lmp:
            rows.append(f"{ts},MCE,{10.0},{_NP15_NODE}")  # must be filtered out
            rows.append(f"{ts},MCC,{5.0},{_NP15_NODE}")
    return "\n".join(rows)


def _make_caiso_rtm_csv(num_intervals=12, include_non_lmp=True) -> str:
    """Build a minimal CAISO OASIS PRC_INTVL_LMP CSV string (5-min real-time)."""
    rows = ["INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE"]
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    for i in range(num_intervals):
        ts = (base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        rows.append(f"{ts},LMP,{50.0 + i * 0.5},{_NP15_NODE}")
        if include_non_lmp:
            rows.append(f"{ts},MCE,{8.0},{_NP15_NODE}")
            rows.append(f"{ts},MCC,{3.0},{_NP15_NODE}")
    return "\n".join(rows)


def _make_caiso_zip(csv_content: str, filename: str = "PRC_LMP_DAM.csv") -> bytes:
    """Wrap CSV text in an in-memory ZIP file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_content)
    return buf.getvalue()


def _make_mock_response(content: bytes = b"", status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# CAISO day-ahead: _parse_zip_response (static helper)
# ---------------------------------------------------------------------------

class TestCAISOParseZip:
    def test_lmp_rows_extracted(self):
        csv = _make_caiso_csv(num_hours=3, include_non_lmp=False)
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE
        )
        assert len(rows) == 3
        for r in rows:
            assert "timestamp" in r
            assert "price_per_mwh" in r
            assert r["region"] == "us-west"

    def test_non_lmp_types_filtered_out(self):
        csv = _make_caiso_csv(num_hours=2, include_non_lmp=True)
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE
        )
        # Only LMP rows — MCE and MCC must be dropped
        assert len(rows) == 2

    def test_bad_zip_returns_empty(self):
        rows = _parse_zip_response(
            b"not a zip file", region="us-west", node=_NP15_NODE
        )
        assert rows == []

    def test_xml_error_in_zip_returns_empty(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("error.xml", "<Error><Message>CAISO OASIS error</Message></Error>")
        rows = _parse_zip_response(
            buf.getvalue(), region="us-west", node=_NP15_NODE
        )
        assert rows == []

    def test_timestamps_are_utc_aware(self):
        csv = _make_caiso_csv(num_hours=2, include_non_lmp=False)
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE
        )
        for r in rows:
            ts = r["timestamp"]
            assert ts.tzinfo is not None, "timestamp must be UTC-aware"
            assert str(ts.tzinfo) == "UTC"

    def test_correct_node_used_as_default(self):
        """Default hub map must use TH_NP15_GEN-APND (standard NP15 trading hub)."""
        provider = CAISOPriceProvider()
        assert provider._hub_map.get("us-west") == "TH_NP15_GEN-APND"

    def test_resultformat_6_in_params(self):
        """resultformat=6 must be present to request ZIP/CSV output explicitly."""
        from aurelius.ingestion.grid_apis.caiso import _fetch_lmp
        with patch("aurelius.ingestion.grid_apis.caiso.requests") as mock_requests:
            csv = _make_caiso_csv(num_hours=1, include_non_lmp=False)
            mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))
            from datetime import timezone
            t0 = datetime(2024, 3, 1, tzinfo=timezone.utc)
            t1 = datetime(2024, 3, 2, tzinfo=timezone.utc)
            _fetch_lmp(
                node=_NP15_NODE, region="us-west", start=t0, end=t1,
                queryname="PRC_LMP", market_run_id="DAM",
                source_name="caiso_oasis_dam", granularity="hourly", floor_to="h",
            )
            call_kwargs = mock_requests.get.call_args
            params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
            assert params.get("resultformat") == "6"


# ---------------------------------------------------------------------------
# CAISO day-ahead: fetch_prices canonical schema
# ---------------------------------------------------------------------------

class TestCAISOFetchPrices:
    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_fetch_prices_canonical_schema(self, mock_requests):
        csv = _make_caiso_csv(num_hours=24)
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-west").all()
        assert df["timestamp"].dtype.tz is not None  # UTC-aware

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_fetch_prices_source_name(self, mock_requests):
        csv = _make_caiso_csv(num_hours=3)
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["source"] == "caiso_oasis_dam").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_fetch_prices_currency_usd(self, mock_requests):
        csv = _make_caiso_csv(num_hours=3)
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["currency"] == "USD").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_source_granularity_is_hourly(self, mock_requests):
        csv = _make_caiso_csv(num_hours=3)
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["source_granularity"] == "hourly").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_unknown_region_returns_empty(self, mock_requests):
        provider = CAISOPriceProvider()
        df = provider.fetch_prices("eu-west", T0, T1)

        assert df.empty
        assert list(df.columns) == PRICE_COLUMNS
        mock_requests.get.assert_not_called()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_empty_response_returns_empty_df(self, mock_requests):
        empty_csv = "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(empty_csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert df.empty

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_price_per_mwh_is_numeric(self, mock_requests):
        csv = _make_caiso_csv(num_hours=5)
        mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))

        provider = CAISOPriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert pd.api.types.is_numeric_dtype(df["price_per_mwh"])
        assert not df["price_per_mwh"].isna().any()


# ---------------------------------------------------------------------------
# CAISO real-time: fetch_prices
# ---------------------------------------------------------------------------

class TestCAISORealtimeFetchPrices:
    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_fetch_prices_canonical_schema(self, mock_requests):
        csv = _make_caiso_rtm_csv(num_intervals=12)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-west").all()
        assert df["timestamp"].dtype.tz is not None  # UTC-aware

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_source_name_is_rtm(self, mock_requests):
        csv = _make_caiso_rtm_csv(num_intervals=3)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["source"] == "caiso_oasis_rtm").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_source_granularity_is_5min(self, mock_requests):
        csv = _make_caiso_rtm_csv(num_intervals=3)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["source_granularity"] == "5min").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_currency_usd(self, mock_requests):
        csv = _make_caiso_rtm_csv(num_intervals=3)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert (df["currency"] == "USD").all()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_5min_timestamps_not_floored_to_hour(self, mock_requests):
        """Real-time timestamps must preserve 5-minute precision, not be floored to hour."""
        csv = _make_caiso_rtm_csv(num_intervals=6, include_non_lmp=False)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert not df.empty
        # Timestamps should span multiple minutes (not all at :00)
        minutes = df["timestamp"].apply(lambda ts: ts.minute)
        assert minutes.nunique() > 1, "RTM timestamps must preserve 5-min intervals"

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_unknown_region_returns_empty(self, mock_requests):
        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("eu-west", T0, T1)

        assert df.empty
        assert list(df.columns) == PRICE_COLUMNS
        mock_requests.get.assert_not_called()

    @patch("aurelius.ingestion.grid_apis.caiso.requests")
    def test_non_lmp_types_filtered_out(self, mock_requests):
        csv = _make_caiso_rtm_csv(num_intervals=4, include_non_lmp=True)
        mock_requests.get.return_value = _make_mock_response(
            _make_caiso_zip(csv, filename="PRC_INTVL_LMP_RTM.csv")
        )

        provider = CAISORealtimePriceProvider()
        df = provider.fetch_prices("us-west", T0, T1)

        assert len(df) == 4  # only LMP rows, not MCE/MCC

    def test_default_node_is_np15_trading_hub(self):
        provider = CAISORealtimePriceProvider()
        assert provider._hub_map.get("us-west") == "TH_NP15_GEN-APND"


# ---------------------------------------------------------------------------
# RTM: parse_zip_response (static helper)
# ---------------------------------------------------------------------------

class TestCAISORealtimeParseZip:
    def test_rtm_rows_extracted(self):
        csv = _make_caiso_rtm_csv(num_intervals=6, include_non_lmp=False)
        from aurelius.ingestion.grid_apis.caiso import _parse_zip_response
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE, floor_to=None
        )
        assert len(rows) == 6

    def test_rtm_timestamps_are_utc_aware(self):
        csv = _make_caiso_rtm_csv(num_intervals=3, include_non_lmp=False)
        from aurelius.ingestion.grid_apis.caiso import _parse_zip_response
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE, floor_to=None
        )
        for r in rows:
            ts = r["timestamp"]
            assert ts.tzinfo is not None
            assert str(ts.tzinfo) == "UTC"

    def test_rtm_non_lmp_filtered(self):
        csv = _make_caiso_rtm_csv(num_intervals=3, include_non_lmp=True)
        from aurelius.ingestion.grid_apis.caiso import _parse_zip_response
        rows = _parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node=_NP15_NODE, floor_to=None
        )
        assert len(rows) == 3  # 3 LMP rows, not 9 (3 * LMP + MCE + MCC)


# ---------------------------------------------------------------------------
# DST-aware timestamp conversion tests
# ---------------------------------------------------------------------------

class TestCAISOTimestampHandling:
    def test_utc_timestamps_already_utc_no_change(self):
        """INTERVALSTARTTIME_GMT is already UTC — tz_convert("UTC") is a safe no-op."""
        ts_utc = pd.Timestamp("2024-03-10T10:00:00+00:00")  # daylight saving transition day
        ts_converted = ts_utc.tz_convert("UTC")
        assert ts_converted == ts_utc
        assert str(ts_converted.tzinfo) == "UTC"

    def test_spring_forward_dst_boundary(self):
        """Timestamps around US spring-forward (Mar 10 2024) convert correctly to UTC."""
        csv_content = (
            "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
            # 2024-03-10T09:00 UTC = 01:00 PST (before spring-forward at 02:00 PST = 10:00 UTC)
            "2024-03-10T09:00:00+00:00,LMP,44.0,TH_NP15_GEN-APND\n"
            # 2024-03-10T10:00 UTC = 03:00 PDT (after spring-forward; 02:00 PST skipped)
            "2024-03-10T10:00:00+00:00,LMP,52.0,TH_NP15_GEN-APND\n"
        )
        raw = pd.read_csv(io.StringIO(csv_content))
        rows = _extract_lmp_rows(raw, "us-west", "test", floor_to="h")
        assert len(rows) == 2
        # Both timestamps must be UTC-aware
        for r in rows:
            assert r["timestamp"].tzinfo is not None
            assert str(r["timestamp"].tzinfo) == "UTC"

    def test_fall_back_dst_boundary(self):
        """Timestamps around US fall-back (Nov 3 2024) convert correctly to UTC."""
        csv_content = (
            "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
            # 2024-11-03T08:00 UTC = 01:00 PDT (before fall-back)
            "2024-11-03T08:00:00+00:00,LMP,38.0,TH_NP15_GEN-APND\n"
            # 2024-11-03T09:00 UTC = 01:00 PST (after fall-back; 01:00 repeated)
            "2024-11-03T09:00:00+00:00,LMP,41.0,TH_NP15_GEN-APND\n"
        )
        raw = pd.read_csv(io.StringIO(csv_content))
        rows = _extract_lmp_rows(raw, "us-west", "test", floor_to="h")
        assert len(rows) == 2
        for r in rows:
            assert r["timestamp"].tzinfo is not None

    def test_no_naive_timestamps_in_output(self):
        """normalize_price_df must never produce naive (tz-unaware) timestamps."""
        csv = _make_caiso_csv(num_hours=3, include_non_lmp=False)
        with patch("aurelius.ingestion.grid_apis.caiso.requests") as mock_requests:
            mock_requests.get.return_value = _make_mock_response(_make_caiso_zip(csv))
            provider = CAISOPriceProvider()
            df = provider.fetch_prices("us-west", T0, T1)

        for ts in df["timestamp"]:
            assert ts.tzinfo is not None, f"Naive timestamp found: {ts}"


# ---------------------------------------------------------------------------
# CAISO: demand/load data must never be mapped to price_per_mwh
# ---------------------------------------------------------------------------

class TestCAISONoDemandAsPrice:
    def test_mw_column_from_prc_lmp_query_is_price(self):
        """CAISO 'MW' column in PRC_LMP response is a price (USD/MWh), not energy."""
        csv_content = (
            "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
            f"2024-03-01T00:00:00+00:00,LMP,55.0,{_NP15_NODE}\n"
        )
        raw = pd.read_csv(io.StringIO(csv_content))
        rows = _extract_lmp_rows(raw, "us-west", "test")
        assert len(rows) == 1
        assert rows[0]["price_per_mwh"] == 55.0

    def test_mw_column_from_prc_intvl_lmp_is_price(self):
        """CAISO 'MW' column in PRC_INTVL_LMP (RTM) response is a price, not energy."""
        csv_content = (
            "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
            f"2024-03-01T00:00:00+00:00,LMP,62.5,{_NP15_NODE}\n"
        )
        raw = pd.read_csv(io.StringIO(csv_content))
        rows = _extract_lmp_rows(raw, "us-west", "test", floor_to=None)
        assert len(rows) == 1
        assert rows[0]["price_per_mwh"] == 62.5

    def test_demand_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("demand")

    def test_load_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("load")

    def test_generation_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("net_generation")

    def test_interchange_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("interchange")

    def test_lmp_label_accepted_by_guard(self):
        assert_price_type_not_demand("day_ahead_lmp")  # must not raise

    def test_real_time_lmp_label_accepted_by_guard(self):
        assert_price_type_not_demand("real_time_lmp")  # must not raise


# ---------------------------------------------------------------------------
# PJM: credential guard
# ---------------------------------------------------------------------------

class TestPJMCredentialGuard:
    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("PJM_API_KEY", raising=False)
        provider = PJMPriceProvider(api_key="")
        with pytest.raises(ProviderConfigError, match="PJM_API_KEY"):
            provider.fetch_prices("us-east", T0, T1)

    def test_env_var_used_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("PJM_API_KEY", "env-key-123")
        provider = PJMPriceProvider()
        assert provider._api_key == "env-key-123"


# ---------------------------------------------------------------------------
# PJM: fixture response → canonical schema
# ---------------------------------------------------------------------------

def _make_pjm_response(num_hours=24, status_code=200):
    items = []
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    for i in range(num_hours):
        ts = base + pd.Timedelta(hours=i)
        items.append({
            "datetime_beginning_utc": ts.strftime("%Y-%m-%dT%H:%M:%S") + " UTC",
            "datetime_ending_utc": (ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S") + " UTC",
            "pnode_name": "WESTERN HUB",
            "total_lmp_da": 42.5 + i,
        })
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"items": items, "totalRows": num_hours}
    resp.raise_for_status = MagicMock()
    return resp


class TestPJMFetchPrices:
    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_fetch_prices_canonical_schema(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_response(num_hours=24)

        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-east").all()
        assert df["timestamp"].dtype.tz is not None

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_fetch_prices_source_name(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_response(num_hours=3)

        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert (df["source"] == "pjm_da_lmp").all()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_fetch_prices_currency_usd(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_response(num_hours=3)

        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert (df["currency"] == "USD").all()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_total_lmp_da_is_mapped_to_price_per_mwh(self, mock_requests):
        """total_lmp_da (a real $/MWh value) must map to price_per_mwh."""
        mock_requests.get.return_value = _make_pjm_response(num_hours=1)

        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert not df.empty
        assert df["price_per_mwh"].iloc[0] == pytest.approx(42.5)

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_unknown_region_returns_empty(self, mock_requests):
        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("eu-west", T0, T1)

        assert df.empty
        assert list(df.columns) == PRICE_COLUMNS
        mock_requests.get.assert_not_called()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_empty_response_returns_empty_df(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": [], "totalRows": 0}
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp

        provider = PJMPriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert df.empty

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_401_raises_config_error(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp

        provider = PJMPriceProvider(api_key="bad-key")
        with pytest.raises(ProviderConfigError, match="PJM API key"):
            provider.fetch_prices("us-east", T0, T1)


# ---------------------------------------------------------------------------
# PJM real-time: fixture response → canonical schema
# ---------------------------------------------------------------------------

def _make_pjm_rt_response(num_intervals=12, freq_minutes=5, status_code=200):
    """Build a fake PJM RT response.

    Mirrors the real API: datetime_beginning_utc is a bare ISO string with no
    "UTC" suffix, and the price lives in total_lmp_rt.
    """
    eastern = ZoneInfo("America/New_York")
    items = []
    base = pd.Timestamp("2026-05-20T04:00:00+00:00")  # 00:00 EPT
    for i in range(num_intervals):
        ts = base + pd.Timedelta(minutes=freq_minutes * i)
        items.append({
            "datetime_beginning_utc": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime_beginning_ept": (ts.tz_convert(eastern)
                                       .strftime("%Y-%m-%dT%H:%M:%S")),
            "pnode_name": "PJM-RTO",
            "total_lmp_rt": 30.0 + i,
        })
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"items": items, "totalRows": num_intervals}
    resp.raise_for_status = MagicMock()
    return resp


class TestPJMRealtimeCredentialGuard:
    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("PJM_API_KEY", raising=False)
        provider = PJMRealtimePriceProvider(api_key="")
        with pytest.raises(ProviderConfigError, match="PJM_API_KEY"):
            provider.fetch_prices("us-east", T0, T1)


class TestPJMRealtimeFetchPrices:
    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_fetch_prices_canonical_schema(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=12)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert list(df.columns) == PRICE_COLUMNS
        assert not df.empty
        assert (df["region"] == "us-east").all()
        assert df["timestamp"].dtype.tz is not None

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_source_name_is_rt(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=3)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert (df["source"] == "pjm_rt_lmp").all()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_default_granularity_is_5min(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=3)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert (df["source_granularity"] == "5min").all()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_default_uses_fivemin_endpoint(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=3)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        provider.fetch_prices("us-east", T0, T1)

        url = mock_requests.get.call_args[0][0]
        assert url.endswith("/rt_fivemin_hrl_lmps")

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_hourly_uses_hourly_endpoint_and_granularity(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(
            num_intervals=3, freq_minutes=60
        )

        provider = PJMRealtimePriceProvider(api_key="test-key", hourly=True)
        df = provider.fetch_prices("us-east", T0, T1)

        url = mock_requests.get.call_args[0][0]
        assert url.endswith("/rt_hrl_lmps")
        assert (df["source_granularity"] == "hourly").all()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_5min_timestamps_not_floored_to_hour(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=12)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        minutes = {ts.minute for ts in df["timestamp"]}
        assert minutes != {0}  # 5-min precision preserved, not collapsed to top-of-hour

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_total_lmp_rt_mapped_to_price(self, mock_requests):
        mock_requests.get.return_value = _make_pjm_rt_response(num_intervals=1)

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert df["price_per_mwh"].iloc[0] == pytest.approx(30.0)

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_unknown_region_returns_empty(self, mock_requests):
        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("eu-west", T0, T1)

        assert df.empty
        assert list(df.columns) == PRICE_COLUMNS
        mock_requests.get.assert_not_called()

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_401_raises_config_error(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp

        provider = PJMRealtimePriceProvider(api_key="bad-key")
        with pytest.raises(ProviderConfigError, match="PJM API key"):
            provider.fetch_prices("us-east", T0, T1)

    @patch("aurelius.ingestion.grid_apis.pjm.requests")
    def test_archive_error_returns_empty_without_retry(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "bad request"
        resp.json.return_value = {
            "errors": [{"field": "datetime", "message": "Archived data ..."}]
        }
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp

        provider = PJMRealtimePriceProvider(api_key="test-key")
        df = provider.fetch_prices("us-east", T0, T1)

        assert df.empty
        assert mock_requests.get.call_count == 1  # 4xx is not retried
