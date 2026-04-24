"""Tests for CAISOPriceProvider and PJMPriceProvider (unit tests with mocked HTTP)."""

import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aurelius.ingestion.grid_apis.base import (
    PRICE_COLUMNS,
    ProviderConfigError,
    empty_price_df,
)
from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider, _extract_lmp_rows
from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider
from aurelius.ingestion.grid_apis.market_registry import assert_price_type_not_demand

UTC = timezone.utc
T0 = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2024, 3, 2, 0, 0, tzinfo=UTC)  # 24-hour window


# ---------------------------------------------------------------------------
# Helpers to build fake CAISO ZIP/CSV responses
# ---------------------------------------------------------------------------

def _make_caiso_csv(num_hours=24, include_non_lmp=True) -> str:
    """Build a minimal CAISO OASIS PRC_LMP CSV string."""
    rows = ["INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE"]
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    for i in range(num_hours):
        ts = (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        rows.append(f"{ts},LMP,{45.0 + i},NP15_7_N001")
        if include_non_lmp:
            rows.append(f"{ts},MCE,{10.0},NP15_7_N001")  # should be filtered out
            rows.append(f"{ts},MCC,{5.0},NP15_7_N001")
    return "\n".join(rows)


def _make_caiso_zip(csv_content: str) -> bytes:
    """Wrap CSV text in an in-memory ZIP file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("PRC_LMP_DAM_20240301.csv", csv_content)
    return buf.getvalue()


def _make_mock_response(content: bytes = b"", status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# CAISO: parse_zip_response
# ---------------------------------------------------------------------------

class TestCAISOParseZip:
    def test_lmp_rows_extracted(self):
        csv = _make_caiso_csv(num_hours=3, include_non_lmp=False)
        rows = CAISOPriceProvider._parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node="NP15_7_N001"
        )
        assert len(rows) == 3
        for r in rows:
            assert "timestamp" in r
            assert "price_per_mwh" in r
            assert r["region"] == "us-west"

    def test_non_lmp_types_filtered_out(self):
        csv = _make_caiso_csv(num_hours=2, include_non_lmp=True)
        rows = CAISOPriceProvider._parse_zip_response(
            _make_caiso_zip(csv), region="us-west", node="NP15_7_N001"
        )
        # Only LMP rows — MCE and MCC rows must be dropped
        assert len(rows) == 2

    def test_bad_zip_returns_empty(self):
        rows = CAISOPriceProvider._parse_zip_response(
            b"not a zip file", region="us-west", node="NP15"
        )
        assert rows == []

    def test_xml_error_in_zip_returns_empty(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("error.xml", "<Error><Message>CAISO OASIS error</Message></Error>")
        rows = CAISOPriceProvider._parse_zip_response(
            buf.getvalue(), region="us-west", node="NP15"
        )
        assert rows == []


# ---------------------------------------------------------------------------
# CAISO: fetch_prices canonical schema
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


# ---------------------------------------------------------------------------
# CAISO: demand/load data must never be mapped to price_per_mwh
# ---------------------------------------------------------------------------

class TestCAISONoDemandAsPrice:
    def test_mw_column_from_prc_lmp_query_is_price(self):
        """CAISO 'MW' column in PRC_LMP response is a price (USD/MWh), not energy."""
        csv_content = (
            "INTERVALSTARTTIME_GMT,LMP_TYPE,MW,NODE\n"
            "2024-03-01T00:00:00+00:00,LMP,55.0,NP15_7_N001\n"
        )
        raw = pd.read_csv(io.StringIO(csv_content))
        rows = _extract_lmp_rows(raw, "us-west", "test")
        assert len(rows) == 1
        assert rows[0]["price_per_mwh"] == 55.0

    def test_demand_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("demand")

    def test_load_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("load")

    def test_generation_type_label_rejected_by_guard(self):
        with pytest.raises(ValueError):
            assert_price_type_not_demand("net_generation")


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
