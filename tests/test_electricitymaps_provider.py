"""Tests for the Electricity Maps provider: sandbox flags, provenance,
endpoint construction, token redaction, and graceful credential handling.

All network access is mocked; no live API calls are made here. The live test
lives under tests/live and is skipped without ELECTRICITYMAPS_API_KEY.
"""

from datetime import datetime, timezone

import pytest

from aurelius.ingestion.grid_apis import electricitymaps as em
from aurelius.ingestion.grid_apis.base import ProviderConfigError
from aurelius.ingestion.market_data_provider import (
    BenchmarkDataError,
    MarketType,
    Provenance,
    Signal,
    assert_benchmark_admissible,
)

UTC = timezone.utc
START = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2025, 1, 1, 3, 0, tzinfo=UTC)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_CARBON_FIXTURE = {
    "zone": "US-CAL-CISO",
    "history": [
        {"datetime": "2025-01-01T00:00:00Z", "carbonIntensity": 210.0},
        {"datetime": "2025-01-01T01:00:00Z", "carbonIntensity": 215.0},
        {"datetime": "2025-01-01T02:00:00Z", "carbonIntensity": 205.0},
    ],
}


@pytest.fixture
def captured_requests(monkeypatch):
    """Patch requests.get and time.sleep; record calls."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        return _FakeResponse(_CARBON_FIXTURE)

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(em.time, "sleep", lambda *_: None)
    return calls


# ---------------------------------------------------------------------------
# Token never printed
# ---------------------------------------------------------------------------

class TestTokenRedaction:
    def test_repr_does_not_leak_token(self):
        p = em.ElectricityMapsProvider(api_key="super-secret-token-123")
        assert "super-secret-token-123" not in repr(p)
        assert "<set>" in repr(p)

    def test_legacy_repr_does_not_leak_token(self):
        p = em.ElectricityMapsCarbonProvider(api_key="super-secret-token-123")
        assert "super-secret-token-123" not in repr(p)


# ---------------------------------------------------------------------------
# Endpoint construction + parsing (fixture-backed)
# ---------------------------------------------------------------------------

class TestEndpointAndParsing:
    def test_endpoint_and_auth_header(self, captured_requests):
        p = em.ElectricityMapsProvider(api_key="tok")
        p.fetch_carbon_series("us-west", START, END)
        assert captured_requests, "expected at least one HTTP call"
        call = captured_requests[0]
        assert call["url"].endswith("/carbon-intensity/past-range")
        assert call["params"]["zone"] == "US-CAL-CISO"
        assert call["headers"]["auth-token"] == "tok"

    def test_fixture_parses_to_carbon_points(self, captured_requests):
        p = em.ElectricityMapsProvider(api_key="tok")
        pts = p.fetch_carbon_series("us-west", START, END)
        assert len(pts) == 3
        assert pts[0].gco2_per_kwh == 210.0
        assert pts[0].region == "us-west"
        assert pts[0].provider == "electricitymaps"
        assert pts[0].provenance == Provenance.AGGREGATED
        assert pts[0].is_sandbox is False
        assert pts[0].benchmark_admissible is True


# ---------------------------------------------------------------------------
# Sandbox flag propagation + benchmark refusal
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_env_enables_sandbox(self, monkeypatch):
        monkeypatch.setenv("ELECTRICITYMAPS_SANDBOX", "true")
        assert em.sandbox_enabled() is True
        p = em.ElectricityMapsProvider(api_key="tok")
        assert p.is_sandbox() is True

    def test_sandbox_points_flagged_and_rejected_for_benchmark(self, captured_requests, monkeypatch):
        monkeypatch.setenv("ELECTRICITYMAPS_SANDBOX", "1")
        p = em.ElectricityMapsProvider(api_key="tok")
        pts = p.fetch_carbon_series("us-west", START, END)
        assert pts and all(pt.is_sandbox for pt in pts)
        assert all(pt.provenance == Provenance.SANDBOX for pt in pts)
        assert all(not pt.benchmark_admissible for pt in pts)
        with pytest.raises(BenchmarkDataError):
            assert_benchmark_admissible(pts)

    def test_production_mode_not_sandbox_by_default(self, monkeypatch):
        monkeypatch.delenv("ELECTRICITYMAPS_SANDBOX", raising=False)
        p = em.ElectricityMapsProvider(api_key="tok")
        assert p.is_sandbox() is False


# ---------------------------------------------------------------------------
# Price: EM is never a price source-of-truth / LMP refusal
# ---------------------------------------------------------------------------

class TestPriceRefusal:
    def test_lmp_request_returns_empty(self):
        p = em.ElectricityMapsProvider(api_key="tok")
        assert p.fetch_price_series("us-west", START, END, MarketType.DAY_AHEAD_LMP) == []

    def test_any_price_request_returns_empty(self):
        p = em.ElectricityMapsProvider(api_key="tok")
        assert p.fetch_price_series("eu-west", START, END, MarketType.DAY_AHEAD_PRICE) == []

    def test_capabilities_declare_no_price_regions(self):
        p = em.ElectricityMapsProvider(api_key="tok")
        caps = {c.signal: c for c in p.get_capabilities()}
        assert caps[Signal.PRICE].regions == ()
        assert caps[Signal.PRICE].production_supported is False
        assert caps[Signal.CARBON].production_supported is True


# ---------------------------------------------------------------------------
# Graceful credential handling
# ---------------------------------------------------------------------------

class TestCredentials:
    def test_validate_credentials_false_when_absent(self, monkeypatch):
        monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
        p = em.ElectricityMapsProvider(api_key="")
        assert p.validate_credentials() is False

    def test_validate_credentials_true_when_present(self):
        assert em.ElectricityMapsProvider(api_key="tok").validate_credentials() is True

    def test_fetch_without_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
        p = em.ElectricityMapsProvider(api_key="")
        with pytest.raises(ProviderConfigError):
            p.fetch_carbon_series("us-west", START, END)
