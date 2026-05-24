"""Tests for the Open-Meteo weather ingestion adapter.

Network calls are NOT made in CI: the canonicalisation, schema, caching and
config logic are tested against synthetic JSON payloads identical in shape to
real Open-Meteo responses. A live smoke test is provided but skipped unless
AURELIUS_OPENMETEO_LIVE=1 is set.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from aurelius.ingestion.weather_provider import (
    CANONICAL_COLS,
    OpenMeteoConfig,
    OpenMeteoWeatherProvider,
    REGION_COORDS,
)


def _fake_payload(n=48, t0=10.0):
    times = pd.date_range("2026-01-07T00:00", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [t0 + i * 0.1 for i in range(n)],
            "relative_humidity_2m": [55.0] * n,
            "wind_speed_10m": [3.5] * n,
        }
    }


def test_canonicalize_produces_canonical_schema():
    df = OpenMeteoWeatherProvider._canonicalize(
        _fake_payload(), "us-south", source="open_meteo_era5",
        t="temperature_2m", rh="relative_humidity_2m", ws="wind_speed_10m",
    )
    assert list(df.columns) == CANONICAL_COLS
    assert len(df) == 48
    assert (df["region"] == "us-south").all()
    assert df["source"].iloc[0] == "open_meteo_era5"
    assert df.isnull().sum().sum() == 0


def test_canonicalize_hdd_cdd_consistency():
    # cold payload → HDD>0, CDD==0
    df = OpenMeteoWeatherProvider._canonicalize(
        _fake_payload(n=24, t0=-5.0), "us-south", source="t",
        t="temperature_2m", rh="relative_humidity_2m", ws="wind_speed_10m",
    )
    assert (df["hdd_f"] > 0).all()
    assert (df["cdd_f"] == 0).all()


def test_canonicalize_empty_payload():
    df = OpenMeteoWeatherProvider._canonicalize(
        {}, "us-south", source="t", t="temperature_2m",
        rh="relative_humidity_2m", ws="wind_speed_10m",
    )
    assert list(df.columns) == CANONICAL_COLS
    assert df.empty


def test_unknown_region_returns_empty():
    p = OpenMeteoWeatherProvider()
    assert p.fetch_historical("mars", "2026-01-01", "2026-01-02").empty


def test_previous_run_lead_day_validation():
    p = OpenMeteoWeatherProvider()
    with pytest.raises(ValueError):
        p.fetch_previous_run_forecast("us-south", "2026-01-01", "2026-01-02", lead_day=9)


def test_cache_roundtrip(tmp_path, monkeypatch):
    """A second identical request is served from cache (no second network call)."""
    p = OpenMeteoWeatherProvider(OpenMeteoConfig(cache_dir=str(tmp_path)))
    calls = {"n": 0}

    def fake_urlopen(url, timeout=0):
        calls["n"] += 1
        import io, json
        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def read(self_): return json.dumps(_fake_payload()).encode()
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df1 = p.fetch_historical("us-south", "2026-01-07", "2026-01-08")
    df2 = p.fetch_historical("us-south", "2026-01-07", "2026-01-08")
    assert calls["n"] == 1  # second call hit the cache
    assert len(df1) == len(df2) == 48


def test_region_coords_cover_benchmark_regions():
    for r in ["us-west", "us-east", "us-south"]:
        assert r in REGION_COORDS


@pytest.mark.skipif(os.environ.get("AURELIUS_OPENMETEO_LIVE") != "1",
                    reason="live network test; set AURELIUS_OPENMETEO_LIVE=1 to run")
def test_live_smoke():
    p = OpenMeteoWeatherProvider()
    df = p.fetch_historical("us-south", "2026-01-26", "2026-01-26")
    assert not df.empty
    assert df["temperature_c"].min() < 0  # Jan 26 2026 Houston freeze
