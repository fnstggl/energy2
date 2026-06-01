#!/usr/bin/env python3
"""Bounded live fetch of energy-market + carbon + GPU-price overlays for the
analysis-tier Economic Overlay corpus.

Writes committable normalized JSONL into
``data/external/economic_overlay/economic_overlay_samples/`` and records a
per-provider success/failure manifest (NO credential values) at
``data/external/economic_overlay/market_fetch_manifest.json``.

Providers:
  - PJM Data Miner DA LMP (live; PJM_API_KEY)
  - ERCOT SPP DA (live; ERCOT_API_KEY + ERCOT_USERNAME/PASSWORD or ID token)
  - CAISO OASIS LMP (live; no key required)
  - WattTime MOER carbon intensity (live; WATTTIME_USERNAME/PASSWORD)
  - afhubbard/gpu-prices multi-day snapshots (HF; CC-BY-4.0)

All credentials read from env only; never printed or committed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT))

OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
SAMPLES = OVERLAY_DIR / "economic_overlay_samples"
RAW = OVERLAY_DIR / "raw"


def _redact(msg: str) -> str:
    """Strip any value that looks like a credential from an error string."""
    out = msg
    for k in ("PJM_API_KEY", "ERCOT_API_KEY", "ERCOT_PASSWORD",
              "WATTTIME_PASSWORD", "HF_TOKEN", "ERCOT_ID_TOKEN"):
        v = os.environ.get(k)
        if v:
            out = out.replace(v, f"<{k}>")
    return out


def fetch_pjm(manifest: dict, days: int = 14) -> None:
    name = "pjm_da_energy_price"
    try:
        from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
            - timedelta(days=1)
        start = end - timedelta(days=days)
        df = PJMPriceProvider().fetch_prices(region="us-east", start=start, end=end)
        out = SAMPLES / f"{name}_{days}day.jsonl"
        df.to_json(out, orient="records", lines=True, date_format="iso")
        manifest[name] = {"status": "success", "rows": int(len(df)),
                          "value_quality": "measured", "market": "PJM",
                          "region": "us-east", "window_days": days,
                          "path": str(out.relative_to(REPO_ROOT))}
        print(f"PJM: {len(df)} rows -> {out.name}")
    except Exception as e:  # noqa: BLE001
        manifest[name] = {"status": "failed", "value_quality": "scenario_prior_fallback",
                          "error": _redact(f"{type(e).__name__}: {e}")[:200]}
        print(f"PJM: FAILED ({type(e).__name__})", file=sys.stderr)


def fetch_ercot(manifest: dict, days: int = 14) -> None:
    name = "ercot_da_energy_price"
    try:
        from aurelius.ingestion.grid_apis.ercot import ERCOTPriceProvider
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
            - timedelta(days=1)
        start = end - timedelta(days=days)
        df = ERCOTPriceProvider().fetch_prices(region="us-south", start=start, end=end)
        out = SAMPLES / f"{name}_{days}day.jsonl"
        df.to_json(out, orient="records", lines=True, date_format="iso")
        manifest[name] = {"status": "success", "rows": int(len(df)),
                          "value_quality": "measured", "market": "ERCOT",
                          "region": "us-south", "window_days": days,
                          "path": str(out.relative_to(REPO_ROOT))}
        print(f"ERCOT: {len(df)} rows -> {out.name}")
    except Exception as e:  # noqa: BLE001
        manifest[name] = {"status": "failed", "value_quality": "scenario_prior_fallback",
                          "error": _redact(f"{type(e).__name__}: {e}")[:200]}
        print(f"ERCOT: FAILED ({type(e).__name__})", file=sys.stderr)


def fetch_caiso(manifest: dict, days: int = 7) -> None:
    name = "caiso_da_energy_price"
    try:
        from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
            - timedelta(days=2)
        start = end - timedelta(days=days)
        df = CAISOPriceProvider().fetch_prices(region="us-west", start=start, end=end)
        out = SAMPLES / f"{name}_{days}day.jsonl"
        df.to_json(out, orient="records", lines=True, date_format="iso")
        manifest[name] = {"status": "success", "rows": int(len(df)),
                          "value_quality": "measured", "market": "CAISO",
                          "region": "us-west", "window_days": days,
                          "path": str(out.relative_to(REPO_ROOT))}
        print(f"CAISO: {len(df)} rows -> {out.name}")
    except Exception as e:  # noqa: BLE001
        manifest[name] = {"status": "failed", "value_quality": "scenario_prior_fallback",
                          "error": _redact(f"{type(e).__name__}: {e}")[:200]}
        print(f"CAISO: FAILED ({type(e).__name__})", file=sys.stderr)


def fetch_watttime(manifest: dict, days: int = 7) -> None:
    name = "watttime_carbon_intensity"
    try:
        from aurelius.ingestion.grid_apis.watttime import WattTimeCarbonProvider
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
            - timedelta(days=1)
        start = end - timedelta(days=days)
        prov = WattTimeCarbonProvider()
        rows = []
        for region in ("us-east", "us-west", "us-south"):
            try:
                df = prov.fetch_carbon(region=region, start=start, end=end)
                df["region"] = region
                rows.append(df)
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            raise RuntimeError("no region returned carbon data")
        import pandas as pd
        allp = pd.concat(rows, ignore_index=True)
        out = SAMPLES / f"{name}_{days}day.jsonl"
        allp.to_json(out, orient="records", lines=True, date_format="iso")
        manifest[name] = {"status": "success", "rows": int(len(allp)),
                          "value_quality": "measured", "provider": "WattTime",
                          "regions": ["us-east", "us-west", "us-south"],
                          "window_days": days, "units": "g_co2_per_kwh",
                          "path": str(out.relative_to(REPO_ROOT))}
        print(f"WattTime: {len(allp)} rows -> {out.name}")
    except Exception as e:  # noqa: BLE001
        manifest[name] = {"status": "failed", "value_quality": "scenario_prior_fallback",
                          "error": _redact(f"{type(e).__name__}: {e}")[:200]}
        print(f"WattTime: FAILED ({type(e).__name__})", file=sys.stderr)


def fetch_gpu_prices_multiday(manifest: dict, n_days: int = 5) -> None:
    name = "gpu_price_overlay_multiday"
    try:
        import pandas as pd
        tok = os.environ.get("HF_TOKEN")
        hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
        info = json.load(urllib.request.urlopen(urllib.request.Request(
            "https://huggingface.co/api/datasets/afhubbard/gpu-prices", headers=hdr),
            timeout=30))
        parq = sorted(s["rfilename"] for s in info.get("siblings", [])
                      if s["rfilename"].endswith(".parquet"))
        chosen = parq[-n_days:]
        RAW.mkdir(parents=True, exist_ok=True)
        frames = []
        for rp in chosen:
            local = RAW / rp.replace("/", "__")
            url = f"https://huggingface.co/datasets/afhubbard/gpu-prices/resolve/main/{rp}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr),
                                        timeout=120) as r, open(local, "wb") as f:
                f.write(r.read())
            frames.append(pd.read_parquet(local))
        df = pd.concat(frames, ignore_index=True)
        gpu = df[(df["gpu_count"] > 0) & (df["gpu_type"] != "Unknown")].copy()
        gpu["price_per_gpu_hour"] = gpu["price_per_hour"] / gpu["gpu_count"]
        gpu["snapshot_date"] = gpu["timestamp"].astype(str).str[:10]
        agg = (gpu.groupby(["snapshot_date", "provider", "gpu_type", "region", "is_spot"],
                           dropna=False)
                  .agg(price_per_gpu_hour_usd=("price_per_gpu_hour", "median"),
                       price_per_hour_usd=("price_per_hour", "median"),
                       gpu_count_typical=("gpu_count", "median"),
                       gpu_memory_gb=("gpu_memory_gb", "median"),
                       sample_rows=("price_per_hour", "size"))
                  .reset_index())
        agg["source_dataset_id"] = "afhubbard/gpu-prices"
        agg["source_license"] = "cc-by-4.0"
        agg["value_quality"] = "prior_public_list_price"
        agg["snapshot_timestamp"] = agg["snapshot_date"] + "T12:00:00Z"
        out = SAMPLES / "gpu_price_overlay_multiday.jsonl"
        agg.to_json(out, orient="records", lines=True, force_ascii=False)
        manifest[name] = {"status": "success", "rows": int(len(agg)),
                          "value_quality": "prior_public_list_price",
                          "source_dataset_id": "afhubbard/gpu-prices",
                          "license": "cc-by-4.0", "snapshots": chosen,
                          "raw_listings_aggregated": int(len(gpu)),
                          "path": str(out.relative_to(REPO_ROOT))}
        print(f"GPU prices: {len(agg)} aggregated rows from {len(chosen)} days "
              f"-> {out.name}")
    except Exception as e:  # noqa: BLE001
        manifest[name] = {"status": "failed",
                          "error": _redact(f"{type(e).__name__}: {e}")[:200]}
        print(f"GPU prices: FAILED ({type(e).__name__})", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pjm-days", type=int, default=14)
    p.add_argument("--ercot-days", type=int, default=14)
    p.add_argument("--caiso-days", type=int, default=7)
    p.add_argument("--watttime-days", type=int, default=7)
    p.add_argument("--gpu-price-days", type=int, default=5)
    args = p.parse_args(argv)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                      "providers": {}}
    fetch_pjm(manifest["providers"], args.pjm_days)
    fetch_ercot(manifest["providers"], args.ercot_days)
    fetch_caiso(manifest["providers"], args.caiso_days)
    fetch_watttime(manifest["providers"], args.watttime_days)
    fetch_gpu_prices_multiday(manifest["providers"], args.gpu_price_days)
    mpath = OVERLAY_DIR / "market_fetch_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nmanifest -> {mpath.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
