"""DCGM/Prometheus GPU telemetry ingestion for Aurelius.

Provides GPU health signals from DCGM (Data Center GPU Manager) metrics
exposed via dcgm-exporter and scraped by Prometheus. Enables Tier 3
GPU/node-level placement intelligence.

Supported ingestion modes:
  1. Fixture (always available): Load pre-recorded Prometheus text format
     from a .prom file — no cluster required. Used for tests and demos.
  2. CSV (always available): Load from a canonical CSV with DCGM columns.
  3. Live Prometheus (optional): Query a real Prometheus endpoint — enabled
     only when PROMETHEUS_URL (or DCGM_EXPORTER_URL) is set.

IMPORTANT — observability vs. control:
  DCGM tells Aurelius WHICH GPUs are hot/degraded/throttled. It does NOT
  automatically route jobs to specific GPUs. Exact GPU placement requires
  scheduler adapter support (Kubernetes node selectors / Slurm GRES / Ray
  resource labels). The gpu_health_data dict (region→timestamp→penalty) is
  used as a region-level routing signal when per-GPU control is unavailable,
  and as a node-selection hint when the scheduler adapter supports it.

Leakage safety:
  get_health_score() and to_dict_lookup() use a "last known ≤ T" lookup,
  never exposing future GPU state to the optimizer.

Environment variables (all optional):
  PROMETHEUS_URL           Base URL of Prometheus server
  DCGM_EXPORTER_URL        Direct dcgm-exporter /metrics URL (alternative)
  PROMETHEUS_BEARER_TOKEN  Bearer token for auth
  PROMETHEUS_USERNAME      Basic-auth username
  PROMETHEUS_PASSWORD      Basic-auth password
  PROMETHEUS_TLS_VERIFY    "false" to disable TLS certificate verification

gpu_health_data lookup format (mirrors price_data / queue_data):
  {region: {timestamp_floor_hour: avg_health_penalty (0.0–1.0)}}
"""

from __future__ import annotations

import bisect
import csv
import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Optional

from aurelius.models import GPUHealthScore, GPUMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DCGM metric names used in Prometheus text format
_DCGM_GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
_DCGM_MEM_USED = "DCGM_FI_DEV_FB_USED"
_DCGM_MEM_TOTAL = "DCGM_FI_DEV_FB_FREE"  # used to derive total
_DCGM_POWER = "DCGM_FI_DEV_POWER_USAGE"
_DCGM_TEMP = "DCGM_FI_DEV_GPU_TEMP"
_DCGM_ECC_SBE = "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL"
_DCGM_ECC_DBE = "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
_DCGM_XID = "DCGM_FI_DEV_XID_ERRORS"
_DCGM_POWER_THROTTLE = "DCGM_FI_DEV_POWER_VIOLATION"
_DCGM_THERMAL_THROTTLE = "DCGM_FI_DEV_THERMAL_VIOLATION"
_DCGM_CLOCK_THROTTLE = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"

# Health-score thresholds
_UTIL_WARN = 80.0        # % above which utilization penalty rises
_TEMP_SAFE = 70.0        # °C below which thermal penalty = 0
_TEMP_CRITICAL = 95.0    # °C above which GPU is unschedulable
_THROTTLE_MAX_US = 1e6   # μs at which throttle penalty reaches 1.0

# CSV canonical columns
_CSV_COLS = {
    "timestamp", "region", "node_id", "gpu_index", "gpu_uuid", "gpu_type",
    "gpu_util_pct", "mem_used_mb", "mem_total_mb", "power_usage_w",
    "gpu_temp_c", "ecc_sbe_count", "ecc_dbe_count", "xid_error_count",
    "power_throttle_us", "thermal_throttle_us", "clock_throttle_reasons",
}


# ---------------------------------------------------------------------------
# GPU health scoring (pure, no I/O)
# ---------------------------------------------------------------------------

def score_gpu_health(metrics: GPUMetrics) -> GPUHealthScore:
    """Compute a health score for a single GPU snapshot.

    Lower health_penalty is better (0.0 = perfectly healthy, 1.0 = severely degraded).

    Component penalties:
      utilization_penalty: rises linearly from 0 at UTIL_WARN% to 1 at 100%
      thermal_penalty:     rises linearly from 0 at TEMP_SAFE°C to 1 at TEMP_CRITICAL°C
      throttle_penalty:    rises linearly from 0 at 0μs to 1 at THROTTLE_MAX_US
      ecc_penalty:         1.0 for any DBE (uncorrectable); 0.5 for ≥5 SBE; 0.0 otherwise

    is_schedulable = True unless:
      - ecc_dbe_count > 0 (GPU should be retired)
      - gpu_temp_c >= TEMP_CRITICAL (thermal runaway risk)
      - xid_error_count > 0 (fatal XID errors signal hardware fault)
    """
    reason_codes: list[str] = []

    # Utilization penalty: how "busy" is the GPU (linear above warn threshold)
    util_frac = metrics.gpu_util_pct / 100.0
    if metrics.gpu_util_pct > _UTIL_WARN:
        util_penalty = (metrics.gpu_util_pct - _UTIL_WARN) / (100.0 - _UTIL_WARN)
        reason_codes.append(f"high_util:{metrics.gpu_util_pct:.0f}%")
    else:
        util_penalty = 0.0

    # Thermal penalty: linear from safe to critical
    if metrics.gpu_temp_c >= _TEMP_CRITICAL:
        thermal_penalty = 1.0
        reason_codes.append(f"critical_temp:{metrics.gpu_temp_c:.0f}C")
    elif metrics.gpu_temp_c > _TEMP_SAFE:
        thermal_penalty = (metrics.gpu_temp_c - _TEMP_SAFE) / (_TEMP_CRITICAL - _TEMP_SAFE)
        reason_codes.append(f"elevated_temp:{metrics.gpu_temp_c:.0f}C")
    else:
        thermal_penalty = 0.0

    # Throttle penalty: combine power and thermal throttle durations
    total_throttle_us = metrics.power_throttle_us + metrics.thermal_throttle_us
    if metrics.clock_throttle_reasons > 0 and total_throttle_us == 0:
        # Throttle reasons set but counters not populated — use bitmask presence
        total_throttle_us = _THROTTLE_MAX_US * 0.5
    throttle_penalty = min(1.0, total_throttle_us / _THROTTLE_MAX_US)
    if throttle_penalty > 0.1:
        reason_codes.append(f"throttling:reasons={metrics.clock_throttle_reasons:#04x}")

    # ECC penalty
    if metrics.ecc_dbe_count > 0:
        ecc_penalty = 1.0
        reason_codes.append(f"ecc_dbe:{metrics.ecc_dbe_count}")
    elif metrics.ecc_sbe_count >= 5:
        ecc_penalty = 0.5
        reason_codes.append(f"ecc_sbe:{metrics.ecc_sbe_count}")
    else:
        ecc_penalty = 0.0

    # Composite health penalty (weighted sum, normalized to 0–1)
    health_penalty = min(1.0, (
        0.30 * util_penalty
        + 0.30 * thermal_penalty
        + 0.30 * throttle_penalty
        + 0.10 * ecc_penalty
    ))

    # Schedulability: hard exclusions
    is_schedulable = (
        metrics.ecc_dbe_count == 0
        and metrics.gpu_temp_c < _TEMP_CRITICAL
        and metrics.xid_error_count == 0
    )
    if not is_schedulable and not reason_codes:
        reason_codes.append("xid_errors" if metrics.xid_error_count > 0 else "unschedulable")

    return GPUHealthScore(
        gpu_uuid=metrics.gpu_uuid,
        node_id=metrics.node_id,
        region=metrics.region,
        timestamp=metrics.timestamp,
        health_penalty=round(health_penalty, 4),
        utilization_penalty=round(util_penalty, 4),
        thermal_penalty=round(thermal_penalty, 4),
        throttle_penalty=round(throttle_penalty, 4),
        ecc_penalty=round(ecc_penalty, 4),
        is_schedulable=is_schedulable,
        reason_codes=reason_codes,
    )


def aggregate_region_health(
    scores: list[GPUHealthScore],
) -> float:
    """Aggregate per-GPU health scores to a region-level penalty (0.0–1.0).

    Returns the mean health_penalty across all schedulable GPUs.
    If all GPUs are unschedulable, returns 1.0 (worst possible).
    If no GPUs, returns 0.0 (unknown → assume healthy, log warning).
    """
    if not scores:
        return 0.0
    schedulable = [s for s in scores if s.is_schedulable]
    if not schedulable:
        return 1.0
    return sum(s.health_penalty for s in schedulable) / len(schedulable)


# ---------------------------------------------------------------------------
# Prometheus text-format parser
# ---------------------------------------------------------------------------

def parse_prometheus_text(
    text: str,
    region: str,
    timestamp: Optional[datetime] = None,
) -> list[GPUMetrics]:
    """Parse Prometheus text-format DCGM metrics into GPUMetrics objects.

    Supports the standard dcgm-exporter metric format where each line is:
      METRIC_NAME{labels} value [timestamp_ms]

    Labels recognized (case-insensitive):
      gpu, gpu_uuid, modelName (or model_name), Hostname (or hostname), node

    Args:
        text: Raw Prometheus text-format metric body
        region: Aurelius region to assign (not in DCGM labels)
        timestamp: Override collection timestamp; defaults to now-UTC if absent

    Returns:
        List of GPUMetrics (one per GPU present in the metric set)
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # {(node_id, gpu_index): {metric_name: value}}
    raw: dict[tuple[str, int], dict[str, float]] = {}
    # {(node_id, gpu_index): {label: value}}
    meta: dict[tuple[str, int], dict[str, str]] = {}

    label_re = re.compile(r'(\w+)="([^"]*)"')

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split: metric_name{labels} value [ts]
        m = re.match(r'^(\w+)\{([^}]*)\}\s+([\d.e+\-]+)', line)
        if not m:
            continue

        metric_name = m.group(1)
        if not metric_name.startswith("DCGM_"):
            continue

        labels = dict(label_re.findall(m.group(2)))
        value = float(m.group(3))

        node_id = (
            labels.get("Hostname")
            or labels.get("hostname")
            or labels.get("node")
            or "unknown-node"
        )
        gpu_index_str = labels.get("gpu", "0")
        try:
            gpu_index = int(gpu_index_str)
        except ValueError:
            gpu_index = 0

        key = (node_id, gpu_index)
        raw.setdefault(key, {})[metric_name] = value
        meta.setdefault(key, {}).update(labels)

    result: list[GPUMetrics] = []
    for (node_id, gpu_index), vals in raw.items():
        labels = meta.get((node_id, gpu_index), {})
        gpu_uuid = (
            labels.get("UUID")
            or labels.get("gpu_uuid")
            or f"GPU-{node_id}-{gpu_index}"
        )
        gpu_type = (
            labels.get("modelName")
            or labels.get("model_name")
            or labels.get("model")
            or "unknown"
        )

        mem_used = vals.get(_DCGM_MEM_USED, 0.0)
        mem_free = vals.get("DCGM_FI_DEV_FB_FREE", 0.0)
        mem_total = mem_used + mem_free if mem_free > 0 else vals.get("DCGM_FI_DEV_FB_TOTAL", mem_used)

        result.append(GPUMetrics(
            timestamp=timestamp,
            region=region,
            node_id=node_id,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_type=gpu_type.lower().replace(" ", "_"),
            gpu_util_pct=float(vals.get(_DCGM_GPU_UTIL, 0.0)),
            mem_used_mb=float(mem_used),
            mem_total_mb=float(mem_total),
            power_usage_w=float(vals.get(_DCGM_POWER, 0.0)),
            gpu_temp_c=float(vals.get(_DCGM_TEMP, 0.0)),
            ecc_sbe_count=int(vals.get(_DCGM_ECC_SBE, 0)),
            ecc_dbe_count=int(vals.get(_DCGM_ECC_DBE, 0)),
            xid_error_count=int(vals.get(_DCGM_XID, 0)),
            power_throttle_us=float(vals.get(_DCGM_POWER_THROTTLE, 0.0)),
            thermal_throttle_us=float(vals.get(_DCGM_THERMAL_THROTTLE, 0.0)),
            clock_throttle_reasons=int(vals.get(_DCGM_CLOCK_THROTTLE, 0)),
        ))

    return result


# ---------------------------------------------------------------------------
# DCGMProvider
# ---------------------------------------------------------------------------

@dataclass
class DCGMProvider:
    """Loads, stores, and queries GPU telemetry from DCGM/Prometheus.

    Supports fixture files (.prom), CSV exports, and optional live Prometheus.
    Region-level health aggregates feed directly into the optimizer objective.

    Usage::

        provider = DCGMProvider.from_prom_fixture("data/fixtures/dcgm_metrics.prom",
                                                   region="us-west")
        provider = DCGMProvider.from_csv("data/gpu_metrics.csv")
        health_data = provider.to_dict_lookup()   # plug into scheduler/objective
    """

    _records: list[GPUMetrics] = field(default_factory=list)
    _health_cache: dict[str, GPUHealthScore] = field(default_factory=dict)

    # Aggregated lookup: {region: sorted [(timestamp, avg_health_penalty)]}
    _lookup: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)
    _lookup_built: bool = False

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_prom_fixture(
        cls,
        path: str,
        region: str,
        timestamp: Optional[datetime] = None,
    ) -> "DCGMProvider":
        """Load GPU metrics from a Prometheus text-format fixture file.

        Args:
            path: Path to .prom fixture file
            region: Aurelius region to assign to all GPUs in this file
            timestamp: Override collection timestamp
        """
        with open(path) as f:
            text = f.read()
        metrics = parse_prometheus_text(text, region=region, timestamp=timestamp)
        obj = cls()
        obj._records = metrics
        logger.info(f"DCGMProvider: loaded {len(metrics)} GPU snapshots from {path} ({region})")
        return obj

    @classmethod
    def from_csv(cls, path: str) -> "DCGMProvider":
        """Load GPU metrics from a canonical CSV file.

        Required columns (see _CSV_COLS):
          timestamp, region, node_id, gpu_index, gpu_uuid, gpu_type,
          gpu_util_pct, mem_used_mb, mem_total_mb, power_usage_w,
          gpu_temp_c, ecc_sbe_count, ecc_dbe_count, xid_error_count,
          power_throttle_us, thermal_throttle_us, clock_throttle_reasons
        """
        records: list[GPUMetrics] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = _CSV_COLS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"DCGMProvider CSV missing columns: {sorted(missing)}")
            for row in reader:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                records.append(GPUMetrics(
                    timestamp=ts,
                    region=row["region"],
                    node_id=row["node_id"],
                    gpu_index=int(row["gpu_index"]),
                    gpu_uuid=row["gpu_uuid"],
                    gpu_type=row["gpu_type"],
                    gpu_util_pct=float(row["gpu_util_pct"]),
                    mem_used_mb=float(row["mem_used_mb"]),
                    mem_total_mb=float(row["mem_total_mb"]),
                    power_usage_w=float(row["power_usage_w"]),
                    gpu_temp_c=float(row["gpu_temp_c"]),
                    ecc_sbe_count=int(row["ecc_sbe_count"]),
                    ecc_dbe_count=int(row["ecc_dbe_count"]),
                    xid_error_count=int(row["xid_error_count"]),
                    power_throttle_us=float(row["power_throttle_us"]),
                    thermal_throttle_us=float(row["thermal_throttle_us"]),
                    clock_throttle_reasons=int(row["clock_throttle_reasons"]),
                ))
        obj = cls()
        obj._records = records
        logger.info(f"DCGMProvider: loaded {len(records)} GPU rows from CSV {path}")
        return obj

    @classmethod
    def from_dataframe(cls, df: "object") -> "DCGMProvider":  # df: pd.DataFrame
        """Construct from a pandas DataFrame with canonical GPU metric columns."""
        import pandas as _pd
        assert isinstance(df, _pd.DataFrame)
        missing = _CSV_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DCGMProvider DataFrame missing columns: {sorted(missing)}")
        records: list[GPUMetrics] = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append(GPUMetrics(
                timestamp=ts,
                region=str(row["region"]),
                node_id=str(row["node_id"]),
                gpu_index=int(row["gpu_index"]),
                gpu_uuid=str(row["gpu_uuid"]),
                gpu_type=str(row["gpu_type"]),
                gpu_util_pct=float(row["gpu_util_pct"]),
                mem_used_mb=float(row["mem_used_mb"]),
                mem_total_mb=float(row["mem_total_mb"]),
                power_usage_w=float(row["power_usage_w"]),
                gpu_temp_c=float(row["gpu_temp_c"]),
                ecc_sbe_count=int(row["ecc_sbe_count"]),
                ecc_dbe_count=int(row["ecc_dbe_count"]),
                xid_error_count=int(row["xid_error_count"]),
                power_throttle_us=float(row["power_throttle_us"]),
                thermal_throttle_us=float(row["thermal_throttle_us"]),
                clock_throttle_reasons=int(row["clock_throttle_reasons"]),
            ))
        obj = cls()
        obj._records = records
        return obj

    # ------------------------------------------------------------------ #
    # Live Prometheus query (optional — requires PROMETHEUS_URL env var)   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_prometheus_live(
        cls,
        region: str,
        prometheus_url: Optional[str] = None,
        dcgm_exporter_url: Optional[str] = None,
    ) -> "DCGMProvider":
        """Query live Prometheus/dcgm-exporter for current GPU metrics.

        Requires either PROMETHEUS_URL or DCGM_EXPORTER_URL env var, or explicit args.
        Falls back to an empty provider if neither is set (logs a warning, no crash).

        This method is intended for production/shadow-mode use. For tests, use
        from_prom_fixture() or from_csv().
        """
        prom_url = prometheus_url or os.environ.get("PROMETHEUS_URL", "")
        exporter_url = dcgm_exporter_url or os.environ.get("DCGM_EXPORTER_URL", "")

        if not prom_url and not exporter_url:
            logger.warning(
                "DCGMProvider.from_prometheus_live: neither PROMETHEUS_URL nor "
                "DCGM_EXPORTER_URL is set — returning empty provider. "
                "Set one of these env vars for live GPU telemetry."
            )
            return cls()

        try:
            import requests
        except ImportError:
            logger.warning("DCGMProvider: requests not installed; cannot query Prometheus")
            return cls()

        bearer = os.environ.get("PROMETHEUS_BEARER_TOKEN", "")
        user = os.environ.get("PROMETHEUS_USERNAME", "")
        pwd = os.environ.get("PROMETHEUS_PASSWORD", "")
        tls_verify = os.environ.get("PROMETHEUS_TLS_VERIFY", "true").lower() != "false"

        auth = None
        headers: dict[str, str] = {}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        elif user and pwd:
            auth = (user, pwd)

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        if exporter_url:
            # Direct dcgm-exporter /metrics scrape
            try:
                resp = requests.get(
                    exporter_url.rstrip("/") + "/metrics",
                    headers=headers, auth=auth, verify=tls_verify, timeout=10,
                )
                resp.raise_for_status()
                metrics = parse_prometheus_text(resp.text, region=region, timestamp=now)
                obj = cls()
                obj._records = metrics
                logger.info(
                    f"DCGMProvider: live scrape from dcgm-exporter: {len(metrics)} GPUs"
                )
                return obj
            except Exception as exc:
                logger.warning(f"DCGMProvider: dcgm-exporter scrape failed: {exc}")
                return cls()

        # Prometheus instant-query for DCGM metrics
        dcgm_metrics = [
            _DCGM_GPU_UTIL, _DCGM_MEM_USED, "DCGM_FI_DEV_FB_FREE",
            _DCGM_POWER, _DCGM_TEMP, _DCGM_ECC_SBE, _DCGM_ECC_DBE,
            _DCGM_XID, _DCGM_POWER_THROTTLE, _DCGM_THERMAL_THROTTLE,
            _DCGM_CLOCK_THROTTLE,
        ]
        # Build a synthetic Prometheus text body from Prometheus API responses
        lines: list[str] = []
        for metric in dcgm_metrics:
            try:
                resp = requests.get(
                    prom_url.rstrip("/") + "/api/v1/query",
                    params={"query": metric},
                    headers=headers, auth=auth, verify=tls_verify, timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                for result in data.get("data", {}).get("result", []):
                    lbl = ",".join(f'{k}="{v}"' for k, v in result["metric"].items())
                    val = result["value"][1]
                    lines.append(f'{metric}{{{lbl}}} {val}')
            except Exception as exc:
                logger.debug(f"DCGMProvider: Prometheus query for {metric} failed: {exc}")

        metrics_list = parse_prometheus_text("\n".join(lines), region=region, timestamp=now)
        obj = cls()
        obj._records = metrics_list
        logger.info(f"DCGMProvider: live Prometheus query: {len(metrics_list)} GPUs")
        return obj

    # ------------------------------------------------------------------ #
    # Querying                                                             #
    # ------------------------------------------------------------------ #

    def _build_lookup(self) -> None:
        """Build aggregated region→time lookup (lazy, called once)."""
        if self._lookup_built:
            return

        # Score all GPUs
        per_region_time: dict[str, dict[datetime, list[GPUHealthScore]]] = {}
        for m in self._records:
            ts_hour = m.timestamp.replace(minute=0, second=0, microsecond=0)
            score = score_gpu_health(m)
            per_region_time.setdefault(m.region, {}).setdefault(ts_hour, []).append(score)

        self._lookup = {}
        for region, time_map in per_region_time.items():
            pairs: list[tuple[datetime, float]] = []
            for ts_hour, scores in sorted(time_map.items()):
                agg = aggregate_region_health(scores)
                pairs.append((ts_hour, agg))
            self._lookup[region] = pairs  # sorted ascending by construction

        self._lookup_built = True

    def get_health_penalty(self, region: str, timestamp: datetime) -> float:
        """Return last known aggregate GPU health penalty at or before timestamp.

        Returns 0.0 (healthy assumption) if no data is available for the region.
        This is leakage-safe: only states observed before `timestamp` are used.
        """
        self._build_lookup()
        pairs = self._lookup.get(region)
        if not pairs:
            return 0.0
        times = [p[0] for p in pairs]
        idx = bisect.bisect_right(times, timestamp) - 1
        if idx < 0:
            return 0.0
        return pairs[idx][1]

    def to_dict_lookup(self) -> dict[str, dict[datetime, float]]:
        """Return {region: {timestamp: avg_health_penalty}} lookup.

        This format mirrors price_data / queue_data and plugs directly into
        the objective function and scheduler.
        """
        self._build_lookup()
        result: dict[str, dict[datetime, float]] = {}
        for region, pairs in self._lookup.items():
            result[region] = {ts: penalty for ts, penalty in pairs}
        return result

    def get_gpu_scores(
        self,
        region: str,
        timestamp: datetime,
    ) -> list[GPUHealthScore]:
        """Return individual GPU health scores for a specific region/time.

        Useful for node-level placement decisions when the scheduler adapter
        supports per-GPU control. Returns the snapshot closest to but not
        exceeding `timestamp`.
        """
        ts_floor = timestamp.replace(minute=0, second=0, microsecond=0)
        # Find the latest snapshot at or before ts_floor
        snaps: dict[datetime, list[GPUMetrics]] = {}
        for m in self._records:
            if m.region != region:
                continue
            mts = m.timestamp.replace(minute=0, second=0, microsecond=0)
            if mts <= ts_floor:
                snaps.setdefault(mts, []).append(m)
        if not snaps:
            return []
        latest_ts = max(snaps.keys())
        return [score_gpu_health(m) for m in snaps[latest_ts]]

    @property
    def regions(self) -> list[str]:
        return sorted({m.region for m in self._records})

    @property
    def record_count(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------ #
    # Fixture generation                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def generate_fixture(
        cls,
        regions: list[str],
        n_nodes_per_region: int = 4,
        n_gpus_per_node: int = 8,
        hours: int = 24,
        start_dt: Optional[datetime] = None,
        seed: int = 42,
        gpu_type: str = "a100",
    ) -> "DCGMProvider":
        """Generate a realistic synthetic GPU telemetry fixture.

        Patterns modelled:
        - Business-hours load cycles (higher utilization 12-20 UTC)
        - Occasional thermal spikes (one per region per day)
        - Small background ECC SBE noise
        - One node per region has slightly higher error rate (hardware variation)

        NOTE: This fixture is SYNTHETIC and must NOT be used for savings claims.
        Use only for tests and integration demos.
        """
        rng = random.Random(seed)
        if start_dt is None:
            start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        records: list[GPUMetrics] = []
        for region_idx, region in enumerate(regions):
            for node_idx in range(n_nodes_per_region):
                node_id = f"gpu-node-{region_idx:02d}-{node_idx:02d}"
                # One "hot" node per region (slightly more errors/throttle)
                is_hot_node = (node_idx == 0)
                for gpu_idx in range(n_gpus_per_node):
                    gpu_uuid = f"GPU-{region}-{node_idx:02d}-{gpu_idx:02d}-{seed}"
                    for h in range(hours):
                        ts = start_dt + timedelta(hours=h)
                        hour_of_day = ts.hour
                        # Business-hours load spike
                        base_util = 40.0 + 40.0 * max(0, (hour_of_day - 8) / 10.0) if 8 <= hour_of_day < 18 else 15.0
                        util = min(100.0, base_util + rng.gauss(0, 8))
                        # Temperature correlated with util
                        base_temp = 45.0 + (util / 100.0) * 35.0
                        temp = base_temp + rng.gauss(0, 3.0)
                        if is_hot_node:
                            temp += 8.0  # hot node runs warmer
                        # Power correlated with util
                        max_power = 400.0 if gpu_type == "a100" else 700.0  # H100 = 700W
                        power = max_power * 0.15 + (util / 100.0) * max_power * 0.80 + rng.gauss(0, 5)
                        # ECC: small background SBE noise, rare DBE
                        ecc_sbe = rng.choices([0, 1, 2], weights=[0.97, 0.02, 0.01])[0]
                        ecc_dbe = 1 if (is_hot_node and rng.random() < 0.002) else 0
                        xid = 1 if (ecc_dbe > 0) else 0
                        # Throttle: occurs when temp high
                        throttle_us = max(0, rng.gauss(0, 100)) if temp > 75 else 0.0
                        if is_hot_node and temp > 80:
                            throttle_us = max(throttle_us, rng.gauss(50000, 20000))
                        mem_total = 80_000.0 if gpu_type == "a100" else 95_000.0
                        mem_used = mem_total * (util / 100.0) * 0.9 + rng.gauss(0, 500)
                        records.append(GPUMetrics(
                            timestamp=ts,
                            region=region,
                            node_id=node_id,
                            gpu_index=gpu_idx,
                            gpu_uuid=gpu_uuid,
                            gpu_type=gpu_type,
                            gpu_util_pct=max(0.0, min(100.0, util)),
                            mem_used_mb=max(0.0, min(mem_total, mem_used)),
                            mem_total_mb=mem_total,
                            power_usage_w=max(0.0, min(max_power * 1.05, power)),
                            gpu_temp_c=max(20.0, min(105.0, temp)),
                            ecc_sbe_count=ecc_sbe,
                            ecc_dbe_count=ecc_dbe,
                            xid_error_count=xid,
                            power_throttle_us=max(0.0, throttle_us),
                            thermal_throttle_us=0.0,
                            clock_throttle_reasons=(1 if throttle_us > 0 else 0),
                        ))
        obj = cls()
        obj._records = records
        logger.info(
            f"DCGMProvider: generated fixture with {len(records)} GPU snapshots "
            f"({len(regions)} regions, {n_nodes_per_region} nodes, "
            f"{n_gpus_per_node} GPUs/node, {hours}h)"
        )
        return obj

    def save_csv(self, path: str) -> None:
        """Save GPU metrics to a canonical CSV file."""
        import csv as _csv
        fieldnames = sorted(_CSV_COLS)
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self._records:
                writer.writerow({
                    "timestamp": m.timestamp.isoformat(),
                    "region": m.region,
                    "node_id": m.node_id,
                    "gpu_index": m.gpu_index,
                    "gpu_uuid": m.gpu_uuid,
                    "gpu_type": m.gpu_type,
                    "gpu_util_pct": m.gpu_util_pct,
                    "mem_used_mb": m.mem_used_mb,
                    "mem_total_mb": m.mem_total_mb,
                    "power_usage_w": m.power_usage_w,
                    "gpu_temp_c": m.gpu_temp_c,
                    "ecc_sbe_count": m.ecc_sbe_count,
                    "ecc_dbe_count": m.ecc_dbe_count,
                    "xid_error_count": m.xid_error_count,
                    "power_throttle_us": m.power_throttle_us,
                    "thermal_throttle_us": m.thermal_throttle_us,
                    "clock_throttle_reasons": m.clock_throttle_reasons,
                })
        logger.info(f"DCGMProvider: saved {len(self._records)} rows to {path}")
