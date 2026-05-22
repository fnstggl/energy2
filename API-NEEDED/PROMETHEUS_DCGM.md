# API-NEEDED: NVIDIA DCGM / Prometheus GPU Telemetry

## Provider
NVIDIA DCGM (Data Center GPU Manager) via dcgm-exporter + Prometheus

## Why needed
GPU telemetry is required for Roadmap Phase 4 (GPU Telemetry & DCGM):
- Real-time GPU utilization (busy vs idle capacity)
- GPU memory utilization (running hot vs headroom)
- GPU power draw (actual energy consumption, more accurate than estimated kW)
- GPU temperature (thermal throttling detection)
- ECC errors (GPU health / reliability risk)
- NVLink / PCIe throughput (migration overhead estimation)

This data feeds the optimizer's cost model. Without it, Aurelius estimates
GPU power consumption from job profile defaults (less accurate).

## Env vars
```
PROMETHEUS_URL=http://localhost:9090
DCGM_EXPORTER_URL=http://localhost:9400/metrics
```

## .env.example entry
```
# Prometheus endpoint for DCGM GPU metrics
# Required only for GPU telemetry integration (Phase 4)
# Default assumes local Prometheus instance
PROMETHEUS_URL=http://localhost:9090

# Optional: direct dcgm-exporter scrape endpoint
DCGM_EXPORTER_URL=http://localhost:9400/metrics
```

## No external API key required
DCGM + dcgm-exporter run on-premises alongside GPUs.
The customer operates Prometheus themselves.
Aurelius connects as a Prometheus client.

## Setup instructions (customer side)
1. Install NVIDIA DCGM: https://docs.nvidia.com/datacenter/dcgm/latest/
2. Install dcgm-exporter: https://github.com/NVIDIA/dcgm-exporter
3. Run dcgm-exporter (exposes metrics at :9400/metrics by default)
4. Add a Prometheus scrape job targeting dcgm-exporter
5. Set PROMETHEUS_URL in Aurelius .env

## Key metrics Aurelius needs
```
DCGM_FI_DEV_GPU_UTIL        GPU utilization (%)
DCGM_FI_DEV_MEM_COPY_UTIL   Memory copy engine utilization (%)
DCGM_FI_DEV_FB_USED         GPU framebuffer memory used (MB)
DCGM_FI_DEV_FB_FREE         GPU framebuffer memory free (MB)
DCGM_FI_DEV_GPU_TEMP        GPU temperature (°C)
DCGM_FI_DEV_POWER_USAGE     GPU power consumption (W)
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION  Cumulative energy (mJ)
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL Double-bit ECC errors (GPU health)
DCGM_FI_PROF_NVLINK_TX_BYTES NVLink TX bandwidth (migration cost signal)
DCGM_FI_PROF_PCIE_TX_BYTES  PCIe TX bandwidth
```

## Planned implementation
- `aurelius/ingestion/gpu_telemetry.py` (not yet implemented)
- `aurelius/monitoring/dcgm_scraper.py` (not yet implemented)
- Feeds into Phase 6 multi-signal cost model

## Is live integration required for CI?
No. When implemented, live DCGM tests will be gated behind
PROMETHEUS_URL being set. All unit tests will use mocked metrics.

## Alternative (lightweight)
For clusters without DCGM, nvidia-smi can provide basic GPU metrics.
However, DCGM is recommended for production accuracy.
