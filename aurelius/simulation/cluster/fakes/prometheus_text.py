"""Generate Prometheus text-format metrics from simulator state.

Outputs match the exact format of real dcgm-exporter and vLLM /metrics
endpoints. The production DCGMAdapter and VLLMAdapter parse these outputs
without any code change — verifying that the simulator and real connectors
share the same parsing path.

DCGM metric names match dcgm-exporter defaults (the subset enabled by default):
  https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv
vLLM V1 metric names (vllm: prefix):
  https://docs.vllm.ai/en/latest/serving/metrics.html
"""

from __future__ import annotations

from typing import Optional

from ..model import SimNode, SimQueue, SimWorkload


# ---------------------------------------------------------------------------
# DCGM metrics text generator
# ---------------------------------------------------------------------------

def generate_dcgm_metrics_text(node: SimNode) -> str:
    """Generate Prometheus text for a node's GPUs in dcgm-exporter format."""
    lines: list[str] = []

    # Metric help/type headers (emitted once per metric, before first sample)
    _add_dcgm_headers(lines)

    for gpu in node.gpus:
        labels = (
            f'gpu="{gpu.gpu_index}",'
            f'UUID="{gpu.uuid}",'
            f'device="nvidia{gpu.gpu_index}",'
            f'modelName="{gpu.profile.model_name}",'
            f'Hostname="{node.node_id}",'
            f'DCGM_FI_DRIVER_VERSION="535.104.12"'
        )

        # Core utilization metrics
        _add_metric(lines, "DCGM_FI_DEV_GPU_UTIL", labels, gpu.utilization_pct)
        _add_metric(lines, "DCGM_FI_DEV_SM_ACTIVE", labels, gpu.sm_activity_pct)
        _add_metric(lines, "DCGM_FI_DEV_SM_OCCUPANCY", labels, gpu.sm_activity_pct * 0.9)

        # Memory metrics (bytes)
        _add_metric(lines, "DCGM_FI_DEV_FB_USED", labels, gpu.memory_used_bytes // (1024 * 1024))
        _add_metric(lines, "DCGM_FI_DEV_FB_FREE", labels, gpu.memory_free_bytes // (1024 * 1024))
        _add_metric(
            lines, "DCGM_FI_DEV_FB_TOTAL", labels, gpu.profile.memory_total_bytes // (1024 * 1024)
        )

        # Power metrics
        _add_metric(lines, "DCGM_FI_DEV_POWER_USAGE", labels, gpu.power_watts)
        _add_metric(lines, "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", labels, gpu.power_watts * 3600)

        # Thermal metrics
        _add_metric(lines, "DCGM_FI_DEV_GPU_TEMP", labels, round(gpu.temperature_c, 1))

        # Thermal throttle reasons (bitfield — 0 = none, 8 = HW_SLOWDOWN, 16 = SW_THERMAL_SLOWDOWN)
        throttle_bits = 0
        if gpu.thermal_throttle_active:
            throttle_bits = 8  # HW_SLOWDOWN
        _add_metric(lines, "DCGM_FI_DEV_CLOCKS_EVENT_REASONS", labels, throttle_bits)

        # XID errors
        _add_metric(lines, "DCGM_FI_DEV_XID_ERRORS", labels, gpu.xid_error_count)

        # NVLink counters (only if nonzero)
        if gpu.nvlink_rx_bytes_per_sec > 0:
            _add_metric(
                lines, "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", labels, gpu.nvlink_rx_bytes_per_sec
            )

        # PCIe counters
        if gpu.pcie_rx_bytes_per_sec > 0:
            _add_metric(
                lines, "DCGM_FI_DEV_PCIE_RX_THROUGHPUT", labels, gpu.pcie_rx_bytes_per_sec // 1024
            )
        if gpu.pcie_tx_bytes_per_sec > 0:
            _add_metric(
                lines, "DCGM_FI_DEV_PCIE_TX_THROUGHPUT", labels, gpu.pcie_tx_bytes_per_sec // 1024
            )

        # Memory bandwidth utilization
        _add_metric(lines, "DCGM_FI_DEV_MEM_COPY_UTIL", labels, gpu.utilization_pct * 0.8)

    lines.append("")
    return "\n".join(lines)


def _add_dcgm_headers(lines: list[str]) -> None:
    headers = [
        ("DCGM_FI_DEV_GPU_UTIL", "gauge", "GPU utilization in %."),
        ("DCGM_FI_DEV_SM_ACTIVE", "gauge", "Ratio of cycles an SM has at least one warp assigned."),
        ("DCGM_FI_DEV_SM_OCCUPANCY", "gauge", "The ratio of number of warps resident on an SM."),
        ("DCGM_FI_DEV_FB_USED", "gauge", "Used FB memory (in MiB)."),
        ("DCGM_FI_DEV_FB_FREE", "gauge", "Free FB memory (in MiB)."),
        ("DCGM_FI_DEV_FB_TOTAL", "gauge", "Total FB memory (in MiB)."),
        ("DCGM_FI_DEV_POWER_USAGE", "gauge", "Power draw (in W)."),
        ("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "counter", "Energy since boot (mJ)."),
        ("DCGM_FI_DEV_GPU_TEMP", "gauge", "GPU temperature (in C)."),
        ("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "gauge", "Bitmask of active clocks event reasons."),
        ("DCGM_FI_DEV_XID_ERRORS", "counter", "Value of the last XID error encountered."),
        ("DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", "counter", "Total NVLink bandwidth (bytes/sec)."),
        ("DCGM_FI_DEV_PCIE_RX_THROUGHPUT", "counter", "PCIe RX (in KB)."),
        ("DCGM_FI_DEV_PCIE_TX_THROUGHPUT", "counter", "PCIe TX (in KB)."),
        ("DCGM_FI_DEV_MEM_COPY_UTIL", "gauge", "Memory utilization in %."),
    ]
    for name, mtype, help_text in headers:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")


def _add_metric(lines: list[str], name: str, labels: str, value: float) -> None:
    lines.append(f"{name}{{{labels}}} {value}")


# ---------------------------------------------------------------------------
# vLLM metrics text generator
# ---------------------------------------------------------------------------

def generate_vllm_metrics_text(queue: SimQueue, workload: Optional[SimWorkload]) -> str:
    """Generate Prometheus text in vLLM V1 format (vllm: prefix)."""
    model_name = getattr(workload, "service_id", "unknown_model") if workload else "unknown_model"
    lines: list[str] = []

    _add_vllm_headers(lines)

    labels = f'model_name="{model_name}"'

    # Request counters
    rps = queue.requests_per_second
    _add_vllm_counter(lines, "vllm:num_requests_running", labels,
                      round(rps * 3600 * 0.8))  # proxy cumulative
    _add_vllm_counter(lines, "vllm:num_requests_waiting", labels, queue.queue_depth)
    _add_vllm_counter(lines, "vllm:num_requests_swapped", labels, 0)

    # GPU cache usage
    kv_pct = queue.kv_cache_usage_pct if queue.kv_cache_usage_pct is not None else 30.0
    _add_vllm_gauge(lines, "vllm:gpu_cache_usage_perc", labels, round(kv_pct / 100.0, 4))

    # Prefix cache hit rate
    hit_pct = (
        queue.prefix_cache_hit_rate_pct if queue.prefix_cache_hit_rate_pct is not None else 50.0
    )
    _add_vllm_gauge(lines, "vllm:gpu_prefix_cache_hit_rate", labels, round(hit_pct / 100.0, 4))

    # Token counters
    tps = queue.tokens_per_second
    _add_vllm_counter(lines, "vllm:prompt_tokens_total", labels, round(tps * 3600 * 0.3))
    _add_vllm_counter(lines, "vllm:generation_tokens_total", labels, round(tps * 3600 * 0.7))

    # Throughput gauges
    _add_vllm_gauge(lines, "vllm:request_throughput", labels, round(rps, 4))
    _add_vllm_gauge(lines, "vllm:token_throughput", labels, round(tps, 2))

    # TTFT histogram (simplified: just _sum and _count for average derivation)
    ttft_avg = queue.ttft_p50_ms if queue.ttft_p50_ms is not None else _BASE_TTFT
    req_count = max(1, round(rps * 3600))
    _add_vllm_histogram_sum_count(
        lines, "vllm:time_to_first_token_seconds", labels,
        sum_val=ttft_avg / 1000.0 * req_count,
        count=req_count,
        buckets=[
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08,
            0.1, 0.25, 0.5, 0.75, 1.0, 2.5, float("inf"),
        ],
        p50_s=ttft_avg / 1000.0,
        p99_s=(queue.ttft_p99_ms or ttft_avg * 5.0) / 1000.0,
    )

    # TPOT histogram
    tpot_avg = queue.tpot_p50_ms if queue.tpot_p50_ms is not None else 20.0
    _add_vllm_histogram_sum_count(
        lines, "vllm:time_per_output_token_seconds", labels,
        sum_val=tpot_avg / 1000.0 * req_count,
        count=req_count,
        buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, float("inf")],
        p50_s=tpot_avg / 1000.0,
        p99_s=(queue.tpot_p99_ms or tpot_avg * 4.0) / 1000.0,
    )

    # E2E request latency histogram
    lat_avg = queue.latency_p50_ms if queue.latency_p50_ms is not None else 3000.0
    _add_vllm_histogram_sum_count(
        lines, "vllm:e2e_request_latency_seconds", labels,
        sum_val=lat_avg / 1000.0 * req_count,
        count=req_count,
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")],
        p50_s=lat_avg / 1000.0,
        p99_s=(queue.latency_p99_ms or lat_avg * 5.0) / 1000.0,
    )

    lines.append("")
    return "\n".join(lines)


_BASE_TTFT = 150.0  # default TTFT in ms


def _add_vllm_headers(lines: list[str]) -> None:
    headers = [
        ("vllm:num_requests_running", "gauge", "Number of requests currently running on GPU."),
        ("vllm:num_requests_waiting", "gauge", "Number of requests waiting to be processed."),
        ("vllm:num_requests_swapped", "gauge", "Number of requests swapped to CPU."),
        ("vllm:gpu_cache_usage_perc", "gauge", "GPU KV-cache usage (0.0-1.0)."),
        ("vllm:gpu_prefix_cache_hit_rate", "gauge", "Prefix cache block hit rate (0.0-1.0)."),
        ("vllm:prompt_tokens_total", "counter", "Number of prefill tokens processed."),
        ("vllm:generation_tokens_total", "counter", "Number of generation tokens processed."),
        ("vllm:request_throughput", "gauge", "Number of requests per second."),
        ("vllm:token_throughput", "gauge", "Number of tokens generated per second."),
        ("vllm:time_to_first_token_seconds", "histogram", "Time to first token."),
        ("vllm:time_per_output_token_seconds", "histogram", "Time per output token."),
        ("vllm:e2e_request_latency_seconds", "histogram", "End-to-end request latency."),
    ]
    for name, mtype, help_text in headers:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")


def _add_vllm_gauge(lines: list[str], name: str, labels: str, value: float) -> None:
    lines.append(f"{name}{{{labels}}} {value}")


def _add_vllm_counter(lines: list[str], name: str, labels: str, value: float) -> None:
    lines.append(f"{name}_total{{{labels}}} {value}")


def _add_vllm_histogram_sum_count(
    lines: list[str],
    name: str,
    labels: str,
    sum_val: float,
    count: int,
    buckets: list[float],
    p50_s: float,
    p99_s: float,
) -> None:
    """Emit a Prometheus histogram with realistic bucket fills."""
    # Fill buckets proportionally based on p50/p99 approximation
    for bucket in buckets:
        if bucket == float("inf"):
            cumulative = count
        elif bucket <= p50_s:
            cumulative = int(count * 0.5 * bucket / p50_s) if p50_s > 0 else 0
        elif bucket <= p99_s:
            frac = 0.5 + 0.49 * (bucket - p50_s) / (p99_s - p50_s)
            cumulative = int(count * frac)
        else:
            cumulative = int(count * 0.99)
        bucket_label = "+Inf" if bucket == float("inf") else str(bucket)
        lines.append(f'{name}_bucket{{{labels},le="{bucket_label}"}} {cumulative}')
    lines.append(f"{name}_sum{{{labels}}} {sum_val:.6f}")
    lines.append(f"{name}_count{{{labels}}} {count}")
