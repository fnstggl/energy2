#!/usr/bin/env python3
"""Targeted HF discovery audit for economic / operational signal pairs.

Mission: find public datasets that can calibrate Aurelius economic scoring
by combining (A) operational AI-infra behavior with (B) economic / cost /
energy / carbon signals — prioritising paired (A+B) datasets, falling back
to B-only that can join existing Aurelius operational traces.

Discovery NEVER downloads data — only metadata via the public HF API
(`https://huggingface.co/api/datasets`). `HF_TOKEN` is read from the
environment and sent via `Authorization: Bearer ...`; the token is never
written to any committed artefact.

Outputs (audit-only, no raw data):
    data/external/hf_discovery/economic_signal_discovery_audit.json
    docs/HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md (regenerated separately)

Run:
    HF_TOKEN=... python3 scripts/discover_hf_economic_signals.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery",
    "economic_signal_discovery_audit.json",
)

HF_API_BASE = "https://huggingface.co/api"
USER_AGENT = "aurelius-economic-signal-discovery/0.1"

logger = logging.getLogger("hf_economic_discovery")


# ---------------------------------------------------------------------------
# Search terms — verbatim from the mission spec.
# ---------------------------------------------------------------------------

PRIMARY_SEARCH_TERMS = [
    # economic / pricing
    "gpu cost", "gpu pricing", "gpu hourly price", "cloud gpu pricing",
    "spot gpu pricing", "gpu spot price", "reserved gpu price",
    "cloud cost telemetry", "cloud billing", "cloud invoice", "chargeback",
    "cost per token", "cost per request", "llm cost telemetry",
    "serving cost", "inference cost",
    # energy / power / carbon
    "gpu energy", "energy per request", "kwh inference", "power draw gpu",
    "gpu power telemetry", "ipmi power gpu", "dcgm power",
    "carbon intensity datacenter", "datacenter energy",
    "electricity price workload", "energy price trace",
    # cloud workload + cluster cost
    "cloud workload cost", "cluster cost", "resource cost",
    "cost aware scheduling", "price aware scheduling",
    "spot instance trace", "preemptible instance trace",
    "cloud spot trace", "aws spot trace", "azure spot trace",
    "gcp preemptible trace", "region cost latency",
    "multi region routing cost", "gpu utilization cost",
    "gpu usage billing", "llm inference billing",
]

COMBINATION_SEARCH_TERMS = [
    "ttft cost", "tpot cost", "latency cost", "queue cost",
    "gpu utilization billing", "gpu power cost", "cache hit cost",
    "kv cache cost", "prefix cache cost", "model loading cost",
    "cold start cost", "autoscaling cost", "replica cost", "migration cost",
    "cloud trace cost", "datacenter trace cost",
    "energy aware scheduling gpu", "carbon aware scheduling gpu",
]

# Single-word probes — HF text search is AND-only, so multi-word queries
# return very few hits. These broaden coverage; we still classify by the
# README/tag/feature matches afterwards.
SINGLE_WORD_PROBES = [
    "gpu-price", "gpu-prices", "carbon-intensity", "electricity-price",
    "energy-consumption", "spot-price", "kwh", "billing", "invoice",
    "chargeback", "cost-per-token", "aws pricing", "ec2 pricing",
    "caiso", "ercot", "pjm", "nordpool", "electricitymaps",
    "codecarbon", "mlperf inference", "nvidia gpu pricing",
    "lambda labs gpu", "runpod", "vast.ai", "gpu rental price",
    "energy aware inference", "carbon aware inference",
    "datacenter pue", "power usage effectiveness",
]

# Known-high-priority seeds (in case keyword searches miss them).
SEED_DATASET_IDS = [
    "afhubbard/gpu-prices",
    "labofsahil/aws-pricing-dataset",
    "ejhusom/llm-inference-energy-consumption",
    "Qinghao/AcmeTrace",
    "optimum-benchmark/llm-perf-leaderboard",
    "memoriant/dgx-spark-kv-cache-benchmark",
    "huggingface/electricity-production",
    "EDS-lab/electricity-demand",
    "tulipa762/electricity_load_diagrams",
    "grm/caiso-curtailment",
    "anonymoususermargin/ercot-rtcb-v1",
]

# Manual overrides for seeds where the HF card is too thin for the text
# detector to find signals that ARE present in the underlying schema
# (verified via the existing HF_DATASET_REGISTRY ingest summaries). The
# override only ADDS signals; it never removes detected ones, and it is
# applied AFTER inspection so the per-dataset evidence remains honest.
SEED_SIGNAL_OVERRIDES = {
    "ejhusom/llm-inference-energy-consumption": {
        "operational": ["ttft", "tpot_itl", "e2e_latency", "throughput"],
        "economic": ["kwh", "energy_per_request"],
        "field_quality": "measured",
        "evidence": (
            "Per-request Ollama timing (total/load/prompt/response_duration_ns) "
            "+ per-request CodeCarbon kWh (CPU/GPU/total). Verified in "
            "data/external/hf/ejhusom__llm-inference-energy-consumption/.../summary.json."
        ),
    },
    "optimum-benchmark/llm-perf-leaderboard": {
        "operational": ["ttft", "tpot_itl", "throughput", "gpu_memory"],
        "economic": ["kwh", "energy_per_request", "gpu_type"],
        "field_quality": "measured",
        "evidence": (
            "Measured prefill (TTFT) + decode (TPOT) p50/p90/p95/p99, "
            "per-request CodeCarbon energy (kWh CPU/RAM/GPU/total), peak VRAM. "
            "Verified in data/external/hf/optimum-benchmark__llm-perf-leaderboard/.../summary.json."
        ),
    },
    "Qinghao/AcmeTrace": {
        "operational": ["queue_wait", "gpu_utilization", "power_draw",
                        "placement_decisions"],
        "economic": [],  # No pricing — cluster trace only.
        "field_quality": "measured",
        "evidence": (
            "Real Shanghai AI Lab Kalos+Seren cluster scheduler trace; "
            "DCGM 15s GPU utilization + IPMI per-host GPU power telemetry. "
            "Already ingested as 4 configs (kalos_jobs / seren_jobs_head / "
            "kalos_gpu_util_head / seren_ipmi_gpu_power_head)."
        ),
    },
    "afhubbard/gpu-prices": {
        "operational": [],  # pricing snapshots, not telemetry
        "economic": ["gpu_hour_price", "gpu_type", "gpu_count", "region_zone",
                     "spot_price", "reserved_ondemand_price"],
        "field_quality": "measured",
        "evidence": (
            "Cross-cloud GPU rental pricing (12+ providers), twice-daily "
            "snapshots in Hive-partitioned Parquet. Schema: timestamp, "
            "provider, instance_type, gpu_type, gpu_count, gpu_memory_gb, "
            "region, price_per_hour, is_spot, availability_zone. CC-BY-4.0."
        ),
    },
    "labofsahil/aws-pricing-dataset": {
        "operational": [],
        "economic": ["gpu_hour_price", "cloud_cost", "region_zone"],
        "field_quality": "measured",
        "evidence": (
            "Official AWS Pricing API export, weekly auto-update. Contains "
            "AmazonEC2.csv with all EC2 instance pricing (incl. GPU SKUs like "
            "p3, p4d, p5, g5). MIT-licensed."
        ),
    },
    "memoriant/dgx-spark-kv-cache-benchmark": {
        "operational": ["gpu_memory", "throughput", "cache_reuse"],
        "economic": ["gpu_type"],
        "field_quality": "measured",
        "evidence": (
            "DGX Spark GB10 KV-cache quantisation benchmark; kv_buffer_mib + "
            "gpu_mem_mib + prompt_tps + gen_tps per (cache_type, context_tokens). "
            "Apache-2.0."
        ),
    },
}


# ---------------------------------------------------------------------------
# Signal taxonomy — what we look for in feature names / tags / descriptions.
# Order: most-specific first so classification is stable.
# ---------------------------------------------------------------------------

# Compute/AI/datacenter context tokens — required for many economic
# signals to count, so we don't false-positive on banking invoice OCR
# or telecom utility billing.
COMPUTE_CONTEXT_TOKENS = re.compile(
    r"\b(gpu|cpu|tpu|cuda|vllm|sglang|triton|inference|llm|model|"
    r"datacent(?:er|re)|cluster|kubernetes|k8s|cloud|aws|azure|gcp|"
    r"compute|hpc|nvidia|amd|h100|a100|h200|b200|a10g?|t4|v100|p100|"
    r"workload|serving|throughput|tokens?|prompt|prefill|decode|"
    r"ml-?perf|deep[ -]?learning|training|benchmark|dcgm|ipmi|nvml|"
    r"codecarbon|rapl|electricitymaps|caiso|ercot|pjm|nordpool|"
    r"wattime|pue)\b",
    re.IGNORECASE,
)


OPERATIONAL_SIGNAL_PATTERNS = {
    "queue_depth": [r"\bqueue[_ ]?depth\b", r"\bnum[_ ]?waiting\b",
                    r"\bwaiting[_ ]?requests\b"],
    "queue_wait": [r"\bqueue[_ ]?wait\b", r"\bwait[_ ]?time\b",
                   r"\bsubmit[_ ]?time\b"],
    "ttft": [r"\bttft\b", r"\bfirst[_ ]?token\b",
             r"\btime[_ ]?to[_ ]?first[_ ]?token\b"],
    "tpot_itl": [r"\btpot\b", r"\bitl\b",
                 r"\binter[_ ]?token[_ ]?latency\b",
                 r"\bdecode[_ ]?latency\b"],
    "e2e_latency": [r"\be2e[_ ]?latency\b",
                    r"\bend[_ ]?to[_ ]?end[_ ]?latency\b"],
    "throughput": [r"\bthroughput\b", r"\btoks?/?s\b",
                   r"\btokens?/(?:sec|second)\b", r"\bprompt_tps\b",
                   r"\bgen_tps\b", r"\brequest_throughput\b"],
    "cache_reuse": [r"\bcache[_ ]?reuse\b", r"\bcache[_ ]?hit\b",
                    r"\bkv[_ ]?reuse\b", r"\bprefix[_ ]?reuse\b",
                    r"\bprefix[_ ]?cache\b"],
    "gpu_utilization": [r"\bgpu[_ ]?util(?:ization)?\b", r"\bdcgm\b",
                        r"\bnvidia-smi\b"],
    "gpu_memory": [r"\bgpu[_ ]?memory\b", r"\bvram\b",
                   r"\bkv[_ ]?cache\b", r"\bmax[_ ]?global[_ ]?vram\b"],
    "power_draw": [r"\bpower[_ ]?draw\b", r"\bipmi\b",
                   r"\bpower[_ ]?telemetry\b", r"\bnvml\b",
                   r"\bpower[_ ]?w\b", r"\bgpu[_ ]?power\b"],
    "migrations": [r"\bmigration\b", r"\bevict\b", r"\bpreempt"],
    "placement_decisions": [r"\bplacement\b",
                            r"\bnode[_ ]?assign", r"\bscheduler\b"],
    "autoscaling_events": [r"\bautoscal", r"\bscale[_ ]?out\b",
                           r"\breplica[_ ]?count\b"],
    "model_load_unload": [r"\bmodel[_ ]?load\b", r"\bmodel[_ ]?unload\b",
                          r"\bcold[_ ]?start\b"],
    "routing_decisions": [r"\brouting\b", r"\bregion[_ ]?route\b",
                          r"\bsession[_ ]?id\b"],
}

# `context_required` patterns only count when the haystack also matches
# COMPUTE_CONTEXT_TOKENS. This prevents banking/telecom/utility datasets
# from being labelled as cloud-cost signals.
ECONOMIC_SIGNAL_PATTERNS = {
    "gpu_type": {
        "patterns": [
            r"\bgpu[_ ]?type\b", r"\baccelerator[_ ]?type\b",
            r"\b(a100|h100|h200|b200|a10g?|t4|v100|p100|l4|l40|gb10|"
            r"mi[0-9]{2,3}|tpu[_ ]?v?\d)\b",
        ],
        "context_required": False,
    },
    "gpu_count": {
        "patterns": [r"\bgpu[_ ]?count\b", r"\bnum[_ ]?gpus?\b",
                     r"\bchip[_ ]?count\b"],
        "context_required": False,
    },
    "gpu_hour_price": {
        "patterns": [r"\bgpu[_ ]?hour(?:ly)?[_ ]?price\b",
                     r"\b\$/(?:gpu[ _]?)?hr?\b",
                     r"\busd/?(?:gpu[ _]?)?hr?\b",
                     r"\bprice[_ ]?per[_ ]?hour\b",
                     r"\bprice[_ ]?per[_ ]?gpu[_ ]?hour\b",
                     r"\binstance[_ ]?type\b.{0,80}\bprice\b",
                     r"\bondemand[_ ]?price\b", r"\bon-?demand[_ ]?price\b",
                     r"\baws[_ ]?pricing\b", r"\bec2[_ ]?pricing\b",
                     r"\bgcp[_ ]?pricing\b", r"\bazure[_ ]?pricing\b"],
        "context_required": True,
    },
    "cloud_cost": {
        "patterns": [r"\bcloud[_ ]?cost\b", r"\bcompute[_ ]?cost\b",
                     r"\bgpu[_ ]?cost\b", r"\binfra[_ ]?cost\b",
                     r"\bcost[_ ]?usd\b"],
        "context_required": True,
    },
    "spot_price": {
        "patterns": [r"\bspot[_ ]?price\b", r"\bspot[_ ]?cost\b",
                     r"\bpreemptible\b", r"\bspot[_ ]?instance\b",
                     r"\bis[_ ]?spot\b"],
        "context_required": True,
    },
    "reserved_ondemand_price": {
        "patterns": [r"\breserved\b.*\bprice", r"\bon[_ ]?demand\b",
                     r"\b1y[_ ]?reserved\b", r"\b3y[_ ]?reserved\b"],
        "context_required": True,
    },
    "energy_per_request": {
        "patterns": [r"\benergy[_ ]?per[_ ]?request\b",
                     r"\benergy[_ ]?per[_ ]?token\b",
                     r"\bj(?:oule)?s?[_ ]?per[_ ]?(?:request|token)\b"],
        "context_required": True,
    },
    "kwh": {
        "patterns": [r"\bkwh\b", r"\bk[_ ]?wh\b", r"\bwatt[_ ]?hours?\b",
                     r"\benergy[_ ]?kwh\b",
                     r"\benergy[_ ]?(?:consumption|use|usage)\b",
                     r"\bpower[_ ]?consumption\b",
                     r"\bcodecarbon\b", r"\brapl\b", r"\bnvml\b"],
        "context_required": True,
    },
    "power_duration": {
        "patterns": [r"\bpower\b.{0,40}\bduration\b",
                     r"\benergy\b.{0,40}\bduration\b"],
        "context_required": True,
    },
    "electricity_price": {
        "patterns": [r"\belectricity[_ ]?price\b",
                     r"\benergy[_ ]?market[_ ]?price\b",
                     r"\bcaiso\b", r"\bercot\b", r"\bpjm\b",
                     r"\bnordpool\b", r"\bclearing[_ ]?price\b"],
        "context_required": False,
    },
    "carbon_intensity": {
        "patterns": [r"\bcarbon[_ ]?intensity\b",
                     r"\bco2[_ ]?per[_ ]?kwh\b",
                     r"\bg[_ ]?co2\b", r"\bgco2/?kwh\b",
                     r"\belectricitymaps?\b", r"\bwattime\b"],
        "context_required": False,
    },
    "carbon_cost": {
        "patterns": [r"\bcarbon[_ ]?cost\b", r"\bcarbon[_ ]?price\b",
                     r"\bcarbon[_ ]?tax\b"],
        "context_required": True,
    },
    "region_zone": {
        "patterns": [r"\bavailability[_ ]?zone\b", r"\bbidding[_ ]?zone\b",
                     r"\b(us-east|us-west|eu-west|ap-(?:south|northeast|"
                     r"southeast))-\d\b", r"\b(europe|northamerica|"
                     r"southamerica|asia)-[a-z]+\d?\b"],
        "context_required": True,
    },
    "datacenter_energy": {
        "patterns": [r"\bdatacent(?:er|re)[_ ]?energy\b",
                     r"\bdatacent(?:er|re)[_ ]?power\b",
                     r"\bfacility[_ ]?power\b",
                     r"\bpue\b", r"\bpower[_ ]?usage[_ ]?effectiveness\b"],
        "context_required": True,
    },
    "cost_per_token": {
        "patterns": [r"\bcost[_ ]?per[_ ]?token\b",
                     r"\b\$/(?:1k)?[_ ]?tokens?\b",
                     r"\busd/?tokens?\b"],
        "context_required": True,
    },
    "cost_per_request": {
        "patterns": [r"\bcost[_ ]?per[_ ]?request\b",
                     r"\b\$/(?:request|req)\b"],
        "context_required": True,
    },
    "billing_chargeback": {
        "patterns": [r"\bcloud[_ ]?billing\b", r"\bcloud[_ ]?invoice\b",
                     r"\bcloud[_ ]?chargeback\b",
                     r"\bcompute[_ ]?invoice\b",
                     r"\bgpu[_ ]?billing\b"],
        "context_required": True,
    },
}

# Existing Aurelius operational traces — used as join targets if a candidate
# is B-only (economics) but joinable.
EXISTING_TRACE_JOIN_KEYS = {
    "CARA": {"gpu_type", "timestamp", "model"},
    "AcmeTrace": {"gpu_type", "timestamp", "region", "cluster"},
    "Optimum": {"gpu_type", "model", "quantization"},
    "BurstGPT": {"timestamp", "model"},
    "Google Cluster": {"timestamp", "region", "machine"},
    "SwissAI": {"timestamp", "model"},
    "AgentPerfBench": {"gpu_type", "model"},
}


# ---------------------------------------------------------------------------
# HF API client (stdlib only).
# ---------------------------------------------------------------------------


class HFClient:
    """Minimal HF API client. Token is read from `HF_TOKEN` and sent via
    `Authorization` header — never written to any file or log line."""

    def __init__(self, *, timeout_s: float = 15.0,
                 rate_limit_sleep_s: float = 0.25):
        self.token = os.environ.get("HF_TOKEN") or None
        self.timeout_s = timeout_s
        self.rate_limit_sleep_s = rate_limit_sleep_s
        self._last_call = 0.0

    def _request(self, url: str) -> Optional[dict | list]:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.rate_limit_sleep_s:
            time.sleep(self.rate_limit_sleep_s - elapsed)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = r.read()
                self._last_call = time.monotonic()
                if not body:
                    return None
                return json.loads(body)
        except urllib.error.HTTPError as e:
            logger.warning("HF API HTTP %s on %s", e.code, _redact(url))
            return None
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as e:
            logger.warning("HF API error on %s: %s", _redact(url), e)
            return None

    def search(self, q: str, *, limit: int = 30) -> list[dict]:
        url = (f"{HF_API_BASE}/datasets?"
               + urllib.parse.urlencode({"search": q, "limit": int(limit)}))
        res = self._request(url)
        return res if isinstance(res, list) else []

    def get(self, dataset_id: str) -> Optional[dict]:
        url = (f"{HF_API_BASE}/datasets/"
               + urllib.parse.quote(dataset_id, safe='/'))
        res = self._request(url)
        return res if isinstance(res, dict) else None

    def readme(self, dataset_id: str) -> str:
        """Best-effort README fetch via the raw content URL. Bounded to
        128 KiB to avoid pulling large docs into memory."""
        url = ("https://huggingface.co/datasets/"
               + urllib.parse.quote(dataset_id, safe='/')
               + "/raw/main/README.md")
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.read(128 * 1024).decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, UnicodeError):
            return ""


def _redact(url: str) -> str:
    return re.sub(r"token=[^&]+", "token=REDACTED", url)


# ---------------------------------------------------------------------------
# Signal detection.
# ---------------------------------------------------------------------------


@dataclass
class SignalMatches:
    operational: dict[str, list[str]] = field(default_factory=dict)
    economic: dict[str, list[str]] = field(default_factory=dict)


def detect_signals(haystack: str) -> SignalMatches:
    matches = SignalMatches()
    lowered = haystack.lower()
    has_compute_context = bool(COMPUTE_CONTEXT_TOKENS.search(lowered))

    for signal, patterns in OPERATIONAL_SIGNAL_PATTERNS.items():
        hits = []
        for pat in patterns:
            for m in re.findall(pat, lowered):
                token = m if isinstance(m, str) else str(m)
                if token and token not in hits:
                    hits.append(token)
        # operational signals always require compute context to count —
        # banking "scheduler" / supply-chain "routing" / utility
        # "queue_depth" telemetry are not what we want.
        if hits and has_compute_context:
            matches.operational[signal] = hits[:5]

    for signal, spec in ECONOMIC_SIGNAL_PATTERNS.items():
        patterns = spec["patterns"]
        context_required = spec["context_required"]
        if context_required and not has_compute_context:
            continue
        hits = []
        for pat in patterns:
            for m in re.findall(pat, lowered):
                token = m if isinstance(m, str) else str(m)
                if token and token not in hits:
                    hits.append(token)
        if hits:
            matches.economic[signal] = hits[:5]
    return matches


def field_quality_label(signals: SignalMatches, license_str: Optional[str],
                        description: str, readme: str) -> str:
    """One of: measured | derived | proxy | synthetic | missing."""
    blob = f"{description or ''}\n{readme or ''}".lower()
    if not signals.operational and not signals.economic:
        return "missing"
    if any(k in blob for k in (
            "synthetic", "simulated", "mock", "fake data", "generated by",
            "llm-generated", "self-declared synthetic")):
        return "synthetic"
    if any(k in blob for k in (
            "proxy", "approximation", "estimated", "derived", "fitted")):
        return "derived"
    if any(k in blob for k in (
            "measured", "real production", "ipmi", "dcgm", "nvidia-smi",
            "codecarbon", "rapl", "wall-clock", "telemetry")):
        return "measured"
    return "proxy"


# ---------------------------------------------------------------------------
# Scoring + classification.
# ---------------------------------------------------------------------------


def score_candidate(meta: dict, signals: SignalMatches,
                    description: str, readme: str,
                    license_str: Optional[str]) -> dict:
    """Score each candidate on 7 dimensions (0-10), then classify A+/A/B/C/D/F."""
    op_count = len(signals.operational)
    ec_count = len(signals.economic)

    blob = f"{description or ''}\n{readme or ''}".lower()
    has_real_pricing = any(
        s in signals.economic for s in (
            "gpu_hour_price", "cloud_cost", "spot_price",
            "reserved_ondemand_price", "cost_per_token", "cost_per_request",
            "electricity_price"))
    has_kwh = "kwh" in signals.economic or "energy_per_request" in signals.economic
    has_carbon = ("carbon_intensity" in signals.economic
                  or "carbon_cost" in signals.economic)

    economic_signal_quality = min(10, ec_count * 2 + (3 if has_real_pricing else 0))
    operational_signal_quality = min(10, op_count * 2)

    join_keys = []
    if "gpu_type" in signals.economic or "gpu_type" in signals.operational:
        join_keys.append("gpu_type")
    if "region_zone" in signals.economic:
        join_keys.append("region")
    if any(k in signals.operational for k in (
            "queue_wait", "ttft", "e2e_latency", "throughput")):
        join_keys.append("timestamp")
    if has_kwh:
        join_keys.append("energy")
    joinability = min(10, len(join_keys) * 2 + (2 if signals.operational else 0))

    license_safe = (license_str or "").lower()
    if any(s in license_safe for s in ("apache", "mit", "cc-by-4", "cc-by-2",
                                       "cc0", "bsd", "cdla-permissive")):
        license_safety = 9
    elif "cc-by-nc" in license_safe or "nc-nd" in license_safe:
        license_safety = 3
    elif license_safe in ("", "none", "unknown") or license_str is None:
        license_safety = 4
    else:
        license_safety = 6

    if any(s in blob for s in ("synthetic", "simulated", "mock")):
        production_similarity = 2
    elif any(s in blob for s in ("benchmark", "leaderboard")):
        production_similarity = 5
    elif any(s in blob for s in ("real production", "pilot", "datacenter",
                                 "azure trace", "google cluster",
                                 "alibaba")):
        production_similarity = 9
    else:
        production_similarity = 6

    downloads = meta.get("downloads") or 0
    uniqueness = 7 if (has_real_pricing or has_kwh or has_carbon) else 4
    if downloads > 1000:
        uniqueness -= 1  # well-known

    expected_scorer_value = round(
        0.45 * economic_signal_quality
        + 0.20 * operational_signal_quality
        + 0.15 * joinability
        + 0.10 * license_safety
        + 0.10 * production_similarity, 2)

    # Core operational signals — only these (not gpu_memory / migrations /
    # gpu_type alone) qualify a dataset as having "operational behavior".
    CORE_OPS = {"queue_depth", "queue_wait", "ttft", "tpot_itl",
                "e2e_latency", "throughput", "cache_reuse",
                "gpu_utilization", "power_draw", "autoscaling_events",
                "model_load_unload"}
    core_op_count = len(set(signals.operational) & CORE_OPS)

    is_synth = any(s in blob for s in (
        "synthetic", "simulated", "mock", "self-declared synthetic"))

    if is_synth and (has_real_pricing or has_kwh):
        classification = "D"
    elif core_op_count >= 1 and has_real_pricing:
        classification = "A_PLUS"
    elif core_op_count >= 1 and has_kwh:
        classification = "A"
    elif (has_real_pricing or has_kwh or has_carbon):
        classification = "B"
    elif core_op_count >= 1:
        classification = "C"
    else:
        classification = "F"

    return {
        "economic_signal_quality": economic_signal_quality,
        "operational_signal_quality": operational_signal_quality,
        "joinability": joinability,
        "license_safety": license_safety,
        "production_similarity": production_similarity,
        "uniqueness": max(0, uniqueness),
        "expected_scorer_value": expected_scorer_value,
        "classification": classification,
        "join_keys_available": join_keys,
        "has_real_pricing": has_real_pricing,
        "has_kwh": has_kwh,
        "has_carbon": has_carbon,
    }


def recommended_action(meta: dict, scores: dict,
                       license_str: Optional[str],
                       signals: SignalMatches,
                       description: str, readme: str) -> str:
    if meta.get("gated") is True:
        return "gated_blocked"
    license_lower = (license_str or "").lower()
    if "nc-nd" in license_lower or "noderivatives" in license_lower:
        return "license_blocked"
    blob = f"{description or ''}\n{readme or ''}".lower()
    is_synth = any(s in blob for s in ("synthetic", "simulated", "mock",
                                       "self-declared synthetic"))
    if is_synth and (scores["has_real_pricing"] or scores["has_kwh"]):
        return "reject_synthetic"
    # No declared license -> conservative non-redistribution policy.
    # The class may still be A/B, but downstream ingest is blocked.
    license_missing = license_str is None or license_lower in (
        "", "none", "unknown")
    if scores["classification"] in ("A_PLUS", "A") and not license_missing:
        return "ingest_now"
    if scores["classification"] in ("A_PLUS", "A") and license_missing:
        return "metadata_only"
    if scores["classification"] == "B" and scores["joinability"] >= 4 and (
            not license_missing):
        return "join_overlay_candidate"
    if scores["classification"] == "C" and not (
            scores["has_real_pricing"] or scores["has_kwh"]):
        return "reject_no_economics"
    if scores["classification"] == "F":
        return "reject_no_ops"
    return "metadata_only"


def can_pair_with_traces(signals: SignalMatches,
                         scores: dict) -> dict[str, bool]:
    out = {}
    keys = set(scores["join_keys_available"])
    for name, trace_keys in EXISTING_TRACE_JOIN_KEYS.items():
        out[name] = bool(keys & trace_keys)
    return out


# ---------------------------------------------------------------------------
# Per-candidate field assembly.
# ---------------------------------------------------------------------------


def _extract_license(meta: dict) -> Optional[str]:
    if meta.get("license"):
        return str(meta["license"])
    card = meta.get("cardData") or {}
    if isinstance(card, dict):
        lic = card.get("license")
        if isinstance(lic, list):
            return ", ".join(str(x) for x in lic if x)
        if lic:
            return str(lic)
    for t in (meta.get("tags") or []):
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def _extract_size(meta: dict) -> tuple[Optional[int], Optional[int]]:
    """Returns (row_count, storage_bytes) best-effort."""
    card = meta.get("cardData") or {}
    rows = None
    storage = meta.get("usedStorage")
    if isinstance(card, dict):
        ds_info = card.get("dataset_info")
        if isinstance(ds_info, dict):
            sp = ds_info.get("splits") or []
            if isinstance(sp, list):
                for s in sp:
                    n = s.get("num_examples")
                    if isinstance(n, int):
                        rows = (rows or 0) + n
        if isinstance(ds_info, list):
            for cfg in ds_info:
                for s in (cfg.get("splits") or []):
                    n = s.get("num_examples")
                    if isinstance(n, int):
                        rows = (rows or 0) + n
    return rows, storage


def _extract_feature_names(meta: dict) -> list[str]:
    out = set()
    card = meta.get("cardData") or {}
    if isinstance(card, dict):
        ds_info = card.get("dataset_info")
        infos = ds_info if isinstance(ds_info, list) else [ds_info] if ds_info else []
        for info in infos:
            if not isinstance(info, dict):
                continue
            feats = info.get("features") or []
            if isinstance(feats, list):
                for f in feats:
                    if isinstance(f, dict) and "name" in f:
                        out.add(str(f["name"]))
    return sorted(out)


def _extract_files(meta: dict) -> list[str]:
    sibs = meta.get("siblings") or []
    out = []
    for s in sibs:
        if isinstance(s, dict) and "rfilename" in s:
            out.append(str(s["rfilename"]))
    return out[:50]


def inspect(client: HFClient, dataset_id: str,
            matched_terms: list[str]) -> Optional[dict]:
    detail = client.get(dataset_id)
    if not detail or "id" not in detail:
        return None
    license_str = _extract_license(detail)
    description = str(detail.get("description") or "")
    tags = detail.get("tags") or []
    rows, storage = _extract_size(detail)
    features = _extract_feature_names(detail)
    files = _extract_files(detail)
    readme = client.readme(dataset_id) if detail.get("gated") is not True else ""

    haystack = "\n".join([
        description, " ".join(str(t) for t in tags),
        " ".join(features), " ".join(files), readme,
    ])
    signals = detect_signals(haystack)
    override = SEED_SIGNAL_OVERRIDES.get(dataset_id)
    override_applied = []
    if override:
        # Override is authoritative for the seeds — replace detected ops
        # with the verified list (so README false positives like
        # `gpu_memory_gb` mentioned only as an SKU spec don't sneak in).
        ops_override = override.get("operational")
        if ops_override is not None:
            for sig in list(signals.operational):
                if sig not in ops_override:
                    del signals.operational[sig]
                    override_applied.append(f"op-drop:{sig}")
            for sig in ops_override:
                if sig not in signals.operational:
                    signals.operational[sig] = ["registry_verified"]
                    override_applied.append(f"op-add:{sig}")
        ec_override = override.get("economic")
        if ec_override is not None:
            for sig in ec_override:
                if sig not in signals.economic:
                    signals.economic[sig] = ["registry_verified"]
                    override_applied.append(f"ec-add:{sig}")
    scores = score_candidate(detail, signals, description, readme,
                             license_str)
    fq = (override.get("field_quality")
          if override and override.get("field_quality")
          else field_quality_label(signals, license_str, description, readme))
    action = recommended_action(detail, scores, license_str, signals,
                                description, readme)
    pairing = can_pair_with_traces(signals, scores)

    return {
        "dataset_id": dataset_id,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": license_str,
        "gated_status": ("gated" if detail.get("gated") is True
                         else ("private" if detail.get("private") is True
                               else "public")),
        "row_count": rows,
        "storage_size_bytes": storage,
        "files_inspected": files,
        "schema_fields": features,
        "matched_search_terms": matched_terms,
        "operational_signals_present": sorted(signals.operational.keys()),
        "economic_signals_present": sorted(signals.economic.keys()),
        "join_keys_available": scores["join_keys_available"],
        "timestamp_available": "timestamp" in scores["join_keys_available"],
        "region_available": "region" in scores["join_keys_available"],
        "gpu_type_available": "gpu_type" in scores["join_keys_available"],
        "price_or_cost_available": scores["has_real_pricing"],
        "energy_available": scores["has_kwh"],
        "carbon_available": scores["has_carbon"],
        "field_quality": fq,
        "trust_tier": _trust_tier(scores, fq),
        "can_pair_with_existing_traces": pairing,
        "recommended_action": action,
        "scores": {
            "economic_signal_quality": scores["economic_signal_quality"],
            "operational_signal_quality": scores["operational_signal_quality"],
            "joinability": scores["joinability"],
            "license_safety": scores["license_safety"],
            "production_similarity": scores["production_similarity"],
            "uniqueness": scores["uniqueness"],
            "expected_scorer_value": scores["expected_scorer_value"],
        },
        "classification": scores["classification"],
        "downloads": detail.get("downloads"),
        "likes": detail.get("likes"),
        "last_modified": detail.get("lastModified"),
        "seed_override_applied": override_applied,
        "seed_override_evidence": (
            override.get("evidence") if override else None),
    }


def _trust_tier(scores: dict, fq: str) -> str:
    if fq == "synthetic":
        return "tier_6_synthetic"
    if scores["classification"] == "A_PLUS":
        return "tier_2_paired_ops_economics"
    if scores["classification"] == "A":
        return "tier_3_ops_plus_energy"
    if scores["classification"] == "B":
        return "tier_4_economics_only_joinable"
    if scores["classification"] == "C":
        return "tier_4_ops_only"
    return "tier_5_metadata_or_other"


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def gather_candidates(client: HFClient, terms: Iterable[str],
                      *, limit_per_query: int = 30) -> dict[str, list[str]]:
    """Returns: dataset_id -> [matched_search_terms]."""
    bucket: dict[str, list[str]] = {}
    for q in terms:
        hits = client.search(q, limit=limit_per_query)
        for h in hits:
            if not isinstance(h, dict) or "id" not in h:
                continue
            ds_id = str(h["id"])
            bucket.setdefault(ds_id, [])
            if q not in bucket[ds_id]:
                bucket[ds_id].append(q)
        logger.info("search %r -> %d hits", q, len(hits))
    return bucket


def build_audit(client: HFClient, *, limit_per_query: int = 30,
                max_inspect: int = 200) -> dict:
    t0 = time.time()
    all_terms = (PRIMARY_SEARCH_TERMS + COMBINATION_SEARCH_TERMS
                 + SINGLE_WORD_PROBES)
    bucket = gather_candidates(client, all_terms,
                               limit_per_query=limit_per_query)
    for seed in SEED_DATASET_IDS:
        bucket.setdefault(seed, []).append("seed::known_high_priority")

    discovered = sorted(bucket.keys())
    logger.info("total unique candidates discovered: %d", len(discovered))

    inspected = []
    for ds_id in discovered[:max_inspect]:
        try:
            rec = inspect(client, ds_id, bucket[ds_id])
        except Exception as e:  # noqa: BLE001 — never fail the whole sweep
            logger.warning("inspect failed for %s: %s", ds_id, e)
            continue
        if rec is None:
            continue
        inspected.append(rec)

    inspected.sort(
        key=lambda r: (
            -r["scores"]["expected_scorer_value"],
            -r["scores"]["economic_signal_quality"],
            r["dataset_id"],
        ),
    )

    # Buckets used by the audit + report.
    a_plus = [r for r in inspected if r["classification"] == "A_PLUS"]
    a = [r for r in inspected if r["classification"] == "A"]
    b = [r for r in inspected if r["classification"] == "B"]
    c = [r for r in inspected if r["classification"] == "C"]
    d = [r for r in inspected if r["classification"] == "D"]
    f = [r for r in inspected if r["classification"] == "F"]

    top20_economic = sorted(
        [r for r in inspected if r["economic_signals_present"]],
        key=lambda r: (
            -r["scores"]["economic_signal_quality"],
            -r["scores"]["expected_scorer_value"],
        ),
    )[:20]

    elapsed = time.time() - t0
    return {
        "doc_version": "economic_signal_discovery_audit_v1",
        "stage": "hf_economic_signal_discovery_v1",
        "production_claim": False,
        "audit_only": True,
        "hf_token_committed": False,
        "raw_data_committed": False,
        "client": "live_hf_api" + (
            " (HF_TOKEN_set)" if client.token else " (anon)"),
        "search_terms": {
            "primary": PRIMARY_SEARCH_TERMS,
            "combinations": COMBINATION_SEARCH_TERMS,
            "single_word_probes": SINGLE_WORD_PROBES,
        },
        "search_summary": {
            "total_terms_searched": len(all_terms),
            "total_unique_datasets_discovered": len(discovered),
            "total_inspected": len(inspected),
            "elapsed_s": round(elapsed, 2),
        },
        "buckets": {
            "A_PLUS_paired_ops_and_pricing": [r["dataset_id"] for r in a_plus],
            "A_ops_plus_energy_only": [r["dataset_id"] for r in a],
            "B_economics_only_joinable": [r["dataset_id"] for r in b],
            "C_ops_only": [r["dataset_id"] for r in c],
            "D_synthetic_economics": [r["dataset_id"] for r in d],
            "F_irrelevant": [r["dataset_id"] for r in f],
        },
        "top_20_economic_candidates": top20_economic,
        "special_audit": _special_audit(inspected),
        "candidates": inspected,
        "operator_policy_only_fields": [
            "energy_price_per_kwh_usd",
            "carbon_price_per_kg_usd",
            "per_gpu_hour_price_usd (fleet-actual, not public list price)",
            "internal_chargeback_rate",
            "reserved_capacity_amortization",
            "datacenter_pue_for_the_operator_facility",
        ],
    }


def _special_audit(records: list[dict]) -> dict:
    CORE_OPS = {"queue_depth", "queue_wait", "ttft", "tpot_itl",
                "e2e_latency", "throughput", "cache_reuse",
                "gpu_utilization", "power_draw"}

    def core_ops(r):
        return CORE_OPS & set(r["operational_signals_present"])

    paired_pricing = [r["dataset_id"] for r in records
                      if r["price_or_cost_available"] and core_ops(r)]
    billing_paired = [r["dataset_id"] for r in records
                      if r["price_or_cost_available"]
                      and (core_ops(r) & {"queue_wait", "ttft", "throughput",
                                          "e2e_latency"})]
    energy_paired = [r["dataset_id"] for r in records
                     if r["energy_available"]
                     and (core_ops(r) & {"ttft", "tpot_itl", "e2e_latency",
                                         "throughput"})]
    calibration_capable = sorted(
        [r["dataset_id"] for r in records
         if r["classification"] in ("A_PLUS", "A")
         and r["field_quality"] in ("measured", "derived")],
    )
    return {
        "q1_real_gpu_hour_pricing_plus_telemetry": paired_pricing,
        "q2_cloud_billing_plus_workload_telemetry": billing_paired,
        "q3_energy_per_request_plus_latency": energy_paired,
        "q4_calibrate_scorer_without_invented_constants": calibration_capable,
        "q5_remaining_operator_policy_only_coefficients": [
            "energy_price_per_kwh_usd",
            "carbon_price_per_kg_usd",
            "per_gpu_hour_price_usd (operator fleet-actual)",
            "internal_chargeback_rate",
        ],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--limit-per-query", type=int, default=30)
    p.add_argument("--max-inspect", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = HFClient()
    audit = build_audit(client, limit_per_query=args.limit_per_query,
                        max_inspect=args.max_inspect)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True)
    logger.info("wrote audit -> %s", args.output)

    # Defensive: token must not appear in any committed artefact.
    token = os.environ.get("HF_TOKEN") or ""
    if token:
        with open(args.output) as fh:
            assert token not in fh.read(), (
                "HF_TOKEN leaked into audit JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
