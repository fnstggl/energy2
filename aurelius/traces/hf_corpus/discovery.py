"""Hugging Face dataset discovery, scoring, and trace-type classification.

The public-facing entry points are:

- ``HFAPIClient`` — minimal stdlib (urllib + json) client for the public
  Hugging Face Dataset API. Honours ``HF_TOKEN`` when present, never logs it,
  and surfaces gated / 401 / 404 explicitly. Pluggable: ``OfflineHFClient``
  serves cached JSON fixtures for tests / hermetic CI.
- ``classify_dataset(meta)`` — maps a dataset metadata blob onto a canonical
  trace type (``aurelius/traces/hf_corpus/schemas.py::CANONICAL_TRACE_TYPES``)
  using a deterministic keyword-matching algorithm. Returns
  ``mixed_or_unknown_trace`` when classification is ambiguous.
- ``score_dataset(meta, classification)`` — produces the per-candidate score
  tuple required by the mission spec.
- ``discover(client, query_groups)`` — runs all keyword group searches,
  merges results by dataset id, classifies + scores each candidate, and
  returns a list of dicts ready for ``hf_dataset_candidates.json``.

Anti-spam: this module favours information-density over volume. A dataset
with measured TTFT / TPOT / queue_wait / gpu_utilization scores higher than
hundreds of plain conversation datasets. See ``docs/HF_DATASET_REGISTRY.md``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

from .schemas import CANONICAL_TRACE_TYPE_TO_TRUST_TIER

logger = logging.getLogger(__name__)


HF_API_BASE = "https://huggingface.co/api"

# Default User-Agent. We never include HF_TOKEN here or anywhere logged.
USER_AGENT = "aurelius-hf-discovery/1.0 (+https://aurelius.energy)"

# Hard search-result cap per query — prevents accidentally pulling tens of
# thousands of low-value datasets and exploding the candidate registry.
DEFAULT_MAX_RESULTS_PER_QUERY = 25


# Default keyword groups. Each group is one HF search query. Groups are
# intentionally narrow because broad queries dilute precision and bias the
# corpus toward conversation-only datasets.
DEFAULT_QUERY_GROUPS = {
    "latency_benchmark": [
        "TTFT TPOT latency benchmark",
        "vLLM SGLang inference benchmark",
        "llm-inference benchmark",
        "agent-perf",
        "AgentPerfBench",
    ],
    "kernel_profile": [
        "GEMM kernel profile",
        "Nsight CUDA benchmark",
        "kernels labeled",
    ],
    "cluster_scheduler": [
        "Alibaba GPU trace",
        "Philly trace",
        "supercloud trace",
        "Azure functions trace",
    ],
    "cache_residency": [
        "prefix cache benchmark",
        "kv cache trace",
        "model residency",
        "prefixbench",
    ],
    "telemetry": [
        "vLLM telemetry",
        "Triton serving telemetry",
        "Ray Serve trace",
        "Kubernetes Prometheus serving",
        "DCGM",
    ],
    "request_shape": [
        "ShareGPT",
        "LMSYS chatbot arena",
        "chatbot arena conversations",
        "sg_52k",
    ],
}


# Keyword -> trace_type classification table. Order = highest-trust-class
# wins (telemetry > scheduler > latency > kernel > residency > shape). The
# classifier picks the most specific match; ties resolve toward higher-trust.
CLASSIFICATION_KEYWORDS = {
    "telemetry_trace": [
        "dcgm", "prometheus_export", "vllm_telemetry", "triton_telemetry",
        "ray_serve_telemetry", "kubernetes_pod_metrics", "autoscaling_trace",
        "real production telemetry", "replica_count", "sla_violation_rate",
    ],
    "cluster_scheduler_trace": [
        "alibaba gpu trace", "philly trace", "supercloud", "azure functions",
        "scheduler trace", "job queue", "queue_wait", "gpu_count", "submit_time",
    ],
    "latency_benchmark_trace": [
        "ttft", "tpot", "itl", "e2el", "e2e latency", "p99 latency",
        "request_throughput", "token_throughput", "llm-inference", "benchmarking",
        "vllm", "sglang", "agent-perf",
    ],
    "kernel_profile_trace": [
        "gemm", "nsight", "kernel profile", "ncu", "cuda kernel",
        "kernels_labeled", "kernel_name", "dram_bytes", "op_type",
    ],
    "cache_residency_trace": [
        "prefix cache", "kv cache", "model residency", "cache hit",
        "prefixbench", "cold start", "cache_affinity",
    ],
    "request_shape_trace": [
        "sharegpt", "lmsys chatbot", "chatbot_arena", "conversation log",
        "user prompts", "instruction tuning", "eval dataset",
    ],
}


# Target-signal keywords used by ``score_dataset`` for the
# ``available_signals`` discovery field. Order matters — denser-information
# signals come first so the JSON output is stable and reviewable.
TARGET_SIGNALS = [
    "ttft", "tpot", "itl", "e2e_latency", "latency_p50", "latency_p90",
    "latency_p95", "latency_p99", "throughput", "request_throughput",
    "token_throughput", "concurrency", "batch_size", "sequence_length",
    "prompt_tokens", "output_tokens", "gpu_type", "gpu_utilization",
    "gpu_memory", "memory_pressure", "queue_wait", "queue_depth", "timeout",
    "sla", "failure", "cold_start", "prefix_cache", "cache_hit",
    "model_residency", "routing", "vllm", "sglang", "triton", "ray_serve",
    "kubernetes", "prometheus", "dcgm", "autoscaling", "replica_count",
    "pod_churn", "kernel_duration", "gemm", "cuda", "nsight",
]


@dataclass(frozen=True)
class HFDatasetMeta:
    """Subset of HF dataset metadata used by classify/score.

    Built from the HF API ``/api/datasets/{id}`` (preferred — has ``cardData``
    + ``siblings``) or the lighter ``/api/datasets?search=...`` listing (only
    has ``tags`` + ``description``). Missing fields are ``None``; classify /
    score never zero-fill.
    """

    dataset_id: str
    dataset_url: str
    gated: Optional[bool]
    private: Optional[bool]
    license: Optional[str]
    description: Optional[str]
    tags: tuple
    downloads: Optional[int]
    likes: Optional[int]
    size_categories: tuple
    configs: tuple                 # config_name strings
    splits: tuple                  # (config, split, num_examples)
    feature_names: tuple           # all feature names across configs
    siblings: tuple                # repo filenames
    last_modified: Optional[str]


class HFAPIClient:
    """Minimal HF API client (stdlib-only).

    The client never raises on auth failure during ``search`` or ``get``: it
    returns ``None`` / ``[]`` and the caller decides whether to mark a
    dataset ``GATED_BLOCKED`` or skip it. The token, when provided, is sent
    via ``Authorization: Bearer <token>`` and is **never** logged.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        timeout_s: float = 15.0,
        rate_limit_sleep_s: float = 0.5,
    ):
        self.token = token or os.environ.get("HF_TOKEN") or None
        self.timeout_s = timeout_s
        self.rate_limit_sleep_s = rate_limit_sleep_s
        self._last_call_ts = 0.0

    def _request(self, url: str) -> Optional[dict]:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.rate_limit_sleep_s:
            time.sleep(self.rate_limit_sleep_s - elapsed)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read()
                self._last_call_ts = time.monotonic()
                if not payload:
                    return None
                return json.loads(payload)
        except urllib.error.HTTPError as e:
            logger.warning("HF API %s -> HTTP %s", _redact_token(url), e.code)
            return None
        except urllib.error.URLError as e:
            logger.warning("HF API %s -> URL error: %s", _redact_token(url), e.reason)
            return None
        except (TimeoutError, json.JSONDecodeError) as e:
            logger.warning("HF API %s -> %s", _redact_token(url), e)
            return None

    def search(
        self, query: str, *, limit: int = DEFAULT_MAX_RESULTS_PER_QUERY,
    ) -> list[dict]:
        url = (
            f"{HF_API_BASE}/datasets?"
            + urllib.parse.urlencode({"search": query, "limit": int(limit)})
        )
        result = self._request(url)
        if result is None:
            return []
        if not isinstance(result, list):
            return []
        return result

    def get(self, dataset_id: str) -> Optional[dict]:
        url = f"{HF_API_BASE}/datasets/{urllib.parse.quote(dataset_id, safe='/')}"
        return self._request(url)


def _redact_token(url: str) -> str:
    """Defensive: even though we send the token in headers not URL, avoid
    accidental log leaks if a caller ever stuffs a token into the URL."""
    if "token=" in url:
        return url.split("token=")[0] + "token=REDACTED"
    return url


class OfflineHFClient:
    """Test / hermetic-CI client.

    Loads dataset listings + per-dataset detail from a local fixtures
    directory:

    - ``<root>/search/<safe_query>.json`` — list[dict] (search response).
    - ``<root>/datasets/<safe_id>.json`` — dict (single dataset detail).

    ``<safe_query>`` and ``<safe_id>`` use ``_safe_name`` so filenames are
    cross-platform stable. Missing fixtures return ``[]`` / ``None``, never
    raise, which matches the live client behaviour on auth failure.
    """

    def __init__(self, root: str):
        self.root = root

    def search(
        self, query: str, *, limit: int = DEFAULT_MAX_RESULTS_PER_QUERY,
    ) -> list[dict]:
        path = os.path.join(self.root, "search", _safe_name(query) + ".json")
        if not os.path.exists(path):
            return []
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        return data[:limit]

    def get(self, dataset_id: str) -> Optional[dict]:
        path = os.path.join(self.root, "datasets", _safe_name(dataset_id) + ".json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch == "/":
            out.append("__")
        else:
            out.append("_")
    return "".join(out).lower()


def safe_dataset_dirname(dataset_id: str) -> str:
    """Map ``namespace/name`` to a filesystem-safe directory name."""
    return _safe_name(dataset_id)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _extract_license(meta: dict) -> Optional[str]:
    card = meta.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        return ",".join(str(x) for x in lic if x)
    if lic:
        return str(lic)
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def _extract_size_categories(meta: dict) -> list[str]:
    cats: list[str] = []
    for t in meta.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("size_categories:"):
            cats.append(t.split(":", 1)[1])
    card = meta.get("cardData") or {}
    cd = card.get("size_categories")
    if isinstance(cd, list):
        for x in cd:
            if x not in cats:
                cats.append(str(x))
    elif cd:
        cats.append(str(cd))
    return cats


def _extract_configs_and_features(meta: dict) -> tuple[list[str], list[tuple], list[str]]:
    """Return (config_names, (config, split, num_examples) tuples, feature_names)."""
    card = meta.get("cardData") or {}
    configs = card.get("configs") or []
    config_names: list[str] = []
    splits: list[tuple] = []
    feature_names: list[str] = []
    seen_features: set = set()
    for c in configs:
        if isinstance(c, dict) and "config_name" in c:
            config_names.append(str(c["config_name"]))
    dataset_info = card.get("dataset_info") or []
    if isinstance(dataset_info, dict):
        dataset_info = [dataset_info]
    for di in dataset_info:
        if not isinstance(di, dict):
            continue
        cname = str(di.get("config_name") or "default")
        for s in di.get("splits") or []:
            if isinstance(s, dict):
                splits.append((cname, str(s.get("name") or ""),
                               int(s.get("num_examples") or 0)))
        for f in di.get("features") or []:
            if isinstance(f, dict) and "name" in f:
                fname = str(f["name"])
                if fname not in seen_features:
                    seen_features.add(fname)
                    feature_names.append(fname)
    return config_names, splits, feature_names


def parse_hf_metadata(raw: dict) -> HFDatasetMeta:
    """Map HF API response to ``HFDatasetMeta``.

    Works for both the lighter ``search`` response and the detailed
    ``get`` response — missing fields stay ``None``.
    """

    if raw is None:
        raise ValueError("parse_hf_metadata called on None")
    dataset_id = str(raw.get("id") or raw.get("modelId") or "")
    if not dataset_id:
        raise ValueError(f"HF metadata missing 'id': {raw}")
    tags = tuple(str(t) for t in (raw.get("tags") or []) if isinstance(t, str))
    config_names, splits, feature_names = _extract_configs_and_features(raw)
    siblings_raw = raw.get("siblings") or []
    siblings = tuple(
        s["rfilename"] for s in siblings_raw
        if isinstance(s, dict) and isinstance(s.get("rfilename"), str)
    )
    gated_raw = raw.get("gated")
    if isinstance(gated_raw, bool):
        gated_norm = gated_raw
    elif isinstance(gated_raw, str) and gated_raw.lower() in ("auto", "manual"):
        gated_norm = True
    else:
        gated_norm = None
    return HFDatasetMeta(
        dataset_id=dataset_id,
        dataset_url=f"https://huggingface.co/datasets/{dataset_id}",
        gated=gated_norm,
        private=(raw.get("private") if isinstance(raw.get("private"), bool) else None),
        license=_extract_license(raw),
        description=(raw.get("description") if isinstance(raw.get("description"), str)
                     else None),
        tags=tags,
        downloads=(int(raw["downloads"]) if isinstance(raw.get("downloads"), (int, float))
                   else None),
        likes=(int(raw["likes"]) if isinstance(raw.get("likes"), (int, float)) else None),
        size_categories=tuple(_extract_size_categories(raw)),
        configs=tuple(config_names),
        splits=tuple(splits),
        feature_names=tuple(feature_names),
        siblings=siblings,
        last_modified=(raw.get("lastModified") if isinstance(raw.get("lastModified"), str)
                       else None),
    )


# ---------------------------------------------------------------------------
# Classification + signals
# ---------------------------------------------------------------------------


def _searchable_text(meta: HFDatasetMeta) -> str:
    parts = [meta.dataset_id]
    if meta.description:
        parts.append(meta.description)
    parts.extend(meta.tags)
    parts.extend(meta.feature_names)
    parts.extend(meta.siblings)
    return " ".join(parts).lower()


def classify_dataset(meta: HFDatasetMeta) -> dict:
    """Return ``{"trace_type": ..., "evidence": {...}}``.

    The classifier picks the most-specific match. Ties resolve toward
    higher-trust trace types so a dataset with both latency and telemetry
    keywords is classified as ``telemetry_trace`` (more conservative under
    federated evaluation). When no keyword matches at all, returns
    ``mixed_or_unknown_trace`` — promotion is blocked at that state.
    """

    text = _searchable_text(meta)
    matches: dict = {}
    for trace_type, kws in CLASSIFICATION_KEYWORDS.items():
        hits = [kw for kw in kws if kw in text]
        if hits:
            matches[trace_type] = hits

    # Trace-type priority (highest trust first; ``mixed`` is the fallback).
    priority = [
        "telemetry_trace",
        "cluster_scheduler_trace",
        "latency_benchmark_trace",
        "kernel_profile_trace",
        "cache_residency_trace",
        "request_shape_trace",
    ]
    chosen = "mixed_or_unknown_trace"
    for tt in priority:
        if tt in matches:
            chosen = tt
            break
    return {
        "trace_type": chosen,
        "evidence": matches,
    }


def available_signals(meta: HFDatasetMeta) -> list[str]:
    """List of ``TARGET_SIGNALS`` whose keyword appears in the metadata."""

    text = _searchable_text(meta)
    out: list[str] = []
    for sig in TARGET_SIGNALS:
        token = sig.replace("_", " ")
        if sig in text or token in text:
            out.append(sig)
    return out


def missing_signals(meta: HFDatasetMeta, present: Iterable[str]) -> list[str]:
    present_set = set(present)
    return [s for s in TARGET_SIGNALS if s not in present_set]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# Signal-density weights for the frontier_value_score. Telemetry-grade
# signals dominate; conversation-shape gets a low ceiling. The cap is 5.
_SIGNAL_WEIGHTS = {
    # Tier 2 telemetry signals.
    "queue_wait": 2.0, "queue_depth": 2.0, "timeout": 1.5, "sla": 1.5,
    "gpu_utilization": 1.5, "gpu_memory": 1.0, "memory_pressure": 1.0,
    "autoscaling": 1.0, "replica_count": 1.0,
    # Tier 4 benchmark signals.
    "ttft": 1.0, "tpot": 1.0, "itl": 0.5, "e2e_latency": 1.0,
    "latency_p95": 0.5, "latency_p99": 0.5, "throughput": 0.5,
    "request_throughput": 0.5, "token_throughput": 0.5,
    "concurrency": 0.5, "batch_size": 0.5,
    # Tier 4-5 metadata.
    "gpu_type": 0.25, "vllm": 0.25, "sglang": 0.25,
    "triton": 0.25, "ray_serve": 0.25, "prometheus": 0.25, "dcgm": 0.25,
    "kubernetes": 0.25,
    # Tier 5 request-shape.
    "prompt_tokens": 0.25, "output_tokens": 0.25,
    "sequence_length": 0.25, "cold_start": 0.5, "prefix_cache": 0.5,
    "cache_hit": 0.5,
}


_LICENSE_PERMISSIVE = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause",
    "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "odc-by",
}


def _ingestion_feasibility(meta: HFDatasetMeta) -> int:
    if meta.gated is True:
        return 1
    if meta.private is True:
        return 1
    # Bias toward bounded / parquet-shaped sources.
    sizes = " ".join(meta.size_categories).lower()
    if any(x in sizes for x in ("1B<", "10b<", "100b<")):
        return 2
    if any(x in sizes for x in ("100m<n<1b", "1m<n<10m", "10m<n<100m")):
        return 3
    if any(x in sizes for x in ("1k<n<10k", "10k<n<100k", "100k<n<1m", "n<1k")):
        return 5
    return 3


def _frontier_value(present_signals: list[str], trace_type: str) -> int:
    raw = sum(_SIGNAL_WEIGHTS.get(s, 0.0) for s in present_signals)
    # Telemetry trace_type gets a small boost; conversation-shape capped at 3.
    if trace_type == "telemetry_trace":
        raw += 1.0
    if trace_type == "request_shape_trace":
        raw = min(raw, 3.0)
    return max(1, min(5, int(round(raw))))


def _schema_quality(meta: HFDatasetMeta) -> int:
    score = 1
    if meta.feature_names:
        score += 2
    if meta.splits:
        score += 1
    if meta.configs:
        score += 1
    return min(5, score)


def _production_similarity(trace_type: str, present_signals: list[str]) -> int:
    if trace_type == "telemetry_trace":
        return 5
    if trace_type == "cluster_scheduler_trace":
        return 4
    if trace_type == "latency_benchmark_trace":
        # Real serving engine + measured latency closer to production than
        # synthetic conversations.
        boost = 0
        if "vllm" in present_signals or "sglang" in present_signals:
            boost += 1
        if "ttft" in present_signals and "tpot" in present_signals:
            boost += 1
        return min(5, 2 + boost)
    if trace_type == "kernel_profile_trace":
        return 2
    if trace_type == "cache_residency_trace":
        return 3
    if trace_type == "request_shape_trace":
        return 2
    return 1


def score_dataset(
    meta: HFDatasetMeta,
    classification: dict,
    matched_keywords: list[str],
) -> dict:
    """Compute the candidate scoring tuple.

    Returns a dict containing exactly the score fields specified by the
    mission prompt:

    - ``ingestion_feasibility_score`` (1-5)
    - ``frontier_value_score`` (1-5)
    - ``schema_quality_score`` (1-5)
    - ``production_similarity_score`` (1-5)
    - ``overall_priority_score`` (weighted mean, 1.0-5.0)
    - ``recommended_action`` (string from the documented enum)

    Recommended-action rules:

    - ``gated_blocked`` overrides everything when ``meta.gated is True``.
    - ``too_large_unbounded`` when no bounded size category and size > 100M
      rows-ish.
    - ``unknown_schema`` when no features at all + no siblings.
    - ``reject_low_value`` when ``frontier_value_score == 1``.
    - ``ingest_now_bounded`` when overall >= 3.5.
    - ``inspect_manually`` otherwise.
    """

    trace_type = classification["trace_type"]
    sigs = available_signals(meta)
    feas = _ingestion_feasibility(meta)
    fv = _frontier_value(sigs, trace_type)
    sq = _schema_quality(meta)
    prod = _production_similarity(trace_type, sigs)

    overall = (
        0.35 * fv + 0.25 * feas + 0.20 * prod + 0.20 * sq
    )
    overall = round(overall, 3)

    action = "inspect_manually"
    if meta.gated is True:
        action = "gated_blocked"
    elif not meta.feature_names and not meta.siblings:
        action = "unknown_schema"
    elif fv <= 1:
        action = "reject_low_value"
    elif overall >= 3.5 and feas >= 3:
        action = "ingest_now_bounded"

    return {
        "trust_level": CANONICAL_TRACE_TYPE_TO_TRUST_TIER.get(
            trace_type, "tier_6_synthetic_benchmark_data"),
        "matched_keywords": list(matched_keywords),
        "candidate_trace_type": trace_type,
        "classification_evidence": classification["evidence"],
        "available_signals": sigs,
        "missing_signals": missing_signals(meta, sigs),
        "ingestion_feasibility_score": feas,
        "frontier_value_score": fv,
        "schema_quality_score": sq,
        "production_similarity_score": prod,
        "overall_priority_score": overall,
        "recommended_action": action,
    }


# ---------------------------------------------------------------------------
# Aurelius use-case routing
# ---------------------------------------------------------------------------


# Each canonical trace type maps to:
# - aurelius_use_case: short string the registry stores.
# - not_recommended_uses: explicit list of what NOT to do with this dataset.
# These mirror the mission spec's routing rules.

AURELIUS_USE_CASES = {
    "request_shape_trace": {
        "use": (
            "Workload replay shape; eval/batch frontier scenarios; request "
            "mix modelling; prompt/output distribution priors."
        ),
        "not_recommended": [
            "Production latency calibration",
            "TTFT / TPOT inference unless measured latency exists",
            "Dynamic frontier calibration",
        ],
    },
    "latency_benchmark_trace": {
        "use": (
            "Performance-surface priors; throughput/latency risk priors; "
            "batch-size + concurrency priors; model/GPU/engine comparison."
        ),
        "not_recommended": [
            "Arrival scheduling unless timestamps exist",
            "Production telemetry substitution",
            "Real queue-wait calibration unless arrival trace exists",
        ],
    },
    "kernel_profile_trace": {
        "use": (
            "Low-level GPU performance priors; model cost estimation; "
            "kernel/memory bottleneck priors."
        ),
        "not_recommended": [
            "Request-level scheduler backtests",
            "Production SLA / queue calibration",
        ],
    },
    "cluster_scheduler_trace": {
        "use": (
            "Constraint-aware scheduler backtests; training/ETL/batch packing; "
            "queue-wait + fragmentation + gang-scheduling priors; placement + "
            "deferral evaluations."
        ),
        "not_recommended": [
            "LLM TTFT / TPOT inference unless present",
            "Direct serving-physics replay",
        ],
    },
    "cache_residency_trace": {
        "use": (
            "Cache / routing / residency / prewarming evaluations; "
            "cold-start risk priors; affinity routing; cache-hit sensitivity."
        ),
        "not_recommended": [
            "Full serving telemetry substitution",
            "Production latency calibration unless latency exists",
        ],
    },
    "telemetry_trace": {
        "use": (
            "Dynamic frontier calibration; shadow evaluation; constraint-aware "
            "scheduler validation; SLA / timeout / queue / replica / GPU-util "
            "risk calibration."
        ),
        "not_recommended": [
            "Treating non-pilot telemetry as Tier 1 pilot calibration",
        ],
    },
    "mixed_or_unknown_trace": {
        "use": "Not eligible — requires manual classification before any use.",
        "not_recommended": [
            "All Aurelius evaluators",
            "Promotion to any canonical corpus until classified",
        ],
    },
}


def aurelius_use_case(trace_type: str) -> dict:
    return AURELIUS_USE_CASES.get(trace_type, AURELIUS_USE_CASES["mixed_or_unknown_trace"])


# ---------------------------------------------------------------------------
# Discovery driver
# ---------------------------------------------------------------------------


def _build_candidate(
    meta: HFDatasetMeta,
    matched_keywords: list[str],
    discovery_timestamp_s: float,
) -> dict:
    classification = classify_dataset(meta)
    scores = score_dataset(meta, classification, matched_keywords)
    use_case = aurelius_use_case(classification["trace_type"])
    return {
        "dataset_id": meta.dataset_id,
        "dataset_url": meta.dataset_url,
        "gated_status": (
            "gated" if meta.gated is True
            else ("private" if meta.private is True else "public")
        ),
        "license": meta.license,
        "estimated_size": list(meta.size_categories),
        "available_splits": [
            {"config": c, "split": s, "num_examples": n}
            for (c, s, n) in meta.splits
        ],
        "schema_available": bool(meta.feature_names),
        "feature_names": list(meta.feature_names),
        "configs": list(meta.configs),
        "downloads": meta.downloads,
        "likes": meta.likes,
        "last_modified": meta.last_modified,
        "aurelius_use_case": use_case["use"],
        "not_recommended_uses": use_case["not_recommended"],
        "discovery_timestamp_s": discovery_timestamp_s,
        **scores,
    }


def discover(
    client,
    query_groups: Optional[dict] = None,
    *,
    extra_seed_ids: Optional[list] = None,
    max_results_per_query: int = DEFAULT_MAX_RESULTS_PER_QUERY,
    now: Optional[float] = None,
) -> list[dict]:
    """Run all keyword searches, merge by dataset id, classify + score.

    The returned list is sorted by ``overall_priority_score`` descending,
    then ``dataset_id`` ascending for determinism. Each candidate carries
    ``matched_keywords`` accumulated across every group that hit it.
    """

    if query_groups is None:
        query_groups = DEFAULT_QUERY_GROUPS

    if now is None:
        now = time.time()

    accumulated: dict = {}
    for group_name, queries in query_groups.items():
        for q in queries:
            for entry in client.search(q, limit=max_results_per_query):
                if not isinstance(entry, dict) or "id" not in entry:
                    continue
                ds_id = str(entry["id"])
                if ds_id not in accumulated:
                    accumulated[ds_id] = {
                        "raw_listing": entry,
                        "matched_keywords": [],
                    }
                kw_tag = f"{group_name}::{q}"
                if kw_tag not in accumulated[ds_id]["matched_keywords"]:
                    accumulated[ds_id]["matched_keywords"].append(kw_tag)

    for seed_id in (extra_seed_ids or []):
        if seed_id not in accumulated:
            accumulated[seed_id] = {
                "raw_listing": {"id": seed_id},
                "matched_keywords": ["seed::known_high_priority"],
            }

    out: list[dict] = []
    for ds_id, info in accumulated.items():
        detail = client.get(ds_id) or info["raw_listing"]
        if not isinstance(detail, dict) or "id" not in detail:
            continue
        try:
            meta = parse_hf_metadata(detail)
        except ValueError as e:
            logger.warning("skipping %s: %s", ds_id, e)
            continue
        out.append(_build_candidate(meta, info["matched_keywords"], now))

    out.sort(key=lambda c: (-float(c["overall_priority_score"]), c["dataset_id"]))
    return out
