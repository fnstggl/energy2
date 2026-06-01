"""TTFT p50 shadow prior adapter for the goodput/$ placement/routing path.

This module is a **thin adapter**, not a parallel economic scorer. It
exposes:

- ``TTFTShadowPrior`` — a per-``(model_size, gpu_type, prompt_token_bin)``
  median-TTFT lookup, fit from CARA train_flat. It is the *shadow* form
  of the calibrated TTFT p50 model (which is ``shadow_ready`` per
  ``docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md``).
- ``refine_service_time_proxy_s(ctx, *, model_size, gpu_type, prompt_tokens, prior)``
  — returns a *new* ``SafetyContext`` whose ``service_time_proxy_s`` is
  replaced by ``max(static_proxy, predicted_ttft_p50)``. The returned
  context is a copy; the input context is not mutated.

Honesty rules (binding):

- Prior is shadow / logging-only. The default is to NOT integrate; the
  refinement happens only when the caller explicitly opts in
  (``apply_to_scorer = True``).
- Only the p50 prior is exposed. p95/p99 ML tails are not used for control
  per the mission spec; tail safety stays on the deterministic Erlang-C
  baseline.
- No imports from the production scheduler, frontier controllers, or
  any executor module (the test
  ``test_ttft_shadow_prior_module_has_no_controller_imports`` pins this).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Pre-registered prompt-token bin boundaries (shared with cara_latency_features).
_PROMPT_BINS = [(0, 50), (50, 200), (200, 800), (800, 3200), (3200, 1_000_000)]


def _bin_label(v, bins=_PROMPT_BINS) -> str:
    if v is None:
        return "missing"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in bins:
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f">={bins[-1][1]}"


def _derive_model_size(instance_type: Optional[str]) -> Optional[str]:
    if not instance_type:
        return None
    head = str(instance_type).split("_", 1)[0]
    parts = head.split("-")
    if len(parts) < 2:
        return None
    return parts[-1].lower()


def _derive_gpu_type(instance_type: Optional[str]) -> Optional[str]:
    if not instance_type:
        return None
    parts = str(instance_type).split("_", 1)
    if len(parts) != 2:
        return None
    return parts[1].lower()


@dataclass
class TTFTShadowPrior:
    """Per-(model_size, gpu_type, prompt_token_bin) median-TTFT prior.

    Fit deterministically by ``fit_from_rows``: each subgroup's prior is
    the empirical p50 of ``actual_ttft_s`` on the fit rows. ``predict()``
    returns the median for the matching subgroup, falling back through
    (model_size, gpu_type) → (gpu_type,) → global median if a subgroup is
    unseen.

    Records ``subgroup_counts`` for the eval script's INSUFFICIENT_SAMPLE
    flagging.
    """

    table: dict = field(default_factory=dict)            # (m, g, bin) -> p50
    by_model_gpu: dict = field(default_factory=dict)     # (m, g) -> p50
    by_gpu: dict = field(default_factory=dict)           # (g,) -> p50
    global_p50: float = float("nan")
    subgroup_counts: dict = field(default_factory=dict)
    fit_row_count: int = 0
    model_version: str = "cara_ttft_p50_shadow_prior_v1"

    def fit_from_rows(self, rows: Iterable[dict]) -> "TTFTShadowPrior":
        ttft = []
        groups: dict = {}
        groups_mg: dict = {}
        groups_g: dict = {}
        for r in rows:
            t = r.get("actual_ttft_s")
            it = r.get("instance_type")
            pt = r.get("num_prompt_tokens")
            if t is None or it is None:
                continue
            try:
                t = float(t)
            except (TypeError, ValueError):
                continue
            m = _derive_model_size(it)
            g = _derive_gpu_type(it)
            if m is None or g is None:
                continue
            bin_label = _bin_label(pt)
            key = (m, g, bin_label)
            groups.setdefault(key, []).append(t)
            groups_mg.setdefault((m, g), []).append(t)
            groups_g.setdefault((g,), []).append(t)
            ttft.append(t)

        self.table = {
            "|".join(k): float(np.median(v))
            for k, v in groups.items() if v
        }
        self.subgroup_counts = {
            "|".join(k): int(len(v)) for k, v in groups.items()
        }
        self.by_model_gpu = {
            "|".join(k): float(np.median(v))
            for k, v in groups_mg.items() if v
        }
        self.by_gpu = {
            k[0]: float(np.median(v)) for k, v in groups_g.items() if v
        }
        self.global_p50 = float(np.median(ttft)) if ttft else float("nan")
        self.fit_row_count = len(ttft)
        return self

    def predict(self, *, model_size: Optional[str], gpu_type: Optional[str],
                prompt_tokens: Optional[float]) -> Optional[float]:
        if model_size is None or gpu_type is None:
            return None
        key3 = f"{model_size}|{gpu_type}|{_bin_label(prompt_tokens)}"
        v = self.table.get(key3)
        if v is not None:
            return v
        key2 = f"{model_size}|{gpu_type}"
        v = self.by_model_gpu.get(key2)
        if v is not None:
            return v
        v = self.by_gpu.get(gpu_type)
        if v is not None:
            return v
        if np.isnan(self.global_p50):
            return None
        return self.global_p50

    def subgroup_n(self, *, model_size, gpu_type, prompt_tokens) -> int:
        key = f"{model_size}|{gpu_type}|{_bin_label(prompt_tokens)}"
        return int(self.subgroup_counts.get(key, 0))

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "fit_row_count": self.fit_row_count,
            "global_p50_s": self.global_p50,
            "by_gpu": self.by_gpu,
            "by_model_gpu": self.by_model_gpu,
            "subgroup_counts": self.subgroup_counts,
            "table_p50_s": self.table,
        }


def refine_service_time_proxy_s(
    ctx, *, model_size: Optional[str], gpu_type: Optional[str],
    prompt_tokens: Optional[float], prior: TTFTShadowPrior,
    apply_to_scorer: bool = False,
    min_subgroup_rows: int = 100,
):
    """Return ``(refined_ctx, refinement_record)``.

    ``apply_to_scorer = False`` (default) returns the input context
    unchanged and records what the prior *would have* changed. This is
    the binding "shadow / logging-only" path.

    ``apply_to_scorer = True`` returns a copy of ``ctx`` with
    ``service_time_proxy_s = max(ctx.service_time_proxy_s,
    predicted_ttft_p50)``. The MAX clamp ensures the prior can only
    *widen* the latency estimate; it never under-predicts vs the static
    proxy (binding safety floor).
    """
    pred = prior.predict(
        model_size=model_size, gpu_type=gpu_type, prompt_tokens=prompt_tokens,
    )
    n = prior.subgroup_n(
        model_size=model_size, gpu_type=gpu_type, prompt_tokens=prompt_tokens,
    )
    record = {
        "model_size": model_size,
        "gpu_type": gpu_type,
        "prompt_token_bin": _bin_label(prompt_tokens),
        "predicted_ttft_p50_s": pred,
        "subgroup_n": n,
        "subgroup_insufficient": n < min_subgroup_rows,
        "static_proxy_s": getattr(ctx, "service_time_proxy_s", None),
        "refined_proxy_s": getattr(ctx, "service_time_proxy_s", None),
        "applied_to_scorer": False,
        "fallback_to_static": False,
    }
    if pred is None or np.isnan(pred):
        record["fallback_to_static"] = True
        return ctx, record
    refined = max(getattr(ctx, "service_time_proxy_s", 0.0), float(pred))
    record["refined_proxy_s"] = refined
    if not apply_to_scorer:
        return ctx, record
    new_ctx = dataclasses.replace(ctx, service_time_proxy_s=refined)
    record["applied_to_scorer"] = True
    return new_ctx, record


def save_prior(prior: TTFTShadowPrior, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(prior.to_dict(), indent=2, sort_keys=True))


def load_prior(path) -> TTFTShadowPrior:
    payload = json.loads(Path(path).read_text())
    p = TTFTShadowPrior(
        table=payload.get("table_p50_s") or {},
        by_model_gpu=payload.get("by_model_gpu") or {},
        by_gpu=payload.get("by_gpu") or {},
        global_p50=float(payload.get("global_p50_s") or float("nan")),
        subgroup_counts=payload.get("subgroup_counts") or {},
        fit_row_count=int(payload.get("fit_row_count") or 0),
        model_version=str(payload.get("model_version")
                          or "cara_ttft_p50_shadow_prior_v1"),
    )
    return p
