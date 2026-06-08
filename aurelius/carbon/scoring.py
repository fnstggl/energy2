"""Carbon-aware candidate scoring.

Phase-5 design decision (documented in code per the brief)
----------------------------------------------------------
A candidate's cost is in USD and its emissions are in kgCO2. **Adding dollars and
kilograms directly is meaningless** unless an explicit carbon price converts one
to the other. Aurelius therefore supports two well-defined modes and NEVER mixes
units implicitly:

* **Option A — normalize (DEFAULT).** Normalize cost and emissions to a common
  [0, 1] scale across the candidate set, then combine with explicit weights::

      score = alpha*norm_cost + beta*norm_emissions + gamma*risk + delta*migration_penalty

  Chosen as the default because it fits Aurelius's existing weighted-objective
  architecture (``OptimizationConfig.alpha/beta/gamma/delta``) and requires no
  external price assumption. Lower score is better.

* **Option B — explicit carbon price.** When the caller provides a
  ``carbon_price_usd_per_tonne``, emissions are converted to dollars and added to
  the energy/risk/migration dollars::

      carbon_cost_usd = emissions_kgco2 / 1000 * carbon_price_usd_per_tonne
      score = energy_cost_usd + carbon_cost_usd + risk_cost_usd + migration_cost_usd

Both are pure functions; the candidate evaluator (``candidate.py``) selects the
mode based on whether a carbon price was supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def carbon_cost_usd(emissions_kgco2: float, carbon_price_usd_per_tonne: float) -> float:
    """Option B conversion: kgCO2 -> USD via an explicit carbon price.

    carbon_cost_usd = emissions_kgco2 / 1000 (tonnes) * price_usd_per_tonne
    """
    return (float(emissions_kgco2) / 1000.0) * float(carbon_price_usd_per_tonne)


@dataclass
class ScoreWeights:
    """Weights for Option A normalized scoring (mirrors OptimizationConfig)."""

    alpha: float = 1.0    # cost
    beta: float = 0.3     # carbon
    gamma: float = 0.05   # risk
    delta: float = 1.0    # migration penalty


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Constant input -> all zeros (no preference)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@dataclass
class CandidateScoreInputs:
    """Raw, un-normalized per-candidate quantities."""

    cost_usd: float
    emissions_kgco2: float
    risk: float = 0.0
    migration_penalty: float = 0.0


def score_candidates_normalized(
    inputs: list[CandidateScoreInputs],
    weights: Optional[ScoreWeights] = None,
) -> list[float]:
    """Option A: normalize cost & emissions across the set, then weight-combine.

    Returns a parallel list of scores (lower is better). Risk and migration
    penalty are assumed already on a comparable scale and are added directly with
    their weights (matching the existing objective's treatment).
    """
    w = weights or ScoreWeights()
    norm_cost = _normalize([i.cost_usd for i in inputs])
    norm_emis = _normalize([i.emissions_kgco2 for i in inputs])
    scores = []
    for i, nc, ne in zip(inputs, norm_cost, norm_emis):
        scores.append(
            w.alpha * nc
            + w.beta * ne
            + w.gamma * i.risk
            + w.delta * i.migration_penalty
        )
    return scores


def score_candidates_carbon_priced(
    inputs: list[CandidateScoreInputs],
    carbon_price_usd_per_tonne: float,
) -> list[float]:
    """Option B: convert emissions to USD via carbon price, sum dollars.

    Returns a parallel list of dollar scores (lower is better).
    """
    scores = []
    for i in inputs:
        scores.append(
            i.cost_usd
            + carbon_cost_usd(i.emissions_kgco2, carbon_price_usd_per_tonne)
            + i.risk
            + i.migration_penalty
        )
    return scores


__all__ = [
    "carbon_cost_usd",
    "ScoreWeights",
    "CandidateScoreInputs",
    "score_candidates_normalized",
    "score_candidates_carbon_priced",
]
