"""Day-ahead → real-time spread risk model for DA-plan / RT-settle scheduling.

In the DA-plan / RT-settle world the optimizer plans against day-ahead (DA)
prices but the customer is billed at real-time (RT). DA is a biased, noisy
predictor of RT: some (region, hour-of-day) slots systematically settle higher
in RT (net-load evening ramps, scarcity events), and some carry large *upside*
spread risk — RT occasionally spikes far above DA even when DA looks cheap.

A planner that minimizes raw DA cost will confidently place load into low-DA
hours that blow out in RT, sometimes losing to a naive reactive baseline. This
model corrects that by turning the DA planning price into a risk-adjusted
estimate of RT:

    adjusted_price(region, ts) = DA(region, ts)
                               + median_spread(region, hour_of_day)        # debias toward E[RT]
                               + lambda * upside_spread(region, hour_of_day) # spike penalty

where spread = RT - DA over the *training* window only (no eval leakage), and
upside_spread is the gap between a high quantile and the median of the spread
(how badly RT tends to overshoot DA in that slot). lambda controls risk
aversion: 0 = debias only, higher = avoid spike-prone slots more aggressively.

The optimizer consumes adjusted prices for *decisions* only; schedules are still
scored on actual realized RT. Because every solver path (greedy, migration,
replan, MILP) reads the same price map, feeding it adjusted prices also gates
migration: relocating a job into a spike-prone region is penalized up front.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class SpreadRiskModel:
    """Learns DA→RT spread debias + upside risk, by hour-of-day and DA level.

    Two upside-risk signals are learned per region from the training window:
      * hour-of-day: some slots (e.g. evening net-load ramps) overshoot RT;
      * DA price level: when DA is already high (scarcity), RT tends to spike
        even harder — a signal hour-of-day alone misses.
    The applied upside penalty is the MAX of the two, so either indicator can
    flag a slot as spike-prone. The central debias term (toward expected RT)
    uses the hour-of-day median spread.
    """

    def __init__(
        self,
        risk_lambda: float = 1.0,
        upside_quantile: float = 0.9,
        min_samples: int = 24,
        n_da_bins: int = 5,
    ) -> None:
        self.risk_lambda = risk_lambda
        self.upside_quantile = upside_quantile
        self.min_samples = min_samples
        self.n_da_bins = n_da_bins
        # hour-of-day signal
        self._hod_median: dict[tuple[str, int], float] = {}
        self._hod_upside: dict[tuple[str, int], float] = {}
        # region-level fallback
        self._region_median: dict[str, float] = {}
        self._region_upside: dict[str, float] = {}
        # DA-price-level signal: per-region quantile edges + per-bin upside
        self._da_edges: dict[str, list[float]] = {}
        self._da_upside: dict[tuple[str, int], float] = {}
        self._fitted = False

    @staticmethod
    def _median_upside(vals: list[float], q: float) -> tuple[float, float]:
        v = sorted(vals)
        median = _quantile(v, 0.5)
        upside = max(0.0, _quantile(v, q) - median)
        return median, upside

    def _da_bin(self, region: str, da_price: float) -> int:
        edges = self._da_edges.get(region)
        if not edges:
            return 0
        b = 0
        for e in edges:
            if da_price >= e:
                b += 1
            else:
                break
        return b

    def fit(
        self,
        plan_data: dict[str, dict[datetime, float]],
        settle_data: dict[str, dict[datetime, float]],
    ) -> "SpreadRiskModel":
        """Fit from aligned DA (plan) and RT (settle) training price maps.

        Only timestamps present in BOTH maps for a region contribute a spread
        sample. Buckets with fewer than ``min_samples`` fall back to the
        region-level adjustment, then to zero.
        """
        hod_spreads: dict[tuple[str, int], list[float]] = defaultdict(list)
        region_spreads: dict[str, list[float]] = defaultdict(list)
        region_da: dict[str, list[float]] = defaultdict(list)
        # collected as (region, da_price, spread) to bin after edges are known
        paired: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for region, plan_map in plan_data.items():
            settle_map = settle_data.get(region)
            if not settle_map:
                continue
            for ts, da_price in plan_map.items():
                rt_price = settle_map.get(ts)
                if rt_price is None:
                    continue
                spread = rt_price - da_price
                hod_spreads[(region, ts.hour)].append(spread)
                region_spreads[region].append(spread)
                region_da[region].append(da_price)
                paired[region].append((da_price, spread))

        for region, vals in region_spreads.items():
            self._region_median[region], self._region_upside[region] = self._median_upside(
                vals, self.upside_quantile
            )

        for key, vals in hod_spreads.items():
            if len(vals) < self.min_samples:
                continue
            self._hod_median[key], self._hod_upside[key] = self._median_upside(
                vals, self.upside_quantile
            )

        # DA-price-level bins: per-region quantile edges, per-bin upside risk.
        for region, da_vals in region_da.items():
            if len(da_vals) < self.min_samples * self.n_da_bins:
                continue
            da_sorted = sorted(da_vals)
            edges = [
                _quantile(da_sorted, i / self.n_da_bins)
                for i in range(1, self.n_da_bins)
            ]
            # strictly-increasing edges only (skip if DA is near-constant)
            if any(edges[i] <= edges[i - 1] for i in range(1, len(edges))):
                continue
            self._da_edges[region] = edges
            bin_spreads: dict[int, list[float]] = defaultdict(list)
            for da_price, spread in paired[region]:
                bin_spreads[self._da_bin(region, da_price)].append(spread)
            # Keep the DA-level signal as a TARGETED scarcity indicator: only the
            # top DA bin contributes, and only the *excess* upside beyond the
            # region's baseline. This protects against high-DA scarcity (DA spikes
            # foreshadowing RT spikes) without broadly inflating normal-price
            # hours, which would over-penalize calm regimes.
            top_bin = self.n_da_bins - 1
            vals = bin_spreads.get(top_bin, [])
            if len(vals) >= self.min_samples:
                _, up = self._median_upside(vals, self.upside_quantile)
                # Only register the DA-level signal when the top DA bin is
                # genuinely riskier than the region's typical upside; its
                # absolute upside then competes with the hour-of-day signal via
                # max() in adjustment().
                if up > self._region_upside.get(region, 0.0):
                    self._da_upside[(region, top_bin)] = up

        self._fitted = bool(self._region_median)
        logger.info(
            "SpreadRiskModel fit: %d hour buckets, %d DA-level buckets, %d regions, lambda=%.2f",
            len(self._hod_median), len(self._da_upside), len(self._region_median),
            self.risk_lambda,
        )
        return self

    def adjustment(self, region: str, ts: datetime, da_price: Optional[float] = None) -> float:
        """Additive $/MWh adjustment for a (region, timestamp[, DA price])."""
        if not self._fitted:
            return 0.0
        median = self._hod_median.get((region, ts.hour))
        if median is None:
            median = self._region_median.get(region, 0.0)
        up_hod = self._hod_upside.get((region, ts.hour))
        if up_hod is None:
            up_hod = self._region_upside.get(region, 0.0)
        up_da = 0.0
        if da_price is not None and region in self._da_edges:
            up_da = self._da_upside.get((region, self._da_bin(region, da_price)), 0.0)
        upside = max(up_hod, up_da)
        return median + self.risk_lambda * upside

    def adjust_price_map(
        self,
        price_map: dict[str, dict[datetime, float]],
    ) -> dict[str, dict[datetime, float]]:
        """Return a risk-adjusted copy of a {region: {ts: price}} planning map.

        The adjustment debiases each slot toward expected RT and adds a
        lambda-weighted upside-spike penalty (the larger of the hour-of-day and
        DA-price-level risk signals). Scoring is always done on actual RT
        outside this function; this only shapes the optimizer's decisions.
        """
        if not self._fitted:
            return price_map
        out: dict[str, dict[datetime, float]] = {}
        for region, ts_map in price_map.items():
            out[region] = {
                ts: price + self.adjustment(region, ts, price)
                for ts, price in ts_map.items()
            }
        return out
