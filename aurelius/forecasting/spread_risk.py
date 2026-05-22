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
    """Learns per-(region, hour-of-day) DA→RT spread debias + upside risk.

    Fit on aligned training-window DA and RT price maps; apply to any planning
    price map to obtain a risk-adjusted RT estimate for the optimizer.
    """

    def __init__(
        self,
        risk_lambda: float = 1.0,
        upside_quantile: float = 0.9,
        min_samples: int = 24,
    ) -> None:
        self.risk_lambda = risk_lambda
        self.upside_quantile = upside_quantile
        self.min_samples = min_samples
        # (region, hour_of_day) -> additive adjustment in $/MWh
        self._adj: dict[tuple[str, int], float] = {}
        # region -> fallback adjustment (median over that region's hours)
        self._region_adj: dict[str, float] = {}
        self._fitted = False

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
        spreads: dict[tuple[str, int], list[float]] = defaultdict(list)
        region_spreads: dict[str, list[float]] = defaultdict(list)

        for region, plan_map in plan_data.items():
            settle_map = settle_data.get(region)
            if not settle_map:
                continue
            for ts, da_price in plan_map.items():
                rt_price = settle_map.get(ts)
                if rt_price is None:
                    continue
                spread = rt_price - da_price
                spreads[(region, ts.hour)].append(spread)
                region_spreads[region].append(spread)

        for region, vals in region_spreads.items():
            vals_sorted = sorted(vals)
            median = _quantile(vals_sorted, 0.5)
            upside = max(0.0, _quantile(vals_sorted, self.upside_quantile) - median)
            self._region_adj[region] = median + self.risk_lambda * upside

        for key, vals in spreads.items():
            if len(vals) < self.min_samples:
                continue
            vals_sorted = sorted(vals)
            median = _quantile(vals_sorted, 0.5)
            upside = max(0.0, _quantile(vals_sorted, self.upside_quantile) - median)
            self._adj[key] = median + self.risk_lambda * upside

        self._fitted = bool(self._region_adj)
        n_buckets = len(self._adj)
        logger.info(
            "SpreadRiskModel fit: %d region-hour buckets, %d regions, lambda=%.2f",
            n_buckets, len(self._region_adj), self.risk_lambda,
        )
        return self

    def adjustment(self, region: str, ts: datetime) -> float:
        """Additive $/MWh adjustment for a (region, timestamp)."""
        if not self._fitted:
            return 0.0
        a = self._adj.get((region, ts.hour))
        if a is None:
            a = self._region_adj.get(region, 0.0)
        return a

    def adjust_price_map(
        self,
        price_map: dict[str, dict[datetime, float]],
    ) -> dict[str, dict[datetime, float]]:
        """Return a risk-adjusted copy of a {region: {ts: price}} planning map.

        Adjusted prices are never pushed below the original DA price by less than
        the additive term; negative market prices (e.g. CAISO oversupply) are
        preserved, but the adjustment only ever raises a slot's effective price
        when the model expects RT to overshoot (upside risk >= 0 with positive
        lambda). A debias term may lower it where RT runs below DA.
        """
        if not self._fitted:
            return price_map
        out: dict[str, dict[datetime, float]] = {}
        for region, ts_map in price_map.items():
            out[region] = {
                ts: price + self.adjustment(region, ts)
                for ts, price in ts_map.items()
            }
        return out
