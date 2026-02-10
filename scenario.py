from __future__ import annotations

# scenario.py
"""
Shock/Scenario engine for the RL market experiment.

Usage:
    from scenario import ShockEngine
    engine = ShockEngine(scenario, spec_costs=spec.costs)
    ...
    for it in range(T):
        engine.tick(ep, it, market, farmers)  # call once per iteration BEFORE agents act
        ...  # rest of the loop
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config import MarketId
from farmer import INFLATION_RULES


@dataclass
class ShockEngine:
    scenario: List[Dict[str, Any]] = field(default_factory=list)
    # Baseline costs (per market), used if you want to do cost inflation by *setting* costs deterministically.
    spec_costs: Optional[Dict[MarketId, float]] = None
    # Internal rules assembled from scenario
    inflation_rules: List[Dict[str, Any]] = field(default_factory=list)
    _applied_once: set[Tuple[int, int]] = field(default_factory=set)

    def _matches_when(self, ev_when: Dict[str, Any], ep: int, it: int) -> bool:
        if ev_when is None:
            return True
        if ev_when.get("episode") is not None and ev_when.get("episode") != ep:
            return False
        if ev_when.get("iter") is not None and ev_when.get("iter") != it:
            return False
        if ev_when.get("from_iter") is not None and it < ev_when["from_iter"]:
            return False
        if ev_when.get("to_iter") is not None and it > ev_when["to_iter"]:
            return False
        return True

    def _apply_price_bump(self, market, payload: Dict[str, float]):
        if "all" in payload:
            delta = float(payload["all"])
            for m in market.spec.markets:
                market.price_add[m] += delta
        for k, v in payload.items():
            if k == "all": continue
            market.price_add[int(k)] += float(v)

    def _apply_technology(self, market, payload: Dict[str, float]):
        if "all" in payload:
            val = float(payload["all"])
            for m in market.spec.markets:
                market.n_multiplier[m] = val
        for k, v in payload.items():
            if k == "all": continue
            market.n_multiplier[int(k)] = float(v)

    def _apply_entrants(self, market, payload: Dict[str, int]):
        if "all" in payload:
            add = int(payload["all"])
            # equal distribution
            per = add // max(1, len(market.spec.markets))
            for m in market.spec.markets:
                market.external_counts[m] += per
        for k, v in payload.items():
            if k == "all": continue
            market.external_counts[int(k)] += int(v)

    def _apply_transaction_cost(self, farmers, payload: Dict[str, float]):
        if "set" in payload:
            val = float(payload["set"])
            for f in farmers:
                f.switch_cost = val
        if "add" in payload:
            delta = float(payload["add"])
            for f in farmers:
                f.switch_cost += delta

    def _apply_cost_shock(self, farmers, payload: Dict[str, float]):
        # DIRECTLY modifies costs per market (same for all farmers)
        def bump(m_id: int, delta: float):
            for f in farmers:
                f.costs[m_id] = float(f.costs[m_id]) + float(delta)
        if "all" in payload:
            for m in farmers[0].costs.keys():
                bump(m, float(payload["all"]))
        for k, v in payload.items():
            if k == "all": continue
            bump(int(k), float(v))

    def _apply_inflation_rule(self, payload: Dict[str, Any]):
        """
        Set linear inflation rules used by Farmer.calculate_profits.

        payload example:
        {"market": 3, "start": 200, "rate": 0.0015}
        {"market": None, "start": 400, "rate": 0.0005}
        """
        market = payload.get("market", None)
        start  = int(payload.get("start", 0))
        rate   = float(payload.get("rate", 0.0))

        if market is None:
            # Global rule for all markets
            INFLATION_RULES[None] = {"start": start, "rate": rate}
        else:
            m = int(market)
            INFLATION_RULES[m] = {"start": start, "rate": rate}

    def tick(self, ep: int, it: int, market, farmers):
        # 1) process events scheduled exactly at this tick
        for idx, ev in enumerate(self.scenario):
            if not self._matches_when(ev.get("when", {}), ep, it):
                continue
            # Apply-once for additive shocks to avoid repeated accumulation
            ev_key = (idx, ep)
            if ev_key in self._applied_once:
                # still allow idempotent events (inflation/technology/set) to re-apply
                if any(k in ev for k in ("price_bump", "entrants", "cost_shock")):
                    continue
            if "price_bump" in ev:
                self._apply_price_bump(market, ev["price_bump"])
                self._applied_once.add(ev_key)
            if "technology" in ev:
                self._apply_technology(market, ev["technology"])
            if "entrants" in ev:
                self._apply_entrants(market, ev["entrants"])
                self._applied_once.add(ev_key)
            if "transaction_cost" in ev:
                self._apply_transaction_cost(farmers, ev["transaction_cost"])
            if "cost_shock" in ev:
                self._apply_cost_shock(farmers, ev["cost_shock"])
                self._applied_once.add(ev_key)
            if "inflation" in ev:
                self._apply_inflation_rule(ev["inflation"])
