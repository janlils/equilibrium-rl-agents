# config.py
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import math

MarketId = int
PriceFn = Callable[[int, Dict[MarketId, int], int], float]  # p(n_i, state, N)

@dataclass(frozen=True)
class MarketSpec:
    markets: Tuple[MarketId, ...]                 # e.g., (1,2,3,4) or (1,2,3,4,5)
    costs: Dict[MarketId, float]                  # c_i for each market i
    price_funcs: Dict[MarketId, PriceFn]          # p_i(n_i, state, N) for each market i

def base_spec(N: int) -> MarketSpec:
    def p1(n, s, N): return (4 + (N // 5)) * 2 - n         
    def p2(n, s, N): return 10 + (N // 5) - n                  
    def p3(n, s, N): return N + 10 - 4 * (N // 5) - n 
    def p4(n, s, N): return 10 + (N // 5) - n           
    markets = (1, 2, 3, 4)
    costs   = {1: 9.0, 2: 8.0, 3: 11.0, 4: 10.0}
    prices  = {1: p1, 2: p2, 3: p3, 4: p4}
    return MarketSpec(markets, costs, prices)

def add_constant_market(spec: MarketSpec, price: float = 1.0, cost: float = 0.0) -> MarketSpec:
    """Return a new spec with one extra market having constant price and given cost."""
    new_id = max(spec.markets) + 1 if spec.markets else 1
    def p_const(n, s, N): return float(price)
    markets = (*spec.markets, new_id)
    costs   = {**spec.costs, new_id: float(cost)}
    prices  = {**spec.price_funcs, new_id: p_const}
    return MarketSpec(markets, costs, prices)

def coordination_spec(N: int) -> MarketSpec:
    """
    Spec for a 3-market environment with coordination / network effects.

    Market 1: "network-effect" market with a hump-shaped price as a function of its own share.
              - Low participation -> low price (no network benefits).
              - Medium participation -> high price (strong network effects).
              - High participation -> price goes down again (congestion).
    Market 2: Standard competitive market with a smoothly decreasing price in n_2.
    Market 3: Safe, almost constant market with low but stable profit.
    """

    # Market 1: hump-shaped price as a function of its own share n1 / N
    def p1(n, s, N):
        if N <= 0:
            return 0.0
        share = n / float(N)  # share of agents on market 1, in [0,1]

        # Piecewise-linear "hump":
        #  - for share in [0, 0.2]  : gently increasing
        #  - for share in (0.2, 0.6]: steeper increase (strong network effects)
        #  - for share in (0.6, 1.0]: decreasing (congestion)
        if share <= 0.05:
            # from 4.0 at share=0  to 5.6 at share=0.2
            return 4.0 + 8.0 * share
        elif share <= 0.6:
            # from 5.6 at share=0.2 to 8.0 at share=0.6
            return 5.6 + 6.0 * (share - 0.05)
        else:
            # from 8.0 at share=0.6 down to 6.0 at share=1.0
            return 8.0 - 10.0 * (share - 0.6)

    # Market 2: standard downward-sloping price in n_2 (more agents -> lower price)
    def p2(n, s, N):
        if N <= 0:
            return 0.0
        share = n / float(N)
        # from 9.0 at share=0  to 6.0 at share=1
        return 9.0 - 3.0 * share

    # Market 3: almost constant "safe" market
    def p3(n, s, N):
        # Slightly decreasing in n, but mostly flat
        return 6.0 - 0.05 * n

    markets = (1, 2, 3)

    # Costs:
    # - Market 1: cost chosen so that profit is positive only for intermediate shares
    #   (network effects needed to make it attractive).
    # - Market 2: solid, standard profits for a broad range of n_2.
    # - Market 3: small but stable profit (safe outside option).
    costs = {
        1: 5.5,   # network-effect market: needs critical mass to be profitable
        2: 5.0,   # standard competitive market
        3: 5.5,   # safe market with low, almost constant profit
    }

    prices = {1: p1, 2: p2, 3: p3}
    return MarketSpec(markets, costs, prices)
