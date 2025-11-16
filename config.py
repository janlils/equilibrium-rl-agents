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
