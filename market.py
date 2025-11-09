# market.py
from typing import Dict
from config import MarketSpec, MarketId

class Market:
    def __init__(self, spec: MarketSpec, initial: Dict[MarketId, int]):
        self.spec = spec
        self.state: Dict[MarketId, int] = {m: int(initial.get(m, 0)) for m in spec.markets}

    def get_state(self) -> Dict[MarketId, int]:
        return self.state #dict(self.state)

    def get_prices(self) -> Dict[MarketId, float]:
        """Compute prices p_i(n_i, state, N) for current state."""
        N = sum(self.state.values())
        prices = {}
        for m in self.spec.markets:
            n_m = self.state[m]
            prices[m] = self.spec.price_funcs[m](n_m, self.state, N)
        return prices

    def update(self, old_market: MarketId, new_market: MarketId) -> Dict[MarketId, int]:
        """Move one producer; return new state."""
        if old_market == new_market:
            return self.get_state()
        if old_market not in self.state or new_market not in self.state:
            raise KeyError(f"Unknown market transition: {old_market}->{new_market}")
        self.state[old_market] -= 1
        self.state[new_market] += 1
        return self.get_state()
