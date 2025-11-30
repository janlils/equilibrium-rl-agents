# market.py
from typing import Dict
from config import MarketSpec, MarketId

class Market:
    def __init__(self, spec: MarketSpec, initial: Dict[MarketId, int]):
        self.spec = spec
        self.state: Dict[MarketId, int] = {m: int(initial.get(m, 0)) for m in spec.markets}
        # --- DYNAMICS / SHOCKS ---
        # Productivity (technology) multiplier per market: effective count = round((state + external) * n_multiplier)
        self.n_multiplier: Dict[MarketId, float] = {m: 1.0 for m in spec.markets}
        # Additive price bump per market (can be temporary or persistent)
        self.price_add: Dict[MarketId, float] = {m: 0.0 for m in spec.markets}
        # Exogenous participants per market (affect prices/N, not tracked as Farmers)
        self.external_counts: Dict[MarketId, int] = {m: 0 for m in spec.markets}

    def get_state(self) -> Dict[MarketId, int]:
        return dict(self.state)

    def get_effective_counts(self) -> Dict[MarketId, int]:
        eff = {}
        for m in self.spec.markets:
            base = self.state[m] + self.external_counts[m]
            eff[m] = int(round(base * self.n_multiplier[m]))
        return eff

    def get_prices(self) -> Dict[MarketId, float]:
        """Compute prices p_i(n_i, N) where:
        - N counts real participants (RL + external),
        - technology affects only local congestion (n_eff), not global N."""
        
        # Base participant counts (RL + external), without technology multiplier
        base_counts = {
            m: self.state[m] + self.external_counts[m]
            for m in self.spec.markets
        }
        N_dyn = sum(base_counts.values())

        # Effective local counts including technology multiplier
        eff_counts = self.get_effective_counts()

        prices: Dict[MarketId, float] = {}
        for m in self.spec.markets:
            n_m_eff = eff_counts[m]

            # Price function uses local effective count (with tech)
            # but global N without tech.
            prices[m] = (
                self.spec.price_funcs[m](n_m_eff, eff_counts, N_dyn)
                + self.price_add[m]
            )

        return prices


    def update(self, old_market: MarketId, new_market: MarketId) -> Dict[MarketId, int]:
        """Move one producer; return new state."""
        if old_market == new_market:
            return self.get_state()
        if old_market not in self.state or new_market not in self.state:
            raise KeyError(f"Unknown market transition: {old_market}->{new_market}")
        if self.state[old_market] <= 0:
            raise ValueError(f"Negative count on market {old_market}")
        self.state[old_market] -= 1
        self.state[new_market] += 1
        return self.get_state()
