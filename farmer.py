# farmer.py
import numpy as np
from typing import Dict, Tuple
from config import MarketId
GAMMA = 0.9
ALPHA = 0.1

class Model:
    def __init__(self, markets: Tuple[MarketId, ...]):
        # Feature dimension: K "action slots" + 1 bias
        self.markets = markets
        self.dim = len(markets) + 1
        self.theta = np.random.randn(self.dim) + 10.0  # keep optimistic init for same behavior

    def _index(self, m: MarketId) -> int:
        return self.markets.index(m)

    def s2x(self, s: Dict[MarketId, int], m: MarketId) -> np.ndarray:
        """Feature: [one-hot(m) * (share_after_join)] over K markets + bias."""
        denom = max(1, sum(s.values()))
        x = np.zeros(self.dim, dtype=float)
        x[self._index(m)] = (s[m] + 1) / denom
        x[-1] = 1.0  # bias
        return x

    def predict(self, s: Dict[MarketId, int], m: MarketId) -> float:
        x = self.s2x(s, m)
        return float(self.theta @ x)

    def grad(self, s: Dict[MarketId, int], m: MarketId) -> np.ndarray:
        return self.s2x(s, m)

class Farmer:
    def __init__(self, id: int, markets: Tuple[MarketId, ...], costs: Dict[MarketId, float]):
        self.id = id
        self.markets = markets
        self.costs = dict(costs)
        self.market: MarketId = int(np.random.choice(markets))
        self.model = Model(markets)

        self.previous_state: Dict[MarketId, int] = {}
        self.next_action: MarketId = self.market

        self.profit: float = 0.0
        self.previous_profit: float = 0.0

    # ---------- helpers ----------
    @staticmethod
    def getQs(model: Model, s: Dict[MarketId, int], markets: Tuple[MarketId, ...]) -> Dict[MarketId, float]:
        """Return dict of Q(s,m) for all markets m."""
        return {m: model.predict(s, m) for m in markets}

    @staticmethod
    def random_argmax(d: Dict[MarketId, float]) -> MarketId:
        """Random tie-break argmax."""
        import numpy as np
        mval = max(d.values())
        keys = [k for k, v in d.items() if v == mval]
        return int(np.random.choice(keys))

    def random_action(self, eps: float, greedy_action: MarketId | None) -> MarketId:
        """ε-randomization: with prob (1-ε) use greedy_action, else uniform random over markets."""
        import numpy as np
        if greedy_action is None or np.random.rand() >= (1 - eps):
            return int(np.random.choice(self.markets))
        return greedy_action

    # ---------- main API  ----------
    def choose_action(self, state: Dict[MarketId, int], it: int = 0) -> MarketId:
        """Pick action for the provided state; set next_action & previous_state; return action."""
        t = 1 + (it // 100)  
        Qs = Farmer.getQs(self.model, state, self.markets)
        greedy = Farmer.random_argmax(Qs)
        a = self.random_action(eps=(0.1 / t), greedy_action=greedy)
        self.next_action = a
        self.previous_state = state #dict(state) 
        return a

    def action(self) -> tuple[MarketId, MarketId]:
        """Apply planned action; return (old_market, new_market)."""
        old_market = self.market
        self.market = self.next_action
        return (old_market, self.market)

    def calculate_profits(self, prices: Dict[MarketId, float], state: Dict[MarketId, int]) -> float:
        """Update profit trackers; reward will be derived as profit delta."""
        self.previous_profit = self.profit
        self.profit = prices[self.market] - self.costs[self.market]
        return self.profit

    def update(self, prices: Dict[MarketId, float], state: Dict[MarketId, int], it: int):
        """update with target form."""
        t = 1 + (it // 100) * 0.01
        alpha = ALPHA / t

        s  = self.previous_state
        s2 = state
        r  = self.profit - self.previous_profit

        a  = self.market
        a2 = self.next_action

        td = (r + GAMMA * self.model.predict(s2, a2)) - self.model.predict(s, a)
        self.model.theta += alpha * td * self.model.grad(s, a)
