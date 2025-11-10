# farmer.py
import numpy as np
from typing import Dict, Tuple
from config import MarketId
GAMMA = 0.9
ALPHA = 0.2

class Model:
    def __init__(self, markets: Tuple[MarketId, ...]):
        # Feature dimension: K "action slots" + 1 bias
        self.markets = markets
        self.dim = len(markets) + 1
        self.theta = np.random.randn(self.dim) * 0.1 + 1.0

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

        # --- SARSA internal state ---
        self.state_t: Dict[MarketId, int] = {}
        self.action_t: MarketId = self.market

        # profit tracking
        self.profit: float = 0.0
        self.previous_profit: float = 0.0

    # ---------- helpers ----------
    @staticmethod
    def getQs(model: Model, state: Dict[MarketId, int], markets: Tuple[MarketId, ...]) -> Dict[MarketId, float]:
        """Compute Q(s,a) for all markets."""
        return {m: model.predict(state, m) for m in markets}

    @staticmethod
    def random_argmax(Qs: Dict[MarketId, float]) -> MarketId:
        """Return random argmax to break ties."""
        maxQ = max(Qs.values())
        best = [m for m, q in Qs.items() if q == maxQ]
        return int(np.random.choice(best))        
        

    # ---------- main API  ----------
    def choose_action(self, state: Dict[MarketId, int], it: int = 0) -> MarketId:
        """ε-greedy policy: choose next market to move to and store (s_t, a_t)."""
        # decaying epsilon schedule
        t = 1 + (it // 1000)
        eps = 0.1 / t
        eps = 0.1

        # compute Q(s, a) for all markets
        Qs = Farmer.getQs(self.model, state, self.markets)
        greedy = Farmer.random_argmax(Qs)

        if np.random.rand() < eps:
            a = int(np.random.choice(self.markets))
        else:
            a = greedy

        # store current state and chosen action for SARSA update
        self.state_t = dict(state)
        self.action_t = a

        return a
   

    def action(self) -> Tuple[MarketId, MarketId]:
        """Execute chosen action (move to selected market)."""
        old_market = self.market
        self.market = self.action_t
        return old_market, self.market


    def calculate_profits(self, prices: Dict[MarketId, float], state: Dict[MarketId, int]) -> float:
        """Compute profit = price - cost."""
        self.previous_profit = self.profit
        self.profit = prices[self.market] - self.costs[self.market]
        return self.profit        

    # ------------------------------
    # SARSA update
    # ------------------------------
    def update(self, prices: Dict[MarketId, float], state: Dict[MarketId, int], it: int):
        """
        SARSA update:
        Q(s_t, a_t) ← Q(s_t, a_t) + α [r_t + γ Q(s_{t+1}, a_{t+1}) − Q(s_t, a_t)]
        """
        # Current (s_t, a_t)
        s_t = self.state_t
        a_t = self.action_t

        # Reward based on current profit
        r_t = self.profit

        # Next state and next action
        s_tp1 = dict(state)
        a_tp1 = self.choose_action(s_tp1, it)

        # Temporal Difference (TD) target
        q_sa = self.model.predict(s_t, a_t)
        q_next = self.model.predict(s_tp1, a_tp1)
        td = (r_t + GAMMA * q_next) - q_sa

        # Gradient step
        self.model.theta += ALPHA * td * self.model.grad(s_t, a_t)

        # Move forward in time
        self.state_t = s_tp1
        self.action_t = a_tp1