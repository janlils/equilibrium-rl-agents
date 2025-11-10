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
        self.dim = len(markets) + 2
        self.theta = np.random.randn(self.dim) * 0.1 + 1.0

    def _index(self, m: MarketId) -> int:
        return self.markets.index(m)

    def s2x(self, s: Dict[MarketId, int], m: MarketId, switch_penalty: float) -> np.ndarray:
        """Feature: [one-hot(m) * (share_after_join)] over K markets + bias."""
        denom = max(1, sum(s.values()))
        x = np.zeros(self.dim, dtype=float)
        x[self._index(m)] = (s[m] + 1) / denom
        
        x[-2] = switch_penalty

        x[-1] = 1.0  # bias
        return x

    def predict(self, s: Dict[MarketId, int], m: MarketId, switch_penalty: float) -> float:
        x = self.s2x(s, m, switch_penalty)
        return float(self.theta @ x)

    def grad(self, s: Dict[MarketId, int], m: MarketId, switch_penalty: float) -> np.ndarray:
        return self.s2x(s, m, switch_penalty)

class Farmer:
    def __init__(self, id: int, markets: Tuple[MarketId, ...], costs: Dict[MarketId, float], switch_cost = 1):
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

        # exploration rate (agent-specific epsilon)
        self.eps: float = 0.1

        self.switch_cost = switch_cost
        self.switched_last_step = False

    # ---------- helpers ----------
    @staticmethod
    def getQs(model, state, markets, current_market, switch_cost: float):
        return {
            m: model.predict(
                state,
                m,
                switch_penalty=(switch_cost if m != current_market else 0.0)
            )
            for m in markets
        }

    @staticmethod
    def random_argmax(Qs: Dict[MarketId, float]) -> MarketId:
        """Return random argmax to break ties."""
        maxQ = max(Qs.values())
        best = [m for m, q in Qs.items() if q == maxQ]
        return int(np.random.choice(best))        
        

    # ---------- main API  ----------
    def choose_action(self, state: Dict[MarketId, int], it: int = 0) -> MarketId:
        """ε-greedy policy: choose next market to move to and store (s_t, a_t)."""
        # compute Q(s, a) for all markets
        Qs = Farmer.getQs(self.model, state, self.markets, self.market, self.switch_cost)
        greedy = Farmer.random_argmax(Qs)

        # agent-specific exploration rate
        eps = float(np.clip(self.eps, 0.0, 1.0))

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

        if self.switched_last_step:
            self.profit -= self.switch_cost

        return self.profit        

    def update_epsilon(self, prices: Dict[MarketId, float]):
        """
        Update exploration rate epsilon based on regret:
        if current profit is much worse than the best achievable profit
        on other markets, increase epsilon to encourage exploration.
        """
        # Compute hypothetical profit on each market
        profits_all = {m: prices[m] - self.costs[m] for m in self.markets}
        best_profit = max(profits_all.values())

        # How much worse am I than the best?
        regret = max(0.0, best_profit - self.profit)

        # Map dissatisfaction in [0, +∞) to epsilon in [EPS_MIN, EPS_MAX]
        EPS_MIN = 0.02
        EPS_MAX = 0.2

        tau = 5.0

        if regret > tau:
            self.eps = EPS_MAX
        else:
            self.eps = EPS_MIN


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
        last_switch_penalty = self.switch_cost if self.switched_last_step else 0

        # Reward based on current profit
        r_t = self.profit

        # Next state and next action
        s_tp1 = dict(state)
        a_tp1 = self.choose_action(s_tp1, it)


        # Temporal Difference (TD) target
        q_sa = self.model.predict(s_t, a_t, last_switch_penalty)

        next_switch_penalty = self.switch_cost if a_tp1 != self.market else 0.0
        q_next = self.model.predict(s_tp1, a_tp1, next_switch_penalty)
        td = (r_t + GAMMA * q_next) - q_sa

        # Gradient step
        self.model.theta += ALPHA * td * self.model.grad(s_t, a_t, last_switch_penalty)

        # Move forward in time
        self.state_t = s_tp1
        self.action_t = a_tp1