# farmer.py
import numpy as np
from typing import Dict, Tuple, Optional
from config import MarketId, PriceFn

INFLATION_RULES: Dict[Optional[int], Dict[str, float]] = {}
GAMMA = 0.9
ALPHA = 0.05
PROFIT_SCALE = 10.0

class Model:
    def __init__(self, markets: Tuple[MarketId, ...], mode: str = "full"):
        """
        Linear function approximator for Q(s, m).

        Modes:
        - "full": use profits (current + counterfactual), advantage and market share.
        - "profit_only": use only profit-based features (no market share).
        - "count_only": use only market share (number of participants) features.

        In all modes we also add:
        - bias_global: global bias term
        - bias_market_m: one-hot market-specific bias (one coordinate per market)
        """
        self.markets = markets
        self.num_markets = len(markets)
        self.mode = mode

        # Decide how many "global" features are used for given mode
        if mode == "full":
            # profit_current, profit_m, advantage_m, share_m, bias_global
            self.global_dim = 5
            base_names = [
                "profit_current",
                "profit_m",
                "advantage_m",
                "share_m",
                "bias_global",
            ]
        elif mode == "profit_only":
            # profit_current, profit_m, advantage_m, bias_global
            self.global_dim = 4
            base_names = [
                "advantage_m",
                "bias_global",
            ]
        elif mode == "count_only":
            # share_m, bias_global
            self.global_dim = 2
            base_names = [
                "share_m",
                "bias_global",
            ]
        else:
            raise ValueError(f"Unknown model mode: {mode}")

        # Total dimension = global features + one bias per market
        self.dim = self.global_dim + self.num_markets

        # Initialize parameters
        self.theta = np.random.randn(self.dim) * 0.1 + 1.0

        # Mapping from market id to index in the one-hot part of the feature vector
        self.market_index = {m: i for i, m in enumerate(self.markets)}

        # Human-readable names for coefficients
        self.feature_names = base_names + [f"bias_market_{m}" for m in self.markets]

    def s2x(
        self,
        s: Dict[MarketId, int],
        m: MarketId,
        current_market: MarketId,
        costs: Dict[MarketId, float],
        price_funcs: Dict[MarketId, PriceFn],
        switch_cost: float,
        profit_scale: float = PROFIT_SCALE,
    ) -> np.ndarray:
        """
        Build feature vector for Q(s, m).

        Monetary values are adaptively scaled to keep the features in a
        numerically stable range, even if switching costs are very large.
        """
        N = max(1, sum(s.values()))

        # --- Raw profits (unscaled) ---
        price_current = price_funcs[current_market](s[current_market], s, N)
        profit_current_raw = price_current - costs[current_market]

        price_m = price_funcs[m](s[m], s, N)
        switch_penalty = switch_cost if m != current_market else 0.0
        profit_m_raw = price_m - costs[m] - switch_penalty

        advantage_raw = profit_m_raw - profit_current_raw

        # --- Adaptive scaling for monetary features ---
        max_abs = max(
            1.0,
            abs(profit_current_raw),
            abs(profit_m_raw),
            abs(advantage_raw),
            abs(switch_cost),
            profit_scale,  # baseline scale from config
        )
        scale = max_abs

        profit_current = profit_current_raw / scale
        profit_m = profit_m_raw / scale
        advantage_m = advantage_raw / scale

        # --- Share on market m (optionally +1 if we move there) ---
        share_m = (s[m] + (1 if m != current_market else 0)) / float(N)

        # --- Build feature vector ---
        x = np.zeros(self.dim, dtype=float)

        # Fill global part depending on the selected mode
        if self.mode == "full":
            # [profit_current, profit_m, advantage_m, share_m, bias_global]
            x[0] = profit_current
            x[1] = profit_m
            x[2] = advantage_m
            x[3] = share_m
            x[4] = 1.0  # global bias
            bias_offset = 5

        elif self.mode == "profit_only":
            # [profit_current, profit_m, advantage_m, bias_global]
            x[0] = profit_current
            x[1] = profit_m
            x[2] = advantage_m
            x[3] = 1.0  # global bias
            bias_offset = 4

        elif self.mode == "count_only":
            # [share_m, bias_global]
            x[0] = share_m
            x[1] = 1.0  # global bias
            bias_offset = 2

        else:
            raise ValueError(f"Unknown model mode in s2x: {self.mode}")

        # Market-specific bias (one-hot)
        idx_market = bias_offset + self.market_index[m]
        x[idx_market] = 1.0

        return x


    def predict(
        self,
        s: Dict[MarketId, int],
        m: MarketId,
        current_market: MarketId,
        costs: Dict[MarketId, float],
        price_funcs: Dict[MarketId, PriceFn],
        switch_cost: float,
        profit_scale: float = PROFIT_SCALE,
    ) -> float:
        """Predict Q(s, m) using the current parameter vector."""
        x = self.s2x(s, m, current_market, costs, price_funcs, switch_cost, profit_scale)
        return float(self.theta @ x)

    def grad(
        self,
        s: Dict[MarketId, int],
        m: MarketId,
        current_market: MarketId,
        costs: Dict[MarketId, float],
        price_funcs: Dict[MarketId, PriceFn],
        switch_cost: float,
        profit_scale: float = PROFIT_SCALE,
    ) -> np.ndarray:
        """Return the feature vector ∂Q/∂θ = x(s, m)."""
        return self.s2x(s, m, current_market, costs, price_funcs, switch_cost, profit_scale)

    def coef_dict(self) -> Dict[str, float]:
        """Return a dictionary mapping feature names to current parameter values."""
        return {name: float(w) for name, w in zip(self.feature_names, self.theta)}

    def pretty_print(self, prefix: str = ""):
        """Print model coefficients in a readable way."""
        print(prefix + "Model coefficients:")
        for name, w in zip(self.feature_names, self.theta):
            print(f"{prefix}  {name:20s} = {w: .6f}")


class Farmer:
    def __init__(
        self,
        id: int,
        markets: Tuple[MarketId, ...],
        costs: Dict[MarketId, float],
        price_funcs: Dict[MarketId, PriceFn],
        switch_cost: float = 1,
        model_type: str = "full",
    ):
        self.id = id
        self.markets = markets
        self.costs = dict(costs)
        self.price_funcs = dict(price_funcs)

        # Initial market is chosen at random
        self.market: MarketId = int(np.random.choice(markets))
        # Market before the last move (used for Q(s_t, a_t))
        self.prev_market: MarketId = self.market

        # Create decision model based on requested type
        # model_type in {"full", "profit_only", "count_only"}
        self.model = Model(markets, mode=model_type)

        # --- SARSA internal state ---
        self.state_t: Dict[MarketId, int] = {}
        self.action_t: MarketId = self.market

        # Profit tracking
        self.profit: float = 0.0
        self.previous_profit: float = 0.0

        # Exploration rate (agent-specific epsilon)
        self.eps: float = 0.1

        self.switch_cost = switch_cost
        self.switched_last_step = False

        self.output_multiplier: Dict[MarketId, float] = {int(m): 1.0 for m in markets}

        self.random_policy = False


    # ---------- helpers ----------
    @staticmethod
    def getQs(
        model: Model,
        state: Dict[MarketId, int],
        markets: Tuple[MarketId, ...],
        current_market: MarketId,
        costs: Dict[MarketId, float],
        price_funcs: Dict[MarketId, PriceFn],
        switch_cost: float,
        profit_scale: float = PROFIT_SCALE,
    ):
        """Compute Q(s, m) for all markets given the current state and model."""
        return {
            m: model.predict(
                state,
                m,
                current_market,
                costs,
                price_funcs,
                switch_cost,
                profit_scale,
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

        # agent-specific exploration rate
        if self.random_policy:
            eps = 1.0
        else:
            eps = float(np.clip(self.eps, 0.0, 1.0))

        if np.random.rand() < eps:
            a = int(np.random.choice(self.markets))
        else:
        # compute Q(s, a) for all markets
            Qs = Farmer.getQs(
                self.model,
                state,
                self.markets,
                self.market,
                self.costs,
                self.price_funcs,
                self.switch_cost,
            )
            greedy = Farmer.random_argmax(Qs)            
            a = greedy

        # store current state and chosen action for SARSA update
        self.state_t = dict(state)
        self.action_t = a

        return a
   

    def action(self) -> Tuple[MarketId, MarketId]:
        """Execute chosen action (move to the selected market)."""
        old_market = self.market
        # Remember the market from which we are moving (used as current_market in s_t)
        self.prev_market = old_market
        self.market = self.action_t
        return old_market, self.market


    def calculate_profits(
        self,
        prices: Dict[MarketId, float],
        state: Dict[MarketId, int],
        it: int,
    ) -> float:
        """
        Compute profit = price * q - production cost.
        q (output per agent) captures technology shocks: agent sells q units but pays cost once.

        Note: Any time-varying costs or inflation should be applied to self.costs[...] upstream
        (e.g., by a shock engine), so here we just read the current cost.
        """
        self.previous_profit = self.profit
        q = float(self.output_multiplier.get(self.market, 1.0))
        price = prices[self.market]

        base_cost = self.costs[self.market]

        # Find market-specific rule or fall back to global rule
        rule = INFLATION_RULES.get(self.market)
        if rule is None:
            rule = INFLATION_RULES.get(None)

        if rule is not None:
            start = int(rule.get("start", 0))
            rate  = float(rule.get("rate", 0.0))
            if it >= start:
                cost = base_cost * (1.0 + rate * float(it - start))
            else:
                cost = base_cost
        else:
            cost = base_cost

        self.profit = price * q - cost

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
        EPS_MIN = 0.001
        EPS_MAX = 0.2

        tau = 1.0

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

        # Reward based on current profit
        r_t = self.profit

        # Next state and next action
        s_tp1 = dict(state)
        a_tp1 = self.choose_action(s_tp1, it)

        # Q(s_t, a_t): use the market before the last move (prev_market)
        q_sa = self.model.predict(
            s_t,
            a_t,
            self.prev_market,
            self.costs,
            self.price_funcs,
            self.switch_cost,
        )

        # Q(s_{t+1}, a_{t+1}): use the current market after the move (self.market)
        q_next = self.model.predict(
            s_tp1,
            a_tp1,
            self.market,
            self.costs,
            self.price_funcs,
            self.switch_cost,
        )

        td = (r_t + GAMMA * q_next) - q_sa

        # Gradient step in parameter space
        self.model.theta += ALPHA * td * self.model.grad(
            s_t,
            a_t,
            self.prev_market,
            self.costs,
            self.price_funcs,
            self.switch_cost,
        )

        # Move forward in time
        self.state_t = s_tp1
        self.action_t = a_tp1
