# simulation.py
import numpy as np, random, pickle
from typing import List
from config import base_spec, add_constant_market
from market import Market
from farmer import Farmer, INFLATION_RULES
import pandas as pd
from pathlib import Path
from datetime import datetime
from scenario import ShockEngine
from optimal_allocation import potential, dp_potential_max

# --- Simulation configuration ---
N = 100              # number of agents
EPISODES = 100       # number of independent runs (experiments)
ITERATIONS = 1000    # iterations per run
EFFICIENT_THRESHOLD = 0.99

def run_experiment(
    N: int,
    T: int,
    episodes: int = EPISODES,
    add_gov: bool = False,
    switch_cost = 0,
    seed: int = 1411,
    scenario: list[dict] | None = None,
    exp_id: str | None = None, 
    random_policy: bool = False,
    model_type: str = "full",
):

    # Reproducibility
    np.random.seed(seed); random.seed(seed)
    INFLATION_RULES.clear()

    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    id_suffix = f"_{exp_id}" if exp_id else ""
    gov_suffix = "_gov" if add_gov else ""
    run_dir_name = f"run_{timestamp}{id_suffix}{gov_suffix}"
    run_dir = base_dir / run_dir_name
    run_dir.mkdir(exist_ok=False)

    spec = base_spec(N)
    if add_gov:
        spec = add_constant_market(spec, price=1.0, cost=0.0)  # adds market K+1

    engine = ShockEngine(
        scenario = scenario or [],   
        spec_costs = spec.costs    
    )

    results_market, results_profits, results_potential = [], [], []

    for ep in range(episodes):          # episodes
        print(f"=== Episode {ep + 1}/{episodes} ===")

        markets_ordered = sorted(spec.markets)
        public_price_board = {m: {} for m in markets_ordered}         

        farmers = [
            Farmer(i, spec.markets, spec.costs, switch_cost=switch_cost, model_type=model_type,)
            for i in range(N)
        ]
        efficient_counter = 0

        init = {m: 0 for m in spec.markets}
        for f in farmers:
            if random_policy:
                f.random_policy = True
            init[f.market] += 1

        market = Market(spec, init)

        for it in range(T):       # iterations
            profits_row = [ep, it + 1] + [0] * N
            markets_row = [ep, it + 1] + [0] * N

            # Apply scheduled shocks for this tick
            engine.tick(ep, it, market, farmers)

            for f in farmers:
                f.output_multiplier = market.n_multiplier.copy()

            random.shuffle(farmers)

            # Asynchronous decisions 
            state = market.get_effective_counts()
            for f in farmers:
                f.public_price_board = public_price_board
                f.choose_action(state, it)
                old_m, new_m = f.action()
                f.switched_last_step = (old_m != new_m)
                market.update(old_m, new_m)
                state = market.get_effective_counts()

            # End-of-round accounting
            state  = market.get_state()
            prices = market.get_prices()

            for f in farmers:
                f.last_prices = dict(prices)

            # --- Theoretical potential for this iteration ---

            # Current allocation of agents (RL outcome)
            alloc = [state[m] for m in markets_ordered]

            # Effective costs in this iteration (cost shocks + inflation)
            costs_iter: list[float] = []
            for m in markets_ordered:
                base_cost = farmers[0].costs[m]  # already includes cost shocks

                rule = INFLATION_RULES.get(m)
                if rule is None:
                    rule = INFLATION_RULES.get(None)

                if rule is not None:
                    start = int(rule.get("start", 0))
                    rate = float(rule.get("rate", 0.0))
                    if it >= start:
                        cost_eff = base_cost * (1.0 + rate * float(it - start))
                    else:
                        cost_eff = base_cost
                else:
                    cost_eff = base_cost

                costs_iter.append(cost_eff)

            # Base participant counts (RL + external), without technology multiplier.
            # This matches how N should be interpreted: number of heads, not productivity.
            base_counts = {
                m: market.state[m] + market.external_counts[m]
                for m in markets_ordered
            }
            N_dyn = sum(base_counts.values())

            # Per-market output multiplier (technology) for this iteration.
            # This captures how many units a single RL agent sells on market m.
            q_iter = {
                m: float(market.n_multiplier[m])
                for m in markets_ordered
            }

            # Build "price" functions for potential calculation.
            # IMPORTANT: here p_i(k) returns *revenue per agent* (price * q),
            # not the unit price itself. Potential then approximates sum of profits:
            # Phi ≈ sum_k (revenue_per_agent(k) - cost).
            def make_p(m_id):
                def p(n, _m=m_id):
                    # Hypothetical number of RL agents on market _m: n
                    # plus exogenous participants
                    raw = n + market.external_counts[_m]

                    # Apply technology multiplier for this market to local congestion
                    n_eff_m = int(round(raw * market.n_multiplier[_m]))

                    # Minimal state dict for compatibility with price functions
                    eff_counts_tmp = {_m: n_eff_m}

                    # Compute unit price as in the market (using dynamic N if you use variant A).
                    unit_price = (
                        market.spec.price_funcs[_m](n_eff_m, eff_counts_tmp, N_dyn)
                        + market.price_add[_m]
                    )

                    # Convert unit price into per-agent revenue using technology multiplier q.
                    revenue_per_agent = unit_price * q_iter[_m]
                    return revenue_per_agent
                return p

            # One "price" function per market: p_i(k) = revenue_per_agent(k)
            p_funcs_iter = [make_p(m) for m in markets_ordered]

            # Current potential for the RL allocation (approximate total profit)
            phi = potential(p_funcs_iter, costs_iter, alloc)

            # Optimal potential and allocation for this game instance
            alloc_max, phi_max = dp_potential_max(N, p_funcs_iter, costs_iter)
            phi_ratio = phi / phi_max if phi_max != 0 else 1.0

            # Actual allocation, optimal allocation and current prices per market
            actual_alloc_row = alloc
            optimal_alloc_row = alloc_max
            price_row = [prices[m] for m in markets_ordered]

            efficient_counter += (1 if phi_ratio >= EFFICIENT_THRESHOLD else 0)

            # Save potential metrics + allocations + prices
            results_potential.append(
                [ep, it + 1, phi, phi_max, phi_ratio]
                + actual_alloc_row
                + optimal_alloc_row
                + price_row
            )

            state_eff = market.get_effective_counts()
            for m in markets_ordered:
                n_eff = int(state_eff[m])
                public_price_board[m][n_eff] = float(prices[m])

            for f in farmers:
                f.public_price_board = public_price_board
                p = f.calculate_profits(prices, state, it)

                # LEARNING
                f.update(state_eff, it)
                f.update_epsilon(prices)
                
                idx = f.id + 2
                markets_row[idx] = f.market
                profits_row[idx] = p

            results_market.append(markets_row)
            results_profits.append(profits_row)

        # Average coefficients across all farmers
        avg_theta = np.mean([f.model.theta for f in farmers], axis=0)
        print(f"\nEpisode {ep}: average coefficients:")
        for name, w in zip(farmers[0].model.feature_names, avg_theta):
            print(f"  {name:15s} = {w: .4f}")

        eff_ratio = efficient_counter / T
        print(f"Episode {ep}: efficient iterations: {efficient_counter}/{T} ({eff_ratio:.2%})")



    suffix = f"_{len(spec.markets)}m"  # np. _4m albo _5m

    # Columns: Round, Iteration, 0..N-1  (agent indices as ints)
    cols = ['Round', 'Iteration'] + list(range(N))

    df_market = pd.DataFrame(results_market, columns=cols)
    df_profits = pd.DataFrame(results_profits, columns=cols)

    # - market IDs as int
    for c in range(N):
        df_market[c] = pd.to_numeric(df_market[c], errors='coerce').astype('Int64')
    # - profits as float
    for c in range(N):
        df_profits[c] = pd.to_numeric(df_profits[c], errors='coerce').astype(float)

    # Round/Iteration as int
    df_market['Round'] = pd.to_numeric(df_market['Round'], errors='coerce').fillna(0).astype(int)
    df_market['Iteration'] = pd.to_numeric(df_market['Iteration'], errors='coerce').fillna(0).astype(int)
    df_profits['Round'] = pd.to_numeric(df_profits['Round'], errors='coerce').fillna(0).astype(int)
    df_profits['Iteration'] = pd.to_numeric(df_profits['Iteration'], errors='coerce').fillna(0).astype(int)

    # Save DataFrames
    with open(run_dir / f"results_market{suffix}.pickle", "wb") as f:
        pickle.dump(df_market, f)

    with open(run_dir / f"results_profits{suffix}.pickle", "wb") as f:
        pickle.dump(df_profits, f)

    markets_ordered = sorted(spec.markets)

    cols_pot = (
        ['Round', 'Iteration', 'Phi', 'Phi_max', 'Phi_ratio']
        + [f"n_market_{m}" for m in markets_ordered]          # actual allocation per market
        + [f"opt_n_market_{m}" for m in markets_ordered]      # optimal allocation per market
        + [f"price_market_{m}" for m in markets_ordered]      # market prices
    )

    df_potential = pd.DataFrame(results_potential, columns=cols_pot)

    with open(run_dir / f"results_potential{suffix}.pickle", "wb") as f:
        pickle.dump(df_potential, f)

    print(f"Wyniki zapisane do: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=False, switch_cost=0
    )