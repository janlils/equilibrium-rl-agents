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

def run_experiment(
    N: int,
    T: int,
    episodes: int = EPISODES,
    add_gov: bool = False,
    switch_cost = 0,
    seed: int = 1411,
    scenario: list[dict] | None = None,   
):
    # Reproducibility
    np.random.seed(seed); random.seed(seed)

    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_gov" if add_gov else ""
    run_dir = base_dir / f"run_{timestamp}{suffix}"
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

        farmers = [
            Farmer(i, spec.markets, spec.costs, spec.price_funcs, switch_cost=switch_cost)
            for i in range(N)
        ]

        init = {m: 0 for m in spec.markets}
        for f in farmers:
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
            state = market.get_state()
            for f in farmers:
                f.choose_action(state, it)
                old_m, new_m = f.action()
                f.switched_last_step = (old_m != new_m)
                state = market.update(old_m, new_m)

            # End-of-round accounting
            state  = market.get_state()
            prices = market.get_prices()

            # --- Dynamic Rosenthal potential for this iteration ---
            markets_ordered = sorted(spec.markets)

            # Current allocation of agents
            alloc = [state[m] for m in markets_ordered]

            # Effective costs in this iteration (cost shocks + inflation)
            costs_iter = []
            for m in markets_ordered:
                base_cost = farmers[0].costs[m]  # already includes cost_shock

                rule = INFLATION_RULES.get(m)
                if rule is None:
                    rule = INFLATION_RULES.get(None)

                if rule is not None:
                    start = int(rule.get("start", 0))
                    rate  = float(rule.get("rate", 0.0))
                    if it >= start:
                        cost_eff = base_cost * (1.0 + rate * float(it - start))
                    else:
                        cost_eff = base_cost
                else:
                    cost_eff = base_cost

                costs_iter.append(cost_eff)

            # Effective counts (technology, entrants, external_counts)
            eff_counts = market.get_effective_counts()

            # Build price functions for potential calculation
            def make_p(m_id):
                def p(n, _m=m_id):
                    # RL + exogenous raw counts
                    raw_counts = {}
                    for j in market.spec.markets:
                        if j == _m:
                            raw_counts[j] = n + market.external_counts[j]
                        else:
                            raw_counts[j] = market.state[j] + market.external_counts[j]

                    # Apply technology multipliers
                    eff_counts_tmp = {}
                    for j in market.spec.markets:
                        eff_counts_tmp[j] = int(round(raw_counts[j] * market.n_multiplier[j]))

                    N_eff_tmp = sum(eff_counts_tmp.values())
                    n_eff_m = eff_counts_tmp[_m]

                    return market.spec.price_funcs[_m](n_eff_m, eff_counts_tmp, N_eff_tmp) + market.price_add[_m]
                return p


            p_funcs_iter = [make_p(m) for m in markets_ordered]

            # Current potential
            phi = potential(p_funcs_iter, costs_iter, alloc)

            # Max potential for this game instance
            _, phi_max = dp_potential_max(N, p_funcs_iter, costs_iter)
            phi_ratio = phi / phi_max if phi_max != 0 else 1.0

            # Save potential metrics
            results_potential.append([ep, it + 1, phi, phi_max, phi_ratio])

            for f in farmers:
                p = f.calculate_profits(prices, state, it)

                # LEARNING
                f.update(prices, state, it)
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

    cols_pot = ['Round', 'Iteration', 'Phi', 'Phi_max', 'Phi_ratio']
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

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0
    )

    # One-off positive price bump on market 2 at iteration 200
    scenario_1 = [
        {"when": {"iter": 200}, "price_bump": {2: +3.0}},
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_1
    )

    # Influx of 15 exogenous participants on market 1 in iteration 300
    scenario_2 = [
        {"when": {"iter": 300}, "entrants": {1: 15}},
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_2
    )    

    # Technology shock: from it>=400 each agent on market 3 effectively counts as 2
    scenario_3 = [
        {"when": {"from_iter": 400}, "technology": {3: 2.0}},
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_3
    )    

    # Increase transaction (switching) cost from it>=500
    scenario_4 = [
        {"when": {"from_iter": 500}, "transaction_cost": {"set": 2.0}},
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_4
    )    

    # Cost shock: from it=600 on market 4 (+2 absolute)
    scenario_5 = [
        {"when": {"iter": 600}, "cost_shock": {4: +2}},
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_5
    )    

    # Cost inflation – linear on market 1 from it=700 at +1% per iteration
    scenario_6 = [
        {"when": {"from_iter": 700}, "inflation": {"market": 1, "start": 700, "rate": 0.01}}
    ]

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_6
    )    

    scenario_7 = scenario_1 + scenario_2 + scenario_3 + scenario_4 + scenario_5 + scenario_6

    run_experiment(
        N=100, T=1000, episodes=10,
        add_gov=True, switch_cost=0, scenario=scenario_7
    )