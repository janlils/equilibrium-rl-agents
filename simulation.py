# simulation.py
import numpy as np, random, pickle
from typing import List
from config import base_spec, add_constant_market
from market import Market
from farmer import Farmer
import pandas as pd

# --- Simulation configuration ---
N = 100              # number of agents
EPISODES = 100       # number of independent runs (experiments)
ITERATIONS = 1000    # iterations per run

def run_experiment(N: int, T: int, add_gov: bool = False, seed: int = 1411):
    # Reproducibility
    np.random.seed(seed); random.seed(seed)

    spec = base_spec(N)
    if add_gov:
        spec = add_constant_market(spec, price=1.0, cost=0.0)  # adds market K+1

    # Random initial allocation over markets
    farmers = [Farmer(i, spec.markets, spec.costs) for i in range(N)]

    init = {m: 0 for m in spec.markets}
    for f in farmers:
        init[f.market] += 1

    market = Market(spec, init)    

    results_market, results_profits = [], []

    for ep in range(EPISODES):          # episodes
        print(f"\n=== Episode {ep + 1}/{EPISODES} ===")

        farmers = [Farmer(i, spec.markets, spec.costs) for i in range(N)]
        init = {m: 0 for m in spec.markets}
        for f in farmers:
            init[f.market] += 1

        market = Market(spec, init)    

        for it in range(T):       # iterations
            profits_row = [ep, it + 1] + [0] * N
            markets_row = [ep, it + 1] + [0] * N

            random.shuffle(farmers)

            # Asynchronous decisions 
            state = market.get_state()
            for f in farmers:
                f.choose_action(state, it)
                old_m, new_m = f.action()
                state = market.update(old_m, new_m)

            # End-of-round accounting
            state  = market.get_state()
            prices = market.get_prices()
            for f in farmers:
                p = f.calculate_profits(prices, state)
                f.update(prices, state, it)
                idx = f.id + 2
                markets_row[idx] = f.market
                profits_row[idx] = p

            results_market.append(markets_row)
            results_profits.append(profits_row)



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
    with open(f"results_market{suffix}.pickle", "wb") as f:
        pickle.dump(df_market, f)

    with open(f"results_profits{suffix}.pickle", "wb") as f:
        pickle.dump(df_profits, f)


if __name__ == "__main__":
    run_experiment(N=N, T=ITERATIONS, add_gov=False)
