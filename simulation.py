# simulation.py
import numpy as np, random, pickle
from typing import List
from config import base_spec, add_constant_market, coordination_spec
from market import Market
from farmer import Farmer
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Simulation configuration ---
N = 100              # number of agents
EPISODES = 100       # number of independent runs (experiments)
ITERATIONS = 1000    # iterations per run

EVAL_FRACTION = 0.0 # last 10% of iterations in each episode = evaluation (no learning)

def run_experiment(N: int, T: int, episodes: int = EPISODES, add_gov: bool = False, switch_cost = 0, seed: int = 1411):
    # Reproducibility
    np.random.seed(seed); random.seed(seed)
    eval_start = int((1.0 - EVAL_FRACTION) * T)    

    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_gov" if add_gov else ""
    run_dir = base_dir / f"run_{timestamp}{suffix}"
    run_dir.mkdir(exist_ok=False)

    spec = base_spec(N)
    if add_gov:
        spec = add_constant_market(spec, price=1.0, cost=0.0)  # adds market K+1

    # spec = coordination_spec(N)


    results_market, results_profits = [], []

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

            if it == eval_start:
                new_state = {m: 0 for m in spec.markets}

                for f in farmers:
                    f.eps = 0.0
                    new_m = random.choice(spec.markets)
                    f.market = new_m
                    new_state[new_m] += 1
                    f.switched_last_step = False

                market.state = new_state

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

            for f in farmers:
                p = f.calculate_profits(prices, state, it)

                if it < eval_start:
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

    print(f"Wyniki zapisane do: {run_dir}")
    return run_dir


if __name__ == "__main__":
    # run_experiment(N=N, T=ITERATIONS, episodes=EPISODES, add_gov=True)
    #1. Basic experiment
    run_experiment(N=100, T=1000, episodes=10, add_gov=False, switch_cost = 0)

    #2. Experiment with goverment subsidies
    run_experiment(N=100, T=1000, episodes=10, add_gov=True, switch_cost = 0)

    #3. Experiment with cost inflation


    #4. Experiment with cost shock

    #5. Experiment with technology shock

    #6. Experiment with agents shock

    #7. Experiment with introduction of transaction cost