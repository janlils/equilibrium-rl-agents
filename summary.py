# summary.py
#
# Usage examples:
#   python summary.py --market results_market_4m.pickle --profits results_profits_4m.pickle
#   python summary.py --market results_market_5m.pickle --profits results_profits_5m.pickle
#
# Outputs (PNG):
#   art_Number_Iter.png      # stacked area: avg number of agents per market by round
#   art_Profit_Iter.png      # lines: avg profit by round (Total bold)
#   art_Decision_chg.png     # line: % agents who changed decision by round
#   art_Profits_chg.png      # scatter+reg: profit vs % of changed decisions

from pathlib import Path
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from matplotlib.colors import ListedColormap
from typing import List, Callable
from optimal_allocation import potential, dp_potential_max
from config import base_spec, add_constant_market
import sys


plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True
})

colors_hex = [ '#24325F', '#82491E', '#B7E4F9', '#E89242','#FB6467', '#69C8EC']
cm = ListedColormap(colors_hex)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_K_and_N(results_market: pd.DataFrame) -> tuple[int, int]:
    """Infer K markets and N agents from market DataFrame."""
    # agents columns are numeric (0..N-1)
    agent_cols = [c for c in results_market.columns if isinstance(c, int)]
    N = len(agent_cols)
    # K = unique markets used
    used = set()
    for _, row in results_market[agent_cols].iterrows():
        for v in row.values:
            try:
                used.add(int(v))
            except Exception:
                pass
    markets = sorted(used)
    return len(markets), N


def build_market_and_profit_tables(results_market: pd.DataFrame,
                                   results_profits: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table with, per (Round, Iteration):
      - mean profit (overall and per market)
      - count (number of agents) per market
      - changes: % agents who changed decision vs previous iteration
    Columns follow the pattern:
      mean, mean_g1..mean_gK, count_g1..count_gK, changes
    """
    # Identify agent columns
    agent_cols = [c for c in results_market.columns if isinstance(c, int)]
    N = len(agent_cols)

    # Calculate mean profit overall
    profits_calc = results_profits[agent_cols]
    market_calc = results_market[agent_cols]

    # Dictionary: market id -> 'g{m}'
    # We use numeric markets 1..K discovered from data
    markets_used = sorted(set(int(x) for x in pd.unique(market_calc.values.ravel()) if pd.notna(x)))
    mg_dict = {m: f"g{m}" for m in markets_used}

    out = results_profits[['Round', 'Iteration']].copy()
    out['mean'] = profits_calc.mean(axis=1)

    # Per-market mean profit and counts
    for m in markets_used:
        mask = market_calc.eq(m)
        out[f"mean_{mg_dict[m]}"] = profits_calc.where(mask).mean(axis=1)
        out[f"count_{mg_dict[m]}"] = market_calc.where(mask).count(axis=1)

    # % of agents who changed decision vs previous iteration (same logic you had)
    # (we set last iter of each Round to NaN, so plot breaks at episode boundary)
    diffs = []
    for rnd in sorted(out['Round'].unique()):
        idx = out['Round'] == rnd
        m_r = market_calc[idx]
        # shift and compare row-wise
        same = (m_r.shift(-1).iloc[:-1].reset_index(drop=True) == m_r.iloc[:-1].reset_index(drop=True)).sum(axis=1)
        diff_pct = (N - same)  # count of changed decisions
        # align back to original indices
        ser = pd.Series(np.nan, index=m_r.index)
        ser.iloc[:-1] = diff_pct.values
        diffs.append(ser)
    out['changes'] = pd.concat(diffs).sort_index()
    # Convert to % of agents
    out['changes'] = out['changes'] / N * 100.0
    out['stability'] = 1.0 - out['changes'] / 100.0

    return out, markets_used, N


def group_by_iteration_means(summary_df: pd.DataFrame, markets_used: list[int]) -> pd.DataFrame:
    """
    Aggregate by Iteration, obtaining:
      mean, mean_g1..mean_gK, count_g1..count_gK, changes
    """
    cols_keep = ['Iteration', 'mean', 'changes', 'stability'] + \
                [f"mean_g{m}" for m in markets_used] + \
                [f"count_g{m}" for m in markets_used]

    agg = summary_df[cols_keep].groupby('Iteration').mean(numeric_only=True)

    agg = agg[['mean'] +
            [f"mean_g{m}" for m in markets_used] +
            [f"count_g{m}" for m in markets_used] +
            ['changes', 'stability']]
    return agg

def build_potential_spec(N: int, markets_used: list[int]) -> tuple[list[int], List[Callable[[int], float]], List[float]]:
    """
    Build (markets_ordered, p_funcs, costs) consistent with the simulation config.
    """
    spec = base_spec(N)

    # If there is a market id not in base_spec, assume it is the gov market
    # added via add_constant_market with price=1.0, cost=0.0
    max_m = max(markets_used)
    if max_m not in spec.markets:
        spec = add_constant_market(spec, price=1.0, cost=0.0)

    markets_ordered = sorted(markets_used)
    costs = [spec.costs[m] for m in markets_ordered]

    # Wrap config.price_funcs[m](n, s, N) into p(n) that only depends on n (N is fixed)
    def make_p(m: int):
        def f(n: int, _m=m) -> float:
            # state argument is irrelevant here because your price functions
            # depend only on n and N
            return spec.price_funcs[_m](n, {}, N)
        return f

    p_funcs = [make_p(m) for m in markets_ordered]
    return markets_ordered, p_funcs, costs


def compute_efficiency_series(results_market: pd.DataFrame,
                              markets_used: list[int]) -> pd.Series:
    """
    For each row (Round, Iteration) compute potential ratio Phi / Phi_max
    based on the allocation of agents across markets.
    Returns a Series indexed like results_market, named 'efficiency'.
    """
    agent_cols = [c for c in results_market.columns if isinstance(c, int)]
    N = len(agent_cols)

    markets_ordered, p_funcs, costs = build_potential_spec(N, markets_used)
    _, phi_max = dp_potential_max(N, p_funcs, costs)
    if phi_max == 0:
        phi_max = 1.0

    ratios = []
    for _, row in results_market[agent_cols].iterrows():
        alloc = [(row == m).sum() for m in markets_ordered]
        phi = potential(p_funcs, costs, alloc)
        ratio = float(phi) / phi_max
        ratios.append(ratio)

    eff = pd.Series(ratios, index=results_market.index, name="efficiency")
    return eff


def compute_efficiency_by_iteration(results_market: pd.DataFrame,
                                    markets_used: list[int]) -> pd.Series:
    """
    Average potential efficiency by iteration (across episodes).
    """
    eff_series = compute_efficiency_series(results_market, markets_used)
    eff_df = results_market[['Round', 'Iteration']].copy()
    eff_df['efficiency'] = eff_series
    eff_by_iter = eff_df.groupby('Iteration')['efficiency'].mean()
    return eff_by_iter


def compute_agent_level_stats(results_market: pd.DataFrame, results_profits: pd.DataFrame):
    """
    For each episode (Round), compute per-agent:
      - average profit across iterations,
      - fraction of decision changes across iterations.
    Returns two lists of lists (one per Round).
    """
    agent_cols = [c for c in results_market.columns if isinstance(c, int)]
    rounds = sorted(results_market['Round'].unique())
    agent_avg_profits = []
    agent_avg_changes = []
    for r in rounds:
        mask_round = results_market['Round'] == r
 
        m = results_market.loc[mask_round, agent_cols].copy()
        p = results_profits.loc[mask_round, agent_cols].copy()

        if len(m) <= 1:
            # za mało kroków w ewaluacji, żeby policzyć zmiany
            agent_avg_profits.append(list(p.mean(axis=0)) if len(p) > 0 else [np.nan] * len(agent_cols))
            agent_avg_changes.append([0.0] * len(agent_cols))
            continue

        # average profit per agent
        agent_avg_profits.append(list(p.mean(axis=0)))

        # fraction of iterations where agent changed decision
        changed = 1 - ((m.shift(-1).iloc[:-1].reset_index(drop=True) ==
                        m.iloc[:-1].reset_index(drop=True)).sum(axis=0) / len(m.iloc[:-1]))
        agent_avg_changes.append(list(changed.fillna(0.0)))

    return agent_avg_profits, agent_avg_changes


def plot_number_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str, N, T):
    tmp = agg_iter[[f"count_g{m}" for m in markets_used]].copy()
    tmp.columns = [f"market_{m}" for m in markets_used]
    ax = tmp.plot(
        title=f'Average number of agents in the market by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        colormap=cm, kind='area', stacked=True, grid=True
    )
    ax.set_ylabel('Number of agents')
    ax.set_xlabel('Iteration')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_profit_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str, N, T):
    tmp = agg_iter[['mean'] + [f"mean_g{m}" for m in markets_used]].copy()
    tmp.columns = ['Total'] + [f"market_{m}" for m in markets_used]
    ax = tmp.plot(
        title=f'Average profit by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        colormap=cm, grid=True
    )
    # bold the Total line 
    for line in ax.get_lines():
        if line.get_label() == 'Total':
            line.set_linewidth(2)
            line.set_zorder(1)
        else:
            line.set_linewidth(1)
            line.set_zorder(0)
    ax.set_ylabel('Profit')
    ax.set_xlabel('Iteration')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_decision_changes(agg_iter: pd.DataFrame, filename: str, N, T):
    ax = agg_iter[['stability']].plot(
        title=f'Stability by iteration \n(n_Agents = {N}, n_Episodes = {T})',
        colormap=cm, grid=True
    )
    ax.set_ylabel('Stability (fraction of agents not changing market)')
    ax.set_xlabel('Iteration')
    ax.set_ylim(0.0, 1.0)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_iter(eff_by_iter: pd.Series, filename: str, N, T):
    """
    Plot average potential efficiency (Phi / Phi_max) by iteration (round).
    """
    ax = eff_by_iter.plot(
        title=f'Average potential efficiency by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        colormap=cm,
        grid=True
    )
    ax.set_ylabel('Potential ratio (Phi / Phi_max)')
    ax.set_xlabel('Iteration')
    ax.set_ylim(0.0, 1.05)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_and_stability_iter(agg_iter: pd.DataFrame,
                                       eff_by_iter: pd.Series,
                                       filename: str,
                                       N,
                                       T):
    """
    Plot efficiency and stability over iterations.
    """
    # Build a DataFrame indexed by iteration
    df = pd.DataFrame({
        'efficiency': eff_by_iter,
        'stability': agg_iter['stability']
    })

    # Remove rows where any of the series is NaN
    df = df.dropna(subset=['efficiency', 'stability'])

    ax = df[['efficiency', 'stability']].plot(
        title=f'Efficiency and stability by iteration\n(n_Agents = {N}, n_Episodes = {T})',
        grid=True
    )

    ax.set_xlabel('Iteration')
    ax.set_ylim(0.0, 1.0)

    ax.legend(['Efficiency (Phi / Phi_max)', 'Stability (1 = no agent changed)'])

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_profits_vs_changes(agent_avg_profits, agent_avg_changes, filename: str):
    # Flatten nested lists
    def flatten(xss):
        return [x for xs in xss for x in xs]

    # agent_avg_changes is a fraction in [0,1]: share of iterations with a change
    changes = flatten(agent_avg_changes)
    stability = [1.0 - v for v in changes]  # 1 = no change, 0 = always changed
    y = flatten(agent_avg_profits)

    ax = sns.regplot(
        x=stability,
        y=y,
        scatter_kws={'s': 1, 'color': '#24325F'},
        line_kws=dict(color='#FB6467')
    )
    ax.set(
        title='Profits vs. stability \n(each dot represents a single agent in one episode)'
    )
    ax.set_xlabel('Stability (fraction of agents not changing market)')
    ax.set_ylabel('Profit')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_efficiency_vs_changes(summary_df: pd.DataFrame,
                               eff_series: pd.Series,
                               filename: str):
    """
    Scatter: each point is one (Round, Iteration).
    X axis: potential ratio Phi / Phi_max.
    Y axis: stability in [0,1], where 1 means no agent changed market.
    """
    # summary_df is expected to contain 'stability' computed earlier
    df = summary_df[['Iteration', 'stability']].copy()
    df['efficiency'] = eff_series.reindex(df.index)

    # Drop rows with NaN (e.g., last iteration in each round)
    df = df.dropna(subset=['stability', 'efficiency'])

    x = df['efficiency']
    y = df['stability']

    ax = sns.regplot(
        x=x,
        y=y,
        scatter_kws={'s': 1, 'color': '#24325F'},
        line_kws=dict(color='#FB6467')
    )
    ax.set_title('Efficiency vs. stability \n(each dot is one iteration in one episode)')
    ax.set_xlabel('Efficiency (potential ratio)')
    ax.set_ylabel('Stability (fraction of agents not changing market)')

    plt.ylim(0.0, 1.0)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_efficiency_hist_last_iter(eff_series: pd.Series,
                                   results_market: pd.DataFrame,
                                   filename: str):
    """
    Plot mean efficiency across all iterations for each episode.
    Display the mean efficiency value on top of each bar.
    """

    # Build dataframe linking Round <-> Efficiency
    df = pd.DataFrame({
        'Round': results_market['Round'],
        'Efficiency': eff_series
    }).dropna()

    # Compute mean efficiency per episode  <-- CHANGED
    df_mean = df.groupby('Round', as_index=False)['Efficiency'].mean().sort_values('Round')

    # Extract values
    rounds = df_mean['Round'].values
    values = df_mean['Efficiency'].values
    x = np.arange(len(rounds))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, values, alpha=0.7, color="#24325F")

    # Add efficiency labels above each bar (rounded)
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        eff = round(values[idx], 2)      # keep your label style
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            str(eff),                    # efficiency on the label
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.xticks(x, [f"Ep {ep}" for ep in rounds])
    plt.ylim(0, 1.05)

    plt.title("Mean efficiency across all iterations (per episode)")
    plt.xlabel("Episode")
    plt.ylabel("Mean efficiency (Phi / Phi_max)")

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Summary of simulation results (plots & stats).")
    ap.add_argument("--market", help="Path to results_market_*.pickle")
    ap.add_argument("--profits", help="Path to results_profits_*.pickle")
    ap.add_argument("--outdir", help="Directory to save PNG plots")
    ap.add_argument(
        "--episode",
        type=int,
        help="If set, only this episode (Round index, starting at 1) will be analyzed and plotted."
    )
    args = ap.parse_args()


    # --- If no pickle paths are provided, automatically select the latest run ---
    if not args.market or not args.profits:
        base_dir = Path("results")
        if not base_dir.exists():
            print("Error: 'results/' directory not found. Run a simulation first.")
            sys.exit(1)

        # Find all subdirectories that look like run_YYYY-MM-DD_HH-MM-SS
        run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            print("Error: No result folders found in 'results/'.")
            sys.exit(1)

        latest_run = run_dirs[-1]
        print(f"Using the most recent results folder: {latest_run}")

        # Look for pickle files inside that folder
        market_files = sorted(latest_run.glob("results_market*.pickle"))
        profit_files = sorted(latest_run.glob("results_profits*.pickle"))

        if not market_files or not profit_files:
            print(f"Error: Missing .pickle files in {latest_run}")
            sys.exit(1)

        # Use the first matching pickle files
        args.market = str(market_files[0])
        args.profits = str(profit_files[0])

        # Default output directory: same folder as the results
        if not args.outdir:
            args.outdir = str(latest_run / "plots")

    # Load
    results_market = load_pickle(args.market)
    results_profits = load_pickle(args.profits)

    potential_path = Path(args.market).with_name(
        Path(args.market).name.replace("results_market", "results_potential")
    )

    df_potential = None
    if potential_path.exists():
        df_potential = load_pickle(potential_path)
        if not isinstance(df_potential, pd.DataFrame):
            df_potential = pd.DataFrame(df_potential)
        print(f"Loaded dynamic potential from {potential_path.name}")
    else:
        print("Warning: results_potential*.pickle not found. Falling back to static potential.")

    # If raw lists were saved, coerce to DataFrame
    if not isinstance(results_market, pd.DataFrame):
        results_market = pd.DataFrame(results_market)
    if not isinstance(results_profits, pd.DataFrame):
        results_profits = pd.DataFrame(results_profits)

    # Optional: restrict analysis to a single episode (Round)
    if args.episode is not None:
        ep = args.episode
        available_rounds = results_market['Round'].unique()

        if ep not in available_rounds:
            print(f"Error: requested episode (Round) {ep} not found in data. Available rounds: {sorted(available_rounds)}")
            sys.exit(1)

        # Filter both market and profits data to the selected episode
        results_market = results_market[results_market['Round'] == ep].copy()
        results_profits = results_profits[results_profits['Round'] == ep].copy()

        print(f"Running summary for a single episode: Round = {ep}")
    else:
        ep = None

    # --- Prepare output directory ---
    market_path = Path(args.market)

    # Base output dir: use args.outdir if given, otherwise "<run_dir>/plots"
    if args.outdir:
        base_outdir = Path(args.outdir)
    else:
        base_outdir = market_path.parent / "plots"

    # If a single episode is selected, put plots into a subdirectory "episode_<ep>"
    if ep is not None:
        outdir = base_outdir / f"episode_{ep}"
    else:
        outdir = base_outdir

    outdir.mkdir(parents=True, exist_ok=True)

    # Basic info
    K, N = infer_K_and_N(results_market)
    T = int(results_market['Round'].nunique())
    T_iter = int(results_market['Iteration'].nunique())

    # Build per-iteration table
    summary_df, markets_used, N = build_market_and_profit_tables(results_market, results_profits)

    # Aggregate by Iteration
    agg_iter = group_by_iteration_means(summary_df, markets_used)

    if df_potential is not None:
        key = ['Round', 'Iteration']
        tmp = results_market[key].merge(
            df_potential[key + ['Phi_ratio']],
            on=key,
            how='left'
        )
        eff_series = tmp['Phi_ratio']
        eff_series.name = "efficiency"

        eff_df = results_market[['Iteration']].copy()
        eff_df['efficiency'] = eff_series
        eff_by_iter = eff_df.groupby('Iteration')['efficiency'].mean()
    else:
        eff_series = compute_efficiency_series(results_market, markets_used)
        eff_by_iter = compute_efficiency_by_iteration(results_market, markets_used)

    # Agent-level stats for the scatter+reg
    agent_avg_profits, agent_avg_changes = compute_agent_level_stats(results_market, results_profits)

    # === Plots ===
    plot_number_iter(agg_iter, markets_used, outdir / 'art_Number_Iter.png', N, T)
    plot_profit_iter(agg_iter, markets_used, outdir / 'art_Profit_Iter.png', N, T)
    # plot_decision_changes(agg_iter, outdir / 'art_Decision_chg.png', N, T)
    # plot_efficiency_iter(eff_by_iter, outdir / 'art_Efficiency_Iter.png', N, T)
    plot_efficiency_and_stability_iter(agg_iter, eff_by_iter, outdir / 'art_Efficiency_Stability_Iter.png', N, T)
    plot_profits_vs_changes(agent_avg_profits, agent_avg_changes, outdir / 'art_Profits_chg.png')
    plot_efficiency_vs_changes(summary_df, eff_series, outdir / 'art_Efficiency_vs_Changes.png')
    plot_efficiency_hist_last_iter(eff_series, results_market, outdir / 'art_Efficiency_LastIter_Hist.png')

    print(f"Saved plots to: {outdir}")

   # === Diagnostic sheet: one row per (Round, Iteration) ===

    agent_cols = [c for c in results_profits.columns if isinstance(c, int)]

    # Basic profit stats per row (all agents)
    profit_stats = pd.DataFrame(index=results_profits.index)
    profit_stats['mean_profit'] = results_profits[agent_cols].mean(axis=1)
    profit_stats['min_profit'] = results_profits[agent_cols].min(axis=1)
    profit_stats['max_profit'] = results_profits[agent_cols].max(axis=1)
    profit_stats['std_profit'] = results_profits[agent_cols].std(axis=1)

    # Base diagnostic frame
    diag = pd.DataFrame({
        'Round': summary_df['Round'],
        'Iteration': summary_df['Iteration'],
        '%changes': summary_df['changes'],
    })

    diag['mean_profit'] = profit_stats['mean_profit']
    diag['min_profit'] = profit_stats['min_profit']
    diag['max_profit'] = profit_stats['max_profit']
    diag['std_profit'] = profit_stats['std_profit']

    # Add dynamic potential if available
    if df_potential is not None:
        key = ['Round', 'Iteration']
        merged = summary_df[key].merge(
            df_potential[key + ['Phi', 'Phi_max', 'Phi_ratio']],
            on=key,
            how='left'
        )

        diag['Phi'] = merged['Phi']
        diag['Phi_max'] = merged['Phi_max']
        diag['efficiency'] = merged['Phi_ratio']   # DYNAMICZNE efficiency
    else:
        diag['efficiency'] = eff_series.reindex(summary_df.index)  # fallback

    # === Per-market counts and average profits ===
    # markets_used: e.g. [1, 2, 3, 4] or [1,2,3,4,5] (with gov)
    market_data = results_market[agent_cols]
    profit_data = results_profits[agent_cols]

    for m in sorted(markets_used):
        # mask: which agents are on market m in each (Round, Iteration)
        mask_m = (market_data == m)

        # number of agents on this market
        n_m = mask_m.sum(axis=1)

        # sum of profits for agents on this market
        sum_profit_m = profit_data.where(mask_m).sum(axis=1)

        # average profit for agents on this market
        mean_profit_m = sum_profit_m / n_m.replace(0, np.nan)

        diag[f'n_market_{m}'] = n_m
        diag[f'mean_profit_market_{m}'] = mean_profit_m

    # Save as CSV (easy to explore in any spreadsheet tool)
    diag_filename = outdir / "diagnostic_episodes.csv"

    diag.to_csv(
        diag_filename,
        index=False,
        sep=';',        
        decimal=',',  
        encoding='utf-8-sig'
    )
    print(f"Saved diagnostic sheet to: {diag_filename}")

if __name__ == "__main__":
    main()
