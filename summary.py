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

    return out, markets_used, N


def group_by_iteration_means(summary_df: pd.DataFrame, markets_used: list[int]) -> pd.DataFrame:
    """
    Aggregate by Iteration, obtaining:
      mean, mean_g1..mean_gK, count_g1..count_gK, changes
    """
    cols_keep = ['Iteration', 'mean', 'changes'] + \
                [f"mean_g{m}" for m in markets_used] + \
                [f"count_g{m}" for m in markets_used]
    agg = summary_df[cols_keep].groupby('Iteration').mean(numeric_only=True)
    # keep column order
    agg = agg[['mean'] + [f"mean_g{m}" for m in markets_used] +
              [f"count_g{m}" for m in markets_used] + ['changes']]
    return agg

def build_potential_spec(N: int, markets_used: list[int]) -> tuple[list[int], List[Callable[[int], float]], List[float]]:
    """
    Build (markets_ordered, p_funcs, costs) consistent with the simulation config.
    markets_ordered: sorted market ids actually used in the data.
    p_funcs: list of p_i(n) functions (price as a function of local n and fixed N).
    costs: list of c_i in the same order.
    """
    # Start from the base 4-market spec
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


def compute_agent_level_stats(results_market: pd.DataFrame, results_profits: pd.DataFrame,
                              eval_start_iter: int):
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
        if eval_start_iter is not None:
            mask_round &= results_market['Iteration'] >= eval_start_iter

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


def plot_number_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str, N, T, eval_start_iter: int):
    tmp = agg_iter[[f"count_g{m}" for m in markets_used]].copy()
    tmp.columns = [f"market_{m}" for m in markets_used]
    ax = tmp.plot(
        title=f'Average number of agents in the market by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        colormap=cm, kind='area', stacked=True, grid=True
    )
    ax.set_ylabel('Number of agents')
    ax.set_xlabel('Iteration')
    ax.axvspan(0, eval_start_iter - 1, alpha=0.35, color='lightgrey')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_profit_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str, N, T, eval_start_iter: int):
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
    ax.axvspan(0, eval_start_iter - 1, alpha=0.35, color='lightgrey')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_decision_changes(agg_iter: pd.DataFrame, filename: str, N, T, eval_start_iter: int):
    ax = agg_iter[['changes']].plot(
        title=f'Average decision changes by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        colormap=cm, grid=True
    )
    ax.set_ylabel('Stability (% of agents who changed their decision)')
    ax.set_xlabel('Iteration')
    ax.axvspan(0, eval_start_iter - 1, alpha=0.35, color='lightgrey')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_iter(eff_by_iter: pd.Series, filename: str, N, T, eval_start_iter: int):
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
    ax.axvspan(0, eval_start_iter - 1, alpha=0.35, color='lightgrey')

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_profits_vs_changes(agent_avg_profits, agent_avg_changes, filename: str):
    # Flatten 
    def flatten(xss):
        return [x for xs in xss for x in xs]

    x = [v * 100 for v in flatten(agent_avg_changes)]  # % changes
    y = flatten(agent_avg_profits)

    ax = sns.regplot(
        x=x, y=y,
        scatter_kws={'s': 1, 'color': '#24325F'},
        line_kws=dict(color='#FB6467')
    )
    ax.set(title='Profits vs. % of changed decisions \n (each dot represents single agent in one evaluation period)')
    ax.set_ylabel('Profit')
    ax.set_xlabel('% of changed decisions')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_vs_changes(summary_df: pd.DataFrame,
                               eff_series: pd.Series,
                               filename: str, eval_start_iter: int):
    """
    Scatter: each point is one (Round, Iteration).
    X axis: potential ratio Phi / Phi_max.
    Y axis: % of agents who changed market between this and next iteration.
    """
    df = summary_df[['Iteration', 'changes']].copy()
    df['efficiency'] = eff_series.reindex(df.index)

    if eval_start_iter is not None:
        df = df[df['Iteration'] >= eval_start_iter]

    # Drop rows with NaN (np. ostatnia iteracja epizodu)
    df = df.dropna(subset=['changes', 'efficiency'])

    x = df['efficiency']
    y = df['changes']

    ax = sns.regplot(
        x=x,
        y=y,
        scatter_kws={'s': 1, 'color': '#24325F'},
        line_kws=dict(color='#FB6467')
    )
    ax.set_title('Efficiency vs. stability \n (each dot is one evaluation period in one episode)')
    ax.set_xlabel('Efficiency (potential ratio)')
    ax.set_ylabel('Stability (% of agents who changed market)')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_hist_last_iter(eff_series: pd.Series,
                                   results_market: pd.DataFrame,
                                   filename: str):
    """
    Histogram of efficiency (Phi / Phi_max) in the final iteration of each episode.
    One observation per episode.
    """
    df = results_market[['Round', 'Iteration']].copy()
    df['efficiency'] = eff_series.reindex(df.index)

    # Last iteration index (assumed common across episodes)
    last_iter = df['Iteration'].max()

    eff_last = df.loc[df['Iteration'] == last_iter, 'efficiency'].dropna()

    plt.hist(eff_last, bins=20, edgecolor='black')
    plt.title('Distribution of efficiency in the final iteration')
    plt.xlabel('Potential ratio (Phi / Phi_max)')
    plt.ylabel('Number of episodes')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()




def main():
    ap = argparse.ArgumentParser(description="Summary of simulation results (plots & stats).")
    ap.add_argument("--market", help="Path to results_market_*.pickle")
    ap.add_argument("--profits", help="Path to results_profits_*.pickle")
    ap.add_argument("--outdir", help="Directory to save PNG plots")
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

    # If raw lists were saved, coerce to DataFrame
    if not isinstance(results_market, pd.DataFrame):
        results_market = pd.DataFrame(results_market)
    if not isinstance(results_profits, pd.DataFrame):
        results_profits = pd.DataFrame(results_profits)

    # --- Prepare output directory ---
    market_path = Path(args.market)
    outdir = Path(args.outdir or market_path.parent)
    outdir.mkdir(parents=True, exist_ok=True)

    # Basic info
    K, N = infer_K_and_N(results_market)
    T = int(results_market['Round'].nunique())
    T_iter = int(results_market['Iteration'].nunique())

    EVAL_FRACTION = 0.1
    train_iters = int((1.0 - EVAL_FRACTION) * T_iter)
    eval_start_iter = train_iters + 1 

    # Build per-iteration table
    summary_df, markets_used, N = build_market_and_profit_tables(results_market, results_profits)

    # Aggregate by Iteration
    agg_iter = group_by_iteration_means(summary_df, markets_used)

    # Efficiency (Phi / Phi_max) per (Round, Iteration)
    eff_series = compute_efficiency_series(results_market, markets_used)

    # Efficiency (Rosenthal potential ratio) per iteration, averaged across episodes
    eff_by_iter = compute_efficiency_by_iteration(results_market, markets_used)

    # Agent-level stats for the scatter+reg
    agent_avg_profits, agent_avg_changes = compute_agent_level_stats(results_market, results_profits, eval_start_iter = eval_start_iter)

    # === Plots ===
    plot_number_iter(agg_iter, markets_used, outdir / 'art_Number_Iter.png', N, T, eval_start_iter = eval_start_iter)
    plot_profit_iter(agg_iter, markets_used, outdir / 'art_Profit_Iter.png', N, T, eval_start_iter = eval_start_iter)
    plot_decision_changes(agg_iter, outdir / 'art_Decision_chg.png', N, T, eval_start_iter = eval_start_iter)
    plot_efficiency_iter(eff_by_iter, outdir / 'art_Efficiency_Iter.png', N, T, eval_start_iter = eval_start_iter)
    plot_profits_vs_changes(agent_avg_profits, agent_avg_changes, outdir / 'art_Profits_chg.png')
    plot_efficiency_vs_changes(summary_df, eff_series, outdir / 'art_Efficiency_vs_Changes.png', eval_start_iter = eval_start_iter)
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

    diag['efficiency'] = eff_series.reindex(diag.index)

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
