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

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from matplotlib.colors import ListedColormap

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
        m = results_market.loc[results_market['Round'] == r, agent_cols].copy()
        p = results_profits.loc[results_profits['Round'] == r, agent_cols].copy()

        # average profit per agent
        agent_avg_profits.append(list(p.mean(axis=0)))

        # fraction of iterations where agent changed decision
        changed = 1 - ((m.shift(-1).iloc[:-1].reset_index(drop=True) ==
                        m.iloc[:-1].reset_index(drop=True)).sum(axis=0) / len(m.iloc[:-1]))
        agent_avg_changes.append(list(changed.fillna(0.0)))
    return agent_avg_profits, agent_avg_changes


def plot_number_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str):
    tmp = agg_iter[[f"count_g{m}" for m in markets_used]].copy()
    tmp.columns = [f"market_{m}" for m in markets_used]
    ax = tmp.plot(
        title='Average number of agents in the market by round \n (n_Agents = 100, n_Iterations = 100)',
        colormap=cm, kind='area', stacked=True, grid=True
    )
    ax.set_ylabel('Number of agents')
    ax.set_xlabel('Round')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_profit_iter(agg_iter: pd.DataFrame, markets_used: list[int], filename: str):
    tmp = agg_iter[['mean'] + [f"mean_g{m}" for m in markets_used]].copy()
    tmp.columns = ['Total'] + [f"market_{m}" for m in markets_used]
    ax = tmp.plot(
        title='Average profit by round \n (n_Agents = 100, n_Iterations = 100)',
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
    ax.set_xlabel('Round')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_decision_changes(agg_iter: pd.DataFrame, filename: str):
    ax = agg_iter[['changes']].plot(
        title='Average decision changes by round \n (n_Agents = 100, n_Iterations = 100)',
        colormap=cm, grid=True
    )
    ax.set_ylabel('% of agents who changed their decision')
    ax.set_xlabel('Round')
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
    ax.set(title='Profits vs. % of changed decisions \n (each dot represents single agent in one experiment run)')
    ax.set_ylabel('Profit')
    ax.set_xlabel('% of changed decisions')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Pretty summary replicating original plots (numeric markets 1..K).")
    ap.add_argument("--market", required=True, help="Path to results_market_*.pickle")
    ap.add_argument("--profits", required=True, help="Path to results_profits_*.pickle")
    args = ap.parse_args()

    # Load
    results_market = load_pickle(args.market)
    results_profits = load_pickle(args.profits)

    # If raw lists were saved, coerce to DataFrame
    if not isinstance(results_market, pd.DataFrame):
        results_market = pd.DataFrame(results_market)
    if not isinstance(results_profits, pd.DataFrame):
        results_profits = pd.DataFrame(results_profits)

    # Basic info
    K, N = infer_K_and_N(results_market)

    # Build per-iteration table with the same columns you used
    summary_df, markets_used, N = build_market_and_profit_tables(results_market, results_profits)

    # Aggregate by Iteration
    agg_iter = group_by_iteration_means(summary_df, markets_used)

    # Agent-level stats for the scatter+reg
    agent_avg_profits, agent_avg_changes = compute_agent_level_stats(results_market, results_profits)

    # === Plots ===
    plot_number_iter(agg_iter, markets_used, 'art_Number_Iter.png')          # (gov variant used 'art_gov_Number_Iter.png')
    plot_profit_iter(agg_iter, markets_used, 'art_Profit_Iter.png')          # (gov: 'art_gov_Profit_Iter.png')
    plot_decision_changes(agg_iter, 'art_Decision_chg.png')                  # (gov: 'art_gov_Decision_chg.png')
    plot_profits_vs_changes(agent_avg_profits, agent_avg_changes, 'art_Profits_chg.png')  # (gov: 'art_gov_Profits_chg.png')

    print(f"Done. Saved: art_Number_Iter.png, art_Profit_Iter.png, art_Decision_chg.png, art_Profits_chg.png")


if __name__ == "__main__":
    main()
