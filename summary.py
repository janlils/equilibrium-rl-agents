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
import json  # for loading scenario metadata from config.json
from typing import List, Callable, Optional, Dict, Any  # if not already imported



plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True
})

colors_hex = [ '#24325F', '#82491E', '#B7E4F9', '#E89242','#FB6467', '#69C8EC']
cm = ListedColormap(colors_hex)

EFFICIENCY_THRESHOLD = 0.99
EQUILIBRIUM_WINDOW = 5


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


def count_adjustment_iterations(
    efficiencies: np.ndarray,
    threshold: float = EFFICIENCY_THRESHOLD,
    window: int = EQUILIBRIUM_WINDOW,
) -> int:
    """
    Count how many iterations within a single episode are spent "adjusting"
    (outside equilibrium). An episode enters equilibrium once it records
    `window` consecutive efficiency values above the threshold and leaves it
    after `window` consecutive values below the threshold.
    """
    values = np.asarray(efficiencies, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return 0

    efficient = values > threshold
    eq_mask = np.zeros(n, dtype=bool)
    out_mask = np.zeros(n, dtype=bool)

    consec_eff = 0
    consec_ineff = 0
    for i, is_eff in enumerate(efficient):
        if is_eff:
            consec_eff += 1
        else:
            consec_eff = 0
        if consec_eff >= window:
            eq_mask[i] = True

        if not is_eff:
            consec_ineff += 1
        else:
            consec_ineff = 0
        if consec_ineff >= window:
            out_mask[i] = True

    adjusting = 0
    in_equilibrium = False
    adjusting_active = True  # start outside equilibrium until we enter it once

    for i in range(n):
        if eq_mask[i]:
            in_equilibrium = True
            adjusting_active = False

        if in_equilibrium and out_mask[i]:
            in_equilibrium = False
            adjusting_active = True

        if adjusting_active:
            adjusting += 1

    return adjusting


def compute_episode_efficiency_stats(
    results_market: pd.DataFrame,
    eff_series: pd.Series,
    threshold: float = EFFICIENCY_THRESHOLD,
    window: int = EQUILIBRIUM_WINDOW,
) -> pd.DataFrame:
    """
    Build a per-episode table with:
      - fraction of iterations above the efficiency threshold,
      - number of iterations spent adjusting (outside equilibrium).
    """
    df = pd.DataFrame({
        'Round': results_market['Round'].values,
        'efficiency': eff_series.values,
    })

    rows = []
    for rnd, grp in df.groupby('Round'):
        eff_vals = grp['efficiency'].dropna().to_numpy(dtype=float)
        total_iters = len(eff_vals)

        if total_iters == 0:
            rows.append({
                'Round': rnd,
                'iterations_total': 0,
                'efficient_iterations': 0,
                'efficient_pct': np.nan,
                'adjustment_iterations': 0,
                'adjustment_pct': np.nan,
            })
            continue

        efficient_mask = eff_vals > threshold
        efficient_iters = int(efficient_mask.sum())
        efficient_pct = (efficient_iters / total_iters) * 100.0
        adjustment_iters = count_adjustment_iterations(eff_vals, threshold, window)
        adjustment_pct = (adjustment_iters / total_iters) * 100.0 if total_iters > 0 else np.nan

        rows.append({
            'Round': rnd,
            'iterations_total': total_iters,
            'efficient_iterations': efficient_iters,
            'efficient_pct': efficient_pct,
            'adjustment_iterations': adjustment_iters,
            'adjustment_pct': adjustment_pct,
        })

    return pd.DataFrame(rows).sort_values('Round').reset_index(drop=True)

def plot_allocation_with_optimal(
    agg_iter: pd.DataFrame,
    df_potential: Optional[pd.DataFrame],
    markets_used: List[int],
    filename: str,
    N: int,
    T: int,
    scenario_events: Optional[List[Dict[str, Any]]] = None,
    scenario_name: Optional[str] = None,
):
    """
    Line plot of average number of agents per market by iteration, with
    optional overlay of optimal allocation and vertical lines for shocks.

    - x-axis: Iteration
    - y-axis: Number of agents
    - solid lines: actual RL allocation (avg across episodes)
    - dashed / lighter lines: optimal allocation (avg across episodes)
    - vertical dashed lines: shock events from the scenario definition
    """
    # Prepare x-axis from aggregated data
    x = agg_iter.index.values

    # Prepare optimal allocation aggregated by Iteration if available
    opt_by_iter = None
    if df_potential is not None:
        # We expect columns: 'Iteration', 'n_market_m', 'opt_n_market_m'
        cols = ['Iteration']
        for m in markets_used:
            cols.append(f"n_market_{m}")
            cols.append(f"opt_n_market_{m}")

        existing = [c for c in cols if c in df_potential.columns]
        if len(existing) > 1:  # at least Iteration + something
            opt_by_iter = (
                df_potential[existing]
                .groupby('Iteration')
                .mean(numeric_only=True)
            )

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot actual allocation for each market
    for i, m in enumerate(sorted(markets_used)):
        color = colors_hex[i % len(colors_hex)]
        col_actual = f"count_g{m}"
        if col_actual not in agg_iter.columns:
            continue

        y_actual = agg_iter[col_actual].values

        # Actual RL allocation: thinner, slightly transparent line
        ax.plot(
            x,
            y_actual,
            label=f"market_{m} actual",
            linewidth=1.0,
            color=color,
            alpha=0.7,
        )

        # Optional: optimal allocation overlay, more visible
        if opt_by_iter is not None:
            col_opt = f"opt_n_market_{m}"
            if col_opt in opt_by_iter.columns:
                # Align by Iteration; reindex on agg_iter.index
                y_opt = opt_by_iter[col_opt].reindex(agg_iter.index).values
                ax.plot(
                    x,
                    y_opt,
                    linewidth=3.5,             # thicker outline
                    linestyle="--",
                    color="black",             # outline color
                    alpha=0.8,
                    zorder=2,
                )
                ax.plot(
                    x,
                    y_opt,
                    label=f"market_{m} optimal",
                    linewidth=2.0,          # thicker line for optimal
                    linestyle="--",
                    color=color,
                    alpha=0.9,
                )

    # Collect shock markers (iteration, label)
    shock_marks = []
    if scenario_events:
        for ev in scenario_events:
            when = ev.get("when", {})
            it = None

            # Single-iteration shock
            if "iter" in when:
                it = int(when["iter"])
            # From-iteration shock: mark the start
            elif "from_iter" in when:
                it = int(when["from_iter"])

            if it is None:
                continue

            # Determine event type: first key that is not "when"
            event_type = next((k for k in ev.keys() if k != "when"), None)
            if event_type is None:
                event_type = "shock"

            # Try to infer market for label, e.g. "price_bump_m2", "inflation_m3"
            market_id = None
            payload = ev.get(event_type, {})

            if isinstance(payload, dict):
                # Case 1: explicit "market" field, e.g. {"market": 3, "rate": ...}
                if "market" in payload:
                    market_id = payload["market"]
                else:
                    # Case 2: dict keyed by market id, e.g. {2: +5.0} or {4: +4}
                    # Take the first key that looks like a market identifier
                    for k in payload.keys():
                        # Try to treat numeric or numeric-string keys as market IDs
                        if isinstance(k, int):
                            market_id = k
                            break
                        if isinstance(k, str) and k.isdigit():
                            market_id = int(k)
                            break

            # Build compact label: event_type[_mX]
            if market_id is not None:
                label = f"{event_type}_m{market_id}"
            else:
                label = event_type

            shock_marks.append((it, label))

    # Draw vertical lines and labels for shocks
    if shock_marks:
        ax.set_ylim(0, 50)
        # Get current limits based on data before adding text
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0

        # Place labels exactly at the top boundary of the plot (ymax)
        label_y = ymax
        x_offset = max(0.005 * (ax.get_xlim()[1] - ax.get_xlim()[0]), 0.2)

        for it, label in shock_marks:
            ax.axvline(
                x=it,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
            ax.text(
                it - x_offset,
                label_y,
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                alpha=0.8,
            )

    ax.set_ylim(0, 50)

    ax.set_title(
        f"Average allocation per market by iteration\n"
        f"(n_Agents = {N}, n_Episodes = {T})"
    )
    ax.set_ylabel("Number of agents")
    ax.set_xlabel("Iteration")

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


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
    ax.set_ylim(-10.0, 10.0)

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

def plot_theta_iter(df_theta: pd.DataFrame, filename: str, N: int, T: int):
    """
    Plot average Q-parameter values by iteration.
    """
    param_cols = [c for c in df_theta.columns if c not in ('Round', 'Iteration')]
    if not param_cols:
        return

    agg = (
        df_theta[['Iteration'] + param_cols]
        .groupby('Iteration')
        .mean(numeric_only=True)
    )

    ax = agg.plot(
        title=f'Average Q-parameters by iteration \n (n_Agents = {N}, n_Episodes = {T})',
        grid=True
    )
    ax.set_ylabel('Average parameter value')
    ax.set_xlabel('Iteration')

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
        eff = round(values[idx], 4)      # keep your label style
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
    ap.add_argument(
        "--exp_id",
        help="Experiment ID used in run directory name (run_<timestamp>_<exp_id>[_...]). "
             "Used only when --market/--profits are not provided."
    )
    args = ap.parse_args()

    # --- If no pickle paths are provided, automatically select the latest run ---
    if not args.market or not args.profits:
        base_dir = Path("results")
        if not base_dir.exists():
            print("Error: 'results/' directory not found. Run a simulation first.")
            sys.exit(1)

        # Find all subdirectories that start with "run_"
        run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            print("Error: No result folders found in 'results/'.")
            sys.exit(1)

        # If experiment ID is provided, filter runs by this ID
        if args.exp_id:
            matching = [d for d in run_dirs if f"_{args.exp_id}" in d.name]
            if not matching:
                print(f"Error: no run directory found for exp_id='{args.exp_id}'.")
                print("Available run directories:")
                for d in run_dirs:
                    print("  -", d.name)
                sys.exit(1)
            latest_run = matching[-1]
            print(f"Using run folder for exp_id='{args.exp_id}': {latest_run}")
        else:
            # Fallback: most recent run by name
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

    theta_path = Path(args.market).with_name(
        Path(args.market).name.replace("results_market", "results_theta")
    )
    df_theta = None
    if theta_path.exists():
        df_theta = load_pickle(theta_path)
        if not isinstance(df_theta, pd.DataFrame):
            df_theta = pd.DataFrame(df_theta)
        print(f"Loaded Q-parameters from {theta_path.name}")
    else:
        print("Warning: results_theta*.pickle not found. Skipping Q-parameter plot.")

    # Try to load scenario metadata (scenario events and name) from config.json
    scenario_events: List[Dict[str, Any]] = []
    scenario_name: Optional[str] = None

    try:
        run_dir = Path(args.market).parent
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Expected structure written by experiments.py
            scenario_events = meta.get("scenario", []) or []
            scenario_name = meta.get("scenario_name") or meta.get("exp_id")
            print(f"Loaded scenario metadata from {cfg_path.name}")
        else:
            print("No config.json found in run directory; skipping shock markers.")
    except Exception as e:
        print(f"Warning: could not load scenario metadata: {e}")
        scenario_events = []
        scenario_name = None


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
        if df_theta is not None:
            df_theta = df_theta[df_theta['Round'] == ep].copy()

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
    # Allocation vs optimal allocation with scenario shocks
    plot_allocation_with_optimal(
        agg_iter=agg_iter,
        df_potential=df_potential,
        markets_used=markets_used,
        filename=outdir / "art_Allocation_Optimal_Iter.png",
        N=N,
        T=T,
        scenario_events=scenario_events,
        scenario_name=scenario_name,
    )
    if df_theta is not None and not df_theta.empty:
        plot_theta_iter(df_theta, outdir / 'art_Q_Params_Iter.png', N, T)

    episode_stats = compute_episode_efficiency_stats(
        results_market,
        eff_series,
        threshold=EFFICIENCY_THRESHOLD,
        window=EQUILIBRIUM_WINDOW,
    )



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

    # === Per-market actual counts and average profits ===
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

    # === Add dynamic potential, optimal allocations and prices if available ===
    if df_potential is not None:
        key = ['Round', 'Iteration']

        # Build list of columns to merge from df_potential
        cols_to_take = ['Phi', 'Phi_max', 'Phi_ratio']
        # Optional per-market columns (created in simulation.py)
        for m in sorted(markets_used):
            for col in (f"opt_n_market_{m}", f"price_market_{m}"):
                if col in df_potential.columns:
                    cols_to_take.append(col)

        merged = summary_df[key].merge(
            df_potential[key + cols_to_take],
            on=key,
            how='left'
        )

        if 'Phi' in merged.columns:
            diag['Phi'] = merged['Phi']
        if 'Phi_max' in merged.columns:
            diag['Phi_max'] = merged['Phi_max']
        if 'Phi_ratio' in merged.columns:
            diag['efficiency'] = merged['Phi_ratio']

        # Copy optimal allocations and prices per market, if present
        for m in sorted(markets_used):
            opt_col = f"opt_n_market_{m}"
            price_col = f"price_market_{m}"
            if opt_col in merged.columns:
                diag[opt_col] = merged[opt_col]
            if price_col in merged.columns:
                diag[price_col] = merged[price_col]
    else:
        # Fallback: only static efficiency available
        diag['efficiency'] = eff_series.reindex(summary_df.index)

    # === Reorder columns: base stats, then per-market blocks: actual, optimal, price, profit ===
    ordered_cols = ['Round', 'Iteration']

    if 'Phi' in diag.columns:
        ordered_cols.append('Phi')
    if 'Phi_max' in diag.columns:
        ordered_cols.append('Phi_max')

    ordered_cols += [
        'efficiency',
        '%changes',
        'mean_profit',
        'min_profit',
        'max_profit',
        'std_profit',
    ]

    for m in sorted(markets_used):
        n_col = f'n_market_{m}'
        opt_col = f"opt_n_market_{m}"
        price_col = f"price_market_{m}"
        mean_p_col = f"mean_profit_market_{m}"

        if n_col in diag.columns:
            ordered_cols.append(n_col)
        if opt_col in diag.columns:
            ordered_cols.append(opt_col)
        if price_col in diag.columns:
            ordered_cols.append(price_col)
        if mean_p_col in diag.columns:
            ordered_cols.append(mean_p_col)

    diag = diag[ordered_cols]

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

    episode_stats_filename = outdir / "episode_efficiency_stats.csv"
    episode_stats.to_csv(
        episode_stats_filename,
        index=False,
        sep=';',
        decimal=',',
        encoding='utf-8-sig'
    )
    print(f"Saved per-episode efficiency stats to: {episode_stats_filename}")

    if not episode_stats.empty:
        print("\nPer-episode efficiency stats:")
        print(episode_stats.to_string(index=False))
        avg_eff_pct = episode_stats['efficient_pct'].mean(skipna=True)
        avg_adj_iters = episode_stats['adjustment_iterations'].mean(skipna=True)
        avg_adj_pct = episode_stats['adjustment_pct'].mean(skipna=True)
        print(
            f"Average share of efficient iterations: "
            f"{avg_eff_pct:.2f}% | "
            f"Average adjustment time: {avg_adj_iters:.1f} iterations "
            f"({avg_adj_pct:.2f}% of iterations)"
        )


def run_summary(
    market: str | None = None,
    profits: str | None = None,
    outdir: str | None = None,
    episode: int | None = None,
    exp_id: str | None = None,
):
    """Programmatic wrapper around the CLI interface.

    It builds a fake sys.argv and calls main(), so that experiments.py
    can trigger the same logic without shelling out.
    """
    import sys

    argv = ["summary.py"]
    if market is not None:
        argv += ["--market", market]
    if profits is not None:
        argv += ["--profits", profits]
    if outdir is not None:
        argv += ["--outdir", outdir]
    if episode is not None:
        argv += ["--episode", str(episode)]
    if exp_id is not None:
        argv += ["--exp_id", exp_id]

    old_argv = sys.argv
    try:
        sys.argv = argv
        return main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
