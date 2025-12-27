import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

EFFICIENCY_THRESHOLD = 0.99
EQUILIBRIUM_WINDOW = 5


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_adjustment_iterations(
    efficiencies: np.ndarray,
    threshold: float = EFFICIENCY_THRESHOLD,
    window: int = EQUILIBRIUM_WINDOW,
) -> int:
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
        consec_eff = consec_eff + 1 if is_eff else 0
        if consec_eff >= window:
            eq_mask[i] = True

        consec_ineff = consec_ineff + 1 if not is_eff else 0
        if consec_ineff >= window:
            out_mask[i] = True

    adjusting = 0
    in_equilibrium = False
    adjusting_active = True

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


def compute_episode_stats(df_potential: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    df = df_potential[['Round', 'Iteration', 'Phi_ratio']].copy()
    df = df.dropna(subset=['Round'])

    for rnd, grp in df.groupby('Round'):
        grp_sorted = grp.sort_values('Iteration').reset_index(drop=True)
        eff_vals = grp_sorted['Phi_ratio'].dropna().to_numpy(dtype=float)
        total_iters = len(eff_vals)
        if total_iters == 0:
            rows.append({
                'Round': rnd,
                'iterations_total': 0,
                'efficient_iterations': 0,
                'efficient_pct': np.nan,
                'adjustment_iterations': 0,
                'adjustment_pct': np.nan,
                'mean_eff': np.nan,
                'min_eff': np.nan,
                'max_eff': np.nan,
                'std_eff': np.nan,
                'eq_start_iter': np.nan,
            })
            continue

        efficient_mask = eff_vals > EFFICIENCY_THRESHOLD
        efficient_iters = int(efficient_mask.sum())
        adjustment_iters = count_adjustment_iterations(eff_vals)
        eq_start_iter = np.nan
        consec = 0
        for it_val, is_eff in zip(grp_sorted['Iteration'], efficient_mask):
            consec = consec + 1 if is_eff else 0
            if consec >= EQUILIBRIUM_WINDOW:
                eq_start_iter = float(it_val - EQUILIBRIUM_WINDOW + 1)
                break

        rows.append({
            'Round': rnd,
            'iterations_total': total_iters,
            'efficient_iterations': efficient_iters,
            'efficient_pct': (efficient_iters / total_iters) * 100.0,
            'adjustment_iterations': adjustment_iters,
            'adjustment_pct': (adjustment_iters / total_iters) * 100.0,
            'mean_eff': float(np.mean(eff_vals)),
            'min_eff': float(np.min(eff_vals)),
            'max_eff': float(np.max(eff_vals)),
            'std_eff': float(np.std(eff_vals)),
            'var_eff': float(np.var(eff_vals)),
            'eq_start_iter': eq_start_iter,
        })

    return pd.DataFrame(rows).sort_values('Round').reset_index(drop=True)


def _find_first(path: Path, pattern: str) -> Optional[Path]:
    files = sorted(path.glob(pattern))
    return files[0] if files else None


def compute_mean_stability(df_market: pd.DataFrame) -> float:
    agent_cols = [c for c in df_market.columns if isinstance(c, int)]
    if not agent_cols:
        return float('nan')

    stabilities: List[float] = []
    for _, grp in df_market.groupby('Round'):
        grp_sorted = grp.sort_values('Iteration')
        if len(grp_sorted) <= 1:
            continue
        data = grp_sorted[agent_cols].fillna(-1).to_numpy()
        prev = data[:-1]
        nxt = data[1:]
        same = (prev == nxt).sum(axis=1) / float(len(agent_cols))
        stabilities.extend(same.tolist())

    if not stabilities:
        return float('nan')
    return float(np.mean(stabilities))


def summarize_run(run_dir: Path) -> Optional[dict]:
    pot_path = _find_first(run_dir, "results_potential*.pickle")
    profits_path = _find_first(run_dir, "results_profits*.pickle")
    market_path = _find_first(run_dir, "results_market*.pickle")
    if pot_path is None or profits_path is None or market_path is None:
        print(f"[WARN] Missing pickle files in {run_dir}. Skipping.")
        return None

    df_pot = load_pickle(pot_path)
    if not isinstance(df_pot, pd.DataFrame):
        df_pot = pd.DataFrame(df_pot)
    if 'Phi_ratio' not in df_pot.columns:
        print(f"[WARN] Phi_ratio not found in {pot_path.name}. Skipping {run_dir.name}.")
        return None

    stats = compute_episode_stats(df_pot)
    if stats.empty:
        print(f"[WARN] No efficiency stats for {run_dir.name}.")
        return None

    total_iters = stats['iterations_total'].sum()
    efficient_iters = stats['efficient_iterations'].sum()
    adjustment_iters = stats['adjustment_iterations'].sum()

    eff_pct = (efficient_iters / total_iters) * 100.0 if total_iters else np.nan
    adj_pct = (adjustment_iters / total_iters) * 100.0 if total_iters else np.nan

    mean_eff = stats['mean_eff'].mean()
    min_eff = stats['mean_eff'].min()
    max_eff = stats['mean_eff'].max()
    std_eff = stats['mean_eff'].std(ddof=0)
    var_eff = stats['var_eff'].mean()
    if 'eq_start_iter' in stats.columns:
        eq_start = float(stats['eq_start_iter'].mean(skipna=True))
    else:
        eq_start = np.nan

    df_pot_sorted = df_pot.sort_values(['Round', 'Iteration'])
    df_market = load_pickle(market_path)
    if not isinstance(df_market, pd.DataFrame):
        df_market = pd.DataFrame(df_market)
    mean_stability = compute_mean_stability(df_market)

    df_profits = load_pickle(profits_path)
    if not isinstance(df_profits, pd.DataFrame):
        df_profits = pd.DataFrame(df_profits)
    agent_cols = [c for c in df_profits.columns if isinstance(c, int)]
    if agent_cols:
        if {'Round', 'Iteration'}.issubset(df_profits.columns):
            profits_sorted = df_profits.sort_values(['Round', 'Iteration']).tail(100)
        else:
            profits_sorted = df_profits.tail(100)
        avg_profit = float(profits_sorted[agent_cols].to_numpy().mean())
    else:
        avg_profit = np.nan

    alloc_cols = [c for c in df_pot_sorted.columns if c.startswith("n_market_")]
    opt_cols = [c for c in df_pot_sorted.columns if c.startswith("opt_n_market_")]
    recent_pot = df_pot_sorted.tail(100)
    alloc_suffixes = [c.split('_')[-1] for c in alloc_cols]
    opt_suffixes = [c.split('_')[-1] for c in opt_cols]
    avg_alloc = {
        f"avg_alloc_{suffix}": float(recent_pot[f"n_market_{suffix}"].mean())
        for suffix in alloc_suffixes
    }
    avg_opt_alloc = {
        f"avg_opt_alloc_{suffix}": float(recent_pot[f"opt_n_market_{suffix}"].mean())
        for suffix in opt_suffixes
    }
    alloc_diff = 0.0
    for suffix in set(alloc_suffixes) & set(opt_suffixes):
        actual = recent_pot[f"n_market_{suffix}"]
        optimal = recent_pot[f"opt_n_market_{suffix}"]
        alloc_diff += float((actual - optimal).abs().sum())

    summary = {
        'run_id': run_dir.name,
        'efficient_pct': eff_pct,
        'adjustment_pct': adj_pct,
        'mean_efficiency': mean_eff,
        'min_efficiency': min_eff,
        'max_efficiency': max_eff,
        'std_efficiency': std_eff,
        'mean_variance_efficiency': var_eff,
        'mean_stability': mean_stability,
        'avg_profit': avg_profit,
        'recent_allocation_gap': alloc_diff,
        'eq_start_iteration': eq_start,
    }
    suffixes = set()
    suffixes.update(k.split('_')[-1] for k in avg_alloc.keys())
    suffixes.update(k.split('_')[-1] for k in avg_opt_alloc.keys())
    for suffix in sorted(suffixes):
        act_key = f"avg_alloc_{suffix}"
        opt_key = f"avg_opt_alloc_{suffix}"
        if act_key in avg_alloc:
            summary[act_key] = avg_alloc[act_key]
        if opt_key in avg_opt_alloc:
            summary[opt_key] = avg_opt_alloc[opt_key]
    return summary


def export_runs_summary_to_excel(run_dirs: Sequence[Path], output_path: Path) -> Path:
    rows: List[dict] = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            print(f"[WARN] Run directory {run_dir} not found. Skipping.")
            continue
        summary = summarize_run(run_dir)
        if summary is not None:
            rows.append(summary)

    if not rows:
        raise ValueError("No valid runs to summarize.")

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    output_path = output_path.with_suffix(".xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="summary")
    print(f"Saved Excel summary to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export summaries for selected run folders into an Excel file."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="List of run directories (relative or absolute paths).",
    )
    parser.add_argument(
        "--runs-file",
        help="Optional text file with one run directory per line (empty lines and '#' comments allowed).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Excel file path (e.g., results/custom_summary.xlsx).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inputs = list(args.runs)
    if args.runs_file:
        runs_path = Path(args.runs_file)
        if not runs_path.exists():
            raise FileNotFoundError(f"runs file not found: {runs_path}")
        extra = []
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            extra.append(line)
        run_inputs.extend(extra)

    if not run_inputs:
        raise ValueError("No run directories provided.")

    run_dirs = [Path(p) for p in run_inputs]
    output_path = Path(args.out)
    export_runs_summary_to_excel(run_dirs, output_path)


if __name__ == "__main__":
    main()
