from __future__ import annotations

"""Central experiment launcher.

This module defines named scenarios and runs:
- the RL simulation (run_experiment),
- the summary generation (run_summary)

for each scenario, with consistent experiment IDs and metadata.
"""

from pathlib import Path
import json
from typing import Dict, Any, List
import pandas as pd

from simulation import run_experiment
from summary import run_summary
from export_runs_excel import export_runs_summary_to_excel

EXPERIMENT_SUMMARIES: List[Dict[str, Any]] = []


# Default parameters shared by all scenarios unless overridden
DEFAULT_PARAMS: Dict[str, Any] = {
    "N": 100,
    "T": 1000,
    "episodes": 100,
    "add_gov": False,
    "switch_cost": 0.0,
    "seed": 1411,
    "random_policy": False,
    "model_type": "full",
}


# Scenarios in the format expected by ShockEngine (see scenario.py).
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "4m_random": {
        "scenario": [],
        "params": {
            "add_gov": False,
            "random_policy": True,            
        },
    },

    "4m_Q1": {
        "scenario": [],
        "params": {
            "add_gov": False,
            "model_type": "Q1",
        },
    },

    "4m_Q2": {
        "scenario": [],
        "params": {
            "add_gov": False,
            "model_type": "Q2",
        },
    },

    "5m_Q1": {
        "scenario": [],
        "params": {
            "add_gov": True,
            "model_type": "Q1",
        },
    },

    "5m_Q2": {
        "scenario": [],
        "params": {
            "add_gov": True,
            "model_type": "Q2",
        },
    },

    "5m_Q1_shocks": {
        "scenario": [
            {"when": {"iter": 400}, "price_bump": {2: +5.0}},
            {"when": {"iter": 500}, "entrants": {1: 15}},
            {"when": {"from_iter": 600}, "inflation": {"market": 3, "start": 600, "rate": 0.002}},            
            {"when": {"from_iter": 700}, "transaction_cost": {"set": 2.0}},
            {"when": {"iter": 800}, "cost_shock": {4: +4}},
            {"when": {"from_iter": 900}, "technology": {3: 1.5}},        
        ],
        "params": {
            "add_gov": True,
            "random_policy": False,
            "model_type": "Q1",            
        },
    },

    "5m_Q2_shocks": {
        "scenario": [
            {"when": {"iter": 400}, "price_bump": {2: +5.0}},
            {"when": {"iter": 500}, "entrants": {1: 15}},
            {"when": {"from_iter": 600}, "inflation": {"market": 3, "start": 600, "rate": 0.002}},            
            {"when": {"from_iter": 700}, "transaction_cost": {"set": 2.0}},
            {"when": {"iter": 800}, "cost_shock": {4: +4}},
            {"when": {"from_iter": 900}, "technology": {3: 1.5}},           
        ],
        "params": {
            "add_gov": True,
            "random_policy": False,
            "model_type": "Q2",            
        },
    },
}


def _merge_params(custom: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge DEFAULT_PARAMS with scenario-specific overrides."""
    params = DEFAULT_PARAMS.copy()
    if custom:
        params.update(custom)
    return params


def _policy_label(params: Dict[str, Any]) -> str:
    """Return human-readable label for the decision policy."""
    return "random" if params.get("random_policy") else params.get("model_type", "full")


def _collect_efficiency_stats(run_dir: Path) -> tuple[float | None, float | None]:
    """Read per-episode stats from summary output and average them."""
    stats_path = run_dir / "plots" / "episode_efficiency_stats.csv"
    if not stats_path.exists():
        return None, None

    try:
        df = pd.read_csv(stats_path, sep=';', decimal=',')
    except Exception as exc:
        print(f"Warning: failed to read {stats_path.name}: {exc}")
        return None, None

    if df.empty:
        return None, None

    eff_pct = float(df['efficient_pct'].mean())
    adjust_iters = float(df['adjustment_iterations'].mean())
    return eff_pct, adjust_iters


def _record_experiment_summary(
    name: str,
    params: Dict[str, Any],
    scenario: List[dict],
    run_dir: Path,
) -> None:
    """Append a row describing this experiment for the final table."""
    eff_pct, adjust_iters = _collect_efficiency_stats(run_dir)
    EXPERIMENT_SUMMARIES.append({
        "scenario": name,
        "gov_market": bool(params.get("add_gov")),
        "policy": _policy_label(params),
        "has_shocks": bool(scenario),
        "efficient_pct": eff_pct,
        "adjustment_iters": adjust_iters,
        "run_dir": str(run_dir),
    })


def _write_experiment_summary_table() -> None:
    """Persist the consolidated summary table after all runs."""
    if not EXPERIMENT_SUMMARIES:
        return

    df = pd.DataFrame(EXPERIMENT_SUMMARIES)
    df.insert(0, "experiment_no", range(1, len(df) + 1))

    # Improve readability of boolean columns
    bool_map = {True: "yes", False: "no"}
    df['gov_market'] = df['gov_market'].map(bool_map)
    df['has_shocks'] = df['has_shocks'].map(bool_map)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiments_summary.csv"
    df.to_csv(out_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"Saved experiment summary table to: {out_path}")


def run_scenario(name: str) -> Path:
    """Run a single named scenario end-to-end.

    - runs the simulation with a stable exp_id equal to `name`
    - writes config.json with parameters and scenario definition
    - calls summary.run_summary for this exp_id (plots + diagnostics)
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[name]
    scenario: List[dict] = cfg.get("scenario", [])
    params = _merge_params(cfg.get("params"))

    exp_id = name

    run_dir: Path = run_experiment(
        N=params["N"],
        T=params["T"],
        episodes=params["episodes"],
        add_gov=params["add_gov"],
        switch_cost=params["switch_cost"],
        seed=params["seed"],
        scenario=scenario,
        exp_id=exp_id,
        random_policy=params["random_policy"],
        model_type=params["model_type"],
    )

    # Store metadata so that each run directory is self-describing
    meta = {
        "exp_id": exp_id,
        "scenario_name": name,
        "params": params,
        "scenario": scenario,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Generate plots and diagnostics for this experiment
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    alloc_band_default = "results/alloc_band_filled.xlsx"
    run_summary(
        exp_id=exp_id,
        outdir=str(plots_dir),
        alloc_band=alloc_band_default,
    )
    _record_experiment_summary(name, params, scenario, run_dir)

    return run_dir


def _write_summary_latest(run_dirs: List[Path]) -> None:
    if not run_dirs:
        return
    out_path = Path("results") / "summary_latest.xlsx"
    export_runs_summary_to_excel(run_dirs, out_path)


def run_all(selected: List[str] | None = None) -> None:
    """Run all scenarios, or a selected subset."""
    names = selected if selected else list(SCENARIOS.keys())
    produced_runs: List[Path] = []
    for name in names:
        print(f"\n========== Running scenario: {name} ==========")
        produced_runs.append(run_scenario(name))
    _write_experiment_summary_table()
    _write_summary_latest(produced_runs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Run multiple RL experiments by scenario name.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of scenario names to run. If omitted, all scenarios are run.",
    )
    args = parser.parse_args()

    if args.only:
        run_all(args.only)
    else:
        run_all()
