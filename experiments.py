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

from simulation import run_experiment
from summary import run_summary


# Default parameters shared by all scenarios unless overridden
DEFAULT_PARAMS: Dict[str, Any] = {
    "N": 100,
    "T": 1000,
    "episodes": 10,
    "add_gov": False,
    "switch_cost": 0.0,
    "seed": 1411,
    "random_policy": False,
    "model_type": "full",
}


# Scenarios in the format expected by ShockEngine (see scenario.py).
# Names roughly mirror the one-off examples from simulation.py __main__.

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "scenario": [],
        "params": {
            "add_gov": False,
        },
    },

    "random_agents_gov": {
        "scenario": [],
        "params": {
            "add_gov": True,
            "random_policy": True,
        },
    },

    "gov_only": {
        "scenario": [],
        "params": {
            "add_gov": True,
        },
    },

    "price_bump_m2_it200": {
        "scenario": [
            {"when": {"iter": 200}, "price_bump": {2: +5.0}},
        ],
        "params": {
            "add_gov": True,
        },
    },

    "entrants_m1_it300": {
        "scenario": [
            {"when": {"iter": 300}, "entrants": {1: 15}},
        ],
        "params": {
            "add_gov": True,
        },
    },

    "inflation_m3_from400": {
        "scenario": [
            {"when": {"from_iter": 400}, "inflation": {"market": 3, "start": 400, "rate": 0.01}},
        ],
        "params": {
            "add_gov": True,
        },    
    },

    "switch_cost_from500": {
        "scenario": [
            {"when": {"from_iter": 500}, "transaction_cost": {"set": 2.0}},
        ],
        "params": {
            "add_gov": True,
        },
    },


    "cost_shock_m4_it600": {
        "scenario": [
            {"when": {"iter": 600}, "cost_shock": {4: +4}},
        ],
        "params": {
            "add_gov": True,
        },
    },

    "tech_m3_from700": {
        "scenario": [
            {"when": {"from_iter": 700}, "technology": {3: 2.0}},
        ],
        "params": {
            "add_gov": True,
        },
    },

    "all_shocks_full": {
        "scenario": [
            {"when": {"iter": 200}, "price_bump": {2: +5.0}},
            {"when": {"iter": 300}, "entrants": {1: 15}},
            {"when": {"from_iter": 400}, "inflation": {"market": 3, "start": 400, "rate": 0.002}},            
            {"when": {"from_iter": 500}, "transaction_cost": {"set": 2.0}},
            {"when": {"iter": 600}, "cost_shock": {4: +4}},
            {"when": {"from_iter": 700}, "technology": {3: 1.5}},            
        ],
        "params": {
            "add_gov": True,
            "random_policy": False,
            "model_type": "full",
        },
    },

    "all_shocks_profit": {
        "scenario": [
            {"when": {"iter": 200}, "price_bump": {2: +5.0}},
            {"when": {"iter": 300}, "entrants": {1: 15}},
            {"when": {"from_iter": 400}, "inflation": {"market": 3, "start": 400, "rate": 0.002}},            
            {"when": {"from_iter": 500}, "transaction_cost": {"set": 2.0}},
            {"when": {"iter": 600}, "cost_shock": {4: +4}},
            {"when": {"from_iter": 700}, "technology": {3: 1.5}},        
        ],
        "params": {
            "add_gov": True,
            "random_policy": False,
            "model_type": "profit_only",            
        },
    },

    "all_shocks_count": {
        "scenario": [
            {"when": {"iter": 200}, "price_bump": {2: +5.0}},
            {"when": {"iter": 300}, "entrants": {1: 15}},
            {"when": {"from_iter": 400}, "inflation": {"market": 3, "start": 400, "rate": 0.002}},            
            {"when": {"from_iter": 500}, "transaction_cost": {"set": 2.0}},
            {"when": {"iter": 600}, "cost_shock": {4: +4}},
            {"when": {"from_iter": 700}, "technology": {3: 1.5}},    
        ],
        "params": {
            "add_gov": True,
            "random_policy": False,
            "model_type": "count_only",            
        },
    },    
}


def _merge_params(custom: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge DEFAULT_PARAMS with scenario-specific overrides."""
    params = DEFAULT_PARAMS.copy()
    if custom:
        params.update(custom)
    return params


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
    run_summary(exp_id=exp_id, outdir=str(plots_dir))

    return run_dir


def run_all(selected: List[str] | None = None) -> None:
    """Run all scenarios, or a selected subset."""
    names = selected if selected else list(SCENARIOS.keys())
    for name in names:
        print(f"\n========== Running scenario: {name} ==========")
        run_scenario(name)


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
