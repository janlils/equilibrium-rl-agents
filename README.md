# equilibrium-rl-agents

Agent-based SARSA simulation for market selection with linear value approximation. Agents learn which market to join or stay on given price-feedback, costs, switching frictions, and exogenous shocks. The simulator tracks actual vs. optimal allocations and produces visual summaries and aggregate statistics.

## Table of Contents
1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Running Simulations](#running-simulations)
4. [Batch Experiments](#batch-experiments)
5. [Analysis & Plots](#analysis--plots)
6. [Excel Export](#excel-export)
7. [Robustness](#robustness)
8. [Directory Structure](#directory-structure)

## Architecture

- `simulation.py` — core engine that instantiates farmers (`farmer.py`), markets (`market.py`), and writes results to `.pickle`.
- `farmer.py` — agent definition with linear Q-function approximation (`full`, `Q2`, `Q1`) and optional random policy.
- `market.py` + `config.py` — market specification, price functions, and costs; includes optional government market.
- `scenario.py` — shock generator (price bumps, entrants, inflation, technology shocks, etc.).
- `optimal_allocation.py` — dynamic programming to compute potential and optimal allocations.
- `summary.py` — reads pickles and produces PNG/CSV summaries.
- `experiments.py` — library of named scenarios and driver for batch simulations.
- `export_runs_excel.py` — aggregates multiple runs and writes a concise Excel report.

## Installation

1. **Python 3.11+** recommended (virtualenv/conda).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `openpyxl`, etc.
3. **NumPy compatibility**: some runs are pickled with NumPy 2.x; ensure your environment has a matching version (`pip install "numpy>=2.0"`), otherwise loading `results_*` may fail with `ModuleNotFoundError: numpy._core`.

## Running Simulations

> **Quick start / Reproducibility for papers:** run `python experiments.py` (optionally with `--only scenario1 scenario2`). This single command launches every configured scenario, produces all plots/diagnostics under each `run_*/plots/`, and writes two tables in `results/`: `experiments_summary.csv` plus `summary_latest.xlsx` (Excel report with one row per run). Each `run_*` folder also stores the exact `config.json` used for that scenario, so reproducing the results for an article amounts to (1) `pip install -r requirements.txt`, (2) `python experiments.py`, and (3) packaging the `results/` directory with the paper.

Simplest entry point:

```bash
python simulation.py
```

Defaults: `N=100` agents, `T=1000` iterations per episode, `EPISODES=10`. Results go into `results/run_<timestamp>[_expid][_gov]`, containing:
- `results_market*.pickle` — each farmer’s market choice per iteration,
- `results_profits*.pickle` — profits per agent per iteration,
- `results_potential*.pickle` — potential metrics (actual vs. optimal allocations),
- `results_theta*.pickle` — average Q-parameters per iteration (if enabled).

Override via `run_experiment(N=..., T=..., add_gov=True, scenario=[...], switch_cost=..., model_type="full"/"Q2"/"Q1", random_policy=True, exp_id="...")`. See `experiments.py` for reference configurations.

## Batch Experiments

`experiments.py` contains named scenarios (e.g., `4m_random`, `5m_Q1_shocks`) and an orchestration flow:

```bash
python experiments.py --scenario 5m_Q1_shocks
# or run an entire suite:
python experiments.py --all
```

For each scenario:
1. `run_experiment` executes with merged parameters.
2. `config.json` (metadata) is stored in the run folder.
3. `summary.py` is invoked automatically to generate plots/CSV diagnostics in `run_.../plots`.
4. Metadata and averaged stats can be appended to `results/experiments_summary.csv`.

## Analysis & Plots

`summary.py` consumes the pickles and produces (include `--alloc-band` if you want the optimal allocation bands generated from `results/alloc_band_filled.xlsx`; you can also pass `--runs-file results/runs.txt` to batch-process many folders):

```bash
python summary.py \
    --market results/run_xxx/results_market_4m.pickle \
    --profits results/run_xxx/results_profits_4m.pickle \
    --outdir results/run_xxx/plots \
    --alloc-band results/alloc_band_filled.xlsx
```

Outputs (PNG + CSV):
- `art_Number_Iter.png` — stacked area chart: average number of agents per market per iteration.
- `art_Profit_Iter.png` — average profits over iterations (Total line bolded, Y-range limited to [-10, 10]).
- `art_Efficiency_Stability_Iter.png` — potential ratio vs. stability (share of agents staying put).
- `art_Allocation_Optimal_Iter.png` — actual vs. optimal allocation with shock annotations (labels near the top).
- `art_Q_Params_Iter.png` — average Q-parameters (if `results_theta*.pickle` is present).
- CSV: `diagnostic_episodes.csv`, `episode_efficiency_stats.csv`, etc.

Other plotting utilities include histograms, profit vs. stability scatter plots, and combined efficiency/stability overlays.

## Excel Export

`export_runs_excel.py` generates a consolidated Excel sheet for selected runs (the same functionality runs automatically at the end of `python experiments.py`, producing `results/summary_latest.xlsx`):

```bash
python export_runs_excel.py run_dir1 run_dir2 --out results/summary.xlsx
# or provide a text file
python export_runs_excel.py --runs-file results/runs.txt --out results/summary.xlsx
```

- `runs` arguments are optional when `--runs-file` is supplied (file lines may include comments `#` and blank lines).
- Each row contains metrics aggregated across episodes/iterations:
  - `efficient_pct`, `adjustment_pct` — share of efficient iterations and share spent adapting (based on potential ratio > 0.99 and a 5-step window).
  - `mean_efficiency`, `min_efficiency`, `max_efficiency`, `std_efficiency`, `mean_variance_efficiency`.
  - `mean_stability` — fraction of farmers who kept the same market between iterations.
  - `avg_profit` — mean profit over the last 100 iterations of every episode.
  - `recent_allocation_gap` — sum of absolute differences between actual and optimal allocations over the last 100 iterations of every episode.
  - `eq_start_iteration` — average iteration (across episodes) when equilibrium (5 consecutive efficient iterations) was first reached.
  - `avg_alloc_mX`, `avg_opt_alloc_mX` — average vs. optimal allocation per market in the last 100 iterations, written side-by-side.

By default, Excel is saved as `results/<name>.xlsx`; adjust `--out` as needed.

## Robustness 

The project includes a dedicated robustness runner that tests **only Q2** and avoids full Cartesian grids:

```bash
# seeds only (baseline params)
python robustness_checks.py --ofat --ofat-seeds --scenarios 5m_Q2_shocks

# hyperparameters only (fixed seed)
python robustness_checks.py --ofat --ofat-params --scenarios 5m_Q2_shocks

# shock variants only (fixed seed)
python robustness_checks.py --ofat --shock-only --scenarios 5m_Q2_shocks --shock-variants 100 --shock-permute-when
```

**Baseline** (edit in `robustness_checks.py`):
- `N=100`, `T=1000`, `episodes=5`
- `alpha=0.05`, `gamma=0.90`, `eps_min=0.01`, `eps_max=0.10`
- `model_type=Q2`, `random_policy=False`

**ranges** (edit in `robustness_checks.py`):
- `N`: 25, 50, 100, 150
- `T`: 1000, 2000
- `episodes`: 5, 10, 30, 50
- `alpha`: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
- `gamma`: 0.7, 0.8, 0.9, 0.95, 0.99
- `eps_pairs`: (0.001,0.05), (0.005,0.05), (0.01,0.10), (0.02,0.20), (0.05,0.30)

**Outputs**
- `results/robustness_summary.xlsx` — merged run summary (auto-append).
- `results/robustness_runs.csv` — manifest with full parameters per run.
- `results/robustness_report/` — aggregated table + plots.

Appendix plots (PNG) can be generated with:

```bash
python robustness_appendix_plots.py
```

## Directory Structure

- `results/`
  - `run_<timestamp>[_expid][...]/`
    - `config.json`
    - `results_market*.pickle`, `results_profits*.pickle`, `results_potential*.pickle`, `results_theta*.pickle`
    - `plots/` with all PNG/CSV outputs.
  - `runs.txt` — sample list of the 10 latest runs (in ascending order, oldest first).
  - `experiments_summary.csv` — optional aggregated table from `experiments.py`.
- `requirements.txt` — Python dependencies.
- `old/` — archived experiments.

## Tips

- **Reproducibility**: use `seed` and `exp_id` parameters to track runs.
- **Baseline policy**: set `random_policy=True` to benchmark learning vs. random behavior.
- **Shock scripting**: define scenarios in `scenario.py` and pass via `run_experiment(..., scenario=SCENARIOS["..."]["scenario"])`.
- **Performance**: prefer `experiments.py` for multi-run batches; it automates summary generation and metadata logging.
- **Excel post-processing**: open the exported file in Excel/LibreOffice to filter/sort by scenario, inspect gaps, or build pivot charts.

## License

This codebase is intended for research/educational purposes. No formal license is attached; contact the repository owner if you plan to reuse the code in other contexts.
