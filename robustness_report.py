from __future__ import annotations

"""
Generate a main robustness table and a few aggregate plots.

Inputs:
  - results/robustness_summary.xlsx (export_runs_excel.py output)
  - results/robustness_runs.csv (manifest with parameters)
Outputs:
  - results/robustness_report/main_table.xlsx
  - results/robustness_report/main_table.csv
  - results/robustness_report/plots/*.png
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_GROUP_COLS = [
    "scenario",
    "ofat_param",
    "ofat_value",
    "add_gov",
    "has_shocks",
]


def _ci95(series: pd.Series) -> tuple[float, float]:
    vals = series.dropna().to_numpy(dtype=float)
    n = len(vals)
    if n <= 1:
        return (np.nan, np.nan)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    half = 1.96 * std / np.sqrt(n)
    return (mean - half, mean + half)


def _agg_group(df: pd.DataFrame, metrics: Iterable[str], group_cols: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for _, grp in df.groupby(group_cols):
        row = {k: grp.iloc[0][k] for k in group_cols}
        row["n_seeds"] = int(grp["seed"].nunique()) if "seed" in grp.columns else len(grp)
        for m in metrics:
            s = grp[m]
            row[f"{m}_mean"] = float(s.mean())
            row[f"{m}_std"] = float(s.std(ddof=1))
            row[f"{m}_median"] = float(s.median())
            lo, hi = _ci95(s)
            row[f"{m}_ci95_low"] = lo
            row[f"{m}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_metric_ofat(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if "ofat_param" not in df.columns:
        return
    params = [p for p in df["ofat_param"].dropna().unique() if p != "baseline"]
    if not params:
        return

    for param in sorted(params):
        sub = df[df["ofat_param"] == param]
        grouped = sub.groupby("ofat_value")[metric].agg(["mean", "std", "count"]).reset_index()
        if grouped.empty:
            continue
        grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
        x = np.arange(len(grouped))
        ax.errorbar(
            x,
            grouped["mean"],
            yerr=grouped["ci"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=param,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["ofat_value"].astype(str), rotation=45, ha="right")

    ax.set_xlabel("OFAT value")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by OFAT parameter (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_report(
    summary_path: Path = Path("results/robustness_summary.xlsx"),
    manifest_path: Path = Path("results/robustness_runs.csv"),
    out_dir: Path = Path("results/robustness_report"),
) -> tuple[Path, Path]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    summary = pd.read_excel(summary_path)
    manifest = pd.read_csv(manifest_path, sep=";", decimal=",")
    if "run_id" not in summary.columns:
        raise ValueError("summary is missing 'run_id' column.")
    if "run_dir" not in manifest.columns:
        raise ValueError("manifest is missing 'run_dir' column.")

    manifest = manifest.copy()
    manifest["run_id"] = manifest["run_dir"].apply(lambda p: Path(p).name)

    merged = summary.merge(manifest, on="run_id", how="left")
    if merged["scenario"].isna().any():
        missing = merged[merged["scenario"].isna()]["run_id"].tolist()
        print(f"[WARN] Missing manifest rows for run_id: {missing}")

    metrics = [
        "efficient_pct",
        "adjustment_pct",
        "mean_efficiency",
        "std_efficiency",
        "mean_stability",
        "avg_profit",
        "recent_allocation_gap",
        "eq_start_iteration",
        "avg_recovery_iters_after_shocks",
    ]
    # keep only existing metrics
    metrics = [m for m in metrics if m in merged.columns]

    group_cols = [c for c in DEFAULT_GROUP_COLS if c in merged.columns]
    main_table = _agg_group(merged, metrics, group_cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_xlsx = out_dir / "main_table.xlsx"
    main_csv = out_dir / "main_table.csv"
    with pd.ExcelWriter(main_xlsx, engine="openpyxl") as writer:
        main_table.to_excel(writer, index=False, sheet_name="main")
    main_table.to_csv(main_csv, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    plots_dir = out_dir / "plots"
    for metric in ["efficient_pct", "adjustment_pct", "mean_stability", "mean_efficiency"]:
        if metric in merged.columns:
            _plot_metric_ofat(merged, metric, plots_dir / f"{metric}_by_ofat.png")
    if "avg_recovery_iters_after_shocks" in merged.columns and merged["avg_recovery_iters_after_shocks"].notna().any():
        _plot_metric_ofat(merged, "avg_recovery_iters_after_shocks", plots_dir / "recovery_by_ofat.png")

    print(f"Saved main table to: {main_xlsx}")
    print(f"Saved plots to: {plots_dir}")
    return main_xlsx, main_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build aggregated robustness report.")
    ap.add_argument("--summary", default="results/robustness_summary.xlsx")
    ap.add_argument("--manifest", default="results/robustness_runs.csv")
    ap.add_argument("--out-dir", default="results/robustness_report")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(
        summary_path=Path(args.summary),
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
