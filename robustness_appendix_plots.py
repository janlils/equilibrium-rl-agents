from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------
# Paths (adjust if needed)
# ---------------------------
INPUT_XLSX = Path("results/robustness_summary.xlsx")
OUTDIR = Path("figures") / "appendix"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Plot style (match summary.py)
# ---------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True
})
colors_hex = ['#24325F', '#82491E', '#B7E4F9', '#E89242', '#FB6467', '#69C8EC']
cm = ListedColormap(colors_hex)


# ---------------------------
# Helpers: parse run_id fields
# ---------------------------
def _find(run_id: str, pattern: str):
    m = re.search(pattern, run_id)
    return m.group(1) if m else None


def p_to_float(s: str | None):
    if s is None:
        return None
    return float(s.replace("p", "."))


def parse_eps(s: str | None):
    if s is None:
        return (None, None)
    a, b = s.split("-")
    return p_to_float(a), p_to_float(b)


def parse_run_id(run_id: str) -> dict:
    N = _find(run_id, r"__N(\d+)__")
    T = _find(run_id, r"__T(\d+)__")
    E = _find(run_id, r"__E(\d+)__")
    seed = _find(run_id, r"__s(\d+)__")
    alpha = _find(run_id, r"__a([^_]+)__")
    gamma = _find(run_id, r"__g([^_]+)__")
    eps = _find(run_id, r"__e([^_]+)")
    v = _find(run_id, r"__v(\d{3})")

    eps_min, eps_max = parse_eps(eps)

    return {
        "N": int(N) if N else None,
        "T": int(T) if T else None,
        "E": int(E) if E else None,
        "seed": int(seed) if seed else None,
        "alpha": p_to_float(alpha),
        "gamma": p_to_float(gamma),
        "eps_min": eps_min,
        "eps_max": eps_max,
        "v": int(v) if v else 0,
    }


# ---------------------------
# Load data
# ---------------------------
df = pd.read_excel(INPUT_XLSX, sheet_name="summary")
parsed = df["run_id"].apply(parse_run_id).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# Metrics
MET_MEAN_EFF = "mean_efficiency"
MET_FIRST_EFF = "eq_start_iteration"
MET_EFF_PCT = "efficient_pct"
MET_ADJ_PCT = "adjustment_pct"
MET_STAB = "mean_stability"
MET_PROFIT = "avg_profit"
MET_RECOVERY = "avg_recovery_iters_after_shocks"


# ---------------------------
# Baseline (must match robustness design)
# ---------------------------
BASELINE = {
    "N": 100,
    "T": 1000,
    "E": 5,
    "seed": 1001,
    "alpha": 0.05,
    "gamma": 0.9,
    "eps_min": 0.01,
    "eps_max": 0.1,
    "v": 0,
}


def baseline_mask(d: pd.DataFrame) -> pd.Series:
    m = (
        (d["N"] == BASELINE["N"]) &
        (d["T"] == BASELINE["T"]) &
        (d["E"] == BASELINE["E"]) &
        (d["alpha"] == BASELINE["alpha"]) &
        (d["gamma"] == BASELINE["gamma"]) &
        (d["eps_min"] == BASELINE["eps_min"]) &
        (d["eps_max"] == BASELINE["eps_max"]) &
        (d["v"] == 0)
    )
    return m


def make_seed_distributions():
    d = df[
        (df["v"] == 0) &
        (df["N"] == BASELINE["N"]) &
        (df["T"] == BASELINE["T"]) &
        (df["E"] == BASELINE["E"]) &
        (df["alpha"] == BASELINE["alpha"]) &
        (df["gamma"] == BASELINE["gamma"]) &
        (df["eps_min"] == BASELINE["eps_min"]) &
        (df["eps_max"] == BASELINE["eps_max"]) &
        (df["seed"].between(1001, 1010))
    ].copy()

    d = d.groupby("seed", as_index=False).agg({
        MET_MEAN_EFF: "mean",
        MET_FIRST_EFF: "mean",
        MET_EFF_PCT: "mean",
        MET_STAB: "mean",
        MET_PROFIT: "mean",
        MET_RECOVERY: "mean"
    }).sort_values("seed")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    axes[0].boxplot(d[MET_MEAN_EFF].values, vert=True, widths=0.5, showfliers=False)
    axes[0].scatter(np.ones(len(d)), d[MET_MEAN_EFF].values, s=14, zorder=3, color=colors_hex[0])
    axes[0].set_title("")
    axes[0].set_ylabel(r"$\mathbb{E}[\phi/\phi_{\max}]$")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Seeds\n(10)"])

    axes[1].boxplot(d[MET_FIRST_EFF].values, vert=True, widths=0.5, showfliers=False)
    axes[1].scatter(np.ones(len(d)), d[MET_FIRST_EFF].values, s=14, zorder=3, color=colors_hex[3])
    axes[1].set_title("")
    axes[1].set_ylabel("Iterations")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Seeds\n(10)"])

    fig.tight_layout()
    out = OUTDIR / "seed_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_rl_param_dependence_plot():
    # RL configs: seed fixed at 1001, v=0
    d = df[(df["v"] == 0) & (df["seed"] == BASELINE["seed"])].copy()

    # Keep N, T, E fixed to isolate hyperparameters
    d = d[(d["N"] == BASELINE["N"]) & (d["T"] == BASELINE["T"]) & (d["E"] == BASELINE["E"])].copy()

    # Drop missing hyperparams
    d = d.dropna(subset=["alpha", "gamma", "eps_min", "eps_max"])

    # Collapse duplicates by hyperparameter tuple
    key_cols = ["alpha", "gamma", "eps_min", "eps_max"]
    d = d.groupby(key_cols, as_index=False).agg({
        MET_MEAN_EFF: "mean",
        MET_EFF_PCT: "mean",
        MET_FIRST_EFF: "mean",
        MET_STAB: "mean",
        MET_PROFIT: "mean",
    })

    gammas = sorted(d["gamma"].unique())
    eps_pairs = sorted({(float(a), float(b)) for a, b in zip(d["eps_min"], d["eps_max"])})

    marker_map = {}
    marker_cycle = ["o", "s", "^", "D", "P", "v", ">", "<"]
    for i, ep in enumerate(eps_pairs):
        marker_map[ep] = marker_cycle[i % len(marker_cycle)]

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.0))

    for g in gammas:
        subg = d[d["gamma"] == g].copy()
        for ep in eps_pairs:
            sub = subg[(subg["eps_min"] == ep[0]) & (subg["eps_max"] == ep[1])].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("alpha")
            x = sub["alpha"].values

            ax.plot(x, sub[MET_MEAN_EFF].values, linewidth=1, alpha=0.7, color=colors_hex[0])
            ax.scatter(
                x, sub[MET_MEAN_EFF].values, s=28,
                marker=marker_map[ep], alpha=0.9,
                label=f"$\\gamma$={g:.2f}, $\\varepsilon$={ep}"
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"Learning rate $\alpha$")
    ax.set_ylabel(r"Mean efficiency $\mathbb{E}[\phi/\phi_{\max}]$")
    ax.axhline(0.99, linewidth=1)

    base = d[
        (d["alpha"] == BASELINE["alpha"]) &
        (d["gamma"] == BASELINE["gamma"]) &
        (d["eps_min"] == BASELINE["eps_min"]) &
        (d["eps_max"] == BASELINE["eps_max"])
    ]
    if not base.empty:
        bx = float(base["alpha"].iloc[0])
        by0 = float(base[MET_MEAN_EFF].iloc[0])
        by1 = float(base[MET_EFF_PCT].iloc[0])

        ax.scatter([bx], [by0], marker="X", s=80, zorder=6)
        ax.annotate("baseline", (bx, by0), xytext=(6, 6), textcoords="offset points")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = OUTDIR / "rl_param_dependence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_NT_dependence_plot():
    # Vary N and T with baseline RL hyperparams (seed fixed, v=0)
    d = df[(df["v"] == 0) & (df["seed"] == BASELINE["seed"])].copy()
    d = d[
        (d["alpha"] == BASELINE["alpha"]) &
        (d["gamma"] == BASELINE["gamma"]) &
        (d["eps_min"] == BASELINE["eps_min"]) &
        (d["eps_max"] == BASELINE["eps_max"]) &
        (d["E"] == BASELINE["E"])
    ].copy()

    d = d.dropna(subset=["N", "T"])
    d = d.groupby(["N", "T"], as_index=False).agg({
        MET_MEAN_EFF: "mean"
    })

    Ns = sorted(d["N"].unique())
    Ts = sorted(d["T"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))

    # Panel A: mean efficiency vs N (for each T)
    for t in Ts:
        sub = d[d["T"] == t].sort_values("N")
        axes[0].plot(sub["N"], sub[MET_MEAN_EFF], marker="o", linewidth=1, label=f"T={t}")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel(r"Mean efficiency $\mathbb{E}[\phi/\phi_{\max}]$")

    # Panel B: mean efficiency vs T (for each N)
    for n in Ns:
        sub = d[d["N"] == n].sort_values("T")
        axes[1].plot(sub["T"], sub[MET_MEAN_EFF], marker="o", linewidth=1, label=f"N={n}")
    axes[1].set_xlabel("T")
    axes[1].set_ylabel(r"Mean efficiency $\mathbb{E}[\phi/\phi_{\max}]$")

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = OUTDIR / "nt_param_dependence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_three_param_panel():
    # Common filter: seed fixed, no shock variants
    d = df[(df["v"] == 0) & (df["seed"] == BASELINE["seed"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    def _plot_param(ax, param, xlabel):
        sub = d.copy()
        if param in ("alpha", "gamma", "eps_pairs"):
            sub = sub[
                (sub["N"] == BASELINE["N"]) &
                (sub["T"] == BASELINE["T"]) &
                (sub["E"] == BASELINE["E"])
            ]
        else:
            sub = sub[
                (sub["alpha"] == BASELINE["alpha"]) &
                (sub["gamma"] == BASELINE["gamma"]) &
                (sub["eps_min"] == BASELINE["eps_min"]) &
                (sub["eps_max"] == BASELINE["eps_max"]) &
                (sub["E"] == BASELINE["E"])
            ]

        if param == "eps_pairs":
            sub = sub.dropna(subset=["eps_min", "eps_max"])
            sub["eps_pair"] = sub.apply(lambda r: f"{r['eps_min']:.3g}:{r['eps_max']:.3g}", axis=1)
            agg = sub.groupby("eps_pair", as_index=False)[MET_MEAN_EFF].mean()
            x = np.arange(len(agg))
            ax.plot(x, agg[MET_MEAN_EFF], marker="o", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(agg["eps_pair"], rotation=30, ha="right")
            bx = f"{BASELINE['eps_min']:.3g}:{BASELINE['eps_max']:.3g}"
            if bx in agg["eps_pair"].values:
                bi = agg.index[agg["eps_pair"] == bx][0]
                by = float(agg.loc[bi, MET_MEAN_EFF])
                ax.scatter([bi], [by], marker="X", s=80, zorder=5, color=colors_hex[4], label="baseline")
        else:
            sub = sub.dropna(subset=[param])
            agg = sub.groupby(param, as_index=False)[MET_MEAN_EFF].mean().sort_values(param)
            ax.plot(agg[param], agg[MET_MEAN_EFF], marker="o", linewidth=1)
            if param in BASELINE:
                bx = BASELINE[param]
                if bx in agg[param].values:
                    by = float(agg.loc[agg[param] == bx, MET_MEAN_EFF].iloc[0])
                    ax.scatter([bx], [by], marker="X", s=80, zorder=5, color=colors_hex[4], label="baseline")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Mean efficiency $\mathbb{E}[\phi/\phi_{\max}]$")

    _plot_param(axes[0], "alpha", r"Learning rate $\alpha$")
    _plot_param(axes[1], "gamma", r"Discount $\gamma$")
    _plot_param(axes[2], "eps_pairs", r"Epsilon $(\min,\max)$")

    # Single shared y-label
    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_ylabel("")

    # Shared legend (include baseline once)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = OUTDIR / "three_param_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_shock_variant_effects():
    variants = df[
        (df["v"] >= 1) &
        (df["seed"] == BASELINE["seed"]) &
        (df["N"] == BASELINE["N"]) &
        (df["T"] == BASELINE["T"]) &
        (df["E"] == BASELINE["E"]) &
        (df["alpha"] == BASELINE["alpha"]) &
        (df["gamma"] == BASELINE["gamma"]) &
        (df["eps_min"] == BASELINE["eps_min"]) &
        (df["eps_max"] == BASELINE["eps_max"])
    ].copy()

    base = df[baseline_mask(df) & (df["seed"] == BASELINE["seed"])].copy()
    if len(base) == 0:
        raise RuntimeError("Baseline run (v=0) not found for shock-variant deltas.")

    base_mean_eff = float(base[MET_MEAN_EFF].mean())
    base_profit = float(base[MET_PROFIT].mean())
    base_first = float(base[MET_FIRST_EFF].mean())
    base_recov = float(base[MET_RECOVERY].mean()) if MET_RECOVERY in base.columns else np.nan

    variants = variants.groupby("v", as_index=False).agg({
        MET_MEAN_EFF: "mean",
        MET_FIRST_EFF: "mean",
        MET_PROFIT: "mean",
        MET_RECOVERY: "mean"
    }).sort_values("v")

    variants["d_mean_eff"] = variants[MET_MEAN_EFF] - base_mean_eff
    variants["d_profit"] = variants[MET_PROFIT] - base_profit
    variants["d_first"] = variants[MET_FIRST_EFF] - base_first
    variants["d_recov"] = variants[MET_RECOVERY] - base_recov if not np.isnan(base_recov) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    axes[0].plot(variants["v"], variants["d_mean_eff"], linewidth=1, color=colors_hex[0])
    axes[0].scatter(variants["v"], variants["d_mean_eff"], s=12, color=colors_hex[0])
    axes[0].axhline(0.0, linewidth=1)
    axes[0].set_title("")
    axes[0].set_xlabel("Variant id")
    axes[0].set_ylabel(r"$\Delta\,\mathbb{E}[\phi/\phi_{\max}]$")

    axes[1].hist(variants["d_profit"].values, bins=12, color=colors_hex[3])
    axes[1].axvline(0.0, linewidth=1)
    axes[1].set_title("")
    axes[1].set_xlabel(r"$\Delta$ profit")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    out = OUTDIR / "shock_variant_effects.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_seed_distributions()
    make_rl_param_dependence_plot()
    make_shock_variant_effects()
    make_NT_dependence_plot()
    make_three_param_panel()
