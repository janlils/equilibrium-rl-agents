from __future__ import annotations

"""
Run OFAT robustness checks and generate summaries/plots.

This script:
- runs one-factor-at-a-time (OFAT) sweeps for seeds/parameters/shocks
- runs simulations for selected scenarios
- runs summary diagnostics (efficiency, stability, plots)
- exports a consolidated Excel summary and OFAT report
"""

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from experiments import SCENARIOS, DEFAULT_PARAMS
from simulation import run_experiment
from summary import run_summary
from export_runs_excel import export_runs_summary_to_excel, summarize_run


def _parse_int_list(raw: str) -> Tuple[int, ...]:
    if not raw:
        return tuple()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _parse_float_list(raw: str) -> Tuple[float, ...]:
    if not raw:
        return tuple()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _parse_str_list(raw: str) -> Tuple[str, ...]:
    if not raw:
        return tuple()
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _parse_float_list_or_default(raw: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if not raw:
        return default
    return _parse_float_list(raw)


def _format_switch_cost(val: float) -> str:
    s = f"{val:.3g}"
    return s.replace(".", "p")


def _merge_params(custom: Dict[str, Any] | None) -> Dict[str, Any]:
    params = DEFAULT_PARAMS.copy()
    if custom:
        params.update(custom)
    return params


def _find_first(run_dir: Path, pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {run_dir}")
    return files[0]


def _build_exp_id(scenario: str, params: Dict[str, Any]) -> str:
    parts = [
        scenario,
        f"N{params['N']}",
        f"T{params['T']}",
        f"E{params['episodes']}",
        f"s{params['seed']}",
        f"sc{_format_switch_cost(float(params['switch_cost']))}",
        f"mt{params['model_type']}",
        f"rp{int(bool(params['random_policy']))}",
        f"a{_format_switch_cost(float(params['alpha']))}",
        f"g{_format_switch_cost(float(params['gamma']))}",
        f"e{_format_switch_cost(float(params['eps_min']))}-{_format_switch_cost(float(params['eps_max']))}",
    ]
    return "__".join(parts)


def _scale_payload(event_type: str, payload: Dict[str, Any], factor: float) -> Dict[str, Any]:
    scaled = {}
    for k, v in payload.items():
        if event_type == "inflation":
            if k == "rate":
                scaled[k] = float(v) * factor
            else:
                scaled[k] = v
            continue

        if event_type == "entrants":
            if k in {"market", "start"}:
                scaled[k] = v
                continue
            scaled[k] = int(round(float(v) * factor))
            continue

        if isinstance(v, (int, float)):
            scaled[k] = float(v) * factor
        else:
            scaled[k] = v
    return scaled


def _apply_scale_to_events(events: List[dict], factor: float) -> List[dict]:
    scaled_events: List[dict] = []
    for ev in events:
        ev_new = copy.deepcopy(ev)
        if factor == 1.0:
            scaled_events.append(ev_new)
            continue
        for key in list(ev_new.keys()):
            if key == "when":
                continue
            payload = ev_new.get(key)
            if isinstance(payload, dict):
                ev_new[key] = _scale_payload(key, payload, factor)
        scaled_events.append(ev_new)
    return scaled_events


def _permute_when(events: List[dict], rng: random.Random) -> List[dict]:
    if not events:
        return events
    whens = [copy.deepcopy(ev.get("when", {})) for ev in events]
    rng.shuffle(whens)
    permuted = []
    for ev, when in zip(events, whens):
        ev_new = copy.deepcopy(ev)
        ev_new["when"] = when
        permuted.append(ev_new)
    return permuted


def _generate_shock_variants(
    base_events: List[dict],
    n_variants: int,
    scale_values: Tuple[float, ...],
    permute_when: bool,
    rng: random.Random,
    per_event_scale: bool = False,
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for i in range(n_variants):
        if per_event_scale:
            events: List[dict] = []
            scale_map: List[float] = []
            for ev in base_events:
                scale = rng.choice(scale_values) if scale_values else 1.0
                scale_map.append(scale)
                events.append(_apply_scale_to_events([ev], scale)[0])
        else:
            scale = rng.choice(scale_values) if scale_values else 1.0
            scale_map = [scale for _ in base_events]
            events = _apply_scale_to_events(base_events, scale)
        if permute_when:
            events = _permute_when(events, rng)
        variants.append({
            "events": events,
            "scale": None if per_event_scale else scale,
            "scale_map": scale_map,
            "permute_when": permute_when,
            "variant_id": i + 1,
        })
    return variants


def _save_run_metadata(run_dir: Path, meta: Dict[str, Any]) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _record_manifest_row(
    rows: List[Dict[str, Any]],
    run_dir: Path,
    scenario_name: str,
    scenario_events: List[dict],
    params: Dict[str, Any],
    variant_id: int | None = None,
    scale_factor: float | None = None,
    permute_when: bool | None = None,
):
    rows.append({
        "run_dir": str(run_dir),
        "scenario": scenario_name,
        "has_shocks": bool(scenario_events),
        "variant_id": variant_id,
        "scale_factor": scale_factor,
        "permute_when": permute_when,
        "N": params["N"],
        "T": params["T"],
        "episodes": params["episodes"],
        "seed": params["seed"],
        "switch_cost": params["switch_cost"],
        "model_type": params["model_type"],
        "random_policy": params["random_policy"],
        "add_gov": params["add_gov"],
        "alpha": params["alpha"],
        "gamma": params["gamma"],
        "eps_min": params["eps_min"],
        "eps_max": params["eps_max"],
        "ofat_param": params.get("ofat_param"),
        "ofat_value": params.get("ofat_value"),
        "scale_map": params.get("scale_map"),
    })


def _ofat_combinations(
    base: Dict[str, Any],
    ranges: Dict[str, Sequence[Any]],
    seeds: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    seen: set[tuple] = set()

    def _key(params: Dict[str, Any]) -> tuple:
        items = tuple((k, params[k]) for k in sorted(params.keys()))
        return items

    for seed in seeds:
        base_params = base.copy()
        base_params["seed"] = seed
        base_params["ofat_param"] = "baseline"
        base_params["ofat_value"] = "baseline"
        k = _key(base_params)
        if k not in seen:
            seen.add(k)
            combos.append(base_params)

        for param, values in ranges.items():
            if isinstance(values, (str, bytes)):
                vals = (values,)
            else:
                try:
                    iter(values)
                    vals = values
                except TypeError:
                    vals = (values,)
            for v in vals:
                params = base.copy()
                params["seed"] = seed
                if param == "eps_pairs":
                    params["eps_min"], params["eps_max"] = v
                    params["ofat_param"] = "eps_pairs"
                    params["ofat_value"] = f"{params['eps_min']}:{params['eps_max']}"
                else:
                    params[param] = v
                    params["ofat_param"] = param
                    params["ofat_value"] = str(v)
                k = _key(params)
                if k in seen:
                    continue
                seen.add(k)
                combos.append(params)

    return combos


def _shock_combinations(base: Dict[str, Any], seeds: Tuple[int, ...]) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for seed in seeds:
        params = base.copy()
        params["seed"] = seed
        params["ofat_param"] = "shock_variants"
        params["ofat_value"] = "baseline"
        combos.append(params)
    return combos


def run_robustness(
    scenarios: Sequence[str],
    combinations: Iterable[Dict[str, Any]],
    out_excel: Path,
    dry_run: bool = False,
    max_runs: int | None = None,
    shock_variants: int = 0,
    shock_scale_values: Tuple[float, ...] = (0.8, 1.0, 1.2),
    shock_permute_when: bool = True,
    shock_seed: int = 20240201,
    shock_per_event_scale: bool = False,
) -> List[Path]:
    run_dirs: List[Path] = []
    manifest_rows: List[Dict[str, Any]] = []

    combos_list = list(combinations)
    total_runs = 0
    for sc_name in scenarios:
        if sc_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{sc_name}'. Available: {list(SCENARIOS.keys())}")
        sc_cfg = SCENARIOS[sc_name]
        scenario_events = sc_cfg.get("scenario", [])
        variants_count = shock_variants if (shock_variants > 0 and scenario_events) else 1
        total_runs += len(combos_list) * variants_count

    count = 0
    rng = random.Random(shock_seed)
    for sc_name in scenarios:
        sc_cfg = SCENARIOS[sc_name]
        scenario_events = sc_cfg.get("scenario", [])
        base_params = _merge_params(sc_cfg.get("params"))

        if shock_variants > 0 and scenario_events:
            variants = _generate_shock_variants(
                base_events=scenario_events,
                n_variants=shock_variants,
                scale_values=shock_scale_values,
                permute_when=shock_permute_when,
                rng=rng,
                per_event_scale=shock_per_event_scale,
            )
        else:
            variants = [{"events": scenario_events, "scale": None, "permute_when": None, "variant_id": None}]

        for combo in combos_list:
            params = base_params.copy()
            params.update(combo)

            for var in variants:
                var_events = var["events"]
                var_id = var["variant_id"]
                suffix = f"__v{var_id:03d}" if var_id is not None else ""
                exp_id = _build_exp_id(sc_name, params) + suffix

                if dry_run:
                    print(f"[DRY-RUN] {exp_id}")
                    count += 1
                    if max_runs is not None and count >= max_runs:
                        break
                    continue

                print(f"[{count + 1}/{total_runs}] scenario={sc_name} seed={params['seed']} "
                      f"ofat={params.get('ofat_param')}:{params.get('ofat_value')} "
                      f"variant={var_id}")

                run_dir = run_experiment(
                    N=params["N"],
                    T=params["T"],
                    episodes=params["episodes"],
                    add_gov=params["add_gov"],
                    switch_cost=params["switch_cost"],
                    seed=params["seed"],
                    scenario=var_events,
                    exp_id=exp_id,
                    random_policy=params["random_policy"],
                    model_type=params["model_type"],
                    alpha=params["alpha"],
                    gamma=params["gamma"],
                    eps_min=params["eps_min"],
                    eps_max=params["eps_max"],
                )

                meta = {
                    "exp_id": exp_id,
                    "scenario_name": sc_name,
                    "params": params,
                    "scenario": var_events,
                    "variant_id": var_id,
                    "scale_factor": var["scale"],
                    "scale_map": var.get("scale_map"),
                    "permute_when": var["permute_when"],
                }
                _save_run_metadata(run_dir, meta)

                plots_dir = run_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                market_path = _find_first(run_dir, "results_market*.pickle")
                profits_path = _find_first(run_dir, "results_profits*.pickle")
                run_summary(
                    market=str(market_path),
                    profits=str(profits_path),
                    outdir=str(plots_dir),
                    alloc_band="results/alloc_band_filled.xlsx",
                )

                run_dirs.append(run_dir)
                params_with_scale = params.copy()
                params_with_scale["scale_map"] = var.get("scale_map")
                _record_manifest_row(
                    manifest_rows,
                    run_dir,
                    sc_name,
                    var_events,
                    params_with_scale,
                    variant_id=var_id,
                    scale_factor=var["scale"],
                    permute_when=var["permute_when"],
                )

                count += 1
                if max_runs is not None and count >= max_runs:
                    break

            if max_runs is not None and count >= max_runs:
                break

    if dry_run:
        print(f"[DRY-RUN] Total planned runs: {count}")
        return []

    if run_dirs:
        out_excel = out_excel.with_suffix(".xlsx")
        import pandas as pd

        # Build new summary rows for this batch
        new_rows: List[dict] = []
        for run_dir in run_dirs:
            summary = summarize_run(Path(run_dir))
            if summary is not None:
                new_rows.append(summary)
        new_df = pd.DataFrame(new_rows)

        if out_excel.exists():
            try:
                old_df = pd.read_excel(out_excel)
            except Exception:
                old_df = pd.DataFrame()
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
            if "run_id" in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=["run_id"], keep="last")
        else:
            merged_df = new_df

        # Write merged summary
        if "id" in merged_df.columns:
            merged_df = merged_df.drop(columns=["id"])
        merged_df.insert(0, "id", range(1, len(merged_df) + 1))
        out_excel.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
            merged_df.to_excel(writer, index=False, sheet_name="summary")
        print(f"Saved Excel summary to {out_excel}")

        # Merge manifest CSV
        manifest_path = Path("results") / "robustness_runs.csv"
        manifest_path.parent.mkdir(exist_ok=True)
        new_manifest = pd.DataFrame(manifest_rows)
        if manifest_path.exists():
            try:
                old_manifest = pd.read_csv(manifest_path, sep=";", decimal=",")
            except Exception:
                old_manifest = pd.DataFrame()
            manifest_df = pd.concat([old_manifest, new_manifest], ignore_index=True)
            if "run_dir" in manifest_df.columns:
                manifest_df = manifest_df.drop_duplicates(subset=["run_dir"], keep="last")
        else:
            manifest_df = new_manifest

        manifest_df.to_csv(
            manifest_path,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        print(f"Saved robustness run manifest to: {manifest_path}")
        try:
            from robustness_report import generate_report
            generate_report(
                summary_path=out_excel,
                manifest_path=manifest_path,
                out_dir=Path("results") / "robustness_report",
            )
        except Exception as exc:
            print(f"[WARN] Failed to build robustness report: {exc}")

    return run_dirs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run OFAT robustness checks for RL experiments.")
    ap.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS.keys()),
        help="Comma-separated scenario names (default: all).",
    )
    ap.add_argument("--seeds", help="Comma-separated seeds (int).")
    ap.add_argument("--ofat", action="store_true", help="Run one-factor-at-a-time sweeps.")
    ap.add_argument("--ofat-seeds", action="store_true", help="Run seed-only sweep (baseline params).")
    ap.add_argument("--ofat-params", action="store_true", help="Run params-only sweep (fixed seed).")
    ap.add_argument("--shock-only", action="store_true", help="Run shock variants only (baseline params, fixed seed).")
    ap.add_argument(
        "--out",
        default="results/robustness_summary.xlsx",
        help="Output Excel file path.",
    )
    ap.add_argument("--shock-variants", type=int, default=0, help="Number of shock variants to generate.")
    ap.add_argument(
        "--shock-scale-values",
        default="0.8,1.0,1.2",
        help="Comma-separated scale factors for shock magnitudes.",
    )
    ap.add_argument(
        "--shock-permute-when",
        action="store_true",
        help="Shuffle event timing across shock events.",
    )
    ap.add_argument(
        "--shock-seed",
        type=int,
        default=20240201,
        help="RNG seed for shock variants.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    ap.add_argument("--max-runs", type=int, help="Optional cap on total runs.")
    return ap.parse_args()


def main():
    args = parse_args()

    scenarios = _parse_str_list(args.scenarios)
    if not args.ofat:
        raise SystemExit("This script now supports OFAT only. Use --ofat.")

    seeds = _parse_int_list(args.seeds) if args.seeds else tuple(range(1001, 1011))
    base_seed = seeds[0] if seeds else 1001

    base = {
        "N": 100,
        "T": 1000,
        "episodes": 5,
        "switch_cost": DEFAULT_PARAMS["switch_cost"],
        "model_type": "Q2",
        "random_policy": False,
        "alpha": 0.05,
        "gamma": 0.90,
        "eps_min": 0.01,
        "eps_max": 0.10,
    }

    # OFAT parameter ranges (edit here if you want different tested values)
    ranges = {
        "N": (50, 100, 150, 500),
        "T": (1000, 2000),
        "episodes": (5,),
        "alpha": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        "gamma": (0.7, 0.8, 0.9, 0.95, 0.99),
        "eps_pairs": (
            (0.001, 0.05),
            (0.005, 0.05),
            (0.01, 0.10),
            (0.02, 0.20),
            (0.05, 0.30),
        ),
    }

    scale_vals = _parse_float_list_or_default(
        args.shock_scale_values, default=(0.8, 1.0, 1.2)
    )

    do_seeds = args.ofat_seeds or (not args.ofat_params and not args.shock_only)
    do_params = args.ofat_params or (not args.ofat_seeds and not args.shock_only)
    do_shocks = args.shock_only or (not args.ofat_seeds and not args.ofat_params)

    if do_seeds:
        seed_combos = _shock_combinations(base, seeds)
        run_robustness(
            scenarios=scenarios,
            combinations=seed_combos,
            out_excel=Path(args.out),
            dry_run=args.dry_run,
            max_runs=args.max_runs,
            shock_variants=0,
            shock_scale_values=scale_vals,
            shock_permute_when=args.shock_permute_when,
            shock_seed=args.shock_seed,
            shock_per_event_scale=False,
        )

    if do_params:
        param_combos = _ofat_combinations(base, ranges, (base_seed,))
        run_robustness(
            scenarios=scenarios,
            combinations=param_combos,
            out_excel=Path(args.out),
            dry_run=args.dry_run,
            max_runs=args.max_runs,
            shock_variants=0,
            shock_scale_values=scale_vals,
            shock_permute_when=args.shock_permute_when,
            shock_seed=args.shock_seed,
            shock_per_event_scale=False,
        )

    if do_shocks and args.shock_variants > 0:
        shock_combos = _shock_combinations(base, (base_seed,))
        run_robustness(
            scenarios=scenarios,
            combinations=shock_combos,
            out_excel=Path(args.out),
            dry_run=args.dry_run,
            max_runs=args.max_runs,
            shock_variants=args.shock_variants,
            shock_scale_values=scale_vals,
            shock_permute_when=args.shock_permute_when,
            shock_seed=args.shock_seed,
            shock_per_event_scale=False,
        )


if __name__ == "__main__":
    main()
