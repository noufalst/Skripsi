
"""
================================================================================
RUNNER — C3 / WEDELIA GSW-DRIVEN GREEN ROOF MODEL
================================================================================

Purpose
-------
Generate the C3 model prediction CSV that `run_c3_quick_validation.py` needs.

This runner reuses the same physical model core as the CAM v7 package but passes
a C3/Wedelia plant object:

    plant_type = "C3"
    H_f        = 0.276 m
    LAI        = 1.07
    rho_f      = 0.438
    tau_f      = 0.20
    alpha_f    = 1 - rho_f - tau_f = 0.362

Default C3 validation window:
    2026-04-09 11:05:00 -> 2026-04-10 14:08:00

Main outputs:
    outputs_c3_gsw_fixed_inputs/
        c3_gsw_prediction_full_with_spinup.csv
        c3_gsw_prediction_eval_window.csv
        c3_gsw_metrics.json
        c3_gsw_validation.png

Important:
    This is a quick thesis/meeting runner. It does not prove sensor mapping.
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import green_roof_c3_gsw as gr


# ==============================================================================
# 01 — DEFAULTS
# ==============================================================================

DEFAULT_BASE_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path("outputs_c3_gsw_fixed_inputs")

DEFAULT_WEATHER_FILE = "weatherfile mar-april.xlsx"
DEFAULT_WEATHER_SHEET = "3-24april"

DEFAULT_NI_FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]

# C3 soil moisture files observed in the project.
DEFAULT_SOIL_SENSOR_1 = "sensor 1 COM5_C3.csv"
DEFAULT_SOIL_SENSOR_2 = "sensor 2 COM6_C3.csv"

# C3 target from old nomenclature/documentation.
# T3Ka = Atap Indoor C3, T3Kd = Ruangan C3.
DEFAULT_TARGET_COL = "T3Ka"
DEFAULT_TIN_COL = "T3Kd"

EVAL_START = "2026-04-09 11:05:00"
EVAL_END = "2026-04-10 14:08:00"
DEFAULT_SPINUP_HOURS = 6.0

# Fixed input guesses.
DEFAULT_PARAMS = {
    "H_g": 0.10,
    "H_slab": 0.10,
    "h_in": 8.0,
    "rho_g": 400.0,
    "theta_sat": 0.90,
    "k_theta_sat": 5e-6,
    "lambda_dry": 0.12,
    "lambda_sat": None,

    # Wedelia/C3 plant properties
    "H_f": 0.276,
    "d_f": 0.060,
    "LAI": 1.07,
    "cover_fraction": 0.324,   # temporary ImageJ-like cover estimate; override if needed
    "rho_f": 0.438,
    "tau_f": 0.20,
    "epsilon_f": 0.95,

    # r_s_min C3 from previous LI-COR extraction/documentation.
    "r_s_c3": 167.2,
}


# ==============================================================================
# 02 — HELPERS
# ==============================================================================

def existing_paths(base_dir: Path, names: Sequence[str]) -> List[Path]:
    paths = []
    for name in names:
        p = base_dir / name
        if p.exists():
            paths.append(p)
        else:
            print(f"WARNING: file not found, skipped: {p}")
    return paths


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_c3_plant(args: argparse.Namespace) -> gr.PlantParameters:
    alpha_f = max(0.0, 1.0 - args.rho_f - args.tau_f)
    return gr.PlantParameters(
        name="Wedelia (C3)",
        plant_type="C3",
        H_f=float(args.H_f),
        d_f=float(args.d_f),
        LAI=float(args.LAI),
        cover_fraction=float(args.cover_fraction),
        rho_f=float(args.rho_f),
        tau_f=float(args.tau_f),
        alpha_f=float(alpha_f),
        epsilon_f=float(args.epsilon_f),
    )


def make_c3_rs_profile(args: argparse.Namespace) -> gr.CAMRsProfile:
    """
    Build a simple C3 stomatal profile from r_s.

    r_s = 1 / (gsw * 0.0224)
    so gsw = 1 / (r_s * 0.0224)

    For C3 quick validation, use one constant daytime-friendly conductance unless
    user overrides r_s. This is intentionally transparent.
    """
    r_s = float(args.r_s_c3)
    gsw = float(gr.r_s_to_gsw_mol_m2_s(r_s))

    hourly = {h: gsw for h in range(24)}
    return gr.CAMRsProfile(
        r_stoma_min_s_m=r_s,
        hourly_gsw_median=hourly,
        fallback_gsw_mol_m2_s=gsw,
        r_s_min_limit_s_m=40.0,
        r_s_max_limit_s_m=8000.0,
        source_summary=(
            f"C3 constant gsw profile from r_s={r_s:.2f} s/m; "
            "use --r-s-c3 to override if LI-COR extraction changes."
        ),
    )


def apply_global_material_params(args: argparse.Namespace) -> None:
    gr.geom.H_g = float(args.H_g)
    gr.slab.H_slab = float(args.H_slab)
    gr.geom.h_in = float(args.h_in)

    gr.substrat.rho_g = float(args.rho_g)
    gr.substrat.theta_sat = float(args.theta_sat)
    gr.substrat.k_theta_sat = float(args.k_theta_sat)
    gr.substrat.lambda_dry = float(args.lambda_dry)

    if args.lambda_sat is None:
        gr.substrat.lambda_sat = float(gr.substrat.lambda_dry + gr.substrat.theta_sat * gr.substrat.lambda_water)
    else:
        gr.substrat.lambda_sat = float(args.lambda_sat)

    gr.num.dt = float(args.final_dt)
    gr.num.Nz_substrate = int(args.final_nz_g)
    gr.num.Nz_slab = int(args.final_nz_s)


def scale_soil_moisture(soil: pd.DataFrame, scale: float, theta_sat: float) -> pd.DataFrame:
    """
    Temporary RK520 sensitivity correction.

    scale=0.70 means raw theta is reduced by 30%.
    This is NOT calibration.
    """
    out = soil.copy()
    for col in ["theta_shallow", "theta_deep"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(scale)
            out[col] = out[col].clip(lower=gr.substrat.theta_min, upper=theta_sat)
    for col in ["theta_shallow_pct", "theta_deep_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(scale)
    return out


def validation_metrics(sim: pd.Series, obs: pd.Series) -> dict:
    sim = sim.sort_index().resample("1min").mean()
    obs = obs.sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)
    df = pd.concat([sim.loc[common].rename("sim"), obs.loc[common].rename("obs")], axis=1).dropna()
    if df.empty:
        return {"n": 0, "bias_C": np.nan, "mae_C": np.nan, "rmse_C": np.nan}
    diff = df["sim"] - df["obs"]
    return {
        "n": int(len(df)),
        "bias_C": float(diff.mean()),
        "mae_C": float(np.mean(np.abs(diff))),
        "rmse_C": float(np.sqrt(np.mean(diff ** 2))),
        "corr": float(np.corrcoef(df["sim"], df["obs"])[0, 1]) if len(df) > 3 else np.nan,
        "amp_model_C": float(df["sim"].quantile(0.95) - df["sim"].quantile(0.05)),
        "amp_measured_C": float(df["obs"].quantile(0.95) - df["obs"].quantile(0.05)),
        "amp_error_C": float((df["sim"].quantile(0.95) - df["sim"].quantile(0.05)) - (df["obs"].quantile(0.95) - df["obs"].quantile(0.05))),
        "peak_error_C": float(df["sim"].max() - df["obs"].max()),
    }


def plot_validation(sim_df, ni_df, target_col, eval_start, eval_end, metrics, output_path):
    sim_eval = sim_df[(sim_df.index >= eval_start) & (sim_df.index <= eval_end)]
    obs = ni_df[target_col].sort_index().resample("1min").mean()
    obs_eval = obs[(obs.index >= eval_start) & (obs.index <= eval_end)]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(obs_eval.index, obs_eval.values, label=f"Measured {target_col}", linewidth=2.0)
    axes[0].plot(sim_eval.index, sim_eval["T_s_in"], label="Model T_s_in", linestyle="--", linewidth=2.0)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(
        "C3 / Wedelia validation\n"
        f"RMSE={metrics.get('rmse_C', np.nan):.3f}°C | "
        f"Bias={metrics.get('bias_C', np.nan):.3f}°C | "
        f"Corr={metrics.get('corr', np.nan):.3f} | "
        f"AmpErr={metrics.get('amp_error_C', np.nan):.3f}°C"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    common = sim_eval.index.intersection(obs_eval.index)
    err = sim_eval.loc[common, "T_s_in"] - obs_eval.loc[common]
    axes[1].plot(err.index, err.values, label="Model - measured")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Error (°C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(sim_eval.index, sim_eval["T_g_top"], label="T_g_top")
    axes[2].plot(sim_eval.index, sim_eval["T_g_bot"], label="T_g_bot")
    axes[2].plot(sim_eval.index, sim_eval["T_s_top"], label="T_s_top")
    axes[2].plot(sim_eval.index, sim_eval["T_s_in"], label="T_s_in")
    axes[2].set_ylabel("Model nodes (°C)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=4, fontsize=8)

    axes[3].plot(sim_eval.index, sim_eval["theta_top"], label="theta_top")
    axes[3].plot(sim_eval.index, sim_eval["theta_bot"], label="theta_bot")
    axes[3].plot(sim_eval.index, sim_eval["j_eva_total"] * 3600, label="ET total kg/m²h")
    axes[3].set_ylabel("Moisture / ET")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# ==============================================================================
# 03 — CLI
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run C3/Wedelia green-roof model and save prediction CSV.")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    p.add_argument("--weather-sheet", default=DEFAULT_WEATHER_SHEET)
    p.add_argument("--ni-files", nargs="*", default=DEFAULT_NI_FILES)
    p.add_argument("--soil-sensor-1", default=DEFAULT_SOIL_SENSOR_1)
    p.add_argument("--soil-sensor-2", default=DEFAULT_SOIL_SENSOR_2)
    p.add_argument("--soil-timestamp-mode", choices=["gmt_to_wib", "local"], default="gmt_to_wib")
    p.add_argument("--swap-depths", action="store_true")
    p.add_argument("--no-soil", action="store_true")

    p.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    p.add_argument("--tin-col", default=DEFAULT_TIN_COL)
    p.add_argument("--eval-start", default=EVAL_START)
    p.add_argument("--eval-end", default=EVAL_END)
    p.add_argument("--spinup-hours", type=float, default=DEFAULT_SPINUP_HOURS)

    p.add_argument("--soil-moisture-scale", type=float, default=0.70,
                   help="Temporary RK520 sensitivity scale. 0.70 = reduce raw VWC by 30%%. Use 1.0 to disable.")

    p.add_argument("--H-g", type=float, default=DEFAULT_PARAMS["H_g"])
    p.add_argument("--H-slab", type=float, default=DEFAULT_PARAMS["H_slab"])
    p.add_argument("--h-in", type=float, default=DEFAULT_PARAMS["h_in"])
    p.add_argument("--rho-g", type=float, default=DEFAULT_PARAMS["rho_g"])
    p.add_argument("--theta-sat", type=float, default=DEFAULT_PARAMS["theta_sat"])
    p.add_argument("--k-theta-sat", type=float, default=DEFAULT_PARAMS["k_theta_sat"])
    p.add_argument("--lambda-dry", type=float, default=DEFAULT_PARAMS["lambda_dry"])
    p.add_argument("--lambda-sat", type=float, default=DEFAULT_PARAMS["lambda_sat"])

    p.add_argument("--H-f", type=float, default=DEFAULT_PARAMS["H_f"])
    p.add_argument("--d-f", type=float, default=DEFAULT_PARAMS["d_f"])
    p.add_argument("--LAI", type=float, default=DEFAULT_PARAMS["LAI"])
    p.add_argument("--cover-fraction", type=float, default=DEFAULT_PARAMS["cover_fraction"])
    p.add_argument("--rho-f", type=float, default=DEFAULT_PARAMS["rho_f"])
    p.add_argument("--tau-f", type=float, default=DEFAULT_PARAMS["tau_f"])
    p.add_argument("--epsilon-f", type=float, default=DEFAULT_PARAMS["epsilon_f"])
    p.add_argument("--r-s-c3", type=float, default=DEFAULT_PARAMS["r_s_c3"])

    p.add_argument("--final-dt", type=float, default=60.0)
    p.add_argument("--final-nz-g", type=int, default=67)
    p.add_argument("--final-nz-s", type=int, default=41)
    p.add_argument("--quiet", action="store_true")
    return p


# ==============================================================================
# 04 — MAIN
# ==============================================================================

def main():
    args = build_parser().parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir
    ensure_output_dir(output_dir)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    sim_start = eval_start - pd.Timedelta(hours=args.spinup_hours)
    sim_end = eval_end

    print("\n" + "=" * 72)
    print("C3 / WEDELIA GSW-DRIVEN MODEL RUNNER")
    print("=" * 72)
    print(f"BASE_DIR    : {base_dir.resolve()}")
    print(f"OUTPUT_DIR  : {output_dir.resolve()}")
    print(f"SIM WINDOW  : {sim_start} -> {sim_end}")
    print(f"EVAL WINDOW : {eval_start} -> {eval_end}")
    print(f"TARGET      : {args.target_col}")
    print(f"T_in        : {args.tin_col}")
    print(f"SM scale    : {args.soil_moisture_scale} (0.70 means -30%)")

    apply_global_material_params(args)
    c3_plant = make_c3_plant(args)
    rs_profile = make_c3_rs_profile(args)

    print("\n=== C3 PLANT PARAMETERS ===")
    print(f"name           : {c3_plant.name}")
    print(f"H_f            : {c3_plant.H_f}")
    print(f"LAI            : {c3_plant.LAI}")
    print(f"cover_fraction : {c3_plant.cover_fraction}")
    print(f"rho_f/tau/alpha: {c3_plant.rho_f:.3f} / {c3_plant.tau_f:.3f} / {c3_plant.alpha_f:.3f}")
    print(f"r_s_C3         : {rs_profile.r_stoma_min_s_m:.2f} s/m")
    print(f"gsw equiv      : {rs_profile.fallback_gsw_mol_m2_s:.4f} mol m-2 s-1")

    weather = gr.load_weather_data(
        base_dir / args.weather_file,
        date_start=str(sim_start),
        date_end=str(sim_end),
        sheet_name=args.weather_sheet,
    )

    ni_paths = existing_paths(base_dir, args.ni_files)
    ni = gr.load_multiple_NI_sensor_data(ni_paths)
    ni = ni[(ni.index >= sim_start) & (ni.index <= sim_end)].copy()
    if ni.empty:
        raise ValueError("NI data empty inside C3 simulation window.")
    if args.target_col not in ni.columns:
        raise ValueError(f"Target column {args.target_col} not found. Available: {ni.columns.tolist()}")
    if args.tin_col not in ni.columns:
        raise ValueError(f"T_in column {args.tin_col} not found. Available: {ni.columns.tolist()}")

    soil = None
    theta_initial = None
    if not args.no_soil:
        p1 = base_dir / args.soil_sensor_1
        p2 = base_dir / args.soil_sensor_2
        if p1.exists() and p2.exists():
            soil = gr.load_cam_soil_moisture(
                p1, p2,
                timestamp_mode=args.soil_timestamp_mode,
                swap_depths=args.swap_depths,
            )
            if args.soil_moisture_scale != 1.0:
                soil = scale_soil_moisture(soil, args.soil_moisture_scale, gr.substrat.theta_sat)
                print(f"Applied temporary soil moisture scale: {args.soil_moisture_scale}")
            theta_initial = gr.get_theta_initial_from_soil(soil, sim_start, gr.substrat)
            print(f"theta_initial from C3 soil: {theta_initial}")
        else:
            print("WARNING: C3 soil files not found; using default theta initial.")

    target_series = ni[args.target_col].sort_index().resample("1min").mean().interpolate("time")
    target_near = target_series[target_series.index >= sim_start].dropna()
    T_s_in_initial_C = float(target_near.iloc[0]) if len(target_near) else None

    T_g_top_initial_C = None
    # If target sensor candidates are available, use a near-surface-looking T3 channel as initial top guess.
    for c in ["T3Ke", "T3Kb", "T3Kc", "T3Ka"]:
        if c in ni.columns:
            s = ni[c].sort_index().resample("1min").mean().interpolate("time")
            near = s[s.index >= sim_start].dropna()
            if len(near):
                T_g_top_initial_C = float(near.iloc[0])
                break

    results = gr.run_cam_simulation(
        weather_df=weather,
        plant=c3_plant,
        substrate=gr.substrat,
        slab_params=gr.slab,
        geom_params=gr.geom,
        num_params=gr.num,
        rs_profile=rs_profile,
        theta_initial=theta_initial,
        T_in_series=ni[args.tin_col],
        T_s_in_initial_C=T_s_in_initial_C,
        T_g_top_initial_C=T_g_top_initial_C,
        verbose=not args.quiet,
    )

    sim_df = gr.results_to_dataframe(results)
    sim_eval = sim_df[(sim_df.index >= eval_start) & (sim_df.index <= eval_end)].copy()
    metrics = validation_metrics(sim_eval["T_s_in"], ni[args.target_col])

    sim_df.to_csv(output_dir / "c3_gsw_prediction_full_with_spinup.csv", index_label="datetime")
    sim_eval.to_csv(output_dir / "c3_gsw_prediction_eval_window.csv", index_label="datetime")

    with open(output_dir / "c3_gsw_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "c3_run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "eval_start": str(eval_start),
            "eval_end": str(eval_end),
            "target_col": args.target_col,
            "tin_col": args.tin_col,
            "soil_moisture_scale": args.soil_moisture_scale,
            "plant": {
                "H_f": c3_plant.H_f,
                "LAI": c3_plant.LAI,
                "cover_fraction": c3_plant.cover_fraction,
                "rho_f": c3_plant.rho_f,
                "tau_f": c3_plant.tau_f,
                "alpha_f": c3_plant.alpha_f,
            },
            "r_s_c3": args.r_s_c3,
            "metrics": metrics,
        }, f, indent=2)

    plot_validation(
        sim_df, ni,
        target_col=args.target_col,
        eval_start=eval_start,
        eval_end=eval_end,
        metrics=metrics,
        output_path=output_dir / "c3_gsw_validation.png",
    )

    print("\n=== DONE ===")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("Saved:")
    print(f"  {output_dir / 'c3_gsw_prediction_eval_window.csv'}")
    print(f"  {output_dir / 'c3_gsw_prediction_full_with_spinup.csv'}")
    print(f"  {output_dir / 'c3_gsw_metrics.json'}")
    print(f"  {output_dir / 'c3_gsw_validation.png'}")


if __name__ == "__main__":
    main()
