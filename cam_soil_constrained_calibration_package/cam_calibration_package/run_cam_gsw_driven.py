"""
================================================================================
RUNNER — CAM DATA-DRIVEN LI-COR gsw/r_s + FIXED-INPUT VALIDATION
================================================================================
Default purpose:
    Run CAM/Bromelia validation only.

Why this runner exists:
    1. Keep model equations separate from execution.
    2. Keep all file paths and calibration switches in one visible section.
    3. Use LI-COR gsw as r_s / r_stoma, not as unused side information.
    4. Use spin-up before validation so the model is not allowed to cheat by
       freely choosing unrealistic initial temperature.
    5. Keep measured/material parameters fixed; optional calibration is limited
       to uncertain boundary/initial-state terms only.

Basic run:
    python run_cam_gsw_driven.py

Fast diagnostic run with custom base folder:
    python run_cam_gsw_driven.py --base-dir "E:\\Pagi\\SKRRRRRRRipsi\\data"

Use specific LI-COR files:
    python run_cam_gsw_driven.py --licor-files "2026-04-01-1616_logdata cam new d1s3.xlsx" "2026-03-31-2148_logdata cam new d1s4.csv"
================================================================================
"""

from __future__ import annotations

# ==============================================================================
# 00 — IMPORTS
# ==============================================================================

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import green_roof_cam_gsw as gr


# ==============================================================================
# 01 — DEFAULT CONFIGURATION
# ==============================================================================

DEFAULT_BASE_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path("outputs_cam_gsw_fixed_inputs")

# Evaluation window: where metrics are calculated.
EVAL_START = "2026-03-31 11:58:00"
EVAL_END = "2026-04-02 21:42:00"

# Spin-up is simulated before EVAL_START, but not used for metrics.
# It helps initialize substrate/slab thermal state without optimizing initial_temp.
DEFAULT_SPINUP_HOURS = 6.0

DEFAULT_WEATHER_FILE = "weatherfile mar-april.xlsx"
DEFAULT_NI_FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]
DEFAULT_SOIL_SENSOR_1 = "sensor 1 COM5_CAM.csv"
DEFAULT_SOIL_SENSOR_2 = "sensor 2 COM6_CAM.csv"
DEFAULT_LICOR_GLOB = "*_logdata cam*"

# CAM target sensor.
# T1Tb is the most defensible inner roof / underside CAM target.
DEFAULT_TARGET_COL = "T1Tb"
DEFAULT_TIN_COL = "T1Ka"

# Scientific guess / measured parameters currently used.
PARAMETER_GUESS = {
    "H_g": 0.10,
    "H_slab": 0.10,
    "h_in": 8.0,
    "rho_g": 400.0,
    "theta_sat": 0.90,
    "k_theta_sat": 5e-6,
    "lambda_dry": 0.12,
    "lambda_sat": None,
    "LAI": 1.95,
    "cover_fraction": 0.95,
    "tau_f": 0.07,
}


# ==============================================================================
# 02 — ARGUMENT PARSER
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAM/Bromelia simulation with data-driven LI-COR gsw/r_s profile."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--weather-sheet", default="3-24april")
    parser.add_argument("--ni-files", nargs="*", default=DEFAULT_NI_FILES)
    parser.add_argument("--soil-sensor-1", default=DEFAULT_SOIL_SENSOR_1)
    parser.add_argument("--soil-sensor-2", default=DEFAULT_SOIL_SENSOR_2)
    parser.add_argument("--licor-files", nargs="*", default=None)
    parser.add_argument("--hourly-gsw-profile", type=Path, default=None,
                        help="Optional CSV/XLSX with columns hour,gsw to override/build hourly CAM gsw profile.")
    parser.add_argument("--manual-gsw", nargs="*", default=None,
                        help="Manual hourly gsw pairs, e.g. --manual-gsw 10:0.03 16:0.2261 21:0.444")

    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--tin-col", default=DEFAULT_TIN_COL)
    parser.add_argument("--eval-start", default=EVAL_START)
    parser.add_argument("--eval-end", default=EVAL_END)
    parser.add_argument("--spinup-hours", type=float, default=DEFAULT_SPINUP_HOURS)

    parser.add_argument("--soil-timestamp-mode", choices=["gmt_to_wib", "local"], default="gmt_to_wib")
    parser.add_argument("--swap-depths", action="store_true")
    parser.add_argument("--no-soil", action="store_true", help="Run without soil moisture initial profile.")

    # Direct parameter overrides for quick sensitivity checks.
    parser.add_argument("--H-g", type=float, default=PARAMETER_GUESS["H_g"])
    parser.add_argument("--H-slab", type=float, default=PARAMETER_GUESS["H_slab"])
    parser.add_argument("--h-in", type=float, default=PARAMETER_GUESS["h_in"])
    parser.add_argument("--rho-g", type=float, default=PARAMETER_GUESS["rho_g"])
    parser.add_argument("--theta-sat", type=float, default=PARAMETER_GUESS["theta_sat"])
    parser.add_argument("--k-theta-sat", type=float, default=PARAMETER_GUESS["k_theta_sat"])
    parser.add_argument("--lambda-dry", type=float, default=PARAMETER_GUESS["lambda_dry"])
    parser.add_argument("--lambda-sat", type=float, default=None)
    parser.add_argument("--LAI", type=float, default=PARAMETER_GUESS["LAI"])
    parser.add_argument("--cover-fraction", type=float, default=PARAMETER_GUESS["cover_fraction"])
    parser.add_argument("--tau-f", type=float, default=PARAMETER_GUESS["tau_f"])

    # Optional manual r_s values if user wants to force values from spreadsheet notes.
    # These are converted into hourly gsw values; they are NOT treated as separate r_stoma_min.
    parser.add_argument("--r-s-night", type=float, default=None)
    parser.add_argument("--r-s-midday", type=float, default=None)
    parser.add_argument("--r-s-late", type=float, default=None)

    # Optional LIMITED calibration.
    # Deliberately does NOT tune H_g, H_slab, LAI, cover_fraction, rho_g, theta_sat, tau_f, or LI-COR r_s.
    # Default limited calibration only searches h_in because indoor convection in the small box is uncertain.
    parser.add_argument("--calibrate-thermal", action="store_true",
                        help="Limited calibration: tune h_in only by default; keep measured/material parameters fixed.")
    parser.add_argument("--calibrate-initial-theta", action="store_true",
                        help="Also tune initial theta scale. Use only when running no-soil/predictive tests or when initial moisture is uncertain.")
    parser.add_argument("--n-trials", type=int, default=40, help="Number of random limited-calibration trials.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--calibration-dt", type=float, default=60.0, help="Coarse dt for calibration trials; final run still uses --final-dt.")
    parser.add_argument("--final-dt", type=float, default=60.0, help="Final simulation dt. Use 1 for paper-style, 60 for faster thesis plotting.")
    parser.add_argument("--calibration-nz-g", type=int, default=35)
    parser.add_argument("--calibration-nz-s", type=int, default=25)
    parser.add_argument("--final-nz-g", type=int, default=67)
    parser.add_argument("--final-nz-s", type=int, default=41)

    parser.add_argument("--quiet", action="store_true")
    return parser


# ==============================================================================
# 03 — FILE DISCOVERY HELPERS
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


def discover_licor_files(base_dir: Path, explicit_files: Optional[Sequence[str]]) -> List[Path]:
    if explicit_files:
        return existing_paths(base_dir, explicit_files)

    # Conservative auto-discovery: include xlsx/csv files with cam and logdata in name.
    candidates = []
    for p in base_dir.glob("*"):
        name = p.name.lower()
        if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"} and "cam" in name and "logdata" in name:
            candidates.append(p)
    return sorted(candidates)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 04 — PLOTTING
# ==============================================================================

def plot_cam_validation(
    sim_df: pd.DataFrame,
    ni_df: pd.DataFrame,
    soil_df: Optional[pd.DataFrame],
    target_col: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    metrics: Dict[str, float],
    save_path: Path,
) -> None:
    """Create CAM validation plot with r_s and separated ET diagnostics."""
    eval_mask = (sim_df.index >= eval_start) & (sim_df.index <= eval_end)
    sim_eval = sim_df.loc[eval_mask]
    obs = ni_df[target_col].sort_index().resample("1min").mean()
    obs_eval = obs[(obs.index >= eval_start) & (obs.index <= eval_end)]

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

    axes[0].plot(obs_eval.index, obs_eval.values, label=f"Measured {target_col}", linewidth=2.2)
    axes[0].plot(sim_eval.index, sim_eval["T_s_in"], label="Model T_s_in", linestyle="--", linewidth=2.0)
    axes[0].set_ylabel("T_s,in (°C)")
    axes[0].set_title(
        "CAM physical validation — measured vs model\n"
        f"RMSE={metrics.get('rmse_C', np.nan):.3f}°C | MAE={metrics.get('mae_C', np.nan):.3f}°C | "
        f"Bias={metrics.get('bias_C', np.nan):.3f}°C | AmpErr={metrics.get('amp_error_C', np.nan):.3f}°C"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    common = sim_eval.index.intersection(obs_eval.index)
    err = sim_eval.loc[common, "T_s_in"] - obs_eval.loc[common]
    axes[1].plot(err.index, err.values, linewidth=1.5, label="Model - measured")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error (°C)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sim_eval.index, sim_eval["r_s_s_m"], label="r_s used", linewidth=1.6)
    axes[2].set_ylabel("r_s (s/m)")
    ax2 = axes[2].twinx()
    ax2.plot(sim_eval.index, sim_eval["gsw_equiv_mol_m2_s"], linestyle="--", label="gsw equiv", linewidth=1.2)
    ax2.set_ylabel("gsw equiv (mol m⁻² s⁻¹)")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(sim_eval.index, np.asarray(sim_eval["j_eva_f"]) * 3600, label="j_eva_f foliage", linewidth=1.6)
    axes[3].plot(sim_eval.index, np.asarray(sim_eval["j_eva_g"]) * 3600, label="j_eva_g substrate", linewidth=1.6)
    axes[3].plot(sim_eval.index, np.asarray(sim_eval["j_eva_total"]) * 3600, label="total ET", linestyle="--", linewidth=1.2)
    axes[3].set_ylabel("ET (kg m⁻² h⁻¹)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(sim_eval.index, sim_eval["T_a"], label="T_air", linewidth=1.2)
    axes[4].plot(sim_eval.index, sim_eval["T_in_used"], label="T_in used", linewidth=1.2)
    axes[4].set_ylabel("Temperature (°C)")
    axes[4].set_xlabel("Datetime")
    axes[4].grid(True, alpha=0.3)

    ax4b = axes[4].twinx()
    ax4b.plot(sim_eval.index, sim_eval["G_sol"], linestyle="--", label="G_sol", linewidth=1.0)
    ax4b.set_ylabel("Solar (W/m²)")
    l1, lab1 = axes[4].get_legend_handles_labels()
    l2, lab2 = ax4b.get_legend_handles_labels()
    axes[4].legend(l1 + l2, lab1 + lab2, loc="upper right")

    if soil_df is not None and not soil_df.empty:
        # Add soil moisture as light markers on axis 3 if available by creating an inset-like second axis.
        # Kept out of main plot to avoid over-cluttering.
        pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cam_soil_and_et(
    sim_df: pd.DataFrame,
    soil_df: Optional[pd.DataFrame],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    save_path: Path,
) -> None:
    """Plot measured soil moisture against simulated theta and ET split."""
    sim = sim_df[(sim_df.index >= eval_start) & (sim_df.index <= eval_end)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(sim.index, sim["theta_top"], label="model theta_top")
    axes[0].plot(sim.index, sim["theta_mid"], label="model theta_mid")
    axes[0].plot(sim.index, sim["theta_bot"], label="model theta_bot")
    if soil_df is not None and not soil_df.empty:
        soil = soil_df[(soil_df.index >= eval_start) & (soil_df.index <= eval_end)].resample("1min").mean()
        if "theta_shallow" in soil:
            axes[0].plot(soil.index, soil["theta_shallow"], "--", label="measured shallow")
        if "theta_deep" in soil:
            axes[0].plot(soil.index, soil["theta_deep"], "--", label="measured deep")
    axes[0].set_ylabel("theta (-)")
    axes[0].set_title("CAM soil moisture: model vs measured")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sim.index, np.asarray(sim["j_eva_f"]) * 3600, label="foliage transpiration")
    axes[1].plot(sim.index, np.asarray(sim["j_eva_g"]) * 3600, label="substrate evaporation")
    axes[1].set_ylabel("kg m⁻² h⁻¹")
    axes[1].set_title("ET split")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sim.index, sim["G_sol"], label="G_sol")
    axb = axes[2].twinx()
    axb.plot(sim.index, sim["VPD_Pa"] / 1000, linestyle="--", label="VPD")
    axes[2].set_ylabel("Solar (W/m²)")
    axb.set_ylabel("VPD (kPa)")
    axes[2].set_xlabel("Datetime")
    l1, lab1 = axes[2].get_legend_handles_labels()
    l2, lab2 = axb.get_legend_handles_labels()
    axes[2].legend(l1 + l2, lab1 + lab2, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 05 — THERMAL CALIBRATION HELPERS
# ==============================================================================

def apply_parameter_dict(params: Dict[str, float]) -> None:
    """Apply a sampled physical/effective parameter set to the global model."""
    gr.apply_cam_parameters(
        H_g=params.get("H_g", gr.geom.H_g),
        H_slab=params.get("H_slab", gr.slab.H_slab),
        h_in=params.get("h_in", gr.geom.h_in),
        rho_g=params.get("rho_g", gr.substrat.rho_g),
        theta_sat=params.get("theta_sat", gr.substrat.theta_sat),
        k_theta_sat=params.get("k_theta_sat", gr.substrat.k_theta_sat),
        lambda_dry=params.get("lambda_dry", gr.substrat.lambda_dry),
        lambda_sat=params.get("lambda_sat", gr.substrat.lambda_sat),
        LAI=params.get("LAI", gr.bromelia.LAI),
        cover_fraction=params.get("cover_fraction", gr.bromelia.cover_fraction),
        tau_f=params.get("tau_f", gr.bromelia.tau_f),
    )


def objective_from_metrics(metrics: Dict[str, float]) -> float:
    """
    Score that punishes flat models.

    RMSE alone can reward an over-damped line. For your CAM case, amplitude and
    peak timing/height matter, so AmpErr and PeakErr are explicitly included.
    """
    if metrics.get("n", 0) <= 0 or not np.isfinite(metrics.get("rmse_C", np.nan)):
        return 1e9
    rmse = abs(metrics.get("rmse_C", np.nan))
    amp = abs(metrics.get("amp_error_C", np.nan))
    peak = abs(metrics.get("peak_error_C", np.nan))
    bias = abs(metrics.get("bias_C", np.nan))
    return float(rmse + 0.85 * amp + 0.35 * peak + 0.20 * bias)


def scale_theta_initial(theta_initial, factor: float, substrate) -> object:
    """Scale only the initial moisture state, not theta_sat/material properties."""
    factor = float(factor)
    if theta_initial is None:
        return None
    if isinstance(theta_initial, dict):
        vals = np.asarray(theta_initial.get("values", []), dtype=float)
        if vals.size == 0:
            return theta_initial
        return {
            "depths": list(theta_initial.get("depths", [0.02, 0.07])),
            "values": [float(np.clip(v * factor, substrate.theta_min, substrate.theta_sat)) for v in vals],
        }
    if isinstance(theta_initial, (list, tuple, np.ndarray)):
        vals = np.asarray(theta_initial, dtype=float)
        return [float(np.clip(v * factor, substrate.theta_min, substrate.theta_sat)) for v in vals]
    return float(np.clip(float(theta_initial) * factor, substrate.theta_min, substrate.theta_sat))


def sample_thermal_params(
    rng: np.random.Generator,
    base: Dict[str, float],
    calibrate_initial_theta: bool = False,
) -> Dict[str, float]:
    """Sample only defensible uncertain terms.

    Fixed during calibration:
        H_g, H_slab, LAI, cover_fraction, rho_g, theta_sat, tau_f,
        k_theta_sat, lambda_dry, lambda_sat, and all LI-COR-derived r_s values.

    Tuned by default:
        h_in only, because natural convection inside a small outdoor test box is
        uncertain and is not directly measured.

    Optional:
        theta_initial_factor, only if --calibrate-initial-theta is used. This
        changes the initial water state, not the material property theta_sat.
    """
    params = dict(base)
    params["h_in"] = float(rng.uniform(1.0, 8.0))
    params["theta_initial_factor"] = 1.0
    if calibrate_initial_theta:
        params["theta_initial_factor"] = float(rng.uniform(0.75, 1.15))
    return params

def run_once_for_score(
    params: Dict[str, float],
    weather: pd.DataFrame,
    ni: pd.DataFrame,
    rs_profile: gr.CAMRsProfile,
    theta_initial,
    target_col: str,
    tin_col: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    T_s_in_initial_C: Optional[float],
    T_g_top_initial_C: Optional[float],
    quiet: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Run one limited-calibration trial and return objective score."""
    apply_parameter_dict(params)
    theta_use = scale_theta_initial(theta_initial, params.get("theta_initial_factor", 1.0), gr.substrat)
    try:
        results = gr.run_cam_simulation(
            weather_df=weather,
            plant=gr.bromelia,
            substrate=gr.substrat,
            slab_params=gr.slab,
            geom_params=gr.geom,
            num_params=gr.num,
            rs_profile=rs_profile,
            theta_initial=theta_use,
            T_in_series=ni[tin_col],
            T_s_in_initial_C=T_s_in_initial_C,
            T_g_top_initial_C=T_g_top_initial_C,
            verbose=not quiet,
        )
        sim_df = gr.results_to_dataframe(results)
        sim_eval = sim_df[(sim_df.index >= eval_start) & (sim_df.index <= eval_end)]
        metrics = gr.validation_metrics(sim_eval["T_s_in"], ni[target_col])
        return objective_from_metrics(metrics), metrics
    except FloatingPointError as exc:
        return 1e9, {"error": f"FloatingPointError: {exc}", "n": 0, "rmse_C": np.nan, "amp_error_C": np.nan}
    except Exception as exc:
        return 1e9, {"error": str(exc), "n": 0, "rmse_C": np.nan, "amp_error_C": np.nan}

def thermal_random_search(
    weather: pd.DataFrame,
    ni: pd.DataFrame,
    rs_profile: gr.CAMRsProfile,
    theta_initial,
    target_col: str,
    tin_col: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    T_s_in_initial_C: Optional[float],
    T_g_top_initial_C: Optional[float],
    n_trials: int,
    seed: int,
    calibrate_initial_theta: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Limited random search while keeping measured/material parameters fixed."""
    rng = np.random.default_rng(seed)
    baseline = {
        "H_g": gr.geom.H_g, "H_slab": gr.slab.H_slab, "h_in": gr.geom.h_in,
        "rho_g": gr.substrat.rho_g, "theta_sat": gr.substrat.theta_sat,
        "k_theta_sat": gr.substrat.k_theta_sat, "lambda_dry": gr.substrat.lambda_dry,
        "lambda_sat": gr.substrat.lambda_sat, "LAI": gr.bromelia.LAI,
        "cover_fraction": gr.bromelia.cover_fraction, "tau_f": gr.bromelia.tau_f,
        "theta_initial_factor": 1.0,
    }

    rows = []
    best_params = dict(baseline)
    best_score, best_metrics = run_once_for_score(
        best_params, weather, ni, rs_profile, theta_initial, target_col, tin_col,
        eval_start, eval_end, T_s_in_initial_C, T_g_top_initial_C, quiet=True,
    )
    rows.append({"trial": 0, "score": best_score, **best_metrics, **best_params})
    print(f"  trial 000 | score={best_score:.3f} | RMSE={best_metrics.get('rmse_C', np.nan):.3f} | AmpErr={best_metrics.get('amp_error_C', np.nan):.3f}")

    for i in range(1, n_trials + 1):
        params = sample_thermal_params(rng, baseline, calibrate_initial_theta=calibrate_initial_theta)
        try:
            score, metrics = run_once_for_score(
                params, weather, ni, rs_profile, theta_initial, target_col, tin_col,
                eval_start, eval_end, T_s_in_initial_C, T_g_top_initial_C, quiet=True,
            )
        except Exception as exc:
            score = 1e9
            metrics = {"error": str(exc), "n": 0, "rmse_C": np.nan, "amp_error_C": np.nan}
        rows.append({"trial": i, "score": score, **metrics, **params})
        if score < best_score:
            best_score = score
            best_params = dict(params)
            best_metrics = dict(metrics)
            print(f"  trial {i:03d} | NEW BEST score={score:.3f} | RMSE={metrics.get('rmse_C', np.nan):.3f} | AmpErr={metrics.get('amp_error_C', np.nan):.3f}")
        elif i % 10 == 0:
            print(f"  trial {i:03d} | score={score:.3f} | best={best_score:.3f}")

    apply_parameter_dict(best_params)
    return best_params, pd.DataFrame(rows).sort_values("score")


# ==============================================================================
# 05 — MAIN WORKFLOW
# ==============================================================================

def main() -> None:
    args = build_parser().parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir
    ensure_output_dir(output_dir)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    sim_start = eval_start - pd.Timedelta(hours=args.spinup_hours)
    sim_end = eval_end

    print("\n" + "=" * 72)
    print("CAM RUNNER — DATA-DRIVEN LI-COR gsw/r_s + FIXED INPUTS")
    print("=" * 72)
    print(f"BASE_DIR    : {base_dir.resolve()}")
    print(f"OUTPUT_DIR  : {output_dir.resolve()}")
    print(f"SIM WINDOW  : {sim_start} -> {sim_end}")
    print(f"EVAL WINDOW : {eval_start} -> {eval_end}")
    print(f"TARGET      : {args.target_col}")
    print(f"T_in        : {args.tin_col}")

    # Apply visible parameter configuration.
    gr.apply_cam_parameters(
        H_g=args.H_g,
        H_slab=args.H_slab,
        h_in=args.h_in,
        rho_g=args.rho_g,
        theta_sat=args.theta_sat,
        k_theta_sat=args.k_theta_sat,
        lambda_dry=args.lambda_dry,
        lambda_sat=args.lambda_sat,
        LAI=args.LAI,
        cover_fraction=args.cover_fraction,
        tau_f=args.tau_f,
    )

    print("\n=== PARAMETER CONFIGURATION ===")
    print(f"H_g            : {gr.geom.H_g} m")
    print(f"H_slab         : {gr.slab.H_slab} m")
    print("Fixed-input note: H_g, H_slab, LAI, cover_fraction, rho_g, theta_sat, tau_f, and LI-COR r_s are fixed during calibration")
    print(f"h_in           : {gr.geom.h_in} W/m2K")
    print(f"rho_g          : {gr.substrat.rho_g} kg/m3")
    print(f"theta_sat      : {gr.substrat.theta_sat}")
    print(f"lambda_dry     : {gr.substrat.lambda_dry} W/mK")
    print(f"lambda_sat     : {gr.substrat.lambda_sat} W/mK")
    print(f"LAI            : {gr.bromelia.LAI}")
    print(f"cover_fraction : {gr.bromelia.cover_fraction}")
    print(f"tau_f          : {gr.bromelia.tau_f}")
    print(f"alpha_f        : {gr.bromelia.alpha_f}")
    if args.calibrate_initial_theta:
        print("Calibration note: initial theta scale is allowed to vary because --calibrate-initial-theta was used.")

    # Load data.
    weather_path = base_dir / args.weather_file
    weather = gr.load_weather_data(
        weather_path,
        date_start=str(sim_start),
        date_end=str(sim_end),
        sheet_name=args.weather_sheet,
    )

    ni_paths = existing_paths(base_dir, args.ni_files)
    ni = gr.load_multiple_NI_sensor_data(ni_paths)
    ni = ni[(ni.index >= sim_start) & (ni.index <= sim_end)].copy()
    if ni.empty:
        raise ValueError("NI data empty inside simulation window.")

    if args.target_col not in ni.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Available: {ni.columns.tolist()}")
    if args.tin_col not in ni.columns:
        raise ValueError(f"T_in column '{args.tin_col}' not found. Available: {ni.columns.tolist()}")

    soil = None
    theta_initial = None
    if not args.no_soil:
        soil_path1 = base_dir / args.soil_sensor_1
        soil_path2 = base_dir / args.soil_sensor_2
        if soil_path1.exists() and soil_path2.exists():
            soil = gr.load_cam_soil_moisture(
                soil_path1,
                soil_path2,
                timestamp_mode=args.soil_timestamp_mode,
                swap_depths=args.swap_depths,
            )
            theta_initial = gr.get_theta_initial_from_soil(soil, sim_start, gr.substrat)
            if theta_initial:
                print(f"theta_initial from soil: {theta_initial}")
            else:
                print("WARNING: could not get theta_initial from soil; fallback to 0.8 theta_sat")
        else:
            print("WARNING: soil files not found; running without measured soil initial profile.")

    # Build r_s profile from LI-COR.
    licor_paths = discover_licor_files(base_dir, args.licor_files)
    if licor_paths:
        print("\n=== LI-COR FILES ===")
        for p in licor_paths:
            print(f"  {p.name}")
        rs_profile = gr.build_cam_rs_profile_from_licor(licor_paths)
    else:
        print("\nWARNING: no LI-COR files found; using default r_s profile.")
        rs_profile = gr.CAMRsProfile()

    # Optional direct hourly gsw profile from spreadsheet summary. This is useful when
    # some LI-COR Excel exports show zeros but you have verified cells manually.
    if args.hourly_gsw_profile is not None:
        profile_path = args.hourly_gsw_profile
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path
        hourly = gr.load_hourly_gsw_profile_csv(profile_path)
        rs_profile.hourly_gsw_median.update(hourly)
        rs_profile.source_summary += f"\nHourly gsw profile loaded from {profile_path.name}"

    if args.manual_gsw:
        rs_profile = gr.profile_from_manual_hourly_gsw(args.manual_gsw, rs_profile)

    # Manual overrides, useful after spreadsheet manual extraction.
    # These set hourly gsw profile values through r_s conversion; they do NOT create
    # three different r_stoma_min values.
    manual_lines = []
    if args.r_s_night is not None:
        gsw = float(gr.r_s_to_gsw_mol_m2_s(args.r_s_night))
        for h in list(range(19, 24)) + list(range(0, 6)):
            rs_profile.hourly_gsw_median[h] = gsw
        manual_lines.append(f"manual night hours r_s={args.r_s_night:.1f} s/m")
    if args.r_s_midday is not None:
        gsw = float(gr.r_s_to_gsw_mol_m2_s(args.r_s_midday))
        for h in range(8, 15):
            rs_profile.hourly_gsw_median[h] = gsw
        manual_lines.append(f"manual midday hours r_s={args.r_s_midday:.1f} s/m")
    if args.r_s_late is not None:
        gsw = float(gr.r_s_to_gsw_mol_m2_s(args.r_s_late))
        for h in range(15, 19):
            rs_profile.hourly_gsw_median[h] = gsw
        manual_lines.append(f"manual late-afternoon hours r_s={args.r_s_late:.1f} s/m")
    if manual_lines:
        rs_profile.source_summary += "\n" + "\n".join(manual_lines)

    print("\n=== CAM DATA-DRIVEN gsw/r_s PROFILE ===")
    print(f"one r_stoma_min_CAM : {rs_profile.r_stoma_min_s_m:.2f} s/m")
    print(f"profile r_s night   : {rs_profile.r_s_night_s_m:.2f} s/m")
    print(f"profile r_s midday  : {rs_profile.r_s_midday_s_m:.2f} s/m")
    print(f"profile r_s late    : {rs_profile.r_s_late_afternoon_s_m:.2f} s/m")
    print("Source summary:")
    print(rs_profile.source_summary)

    # Initial temperature should not be optimized freely.
    target_series = ni[args.target_col].sort_index().resample("1min").mean().interpolate("time")
    target_near_start = target_series[target_series.index >= sim_start].dropna()
    T_s_in_initial_C = float(target_near_start.iloc[0]) if len(target_near_start) else None

    T_g_top_initial_C = None
    if "T1Ke" in ni.columns:
        gtop = ni["T1Ke"].sort_index().resample("1min").mean().interpolate("time")
        gtop_near = gtop[gtop.index >= sim_start].dropna()
        if len(gtop_near):
            T_g_top_initial_C = float(gtop_near.iloc[0])

    # Optional thermal calibration. This targets amplitude/peak by changing thermal/radiation
    # parameters only; the LI-COR gsw/r_s profile is kept fixed.
    best_params = None
    trials_df = None
    if args.calibrate_thermal:
        print("\n=== LIMITED CALIBRATION ===")
        print("Keeping geometry, morphology, material properties, and LI-COR r_s fixed; tuning h_in only by default.")
        old_dt, old_nzg, old_nzs = gr.num.dt, gr.num.Nz_substrate, gr.num.Nz_slab
        gr.num.dt = float(args.calibration_dt)
        gr.num.Nz_substrate = int(args.calibration_nz_g)
        gr.num.Nz_slab = int(args.calibration_nz_s)
        best_params, trials_df = thermal_random_search(
            weather=weather, ni=ni, rs_profile=rs_profile, theta_initial=theta_initial,
            target_col=args.target_col, tin_col=args.tin_col, eval_start=eval_start, eval_end=eval_end,
            T_s_in_initial_C=T_s_in_initial_C, T_g_top_initial_C=T_g_top_initial_C,
            n_trials=args.n_trials, seed=args.random_seed,
            calibrate_initial_theta=args.calibrate_initial_theta,
        )
        trials_df.to_csv(output_dir / "cam_thermal_calibration_trials.csv", index=False)
        with open(output_dir / "cam_best_thermal_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print("Best thermal parameters:")
        print(json.dumps(best_params, indent=2))

    # Final run resolution.
    gr.num.dt = float(args.final_dt)
    gr.num.Nz_substrate = int(args.final_nz_g)
    gr.num.Nz_slab = int(args.final_nz_s)

    theta_initial_final = theta_initial
    if best_params is not None:
        theta_initial_final = scale_theta_initial(
            theta_initial, best_params.get("theta_initial_factor", 1.0), gr.substrat
        )

    results = gr.run_cam_simulation(
        weather_df=weather,
        plant=gr.bromelia,
        substrate=gr.substrat,
        slab_params=gr.slab,
        geom_params=gr.geom,
        num_params=gr.num,
        rs_profile=rs_profile,
        theta_initial=theta_initial_final,
        T_in_series=ni[args.tin_col],
        T_s_in_initial_C=T_s_in_initial_C,
        T_g_top_initial_C=T_g_top_initial_C,
        verbose=not args.quiet,
    )

    sim_df = gr.results_to_dataframe(results)
    sim_eval = sim_df[(sim_df.index >= eval_start) & (sim_df.index <= eval_end)]
    metrics = gr.validation_metrics(sim_eval["T_s_in"], ni[args.target_col])

    # Save outputs.
    sim_df.to_csv(output_dir / "cam_gsw_prediction_full_with_spinup.csv", index_label="datetime")
    sim_eval.to_csv(output_dir / "cam_gsw_prediction_eval_window.csv", index_label="datetime")

    with open(output_dir / "cam_gsw_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "cam_gsw_rs_profile.json", "w", encoding="utf-8") as f:
        json.dump({
            "r_stoma_min_CAM_s_m": rs_profile.r_stoma_min_s_m,
            "profile_r_s_night_s_m": rs_profile.r_s_night_s_m,
            "profile_r_s_midday_s_m": rs_profile.r_s_midday_s_m,
            "profile_r_s_late_afternoon_s_m": rs_profile.r_s_late_afternoon_s_m,
            "hourly_gsw_median": rs_profile.hourly_gsw_median,
            "hourly_gsw_n": rs_profile.hourly_gsw_n,
            "source_summary": rs_profile.source_summary,
        }, f, indent=2)

    plot_cam_validation(
        sim_df=sim_df,
        ni_df=ni,
        soil_df=soil,
        target_col=args.target_col,
        eval_start=eval_start,
        eval_end=eval_end,
        metrics=metrics,
        save_path=output_dir / "cam_gsw_validation.png",
    )
    plot_cam_soil_and_et(
        sim_df=sim_df,
        soil_df=soil,
        eval_start=eval_start,
        eval_end=eval_end,
        save_path=output_dir / "cam_gsw_soil_et_split.png",
    )

    print("\n=== METRICS ===")
    print(metrics)
    print("\nSaved outputs:")
    print(f"  {output_dir / 'cam_gsw_validation.png'}")
    print(f"  {output_dir / 'cam_gsw_soil_et_split.png'}")
    print(f"  {output_dir / 'cam_gsw_prediction_eval_window.csv'}")
    print(f"  {output_dir / 'cam_gsw_metrics.json'}")
    print(f"  {output_dir / 'cam_gsw_rs_profile.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
