"""
Clean runner for green-roof validation.

Use this file as the main script. It keeps the professor-preferred physics model,
but avoids running extra/debug segments unless explicitly enabled.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # force PNG saving even when no GUI/backend is available
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import new_baru_revised_same_structure_v3_channelmap as gr


# =============================================================================
# USER CONFIG
# =============================================================================
# Always resolve paths relative to this runner file, not terminal working directory.
# Put this runner in the same folder as weatherfile, NI xlsx, and datasoilmoisture.zip.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CAM = True                    # run CAM / Bromelia as parallel validation case
RUN_C3_MAIN = True
RUN_C3_EARLY = False              # keep False unless you intentionally discuss early/anomaly segment

# Known scheduled watering: 07:00-07:04 and 16:30-16:34.
# If pump flow is not measured, leave 0.0: the code records irrigation_flag for ANN
# but does not invent water input in the physics model.
IRRIGATION_WINDOWS = gr.DEFAULT_IRRIGATION_WINDOWS
IRRIGATION_MM_PER_MIN = 0.0       # update only if you have measured/applicable flow rate


# =============================================================================
# SCIENTIFIC-GUESS PARAMETERS TO LOCK BEFORE RUNNING
# =============================================================================
gr.apply_scientific_guess_parameters(
    rho_g=400.0,
    H_slab=0.10,
    H_g=0.10,              # your later correction: substrate is 10 cm, not 6 cm
    theta_sat=0.90,
    k_theta_sat=5e-6,
    lambda_dry=0.12,
    h_in=8.0,
)

gr.geom.A_roof = 1.0

# Cover fraction from visual/ImageJ discussion. Adjust only with documented ImageJ value.
gr.bromelia.cover_fraction = 0.70
gr.wedelia.cover_fraction = 0.40

gr.geom.dynamic_h_in = False

# Validation targets. Keep explicit to avoid accidental channel switching.
CAM_TARGET = "T2A2"   # mapping terbaru: Atap Indoor CAM
C3_TARGET = "T3Ka"     # atap indoor C3

gr.VALIDATION_TARGETS["CAM"] = CAM_TARGET
gr.VALIDATION_TARGETS["C3"] = C3_TARGET


def plot_validation_single(results, ni, plant_type, save_path=None):
    """Plot measured vs physics-model T_s,in and print validation metrics."""
    plant_type = plant_type.upper()

    if plant_type == "CAM":
        target_col = gr.VALIDATION_TARGETS.get("CAM", "T_s_in_CAM")
        title = "CAM / Bromelia"
    elif plant_type == "C3":
        target_col = gr.VALIDATION_TARGETS.get("C3", "T_s_in_C3")
        title = "C3 / Wedelia"
    else:
        raise ValueError("plant_type must be 'CAM' or 'C3'")

    # Allow either raw channel names or descriptive aliases.
    if target_col not in ni.columns:
        alias = "T_s_in_CAM" if plant_type == "CAM" else "T_s_in_C3"
        if alias in ni.columns:
            target_col = alias
        else:
            raise ValueError(f"Target column not found: {target_col}")

    sim = pd.Series(
        results["T_s_in"],
        index=pd.to_datetime(results["datetime"]),
        name="Model"
    ).sort_index().resample("1min").mean()

    obs = ni[target_col].sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)
    if len(common) == 0:
        raise ValueError("No common timestamp between simulation and NI observation.")

    sim_common = sim.loc[common]
    obs_common = obs.loc[common]
    err = (sim_common - obs_common).dropna()

    bias = err.mean()
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    amp_measured = obs_common.max() - obs_common.min()
    amp_model = sim_common.max() - sim_common.min()
    amp_error = amp_model - amp_measured
    peak_error = sim_common.max() - obs_common.max()
    min_error = sim_common.min() - obs_common.min()

    print(f"\nAmplitude check {plant_type}:")
    print(f"  Target channel      : {target_col}")
    print(f"  Measured amplitude : {amp_measured:.2f} °C")
    print(f"  Model amplitude    : {amp_model:.2f} °C")
    print(f"  Amplitude error    : {amp_error:.2f} °C")
    print(f"  Peak error         : {peak_error:.2f} °C")
    print(f"  Min error          : {min_error:.2f} °C")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(obs_common.index, obs_common.values, label=f"Measured NI ({target_col})", linewidth=2)
    axes[0].plot(sim_common.index, sim_common.values, label="Physics model", linestyle="--", linewidth=2)
    axes[0].set_ylabel("T_s,in (°C)")
    axes[0].set_title(
        f"Validation {title}: Inner Roof Surface Temperature\n"
        f"Bias={bias:.2f}°C | MAE={mae:.2f}°C | RMSE={rmse:.2f}°C | "
        f"AmpErr={amp_error:.2f}°C | n={len(err)}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(err.index, err.values, linewidth=1.5)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error (°C)")
    axes[1].set_title("Residual: Physics model - Measured")
    axes[1].grid(True, alpha=0.3)

    t = pd.to_datetime(results["datetime"])
    axes[2].plot(t, results["T_a"], label="T_a", linewidth=1.5)
    axes[2].set_ylabel("T_a (°C)")
    axes[2].set_xlabel("Datetime")
    axes[2].grid(True, alpha=0.3)

    ax2 = axes[2].twinx()
    ax2.plot(t, results["G_sol"], label="G_sol", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("G_sol (W/m²)")

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
        exists = save_path.exists()
        size = save_path.stat().st_size if exists else 0
        print(f"Saved plot: {save_path.resolve()} | exists={exists} | size={size} bytes")
        if not exists or size == 0:
            raise RuntimeError(f"Plot was not saved correctly: {save_path}")
    plt.close(fig)

    return {
        "target_col": target_col,
        "bias_C": float(bias),
        "mae_C": float(mae),
        "rmse_C": float(rmse),
        "amp_measured_C": float(amp_measured),
        "amp_model_C": float(amp_model),
        "amp_error_C": float(amp_error),
        "peak_error_C": float(peak_error),
        "min_error_C": float(min_error),
        "n": int(len(err)),
    }


def run_one_case(plant_type: str, tag: str):
    """Run one validation case, save plot and ANN residual dataset."""
    plant_type = plant_type.upper()
    print(f"\n=== RUNNING {tag} ({plant_type}) ===")

    results, metrics = gr.run_validation_case(
        plant_type=plant_type,
        base_dir=str(BASE_DIR),
        calibrate_lai=False,
        irrigation_windows=IRRIGATION_WINDOWS,
        irrigation_mm_per_min=IRRIGATION_MM_PER_MIN,
    )

    print(f"\n{tag} metrics from model module:")
    print(metrics)

    _, ni, _, _ = gr.prepare_validation_case(plant_type, base_dir=str(BASE_DIR))

    plot_metrics = plot_validation_single(
        results,
        ni,
        plant_type=plant_type,
        save_path=OUTPUT_DIR / f"validation_{tag}_model_vs_measured.png",
    )

    target_col = gr.VALIDATION_TARGETS.get(plant_type, "T_s_in_C3" if plant_type == "C3" else "T_s_in_CAM")
    ann_df = gr.export_ann_residual_dataset(
        results,
        ni,
        target_col=target_col,
        out_path=OUTPUT_DIR / f"ann_residual_dataset_{tag}.csv",
        plant_type=plant_type,
    )

    return results, ni, metrics, plot_metrics, ann_df


def combine_ann_datasets(outputs, out_path):
    """Combine per-case ANN residual datasets into one file for later ANN training."""
    frames = []
    for key, value in outputs.items():
        if len(value) >= 5 and value[4] is not None:
            df = value[4].copy()
            df["case"] = key
            frames.append(df)

    if not frames:
        print("No ANN residual datasets to combine.")
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Simple model-ready indicators; keep plant_type string too for one-hot encoding later.
    if "plant_type" in combined.columns:
        combined["is_CAM"] = (combined["plant_type"].astype(str).str.upper() == "CAM").astype(int)
        combined["is_C3"] = (combined["plant_type"].astype(str).str.upper() == "C3").astype(int)

    combined.to_csv(out_path, index=False)
    print(f"Combined ANN residual dataset saved: {out_path} | rows={len(combined)}")
    return combined


def main():
    print(f"\nBASE_DIR   = {BASE_DIR.resolve()}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")

    # Optional weather summary, only if the required file exists.
    weather_path = BASE_DIR / "weatherfile mar-april.xlsx"
    if weather_path.exists():
        print("\n=== WEATHER SUMMARY ===")
        summary = gr.summarize_weather_windows(base_dir=str(BASE_DIR))
        print(summary)
        summary.to_excel(OUTPUT_DIR / "weather_summary.xlsx", index=False)
    else:
        print("\nWeather file not found; skipping weather summary.")

    outputs = {}

    if RUN_CAM:
        # CAM window remains defined in the model module:
        # 2026-03-31 11:58 to 2026-04-02 21:42
        # Target column is controlled by CAM_TARGET above.
        outputs["CAM_main"] = run_one_case("CAM", "CAM_main")

    if RUN_C3_MAIN:
        gr.VALIDATION_WINDOWS["C3"] = (
            pd.Timestamp("2026-04-09 11:05:00"),
            pd.Timestamp("2026-04-10 14:08:00"),
        )
        outputs["C3_main"] = run_one_case("C3", "C3_main")

    if RUN_C3_EARLY:
        gr.VALIDATION_WINDOWS["C3"] = (
            pd.Timestamp("2026-04-06 17:38:00"),  # after known early moisture glitch
            pd.Timestamp("2026-04-07 10:06:00"),
        )
        outputs["C3_early_after_anomaly"] = run_one_case("C3", "C3_early_after_anomaly")

    # Combine CAM + C3 residual rows for later ANN residual correction / optimization.
    combine_ann_datasets(outputs, OUTPUT_DIR / "ann_residual_dataset_ALL.csv")

    print("\nDONE. Outputs saved in:", OUTPUT_DIR.resolve())
    return outputs


if __name__ == "__main__":
    main()
