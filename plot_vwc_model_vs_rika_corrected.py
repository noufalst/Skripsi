"""
================================================================================
PLOT VWC MODEL vs RIKA SENSOR — WITH TEMPORARY RK520 CORRECTION FACTOR
================================================================================

Purpose
-------
Compare model VWC with RIKA soil moisture data after applying a correction factor:

    theta_corrected = theta_raw * correction_factor

Default correction factor:
    0.70  -> reduce RK520 reading by 30%

This is NOT calibration. It is only a sensitivity check for RK520-01 readings in
soil + rice husk substrate.

Designed for the old-good code pair:
    new_baru_revised_same_structure_v2.py
    run_green_roof.py

Usage
-----
CAM:
    python plot_vwc_model_vs_rika_corrected.py --plant CAM --factor 0.70

C3:
    python plot_vwc_model_vs_rika_corrected.py --plant C3 --factor 0.70

Only raw/corrected RIKA, no model rerun:
    python plot_vwc_model_vs_rika_corrected.py --plant CAM --factor 0.70 --no-model

Outputs
-------
    vwc_comparison_CAM_factor_0p70.png
    vwc_comparison_CAM_factor_0p70.csv
    vwc_comparison_CAM_factor_0p70_summary.txt
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import new_baru_revised_same_structure_v2 as gr


# ==============================================================================
# SECTION 1 — PROJECT DEFAULTS
# ==============================================================================

def apply_project_defaults() -> None:
    """Apply the same core defaults used in your old-good run_green_roof.py."""
    gr.apply_scientific_guess_parameters(
        rho_g=400.0,
        H_slab=0.10,
        H_g=0.10,
        theta_sat=0.90,
        k_theta_sat=5e-6,
        lambda_dry=0.12,
    )

    gr.geom.A_roof = 1.0
    gr.bromelia.cover_fraction = 0.7
    gr.wedelia.cover_fraction = 0.4
    gr.geom.dynamic_h_in = False


def apply_vwc_correction_to_rika(
    rika: pd.DataFrame,
    factor: float,
    theta_sat: float,
    theta_min: float,
) -> pd.DataFrame:
    """
    Apply temporary multiplicative correction to RIKA VWC.

    factor = 0.70 means 80% raw -> 56% corrected.
    This is not calibration.
    """
    out = rika.copy()

    for theta_col, pct_col in [
        ("theta_2cm", "moisture_2cm_pct"),
        ("theta_7cm", "moisture_7cm_pct"),
    ]:
        if theta_col in out.columns:
            out[theta_col + "_raw"] = out[theta_col]
            out[theta_col] = pd.to_numeric(out[theta_col], errors="coerce") * factor
            out[theta_col] = out[theta_col].clip(lower=theta_min, upper=theta_sat)

        if pct_col in out.columns:
            out[pct_col + "_raw"] = out[pct_col]
            out[pct_col] = pd.to_numeric(out[pct_col], errors="coerce") * factor
            out[pct_col] = out[pct_col].clip(lower=theta_min * 100, upper=theta_sat * 100)

    return out


# ==============================================================================
# SECTION 2 — RUN MODEL WITH CORRECTED INITIAL VWC
# ==============================================================================

def run_model_with_corrected_vwc(
    plant_type: str,
    base_dir: str,
    correction_factor: float,
    cam_target: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load weather/NI/RIKA, correct RIKA VWC, then run model using corrected theta_initial."""
    plant_type = plant_type.upper()
    if plant_type not in ["CAM", "C3"]:
        raise ValueError("plant must be CAM or C3")

    apply_project_defaults()

    if cam_target:
        gr.VALIDATION_TARGETS["CAM"] = cam_target

    plant = gr.bromelia if plant_type == "CAM" else gr.wedelia

    weather, ni, rika_raw, _theta_unused = gr.prepare_validation_case(plant_type, base_dir=base_dir)
    start, end = gr.VALIDATION_WINDOWS[plant_type]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    rika_corr = apply_vwc_correction_to_rika(
        rika_raw,
        factor=correction_factor,
        theta_sat=gr.substrat.theta_sat,
        theta_min=gr.substrat.theta_min,
    )

    theta_initial = gr.get_theta_initial_from_rika(rika_corr, start)

    if plant_type == "CAM":
        T_in_series = ni["T_in_CAM"]
        target_col = gr.VALIDATION_TARGETS.get("CAM", "T1Tb")
        T_g_top_initial = (
            float(ni["T_g_top_CAM"].dropna().iloc[0])
            if "T_g_top_CAM" in ni and not ni["T_g_top_CAM"].dropna().empty
            else None
        )
    else:
        T_in_series = ni["T_in_C3"]
        target_col = gr.VALIDATION_TARGETS.get("C3", "T3Ka")
        T_g_top_initial = None

    if target_col not in ni.columns:
        raise ValueError(f"Target column {target_col} not found in NI data.")

    T_s_in_initial = float(ni[target_col].dropna().iloc[0])

    results = gr.run_simulation(
        weather_df=weather,
        plant=plant,
        substrate=gr.substrat,
        slab=gr.slab,
        geom=gr.geom,
        num=gr.num,
        theta_initial=theta_initial,
        T_in_series=T_in_series,
        T_g_top_initial_C=T_g_top_initial,
        T_s_in_initial_C=T_s_in_initial,
    )

    meta = {
        "plant_type": plant_type,
        "start": str(start),
        "end": str(end),
        "correction_factor": correction_factor,
        "theta_initial": theta_initial,
        "target_col": target_col,
        "T_in_col": "T_in_CAM" if plant_type == "CAM" else "T_in_C3",
        "LAI": float(plant.LAI),
        "cover_fraction": float(plant.cover_fraction),
        "theta_sat": float(gr.substrat.theta_sat),
        "theta_min": float(gr.substrat.theta_min),
    }

    return results, weather, ni, rika_raw, rika_corr, meta


# ==============================================================================
# SECTION 3 — PLOTTING AND EXPORT
# ==============================================================================

def results_theta_dataframe(results: dict) -> pd.DataFrame:
    """Convert model result dictionary to timestamped theta dataframe."""
    t = pd.to_datetime(results["datetime"])
    df = pd.DataFrame(
        {
            "model_theta_top": results.get("theta_top", np.nan),
            "model_theta_mid": results.get("theta_mid", np.nan),
            "model_theta_bot": results.get("theta_bot", np.nan),
            "model_theta_mean": results.get("theta_mean", np.nan),
        },
        index=t,
    )
    df.index.name = "datetime"
    return df.sort_index()


def make_vwc_comparison_dataframe(
    model_df: pd.DataFrame | None,
    rika_raw: pd.DataFrame,
    rika_corr: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model theta and corrected/raw RIKA theta on 1-minute grid."""
    frames = []

    if model_df is not None:
        frames.append(model_df.resample("1min").mean())

    rika_keep = pd.DataFrame(index=rika_corr.index)

    for col in ["theta_2cm", "theta_7cm"]:
        if col in rika_corr.columns:
            rika_keep[f"rika_corrected_{col}"] = rika_corr[col]
        if col in rika_raw.columns:
            rika_keep[f"rika_raw_{col}"] = rika_raw[col]

    frames.append(rika_keep.resample("1min").mean().interpolate("time", limit=15))
    return pd.concat(frames, axis=1).sort_index()


def summarize_vwc(df: pd.DataFrame, meta: dict) -> str:
    lines = []
    lines.append("VWC MODEL vs RIKA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Plant type        : {meta['plant_type']}")
    lines.append(f"Window            : {meta['start']} -> {meta['end']}")
    lines.append(f"Correction factor : {meta['correction_factor']}")
    lines.append(f"theta_sat         : {meta['theta_sat']}")
    lines.append(f"theta_min         : {meta['theta_min']}")
    lines.append(f"theta_initial     : {meta['theta_initial']}")
    lines.append(f"Target col        : {meta['target_col']}")
    lines.append(f"T_in col          : {meta['T_in_col']}")
    lines.append("")

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        lines.append(
            f"{col:25s}: min={s.min():.3f}, mean={s.mean():.3f}, max={s.max():.3f}, n={len(s)}"
        )

    return "\n".join(lines)


def plot_vwc(
    df: pd.DataFrame,
    plant_type: str,
    factor: float,
    out_png: Path,
    show_raw: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: corrected RIKA vs model top/bottom
    if "model_theta_top" in df:
        axes[0].plot(df.index, df["model_theta_top"], label="Model θ_top", linewidth=2)
    if "model_theta_bot" in df:
        axes[0].plot(df.index, df["model_theta_bot"], label="Model θ_bot", linewidth=2)
    if "rika_corrected_theta_2cm" in df:
        axes[0].plot(df.index, df["rika_corrected_theta_2cm"], "--", label=f"RIKA θ_2cm corrected ×{factor}", linewidth=1.6)
    if "rika_corrected_theta_7cm" in df:
        axes[0].plot(df.index, df["rika_corrected_theta_7cm"], "--", label=f"RIKA θ_7cm corrected ×{factor}", linewidth=1.6)

    axes[0].set_title(f"{plant_type} VWC comparison: model vs corrected RIKA")
    axes[0].set_ylabel("VWC θ (m³/m³)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    # Panel 2: raw vs corrected RIKA only
    if show_raw:
        if "rika_raw_theta_2cm" in df:
            axes[1].plot(df.index, df["rika_raw_theta_2cm"], label="RIKA raw θ_2cm", alpha=0.45)
        if "rika_raw_theta_7cm" in df:
            axes[1].plot(df.index, df["rika_raw_theta_7cm"], label="RIKA raw θ_7cm", alpha=0.45)
    if "rika_corrected_theta_2cm" in df:
        axes[1].plot(df.index, df["rika_corrected_theta_2cm"], "--", label="RIKA corrected θ_2cm", linewidth=1.8)
    if "rika_corrected_theta_7cm" in df:
        axes[1].plot(df.index, df["rika_corrected_theta_7cm"], "--", label="RIKA corrected θ_7cm", linewidth=1.8)

    axes[1].set_title("RIKA raw vs corrected")
    axes[1].set_ylabel("VWC θ (m³/m³)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    # Panel 3: residual model - corrected sensor
    if "model_theta_top" in df and "rika_corrected_theta_2cm" in df:
        axes[2].plot(df.index, df["model_theta_top"] - df["rika_corrected_theta_2cm"],
                     label="Model θ_top - corrected RIKA θ_2cm")
    if "model_theta_bot" in df and "rika_corrected_theta_7cm" in df:
        axes[2].plot(df.index, df["model_theta_bot"] - df["rika_corrected_theta_7cm"],
                     label="Model θ_bot - corrected RIKA θ_7cm")
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_title("VWC residual")
    axes[2].set_ylabel("Δθ (m³/m³)")
    axes[2].set_xlabel("Datetime")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# SECTION 4 — MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot model VWC vs corrected RIKA VWC.")
    parser.add_argument("--plant", choices=["CAM", "C3"], default="CAM")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--factor", type=float, default=0.70,
                        help="Correction factor. 0.70 means reduce raw RK520 by 30%.")
    parser.add_argument("--cam-target", default=None,
                        help="Optional CAM target override, e.g. T1Tb or T2A2.")
    parser.add_argument("--no-model", action="store_true",
                        help="Only plot raw/corrected RIKA; do not rerun thermal model.")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    plant_type = args.plant.upper()
    factor_text = f"{args.factor:.2f}".replace(".", "p")
    out_prefix = args.out_prefix or f"vwc_comparison_{plant_type}_factor_{factor_text}"

    apply_project_defaults()

    if args.no_model:
        weather, ni, rika_raw, _ = gr.prepare_validation_case(plant_type, base_dir=args.base_dir)
        start, end = gr.VALIDATION_WINDOWS[plant_type]
        rika_corr = apply_vwc_correction_to_rika(
            rika_raw,
            factor=args.factor,
            theta_sat=gr.substrat.theta_sat,
            theta_min=gr.substrat.theta_min,
        )
        model_df = None
        meta = {
            "plant_type": plant_type,
            "start": str(start),
            "end": str(end),
            "correction_factor": args.factor,
            "theta_initial": "not used (--no-model)",
            "target_col": "not used",
            "T_in_col": "not used",
            "LAI": np.nan,
            "cover_fraction": np.nan,
            "theta_sat": float(gr.substrat.theta_sat),
            "theta_min": float(gr.substrat.theta_min),
        }
    else:
        results, weather, ni, rika_raw, rika_corr, meta = run_model_with_corrected_vwc(
            plant_type=plant_type,
            base_dir=args.base_dir,
            correction_factor=args.factor,
            cam_target=args.cam_target,
        )
        model_df = results_theta_dataframe(results)

    comp = make_vwc_comparison_dataframe(model_df, rika_raw, rika_corr)

    out_png = Path(f"{out_prefix}.png")
    out_csv = Path(f"{out_prefix}.csv")
    out_txt = Path(f"{out_prefix}_summary.txt")

    plot_vwc(comp, plant_type, args.factor, out_png)
    comp.to_csv(out_csv, index_label="datetime")
    out_txt.write_text(summarize_vwc(comp, meta), encoding="utf-8")

    print("\nDone.")
    print(f"Saved plot   : {out_png}")
    print(f"Saved data   : {out_csv}")
    print(f"Saved summary: {out_txt}")
    print("\nReminder:")
    print("  Corrected RIKA = raw RIKA × factor")
    print("  This is only sensitivity testing, not calibrated VWC.")


if __name__ == "__main__":
    main()
