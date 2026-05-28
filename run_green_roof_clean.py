"""
run_green_roof_clean.py

Runner for green_roof_model_clean.py.

Default behaviour:
- Runs the model.
- Saves prediction CSV, metrics JSON, and comparison plot when target exists.
- Does NOT create extra diagnostics unless `--diagnostics` is passed.

Example C3:
    python run_green_roof_clean.py --input data_clean.csv --plant c3 --target-col C3 \
        --output-dir outputs/c3

Example CAM with damped amplitude:
    python run_green_roof_clean.py --input data_clean.csv --plant cam --target-col CAM \
        --cam-amplitude-scale 0.45 --output-dir outputs/cam

If your NI file already has dynamic indoor temperature, pass its column:
    python run_green_roof_clean.py --input data.csv --plant c3 --tin-col T_in_boundary
"""

from __future__ import annotations

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from green_roof_model_clean import (
    build_default_config,
    compute_metrics,
    green_roof_model_clean,
    with_updated_plant,
)


# ============================================================
# SECTION 1 — COLUMN DETECTION / DATA CLEANING
# ============================================================

COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "time": (
        "datetime", "date_time", "timestamp", "time", "tanggal", "waktu", "DateTime", "Time",
    ),
    "T_air_C": (
        "T_air_C", "T_air", "Tamb", "T_amb", "ambient", "ambient_C", "outdoor", "outdoor_C",
        "temperature_2m", "Temp Out", "Outdoor Temperature", "T_out", "Tout",
    ),
    "T_in_C": (
        "T_in_C", "T_in", "Tin", "indoor", "indoor_C", "room", "room_C", "NI_T_in",
        "T_in_boundary", "Indoor Temperature",
    ),
    "solar_W_m2": (
        "solar_W_m2", "shortwave_radiation", "shortwave", "GHI", "radiation", "Solar Radiation",
        "Rs", "R_s", "I_solar", "SW_in",
    ),
    "RH_pct": (
        "RH_pct", "RH", "relative_humidity", "relative_humidity_2m", "humidity", "Hum", "H_out",
    ),
    "soil_moisture": (
        "soil_moisture", "SM", "VWC", "soil_water", "water_content", "theta", "Theta",
    ),
}


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(df: pd.DataFrame, aliases: Iterable[str], explicit: Optional[str] = None) -> Optional[str]:
    """Find a dataframe column using an explicit name or aliases."""
    if explicit:
        if explicit in df.columns:
            return explicit
        normalized_explicit = _normalize_name(explicit)
        for col in df.columns:
            if _normalize_name(col) == normalized_explicit:
                return col
        raise ValueError(f"Column '{explicit}' was requested but not found in input file.")

    normalized_map = {_normalize_name(col): col for col in df.columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized_map:
            return normalized_map[key]
    return None


def find_target_column(df: pd.DataFrame, plant: str, explicit: Optional[str] = None) -> Optional[str]:
    """Choose target column. Explicit target always wins."""
    if explicit:
        return find_column(df, [explicit], explicit=explicit)

    plant_key = plant.lower().strip()
    normalized_cols = {_normalize_name(col): col for col in df.columns}

    preferred_tokens = {
        "c3": ("c3", "c3indoor", "c3target", "tc3", "c3temp"),
        "cam": ("cam", "camindoor", "camtarget", "tcam", "camtemp"),
    }.get(plant_key, ())

    # Avoid accidentally using forcing columns as target.
    forbidden = {
        _normalize_name(x)
        for x in [
            "T_air_C", "T_air", "Tamb", "T_amb", "temperature_2m", "T_in_C", "T_in", "Tin",
            "solar_W_m2", "shortwave_radiation", "RH_pct", "soil_moisture",
        ]
    }

    for key, original in normalized_cols.items():
        if key in forbidden:
            continue
        if any(token in key for token in preferred_tokens):
            return original

    return None


def load_and_standardize_data(
    input_path: Path,
    plant: str,
    time_col: Optional[str] = None,
    tair_col: Optional[str] = None,
    tin_col: Optional[str] = None,
    solar_col: Optional[str] = None,
    rh_col: Optional[str] = None,
    soil_moisture_col: Optional[str] = None,
    target_col: Optional[str] = None,
    tin_constant: float = 27.0,
) -> tuple[pd.DataFrame, Optional[str]]:
    """Read CSV/XLSX and convert important columns to model names."""
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(input_path)
    else:
        raw = pd.read_csv(input_path)

    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    detected_time = find_column(raw, COLUMN_ALIASES["time"], explicit=time_col)
    if detected_time:
        raw[detected_time] = pd.to_datetime(raw[detected_time], errors="coerce")
        raw = raw.dropna(subset=[detected_time]).sort_values(detected_time)
        raw = raw.set_index(detected_time)

    detected_tair = find_column(raw, COLUMN_ALIASES["T_air_C"], explicit=tair_col)
    if detected_tair is None:
        raise ValueError(
            "Could not detect outdoor/ambient temperature column. "
            "Use --tair-col to specify it."
        )

    detected_tin = find_column(raw, COLUMN_ALIASES["T_in_C"], explicit=tin_col) if tin_col else find_column(raw, COLUMN_ALIASES["T_in_C"])
    detected_solar = find_column(raw, COLUMN_ALIASES["solar_W_m2"], explicit=solar_col) if solar_col else find_column(raw, COLUMN_ALIASES["solar_W_m2"])
    detected_rh = find_column(raw, COLUMN_ALIASES["RH_pct"], explicit=rh_col) if rh_col else find_column(raw, COLUMN_ALIASES["RH_pct"])
    detected_sm = (
        find_column(raw, COLUMN_ALIASES["soil_moisture"], explicit=soil_moisture_col)
        if soil_moisture_col
        else find_column(raw, COLUMN_ALIASES["soil_moisture"])
    )
    detected_target = find_target_column(raw, plant=plant, explicit=target_col)

    model_df = pd.DataFrame(index=raw.index)
    model_df["T_air_C"] = pd.to_numeric(raw[detected_tair], errors="coerce")

    if detected_tin:
        model_df["T_in_C"] = pd.to_numeric(raw[detected_tin], errors="coerce")
    else:
        model_df["T_in_C"] = float(tin_constant)

    if detected_solar:
        model_df["solar_W_m2"] = pd.to_numeric(raw[detected_solar], errors="coerce").clip(lower=0)
    else:
        model_df["solar_W_m2"] = 0.0

    if detected_rh:
        model_df["RH_pct"] = pd.to_numeric(raw[detected_rh], errors="coerce")

    if detected_sm:
        model_df["soil_moisture"] = pd.to_numeric(raw[detected_sm], errors="coerce")

    target_name_for_model = None
    if detected_target:
        target_name_for_model = "T_target_C"
        model_df[target_name_for_model] = pd.to_numeric(raw[detected_target], errors="coerce")

    # Interpolate numeric forcing values. This avoids spikes from small missing chunks.
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.interpolate(method="time" if isinstance(model_df.index, pd.DatetimeIndex) else "linear")
    model_df = model_df.ffill().bfill()

    model_df.attrs["detected_columns"] = {
        "time": detected_time,
        "T_air_C": detected_tair,
        "T_in_C": detected_tin or f"constant {tin_constant}",
        "solar_W_m2": detected_solar or "not found -> 0",
        "RH_pct": detected_rh or "not found",
        "soil_moisture": detected_sm or "not found",
        "target": detected_target or "not found",
    }
    return model_df, target_name_for_model


# ============================================================
# SECTION 2 — OUTPUT HELPERS
# ============================================================

def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            clean_metrics[key] = None
        else:
            clean_metrics[key] = value
    output_path.write_text(json.dumps(clean_metrics, indent=2), encoding="utf-8")


def make_main_plot(result: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if "T_target_C" in result.columns:
        ax.plot(result.index, result["T_target_C"], label="target")
    ax.plot(result.index, result["T_pred_C"], label="model")
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("temperature [°C]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_diagnostic_plots(result: pd.DataFrame, forcing: pd.DataFrame, output_dir: Path, plant: str) -> None:
    """Save optional diagnostic plots only when requested."""
    diag = pd.concat(
        [
            forcing[[c for c in ["T_air_C", "T_in_C", "solar_W_m2"] if c in forcing.columns]],
            result[["T_surface_C", "T_pred_C", "q_solar_W_m2", "q_plant_W_m2", "q_top_conv_W_m2", "q_bottom_W_m2"]],
        ],
        axis=1,
    )
    diag.to_csv(output_dir / f"diagnostics_{plant}.csv", index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ["T_air_C", "T_in_C", "T_surface_C", "T_pred_C"]:
        if col in diag.columns:
            ax.plot(diag.index, diag[col], label=col)
    ax.set_title(f"Temperature diagnostics — {plant.upper()}")
    ax.set_xlabel("time")
    ax.set_ylabel("temperature [°C]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"diagnostics_temperature_{plant}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ["q_solar_W_m2", "q_plant_W_m2", "q_top_conv_W_m2", "q_bottom_W_m2"]:
        if col in diag.columns:
            ax.plot(diag.index, diag[col], label=col)
    ax.set_title(f"Flux diagnostics — {plant.upper()}")
    ax.set_xlabel("time")
    ax.set_ylabel("heat flux [W/m²]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"diagnostics_flux_{plant}.png", dpi=180)
    plt.close(fig)


# ============================================================
# SECTION 3 — RUNNER
# ============================================================

def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    forcing, target_col_for_model = load_and_standardize_data(
        input_path=input_path,
        plant=args.plant,
        time_col=args.time_col,
        tair_col=args.tair_col,
        tin_col=args.tin_col,
        solar_col=args.solar_col,
        rh_col=args.rh_col,
        soil_moisture_col=args.soil_moisture_col,
        target_col=args.target_col,
        tin_constant=args.tin_constant,
    )

    config = build_default_config(
        plant=args.plant,
        soil_thickness_m=args.soil_thickness,
        concrete_thickness_m=args.concrete_thickness,
        lai=args.lai,
        cover_fraction=args.cover_fraction,
        cam_amplitude_scale=args.cam_amplitude_scale,
        c3_amplitude_scale=args.c3_amplitude_scale,
        h_in_W_m2K=args.h_in,
        h_out_W_m2K=args.h_out,
        solar_absorptivity=args.solar_absorptivity,
        target_col=target_col_for_model,
    )

    # Optional direct override useful for controlled sweeps.
    if args.plant_max_flux is not None:
        config = with_updated_plant(config, q_plant_cap_W_m2=args.plant_max_flux)

    result = green_roof_model_clean(forcing, config=config, diagnostics=args.diagnostics)

    plant = args.plant.lower().strip()
    prediction_path = output_dir / f"prediction_{plant}.csv"
    result.to_csv(prediction_path, index=True)

    metrics: Dict[str, float] = {
        "plant": plant,
        "input": str(input_path),
        "n_rows": int(len(result)),
        "soil_thickness_m": float(args.soil_thickness),
        "concrete_thickness_m": float(args.concrete_thickness),
        "h_in_W_m2K": float(args.h_in),
        "h_out_W_m2K": float(args.h_out),
        "cover_fraction": float(args.cover_fraction),
        "lai": float(config.plant_params.lai),
        "plant_amplitude_scale": float(config.plant_params.amplitude_scale),
        "detected_columns": forcing.attrs.get("detected_columns", {}),
    }
    if "T_target_C" in result.columns:
        metrics.update(compute_metrics(result["T_target_C"], result["T_pred_C"]))

    metrics_path = output_dir / f"metrics_{plant}.json"
    save_metrics(metrics, metrics_path)

    plot_path = output_dir / f"comparison_{plant}.png"
    make_main_plot(result, plot_path, title=f"Green roof model — {plant.upper()}")

    if args.diagnostics:
        make_diagnostic_plots(result, forcing, output_dir, plant)

    print("Done.")
    print(f"Prediction CSV : {prediction_path}")
    print(f"Metrics JSON   : {metrics_path}")
    print(f"Main plot      : {plot_path}")
    if args.diagnostics:
        print(f"Diagnostics    : {output_dir}")
    print("Detected columns:")
    for key, value in metrics["detected_columns"].items():
        print(f"  {key}: {value}")
    if "rmse_C" in metrics:
        print(f"RMSE: {metrics['rmse_C']:.4f} °C | Bias: {metrics['bias_C']:.4f} °C | MAE: {metrics['mae_C']:.4f} °C")


# ============================================================
# SECTION 4 — CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean green-roof model for C3/CAM data.")

    parser.add_argument("--input", required=True, help="Input CSV/XLSX path.")
    parser.add_argument("--output-dir", default="outputs_green_roof", help="Output directory.")
    parser.add_argument("--plant", choices=["c3", "cam", "none", "bare"], default="c3", help="Plant type.")

    # Explicit column mapping. Auto-detection is used when these are omitted.
    parser.add_argument("--time-col", default=None, help="Datetime column name.")
    parser.add_argument("--tair-col", default=None, help="Outdoor/ambient temperature column name.")
    parser.add_argument("--tin-col", default=None, help="Dynamic indoor boundary temperature column name.")
    parser.add_argument("--solar-col", default=None, help="Solar radiation column name.")
    parser.add_argument("--rh-col", default=None, help="Relative humidity column name.")
    parser.add_argument("--soil-moisture-col", default=None, help="Soil moisture column name.")
    parser.add_argument("--target-col", default=None, help="Measured target temperature column for validation.")

    # Physical/model parameters.
    parser.add_argument("--soil-thickness", type=float, default=0.10, help="Soil/substrate thickness [m].")
    parser.add_argument("--concrete-thickness", type=float, default=0.12, help="Concrete thickness [m].")
    parser.add_argument("--h-in", type=float, default=8.0, help="Indoor convection coefficient [W/m2K].")
    parser.add_argument("--h-out", type=float, default=12.0, help="Outdoor convection coefficient [W/m2K].")
    parser.add_argument("--solar-absorptivity", type=float, default=0.68, help="Absorbed fraction of shortwave radiation.")
    parser.add_argument("--tin-constant", type=float, default=27.0, help="Indoor boundary temp if no T_in column exists [degC].")

    # Plant parameters.
    parser.add_argument("--lai", type=float, default=None, help="Leaf area index. Default: C3=1.07, CAM=0.8.")
    parser.add_argument("--cover-fraction", type=float, default=0.31, help="Canopy/cover fraction 0..1.")
    parser.add_argument("--c3-amplitude-scale", type=float, default=1.0, help="C3 plant cooling amplitude scale.")
    parser.add_argument(
        "--cam-amplitude-scale",
        type=float,
        default=0.55,
        help="CAM plant cooling amplitude scale. Lower this if CAM oscillation is too large.",
    )
    parser.add_argument("--plant-max-flux", type=float, default=None, help="Optional cap for plant cooling flux [W/m2].")

    # Diagnostics are optional by design.
    parser.add_argument("--diagnostics", action="store_true", help="Save extra diagnostic CSV and plots.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
