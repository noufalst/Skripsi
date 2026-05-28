"""
Final CAM runner using the current sensor interpretation.

Current CAM sensor interpretation:
- T1Ka = confirmed inner-roof / indoor-side roof surface temperature, T_s_in.
- T1Ke = upper-zone indoor air proxy, T_air_upper. It is hanging in the cabin,
  not a room-average air temperature sensor.
- T2A2 = candidate outer-roof interface temperature, located conceptually above
  the slab and below the substrate. It is NOT treated as an exposed outdoor
  surface temperature.
- T2A = uncertain / not preferred as outer-roof candidate based on phase and
  weather response checks.

Purpose:
1) Run CAM simulation with T1Ka as the main validation target.
2) Use T1Ke as the dynamic indoor air boundary / T_in proxy.
3) Sweep indoor convection coefficient h_in because the cabin had no fan or
   ventilation, so natural/stagnant convection is more defensible than strong
   forced convection.
4) Use T2A2 as secondary observational diagnostic for outer-to-inner roof lag,
   not as a hard boundary condition.

Run:
    py run_green_roof_cam_final_mapping.py

Expected outputs:
    outputs_cam_final_mapping/
        cam_final_hin_sweep_metrics.csv
        cam_final_sensor_stats.csv
        cam_observed_outer_to_inner_lag_summary.csv
        cam_final_best_case_timeseries.csv
        cam_final_selected_case_summary.txt
        01_CAM_hin_sweep_T1Ka_target.png
        02_CAM_selected_case_detail.png
        03_CAM_T2A2_outer_to_T1Ka_inner_lag.png
        04_CAM_T1Ke_upper_air_diagnostic.png

Required in same folder:
    one compatible model module:
        new_baru_revised_same_structure_v3_t2a_subs_candidate_foliage_guard.py
        new_baru_revised_same_structure_v3_t2a_subs_candidate.py
        new_baru_revised_same_structure_v3_channelmap.py
        new_baru_revised_same_structure_v3_fixed.py
    weatherfile mar-april.xlsx
    Pengukuran 30_1 Maret 2026.xlsx
    Pengukuran 30_2 Maret 2026.xlsx
    Pengukuran 3 April 2026.xlsx
    datasoilmoisture.zip
"""

from __future__ import annotations

from pathlib import Path
import importlib
import json
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs_cam_final_mapping"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# IMPORT MODEL MODULE
# =============================================================================
MODULE_CANDIDATES = [
    "new_baru_revised_same_structure_v3_t2a_subs_candidate_foliage_guard",
    "new_baru_revised_same_structure_v3_t2a_subs_candidate",
    "new_baru_revised_same_structure_v3_channelmap",
    "new_baru_revised_same_structure_v3_fixed",
]

gr = None
MODEL_MODULE_NAME = None
for module_name in MODULE_CANDIDATES:
    try:
        gr = importlib.import_module(module_name)
        MODEL_MODULE_NAME = module_name
        print(f"Using model module: {module_name}")
        break
    except ModuleNotFoundError:
        continue

if gr is None:
    raise ModuleNotFoundError(
        "No compatible green-roof model module found in this folder. "
        f"Tried: {MODULE_CANDIDATES}"
    )


# =============================================================================
# CONFIG: SENSOR MAPPING AND RUN OPTIONS
# =============================================================================
CAM_TARGET = "T1Ka"                 # confirmed inner-roof / indoor-side roof surface
INDOOR_AIR_PROXY = "T1Ke"           # upper-zone indoor air proxy, not room-average air
OUTER_ROOF_CANDIDATE = "T2A2"       # candidate above-slab / below-substrate interface
LESS_PREFERRED_OUTER_CANDIDATE = "T2A"

# Since there was no fan/ventilation inside the cabin, sweep natural/stagnant
# indoor convection values. Keep 8.0 as previous reference.
H_IN_SWEEP = [2.0, 3.0, 4.0, 6.0, 8.0]

# Main physical parameters. These follow the previous runner style.
BASE_PARAMETER_GUESS = dict(
    rho_g=400.0,
    H_slab=0.10,
    H_g=0.10,
    theta_sat=0.90,
    k_theta_sat=5e-6,
    lambda_dry=0.12,
)

IRRIGATION_WINDOWS = getattr(
    gr,
    "DEFAULT_IRRIGATION_WINDOWS",
    (("07:00", "07:04"), ("16:30", "16:34")),
)
IRRIGATION_MM_PER_MIN = 0.0

# Do not set T2A2 as model boundary here. It is an observational diagnostic.
# Do not set T1Ke as substrate temperature; it is an indoor-air proxy.
T_G_TOP_INITIAL_C = None

SENSOR_LABELS = {
    "T1Ka": "T1Ka = confirmed inner roof / T_s_in target",
    "T1Ke": "T1Ke = upper-zone indoor air proxy / T_air_upper",
    "T2A2": "T2A2 = candidate outer roof interface above slab, below substrate",
    "T2A": "T2A = uncertain candidate, not preferred for outer roof",
    "T1Kd": "T1Kd = floor-related sensor",
    "T1Kb": "T1Kb = east-side related sensor",
    "T1Kc": "T1Kc = west-side related sensor",
}


def sensor_label(col: str | None) -> str:
    if col is None:
        return "None / default"
    return SENSOR_LABELS.get(col, col)


# Use CAM validation window from model file, but set its target explicitly.
gr.VALIDATION_TARGETS["CAM"] = CAM_TARGET


# =============================================================================
# BASIC HELPERS
# =============================================================================
def as_1min_series(series: pd.Series, name: str | None = None) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")
    s = s.sort_index().resample("1min").mean()
    if name is not None:
        s.name = name
    return s


def make_result_series(results: dict, key: str, name: str | None = None) -> pd.Series:
    if key not in results:
        raise KeyError(f"Result key not found: {key}. Available keys: {list(results.keys())}")
    return pd.Series(
        results[key],
        index=pd.to_datetime(results["datetime"]),
        name=name or key,
    ).sort_index().resample("1min").mean()


def common_frame(*series: pd.Series) -> pd.DataFrame:
    df = pd.concat(series, axis=1)
    return df.dropna()


def amplitude_p95_p05(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(0.95) - s.quantile(0.05))


def amplitude_max_min(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.max() - s.min())


def compute_metrics(sim: pd.Series, obs: pd.Series) -> dict:
    sim = as_1min_series(sim, "sim")
    obs = as_1min_series(obs, "obs")
    df = common_frame(sim, obs)
    if df.empty:
        raise ValueError("No valid overlap between simulation and observation.")

    err = df["sim"] - df["obs"]
    obs_amp = amplitude_max_min(df["obs"])
    sim_amp = amplitude_max_min(df["sim"])
    return {
        "n": int(len(df)),
        "bias_C": float(err.mean()),
        "mae_C": float(err.abs().mean()),
        "rmse_C": float(np.sqrt((err ** 2).mean())),
        "corr": float(df["sim"].corr(df["obs"])),
        "obs_min_C": float(df["obs"].min()),
        "obs_max_C": float(df["obs"].max()),
        "obs_amp_C": float(obs_amp),
        "sim_min_C": float(df["sim"].min()),
        "sim_max_C": float(df["sim"].max()),
        "sim_amp_C": float(sim_amp),
        "amp_error_C": float(sim_amp - obs_amp),
        "abs_amp_error_C": float(abs(sim_amp - obs_amp)),
        "obs_peak_time": str(df["obs"].idxmax()),
        "sim_peak_time": str(df["sim"].idxmax()),
    }


def estimate_lag_minutes(reference: pd.Series, candidate: pd.Series, max_lag_min: int = 360, step_min: int = 5) -> dict:
    """Estimate candidate lead/lag relative to reference by cross-correlation.

    Positive lead_min means candidate leads reference by that many minutes.
    This is implemented by shifting the candidate forward/backward and finding
    the shift that maximizes correlation with reference.
    """
    ref = as_1min_series(reference, "reference")
    cand = as_1min_series(candidate, "candidate")
    rows = []

    for lead in range(-max_lag_min, max_lag_min + 1, step_min):
        # If candidate leads reference by +lead, then candidate(t) corresponds
        # to reference(t+lead). Shift candidate later to align with reference.
        shifted = cand.shift(freq=f"{lead}min")
        df = common_frame(ref, shifted.rename("candidate_shifted"))
        if len(df) < 30:
            continue
        rows.append({"candidate_leads_reference_min": lead, "corr": float(df.iloc[:, 0].corr(df.iloc[:, 1]))})

    curve = pd.DataFrame(rows)
    if curve.empty:
        return {
            "best_lead_min": np.nan,
            "best_corr": np.nan,
            "curve": curve,
        }

    best = curve.loc[curve["corr"].idxmax()]
    return {
        "best_lead_min": float(best["candidate_leads_reference_min"]),
        "best_corr": float(best["corr"]),
        "curve": curve,
    }


def daily_peak_lag_minutes(reference: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    """Daily peak timing check: candidate peak minus reference peak.

    Negative candidate_minus_reference_peak_min means candidate peaks earlier.
    """
    ref = as_1min_series(reference, "reference")
    cand = as_1min_series(candidate, "candidate")
    df = common_frame(ref, cand)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for day, g in df.groupby(df.index.date):
        if len(g) < 60:
            continue
        ref_peak = g["reference"].idxmax()
        cand_peak = g["candidate"].idxmax()
        rows.append({
            "date": str(day),
            "reference_peak_time": str(ref_peak),
            "candidate_peak_time": str(cand_peak),
            "candidate_minus_reference_peak_min": float((cand_peak - ref_peak).total_seconds() / 60.0),
            "candidate_leads_reference_by_peak_min": float((ref_peak - cand_peak).total_seconds() / 60.0),
            "reference_peak_C": float(g["reference"].max()),
            "candidate_peak_C": float(g["candidate"].max()),
        })
    return pd.DataFrame(rows)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return s * np.nan
    return (s - s.mean()) / std


def safe_weather_series(weather: pd.DataFrame, col: str) -> pd.Series | None:
    if weather is None or col not in weather.columns:
        return None
    return as_1min_series(weather[col], col)


def finite_min_max(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.min()), float(arr.max())


# =============================================================================
# DIAGNOSTICS
# =============================================================================
def save_sensor_stats(ni: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    focus_cols = [
        "T1Ka", "T1Ke", "T2A2", "T2A",
        "T3Ka", "T2Ka", "T1Kd", "T1Kb", "T1Kc", "T1Ta2", "T1Tb", "T1Ta",
    ]
    focus_cols = [c for c in focus_cols if c in ni.columns]
    rows = []
    G = safe_weather_series(weather, "G_sol") if weather is not None else None
    Ta = safe_weather_series(weather, "T_a") if weather is not None else None

    for c in focus_cols:
        s = as_1min_series(ni[c], c).dropna()
        if s.empty:
            continue
        row = {
            "channel": c,
            "current_interpretation": sensor_label(c),
            "n": int(len(s)),
            "min_C": float(s.min()),
            "mean_C": float(s.mean()),
            "max_C": float(s.max()),
            "amp_max_min_C": amplitude_max_min(s),
            "amp_p95_p05_C": amplitude_p95_p05(s),
            "peak_time": str(s.idxmax()),
        }
        if G is not None:
            dg = common_frame(s.rename(c), G.rename("G_sol"))
            row["corr_with_G_sol"] = float(dg[c].corr(dg["G_sol"])) if len(dg) > 30 else np.nan
        if Ta is not None:
            dt = common_frame(s.rename(c), Ta.rename("T_a"))
            row["corr_with_T_a"] = float(dt[c].corr(dt["T_a"])) if len(dt) > 30 else np.nan
        rows.append(row)

    stats = pd.DataFrame(rows)
    stats.to_csv(OUTPUT_DIR / "cam_final_sensor_stats.csv", index=False)
    return stats


def diagnose_upper_air(ni: pd.DataFrame) -> pd.DataFrame:
    if CAM_TARGET not in ni.columns or INDOOR_AIR_PROXY not in ni.columns:
        print("Skipping T1Ke diagnostic because T1Ka or T1Ke is missing.")
        return pd.DataFrame()

    roof = as_1min_series(ni[CAM_TARGET], "T1Ka_inner_roof")
    air = as_1min_series(ni[INDOOR_AIR_PROXY], "T1Ke_upper_air")
    df = common_frame(roof, air)
    if df.empty:
        return pd.DataFrame()

    df["T1Ke_minus_T1Ka_C"] = df["T1Ke_upper_air"] - df["T1Ka_inner_roof"]
    metrics = pd.DataFrame([{
        "n": int(len(df)),
        "corr_T1Ka_T1Ke": float(df["T1Ka_inner_roof"].corr(df["T1Ke_upper_air"])),
        "mean_T1Ke_minus_T1Ka_C": float(df["T1Ke_minus_T1Ka_C"].mean()),
        "median_T1Ke_minus_T1Ka_C": float(df["T1Ke_minus_T1Ka_C"].median()),
        "T1Ka_amp_C": amplitude_max_min(df["T1Ka_inner_roof"]),
        "T1Ke_amp_C": amplitude_max_min(df["T1Ke_upper_air"]),
        "T1Ke_over_T1Ka_amp_ratio": float(
            amplitude_max_min(df["T1Ke_upper_air"]) / amplitude_max_min(df["T1Ka_inner_roof"])
        ) if amplitude_max_min(df["T1Ka_inner_roof"]) != 0 else np.nan,
    }])
    metrics.to_csv(OUTPUT_DIR / "cam_upper_air_T1Ke_diagnostic_metrics.csv", index=False)
    df.to_csv(OUTPUT_DIR / "cam_upper_air_T1Ke_timeseries.csv")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    axes[0].plot(df.index, df["T1Ka_inner_roof"], label=sensor_label(CAM_TARGET), linewidth=2)
    axes[0].plot(df.index, df["T1Ke_upper_air"], label=sensor_label(INDOOR_AIR_PROXY), linewidth=2)
    axes[0].set_title("CAM diagnostic: T1Ke as upper-zone indoor air proxy")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["T1Ke_minus_T1Ka_C"], linewidth=1.4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("T1Ke - T1Ka (°C)")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(df["T1Ka_inner_roof"], df["T1Ke_upper_air"], s=8, alpha=0.45)
    axes[2].set_xlabel("T1Ka inner roof / T_s_in (°C)")
    axes[2].set_ylabel("T1Ke upper-zone air proxy (°C)")
    axes[2].set_title(f"Scatter check, corr = {metrics.loc[0, 'corr_T1Ka_T1Ke']:.3f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_CAM_T1Ke_upper_air_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics


def diagnose_outer_to_inner_lag(ni: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    if CAM_TARGET not in ni.columns or OUTER_ROOF_CANDIDATE not in ni.columns:
        print("Skipping T2A2 lag diagnostic because T1Ka or T2A2 is missing.")
        return pd.DataFrame()

    inner = as_1min_series(ni[CAM_TARGET], "T1Ka_inner_roof")
    outer = as_1min_series(ni[OUTER_ROOF_CANDIDATE], "T2A2_outer_interface")
    df = common_frame(inner, outer)
    if df.empty:
        return pd.DataFrame()

    lag = estimate_lag_minutes(inner, outer, max_lag_min=360, step_min=5)
    lag_curve = lag["curve"]
    lag_curve.to_csv(OUTPUT_DIR / "cam_T2A2_to_T1Ka_lag_correlation_curve.csv", index=False)

    daily = daily_peak_lag_minutes(inner, outer)
    daily.to_csv(OUTPUT_DIR / "cam_T2A2_to_T1Ka_daily_peak_lag.csv", index=False)

    G = safe_weather_series(weather, "G_sol") if weather is not None else None
    Ta = safe_weather_series(weather, "T_a") if weather is not None else None
    RH = safe_weather_series(weather, "RH") if weather is not None else None

    corr_g = np.nan
    corr_ta = np.nan
    corr_rh = np.nan
    if G is not None:
        dg = common_frame(outer.rename("T2A2"), G.rename("G_sol"))
        corr_g = float(dg["T2A2"].corr(dg["G_sol"])) if len(dg) > 30 else np.nan
    if Ta is not None:
        dt = common_frame(outer.rename("T2A2"), Ta.rename("T_a"))
        corr_ta = float(dt["T2A2"].corr(dt["T_a"])) if len(dt) > 30 else np.nan
    if RH is not None:
        dr = common_frame(outer.rename("T2A2"), RH.rename("RH"))
        corr_rh = float(dr["T2A2"].corr(dr["RH"])) if len(dr) > 30 else np.nan

    summary = pd.DataFrame([{
        "outer_candidate": OUTER_ROOF_CANDIDATE,
        "outer_role": sensor_label(OUTER_ROOF_CANDIDATE),
        "inner_reference": CAM_TARGET,
        "inner_role": sensor_label(CAM_TARGET),
        "n": int(len(df)),
        "corr_T2A2_T1Ka_zero_lag": float(df["T2A2_outer_interface"].corr(df["T1Ka_inner_roof"])),
        "T2A2_amp_p95_p05_C": amplitude_p95_p05(df["T2A2_outer_interface"]),
        "T1Ka_amp_p95_p05_C": amplitude_p95_p05(df["T1Ka_inner_roof"]),
        "T2A2_amp_ratio_vs_T1Ka": float(
            amplitude_p95_p05(df["T2A2_outer_interface"]) / amplitude_p95_p05(df["T1Ka_inner_roof"])
        ) if amplitude_p95_p05(df["T1Ka_inner_roof"]) != 0 else np.nan,
        "xcorr_T2A2_leads_T1Ka_min": lag["best_lead_min"],
        "xcorr_best_corr": lag["best_corr"],
        "daily_median_T2A2_leads_T1Ka_by_peak_min": float(daily["candidate_leads_reference_by_peak_min"].median()) if not daily.empty else np.nan,
        "daily_mean_T2A2_leads_T1Ka_by_peak_min": float(daily["candidate_leads_reference_by_peak_min"].mean()) if not daily.empty else np.nan,
        "corr_T2A2_with_G_sol": corr_g,
        "corr_T2A2_with_T_a": corr_ta,
        "corr_T2A2_with_RH": corr_rh,
    }])
    summary.to_csv(OUTPUT_DIR / "cam_observed_outer_to_inner_lag_summary.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(df.index, df["T2A2_outer_interface"], label=sensor_label(OUTER_ROOF_CANDIDATE), linewidth=2)
    axes[0].plot(df.index, df["T1Ka_inner_roof"], label=sensor_label(CAM_TARGET), linewidth=2)
    axes[0].set_title("Observed CAM lag: outer roof interface candidate T2A2 vs inner roof T1Ka")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, zscore(df["T2A2_outer_interface"]), label="T2A2 z-score", linewidth=1.8)
    axes[1].plot(df.index, zscore(df["T1Ka_inner_roof"]), label="T1Ka z-score", linewidth=1.8)
    if G is not None:
        gz = zscore(G).dropna()
        axes[1].plot(gz.index, gz.values, label="G_sol z-score", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Z-score")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if not lag_curve.empty:
        axes[2].plot(lag_curve["candidate_leads_reference_min"], lag_curve["corr"], linewidth=2)
        axes[2].axvline(0, linestyle="--", linewidth=1)
        axes[2].axvline(lag["best_lead_min"], linestyle=":", linewidth=1.4)
        axes[2].set_xlabel("T2A2 leads T1Ka by this many minutes")
        axes[2].set_ylabel("Correlation after shifting T2A2")
        axes[2].set_title(
            f"Best xcorr lead = {lag['best_lead_min']:.0f} min, corr = {lag['best_corr']:.3f}"
        )
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_CAM_T2A2_outer_to_T1Ka_inner_lag.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return summary


# =============================================================================
# PLOTS FOR SIMULATION RESULTS
# =============================================================================
def plot_hin_sweep(all_results: dict, metrics: pd.DataFrame, ni: pd.DataFrame, weather: pd.DataFrame | None) -> None:
    obs = as_1min_series(ni[CAM_TARGET], "T1Ka")
    t1ke = as_1min_series(ni[INDOOR_AIR_PROXY], "T1Ke") if INDOOR_AIR_PROXY in ni.columns else None
    t2a2 = as_1min_series(ni[OUTER_ROOF_CANDIDATE], "T2A2") if OUTER_ROOF_CANDIDATE in ni.columns else None
    G = safe_weather_series(weather, "G_sol") if weather is not None else None
    Ta = safe_weather_series(weather, "T_a") if weather is not None else None
    RH = safe_weather_series(weather, "RH") if weather is not None else None
    rain = safe_weather_series(weather, "rain") if weather is not None else None

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.35, 1.25, 1.0]},
    )

    ax = axes[0]
    ax.plot(obs.index, obs.values, label=sensor_label(CAM_TARGET), linewidth=2.8)
    for tag, results in all_results.items():
        sim = make_result_series(results, "T_s_in", tag)
        row = metrics.loc[metrics["case"] == tag]
        if not row.empty:
            label = f"{tag} | RMSE={row.iloc[0]['rmse_C']:.2f}, amp err={row.iloc[0]['amp_error_C']:.2f}"
        else:
            label = tag
        ax.plot(sim.index, sim.values, linestyle="--", linewidth=1.4, label=label)
    ax.set_title("CAM final: h_in sweep using T1Ke upper-zone indoor air as T_in boundary")
    ax.set_ylabel("T_s,in (°C)")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if t2a2 is not None:
        ax.plot(t2a2.index, t2a2.values, label=sensor_label(OUTER_ROOF_CANDIDATE), linewidth=1.8)
    ax.plot(obs.index, obs.values, label=sensor_label(CAM_TARGET), linewidth=1.8)
    if t1ke is not None:
        ax.plot(t1ke.index, t1ke.values, label=sensor_label(INDOOR_AIR_PROXY), linewidth=1.4)
    ax.set_ylabel("NI sensors (°C)")
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if Ta is not None:
        ax.plot(Ta.index, Ta.values, label="Weather T_a", linewidth=1.5)
    ax.set_ylabel("T_a (°C)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if G is not None:
        ax2.plot(G.index, G.values, label="G_sol", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("G_sol (W/m²)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax = axes[3]
    if RH is not None:
        ax.plot(RH.index, RH.values, label="RH", linewidth=1.4)
    ax.set_ylabel("RH (%)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if rain is not None:
        ax2.bar(rain.index, rain.values, width=2/1440, alpha=0.25, label="Rain")
    ax2.set_ylabel("Rain")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Mark irrigation times.
    try:
        start = obs.index.min().floor("D")
        end = obs.index.max().ceil("D")
        for day in pd.date_range(start, end, freq="D"):
            for hhmm in [w[0] for w in IRRIGATION_WINDOWS]:
                ts = pd.Timestamp(f"{day.date()} {hhmm}")
                if obs.index.min() <= ts <= obs.index.max():
                    for a in axes:
                        a.axvline(ts, linewidth=0.8, linestyle=":", alpha=0.35)
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_CAM_hin_sweep_T1Ka_target.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_selected_case(selected_tag: str, selected_results: dict, ni: pd.DataFrame, weather: pd.DataFrame | None) -> None:
    sim = make_result_series(selected_results, "T_s_in", "sim_T_s_in")
    obs = as_1min_series(ni[CAM_TARGET], "obs_T1Ka")
    df = common_frame(sim, obs)
    if df.empty:
        return
    df["error_C"] = df["sim_T_s_in"] - df["obs_T1Ka"]

    t1ke = as_1min_series(ni[INDOOR_AIR_PROXY], "T1Ke") if INDOOR_AIR_PROXY in ni.columns else None
    t2a2 = as_1min_series(ni[OUTER_ROOF_CANDIDATE], "T2A2") if OUTER_ROOF_CANDIDATE in ni.columns else None
    G = safe_weather_series(weather, "G_sol") if weather is not None else None
    Ta = safe_weather_series(weather, "T_a") if weather is not None else None

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df.index, df["obs_T1Ka"], label=sensor_label(CAM_TARGET), linewidth=2.6)
    axes[0].plot(df.index, df["sim_T_s_in"], label=f"Model T_s_in | {selected_tag}", linestyle="--", linewidth=2.0)
    axes[0].set_title("CAM selected case: inner roof validation target")
    axes[0].set_ylabel("T_s,in (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["error_C"], linewidth=1.4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Model - T1Ka (°C)")
    axes[1].grid(True, alpha=0.3)

    if t2a2 is not None:
        axes[2].plot(t2a2.index, t2a2.values, label=sensor_label(OUTER_ROOF_CANDIDATE), linewidth=1.8)
    axes[2].plot(df.index, df["obs_T1Ka"], label=sensor_label(CAM_TARGET), linewidth=1.8)
    if t1ke is not None:
        axes[2].plot(t1ke.index, t1ke.values, label=sensor_label(INDOOR_AIR_PROXY), linewidth=1.4)
    axes[2].set_ylabel("Observed sensors (°C)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    if Ta is not None:
        axes[3].plot(Ta.index, Ta.values, label="Weather T_a", linewidth=1.4)
    axes[3].set_ylabel("T_a (°C)")
    axes[3].set_xlabel("Datetime")
    axes[3].grid(True, alpha=0.3)
    ax2 = axes[3].twinx()
    if G is not None:
        ax2.plot(G.index, G.values, label="G_sol", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("G_sol (W/m²)")
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_CAM_selected_case_detail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save selected case timeseries.
    out = df.copy()
    if t1ke is not None:
        out = out.join(t1ke.rename("T1Ke_upper_air"), how="left")
    if t2a2 is not None:
        out = out.join(t2a2.rename("T2A2_outer_interface"), how="left")
    if Ta is not None:
        out = out.join(Ta.rename("weather_T_a"), how="left")
    if G is not None:
        out = out.join(G.rename("weather_G_sol"), how="left")
    out.to_csv(OUTPUT_DIR / "cam_final_best_case_timeseries.csv")


# =============================================================================
# RUN SIMULATION
# =============================================================================
def configure_parameters_for_hin(h_in: float) -> None:
    gr.apply_scientific_guess_parameters(**BASE_PARAMETER_GUESS, h_in=float(h_in))
    gr.geom.A_roof = 1.0
    gr.bromelia.cover_fraction = 0.70
    gr.wedelia.cover_fraction = 0.40
    gr.geom.dynamic_h_in = False


def run_one_case(h_in: float, weather: pd.DataFrame, ni: pd.DataFrame, theta_initial) -> tuple[str, dict, dict]:
    configure_parameters_for_hin(h_in)
    tag = f"T1Ke_upper_air_hin_{h_in:g}"

    if INDOOR_AIR_PROXY not in ni.columns:
        raise ValueError(f"Indoor-air proxy column {INDOOR_AIR_PROXY} not found. Available NI columns: {ni.columns.tolist()}")

    T_in_series = ni[INDOOR_AIR_PROXY]
    T_s_in_initial = float(pd.to_numeric(ni[CAM_TARGET], errors="coerce").dropna().iloc[0])

    print("\n" + "=" * 80)
    print(f"Running CAM case: {tag}")
    print(f"Target          : {sensor_label(CAM_TARGET)}")
    print(f"T_in boundary   : {sensor_label(INDOOR_AIR_PROXY)}")
    print(f"h_in            : {h_in:g} W/m²K")
    print(f"T_g_top_initial : {T_G_TOP_INITIAL_C} (None = model default; do not use T1Ke here)")
    print("=" * 80)

    results = gr.run_simulation(
        weather_df=weather,
        plant=gr.bromelia,
        substrate=gr.substrat,
        slab=gr.slab,
        geom=gr.geom,
        num=gr.num,
        theta_initial=theta_initial,
        T_in_series=T_in_series,
        T_g_top_initial_C=T_G_TOP_INITIAL_C,
        T_s_in_initial_C=T_s_in_initial,
        irrigation_windows=IRRIGATION_WINDOWS,
        irrigation_mm_per_min=IRRIGATION_MM_PER_MIN,
        save_every_s=60,
    )

    sim = make_result_series(results, "T_s_in", "sim")
    obs = as_1min_series(ni[CAM_TARGET], "obs")
    metrics = compute_metrics(sim, obs)

    tf_min, tf_max = finite_min_max(results.get("T_f", []))
    tin_min, tin_max = finite_min_max(results.get("T_in_used", []))
    metrics.update({
        "case": tag,
        "h_in_W_m2K": float(h_in),
        "target_col": CAM_TARGET,
        "target_role": sensor_label(CAM_TARGET),
        "tin_proxy_col": INDOOR_AIR_PROXY,
        "tin_proxy_role": sensor_label(INDOOR_AIR_PROXY),
        "outer_roof_candidate_col": OUTER_ROOF_CANDIDATE,
        "outer_roof_candidate_role": sensor_label(OUTER_ROOF_CANDIDATE),
        "Tf_min_C": tf_min,
        "Tf_max_C": tf_max,
        "Tin_used_min_C": tin_min,
        "Tin_used_max_C": tin_max,
    })

    print(pd.Series(metrics).to_string())
    return tag, results, metrics


def choose_best_case(metrics: pd.DataFrame) -> pd.Series:
    """Choose selected case using a transparent combined score.

    We do not optimize by RMSE alone because a too-flat curve can look acceptable
    in RMSE while failing the observed amplitude. The score prioritizes RMSE,
    amplitude consistency, and correlation.
    """
    m = metrics.copy()
    # Normalize defensively.
    rmse = m["rmse_C"].astype(float)
    amp = m["abs_amp_error_C"].astype(float)
    corr_penalty = (1.0 - m["corr"].astype(float)).clip(lower=0)

    # Main units are Celsius, so rmse and amp_error can be combined directly.
    m["selection_score"] = rmse + 0.50 * amp + 0.25 * corr_penalty
    return m.sort_values("selection_score").iloc[0]


def write_summary(selected: pd.Series, metrics: pd.DataFrame, lag_summary: pd.DataFrame, upper_air_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("CAM final sensor-mapping summary")
    lines.append("=" * 70)
    lines.append(f"Model module: {MODEL_MODULE_NAME}")
    lines.append("")
    lines.append("Sensor interpretation used in this runner:")
    lines.append(f"- {sensor_label(CAM_TARGET)}")
    lines.append(f"- {sensor_label(INDOOR_AIR_PROXY)}")
    lines.append(f"- {sensor_label(OUTER_ROOF_CANDIDATE)}")
    lines.append(f"- {sensor_label(LESS_PREFERRED_OUTER_CANDIDATE)}")
    lines.append("")
    lines.append("Selected CAM case:")
    for key in ["case", "h_in_W_m2K", "rmse_C", "mae_C", "bias_C", "corr", "obs_amp_C", "sim_amp_C", "amp_error_C", "selection_score"]:
        if key in selected:
            lines.append(f"- {key}: {selected[key]}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Main validation target is T1Ka because it is the confirmed inner-roof / indoor-side roof surface temperature.")
    lines.append("- T1Ke is used as a dynamic upper-zone indoor air boundary, not as room-average air temperature.")
    lines.append("- T2A2 is used as secondary observational evidence for outer-to-inner roof lag. It is not an exposed outdoor surface; it represents the above-slab / below-substrate interface candidate.")
    lines.append("- h_in is swept because the cabin had no fan or ventilation, making natural/stagnant indoor convection more defensible.")
    lines.append("")

    if lag_summary is not None and not lag_summary.empty:
        row = lag_summary.iloc[0]
        lines.append("Observed outer-to-inner lag diagnostic:")
        for key in [
            "xcorr_T2A2_leads_T1Ka_min",
            "xcorr_best_corr",
            "daily_median_T2A2_leads_T1Ka_by_peak_min",
            "T2A2_amp_ratio_vs_T1Ka",
            "corr_T2A2_with_G_sol",
            "corr_T2A2_with_T_a",
        ]:
            if key in row:
                lines.append(f"- {key}: {row[key]}")
        lines.append("")

    if upper_air_summary is not None and not upper_air_summary.empty:
        row = upper_air_summary.iloc[0]
        lines.append("T1Ke upper-zone air diagnostic:")
        for key in [
            "corr_T1Ka_T1Ke",
            "mean_T1Ke_minus_T1Ka_C",
            "T1Ke_over_T1Ka_amp_ratio",
        ]:
            if key in row:
                lines.append(f"- {key}: {row[key]}")
        lines.append("")

    lines.append("Suggested thesis phrasing:")
    lines.append(
        "The CAM model was primarily validated against T1Ka, which represents the confirmed indoor-side roof surface temperature. "
        "T1Ke was applied as an upper-zone indoor air temperature proxy for the inner convective boundary because the sensor was suspended inside a closed, unventilated cabin. "
        "T2A2 was evaluated as a candidate outer-roof interface temperature above the slab and below the substrate. "
        "The observed phase lead of T2A2 relative to T1Ka indicates thermal delay through the roof/slab assembly."
    )

    text = "\n".join(lines)
    (OUTPUT_DIR / "cam_final_selected_case_summary.txt").write_text(text, encoding="utf-8")

    md = text.replace("CAM final sensor-mapping summary\n" + "=" * 70, "# CAM final sensor-mapping summary")
    (OUTPUT_DIR / "cam_final_selected_case_summary.md").write_text(md, encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\n=== CAM final mapping runner ===")
    print(f"Base dir : {BASE_DIR}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Target   : {CAM_TARGET} -> {sensor_label(CAM_TARGET)}")
    print(f"T_in     : {INDOOR_AIR_PROXY} -> {sensor_label(INDOOR_AIR_PROXY)}")
    print(f"Outer obs: {OUTER_ROOF_CANDIDATE} -> {sensor_label(OUTER_ROOF_CANDIDATE)}")

    weather, ni, rika, theta_initial = gr.prepare_validation_case("CAM", base_dir=str(BASE_DIR))
    ni = ni.copy()

    required = [CAM_TARGET, INDOOR_AIR_PROXY]
    missing = [c for c in required if c not in ni.columns]
    if missing:
        raise ValueError(f"Missing required NI columns: {missing}. Available: {ni.columns.tolist()}")

    save_sensor_stats(ni, weather)
    upper_air_summary = diagnose_upper_air(ni)
    lag_summary = diagnose_outer_to_inner_lag(ni, weather)

    all_results = {}
    rows = []
    for h_in in H_IN_SWEEP:
        tag, results, metrics = run_one_case(h_in, weather, ni, theta_initial)
        all_results[tag] = results
        rows.append(metrics)

    metrics = pd.DataFrame(rows)
    cols_front = [
        "case", "h_in_W_m2K", "target_col", "tin_proxy_col", "outer_roof_candidate_col",
        "n", "bias_C", "mae_C", "rmse_C", "corr", "obs_amp_C", "sim_amp_C", "amp_error_C", "abs_amp_error_C",
        "obs_peak_time", "sim_peak_time", "Tf_min_C", "Tf_max_C", "Tin_used_min_C", "Tin_used_max_C",
    ]
    cols = [c for c in cols_front if c in metrics.columns] + [c for c in metrics.columns if c not in cols_front]
    metrics = metrics[cols]

    # Add transparent selection score.
    scored = metrics.copy()
    rmse = scored["rmse_C"].astype(float)
    amp = scored["abs_amp_error_C"].astype(float)
    corr_penalty = (1.0 - scored["corr"].astype(float)).clip(lower=0)
    scored["selection_score"] = rmse + 0.50 * amp + 0.25 * corr_penalty
    scored = scored.sort_values("selection_score")
    scored.to_csv(OUTPUT_DIR / "cam_final_hin_sweep_metrics.csv", index=False)

    selected = scored.iloc[0]
    selected_tag = str(selected["case"])
    print("\n=== CAM h_in sweep metrics ===")
    print(scored.to_string(index=False))
    print("\n=== Selected case ===")
    print(selected.to_string())

    plot_hin_sweep(all_results, scored, ni, weather)
    plot_selected_case(selected_tag, all_results[selected_tag], ni, weather)
    write_summary(selected, scored, lag_summary, upper_air_summary)

    # Save result keys for debugging/possible secondary model-output matching.
    key_dump = {
        tag: sorted([str(k) for k in results.keys()])
        for tag, results in all_results.items()
    }
    (OUTPUT_DIR / "cam_result_keys.json").write_text(json.dumps(key_dump, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    return all_results, scored


if __name__ == "__main__":
    main()
