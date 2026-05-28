"""
Check whether T2A / T2A2 behave like outdoor-side slab / roof temperature candidates.

Place this file in the same folder as your green-roof model runner and data files, then run:

    py check_t2a_t2a2_outdoor_slab_candidates.py

What it does:
- Loads CAM validation data using your model module's prepare_validation_case("CAM").
- Uses T1Ka as the confirmed indoor roof surface temperature / T_S_in.
- Tests T2A and T2A2 as possible outdoor-side slab/roof temperature sensors.
- Compares each candidate against T1Ka and weather drivers, especially solar radiation.
- Saves CSV metrics and diagnostic plots.

Interpretation:
- A likely outdoor-side slab/roof sensor should generally have stronger solar response,
  larger or earlier daily peak than T1Ka, and should often lead T1Ka in phase.
- This script does not confirm physical placement by itself. It ranks candidates from data behavior.
"""

from pathlib import Path
import importlib
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs_outdoor_slab_sensor_check"
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
for module_name in MODULE_CANDIDATES:
    try:
        gr = importlib.import_module(module_name)
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
# CONFIG
# =============================================================================
CONFIRMED_INNER_ROOF = "T1Ka"      # confirmed indoor-side roof surface / T_S_in
UPPER_INDOOR_AIR = "T1Ke"          # optional reference only, upper-zone indoor air proxy
OUTDOOR_CANDIDATES = ["T2A", "T2A2"]

# Add more if you want broader search.
OPTIONAL_EXTRA_CANDIDATES = [
    # "T2Ka", "T3Ka", "T1Ta", "T1Ta2", "T1Kb", "T1Kc", "T1Kd"
]

RESAMPLE_RULE = "1min"
DAYTIME_GSOL_THRESHOLD = 80.0       # W/m2; used to split day/night if G_sol exists
MAX_LAG_MIN = 360                   # search +/- 6 hours for thermal lag
LAG_STEP_MIN = 5
TEMP_REASONABLE_MIN = 5.0
TEMP_REASONABLE_MAX = 80.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def as_1min_series(df, col, name=None):
    if df is None or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.sort_index().resample(RESAMPLE_RULE).mean()
    s.name = name or col
    return s


def clean_temp_series(s):
    s = s.copy()
    s[(s < TEMP_REASONABLE_MIN) | (s > TEMP_REASONABLE_MAX)] = np.nan
    return s


def amp_p95_p05(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(0.95) - s.quantile(0.05))


def safe_corr(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def lag_candidate_leads_reference(candidate, reference, max_lag_min=360, step_min=5):
    """
    Estimate phase relation between candidate and reference.

    Positive returned lag means:
        candidate leads reference by that many minutes.

    Method:
        Shift candidate forward in time by lag minutes and find the lag that maximizes
        correlation with the reference. If shifting candidate forward improves alignment,
        candidate originally happened earlier, so candidate leads the reference.
    """
    rows = []
    for lag in np.arange(-max_lag_min, max_lag_min + step_min, step_min):
        shifted_candidate = candidate.shift(freq=f"{int(lag)}min")
        df = pd.concat([shifted_candidate.rename("candidate_shifted"), reference.rename("reference")], axis=1).dropna()
        if len(df) < 30:
            continue
        corr = df["candidate_shifted"].corr(df["reference"])
        rows.append({"candidate_leads_reference_min": float(lag), "corr": float(corr)})

    if not rows:
        return np.nan, np.nan, pd.DataFrame()

    lag_df = pd.DataFrame(rows)
    # Use maximum positive correlation, not absolute correlation, because temperature waves should be positively related.
    best = lag_df.loc[lag_df["corr"].idxmax()]
    return float(best["candidate_leads_reference_min"]), float(best["corr"]), lag_df


def daily_peak_lag(candidate, reference):
    """
    Daily peak timing difference.

    Negative candidate_minus_reference means candidate peaks earlier than reference.
    Positive means candidate peaks later.
    """
    df = pd.concat([candidate.rename("candidate"), reference.rename("reference")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for day, g in df.groupby(df.index.date):
        if len(g) < 60:
            continue
        cand_amp = g["candidate"].quantile(0.95) - g["candidate"].quantile(0.05)
        ref_amp = g["reference"].quantile(0.95) - g["reference"].quantile(0.05)
        if cand_amp < 0.25 or ref_amp < 0.25:
            continue
        cand_peak = g["candidate"].idxmax()
        ref_peak = g["reference"].idxmax()
        lag_min = (cand_peak - ref_peak).total_seconds() / 60.0
        rows.append({
            "date": str(day),
            "candidate_peak_time": str(cand_peak),
            "T1Ka_peak_time": str(ref_peak),
            "candidate_minus_T1Ka_peak_min": float(lag_min),
            "candidate_amp_C": float(cand_amp),
            "T1Ka_amp_C": float(ref_amp),
        })
    return pd.DataFrame(rows)


def daytime_night_stats(candidate, t1ka, gsol):
    if gsol is None:
        return {}

    df = pd.concat([
        candidate.rename("candidate"),
        t1ka.rename("T1Ka"),
        gsol.rename("G_sol"),
    ], axis=1).dropna()

    if df.empty:
        return {}

    day = df[df["G_sol"] >= DAYTIME_GSOL_THRESHOLD]
    night = df[df["G_sol"] < DAYTIME_GSOL_THRESHOLD]

    out = {}
    if len(day) > 20:
        out.update({
            "day_mean_candidate_C": float(day["candidate"].mean()),
            "day_mean_T1Ka_C": float(day["T1Ka"].mean()),
            "day_mean_candidate_minus_T1Ka_C": float((day["candidate"] - day["T1Ka"]).mean()),
        })
    if len(night) > 20:
        out.update({
            "night_mean_candidate_C": float(night["candidate"].mean()),
            "night_mean_T1Ka_C": float(night["T1Ka"].mean()),
            "night_mean_candidate_minus_T1Ka_C": float((night["candidate"] - night["T1Ka"]).mean()),
        })
    return out


def score_candidate(row):
    """
    Heuristic score. Higher score means more consistent with outdoor-side slab/roof behavior.
    This is not proof; it only ranks candidates.
    """
    score = 0
    notes = []

    amp_ratio = row.get("amp_ratio_vs_T1Ka", np.nan)
    corr_g = row.get("corr_with_G_sol", np.nan)
    lead = row.get("xcorr_candidate_leads_T1Ka_min", np.nan)
    peak_lag = row.get("median_daily_candidate_minus_T1Ka_peak_min", np.nan)
    day_delta = row.get("day_mean_candidate_minus_T1Ka_C", np.nan)

    if np.isfinite(amp_ratio):
        if amp_ratio >= 1.10:
            score += 2
            notes.append("amplitude clearly larger than T1Ka")
        elif amp_ratio >= 0.80:
            score += 1
            notes.append("amplitude comparable to T1Ka")
        else:
            notes.append("amplitude much lower than T1Ka")

    if np.isfinite(corr_g):
        if corr_g >= 0.50:
            score += 2
            notes.append("strong solar relation")
        elif corr_g >= 0.25:
            score += 1
            notes.append("moderate solar relation")
        else:
            notes.append("weak solar relation")

    if np.isfinite(lead):
        if lead >= 30:
            score += 2
            notes.append("candidate leads T1Ka by cross-correlation")
        elif lead >= -15:
            score += 1
            notes.append("candidate roughly in phase with T1Ka")
        else:
            notes.append("candidate lags T1Ka; less likely outdoor-side")

    if np.isfinite(peak_lag):
        if peak_lag <= -30:
            score += 2
            notes.append("daily peak occurs before T1Ka")
        elif peak_lag <= 30:
            score += 1
            notes.append("daily peak near T1Ka")
        else:
            notes.append("daily peak occurs after T1Ka")

    if np.isfinite(day_delta):
        if day_delta > 0.5:
            score += 1
            notes.append("warmer than T1Ka during daytime")
        elif day_delta < -1.0:
            notes.append("cooler than T1Ka during daytime")

    return score, "; ".join(notes)


def plot_timeseries(ni, weather, candidate_cols, t1ka):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.0]})

    ax = axes[0]
    ax.plot(t1ka.index, t1ka.values, label="T1Ka confirmed inner roof / T_S_in", linewidth=2.5)

    if UPPER_INDOOR_AIR in ni.columns:
        t1ke = clean_temp_series(as_1min_series(ni, UPPER_INDOOR_AIR))
        ax.plot(t1ke.index, t1ke.values, label="T1Ke upper-zone indoor air proxy", linewidth=1.4, alpha=0.85)

    for c in candidate_cols:
        s = clean_temp_series(as_1min_series(ni, c))
        ax.plot(s.index, s.values, label=f"{c} outdoor-side candidate", linewidth=1.7)

    ax.set_title("T2A / T2A2 outdoor-side slab/roof candidate check")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    gsol = as_1min_series(weather, "G_sol") if weather is not None and "G_sol" in weather.columns else None
    ta = as_1min_series(weather, "T_a") if weather is not None and "T_a" in weather.columns else None
    if ta is not None:
        ax.plot(ta.index, ta.values, label="Weather T_a", linewidth=1.4)
    ax.set_ylabel("T_a (°C)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if gsol is not None:
        ax2.plot(gsol.index, gsol.values, label="G_sol", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("G_sol (W/m²)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    ax = axes[2]
    rain = as_1min_series(weather, "rain") if weather is not None and "rain" in weather.columns else None
    rh = as_1min_series(weather, "RH") if weather is not None and "RH" in weather.columns else None
    if rh is not None:
        ax.plot(rh.index, rh.values, label="RH", linewidth=1.2)
    ax.set_ylabel("RH (%)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if rain is not None:
        ax2.bar(rain.index, rain.values, width=2/1440, alpha=0.25, label="Rain")
    ax2.set_ylabel("Rain")
    ax.set_xlabel("Datetime")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    plt.tight_layout()
    path = OUTPUT_DIR / "01_t2a_t2a2_vs_t1ka_weather.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_normalized_phase(ni, weather, candidate_cols, t1ka):
    fig, ax = plt.subplots(figsize=(14, 6))

    def zscore(s):
        s = s.dropna()
        if s.std() == 0 or np.isnan(s.std()):
            return s * np.nan
        return (s - s.mean()) / s.std()

    ax.plot(zscore(t1ka).index, zscore(t1ka).values, label="T1Ka z-score", linewidth=2.5)

    for c in candidate_cols:
        s = clean_temp_series(as_1min_series(ni, c))
        ax.plot(zscore(s).index, zscore(s).values, label=f"{c} z-score", linewidth=1.7)

    if weather is not None and "G_sol" in weather.columns:
        gsol = as_1min_series(weather, "G_sol")
        ax.plot(zscore(gsol).index, zscore(gsol).values, label="G_sol z-score", linestyle="--", linewidth=1.2)

    ax.set_title("Normalized phase comparison: candidates vs T1Ka vs solar radiation")
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = OUTPUT_DIR / "02_normalized_phase_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_lag_curves(lag_curve_dict):
    if not lag_curve_dict:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, lag_df in lag_curve_dict.items():
        if lag_df.empty:
            continue
        ax.plot(lag_df["candidate_leads_reference_min"], lag_df["corr"], label=c, linewidth=1.8)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Candidate leads T1Ka by this many minutes")
    ax.set_ylabel("Correlation after shifting candidate")
    ax.set_title("Cross-correlation lag check")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "03_lag_correlation_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== Outdoor-side slab/roof sensor candidate check ===")
    print(f"Base dir : {BASE_DIR}")
    print(f"Output   : {OUTPUT_DIR}")

    weather, ni, rika, theta_initial = gr.prepare_validation_case("CAM", base_dir=str(BASE_DIR))
    ni = ni.copy()
    weather = weather.copy() if weather is not None else None

    if CONFIRMED_INNER_ROOF not in ni.columns:
        raise ValueError(f"{CONFIRMED_INNER_ROOF} not found in NI columns: {ni.columns.tolist()}")

    candidate_cols = [c for c in OUTDOOR_CANDIDATES + OPTIONAL_EXTRA_CANDIDATES if c in ni.columns]
    if not candidate_cols:
        raise ValueError(
            f"None of the candidate columns were found. Tried: {OUTDOOR_CANDIDATES + OPTIONAL_EXTRA_CANDIDATES}\n"
            f"Available NI columns: {ni.columns.tolist()}"
        )

    print(f"Confirmed inner roof target: {CONFIRMED_INNER_ROOF}")
    print(f"Candidates checked: {candidate_cols}")

    t1ka = clean_temp_series(as_1min_series(ni, CONFIRMED_INNER_ROOF, "T1Ka"))
    gsol = as_1min_series(weather, "G_sol") if weather is not None and "G_sol" in weather.columns else None
    ta = as_1min_series(weather, "T_a") if weather is not None and "T_a" in weather.columns else None
    rh = as_1min_series(weather, "RH") if weather is not None and "RH" in weather.columns else None

    rows = []
    daily_peak_rows = []
    lag_curve_dict = {}

    amp_t1ka = amp_p95_p05(t1ka)

    for c in candidate_cols:
        s = clean_temp_series(as_1min_series(ni, c, c))
        df = pd.concat([s.rename(c), t1ka.rename("T1Ka")], axis=1).dropna()
        if df.empty:
            print(f"Skipping {c}: no overlap with T1Ka")
            continue

        lead_min, lead_corr, lag_df = lag_candidate_leads_reference(s, t1ka, MAX_LAG_MIN, LAG_STEP_MIN)
        lag_curve_dict[c] = lag_df

        peak_df = daily_peak_lag(s, t1ka)
        if not peak_df.empty:
            peak_df.insert(0, "candidate", c)
            daily_peak_rows.append(peak_df)
            median_peak_lag = float(peak_df["candidate_minus_T1Ka_peak_min"].median())
        else:
            median_peak_lag = np.nan

        row = {
            "candidate": c,
            "n_overlap": int(len(df)),
            "mean_C": float(df[c].mean()),
            "min_C": float(df[c].min()),
            "max_C": float(df[c].max()),
            "amp_p95_p05_C": amp_p95_p05(df[c]),
            "T1Ka_amp_p95_p05_C": amp_t1ka,
            "amp_ratio_vs_T1Ka": amp_p95_p05(df[c]) / amp_t1ka if amp_t1ka and np.isfinite(amp_t1ka) else np.nan,
            "mean_candidate_minus_T1Ka_C": float((df[c] - df["T1Ka"]).mean()),
            "corr_with_T1Ka_zero_lag": safe_corr(s, t1ka),
            "xcorr_candidate_leads_T1Ka_min": lead_min,
            "xcorr_best_corr": lead_corr,
            "median_daily_candidate_minus_T1Ka_peak_min": median_peak_lag,
            "corr_with_G_sol": safe_corr(s, gsol) if gsol is not None else np.nan,
            "corr_with_T_a": safe_corr(s, ta) if ta is not None else np.nan,
            "corr_with_RH": safe_corr(s, rh) if rh is not None else np.nan,
            "candidate_peak_time_global": str(df[c].idxmax()),
            "T1Ka_peak_time_global": str(df["T1Ka"].idxmax()),
        }

        row.update(daytime_night_stats(s, t1ka, gsol))
        score, notes = score_candidate(row)
        row["outdoor_side_behavior_score"] = score
        row["interpretation_notes"] = notes
        rows.append(row)

        # Save aligned time series for this candidate.
        aligned = pd.concat([
            s.rename(c),
            t1ka.rename("T1Ka_inner_roof"),
            gsol.rename("G_sol") if gsol is not None else None,
            ta.rename("T_a") if ta is not None else None,
            rh.rename("RH") if rh is not None else None,
        ], axis=1)
        aligned.to_csv(OUTPUT_DIR / f"aligned_timeseries_{c}_vs_T1Ka_weather.csv")

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No metrics were computed. Check data overlap and candidate names.")

    metrics = metrics.sort_values(["outdoor_side_behavior_score", "corr_with_G_sol", "amp_ratio_vs_T1Ka"], ascending=[False, False, False])
    metrics.to_csv(OUTPUT_DIR / "t2a_t2a2_outdoor_slab_candidate_metrics.csv", index=False)

    if daily_peak_rows:
        daily_peaks = pd.concat(daily_peak_rows, ignore_index=True)
        daily_peaks.to_csv(OUTPUT_DIR / "daily_peak_lag_candidates_vs_T1Ka.csv", index=False)
    else:
        daily_peaks = pd.DataFrame()

    plot_timeseries(ni, weather, candidate_cols, t1ka)
    plot_normalized_phase(ni, weather, candidate_cols, t1ka)
    plot_lag_curves(lag_curve_dict)

    print("\n=== CANDIDATE METRICS SUMMARY ===")
    print(metrics.to_string(index=False))

    print("\n=== QUICK INTERPRETATION ===")
    best = metrics.iloc[0]
    print(f"Best behavioral candidate: {best['candidate']}")
    print(f"Score: {best['outdoor_side_behavior_score']}")
    print(f"Notes: {best['interpretation_notes']}")
    print("\nUse this as behavioral evidence only. Physical cable tracing is still the final confirmation.")

    print("\nDONE")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    return metrics, daily_peaks


if __name__ == "__main__":
    main()
