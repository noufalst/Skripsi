"""
CAM hypothesis simulation with weather overlay:
- Treat T1Ka as confirmed CAM indoor-roof surface temperature / T_s_in target.
- Treat T1Ke as the tested indoor-air temperature proxy / T_air_in boundary.
- Do NOT use T1Ka as T_in boundary, except for the optional circular diagnostic case.
- Do NOT use T1Ke as substrate-top initial temperature in this version, because here
  T1Ke is being tested as indoor air temperature.

Run:
    py run_green_roof_cam_t1ke_indoor_air.py

Expected files in:
    outputs_cam_t1ke_indoor_air_weather/

Required in same folder:
    one of these model modules:
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

from pathlib import Path
import importlib
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
OUTPUT_DIR = SCRIPT_DIR / "outputs_cam_t1ke_indoor_air_weather"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# IMPORT MODEL MODULE — prefer latest guarded model if available
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
IRRIGATION_WINDOWS = getattr(gr, "DEFAULT_IRRIGATION_WINDOWS", (("07:00", "07:04"), ("16:30", "16:34")))
IRRIGATION_MM_PER_MIN = 0.0

# Main hypothesis / current sensor interpretation
CAM_TARGET = "T1Ka"          # confirmed CAM indoor-roof surface temperature, T_s_in
INDOOR_AIR_PROXY = "T1Ke"    # tested as indoor air temperature, T_air_in / T_in

SENSOR_LABELS = {
    "T1Ka": "T1Ka = measured T_s_in / indoor roof surface",
    "T1Ke": "T1Ke = tested T_air_in / indoor air proxy",
    "T1Kd": "T1Kd = floor-related sensor",
    "T1Kb": "T1Kb = east-side related sensor",
    "T1Kc": "T1Kc = west-side related sensor",
}


def sensor_label(col):
    return SENSOR_LABELS.get(col, col)


# T_in candidates. T1Ke is now the main tested indoor-air boundary.
# Other cases are kept only for comparison/sensitivity.
TIN_PROXY_CANDIDATES = {
    "T1Ke_indoor_air_proxy": INDOOR_AIR_PROXY,
    "T1Kd_internal_floor_proxy": "T1Kd",
    "T1Kb_wall_proxy": "T1Kb",
    "T1Kc_wall_proxy": "T1Kc",
    "default_no_dynamic_Tin": None,
    # This one is intentionally circular and should NOT be used as final.
    # It is included only as a diagnostic upper-bound fit check.
    "T1Ka_circular_do_not_use": "T1Ka",
}

# =============================================================================
# SCIENTIFIC-GUESS PARAMETERS — same style as previous runner
# =============================================================================
gr.apply_scientific_guess_parameters(
    rho_g=400.0,
    H_slab=0.10,
    H_g=0.10,
    theta_sat=0.90,
    k_theta_sat=5e-6,
    lambda_dry=0.12,
    h_in=8.0,
)

gr.geom.A_roof = 1.0
gr.bromelia.cover_fraction = 0.70
gr.wedelia.cover_fraction = 0.40
gr.geom.dynamic_h_in = False

# Use CAM validation window from model file.
gr.VALIDATION_TARGETS["CAM"] = CAM_TARGET


def make_series(results, key):
    return pd.Series(
        results[key],
        index=pd.to_datetime(results["datetime"]),
        name=key,
    ).sort_index().resample("1min").mean()


def compute_metrics(sim, obs):
    sim = sim.sort_index().resample("1min").mean()
    obs = obs.sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)
    df = pd.concat([sim.loc[common].rename("sim"), obs.loc[common].rename("obs")], axis=1).dropna()
    if df.empty:
        raise ValueError("No valid overlap between simulation and observation.")
    err = df["sim"] - df["obs"]
    return {
        "n": int(len(df)),
        "bias_C": float(err.mean()),
        "mae_C": float(err.abs().mean()),
        "rmse_C": float(np.sqrt((err ** 2).mean())),
        "corr": float(df["sim"].corr(df["obs"])),
        "obs_min_C": float(df["obs"].min()),
        "obs_max_C": float(df["obs"].max()),
        "obs_amp_C": float(df["obs"].max() - df["obs"].min()),
        "sim_min_C": float(df["sim"].min()),
        "sim_max_C": float(df["sim"].max()),
        "sim_amp_C": float(df["sim"].max() - df["sim"].min()),
        "amp_error_C": float((df["sim"].max() - df["sim"].min()) - (df["obs"].max() - df["obs"].min())),
        "obs_peak_time": str(df["obs"].idxmax()),
        "sim_peak_time": str(df["sim"].idxmax()),
    }


def plot_one_case(tag, results, ni, weather, target_col, tin_col):
    sim = make_series(results, "T_s_in")
    obs = ni[target_col].sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)
    sim = sim.loc[common]
    obs = obs.loc[common]
    err = (sim - obs).dropna()

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(obs.index, obs.values, label=sensor_label(target_col), linewidth=2)
    axes[0].plot(sim.index, sim.values, label="Physics model T_s_in", linestyle="--", linewidth=2)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"CAM T1Ka-as-T_s_in target | T_in boundary = {sensor_label(tin_col) if tin_col else 'default geom.T_in_default'}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(err.index, err.values, linewidth=1.4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Model - measured (°C)")
    axes[1].grid(True, alpha=0.3)

    # Compare candidate target with selected T_in proxy and relevant sensors.
    sensor_cols = [target_col, INDOOR_AIR_PROXY, "T2A2", "T2A", "T1Ta2"]
    if tin_col is not None and tin_col not in sensor_cols:
        sensor_cols.append(tin_col)
    sensor_cols = [c for c in sensor_cols if c in ni.columns]
    for c in sensor_cols:
        s = ni[c].resample("1min").mean()
        axes[2].plot(s.index, s.values, label=sensor_label(c), linewidth=1.2)
    axes[2].set_ylabel("NI sensors (°C)")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    t = pd.to_datetime(results["datetime"])
    axes[3].plot(t, results["T_a"], label="T_a / outdoor air from weather", linewidth=1.3)
    axes[3].plot(t, results["T_in_used"], label=f"T_in_used ({sensor_label(tin_col) if tin_col else 'default'})", linewidth=1.3)
    axes[3].set_ylabel("T (°C)")
    axes[3].set_xlabel("Datetime")
    axes[3].grid(True, alpha=0.3)
    ax2 = axes[3].twinx()
    ax2.plot(t, results["G_sol"], label="G_sol", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("G_sol (W/m²)")
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = OUTPUT_DIR / f"validation_CAM_T1Ka_target__Tin_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path} | size={path.stat().st_size if path.exists() else 0} bytes")


def _safe_weather_series(weather, col):
    """Return a 1-minute weather series if the column exists."""
    if weather is None or col not in weather.columns:
        return None
    return weather[col].sort_index().resample("1min").mean()


def plot_overlay(all_results, ni, target_col, weather=None):
    """Overlay CAM T1Ka hypothesis cases and add weather drivers.

    The first panel keeps the original model-vs-measured comparison.
    The second and third panels show weather drivers so day-to-day differences
    in model behavior can be interpreted against solar radiation, ambient
    temperature, RH, and rain.
    """
    obs = ni[target_col].sort_index().resample("1min").mean()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.25, 1.0]}
    )

    ax = axes[0]
    ax.plot(obs.index, obs.values, label=sensor_label(target_col), linewidth=2.6)
    for tag, results in all_results.items():
        sim = make_series(results, "T_s_in")
        ax.plot(sim.index, sim.values, linestyle="--", linewidth=1.5, label=tag)
    ax.set_title("CAM overlay: T1Ka as T_s_in target, T1Ke as indoor-air T_in case + sensitivity cases")
    ax.set_ylabel("T_s,in / model (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Weather panel 1: ambient air and solar radiation.
    ax = axes[1]
    Ta = _safe_weather_series(weather, "T_a")
    G = _safe_weather_series(weather, "G_sol")
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

    # Weather panel 2: RH and rain/irrigation timing context.
    ax = axes[2]
    RH = _safe_weather_series(weather, "RH")
    rain = _safe_weather_series(weather, "rain")
    if RH is not None:
        ax.plot(RH.index, RH.values, label="RH", linewidth=1.3)
    ax.set_ylabel("RH (%)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if rain is not None:
        # Bar width in days: 1 minute = 1/1440 day. Use a slightly wider bar for visibility.
        ax2.bar(rain.index, rain.values, width=2/1440, alpha=0.25, label="Rain")
    ax2.set_ylabel("Rain (mm/min record)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Mark known irrigation windows as light vertical lines on all panels.
    try:
        start = obs.index.min().floor("D")
        end = obs.index.max().ceil("D")
        days = pd.date_range(start, end, freq="D")
        for day in days:
            for hhmm in ["07:00", "16:30"]:
                ts = pd.Timestamp(f"{day.date()} {hhmm}")
                if obs.index.min() <= ts <= obs.index.max():
                    for a in axes:
                        a.axvline(ts, linewidth=0.8, linestyle=":", alpha=0.35)
    except Exception:
        pass

    plt.tight_layout()
    path = OUTPUT_DIR / "overlay_CAM_T1Ka_target_all_Tin_proxies_with_weather.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved weather overlay: {path}")

    # Keep the original compact overlay too, for comparison.
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(obs.index, obs.values, label=sensor_label(target_col), linewidth=2.5)
    for tag, results in all_results.items():
        sim = make_series(results, "T_s_in")
        ax.plot(sim.index, sim.values, linestyle="--", linewidth=1.5, label=tag)
    ax.set_title("CAM overlay: T1Ka target with T1Ke indoor-air T_in case")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / "overlay_CAM_T1Ka_target_all_Tin_proxies.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compact overlay: {path}")


def plot_t1ke_indoor_air_diagnostic(ni):
    """Quick visual/statistical check for using T1Ke as indoor-air proxy."""
    if CAM_TARGET not in ni.columns or INDOOR_AIR_PROXY not in ni.columns:
        print(f"Skipping T1Ke diagnostic: need {CAM_TARGET} and {INDOOR_AIR_PROXY} in NI columns.")
        return

    s_roof = ni[CAM_TARGET].sort_index().resample("1min").mean()
    s_air = ni[INDOOR_AIR_PROXY].sort_index().resample("1min").mean()
    df = pd.concat([
        s_roof.rename("T1Ka_T_s_in"),
        s_air.rename("T1Ke_T_air_in_proxy"),
    ], axis=1).dropna()

    if df.empty:
        print("Skipping T1Ke diagnostic: no valid overlap between T1Ka and T1Ke.")
        return

    df["T1Ke_minus_T1Ka_C"] = df["T1Ke_T_air_in_proxy"] - df["T1Ka_T_s_in"]
    diagnostic_metrics = {
        "n": int(len(df)),
        "corr_T1Ka_T1Ke": float(df["T1Ka_T_s_in"].corr(df["T1Ke_T_air_in_proxy"])),
        "mean_T1Ke_minus_T1Ka_C": float(df["T1Ke_minus_T1Ka_C"].mean()),
        "median_T1Ke_minus_T1Ka_C": float(df["T1Ke_minus_T1Ka_C"].median()),
        "T1Ka_amp_C": float(df["T1Ka_T_s_in"].max() - df["T1Ka_T_s_in"].min()),
        "T1Ke_amp_C": float(df["T1Ke_T_air_in_proxy"].max() - df["T1Ke_T_air_in_proxy"].min()),
        "T1Ke_over_T1Ka_amp_ratio": float(
            (df["T1Ke_T_air_in_proxy"].max() - df["T1Ke_T_air_in_proxy"].min()) /
            (df["T1Ka_T_s_in"].max() - df["T1Ka_T_s_in"].min())
        ) if (df["T1Ka_T_s_in"].max() - df["T1Ka_T_s_in"].min()) != 0 else np.nan,
    }

    pd.Series(diagnostic_metrics).to_csv(OUTPUT_DIR / "diagnostic_T1Ke_as_indoor_air_metrics.csv")
    df.to_csv(OUTPUT_DIR / "diagnostic_T1Ka_T1Ke_timeseries.csv")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    axes[0].plot(df.index, df["T1Ka_T_s_in"], label=sensor_label(CAM_TARGET), linewidth=2)
    axes[0].plot(df.index, df["T1Ke_T_air_in_proxy"], label=sensor_label(INDOOR_AIR_PROXY), linewidth=2)
    axes[0].set_title("Diagnostic: T1Ke treated as indoor-air temperature proxy")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["T1Ke_minus_T1Ka_C"], linewidth=1.4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("T1Ke - T1Ka (°C)")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(df["T1Ka_T_s_in"], df["T1Ke_T_air_in_proxy"], s=8, alpha=0.45)
    axes[2].set_xlabel("T1Ka measured T_s_in / indoor roof surface (°C)")
    axes[2].set_ylabel("T1Ke tested T_air_in proxy (°C)")
    axes[2].set_title(f"Scatter check, corr = {diagnostic_metrics['corr_T1Ka_T1Ke']:.3f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "diagnostic_T1Ke_as_indoor_air_proxy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved T1Ke indoor-air diagnostic: {path}")


def main():
    print("\n=== CAM T1Ka target + T1Ke as indoor-air T_in test ===")
    print(f"Base dir : {BASE_DIR}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Target   : {CAM_TARGET} ({sensor_label(CAM_TARGET)})")
    print(f"T_in test: {INDOOR_AIR_PROXY} ({sensor_label(INDOOR_AIR_PROXY)})")

    weather, ni, rika, theta_initial = gr.prepare_validation_case("CAM", base_dir=str(BASE_DIR))

    if CAM_TARGET not in ni.columns:
        raise ValueError(f"Target {CAM_TARGET} not found. Available columns: {ni.columns.tolist()}")

    # Make explicit aliases for output/debugging only.
    ni = ni.copy()
    ni["T_s_in_CAM_hypothesis_T1Ka"] = ni["T1Ka"]
    if INDOOR_AIR_PROXY in ni.columns:
        ni["T_air_in_CAM_proxy_T1Ke"] = ni[INDOOR_AIR_PROXY]
    if "T1Kd" in ni.columns:
        ni["T_in_CAM_proxy_T1Kd"] = ni["T1Kd"]

    plot_t1ke_indoor_air_diagnostic(ni)

    # Initial condition from measured indoor roof surface target.
    T_s_in_initial = float(ni[CAM_TARGET].dropna().iloc[0])

    # Important: T1Ke is now being tested as indoor air temperature.
    # Therefore, do not also use T1Ke as substrate/top-soil initial temperature.
    # Leave this as None unless you later identify a verified substrate-top sensor.
    T_g_top_initial = None

    # Save raw sensor stats for this window.
    focus_cols = ["T1Ka", "T3Ka", "T2Ka", "T1Kd", "T1Kb", "T1Kc", "T2A2", "T2A", "T1Ta2", "T1Ke", "T1Tb", "T1Ta"]
    focus_cols = [c for c in focus_cols if c in ni.columns]
    stats = []
    for c in focus_cols:
        s = ni[c].dropna()
        stats.append({
            "channel": c,
            "role_in_this_runner": sensor_label(c),
            "min_C": float(s.min()),
            "mean_C": float(s.mean()),
            "max_C": float(s.max()),
            "amp_C": float(s.max() - s.min()),
            "peak_time": str(s.idxmax()),
        })
    pd.DataFrame(stats).to_csv(OUTPUT_DIR / "cam_window_sensor_stats.csv", index=False)

    all_results = {}
    metrics_rows = []

    for tag, tin_col in TIN_PROXY_CANDIDATES.items():
        print("\n" + "="*72)
        print(f"Running case: {tag} | T_in boundary = {sensor_label(tin_col) if tin_col else 'default geom.T_in_default'}")
        print("="*72)

        if tin_col is not None:
            if tin_col not in ni.columns:
                print(f"Skipping {tag}: column {tin_col} not found")
                continue
            T_in_series = ni[tin_col]
        else:
            T_in_series = None

        results = gr.run_simulation(
            weather_df=weather,
            plant=gr.bromelia,
            substrate=gr.substrat,
            slab=gr.slab,
            geom=gr.geom,
            num=gr.num,
            theta_initial=theta_initial,
            T_in_series=T_in_series,
            T_g_top_initial_C=T_g_top_initial,
            T_s_in_initial_C=T_s_in_initial,
            irrigation_windows=IRRIGATION_WINDOWS,
            irrigation_mm_per_min=IRRIGATION_MM_PER_MIN,
            save_every_s=60,
        )

        all_results[tag] = results
        sim = make_series(results, "T_s_in")
        obs = ni[CAM_TARGET]
        m = compute_metrics(sim, obs)
        m.update({
            "case": tag,
            "target_col": CAM_TARGET,
            "tin_proxy_col": tin_col if tin_col is not None else "geom.T_in_default",
            "tin_proxy_role": sensor_label(tin_col) if tin_col is not None else "default geom.T_in_default",
            "is_circular": bool(tin_col == CAM_TARGET),
            "Tf_min_C": float(np.nanmin(results["T_f"])),
            "Tf_max_C": float(np.nanmax(results["T_f"])),
            "Tin_used_min_C": float(np.nanmin(results["T_in_used"])),
            "Tin_used_max_C": float(np.nanmax(results["T_in_used"])),
        })
        metrics_rows.append(m)
        print(pd.Series(m).to_string())
        plot_one_case(tag, results, ni, weather, CAM_TARGET, tin_col)

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[["case", "target_col", "tin_proxy_col", "tin_proxy_role", "is_circular", "n", "bias_C", "mae_C", "rmse_C", "corr", "obs_amp_C", "sim_amp_C", "amp_error_C", "obs_peak_time", "sim_peak_time", "Tf_min_C", "Tf_max_C", "Tin_used_min_C", "Tin_used_max_C"]]
    metrics = metrics.sort_values(["is_circular", "rmse_C"])
    metrics.to_csv(OUTPUT_DIR / "cam_t1ka_target_tin_proxy_metrics.csv", index=False)
    print("\n=== METRICS SUMMARY ===")
    print(metrics.to_string(index=False))

    if all_results:
        plot_overlay(all_results, ni, CAM_TARGET, weather=weather)

    print("\nDONE")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    return all_results, metrics


if __name__ == "__main__":
    main()
