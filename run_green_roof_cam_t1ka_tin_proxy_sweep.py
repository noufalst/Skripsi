"""
CAM hypothesis simulation sweep:
- Treat T1Ka as candidate CAM indoor-roof temperature / T_s_in target.
- Do NOT use T1Ka as T_in boundary, to avoid circular validation.
- Sweep several CAM internal-temperature proxy candidates for T_in.

Run:
    py run_green_roof_cam_t1ka_tin_proxy_sweep.py

Expected files in:
    outputs_cam_t1ka_tin_proxy_sweep/

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
OUTPUT_DIR = SCRIPT_DIR / "outputs_cam_t1ka_tin_proxy_sweep"
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

# Main hypothesis
CAM_TARGET = "T1Ka"  # candidate CAM T_s_in based on T1Ka ~ T3Ka pattern

# T_in proxy candidates. These are not confirmed air sensors; they are tests.
# Keep the list short first because each simulation can take some time.
TIN_PROXY_CANDIDATES = {
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

    axes[0].plot(obs.index, obs.values, label=f"Measured candidate T_s_in ({target_col})", linewidth=2)
    axes[0].plot(sim.index, sim.values, label="Physics model T_s_in", linestyle="--", linewidth=2)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"CAM T1Ka-as-T_s_in hypothesis | T_in proxy = {tin_col if tin_col else 'default'}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(err.index, err.values, linewidth=1.4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Model - measured (°C)")
    axes[1].grid(True, alpha=0.3)

    # Compare candidate target with selected T_in proxy and relevant sensors.
    sensor_cols = [target_col, "T2A2", "T2A", "T1Ta2", "T1Ke"]
    if tin_col is not None and tin_col not in sensor_cols:
        sensor_cols.append(tin_col)
    sensor_cols = [c for c in sensor_cols if c in ni.columns]
    for c in sensor_cols:
        axes[2].plot(ni[c].resample("1min").mean().index, ni[c].resample("1min").mean().values, label=c, linewidth=1.2)
    axes[2].set_ylabel("NI sensors (°C)")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    t = pd.to_datetime(results["datetime"])
    axes[3].plot(t, results["T_a"], label="T_a", linewidth=1.3)
    axes[3].plot(t, results["T_in_used"], label="T_in_used", linewidth=1.3)
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


def plot_overlay(all_results, ni, target_col):
    obs = ni[target_col].sort_index().resample("1min").mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(obs.index, obs.values, label=f"Measured target {target_col}", linewidth=2.5)
    for tag, results in all_results.items():
        sim = make_series(results, "T_s_in")
        ax.plot(sim.index, sim.values, linestyle="--", linewidth=1.5, label=tag)
    ax.set_title("CAM hypothesis overlay: T1Ka as T_s_in target, different T_in proxies")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / "overlay_CAM_T1Ka_target_all_Tin_proxies.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {path}")


def main():
    print("\n=== CAM T1Ka-as-T_s_in hypothesis simulation ===")
    print(f"Base dir : {BASE_DIR}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Target   : {CAM_TARGET}")

    weather, ni, rika, theta_initial = gr.prepare_validation_case("CAM", base_dir=str(BASE_DIR))

    if CAM_TARGET not in ni.columns:
        raise ValueError(f"Target {CAM_TARGET} not found. Available columns: {ni.columns.tolist()}")

    # Make explicit aliases for output/debugging only.
    ni = ni.copy()
    ni["T_s_in_CAM_hypothesis_T1Ka"] = ni["T1Ka"]
    if "T1Kd" in ni.columns:
        ni["T_in_CAM_proxy_T1Kd"] = ni["T1Kd"]

    # Initial conditions from measured candidate target and top substrate.
    T_s_in_initial = float(ni[CAM_TARGET].dropna().iloc[0])
    T_g_top_initial = float(ni["T1Ke"].dropna().iloc[0]) if "T1Ke" in ni and not ni["T1Ke"].dropna().empty else None

    # Save raw sensor stats for this window.
    focus_cols = ["T1Ka", "T3Ka", "T2Ka", "T1Kd", "T1Kb", "T1Kc", "T2A2", "T2A", "T1Ta2", "T1Ke", "T1Tb", "T1Ta"]
    focus_cols = [c for c in focus_cols if c in ni.columns]
    stats = []
    for c in focus_cols:
        s = ni[c].dropna()
        stats.append({
            "channel": c,
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
        print(f"Running case: {tag} | T_in proxy = {tin_col if tin_col else 'default geom.T_in_default'}")
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
    metrics = metrics[["case", "target_col", "tin_proxy_col", "is_circular", "n", "bias_C", "mae_C", "rmse_C", "corr", "obs_amp_C", "sim_amp_C", "amp_error_C", "obs_peak_time", "sim_peak_time", "Tf_min_C", "Tf_max_C", "Tin_used_min_C", "Tin_used_max_C"]]
    metrics = metrics.sort_values(["is_circular", "rmse_C"])
    metrics.to_csv(OUTPUT_DIR / "cam_t1ka_target_tin_proxy_metrics.csv", index=False)
    print("\n=== METRICS SUMMARY ===")
    print(metrics.to_string(index=False))

    if all_results:
        plot_overlay(all_results, ni, CAM_TARGET)

    print("\nDONE")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    return all_results, metrics


if __name__ == "__main__":
    main()
