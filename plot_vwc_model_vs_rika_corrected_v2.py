
"""
================================================================================
VWC MODEL vs RIKA — v2 MOISTURE LIMITER
================================================================================

What is fixed?
--------------
The previous free-running moisture model could let surface theta_top collapse
to theta_min (= 0.05) unrealistically. This v2 adds a conservative limiter to the
surface evaporation sink so the model cannot remove more water from the top cell
than is physically available in one timestep.

This still allows sudden upward spikes after rain/irrigation input, but prevents
unphysical instant drying to theta_min.

Important:
    - This is not RK520 calibration.
    - Corrected RIKA = raw RIKA × factor remains a sensitivity assumption.
    - The limiter only prevents numerical/physical over-extraction of water.

Usage:
    python plot_vwc_model_vs_rika_corrected_v2.py --plant CAM --factor 0.70

Optional:
    python plot_vwc_model_vs_rika_corrected_v2.py --plant CAM --factor 0.70 --mode free_limited

    python plot_vwc_model_vs_rika_corrected_v2.py --plant CAM --factor 0.70 --mode rika_nudged

Modes:
    free_limited:
        model evolves freely, but surface evaporation is limited by available water.
        Recommended first.

    rika_nudged:
        same as free_limited, then weakly relaxes model theta toward corrected RIKA
        with a user-set time constant. This is diagnostic only, not validation.

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
# SECTION 2 — PATCHED MOISTURE SOLVER
# ==============================================================================

def patch_moisture_solver_with_surface_limiter(
    safety: float = 0.85,
    evap_scale: float = 1.0,
) -> None:
    """
    Monkey-patch gr.solve_substrate_moisture with a safer version.

    Reason:
        Original surface water balance can over-extract the top control volume,
        causing theta_top to hit theta_min very quickly.

    Fix:
        Limit evaporation sink using available water in top control volume:
            available_depth = max(theta_top - theta_min, 0) * dz/2
            max_sink_ms = safety * available_depth / dt

        Then:
            j_eva_ms_limited = min(j_eva_ms, j_irrig_ms + max_sink_ms)

    Notes:
        - dz/2 is used because boundary top node is a half control volume.
        - rainfall/irrigation can still create sudden upward spikes.
        - this does not force model to match RIKA.
    """

    def solve_substrate_moisture_limited(theta: np.ndarray,
                                         T_f: float,
                                         T_g_surface: float,
                                         T_a_K: float,
                                         RH: float,
                                         u: float,
                                         G_sol: float,
                                         j_irrigation: float,
                                         plant: gr.PlantParameters,
                                         substrate: gr.SubstrateParameters,
                                         H_g: float,
                                         dt: float):
        gamma = gr.psychrometric_constant()
        rho_air = 1.2
        cp_air = 1005.0

        theta = np.asarray(theta, dtype=float)
        Nz = len(theta)
        dz = H_g / (Nz - 1)

        _, _, k_theta, D_theta = gr.compute_substrate_properties(theta, substrate)

        P_a = gr.ambient_vapor_pressure(T_a_K, RH)
        P_f_sat = gr.saturation_pressure(T_f)
        P_g_sat = gr.saturation_pressure(T_g_surface)

        r_a = gr.compute_aerodynamic_resistance(u, plant)
        r_stoma = gr.compute_stomatal_resistance(
            G_sol,
            T_f,
            P_f_sat,
            P_a,
            float(np.mean(theta)),
            plant,
            substrate,
        )
        r_vap = gr.substrate_vapor_resistance(float(theta[0]), substrate)

        h_eva_f = plant.LAI / gamma * (rho_air * cp_air) / (r_a + r_stoma)
        h_eva_g = plant.LAI / gamma * (rho_air * cp_air) / (r_a + r_vap)

        j_eva_f = max(0.0, h_eva_f * (P_f_sat - P_a) / substrate.l_fg)
        j_eva_g = max(0.0, h_eva_g * (P_g_sat - P_a) / substrate.l_fg)

        j_eva = evap_scale * (j_eva_f + j_eva_g)

        rho_water = 1000.0
        j_eva_ms_raw = j_eva / rho_water
        j_irrig_ms = max(float(j_irrigation), 0.0) / rho_water

        # ------------------------------------------------------------------
        # NEW LIMITER: do not let top node lose more water than available.
        # ------------------------------------------------------------------
        theta_top_excess = max(float(theta[0]) - float(substrate.theta_min), 0.0)
        available_water_depth_m = theta_top_excess * (dz / 2.0)
        max_net_sink_ms = safety * available_water_depth_m / max(float(dt), 1e-12)

        # rainfall/irrigation can offset evaporation
        j_eva_ms_limited = min(j_eva_ms_raw, j_irrig_ms + max_net_sink_ms)

        # return the actually used evapotranspiration flux in kg/m2/s
        j_eva_used = j_eva_ms_limited * rho_water

        aw = np.zeros(Nz)
        ap = np.zeros(Nz)
        ae = np.zeros(Nz)
        bv = np.zeros(Nz)

        for i in range(1, Nz - 1):
            Dw = 0.5 * (D_theta[i - 1] + D_theta[i])
            De = 0.5 * (D_theta[i] + D_theta[i + 1])
            kw = 0.5 * (k_theta[i - 1] + k_theta[i])
            ke = 0.5 * (k_theta[i] + k_theta[i + 1])

            aw[i] = Dw / dz**2
            ae[i] = De / dz**2
            ap[i] = 1.0 / dt + aw[i] + ae[i]
            bv[i] = theta[i] / dt + (ke - kw) / dz

        # Top boundary: positive j_net adds water, negative dries.
        j_net = j_irrig_ms - j_eva_ms_limited

        ae[0] = D_theta[0] / dz**2
        ap[0] = 1.0 / dt + ae[0]
        bv[0] = theta[0] / dt + (j_net - k_theta[0]) / dz

        # Bottom boundary: free drainage / unit-gradient drainage.
        aw[-1] = D_theta[-1] / dz**2
        ap[-1] = 1.0 / dt + aw[-1]
        bv[-1] = theta[-1] / dt - k_theta[-1] / dz

        theta_new = gr.tdma_solver(-aw, ap, -ae, bv)
        theta_new = np.clip(theta_new, substrate.theta_min, substrate.theta_sat)

        return theta_new, float(j_eva_used)

    gr.solve_substrate_moisture = solve_substrate_moisture_limited


# ==============================================================================
# SECTION 3 — OPTIONAL RIKA NUDGING
# ==============================================================================

def make_rika_profile_series(
    rika_corr: pd.DataFrame,
    H_g: float,
    Nz: int,
    substrate,
) -> pd.DataFrame:
    """
    Interpolate corrected RIKA 2cm/7cm into model depth grid over time.

    Used only for optional diagnostic nudging.
    """
    z = np.linspace(0, H_g, Nz)
    r = rika_corr[["theta_2cm", "theta_7cm"]].copy()
    r = r.resample("1min").mean().interpolate("time", limit=15, limit_direction="both")

    rows = []
    for _, row in r.iterrows():
        if pd.isna(row["theta_2cm"]) or pd.isna(row["theta_7cm"]):
            rows.append(np.full(Nz, np.nan))
        else:
            prof = np.interp(
                z,
                [0.02, 0.07],
                [float(row["theta_2cm"]), float(row["theta_7cm"])],
                left=float(row["theta_2cm"]),
                right=float(row["theta_7cm"]),
            )
            prof = np.clip(prof, substrate.theta_min, substrate.theta_sat)
            rows.append(prof)

    out = pd.DataFrame(rows, index=r.index, columns=[f"z{i}" for i in range(Nz)])
    return out


def run_simulation_with_rika_nudging(
    weather_df: pd.DataFrame,
    plant,
    substrate,
    slab,
    geom,
    num,
    theta_initial,
    T_in_series,
    T_g_top_initial_C,
    T_s_in_initial_C,
    rika_corr: pd.DataFrame,
    tau_hours: float = 3.0,
    save_every_s: int = 60,
) -> dict:
    """
    Same structure as gr.run_simulation, but after solving moisture,
    theta is weakly relaxed toward corrected RIKA.

    This is diagnostic only. It uses measured RIKA as information during the run,
    so do not call it independent moisture validation.
    """
    print(f"\n{'='*60}")
    print(f"Simulasi with RIKA nudging: {plant.name} ({plant.plant_type})")
    print(f"nudging tau = {tau_hours:.2f} hours")
    print(f"{'='*60}")

    Nz_g = num.Nz_substrate
    Nz_s = num.Nz_slab
    dt = num.dt

    weather_1s = gr.prepare_weather_1s(weather_df, dt=dt)
    N_steps = len(weather_1s)

    T_in_1s = None
    if T_in_series is not None:
        T_in_1s = (
            T_in_series.sort_index()
            .resample(f"{int(dt)}s").interpolate("time")
            .reindex(weather_1s.index).interpolate("time")
        )

    rika_profile_1min = make_rika_profile_series(rika_corr, geom.H_g, Nz_g, substrate)

    T_a_init = float(weather_1s["T_a"].iloc[0]) + 273.15
    T_g_top_init = T_a_init if T_g_top_initial_C is None else T_g_top_initial_C + 273.15
    T_s_in_init = T_a_init if T_s_in_initial_C is None else T_s_in_initial_C + 273.15

    T_g = np.linspace(T_g_top_init, T_a_init, Nz_g)
    T_s = np.linspace(T_a_init, T_s_in_init, Nz_s)
    T_f = T_a_init
    theta = gr.make_theta_initial_profile(theta_initial, geom.H_g, Nz_g, substrate)

    alpha = 1.0 - np.exp(-dt / (tau_hours * 3600.0))

    results = {
        "datetime": [],
        "time": [],
        "T_f": [],
        "T_g_top": [],
        "T_g_mid": [],
        "T_g_bot": [],
        "T_s_in": [],
        "theta_top": [],
        "theta_mid": [],
        "theta_bot": [],
        "q_s_in": [],
        "j_eva": [],
        "T_a": [],
        "G_sol": [],
        "T_in_used": [],
        "j_pr": [],
        "theta_mean": [],
    }

    for step, (ts, row) in enumerate(weather_1s.iterrows()):
        T_a_K = float(row["T_a"]) + 273.15
        G_sol = max(float(row["G_sol"]), 0.0)
        RH = float(np.clip(row["RH"], 1, 99))
        u = max(float(row["u"]), 0.1)
        j_pr = max(float(row.get("rain_flux", 0.0)), 0.0)

        if T_in_1s is not None and ts in T_in_1s.index and not pd.isna(T_in_1s.loc[ts]):
            T_in_current = float(T_in_1s.loc[ts]) + 273.15
        else:
            T_in_current = geom.T_in_default

        theta_avg = float(np.mean(theta))

        T_f = gr.solve_foliage_temperature(
            T_f_prev=T_f,
            T_a_K=T_a_K,
            T_g_surface=T_g[0],
            G_sol=G_sol,
            RH=RH,
            u=u,
            theta_avg=theta_avg,
            plant=plant,
            substrate=substrate,
            dt=dt,
        )

        T_g = gr.solve_substrate_heat(
            T_g=T_g,
            T_f=T_f,
            T_slab_top=T_s[0],
            G_sol=G_sol,
            T_a_K=T_a_K,
            RH=RH,
            u=u,
            theta=theta,
            plant=plant,
            substrate=substrate,
            H_g=geom.H_g,
            dt=dt,
        )

        theta, j_eva = gr.solve_substrate_moisture(
            theta=theta,
            T_f=T_f,
            T_g_surface=T_g[0],
            T_a_K=T_a_K,
            RH=RH,
            u=u,
            G_sol=G_sol,
            j_irrigation=j_pr,
            plant=plant,
            substrate=substrate,
            H_g=geom.H_g,
            dt=dt,
        )

        # Weak nudging every timestep using nearest 1-min corrected RIKA profile.
        ts_min = ts.floor("min")
        if ts_min in rika_profile_1min.index:
            prof = rika_profile_1min.loc[ts_min].to_numpy(dtype=float)
            if np.isfinite(prof).all():
                theta = (1.0 - alpha) * theta + alpha * prof
                theta = np.clip(theta, substrate.theta_min, substrate.theta_sat)

        _, lambda_g, _, _ = gr.compute_substrate_properties(theta, substrate)
        T_s, q_s_in = gr.solve_slab_heat(
            T_s=T_s,
            T_g_bottom=T_g[-1],
            lambda_g_bottom=lambda_g[-1],
            slab=slab,
            geom=geom,
            dt=dt,
            T_in_K=T_in_current,
        )

        if step % int(save_every_s / dt) == 0:
            results["datetime"].append(ts)
            results["time"].append(step * dt)
            results["T_f"].append(T_f - 273.15)
            results["T_g_top"].append(T_g[0] - 273.15)
            results["T_g_mid"].append(T_g[Nz_g // 2] - 273.15)
            results["T_g_bot"].append(T_g[-1] - 273.15)
            results["T_s_in"].append(T_s[-1] - 273.15)
            results["theta_top"].append(float(theta[0]))
            results["theta_mid"].append(float(theta[Nz_g // 2]))
            results["theta_bot"].append(float(theta[-1]))
            results["q_s_in"].append(float(q_s_in))
            results["j_eva"].append(float(j_eva))
            results["T_a"].append(T_a_K - 273.15)
            results["G_sol"].append(G_sol)
            results["T_in_used"].append(T_in_current - 273.15)
            results["j_pr"].append(float(j_pr))
            results["theta_mean"].append(float(np.mean(theta)))

    results["Q_gain"] = float(np.trapz(results["q_s_in"], dx=save_every_s))
    return results


# ==============================================================================
# SECTION 4 — RUN MODEL WITH CORRECTED INITIAL VWC
# ==============================================================================

def run_model_with_corrected_vwc(
    plant_type: str,
    base_dir: str,
    correction_factor: float,
    cam_target: str | None,
    mode: str,
    limiter_safety: float,
    evap_scale: float,
    nudging_tau_h: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load weather/NI/RIKA, correct RIKA VWC, then run model."""
    plant_type = plant_type.upper()
    if plant_type not in ["CAM", "C3"]:
        raise ValueError("plant must be CAM or C3")

    apply_project_defaults()

    if cam_target:
        gr.VALIDATION_TARGETS["CAM"] = cam_target

    patch_moisture_solver_with_surface_limiter(
        safety=limiter_safety,
        evap_scale=evap_scale,
    )

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

    if mode == "free_limited":
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
    elif mode == "rika_nudged":
        results = run_simulation_with_rika_nudging(
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
            rika_corr=rika_corr,
            tau_hours=nudging_tau_h,
        )
    else:
        raise ValueError("mode must be free_limited or rika_nudged")

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
        "mode": mode,
        "limiter_safety": limiter_safety,
        "evap_scale": evap_scale,
        "nudging_tau_h": nudging_tau_h,
    }

    return results, weather, ni, rika_raw, rika_corr, meta


# ==============================================================================
# SECTION 5 — PLOTTING AND EXPORT
# ==============================================================================

def results_theta_dataframe(results: dict) -> pd.DataFrame:
    t = pd.to_datetime(results["datetime"])
    df = pd.DataFrame(
        {
            "model_theta_top": results.get("theta_top", np.nan),
            "model_theta_mid": results.get("theta_mid", np.nan),
            "model_theta_bot": results.get("theta_bot", np.nan),
            "model_theta_mean": results.get("theta_mean", np.nan),
            "j_eva": results.get("j_eva", np.nan),
            "j_pr": results.get("j_pr", np.nan),
            "G_sol": results.get("G_sol", np.nan),
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
    lines.append("VWC MODEL vs RIKA SUMMARY — v2 MOISTURE LIMITER")
    lines.append("=" * 70)
    for k, v in meta.items():
        lines.append(f"{k:20s}: {v}")
    lines.append("")

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        lines.append(
            f"{col:28s}: min={s.min():.4f}, mean={s.mean():.4f}, max={s.max():.4f}, n={len(s)}"
        )

    lines.append("")
    lines.append("Interpretation note:")
    lines.append("- Corrected RIKA = raw RIKA × factor, not calibrated VWC.")
    lines.append("- free_limited mode prevents unphysical instant drying to theta_min.")
    lines.append("- rika_nudged mode uses measured RIKA during simulation, so it is diagnostic, not independent validation.")
    return "\n".join(lines)


def plot_vwc(
    df: pd.DataFrame,
    plant_type: str,
    factor: float,
    out_png: Path,
    meta: dict,
    show_raw: bool = True,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Panel 1: corrected RIKA vs model top/bottom
    if "model_theta_top" in df:
        axes[0].plot(df.index, df["model_theta_top"], label="Model θ_top", linewidth=2)
    if "model_theta_bot" in df:
        axes[0].plot(df.index, df["model_theta_bot"], label="Model θ_bot", linewidth=2)
    if "rika_corrected_theta_2cm" in df:
        axes[0].plot(df.index, df["rika_corrected_theta_2cm"], "--", label=f"RIKA θ_2cm corrected ×{factor}", linewidth=1.6)
    if "rika_corrected_theta_7cm" in df:
        axes[0].plot(df.index, df["rika_corrected_theta_7cm"], "--", label=f"RIKA θ_7cm corrected ×{factor}", linewidth=1.6)

    axes[0].set_title(
        f"{plant_type} VWC comparison: model vs corrected RIKA | "
        f"mode={meta.get('mode')} | limiter safety={meta.get('limiter_safety')}"
    )
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
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    # Panel 4: forcing context
    if "j_eva" in df:
        axes[3].plot(df.index, df["j_eva"] * 3600, label="Model ET (kg m⁻² h⁻¹)")
    if "j_pr" in df:
        axes[3].plot(df.index, df["j_pr"] * 3600, label="Rain/input flux (kg m⁻² h⁻¹)")
    if "G_sol" in df:
        ax2 = axes[3].twinx()
        ax2.plot(df.index, df["G_sol"], "--", label="G_sol", alpha=0.6)
        ax2.set_ylabel("G_sol (W/m²)")
        l1, lab1 = axes[3].get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        axes[3].legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
    else:
        axes[3].legend(fontsize=8)

    axes[3].set_title("Water/solar forcing context")
    axes[3].set_ylabel("Water flux")
    axes[3].set_xlabel("Datetime")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# SECTION 6 — MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot model VWC vs corrected RIKA VWC with moisture limiter.")
    parser.add_argument("--plant", choices=["CAM", "C3"], default="CAM")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--factor", type=float, default=0.70,
                        help="Correction factor. 0.70 means reduce raw RK520 by 30%.")
    parser.add_argument("--cam-target", default=None,
                        help="Optional CAM target override, e.g. T1Tb or T2A2.")
    parser.add_argument("--mode", choices=["free_limited", "rika_nudged"], default="free_limited")
    parser.add_argument("--limiter-safety", type=float, default=0.85,
                        help="Available-water sink limiter safety. Lower = more conservative drying.")
    parser.add_argument("--evap-scale", type=float, default=1.0,
                        help="Optional scale for model ET sink. Use 1.0 normally.")
    parser.add_argument("--nudging-tau-h", type=float, default=3.0,
                        help="Nudging time constant in hours for rika_nudged mode.")
    parser.add_argument("--no-model", action="store_true",
                        help="Only plot raw/corrected RIKA; do not rerun thermal model.")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    plant_type = args.plant.upper()
    factor_text = f"{args.factor:.2f}".replace(".", "p")
    out_prefix = args.out_prefix or f"vwc_comparison_{plant_type}_{args.mode}_factor_{factor_text}"

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
            "mode": "no_model",
            "limiter_safety": np.nan,
            "evap_scale": np.nan,
            "nudging_tau_h": np.nan,
        }
    else:
        results, weather, ni, rika_raw, rika_corr, meta = run_model_with_corrected_vwc(
            plant_type=plant_type,
            base_dir=args.base_dir,
            correction_factor=args.factor,
            cam_target=args.cam_target,
            mode=args.mode,
            limiter_safety=args.limiter_safety,
            evap_scale=args.evap_scale,
            nudging_tau_h=args.nudging_tau_h,
        )
        model_df = results_theta_dataframe(results)

    comp = make_vwc_comparison_dataframe(model_df, rika_raw, rika_corr)

    out_png = Path(f"{out_prefix}.png")
    out_csv = Path(f"{out_prefix}.csv")
    out_txt = Path(f"{out_prefix}_summary.txt")

    plot_vwc(comp, plant_type, args.factor, out_png, meta=meta)
    comp.to_csv(out_csv, index_label="datetime")
    out_txt.write_text(summarize_vwc(comp, meta), encoding="utf-8")

    print("\nDone.")
    print(f"Saved plot   : {out_png}")
    print(f"Saved data   : {out_csv}")
    print(f"Saved summary: {out_txt}")
    print("\nReminder:")
    print("  Corrected RIKA = raw RIKA × factor")
    print("  free_limited prevents impossible instant drying to theta_min.")
    print("  rika_nudged is diagnostic only, not independent validation.")


if __name__ == "__main__":
    main()
