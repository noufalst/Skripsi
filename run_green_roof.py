import new_baru_revised_same_structure_v2 as gr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_validation_single(results, ni, plant_type, save_path=None):
    """
    Plot model vs data asli NI untuk validasi T_s,in.
    Sensor:
    CAM = T1Tb / T_s_in_CAM
    C3  = T3Ka / T_s_in_C3
    """
    plant_type = plant_type.upper()

    if plant_type == "CAM":
        target_col = "T_s_in_CAM"
        title = "CAM / Bromelia"
    elif plant_type == "C3":
        target_col = "T_s_in_C3"
        title = "C3 / Wedelia"
    else:
        raise ValueError("plant_type harus 'CAM' atau 'C3'")

    sim = pd.Series(
        results["T_s_in"],
        index=pd.to_datetime(results["datetime"]),
        name="Model"
    ).sort_index().resample("1min").mean()

    obs = ni[target_col].sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)

    sim_common = sim.loc[common]
    obs_common = obs.loc[common]
    err = sim_common - obs_common

    bias = err.mean()
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    amp_measured = obs_common.max() - obs_common.min()
    amp_model = sim_common.max() - sim_common.min()
    amp_error = amp_model - amp_measured

    peak_error = sim_common.max() - obs_common.max()
    min_error = sim_common.min() - obs_common.min()

    print(f"\nAmplitude check {plant_type}:")
    print(f"  Measured amplitude : {amp_measured:.2f} °C")
    print(f"  Model amplitude    : {amp_model:.2f} °C")
    print(f"  Amplitude error    : {amp_error:.2f} °C")
    print(f"  Peak error         : {peak_error:.2f} °C")
    print(f"  Min error          : {min_error:.2f} °C")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: Model vs measured
    axes[0].plot(obs_common.index, obs_common.values, label="Measured NI", linewidth=2)
    axes[0].plot(sim_common.index, sim_common.values, label="Model", linestyle="--", linewidth=2)
    axes[0].set_ylabel("T_s,in (°C)")
    axes[0].set_title(
        f"Validation {title}: Inner Roof Surface Temperature\n"
        f"Bias={bias:.2f}°C | MAE={mae:.2f}°C | RMSE={rmse:.2f}°C | AmpErr={amp_error:.2f}°C | n={len(common)}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Residual
    axes[1].plot(err.index, err.values, linewidth=1.5)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error (°C)")
    axes[1].set_title("Residual: Model - Measured")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Weather driver
    axes[2].plot(pd.to_datetime(results["datetime"]), results["T_a"], label="T_a", linewidth=1.5)
    axes[2].set_ylabel("T_a (°C)")
    axes[2].set_xlabel("Datetime")
    axes[2].grid(True, alpha=0.3)

    ax2 = axes[2].twinx()
    ax2.plot(pd.to_datetime(results["datetime"]), results["G_sol"], label="G_sol", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("G_sol (W/m²)")

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()
# =========================
# PARAMETER SEMENTARA
# =========================
gr.apply_scientific_guess_parameters(
    rho_g=400.0,
    H_slab=0.10,
    H_g=0.10,
    theta_sat=0.90,
    k_theta_sat=5e-6,
    lambda_dry=0.12
)

gr.geom.A_roof = 1.0

# cover fraction awal
gr.bromelia.cover_fraction = 0.7
gr.wedelia.cover_fraction = 0.4
# Untuk validasi CAM ke T1Tb
# gr.VALIDATION_TARGETS["CAM"] = "T1Tb"

# Untuk validasi CAM ke T2A2
gr.VALIDATION_TARGETS["CAM"] = "T2A2"

# Scientific dynamic indoor convection for underside ceiling
gr.geom.dynamic_h_in = False  # Kalau True, h_in dihitung dinamis dari suhu interior T_s[-1] dan T_in
# =========================
# RINGKAS WEATHER
# =========================
print("\n=== WEATHER SUMMARY ===")
summary = gr.summarize_weather_windows(base_dir=".")
print(summary)

summary.to_excel("weather_summary.xlsx", index=False)
print("\nWeather summary saved to weather_summary.xlsx")

# # ============================================================
# # DEBUG STEP 1 — CHECK CAM SENSOR CHANNELS
# # ============================================================

# print("\n=== DEBUG: CHECK CAM SENSOR CHANNELS ===")

# # Pastikan window CAM benar
# gr.VALIDATION_WINDOWS["CAM"] = (
#     "2026-03-31 11:58:00",
#     "2026-04-02 21:42:00"
# )

# # Ambil data NI pada window CAM
# _, ni_cam_check, _, _ = gr.prepare_validation_case("CAM", base_dir=".")

# # Pilih channel yang mau dicek
# cam_cols = [
#     "T1Ta",       # kemungkinan atap/surface tertentu CAM
#     "T1Tb",       # target sementara: atap indoor CAM
#     "T1Ka",       # interior/air CAM
#     "T1Ke",       # soil/top CAM
#     "T2A",        # soil/bottom CAM kalau ada
#     "T2Ka",       # reference roof / control
#     "T2A2",       # coba aslinya bukan ini
# ]

# print("dynamic_h_in    =", getattr(gr.geom, "dynamic_h_in", False))
# print("h_in constant   =", gr.geom.h_in)

# # Ambil hanya kolom yang memang ada di file
# cam_cols = [c for c in cam_cols if c in ni_cam_check.columns]

# fig, ax = plt.subplots(figsize=(14, 7))

# for col in cam_cols:
#     ax.plot(
#         ni_cam_check.index,
#         ni_cam_check[col],
#         label=col,
#         linewidth=1.5
#     )

# ax.set_title("CAM Sensor Channel Check — NI Data")
# ax.set_ylabel("Temperature (°C)")
# ax.set_xlabel("Datetime")
# ax.grid(True, alpha=0.3)
# ax.legend()

# plt.tight_layout()
# plt.savefig("debug_CAM_all_channels.png", dpi=200, bbox_inches="tight")
# plt.close(fig)

# print("Saved plot: debug_CAM_all_channels.png")

# # =========================
# # RUN CAM
# # =========================
# print("\n=== RUNNING CAM ===")
# res_cam, metrics_cam = gr.run_validation_case(
#     plant_type="CAM",
#     base_dir=".",
#     calibrate_lai=False
# )

# print("\nCAM metrics:")
# print(metrics_cam)

# # debug
# def plot_cam_transfer_path(results, ni_cam, save_path="debug_CAM_transfer_path.png"):
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     t = pd.to_datetime(results["datetime"])

#     fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

#     # 1. Target inner roof
#     axes[0].plot(t, results["T_s_in"], label="Model T_s,in", linewidth=2)
#     if "T1Tb" in ni_cam.columns:
#         axes[0].plot(ni_cam.index, ni_cam["T1Tb"], "--", label="Measured T1Tb", linewidth=1.4)
#     if "T2A2" in ni_cam.columns:
#         axes[0].plot(ni_cam.index, ni_cam["T2A2"], "--", label="Measured T2A2", linewidth=1.4)
#     axes[0].set_ylabel("Inner temp (°C)")
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     # 2. Top/substrate-side temperatures
#     if "T_g_surface" in results:
#         axes[1].plot(t, results["T_g_surface"], label="Model T_g_surface", linewidth=2)
#     if "T1Ta" in ni_cam.columns:
#         axes[1].plot(ni_cam.index, ni_cam["T1Ta"], "--", label="Measured T1Ta", linewidth=1.4)
#     if "T1Ke" in ni_cam.columns:
#         axes[1].plot(ni_cam.index, ni_cam["T1Ke"], "--", label="Measured T1Ke", linewidth=1.4)
#     axes[1].set_ylabel("Top/substrate (°C)")
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)

#     # 3. Moisture state
#     for col in ["theta_top", "theta_mean", "theta_bot"]:
#         if col in results:
#             axes[2].plot(t, results[col], label=col)
#     axes[2].set_ylabel("theta (-)")
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)

#     # 4. Solar forcing
#     axes[3].plot(t, results["G_sol"], label="G_sol")
#     axes[3].set_ylabel("Solar (W/m²)")
#     axes[3].set_xlabel("Datetime")
#     axes[3].legend()
#     axes[3].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Saved plot: {save_path}")

# # Load NI CAM untuk plot actual vs model
# _, ni_cam, _, _ = gr.prepare_validation_case("CAM", base_dir=".")

# plot_cam_transfer_path(
#     res_cam,
#     ni_cam,
#     save_path="debug_CAM_transfer_path.png"
# )

# # Hitung amplitude dan phase secara saintifik untuk CAM
# def harmonic_daily(series, name="signal"):
#     import numpy as np
#     import pandas as pd

#     s = series.dropna().sort_index().resample("1min").mean().dropna()

#     if len(s) < 60:
#         return None

#     t = (s.index - s.index[0]).total_seconds().to_numpy()
#     y = s.to_numpy()

#     omega = 2 * np.pi / (24 * 3600)

#     X = np.column_stack([
#         np.ones_like(t),
#         np.sin(omega * t),
#         np.cos(omega * t)
#     ])

#     coef, *_ = np.linalg.lstsq(X, y, rcond=None)
#     offset, a, b = coef

#     amp = np.sqrt(a**2 + b**2)
#     phase = np.arctan2(b, a)

#     # peak time relative to start
#     t_peak_s = ((np.pi / 2 - phase) % (2 * np.pi)) / omega
#     t_peak_h = t_peak_s / 3600

#     return {
#         "name": name,
#         "mean": float(offset),
#         "amplitude_C": float(amp),
#         "peak_hour_from_start": float(t_peak_h)
#     }


# def print_cam_harmonic_diagnostics(results, ni_cam):
#     import pandas as pd

#     t = pd.to_datetime(results["datetime"])

#     signals = []

#     signals.append(harmonic_daily(
#         pd.Series(results["T_s_in"], index=t),
#         "Model T_s,in"
#     ))

#     if "T_g_surface" in results:
#         signals.append(harmonic_daily(
#             pd.Series(results["T_g_surface"], index=t),
#             "Model T_g_surface"
#         ))

#     for col in ["T1Ta", "T1Ke", "T1Tb", "T2A2", "T1Ka"]:
#         if col in ni_cam.columns:
#             signals.append(harmonic_daily(ni_cam[col], f"Measured {col}"))

#     print("\n=== CAM HARMONIC DAILY DIAGNOSTIC ===")
#     for item in signals:
#         if item is None:
#             continue
#         print(
#             f"{item['name']:20s} | "
#             f"mean={item['mean']:.2f}°C | "
#             f"amp={item['amplitude_C']:.2f}°C | "
#             f"peak_hour={item['peak_hour_from_start']:.2f} h"
#         )

# t = pd.to_datetime(res_cam["datetime"])
# j_pr = pd.Series(res_cam["j_pr"], index=t)
# j_eva = pd.Series(res_cam["j_eva"], index=t)
# theta_mean = pd.Series(res_cam["theta_mean"], index=t)

# print("\n=== CAM WATER BALANCE CHECK ===")
# print("theta_mean min/max:", theta_mean.min(), theta_mean.max())
# print("total input water j_pr [mm]:", j_pr.sum() * 60 / 1000 if False else "check unit")
# print("j_pr max:", j_pr.max())
# print("j_eva max:", j_eva.max())
# print_cam_harmonic_diagnostics(res_cam, ni_cam)

# dt_save = 60  # karena save_every_s=60
# total_input_mm = (j_pr * dt_save).sum()  # kg/m2 = mm water
# total_evap_mm = (j_eva * dt_save).sum()  # kg/m2 = mm water

# print("Total input water:", total_input_mm, "mm")
# print("Total evap water :", total_evap_mm, "mm")

# def forced_slab_from_measured_top(
#     ni,
#     top_col="T1Ta",
#     target_col="T1Tb",
#     T_in_col="T1Ka",
#     H_slab=0.10,
#     lambda_s=1.74,
#     rho_s=2300.0,
#     cp_s=840.0,
#     h_in=8.0,
#     Nz=67,
#     dt_s=60,
#     save_path="forced_slab_CAM_T1Ta_to_T1Tb.png"
# ):
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     ni_1min = ni[[top_col, target_col, T_in_col]].dropna().resample("1min").mean().dropna()

#     dz = H_slab / (Nz - 1)
#     alpha = lambda_s / (rho_s * cp_s)

#     T = np.full(Nz, ni_1min[target_col].iloc[0] + 273.15)

#     times = []
#     pred = []

#     for ts, row in ni_1min.iterrows():
#         T_top = row[top_col] + 273.15
#         T_in = row[T_in_col] + 273.15

#         T_old = T.copy()

#         # Explicit internal conduction, stable enough for dt=60? 
#         # To be safer, substep 60 times with dt=1s.
#         for _ in range(int(dt_s)):
#             Tn = T.copy()

#             # Dirichlet top boundary from measured T1Ta
#             Tn[0] = T_top

#             # internal nodes
#             Fo = alpha * 1.0 / dz**2
#             Tn[1:-1] = T[1:-1] + Fo * (T[2:] - 2*T[1:-1] + T[:-2])

#             # bottom convective boundary
#             # -lambda dT/dz = h_in (T_surface - T_in)
#             # ghost-node style approximation
#             T_ghost = T[-2] - 2 * dz * h_in / lambda_s * (T[-1] - T_in)
#             Tn[-1] = T[-1] + Fo * (T_ghost - 2*T[-1] + T[-2])

#             T = Tn

#         times.append(ts)
#         pred.append(T[-1] - 273.15)

#     sim = pd.Series(pred, index=pd.to_datetime(times), name="Forced slab model")
#     obs = ni_1min[target_col]

#     common = sim.index.intersection(obs.index)
#     err = sim.loc[common] - obs.loc[common]

#     bias = err.mean()
#     mae = err.abs().mean()
#     rmse = np.sqrt((err**2).mean())
#     amp_err = (sim.loc[common].max() - sim.loc[common].min()) - (obs.loc[common].max() - obs.loc[common].min())

#     fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

#     axes[0].plot(obs.index, obs.values, label=f"Measured {target_col}", linewidth=2)
#     axes[0].plot(sim.index, sim.values, "--", label=f"Forced slab from {top_col}", linewidth=2)
#     axes[0].set_title(
#         f"Forced slab diagnostic: {top_col} → {target_col}\n"
#         f"Bias={bias:.2f}°C | MAE={mae:.2f}°C | RMSE={rmse:.2f}°C | AmpErr={amp_err:.2f}°C"
#     )
#     axes[0].set_ylabel("Temperature (°C)")
#     axes[0].grid(True, alpha=0.3)
#     axes[0].legend()

#     axes[1].plot(err.index, err.values)
#     axes[1].axhline(0, linestyle="--")
#     axes[1].set_ylabel("Error (°C)")
#     axes[1].set_xlabel("Datetime")
#     axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print("\n=== FORCED SLAB DIAGNOSTIC ===")
#     print(f"Top boundary : {top_col}")
#     print(f"Target       : {target_col}")
#     print(f"Bias         : {bias:.2f} °C")
#     print(f"MAE          : {mae:.2f} °C")
#     print(f"RMSE         : {rmse:.2f} °C")
#     print(f"AmpErr       : {amp_err:.2f} °C")
#     print(f"Saved plot   : {save_path}")

#     return sim

# forced_slab_from_measured_top(
#     ni_cam,
#     top_col="T1Ta",
#     target_col="T1Tb",
#     T_in_col="T1Ka",
#     save_path="forced_slab_T1Ta_to_T1Tb.png"
# )

# forced_slab_from_measured_top(
#     ni_cam,
#     top_col="T1Ta",
#     target_col="T2A2",
#     T_in_col="T1Ka",
#     save_path="forced_slab_T1Ta_to_T2A2.png"
# )

# forced_slab_from_measured_top(
#     ni_cam,
#     top_col="T1Ta",
#     target_col="T1Tb",
#     T_in_col="T1Ka",
#     save_path="forced_slab_T1Ta_to_T1Tb.png"
# )

# forced_slab_from_measured_top(
#     ni_cam,
#     top_col="T2A",
#     target_col="T2A2",
#     T_in_col="T1Ka",
#     save_path="forced_slab_T2A_to_T2A2.png"
# )

# plot_validation_single(
#     res_cam,
#     ni_cam,
#     plant_type="CAM",
#     save_path="validation_CAM_model_vs_measured.png"
# )



# =========================
# RUN C3
# =========================
print("\n=== RUNNING C3 ===")
res_c3, metrics_c3 = gr.run_validation_case(
    plant_type="C3",
    base_dir=".",
    calibrate_lai=False
)

print("\nC3 metrics:")
print(metrics_c3)

# Load NI C3 untuk plot actual vs model
_, ni_c3, _, _ = gr.prepare_validation_case("C3", base_dir=".")

plot_validation_single(
    res_c3,
    ni_c3,
    plant_type="C3",
    save_path="validation_C3_model_vs_measured2.png"
)

# =========================
# RUN C3 EARLY SEGMENT
# =========================
print("\n=== RUNNING C3 EARLY SEGMENT ===")

# Override window C3 sementara
gr.VALIDATION_WINDOWS["C3"] = (
    "2026-04-06 08:25:00",
    "2026-04-07 10:06:00"
)

res_c3_early, metrics_c3_early = gr.run_validation_case(
    "C3",
    base_dir=".",
    calibrate_lai=False
)

print("\nC3 early metrics:")
print(metrics_c3_early)

# Load NI C3 early untuk plot actual vs model
_, ni_c3_early, _, _ = gr.prepare_validation_case("C3", base_dir=".")

plot_validation_single(
    res_c3_early,
    ni_c3_early,
    plant_type="C3",
    save_path="validation_C3_early_model_vs_measured2.png"
)

def combined_metrics_from_results(result_list, ni_list, plant_type="C3"):
    import pandas as pd
    import numpy as np

    target_col = "T_s_in_C3" if plant_type == "C3" else "T_s_in_CAM"

    all_err = []

    for results, ni in zip(result_list, ni_list):
        sim = pd.Series(
            results["T_s_in"],
            index=pd.to_datetime(results["datetime"]),
            name="Model"
        ).sort_index().resample("1min").mean()

        obs = ni[target_col].sort_index().resample("1min").mean()

        common = sim.index.intersection(obs.index)
        err = sim.loc[common] - obs.loc[common]
        all_err.append(err)

    err_all = pd.concat(all_err).dropna()

    return {
        "n": len(err_all),
        "bias_C": err_all.mean(),
        "mae_C": err_all.abs().mean(),
        "rmse_C": np.sqrt((err_all ** 2).mean()),
        "max_error_C": err_all.max(),
        "min_error_C": err_all.min(),
    }

combined_c3 = combined_metrics_from_results(
    result_list=[res_c3_early, res_c3],
    ni_list=[ni_c3_early, ni_c3],
    plant_type="C3"
)

print("\nCombined C3 metrics:")
print(combined_c3)
print("\nDONE.")