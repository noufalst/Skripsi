"""
Runner validasi green roof.

File ini sengaja dibuat sebagai runner terpisah dari model utama.
Tujuannya:
1. model bisa di-import tanpa langsung menjalankan simulasi,
2. parameter uji terkumpul di satu section,
3. fungsi plotting/diagnostic tidak bercampur dengan eksekusi utama.

Cara pakai umum:
    python run_green_roof_clean.py

Pastikan file data berada di BASE_DIR, atau ubah BASE_DIR di section konfigurasi.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import green_roof_model_clean as gr


# ==============================================================================
# 01 — KONFIGURASI RUN
# ==============================================================================

BASE_DIR = Path(".")
OUTPUT_DIR = Path("outputs")

# Toggle bagian yang ingin dijalankan.
RUN_WEATHER_SUMMARY = True
RUN_CAM = True
RUN_C3 = True
RUN_C3_EARLY = True
RUN_CAM_SENSOR_CHECK = True
RUN_CAM_TRANSFER_DIAGNOSTIC = True
RUN_FORCED_SLAB_DIAGNOSTIC = True

# Target validasi. Ganti di sini kalau ternyata sensor target berubah.
VALIDATION_TARGETS: Dict[str, str] = {
    "CAM": "T2A2",   # alternatif yang sering dicek: "T1Tb"
    "C3": "T3Ka",
}

# Window validasi. Data CAM dan C3 tidak harus berada pada hari yang sama.
VALIDATION_WINDOWS: Dict[str, Tuple[str, str]] = {
    "CAM": ("2026-03-31 11:58:00", "2026-04-02 21:42:00"),
    "C3": ("2026-04-09 11:05:00", "2026-04-10 14:08:00"),
}

VALIDATION_WINDOWS_ALT: Dict[str, Tuple[str, str]] = {
    "C3_EARLY": ("2026-04-06 08:25:00", "2026-04-07 10:06:00"),
}

# Parameter sementara / scientific guess.
PARAMETER_GUESS = {
    "rho_g": 400.0,
    "H_slab": 0.10,
    "H_g": 0.10,
    "theta_sat": 0.90,
    "k_theta_sat": 5e-6,
    "lambda_dry": 0.12,
    "A_roof": 1.0,
}

COVER_FRACTION = {
    "CAM": 0.95,
    "C3": 0.45,
}

# Kalau True, h_in dihitung dinamis dari beda suhu plafon bawah dan udara indoor.
# Untuk validasi awal biasanya False agar lebih stabil dan mudah dibandingkan.
DYNAMIC_H_IN = False


# ==============================================================================
# 02 — HELPER UMUM
# ==============================================================================

def ensure_output_dir() -> None:
    """Buat folder output kalau belum ada."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_runner_configuration() -> None:
    """Kirim konfigurasi runner ke modul model."""
    gr.apply_scientific_guess_parameters(**PARAMETER_GUESS)

    gr.VALIDATION_TARGETS.update(VALIDATION_TARGETS)
    for plant_type, window in VALIDATION_WINDOWS.items():
        gr.VALIDATION_WINDOWS[plant_type] = tuple(pd.Timestamp(x) for x in window)

    gr.geom.dynamic_h_in = DYNAMIC_H_IN
    gr.bromelia.cover_fraction = COVER_FRACTION["CAM"]
    gr.wedelia.cover_fraction = COVER_FRACTION["C3"]


def print_runner_configuration() -> None:
    """Cetak konfigurasi penting supaya run mudah dicek ulang."""
    print("\n=== RUNNER CONFIGURATION ===")
    print(f"BASE_DIR       : {BASE_DIR.resolve()}")
    print(f"OUTPUT_DIR     : {OUTPUT_DIR.resolve()}")
    print(f"CAM target     : {gr.VALIDATION_TARGETS.get('CAM')}")
    print(f"C3 target      : {gr.VALIDATION_TARGETS.get('C3')}")
    print(f"dynamic_h_in   : {gr.geom.dynamic_h_in}")
    print(f"h_in constant  : {gr.geom.h_in} W/m²K")
    print(f"H_g            : {gr.geom.H_g} m")
    print(f"H_slab         : {gr.geom.H_slab} m")
    print(f"theta_sat      : {gr.substrat.theta_sat}")
    print(f"lambda_dry     : {gr.substrat.lambda_dry} W/mK")
    print(f"lambda_sat     : {gr.substrat.lambda_sat:.3f} W/mK")


# ==============================================================================
# 03 — PLOT VALIDASI MODEL VS SENSOR
# ==============================================================================

def plot_validation_single(
    results: dict,
    ni: pd.DataFrame,
    plant_type: str,
    target_col: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Plot model T_s,in vs sensor NI target.

    Output panel:
    1. model vs measured,
    2. residual model - measured,
    3. weather driver T_a dan G_sol.
    """
    plant_type = plant_type.upper()
    title_map = {"CAM": "CAM / Bromelia", "C3": "C3 / Wedelia"}

    if plant_type not in title_map:
        raise ValueError("plant_type harus 'CAM' atau 'C3'")

    if target_col is None:
        target_col = gr.VALIDATION_TARGETS.get(plant_type)

    if target_col not in ni.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan. Kolom tersedia: {ni.columns.tolist()}"
        )

    sim = pd.Series(
        results["T_s_in"],
        index=pd.to_datetime(results["datetime"]),
        name="Model",
    ).sort_index().resample("1min").mean()

    obs = ni[target_col].sort_index().resample("1min").mean()
    common = sim.index.intersection(obs.index)

    sim_common = sim.loc[common]
    obs_common = obs.loc[common]
    err = sim_common - obs_common

    metrics = {
        "n": int(err.count()),
        "bias_C": float(err.mean()),
        "mae_C": float(err.abs().mean()),
        "rmse_C": float(np.sqrt((err**2).mean())),
        "amp_measured_C": float(obs_common.max() - obs_common.min()),
        "amp_model_C": float(sim_common.max() - sim_common.min()),
        "peak_error_C": float(sim_common.max() - obs_common.max()),
        "min_error_C": float(sim_common.min() - obs_common.min()),
    }
    metrics["amp_error_C"] = metrics["amp_model_C"] - metrics["amp_measured_C"]

    print(f"\n=== VALIDATION PLOT: {plant_type} ===")
    print(f"Target sensor       : {target_col}")
    print(f"Measured amplitude  : {metrics['amp_measured_C']:.2f} °C")
    print(f"Model amplitude     : {metrics['amp_model_C']:.2f} °C")
    print(f"Amplitude error     : {metrics['amp_error_C']:.2f} °C")
    print(f"Peak error          : {metrics['peak_error_C']:.2f} °C")
    print(f"Min error           : {metrics['min_error_C']:.2f} °C")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(obs_common.index, obs_common.values, label=f"Measured {target_col}", linewidth=2)
    axes[0].plot(sim_common.index, sim_common.values, label="Model T_s,in", linestyle="--", linewidth=2)
    axes[0].set_ylabel("T_s,in (°C)")
    axes[0].set_title(
        f"Validation {title_map[plant_type]} — Inner Roof Surface Temperature\n"
        f"Bias={metrics['bias_C']:.2f}°C | MAE={metrics['mae_C']:.2f}°C | "
        f"RMSE={metrics['rmse_C']:.2f}°C | AmpErr={metrics['amp_error_C']:.2f}°C | "
        f"n={metrics['n']}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(err.index, err.values, linewidth=1.5)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error (°C)")
    axes[1].set_title("Residual: Model - Measured")
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
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close(fig)

    return metrics


# ==============================================================================
# 04 — DIAGNOSTIC SENSOR CAM
# ==============================================================================

def plot_cam_sensor_channels(ni_cam: pd.DataFrame, save_path: Path) -> None:
    """Plot beberapa channel sensor CAM untuk memilih target validasi yang masuk akal."""
    cam_cols = [
        "T1Ta",  # kandidat top/surface CAM
        "T1Tb",  # kandidat inner roof CAM
        "T1Ka",  # kandidat air/interior CAM
        "T1Ke",  # kandidat soil/top CAM
        "T2A",   # kandidat soil/bottom CAM
        "T2Ka",  # reference roof/control
        "T2A2",  # kandidat target alternatif
    ]
    cam_cols = [col for col in cam_cols if col in ni_cam.columns]

    if not cam_cols:
        print("Tidak ada kolom CAM yang cocok untuk sensor-channel check.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in cam_cols:
        ax.plot(ni_cam.index, ni_cam[col], label=col, linewidth=1.5)

    ax.set_title("CAM Sensor Channel Check — NI Data")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_cam_transfer_path(results: dict, ni_cam: pd.DataFrame, save_path: Path) -> None:
    """Plot jalur transfer panas CAM: top/substrate → slab → inner surface."""
    t = pd.to_datetime(results["datetime"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(t, results["T_s_in"], label="Model T_s,in", linewidth=2)
    for col in ["T1Tb", "T2A2"]:
        if col in ni_cam.columns:
            axes[0].plot(ni_cam.index, ni_cam[col], "--", label=f"Measured {col}", linewidth=1.4)
    axes[0].set_ylabel("Inner temp (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "T_g_surface" in results:
        axes[1].plot(t, results["T_g_surface"], label="Model T_g_surface", linewidth=2)
    for col in ["T1Ta", "T1Ke"]:
        if col in ni_cam.columns:
            axes[1].plot(ni_cam.index, ni_cam[col], "--", label=f"Measured {col}", linewidth=1.4)
    axes[1].set_ylabel("Top/substrate (°C)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for col in ["theta_top", "theta_mean", "theta_bot"]:
        if col in results:
            axes[2].plot(t, results[col], label=col)
    axes[2].set_ylabel("theta (-)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, results["G_sol"], label="G_sol")
    axes[3].set_ylabel("Solar (W/m²)")
    axes[3].set_xlabel("Datetime")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


# ==============================================================================
# 05 — DIAGNOSTIC HARMONIK DAN WATER BALANCE
# ==============================================================================

def harmonic_daily(series: pd.Series, name: str = "signal") -> Optional[dict]:
    """Fit sin/cos 24 jam untuk estimasi mean, amplitudo, dan waktu puncak."""
    s = series.dropna().sort_index().resample("1min").mean().dropna()
    if len(s) < 60:
        return None

    t = (s.index - s.index[0]).total_seconds().to_numpy()
    y = s.to_numpy()
    omega = 2 * np.pi / (24 * 3600)

    X = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    offset, a_sin, b_cos = coef

    amp = np.sqrt(a_sin**2 + b_cos**2)
    phase = np.arctan2(b_cos, a_sin)
    t_peak_s = ((np.pi / 2 - phase) % (2 * np.pi)) / omega

    return {
        "name": name,
        "mean": float(offset),
        "amplitude_C": float(amp),
        "peak_hour_from_start": float(t_peak_s / 3600),
    }


def print_cam_harmonic_diagnostics(results: dict, ni_cam: pd.DataFrame) -> None:
    """Cetak ringkasan amplitudo dan phase harian untuk model/sensor CAM."""
    t = pd.to_datetime(results["datetime"])
    signals = [
        harmonic_daily(pd.Series(results["T_s_in"], index=t), "Model T_s,in"),
    ]

    if "T_g_surface" in results:
        signals.append(harmonic_daily(pd.Series(results["T_g_surface"], index=t), "Model T_g_surface"))

    for col in ["T1Ta", "T1Ke", "T1Tb", "T2A2", "T1Ka"]:
        if col in ni_cam.columns:
            signals.append(harmonic_daily(ni_cam[col], f"Measured {col}"))

    print("\n=== CAM HARMONIC DAILY DIAGNOSTIC ===")
    for item in signals:
        if item is None:
            continue
        print(
            f"{item['name']:20s} | "
            f"mean={item['mean']:.2f}°C | "
            f"amp={item['amplitude_C']:.2f}°C | "
            f"peak_hour={item['peak_hour_from_start']:.2f} h"
        )


def print_water_balance_check(results: dict, label: str = "CAM", dt_save_s: float = 60.0) -> None:
    """Cetak cek sederhana input air, evap, dan theta_mean."""
    t = pd.to_datetime(results["datetime"])
    j_pr = pd.Series(results["j_pr"], index=t)
    j_eva = pd.Series(results["j_eva"], index=t)
    theta_mean = pd.Series(results["theta_mean"], index=t)

    total_input_mm = float((j_pr * dt_save_s).sum())
    total_evap_mm = float((j_eva * dt_save_s).sum())

    print(f"\n=== {label} WATER BALANCE CHECK ===")
    print(f"theta_mean min/max : {theta_mean.min():.3f} / {theta_mean.max():.3f}")
    print(f"j_pr max           : {j_pr.max():.6g}")
    print(f"j_eva max          : {j_eva.max():.6g}")
    print(f"Total input water  : {total_input_mm:.3f} mm")
    print(f"Total evap water   : {total_evap_mm:.3f} mm")


# ==============================================================================
# 06 — DIAGNOSTIC FORCED SLAB
# ==============================================================================

def forced_slab_from_measured_top(
    ni: pd.DataFrame,
    top_col: str = "T1Ta",
    target_col: str = "T1Tb",
    T_in_col: str = "T1Ka",
    H_slab: float = 0.10,
    lambda_s: float = 1.74,
    rho_s: float = 2300.0,
    cp_s: float = 840.0,
    h_in: float = 8.0,
    Nz: int = 67,
    dt_s: int = 60,
    save_path: Optional[Path] = None,
) -> pd.Series:
    """
    Diagnostic konduksi slab paksa.

    Top boundary dipaksa mengikuti sensor top_col, lalu prediksi node bawah
    dibandingkan dengan target_col. Ini berguna untuk mengecek apakah masalah
    utama berasal dari model substrat/foliage atau dari slab/boundary indoor.
    """
    required_cols = [top_col, target_col, T_in_col]
    missing = [col for col in required_cols if col not in ni.columns]
    if missing:
        raise ValueError(f"Kolom tidak tersedia untuk forced slab diagnostic: {missing}")

    ni_1min = ni[required_cols].dropna().resample("1min").mean().dropna()
    if ni_1min.empty:
        raise ValueError("Data forced slab kosong setelah dropna/resample.")

    dz = H_slab / (Nz - 1)
    alpha = lambda_s / (rho_s * cp_s)
    T = np.full(Nz, ni_1min[target_col].iloc[0] + 273.15)

    times = []
    pred = []

    for ts, row in ni_1min.iterrows():
        T_top = row[top_col] + 273.15
        T_in = row[T_in_col] + 273.15

        for _ in range(int(dt_s)):
            T_next = T.copy()
            Fo = alpha / dz**2

            T_next[0] = T_top
            T_next[1:-1] = T[1:-1] + Fo * (T[2:] - 2 * T[1:-1] + T[:-2])

            # Boundary bawah: konveksi menuju udara indoor.
            T_ghost = T[-2] - 2 * dz * h_in / lambda_s * (T[-1] - T_in)
            T_next[-1] = T[-1] + Fo * (T_ghost - 2 * T[-1] + T[-2])

            T = T_next

        times.append(ts)
        pred.append(T[-1] - 273.15)

    sim = pd.Series(pred, index=pd.to_datetime(times), name="Forced slab model")
    obs = ni_1min[target_col]
    common = sim.index.intersection(obs.index)
    err = sim.loc[common] - obs.loc[common]

    bias = float(err.mean())
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    amp_err = float((sim.loc[common].max() - sim.loc[common].min()) - (obs.loc[common].max() - obs.loc[common].min()))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(obs.index, obs.values, label=f"Measured {target_col}", linewidth=2)
    axes[0].plot(sim.index, sim.values, "--", label=f"Forced slab from {top_col}", linewidth=2)
    axes[0].set_title(
        f"Forced slab diagnostic: {top_col} → {target_col}\n"
        f"Bias={bias:.2f}°C | MAE={mae:.2f}°C | RMSE={rmse:.2f}°C | AmpErr={amp_err:.2f}°C"
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(err.index, err.values)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("Error (°C)")
    axes[1].set_xlabel("Datetime")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== FORCED SLAB DIAGNOSTIC ===")
    print(f"Top boundary : {top_col}")
    print(f"Target       : {target_col}")
    print(f"Bias         : {bias:.2f} °C")
    print(f"MAE          : {mae:.2f} °C")
    print(f"RMSE         : {rmse:.2f} °C")
    print(f"AmpErr       : {amp_err:.2f} °C")
    if save_path is not None:
        print(f"Saved plot   : {save_path}")

    return sim


# ==============================================================================
# 07 — WORKFLOW CAM
# ==============================================================================

def run_cam_workflow() -> Tuple[dict, dict, pd.DataFrame]:
    """Jalankan semua step validasi CAM yang aktif di konfigurasi."""
    print("\n=== PREPARE CAM DATA ===")
    _, ni_cam, _, _ = gr.prepare_validation_case("CAM", base_dir=str(BASE_DIR))

    if RUN_CAM_SENSOR_CHECK:
        plot_cam_sensor_channels(
            ni_cam,
            save_path=OUTPUT_DIR / "debug_CAM_all_channels.png",
        )

    print("\n=== RUNNING CAM ===")
    res_cam, metrics_cam = gr.run_validation_case(
        plant_type="CAM",
        base_dir=str(BASE_DIR),
        calibrate_lai=False,
    )

    print("\nCAM metrics:")
    print(metrics_cam)

    if RUN_CAM_TRANSFER_DIAGNOSTIC:
        plot_cam_transfer_path(
            res_cam,
            ni_cam,
            save_path=OUTPUT_DIR / "debug_CAM_transfer_path.png",
        )

    print_water_balance_check(res_cam, label="CAM")
    print_cam_harmonic_diagnostics(res_cam, ni_cam)

    if RUN_FORCED_SLAB_DIAGNOSTIC:
        diagnostic_cases = [
            ("T1Ta", "T1Tb", "T1Ka", "forced_slab_T1Ta_to_T1Tb.png"),
            ("T1Ta", "T2A2", "T1Ka", "forced_slab_T1Ta_to_T2A2.png"),
            ("T2A", "T2A2", "T1Ka", "forced_slab_T2A_to_T2A2.png"),
        ]
        for top_col, target_col, T_in_col, filename in diagnostic_cases:
            try:
                forced_slab_from_measured_top(
                    ni_cam,
                    top_col=top_col,
                    target_col=target_col,
                    T_in_col=T_in_col,
                    save_path=OUTPUT_DIR / filename,
                )
            except ValueError as exc:
                print(f"Skip forced slab {top_col} → {target_col}: {exc}")

    plot_validation_single(
        res_cam,
        ni_cam,
        plant_type="CAM",
        target_col=gr.VALIDATION_TARGETS["CAM"],
        save_path=OUTPUT_DIR / "validation_CAM_model_vs_measured.png",
    )

    return res_cam, metrics_cam, ni_cam


# ==============================================================================
# 08 — WORKFLOW C3
# ==============================================================================

def run_c3_workflow(window_override: Optional[Tuple[str, str]] = None, label: str = "C3") -> Tuple[dict, dict, pd.DataFrame]:
    """Jalankan validasi C3. Bisa override window untuk segment early."""
    original_window = gr.VALIDATION_WINDOWS["C3"]

    if window_override is not None:
        gr.VALIDATION_WINDOWS["C3"] = tuple(pd.Timestamp(x) for x in window_override)

    print(f"\n=== RUNNING {label} ===")
    res_c3, metrics_c3 = gr.run_validation_case(
        plant_type="C3",
        base_dir=str(BASE_DIR),
        calibrate_lai=False,
    )

    print(f"\n{label} metrics:")
    print(metrics_c3)

    _, ni_c3, _, _ = gr.prepare_validation_case("C3", base_dir=str(BASE_DIR))

    safe_label = label.replace(" ", "_")
    plot_validation_single(
        res_c3,
        ni_c3,
        plant_type="C3",
        target_col=gr.VALIDATION_TARGETS["C3"],
        save_path=OUTPUT_DIR / f"validation_{safe_label}_model_vs_measured.png",
    )

    if window_override is not None:
        gr.VALIDATION_WINDOWS["C3"] = original_window

    return res_c3, metrics_c3, ni_c3


def combined_metrics_from_results(
    result_list: Iterable[dict],
    ni_list: Iterable[pd.DataFrame],
    plant_type: str = "C3",
) -> Dict[str, float]:
    """Gabungkan error dari beberapa segment validasi."""
    plant_type = plant_type.upper()
    target_col = gr.VALIDATION_TARGETS[plant_type]

    all_err = []
    for results, ni in zip(result_list, ni_list):
        sim = pd.Series(
            results["T_s_in"],
            index=pd.to_datetime(results["datetime"]),
            name="Model",
        ).sort_index().resample("1min").mean()

        obs = ni[target_col].sort_index().resample("1min").mean()
        common = sim.index.intersection(obs.index)
        all_err.append(sim.loc[common] - obs.loc[common])

    err_all = pd.concat(all_err).dropna()
    return {
        "n": int(len(err_all)),
        "bias_C": float(err_all.mean()),
        "mae_C": float(err_all.abs().mean()),
        "rmse_C": float(np.sqrt((err_all**2).mean())),
        "max_error_C": float(err_all.max()),
        "min_error_C": float(err_all.min()),
    }


# ==============================================================================
# 09 — WEATHER SUMMARY
# ==============================================================================

def export_weather_summary() -> pd.DataFrame:
    """Ringkas window cuaca CAM/C3 dan simpan ke Excel."""
    print("\n=== WEATHER SUMMARY ===")
    summary = gr.summarize_weather_windows(base_dir=str(BASE_DIR))
    print(summary)

    save_path = OUTPUT_DIR / "weather_summary.xlsx"
    summary.to_excel(save_path, index=False)
    print(f"Weather summary saved: {save_path}")
    return summary


# ==============================================================================
# 10 — MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    ensure_output_dir()
    apply_runner_configuration()
    print_runner_configuration()

    if RUN_WEATHER_SUMMARY:
        export_weather_summary()

    if RUN_CAM:
        run_cam_workflow()

    c3_results = []
    c3_ni = []

    if RUN_C3:
        res_c3, _, ni_c3 = run_c3_workflow(label="C3")
        c3_results.append(res_c3)
        c3_ni.append(ni_c3)

    if RUN_C3_EARLY:
        res_c3_early, _, ni_c3_early = run_c3_workflow(
            window_override=VALIDATION_WINDOWS_ALT["C3_EARLY"],
            label="C3_EARLY",
        )
        c3_results.append(res_c3_early)
        c3_ni.append(ni_c3_early)

    if len(c3_results) >= 2:
        combined_c3 = combined_metrics_from_results(c3_results, c3_ni, plant_type="C3")
        print("\nCombined C3 metrics:")
        print(combined_c3)

    print("\nDONE.")


if __name__ == "__main__":
    main()
