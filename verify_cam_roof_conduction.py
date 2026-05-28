"""
VERIFY CAM ROOF OUTDOOR-INDOOR SENSOR RELATION
================================================

Tujuan:
    Mengecek apakah sensor atap outdoor CAM dan atap indoor CAM nyambung secara
    fisik lewat model konduksi 1D transient pada slab/atap.

Mapping header NI Maret/April 2026 yang dipakai:
    T2A2 = Atap outdoor CAM
    T1Tb = Atap indoor CAM
    T1Ka = Udara/ruangan CAM

Catatan penting:
    - Script ini adalah diagnostic/sanity check sensor, bukan model green roof penuh.
    - Top boundary dipaksa mengikuti data measured T2A2.
    - Suhu sisi bawah slab diprediksi dengan konduksi transient.
    - Boundary bawah default memakai konveksi ke udara ruangan T1Ka.
    - Kalau ingin konduksi murni tanpa konveksi bawah, set BOTTOM_BC = "adiabatic".

Cara pakai cepat:
    1) Taruh file NI Maret/April 2026 di folder yang sama dengan script ini.
    2) Sesuaikan NI_FILES di Section 01 kalau nama file berbeda.
    3) Jalankan:
          python verify_cam_roof_conduction.py

Output:
    outputs/forced_slab_CAM_T2A2_to_T1Tb.png
    outputs/raw_CAM_roof_sensors.png
    outputs/forced_slab_CAM_metrics.csv
    outputs/forced_slab_CAM_sensitivity.csv   (jika RUN_SENSITIVITY = True)
    outputs/forced_slab_CAM_sensitivity.png   (jika RUN_SENSITIVITY = True)
"""

# ==============================================================================
# 00 — IMPORTS
# ==============================================================================

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 01 — USER CONFIGURATION
# ==============================================================================

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"

# File NI Maret/April 2026. Ganti kalau nama file kamu berbeda.
NI_FILES = [
    BASE_DIR / "Pengukuran 30_1 Maret 2026.xlsx",
    BASE_DIR / "Pengukuran 30_2 Maret 2026.xlsx",
    BASE_DIR / "Pengukuran 3 April 2026.xlsx",
]

# Window analisis. Set None kalau mau pakai semua data.
# Contoh: WINDOW_START = "2026-04-09 11:05:00"
WINDOW_START: Optional[str] = None
WINDOW_END: Optional[str] = None

# Mapping sensor CAM menurut header NI Maret/April 2026.
TOP_COL = "T2A2"       # Atap outdoor CAM
TARGET_COL = "T1Tb"    # Atap indoor CAM
T_IN_COL = "T1Ka"      # Udara/ruangan CAM

# Parameter slab/atap. Sesuaikan dengan struktur fisik eksperimenmu.
H_SLAB = 0.10          # m, tebal slab/beton/atap yang diuji
LAMBDA_S = 1.74        # W/m.K, konduktivitas termal beton normal kira-kira 1.4–2.0
RHO_S = 2300.0         # kg/m3
CP_S = 840.0           # J/kg.K
H_IN = 8.0             # W/m2.K, coba 2–4 untuk natural convection yang lebih lemah
NZ = 51                # jumlah node 1D
DT_SECONDS = 60.0      # timestep setelah resample 1 menit

# Boundary bawah:
#   "convective" = bawah slab konveksi ke T1Ka, recommended untuk prediksi T1Tb
#   "adiabatic"  = bawah slab tanpa heat loss; diagnostic konduksi murni yang sangat ideal
BOTTOM_BC = "convective"

# Sensitivity sweep untuk melihat apakah model terlalu damping.
RUN_SENSITIVITY = True
SENSITIVITY_H_IN = [1.5, 2.0, 4.0, 8.0]
SENSITIVITY_H_SLAB = [0.05, 0.07, 0.10]
SENSITIVITY_LAMBDA = [1.0, 1.4, 1.74, 2.2]


# ==============================================================================
# 02 — NI HEADER LIST DAN SEMANTIC MAP
# ==============================================================================

# Header Maret/April tetap dipakai apa adanya agar cocok dengan file raw NI.
NI_CHANNEL_NAMES = [
    "timestamp_serial",

    # CAM
    "T1Kd",    # Lantai CAM
    "T1Kb",    # Dinding timur CAM
    "T1Ke",    # Tanah atas CAM
    "T1Ka",    # Ruangan CAM
    "T1Kc",    # Dinding barat CAM

    # C3
    "T3Kb",    # Dinding barat C3
    "T3Kd",    # Ruangan C3
    "T3Ke",    # Lantai C3
    "T3Kc",    # Dinding timur C3
    "T3Ka",    # Atap indoor C3

    # RR / tradisional
    "T2Ka",    # Atap indoor RR
    "T2Kd",    # Dinding timur RR
    "T2Kc",    # Dinding barat RR
    "T2Ke",    # Lantai RR. Di beberapa nomenclature bisa tertulis "2Ke"

    # Outdoor / additional channels
    "T1A",     # Atap outdoor C3
    "T2A",     # Tanah bawah CAM
    "T2A2",    # Atap outdoor CAM
    "T1Tb",    # Atap indoor CAM
    "T1Ta",    # Ruangan RR
    "T1Ta2",   # Atap outdoor RR
]

CAM_SENSOR_MAP = {
    "T_air_in": "T1Ka",
    "T_soil_top": "T1Ke",
    "T_soil_bottom": "T2A",
    "T_roof_out": "T2A2",
    "T_roof_in": "T1Tb",
}


# ==============================================================================
# 03 — DATA CLASSES
# ==============================================================================

@dataclass(frozen=True)
class SlabParams:
    """Parameter fisik slab/atap untuk model konduksi 1D."""

    H: float = H_SLAB
    k: float = LAMBDA_S
    rho: float = RHO_S
    cp: float = CP_S
    h_in: float = H_IN
    nz: int = NZ
    dt: float = DT_SECONDS
    bottom_bc: str = BOTTOM_BC

    @property
    def alpha(self) -> float:
        """Thermal diffusivity, alpha = k / (rho cp)."""
        return self.k / (self.rho * self.cp)

    @property
    def dz(self) -> float:
        return self.H / (self.nz - 1)

    @property
    def fourier_number(self) -> float:
        return self.alpha * self.dt / self.dz**2


# ==============================================================================
# 04 — LOW-LEVEL XLSX LOADER UNTUK FILE NI LABVIEW
# ==============================================================================

def excel_col_to_index(cell_ref: str) -> int:
    """Convert Excel column letters in cell reference to zero-based column index."""
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    idx = 0
    for char in letters:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheet1_numeric_rows(filepath: Path, expected_cols: int) -> list[list[float]]:
    """
    Read numeric rows from sheet1.xml in an .xlsx file.

    Kenapa tidak langsung pandas.read_excel?
        File LabVIEW kadang besar dan formatnya agak tidak rapi. Parser XML langsung
        biasanya lebih cepat dan menghindari masalah header/string.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    with zipfile.ZipFile(filepath, "r") as zf:
        with zf.open("xl/worksheets/sheet1.xml") as f:
            root = ET.fromstring(f.read())

    rows: list[list[float]] = []
    rows_xml = root.findall(f".//{ns}row")

    for row_xml in rows_xml[1:]:  # skip header row
        values = [np.nan] * expected_cols

        for cell in row_xml.findall(f"{ns}c"):
            cell_ref = cell.attrib.get("r", "")
            col_idx = excel_col_to_index(cell_ref)
            if col_idx < 0 or col_idx >= expected_cols:
                continue

            value_tag = cell.find(f"{ns}v")
            if value_tag is None or value_tag.text is None:
                continue

            try:
                values[col_idx] = float(value_tag.text)
            except ValueError:
                values[col_idx] = np.nan

        # Ambil row yang minimal punya timestamp dan sebagian besar kolom numerik.
        if not np.isnan(values[0]) and np.count_nonzero(~pd.isna(values)) >= expected_cols - 2:
            rows.append(values)

    return rows


def labview_serial_to_datetime(serial: pd.Series) -> pd.Series:
    """Convert LabVIEW day serial since 1904-01-01 to pandas datetime."""
    return pd.Timestamp("1904-01-01") + pd.to_timedelta(serial, unit="D")


def load_ni_file(filepath: Path) -> pd.DataFrame:
    """Load one NI LabVIEW Excel file using the fixed Maret/April header mapping."""
    filepath = Path(filepath)
    rows = read_xlsx_sheet1_numeric_rows(filepath, expected_cols=len(NI_CHANNEL_NAMES))

    if not rows:
        raise ValueError(f"Tidak ada row numerik yang terbaca dari {filepath}")

    df = pd.DataFrame(rows, columns=NI_CHANNEL_NAMES)
    df["timestamp"] = labview_serial_to_datetime(df["timestamp_serial"])
    df = df.drop(columns=["timestamp_serial"]).set_index("timestamp").sort_index()

    return df


def clean_temperature_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean anomaly suhu sederhana.

    Nilai di luar -10 sampai 80 °C dianggap sensor disconnect/glitch dan
    diganti NaN, lalu diinterpolasi time-based untuk gap pendek.
    """
    df = df.copy()
    temp_cols = [c for c in df.columns if c.startswith("T")]

    for col in temp_cols:
        bad = (df[col] < -10) | (df[col] > 80)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"Cleaning anomaly {col}: {n_bad} points")
            df.loc[bad, col] = np.nan
            df[col] = df[col].interpolate(
                method="time",
                limit=30,
                limit_direction="both",
            )

    return df


def load_multiple_ni_files(filepaths: Iterable[Path]) -> pd.DataFrame:
    """Load, combine, clean, sort, and deduplicate NI files."""
    dfs = []

    for fp in filepaths:
        fp = Path(fp)
        if not fp.exists():
            print(f"WARNING: file tidak ditemukan, skip: {fp}")
            continue

        print(f"Loading NI file: {fp}")
        dfs.append(load_ni_file(fp))

    if not dfs:
        raise FileNotFoundError(
            "Tidak ada file NI yang berhasil dibaca. Cek NI_FILES / BASE_DIR."
        )

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = clean_temperature_channels(df)

    if WINDOW_START is not None:
        df = df[df.index >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        df = df[df.index <= pd.Timestamp(WINDOW_END)]

    print(f"\nNI combined period : {df.index.min()} → {df.index.max()}")
    print(f"NI combined records: {len(df)}")

    return df


# ==============================================================================
# 05 — PREPROCESS SENSOR UNTUK FORCED SLAB
# ==============================================================================

def prepare_forced_slab_input(
    ni: pd.DataFrame,
    top_col: str = TOP_COL,
    target_col: str = TARGET_COL,
    t_in_col: str = T_IN_COL,
) -> pd.DataFrame:
    """Ambil 3 channel penting, resample ke 1 menit, dan interpolate gap pendek."""
    required_cols = [top_col, target_col, t_in_col]
    missing = [col for col in required_cols if col not in ni.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di NI data: {missing}")

    data = ni[required_cols].copy().sort_index()
    data = data.resample("1min").mean()
    data = data.interpolate(method="time", limit=10, limit_direction="both")
    data = data.dropna()

    if data.empty:
        raise ValueError("Data forced slab kosong setelah resample/dropna.")

    print("\n=== SENSOR SUMMARY ===")
    for col in required_cols:
        s = data[col].dropna()
        print(
            f"{col:>6s}: min={s.min():6.2f} °C | max={s.max():6.2f} °C | "
            f"mean={s.mean():6.2f} °C | amp={s.max() - s.min():6.2f} °C"
        )

    return data


# ==============================================================================
# 06 — NUMERICAL CORE: IMPLICIT 1D TRANSIENT CONDUCTION
# ==============================================================================

def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm untuk sistem tridiagonal.

    a = lower diagonal, a[0] unused/0
    b = main diagonal
    c = upper diagonal, c[-1] unused/0
    d = right-hand side
    """
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))

    for i in range(1, n):
        if abs(bc[i - 1]) < 1e-14:
            raise ZeroDivisionError("Zero pivot in tridiagonal solver.")
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


def step_slab_implicit(
    T: np.ndarray,
    T_top: float,
    T_air_in: float,
    params: SlabParams,
) -> np.ndarray:
    """
    One timestep backward-Euler untuk slab 1D.

    Boundary atas:
        Dirichlet, T(x=0) = measured T_top.

    Boundary bawah:
        convective:
            -k dT/dx = h_in (T_bottom - T_air_in)
        adiabatic:
            dT/dx = 0
    """
    n = params.nz
    fo = params.fourier_number
    dz = params.dz

    a = np.zeros(n)  # lower diagonal
    b = np.zeros(n)  # main diagonal
    c = np.zeros(n)  # upper diagonal
    d = np.zeros(n)  # RHS

    # Top boundary: prescribed measured outdoor roof temperature.
    b[0] = 1.0
    d[0] = T_top

    # Interior nodes.
    for i in range(1, n - 1):
        a[i] = -fo
        b[i] = 1.0 + 2.0 * fo
        c[i] = -fo
        d[i] = T[i]

    # Bottom boundary.
    if params.bottom_bc.lower() == "convective":
        bi = dz * params.h_in / params.k
        a[-1] = -2.0 * fo
        b[-1] = 1.0 + 2.0 * fo * (1.0 + bi)
        c[-1] = 0.0
        d[-1] = T[-1] + 2.0 * fo * bi * T_air_in
    elif params.bottom_bc.lower() == "adiabatic":
        a[-1] = -2.0 * fo
        b[-1] = 1.0 + 2.0 * fo
        c[-1] = 0.0
        d[-1] = T[-1]
    else:
        raise ValueError("bottom_bc harus 'convective' atau 'adiabatic'.")

    return solve_tridiagonal(a, b, c, d)


def run_forced_slab(
    data: pd.DataFrame,
    params: SlabParams,
    top_col: str = TOP_COL,
    target_col: str = TARGET_COL,
    t_in_col: str = T_IN_COL,
) -> pd.DataFrame:
    """
    Run forced slab diagnostic.

    Return dataframe berisi measured top, measured target, indoor air, model bottom,
    residual, dan heat flux bawah.
    """
    # Initial condition: uniform slab at first measured indoor-roof temperature.
    T = np.full(params.nz, float(data[target_col].iloc[0]))

    pred = []
    q_in = []

    for _, row in data.iterrows():
        T = step_slab_implicit(
            T=T,
            T_top=float(row[top_col]),
            T_air_in=float(row[t_in_col]),
            params=params,
        )
        pred.append(T[-1])

        # Positive q_in = heat transfer from inner roof surface to indoor air.
        if params.bottom_bc.lower() == "convective":
            q_in.append(params.h_in * (T[-1] - float(row[t_in_col])))
        else:
            q_in.append(0.0)

    result = data.copy()
    result["T_bottom_model"] = np.asarray(pred)
    result["residual_model_minus_measured"] = result["T_bottom_model"] - result[target_col]
    result["q_inner_to_air_W_m2"] = np.asarray(q_in)

    return result


# ==============================================================================
# 07 — METRICS DAN LAG CHECK
# ==============================================================================

def estimate_lag_minutes(reference: pd.Series, response: pd.Series, max_lag_min: int = 360) -> int:
    """
    Estimate lag response terhadap reference via cross-correlation.

    Positive lag berarti response tertinggal dari reference.
    """
    ref = reference.dropna().resample("1min").mean()
    rsp = response.dropna().resample("1min").mean()
    idx = ref.index.intersection(rsp.index)

    if len(idx) < 60:
        return 0

    x = ref.loc[idx].to_numpy(dtype=float)
    y = rsp.loc[idx].to_numpy(dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)

    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag_min, max_lag_min + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: len(xs)]
        elif lag > 0:
            xs = x[:-lag]
            ys = y[lag:]
        else:
            xs = x
            ys = y

        if len(xs) < 30:
            continue

        denom = np.std(xs) * np.std(ys)
        if denom == 0:
            continue

        corr = float(np.mean((xs - xs.mean()) * (ys - ys.mean())) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return int(best_lag)


def calculate_metrics(result: pd.DataFrame) -> pd.DataFrame:
    """Hitung metrik hubungan measured top-target dan forced slab result."""
    err = result["residual_model_minus_measured"].dropna()

    metrics = {
        "n": int(err.count()),
        "top_col": TOP_COL,
        "target_col": TARGET_COL,
        "t_in_col": T_IN_COL,
        "bottom_bc": BOTTOM_BC,
        "H_slab_m": H_SLAB,
        "lambda_W_mK": LAMBDA_S,
        "rho_kg_m3": RHO_S,
        "cp_J_kgK": CP_S,
        "h_in_W_m2K": H_IN,
        "Fo": SlabParams().fourier_number,
        "bias_C": float(err.mean()),
        "mae_C": float(err.abs().mean()),
        "rmse_C": float(np.sqrt(np.mean(err**2))),
        "amp_top_C": float(result[TOP_COL].max() - result[TOP_COL].min()),
        "amp_measured_target_C": float(result[TARGET_COL].max() - result[TARGET_COL].min()),
        "amp_model_C": float(result["T_bottom_model"].max() - result["T_bottom_model"].min()),
        "amp_error_C": float(
            (result["T_bottom_model"].max() - result["T_bottom_model"].min())
            - (result[TARGET_COL].max() - result[TARGET_COL].min())
        ),
        "peak_top_time": str(result[TOP_COL].idxmax()),
        "peak_measured_target_time": str(result[TARGET_COL].idxmax()),
        "peak_model_time": str(result["T_bottom_model"].idxmax()),
        "lag_measured_target_vs_top_min": estimate_lag_minutes(result[TOP_COL], result[TARGET_COL]),
        "lag_model_vs_top_min": estimate_lag_minutes(result[TOP_COL], result["T_bottom_model"]),
    }

    return pd.DataFrame([metrics])


# ==============================================================================
# 08 — PLOTTING
# ==============================================================================

def plot_raw_sensors(data: pd.DataFrame, save_path: Path) -> None:
    """Plot raw sensor relationship: outdoor roof, indoor roof, indoor air."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data[TOP_COL], label=f"{TOP_COL} — Atap outdoor CAM", linewidth=1.8)
    ax.plot(data.index, data[TARGET_COL], label=f"{TARGET_COL} — Atap indoor CAM", linewidth=1.8)
    ax.plot(data.index, data[T_IN_COL], label=f"{T_IN_COL} — Ruangan CAM", linewidth=1.4, linestyle="--")

    ax.set_title("Raw sensor check: CAM roof outdoor vs indoor")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_forced_slab(result: pd.DataFrame, metrics: pd.DataFrame, save_path: Path) -> None:
    """Plot forced slab result and residual."""
    m = metrics.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(result.index, result[TOP_COL], label=f"Measured {TOP_COL} — outdoor roof", linewidth=1.5)
    axes[0].plot(result.index, result[TARGET_COL], label=f"Measured {TARGET_COL} — indoor roof", linewidth=2.0)
    axes[0].plot(result.index, result[T_IN_COL], label=f"Measured {T_IN_COL} — indoor air", linewidth=1.3, linestyle="--")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Measured CAM channels")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result.index, result[TARGET_COL], label=f"Measured {TARGET_COL}", linewidth=2.0)
    axes[1].plot(result.index, result["T_bottom_model"], label=f"Forced slab from {TOP_COL}", linewidth=2.0, linestyle="--")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title(
        "Forced slab diagnostic: "
        f"{TOP_COL} → {TARGET_COL} | "
        f"Bias={m['bias_C']:.2f}°C, MAE={m['mae_C']:.2f}°C, "
        f"RMSE={m['rmse_C']:.2f}°C, AmpErr={m['amp_error_C']:.2f}°C"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(result.index, result["residual_model_minus_measured"], label="Model - measured", linewidth=1.4)
    axes[2].axhline(0, linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Residual (°C)")
    axes[2].set_xlabel("Datetime")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_sensitivity(sensitivity_df: pd.DataFrame, save_path: Path) -> None:
    """Plot sensitivity summary as scatter-like line by RMSE and AmpErr."""
    if sensitivity_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [
        f"H={row.H_slab_m:.2f}, k={row.lambda_W_mK:.2f}, h={row.h_in_W_m2K:.1f}"
        for row in sensitivity_df.itertuples()
    ]
    x = np.arange(len(sensitivity_df))

    ax.plot(x, sensitivity_df["rmse_C"], marker="o", label="RMSE")
    ax.plot(x, sensitivity_df["amp_error_C"].abs(), marker="o", label="|AmpErr|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Error (°C)")
    ax.set_title("Forced slab sensitivity — lower is better")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ==============================================================================
# 09 — SENSITIVITY SWEEP
# ==============================================================================

def run_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep beberapa nilai H, k, dan h_in untuk melihat apakah forced slab terlalu flat.

    Interpretasi cepat:
        - AmpErr negatif besar: model terlalu flat / amplitudo terlalu kecil.
        - H lebih kecil atau k lebih besar biasanya membuat respons lebih agresif.
        - h_in lebih kecil biasanya membuat sisi indoor roof tidak terlalu terkunci ke udara ruangan.
    """
    rows = []

    for H in SENSITIVITY_H_SLAB:
        for k in SENSITIVITY_LAMBDA:
            for h in SENSITIVITY_H_IN:
                params = SlabParams(H=H, k=k, rho=RHO_S, cp=CP_S, h_in=h, nz=NZ, dt=DT_SECONDS, bottom_bc=BOTTOM_BC)
                result = run_forced_slab(data, params=params)
                err = result["residual_model_minus_measured"].dropna()

                rows.append({
                    "H_slab_m": H,
                    "lambda_W_mK": k,
                    "h_in_W_m2K": h,
                    "Fo": params.fourier_number,
                    "bias_C": float(err.mean()),
                    "mae_C": float(err.abs().mean()),
                    "rmse_C": float(np.sqrt(np.mean(err**2))),
                    "amp_measured_target_C": float(result[TARGET_COL].max() - result[TARGET_COL].min()),
                    "amp_model_C": float(result["T_bottom_model"].max() - result["T_bottom_model"].min()),
                    "amp_error_C": float(
                        (result["T_bottom_model"].max() - result["T_bottom_model"].min())
                        - (result[TARGET_COL].max() - result[TARGET_COL].min())
                    ),
                    "lag_model_vs_top_min": estimate_lag_minutes(result[TOP_COL], result["T_bottom_model"]),
                })

    sensitivity = pd.DataFrame(rows)
    sensitivity = sensitivity.sort_values(["rmse_C", "amp_error_C"], key=lambda s: s.abs() if s.name == "amp_error_C" else s)
    return sensitivity


# ==============================================================================
# 10 — MAIN WORKFLOW
# ==============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== CAM ROOF CONDUCTION VERIFICATION ===")
    print(f"Top boundary    : {TOP_COL}  = Atap outdoor CAM")
    print(f"Target measured : {TARGET_COL} = Atap indoor CAM")
    print(f"Indoor air      : {T_IN_COL}  = Ruangan CAM")
    print(f"Bottom BC       : {BOTTOM_BC}")

    ni = load_multiple_ni_files(NI_FILES)
    data = prepare_forced_slab_input(ni, TOP_COL, TARGET_COL, T_IN_COL)

    params = SlabParams()
    print("\n=== SLAB PARAMETER SUMMARY ===")
    print(f"H      = {params.H:.3f} m")
    print(f"k      = {params.k:.3f} W/m.K")
    print(f"rho    = {params.rho:.1f} kg/m3")
    print(f"cp     = {params.cp:.1f} J/kg.K")
    print(f"alpha  = {params.alpha:.3e} m2/s")
    print(f"h_in   = {params.h_in:.2f} W/m2.K")
    print(f"Nz     = {params.nz}")
    print(f"dt     = {params.dt:.1f} s")
    print(f"Fo     = {params.fourier_number:.4f}  (implicit scheme, stable even if > 0.5)")

    result = run_forced_slab(data, params=params)
    metrics = calculate_metrics(result)

    print("\n=== FORCED SLAB METRICS ===")
    for key, value in metrics.iloc[0].items():
        print(f"{key}: {value}")

    metrics_path = OUTPUT_DIR / "forced_slab_CAM_metrics.csv"
    result_path = OUTPUT_DIR / "forced_slab_CAM_timeseries.csv"
    metrics.to_csv(metrics_path, index=False)
    result.to_csv(result_path, index=True)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {result_path}")

    plot_raw_sensors(data, OUTPUT_DIR / "raw_CAM_roof_sensors.png")
    plot_forced_slab(result, metrics, OUTPUT_DIR / "forced_slab_CAM_T2A2_to_T1Tb.png")

    if RUN_SENSITIVITY:
        print("\n=== RUNNING SENSITIVITY SWEEP ===")
        sensitivity = run_sensitivity(data)
        sensitivity_path = OUTPUT_DIR / "forced_slab_CAM_sensitivity.csv"
        sensitivity.to_csv(sensitivity_path, index=False)
        print(f"Saved: {sensitivity_path}")
        print("\nTop 10 sensitivity cases by RMSE:")
        print(sensitivity.head(10).to_string(index=False))
        plot_sensitivity(sensitivity.head(24), OUTPUT_DIR / "forced_slab_CAM_sensitivity.png")

    print("\nDONE.")


if __name__ == "__main__":
    main()
