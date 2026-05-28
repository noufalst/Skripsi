"""
CAM SENSOR / HEAT-PATH DIAGNOSTIC
=================================

Purpose
-------
This script is NOT a calibration runner. It diagnoses why the CAM model can become
too flat by checking whether the measured NI target sensor is consistent with the
model heat path:

    foliage / outdoor surface -> substrate top -> substrate bottom -> slab inner surface

Run this AFTER running run_cam_gsw_driven.py, because it reads the prediction CSV
from outputs_cam_gsw_fixed_inputs/ by default.

Default use:
    python cam_sensor_path_diagnostic.py

If your data are elsewhere:
    python cam_sensor_path_diagnostic.py --base-dir "E:\\Pagi\\SKRRRRRRRipsi\\data"

If your CAM model output folder is different:
    python cam_sensor_path_diagnostic.py --model-output-dir outputs_cam_gsw_fixed_inputs

Outputs
-------
outputs_cam_path_diagnostic/
    cam_sensor_path_overlay.png
    cam_model_vs_sensor_matrix.csv
    cam_measured_sensor_relationships.csv
    cam_amplitude_path_summary.csv
    cam_forced_slab_candidates.csv
    cam_forced_slab_thickness_sweep.csv
    cam_forced_slab_diagnostic.png
    cam_path_diagnostic_summary.json

Interpretation
--------------
1) If model T_g_top is flat already:
   problem is likely canopy / solar transmission / substrate top boundary.

2) If model T_g_top has amplitude but model T_s_in is flat:
   problem is likely slab conduction, indoor boundary, or target sensor mismatch.

3) If measured T1Tb resembles T1Ta/T1Ke more than model T_s_in:
   T1Tb may not represent the same physical node as model T_s_in.

4) If forced-slab using measured top sensors cannot reproduce T1Tb unless slab
   thickness is unrealistically small:
   T1Tb is probably not behaving like the bottom of a 10 cm concrete slab.
"""

from __future__ import annotations

# ==============================================================================
# 00 — IMPORTS
# ==============================================================================

import argparse
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# 01 — DEFAULT CONFIGURATION
# ==============================================================================

DEFAULT_BASE_DIR = Path(".")
DEFAULT_MODEL_OUTPUT_DIR = Path("outputs_cam_gsw_fixed_inputs")
DEFAULT_DIAGNOSTIC_OUTPUT_DIR = Path("outputs_cam_path_diagnostic")

DEFAULT_NI_FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]

DEFAULT_EVAL_START = "2026-03-31 11:58:00"
DEFAULT_EVAL_END = "2026-04-02 21:42:00"

DEFAULT_TARGET_COL = "T1Tb"
DEFAULT_TIN_COL = "T1Ka"

# Measured CAM channels to compare along the likely heat path.
CAM_SENSOR_PATH = [
    "T1Ta",  # Atap outdoor CAM / possible outside roof surface
    "T1Ke",  # Tanah atas CAM / top substrate candidate
    "T2A",   # Tanah bawah CAM / bottom substrate candidate
    "T2A2",  # alternative CAM candidate
    "T1Tb",  # Atap indoor CAM / current target
    "T1Ka",  # Ruangan CAM / indoor air boundary
]

# Model columns saved by green_roof_cam_gsw.py.
MODEL_PATH = [
    "T_f",
    "T_g_top",
    "T_g_mid",
    "T_g_bot",
    "T_s_in",
    "T_in_used",
]


# ==============================================================================
# 02 — DATA CLASSES
# ==============================================================================

@dataclass
class SeriesStats:
    name: str
    n: int
    mean_C: float
    min_C: float
    max_C: float
    amplitude_C: float
    peak_time: str
    trough_time: str


@dataclass
class PairMetrics:
    reference: str
    candidate: str
    n: int
    bias_C: float
    mae_C: float
    rmse_C: float
    corr: float
    amp_reference_C: float
    amp_candidate_C: float
    amp_error_C: float
    peak_time_reference: str
    peak_time_candidate: str
    peak_lag_min: float


# ==============================================================================
# 03 — ARGUMENT PARSER
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose CAM sensor mapping and model heat-path damping."
    )

    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--model-output-dir", type=Path, default=DEFAULT_MODEL_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTIC_OUTPUT_DIR)

    parser.add_argument("--ni-files", nargs="*", default=DEFAULT_NI_FILES)
    parser.add_argument("--prediction-file", default=None,
                        help="Optional explicit prediction CSV path. If omitted, auto-detects in model output dir.")

    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--eval-end", default=DEFAULT_EVAL_END)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--tin-col", default=DEFAULT_TIN_COL)

    parser.add_argument("--H-slab", type=float, default=0.10, help="Measured slab thickness [m]. Fixed diagnostic input.")
    parser.add_argument("--lambda-s", type=float, default=1.74, help="Concrete conductivity [W/mK].")
    parser.add_argument("--rho-s", type=float, default=2300.0, help="Concrete density [kg/m3].")
    parser.add_argument("--cp-s", type=float, default=840.0, help="Concrete heat capacity [J/kgK].")
    parser.add_argument("--h-in", type=float, default=8.0, help="Indoor convection coefficient [W/m2K].")

    parser.add_argument("--forced-top-cols", nargs="*", default=["T1Ta", "T1Ke", "T2A", "T2A2"],
                        help="Measured channels to use as forced slab top boundary.")
    parser.add_argument("--forced-spinup-hours", type=float, default=6.0,
                        help="Start forced slab this many hours before eval start when data exist.")
    parser.add_argument("--thickness-min", type=float, default=0.01)
    parser.add_argument("--thickness-max", type=float, default=0.12)
    parser.add_argument("--thickness-step", type=float, default=0.005)

    return parser


# ==============================================================================
# 04 — ROBUST LOADERS
# ==============================================================================


def resolve_paths(base_dir: Path, file_names: Sequence[str]) -> List[Path]:
    paths = []
    for name in file_names:
        p = Path(name)
        if not p.is_absolute():
            p = base_dir / p
        paths.append(p)
    return paths



def load_NI_sensor_data_fallback(filepath: Path) -> pd.DataFrame:
    """
    Minimal NI Excel XML loader.

    This fallback mirrors the channel order used in the model code. It avoids
    needing openpyxl and is robust to large LabVIEW-exported xlsx files.
    """
    ni_channel_names = [
        "timestamp_serial",
        "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
        "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
        "T2Ka", "T2Kd", "T2Kc", "2Ke", "T1A",
        "T2A", "T2A2", "T1Tb", "T1Ta", "T1Ta2",
    ]

    with zipfile.ZipFile(filepath, "r") as z:
        with z.open("xl/worksheets/sheet1.xml") as f:
            xml_content = f.read()

    tree = ET.fromstring(xml_content)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    rows_xml = tree.findall(f".//{ns}row")
    data = []
    for row in rows_xml[1:]:
        vals = []
        for cell in row.findall(f"{ns}c"):
            v = cell.find(f"{ns}v")
            try:
                vals.append(float(v.text) if v is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        if len(vals) == 21:
            data.append(vals)

    if not data:
        raise ValueError(f"No numeric NI rows parsed from {filepath}")

    df = pd.DataFrame(data, columns=ni_channel_names)
    labview_epoch = pd.Timestamp("1904-01-01")
    df["timestamp"] = labview_epoch + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"])
    df = df.sort_index()

    # Basic anomaly filter for CAM bottom soil sensor candidate.
    if "T2A" in df.columns:
        mask = (df["T2A"] < -10) | (df["T2A"] > 80)
        if mask.any():
            df.loc[mask, "T2A"] = np.nan
            df["T2A"] = df["T2A"].interpolate(method="time", limit=30, limit_direction="both")

    # Descriptive aliases.
    if "T1Ke" in df.columns:
        df["T_g_top_CAM"] = df["T1Ke"]
    if "T2A" in df.columns:
        df["T_g_bot_CAM"] = df["T2A"]
    if "T1Tb" in df.columns:
        df["T_s_in_CAM"] = df["T1Tb"]
    if "T1Ta" in df.columns:
        df["T_s_ext_CAM"] = df["T1Ta"]
    if "T1Ka" in df.columns:
        df["T_in_CAM"] = df["T1Ka"]

    return df



def load_multiple_NI_sensor_data(filepaths: Sequence[Path]) -> pd.DataFrame:
    """Load multiple NI files; prefer green_roof_cam_gsw loader if available."""
    try:
        import green_roof_cam_gsw as gr  # local module if script is in same folder
        if hasattr(gr, "load_multiple_NI_sensor_data"):
            return gr.load_multiple_NI_sensor_data(filepaths)
    except Exception:
        pass

    frames = []
    for fp in filepaths:
        if not fp.exists():
            raise FileNotFoundError(f"NI file not found: {fp}")
        frames.append(load_NI_sensor_data_fallback(fp))

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out



def load_prediction_csv(model_output_dir: Path, prediction_file: Optional[str]) -> pd.DataFrame:
    """Load model prediction CSV from explicit path or common output names."""
    candidates: List[Path] = []
    if prediction_file:
        candidates.append(Path(prediction_file))
    else:
        candidates.extend([
            model_output_dir / "cam_gsw_prediction_full_with_spinup.csv",
            model_output_dir / "cam_gsw_prediction_eval_window.csv",
            model_output_dir / "cam_physical_prediction_full_with_spinup.csv",
            model_output_dir / "cam_prediction_calibrated.csv",
        ])

    for p in candidates:
        if not p.is_absolute():
            p = Path(".") / p
        if p.exists():
            df = pd.read_csv(p)
            # Find datetime column.
            dt_col = None
            for c in ["datetime", "timestamp", "time", "index"]:
                if c in df.columns:
                    dt_col = c
                    break
            if dt_col is None:
                dt_col = df.columns[0]
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
            return df

    raise FileNotFoundError(
        "Prediction CSV not found. Run run_cam_gsw_driven.py first, or pass --prediction-file. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ==============================================================================
# 05 — METRICS
# ==============================================================================


def to_1min(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy().sort_index()
    out = out[(out.index >= start) & (out.index <= end)]
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out = out[numeric_cols].resample("1min").mean()
    return out



def clean_pair(reference: pd.Series, candidate: pd.Series) -> Tuple[pd.Series, pd.Series]:
    common = reference.index.intersection(candidate.index)
    a = reference.loc[common]
    b = candidate.loc[common]
    mask = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    return a.loc[mask], b.loc[mask]



def safe_stats(series: pd.Series, name: str) -> SeriesStats:
    s = series.dropna().sort_index()
    if s.empty:
        return SeriesStats(name, 0, np.nan, np.nan, np.nan, np.nan, "", "")
    peak_time = str(s.idxmax())
    trough_time = str(s.idxmin())
    return SeriesStats(
        name=name,
        n=int(s.count()),
        mean_C=float(s.mean()),
        min_C=float(s.min()),
        max_C=float(s.max()),
        amplitude_C=float(s.max() - s.min()),
        peak_time=peak_time,
        trough_time=trough_time,
    )



def pair_metrics(reference: pd.Series, candidate: pd.Series, reference_name: str, candidate_name: str) -> PairMetrics:
    ref, cand = clean_pair(reference, candidate)
    if len(ref) < 3:
        return PairMetrics(reference_name, candidate_name, 0, np.nan, np.nan, np.nan, np.nan,
                           np.nan, np.nan, np.nan, "", "", np.nan)

    err = cand - ref
    corr = float(np.corrcoef(ref.to_numpy(dtype=float), cand.to_numpy(dtype=float))[0, 1])
    amp_ref = float(ref.max() - ref.min())
    amp_cand = float(cand.max() - cand.min())
    peak_ref = ref.idxmax()
    peak_cand = cand.idxmax()
    peak_lag_min = float((peak_cand - peak_ref).total_seconds() / 60.0)

    return PairMetrics(
        reference=reference_name,
        candidate=candidate_name,
        n=int(len(err)),
        bias_C=float(err.mean()),
        mae_C=float(err.abs().mean()),
        rmse_C=float(np.sqrt((err ** 2).mean())),
        corr=corr,
        amp_reference_C=amp_ref,
        amp_candidate_C=amp_cand,
        amp_error_C=float(amp_cand - amp_ref),
        peak_time_reference=str(peak_ref),
        peak_time_candidate=str(peak_cand),
        peak_lag_min=peak_lag_min,
    )



def metrics_to_dict(metric: PairMetrics) -> Dict[str, object]:
    return asdict(metric)



def stats_to_dict(stats: SeriesStats) -> Dict[str, object]:
    return asdict(stats)


# ==============================================================================
# 06 — FORCED SLAB DIAGNOSTIC
# ==============================================================================


def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    ac, bc, cc, dc = map(lambda x: x.astype(float).copy(), (a, b, c, d))
    for i in range(1, n):
        denom = bc[i - 1] if abs(bc[i - 1]) > 1e-30 else 1e-30
        m = ac[i] / denom
        bc[i] = bc[i] - m * cc[i - 1]
        dc[i] = dc[i] - m * dc[i - 1]
    x = np.zeros(n)
    x[-1] = dc[-1] / (bc[-1] if abs(bc[-1]) > 1e-30 else 1e-30)
    for i in range(n - 2, -1, -1):
        denom = bc[i] if abs(bc[i]) > 1e-30 else 1e-30
        x[i] = (dc[i] - cc[i] * x[i + 1]) / denom
    return x



def forced_slab_from_measured_top(
    ni_1min: pd.DataFrame,
    top_col: str,
    target_col: str,
    tin_col: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    H_slab: float,
    lambda_s: float,
    rho_s: float,
    cp_s: float,
    h_in: float,
    Nz: int = 67,
    initial_col: Optional[str] = None,
) -> Tuple[pd.Series, PairMetrics]:
    """
    Slab-only diagnostic.

    Top boundary is forced from a measured NI channel. Bottom boundary uses indoor
    convection to T_in. Then the predicted bottom slab surface is compared with target.
    """
    required = [top_col, tin_col, target_col]
    missing = [c for c in required if c not in ni_1min.columns]
    if missing:
        raise ValueError(f"Missing columns for forced slab: {missing}")

    data = ni_1min[required].dropna().copy()
    if data.empty:
        raise ValueError(f"Empty data for forced slab top={top_col}")

    dz = H_slab / (Nz - 1)
    alpha = lambda_s / (rho_s * cp_s)
    dt_s = 60.0
    Fo = alpha * dt_s / dz ** 2

    if initial_col and initial_col in data.columns and pd.notna(data[initial_col].iloc[0]):
        initial_bottom = float(data[initial_col].iloc[0]) + 273.15
    else:
        initial_bottom = float(data[target_col].iloc[0]) + 273.15
    initial_top = float(data[top_col].iloc[0]) + 273.15
    T = np.linspace(initial_top, initial_bottom, Nz)

    pred = []
    times = []
    for ts, row in data.iterrows():
        top_K = float(row[top_col]) + 273.15
        tin_K = float(row[tin_col]) + 273.15

        a = np.zeros(Nz)
        b = np.zeros(Nz)
        c = np.zeros(Nz)
        d = np.zeros(Nz)

        # Dirichlet top boundary: T[0] = measured top.
        b[0] = 1.0
        d[0] = top_K

        # Interior implicit diffusion.
        for i in range(1, Nz - 1):
            a[i] = -Fo
            b[i] = 1.0 + 2.0 * Fo
            c[i] = -Fo
            d[i] = T[i]

        # Bottom convective boundary: -lambda dT/dz = h(T_surface - T_in).
        # Algebraic boundary equation.
        a[-1] = -lambda_s / dz
        b[-1] = lambda_s / dz + h_in
        d[-1] = h_in * tin_K

        T = tdma(a, b, c, d)
        pred.append(T[-1] - 273.15)
        times.append(ts)

    sim = pd.Series(pred, index=pd.to_datetime(times), name=f"forced_{top_col}_to_{target_col}")
    eval_sim = sim[(sim.index >= eval_start) & (sim.index <= eval_end)]
    eval_obs = ni_1min[target_col][(ni_1min.index >= eval_start) & (ni_1min.index <= eval_end)]
    metric = pair_metrics(eval_obs, eval_sim, target_col, f"forced_slab_from_{top_col}")
    return sim, metric


# ==============================================================================
# 07 — DIAGNOSTIC TABLES
# ==============================================================================


def build_model_sensor_matrix(model_1min: pd.DataFrame, ni_1min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = [c for c in MODEL_PATH if c in model_1min.columns]
    sensor_cols = [c for c in CAM_SENSOR_PATH if c in ni_1min.columns]

    for sensor in sensor_cols:
        for model_col in model_cols:
            m = pair_metrics(ni_1min[sensor], model_1min[model_col], sensor, model_col)
            rows.append(metrics_to_dict(m))
    return pd.DataFrame(rows)



def build_measured_relationship_matrix(ni_1min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sensor_cols = [c for c in CAM_SENSOR_PATH if c in ni_1min.columns]
    for ref in sensor_cols:
        for cand in sensor_cols:
            if ref == cand:
                continue
            m = pair_metrics(ni_1min[ref], ni_1min[cand], ref, cand)
            rows.append(metrics_to_dict(m))
    return pd.DataFrame(rows)



def build_amplitude_summary(model_1min: pd.DataFrame, ni_1min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in CAM_SENSOR_PATH if c in ni_1min.columns]:
        d = stats_to_dict(safe_stats(ni_1min[col], f"measured_{col}"))
        d["source"] = "measured"
        d["variable"] = col
        rows.append(d)
    for col in [c for c in MODEL_PATH if c in model_1min.columns]:
        d = stats_to_dict(safe_stats(model_1min[col], f"model_{col}"))
        d["source"] = "model"
        d["variable"] = col
        rows.append(d)
    return pd.DataFrame(rows)


# ==============================================================================
# 08 — PLOTS
# ==============================================================================


def plot_sensor_path_overlay(
    model_1min: pd.DataFrame,
    ni_1min: pd.DataFrame,
    target_col: str,
    tin_col: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    # Panel 1: measured CAM path.
    for col in [c for c in CAM_SENSOR_PATH if c in ni_1min.columns]:
        axes[0].plot(ni_1min.index, ni_1min[col], label=col, linewidth=1.4)
    axes[0].set_title("Measured CAM NI channels — possible heat path")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    # Panel 2: model path.
    for col in [c for c in MODEL_PATH if c in model_1min.columns]:
        axes[1].plot(model_1min.index, model_1min[col], label=col, linewidth=1.5)
    axes[1].set_title("Model heat path")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)

    # Panel 3: current target against closest physical candidates.
    if target_col in ni_1min.columns:
        axes[2].plot(ni_1min.index, ni_1min[target_col], label=f"Measured target {target_col}", linewidth=2)
    for col in ["T1Ta", "T1Ke", "T2A", "T2A2", tin_col]:
        if col in ni_1min.columns and col != target_col:
            axes[2].plot(ni_1min.index, ni_1min[col], label=f"Measured {col}", linewidth=1.0, alpha=0.8)
    if "T_s_in" in model_1min.columns:
        axes[2].plot(model_1min.index, model_1min["T_s_in"], "--", label="Model T_s_in", linewidth=2)
    if "T_g_bot" in model_1min.columns:
        axes[2].plot(model_1min.index, model_1min["T_g_bot"], "--", label="Model T_g_bot", linewidth=1.4)
    axes[2].set_title("Target sensor check")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3, fontsize=8)

    # Panel 4: drivers.
    if "T_a" in model_1min.columns:
        axes[3].plot(model_1min.index, model_1min["T_a"], label="T_a")
    if "T_in_used" in model_1min.columns:
        axes[3].plot(model_1min.index, model_1min["T_in_used"], label="T_in_used")
    elif tin_col in ni_1min.columns:
        axes[3].plot(ni_1min.index, ni_1min[tin_col], label=f"T_in {tin_col}")
    axes[3].set_ylabel("Temperature (°C)")
    axes[3].set_xlabel("Datetime")
    axes[3].grid(True, alpha=0.3)

    if "G_sol" in model_1min.columns:
        ax2 = axes[3].twinx()
        ax2.plot(model_1min.index, model_1min["G_sol"], linestyle="--", label="G_sol")
        ax2.set_ylabel("Solar (W/m²)")
        lines1, labels1 = axes[3].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[3].legend(lines1 + lines2, labels1 + labels2, ncol=3, fontsize=8)
    else:
        axes[3].legend(ncol=3, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_amplitude_summary(summary: pd.DataFrame, save_path: Path) -> None:
    if summary.empty:
        return
    df = summary.dropna(subset=["amplitude_C"]).copy()
    labels = [f"{s}:{v}" for s, v in zip(df["source"], df["variable"])]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(df)), df["amplitude_C"].to_numpy())
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Amplitude max-min (°C)")
    ax.set_title("Amplitude along measured sensors and model path")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_similarity_heatmap(matrix: pd.DataFrame, save_path: Path, value_col: str = "corr") -> None:
    if matrix.empty or value_col not in matrix.columns:
        return
    piv = matrix.pivot_table(index="reference", columns="candidate", values=value_col, aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title(f"Sensor/model similarity heatmap: {value_col}")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_forced_slab(
    ni_1min: pd.DataFrame,
    forced_predictions: Dict[str, pd.Series],
    target_col: str,
    save_path: Path,
) -> None:
    if target_col not in ni_1min.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(ni_1min.index, ni_1min[target_col], label=f"Measured {target_col}", linewidth=2)
    for name, s in forced_predictions.items():
        axes[0].plot(s.index, s.values, "--", label=name, linewidth=1.4)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Forced slab diagnostic: measured top boundary -> predicted bottom surface")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    for name, s in forced_predictions.items():
        ref, cand = clean_pair(ni_1min[target_col], s)
        if len(ref) > 0:
            axes[1].plot(cand.index, cand - ref, label=name, linewidth=1.2)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Forced slab error (°C)")
    axes[1].set_xlabel("Datetime")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 09 — MAIN WORKFLOW
# ==============================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    model_output_dir: Path = args.model_output_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    forced_start = eval_start - pd.Timedelta(hours=float(args.forced_spinup_hours))

    print("\n=== CAM SENSOR / PATH DIAGNOSTIC ===")
    print(f"BASE_DIR           : {base_dir.resolve()}")
    print(f"MODEL_OUTPUT_DIR   : {model_output_dir.resolve()}")
    print(f"OUTPUT_DIR         : {output_dir.resolve()}")
    print(f"EVAL WINDOW        : {eval_start} -> {eval_end}")

    # Load data.
    ni_paths = resolve_paths(base_dir, args.ni_files)
    print("\nLoading NI files:")
    for p in ni_paths:
        print(f"  - {p}")
    ni = load_multiple_NI_sensor_data(ni_paths)
    sim = load_prediction_csv(model_output_dir, args.prediction_file)

    # Resample / window.
    ni_eval = to_1min(ni, eval_start, eval_end)
    sim_eval = to_1min(sim, eval_start, eval_end)

    # For forced slab, use extra spin-up range if available.
    ni_forced = to_1min(ni, forced_start, eval_end)

    print(f"\nNI eval range       : {ni_eval.index.min()} -> {ni_eval.index.max()} | n={len(ni_eval)}")
    print(f"Model eval range    : {sim_eval.index.min()} -> {sim_eval.index.max()} | n={len(sim_eval)}")

    # Build tables.
    model_sensor_matrix = build_model_sensor_matrix(sim_eval, ni_eval)
    measured_relationships = build_measured_relationship_matrix(ni_eval)
    amplitude_summary = build_amplitude_summary(sim_eval, ni_eval)

    model_sensor_matrix.to_csv(output_dir / "cam_model_vs_sensor_matrix.csv", index=False)
    measured_relationships.to_csv(output_dir / "cam_measured_sensor_relationships.csv", index=False)
    amplitude_summary.to_csv(output_dir / "cam_amplitude_path_summary.csv", index=False)

    # Forced slab candidates.
    forced_rows = []
    forced_predictions: Dict[str, pd.Series] = {}
    for top_col in args.forced_top_cols:
        if top_col not in ni_forced.columns:
            continue
        try:
            pred, met = forced_slab_from_measured_top(
                ni_1min=ni_forced,
                top_col=top_col,
                target_col=args.target_col,
                tin_col=args.tin_col,
                eval_start=eval_start,
                eval_end=eval_end,
                H_slab=args.H_slab,
                lambda_s=args.lambda_s,
                rho_s=args.rho_s,
                cp_s=args.cp_s,
                h_in=args.h_in,
                Nz=67,
                initial_col=args.target_col,
            )
            forced_predictions[f"forced_from_{top_col}"] = pred[(pred.index >= eval_start) & (pred.index <= eval_end)]
            d = metrics_to_dict(met)
            d["top_col"] = top_col
            d["H_slab_m"] = args.H_slab
            forced_rows.append(d)
        except Exception as exc:
            forced_rows.append({"top_col": top_col, "error": str(exc), "H_slab_m": args.H_slab})

    forced_df = pd.DataFrame(forced_rows)
    forced_df.to_csv(output_dir / "cam_forced_slab_candidates.csv", index=False)

    # Thickness sweep: diagnostic only, not calibration.
    sweep_rows = []
    H_values = np.arange(args.thickness_min, args.thickness_max + args.thickness_step / 2, args.thickness_step)
    for top_col in args.forced_top_cols:
        if top_col not in ni_forced.columns:
            continue
        for H in H_values:
            try:
                _, met = forced_slab_from_measured_top(
                    ni_1min=ni_forced,
                    top_col=top_col,
                    target_col=args.target_col,
                    tin_col=args.tin_col,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    H_slab=float(H),
                    lambda_s=args.lambda_s,
                    rho_s=args.rho_s,
                    cp_s=args.cp_s,
                    h_in=args.h_in,
                    Nz=67,
                    initial_col=args.target_col,
                )
                d = metrics_to_dict(met)
                d["top_col"] = top_col
                d["H_slab_m"] = float(H)
                sweep_rows.append(d)
            except Exception as exc:
                sweep_rows.append({"top_col": top_col, "H_slab_m": float(H), "error": str(exc)})

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "cam_forced_slab_thickness_sweep.csv", index=False)

    # Plots.
    plot_sensor_path_overlay(
        model_1min=sim_eval,
        ni_1min=ni_eval,
        target_col=args.target_col,
        tin_col=args.tin_col,
        save_path=output_dir / "cam_sensor_path_overlay.png",
    )
    plot_amplitude_summary(amplitude_summary, output_dir / "cam_amplitude_path_bar.png")
    plot_similarity_heatmap(model_sensor_matrix, output_dir / "cam_model_vs_sensor_corr_heatmap.png", value_col="corr")
    plot_similarity_heatmap(model_sensor_matrix, output_dir / "cam_model_vs_sensor_rmse_heatmap.png", value_col="rmse_C")
    plot_forced_slab(ni_eval, forced_predictions, args.target_col, output_dir / "cam_forced_slab_diagnostic.png")

    # Summary recommendations.
    summary: Dict[str, object] = {
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "target_col": args.target_col,
        "tin_col": args.tin_col,
    }

    if not model_sensor_matrix.empty:
        target_rows = model_sensor_matrix[model_sensor_matrix["reference"] == args.target_col].copy()
        if not target_rows.empty:
            target_rows_valid = target_rows.dropna(subset=["rmse_C"])
            if not target_rows_valid.empty:
                best_model_rmse = target_rows_valid.sort_values("rmse_C").iloc[0].to_dict()
                summary["best_model_node_for_target_by_rmse"] = best_model_rmse

            target_rows_valid_corr = target_rows.dropna(subset=["corr"])
            if not target_rows_valid_corr.empty:
                best_model_corr = target_rows_valid_corr.sort_values("corr", ascending=False).iloc[0].to_dict()
                summary["best_model_node_for_target_by_corr"] = best_model_corr

    if not measured_relationships.empty:
        rel = measured_relationships[measured_relationships["reference"] == args.target_col].dropna(subset=["rmse_C"])
        if not rel.empty:
            summary["measured_sensor_most_similar_to_target_by_rmse"] = rel.sort_values("rmse_C").iloc[0].to_dict()
            summary["measured_sensor_most_similar_to_target_by_corr"] = rel.dropna(subset=["corr"]).sort_values("corr", ascending=False).iloc[0].to_dict()

    if not forced_df.empty and "rmse_C" in forced_df.columns:
        fvalid = forced_df.dropna(subset=["rmse_C"])
        if not fvalid.empty:
            summary["best_forced_slab_candidate_at_measured_H"] = fvalid.sort_values("rmse_C").iloc[0].to_dict()

    if not sweep_df.empty and "rmse_C" in sweep_df.columns:
        svalid = sweep_df.dropna(subset=["rmse_C"])
        if not svalid.empty:
            summary["best_forced_slab_effective_thickness"] = svalid.sort_values("rmse_C").iloc[0].to_dict()

    # Plain-language diagnostic conclusion.
    notes: List[str] = []
    amp_df = amplitude_summary.copy()
    if not amp_df.empty:
        target_amp = amp_df[(amp_df["source"] == "measured") & (amp_df["variable"] == args.target_col)]["amplitude_C"]
        model_amp = amp_df[(amp_df["source"] == "model") & (amp_df["variable"] == "T_s_in")]["amplitude_C"]
        gtop_amp = amp_df[(amp_df["source"] == "model") & (amp_df["variable"] == "T_g_top")]["amplitude_C"]
        if not target_amp.empty and not model_amp.empty:
            ta = float(target_amp.iloc[0])
            ma = float(model_amp.iloc[0])
            notes.append(f"Measured {args.target_col} amplitude = {ta:.2f} C; model T_s_in amplitude = {ma:.2f} C.")
            if ma < 0.5 * ta:
                notes.append("Model T_s_in is strongly damped relative to the measured target.")
        if not target_amp.empty and not gtop_amp.empty:
            ga = float(gtop_amp.iloc[0])
            notes.append(f"Model T_g_top amplitude = {ga:.2f} C.")
            if ga < 0.5 * float(target_amp.iloc[0]):
                notes.append("Amplitude is already lost near the modeled substrate/top boundary, not only in slab conduction.")
            else:
                notes.append("Modeled upper layer has amplitude; damping likely occurs between substrate and T_s_in or due to target mismatch.")

    if "best_forced_slab_effective_thickness" in summary:
        best_H = summary["best_forced_slab_effective_thickness"].get("H_slab_m", np.nan)
        if np.isfinite(best_H):
            notes.append(f"Best forced-slab effective thickness in sweep = {best_H:.3f} m. Compare this with measured H_slab={args.H_slab:.3f} m.")
            if best_H < 0.5 * args.H_slab:
                notes.append("If only a much thinner effective slab matches the target, the target sensor may not represent the bottom of the measured slab thickness.")

    summary["diagnostic_notes"] = notes

    with open(output_dir / "cam_path_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== KEY OUTPUTS ===")
    for fname in [
        "cam_sensor_path_overlay.png",
        "cam_amplitude_path_bar.png",
        "cam_model_vs_sensor_matrix.csv",
        "cam_measured_sensor_relationships.csv",
        "cam_forced_slab_candidates.csv",
        "cam_forced_slab_thickness_sweep.csv",
        "cam_forced_slab_diagnostic.png",
        "cam_path_diagnostic_summary.json",
    ]:
        print(f"  {output_dir / fname}")

    print("\n=== DIAGNOSTIC NOTES ===")
    for note in notes:
        print(f"- {note}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
