
"""
================================================================================
CAM NODE–SENSOR TARGET MATRIX
================================================================================
Purpose
-------
Compare measured CAM NI sensors against model nodes to decide which measured
sensor should be paired with which physical model output.

Main question:
    Should T2A be compared with model T_g_bot or model T_s_in?

This script computes, for each measured sensor vs each model node:
    - RMSE
    - Bias
    - MAE
    - Correlation
    - Robust amplitude error
    - Best lag and lagged correlation

It also creates a focused report for T2A.

This script does NOT calibrate anything.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

NI_CHANNEL_NAMES = [
    "timestamp_serial",
    "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
    "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
    "T2Ka", "T2Kd", "T2Kc", "2Ke",  "T1A",
    "T2A",  "T2A2", "T1Tb", "T1Ta", "T1Ta2"
]

DEFAULT_SENSORS = ["T1Ka", "T1Ke", "T1Ta", "T1Tb", "T2A", "T2A2"]

PREFERRED_MODEL_NODE_ORDER = [
    "T_f",
    "T_g_top",
    "T_g_mid",
    "T_g_bot",
    "T_s_top",
    "T_s_in",
    "T_in_used",
]


# ==============================================================================
# SECTION 2: SAFE HELPERS
# ==============================================================================

def numeric_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(axis=1, how="all")


def safe_resample_1min(df: pd.DataFrame, interpolate_limit: int = 30) -> pd.DataFrame:
    out = numeric_only_frame(df)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.resample("1min").mean(numeric_only=True)
    out = out.interpolate(method="time", limit=interpolate_limit)
    return out


def robust_amp(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 5:
        return np.nan
    return float(x.quantile(0.95) - x.quantile(0.05))


def align_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return x["a"], x["b"]


def corr(a: pd.Series, b: pd.Series) -> float:
    x, y = align_series(a, b)
    if len(x) < 5:
        return np.nan
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x.values, y.values)[0, 1])


def rmse(a: pd.Series, b: pd.Series) -> float:
    x, y = align_series(a, b)
    if len(x) < 5:
        return np.nan
    return float(np.sqrt(np.mean((x.values - y.values) ** 2)))


def mae(a: pd.Series, b: pd.Series) -> float:
    x, y = align_series(a, b)
    if len(x) < 5:
        return np.nan
    return float(np.mean(np.abs(x.values - y.values)))


def bias(model: pd.Series, sensor: pd.Series) -> float:
    x, y = align_series(model, sensor)
    if len(x) < 5:
        return np.nan
    return float(np.mean(x.values - y.values))


def best_lag_metrics(sensor: pd.Series,
                     model: pd.Series,
                     max_lag_min: int = 360,
                     step_min: int = 5) -> Tuple[float, int, float]:
    """
    Find the lag that maximizes absolute correlation.
    Positive lag means model is shifted later relative to sensor.

    Returns:
        best_corr, best_lag_min, rmse_at_best_lag
    """
    best_c = np.nan
    best_lag = 0
    best_rmse = np.nan

    for lag in range(-max_lag_min, max_lag_min + 1, step_min):
        shifted_model = model.shift(lag, freq="1min")
        c = corr(sensor, shifted_model)
        if np.isfinite(c) and (not np.isfinite(best_c) or abs(c) > abs(best_c)):
            best_c = c
            best_lag = lag
            best_rmse = rmse(sensor, shifted_model)

    return (
        float(best_c) if np.isfinite(best_c) else np.nan,
        int(best_lag),
        float(best_rmse) if np.isfinite(best_rmse) else np.nan,
    )


# ==============================================================================
# SECTION 3: LOADERS
# ==============================================================================

def read_ni_xlsx_xml(filepath: Path) -> pd.DataFrame:
    filepath = Path(filepath)
    with zipfile.ZipFile(filepath, "r") as z:
        sheet_names = [
            n for n in z.namelist()
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError(f"No worksheet XML found in {filepath}")
        with z.open(sheet_names[0]) as f:
            xml_content = f.read()

    root = ET.fromstring(xml_content)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows_xml = root.findall(f".//{ns}row")

    data = []
    for row in rows_xml[1:]:
        vals = []
        for cell in row.findall(f"{ns}c"):
            v = cell.find(f"{ns}v")
            try:
                vals.append(float(v.text) if v is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        if len(vals) >= 21:
            data.append(vals[:21])

    if not data:
        raise ValueError(f"No numeric rows found in {filepath}")

    df = pd.DataFrame(data, columns=NI_CHANNEL_NAMES)

    labview_epoch = pd.Timestamp("1904-01-01")
    df["timestamp"] = labview_epoch + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"])
    df = df.sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[(df[c] < -20) | (df[c] > 100), c] = np.nan

    return df


def load_ni_files(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        f = Path(f)
        if not f.exists():
            warnings.warn(f"NI file not found: {f}")
            continue
        print(f"Loading NI: {f}")
        frames.append(read_ni_xlsx_xml(f))

    if not frames:
        raise FileNotFoundError("No NI files could be loaded.")

    df = pd.concat(frames).sort_index()
    return safe_resample_1min(df, interpolate_limit=30)


def load_model_prediction(model_output_dir: Path,
                          explicit_file: Optional[str] = None) -> pd.DataFrame:
    model_output_dir = Path(model_output_dir)

    candidates = []
    if explicit_file:
        candidates.append(model_output_dir / explicit_file)

    candidates += [
        model_output_dir / "cam_gsw_prediction_eval_window.csv",
        model_output_dir / "cam_gsw_prediction_full_with_spinup.csv",
        model_output_dir / "cam_physical_prediction_eval_window.csv",
        model_output_dir / "cam_prediction_calibrated.csv",
        model_output_dir / "cam_prediction_baseline.csv",
    ]

    found = None
    for f in candidates:
        if f.exists():
            found = f
            break

    if found is None:
        raise FileNotFoundError(
            f"No model prediction CSV found in {model_output_dir}. "
            "Expected cam_gsw_prediction_eval_window.csv or pass --model-file."
        )

    print(f"Loading model prediction: {found}")
    df = pd.read_csv(found)

    time_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ["datetime", "timestamp", "time", "date_time"]:
            time_col = c
            break

    if time_col is None:
        # Try first column
        time_col = df.columns[0]

    idx = pd.to_datetime(df[time_col], errors="coerce")
    df = df.drop(columns=[time_col])
    df.index = idx
    df = df[~df.index.isna()]
    return safe_resample_1min(df, interpolate_limit=30)


def select_model_nodes(model: pd.DataFrame) -> List[str]:
    available = []
    for n in PREFERRED_MODEL_NODE_ORDER:
        if n in model.columns:
            available.append(n)

    # Extra fallback: include other temperature-like model columns
    for c in model.columns:
        if c not in available and (
            str(c).startswith("T_") or "temp" in str(c).lower()
        ):
            available.append(c)

    return available


# ==============================================================================
# SECTION 4: METRICS TABLES
# ==============================================================================

def compute_matrix(ni: pd.DataFrame,
                   model: pd.DataFrame,
                   sensors: List[str],
                   model_nodes: List[str],
                   max_lag_min: int) -> pd.DataFrame:
    rows = []

    for sensor in sensors:
        if sensor not in ni.columns:
            continue
        for node in model_nodes:
            if node not in model.columns:
                continue

            s = ni[sensor]
            m = model[node]

            base_rmse = rmse(s, m)
            base_corr = corr(s, m)
            best_c, best_lag, best_lag_rmse = best_lag_metrics(
                sensor=s,
                model=m,
                max_lag_min=max_lag_min,
                step_min=5,
            )

            amp_s = robust_amp(s)
            amp_m = robust_amp(m)

            rows.append({
                "sensor": sensor,
                "model_node": node,
                "rmse_C": base_rmse,
                "mae_C": mae(s, m),
                "bias_model_minus_sensor_C": bias(m, s),
                "corr_no_lag": base_corr,
                "best_abs_corr": best_c,
                "best_lag_min_model_shift": best_lag,
                "rmse_at_best_lag_C": best_lag_rmse,
                "sensor_amp_95_05_C": amp_s,
                "model_amp_95_05_C": amp_m,
                "amp_error_model_minus_sensor_C": amp_m - amp_s if np.isfinite(amp_s) and np.isfinite(amp_m) else np.nan,
                "n_overlap": len(pd.concat([s.rename("s"), m.rename("m")], axis=1).dropna()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ranking: low RMSE + high abs corr + low amplitude error
    # Normalize simply, heuristic.
    rmse_med = out["rmse_C"].median(skipna=True)
    amp_med = out["amp_error_model_minus_sensor_C"].abs().median(skipna=True)
    rmse_scale = rmse_med if np.isfinite(rmse_med) and rmse_med > 0 else 3.0
    amp_scale = amp_med if np.isfinite(amp_med) and amp_med > 0 else 5.0

    out["fit_score"] = (
        np.exp(-out["rmse_C"] / rmse_scale)
        + out["best_abs_corr"].abs().fillna(0)
        + np.exp(-out["amp_error_model_minus_sensor_C"].abs() / amp_scale)
    )

    out["rank_for_sensor"] = out.groupby("sensor")["fit_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return out.sort_values(["sensor", "rank_for_sensor", "model_node"])


# ==============================================================================
# SECTION 5: PLOTS
# ==============================================================================

def plot_matrix_heatmap(table: pd.DataFrame,
                        value_col: str,
                        outpath: Path,
                        title: str) -> None:
    if table.empty:
        return

    pivot = table.pivot(index="sensor", columns="model_node", values=value_col)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values.astype(float), aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_sensor_vs_top_nodes(ni: pd.DataFrame,
                             model: pd.DataFrame,
                             table: pd.DataFrame,
                             sensor: str,
                             outpath: Path,
                             top_n: int = 4) -> None:
    if sensor not in ni.columns:
        return

    sub = table[table["sensor"] == sensor].sort_values("rank_for_sensor").head(top_n)
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(ni.index, ni[sensor], lw=2.0, label=f"Measured {sensor}", color="black")

    for _, row in sub.iterrows():
        node = row["model_node"]
        if node in model.columns:
            ax.plot(model.index, model[node], lw=1.2, label=f"{node} | rank {row['rank_for_sensor']} | RMSE {row['rmse_C']:.2f}")

    ax.set_title(f"Measured {sensor} vs best matching model nodes")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_all_sensor_best_matches(ni: pd.DataFrame,
                                 model: pd.DataFrame,
                                 table: pd.DataFrame,
                                 outpath: Path) -> None:
    sensors = list(table["sensor"].unique())
    fig, axes = plt.subplots(len(sensors), 1, figsize=(15, 3.0 * len(sensors)), sharex=True)
    if len(sensors) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensors):
        if sensor not in ni.columns:
            continue
        best = table[table["sensor"] == sensor].sort_values("rank_for_sensor").head(1)
        ax.plot(ni.index, ni[sensor], color="black", lw=1.5, label=f"Measured {sensor}")
        if not best.empty:
            node = best.iloc[0]["model_node"]
            if node in model.columns:
                ax.plot(model.index, model[node], lw=1.2, label=f"Best node: {node}")
                ax.set_title(
                    f"{sensor}: best={node}, RMSE={best.iloc[0]['rmse_C']:.2f}°C, "
                    f"corr={best.iloc[0]['corr_no_lag']:.2f}, amp_err={best.iloc[0]['amp_error_model_minus_sensor_C']:.2f}°C"
                )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ==============================================================================
# SECTION 6: MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAM NI sensors against model nodes to choose correct validation target."
    )
    parser.add_argument("--base-dir", default=".", help="Folder containing NI files and model output folder")
    parser.add_argument(
        "--ni-files",
        nargs="*",
        default=[
            "Pengukuran 30_1 Maret 2026.xlsx",
            "Pengukuran 30_2 Maret 2026.xlsx",
            "Pengukuran 3 April 2026.xlsx",
        ],
        help="NI XLSX files",
    )
    parser.add_argument("--model-output-dir", default="outputs_cam_gsw_fixed_inputs", help="Folder containing model prediction CSV")
    parser.add_argument("--model-file", default=None, help="Specific model CSV filename inside model-output-dir")
    parser.add_argument("--start", default="2026-03-31 11:58:00", help="Evaluation window start")
    parser.add_argument("--end", default="2026-04-02 21:42:00", help="Evaluation window end")
    parser.add_argument("--sensors", nargs="*", default=DEFAULT_SENSORS, help="Measured sensors to compare")
    parser.add_argument("--focus-sensor", default="T2A", help="Sensor to make focused report/plot for")
    parser.add_argument("--max-lag-min", type=int, default=360, help="Maximum lag scan in minutes")
    parser.add_argument("--output-dir", default="outputs_cam_node_target_matrix", help="Output folder")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    ni_paths = [base_dir / f for f in args.ni_files]
    ni = load_ni_files(ni_paths)
    ni = ni[(ni.index >= start) & (ni.index <= end)].copy()

    if ni.empty:
        raise ValueError(f"No NI data inside {start} -> {end}")

    model = load_model_prediction(base_dir / args.model_output_dir, explicit_file=args.model_file)
    model = model[(model.index >= start) & (model.index <= end)].copy()

    if model.empty:
        raise ValueError(f"No model data inside {start} -> {end}")

    sensors = [s for s in args.sensors if s in ni.columns]
    model_nodes = select_model_nodes(model)

    print("\nMeasured sensors:", ", ".join(sensors))
    print("Model nodes:", ", ".join(model_nodes))
    print(f"Window NI   : {ni.index.min()} -> {ni.index.max()} | rows={len(ni)}")
    print(f"Window model: {model.index.min()} -> {model.index.max()} | rows={len(model)}")

    table = compute_matrix(
        ni=ni,
        model=model,
        sensors=sensors,
        model_nodes=model_nodes,
        max_lag_min=args.max_lag_min,
    )

    if table.empty:
        raise ValueError("No metrics could be computed. Check model nodes and sensor names.")

    table.to_csv(out_dir / "cam_node_sensor_full_metrics.csv", index=False)

    # Focus report
    focus = table[table["sensor"] == args.focus_sensor].sort_values("rank_for_sensor")
    focus.to_csv(out_dir / f"cam_{args.focus_sensor}_candidate_model_nodes.csv", index=False)

    # Best per sensor
    best = table.sort_values("rank_for_sensor").groupby("sensor", as_index=False).first()
    best.to_csv(out_dir / "cam_best_model_node_per_sensor.csv", index=False)

    # Heatmaps
    plot_matrix_heatmap(
        table, "rmse_C", out_dir / "cam_node_sensor_rmse_heatmap.png",
        "RMSE: measured sensor vs model node (°C)"
    )
    plot_matrix_heatmap(
        table, "corr_no_lag", out_dir / "cam_node_sensor_corr_heatmap.png",
        "Correlation: measured sensor vs model node"
    )
    plot_matrix_heatmap(
        table, "amp_error_model_minus_sensor_C", out_dir / "cam_node_sensor_amp_error_heatmap.png",
        "Amplitude error: model amp - sensor amp (°C)"
    )
    plot_matrix_heatmap(
        table, "fit_score", out_dir / "cam_node_sensor_fit_score_heatmap.png",
        "Heuristic fit score"
    )

    # Focus plots
    plot_sensor_vs_top_nodes(
        ni, model, table, sensor=args.focus_sensor,
        outpath=out_dir / f"cam_{args.focus_sensor}_vs_candidate_model_nodes.png"
    )
    plot_all_sensor_best_matches(
        ni, model, table,
        outpath=out_dir / "cam_all_sensors_best_model_matches.png"
    )

    summary = {
        "window": {"start": str(start), "end": str(end)},
        "focus_sensor": args.focus_sensor,
        "focus_sensor_ranked_candidates": focus.head(10).to_dict("records"),
        "best_model_node_per_sensor": best.to_dict("records"),
        "interpretation_notes": [
            "T2A should be first compared to T_g_bot if it is physically lower substrate.",
            "If T2A matches T_s_in better than T_g_bot, the original sensor mapping may be wrong or T2A may be closer to slab/underside behavior.",
            "Use RMSE, correlation, amplitude error, and lag together; do not rely on a single metric.",
            "This script diagnoses target mapping only; it does not calibrate the model."
        ],
    }
    with open(out_dir / "cam_node_target_matrix_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print(f"\n{args.focus_sensor} ranked model-node candidates:")
    cols = [
        "model_node", "rank_for_sensor", "rmse_C", "corr_no_lag",
        "best_abs_corr", "best_lag_min_model_shift",
        "sensor_amp_95_05_C", "model_amp_95_05_C",
        "amp_error_model_minus_sensor_C"
    ]
    print(focus[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
