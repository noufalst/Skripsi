
from __future__ import annotations

import argparse
import json
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: NI CHANNEL CONFIGURATION
# ==============================================================================

NI_CHANNEL_NAMES = [
    "timestamp_serial",
    "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
    "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
    "T2Ka", "T2Kd", "T2Kc", "2Ke", "T1A",
    "T2A", "T2A2", "T1Tb", "T1Ta", "T1Ta2",
]

DEFAULT_NI_FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]

# Conservative C3 candidates based on the NI header family.
# This script does NOT assume the mapping is proven.
DEFAULT_C3_SENSOR_CANDIDATES = [
    "T3Ka", "T3Kb", "T3Kc", "T3Kd", "T3Ke",
    "T2Ka", "T2Kc", "T2Kd", "2Ke", "T1A",
]

MODEL_NODE_CANDIDATES = [
    "T_s_in", "T_s_top", "T_g_bot", "T_g_mid", "T_g_top", "T_in_used", "T_a",
]


# ==============================================================================
# SECTION 2: BASIC HELPERS
# ==============================================================================

def robust_amp(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 5:
        return np.nan
    return float(x.quantile(0.95) - x.quantile(0.05))


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


def align(measured: pd.Series, model: pd.Series) -> pd.DataFrame:
    return pd.concat([measured.rename("measured"), model.rename("model")], axis=1).dropna()


def corr(a: pd.Series, b: pd.Series) -> float:
    x = align(a, b)
    if len(x) < 5:
        return np.nan
    if x["measured"].std() == 0 or x["model"].std() == 0:
        return np.nan
    return float(np.corrcoef(x["measured"], x["model"])[0, 1])


def metrics(measured: pd.Series, model: pd.Series, max_lag_min: int = 360) -> dict:
    x = align(measured, model)
    if len(x) < 5:
        return {
            "available": False,
            "n": len(x),
            "rmse_C": np.nan,
            "mae_C": np.nan,
            "bias_C": np.nan,
            "corr": np.nan,
            "best_abs_corr": np.nan,
            "best_lag_min_model_shift": np.nan,
            "rmse_at_best_lag_C": np.nan,
            "measured_amp_C": np.nan,
            "model_amp_C": np.nan,
            "amp_error_C": np.nan,
            "measured_peak_C": np.nan,
            "model_peak_C": np.nan,
            "peak_error_C": np.nan,
        }

    err = x["model"] - x["measured"]
    measured_amp = robust_amp(x["measured"])
    model_amp = robust_amp(x["model"])

    best_c = np.nan
    best_lag = 0
    best_rmse = np.nan
    for lag in range(-max_lag_min, max_lag_min + 1, 5):
        shifted = x["model"].shift(lag, freq="1min")
        xx = align(x["measured"], shifted)
        if len(xx) < 5:
            continue
        c = corr(xx["measured"], xx["model"])
        if np.isfinite(c) and (not np.isfinite(best_c) or abs(c) > abs(best_c)):
            best_c = c
            best_lag = lag
            ee = xx["model"] - xx["measured"]
            best_rmse = float(np.sqrt(np.mean(ee.to_numpy() ** 2)))

    return {
        "available": True,
        "n": int(len(x)),
        "rmse_C": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "mae_C": float(np.mean(np.abs(err.to_numpy()))),
        "bias_C": float(err.mean()),
        "corr": corr(x["measured"], x["model"]),
        "best_abs_corr": float(best_c) if np.isfinite(best_c) else np.nan,
        "best_lag_min_model_shift": int(best_lag),
        "rmse_at_best_lag_C": float(best_rmse) if np.isfinite(best_rmse) else np.nan,
        "measured_amp_C": measured_amp,
        "model_amp_C": model_amp,
        "amp_error_C": model_amp - measured_amp if np.isfinite(measured_amp) and np.isfinite(model_amp) else np.nan,
        "measured_mean_C": float(x["measured"].mean()),
        "model_mean_C": float(x["model"].mean()),
        "measured_peak_C": float(x["measured"].max()),
        "model_peak_C": float(x["model"].max()),
        "peak_error_C": float(x["model"].max() - x["measured"].max()),
        "measured_peak_time": str(x["measured"].idxmax()),
        "model_peak_time": str(x["model"].idxmax()),
        "peak_lag_min": float((x["model"].idxmax() - x["measured"].idxmax()).total_seconds() / 60),
    }


def bias_correct(measured: pd.Series, model: pd.Series) -> tuple[pd.Series, float]:
    x = align(measured, model)
    if len(x) < 5:
        return model.copy() * np.nan, np.nan
    bias = float((x["model"] - x["measured"]).mean())
    return model - bias, bias


# ==============================================================================
# SECTION 3: LOADERS
# ==============================================================================

def read_ni_xlsx_xml(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path, "r") as z:
        sheet_names = [
            n for n in z.namelist()
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError(f"No worksheet XML found in {path}")
        xml = z.read(sheet_names[0])

    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows_xml = root.findall(f".//{ns}row")

    rows = []
    for row in rows_xml[1:]:
        vals = []
        for cell in row.findall(f"{ns}c"):
            v = cell.find(f"{ns}v")
            try:
                vals.append(float(v.text) if v is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        if len(vals) >= 21:
            rows.append(vals[:21])

    if not rows:
        raise ValueError(f"No numeric NI rows found in {path}")

    df = pd.DataFrame(rows, columns=NI_CHANNEL_NAMES)
    df["timestamp"] = pd.Timestamp("1904-01-01") + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"]).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[(df[c] < -20) | (df[c] > 100), c] = np.nan

    return df


def load_ni(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        f = Path(f)
        if not f.exists():
            warnings.warn(f"NI file not found: {f}")
            continue
        print(f"Loading NI: {f}")
        frames.append(read_ni_xlsx_xml(f))

    if not frames:
        raise FileNotFoundError("No NI files were loaded.")

    return safe_resample_1min(pd.concat(frames).sort_index())



def _score_model_candidate(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if "c3" in s:
        score += 50
    if "prediction" in s:
        score += 30
    if "eval_window" in s:
        score += 20
    if "full_with_spinup" in s:
        score += 10
    if "output" in s:
        score += 5
    if "cam" in s and "c3" not in s:
        score -= 30
    if "metrics" in s or "summary" in s or "timeseries" in s:
        score -= 20
    return score


def find_model_csv_candidates(base_dir: Path) -> list[Path]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    candidates = []
    for p in base_dir.rglob("*.csv"):
        name = p.name.lower()
        parent = str(p.parent).lower()
        if any(k in name for k in ["prediction", "eval_window", "full_with_spinup"]):
            candidates.append(p)
        elif "outputs" in parent and any(k in name for k in ["c3", "model", "sim"]):
            candidates.append(p)

    return sorted(candidates, key=lambda p: _score_model_candidate(p), reverse=True)


def load_model(folder: Path, model_file: str | None = None, base_dir: Path | None = None) -> tuple[pd.DataFrame, Path]:
    folder = Path(folder)
    base_dir = Path(base_dir) if base_dir is not None else Path(".")

    candidates = []
    if model_file:
        mf = Path(model_file)
        if mf.exists():
            candidates.append(mf)
        candidates.append(folder / model_file)

    candidates += [
        folder / "c3_gsw_prediction_eval_window.csv",
        folder / "c3_gsw_prediction_full_with_spinup.csv",
        folder / "c3_prediction_eval_window.csv",
        folder / "c3_prediction_full_with_spinup.csv",
        folder / "c3_prediction_calibrated.csv",
        folder / "c3_prediction_baseline.csv",
        folder / "prediction_eval_window.csv",
        folder / "final_prediction_eval_window.csv",
    ]

    found = next((p for p in candidates if p.exists()), None)

    if found is None:
        recursive_candidates = find_model_csv_candidates(base_dir)
        if recursive_candidates:
            print("\nNo exact C3 model CSV found in the requested folder.")
            print("Using the best auto-detected candidate instead:")
            print(f"  {recursive_candidates[0]}")
            print("\nOther candidates:")
            for p in recursive_candidates[:15]:
                print(f"  score={_score_model_candidate(p):>3} | {p}")
            found = recursive_candidates[0]
        else:
            raise FileNotFoundError(
                f"No C3 model prediction CSV found in {folder} and no candidates found under {base_dir}.\n\n"
                "This means the C3 simulation has probably not been run yet, or the output CSV has a different name.\n\n"
                "Quick checks on Windows:\n"
                "  dir /s /b *.csv\n"
                "  dir /s /b *prediction*.csv\n"
                "  dir /s /b *c3*.csv\n\n"
                "Then rerun with:\n"
                "  python run_c3_quick_validation.py --model-output-dir \"FOLDER_NAME\" --model-file \"CSV_NAME.csv\"\n"
            )

    print(f"Loading C3 model: {found}")
    df = pd.read_csv(found)

    time_col = None
    for c in df.columns:
        if str(c).lower() in ["datetime", "timestamp", "time", "date_time"]:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]

    idx = pd.to_datetime(df[time_col], errors="coerce")
    df = df.drop(columns=[time_col])
    df.index = idx
    df = df[~df.index.isna()]

    return safe_resample_1min(df), found


def auto_find_model_folder(base_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return base_dir / explicit

    candidates = [
        "outputs_c3_gsw_fixed_inputs",
        "outputs_c3_model",
        "outputs_c3_validation",
        "outputs_c3_final_validation",
        "outputs_c3",
        "outputs",
    ]

    for c in candidates:
        p = base_dir / c
        if p.exists():
            return p

    return base_dir / "outputs_c3_gsw_fixed_inputs"


# ==============================================================================
# SECTION 4: TARGET MATRIX AND PLOTS
# ==============================================================================

def build_target_matrix(ni: pd.DataFrame,
                        model: pd.DataFrame,
                        sensors: list[str],
                        nodes: list[str],
                        max_lag_min: int) -> pd.DataFrame:
    rows = []

    for sensor in sensors:
        if sensor not in ni.columns:
            continue
        for node in nodes:
            if node not in model.columns:
                continue

            raw = metrics(ni[sensor], model[node], max_lag_min=max_lag_min)
            corrected, b = bias_correct(ni[sensor], model[node])
            bc = metrics(ni[sensor], corrected, max_lag_min=max_lag_min)

            row = {
                "sensor": sensor,
                "model_node": node,
                **{f"raw_{k}": v for k, v in raw.items()},
                "bias_correction_C": b,
                **{f"bc_{k}": v for k, v in bc.items()},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rmse_scale = df["bc_rmse_C"].median(skipna=True)
    if not np.isfinite(rmse_scale) or rmse_scale <= 0:
        rmse_scale = 1.0

    amp_scale = df["bc_amp_error_C"].abs().median(skipna=True)
    if not np.isfinite(amp_scale) or amp_scale <= 0:
        amp_scale = 1.0

    df["bc_fit_score"] = (
        np.exp(-df["bc_rmse_C"] / rmse_scale)
        + df["bc_corr"].abs().fillna(0)
        + np.exp(-df["bc_amp_error_C"].abs() / amp_scale)
    )

    df["rank_for_model_node"] = df.groupby("model_node")["bc_fit_score"].rank(ascending=False, method="min")
    df["rank_for_sensor"] = df.groupby("sensor")["bc_fit_score"].rank(ascending=False, method="min")

    return df.sort_values(["model_node", "rank_for_model_node", "sensor"])


def choose_target(matrix: pd.DataFrame,
                  target_node: str,
                  target_sensor: str | None = None) -> tuple[str, str]:
    if target_sensor:
        sub = matrix[(matrix["sensor"] == target_sensor) & (matrix["model_node"] == target_node)]
        if not sub.empty:
            return target_sensor, target_node

    sub = matrix[matrix["model_node"] == target_node].copy()
    if sub.empty:
        # fallback: best overall
        best = matrix.sort_values("bc_fit_score", ascending=False).iloc[0]
        return str(best["sensor"]), str(best["model_node"])

    best = sub.sort_values("bc_fit_score", ascending=False).iloc[0]
    return str(best["sensor"]), target_node


def plot_target(ts: pd.DataFrame,
                measured_col: str,
                model_col: str,
                title: str,
                outpath: Path) -> tuple[dict, dict, pd.Series]:
    measured = ts[measured_col]
    model = ts[model_col]
    model_bc, b = bias_correct(measured, model)

    raw = metrics(measured, model)
    bc = metrics(measured, model_bc)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(ts.index, measured, label=f"Measured {measured_col}")
    axes[0].plot(ts.index, model, label=f"Model {model_col} raw")
    axes[0].plot(ts.index, model_bc, "--", label=f"Bias-corrected model ({b:.2f} °C removed)")
    axes[0].set_title(title)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].text(
        0.01, 0.97,
        f"Bias-corrected\nRMSE={bc['rmse_C']:.2f}°C\nCorr={bc['corr']:.2f}\nAmpErr={bc['amp_error_C']:.2f}°C",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
        fontsize=9,
    )

    axes[1].plot(ts.index, model - measured, label="Raw error")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Raw error (°C)")

    axes[2].plot(ts.index, model_bc - measured, label="Bias-corrected error")
    axes[2].axhline(0, linewidth=0.8)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("Corrected error (°C)")
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    return raw, bc, model_bc


def plot_matrix_heatmap(matrix: pd.DataFrame, value_col: str, outpath: Path, title: str):
    if matrix.empty or value_col not in matrix.columns:
        return

    pivot = matrix.pivot_table(index="sensor", columns="model_node", values=value_col, aggfunc="first")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.values.astype(float), aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


# ==============================================================================
# SECTION 5: MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="C3 validation / target-finding script.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--ni-files", nargs="*", default=DEFAULT_NI_FILES)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--start", default="2026-04-09 11:05:00")
    parser.add_argument("--end", default="2026-04-10 14:08:00")
    parser.add_argument("--sensors", nargs="*", default=DEFAULT_C3_SENSOR_CANDIDATES)
    parser.add_argument("--target-node", default="T_s_in")
    parser.add_argument("--target-sensor", default=None, help="Optional forced C3 measured target sensor")
    parser.add_argument("--output-dir", default="outputs_c3_stack_validation")
    parser.add_argument("--max-lag-min", type=int, default=360)
    parser.add_argument("--list-model-csvs", action="store_true", help="List possible model CSVs under base-dir and exit")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.list_model_csvs:
        candidates = find_model_csv_candidates(base_dir)
        print(f"Found {len(candidates)} possible model CSV candidates under {base_dir}:")
        for p in candidates[:50]:
            print(f"  score={_score_model_candidate(p):>3} | {p}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    ni = load_ni([base_dir / f for f in args.ni_files])
    ni = ni[(ni.index >= start) & (ni.index <= end)].copy()
    if ni.empty:
        raise ValueError(
            f"No NI data inside selected C3 window {start} -> {end}. "
            "Set --start and --end to your C3 period."
        )

    model_folder = auto_find_model_folder(base_dir, args.model_output_dir)
    model, model_file = load_model(model_folder, model_file=args.model_file, base_dir=base_dir)
    model = model[(model.index >= start) & (model.index <= end)].copy()
    if model.empty:
        raise ValueError(
            f"No model data inside selected C3 window {start} -> {end}. "
            "Check C3 model output time range."
        )

    sensors = [s for s in args.sensors if s in ni.columns]
    nodes = [n for n in MODEL_NODE_CANDIDATES if n in model.columns]

    if not sensors:
        raise ValueError(f"None of the requested C3 sensors found in NI. Available: {list(ni.columns)}")
    if not nodes:
        raise ValueError(f"None of the target model nodes found. Available: {list(model.columns)}")

    matrix = build_target_matrix(ni, model, sensors, nodes, max_lag_min=args.max_lag_min)
    matrix.to_csv(out_dir / "c3_target_matrix_metrics.csv", index=False)

    target_sensor, target_node = choose_target(matrix, target_node=args.target_node, target_sensor=args.target_sensor)

    print(f"\nChosen C3 target: measured {target_sensor} vs model {target_node}")

    ts = pd.concat([
        ni[target_sensor].rename(f"c3_main__measured__{target_sensor}"),
        model[target_node].rename(f"c3_main__model__{target_node}"),
    ], axis=1).dropna()

    raw, bc, model_bc = plot_target(
        ts,
        measured_col=f"c3_main__measured__{target_sensor}",
        model_col=f"c3_main__model__{target_node}",
        title=f"C3 validation: measured {target_sensor} vs model {target_node}",
        outpath=out_dir / "c3_main_validation.png",
    )

    # Save timeseries compatible with the combined professor plotter.
    ts[f"c3_main__model_bias_corrected__{target_node}"] = model_bc
    ts.to_csv(out_dir / "c3_stack_pair_timeseries.csv")

    plot_matrix_heatmap(matrix, "bc_rmse_C", out_dir / "c3_target_matrix_bc_rmse.png", "C3 target matrix: bias-corrected RMSE")
    plot_matrix_heatmap(matrix, "bc_corr", out_dir / "c3_target_matrix_bc_corr.png", "C3 target matrix: bias-corrected correlation")
    plot_matrix_heatmap(matrix, "bc_amp_error_C", out_dir / "c3_target_matrix_bc_amp_error.png", "C3 target matrix: amplitude error")

    top_by_node = {}
    if not matrix.empty:
        for node, sub in matrix.groupby("model_node"):
            top_by_node[node] = sub.sort_values("bc_fit_score", ascending=False).head(5).to_dict("records")

    summary = {
        "window": {"start": str(start), "end": str(end)},
        "model_file": str(model_file),
        "available_sensors_used": sensors,
        "available_model_nodes_used": nodes,
        "chosen_target_sensor": target_sensor,
        "chosen_model_node": target_node,
        "raw_metrics": raw,
        "bias_corrected_metrics": bc,
        "top_candidates_by_model_node": top_by_node,
        "notes": [
            "This is a quick C3 validation/target-finding script.",
            "If target-sensor is not forced, it chooses the best measured sensor for the requested model node using bias-corrected RMSE, correlation, and amplitude error.",
            "Use --target-sensor if the physical C3 sensor mapping is already known.",
        ],
    }

    with open(out_dir / "c3_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("\nTop candidates for target node:")
    print(
        matrix[matrix["model_node"] == target_node]
        .sort_values("bc_fit_score", ascending=False)
        [["sensor", "model_node", "bc_rmse_C", "bc_corr", "bc_amp_error_C", "bc_fit_score"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
