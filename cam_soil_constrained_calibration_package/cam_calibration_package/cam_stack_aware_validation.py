
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

PHYSICAL_PAIRS = {
    "soil_top":    ("T1Ke", "T_g_top",  "tanah atas -> substrate top"),
    "soil_bottom": ("T2A",  "T_g_bot",  "tanah bawah -> substrate bottom"),
    "slab_top":    ("T1Ta", "T_s_top",  "atap outdoor -> top slab under substrate"),
    "slab_bottom": ("T1Tb", "T_s_in",   "atap indoor -> bottom slab / inner roof"),
    "room_air":    ("T1Ka", "T_in_used", "ruangan -> indoor air boundary"),
}

ALT_PAIRS = [
    ("T2A", "T_g_bot", "T2A expected lower substrate"),
    ("T2A", "T_s_in", "T2A possible damped inside path"),
    ("T2A", "T_in_used", "T2A possible room coupling"),
    ("T1Tb", "T_s_in", "T1Tb expected atap indoor"),
    ("T1Tb", "T_s_top", "T1Tb possible short slab path"),
    ("T1Tb", "T_g_bot", "T1Tb possible lower substrate-like"),
    ("T1Ta", "T_s_top", "T1Ta expected atap outdoor/top slab"),
    ("T1Ta", "T_g_top", "T1Ta possible upper heat path"),
    ("T1Ta", "T_a", "T1Ta possible ambient-like"),
    ("T1Ke", "T_g_top", "T1Ke expected tanah atas"),
    ("T1Ke", "T_s_top", "T1Ke possible slab-top-like"),
    ("T2A2", "T_g_bot", "T2A2 vs substrate bottom"),
    ("T2A2", "T_s_in", "T2A2 vs slab bottom"),
    ("T2A2", "T_in_used", "T2A2 vs room"),
]

SENSOR_CONTEXT = ["T1Ta", "T1Ke", "T2A", "T1Tb", "T2A2", "T1Ka"]
MODEL_CONTEXT = ["T_a", "T_f", "T_g_top", "T_g_mid", "T_g_bot", "T_s_top", "T_s_mid", "T_s_in", "T_in_used"]


def safe_resample_1min(df: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.resample("1min").mean(numeric_only=True)
    return out.interpolate(method="time", limit=limit)


def robust_amp(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 5:
        return np.nan
    return float(x.quantile(0.95) - x.quantile(0.05))


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.concat([a.rename("measured"), b.rename("model")], axis=1).dropna()


def corr(a: pd.Series, b: pd.Series) -> float:
    x = align(a, b)
    if len(x) < 5 or x["measured"].std() == 0 or x["model"].std() == 0:
        return np.nan
    return float(np.corrcoef(x["measured"], x["model"])[0, 1])


def metrics(measured: pd.Series, model: pd.Series, label: str, max_lag: int = 360) -> dict:
    x = align(measured, model)
    if len(x) < 5:
        return {"label": label, "available": False, "reason": "not enough overlap"}

    err = x["model"] - x["measured"]
    ma = robust_amp(x["measured"])
    moa = robust_amp(x["model"])

    best_c, best_lag, best_rmse = np.nan, 0, np.nan
    for lag in range(-max_lag, max_lag + 1, 5):
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
        "label": label,
        "available": True,
        "n": int(len(x)),
        "rmse_C": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "mae_C": float(np.mean(np.abs(err.to_numpy()))),
        "bias_model_minus_measured_C": float(err.mean()),
        "corr": corr(x["measured"], x["model"]),
        "best_abs_corr": float(best_c) if np.isfinite(best_c) else np.nan,
        "best_lag_min_model_shift": int(best_lag),
        "rmse_at_best_lag_C": best_rmse,
        "measured_amp_95_05_C": ma,
        "model_amp_95_05_C": moa,
        "amp_error_model_minus_measured_C": moa - ma if np.isfinite(ma) and np.isfinite(moa) else np.nan,
        "measured_mean_C": float(x["measured"].mean()),
        "model_mean_C": float(x["model"].mean()),
        "measured_peak_C": float(x["measured"].max()),
        "model_peak_C": float(x["model"].max()),
        "peak_error_model_minus_measured_C": float(x["model"].max() - x["measured"].max()),
        "measured_peak_time": str(x["measured"].idxmax()),
        "model_peak_time": str(x["model"].idxmax()),
        "peak_lag_min": float((x["model"].idxmax() - x["measured"].idxmax()).total_seconds() / 60),
    }


def read_ni_xlsx_xml(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path, "r") as z:
        sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        if not sheets:
            raise ValueError(f"No worksheet XML in {path}")
        xml = z.read(sheets[0])

    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows = root.findall(f".//{ns}row")
    data = []
    for row in rows[1:]:
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
        raise ValueError(f"No numeric rows in {path}")

    df = pd.DataFrame(data, columns=NI_CHANNEL_NAMES)
    df["timestamp"] = pd.Timestamp("1904-01-01") + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"]).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[(df[c] < -20) | (df[c] > 100), c] = np.nan
    return df


def load_ni(files) -> pd.DataFrame:
    frames = []
    for f in files:
        f = Path(f)
        if not f.exists():
            warnings.warn(f"NI file not found: {f}")
            continue
        print(f"Loading NI: {f}")
        frames.append(read_ni_xlsx_xml(f))
    if not frames:
        raise FileNotFoundError("No NI files loaded.")
    return safe_resample_1min(pd.concat(frames).sort_index())


def load_model(folder: Path, model_file: str | None = None) -> pd.DataFrame:
    folder = Path(folder)
    candidates = []
    if model_file:
        candidates.append(folder / model_file)
    candidates += [
        folder / "cam_gsw_prediction_eval_window.csv",
        folder / "cam_gsw_prediction_full_with_spinup.csv",
        folder / "cam_physical_prediction_eval_window.csv",
        folder / "cam_prediction_calibrated.csv",
        folder / "cam_prediction_baseline.csv",
    ]
    f = next((p for p in candidates if p.exists()), None)
    if f is None:
        raise FileNotFoundError(f"No model prediction CSV found in {folder}")

    print(f"Loading model: {f}")
    df = pd.read_csv(f)
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
    return safe_resample_1min(df)


def make_pair_tables(ni, model, max_lag):
    raw_rows, bc_rows, ts_cols = [], [], []
    for role, (sensor, node, meaning) in PHYSICAL_PAIRS.items():
        base = {"role": role, "meaning": meaning, "sensor": sensor, "model_node": node}
        if sensor not in ni.columns:
            raw_rows.append({**base, "available": False, "reason": f"missing sensor {sensor}"})
            bc_rows.append({**base, "available": False, "reason": f"missing sensor {sensor}"})
            continue
        if node not in model.columns:
            raw_rows.append({**base, "available": False, "reason": f"missing model node {node}"})
            bc_rows.append({**base, "available": False, "reason": f"missing model node {node}"})
            continue

        m = ni[sensor]
        y = model[node]
        raw = metrics(m, y, f"{role}_raw", max_lag=max_lag)
        raw_rows.append({**base, **raw})

        x = align(m, y)
        b = float((x["model"] - x["measured"]).mean()) if len(x) else np.nan
        ybc = y - b
        bc = metrics(m, ybc, f"{role}_bias_corrected", max_lag=max_lag)
        bc_rows.append({**base, "bias_correction_C": b, **bc})

        ts_cols += [
            m.rename(f"{role}__measured__{sensor}"),
            y.rename(f"{role}__model__{node}"),
            ybc.rename(f"{role}__model_bias_corrected__{node}"),
        ]

    return pd.DataFrame(raw_rows), pd.DataFrame(bc_rows), pd.concat(ts_cols, axis=1) if ts_cols else pd.DataFrame()


def make_alt_table(ni, model, max_lag):
    rows = []
    for sensor, node, label in ALT_PAIRS:
        base = {"sensor": sensor, "model_node": node, "label": label}
        if sensor not in ni.columns:
            rows.append({**base, "available": False, "reason": f"missing sensor {sensor}"})
            continue
        if node not in model.columns:
            rows.append({**base, "available": False, "reason": f"missing model node {node}"})
            continue
        raw = metrics(ni[sensor], model[node], label, max_lag=max_lag)
        x = align(ni[sensor], model[node])
        b = float((x["model"] - x["measured"]).mean()) if len(x) else np.nan
        bc = metrics(ni[sensor], model[node] - b, label + "_bias_corrected", max_lag=max_lag)
        rows.append({
            **base,
            **{f"raw_{k}": v for k, v in raw.items() if k != "label"},
            "bias_correction_C": b,
            **{f"bc_{k}": v for k, v in bc.items() if k != "label"},
        })
    df = pd.DataFrame(rows)
    if "raw_rmse_C" in df.columns:
        rmse_scale = df["raw_rmse_C"].median(skipna=True)
        if not np.isfinite(rmse_scale) or rmse_scale <= 0:
            rmse_scale = 3
        amp_scale = df["raw_amp_error_model_minus_measured_C"].abs().median(skipna=True)
        if not np.isfinite(amp_scale) or amp_scale <= 0:
            amp_scale = 3
        df["raw_fit_score"] = (
            np.exp(-df["raw_rmse_C"] / rmse_scale)
            + df["raw_corr"].abs().fillna(0)
            + np.exp(-df["raw_amp_error_model_minus_measured_C"].abs() / amp_scale)
        )
        df["rank_within_sensor"] = df.groupby("sensor")["raw_fit_score"].rank(ascending=False, method="min")
    return df


def plot_physical(ts, metric_df, outpath, bias_corrected=False):
    roles = list(PHYSICAL_PAIRS.keys())
    fig, axes = plt.subplots(len(roles), 1, figsize=(16, 3.1 * len(roles)), sharex=True)
    if len(roles) == 1:
        axes = [axes]
    for ax, role in zip(axes, roles):
        sensor, node, meaning = PHYSICAL_PAIRS[role]
        meas_col = f"{role}__measured__{sensor}"
        mod_col = f"{role}__model_bias_corrected__{node}" if bias_corrected else f"{role}__model__{node}"
        if meas_col in ts.columns:
            ax.plot(ts.index, ts[meas_col], label=f"Measured {sensor}", lw=1.8, color="black")
        if mod_col in ts.columns:
            ax.plot(ts.index, ts[mod_col], label=f"Model {node}", lw=1.4)
        row = metric_df[metric_df["role"] == role]
        if not row.empty and bool(row.iloc[0].get("available", False)):
            r = row.iloc[0]
            ax.set_title(
                f"{role}: {sensor} vs {node} | RMSE={r.get('rmse_C', np.nan):.2f}°C, "
                f"corr={r.get('corr', np.nan):.2f}, amp_err={r.get('amp_error_model_minus_measured_C', np.nan):.2f}°C"
            )
        else:
            reason = row.iloc[0].get("reason", "unavailable") if not row.empty else "unavailable"
            ax.set_title(f"{role}: {sensor} vs {node} unavailable ({reason})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylabel("T (°C)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_normalized(ts, outpath):
    roles = list(PHYSICAL_PAIRS.keys())
    fig, axes = plt.subplots(len(roles), 1, figsize=(16, 3.0 * len(roles)), sharex=True)
    if len(roles) == 1:
        axes = [axes]
    for ax, role in zip(axes, roles):
        sensor, node, _ = PHYSICAL_PAIRS[role]
        for col, label, style in [
            (f"{role}__measured__{sensor}", f"Measured {sensor}", "-"),
            (f"{role}__model__{node}", f"Model {node}", "--"),
        ]:
            if col in ts.columns:
                s = ts[col]
                amp = robust_amp(s)
                if np.isfinite(amp) and amp > 0:
                    ax.plot(ts.index, (s - s.median()) / amp, style, label=label, lw=1.3)
        ax.set_title(f"Normalized shape: {role}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_context(df, cols, title, outpath):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for c in cols:
        axes[0].plot(df.index, df[c], label=c, lw=1.2)
    axes[0].set_title(title)
    axes[0].set_ylabel("T (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)
    for c in cols:
        s = df[c]
        amp = robust_amp(s)
        if np.isfinite(amp) and amp > 0:
            axes[1].plot(df.index, (s - s.median()) / amp, label=c, lw=1.2)
    axes[1].set_title(title + " — normalized")
    axes[1].set_ylabel("normalized")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_alt(ni, model, alt, outpath):
    sensors = ["T2A", "T1Tb", "T1Ta", "T1Ke", "T2A2"]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(16, 3.2 * len(sensors)), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        if sensor not in ni.columns:
            ax.set_title(f"{sensor} missing")
            continue
        ax.plot(ni.index, ni[sensor], color="black", lw=1.8, label=f"Measured {sensor}")
        sub = alt[alt["sensor"] == sensor].copy()
        if "rank_within_sensor" in sub.columns:
            sub = sub.sort_values("rank_within_sensor").head(3)
        else:
            sub = sub.head(3)
        for _, r in sub.iterrows():
            node = r["model_node"]
            if node in model.columns:
                ax.plot(model.index, model[node], lw=1.1, label=f"{node} | RMSE={r.get('raw_rmse_C', np.nan):.2f}, corr={r.get('raw_corr', np.nan):.2f}")
        ax.set_title(f"Alternative checks for {sensor}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylabel("T (°C)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="CAM stack-aware validation using physical mapping.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--ni-files", nargs="*", default=DEFAULT_NI_FILES)
    p.add_argument("--model-output-dir", default="outputs_cam_gsw_fixed_inputs")
    p.add_argument("--model-file", default=None)
    p.add_argument("--start", default="2026-03-31 11:58:00")
    p.add_argument("--end", default="2026-04-02 21:42:00")
    p.add_argument("--output-dir", default="outputs_cam_stack_validation")
    p.add_argument("--max-lag-min", type=int, default=360)
    args = p.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    ni = load_ni([base / f for f in args.ni_files])
    ni = ni[(ni.index >= start) & (ni.index <= end)].copy()
    if ni.empty:
        raise ValueError(f"No NI data inside {start} -> {end}")

    model = load_model(base / args.model_output_dir, args.model_file)
    model = model[(model.index >= start) & (model.index <= end)].copy()
    if model.empty:
        raise ValueError(f"No model data inside {start} -> {end}")

    print("\nAvailable measured sensors:", ", ".join([c for c in SENSOR_CONTEXT if c in ni.columns]))
    print("Available model nodes:", ", ".join([c for c in MODEL_CONTEXT if c in model.columns]))

    raw, bc, ts = make_pair_tables(ni, model, args.max_lag_min)
    alt = make_alt_table(ni, model, args.max_lag_min)

    raw.to_csv(out / "cam_stack_physical_pair_metrics.csv", index=False)
    bc.to_csv(out / "cam_stack_physical_pair_metrics_bias_corrected.csv", index=False)
    alt.to_csv(out / "cam_stack_alternative_pair_metrics.csv", index=False)
    ts.to_csv(out / "cam_stack_pair_timeseries.csv")

    plot_physical(ts, raw, out / "cam_stack_physical_pairs_overlay.png", bias_corrected=False)
    plot_physical(ts, bc, out / "cam_stack_physical_pairs_bias_corrected.png", bias_corrected=True)
    plot_normalized(ts, out / "cam_stack_physical_pairs_normalized.png")
    plot_context(ni, SENSOR_CONTEXT, "CAM measured sensors along stack", out / "cam_stack_measured_sensors_context.png")
    plot_context(model, MODEL_CONTEXT, "CAM model nodes along stack", out / "cam_stack_model_nodes_context.png")
    plot_alt(ni, model, alt, out / "cam_stack_alternative_checks.png")

    summary = {
        "window": {"start": str(start), "end": str(end)},
        "physical_pairs": {k: {"sensor": v[0], "model": v[1], "meaning": v[2]} for k, v in PHYSICAL_PAIRS.items()},
        "notes": [
            "No sensor is assumed truly exposed.",
            "High solar response should be interpreted as fast/upper heat-path response, not direct exposure.",
            "Physical stack mapping is evaluated before using statistical best-match mapping.",
            "If T_s_top is missing, patch the CAM model runner to export slab top temperature.",
        ],
        "physical_pair_raw_metrics": raw.to_dict("records"),
        "physical_pair_bias_corrected_metrics": bc.to_dict("records"),
        "top_alternative_pairs_by_sensor": {},
    }
    if "rank_within_sensor" in alt.columns:
        for sensor, sub in alt.groupby("sensor"):
            summary["top_alternative_pairs_by_sensor"][sensor] = sub.sort_values("rank_within_sensor").head(3).to_dict("records")

    with open(out / "cam_stack_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Outputs saved to: {out.resolve()}")
    print("\nPhysical-pair raw metrics:")
    show = ["role", "sensor", "model_node", "available", "reason", "rmse_C", "corr", "bias_model_minus_measured_C", "amp_error_model_minus_measured_C"]
    print(raw[[c for c in show if c in raw.columns]].to_string(index=False))
    print("\nPhysical-pair bias-corrected metrics:")
    show2 = ["role", "sensor", "model_node", "available", "bias_correction_C", "rmse_C", "corr", "bias_model_minus_measured_C", "amp_error_model_minus_measured_C"]
    print(bc[[c for c in show2 if c in bc.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
