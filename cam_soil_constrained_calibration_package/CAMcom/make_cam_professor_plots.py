from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def robust_amp(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 5:
        return np.nan
    return float(x.quantile(0.95) - x.quantile(0.05))


def align(measured: pd.Series, model: pd.Series) -> pd.DataFrame:
    return pd.concat([measured.rename("measured"), model.rename("model")], axis=1).dropna()


def calc_metrics(measured: pd.Series, model: pd.Series) -> dict:
    df = align(measured, model)
    if len(df) < 5:
        return {"n": len(df), "rmse_C": np.nan, "mae_C": np.nan, "bias_C": np.nan, "corr": np.nan,
                "measured_amp_C": np.nan, "model_amp_C": np.nan, "amp_error_C": np.nan,
                "measured_peak_C": np.nan, "model_peak_C": np.nan, "peak_error_C": np.nan}
    err = df["model"] - df["measured"]
    ma = robust_amp(df["measured"])
    moa = robust_amp(df["model"])
    return {
        "n": int(len(df)),
        "rmse_C": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "mae_C": float(np.mean(np.abs(err.to_numpy()))),
        "bias_C": float(err.mean()),
        "corr": float(np.corrcoef(df["measured"], df["model"])[0, 1]),
        "measured_amp_C": ma,
        "model_amp_C": moa,
        "amp_error_C": moa - ma,
        "measured_peak_C": float(df["measured"].max()),
        "model_peak_C": float(df["model"].max()),
        "peak_error_C": float(df["model"].max() - df["measured"].max()),
    }


def bias_correct(measured: pd.Series, model: pd.Series) -> tuple[pd.Series, float]:
    df = align(measured, model)
    bias = float((df["model"] - df["measured"]).mean())
    return model - bias, bias


def find_col(columns, parts):
    for col in columns:
        s = str(col)
        if all(p in s for p in parts):
            return col
    raise KeyError(f"No column contains all of {parts}")


def annotate(ax, title, met):
    msg = (f"{title}\nRMSE = {met['rmse_C']:.2f} °C\n"
           f"Bias = {met['bias_C']:.2f} °C\nCorr = {met['corr']:.2f}\n"
           f"Amp error = {met['amp_error_C']:.2f} °C")
    ax.text(0.01, 0.97, msg, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.15), fontsize=9)


def plot_t2a(ts: pd.DataFrame, out: Path):
    measured = ts[find_col(ts.columns, ["soil_bottom", "measured", "T2A"])]
    model = ts[find_col(ts.columns, ["slab_bottom", "model", "T_s_in"])]
    model_bc, b = bias_correct(measured, model)
    raw = calc_metrics(measured, model)
    bc = calc_metrics(measured, model_bc)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(ts.index, measured, label="Measured T2A")
    axes[0].plot(ts.index, model, label="Model T_s_in raw")
    axes[0].plot(ts.index, model_bc, linestyle="--", label=f"Model T_s_in bias-corrected ({b:.2f} °C removed)")
    axes[0].set_title("Main CAM validation: measured T2A vs model T_s_in")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    annotate(axes[0], "Bias-corrected", bc)

    axes[1].plot(ts.index, model - measured, label="Raw error")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Error (°C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    annotate(axes[1], "Raw", raw)

    axes[2].plot(ts.index, model_bc - measured, label="Bias-corrected error")
    axes[2].axhline(0, linewidth=0.8)
    axes[2].set_ylabel("Error (°C)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "01_main_T2A_vs_Ts_in_validation.png", dpi=180)
    plt.close(fig)
    return raw, bc


def plot_t1tb(ts: pd.DataFrame, out: Path):
    measured = ts[find_col(ts.columns, ["slab_bottom", "measured", "T1Tb"])]
    model = ts[find_col(ts.columns, ["slab_bottom", "model", "T_s_in"])]
    model_bc, b = bias_correct(measured, model)
    raw = calc_metrics(measured, model)
    bc = calc_metrics(measured, model_bc)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(ts.index, measured, label="Measured T1Tb / atap indoor")
    axes[0].plot(ts.index, model, label="Model T_s_in")
    axes[0].plot(ts.index, model_bc, linestyle="--", label=f"Bias-corrected T_s_in ({b:.2f} °C removed)")
    axes[0].set_title("Diagnostic: measured T1Tb is much more dynamic than model T_s_in")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    annotate(axes[0], "Raw diagnostic", raw)

    for series, label, style in [(measured, "Measured T1Tb normalized", "-"), (model, "Model T_s_in normalized", "--")]:
        amp = robust_amp(series)
        if np.isfinite(amp) and amp > 0:
            axes[1].plot(ts.index, (series - series.median()) / amp, linestyle=style, label=label)
    axes[1].set_title("Normalized shape comparison")
    axes[1].set_ylabel("Normalized signal")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "02_diagnostic_T1Tb_vs_Ts_in.png", dpi=180)
    plt.close(fig)
    return raw, bc


def plot_stack(ts: pd.DataFrame, out: Path) -> pd.DataFrame:
    pairs = [
        ("soil_top", "T1Ke", "T_g_top", "Tanah atas"),
        ("soil_bottom", "T2A", "T_g_bot", "Tanah bawah"),
        ("slab_top", "T1Ta", "T_s_top", "Atap outdoor / top slab"),
        ("slab_bottom", "T1Tb", "T_s_in", "Atap indoor / bottom slab"),
        ("room_air", "T1Ka", "T_in_used", "Ruangan"),
    ]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(15, 3 * len(pairs)), sharex=True)
    rows = []
    for ax, (role, sensor, node, title) in zip(axes, pairs):
        try:
            mcol = find_col(ts.columns, [role, "measured", sensor])
            ycol = find_col(ts.columns, [role, "model", node])
        except KeyError as e:
            ax.set_title(f"{title}: missing column ({e})")
            ax.axis("off")
            continue
        measured = ts[mcol]
        model = ts[ycol]
        met = calc_metrics(measured, model)
        rows.append({"pair": title, "sensor": sensor, "model_node": node, **met})
        ax.plot(ts.index, measured, label=f"Measured {sensor}")
        ax.plot(ts.index, model, label=f"Model {node}")
        ax.set_title(f"{title}: {sensor} vs {node} | RMSE {met['rmse_C']:.2f}°C, Corr {met['corr']:.2f}, AmpErr {met['amp_error_C']:.2f}°C")
        ax.set_ylabel("T (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out / "03_stack_aware_physical_pairs.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def summary_table(metrics_df: pd.DataFrame, out: Path):
    cols = ["pair", "sensor", "model_node", "rmse_C", "corr", "bias_C", "measured_amp_C", "model_amp_C", "amp_error_C"]
    show = metrics_df[cols].copy()
    for col in cols[3:]:
        show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    table = ax.table(cellText=show.values, colLabels=["Pair", "Sensor", "Model", "RMSE", "Corr", "Bias", "Amp meas", "Amp model", "Amp err"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title("CAM stack-aware validation summary", pad=20)
    fig.tight_layout()
    fig.savefig(out / "04_stack_summary_table.png", dpi=180)
    plt.close(fig)


def talking_points(out: Path, t2a_raw, t2a_bc, t1tb_raw):
    txt = f"""# CAM Plot Talking Points for Professor

## Main message

The model should not be judged only from `T1Tb vs T_s_in`, because measured `T1Tb` is much more dynamic than the 1D slab-bottom model node.

For tomorrow, present `T2A vs T_s_in` as the temporary damped-path validation target, with the caveat that physical sensor mapping still needs plug-off verification.

## Plot 1 — Main validation: T2A vs T_s_in

Raw:
- RMSE = {t2a_raw['rmse_C']:.2f} °C
- Bias = {t2a_raw['bias_C']:.2f} °C
- Corr = {t2a_raw['corr']:.2f}
- Amplitude error = {t2a_raw['amp_error_C']:.2f} °C

Bias-corrected:
- RMSE = {t2a_bc['rmse_C']:.2f} °C
- Bias ≈ {t2a_bc['bias_C']:.2f} °C
- Corr = {t2a_bc['corr']:.2f}
- Amplitude error = {t2a_bc['amp_error_C']:.2f} °C

Interpretation: the model captures a damped thermal response reasonably after removing systematic offset. This is not hidden calibration; the bias correction is reported separately.

## Plot 2 — Diagnostic: T1Tb vs T_s_in

- Measured T1Tb amplitude = {t1tb_raw['measured_amp_C']:.2f} °C
- Model T_s_in amplitude = {t1tb_raw['model_amp_C']:.2f} °C
- Amplitude error = {t1tb_raw['amp_error_C']:.2f} °C

Interpretation: T1Tb is too dynamic to be represented by the current 1D slab-bottom node. This suggests sensor/path mismatch, edge conduction, or a heat path not captured by the model.

## Suggested wording

“Pak, saya awalnya validasi CAM terhadap sensor atap indoor T1Tb, tetapi hasil model terlalu flat. Saya kemudian lakukan stack-aware validation. Hasilnya menunjukkan T1Tb kemungkinan tidak merepresentasikan node T_s_in model 1D secara bersih karena amplitudonya jauh lebih besar. Untuk sementara saya gunakan T2A sebagai target validasi damped thermal response karena bentuk dan amplitudonya lebih dekat dengan node model. Namun mapping sensor masih perlu saya verifikasi dengan plug-off test.”

## Next step

- Plug-off test sensor untuk memastikan channel fisik.
- Cek apakah T1Ta/T1Tb terkena side/edge heat path.
- Setelah mapping fix, rerun stack-aware validation.
"""
    (out / "05_talking_points_for_prof.md").write_text(txt, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Make professor-ready CAM plots.")
    p.add_argument("--stack-output-dir", default="outputs_cam_stack_validation")
    p.add_argument("--out-dir", default="outputs_prof_meeting_cam")
    args = p.parse_args()
    stack = Path(args.stack_output_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_path = stack / "cam_stack_pair_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing {ts_path}. Run cam_stack_aware_validation.py first.")
    ts = pd.read_csv(ts_path)
    time_col = ts.columns[0]
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts = ts.dropna(subset=[time_col]).set_index(time_col).sort_index()
    t2a_raw, t2a_bc = plot_t2a(ts, out)
    t1tb_raw, t1tb_bc = plot_t1tb(ts, out)
    stack_df = plot_stack(ts, out)
    summary_table(stack_df, out)
    pd.DataFrame([
        {"case": "main_T2A_vs_T_s_in_raw", **t2a_raw},
        {"case": "main_T2A_vs_T_s_in_bias_corrected", **t2a_bc},
        {"case": "diagnostic_T1Tb_vs_T_s_in_raw", **t1tb_raw},
        {"case": "diagnostic_T1Tb_vs_T_s_in_bias_corrected", **t1tb_bc},
    ]).to_csv(out / "prof_cam_key_metrics.csv", index=False)
    stack_df.to_csv(out / "prof_cam_stack_summary_metrics.csv", index=False)
    talking_points(out, t2a_raw, t2a_bc, t1tb_raw)
    print("Done.")
    print(f"Output folder: {out.resolve()}")
    for name in ["01_main_T2A_vs_Ts_in_validation.png", "02_diagnostic_T1Tb_vs_Ts_in.png", "03_stack_aware_physical_pairs.png", "04_stack_summary_table.png", "05_talking_points_for_prof.md", "prof_cam_key_metrics.csv"]:
        print("-", out / name)


if __name__ == "__main__":
    main()
