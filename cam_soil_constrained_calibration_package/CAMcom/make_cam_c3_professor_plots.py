
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


def load_timeseries(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ts = pd.read_csv(path)
    time_col = ts.columns[0]
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts = ts.dropna(subset=[time_col]).set_index(time_col).sort_index()

    for c in ts.columns:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")

    return ts


def find_col(cols, tokens, exclude=None):
    exclude = exclude or []
    for c in cols:
        s = str(c)
        if all(t in s for t in tokens) and not any(e in s for e in exclude):
            return c
    return None


def calc_metrics(measured: pd.Series, model: pd.Series) -> dict:
    df = pd.concat([measured.rename("measured"), model.rename("model")], axis=1).dropna()
    if len(df) < 5:
        return {"rmse_C": np.nan, "bias_C": np.nan, "corr": np.nan, "amp_error_C": np.nan, "measured_amp_C": np.nan, "model_amp_C": np.nan}
    err = df["model"] - df["measured"]
    ma = robust_amp(df["measured"])
    moa = robust_amp(df["model"])
    return {
        "rmse_C": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "bias_C": float(err.mean()),
        "corr": float(np.corrcoef(df["measured"], df["model"])[0, 1]),
        "measured_amp_C": ma,
        "model_amp_C": moa,
        "amp_error_C": moa - ma,
    }


def bias_correct(measured: pd.Series, model: pd.Series) -> tuple[pd.Series, float]:
    df = pd.concat([measured.rename("measured"), model.rename("model")], axis=1).dropna()
    b = float((df["model"] - df["measured"]).mean())
    return model - b, b


def plot_validation(ts, measured_col, model_col, title, outpath):
    measured = ts[measured_col]
    model = ts[model_col]
    model_bc, bias = bias_correct(measured, model)
    raw = calc_metrics(measured, model)
    bc = calc_metrics(measured, model_bc)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(ts.index, measured, label=f"Measured: {measured_col}")
    axes[0].plot(ts.index, model, label=f"Model raw: {model_col}")
    axes[0].plot(ts.index, model_bc, "--", label=f"Bias-corrected model ({bias:.2f} °C removed)")
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
    axes[1].set_ylabel("Raw error (°C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(ts.index, model_bc - measured, label="Bias-corrected error")
    axes[2].axhline(0, linewidth=0.8)
    axes[2].set_ylabel("Corrected error (°C)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    return raw, bc, model_bc


def make_comparison(cam_ts, cam_meas, cam_model_bc, c3_ts, c3_meas, c3_model_bc, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

    axes[0].plot(cam_ts.index, cam_ts[cam_meas], label="CAM measured T2A")
    axes[0].plot(cam_ts.index, cam_model_bc, "--", label="CAM model bias-corrected")
    axes[0].set_title("CAM temporary damped-path validation")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(c3_ts.index, c3_ts[c3_meas], label="C3 measured target")
    axes[1].plot(c3_ts.index, c3_model_bc, "--", label="C3 model bias-corrected")
    axes[1].set_title("C3 validation")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_summary_table(df: pd.DataFrame, outpath: Path):
    show = df[["case", "target", "model", "rmse_C", "corr", "bias_C", "measured_amp_C", "model_amp_C", "amp_error_C"]].copy()
    for col in ["rmse_C", "corr", "bias_C", "measured_amp_C", "model_amp_C", "amp_error_C"]:
        show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=show.values,
        colLabels=["Case", "Target", "Model", "RMSE", "Corr", "Bias", "Amp meas", "Amp model", "Amp err"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title("CAM + C3 validation summary", pad=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create CAM+C3 professor-ready plots after C3 validation exists.")
    parser.add_argument("--cam-timeseries", default="outputs_cam_stack_validation/cam_stack_pair_timeseries.csv")
    parser.add_argument("--c3-timeseries", default="outputs_c3_stack_validation/c3_stack_pair_timeseries.csv")
    parser.add_argument("--out-dir", default="outputs_prof_meeting_cam_c3")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cam_ts = load_timeseries(Path(args.cam_timeseries))
    c3_ts = load_timeseries(Path(args.c3_timeseries))

    cam_meas = find_col(cam_ts.columns, ["soil_bottom", "measured", "T2A"])
    cam_model = find_col(cam_ts.columns, ["slab_bottom", "model", "T_s_in"], exclude=["bias_corrected"])
    if cam_meas is None:
        cam_meas = find_col(cam_ts.columns, ["T2A"])
    if cam_model is None:
        cam_model = find_col(cam_ts.columns, ["T_s_in"], exclude=["bias_corrected"])

    c3_meas = find_col(c3_ts.columns, ["c3_main", "measured"])
    c3_model = find_col(c3_ts.columns, ["c3_main", "model"], exclude=["bias_corrected"])

    if cam_meas is None or cam_model is None:
        raise KeyError("Could not detect CAM columns.")
    if c3_meas is None or c3_model is None:
        raise KeyError("Could not detect C3 columns.")

    cam_raw, cam_bc, cam_model_bc = plot_validation(
        cam_ts, cam_meas, cam_model,
        "CAM main validation: measured T2A vs model T_s_in",
        out / "01_CAM_main_T2A_vs_Ts_in.png",
    )

    c3_raw, c3_bc, c3_model_bc = plot_validation(
        c3_ts, c3_meas, c3_model,
        "C3 validation",
        out / "02_C3_validation.png",
    )

    make_comparison(
        cam_ts, cam_meas, cam_model_bc,
        c3_ts, c3_meas, c3_model_bc,
        out / "03_CAM_vs_C3_bias_corrected_comparison.png",
    )

    rows = [
        {"case": "CAM raw", "target": cam_meas, "model": cam_model, **cam_raw},
        {"case": "CAM bias-corrected", "target": cam_meas, "model": cam_model, **cam_bc},
        {"case": "C3 raw", "target": c3_meas, "model": c3_model, **c3_raw},
        {"case": "C3 bias-corrected", "target": c3_meas, "model": c3_model, **c3_bc},
    ]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "prof_cam_c3_key_metrics.csv", index=False)
    make_summary_table(summary_df, out / "04_CAM_C3_summary_table.png")

    talking = f"""# CAM + C3 talking points

## CAM
Use `01_CAM_main_T2A_vs_Ts_in.png`.

CAM bias-corrected:
- RMSE = {cam_bc['rmse_C']:.2f} °C
- Corr = {cam_bc['corr']:.2f}
- Amplitude error = {cam_bc['amp_error_C']:.2f} °C

## C3
Use `02_C3_validation.png`.

C3 bias-corrected:
- RMSE = {c3_bc['rmse_C']:.2f} °C
- Corr = {c3_bc['corr']:.2f}
- Amplitude error = {c3_bc['amp_error_C']:.2f} °C

## Suggested wording
Pak, untuk CAM saya gunakan T2A sebagai target sementara validasi damped thermal response karena T1Tb terlalu dinamis terhadap node model T_s_in. Untuk C3, saya jalankan validasi terpisah dengan target yang paling cocok terhadap node model, sambil tetap menyatakan bahwa mapping sensor perlu diverifikasi fisik.
"""
    (out / "05_talking_points_CAM_C3.md").write_text(talking, encoding="utf-8")

    print("Done.")
    print(f"Output folder: {out.resolve()}")
    for p in sorted(out.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
