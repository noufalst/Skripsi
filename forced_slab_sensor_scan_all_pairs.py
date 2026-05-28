"""
Forced slab sensor-pair scanner
================================
Purpose:
    Scan ALL raw NI temperature sensor pairs using a 1D forced-slab model.
    For each pair:
        top_col  -> used as measured top boundary temperature
        target_col -> used as measured bottom/indoor roof temperature

    The code ranks which sensor pairs are thermally consistent with a slab model
    by RMSE, MAE, amplitude error, and correlation.

How to use in VSCode terminal:
    1) Put this file in the same folder as:
       - new_baru_revised_same_structure_v2.py
       - Pengukuran 30_1 Maret 2026.xlsx
       - Pengukuran 30_2 Maret 2026.xlsx
       - Pengukuran 3 April 2026.xlsx

    2) Run:
       py forced_slab_sensor_scan.py

Outputs:
    - forced_slab_pair_scan_CAM_WINDOW.csv
    - forced_slab_pair_scan_C3_WINDOW.csv
    - forced_slab_pair_scan_TOP_<N>_<window>.png
    - forced_slab_best_pairs_<window>/ individual PNGs

Notes:
    - This is diagnostic only. It does not prove physical sensor placement by itself.
    - Use it together with NI setup documentation and sensor installation notes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER CONFIG
# ============================================================

BASE_DIR = Path(".")
MODULE_NAME = "new_baru_revised_same_structure_v2"

NI_FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]

WINDOWS = {
    "CAM_WINDOW": ("2026-03-31 11:58:00", "2026-04-02 21:42:00"),
    "C3_WINDOW":  ("2026-04-09 11:05:00", "2026-04-10 14:08:00"),
}

# Slab properties. Default normal concrete / paper reference.
SLAB = {
    "H_slab": 0.10,      # m
    "lambda_s": 1.74,   # W/mK
    "rho_s": 2300.0,    # kg/m3
    "cp_s": 840.0,      # J/kgK
    "h_in": 8.0,        # W/m2K, bottom convective coefficient for forced diagnostic
    "Nz": 67,
}

# If True, every possible pair among SENSOR_CANDIDATES will be tested.
# If False, only TOP_CANDIDATES -> TARGET_CANDIDATES will be tested.
SCAN_ALL_PAIRS = True

# Candidate sensors. Edit these if your NI columns differ.
# Put outdoor/top-like sensors in TOP_CANDIDATES and indoor/bottom-like sensors in TARGET_CANDIDATES.
TOP_CANDIDATES = [
    "T1Ta", "T1Ta2", "T1Ke", "T2A", "T2A1", "T2Ka", "T3Ka", "T3Kd",
]

TARGET_CANDIDATES = [
    "T1Tb", "T2A2", "T1Ka", "T1Kd", "T2Ka", "T3Ka", "T3Kd",
]

# Use this as interior air boundary for all forced slab tests if present.
# For CAM, T1Ka is often a reasonable first proxy. Change if needed.
T_IN_COL_DEFAULT = "T1Ka"

# Plot only best N pairs per window.
PLOT_TOP_N = 20

# Use 1-minute resampled data.
RESAMPLE_RULE = "1min"


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal system."""
    n = len(d)
    ac, bc, cc, dc = map(lambda x: np.array(x, dtype=float).copy(), (a, b, c, d))

    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


def forced_slab_implicit(
    df: pd.DataFrame,
    top_col: str,
    target_col: str,
    T_in_col: Optional[str] = None,
    H_slab: float = 0.10,
    lambda_s: float = 1.74,
    rho_s: float = 2300.0,
    cp_s: float = 840.0,
    h_in: float = 8.0,
    Nz: int = 67,
    resample_rule: str = "1min",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    1D implicit forced slab model.

    Top boundary is measured top_col, imposed as Dirichlet T(0,t)=T_top(t).
    Bottom boundary is convection to T_in_col if available; otherwise uses target initial/mean as weak proxy.

    Returns:
        sim_bottom_C, obs_target_C, err_C
    """
    use_cols = [top_col, target_col]
    if T_in_col is not None and T_in_col in df.columns:
        use_cols.append(T_in_col)

    data = df[use_cols].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.resample(resample_rule).mean().interpolate(method="time", limit=10)
    data = data.dropna(subset=[top_col, target_col])

    if len(data) < 20:
        raise ValueError(f"Not enough data for pair {top_col}->{target_col}")

    # Time step from index
    dt = float(data.index.to_series().diff().dt.total_seconds().median())
    if not np.isfinite(dt) or dt <= 0:
        dt = 60.0

    dz = H_slab / (Nz - 1)
    alpha = lambda_s / (rho_s * cp_s)
    Fo = alpha * dt / dz**2

    # Unknown nodes are 1..Nz-1 because node 0 is fixed by measured top boundary.
    m = Nz - 1

    # Initial profile: linear between first top and first target.
    T_top0 = float(data[top_col].iloc[0]) + 273.15
    T_bot0 = float(data[target_col].iloc[0]) + 273.15
    T = np.linspace(T_top0, T_bot0, Nz)

    sim_vals = []
    times = []

    for ts, row in data.iterrows():
        T_top = float(row[top_col]) + 273.15

        if T_in_col is not None and T_in_col in data.columns and np.isfinite(row[T_in_col]):
            T_in = float(row[T_in_col]) + 273.15
        else:
            # Fallback: use previous bottom as weak air proxy.
            T_in = T[-1]

        # Set up tridiagonal for unknown vector U = T[1:]
        a = np.zeros(m)  # lower
        b = np.zeros(m)  # diag
        c = np.zeros(m)  # upper
        rhs = np.zeros(m)

        # Internal unknowns corresponding to physical nodes j=1..Nz-2
        for k in range(m - 1):
            # physical node j = k + 1
            a[k] = -Fo if k > 0 else 0.0
            b[k] = 1.0 + 2.0 * Fo
            c[k] = -Fo
            rhs[k] = T[k + 1]

            # contribution from fixed top boundary for node j=1
            if k == 0:
                rhs[k] += Fo * T_top

        # Bottom convective boundary for physical node j=Nz-1, unknown index k=m-1
        k = m - 1
        gamma = 2.0 * alpha * dt * h_in / (lambda_s * dz)
        a[k] = -2.0 * Fo
        b[k] = 1.0 + 2.0 * Fo + gamma
        c[k] = 0.0
        rhs[k] = T[-1] + gamma * T_in

        U = tdma(a, b, c, rhs)
        T[0] = T_top
        T[1:] = U

        times.append(ts)
        sim_vals.append(T[-1] - 273.15)

    sim = pd.Series(sim_vals, index=pd.to_datetime(times), name=f"model_{top_col}_to_{target_col}")
    obs = data[target_col].copy()
    obs.name = target_col

    common = sim.index.intersection(obs.index)
    sim = sim.loc[common]
    obs = obs.loc[common]
    err = sim - obs
    err.name = "error"

    return sim, obs, err


# ============================================================
# METRICS AND PLOTTING
# ============================================================

def pair_metrics(
    sim: pd.Series,
    obs: pd.Series,
    err: pd.Series,
    top_col: str,
    target_col: str,
    T_in_col: Optional[str],
    window_name: str,
) -> Dict[str, object]:
    sim_amp = float(sim.max() - sim.min())
    obs_amp = float(obs.max() - obs.min())
    corr = float(sim.corr(obs)) if len(sim) > 3 else np.nan

    # Composite score: low RMSE and low amplitude error are both desired.
    rmse = float(np.sqrt(np.mean(err.values ** 2)))
    amp_err = sim_amp - obs_amp
    score = rmse + 0.5 * abs(amp_err)

    return {
        "window": window_name,
        "top_col": top_col,
        "target_col": target_col,
        "T_in_col": T_in_col,
        "n": int(len(err)),
        "bias_C": float(err.mean()),
        "mae_C": float(err.abs().mean()),
        "rmse_C": rmse,
        "max_abs_error_C": float(err.abs().max()),
        "sim_amp_C": sim_amp,
        "obs_amp_C": obs_amp,
        "amp_err_C": float(amp_err),
        "corr": corr,
        "score": float(score),
    }


def plot_pair(
    sim: pd.Series,
    obs: pd.Series,
    err: pd.Series,
    metrics: Dict[str, object],
    save_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    top_col = metrics["top_col"]
    target_col = metrics["target_col"]

    axes[0].plot(obs.index, obs.values, label=f"Measured {target_col}", linewidth=2)
    axes[0].plot(sim.index, sim.values, "--", label=f"Forced slab from {top_col}", linewidth=2)
    axes[0].set_title(
        f"Forced slab: {top_col} → {target_col}\n"
        f"RMSE={metrics['rmse_C']:.2f}°C | MAE={metrics['mae_C']:.2f}°C | "
        f"Bias={metrics['bias_C']:.2f}°C | AmpErr={metrics['amp_err_C']:.2f}°C | "
        f"Corr={metrics['corr']:.3f}"
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(err.index, err.values, linewidth=1.5)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error (°C)")
    axes[1].set_xlabel("Datetime")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_summary(summary: pd.DataFrame, window_name: str, save_path: Path, top_n: int = 10):
    top = summary.head(top_n).copy()
    labels = [f"{r.top_col}→{r.target_col}" for r in top.itertuples(index=False)]

    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, top["rmse_C"], width, label="RMSE")
    ax.bar(x + width/2, top["amp_err_C"].abs(), width, label="|AmpErr|")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("°C")
    ax.set_title(f"Top forced-slab sensor pairs — {window_name}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DATA LOADING
# ============================================================

def load_ni_data(base_dir: Path) -> pd.DataFrame:
    """Load NI data using existing project module loader."""
    import importlib

    gr = importlib.import_module(MODULE_NAME)

    # Try common signatures used in the project.
    try:
        return gr.load_multiple_NI_sensor_data(base_dir=str(base_dir))
    except TypeError:
        pass

    try:
        return gr.load_multiple_NI_sensor_data([str(base_dir / f) for f in NI_FILES])
    except TypeError:
        pass

    try:
        return gr.load_multiple_NI_sensor_data()
    except Exception as e:
        raise RuntimeError(
            "Could not load NI data using project module. "
            "Check load_multiple_NI_sensor_data signature."
        ) from e


# ============================================================
# MAIN SCAN
# ============================================================

def run_scan_for_window(
    ni_all: pd.DataFrame,
    window_name: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    ni = ni_all.loc[start_ts:end_ts].copy()
    if ni.empty:
        raise ValueError(f"NI data empty for {window_name}: {start} -> {end}")

    numeric_cols = []
    raw_temp_pattern = re.compile(r"^T\d+[A-Za-z]+\d*$")

    for c in ni.columns:
        # Keep raw NI sensor channels such as T1Ta, T1Ta2, T2A, T2A2, T3Kd.
        # Exclude derived aliases such as T_s_in_CAM, T_g_top_CAM, etc.
        if not raw_temp_pattern.match(str(c)):
            continue

        s = pd.to_numeric(ni[c], errors="coerce")

        # Sensor-like temperature columns: enough valid data and plausible values.
        if s.notna().sum() > 50:
            q01, q99 = s.quantile([0.01, 0.99])
            if -20 < q01 < 80 and -20 < q99 < 80:
                numeric_cols.append(c)

    numeric_cols = sorted(set(numeric_cols))

    if SCAN_ALL_PAIRS:
        top_candidates = numeric_cols
        target_candidates = numeric_cols
    else:
        top_candidates = [c for c in TOP_CANDIDATES if c in ni.columns]
        target_candidates = [c for c in TARGET_CANDIDATES if c in ni.columns]

    print(f"\n=== FORCED SLAB SENSOR SCAN: {window_name} ===")
    print(f"Window       : {start_ts} -> {end_ts}")
    print(f"Rows         : {len(ni)}")
    print(f"Top candidates    : {top_candidates}")
    print(f"Target candidates : {target_candidates}")
    print(f"T_in default      : {T_IN_COL_DEFAULT if T_IN_COL_DEFAULT in ni.columns else 'None'}")

    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series, pd.Series, Dict[str, object]]] = {}

    T_in_col = T_IN_COL_DEFAULT if T_IN_COL_DEFAULT in ni.columns else None

    for top_col in top_candidates:
        for target_col in target_candidates:
            if top_col == target_col:
                continue

            try:
                sim, obs, err = forced_slab_implicit(
                    ni,
                    top_col=top_col,
                    target_col=target_col,
                    T_in_col=T_in_col,
                    H_slab=SLAB["H_slab"],
                    lambda_s=SLAB["lambda_s"],
                    rho_s=SLAB["rho_s"],
                    cp_s=SLAB["cp_s"],
                    h_in=SLAB["h_in"],
                    Nz=int(SLAB["Nz"]),
                    resample_rule=RESAMPLE_RULE,
                )

                metrics = pair_metrics(sim, obs, err, top_col, target_col, T_in_col, window_name)
                rows.append(metrics)
                cache[(top_col, target_col)] = (sim, obs, err, metrics)

            except Exception as e:
                print(f"  SKIP {top_col}->{target_col}: {e}")

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError(f"No valid pair results for {window_name}")

    summary = summary.sort_values(["score", "rmse_C", "mae_C"]).reset_index(drop=True)

    csv_path = Path(f"forced_slab_pair_scan_{window_name}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\nTop pairs:")
    print(summary.head(PLOT_TOP_N)[[
        "top_col", "target_col", "rmse_C", "mae_C", "bias_C", "amp_err_C", "corr", "score"
    ]].to_string(index=False))

    # Summary bar plot.
    plot_top_summary(
        summary,
        window_name=window_name,
        save_path=Path(f"forced_slab_pair_scan_TOP_{PLOT_TOP_N}_{window_name}.png"),
        top_n=PLOT_TOP_N,
    )

    # Individual plots for best pairs.
    out_dir = Path(f"forced_slab_best_pairs_{window_name}")
    out_dir.mkdir(exist_ok=True)

    for r in summary.head(PLOT_TOP_N).itertuples(index=False):
        key = (r.top_col, r.target_col)
        if key not in cache:
            continue
        sim, obs, err, metrics = cache[key]
        save_path = out_dir / f"forced_slab_{r.top_col}_to_{r.target_col}.png"
        plot_pair(sim, obs, err, metrics, save_path)

    print(f"Saved best-pair plots to: {out_dir}")
    return summary


def main():
    print("Loading NI data...")
    ni_all = load_ni_data(BASE_DIR)
    print(f"NI period: {ni_all.index.min()} -> {ni_all.index.max()} | rows={len(ni_all)}")

    all_summaries = []
    for window_name, (start, end) in WINDOWS.items():
        summary = run_scan_for_window(ni_all, window_name, start, end)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv("forced_slab_pair_scan_ALL_WINDOWS.csv", index=False)
    print("\nSaved combined CSV: forced_slab_pair_scan_ALL_WINDOWS.csv")
    print("Done.")


if __name__ == "__main__":
    main()
