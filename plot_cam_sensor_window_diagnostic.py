"""
CAM sensor diagnostic plots for the validation window.

Purpose:
- Check whether T2A2 behaves like true CAM inner roof temperature.
- Compare raw vs cleaned NI channels.
- Generate overview, focused, normalized-shape, individual, jump, and correlation plots.

How to run:
    py plot_cam_sensor_window_diagnostic.py

Put this file in the same folder as:
    new_baru_revised_same_structure_v3_channelmap.py
    Pengukuran 30_1 Maret 2026.xlsx
    Pengukuran 30_2 Maret 2026.xlsx
    Pengukuran 3 April 2026.xlsx
Optional:
    weatherfile mar-april.xlsx
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import new_baru_revised_same_structure_v3_channelmap as gr


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cam_sensor_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NI_FILES = [
    BASE_DIR / "Pengukuran 30_1 Maret 2026.xlsx",
    BASE_DIR / "Pengukuran 30_2 Maret 2026.xlsx",
    BASE_DIR / "Pengukuran 3 April 2026.xlsx",
]
WEATHER_FILE = BASE_DIR / "weatherfile mar-april.xlsx"
WEATHER_CACHE = BASE_DIR / "weather_clean_cache.csv"

# CAM validation window from model module
WINDOW_START, WINDOW_END = gr.VALIDATION_WINDOWS["CAM"]
WINDOW_START = pd.Timestamp(WINDOW_START)
WINDOW_END = pd.Timestamp(WINDOW_END)

# Raw NI names from the LabVIEW Excel order.
NI_CHANNEL_NAMES = [
    "timestamp_serial",
    "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
    "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
    "T2Ka", "T2Kd", "T2Kc", "2Ke", "T1A",
    "T2A", "T2A2", "T1Tb", "T1Ta", "T1Ta2",
]

# Latest confirmed physical mapping for useful aliases.
ALIAS_MAP = {
    "T_g_top_CAM": "T1Ke",      # tanah atas CAM
    "T_g_bot_CAM": "T1Ta2",     # tanah bawah CAM
    "T_s_ext_CAM": "T2A",       # atap outdoor CAM, has known anomalies
    "T_s_in_CAM": "T2A2",       # atap indoor CAM
    "T_in_CAM": "T1Ka",         # ruangan CAM
    "T_s_ext_C3": "T1A",        # atap outdoor C3 / bawah substrat C3
    "T_s_in_C3": "T3Ka",        # atap indoor C3
    "T_r_floor_RR": "2Ke",      # RR lantai
    "T_room_RR": "T1Tb",        # ruangan RR
    "T_r_ext_RR": "T1Ta",       # atap outdoor RR
    "T_r_in_RR": "T2Ka",        # atap indoor RR
}


# =============================================================================
# LOAD RAW NI WITHOUT CLEANING
# =============================================================================
def load_ni_raw_xml(filepath: Path) -> pd.DataFrame:
    """Load NI Excel using the same XML method as the model, but without cleaning."""
    with zipfile.ZipFile(filepath, "r") as z:
        with z.open("xl/worksheets/sheet1.xml") as f:
            xml_content = f.read()

    tree = ET.fromstring(xml_content)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows_xml = tree.findall(f".//{ns}row")

    data = []
    for row in rows_xml[1:]:  # skip header row
        vals = []
        for cell in row.findall(f"{ns}c"):
            v = cell.find(f"{ns}v")
            vals.append(float(v.text) if v is not None else np.nan)
        if len(vals) == 21:
            data.append(vals)

    df = pd.DataFrame(data, columns=NI_CHANNEL_NAMES)
    labview_epoch = pd.Timestamp("1904-01-01")
    df["timestamp"] = labview_epoch + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"]).sort_index()

    for alias, raw_col in ALIAS_MAP.items():
        if raw_col in df.columns:
            df[alias] = df[raw_col]

    return df


def load_multiple_ni_raw(filepaths) -> pd.DataFrame:
    frames = []
    for fp in filepaths:
        if fp.exists():
            print(f"Loading RAW NI: {fp.name}")
            frames.append(load_ni_raw_xml(fp))
        else:
            print(f"WARNING: NI file not found: {fp}")
    if not frames:
        raise FileNotFoundError("No NI files found. Put this script in the same folder as the NI Excel files.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_clean_ni(filepaths) -> pd.DataFrame:
    existing = [str(fp) for fp in filepaths if fp.exists()]
    if not existing:
        raise FileNotFoundError("No NI files found for cleaned loader.")
    return gr.load_multiple_NI_sensor_data(existing)


def load_weather_optional() -> pd.DataFrame | None:
    if not WEATHER_FILE.exists() and not WEATHER_CACHE.exists():
        print("Weather file/cache not found; weather overlay will be skipped.")
        return None
    try:
        return gr.load_weather_cache_or_excel(
            str(WEATHER_FILE),
            cache_path=str(WEATHER_CACHE),
            date_start=str(WINDOW_START),
            date_end=str(WINDOW_END),
        )
    except Exception as exc:
        print(f"Weather load failed; weather overlay skipped. Reason: {exc}")
        return None


# =============================================================================
# DIAGNOSTIC HELPERS
# =============================================================================
def window_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.index >= WINDOW_START) & (df.index <= WINDOW_END)].copy()


def sensor_summary(df_raw_w: pd.DataFrame, df_clean_w: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df_raw_w.columns:
            continue
        raw = df_raw_w[col].copy()
        clean = df_clean_w[col].copy() if col in df_clean_w.columns else raw.copy()
        diff = raw.resample("1min").mean().diff().abs()

        # flatline: longest run with very small per-minute change
        small_change = diff.fillna(np.inf) < 0.01
        longest = 0
        run = 0
        for val in small_change.values:
            if bool(val):
                run += 1
                longest = max(longest, run)
            else:
                run = 0

        rows.append({
            "channel": col,
            "n_raw": int(raw.count()),
            "n_clean": int(clean.count()),
            "raw_min_C": float(raw.min()) if raw.count() else np.nan,
            "raw_mean_C": float(raw.mean()) if raw.count() else np.nan,
            "raw_max_C": float(raw.max()) if raw.count() else np.nan,
            "clean_min_C": float(clean.min()) if clean.count() else np.nan,
            "clean_mean_C": float(clean.mean()) if clean.count() else np.nan,
            "clean_max_C": float(clean.max()) if clean.count() else np.nan,
            "clean_amplitude_C": float(clean.max() - clean.min()) if clean.count() else np.nan,
            "raw_out_of_range_n": int(((raw < -10) | (raw > 80)).sum()),
            "raw_missing_n": int(raw.isna().sum()),
            "max_abs_jump_C_per_min_raw": float(diff.max()) if diff.count() else np.nan,
            "large_jump_gt_1C_n": int((diff > 1.0).sum()),
            "large_jump_gt_2C_n": int((diff > 2.0).sum()),
            "longest_flatline_min_clean_est": int(longest),
        })
    return pd.DataFrame(rows)


def savefig(fig, filename: str):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path} | size={path.stat().st_size if path.exists() else 0} bytes")


def existing_cols(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def plot_lines(df, cols, title, filename, ylabel="Temperature (°C)", weather=None, include_gsol=True):
    cols = existing_cols(df, cols)
    if not cols:
        print(f"Skip {filename}: no valid columns")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    d = df[cols].resample("1min").mean()
    for col in cols:
        ax.plot(d.index, d[col].values, label=col, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    if weather is not None and include_gsol and "G_sol" in weather.columns:
        w = weather["G_sol"].resample("1min").mean()
        ax2 = ax.twinx()
        ax2.plot(w.index, w.values, linestyle="--", alpha=0.5, label="G_sol")
        ax2.set_ylabel("G_sol (W/m²)")

    fig.autofmt_xdate()
    savefig(fig, filename)


def plot_raw_vs_clean(df_raw, df_clean, col, filename):
    if col not in df_raw.columns or col not in df_clean.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    raw = df_raw[col].resample("1min").mean()
    clean = df_clean[col].resample("1min").mean()
    ax.plot(raw.index, raw.values, label=f"{col} raw", linewidth=1.2)
    ax.plot(clean.index, clean.values, label=f"{col} cleaned", linewidth=1.8, linestyle="--")
    ax.set_title(f"Raw vs cleaned: {col}")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    savefig(fig, filename)


def plot_normalized_shape(df, cols, title, filename):
    cols = existing_cols(df, cols)
    if not cols:
        return
    d = df[cols].resample("1min").mean()
    norm = pd.DataFrame(index=d.index)
    for col in cols:
        s = d[col]
        amp = s.max() - s.min()
        if pd.notna(amp) and amp > 1e-9:
            norm[col] = (s - s.min()) / amp
    if norm.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in norm.columns:
        ax.plot(norm.index, norm[col].values, label=col, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Normalized shape, 0–1")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    savefig(fig, filename)


def plot_jumps(df, cols, title, filename):
    cols = existing_cols(df, cols)
    if not cols:
        return
    d = df[cols].resample("1min").mean().diff()
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in cols:
        ax.plot(d.index, d[col].values, label=col, linewidth=1.2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axhline(-1.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("ΔT per minute (°C/min)")
    ax.set_xlabel("Datetime")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    savefig(fig, filename)


def plot_corr(df, cols, title, filename):
    cols = existing_cols(df, cols)
    if len(cols) < 2:
        return
    d = df[cols].resample("1min").mean().dropna(how="all")
    corr = d.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    savefig(fig, filename)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("CAM SENSOR DIAGNOSTIC PLOTS")
    print("=" * 80)
    print(f"Window: {WINDOW_START} -> {WINDOW_END}")
    print(f"Output: {OUTPUT_DIR}")

    df_raw = load_multiple_ni_raw(NI_FILES)
    df_clean = load_clean_ni(NI_FILES)
    weather = load_weather_optional()

    df_raw_w = window_slice(df_raw)
    df_clean_w = window_slice(df_clean)
    if df_clean_w.empty:
        raise ValueError("No cleaned NI data inside CAM validation window.")

    # Main channel groups
    cam_box_cols = ["T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc"]
    new_mapping_cols = ["2Ke", "T1A", "T2A", "T2A2", "T1Tb", "T1Ta", "T1Ta2"]
    cam_focus_cols = ["T2A", "T2A2", "T1Ta2", "T1Ke", "T1Ka"]
    alias_focus_cols = ["T_s_ext_CAM", "T_s_in_CAM", "T_g_bot_CAM", "T_g_top_CAM", "T_in_CAM"]
    all_diagnostic_cols = sorted(set(cam_box_cols + new_mapping_cols + cam_focus_cols))

    summary = sensor_summary(df_raw_w, df_clean_w, all_diagnostic_cols)
    summary_path = OUTPUT_DIR / "cam_sensor_diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))

    # Overview plots using cleaned data
    plot_lines(
        df_clean_w, cam_box_cols,
        "CAM box sensors, cleaned NI data",
        "01_cam_box_sensors_cleaned.png",
        weather=weather,
    )
    plot_lines(
        df_clean_w, new_mapping_cols,
        "Col 14–20 sensors, cleaned NI data, latest mapping check",
        "02_col14_20_sensors_cleaned.png",
        weather=weather,
    )
    plot_lines(
        df_clean_w, cam_focus_cols,
        "CAM focused channels: roof, soil, and room, cleaned NI data",
        "03_cam_roof_soil_room_focus_cleaned.png",
        weather=weather,
    )
    plot_lines(
        df_clean_w, alias_focus_cols,
        "CAM aliases from latest channel map, cleaned NI data",
        "04_cam_aliases_latest_mapping_cleaned.png",
        weather=weather,
    )

    # Shape/correlation plots
    plot_normalized_shape(
        df_clean_w, cam_focus_cols,
        "Normalized shape comparison: does T2A2 look like roof/soil/room?",
        "05_cam_focus_normalized_shape.png",
    )
    plot_normalized_shape(
        df_clean_w, new_mapping_cols,
        "Normalized shape comparison: Col 14–20",
        "06_col14_20_normalized_shape.png",
    )
    plot_jumps(
        df_clean_w, cam_focus_cols,
        "Jump check after cleaning: CAM focused channels",
        "07_cam_focus_deltaT_per_min_cleaned.png",
    )
    plot_jumps(
        df_raw_w, cam_focus_cols,
        "Jump check raw data: CAM focused channels",
        "08_cam_focus_deltaT_per_min_raw.png",
    )
    plot_corr(
        df_clean_w, cam_focus_cols,
        "Correlation matrix: CAM focused channels, cleaned",
        "09_cam_focus_correlation_cleaned.png",
    )

    # Raw vs cleaned checks for the most important/suspicious channels
    for col in ["T2A", "T2A2", "T1Ta2", "T1Ke", "T1Ka", "T1Tb", "T1Ta"]:
        plot_raw_vs_clean(df_raw_w, df_clean_w, col, f"raw_vs_clean_{col}.png")

    # Individual plots for every relevant channel
    for col in all_diagnostic_cols:
        plot_lines(
            df_clean_w, [col],
            f"Individual sensor check: {col}, cleaned NI data",
            f"individual_{col}_cleaned.png",
            weather=weather,
        )

    print("\nDONE.")
    print(f"Open this folder: {OUTPUT_DIR}")
    print("Most important files to inspect first:")
    print("  03_cam_roof_soil_room_focus_cleaned.png")
    print("  05_cam_focus_normalized_shape.png")
    print("  07_cam_focus_deltaT_per_min_cleaned.png")
    print("  raw_vs_clean_T2A2.png")
    print("  raw_vs_clean_T2A.png")
    print("  cam_sensor_diagnostic_summary.csv")


if __name__ == "__main__":
    main()
