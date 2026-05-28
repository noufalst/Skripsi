"""
Plot all NI temperature sensor channels with inline labels at the end of each line.

This FIXED version avoids openpyxl entirely.
Why: some LabVIEW/Excel files have workbook styles that can trigger:
    TypeError: expected <class 'openpyxl.styles.fills.Fill'>

Use:
    python plot_all_ni_sensors_inline_labels_FIXED.py

Edit FILES and optional START_TIME / END_TIME below.
"""

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

FILES = [
    "Pengukuran 30_1 Maret 2026.xlsx",
    "Pengukuran 30_2 Maret 2026.xlsx",
    "Pengukuran 3 April 2026.xlsx",
]

OUTPUT_DIR = Path("outputs_sensor_check")
OUTPUT_DIR.mkdir(exist_ok=True)

# Keep names as used in the March/April NI files.
NI_CHANNEL_NAMES = [
    "timestamp_serial",
    "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
    "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
    "T2Ka", "T2Kd", "T2Kc", "2Ke",  "T1A",
    "T2A",  "T2A2", "T1Tb", "T1Ta", "T1Ta2",
]

SENSOR_COLS = NI_CHANNEL_NAMES[1:]

# Optional time filter. Set to None for full data.
START_TIME = None
END_TIME = None

# Example:
# START_TIME = "2026-03-31 00:00:00"
# END_TIME   = "2026-04-03 00:00:00"

# For raw audit, keep False first.
AUTO_CONVERT_F_TO_C = False

# For raw audit, keep False first.
MASK_UNPHYSICAL = False
PHYSICAL_MIN = -10
PHYSICAL_MAX = 120


# =============================================================================
# LOW-LEVEL XLSX XML PARSER — NO OPENPYXL
# =============================================================================

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _column_letters_to_index(cell_ref: str) -> int:
    """Convert Excel cell ref like A1, B1, AA1 to zero-based column index."""
    match = re.match(r"([A-Z]+)", str(cell_ref).upper())
    if not match:
        return -1
    letters = match.group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _load_shared_strings(z: zipfile.ZipFile) -> list[str]:
    """Load sharedStrings.xml if present. Numeric NI data usually does not need this."""
    if "xl/sharedStrings.xml" not in z.namelist():
        return []

    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall(f".//{NS_MAIN}si"):
        texts = []
        for t in si.findall(f".//{NS_MAIN}t"):
            texts.append(t.text or "")
        strings.append("".join(texts))
    return strings


def _cell_value(cell, shared_strings: list[str]):
    """Return raw cell value from sheet XML cell element."""
    cell_type = cell.attrib.get("t")

    # Inline string
    if cell_type == "inlineStr":
        t = cell.find(f".//{NS_MAIN}t")
        return t.text if t is not None else None

    v = cell.find(f"{NS_MAIN}v")
    if v is None:
        return None

    text = v.text
    if text is None:
        return None

    # Shared string index
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except Exception:
            return text

    return text


def _get_first_worksheet_xml_name(z: zipfile.ZipFile) -> str:
    """Prefer sheet1.xml, otherwise first worksheet XML."""
    names = z.namelist()
    if "xl/worksheets/sheet1.xml" in names:
        return "xl/worksheets/sheet1.xml"

    candidates = sorted([name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")])
    if not candidates:
        raise ValueError("No worksheet XML found inside XLSX file.")
    return candidates[0]


def load_ni_excel_xml(filepath) -> pd.DataFrame:
    """Load one NI LabVIEW Excel file by parsing XLSX XML directly.

    Expected structure:
        col 0      = LabVIEW timestamp serial
        col 1..20  = temperature channels

    Timestamp convention:
        LabVIEW days since 1904-01-01.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"WARNING: file not found: {filepath}")
        return pd.DataFrame()

    print(f"Loading XML: {filepath}")

    rows = []
    with zipfile.ZipFile(filepath, "r") as z:
        shared_strings = _load_shared_strings(z)
        sheet_name = _get_first_worksheet_xml_name(z)
        root = ET.fromstring(z.read(sheet_name))

        for row in root.findall(f".//{NS_MAIN}row"):
            values = [np.nan] * len(NI_CHANNEL_NAMES)

            for cell in row.findall(f"{NS_MAIN}c"):
                ref = cell.attrib.get("r", "")
                col_idx = _column_letters_to_index(ref)
                if col_idx < 0 or col_idx >= len(NI_CHANNEL_NAMES):
                    continue

                raw = _cell_value(cell, shared_strings)
                try:
                    values[col_idx] = float(raw)
                except (TypeError, ValueError):
                    values[col_idx] = np.nan

            # Skip header / blank rows.
            if not np.isfinite(values[0]):
                continue

            rows.append(values)

    if not rows:
        raise ValueError(f"No numeric NI rows found in {filepath}")

    df = pd.DataFrame(rows, columns=NI_CHANNEL_NAMES)

    labview_epoch = pd.Timestamp("1904-01-01")
    df["datetime"] = labview_epoch + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.drop(columns=["timestamp_serial"]).set_index("datetime").sort_index()

    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  rows: {len(df)} | {df.index.min()} → {df.index.max()}")
    return df


def load_multiple_ni(files) -> pd.DataFrame:
    dfs = []
    for fp in files:
        df_i = load_ni_excel_xml(fp)
        if not df_i.empty:
            dfs.append(df_i)

    if not dfs:
        raise FileNotFoundError("No NI files were loaded. Check FILES paths.")

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if START_TIME is not None:
        df = df[df.index >= pd.Timestamp(START_TIME)]
    if END_TIME is not None:
        df = df[df.index <= pd.Timestamp(END_TIME)]

    if MASK_UNPHYSICAL:
        for col in SENSOR_COLS:
            bad = (df[col] < PHYSICAL_MIN) | (df[col] > PHYSICAL_MAX)
            df.loc[bad, col] = np.nan

    if AUTO_CONVERT_F_TO_C:
        df = auto_convert_possible_fahrenheit(df)

    print(f"\nCombined period: {df.index.min()} → {df.index.max()}")
    print(f"Records: {len(df)}")
    return df


# =============================================================================
# OPTIONAL UNIT CHECK
# =============================================================================

def auto_convert_possible_fahrenheit(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic conversion for channels that look like Fahrenheit.

    Use carefully. For raw sensor audit, keep AUTO_CONVERT_F_TO_C = False.
    """
    out = df.copy()

    print("\nUnit check:")
    for col in SENSOR_COLS:
        s = out[col].dropna()
        if s.empty:
            continue

        p05 = float(s.quantile(0.05))
        med = float(s.median())
        p95 = float(s.quantile(0.95))

        looks_f = (50 < p05 < 100) and (60 < med < 110) and (70 < p95 < 130)

        if looks_f:
            converted = (out[col] - 32.0) * 5.0 / 9.0
            c_med = converted.dropna().median()

            if 15 <= c_med <= 50:
                out[col] = converted
                print(f"  {col}: converted F → C | raw median={med:.1f}, converted median={c_med:.1f}")
            else:
                print(f"  {col}: looks high but not converted | median={med:.1f}")
        else:
            print(f"  {col}: kept raw | p05={p05:.1f}, median={med:.1f}, p95={p95:.1f}")

    return out


def sensor_summary(df: pd.DataFrame, save_path=None) -> pd.DataFrame:
    rows = []
    for col in SENSOR_COLS:
        s = df[col].dropna()
        if s.empty:
            rows.append({
                "sensor": col,
                "n": 0,
                "min": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "max": np.nan,
                "amplitude": np.nan,
            })
            continue

        rows.append({
            "sensor": col,
            "n": int(s.count()),
            "min": float(s.min()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "max": float(s.max()),
            "amplitude": float(s.max() - s.min()),
        })

    summary = pd.DataFrame(rows)

    if save_path is not None:
        summary.to_csv(save_path, index=False)
        print(f"Saved summary: {save_path}")

    return summary


# =============================================================================
# PLOTTING WITH END-OF-LINE LABELS
# =============================================================================

def _spread_label_positions(y_values: dict[str, float], min_gap: float) -> dict[str, float]:
    """Spread label y-positions to reduce overlap at the right edge."""
    if not y_values:
        return {}

    items = sorted(y_values.items(), key=lambda kv: kv[1])
    adjusted = {}
    last_y = -np.inf

    for name, y in items:
        y_new = max(y, last_y + min_gap)
        adjusted[name] = y_new
        last_y = y_new

    original_max = max(y_values.values())
    adjusted_max = max(adjusted.values())
    overshoot = adjusted_max - original_max

    if overshoot > 0:
        for name in adjusted:
            adjusted[name] -= overshoot * 0.5

    return adjusted


def plot_inline_labels(
    df: pd.DataFrame,
    cols: list[str],
    title: str,
    save_path,
    ylabel: str = "Temperature",
    resample_rule: str | None = "1min",
    figsize=(16, 8),
):
    """Plot selected columns and write each label at the right end of its line."""
    cols = [col for col in cols if col in df.columns]
    data = df[cols].copy()

    if resample_rule is not None:
        data = data.resample(resample_rule).mean()

    fig, ax = plt.subplots(figsize=figsize)

    lines = {}
    for col in cols:
        s = data[col].dropna()
        if s.empty:
            continue
        line, = ax.plot(s.index, s.values, linewidth=1.4, alpha=0.95)
        lines[col] = line

    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if not lines:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {save_path}")
        return

    # Extend right side to make room for inline labels.
    x_min = data.index.min()
    x_max = data.index.max()
    duration = x_max - x_min
    pad = duration * 0.11 if duration > pd.Timedelta(0) else pd.Timedelta(hours=1)
    x_label = x_max + pad * 0.12
    ax.set_xlim(x_min, x_max + pad)

    # Compute final y for every plotted line.
    final_y = {}
    final_x = {}
    for col, line in lines.items():
        s = data[col].dropna()
        final_x[col] = s.index[-1]
        final_y[col] = float(s.iloc[-1])

    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1.0)
    min_gap = y_span * 0.025
    label_y = _spread_label_positions(final_y, min_gap=min_gap)

    for col, line in lines.items():
        color = line.get_color()
        y0 = final_y[col]
        y1 = label_y[col]

        ax.plot([final_x[col], x_label], [y0, y1], color=color, linewidth=0.8, alpha=0.55)
        ax.text(
            x_label,
            y1,
            col,
            color=color,
            fontsize=9,
            va="center",
            ha="left",
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = load_multiple_ni(FILES)

    summary = sensor_summary(df, OUTPUT_DIR / "sensor_summary.csv")
    print("\nSensor summary:")
    print(summary.to_string(index=False))

    plot_inline_labels(
        df,
        cols=SENSOR_COLS,
        title="All NI Sensor Channels — Inline End Labels",
        save_path=OUTPUT_DIR / "all_NI_sensors_inline_labels.png",
        ylabel="Temperature",
        resample_rule="1min",
        figsize=(18, 9),
    )

    groups = {
        "CAM_box": ["T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc", "T2A", "T2A2", "T1Tb"],
        "C3_box": ["T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka", "T1A"],
        "RR_box": ["T2Ka", "T2Kd", "T2Kc", "2Ke", "T1Ta", "T1Ta2"],
        "roof_candidates": ["T2A2", "T1Tb", "T1A", "T3Ka", "T1Ta2", "T2Ka"],
        "room_air_candidates": ["T1Ka", "T3Kd", "T1Ta"],
    }

    for name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if cols:
            plot_inline_labels(
                df,
                cols=cols,
                title=f"NI Sensor Check — {name}",
                save_path=OUTPUT_DIR / f"{name}_inline_labels.png",
                ylabel="Temperature",
                resample_rule="1min",
                figsize=(16, 7),
            )

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
