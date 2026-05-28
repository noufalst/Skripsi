
"""
================================================================================
CAM SENSOR IDENTITY INFERENCE
================================================================================
Purpose
-------
Infer which NI temperature channels most likely represent:
    - outdoor/exposed roof surface
    - upper substrate / near-surface soil
    - lower substrate / damped soil
    - indoor air
    - inner roof/slab surface

This script does NOT prove sensor identity. It ranks candidates using signal behavior:
    - amplitude
    - correlation with solar radiation
    - correlation with ambient air
    - correlation with indoor air
    - time lag against solar radiation and exposed roof sensor
    - similarity to model outputs, if model prediction CSV is available

Why this is useful
------------------
If the model T_s_in is too flat compared with measured T1Tb, the issue may be:
    1. wrong physical target mapping,
    2. missing heat path,
    3. sensor not located where assumed,
    4. real system has edge/direct conduction not represented by the 1D model.

The script helps decide which measured sensor behaves like which model node.

Author: ChatGPT-assisted diagnostic
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: DEFAULT CHANNEL MAP
# ==============================================================================

NI_CHANNEL_NAMES = [
    "timestamp_serial",
    "T1Kd", "T1Kb", "T1Ke", "T1Ka", "T1Kc",
    "T3Kb", "T3Kd", "T3Ke", "T3Kc", "T3Ka",
    "T2Ka", "T2Kd", "T2Kc", "2Ke",  "T1A",
    "T2A",  "T2A2", "T1Tb", "T1Ta", "T1Ta2"
]

DEFAULT_CAM_CANDIDATES = ["T1Ka", "T1Ke", "T1Ta", "T1Tb", "T2A", "T2A2"]

DEFAULT_ROLE_HINTS = {
    "outdoor_exposed_surface": {
        "expected": "large amplitude, strong solar correlation, short solar lag",
        "likely_channels": ["T1Ta"],
    },
    "upper_substrate": {
        "expected": "moderate amplitude, solar-related but damped vs exposed surface",
        "likely_channels": ["T1Ke"],
    },
    "lower_substrate": {
        "expected": "low amplitude, delayed response, smoother",
        "likely_channels": ["T2A", "T2A2"],
    },
    "indoor_air": {
        "expected": "smooth, similar to room boundary, not directly solar-spiky",
        "likely_channels": ["T1Ka"],
    },
    "inner_roof_surface": {
        "expected": "damped response compared with exposed surface, lagged but not as flat as indoor air",
        "likely_channels": ["T1Tb"],
    },
}


# ==============================================================================
# SECTION 1B: SAFE DATAFRAME HELPERS
# ==============================================================================

def numeric_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric and drop columns that are fully non-numeric.
    This prevents pandas resample().mean() from crashing on string/object columns.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out


def safe_resample_1min(df: pd.DataFrame, interpolate_limit: int = 30) -> pd.DataFrame:
    """
    Robust 1-minute resample for mixed-type dataframes.
    """
    out = numeric_only_frame(df)
    if out.empty:
        return out
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.resample("1min").mean(numeric_only=True)
    out = out.interpolate(method="time", limit=interpolate_limit)
    return out



# ==============================================================================
# SECTION 2: LOADERS
# ==============================================================================

def _read_xlsx_numeric_xml(filepath: Path) -> pd.DataFrame:
    """
    Read LabVIEW-exported XLSX through XML directly.
    This avoids normal Excel parsing issues and matches the style of the existing model.
    """
    with zipfile.ZipFile(filepath, "r") as z:
        sheet_names = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        if not sheet_names:
            raise ValueError(f"No worksheet XML found in {filepath}")
        # Use first sheet by default
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
        raise ValueError(f"No numeric NI rows found in {filepath}")

    df = pd.DataFrame(data, columns=NI_CHANNEL_NAMES)

    labview_epoch = pd.Timestamp("1904-01-01")
    df["timestamp"] = labview_epoch + pd.to_timedelta(df["timestamp_serial"], unit="D")
    df = df.set_index("timestamp").drop(columns=["timestamp_serial"])
    df = df.sort_index()

    # Remove physically impossible extreme values, but do not aggressively interpolate here.
    for col in df.columns:
        if col.startswith("T") or col == "2Ke":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[(df[col] < -20) | (df[col] > 100), col] = np.nan

    return df


def load_ni_files(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        f = Path(f)
        if not f.exists():
            warnings.warn(f"NI file not found: {f}")
            continue
        print(f"Loading NI: {f}")
        frames.append(_read_xlsx_numeric_xml(f))

    if not frames:
        raise FileNotFoundError("No NI files could be loaded.")

    df = pd.concat(frames).sort_index()
    df = safe_resample_1min(df, interpolate_limit=30)

    return df



def _normalize_colname(x) -> str:
    """Normalize a column/header string for fuzzy weather matching."""
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def _excel_numeric_datetime(date_s: pd.Series, time_s: Optional[pd.Series] = None) -> pd.Series:
    """
    Robust conversion for Excel/Davis date + time columns.
    Handles:
      - normal date strings
      - Excel serial date numbers
      - separate Excel time fractions
      - normal time strings
    """
    date_raw = date_s.copy()

    # Case 1: Excel serial date number
    date_num = pd.to_numeric(date_raw, errors="coerce")
    if date_num.notna().mean() > 0.7 and date_num.dropna().median() > 20000:
        dt = pd.to_datetime(date_num, unit="D", origin="1899-12-30", errors="coerce")
    else:
        dt = pd.to_datetime(date_raw, errors="coerce")

    if time_s is None:
        return dt

    time_num = pd.to_numeric(time_s, errors="coerce")
    if time_num.notna().mean() > 0.7:
        # Excel time fraction or seconds/days-like numeric
        # Values < 2 are usually fractions of a day.
        if time_num.dropna().median() < 2:
            td = pd.to_timedelta(time_num.fillna(0), unit="D")
        else:
            td = pd.to_timedelta(time_num.fillna(0), unit="s")
        return dt.dt.normalize() + td

    # String time
    combined = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()
    dt2 = pd.to_datetime(combined, errors="coerce")
    if dt2.notna().sum() >= dt.notna().sum():
        return dt2
    return dt


def _find_header_row_for_weather(raw: pd.DataFrame, max_scan_rows: int = 40) -> Optional[int]:
    """
    Search a header row containing weather-like labels.
    Useful when Davis/WeatherLink Excel exports have metadata rows before the table.
    """
    target_words = ["date", "time", "temp", "hum", "wind", "rain", "solar", "rad"]
    best_row = None
    best_score = 0

    scan_n = min(max_scan_rows, len(raw))
    for i in range(scan_n):
        values = [_normalize_colname(v) for v in raw.iloc[i].tolist()]
        row_text = " ".join(values)
        score = sum(1 for w in target_words if w in row_text)
        if score > best_score:
            best_score = score
            best_row = i

    return best_row if best_score >= 3 else None


def _weather_from_position_fallback(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Last-resort Davis layout fallback:
      Date(0), Time(1), T_a(2), RH(5), u(7), rain(17), G_sol(19)
    """
    if raw.shape[1] < 20:
        raise ValueError("Position fallback needs at least 20 columns.")

    out = pd.DataFrame({
        "date": raw.iloc[:, 0],
        "time": raw.iloc[:, 1],
        "T_a": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
        "RH": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
        "u": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
        "rain": pd.to_numeric(raw.iloc[:, 17], errors="coerce"),
        "G_sol": pd.to_numeric(raw.iloc[:, 19], errors="coerce"),
    })
    idx = _excel_numeric_datetime(out["date"], out["time"])
    out.index = idx
    out = out[~out.index.isna()]
    out = out.drop(columns=["date", "time"])
    return out





def _coerce_weather_timestamp(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    """
    Convert weather Date + Time to datetime.

    IMPORTANT:
    The uploaded weatherfile uses Indonesian/WeatherLink style date order:
        DD/MM/YY

    Example:
        31/03/26 = 31 March 2026
        24/04/26 = 24 April 2026

    The earlier loader assumed MM/DD/YY, so dates after the 12th day were dropped
    and the weather plot became short/incomplete.
    """
    date_raw = date_col.copy()
    time_raw = time_col.copy()

    date_str = date_raw.astype(str).str.strip()
    time_str = time_raw.astype(str).str.strip()

    # 1) Correct format for this file: DD/MM/YY HH:MM
    ts_dayfirst_exact = pd.to_datetime(
        date_str + " " + time_str,
        format="%d/%m/%y %H:%M",
        errors="coerce",
    )

    # 2) Fallback: generic dayfirst parse, handles "0:01" etc.
    ts_dayfirst_generic = pd.to_datetime(
        date_str + " " + time_str,
        dayfirst=True,
        errors="coerce",
    )

    # 3) Excel serial date + time fraction fallback.
    date_num = pd.to_numeric(date_raw, errors="coerce")
    time_num = pd.to_numeric(time_raw, errors="coerce")
    excel_date = pd.to_datetime(date_num, unit="D", origin="1899-12-30", errors="coerce")

    if time_num.notna().any():
        median_time = time_num.dropna().median() if time_num.notna().any() else 0
        if median_time < 2:
            excel_time = pd.to_timedelta(time_num.fillna(0), unit="D")
        else:
            excel_time = pd.to_timedelta(time_num.fillna(0), unit="s")
    else:
        excel_time = pd.to_timedelta(0, unit="s")

    ts_excel = excel_date.dt.normalize() + excel_time

    # Combine in priority order.
    ts = ts_dayfirst_exact.copy()
    ts = ts.where(ts.notna(), ts_dayfirst_generic)
    ts = ts.where(ts.notna(), ts_excel)

    return ts


def load_weather_file(filepath: Path) -> pd.DataFrame:
    """
    Exact + fast loader for user's `weatherfile mar-april.xlsx`.

    File structure observed:
      Sheet: 3-24april
      Row 0: Column1, Column2, ...
      Row 1: grouped header
      Row 2: actual header names
      Row 3+: data

    Fast read:
      skiprows=3 and usecols only needed Davis columns.

    Column mapping, zero-based:
      0  Date
      1  Time
      2  Temp Out       -> T_a
      5  Out Hum        -> RH
      7  Wind Speed     -> u
      17 Rain           -> rain
      19 Solar Rad.     -> G_sol

    v5 fix:
      Date cells can switch between strings and Excel serial numbers.
      This loader now handles both, preventing the weather series from stopping
      halfway through the CAM window.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    print(f"Loading weather: {filepath}")

    if filepath.suffix.lower() in [".xlsx", ".xls"]:
        try:
            raw = pd.read_excel(
                filepath,
                sheet_name="3-24april",
                header=None,
                skiprows=3,
                usecols=[0, 1, 2, 5, 7, 17, 19],
            )
        except ValueError:
            raw = pd.read_excel(
                filepath,
                sheet_name=0,
                header=None,
                skiprows=3,
                usecols=[0, 1, 2, 5, 7, 17, 19],
            )

        raw.columns = ["date", "time", "T_a", "RH", "u", "rain", "G_sol"]

        out = raw.copy()
        out["timestamp"] = _coerce_weather_timestamp(out["date"], out["time"])
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
        out = out.drop(columns=["date", "time"])

    else:
        raw = pd.read_csv(
            filepath,
            sep="\t",
            skiprows=2,
            header=None,
            na_values=["---", "  ---", " ---", "---  "],
            low_memory=False,
            encoding_errors="ignore",
            usecols=[0, 1, 2, 5, 7, 17, 19],
        )
        raw.columns = ["date", "time", "T_a", "RH", "u", "rain", "G_sol"]

        out = raw.copy()
        out["timestamp"] = _coerce_weather_timestamp(out["date"], out["time"])
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
        out = out.drop(columns=["date", "time"])

    for col in ["T_a", "RH", "u", "rain", "G_sol"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["G_sol"] = out["G_sol"].clip(lower=0)
    out["RH"] = out["RH"].clip(0, 100)
    out["u"] = out["u"].clip(lower=0.1)
    out["rain"] = out["rain"].fillna(0).clip(lower=0)

    out = safe_resample_1min(out, interpolate_limit=30)

    print(f"  Weather range : {out.index.min()} -> {out.index.max()} | rows={len(out)}")
    print(f"  G_sol valid   : {out['G_sol'].notna().sum()} rows | max={out['G_sol'].max():.1f} W/m²")
    print(f"  T_a valid     : {out['T_a'].notna().sum()} rows | min={out['T_a'].min():.1f}, max={out['T_a'].max():.1f} °C")

    return out


def load_model_prediction(folder: Path) -> Optional[pd.DataFrame]:
    """
    Try to load model output from previous CAM runs.
    Accepts common filenames.
    """
    folder = Path(folder)
    candidates = [
        "cam_gsw_prediction_eval_window.csv",
        "cam_gsw_prediction_full_with_spinup.csv",
        "cam_physical_prediction_eval_window.csv",
        "cam_prediction_calibrated.csv",
        "cam_prediction_baseline.csv",
    ]
    for name in candidates:
        f = folder / name
        if f.exists():
            print(f"Loading model prediction: {f}")
            df = pd.read_csv(f)
            time_col = next((c for c in df.columns if str(c).lower() in ["datetime", "timestamp", "time"]), None)
            if time_col is None:
                # Try first col
                time_col = df.columns[0]
            idx = pd.to_datetime(df[time_col], errors="coerce")
            df = df.drop(columns=[time_col])
            df.index = idx
            df = df[~df.index.isna()].sort_index()
            return safe_resample_1min(df, interpolate_limit=30)
    return None


# ==============================================================================
# SECTION 3: METRICS
# ==============================================================================

def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return x["a"], x["b"]


def rmse(a: pd.Series, b: pd.Series) -> float:
    x, y = _align(a, b)
    if len(x) < 5:
        return np.nan
    return float(np.sqrt(np.mean((x.values - y.values) ** 2)))


def corr(a: pd.Series, b: pd.Series) -> float:
    x, y = _align(a, b)
    if len(x) < 5:
        return np.nan
    return float(np.corrcoef(x.values, y.values)[0, 1])


def amplitude(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) == 0:
        return np.nan
    # Robust amplitude avoids one-sample spikes
    return float(v.quantile(0.95) - v.quantile(0.05))


def lagged_corr(a: pd.Series, b: pd.Series, max_lag_min: int = 360) -> Tuple[float, int]:
    """
    Return best absolute correlation and lag in minutes.
    Positive lag means b shifted later relative to a.
    """
    best_c = np.nan
    best_lag = 0
    for lag in range(-max_lag_min, max_lag_min + 1, 5):
        b_shifted = b.shift(lag, freq="1min")
        c = corr(a, b_shifted)
        if np.isfinite(c) and (not np.isfinite(best_c) or abs(c) > abs(best_c)):
            best_c = c
            best_lag = lag
    return float(best_c) if np.isfinite(best_c) else np.nan, int(best_lag)


def normalized_score(value: float, low_is_good: bool = False, scale: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    if low_is_good:
        return float(np.exp(-abs(value) / scale))
    return float(value)


# ==============================================================================
# SECTION 4: ROLE INFERENCE
# ==============================================================================

def infer_roles(data: pd.DataFrame,
                weather: Optional[pd.DataFrame],
                model: Optional[pd.DataFrame],
                candidates: List[str]) -> pd.DataFrame:
    """
    Score each measured sensor for likely physical roles.
    Scores are heuristic, not proof.
    """
    rows = []

    solar = weather["G_sol"] if weather is not None and "G_sol" in weather.columns else None
    Tair = weather["T_a"] if weather is not None and "T_a" in weather.columns else None

    for ch in candidates:
        if ch not in data.columns:
            continue

        s = data[ch]
        amp = amplitude(s)
        c_solar, lag_solar = (np.nan, np.nan)
        c_tair = np.nan
        if solar is not None:
            c_solar, lag_solar = lagged_corr(solar, s, max_lag_min=360)
        if Tair is not None:
            c_tair = corr(Tair, s)

        # Relationships with known/assumed channels if present
        c_t1ta = corr(data["T1Ta"], s) if "T1Ta" in data.columns and ch != "T1Ta" else np.nan
        c_t1ka = corr(data["T1Ka"], s) if "T1Ka" in data.columns and ch != "T1Ka" else np.nan

        # Role scoring heuristics
        # Exposed: high amplitude, high solar corr, low lag
        exposed_score = (
            normalized_score(amp, scale=10)
            + normalized_score(abs(c_solar) if np.isfinite(c_solar) else 0)
            + normalized_score(abs(lag_solar) if np.isfinite(lag_solar) else 999, low_is_good=True, scale=90)
        )

        # Indoor air: low/moderate amplitude, high similarity to T1Ka or smooth indoor boundary
        indoor_score = (
            normalized_score(amp - 4, low_is_good=True, scale=5)
            + normalized_score(abs(c_t1ka) if np.isfinite(c_t1ka) else (1.0 if ch == "T1Ka" else 0))
        )

        # Upper substrate: moderate amplitude, positive solar relation, lagged but not extreme
        upper_score = (
            normalized_score(abs(amp - 8), low_is_good=True, scale=5)
            + normalized_score(abs(c_solar) if np.isfinite(c_solar) else 0)
            + normalized_score(abs((lag_solar if np.isfinite(lag_solar) else 180) - 60), low_is_good=True, scale=180)
        )

        # Lower substrate: low amplitude, delayed, smooth
        lower_score = (
            normalized_score(amp, low_is_good=True, scale=6)
            + normalized_score(abs((lag_solar if np.isfinite(lag_solar) else 240) - 180), low_is_good=True, scale=240)
        )

        # Inner roof: moderate-high amplitude, high correlation with exposed surface, but damped relative to exposed
        inner_score = (
            normalized_score(abs(amp - 12), low_is_good=True, scale=8)
            + normalized_score(abs(c_t1ta) if np.isfinite(c_t1ta) else 0)
        )

        rows.append({
            "channel": ch,
            "amp_95_05_C": amp,
            "corr_with_solar_best": c_solar,
            "lag_from_solar_min": lag_solar,
            "corr_with_Tair": c_tair,
            "corr_with_T1Ta": c_t1ta,
            "corr_with_T1Ka": c_t1ka,
            "score_exposed_surface": exposed_score,
            "score_upper_substrate": upper_score,
            "score_lower_substrate": lower_score,
            "score_indoor_air": indoor_score,
            "score_inner_roof_surface": inner_score,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    for role in [
        "exposed_surface", "upper_substrate", "lower_substrate",
        "indoor_air", "inner_roof_surface"
    ]:
        score_col = f"score_{role}"
        if score_col in out.columns:
            out[f"rank_{role}"] = out[score_col].rank(ascending=False, method="min").astype(int)

    return out.sort_values("channel")


def model_sensor_matrix(data: pd.DataFrame,
                        model: Optional[pd.DataFrame],
                        candidates: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model is None:
        return pd.DataFrame(), pd.DataFrame()

    model_cols = [c for c in model.columns if any(k in c for k in [
        "T_f", "T_g_top", "T_g_mid", "T_g_bot", "T_s_in", "T_in_used"
    ])]
    if not model_cols:
        return pd.DataFrame(), pd.DataFrame()

    corr_rows = []
    rmse_rows = []
    for ch in candidates:
        if ch not in data.columns:
            continue
        corr_row = {"sensor": ch}
        rmse_row = {"sensor": ch}
        for mc in model_cols:
            corr_row[mc] = corr(data[ch], model[mc])
            rmse_row[mc] = rmse(data[ch], model[mc])
        corr_rows.append(corr_row)
        rmse_rows.append(rmse_row)

    return pd.DataFrame(corr_rows).set_index("sensor"), pd.DataFrame(rmse_rows).set_index("sensor")


# ==============================================================================
# SECTION 5: PLOTS
# ==============================================================================

def plot_sensor_overview(data: pd.DataFrame,
                         weather: Optional[pd.DataFrame],
                         candidates: List[str],
                         outpath: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    for ch in candidates:
        if ch in data.columns:
            ax.plot(data.index, data[ch], label=ch, lw=1.2)
    ax.set_ylabel("Measured NI temperature (°C)")
    ax.set_title("CAM measured NI channels — identity/path inference")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)

    ax = axes[1]
    # Plot normalized signals for shape comparison
    for ch in candidates:
        if ch in data.columns:
            s = data[ch]
            span = s.quantile(0.95) - s.quantile(0.05)
            if np.isfinite(span) and span > 0:
                z = (s - s.median()) / span
                ax.plot(data.index, z, label=ch, lw=1.0)
    ax.set_ylabel("Normalized shape")
    ax.set_title("Shape comparison, normalized by robust amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)

    ax = axes[2]
    if weather is not None:
        if "T_a" in weather:
            ax.plot(weather.index, weather["T_a"], label="T_a", lw=1.2)
        if "G_sol" in weather:
            ax2 = ax.twinx()
            ax2.plot(weather.index, weather["G_sol"], linestyle="--", label="G_sol", lw=1.0)
            ax2.set_ylabel("Solar (W/m²)")
            ax2.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("Weather")
    ax.set_title("Weather drivers")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_role_scores(scores: pd.DataFrame, outpath: Path) -> None:
    if scores.empty:
        return
    role_cols = [c for c in scores.columns if c.startswith("score_")]
    fig, axes = plt.subplots(len(role_cols), 1, figsize=(12, 3 * len(role_cols)), sharex=True)
    if len(role_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, role_cols):
        ax.bar(scores["channel"], scores[col])
        ax.set_title(col.replace("score_", "Likely role: "))
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("NI channel")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, title: str, outpath: Path) -> None:
    if matrix.empty:
        return
    arr = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ==============================================================================
# SECTION 6: MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer likely CAM sensor identities from NI signal behavior."
    )
    parser.add_argument("--base-dir", default=".", help="Folder containing data files")
    parser.add_argument(
        "--ni-files",
        nargs="*",
        default=[
            "Pengukuran 30_1 Maret 2026.xlsx",
            "Pengukuran 30_2 Maret 2026.xlsx",
            "Pengukuran 3 April 2026.xlsx",
        ],
        help="NI XLSX files to load",
    )
    parser.add_argument("--weather-file", default="weatherfile mar-april.xlsx", help="Weather file, optional")
    parser.add_argument("--model-output-dir", default="outputs_cam_gsw_fixed_inputs", help="Folder with model prediction CSV, optional")
    parser.add_argument("--start", default="2026-03-31 11:58:00", help="Start datetime")
    parser.add_argument("--end", default="2026-04-02 21:42:00", help="End datetime")
    parser.add_argument("--output-dir", default="outputs_cam_sensor_identity", help="Output folder")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=DEFAULT_CAM_CANDIDATES,
        help="Candidate NI channels to rank",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ni_paths = [base_dir / f for f in args.ni_files]
    ni = load_ni_files(ni_paths)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    ni = ni[(ni.index >= start) & (ni.index <= end)].copy()

    if ni.empty:
        raise ValueError(f"No NI data in selected window: {start} -> {end}")

    weather = None
    weather_path = base_dir / args.weather_file
    if weather_path.exists():
        try:
            weather = load_weather_file(weather_path)
            weather = weather[(weather.index >= start) & (weather.index <= end)].copy()
            if weather.empty:
                warnings.warn(
                    f"Weather loaded, but no overlap with selected window {start} -> {end}. "
                    "Solar correlation will be NaN."
                )
            elif "G_sol" in weather.columns:
                print(f"  Window G_sol valid: {weather['G_sol'].notna().sum()} / {len(weather)}")
        except Exception as e:
            warnings.warn(f"Weather could not be loaded: {e}")
            weather = None
    else:
        warnings.warn(f"Weather file not found, continuing sensor-only: {weather_path}")

    model = load_model_prediction(base_dir / args.model_output_dir)
    if model is not None:
        model = model[(model.index >= start) & (model.index <= end)].copy()

    candidates = [c for c in args.candidates if c in ni.columns]
    if not candidates:
        raise ValueError("None of the candidate channels exist in NI data.")

    print("\nCandidate channels:", ", ".join(candidates))
    print(f"Window: {ni.index.min()} -> {ni.index.max()} | rows={len(ni)}")

    scores = infer_roles(ni, weather, model, candidates)
    scores.to_csv(out_dir / "cam_sensor_identity_role_scores.csv", index=False)

    corr_mat, rmse_mat = model_sensor_matrix(ni, model, candidates)
    if not corr_mat.empty:
        corr_mat.to_csv(out_dir / "cam_model_sensor_corr_matrix.csv")
        rmse_mat.to_csv(out_dir / "cam_model_sensor_rmse_matrix.csv")

    # Pairwise measured relationships
    pair_rows = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            pair_rows.append({
                "sensor_a": a,
                "sensor_b": b,
                "corr": corr(ni[a], ni[b]),
                "rmse_C": rmse(ni[a], ni[b]),
                "amp_a_C": amplitude(ni[a]),
                "amp_b_C": amplitude(ni[b]),
                "amp_ratio_b_over_a": amplitude(ni[b]) / amplitude(ni[a]) if amplitude(ni[a]) else np.nan,
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "cam_measured_pairwise_relationships.csv", index=False)

    # JSON summary
    summary = {
        "window": {"start": str(start), "end": str(end), "rows": int(len(ni))},
        "candidates": candidates,
        "top_role_candidates": {},
        "interpretation_notes": [
            "This is a heuristic ranking, not proof.",
            "A plug-off test remains the best direct verification.",
            "High amplitude + high solar correlation suggests exposed/near-surface sensor.",
            "Low amplitude + lagged response suggests deeper substrate or indoor air.",
            "If measured inner roof sensor follows exposed surface strongly, target mapping may be wrong for full 1D green-roof T_s_in.",
        ],
    }

    for role in [
        "exposed_surface", "upper_substrate", "lower_substrate",
        "indoor_air", "inner_roof_surface"
    ]:
        score_col = f"score_{role}"
        if score_col in scores.columns:
            top = scores.sort_values(score_col, ascending=False).head(3)
            summary["top_role_candidates"][role] = top[["channel", score_col, "amp_95_05_C", "corr_with_solar_best", "lag_from_solar_min"]].to_dict("records")

    with open(out_dir / "cam_sensor_identity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_sensor_overview(ni, weather, candidates, out_dir / "cam_sensor_identity_overview.png")
    plot_role_scores(scores, out_dir / "cam_sensor_identity_role_scores.png")
    plot_heatmap(corr_mat, "Measured sensor vs model output: correlation", out_dir / "cam_model_sensor_corr_heatmap.png")
    plot_heatmap(rmse_mat, "Measured sensor vs model output: RMSE (°C)", out_dir / "cam_model_sensor_rmse_heatmap.png")

    print("\nDone.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("\nQuick top candidates:")
    for role, items in summary["top_role_candidates"].items():
        if items:
            print(f"  {role}: " + ", ".join([f"{x['channel']} ({x[f'score_{role}']:.2f})" for x in items[:3]]))


if __name__ == "__main__":
    main()
