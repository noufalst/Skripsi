"""
Shrink Davis WeatherLink TXT to a small weather CSV/ZIP for the green roof model.

IMPORTANT:
Your weather TXT uses DD/MM/YYYY date format.

Usage:
1) Put this file in the same folder as your large Davis TXT file.
2) Edit INPUT_FILE if needed.
3) Run:
   python shrink_weather_file_ddmmyyyy.py

Output:
- weather_clean_2026-04-03_to_2026-04-10.csv
- weather_clean_2026-04-03_to_2026-04-10.zip

Output columns:
timestamp, T_a, RH, u, rain, G_sol
"""

from pathlib import Path
import zipfile
import pandas as pd

# === EDIT THIS IF YOUR FILE NAME IS DIFFERENT ===
INPUT_FILE = "3-24april.txt"

# Overlap period with NI sensor data
DATE_START = "2026-04-03 14:58:00"
DATE_END   = "2026-04-10 16:15:59"

OUTPUT_CSV = "weather_clean_2026-04-03_to_2026-04-10.csv"
OUTPUT_ZIP = "weather_clean_2026-04-03_to_2026-04-10.zip"


def parse_weather_datetime(date_series, time_series):
    """
    Parse weather datetime using DD/MM/YYYY first.
    This prevents 03/04/2026 from being misread as March 4 instead of April 3.
    """
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()

    # DD/MM first because your file uses day/month/year
    formats = [
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%y %I:%M %p",

        # fallback only, in case some export uses month/day
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %I:%M %p",
    ]

    best_dt = None
    best_valid_count = -1
    best_format = None

    for fmt in formats:
        dt = pd.to_datetime(combined, format=fmt, errors="coerce")
        valid_count = dt.notna().sum()
        if valid_count > best_valid_count:
            best_dt = dt
            best_valid_count = valid_count
            best_format = fmt

    if best_valid_count <= 0:
        # Final fallback: dayfirst=True
        best_dt = pd.to_datetime(combined, errors="coerce", dayfirst=True)
        best_format = "pandas infer, dayfirst=True"

    print(f"Datetime format used: {best_format}")
    return best_dt


def shrink_weather_file(input_file):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path.resolve()}\n"
            "Put this script in the same folder as your Davis TXT file, "
            "or edit INPUT_FILE with the full path."
        )

    print(f"Reading: {input_path.resolve()}")

    # Davis WeatherLink export expected by your model:
    # tab-separated, 2 header rows, needed columns:
    # Date(0), Time(1), T_a(2), RH(5), u(7), rain(17), G_sol(19)
    df_raw = pd.read_csv(
        input_path,
        sep="\t",
        skiprows=2,
        header=None,
        na_values=["---", "  ---", " ---", "---  "],
        low_memory=False,
        encoding_errors="replace",
    )

    if df_raw.shape[1] < 20:
        raise ValueError(
            f"Expected at least 20 columns, found {df_raw.shape[1]}.\n"
            "Check if the file is really tab-separated Davis WeatherLink TXT."
        )

    df = pd.DataFrame({
        "timestamp": parse_weather_datetime(df_raw.iloc[:, 0], df_raw.iloc[:, 1]),
        "T_a":   pd.to_numeric(df_raw.iloc[:, 2],  errors="coerce"),   # Temp Out
        "RH":    pd.to_numeric(df_raw.iloc[:, 5],  errors="coerce"),   # Out Hum
        "u":     pd.to_numeric(df_raw.iloc[:, 7],  errors="coerce"),   # Wind Speed
        "rain":  pd.to_numeric(df_raw.iloc[:, 17], errors="coerce"),   # Rain
        "G_sol": pd.to_numeric(df_raw.iloc[:, 19], errors="coerce"),   # Solar Radiation
    })

    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # Basic physical cleaning
    df["G_sol"] = df["G_sol"].clip(lower=0)
    df["u"] = df["u"].clip(lower=0.1)
    df["RH"] = df["RH"].clip(lower=0, upper=100)
    df["rain"] = df["rain"].fillna(0).clip(lower=0)

    for col in ["T_a", "G_sol", "RH", "u"]:
        df[col] = df[col].interpolate(method="time")

    df = df.dropna(subset=["T_a", "G_sol", "RH", "u"])

    print(f"Full parsed period: {df.index.min()} -> {df.index.max()}")

    # Filter to overlap with NI data
    df = df.loc[DATE_START:DATE_END].copy()

    if df.empty:
        raise ValueError(
            "No data left after date filtering.\n"
            f"Parsed period may not overlap DATE_START={DATE_START} and DATE_END={DATE_END}.\n"
            "Check the printed 'Full parsed period' above."
        )

    df = df.reset_index()

    df.to_csv(OUTPUT_CSV, index=False)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUTPUT_CSV)

    print("")
    print(f"Rows after filter: {len(df)}")
    print(f"Filtered period: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"T_a: {df['T_a'].min():.1f} to {df['T_a'].max():.1f}")
    print(f"RH : {df['RH'].min():.1f} to {df['RH'].max():.1f}")
    print(f"G  : {df['G_sol'].min():.1f} to {df['G_sol'].max():.1f}")
    print("")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_ZIP}")
    print("")
    print("Upload the ZIP or CSV to ChatGPT instead of the original huge TXT.")

    return df


if __name__ == "__main__":
    shrink_weather_file(INPUT_FILE)
