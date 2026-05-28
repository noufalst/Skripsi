"""
Shrink Davis WeatherLink TXT to a small weather CSV/ZIP for the green roof model.

Usage:
1) Put this file in the same folder as your large Davis TXT file.
2) Edit INPUT_FILE if needed.
3) Run: python shrink_weather_file.py

Output columns:
timestamp, T_a, RH, u, rain, G_sol
"""

from pathlib import Path
import zipfile
import pandas as pd

INPUT_FILE = "3-24april.txt"
DATE_START = "2026-04-03 14:58:00"
DATE_END   = "2026-04-10 16:15:59"

OUTPUT_CSV = "weather_clean_2026-04-03_to_2026-04-10.csv"
OUTPUT_ZIP = "weather_clean_2026-04-03_to_2026-04-10.zip"


def parse_weather_datetime(date_series, time_series):
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    formats = [
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M",
    ]
    for fmt in formats:
        dt = pd.to_datetime(combined, format=fmt, errors="coerce")
        if dt.notna().sum() > 0:
            return dt
    return pd.to_datetime(combined, errors="coerce")


def shrink_weather_file(input_file):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path.resolve()}\n"
            "Put this script in the same folder as your Davis TXT file, "
            "or edit INPUT_FILE with the full path."
        )

    print(f"Reading: {input_path.resolve()}")

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
            f"Expected at least 20 columns, found {df_raw.shape[1]}. "
            "Check if the file is tab-separated Davis WeatherLink TXT."
        )

    df = pd.DataFrame({
        "timestamp": parse_weather_datetime(df_raw.iloc[:, 0], df_raw.iloc[:, 1]),
        "T_a":   pd.to_numeric(df_raw.iloc[:, 2],  errors="coerce"),
        "RH":    pd.to_numeric(df_raw.iloc[:, 5],  errors="coerce"),
        "u":     pd.to_numeric(df_raw.iloc[:, 7],  errors="coerce"),
        "rain":  pd.to_numeric(df_raw.iloc[:, 17], errors="coerce"),
        "G_sol": pd.to_numeric(df_raw.iloc[:, 19], errors="coerce"),
    })

    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    df["G_sol"] = df["G_sol"].clip(lower=0)
    df["u"] = df["u"].clip(lower=0.1)
    df["RH"] = df["RH"].clip(lower=0, upper=100)
    df["rain"] = df["rain"].fillna(0).clip(lower=0)

    for col in ["T_a", "G_sol", "RH", "u"]:
        df[col] = df[col].interpolate(method="time")

    df = df.dropna(subset=["T_a", "G_sol", "RH", "u"])
    df = df.loc[DATE_START:DATE_END].copy()

    if df.empty:
        raise ValueError(
            "No data left after date filtering. Check DATE_START/DATE_END "
            "or date format in the weather file."
        )

    df = df.reset_index()
    df.to_csv(OUTPUT_CSV, index=False)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUTPUT_CSV)

    print(f"Rows after filter: {len(df)}")
    print(f"Period: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_ZIP}")
    print("Upload the ZIP or CSV instead of the original huge TXT.")

    return df


if __name__ == "__main__":
    shrink_weather_file(INPUT_FILE)
