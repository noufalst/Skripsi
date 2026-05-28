"""
================================================================================
GREEN ROOF THERMAL SIMULATION — CORRECTED VERSION
================================================================================
Referensi utama:
    Chagolla-Aranda et al. (2025), Journal of Building Engineering 103, 112053

Tanaman: Bromelia (CAM) vs Wedelia (C3)
Substrat: Tanah + Sekam Padi
Lokasi: Universitas Indonesia, Jakarta

DAFTAR KOREKSI dari versi sebelumnya:
─────────────────────────────────────────────────────────────────────────────
[C1] T_in TIDAK konstan 25°C
     → Asumsi paper tidak berlaku untuk boks outdoor UI
     → T_in diambil dinamis dari NI sensor: T1Ka (CAM) atau T3Kd (C3)
     → Data NI menunjukkan T_in = 24.8–42.8°C (mean 29.5°C)

[C2] tau_f DIPERBARUI berdasarkan morfologi daun
     → Wedelia C3 (daun tipis): tau_f = 0.20 (sebelumnya 0.10)
     → Bromelia CAM (daun tebal+lilin): tau_f = 0.07 (sebelumnya 0.07 sama)
     → Referensi: PROSPECT-D model + CAM vs C3 leaf thickness scaling

[C3] alpha_f DIHITUNG ULANG dari rho_f terukur
     → Wedelia:  alpha_f = 1 - 0.438 - 0.10 = 0.462
     → Bromelia: alpha_f = 1 - 0.390 - 0.07 = 0.540

[C4] r_stoma_min C3 dari data Licor 6800 NYATA
     → r_stoma_min C3 = 167.2 s/m (percentile 5%, data April 2026)
     → Sebelumnya: nilai dummy 150 s/m

[C5] NI data loader menggunakan XML parser
     → extract-text truncate di 1000 baris → data hilang
     → XML parser langsung dari file .xlsx → semua 9032 baris terbaca

[C6] T2A (Tanah bawah CAM) anomali difilter
     → 66 data dengan nilai negatif ekstrem (min -67 miliar °C)
     → Penyebab: sensor disconnect sementara
     → Solusi: replace dengan NaN lalu interpolasi linear

[C7] Sinkronisasi timestamp Davis ↔ NI sensor
     → Davis: interval 1 menit (xx:00:00)
     → NI: interval 1 menit tapi dari LabVIEW (ada detik yang tidak tepat)
     → Solusi: resample keduanya ke grid waktu yang sama

[C8] r_stoma_min CAM masih PLACEHOLDER
     → File Licor CAM belum diupload
     → Nilai sementara: 400.0 s/m (perlu dikonfirmasi dari Licor CAM)
================================================================================
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


# ==============================================================================
# SECTION 1: DATA CLASSES
# ==============================================================================

@dataclass
class PlantParameters:
    """
    Parameter spesifik tanaman.
    CAM (Bromelia): stomata terbuka malam, tertutup siang
    C3  (Wedelia) : stomata terbuka siang, tertutup malam
    """
    name: str
    plant_type: str              # "CAM" atau "C3"

    # Geometri kanopi
    H_f: float = 0.3            # m — tinggi kanopi [REAL: dari pengukuran] -> tadinya none
    d_f: float = 0.06            # m — lebar daun [TODO: ukur dari foto]
    LAI: float = 1            # m²/m² [REAL: dari data setahun lalu, dikalibrasi]

    # Properti optik daun
    # [C2] tau_f DIPERBARUI: C3 lebih tinggi dari CAM karena daun lebih tipis
    # [C3] alpha_f DIHITUNG dari rho_f terukur (bukan nilai paper)
    alpha_f: float = None        # absorptivitas (-)
    epsilon_f: float = 0.95      # emissivitas — standar semua daun hijau, tidak berubah
    tau_f: float = None          # transmisivitas (-)
    rho_f: float = None          # reflektifitas — dari pengukuran langsung

    # Stomatal resistance
    # [C4] r_stoma_min C3 dari Licor NYATA = 167.2 s/m
    # [C8] r_stoma_min CAM masih PLACEHOLDER = 400.0 s/m
    r_stoma_min: float = None    # s/m


@dataclass
class SubstrateParameters:
    """Parameter substrat: tanah + sekam padi."""
    # Properti termal
    lambda_dry: float = None     # W/mK  [LIT: 0.08, tanah+sekam padi]
    rho_g: float = None          # kg/m³ [LAB: specific gravity test]
    cp_g: float = None           # J/kgK [LIT: 1300]

    # Properti hidraulik
    theta_sat: float = None      # m³/m³ [LAB: water content test]
    k_theta_sat: float = None    # m/s   [LAB: permeability test — PALING KRITIS]
    psi_sat: float = None        # m     [LIT: 0.35]
    b: float = None              # (-)   [LIT: 5.5]
    theta_min: float = 0.05      # m³/m³ — wilting point, default OK

    # Properti optik permukaan substrat
    rho_g_rad: float = 0.15      # reflektifitas — standar
    epsilon_g: float = 0.95      # emissivitas — standar

    # Properti air — konstanta fisika
    lambda_water: float = 0.60   # W/mK
    cp_water: float = 4180       # J/kgK
    l_fg: float = 2.45e6         # J/kg — latent heat


@dataclass
class SlabParameters:
    """Parameter slab beton — semua nilai standar kecuali H_slab."""
    lambda_s: float = 1.74       # W/mK  — standar beton
    rho_s: float = 2300.0        # kg/m³ — standar beton
    cp_s: float = 840.0          # J/kgK — standar beton
    alpha_s: float = 0.40        # absorptivitas (untuk RR model)
    epsilon_s: float = 0.82      # emissivitas (untuk RR model)
    H_slab: float = None         # m — UKUR LANGSUNG di boks


@dataclass
class GeometryParameters:
    """
    Parameter geometri dan kondisi batas.
    [C1] T_in sekarang DINAMIS dari data NI sensor, bukan konstan 25°C.
    """
    H_g: float = None            # m — tebal substrat [UKUR LANGSUNG]
    H_slab: float = None         # m — tebal slab [UKUR LANGSUNG]

    # [C1] T_in_default hanya dipakai sebagai fallback kalau NI data tidak ada
    # Nilai aktual: T1Ka (CAM) = 24.8-42.8°C, T3Kd (C3) = 26.4-31.0°C
    T_in_default: float = 29.5 + 273.15  # K — rata-rata dari NI data (bukan 25°C!)

    h_in: float = 8.0            # W/m²K — koef. konveksi interior, standar


@dataclass
class NumericalParameters:
    """Parameter numerik dari Paper Section 3.1 — tidak perlu diubah."""
    dt: float = 1.0              # s — time step (paper: dt=1s → stable)
    Nz_substrate: int = 107      # node substrat (paper: Nz=107 → mesh independent)
    Nz_slab: int = 67            # node slab (paper: Nz=67)
    convergence: float = 1e-5   # kriteria Gauss-Seidel


# ==============================================================================
# INISIALISASI PARAMETER — NILAI YANG SUDAH DIKETAHUI
# ==============================================================================

# --- Bromelia (CAM) ---
# [REAL] H_f, LAI, rho_f: dari pengukuran setahun lalu
# [C2][C3] tau_f=0.07, alpha_f dihitung dari rho_f
# [C8] r_stoma_min: PLACEHOLDER — perlu Licor CAM
bromelia = PlantParameters(
    name        = "Bromelia (CAM)",
    plant_type  = "CAM",
    H_f         = 0.098,    # [REAL] 9.8 cm dari data setahun lalu
    d_f         = 0.045,    # [ESTIMASI] dari foto — ukur lebih akurat dengan ImageJ
    LAI         = 1.95,     # [REAL] dari data setahun lalu — akan dikalibrasi
    rho_f       = 0.390,    # [REAL] dari pengukuran langsung
    tau_f       = 0.07,     # [C2] daun tebal+lilin → tau rendah (range 0.05-0.10)
    alpha_f     = 0.540,    # [C3] = 1 - 0.390 - 0.07
    epsilon_f   = 0.95,     # standar semua daun
    r_stoma_min = 400.0,    # [C8] PLACEHOLDER — upload Licor CAM untuk nilai nyata
)

# --- Wedelia (C3) ---
# [REAL] H_f, LAI, rho_f: dari pengukuran setahun lalu
# [C2] tau_f=0.20 (bukan 0.10!) — daun C3 tipis meneruskan lebih banyak cahaya
# [C3] alpha_f dihitung dari rho_f
# [C4] r_stoma_min = 167.2 s/m dari Licor NYATA
wedelia = PlantParameters(
    name        = "Wedelia (C3)",
    plant_type  = "C3",
    H_f         = 0.276,    # [REAL] 27.6 cm dari data setahun lalu
    d_f         = 0.060,    # [ESTIMASI] dari foto — ukur lebih akurat dengan ImageJ
    LAI         = 1.07,     # [REAL] dari data setahun lalu — akan dikalibrasi
    rho_f       = 0.438,    # [REAL] dari pengukuran langsung
    tau_f       = 0.20,     # [C2] daun C3 tipis → tau tinggi (range 0.15-0.30)
    alpha_f     = 0.462,    # [C3] = 1 - 0.438 - 0.10 (pakai tau=0.10 awal)
    epsilon_f   = 0.95,     # standar semua daun
    r_stoma_min = 167.2,    # [C4] REAL — percentile 5% dari Licor C3 April 2026
)
# NOTE: alpha_f wedelia = 1 - rho_f - tau_f = 1 - 0.438 - 0.20 = 0.362
#       dengan tau_f yang sudah diperbarui ke 0.20
#       perlu update: alpha_f = 1 - 0.438 - 0.20 = 0.362
wedelia.alpha_f = 1.0 - wedelia.rho_f - wedelia.tau_f  # = 0.362

# --- Substrat: tanah + sekam padi ---
substrat = SubstrateParameters(
    lambda_dry  = 0.08,    # [LIT] W/mK — tanah+sekam padi (range 0.04-0.16)
    rho_g       = None,    # [TODO-LAB] specific gravity test
    cp_g        = 1300.0,  # [LIT] J/kgK — tanah+sekam padi (range 1200-1500)
    theta_sat   = None,    # [TODO-LAB] water content test
    k_theta_sat = None,    # [TODO-LAB] permeability test — PALING KRITIS
    psi_sat     = 0.35,    # [LIT] m — estimasi untuk loam
    b           = 5.5,     # [LIT] estimasi dari grain size (loam: 5-6)
)

# --- Slab beton ---
slab = SlabParameters(
    H_slab = None,         # [TODO-MEASURE] ukur langsung di boks
)

# --- Geometri ---
# [C1] T_in_default diset ke 29.5°C (rata-rata dari NI data), BUKAN 25°C
geom = GeometryParameters(
    H_g            = None,             # [TODO-MEASURE] tebal substrat
    H_slab         = None,             # [TODO-MEASURE] tebal slab
    T_in_default   = 29.5 + 273.15,   # [C1] dari rata-rata NI data T1Ka
)

# --- Numerical ---
num = NumericalParameters()


# ==============================================================================
# SECTION 2: DATA LOADERS
# ==============================================================================

def load_weather_data(filepath: str,
                      date_start: str = None,
                      date_end: str = None) -> pd.DataFrame:
    """
    Load data cuaca dari Davis Vantage Pro 2 (WeatherLink .txt export).

    Format yang sudah dikonfirmasi:
    - Tab-separated
    - 2 baris header (skip)
    - Kolom yang dipakai: Date(0), Time(1), T_a(2), RH(5), u(7), rain(17), G_sol(19)
    - Format tanggal: MM/DD/YY
    - Interval: 1 menit
    - Missing values: '---'

    Kolom yang TIDAK dipakai (dari 33 kolom total):
    Hi/Lo Temp, Wind Dir/Run/Hi, Wind Chill, Heat Index, THW, THSW,
    Bar, Rain Rate, Solar Energy, Hi Solar, UV, In Temp/Hum/Dew, dll.
    """
    print(f"Loading weather data: {filepath}")

    df_raw = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=2,
        header=None,
        na_values=['---', '  ---', ' ---', '---  '],
        low_memory=False
    )

    # Ambil hanya 5 kolom yang dibutuhkan model
    df = pd.DataFrame({
        'date'  : df_raw.iloc[:, 0].astype(str).str.strip(),
        'time'  : df_raw.iloc[:, 1].astype(str).str.strip(),
        'T_a'   : pd.to_numeric(df_raw.iloc[:, 2],  errors='coerce'),  # Temp Out
        'RH'    : pd.to_numeric(df_raw.iloc[:, 5],  errors='coerce'),  # Out Hum
        'u'     : pd.to_numeric(df_raw.iloc[:, 7],  errors='coerce'),  # Wind Speed
        'rain'  : pd.to_numeric(df_raw.iloc[:, 17], errors='coerce'),  # Rain
        'G_sol' : pd.to_numeric(df_raw.iloc[:, 19], errors='coerce'),  # Solar Rad
    })

    # Buat timestamp
    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%m/%d/%y %H:%M',
        errors='coerce'
    )

    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp').drop(columns=['date', 'time'])

    # Koreksi fisik
    df['G_sol'] = df['G_sol'].clip(lower=0)
    df['u']     = df['u'].clip(lower=0.1)    # min 0.1 untuk hindari div/0
    df['RH']    = df['RH'].clip(0, 100)
    df['rain']  = df['rain'].fillna(0).clip(lower=0)

    # Interpolasi missing values
    for col in ['T_a', 'G_sol', 'RH', 'u']:
        df[col] = df[col].interpolate(method='time')

    df = df.dropna(subset=['T_a', 'G_sol', 'RH', 'u'])

    # Filter periode
    if date_start:
        df = df[df.index >= date_start]
    if date_end:
        df = df[df.index <= date_end]

    print(f"  Periode : {df.index[0]} → {df.index[-1]}")
    print(f"  Records : {len(df)} ({len(df)/60:.1f} jam)")
    print(f"  T_a     : {df['T_a'].min():.1f}–{df['T_a'].max():.1f} °C")
    print(f"  G_sol   : {df['G_sol'].min():.0f}–{df['G_sol'].max():.0f} W/m²")
    print(f"  RH      : {df['RH'].min():.0f}–{df['RH'].max():.0f} %")

    return df


def load_NI_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load data suhu dari NI DAQ (LabVIEW → Excel).

    [C5] Menggunakan XML parser langsung — bukan extract-text yang truncate di 1000 baris.
    [C6] T2A (Tanah bawah CAM) difilter dari 66 anomali ekstrem (< -10°C atau > 80°C).
    [C7] Timestamp dikonversi dari LabVIEW serial (hari sejak 1 Jan 1904).

    Channel mapping (dari nomenclature image yang sudah dikonfirmasi):
    Col  1 → T1Kd  : Lantai CAM
    Col  2 → T1Kb  : Dinding Timur CAM
    Col  3 → T1Ke  : Tanah Atas CAM      → T_g_top (kalibrasi bromelia)
    Col  4 → T1Ka  : Ruangan CAM         → T_in dinamis untuk CAM [C1]
    Col  5 → T1Kc  : Dinding Barat CAM
    Col  6 → T3Kb  : Dinding Barat C3
    Col  7 → T3Kd  : Ruangan C3          → T_in dinamis untuk C3 [C1]
    Col  8 → T3Ke  : Lantai C3
    Col  9 → T3Kc  : Dinding Timur C3
    Col 10 → T3Ka  : Atap Indoor C3      → T_s_in (kalibrasi wedelia)
    Col 11 → T2Ka  : Atap Indoor RR      → referensi conventional roof
    Col 12 → T2Kd  : Dinding Timur RR
    Col 13 → T2Kc  : Dinding Barat RR
    Col 14 → 2Ke   : TBD
    Col 15 → T1A   : TBD
    Col 16 → T2A   : Tanah Bawah CAM     → T_g_bot [C6: ada 66 anomali!]
    Col 17 → T2A2  : TBD
    Col 18 → T1Tb  : Atap Indoor CAM     → T_s_in (kalibrasi bromelia)
    Col 19 → T1Ta  : Atap Outdoor CAM
    Col 20 → T1Ta2 : TBD
    """
    print(f"Loading NI sensor data: {filepath}")

    # [C5] Parse XML langsung — tidak pakai extract-text
    import zipfile
    import io

    # Excel (.xlsx) sebenarnya adalah ZIP berisi XML
    with zipfile.ZipFile(filepath, 'r') as z:
        with z.open('xl/worksheets/sheet1.xml') as f:
            xml_content = f.read()

    tree = ET.fromstring(xml_content)
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

    rows_xml = tree.findall(f'.//{ns}row')
    print(f"  Total rows di XML: {len(rows_xml)} (termasuk 1 header)")

    ni_channel_names = [
        'timestamp_serial',
        'T1Kd', 'T1Kb', 'T1Ke', 'T1Ka', 'T1Kc',
        'T3Kb', 'T3Kd', 'T3Ke', 'T3Kc', 'T3Ka',
        'T2Ka', 'T2Kd', 'T2Kc', '2Ke',  'T1A',
        'T2A',  'T2A2', 'T1Tb', 'T1Ta', 'T1Ta2'
    ]

    data = []
    for row in rows_xml[1:]:   # skip header row
        vals = []
        for cell in row.findall(f'{ns}c'):
            v = cell.find(f'{ns}v')
            vals.append(float(v.text) if v is not None else np.nan)
        if len(vals) == 21:
            data.append(vals)

    df = pd.DataFrame(data, columns=ni_channel_names)

    # [C7] Konversi LabVIEW timestamp (hari sejak 1 Jan 1904) ke datetime
    labview_epoch = pd.Timestamp('1904-01-01')
    df['timestamp'] = labview_epoch + pd.to_timedelta(
        df['timestamp_serial'], unit='D'
    )
    df = df.set_index('timestamp').drop(columns=['timestamp_serial'])

    # [C6] Filter anomali T2A — 66 data dengan nilai ekstrem
    # Nilai < -10°C atau > 80°C adalah tidak fisik → sensor disconnect
    print(f"\n  [C6] Filtering anomali T2A...")
    mask_anomaly = (df['T2A'] < -10) | (df['T2A'] > 80)
    n_anomaly = mask_anomaly.sum()
    print(f"  Anomali T2A: {n_anomaly} data")

    if n_anomaly > 0:
        # Cek max cluster
        clusters = []
        count = 0
        for val in mask_anomaly:
            if val:
                count += 1
            else:
                if count > 0:
                    clusters.append(count)
                    count = 0
        if count > 0:
            clusters.append(count)
        max_cluster = max(clusters) if clusters else 0
        print(f"  Max berturutan: {max_cluster} data")

        # Replace dengan NaN lalu interpolasi
        df.loc[mask_anomaly, 'T2A'] = np.nan
        df['T2A'] = df['T2A'].interpolate(
            method='linear',
            limit=30,           # max 30 menit gap
            limit_direction='both'
        )
        print(f"  Status: replaced dengan NaN → interpolasi linear ✓")

    # Buat kolom deskriptif untuk simulasi
    df['T_g_top_CAM'] = df['T1Ke']   # Tanah atas CAM → kalibrasi bromelia (dual)
    df['T_g_bot_CAM'] = df['T2A']    # Tanah bawah CAM → validasi profil
    df['T_s_in_CAM']  = df['T1Tb']   # Atap indoor CAM → kalibrasi bromelia (dual)
    df['T_s_ext_CAM'] = df['T1Ta']   # Atap outdoor CAM
    df['T_s_in_C3']   = df['T3Ka']   # Atap indoor C3 → kalibrasi wedelia
    # [C1] T_in dinamis — BUKAN 25°C konstan
    df['T_in_CAM']    = df['T1Ka']   # Ruangan CAM → dipakai sebagai T_in dinamis
    df['T_in_C3']     = df['T3Kd']   # Ruangan C3 → dipakai sebagai T_in dinamis
    df['T_r_in_RR']   = df['T2Ka']   # Referensi conventional roof

    interval = (df.index[1] - df.index[0]).total_seconds()

    print(f"\n  Periode : {df.index[0]} → {df.index[-1]}")
    print(f"  Records : {len(df)} ({len(df)/60:.1f} jam)")
    print(f"  Interval: {interval:.0f} detik")
    print(f"\n  Key variables (setelah cleaning):")
    for col, label in [
        ('T_g_top_CAM', 'T_g_top CAM (T1Ke)'),
        ('T_g_bot_CAM', 'T_g_bot CAM (T2A) '),
        ('T_s_in_CAM',  'T_s_in CAM (T1Tb) '),
        ('T_s_in_C3',   'T_s_in C3  (T3Ka) '),
        ('T_in_CAM',    '[C1] T_in CAM (T1Ka)'),
        ('T_in_C3',     '[C1] T_in C3  (T3Kd)'),
    ]:
        s = df[col].dropna()
        print(f"    {label}: {s.min():.1f}–{s.max():.1f}°C, mean={s.mean():.1f}")

    return df


def load_licor_data(filepath: str) -> pd.DataFrame:
    """
    Load data gas exchange dari Licor 6800.
    Header kompleks: 14 baris sebelum data, baris 15 adalah satuan (skip).
    """
    print(f"Loading Licor data: {filepath}")

    df = pd.read_excel(filepath, header=14, skiprows=[15])

    for col in ['A', 'E', 'gsw', 'TleafEB', 'VPDleaf', 'Qin']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Records: {len(df)}")
    if 'gsw' in df.columns:
        gsw_valid = df[(df['gsw'] > 0.001) & (df.get('A', pd.Series([1]*len(df))) > 0)]['gsw']
        print(f"  Valid gsw (>0.001, A>0): {len(gsw_valid)}")
        if len(gsw_valid) > 0:
            r = 1 / (gsw_valid * 0.0224)
            print(f"  r_stoma: {r.min():.1f}–{r.max():.1f} s/m")

    return df


def extract_r_stoma_min(licor_df: pd.DataFrame,
                         plant_type: str = "C3",
                         percentile: float = 5.0) -> float:
    """
    Ekstrak r_stoma_min dari data Licor.
    r_stoma (s/m) = 1 / (gsw_mol × 0.0224)
    Pakai percentile ke-5 (bukan minimum) untuk stabilitas.
    [C4] Untuk C3: sudah dikonfirmasi = 167.2 s/m dari data April 2026
    """
    mask = (licor_df['gsw'] > 0.001)
    if 'A' in licor_df.columns:
        mask = mask & (licor_df['A'] > 0)

    gsw_valid = licor_df[mask]['gsw']

    if len(gsw_valid) == 0:
        print(f"WARNING: Tidak ada gsw valid! Pakai nilai default.")
        return 167.2 if plant_type == "C3" else 400.0

    r_stoma = 1 / (gsw_valid * 0.0224)
    r_stoma_min = float(np.percentile(r_stoma, percentile))

    print(f"\nr_stoma_min {plant_type}:")
    print(f"  Min absolut     : {r_stoma.min():.1f} s/m")
    print(f"  Percentile {percentile:.0f}%  : {r_stoma_min:.1f} s/m  ← dipakai")
    print(f"  Mean            : {r_stoma.mean():.1f} s/m")

    return r_stoma_min


def load_soil_moisture_data(filepath: str) -> pd.DataFrame:
    """
    Load data soil moisture sensor (kedalaman 2cm dan 7cm).
    TODO: Implementasi setelah data diekstrak dari logger.

    Format yang perlu diketahui:
    - Apakah CSV atau Excel?
    - Nama kolom untuk 2cm dan 7cm?
    - Satuan (m³/m³ atau % volumetric?)
    - Interval waktu?
    - CAM dan C3 pada hari berbeda
    """
    raise NotImplementedError(
        "Soil moisture data belum diekstrak.\n"
        "Konfirmasi format file dari data logger dulu.\n"
        "Sementara pakai theta_initial = theta_sat × 0.80"
    )


def synchronize_datasets(weather_df: pd.DataFrame,
                          ni_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    [C7] Sinkronisasi timestamp Davis dan NI sensor.

    Masalah: Davis interval 1 menit tepat (xx:00),
    NI dari LabVIEW mungkin ada detik tidak tepat (xx:58:16).
    Solusi: resample keduanya ke grid 1 menit yang sama.
    """
    print("\nSinkronisasi dataset...")

    # Temukan periode overlap
    start = max(weather_df.index[0], ni_df.index[0])
    end   = min(weather_df.index[-1], ni_df.index[-1])

    print(f"  Overlap: {start} → {end}")
    duration_h = (end - start).total_seconds() / 3600
    print(f"  Durasi : {duration_h:.1f} jam = {duration_h/24:.1f} hari")

    # Filter ke periode overlap
    w = weather_df[(weather_df.index >= start) & (weather_df.index <= end)].copy()
    n = ni_df[(ni_df.index >= start) & (ni_df.index <= end)].copy()

    # Resample NI ke grid 1 menit aligned dengan Davis
    # (menangani slight timestamp drift dari LabVIEW)
    n_resampled = n.resample('1min').mean()

    print(f"  Weather records : {len(w)}")
    print(f"  NI records      : {len(n)} → {len(n_resampled)} (setelah resample)")

    return w, n_resampled


def resample_for_simulation(df: pd.DataFrame,
                             dt: float = 1.0) -> pd.DataFrame:
    """Resample data ke interval dt detik untuk simulasi."""
    rule = f'{int(dt)}s'
    df_resampled = df.resample(rule).interpolate(method='time')
    print(f"  Resample: {len(df)} → {len(df_resampled)} records ({dt:.0f}s interval)")
    return df_resampled


# ==============================================================================
# SECTION 3: HELPER TERMODINAMIKA
# ==============================================================================

def saturation_pressure(T_K: float) -> float:
    """Tekanan uap saturasi [Pa] via persamaan Magnus."""
    T_C = T_K - 273.15
    return 610.78 * np.exp(17.269 * T_C / (T_C + 237.29))


def sky_temperature(T_a_K: float, RH: float) -> float:
    """
    Suhu langit efektif [K].
    Jakarta (RH tinggi ~85%) → T_sky ≈ T_a → pendinginan radiatif kecil.
    """
    epsilon_sky = 0.711 + 0.56*(RH/100) + 0.73*(RH/100)**2
    epsilon_sky = min(epsilon_sky, 1.0)
    return epsilon_sky**0.25 * T_a_K


def ambient_vapor_pressure(T_a_K: float, RH: float) -> float:
    """Tekanan uap aktual udara [Pa]. VPD = P_sat - P_a."""
    return (RH/100) * saturation_pressure(T_a_K)


def psychrometric_constant() -> float:
    """Konstanta psikrometrik γ ≈ 66.8 Pa/K."""
    return 1005 * 101325 / (0.622 * 2.45e6)


# ==============================================================================
# SECTION 4: BLOK 1 — FOLIAGE MODEL
# ==============================================================================

def compute_aerodynamic_resistance(u: float,
                                   plant: PlantParameters) -> float:
    """
    Resistansi aerodinamik r_a [s/m]. Persamaan A.2, A.3, A.4.
    Berbeda antara CAM dan C3 karena H_f berbeda.
    """
    k = 0.41
    d0 = 0.701 * plant.H_f**0.975    # A.3
    Z0 = 0.131 * plant.H_f**0.997    # A.4
    z_ref = plant.H_f + 2.0
    ratio = max((z_ref - d0) / Z0, 1.01)
    r_a = 1 / (k**2 * max(u, 0.1)) * (np.log(ratio))**2    # A.2
    return max(r_a, 5.0)


def compute_stomatal_resistance(G_sol: float,
                                T_f_K: float,
                                P_f_sat: float,
                                P_a: float,
                                theta: float,
                                plant: PlantParameters,
                                substrate: SubstrateParameters) -> float:
    """
    Stomatal resistance aktual r_stoma [s/m]. Persamaan A.6.
    Perbedaan CAM vs C3: modifier pada G_sol > 50 W/m².
    [C8] CAM factor = 12.0 masih placeholder → dari Licor CAM nanti.
    """
    # f1–f4: fungsi modifikasi A.7–A.10
    f1 = 1 + np.exp(-0.034 * (G_sol - 3.5))
    T_C = T_f_K - 273.15
    f2 = (np.exp(0.3*T_C) + 258) / (np.exp(0.3*T_C) + 27)
    VPD = max(P_f_sat - P_a, 0)
    f3 = 4e-3 + np.exp(-0.73 * 0.622e3/101325 * VPD)
    theta_safe = max(theta, substrate.theta_min + 0.001)
    f4 = substrate.theta_sat / theta_safe

    r_stoma = (plant.r_stoma_min / plant.LAI) * f1 * f2 * f3 * f4

    # Modifikasi CAM vs C3
    if plant.plant_type == "CAM":
        if G_sol > 50:    # siang hari
            # [C8] 12.0 = PLACEHOLDER — akan diupdate dari Licor CAM
            r_stoma *= 12.0
        else:             # malam hari
            r_stoma *= 0.5

    return max(r_stoma, 5.0)


def solve_foliage_temperature(T_f_prev: float,
                              T_a_K: float,
                              T_g_surface: float,
                              G_sol: float,
                              RH: float,
                              u: float,
                              theta_avg: float,
                              plant: PlantParameters,
                              substrate: SubstrateParameters,
                              dt: float) -> float:
    """
    Solve T_f baru dengan implicit scheme. Persamaan (1)(2)(3).
    """
    sigma   = 5.67e-8
    rho_air = 1.2
    cp_air  = 1005
    gamma   = psychrometric_constant()

    T_sky   = sky_temperature(T_a_K, RH)
    P_a     = ambient_vapor_pressure(T_a_K, RH)
    P_f_sat = saturation_pressure(T_f_prev)

    epsilon_fg = 1 / (1/plant.epsilon_f + 1/substrate.epsilon_g - 1)    # Eq.3

    R_f_net = (plant.alpha_f * G_sol
               - plant.epsilon_f * sigma * (T_f_prev**4 - T_sky**4)
               - epsilon_fg * sigma * (T_f_prev**4 - T_g_surface**4))   # Eq.2

    r_a      = compute_aerodynamic_resistance(u, plant)
    r_stoma  = compute_stomatal_resistance(G_sol, T_f_prev, P_f_sat, P_a,
                                           theta_avg, plant, substrate)
    h_conv_f = plant.LAI * (rho_air * cp_air) / r_a                     # A.1
    h_eva_f  = plant.LAI / gamma * (rho_air * cp_air) / (r_a + r_stoma) # A.5

    rho_cp_f = rho_air * cp_air
    d_eff    = 0.001

    dPdT = saturation_pressure(T_f_prev+0.5) - saturation_pressure(T_f_prev-0.5)

    A = rho_cp_f*d_eff/dt + h_conv_f + h_eva_f*dPdT
    B = (rho_cp_f*d_eff/dt * T_f_prev + R_f_net
         + h_conv_f*T_a_K + h_eva_f*(P_f_sat - dPdT*T_f_prev - P_a))

    return B / A


# ==============================================================================
# SECTION 5: BLOK 2 — SUBSTRATE MODEL
# ==============================================================================

def compute_substrate_properties(theta: np.ndarray,
                                  substrate: SubstrateParameters
                                  ) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """Properti substrat sebagai fungsi VWC. Persamaan B.8–B.11."""
    b = substrate.b

    rho_cp_g = substrate.cp_g * (0.2 + theta) * substrate.rho_g         # B.8
    k_theta  = substrate.k_theta_sat * (theta/substrate.theta_sat)**(2*b+3)  # B.10
    lambda_g = substrate.lambda_dry + theta * k_theta                    # B.9
    theta_s  = np.maximum(theta, substrate.theta_min + 1e-6)
    D_theta  = (-b * substrate.k_theta_sat * substrate.psi_sat
                / theta_s * (theta/substrate.theta_sat)**(b+3))          # B.11

    return rho_cp_g, lambda_g, k_theta, D_theta


def tdma_solver(a: np.ndarray, b: np.ndarray,
                c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Tridiagonal Matrix Algorithm (Thomas Algorithm).
    O(n) complexity — lebih efisien dari O(n³) matrix inversion.
    """
    n   = len(d)
    c_  = np.zeros(n)
    d_  = np.zeros(n)
    x   = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i-1]
        denom = denom if abs(denom) > 1e-30 else 1e-30
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom

    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x


def solve_substrate_heat(T_g: np.ndarray,
                          T_f: float,
                          T_slab_top: float,
                          G_sol: float,
                          T_a_K: float,
                          RH: float,
                          u: float,
                          theta: np.ndarray,
                          plant: PlantParameters,
                          substrate: SubstrateParameters,
                          H_g: float,
                          dt: float) -> np.ndarray:
    """Solve T_g[z] via FVM + TDMA. Persamaan (4)(5)(6)(7)."""
    sigma   = 5.67e-8
    gamma   = psychrometric_constant()
    rho_air = 1.2
    cp_air  = 1005

    Nz = len(T_g)
    dz = H_g / (Nz - 1)

    rho_cp_g, lambda_g, k_theta, _ = compute_substrate_properties(theta, substrate)

    # Boundary atas
    T_sky   = sky_temperature(T_a_K, RH)
    P_a     = ambient_vapor_pressure(T_a_K, RH)
    P_g_sat = saturation_pressure(T_g[0])
    epsilon_fg = 1 / (1/plant.epsilon_f + 1/substrate.epsilon_g - 1)

    R_g_net = ((1 - substrate.rho_g_rad) * plant.tau_f * G_sol          # Eq.6
               + epsilon_fg * sigma * (T_f**4 - T_g[0]**4))

    d0  = 0.701 * plant.H_f**0.975
    Z_M = 0.131 * plant.H_f**0.997
    Z_u = plant.H_f + 2.0
    k_vk = 0.41

    log_r = np.log(max((plant.H_f - d0)/Z_M, 1.01)) / np.log(max((Z_u - d0)/Z_M, 1.01))
    u_f = u * max(log_r, 0.01)
    a_drag = (0.28 * plant.LAI * plant.H_f * plant.d_f)**0.5
    u_g = max(u_f * np.exp(-a_drag * (1 - 0.05/max(plant.H_f, 0.01))), 0.01)

    r_c   = 1 / (0.004 + 0.012 * u_g)                                   # B.3
    r_vap = substrate.lambda_dry * 50
    r_a   = compute_aerodynamic_resistance(u, plant)

    h_conv_g = plant.LAI * (rho_air * cp_air) / (r_a + r_c)             # B.1
    h_eva_g  = plant.LAI / gamma * (rho_air * cp_air) / (r_a + r_vap)   # B.2

    q_top = R_g_net - h_conv_g*(T_g[0]-T_a_K) - h_eva_g*(P_g_sat-P_a)  # Eq.5

    # FVM tridiagonal
    aw = np.zeros(Nz); ap = np.zeros(Nz)
    ae = np.zeros(Nz); bv = np.zeros(Nz)

    for i in range(1, Nz-1):
        lw = 0.5*(lambda_g[i-1]+lambda_g[i])
        le = 0.5*(lambda_g[i]+lambda_g[i+1])
        aw[i] = lw/dz**2
        ae[i] = le/dz**2
        ap[i] = rho_cp_g[i]/dt + aw[i] + ae[i]
        bv[i] = rho_cp_g[i]/dt * T_g[i]

    ae[0]  = lambda_g[0]/dz**2
    ap[0]  = rho_cp_g[0]/dt + ae[0]
    bv[0]  = rho_cp_g[0]/dt * T_g[0] + q_top/dz

    aw[-1] = lambda_g[-1]/dz**2
    ap[-1] = rho_cp_g[-1]/dt + aw[-1]
    bv[-1] = rho_cp_g[-1]/dt*T_g[-1] + lambda_g[-1]/dz*T_slab_top/dz

    return tdma_solver(aw, ap, ae, bv)


def solve_substrate_moisture(theta: np.ndarray,
                              T_f: float,
                              T_g_surface: float,
                              T_a_K: float,
                              RH: float,
                              u: float,
                              j_irrigation: float,
                              plant: PlantParameters,
                              substrate: SubstrateParameters,
                              H_g: float,
                              dt: float) -> Tuple[np.ndarray, float]:
    """Solve θ[z] via Richards equation + TDMA. Persamaan (8)(9)(10)(11)."""
    gamma   = psychrometric_constant()
    rho_air = 1.2
    cp_air  = 1005

    Nz = len(theta)
    dz = H_g / (Nz - 1)

    _, _, k_theta, D_theta = compute_substrate_properties(theta, substrate)

    P_a     = ambient_vapor_pressure(T_a_K, RH)
    P_f_sat = saturation_pressure(T_f)
    P_g_sat = saturation_pressure(T_g_surface)
    r_a     = compute_aerodynamic_resistance(u, plant)
    r_stoma = compute_stomatal_resistance(0, T_f, P_f_sat, P_a,
                                          np.mean(theta), plant, substrate)
    r_vap   = substrate.lambda_dry * 50

    h_eva_f = plant.LAI / gamma * (rho_air*cp_air) / (r_a + r_stoma)
    h_eva_g = plant.LAI / gamma * (rho_air*cp_air) / (r_a + r_vap)

    j_eva_f = h_eva_f * (P_f_sat - P_a) / substrate.l_fg               # Eq.10
    j_eva_g = h_eva_g * (P_g_sat - P_a) / substrate.l_fg
    j_eva   = j_eva_f + j_eva_g

    aw = np.zeros(Nz); ap = np.zeros(Nz)
    ae = np.zeros(Nz); bv = np.zeros(Nz)

    for i in range(1, Nz-1):
        Dw = 0.5*(D_theta[i-1]+D_theta[i])
        De = 0.5*(D_theta[i]+D_theta[i+1])
        kw = 0.5*(k_theta[i-1]+k_theta[i])
        ke = 0.5*(k_theta[i]+k_theta[i+1])
        aw[i] = Dw/dz**2
        ae[i] = De/dz**2
        ap[i] = 1/dt + aw[i] + ae[i]
        bv[i] = theta[i]/dt + (ke-kw)/dz

    j_net  = j_irrigation - j_eva                                        # Eq.9
    ae[0]  = D_theta[0]/dz**2
    ap[0]  = 1/dt + ae[0]
    bv[0]  = theta[0]/dt + j_net/dz + k_theta[0]/dz

    if theta[-1] >= substrate.theta_sat*0.99:                            # Eq.11
        aw[-1] = D_theta[-1]/dz**2
        ap[-1] = 1/dt + aw[-1]
        bv[-1] = theta[-1]/dt - k_theta[-1]/dz
    else:
        aw[-1] = D_theta[-1]/dz**2
        ap[-1] = 1/dt + aw[-1]
        bv[-1] = theta[-1]/dt

    theta_new = tdma_solver(aw, ap, ae, bv)
    theta_new = np.clip(theta_new, substrate.theta_min, substrate.theta_sat)

    return theta_new, j_eva


# ==============================================================================
# SECTION 6: BLOK 3 — SLAB MODEL
# ==============================================================================

def solve_slab_heat(T_s: np.ndarray,
                    T_g_bottom: float,
                    lambda_g_bottom: float,
                    slab: SlabParameters,
                    geom: GeometryParameters,
                    dt: float,
                    T_in_K: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Solve T_s[z] dan q_s_in via FVM + TDMA. Persamaan (12)(13)(14).

    [C1] T_in_K sekarang PARAMETER DINAMIS:
    - Kalau T_in_K diberikan → pakai nilai dari NI sensor saat itu
    - Kalau None → fallback ke geom.T_in_default (rata-rata 29.5°C, bukan 25°C!)
    """
    Nz = len(T_s)
    dz = slab.H_slab / (Nz - 1)
    rho_cp_s = slab.rho_s * slab.cp_s

    # [C1] Gunakan T_in dinamis atau fallback ke default
    T_in = T_in_K if T_in_K is not None else geom.T_in_default

    aw = np.zeros(Nz); ap = np.zeros(Nz)
    ae = np.zeros(Nz); bv = np.zeros(Nz)

    for i in range(1, Nz-1):
        aw[i] = slab.lambda_s/dz**2
        ae[i] = slab.lambda_s/dz**2
        ap[i] = rho_cp_s/dt + aw[i] + ae[i]
        bv[i] = rho_cp_s/dt * T_s[i]

    ae[0]  = slab.lambda_s/dz**2
    ap[0]  = rho_cp_s/dt + ae[0]
    bv[0]  = rho_cp_s/dt*T_s[0] + lambda_g_bottom*T_g_bottom/dz**2     # Eq.13

    h_in   = geom.h_in
    aw[-1] = slab.lambda_s/dz**2
    ap[-1] = rho_cp_s/dt + aw[-1] + h_in/dz
    bv[-1] = rho_cp_s/dt*T_s[-1] + h_in*T_in/dz                        # Eq.14

    T_s_new = tdma_solver(aw, ap, ae, bv)

    q_s_in = h_in * (T_s_new[-1] - T_in)   # Eq.14: positif=masuk, negatif=keluar

    return T_s_new, q_s_in


# ==============================================================================
# SECTION 7: MAIN SIMULATION LOOP
# ==============================================================================

def run_simulation(weather_df: pd.DataFrame,
                   plant: PlantParameters,
                   substrate: SubstrateParameters,
                   slab: SlabParameters,
                   geom: GeometryParameters,
                   num: NumericalParameters,
                   theta_initial: float = None,
                   j_irrigation: float = 0.0,
                   T_in_series: pd.Series = None) -> dict:  # [C1] parameter baru
    """
    Main simulation loop mengikuti Figure 2 flowchart dari paper.

    [C1] Parameter baru: T_in_series
    - Kalau diberikan: T_in berubah setiap timestep dari NI data
    - Kalau None: pakai geom.T_in_default (29.5°C, bukan 25°C)

    Urutan sequential per timestep:
    1. Baca kondisi cuaca
    2. Solve T_f  → Blok 1
    3. Solve T_g  → Blok 2a
    4. Solve θ    → Blok 2b
    5. Solve T_s + q_s_in → Blok 3
    6. Simpan setiap 60 detik
    """
    print(f"\n{'='*60}")
    print(f"Simulasi: {plant.name} ({plant.plant_type})")
    if T_in_series is not None:
        print(f"[C1] T_in: DINAMIS dari NI data "
              f"({T_in_series.min():.1f}–{T_in_series.max():.1f}°C)")
    else:
        print(f"[C1] T_in: default {geom.T_in_default-273.15:.1f}°C")
    print(f"{'='*60}")

    Nz_g = num.Nz_substrate
    Nz_s = num.Nz_slab
    dt   = num.dt

    # Resample weather ke 1 detik
    print("Resample weather data ke 1 detik...")
    weather_1s = weather_df.resample('1s').interpolate(method='time')
    N_steps = len(weather_1s)
    print(f"Total timesteps: {N_steps:,} ({N_steps/3600:.1f} jam)")

    # [C1] Resample T_in series ke 1 detik jika ada
    T_in_1s = None
    if T_in_series is not None:
        T_in_1s = T_in_series.resample('1s').interpolate(method='time')

    # Kondisi awal
    T_a_init = weather_1s['T_a'].iloc[0] + 273.15

    if theta_initial is None:
        theta_initial = substrate.theta_sat * 0.80
        print(f"theta_initial tidak ada → pakai {theta_initial:.2f} (80% sat)")

    T_g   = np.full(Nz_g, T_a_init)
    T_s   = np.full(Nz_s, T_a_init)
    T_f   = T_a_init
    theta = np.full(Nz_g, theta_initial)

    # Storage hasil
    results = {
        'time'      : [], 'T_f'    : [], 'T_g_top': [],
        'T_g_mid'   : [], 'T_g_bot': [], 'T_s_in' : [],
        'theta_top' : [], 'theta_mid': [], 'theta_bot': [],
        'q_s_in'    : [], 'j_eva'  : [],
        'T_a'       : [], 'G_sol'  : [],
        'T_in_used' : [],   # [C1] simpan T_in yang dipakai setiap step
    }

    print("Mulai simulasi...\n")

    for step in range(N_steps):

        # Baca kondisi cuaca
        row   = weather_1s.iloc[step]
        T_a_K = row['T_a'] + 273.15
        G_sol = max(row['G_sol'], 0.0)
        RH    = float(np.clip(row['RH'], 1, 99))
        u     = max(row['u'], 0.1)
        rain  = max(row.get('rain', 0), 0)
        j_rain = rain / 60.0
        j_pr   = j_irrigation + j_rain

        # [C1] Ambil T_in dinamis dari NI data
        if T_in_1s is not None:
            try:
                T_in_current = T_in_1s.iloc[step] + 273.15
            except (IndexError, KeyError):
                T_in_current = geom.T_in_default
        else:
            T_in_current = geom.T_in_default

        theta_avg = float(np.mean(theta))

        # BLOK 1: Foliage
        T_f = solve_foliage_temperature(
            T_f_prev=T_f, T_a_K=T_a_K, T_g_surface=T_g[0],
            G_sol=G_sol, RH=RH, u=u, theta_avg=theta_avg,
            plant=plant, substrate=substrate, dt=dt
        )

        # BLOK 2a: Substrat heat
        T_g = solve_substrate_heat(
            T_g=T_g, T_f=T_f, T_slab_top=T_s[0],
            G_sol=G_sol, T_a_K=T_a_K, RH=RH, u=u, theta=theta,
            plant=plant, substrate=substrate, H_g=geom.H_g, dt=dt
        )

        # BLOK 2b: Substrat moisture
        theta, j_eva = solve_substrate_moisture(
            theta=theta, T_f=T_f, T_g_surface=T_g[0],
            T_a_K=T_a_K, RH=RH, u=u, j_irrigation=j_pr,
            plant=plant, substrate=substrate, H_g=geom.H_g, dt=dt
        )

        # BLOK 3: Slab + q_s_in
        _, lambda_g, _, _ = compute_substrate_properties(theta, substrate)
        T_s, q_s_in = solve_slab_heat(
            T_s=T_s, T_g_bottom=T_g[-1],
            lambda_g_bottom=lambda_g[-1],
            slab=slab, geom=geom, dt=dt,
            T_in_K=T_in_current    # [C1] T_in dinamis
        )

        # Simpan setiap 60 detik
        if step % 60 == 0:
            results['time'].append(step * dt)
            results['T_f'].append(T_f - 273.15)
            results['T_g_top'].append(T_g[0] - 273.15)
            results['T_g_mid'].append(T_g[Nz_g//2] - 273.15)
            results['T_g_bot'].append(T_g[-1] - 273.15)
            results['T_s_in'].append(T_s[-1] - 273.15)
            results['theta_top'].append(float(theta[0]))
            results['theta_mid'].append(float(theta[Nz_g//2]))
            results['theta_bot'].append(float(theta[-1]))
            results['q_s_in'].append(float(q_s_in))
            results['j_eva'].append(float(j_eva))
            results['T_a'].append(T_a_K - 273.15)
            results['G_sol'].append(G_sol)
            results['T_in_used'].append(T_in_current - 273.15)    # [C1]

        # Progress setiap 6 jam
        if step % int(6*3600/dt) == 0:
            print(f"  t={step*dt/3600:5.1f}h | "
                  f"T_f={T_f-273.15:5.1f}°C | "
                  f"T_g={T_g[0]-273.15:5.1f}°C | "
                  f"T_s_in={T_s[-1]-273.15:5.1f}°C | "
                  f"θ={theta[0]:.3f} | "
                  f"T_in={T_in_current-273.15:.1f}°C | "   # [C1] tampilkan T_in aktual
                  f"q={q_s_in:6.1f} W/m²")

    # Q_gain — Persamaan (18)
    Q_gain = float(np.trapz(results['q_s_in'], dx=60.0))
    results['Q_gain'] = Q_gain
    print(f"\nQ_gain = {Q_gain:.2f} J/m²  "
          f"({'heat sink ✅' if Q_gain < 0 else 'heat gain ⚠'})")

    return results


# ==============================================================================
# SECTION 8: KALIBRASI LAI
# ==============================================================================

def calibrate_LAI(weather_df: pd.DataFrame,
                   NI_data: pd.DataFrame,
                   plant: PlantParameters,
                   substrate: SubstrateParameters,
                   slab: SlabParameters,
                   geom: GeometryParameters,
                   num: NumericalParameters,
                   LAI_bounds: Tuple[float, float] = (0.1, 5.0),
                   theta_initial: float = None) -> Tuple[float, dict]:
    """
    Kalibrasi LAI dengan meminimalkan error antara simulasi dan NI data.

    Bromelia (CAM): dual-target → T_g_top (T1Ke) + T_s_in (T1Tb)
    Wedelia  (C3) : single-target → T_s_in (T3Ka) saja

    [C1] T_in dinamis dari NI data dipakai otomatis.
    """
    from scipy.optimize import minimize_scalar

    # Siapkan T_in series dari NI data [C1]
    if plant.plant_type == "CAM":
        T_in_series = NI_data['T_in_CAM']
        target_vars = ['T_g_top', 'T_s_in']
        obs_map     = {'T_g_top': 'T_g_top_CAM', 'T_s_in': 'T_s_in_CAM'}
        print("[C1] Menggunakan T_in_CAM (T1Ka) dari NI data")
        print("Kalibrasi dual-target: T_g_top + T_s_in")
    else:
        T_in_series = NI_data['T_in_C3']
        target_vars = ['T_s_in']
        obs_map     = {'T_s_in': 'T_s_in_C3'}
        print("[C1] Menggunakan T_in_C3 (T3Kd) dari NI data")
        print("Kalibrasi single-target: T_s_in saja")

    # Sync T_in dengan weather
    T_in_synced, _ = synchronize_datasets(
        pd.DataFrame({'T_in': T_in_series}),
        weather_df
    )
    T_in_series_synced = T_in_synced['T_in']

    iteration = [0]

    def objective(LAI):
        iteration[0] += 1
        plant.LAI = LAI

        results = run_simulation(
            weather_df=weather_df, plant=plant,
            substrate=substrate, slab=slab,
            geom=geom, num=num,
            theta_initial=theta_initial,
            T_in_series=T_in_series_synced    # [C1]
        )

        total_error = 0.0
        for var in target_vars:
            sim_vals = np.array(results[var])
            obs_col  = obs_map[var]
            obs_vals = NI_data[obs_col].resample('1min').mean().values
            # Trim ke panjang yang sama
            n = min(len(sim_vals), len(obs_vals))
            mse = np.mean((sim_vals[:n] - obs_vals[:n])**2)
            total_error += mse

        print(f"  Iterasi {iteration[0]:2d} | LAI={LAI:.3f} | Error={total_error:.4f}")
        return total_error

    result = minimize_scalar(objective, bounds=LAI_bounds,
                              method='bounded', options={'xatol': 0.01})

    LAI_optimal = result.x
    plant.LAI   = LAI_optimal
    print(f"\nLAI optimal ({plant.name}) = {LAI_optimal:.3f}")

    cal_results = run_simulation(
        weather_df=weather_df, plant=plant,
        substrate=substrate, slab=slab,
        geom=geom, num=num,
        theta_initial=theta_initial,
        T_in_series=T_in_series_synced
    )

    return LAI_optimal, cal_results


# ==============================================================================
# SECTION 9: VISUALISASI
# ==============================================================================

def plot_results(results_cam: dict,
                  results_c3: dict,
                  weather_df: pd.DataFrame,
                  NI_data: pd.DataFrame = None,
                  save_path: str = None):
    """
    6-panel comparison plot: CAM vs C3.
    [C1] Panel tambahan untuk T_in aktual vs 25°C asumsi paper.
    """
    t_cam = np.array(results_cam['time']) / 3600
    t_c3  = np.array(results_c3['time'])  / 3600

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Green Roof Thermal Simulation — CAM (Bromelia) vs C3 (Wedelia)\n'
                 'Universitas Indonesia | Koreksi: T_in dinamis, tau_f diperbarui',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # (a) Suhu CAM
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_cam, results_cam['T_a'],     'k--', lw=1.2, label='T_a', alpha=0.7)
    ax.plot(t_cam, results_cam['T_f'],     'b-',  lw=1.5, label='T_f (foliage)')
    ax.plot(t_cam, results_cam['T_g_top'], 'g-o', lw=1.2, label='T_g_top',
            markevery=60, ms=3)
    ax.plot(t_cam, results_cam['T_s_in'],  'r-^', lw=1.2, label='T_s_in',
            markevery=60, ms=3)
    if NI_data is not None:
        t_ni = np.arange(len(NI_data)) / 60
        ax.plot(t_ni, NI_data['T_g_top_CAM'].values, 'g.', ms=2, alpha=0.4,
                label='T_g_top (NI)')
        ax.plot(t_ni, NI_data['T_s_in_CAM'].values,  'r.', ms=2, alpha=0.4,
                label='T_s_in (NI)')
    ax.set_title('(a) Suhu — Bromelia (CAM)')
    ax.set_ylabel('Suhu (°C)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (b) Suhu C3
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_c3,  results_c3['T_a'],     'k--', lw=1.2, label='T_a', alpha=0.7)
    ax.plot(t_c3,  results_c3['T_f'],     'b-',  lw=1.5, label='T_f (foliage)')
    ax.plot(t_c3,  results_c3['T_g_top'], 'g-o', lw=1.2, label='T_g_top',
            markevery=60, ms=3)
    ax.plot(t_c3,  results_c3['T_s_in'],  'r-^', lw=1.2, label='T_s_in',
            markevery=60, ms=3)
    if NI_data is not None:
        ax.plot(t_ni, NI_data['T_s_in_C3'].values, 'r.', ms=2, alpha=0.4,
                label='T_s_in (NI)')
    ax.set_title('(b) Suhu — Wedelia (C3)')
    ax.set_ylabel('Suhu (°C)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (c) Heat flux
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_cam, results_cam['q_s_in'], 'b-', lw=1.5, label='CAM (Bromelia)')
    ax.plot(t_c3,  results_c3['q_s_in'],  'r-', lw=1.5, label='C3 (Wedelia)')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.fill_between(t_cam, results_cam['q_s_in'], 0,
                    where=np.array(results_cam['q_s_in'])>0,
                    alpha=0.15, color='blue')
    Q_cam = results_cam['Q_gain']
    Q_c3  = results_c3['Q_gain']
    ax.text(0.02, 0.97, f"Q_gain CAM = {Q_cam:.0f} J/m²\nQ_gain C3  = {Q_c3:.0f} J/m²",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_title('(c) Heat Flux ke Interior')
    ax.set_ylabel('q_s_in (W/m²)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d) VWC
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_cam, results_cam['theta_top'], 'b-',  lw=1.5, label='CAM θ_top')
    ax.plot(t_cam, results_cam['theta_bot'], 'b--', lw=1.0, label='CAM θ_bot')
    ax.plot(t_c3,  results_c3['theta_top'],  'r-',  lw=1.5, label='C3 θ_top')
    ax.plot(t_c3,  results_c3['theta_bot'],  'r--', lw=1.0, label='C3 θ_bot')
    ax.set_title('(d) Volumetric Water Content')
    ax.set_ylabel('VWC (m³/m³)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (e) Evapotranspirasi
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_cam, np.array(results_cam['j_eva'])*3600, 'b-', lw=1.5,
            label='CAM')
    ax.plot(t_c3,  np.array(results_c3['j_eva'])*3600,  'r-', lw=1.5,
            label='C3')
    ax.set_title('(e) Evapotranspirasi')
    ax.set_ylabel('ET (kg/m²jam)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (f) [C1] T_in aktual vs asumsi 25°C
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_cam, results_cam['T_in_used'], 'b-', lw=1.5, label='T_in CAM (T1Ka)')
    ax.plot(t_c3,  results_c3['T_in_used'],  'r-', lw=1.5, label='T_in C3 (T3Kd)')
    ax.axhline(25.0, color='gray', ls='--', alpha=0.7, label='Asumsi paper (25°C)')
    ax.set_title('[C1] T_in Aktual vs Asumsi Paper')
    ax.set_ylabel('T_in (°C)'); ax.set_xlabel('Waktu (jam)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97, 'Bukan 25°C!', transform=ax.transAxes,
            va='top', color='red', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot disimpan: {save_path}")

    plt.show()


# ==============================================================================
# SECTION 10: MAIN — CARA PEMAKAIAN
# ==============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GREEN ROOF SIMULATION — CORRECTED VERSION")
    print("=" * 60)

    print("\n[STATUS KOREKSI]")
    print("[C1] T_in: DINAMIS dari NI data (bukan 25°C)")
    print("[C2] tau_f C3: 0.20 (bukan 0.10)")
    print("[C3] alpha_f: dihitung dari rho_f terukur")
    print("[C4] r_stoma_min C3: 167.2 s/m dari Licor NYATA")
    print("[C5] NI loader: XML parser (bukan extract-text)")
    print("[C6] T2A anomali: 66 data diinterpolasi")
    print("[C7] Timestamp: sinkronisasi Davis ↔ NI")
    print("[C8] r_stoma_min CAM: 400.0 s/m (MASIH PLACEHOLDER)")

    print("\n[PARAMETER YANG SUDAH REAL]")
    for plant in [bromelia, wedelia]:
        print(f"\n{plant.name}:")
        print(f"  H_f         = {plant.H_f} m")
        print(f"  LAI         = {plant.LAI} (akan dikalibrasi)")
        print(f"  rho_f       = {plant.rho_f}")
        print(f"  tau_f [C2]  = {plant.tau_f}")
        print(f"  alpha_f [C3]= {plant.alpha_f:.3f}")
        print(f"  r_stoma_min = {plant.r_stoma_min} s/m"
              + (" [C4 REAL]" if plant.plant_type=="C3" else " [C8 PLACEHOLDER]"))

    print("\n[PARAMETER YANG MASIH TODO]")
    todos = [
        ("H_g",           "ukur tebal substrat di boks"),
        ("H_slab",        "ukur tebal slab di boks"),
        ("d_f",           "ukur lebar daun dengan ImageJ"),
        ("rho_g",         "lab mekanika tanah: specific gravity"),
        ("theta_sat",     "lab mekanika tanah: water content"),
        ("k_theta_sat",   "lab mekanika tanah: permeability [KRITIS]"),
        ("r_stoma_min CAM", "upload file Licor CAM [C8]"),
        ("theta_initial", "ekstrak data soil moisture"),
    ]
    for param, keterangan in todos:
        print(f"  ✗ {param:<20} → {keterangan}")

    print("\n[CARA PAKAI SETELAH DATA LENGKAP]")
    print("""
# 1. Load data
weather = load_weather_data('3-24april.txt',
                             date_start='2026-04-03',
                             date_end='2026-04-10')
ni_data = load_NI_sensor_data('Pengukuran_3_April_2026.xlsx')

# 2. Sinkronisasi
weather_sync, ni_sync = synchronize_datasets(weather, ni_data)

# 3. Isi parameter yang masih None
substrat.rho_g       = 950.0    # dari lab
substrat.theta_sat   = 0.55     # dari lab
substrat.k_theta_sat = 5e-6     # dari lab
slab.H_slab          = 0.10     # dari pengukuran
geom.H_g             = 0.08     # dari pengukuran
bromelia.d_f         = 0.045    # dari ImageJ
wedelia.d_f          = 0.060    # dari ImageJ

# 4. Update r_stoma_min CAM dari Licor
licor_cam = load_licor_data('licor_CAM_siang.xlsx')
bromelia.r_stoma_min = extract_r_stoma_min(licor_cam, plant_type='CAM')

# 5. [C1] Siapkan T_in series dari NI data
T_in_CAM = ni_sync['T_in_CAM']   # T1Ka
T_in_C3  = ni_sync['T_in_C3']    # T3Kd

# 6. Kalibrasi LAI
LAI_bromelia, res_cam = calibrate_LAI(
    weather_sync, ni_sync, bromelia,
    substrat, slab, geom, num
)
LAI_wedelia, res_c3 = calibrate_LAI(
    weather_sync, ni_sync, wedelia,
    substrat, slab, geom, num
)

# 7. Plot hasil
plot_results(res_cam, res_c3, weather_sync, ni_sync,
             save_path='hasil_simulasi.png')
    """)
