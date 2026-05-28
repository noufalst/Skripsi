# CAM GSW-Driven Green Roof Model

This package is CAM-only and focuses on using LI-COR `gsw` in the model instead of treating it as side data.

## Core scientific decision

The model uses:

```text
LI-COR gsw -> hourly CAM gsw profile -> r_s(t) = 1/(gsw*0.0224)
```

It keeps **one** `r_stoma_min_CAM`, derived from robust high-end conductance (95th percentile gsw), and uses the hourly profile for actual `r_s(t)`.

This avoids the oversimplified assumption:

```text
night = always open, day = always closed
```

because your CAM data suggests the upper-tail day/night gsw can be similar, while median/integrated behavior can still differ.

## Main files

- `green_roof_cam_gsw.py` — CAM model module, sectioned like your preferred `green_roof_model.py` format.
- `run_cam_gsw_driven.py` — runner, plotting, validation, optional thermal calibration.
- `cam_hourly_gsw_profile_template.csv` — optional manual gsw profile template.

## Basic run

Put the code in the same folder as your files:

```text
weatherfile mar-april.xlsx
Pengukuran 30_1 Maret 2026.xlsx
Pengukuran 30_2 Maret 2026.xlsx
Pengukuran 3 April 2026.xlsx
sensor 1 COM5_CAM.csv
sensor 2 COM6_CAM.csv
LI-COR CAM files
```

Run:

```bash
python run_cam_gsw_driven.py
```

## If LI-COR Excel exports read as zero

Use manual hourly gsw values from your verified spreadsheet cells:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444
```

Or use the template CSV:

```bash
python run_cam_gsw_driven.py --hourly-gsw-profile cam_hourly_gsw_profile_template.csv
```

## Improve temperature amplitude scientifically

Do not tune stomata to fix the graph. Keep `gsw/r_s` fixed and tune thermal/radiation parameters:

```bash
python run_cam_gsw_driven.py --calibrate-thermal --n-trials 60
```

This changes parameters such as effective `H_slab`, `H_g`, `cover_fraction`, `tau_f`, `lambda_dry`, and `rho_g`, while keeping LI-COR-derived stomatal behavior fixed.

## Outputs

```text
outputs_cam_physical/
  cam_physical_validation.png
  cam_physical_soil_et_split.png
  cam_physical_prediction_full_with_spinup.csv
  cam_physical_prediction_eval_window.csv
  cam_physical_metrics.json
  cam_gsw_rs_profile.json
  cam_thermal_calibration_trials.csv          # if --calibrate-thermal
  cam_best_thermal_params.json                # if --calibrate-thermal
```

## Suggested manual profile from the current discussion

Use this only as a starting sensitivity case:

```text
10:0.03     # low morning/midday CAM gsw
16:0.2261   # late-afternoon partial reopening
21:0.444    # night/open session median from the uploaded CSV
```

Run:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444 --calibrate-thermal --n-trials 60
```
