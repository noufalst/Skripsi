# CAM Soil-Moisture-Constrained Calibration

This package is for improving the CAM simulation graph without forcing unrealistic daytime evapotranspiration.

## Files

- `green_roof_model_clean.py`  
  Clean 1-D green roof model. C3 and CAM are separated through plant-profile parameters only.

- `run_cam_soil_constrained_calibration.py`  
  CAM calibration runner. It minimizes temperature error against measured CAM NI data and applies a soft penalty when CAM daytime plant-cooling flux becomes too high relative to the observed CAM soil-moisture behavior.

## Recommended raw file placement

Place these files in the same folder:

```text
green_roof_model_clean.py
run_cam_soil_constrained_calibration.py
weatherfile mar-april.xlsx
Pengukuran 30_1 Maret 2026.xlsx
Pengukuran 30_2 Maret 2026.xlsx
Pengukuran 3 April 2026.xlsx
sensor 1 COM5_CAM.csv
sensor 2 COM6_CAM.csv
```

Then run:

```bash
python run_cam_soil_constrained_calibration.py
```

Default settings:

```text
CAM window : 2026-03-31 11:58:00 to 2026-04-02 21:42:00
Target     : T1Tb  (recommended CAM inner roof / underside target)
T_in       : T1Ka  (CAM indoor air boundary)
S1/S2      : S1 shallow, S2 deep
Constraint : soft soil-moisture constraint
```

## If you want to compare to T2A2 instead

```bash
python run_cam_soil_constrained_calibration.py --target-col T2A2
```

## If S1/S2 depth assumption is swapped

```bash
python run_cam_soil_constrained_calibration.py --swap-depths
```

## If RIKA timestamp is already local time

The default assumes the CSV timestamp is GMT and converts it to WIB.

```bash
python run_cam_soil_constrained_calibration.py --timestamp-mode local
```

## If you already have a merged file

Use a CSV/XLSX with these columns:

```text
T_air_C
T_in_C
solar_W_m2
T_target_C
soil_moisture        optional
theta_shallow_pct    optional
theta_deep_pct       optional
```

Run:

```bash
python run_cam_soil_constrained_calibration.py --input merged_cam.csv
```

## Outputs

The default output folder is `outputs_cam_calibrated/`.

Important files:

```text
cam_calibrated_vs_measured.png
cam_calibrated_flux_check.png
cam_prediction_calibrated.csv
cam_prediction_baseline.csv
cam_best_params.json
cam_metrics_calibrated.json
cam_metrics_baseline.json
cam_calibration_trials.csv
cam_soil_constraint_summary.json
cam_calibration_input_used.csv
```

## What “least error” means here

The script minimizes a bounded objective:

```text
objective = RMSE + amplitude_weight * amplitude_error + soil-moisture penalty
```

So the result is the best fit within the selected parameter bounds and physical constraint, not a claim of a universal/global truth.

## Important scientific interpretation

Do not tune CAM by simply increasing daytime ET. Your CAM soil-moisture data does not show a consistent daytime drydown, so a physically safer CAM fit should improve the graph mainly through shading, solar absorptivity, thermal inertia, and boundary parameters while keeping daytime plant-cooling flux limited.
