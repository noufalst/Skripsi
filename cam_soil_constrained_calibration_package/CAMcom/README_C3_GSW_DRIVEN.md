
# C3 GSW-Driven Simulation + Validation Package

This package gives the missing files needed before running `run_c3_quick_validation.py`.

## Files

```text
green_roof_c3_gsw.py
run_c3_gsw_driven.py
run_c3_quick_validation.py
make_cam_c3_professor_plots.py
```

## Required experiment files in the same folder

```text
weatherfile mar-april.xlsx
Pengukuran 30_1 Maret 2026.xlsx
Pengukuran 30_2 Maret 2026.xlsx
Pengukuran 3 April 2026.xlsx
sensor 1 COM5_C3.csv
sensor 2 COM6_C3.csv
```

## Step 1 — Generate C3 model output

```bash
rmdir /s /q __pycache__
python run_c3_gsw_driven.py
```

This creates:

```text
outputs_c3_gsw_fixed_inputs/
  c3_gsw_prediction_eval_window.csv
  c3_gsw_prediction_full_with_spinup.csv
  c3_gsw_metrics.json
  c3_gsw_validation.png
```

## Step 2 — Run C3 quick validation / target finder

```bash
python run_c3_quick_validation.py --model-output-dir outputs_c3_gsw_fixed_inputs
```

This creates:

```text
outputs_c3_stack_validation/
  c3_stack_pair_timeseries.csv
  c3_main_validation.png
  c3_validation_summary.json
```

## Step 3 — Make CAM + C3 professor plots

Make sure CAM stack validation output already exists:

```text
outputs_cam_stack_validation/cam_stack_pair_timeseries.csv
```

Then run:

```bash
python make_cam_c3_professor_plots.py
```

This creates:

```text
outputs_prof_meeting_cam_c3/
  01_CAM_main_T2A_vs_Ts_in.png
  02_C3_validation.png
  03_CAM_vs_C3_bias_corrected_comparison.png
  04_CAM_C3_summary_table.png
  05_talking_points_CAM_C3.md
```

## Important defaults

C3 default window:

```text
2026-04-09 11:05:00 -> 2026-04-10 14:08:00
```

C3 default measured target:

```text
T3Ka
```

C3 default dynamic indoor boundary:

```text
T3Kd
```

Temporary soil moisture correction is ON by default:

```text
--soil-moisture-scale 0.70
```

This means RK520 reading is reduced by 30 percent as a sensitivity test. It is not a calibration.

To disable:

```bash
python run_c3_gsw_driven.py --soil-moisture-scale 1.0
```

## Useful overrides

If your C3 validation window is different:

```bash
python run_c3_gsw_driven.py --eval-start "YYYY-MM-DD HH:MM:SS" --eval-end "YYYY-MM-DD HH:MM:SS"
```

If your target sensor is different:

```bash
python run_c3_gsw_driven.py --target-col T3Ka --tin-col T3Kd
```

If you want to force C3 r_s from a new LI-COR result:

```bash
python run_c3_gsw_driven.py --r-s-c3 167.2
```
