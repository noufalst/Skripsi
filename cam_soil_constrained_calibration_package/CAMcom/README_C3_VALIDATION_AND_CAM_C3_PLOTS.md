# C3 Validation and CAM+C3 Professor Pack

You were right: the previous combined plotter expected a C3 validation output, but we had not made the C3 validation script yet.

This package contains:

```text
run_c3_quick_validation.py
make_cam_c3_professor_plots.py
```

## Step 1 — Run C3 validation

Default:

```bash
python run_c3_quick_validation.py
```

Important: the default C3 window is:

```text
2026-04-09 11:05 -> 2026-04-10 14:08
```

If your C3 window is different:

```bash
python run_c3_quick_validation.py --start "YYYY-MM-DD HH:MM:SS" --end "YYYY-MM-DD HH:MM:SS"
```

If your C3 model output folder is different:

```bash
python run_c3_quick_validation.py --model-output-dir outputs_c3_gsw_fixed_inputs
```

If your model CSV name is different:

```bash
python run_c3_quick_validation.py --model-output-dir outputs_c3_gsw_fixed_inputs --model-file c3_prediction_eval_window.csv
```

If you already know the correct measured C3 target sensor:

```bash
python run_c3_quick_validation.py --target-sensor T3Ka
```

## Step 2 — Make CAM+C3 professor plots

After CAM stack validation and C3 quick validation exist:

```bash
python make_cam_c3_professor_plots.py
```

Inputs expected:

```text
outputs_cam_stack_validation/cam_stack_pair_timeseries.csv
outputs_c3_stack_validation/c3_stack_pair_timeseries.csv
```

Outputs:

```text
outputs_prof_meeting_cam_c3/
  01_CAM_main_T2A_vs_Ts_in.png
  02_C3_validation.png
  03_CAM_vs_C3_bias_corrected_comparison.png
  04_CAM_C3_summary_table.png
  05_talking_points_CAM_C3.md
  prof_cam_c3_key_metrics.csv
```

## What the C3 validation does

It scans possible C3 NI sensors against available model nodes and chooses the best target for `T_s_in`, unless you force a target sensor using `--target-sensor`.

This is a practical quick validation for tomorrow's meeting, not the final sensor-mapping proof.
