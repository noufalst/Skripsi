# CAM gsw-driven model — fixed-input version

This version is intentionally conservative for thesis validation.

## Main principle

LI-COR `gsw` is used to build the CAM stomatal resistance profile:

```text
gsw -> r_s = 1 / (gsw * 0.0224)
```

The simulation then computes foliage evapotranspiration, substrate evaporation,
moisture balance, and slab temperature. LI-COR is not used as a continuous
weather-like input; it is used as physiological parameter evidence.

## What is fixed during calibration

The limited calibration does **not** tune:

- `H_g`
- `H_slab`
- `LAI`
- `cover_fraction`
- `rho_g`
- `theta_sat`
- `tau_f`
- `k_theta_sat`
- `lambda_dry`
- `lambda_sat`
- LI-COR-derived `r_s`

Those are treated as measured, literature, or explicitly selected input values.
If they are uncertain, change them manually and call it sensitivity analysis,
not hidden calibration.

## What can be calibrated

By default, `--calibrate-thermal` only tunes:

- `h_in`

Reason: indoor natural convection inside the small outdoor box is uncertain and
not directly measured.

Optional:

- `--calibrate-initial-theta`

This tunes only a scale factor on the initial moisture state. It does **not**
change `theta_sat`. Use this only when the initial moisture condition is
uncertain, especially for no-soil-sensor/predictive tests.

## Recommended commands

Fixed-physics validation:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444
```

Limited calibration, h_in only:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444 --calibrate-thermal --n-trials 40
```

Limited calibration plus uncertain initial theta:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444 --calibrate-thermal --calibrate-initial-theta --n-trials 40
```

Run without soil sensor input:

```bash
python run_cam_gsw_driven.py --manual-gsw 10:0.03 16:0.2261 21:0.444 --no-soil
```

## Main outputs

The default output folder is:

```text
outputs_cam_gsw_fixed_inputs/
```

Important files:

- `cam_gsw_validation.png`
- `cam_gsw_soil_et_split.png`
- `cam_gsw_prediction_full_with_spinup.csv`
- `cam_gsw_prediction_eval_window.csv`
- `cam_gsw_metrics.json`
- `cam_gsw_rs_profile.json`
- `cam_thermal_calibration_trials.csv` if calibration is used
- `cam_best_thermal_params.json` if calibration is used

## Why this version exists

Previous calibration versions tuned too many variables, which can make the graph
look better but weakens scientific defensibility. This version keeps the physical
experiment fixed and only permits a very small calibration of uncertain boundary
conditions.
