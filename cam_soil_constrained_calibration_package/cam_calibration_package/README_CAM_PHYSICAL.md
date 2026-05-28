# CAM Physical Green Roof Model — LI-COR `r_s` Version

This package is CAM-only for now. It follows the readable section style of your previous `green_roof_model.py`, but the physiology is changed so LI-COR gas-exchange data is actually used.

## Files

```text
green_roof_cam_physical.py   # model functions and equations
run_cam_physical.py          # runner / validation workflow
README_CAM_PHYSICAL.md       # this note
```

## Main scientific change

Previous reduced calibration versions used an empirical plant cooling term. This version uses:

```text
LI-COR gsw
→ r_s = 1 / (gsw × 0.0224)
→ h_eva_f = LAI/gamma × rho_air cp_air / (r_a + r_s)
→ j_eva_f
→ foliage/substrate/slab temperature response
```

So `gsw` is not just reported. It directly affects evapotranspiration and temperature.

## Expected data files

Put these in the same folder, or use `--base-dir`:

```text
weatherfile mar-april.xlsx
Pengukuran 30_1 Maret 2026.xlsx
Pengukuran 30_2 Maret 2026.xlsx
Pengukuran 3 April 2026.xlsx
sensor 1 COM5_CAM.csv
sensor 2 COM6_CAM.csv
LI-COR CAM files, e.g.
  2026-03-31-2148_logdata cam new d1s4.csv
  2026-04-01-1616_logdata cam new d1s3.xlsx
```

## Basic run

```bash
python run_cam_physical.py
```

## Run with your data folder

```bash
python run_cam_physical.py --base-dir "E:\Pagi\SKRRRRRRRipsi\data"
```

## Important: d1s3 late-afternoon gsw

If your local d1s3 Excel has valid `gsw` around 0.2261 mol m-2 s-1 but the exported file is read as zero, directly pass the derived late-afternoon resistance:

```bash
python run_cam_physical.py --r-s-late 197
```

You can also force all phase values:

```bash
python run_cam_physical.py --r-s-night 100.5 --r-s-midday 500 --r-s-late 197
```

## Outputs

Saved to `outputs_cam_physical/` by default:

```text
cam_physical_validation.png
cam_physical_soil_et_split.png
cam_physical_prediction_full_with_spinup.csv
cam_physical_prediction_eval_window.csv
cam_physical_metrics.json
cam_rs_profile.json
```

## Why spin-up is used

The model simulates before the validation window and calculates metrics only during the evaluation window. This avoids the previous problem where initial temperature was optimized freely and created an artificial descending start.

Default:

```text
spin-up = 6 hours
validation = 2026-03-31 11:58 → 2026-04-02 21:42
```

Change it with:

```bash
python run_cam_physical.py --spinup-hours 12
```

## Notes

This is not yet the final C3+CAM comparison model. It is the CAM physical core first, because we need to verify that:

1. `r_s` behaves correctly,
2. `j_eva_f` and `j_eva_g` are separated,
3. CAM soil moisture does not get contradicted,
4. the graph improves for the right reason, not from arbitrary plant cooling.
