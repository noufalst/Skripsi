# Green Roof Skripsi — Codex Context

## Project goal

Saya sedang mengerjakan skripsi simulasi termal green roof menggunakan Python. Fokus utama saat ini adalah memperbaiki simulasi CAM/Bromelia karena hasil model `T_s_in` terlalu flat dibanding data sensor indoor.

Referensi utama model:
- Chagolla-Aranda et al. (2025), *Journal of Building Engineering*, 103, 112053.

Tanaman:
- CAM / Bromelia / Neoregelia
- C3 / Wedelia

Substrat:
- Tanah + sekam padi
- Tebal substrat aktual: 10 cm
- Tebal slab/atap sementara: 10 cm

---

## Current important files

Main files:
- `new_baru_revised_same_structure_v2.py` or cleaned equivalent  
  Main model file. Contains plant parameters, substrate/slab parameters, data loaders, simulation functions, validation functions, and plotting helpers.

- `run_green_roof.py` or cleaned runner  
  Runner for CAM/C3 validation, diagnostics, forced slab checks, and plotting.

Diagnostic scripts may exist:
- `verify_cam_roof_conduction.py`
- `plot_all_ni_sensors_inline_labels_OMIT_ANOMALIES.py`

---

## Main problem

CAM simulation output has too little amplitude:

- Model `T_s_in` is too flat.
- The model likely has excessive damping.
- Do not immediately assume the discretization is wrong.
- The first suspects are physical/empirical parameters and sensor mapping.

Main suspects:

1. `bromelia.cover_fraction = 0.95` is too high.
2. `h_in = 8 W/m²K` may be too high for a passive test box with natural convection.
3. Substrate/slab thermal mass and moisture may be too damping.
4. Target sensor mapping still has uncertainty.
5. The CAM green roof may have been dismantled around April 7 or April 10, so data after dismantling must not be treated as continuous CAM green roof data.

---

## Important distinction: LAI is not cover fraction

LAI and projected canopy cover fraction are different.

Known/assumed CAM values:

```python
bromelia.LAI = 1.95
bromelia.tau_f = 0.07
bromelia.rho_f = 0.390
bromelia.alpha_f = 1.0 - bromelia.rho_f - bromelia.tau_f  # 0.540
```

From ImageJ / visual canopy segmentation, CAM projected cover fraction looks closer to about `0.60–0.65`, not `0.95`.

The current model calculates solar radiation reaching substrate approximately as:

```python
G_to_substrate = ((1 - cover) + cover * tau_f) * G_sol
```

For CAM:

```python
tau_f = 0.07
```

If:

```python
cover = 0.95
```

then:

```text
G_to_substrate = (0.05 + 0.95 * 0.07) * G_sol
               = 0.1165 * G_sol
```

Only about 11.65% of solar radiation reaches the substrate.

If:

```python
cover = 0.60
```

then:

```text
G_to_substrate = (0.40 + 0.60 * 0.07) * G_sol
               = 0.442 * G_sol
```

About 44.2% of solar radiation reaches the substrate.

This is likely a major reason why the CAM simulation becomes too flat when `cover_fraction = 0.95`.

---

## Working NI sensor mapping for March/April 2026 data

Use March/April NI headers as-is, but interpret them using the September nomenclature.

### CAM box

| Header | Working interpretation |
|---|---|
| `T1Ka` | Ruangan CAM / indoor air candidate / boundary candidate |
| `T1Ke` | Tanah atas CAM |
| `T2A` | Tanah bawah CAM |
| `T2A2` | Atap outdoor CAM candidate |
| `T1Tb` | Atap indoor CAM candidate |

### C3 box

| Header | Working interpretation |
|---|---|
| `T3Kd` | Ruangan C3 |
| `T3Ka` | Atap indoor C3 |
| `T1A` | Atap outdoor C3 |

### RR / reference roof box

| Header | Working interpretation |
|---|---|
| `T1Ta` | Ruangan RR |
| `T1Ta2` | Atap outdoor RR |
| `T2Ka` | Atap indoor RR |

---

## Important experimental note

CAM may have been dismantled around April 7 or April 10.

Therefore:

- Do not treat data after dismantling as continuous CAM green roof data.
- Use CAM validation window before dismantling only.
- After dismantling, the three boxes may effectively become reference roofs.
- Data after dismantling can be used for sensor audit or reference roof comparison, but not for CAM green roof validation.

Recommended CAM validation window:

```python
CAM_VALID_START = "2026-03-31 11:58:00"
CAM_VALID_END   = "2026-04-02 21:42:00"
```

---

## Current validation issue

The model output is:

```python
T_s_in
```

This represents inner roof/slab surface temperature.

The official primary target candidate for CAM should ideally be:

```python
target_col = "T1Tb"  # atap indoor CAM candidate
```

Boundary candidate:

```python
T_in_series = ni["T1Ka"]  # ruangan CAM / indoor air candidate
```

However, there is uncertainty:

- The sensor labeled as room / indoor air may visually have more reasonable amplitude.
- If using a room-labeled sensor as an alternative target, treat it as an alternative sensor mapping test, not final truth.
- Avoid using the same sensor as both `T_in_series` input boundary and `T_s_in` validation target.

Possible scenario tests:

```text
Case A:
target = T1Tb
T_in   = T1Ka

Case B:
target = T1Ka
T_in   = None

Case C:
target = T1Ka
T_in   = T1Tb
```

Case B is cleaner if `T1Ka` is assumed temporarily to be the correct indoor roof sensor, because the target is not also used as the lower boundary input.

---

## Recommended CAM parameter test

Start with:

```python
gr.apply_scientific_guess_parameters(
    rho_g=400.0,
    H_slab=0.10,
    H_g=0.10,
    theta_sat=0.85,
    k_theta_sat=5e-6,
    lambda_dry=0.12,
    h_in=2.0,
)

gr.geom.dynamic_h_in = False

gr.bromelia.LAI = 1.95
gr.bromelia.cover_fraction = 0.60
gr.bromelia.tau_f = 0.07
gr.bromelia.alpha_f = 1.0 - gr.bromelia.rho_f - gr.bromelia.tau_f
```

Then sweep:

```python
cover_values = [0.50, 0.60, 0.65, 0.70, 0.80, 0.95]
h_in_values = [1.5, 2.0, 3.0, 4.0, 8.0]
theta_sat_values = [0.75, 0.85, 0.90]
```

Do not tune everything at once. Start with `cover_fraction` and `h_in`.

---

## Desired outputs for professor update

Generate clear and defensible plots:

1. CAM baseline vs improved model vs measured target.
2. Sensitivity of `cover_fraction`.
3. Sensitivity of `h_in`.
4. Metrics table.

Metrics table should include:

- Bias
- MAE
- RMSE
- measured amplitude
- model amplitude
- amplitude error
- peak error
- minimum error

Suggested narration:

```text
The initial CAM model showed excessive damping in the predicted inner roof temperature.
The first investigation indicates that the damping is strongly affected by the assumed effective canopy cover and the indoor convection coefficient.
The initial cover fraction of 0.95 allowed only about 11.65% of solar radiation to reach the substrate.
Based on ImageJ segmentation, the projected canopy cover is closer to 0.60–0.65, which increases solar forcing to the substrate and improves the model amplitude.
```

---

## Coding style requested

Please keep code organized by section:

1. Imports and constants
2. Configuration
3. Data loading
4. Cleaning/anomaly handling
5. Simulation runner
6. Metrics
7. Plotting
8. Main execution

Avoid:

- messy duplicated top-level code
- auto-running heavy simulation when imported
- changing too many parameters at once
- using the same sensor as both input boundary and validation target unless explicitly testing that assumption

Use:

```python
if __name__ == "__main__":
    main()
```

---

## Suggested first Codex task

Read this file first, then inspect the Python scripts.

Main task:

```text
Refactor the CAM validation runner so it can compare multiple sensor mapping scenarios and run sensitivity analysis for cover_fraction and h_in. Keep the code organized by section, avoid duplicated top-level code, and output clear plots plus CSV metrics for professor update.
```

Suggested scenarios:

```python
SCENARIOS = {
    "official_T1Tb_target": {
        "target_col": "T1Tb",
        "T_in_col": "T1Ka",
        "description": "Official mapping: T1Tb as atap indoor CAM, T1Ka as room air boundary.",
    },
    "alternative_T1Ka_target_no_Tin": {
        "target_col": "T1Ka",
        "T_in_col": None,
        "description": "Alternative mapping: T1Ka temporarily tested as indoor roof target; no dynamic T_in input to avoid circular validation.",
    },
    "alternative_T1Ka_target_T1Tb_boundary": {
        "target_col": "T1Ka",
        "T_in_col": "T1Tb",
        "description": "Alternative mapping: T1Ka as target and T1Tb as lower boundary candidate.",
    },
}
```

Expected outputs:

```text
outputs_cam_update/
├── cam_baseline_vs_improved.png
├── cam_cover_fraction_sensitivity.png
├── cam_h_in_sensitivity.png
├── cam_scenario_metrics.csv
├── cam_cover_sensitivity_metrics.csv
└── cam_h_in_sensitivity_metrics.csv
```
