"""
green_roof_model_clean.py

Clean 1-D green-roof heat-transfer model for C3 and CAM cases.

Main design choices:
- Heat model is kept independent from plotting/diagnostics.
- Diagnostics are available, but optional.
- Plant behaviour only changes the TOP surface energy balance.
  It does not touch the indoor boundary, so a previously good C3 result
  should not be damaged by CAM-specific tuning.
- CAM amplitude is controlled by a bounded `amplitude_scale`, so CAM can be
  damped without changing C3 parameters.

Typical usage:
    from green_roof_model_clean import green_roof_model_clean, build_default_config

    config = build_default_config(plant="c3", soil_thickness_m=0.10)
    result = green_roof_model_clean(weather_df, config=config, diagnostics=False)

Expected dataframe columns, case-insensitive aliases are handled in runner:
    datetime index preferred, or any normal row order
    T_air_C          outdoor/ambient air temperature [degC]
    T_in_C           indoor boundary temperature [degC], optional
    solar_W_m2       shortwave radiation [W/m2], optional
    RH_pct           relative humidity [%], optional
    soil_moisture    volumetric or normalized soil moisture, optional

Output columns:
    T_pred_C          predicted inner/bottom roof temperature [degC]
    T_surface_C       predicted green-roof top surface node temperature [degC]
    T_bottom_C        same as T_pred_C, explicit name
    q_plant_W_m2      latent/plant cooling flux removed from top surface [W/m2]
    q_solar_W_m2      absorbed shortwave flux at top surface [W/m2]
    q_top_conv_W_m2   outdoor convection flux into top surface [W/m2]
    q_bottom_W_m2     heat flux through bottom boundary into indoor air [W/m2]
"""

from __future__ import annotations

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# SECTION 1 — DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Layer:
    """Material layer for a 1-D finite-volume roof model."""

    name: str
    thickness_m: float
    k_W_mK: float
    rho_kg_m3: float
    cp_J_kgK: float
    n_nodes: int

    def validate(self) -> None:
        if self.thickness_m <= 0:
            raise ValueError(f"Layer {self.name}: thickness_m must be positive.")
        if self.k_W_mK <= 0:
            raise ValueError(f"Layer {self.name}: k_W_mK must be positive.")
        if self.rho_kg_m3 <= 0:
            raise ValueError(f"Layer {self.name}: rho_kg_m3 must be positive.")
        if self.cp_J_kgK <= 0:
            raise ValueError(f"Layer {self.name}: cp_J_kgK must be positive.")
        if self.n_nodes < 1:
            raise ValueError(f"Layer {self.name}: n_nodes must be >= 1.")


@dataclass(frozen=True)
class PlantProfileParams:
    """Plant cooling profile for the top surface energy balance.

    The flux is positive when the plant removes heat from the roof surface.
    `amplitude_scale` is intentionally separate so CAM amplitude can be tuned
    without changing the physics of the C3 case.
    """

    plant: str = "c3"
    lai: float = 1.0
    cover_fraction: float = 0.31
    amplitude_scale: float = 1.0
    c3_day_max_W_m2: float = 26.0
    c3_night_max_W_m2: float = 2.0
    cam_day_max_W_m2: float = 8.0
    cam_night_max_W_m2: float = 14.0
    extinction_coeff: float = 0.55
    q_plant_cap_W_m2: float = 45.0

    def validate(self) -> None:
        plant = self.plant.lower().strip()
        if plant not in {"c3", "cam", "none", "bare"}:
            raise ValueError("plant must be one of: 'c3', 'cam', 'none', 'bare'.")
        if self.lai < 0:
            raise ValueError("lai must be >= 0.")
        if not 0 <= self.cover_fraction <= 1:
            raise ValueError("cover_fraction must be between 0 and 1.")
        if self.amplitude_scale < 0:
            raise ValueError("amplitude_scale must be >= 0.")
        if self.q_plant_cap_W_m2 < 0:
            raise ValueError("q_plant_cap_W_m2 must be >= 0.")


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary condition parameters."""

    h_out_W_m2K: float = 12.0
    h_in_W_m2K: float = 8.0
    solar_absorptivity: float = 0.68
    t_in_const_C: float = 27.0
    initial_temp_C: Optional[float] = None

    def validate(self) -> None:
        if self.h_out_W_m2K <= 0:
            raise ValueError("h_out_W_m2K must be positive.")
        if self.h_in_W_m2K <= 0:
            raise ValueError("h_in_W_m2K must be positive.")
        if not 0 <= self.solar_absorptivity <= 1.2:
            raise ValueError("solar_absorptivity should be between 0 and 1.2.")


@dataclass(frozen=True)
class SimulationConfig:
    """Full model configuration."""

    layers: Tuple[Layer, ...]
    plant_params: PlantProfileParams = PlantProfileParams()
    boundary: BoundaryConfig = BoundaryConfig()
    dt_s: Optional[float] = None
    spinup_steps: int = 0
    target_col: Optional[str] = None

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("At least one layer is required.")
        for layer in self.layers:
            layer.validate()
        self.plant_params.validate()
        self.boundary.validate()
        if self.dt_s is not None and self.dt_s <= 0:
            raise ValueError("dt_s must be positive when provided.")
        if self.spinup_steps < 0:
            raise ValueError("spinup_steps must be >= 0.")


# ============================================================
# SECTION 2 — SMALL NUMERICAL UTILITIES
# ============================================================

def _as_float_array(values: Iterable[float], fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if np.all(np.isnan(arr)):
        arr[:] = fallback
    else:
        idx = np.isnan(arr)
        if idx.any():
            good = np.where(~idx)[0]
            bad = np.where(idx)[0]
            arr[idx] = np.interp(bad, good, arr[good])
    return arr


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _safe_col(df: pd.DataFrame, col: str, fallback: float) -> np.ndarray:
    if col in df.columns:
        return _as_float_array(df[col].to_numpy(), fallback=fallback)
    return np.full(len(df), fallback, dtype=float)


def _infer_dt_s(index: pd.Index, default_dt_s: float = 300.0) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        diffs = index.to_series().diff().dt.total_seconds().dropna().to_numpy()
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return float(default_dt_s)


def _hour_of_day(index: pd.Index, n: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    # If no datetime is available, assume samples span a daily cycle repeatedly.
    return np.mod(np.arange(n, dtype=float), 24.0)


def _circular_gaussian_hour(hour: np.ndarray, center: float, width_h: float) -> np.ndarray:
    """Smooth gate for daily profiles with circular 24-hour distance."""
    d = np.abs(hour - center)
    d = np.minimum(d, 24.0 - d)
    return np.exp(-0.5 * (d / max(width_h, 1e-6)) ** 2)


# ============================================================
# SECTION 3 — DEFAULT CONFIGURATION
# ============================================================

def build_default_config(
    plant: str = "c3",
    soil_thickness_m: float = 0.10,
    concrete_thickness_m: float = 0.12,
    lai: Optional[float] = None,
    cover_fraction: float = 0.31,
    cam_amplitude_scale: float = 0.55,
    c3_amplitude_scale: float = 1.0,
    h_in_W_m2K: float = 8.0,
    h_out_W_m2K: float = 12.0,
    solar_absorptivity: float = 0.68,
    target_col: Optional[str] = None,
) -> SimulationConfig:
    """Create a practical default config for the green-roof setup.

    Defaults are intentionally conservative. C3 and CAM differ only through
    plant cooling profile parameters; material layers and indoor boundary stay
    identical unless you explicitly change them.
    """
    plant_key = plant.lower().strip()
    if plant_key in {"none", "bare"}:
        lai_default = 0.0
        amp = 0.0
    elif plant_key == "cam":
        lai_default = 0.8
        amp = cam_amplitude_scale
    else:
        lai_default = 1.07
        amp = c3_amplitude_scale

    plant_params = PlantProfileParams(
        plant=plant_key,
        lai=lai_default if lai is None else float(lai),
        cover_fraction=float(cover_fraction),
        amplitude_scale=float(amp),
    )

    layers = (
        Layer(
            name="soil_substrate",
            thickness_m=float(soil_thickness_m),
            k_W_mK=0.55,
            rho_kg_m3=1300.0,
            cp_J_kgK=1450.0,
            n_nodes=6,
        ),
        Layer(
            name="concrete_slab",
            thickness_m=float(concrete_thickness_m),
            k_W_mK=1.40,
            rho_kg_m3=2200.0,
            cp_J_kgK=880.0,
            n_nodes=7,
        ),
    )

    boundary = BoundaryConfig(
        h_out_W_m2K=float(h_out_W_m2K),
        h_in_W_m2K=float(h_in_W_m2K),
        solar_absorptivity=float(solar_absorptivity),
    )

    config = SimulationConfig(
        layers=layers,
        plant_params=plant_params,
        boundary=boundary,
        target_col=target_col,
    )
    config.validate()
    return config


# ============================================================
# SECTION 4 — GRID BUILDER
# ============================================================

def build_grid(layers: Tuple[Layer, ...]) -> Dict[str, np.ndarray]:
    """Build finite-volume grid arrays from material layers."""
    dx: List[float] = []
    k: List[float] = []
    rho: List[float] = []
    cp: List[float] = []
    layer_name: List[str] = []

    for layer in layers:
        layer.validate()
        local_dx = layer.thickness_m / layer.n_nodes
        for _ in range(layer.n_nodes):
            dx.append(local_dx)
            k.append(layer.k_W_mK)
            rho.append(layer.rho_kg_m3)
            cp.append(layer.cp_J_kgK)
            layer_name.append(layer.name)

    dx_arr = np.asarray(dx, dtype=float)
    x_center = np.cumsum(dx_arr) - 0.5 * dx_arr
    heat_capacity = np.asarray(rho, dtype=float) * np.asarray(cp, dtype=float) * dx_arr

    # Interface conductance per unit area [W/m2K].
    k_arr = np.asarray(k, dtype=float)
    g_interface = np.zeros(len(dx_arr) - 1, dtype=float)
    for i in range(len(g_interface)):
        resistance = dx_arr[i] / (2.0 * k_arr[i]) + dx_arr[i + 1] / (2.0 * k_arr[i + 1])
        g_interface[i] = 1.0 / resistance

    return {
        "dx_m": dx_arr,
        "k_W_mK": k_arr,
        "rho_cp_dx_J_m2K": heat_capacity,
        "x_center_m": x_center,
        "g_interface_W_m2K": g_interface,
        "layer_name": np.asarray(layer_name, dtype=object),
    }


# ============================================================
# SECTION 5 — PLANT / LATENT COOLING PROFILE
# ============================================================

def compute_plant_cooling_flux(
    df: pd.DataFrame,
    params: PlantProfileParams,
) -> np.ndarray:
    """Compute plant cooling flux [W/m2] removed from top surface.

    This is deliberately bounded and smooth. It is not intended to replace a
    full stomatal conductance/photosynthesis model; it is a clean forcing term
    for the roof heat balance.
    """
    params.validate()
    n = len(df)
    plant = params.plant.lower().strip()
    if plant in {"none", "bare"} or n == 0 or params.cover_fraction <= 0 or params.lai <= 0:
        return np.zeros(n, dtype=float)

    t_air = _safe_col(df, "T_air_C", fallback=27.0)
    solar = np.maximum(_safe_col(df, "solar_W_m2", fallback=0.0), 0.0)
    hour = _hour_of_day(df.index, n)

    # Smooth response factors.
    solar_factor = solar / (solar + 120.0)  # robust 0..1, no hard jump at sunrise
    temp_factor = _clip01((t_air - 16.0) / 16.0)

    # Moisture factor is optional. If present and raw values look like %,
    # normalize them gently; otherwise assume the available data are already 0..1.
    if "soil_moisture" in df.columns:
        sm = _safe_col(df, "soil_moisture", fallback=np.nan)
        if np.nanmax(sm) > 1.5:
            sm = sm / 100.0
        moisture_factor = _clip01((sm - 0.08) / 0.22)
        moisture_factor = np.where(np.isfinite(moisture_factor), moisture_factor, 1.0)
    else:
        moisture_factor = np.ones(n, dtype=float)

    # LAI/cover factor. This is intentionally less aggressive than raw LAI.
    lai_factor = 1.0 - np.exp(-params.extinction_coeff * params.lai)
    canopy_factor = params.cover_fraction * lai_factor

    if plant == "c3":
        # C3: mostly daytime transpiration. Tiny night term only prevents a hard discontinuity.
        day_flux = params.c3_day_max_W_m2 * solar_factor * temp_factor
        night_flux = params.c3_night_max_W_m2 * (1.0 - solar_factor) * temp_factor
        q_plant = params.amplitude_scale * canopy_factor * moisture_factor * (day_flux + night_flux)
    elif plant == "cam":
        # CAM: weaker daytime cooling and a smooth nocturnal activity window.
        # The amplitude scale defaults lower than C3 to prevent exaggerated oscillation.
        night_gate = _circular_gaussian_hour(hour, center=2.0, width_h=4.5)
        day_flux = params.cam_day_max_W_m2 * solar_factor * temp_factor
        night_flux = params.cam_night_max_W_m2 * night_gate * (1.0 - 0.55 * solar_factor) * temp_factor
        q_plant = params.amplitude_scale * canopy_factor * moisture_factor * (day_flux + night_flux)
    else:
        raise ValueError("plant must be one of: 'c3', 'cam', 'none', 'bare'.")

    return np.clip(q_plant, 0.0, params.q_plant_cap_W_m2)


# ============================================================
# SECTION 6 — IMPLICIT FINITE-VOLUME SOLVER
# ============================================================

def _solve_one_step(
    temp_old: np.ndarray,
    dt_s: float,
    grid: Dict[str, np.ndarray],
    t_air_C: float,
    t_in_C: float,
    solar_W_m2: float,
    q_plant_W_m2: float,
    boundary: BoundaryConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve one implicit time step."""
    c = grid["rho_cp_dx_J_m2K"]
    g = grid["g_interface_W_m2K"]
    n = len(c)

    a = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i in range(n):
        a[i, i] = c[i] / dt_s
        b[i] = c[i] / dt_s * temp_old[i]

    # Internal conduction interfaces.
    for i in range(n - 1):
        gij = g[i]
        a[i, i] += gij
        a[i, i + 1] -= gij
        a[i + 1, i + 1] += gij
        a[i + 1, i] -= gij

    # Top boundary: convection + absorbed solar - plant latent cooling.
    h_out = boundary.h_out_W_m2K
    q_solar = boundary.solar_absorptivity * max(float(solar_W_m2), 0.0)
    a[0, 0] += h_out
    b[0] += h_out * float(t_air_C) + q_solar - float(q_plant_W_m2)

    # Bottom boundary: dynamic indoor air temperature, if supplied.
    h_in = boundary.h_in_W_m2K
    a[-1, -1] += h_in
    b[-1] += h_in * float(t_in_C)

    temp_new = np.linalg.solve(a, b)

    q_top_conv = h_out * (float(t_air_C) - temp_new[0])
    q_bottom = h_in * (temp_new[-1] - float(t_in_C))
    fluxes = {
        "q_solar_W_m2": q_solar,
        "q_plant_W_m2": float(q_plant_W_m2),
        "q_top_conv_W_m2": float(q_top_conv),
        "q_bottom_W_m2": float(q_bottom),
    }
    return temp_new, fluxes


def simulate_green_roof(
    df: pd.DataFrame,
    config: SimulationConfig,
    diagnostics: bool = False,
) -> pd.DataFrame:
    """Run the green-roof model and return a prediction dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if len(df) < 2:
        raise ValueError("df must contain at least 2 rows.")

    config.validate()
    work = df.copy()

    if "T_air_C" not in work.columns:
        raise ValueError("Input dataframe must contain 'T_air_C'.")

    dt_s = float(config.dt_s or _infer_dt_s(work.index))
    grid = build_grid(config.layers)
    n_nodes = len(grid["dx_m"])

    t_air = _safe_col(work, "T_air_C", fallback=27.0)
    t_in = _safe_col(work, "T_in_C", fallback=config.boundary.t_in_const_C)
    solar = np.maximum(_safe_col(work, "solar_W_m2", fallback=0.0), 0.0)
    q_plant = compute_plant_cooling_flux(work, config.plant_params)

    if config.boundary.initial_temp_C is not None:
        initial_temp = float(config.boundary.initial_temp_C)
        temp = np.full(n_nodes, initial_temp, dtype=float)
    else:
        # Initial profile from outdoor surface side to indoor side.
        temp = np.linspace(float(t_air[0]), float(t_in[0]), n_nodes)

    rows: List[Dict[str, float]] = []

    # Optional spin-up using first forcing row. This improves start-up without
    # changing the actual time range/output length.
    for _ in range(config.spinup_steps):
        temp, _ = _solve_one_step(
            temp_old=temp,
            dt_s=dt_s,
            grid=grid,
            t_air_C=float(t_air[0]),
            t_in_C=float(t_in[0]),
            solar_W_m2=float(solar[0]),
            q_plant_W_m2=float(q_plant[0]),
            boundary=config.boundary,
        )

    # Save initial state at the first timestamp.
    rows.append(
        {
            "T_pred_C": float(temp[-1]),
            "T_surface_C": float(temp[0]),
            "T_bottom_C": float(temp[-1]),
            "q_solar_W_m2": float(config.boundary.solar_absorptivity * solar[0]),
            "q_plant_W_m2": float(q_plant[0]),
            "q_top_conv_W_m2": float(config.boundary.h_out_W_m2K * (t_air[0] - temp[0])),
            "q_bottom_W_m2": float(config.boundary.h_in_W_m2K * (temp[-1] - t_in[0])),
        }
    )

    for j in range(1, len(work)):
        temp, fluxes = _solve_one_step(
            temp_old=temp,
            dt_s=dt_s,
            grid=grid,
            t_air_C=float(t_air[j]),
            t_in_C=float(t_in[j]),
            solar_W_m2=float(solar[j]),
            q_plant_W_m2=float(q_plant[j]),
            boundary=config.boundary,
        )
        rows.append(
            {
                "T_pred_C": float(temp[-1]),
                "T_surface_C": float(temp[0]),
                "T_bottom_C": float(temp[-1]),
                **fluxes,
            }
        )

    out = pd.DataFrame(rows, index=work.index)

    if config.target_col and config.target_col in work.columns:
        out["T_target_C"] = _safe_col(work, config.target_col, fallback=np.nan)
        metrics = compute_metrics(out["T_target_C"], out["T_pred_C"])
        for key, value in metrics.items():
            out.attrs[key] = value

    if diagnostics:
        out.attrs["grid"] = grid
        out.attrs["config"] = config
        out.attrs["dt_s"] = dt_s

    return out


# ============================================================
# SECTION 7 — METRICS
# ============================================================

def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Return basic error metrics for temperature prediction."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return {"n": 0, "bias_C": np.nan, "mae_C": np.nan, "rmse_C": np.nan, "r_C": np.nan}

    err = yp[mask] - yt[mask]
    if np.sum(mask) >= 2 and np.std(yt[mask]) > 0 and np.std(yp[mask]) > 0:
        r = float(np.corrcoef(yt[mask], yp[mask])[0, 1])
    else:
        r = np.nan

    return {
        "n": int(np.sum(mask)),
        "bias_C": float(np.mean(err)),
        "mae_C": float(np.mean(np.abs(err))),
        "rmse_C": float(np.sqrt(np.mean(err ** 2))),
        "r_C": r,
    }


# ============================================================
# SECTION 8 — CONFIG UPDATE HELPERS
# ============================================================

def with_updated_plant(config: SimulationConfig, **kwargs) -> SimulationConfig:
    """Return a copy of config with updated plant parameters."""
    return replace(config, plant_params=replace(config.plant_params, **kwargs))


def with_updated_boundary(config: SimulationConfig, **kwargs) -> SimulationConfig:
    """Return a copy of config with updated boundary parameters."""
    return replace(config, boundary=replace(config.boundary, **kwargs))


# ============================================================
# SECTION 9 — PUBLIC API WRAPPER
# ============================================================

def green_roof_model_clean(
    df: pd.DataFrame,
    config: Optional[SimulationConfig] = None,
    plant: str = "c3",
    diagnostics: bool = False,
    **config_kwargs,
) -> pd.DataFrame:
    """Public wrapper used by the runner.

    You can either pass a ready `SimulationConfig`, or pass `plant` plus
    supported `build_default_config()` keyword arguments.
    """
    if config is None:
        config = build_default_config(plant=plant, **config_kwargs)
    return simulate_green_roof(df=df, config=config, diagnostics=diagnostics)
