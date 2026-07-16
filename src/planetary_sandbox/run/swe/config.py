"""Resolved shallow-water (swe) run configuration.

Deliberately minimal (the spec for the first shallow-water milestone):
gravity, planetary radius, rotation rate, mean fluid depth, spectral
resolution, duration, and the snapshot schedule. No presets, no topography,
no forcing, no expression-based initial conditions. Import-light (stdlib
only) so ``--help`` and validation never touch CuPy.

Snapshot semantics are shared with the BVE (count mode canonical, interval
mode supported); the schedule machinery lives in ``run.engine``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from ..engine import (SECONDS_PER_DAY, _require_finite_positive,
                      count_snapshot_times, interval_snapshot_times)
from ..bve.config import (GRID_TYPES, MIN_NLAT, MIN_NLON,
                          scientific_config_subset)

#: Only image product the SWE runner renders (no snapshot viewer yet).
SWE_PLOT_TYPES = ("diagnostics",)

#: Default sidereal day (hours): 2*pi / 7.292e-5 s^-1, i.e. Earth's rotation
#: rate. Unlike the BVE (whose historical default is non-rotating), the
#: shallow-water core defaults to a rotating planet.
DEFAULT_DAY_HOURS = 23.9345

#: Default mean fluid depth (m); a typical barotropic test-suite depth.
#: The canonical Williamson-2 configuration (g*h0 = 2.94e4 m^2/s^2) is
#: obtained by passing the mean depth explicitly (see tests/test_williamson2).
DEFAULT_MEAN_DEPTH_M = 3000.0

#: Standard gravity of the Williamson et al. (1992) suite (m/s^2).
DEFAULT_GRAVITY = 9.80616

DEFAULT_N_SNAPSHOTS = 5

#: Available initial-condition scenarios (must match run/swe/
#: initial_conditions.SWE_INITIAL_CONDITIONS; kept as a plain mapping here
#: because that module imports CuPy at import time).
SWE_SCENARIOS = {
    "rest": "Resting atmosphere: zeta = delta = phi = 0 (all tendencies zero).",
    "gravity_wave": "Small-amplitude Y_4^2 geopotential perturbation "
                    "(linear gravity-wave test).",
    "williamson2": "Williamson et al. (1992) case 2: steady nonlinear "
                   "zonal geostrophic flow.",
}

_MAX_T_END_SECONDS = 1e12

SWE_BASE_DEFAULTS: dict = {
    "lmax": 21,
    "grid": "geodesic",
    "resolution": 4,
    "nlat": 128,
    "nlon": 256,
    "day_hours": DEFAULT_DAY_HOURS,
    "radius_earth_units": 1.0,
    "duration_days": 1.0,
    "gravity": DEFAULT_GRAVITY,
    "mean_depth_m": DEFAULT_MEAN_DEPTH_M,
    "scenario": "williamson2",
    "out": "runs",
    "experiment": None,
    "overwrite": False,
}


@dataclass(frozen=True)
class SWERunConfig:
    """Fully resolved configuration for one shallow-water run."""

    lmax: int = 21
    grid: str = "geodesic"
    resolution: int = 4
    nlat: int = 128
    nlon: int = 256
    day_hours: float = DEFAULT_DAY_HOURS
    radius_earth_units: float = 1.0
    duration_days: float = 1.0
    gravity: float = DEFAULT_GRAVITY
    mean_depth_m: float = DEFAULT_MEAN_DEPTH_M
    scenario: str = "williamson2"
    dt_snapshots: Optional[float] = None
    snapshot_mode: str = "count"
    n_snapshots: Optional[int] = DEFAULT_N_SNAPSHOTS
    plots: tuple[str, ...] = SWE_PLOT_TYPES
    out: str = "runs"
    experiment: Optional[str] = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.grid not in GRID_TYPES:
            raise ValueError(f"grid must be one of {GRID_TYPES}, got {self.grid!r}")
        if self.lmax < 1:
            raise ValueError(f"lmax must be >= 1, got {self.lmax}")
        if self.resolution < 0:
            raise ValueError(f"resolution must be >= 0, got {self.resolution}")
        if self.grid == "latlon":
            if self.nlat < MIN_NLAT:
                raise ValueError(
                    f"lat-lon backend requires nlat >= {MIN_NLAT}, got {self.nlat}")
            if self.nlon < MIN_NLON:
                raise ValueError(
                    f"lat-lon backend requires nlon >= {MIN_NLON}, got {self.nlon}")
        elif self.nlat < 1 or self.nlon < 1:
            raise ValueError(f"nlat/nlon must be positive, got {self.nlat}x{self.nlon}")

        if self.scenario not in SWE_SCENARIOS:
            raise ValueError(
                f"unknown swe scenario {self.scenario!r}; choose from "
                f"{', '.join(sorted(SWE_SCENARIOS))}")

        duration = _require_finite_positive("duration_days", self.duration_days)
        t_end = duration * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end <= 0 or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {self.duration_days} overflows t_end "
                f"(got {t_end} s, cap {_MAX_T_END_SECONDS} s)")
        _require_finite_positive("radius_earth_units", self.radius_earth_units)
        _require_finite_positive("gravity", self.gravity)
        _require_finite_positive("mean_depth_m", self.mean_depth_m)
        if self.day_hours != math.inf:
            _require_finite_positive("day_hours", self.day_hours)

        if self.snapshot_mode == "interval":
            if self.dt_snapshots is None:
                raise ValueError("snapshot interval must be provided in interval mode")
            _require_finite_positive("snapshot interval", self.dt_snapshots)
            if self.n_snapshots is not None:
                raise ValueError("n_snapshots must be None in interval mode")
        elif self.snapshot_mode == "count":
            if not isinstance(self.n_snapshots, int) or isinstance(self.n_snapshots, bool):
                raise ValueError(
                    f"snapshot count must be an integer, got {self.n_snapshots!r}")
            if self.n_snapshots < 0:
                raise ValueError(
                    f"snapshot count must be >= 0, got {self.n_snapshots}")
            if self.n_snapshots >= 2:
                expected = t_end / (self.n_snapshots - 1)
                if self.dt_snapshots is None or not math.isclose(
                        self.dt_snapshots, expected, rel_tol=1e-12):
                    raise ValueError(
                        "dt_snapshots must equal duration/(N-1) in count mode")
            elif self.dt_snapshots is not None:
                raise ValueError("dt_snapshots must be None for N in {0, 1}")
        else:
            raise ValueError(f"unknown snapshot_mode: {self.snapshot_mode!r}")

        unknown_plots = set(self.plots) - set(SWE_PLOT_TYPES)
        if unknown_plots:
            raise ValueError(f"unknown plot types: {sorted(unknown_plots)}")

    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls, explicit: Mapping) -> "SWERunConfig":
        """Layer explicit (user-supplied) values over the ordinary defaults."""
        explicit = {k: v for k, v in dict(explicit).items() if v is not None}

        allowed = set(SWE_BASE_DEFAULTS) | {
            "n_snapshots", "dt_snapshots", "no_plots"}
        unknown = set(explicit) - allowed
        if unknown:
            raise ValueError(f"unknown explicit settings: {sorted(unknown)}")

        settings = dict(SWE_BASE_DEFAULTS)
        settings.update({k: v for k, v in explicit.items()
                         if k in SWE_BASE_DEFAULTS})
        if settings["grid"] == "gauss-latlon":  # user-facing alias
            settings["grid"] = "latlon"

        duration_days = _require_finite_positive(
            "duration_days", settings["duration_days"])
        t_end = duration_days * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {duration_days} overflows t_end (got {t_end} s)")

        n = explicit.get("n_snapshots")
        interval = explicit.get("dt_snapshots")
        if n is not None and interval is not None:
            raise ValueError(
                "snapshot count and snapshot interval are mutually exclusive; "
                "provide at most one")
        if n is None and interval is None:
            n = DEFAULT_N_SNAPSHOTS
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(f"snapshot count must be an integer, got {n!r}")
            if n < 0:
                raise ValueError(f"snapshot count must be >= 0, got {n}")
            snapshot_mode = "count"
            dt = t_end / (n - 1) if n >= 2 else None
        else:
            interval = _require_finite_positive("snapshot interval", interval)
            snapshot_mode = "interval"
            dt = interval

        plots = () if explicit.get("no_plots") else SWE_PLOT_TYPES

        return cls(dt_snapshots=dt, snapshot_mode=snapshot_mode,
                   n_snapshots=n, plots=plots, **settings)

    # ------------------------------------------------------------------

    def snapshot_times_seconds(self) -> list[float]:
        t_end = self.duration_days * SECONDS_PER_DAY
        if self.snapshot_mode == "count":
            return count_snapshot_times(self.n_snapshots, t_end)
        return interval_snapshot_times(self.dt_snapshots, t_end)

    def scientific_config_dict(self) -> dict:
        return scientific_config_subset(self.to_run_config_dict())

    def to_run_config_dict(self) -> dict:
        """Config dict for make_run_id, config.json, and manifest.json."""
        return {
            "solver": "swe",
            "lmax": self.lmax,
            "grid": self.grid,
            "resolution": self.resolution,
            "nlat": self.nlat,
            "nlon": self.nlon,
            "day_hours": self.day_hours,
            "radius_earth_units": self.radius_earth_units,
            "duration_days": self.duration_days,
            "gravity": self.gravity,
            "mean_depth_m": self.mean_depth_m,
            "dt_snapshots": self.dt_snapshots,
            "scenario": self.scenario,
            # Frozen for the shallow-water core: nonlinear products always
            # use the backend's fine (overresolved / 3/2-rule) sampling.
            "product_quadrature": "fine",
            "out": self.out,
            "experiment": self.experiment,
            "overwrite": self.overwrite,
            "snapshot_mode": self.snapshot_mode,
            "n_snapshots": self.n_snapshots,
            "snapshot_times": self.snapshot_times_seconds(),
            "plots": list(self.plots),
        }

    def summary_lines(self) -> list[str]:
        """Concise resolved-configuration summary (no CUDA involved)."""
        times = self.snapshot_times_seconds()
        day = ("inf (non-rotating)" if self.day_hours == math.inf
               else f"{self.day_hours:g} h")
        if self.snapshot_mode == "count":
            schedule = f"{self.n_snapshots} states (count mode"
            if self.dt_snapshots is not None:
                schedule += f", every {self.dt_snapshots:g} s"
            schedule += ")"
        else:
            schedule = (f"every {self.dt_snapshots:g} s "
                        f"(interval mode, {len(times)} states)")
        out = (self.out if self.experiment is None
               else f"{self.out} (experiment: {self.experiment})")
        lines = ["Resolved run configuration:", "  solver              swe"]
        lines.append(f"  backend/grid        {self.grid}")
        if self.grid == "geodesic":
            lines.append(f"  resolution          {self.resolution} "
                         "(geodesic subdivision level)")
        else:
            lines.append(f"  nlat x nlon         {self.nlat} x {self.nlon} "
                         "(Gauss-Legendre latitudes x uniform longitudes)")
        lines += [
            f"  l_max               {self.lmax}",
            f"  scenario            {self.scenario}",
            f"  day length          {day}",
            f"  radius              {self.radius_earth_units:g} Earth radii",
            f"  gravity             {self.gravity:g} m/s^2",
            f"  mean depth          {self.mean_depth_m:g} m "
            f"(Phi0 = {self.gravity * self.mean_depth_m:g} m^2/s^2)",
            f"  duration            {self.duration_days:g} days",
            f"  snapshots           {schedule}",
            f"  plots               {', '.join(self.plots) if self.plots else 'none'}",
            f"  output base         {out}",
        ]
        return lines
