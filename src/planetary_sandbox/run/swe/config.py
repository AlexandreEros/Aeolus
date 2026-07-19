"""Resolved shallow-water (swe) run configuration.

Deliberately minimal: gravity, planetary radius, rotation rate, mean fluid
depth, spectral resolution, duration, the snapshot schedule, and fixed
analytic bottom topography (flat by default, or one Gaussian mountain). No
presets, no forcing, no expression-based initial conditions, no terrain
files. Import-light (stdlib only) so ``--help`` and validation never touch
CuPy.

Snapshot semantics are shared with the BVE (count mode canonical, interval
mode supported); the schedule machinery lives in ``run.engine``.

Topography config schema (additive)
-----------------------------------
``to_run_config_dict()`` emits the topography keys (``topography`` plus the
four ``mountain_*`` parameters) ONLY when the resolved topography is not
``flat``. A flat-bottom run therefore produces exactly the historical
config dict — and thus exactly the historical scientific hash and run id —
while any non-flat terrain participates fully in the scientific identity.
Old manifests without a ``topography`` key are unambiguously flat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from ..engine import (SECONDS_PER_DAY, _require_finite_number,
                      _require_finite_positive, count_snapshot_times,
                      interval_snapshot_times)
from ..bve.config import (GRID_TYPES, MIN_NLAT, MIN_NLON,
                          scientific_config_subset)

#: Image products in deterministic execution order.  The summary requires at
#: least one persisted state; diagnostics remain available for N=0 runs.
SWE_PLOT_TYPES = ("diagnostics", "snapshots", "summary")
_SWE_PLOTS_REQUIRING_SNAPSHOTS = ("snapshots", "summary")

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
    "rest": "Resting atmosphere: zero velocity, constant free surface "
            "(exact lake-at-rest over topography; all tendencies zero).",
    "gravity_wave": "Small-amplitude Y_4^2 free-surface perturbation "
                    "(linear gravity-wave test).",
    "williamson2": "Williamson et al. (1992) case 2: steady nonlinear "
                   "zonal geostrophic flow (over a mountain: a smooth "
                   "mountain-flow experiment, not steady).",
}

#: Available bottom-topography presets (must match
#: physics/topography.TOPOGRAPHY_PRESETS; duplicated here because that
#: module imports CuPy at import time).
SWE_TOPOGRAPHIES = {
    "flat": "Flat bottom (canonical default; identical to the historical "
            "flat-bottom solver).",
    "mountain": "One smooth isolated Gaussian mountain, band-limited at "
                "the model truncation.",
}

#: Default Gaussian-mountain parameters, applied when --topography mountain
#: is selected and a parameter is not given explicitly.
DEFAULT_MOUNTAIN_HEIGHT_M = 2000.0
DEFAULT_MOUNTAIN_LAT_DEG = 30.0
DEFAULT_MOUNTAIN_LON_DEG = 90.0
DEFAULT_MOUNTAIN_WIDTH_DEG = 20.0

_MOUNTAIN_PARAM_FIELDS = ("mountain_height_m", "mountain_lat_deg",
                          "mountain_lon_deg", "mountain_width_deg")

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
    "topography": "flat",
    "mountain_height_m": None,
    "mountain_lat_deg": None,
    "mountain_lon_deg": None,
    "mountain_width_deg": None,
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
    topography: str = "flat"
    mountain_height_m: Optional[float] = None
    mountain_lat_deg: Optional[float] = None
    mountain_lon_deg: Optional[float] = None
    mountain_width_deg: Optional[float] = None
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

        if self.topography not in SWE_TOPOGRAPHIES:
            raise ValueError(
                f"unknown topography {self.topography!r}; choose from "
                f"{', '.join(sorted(SWE_TOPOGRAPHIES))}")
        if self.topography == "flat":
            given = [name for name in _MOUNTAIN_PARAM_FIELDS
                     if getattr(self, name) is not None]
            if given:
                raise ValueError(
                    f"mountain parameter(s) {given} require "
                    "topography='mountain' (the default topography is flat)")
        else:  # mountain: every parameter must be resolved and valid
            missing = [name for name in _MOUNTAIN_PARAM_FIELDS
                       if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"topography='mountain' requires resolved parameter(s) "
                    f"{missing} (SWERunConfig.resolve applies the defaults)")
            _require_finite_positive("mountain_height_m",
                                     self.mountain_height_m)
            _require_finite_positive("mountain_width_deg",
                                     self.mountain_width_deg)
            if self.mountain_width_deg > 90.0:
                raise ValueError(
                    f"mountain_width_deg must be <= 90, got "
                    f"{self.mountain_width_deg}")
            lat = _require_finite_number("mountain_lat_deg",
                                         self.mountain_lat_deg)
            if not -90.0 <= lat <= 90.0:
                raise ValueError(
                    f"mountain_lat_deg must be in [-90, 90], got {lat}")
            lon = _require_finite_number("mountain_lon_deg",
                                         self.mountain_lon_deg)
            if not -360.0 <= lon <= 360.0:
                raise ValueError(
                    f"mountain_lon_deg must be in [-360, 360], got {lon}")

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
        if list(self.plots) != [p for p in SWE_PLOT_TYPES if p in self.plots]:
            raise ValueError("plots must be deduplicated and in canonical order")
        if self.n_snapshots == 0 and any(
                plot in _SWE_PLOTS_REQUIRING_SNAPSHOTS for plot in self.plots):
            raise ValueError("SWE snapshot visualization requires a stored state")

    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls, explicit: Mapping) -> "SWERunConfig":
        """Layer explicit (user-supplied) values over the ordinary defaults."""
        explicit = {k: v for k, v in dict(explicit).items() if v is not None}

        allowed = set(SWE_BASE_DEFAULTS) | {
            "n_snapshots", "dt_snapshots", "plots", "no_plots"}
        unknown = set(explicit) - allowed
        if unknown:
            raise ValueError(f"unknown explicit settings: {sorted(unknown)}")

        settings = dict(SWE_BASE_DEFAULTS)
        settings.update({k: v for k, v in explicit.items()
                         if k in SWE_BASE_DEFAULTS})
        if settings["grid"] == "gauss-latlon":  # user-facing alias
            settings["grid"] = "latlon"

        # Resolve the mountain parameters: defaults apply only when the
        # mountain preset is selected; supplying them with a flat bottom is
        # an error (caught by __post_init__, with an early clear message
        # here for the common CLI path).
        if settings["topography"] == "mountain":
            mountain_defaults = {
                "mountain_height_m": DEFAULT_MOUNTAIN_HEIGHT_M,
                "mountain_lat_deg": DEFAULT_MOUNTAIN_LAT_DEG,
                "mountain_lon_deg": DEFAULT_MOUNTAIN_LON_DEG,
                "mountain_width_deg": DEFAULT_MOUNTAIN_WIDTH_DEG,
            }
            for name, default in mountain_defaults.items():
                if settings[name] is None:
                    settings[name] = default
        else:
            given = [name for name in _MOUNTAIN_PARAM_FIELDS
                     if settings[name] is not None]
            if given:
                raise ValueError(
                    f"mountain parameter(s) {given} require "
                    "--topography mountain")

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

        has_snapshots = not (snapshot_mode == "count" and n == 0)
        plots = cls._resolve_plots(
            explicit.get("plots"), explicit.get("no_plots"),
            has_snapshots=has_snapshots)

        return cls(dt_snapshots=dt, snapshot_mode=snapshot_mode,
                   n_snapshots=n, plots=plots, **settings)

    @staticmethod
    def _resolve_plots(requested, no_plots, *,
                       has_snapshots: bool) -> tuple[str, ...]:
        """Resolve SWE plot selection in canonical execution order."""
        if no_plots and requested:
            raise ValueError("--no-plots and --plot are mutually exclusive")
        if no_plots:
            return ()
        if requested is None:
            if has_snapshots:
                return SWE_PLOT_TYPES
            return tuple(
                plot for plot in SWE_PLOT_TYPES
                if plot not in _SWE_PLOTS_REQUIRING_SNAPSHOTS)
        selected = set()
        for name in requested:
            if name == "all":
                selected.update(SWE_PLOT_TYPES)
            elif name in SWE_PLOT_TYPES:
                selected.add(name)
            else:
                raise ValueError(
                    f"unknown SWE plot type {name!r}; choose from "
                    f"{', '.join(SWE_PLOT_TYPES)} or 'all'")
        return tuple(plot for plot in SWE_PLOT_TYPES if plot in selected)

    # ------------------------------------------------------------------

    def snapshot_times_seconds(self) -> list[float]:
        t_end = self.duration_days * SECONDS_PER_DAY
        if self.snapshot_mode == "count":
            return count_snapshot_times(self.n_snapshots, t_end)
        return interval_snapshot_times(self.dt_snapshots, t_end)

    def scientific_config_dict(self) -> dict:
        return scientific_config_subset(self.to_run_config_dict())

    def to_run_config_dict(self) -> dict:
        """Config dict for make_run_id, config.json, and manifest.json.

        The topography keys are ADDITIVE and emitted only for a non-flat
        bottom (module docstring): a flat run's dict — and therefore its
        scientific hash and run id — is exactly the historical one, while
        resolved terrain parameters participate fully in the scientific
        identity of every non-flat run.
        """
        config = {
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
        if self.topography != "flat":
            config["topography"] = self.topography
            config["mountain_height_m"] = self.mountain_height_m
            config["mountain_lat_deg"] = self.mountain_lat_deg
            config["mountain_lon_deg"] = self.mountain_lon_deg
            config["mountain_width_deg"] = self.mountain_width_deg
        return config

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
        if self.topography == "mountain":
            topo = (f"mountain (h={self.mountain_height_m:g} m at "
                    f"lat {self.mountain_lat_deg:g} deg, "
                    f"lon {self.mountain_lon_deg:g} deg, "
                    f"width {self.mountain_width_deg:g} deg)")
        else:
            topo = "flat"
        lines += [
            f"  l_max               {self.lmax}",
            f"  scenario            {self.scenario}",
            f"  topography          {topo}",
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
