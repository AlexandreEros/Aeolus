"""Resolved dry primitive-equation (pe) run configuration.

The smallest scientifically honest configuration for the first runnable dry
hydrostatic PE experiment. It follows the SWE config as closely as possible
(same snapshot machinery, same resolve/validate/hash contract) and adds the
parameters the primitive-equation core needs:

* a **fixed** integration timestep ``dt_seconds`` (this runner does NOT use
  the adaptive advective-CFL ceiling the BVE/SWE runners do — see
  ``run.pe.runner``);
* the vertical grid, given either as a uniform-sigma ``nlev`` level count or
  as explicit ``sigma_interfaces`` (validated through
  :class:`~planetary_sandbox.physics.sigma_coordinate.SigmaGrid`);
* the configurable dry gas constants ``r_dry`` / ``cp_dry``;
* the initial-condition thermodynamic parameters ``temperature`` /
  ``surface_pressure`` (and, for ``thermal_wave``, ``thermal_amplitude``).

The dry sigma core has no separate reference pressure (sigma = p/p_s), so the
configurable thermodynamic identity is exactly (r_dry, cp_dry) plus the
initial surface pressure. Every one of these PE-specific values participates
in the scientific-configuration hash: ``scientific_config_subset`` drops only
the locational/control keys (``out``, ``experiment``, ``overwrite``,
``plots``), so ``dt_seconds``, ``nlev``, ``sigma_interfaces``, ``r_dry``,
``cp_dry`` and the IC parameters are all part of a run's scientific identity.

Import-light on purpose (stdlib + numpy-only ``SigmaGrid``); parsing,
``--help``, ``list`` and validation never touch CuPy or matplotlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
from ..bve.config import (GRID_TYPES, MIN_NLAT, MIN_NLON,
                          scientific_config_subset)
from ..engine import (SECONDS_PER_DAY, _require_finite_positive,
                      count_snapshot_times, interval_snapshot_times)

#: Image products in deterministic execution order. ``summary`` needs at
#: least one persisted state; ``diagnostics`` remains available for N=0 runs.
PE_PLOT_TYPES = ("diagnostics", "summary")
_PE_PLOTS_REQUIRING_SNAPSHOTS = ("summary",)

#: Dry-air constants mirrored from ``physics.primitive_equations`` (imported
#: literally here, not from that module, so the config stays CuPy-free). The
#: model rejects a run whose r_dry/cp_dry disagree with these unless the user
#: deliberately overrides them; the values are the documented dry defaults.
DEFAULT_R_DRY = 287.04          # J kg^-1 K^-1
DEFAULT_CP_DRY = 1004.64        # J kg^-1 K^-1

#: Default resting thermodynamic state (a mid-tropospheric isothermal value
#: and standard sea-level pressure).
DEFAULT_TEMPERATURE = 260.0     # K
DEFAULT_SURFACE_PRESSURE = 101325.0  # Pa

#: Default thermal-wave perturbation amplitude: the spectral coefficient
#: placed on the degree-2 mode (see run.pe.initial_conditions). ~1 K keeps
#: the perturbed temperature positive everywhere.
DEFAULT_THERMAL_AMPLITUDE = 1.0  # K

#: A deliberately tiny, stable demonstration: a coarse geodesic grid, a short
#: fixed step, and half an hour of simulated time (six RK4 steps).
DEFAULT_NLEV = 8
DEFAULT_DT_SECONDS = 300.0
DEFAULT_DURATION_DAYS = 1800.0 / SECONDS_PER_DAY  # 30 minutes
DEFAULT_N_SNAPSHOTS = 3

_MAX_T_END_SECONDS = 1e12

#: Available initial-condition presets (must match
#: run.pe.initial_conditions.PE_INITIAL_CONDITIONS; kept as a plain mapping
#: here because that module imports CuPy at import time).
PE_SCENARIOS = {
    "isothermal_rest": "Exactly resting, horizontally uniform isothermal "
                       "atmosphere (exercises the exact-rest property).",
    "thermal_wave": "Resting atmosphere with a small deterministic degree-2 "
                    "temperature perturbation (smooth finite response).",
}

#: Presets that require a resolvable degree-2 harmonic.
_SCENARIOS_NEEDING_L2 = ("thermal_wave",)

PE_BASE_DEFAULTS: dict = {
    "lmax": 10,
    "grid": "geodesic",
    "resolution": 3,
    "nlat": 32,
    "nlon": 64,
    "day_hours": 24.0,
    "radius_earth_units": 1.0,
    "nlev": DEFAULT_NLEV,
    "sigma_interfaces": None,
    "r_dry": DEFAULT_R_DRY,
    "cp_dry": DEFAULT_CP_DRY,
    "duration_days": DEFAULT_DURATION_DAYS,
    "dt_seconds": DEFAULT_DT_SECONDS,
    "scenario": "thermal_wave",
    "temperature": DEFAULT_TEMPERATURE,
    "surface_pressure": DEFAULT_SURFACE_PRESSURE,
    "thermal_amplitude": DEFAULT_THERMAL_AMPLITUDE,
    "out": "runs",
    "experiment": None,
    "overwrite": False,
}


@dataclass(frozen=True)
class PERunConfig:
    """Fully resolved configuration for one dry primitive-equation run."""

    lmax: int = 10
    grid: str = "geodesic"
    resolution: int = 3
    nlat: int = 32
    nlon: int = 64
    day_hours: float = 24.0
    radius_earth_units: float = 1.0
    nlev: int = DEFAULT_NLEV
    sigma_interfaces: Optional[tuple[float, ...]] = None
    r_dry: float = DEFAULT_R_DRY
    cp_dry: float = DEFAULT_CP_DRY
    duration_days: float = DEFAULT_DURATION_DAYS
    dt_seconds: float = DEFAULT_DT_SECONDS
    scenario: str = "thermal_wave"
    temperature: float = DEFAULT_TEMPERATURE
    surface_pressure: float = DEFAULT_SURFACE_PRESSURE
    thermal_amplitude: float = DEFAULT_THERMAL_AMPLITUDE
    dt_snapshots: Optional[float] = None
    snapshot_mode: str = "count"
    n_snapshots: Optional[int] = DEFAULT_N_SNAPSHOTS
    plots: tuple[str, ...] = PE_PLOT_TYPES
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

        if self.scenario not in PE_SCENARIOS:
            raise ValueError(
                f"unknown pe scenario {self.scenario!r}; choose from "
                f"{', '.join(sorted(PE_SCENARIOS))}")
        if self.scenario in _SCENARIOS_NEEDING_L2 and self.lmax < 2:
            raise ValueError(
                f"scenario {self.scenario!r} needs lmax >= 2 for its degree-2 "
                f"perturbation, got {self.lmax}")

        # Vertical grid: SigmaGrid validates interfaces (finite, strictly
        # increasing, exact 0/1 endpoints). A resolved interface list is the
        # single source of truth; nlev must agree with it.
        interfaces = self.sigma_interfaces_resolved()
        if self.sigma_interfaces is not None and self.nlev != len(interfaces) - 1:
            raise ValueError(
                f"nlev={self.nlev} disagrees with the {len(interfaces)} "
                f"explicit sigma interfaces ({len(interfaces) - 1} layers)")
        if self.nlev < 1:
            raise ValueError(f"nlev must be >= 1, got {self.nlev}")

        duration = _require_finite_positive("duration_days", self.duration_days)
        t_end = duration * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end <= 0 or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {self.duration_days} overflows t_end "
                f"(got {t_end} s, cap {_MAX_T_END_SECONDS} s)")
        dt = _require_finite_positive("dt_seconds", self.dt_seconds)
        if dt > t_end:
            raise ValueError(
                f"fixed timestep dt_seconds={dt} exceeds the run duration "
                f"({t_end} s); choose dt_seconds <= duration")

        _require_finite_positive("radius_earth_units", self.radius_earth_units)
        _require_finite_positive("temperature", self.temperature)
        _require_finite_positive("surface_pressure", self.surface_pressure)
        _require_finite_positive("r_dry", self.r_dry)
        cp_dry = _require_finite_positive("cp_dry", self.cp_dry)
        if not cp_dry > self.r_dry:
            raise ValueError(
                f"cp_dry must be > r_dry, got cp_dry={cp_dry}, r_dry={self.r_dry}")
        amp = float(self.thermal_amplitude)
        if not (math.isfinite(amp) and amp >= 0.0):
            raise ValueError(
                f"thermal_amplitude must be finite and >= 0, got {amp}")
        if self.day_hours != math.inf:
            _require_finite_positive("day_hours", self.day_hours)

        self._validate_snapshot_controls(t_end)

        unknown_plots = set(self.plots) - set(PE_PLOT_TYPES)
        if unknown_plots:
            raise ValueError(f"unknown plot types: {sorted(unknown_plots)}")
        if list(self.plots) != [p for p in PE_PLOT_TYPES if p in self.plots]:
            raise ValueError("plots must be deduplicated and in canonical order")
        if self.n_snapshots == 0 and any(
                plot in _PE_PLOTS_REQUIRING_SNAPSHOTS for plot in self.plots):
            raise ValueError("PE summary visualization requires a stored state")

    def _validate_snapshot_controls(self, t_end: float) -> None:
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

    # ------------------------------------------------------------------

    def sigma_interfaces_resolved(self) -> tuple[float, ...]:
        """The K+1 interface coordinates, resolved from nlev or explicit list.

        Always validated by :class:`SigmaGrid` (finite, strictly increasing,
        exact 0.0/1.0 endpoints), so an invalid vertical grid is rejected at
        configuration time rather than deep in the model.
        """
        if self.sigma_interfaces is not None:
            return SigmaGrid(tuple(self.sigma_interfaces)).interfaces
        return SigmaGrid.uniform(int(self.nlev)).interfaces

    @classmethod
    def resolve(cls, explicit: Mapping) -> "PERunConfig":
        """Layer explicit (user-supplied) values over the ordinary defaults."""
        explicit = {k: v for k, v in dict(explicit).items() if v is not None}

        allowed = set(PE_BASE_DEFAULTS) | {
            "n_snapshots", "dt_snapshots", "plots", "no_plots"}
        unknown = set(explicit) - allowed
        if unknown:
            raise ValueError(f"unknown explicit settings: {sorted(unknown)}")

        settings = dict(PE_BASE_DEFAULTS)
        settings.update({k: v for k, v in explicit.items()
                         if k in PE_BASE_DEFAULTS})
        if settings["grid"] == "gauss-latlon":  # user-facing alias
            settings["grid"] = "latlon"

        # Explicit sigma interfaces fix the level count; keep nlev consistent.
        if settings["sigma_interfaces"] is not None:
            settings["sigma_interfaces"] = tuple(
                float(s) for s in settings["sigma_interfaces"])
            settings["nlev"] = len(settings["sigma_interfaces"]) - 1

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
        if no_plots and requested:
            raise ValueError("--no-plots and --plot are mutually exclusive")
        if no_plots:
            return ()
        if requested is None:
            if has_snapshots:
                return PE_PLOT_TYPES
            return tuple(
                plot for plot in PE_PLOT_TYPES
                if plot not in _PE_PLOTS_REQUIRING_SNAPSHOTS)
        selected = set()
        for name in requested:
            if name == "all":
                selected.update(PE_PLOT_TYPES)
            elif name in PE_PLOT_TYPES:
                selected.add(name)
            else:
                raise ValueError(
                    f"unknown PE plot type {name!r}; choose from "
                    f"{', '.join(PE_PLOT_TYPES)} or 'all'")
        return tuple(plot for plot in PE_PLOT_TYPES if plot in selected)

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
            "solver": "pe",
            "lmax": self.lmax,
            "grid": self.grid,
            "resolution": self.resolution,
            "nlat": self.nlat,
            "nlon": self.nlon,
            "day_hours": self.day_hours,
            "radius_earth_units": self.radius_earth_units,
            "nlev": self.nlev,
            "sigma_interfaces": list(self.sigma_interfaces_resolved()),
            "r_dry": self.r_dry,
            "cp_dry": self.cp_dry,
            "duration_days": self.duration_days,
            "dt_seconds": self.dt_seconds,
            "scenario": self.scenario,
            "temperature": self.temperature,
            "surface_pressure": self.surface_pressure,
            "thermal_amplitude": self.thermal_amplitude,
            # Frozen for the PE core: nonlinear products always use the
            # backend's fine (overresolved / 3/2-rule) sampling.
            "product_quadrature": "fine",
            "dt_snapshots": self.dt_snapshots,
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
        interfaces = self.sigma_interfaces_resolved()
        t_end = self.duration_days * SECONDS_PER_DAY
        n_steps = math.ceil(t_end / self.dt_seconds)
        lines = ["Resolved run configuration:", "  solver              pe"]
        lines.append(f"  backend/grid        {self.grid}")
        if self.grid == "geodesic":
            lines.append(f"  resolution          {self.resolution} "
                         "(geodesic subdivision level)")
        else:
            lines.append(f"  nlat x nlon         {self.nlat} x {self.nlon} "
                         "(Gauss-Legendre latitudes x uniform longitudes)")
        lines += [
            f"  l_max               {self.lmax}",
            f"  sigma levels        {self.nlev} "
            f"(interfaces {', '.join(f'{s:g}' for s in interfaces)})",
            f"  scenario            {self.scenario}",
            f"  day length          {day}",
            f"  radius              {self.radius_earth_units:g} Earth radii",
            f"  R_d / c_p           {self.r_dry:g} / {self.cp_dry:g} J/kg/K",
            f"  initial T           {self.temperature:g} K",
            f"  initial p_s         {self.surface_pressure:g} Pa",
        ]
        if self.scenario == "thermal_wave":
            lines.append(f"  thermal amplitude   {self.thermal_amplitude:g} K "
                         "(degree-2 mode)")
        lines += [
            f"  fixed timestep      {self.dt_seconds:g} s "
            f"({n_steps} steps over {self.duration_days:g} days)",
            f"  snapshots           {schedule}",
            f"  plots               {', '.join(self.plots) if self.plots else 'none'}",
            f"  output base         {out}",
        ]
        return lines
