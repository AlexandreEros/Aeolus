"""Resolved BVE run configuration.

This module owns configuration *resolution*: layering explicit CLI values
over preset values over ordinary defaults, resolving the snapshot schedule
and plot selection, validating cross-field constraints, and emitting the
config dict consumed by the runner and the provenance system.

It is independent of argparse (it consumes plain dicts of already-parsed
values) and import-light on purpose (stdlib only), so help/list/inspect
commands and configuration validation never touch CuPy or matplotlib.

Snapshot semantics
------------------
Two mutually exclusive controls, both parsed as ``None`` by the CLI:

``n_snapshots`` (count mode, canonical):
    N = 0   store no field snapshots
    N = 1   store only the final state at t_end
    N >= 2  store exactly N evenly spaced states including t=0 and t_end

``dt_snapshots`` (interval mode, legacy):
    store t=0 and every interval boundary up to t_end; the final state is
    stored only when the duration is a multiple of the interval (historical
    psx-bve behavior, preserved).

When neither is supplied, the default depends on the calling interface:
``aeolus run bve`` uses N=5 (which reproduces the historical 0/6/12/18/24 h
states for the default one-day run); legacy ``psx-bve`` keeps the historical
21600 s interval.

Config-dict schema
------------------
``to_run_config_dict()`` emits the historical psx-bve key set unchanged,
plus four additive keys::

    snapshot_mode    "count" | "interval"
    n_snapshots      requested count (None in interval mode)
    snapshot_times   resolved schedule in seconds (authoritative)
    plots            resolved plot products, deterministic order

``dt_snapshots`` stays the interval in seconds for interval mode and for
count mode with N >= 2 (the uniform spacing); it is None for N in {0, 1}
rather than inventing a fake interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

SECONDS_PER_DAY = 86400.0

#: Historical psx-bve snapshot cadence (seconds); the interval-mode default.
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 21600.0

#: Canonical count-mode default (aeolus run bve): five stored states,
#: reproducing 0/6/12/18/24 h for the default one-day run.
DEFAULT_N_SNAPSHOTS = 5

GRID_TYPES = ("geodesic", "latlon")
PRODUCT_QUADRATURES = ("fine", "coarse")

#: Minimum lat-lon dimensions accepted by the CLI. The transforms need at
#: least a two-point latitude sample and a longitude wraparound; smaller
#: grids trip assertions deep in the numerics without a useful message.
MIN_NLAT = 2
MIN_NLON = 4

#: All currently implemented plot products, in the fixed deterministic
#: execution order. 'diagnostics' renders figures/ from the diagnostics CSV
#: (works with any snapshot count); 'snapshots' renders the per-snapshot
#: panel figure and 'summary' renders bve_summary.png (both need >= 1
#: stored state).
PLOT_TYPES = ("diagnostics", "snapshots", "summary")
_PLOTS_REQUIRING_SNAPSHOTS = ("snapshots", "summary")

#: Ordinary defaults for run-bve settings, identical to the historical
#: psx-bve argparse defaults. Snapshot and plot controls are deliberately
#: absent: their defaults depend on the calling interface / schedule.
BASE_DEFAULTS: dict = {
    "lmax": 21,
    "grid": "geodesic",
    "resolution": 4,
    "nlat": 128,
    "nlon": 256,
    "day_hours": math.inf,
    "radius_earth_units": 1.0,
    "duration_days": 1.0,
    "scenario": "two_vortices",
    "viscosity": 0.0,
    "product_quadrature": "fine",
    "out": "runs",
    "experiment": None,
    "overwrite": False,
}

#: Keys a preset may set on top of BASE_DEFAULTS. Plot selection is a
#: per-invocation user choice (`--plot` / `--no-plots`) and never comes
#: from a preset; only these snapshot controls are additionally allowed.
_PRESET_ONLY_KEYS = frozenset({"n_snapshots", "dt_snapshots"})
_EXPLICIT_ONLY_KEYS = _PRESET_ONLY_KEYS | frozenset({"plots", "no_plots"})

#: Cap on t_end (seconds). A duration this large trips scale-aware
#: tolerances and downstream time bookkeeping; it is far above any
#: physically motivated Aeolus run and rejects overflow explicitly.
_MAX_T_END_SECONDS = 1e12  # ~31,700 years


def _require_finite_number(name: str, value) -> float:
    """Reject NaN, +/-inf, and non-numeric values with a clear message."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {value}")
    return f


def _require_finite_positive(name: str, value) -> float:
    f = _require_finite_number(name, value)
    if not f > 0:
        raise ValueError(f"{name} must be > 0, got {f}")
    return f


def _require_finite_nonneg(name: str, value) -> float:
    f = _require_finite_number(name, value)
    if f < 0:
        raise ValueError(f"{name} must be >= 0, got {f}")
    return f


def interval_snapshot_times(dt_snapshots: float, t_end: float) -> list[float]:
    """Legacy interval-mode schedule: t=0 and every boundary up to t_end.

    Mirrors the historical runner countdown: the final state appears only
    when the duration is a multiple of the interval (within the historical
    ``1e-6 * dt`` tolerance).
    """
    tol = 1e-6 * dt_snapshots
    times: list[float] = []
    k = 0
    while k * dt_snapshots <= t_end + tol:
        times.append(min(k * dt_snapshots, t_end))
        k += 1
    return times


def count_snapshot_times(n_snapshots: int, t_end: float) -> list[float]:
    """Count-mode schedule: N evenly spaced states including both endpoints.

    N=0 -> []; N=1 -> [t_end]; N>=2 -> [0, ..., t_end] with exact endpoints.
    """
    if n_snapshots == 0:
        return []
    if n_snapshots == 1:
        return [t_end]
    spacing = t_end / (n_snapshots - 1)
    times = [i * spacing for i in range(n_snapshots)]
    times[-1] = t_end  # exact, no accumulated float error
    return times


def scheduler_tolerance(t_end: float, times: Sequence[float]) -> float:
    """Scale/gap-aware tolerance for matching schedule entries at runtime.

    Small enough that two distinct entries are never coalesced (a fraction
    of the smallest positive inter-entry gap), and small enough that a
    short simulation is not entirely consumed (a fraction of t_end), while
    staying well above floating-point noise (>= 1e-12 s).
    """
    tol = max(1e-12, 1e-9 * t_end) if t_end > 0 else 1e-12
    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if gaps:
            tol = min(tol, 0.25 * min(gaps))
    return tol


def validate_snapshot_schedule(times: Sequence[float], t_end: float) -> list[float]:
    """Return a clean schedule; raise ValueError on any anomaly.

    Entries must be finite, strictly increasing, non-duplicated, and lie in
    [0, t_end] (allowing a scale-aware slack against float noise on the
    endpoints; interior duplicates are always rejected).
    """
    if t_end <= 0 or not math.isfinite(t_end):
        raise ValueError(f"t_end must be finite and positive, got {t_end}")
    cleaned: list[float] = []
    slack = max(1e-9, 1e-9 * t_end)
    for i, t in enumerate(times):
        if not math.isfinite(t):
            raise ValueError(
                f"snapshot_times[{i}] = {t} is not finite")
        if t < -slack or t > t_end + slack:
            raise ValueError(
                f"snapshot_times[{i}] = {t} is outside [0, {t_end}]")
        clipped = max(0.0, min(t_end, float(t)))
        if cleaned and not clipped > cleaned[-1]:
            raise ValueError(
                "snapshot_times must be strictly increasing without "
                f"duplicates; got {times!r}")
        cleaned.append(clipped)
    return cleaned


@dataclass(frozen=True)
class BVERunConfig:
    """Fully resolved configuration for one BVE run.

    Build instances with :meth:`resolve`, which applies the
    explicit > preset > ordinary-default precedence. The field names and
    the ``to_run_config_dict()`` legacy key set are frozen: they feed
    ``make_run_id`` (io.py) and the on-disk ``config.json`` schema.
    """

    lmax: int = 21
    grid: str = "geodesic"
    resolution: int = 4
    nlat: int = 128
    nlon: int = 256
    day_hours: float = math.inf
    radius_earth_units: float = 1.0
    duration_days: float = 1.0
    dt_snapshots: Optional[float] = DEFAULT_SNAPSHOT_INTERVAL_SECONDS
    scenario: str = "two_vortices"
    viscosity: float = 0.0
    product_quadrature: str = "fine"
    out: str = "runs"
    experiment: Optional[str] = None
    overwrite: bool = False
    snapshot_mode: str = "interval"
    n_snapshots: Optional[int] = None
    plots: tuple[str, ...] = PLOT_TYPES

    def __post_init__(self) -> None:
        if self.grid not in GRID_TYPES:
            raise ValueError(f"grid must be one of {GRID_TYPES}, got {self.grid!r}")
        if self.product_quadrature not in PRODUCT_QUADRATURES:
            raise ValueError(
                f"product_quadrature must be one of {PRODUCT_QUADRATURES}, "
                f"got {self.product_quadrature!r}")
        if self.lmax < 1:
            raise ValueError(f"lmax must be >= 1, got {self.lmax}")
        if self.resolution < 0:
            raise ValueError(f"resolution must be >= 0, got {self.resolution}")

        # Lat-lon dimensions matter only when the lat-lon backend is used,
        # but the values live in the config either way; enforce the
        # backend-relevant floor to reject transforms that would crash
        # deep inside the numerics with an unhelpful assertion.
        if self.grid == "latlon":
            if self.nlat < MIN_NLAT:
                raise ValueError(
                    f"lat-lon backend requires nlat >= {MIN_NLAT}, got {self.nlat}")
            if self.nlon < MIN_NLON:
                raise ValueError(
                    f"lat-lon backend requires nlon >= {MIN_NLON}, got {self.nlon}")
        elif self.nlat < 1 or self.nlon < 1:
            # Sanity check even for backends that don't consume these,
            # since they still land in config.json.
            raise ValueError(f"nlat/nlon must be positive, got {self.nlat}x{self.nlon}")

        duration = _require_finite_positive("duration_days", self.duration_days)
        t_end = duration * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end <= 0 or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {self.duration_days} overflows t_end "
                f"(got {t_end} s, cap {_MAX_T_END_SECONDS} s)")

        _require_finite_positive("radius_earth_units", self.radius_earth_units)
        _require_finite_nonneg("viscosity", self.viscosity)

        # day_hours == +inf is the sentinel for the non-rotating mode
        # (f_lm == 0); every other non-finite or non-positive value is a
        # user error and shouldn't silently coast into the solver.
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

        unknown_plots = set(self.plots) - set(PLOT_TYPES)
        if unknown_plots:
            raise ValueError(f"unknown plot types: {sorted(unknown_plots)}")
        if list(self.plots) != [p for p in PLOT_TYPES if p in self.plots]:
            raise ValueError("plots must be deduplicated and in canonical order")
        needing = [p for p in self.plots if p in _PLOTS_REQUIRING_SNAPSHOTS]
        if needing and not self.snapshot_times_seconds():
            raise ValueError(
                f"plot type(s) {needing} require at least one stored snapshot; "
                "increase --n-snapshots or drop the plot")

    # ------------------------------------------------------------------
    # Resolution (explicit > preset > ordinary default)
    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls,
                explicit: Mapping,
                preset: Optional[Mapping] = None,
                snapshot_default: str = "count") -> "BVERunConfig":
        """Layer explicit values over a preset over ordinary defaults.

        ``explicit`` must contain only values the user actually supplied
        (the CLI parses every option with a None default and filters).
        ``snapshot_default`` selects the interface default when neither
        snapshot control is supplied: "count" (aeolus, N=5) or "interval"
        (legacy psx-bve, 21600 s).
        """
        explicit = {k: v for k, v in dict(explicit).items() if v is not None}
        preset = dict(preset) if preset else {}

        allowed_explicit = set(BASE_DEFAULTS) | _EXPLICIT_ONLY_KEYS
        allowed_preset = set(BASE_DEFAULTS) | _PRESET_ONLY_KEYS
        unknown = set(explicit) - allowed_explicit
        if unknown:
            raise ValueError(f"unknown explicit settings: {sorted(unknown)}")
        unknown = set(preset) - allowed_preset
        if unknown:
            raise ValueError(f"unknown preset settings: {sorted(unknown)}")

        # The two snapshot controls are one mutually exclusive choice: an
        # explicit flag replaces whichever form the preset used.
        if "n_snapshots" in explicit or "dt_snapshots" in explicit:
            preset.pop("n_snapshots", None)
            preset.pop("dt_snapshots", None)

        settings = dict(BASE_DEFAULTS)
        settings.update({k: v for k, v in preset.items()
                         if k in BASE_DEFAULTS})
        settings.update({k: v for k, v in explicit.items()
                         if k in BASE_DEFAULTS})

        if settings["grid"] == "gauss-latlon":  # user-facing alias
            settings["grid"] = "latlon"

        merged = dict(preset)
        merged.update(explicit)
        n = merged.get("n_snapshots")
        interval = merged.get("dt_snapshots")

        # Basic domain validation on values that participate in schedule
        # arithmetic so ordinary user errors fail here, before the full
        # dataclass constructor is called with a nonsensical dt.
        duration_days = _require_finite_positive(
            "duration_days", settings["duration_days"])
        t_end = duration_days * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {duration_days} overflows t_end (got {t_end} s)")

        if n is not None and interval is not None:
            raise ValueError(
                "snapshot count and snapshot interval are mutually exclusive; "
                "provide at most one")
        if n is None and interval is None:
            if snapshot_default == "count":
                n = DEFAULT_N_SNAPSHOTS
            elif snapshot_default == "interval":
                interval = DEFAULT_SNAPSHOT_INTERVAL_SECONDS
            else:
                raise ValueError(
                    f"unknown snapshot_default: {snapshot_default!r}")

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

        plots = cls._resolve_plots(
            explicit.get("plots"), explicit.get("no_plots"),
            has_snapshots=bool(
                count_snapshot_times(n, t_end) if snapshot_mode == "count"
                else interval_snapshot_times(dt, t_end)))

        return cls(dt_snapshots=dt, snapshot_mode=snapshot_mode,
                   n_snapshots=n, plots=plots, **settings)

    @staticmethod
    def _resolve_plots(requested, no_plots, *, has_snapshots: bool) -> tuple[str, ...]:
        """Resolve --plot/--no-plots into a deterministic plot tuple.

        No selection: current default behavior — every product the schedule
        supports. Explicit selection: exactly that set (deduplicated, in
        canonical order); incompatibility with the schedule is an error,
        raised later by __post_init__.
        """
        if no_plots and requested:
            raise ValueError("--no-plots and --plot are mutually exclusive")
        if no_plots:
            return ()
        if requested is None:
            if has_snapshots:
                return PLOT_TYPES
            return tuple(p for p in PLOT_TYPES
                         if p not in _PLOTS_REQUIRING_SNAPSHOTS)
        selected = set()
        for name in requested:
            if name == "all":
                selected.update(PLOT_TYPES)
            elif name in PLOT_TYPES:
                selected.add(name)
            else:
                raise ValueError(
                    f"unknown plot type {name!r}; choose from "
                    f"{', '.join(PLOT_TYPES)} or 'all'")
        return tuple(p for p in PLOT_TYPES if p in selected)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def snapshot_times_seconds(self) -> list[float]:
        """The authoritative snapshot schedule in seconds."""
        t_end = self.duration_days * SECONDS_PER_DAY
        if self.snapshot_mode == "count":
            return count_snapshot_times(self.n_snapshots, t_end)
        return interval_snapshot_times(self.dt_snapshots, t_end)

    @property
    def includes_final_state(self) -> bool:
        times = self.snapshot_times_seconds()
        t_end = self.duration_days * SECONDS_PER_DAY
        tol = scheduler_tolerance(t_end, times)
        return bool(times) and abs(times[-1] - t_end) <= tol

    def scientific_config_dict(self) -> dict:
        """Config subset used to derive the run-id hash.

        Deliberately excludes purely locational/control values (out,
        experiment, overwrite) so that the same scientific configuration
        gets the same hash regardless of where it is written or whether
        it is being re-run. Plot selection also stays out — figures are
        derived artifacts, not part of the scientific state.
        """
        d = self.to_run_config_dict()
        for key in ("out", "experiment", "overwrite", "plots"):
            d.pop(key, None)
        return d

    def to_run_config_dict(self) -> dict:
        """Config dict for make_run_id, config.json, and manifest.json.

        The historical psx-bve key set is preserved unchanged; the four
        trailing keys (snapshot_mode, n_snapshots, snapshot_times, plots)
        are additive (see module docstring).
        """
        return {
            "lmax": self.lmax,
            "grid": self.grid,
            "resolution": self.resolution,
            "nlat": self.nlat,
            "nlon": self.nlon,
            "day_hours": self.day_hours,
            "radius_earth_units": self.radius_earth_units,
            "duration_days": self.duration_days,
            "dt_snapshots": self.dt_snapshots,
            "scenario": self.scenario,
            "viscosity": self.viscosity,
            "product_quadrature": self.product_quadrature,
            "out": self.out,
            "experiment": self.experiment,
            "overwrite": self.overwrite,
            "snapshot_mode": self.snapshot_mode,
            "n_snapshots": self.n_snapshots,
            "snapshot_times": self.snapshot_times_seconds(),
            "plots": list(self.plots),
        }

    def summary_lines(self, preset: Optional[str] = None) -> list[str]:
        """Concise resolved-configuration summary (no CUDA involved)."""
        times = self.snapshot_times_seconds()
        lines = ["Resolved run configuration:", "  solver              bve"]
        if preset:
            lines.append(f"  preset              {preset}")
        lines.append(f"  backend/grid        {self.grid}")
        if self.grid == "geodesic":
            lines.append(f"  resolution          {self.resolution} "
                         "(geodesic subdivision level)")
        else:
            lines.append(f"  nlat x nlon         {self.nlat} x {self.nlon} "
                         "(Gauss-Legendre latitudes x uniform longitudes)")
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
        lines += [
            f"  l_max               {self.lmax}",
            f"  scenario            {self.scenario}",
            f"  day length          {day}",
            f"  radius              {self.radius_earth_units:g} Earth radii",
            f"  duration            {self.duration_days:g} days",
            f"  snapshots           {schedule}",
            f"  plots               {', '.join(self.plots) if self.plots else 'none'}",
            f"  viscosity           {self.viscosity:g} m^2/s",
            f"  product quadrature  {self.product_quadrature}",
            f"  output base         {out}",
        ]
        if self.snapshot_mode == "interval" and not self.includes_final_state:
            lines.append(
                "  note: the duration is not a multiple of the snapshot "
                "interval, so the final state will not be stored (use "
                "--n-snapshots to include it).")
        return lines
