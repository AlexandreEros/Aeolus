"""aeolus — the public command-line interface for Aeolus.

Command tree::

    aeolus run bve [...]        run the barotropic vorticity solver
    aeolus run swe [...]        run the rotating shallow-water solver
    aeolus run pe  [...]        run the dry primitive-equation solver
    aeolus list presets         named run configurations
    aeolus list scenarios       initial-condition scenarios
    aeolus inspect RUN_PATH     summarize a finished run from its manifest
    aeolus gen [...]            demo planet + summary plot (psx-gen)
    aeolus recompile [...]      clear/verify the CuPy kernel cache (psx-recompile)

``aeolus validate`` and ``aeolus list planets`` are reserved: there is no
automated validation scorer and no planet catalog yet, so they are neither
implemented nor advertised.

The ``psx-bve`` / ``psx-gen`` / ``psx-recompile`` commands are kept as
compatibility entry points and delegate to the same implementations.

Design rules for this module:

- Import-light: parsing, ``--help``, ``list``, and ``inspect`` must never
  import CuPy, matplotlib, Planet, the runner, or visualization modules.
  Heavy imports happen inside command handlers, after validation.
- Every run-bve option parses with a ``None`` default so that explicit
  values are distinguishable from defaults; the documented defaults live in
  ``run.bve.config.BASE_DEFAULTS`` and are applied during resolution, which
  enforces the precedence: explicit flag > preset value > ordinary default.
- Resolution, cross-field validation, snapshot schedules, and plot
  selection live in ``planetary_sandbox.run.bve.config`` (BVERunConfig);
  this module owns only parsing, aliases, and the preset registry.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from planetary_sandbox.cli import clear_cache, generate_planet
from planetary_sandbox.run.bve.config import (  # import-light (stdlib only)
    BASE_DEFAULTS, DEFAULT_SNAPSHOT_INTERVAL_SECONDS, PLOT_TYPES, BVERunConfig)

# ---------------------------------------------------------------------------
# Choices and presets
# ---------------------------------------------------------------------------

#: Initial-condition scenarios. Must match INITIAL_CONDITIONS in
#: run/bve/initial_conditions.py (kept as a plain mapping here because that
#: module imports CuPy at import time; parity is enforced by a test).
SCENARIOS = {
    "two_vortices": "Two opposite-signed Gaussian vortices at +/-33 deg latitude.",
    "inverted_vortices": "Sign-flipped variant of two_vortices.",
    "polar_vortices": "Opposite-signed Gaussian vortices at the poles.",
    "inverted_polar_vortices": "Sign-flipped variant of polar_vortices.",
    "equatorial_vortices": "Opposite-signed Gaussian vortices on the equator.",
    "inverted_equatorial_vortices": "Sign-flipped variant of equatorial_vortices.",
    "random_low_l": "Random low-degree (l <= 10) spectral vorticity field.",
    "rh4": "Rossby-Haurwitz wavenumber-4 traveling wave (validation case).",
}

#: User-facing backend spellings; 'gauss-latlon' normalizes to 'latlon'.
BACKEND_CHOICES = ("geodesic", "latlon", "gauss-latlon")

#: Named bundles of run-bve settings, corresponding to configurations
#: already documented in the README / docs/VALIDATION.md. Keys are argparse
#: dest names; explicit user flags always override preset values.
PRESETS = {
    "rh4": {
        "description": "One-day Rossby-Haurwitz wavenumber-4 validation run "
                       "(the docs/VALIDATION.md configuration).",
        "settings": {
            "scenario": "rh4",
            "lmax": 21,
            "resolution": 4,
            "day_hours": 24.0,
            "duration_days": 1.0,
            "dt_snapshots": 21600.0,
            "product_quadrature": "fine",
            "viscosity": 0.0,
            "experiment": "validation-rh4",
        },
    },
    "two-vortices-quick": {
        "description": "Small, fast two-vortex smoke run "
                       "(the README quickstart configuration).",
        "settings": {
            "scenario": "two_vortices",
            "lmax": 8,
            "resolution": 3,
            "nlat": 12,
            "nlon": 24,
            "duration_days": 0.02,
            "dt_snapshots": 864.0,
            "experiment": "quickstart",
        },
    },
}


# ---------------------------------------------------------------------------
# run bve: arguments and dispatch
# ---------------------------------------------------------------------------

_BVE_EXAMPLES = """\
examples:
  aeolus run bve                          geodesic grid, two_vortices, 1 day, 5 snapshots
  aeolus run bve --preset rh4             documented RH4 validation configuration
  aeolus run bve --days 1 --n-snapshots 9
  aeolus run bve --n-snapshots 20 --no-plots
  aeolus run bve --n-snapshots 1 --plot summary
  aeolus run bve --backend gauss-latlon --nlat 12 --nlon 24 --l-max 8 --days 0.02
"""


def add_bve_arguments(parser: argparse.ArgumentParser) -> None:
    """All `run bve` options. Every default is None (see module docstring)."""
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="Named bundle of settings (see 'aeolus list presets'). "
             "Explicit options override the preset.")
    parser.add_argument(
        "--backend", "--grid", dest="grid", choices=list(BACKEND_CHOICES),
        default=None,
        help="Numerical backend / grid family [default: geodesic]. "
             "'gauss-latlon' is an alias for 'latlon' (Gauss-Legendre "
             "latitudes, uniform longitudes). 'geodesic' uses --resolution; "
             "'latlon' uses --nlat/--nlon.")
    parser.add_argument(
        "--l-max", "--lmax", dest="lmax", type=int, default=None,
        help="Maximum spherical harmonic degree [default: 21].")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Geodesic grid subdivision level [default: 4]. "
             "The (resolution=4, l_max=21) default keeps ~10 grid points per "
             "SH basis function (docs/KNOWN_RISKS.md R-2).")
    parser.add_argument(
        "--nlat", type=int, default=None,
        help="Lat-lon backend: number of Gauss-Legendre latitudes [default: 128].")
    parser.add_argument(
        "--nlon", type=int, default=None,
        help="Lat-lon backend: number of uniform longitudes [default: 256].")
    parser.add_argument(
        "--day-hours", type=float, default=None,
        help="Sidereal day length in hours; 'inf' = non-rotating [default: inf].")
    parser.add_argument(
        "--radius-earth-units", type=float, default=None,
        help="Planet radius in Earth radii [default: 1.0].")
    parser.add_argument(
        "--days", "--duration-days", dest="duration_days", type=float,
        default=None,
        help="Simulated duration in days [default: 1.0].")

    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument(
        "--n-snapshots", type=int, metavar="N", default=None,
        help="Store N field states evenly spaced over the duration "
             "[default: 5]. N=0: none; N=1: only the final state; N>=2: "
             "exactly N states including both t=0 and t_end. Mutually "
             "exclusive with --snapshot-interval-seconds.")
    snapshots.add_argument(
        "--snapshot-interval-seconds", "--dt-snapshots",
        dest="dt_snapshots", type=float, metavar="SECONDS", default=None,
        help="Store a state every SECONDS of simulated time instead of a "
             "count. The initial state is always stored; the final state is "
             "stored only if the duration is a multiple of the interval. "
             "--dt-snapshots is a compatibility alias (and the legacy "
             "psx-bve default: 21600 s).")

    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default=None,
        help="Initial-condition scenario (see 'aeolus list scenarios') "
             "[default: two_vortices].")
    parser.add_argument(
        "--viscosity", type=float, default=None,
        help="Kinematic viscosity in m^2/s [default: 0.0].")
    parser.add_argument(
        "--product-quadrature", choices=["fine", "coarse"], default=None,
        help="Where nonlinear (pseudospectral) products are evaluated. "
             "'fine' [default]: on a reusable resolution-(r+1) product grid "
             "('overresolved product quadrature', docs/KNOWN_RISKS.md R-3). "
             "'coarse': the historical state-grid path, kept for A/B "
             "comparisons. An unsupported combination raises at startup.")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plots", action="append", metavar="TYPE",
        choices=list(PLOT_TYPES) + ["all"], default=None,
        help="Generate only the named image product; repeatable "
             f"({', '.join(PLOT_TYPES)}, or 'all'). Duplicates are ignored "
             "and execution order is fixed. Without any plot option, every "
             "product the snapshot schedule supports is generated. Field "
             "snapshots and numerical diagnostics are always written "
             "regardless of plot selection.")
    plot_group.add_argument(
        "--no-plots", action="store_true", default=None,
        help="Generate no image files (field snapshots and numerical "
             "diagnostics are still written).")

    parser.add_argument(
        "--out", type=str, default=None,
        help="Base directory for run outputs [default: runs]. Each run "
             "creates a unique subdirectory under this (or under "
             "<out>/<experiment>/ if --experiment is given).")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Optional grouping name; runs go to <out>/<experiment>/<run_id>/.")
    parser.add_argument(
        "--overwrite", action="store_true", default=None,
        help="Reuse an existing run directory if the auto-generated run ID "
             "collides (same command in the same second). Off by default to "
             "keep runs immutable.")


def build_bve_parser(prog: str = "aeolus run bve",
                     apply_defaults: bool = False) -> argparse.ArgumentParser:
    """Standalone `run bve` parser.

    ``apply_defaults=True`` fills in the ordinary defaults directly on the
    parser (kept for the legacy ``planetary_sandbox.cli.bve.build_parser``
    import surface). To match the historical psx-bve parser surface exactly,
    it also applies the legacy ``dt_snapshots`` default (21600 s) so that
    ``build_parser().parse_args([]).dt_snapshots == 21600.0``; the plot
    controls stay None. The canonical aeolus parser (``apply_defaults=False``)
    leaves *all* snapshot controls None so count mode (N=5) remains the
    resolved default — defaults are otherwise applied in config resolution.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the barotropic vorticity equation on a planet.",
        epilog=_BVE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bve_arguments(parser)
    if apply_defaults:
        parser.set_defaults(**BASE_DEFAULTS,
                            dt_snapshots=DEFAULT_SNAPSHOT_INTERVAL_SECONDS)
    return parser


#: Parsed-namespace keys forwarded into configuration resolution.
_EXPLICIT_KEYS = tuple(BASE_DEFAULTS) + (
    "n_snapshots", "dt_snapshots", "plots", "no_plots")


def run_bve_command(args: argparse.Namespace,
                    parser: argparse.ArgumentParser,
                    snapshot_default: str = "count") -> int:
    """Resolve, print the resolved configuration, and dispatch the run.

    ``snapshot_default`` selects the interface default when neither
    snapshot control is given: "count" (aeolus, N=5) or "interval"
    (legacy psx-bve, 21600 s).
    """
    preset_name = getattr(args, "preset", None)
    explicit = {k: getattr(args, k, None) for k in _EXPLICIT_KEYS}
    try:
        cfg = BVERunConfig.resolve(
            explicit=explicit,
            preset=PRESETS[preset_name]["settings"] if preset_name else None,
            snapshot_default=snapshot_default)
    except ValueError as err:
        parser.error(str(err))
    print("\n".join(cfg.summary_lines(preset=preset_name)))
    # Heavy imports (CuPy, matplotlib) happen inside execute_run.
    from planetary_sandbox.cli import bve as bve_module
    return bve_module.execute_run(cfg)


def _cmd_run_bve(args: argparse.Namespace) -> int:
    return run_bve_command(args, args._parser, snapshot_default="count")


# ---------------------------------------------------------------------------
# run swe: arguments and dispatch
# ---------------------------------------------------------------------------

_SWE_EXAMPLES = """\
examples:
  aeolus run swe                          Williamson-2 steady flow, 1 day, geodesic grid
  aeolus run swe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15
  aeolus run swe --scenario gravity_wave --day-hours inf --mean-depth 1000
  aeolus run swe --days 5 --n-snapshots 11 --no-plots
  aeolus run swe --topography mountain --mountain-height-m 2000 --days 2
"""


def add_swe_arguments(parser: argparse.ArgumentParser) -> None:
    """All `run swe` options. Every default is None (resolution applies them)."""
    from planetary_sandbox.run.swe.config import (  # import-light
        SWE_PLOT_TYPES, SWE_SCENARIOS, SWE_TOPOGRAPHIES)

    parser.add_argument(
        "--backend", "--grid", dest="grid", choices=list(BACKEND_CHOICES),
        default=None,
        help="Numerical backend / grid family [default: geodesic]. "
             "'gauss-latlon' is an alias for 'latlon'.")
    parser.add_argument(
        "--l-max", "--lmax", dest="lmax", type=int, default=None,
        help="Maximum spherical harmonic degree [default: 21].")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Geodesic grid subdivision level [default: 4].")
    parser.add_argument(
        "--nlat", type=int, default=None,
        help="Lat-lon backend: number of Gauss-Legendre latitudes [default: 128].")
    parser.add_argument(
        "--nlon", type=int, default=None,
        help="Lat-lon backend: number of uniform longitudes [default: 256].")
    parser.add_argument(
        "--day-hours", type=float, default=None,
        help="Sidereal day length in hours; 'inf' = non-rotating "
             "[default: 23.9345, i.e. Earth's rotation rate].")
    parser.add_argument(
        "--radius-earth-units", type=float, default=None,
        help="Planet radius in Earth radii [default: 1.0].")
    parser.add_argument(
        "--gravity", type=float, default=None,
        help="Surface gravity in m/s^2 [default: 9.80616].")
    parser.add_argument(
        "--mean-depth", dest="mean_depth_m", type=float, default=None,
        help="Mean (resting) fluid depth H in meters [default: 3000]. "
             "The resting geopotential is Phi0 = gravity * H.")
    parser.add_argument(
        "--days", "--duration-days", dest="duration_days", type=float,
        default=None,
        help="Simulated duration in days [default: 1.0].")

    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument(
        "--n-snapshots", type=int, metavar="N", default=None,
        help="Store N spectral states evenly spaced over the duration "
             "[default: 5]. Same semantics as run bve.")
    snapshots.add_argument(
        "--snapshot-interval-seconds", "--dt-snapshots",
        dest="dt_snapshots", type=float, metavar="SECONDS", default=None,
        help="Store a state every SECONDS of simulated time instead of a count.")

    parser.add_argument(
        "--scenario", choices=sorted(SWE_SCENARIOS), default=None,
        help="Initial-condition scenario [default: williamson2].")
    parser.add_argument(
        "--topography", choices=sorted(SWE_TOPOGRAPHIES), default=None,
        help="Fixed bottom topography [default: flat]. 'mountain' is one "
             "smooth Gaussian mountain, band-limited at the model "
             "truncation; its resolved parameters participate in the "
             "scientific run identity.")
    parser.add_argument(
        "--mountain-height-m", dest="mountain_height_m", type=float,
        metavar="METERS", default=None,
        help="Mountain peak elevation in meters [default: 2000]. "
             "Requires --topography mountain.")
    parser.add_argument(
        "--mountain-lat-deg", dest="mountain_lat_deg", type=float,
        metavar="DEG", default=None,
        help="Mountain center latitude in degrees [-90, 90] [default: 30]. "
             "Requires --topography mountain.")
    parser.add_argument(
        "--mountain-lon-deg", dest="mountain_lon_deg", type=float,
        metavar="DEG", default=None,
        help="Mountain center longitude in degrees [default: 90]. "
             "Requires --topography mountain.")
    parser.add_argument(
        "--mountain-width-deg", dest="mountain_width_deg", type=float,
        metavar="DEG", default=None,
        help="Mountain Gaussian e-folding half-width in degrees (0, 90] "
             "[default: 20]. Requires --topography mountain.")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plots", action="append", metavar="TYPE",
        choices=list(SWE_PLOT_TYPES) + ["all"], default=None,
        help="Generate only the named image product; repeatable "
             f"({', '.join(SWE_PLOT_TYPES)}, or 'all'). Duplicates are "
             "ignored and execution order is fixed.")
    plot_group.add_argument(
        "--no-plots", action="store_true", default=None,
        help="Generate no image files (spectral snapshots and numerical "
             "diagnostics are still written).")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Base directory for run outputs [default: runs].")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Optional grouping name; runs go to <out>/<experiment>/<run_id>/.")
    parser.add_argument(
        "--overwrite", action="store_true", default=None,
        help="Reuse an existing run directory on a run-id collision.")


_SWE_EXPLICIT_KEYS = (
    "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
    "radius_earth_units", "duration_days", "gravity", "mean_depth_m",
    "scenario", "topography", "mountain_height_m", "mountain_lat_deg",
    "mountain_lon_deg", "mountain_width_deg", "n_snapshots", "dt_snapshots",
    "plots", "no_plots", "out", "experiment", "overwrite")


def _cmd_run_swe(args: argparse.Namespace) -> int:
    from planetary_sandbox.run.swe.config import SWERunConfig  # import-light

    explicit = {k: getattr(args, k, None) for k in _SWE_EXPLICIT_KEYS}
    try:
        cfg = SWERunConfig.resolve(explicit)
    except ValueError as err:
        args._parser.error(str(err))
    print("\n".join(cfg.summary_lines()))
    # Heavy imports (CuPy, matplotlib) happen inside execute_run.
    from planetary_sandbox.cli import swe as swe_module
    return swe_module.execute_run(cfg)


# ---------------------------------------------------------------------------
# run pe: arguments and dispatch
# ---------------------------------------------------------------------------

_PE_EXAMPLES = """\
examples:
  aeolus run pe                          thermal_wave, tiny geodesic demo, fixed 300 s step
  aeolus run pe --scenario isothermal_rest    verify the exact-rest property
  aeolus run pe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15
  aeolus run pe --levels 12 --dt-seconds 200 --days 0.05 --n-snapshots 4
  aeolus run pe --sigma-interfaces 0,0.25,0.6,1.0 --temperature 250
"""


def _parse_sigma_interfaces(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of sigma interface coordinates."""
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"sigma interfaces must be comma-separated numbers, got {text!r}"
        ) from err
    if len(values) < 2:
        raise argparse.ArgumentTypeError(
            "need at least 2 sigma interfaces (1 layer)")
    return values


def add_pe_arguments(parser: argparse.ArgumentParser) -> None:
    """All `run pe` options. Every default is None (resolution applies them)."""
    from planetary_sandbox.run.pe.config import (  # import-light
        PE_PLOT_TYPES, PE_SCENARIOS)

    parser.add_argument(
        "--backend", "--grid", dest="grid", choices=list(BACKEND_CHOICES),
        default=None,
        help="Numerical backend / grid family [default: geodesic]. "
             "'gauss-latlon' is an alias for 'latlon'.")
    parser.add_argument(
        "--l-max", "--lmax", dest="lmax", type=int, default=None,
        help="Maximum spherical harmonic degree [default: 10].")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Geodesic grid subdivision level [default: 3].")
    parser.add_argument(
        "--nlat", type=int, default=None,
        help="Lat-lon backend: number of Gauss-Legendre latitudes [default: 32].")
    parser.add_argument(
        "--nlon", type=int, default=None,
        help="Lat-lon backend: number of uniform longitudes [default: 64].")
    parser.add_argument(
        "--day-hours", type=float, default=None,
        help="Sidereal day length in hours; 'inf' = non-rotating [default: 24].")
    parser.add_argument(
        "--radius-earth-units", type=float, default=None,
        help="Planet radius in Earth radii [default: 1.0].")

    levels = parser.add_mutually_exclusive_group()
    levels.add_argument(
        "--levels", dest="nlev", type=int, metavar="K", default=None,
        help="Number of uniform sigma layers [default: 8]. Mutually "
             "exclusive with --sigma-interfaces.")
    levels.add_argument(
        "--sigma-interfaces", dest="sigma_interfaces",
        type=_parse_sigma_interfaces, metavar="S0,...,SK", default=None,
        help="Explicit sigma interface coordinates (comma-separated, from "
             "0.0 to 1.0). Overrides --levels.")

    parser.add_argument(
        "--r-dry", dest="r_dry", type=float, default=None,
        help="Dry-air gas constant R_d in J/kg/K [default: 287.04].")
    parser.add_argument(
        "--cp-dry", dest="cp_dry", type=float, default=None,
        help="Dry-air specific heat c_p in J/kg/K [default: 1004.64].")

    parser.add_argument(
        "--scenario", choices=sorted(PE_SCENARIOS), default=None,
        help="Initial-condition preset [default: thermal_wave].")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Initial (resting) temperature in K [default: 260].")
    parser.add_argument(
        "--surface-pressure", dest="surface_pressure", type=float,
        default=None,
        help="Initial (uniform) surface pressure in Pa [default: 101325].")
    parser.add_argument(
        "--thermal-amplitude", dest="thermal_amplitude", type=float,
        default=None,
        help="thermal_wave degree-2 perturbation amplitude in K [default: 1].")

    parser.add_argument(
        "--dt-seconds", dest="dt_seconds", type=float, metavar="SECONDS",
        default=None,
        help="FIXED integration timestep in seconds [default: 300]. This "
             "runner uses a user-supplied conservative fixed step, not an "
             "adaptive CFL controller.")
    parser.add_argument(
        "--days", "--duration-days", dest="duration_days", type=float,
        default=None,
        help="Simulated duration in days [default: ~0.0208 (30 minutes)].")

    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument(
        "--n-snapshots", type=int, metavar="N", default=None,
        help="Store N spectral states evenly spaced over the duration "
             "[default: 3]. Same semantics as run bve/swe.")
    snapshots.add_argument(
        "--snapshot-interval-seconds", "--dt-snapshots",
        dest="dt_snapshots", type=float, metavar="SECONDS", default=None,
        help="Store a state every SECONDS of simulated time instead of a count.")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plots", action="append", metavar="TYPE",
        choices=list(PE_PLOT_TYPES) + ["all"], default=None,
        help="Generate only the named image product; repeatable "
             f"({', '.join(PE_PLOT_TYPES)}, or 'all').")
    plot_group.add_argument(
        "--no-plots", action="store_true", default=None,
        help="Generate no image files (spectral snapshots and numerical "
             "diagnostics are still written).")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Base directory for run outputs [default: runs].")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Optional grouping name; runs go to <out>/<experiment>/<run_id>/.")
    parser.add_argument(
        "--overwrite", action="store_true", default=None,
        help="Reuse an existing run directory on a run-id collision.")


_PE_EXPLICIT_KEYS = (
    "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
    "radius_earth_units", "nlev", "sigma_interfaces", "r_dry", "cp_dry",
    "duration_days", "dt_seconds", "scenario", "temperature",
    "surface_pressure", "thermal_amplitude", "n_snapshots", "dt_snapshots",
    "plots", "no_plots", "out", "experiment", "overwrite")


def _cmd_run_pe(args: argparse.Namespace) -> int:
    from planetary_sandbox.run.pe.config import PERunConfig  # import-light

    explicit = {k: getattr(args, k, None) for k in _PE_EXPLICIT_KEYS}
    try:
        cfg = PERunConfig.resolve(explicit)
    except ValueError as err:
        args._parser.error(str(err))
    print("\n".join(cfg.summary_lines()))
    # Heavy imports (CuPy, matplotlib) happen inside execute_run.
    from planetary_sandbox.cli import pe as pe_module
    return pe_module.execute_run(cfg)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def _cmd_list_presets(args: argparse.Namespace) -> int:
    for name in sorted(PRESETS):
        entry = PRESETS[name]
        print(f"{name}")
        print(f"    {entry['description']}")
        settings = ", ".join(f"{k}={v}" for k, v in entry["settings"].items())
        print(f"    sets: {settings}")
    return 0


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    from planetary_sandbox.run.swe.config import SWE_SCENARIOS  # import-light
    from planetary_sandbox.run.pe.config import PE_SCENARIOS  # import-light

    for title, catalog in (("bve", SCENARIOS), ("swe", SWE_SCENARIOS),
                           ("pe", PE_SCENARIOS)):
        print(f"{title} scenarios:")
        width = max(len(name) for name in catalog)
        for name in sorted(catalog):
            print(f"  {name:<{width}}  {catalog[name]}")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _resolve_inspect_target(target):
    """Resolve RUN_PATH into a concrete run directory.

    Accepts a run directory, a base directory with latest_run.txt, or an
    experiment directory (picks the newest run, which the timestamped
    run-id naming makes the lexically greatest).
    """
    import pathlib

    target = pathlib.Path(target)
    if (target / "manifest.json").exists() or (target / "config.json").exists():
        return target, None

    pointer_file = target / "latest_run.txt"
    if pointer_file.exists():
        pointer = pointer_file.read_text(encoding="utf-8").strip()
        pointed = pathlib.Path(pointer)
        return (pointed if pointed.is_absolute() else target / pointer), None

    if target.is_dir():
        runs = sorted(
            child for child in target.iterdir()
            if child.is_dir() and ((child / "manifest.json").exists()
                                   or (child / "config.json").exists()))
        if runs:
            note = (f"(newest of {len(runs)} runs in {target})"
                    if len(runs) > 1 else None)
            return runs[-1], note
    return None, None


def _cmd_inspect(args: argparse.Namespace) -> int:
    import json

    run_dir, note = _resolve_inspect_target(args.run_path)
    if run_dir is None:
        return _error(
            f"no run found under: {args.run_path} (expected a run directory, "
            "an experiment directory, or a base directory with latest_run.txt)")

    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.exists() and not config_path.exists():
        return _error(f"no manifest.json or config.json found under: {run_dir}")

    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return _error(
                    f"malformed manifest.json under {run_dir}: "
                    f"expected an object, got {type(manifest).__name__}")
            raw_run_config = manifest.get("run_config")
            if raw_run_config is None:
                run_config = {}
            elif isinstance(raw_run_config, dict):
                run_config = raw_run_config
            else:
                return _error(
                    f"malformed manifest.json under {run_dir}: "
                    f"'run_config' is {type(raw_run_config).__name__}, "
                    "expected an object")
        else:
            manifest = {}
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(run_config, dict):
                return _error(
                    f"malformed config.json under {run_dir}: "
                    f"expected an object, got {type(run_config).__name__}")
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        return _error(f"malformed run metadata under {run_dir}: {err}")
    except OSError as err:
        return _error(f"could not read run metadata under {run_dir}: {err}")

    print(f"Run directory: {run_dir}")
    if note:
        print(f"  {note}")

    def show(label: str, value) -> None:
        if value not in (None, "", {}):
            print(f"  {label:<18}{value}")

    show("run id", manifest.get("run_id") or run_config.get("run_id"))
    show("status", manifest.get("status"))
    show("created (UTC)", manifest.get("created_utc"))
    if manifest.get("updated_utc"):
        show("updated (UTC)", manifest.get("updated_utc"))
    show("experiment", manifest.get("experiment") or run_config.get("experiment"))
    err = manifest.get("error") or {}
    if isinstance(err, dict) and err.get("type"):
        show("failure", f"{err['type']}: {err.get('message', '')}".rstrip(": "))

    git = manifest.get("git") or {}
    if git.get("commit"):
        commit = git["commit"][:8]
        if git.get("dirty"):
            commit += " (dirty)"
        if git.get("branch"):
            commit += f" on {git['branch']}"
        show("commit", commit)

    grid = run_config.get("grid")
    if grid == "latlon":
        show("grid", f"latlon {run_config.get('nlat')} x {run_config.get('nlon')}")
    elif grid:
        show("grid", f"{grid} r{run_config.get('resolution')}")
    show("l_max", run_config.get("lmax"))
    show("scenario", run_config.get("scenario"))
    if run_config.get("solver") == "swe":
        # Additive schema: manifests without a topography key are flat runs.
        if run_config.get("topography", "flat") == "mountain":
            show("topography",
                 f"mountain (h={run_config.get('mountain_height_m')} m at "
                 f"lat {run_config.get('mountain_lat_deg')} deg, "
                 f"lon {run_config.get('mountain_lon_deg')} deg, "
                 f"width {run_config.get('mountain_width_deg')} deg)")
        else:
            show("topography", "flat")
    day_hours = run_config.get("day_hours")
    if day_hours is not None:
        show("day length", "non-rotating" if day_hours in (float("inf"), "Infinity")
             else f"{day_hours} h")
    if run_config.get("duration_days") is not None:
        show("duration", f"{run_config['duration_days']} days")

    mode = run_config.get("snapshot_mode")
    times = run_config.get("snapshot_times")
    if mode == "count":
        show("snapshots", f"count mode, N={run_config.get('n_snapshots')}")
    elif mode == "interval" or run_config.get("dt_snapshots") is not None:
        show("snapshots", f"interval mode, every {run_config.get('dt_snapshots')} s")
    if isinstance(times, list) and times:
        hours = ", ".join(f"{s/3600.0:g}" for s in times[:8])
        if len(times) > 8:
            hours += f", ... ({len(times)} total)"
        show("snapshot times", f"{hours} h")
    plots = run_config.get("plots")
    if plots is not None:
        show("plots", ", ".join(plots) if plots else "none")

    show("viscosity", run_config.get("viscosity"))
    numerics = manifest.get("numerics") or {}
    show("backend", numerics.get("backend"))
    show("product sampling", numerics.get("product_sampling"))
    show("gpu", manifest.get("gpu"))
    versions = manifest.get("versions") or {}
    if versions:
        show("versions", ", ".join(f"{k} {v}" for k, v in versions.items() if v))

    files = sorted(p.name + ("/" if p.is_dir() else "")
                   for p in run_dir.iterdir())
    if files:
        show("output files", ", ".join(files))
    return 0


# ---------------------------------------------------------------------------
# gen / recompile
# ---------------------------------------------------------------------------

def _cmd_gen(args: argparse.Namespace) -> int:
    return generate_planet.run(args)


def _cmd_recompile(args: argparse.Namespace) -> int:
    return clear_cache.run(args)


# ---------------------------------------------------------------------------
# Parser tree and entry point
# ---------------------------------------------------------------------------

_TOP_EXAMPLES = """\
examples:
  aeolus run bve --preset rh4
  aeolus run bve --backend gauss-latlon --scenario two_vortices --days 10
  aeolus run bve --days 1 --n-snapshots 9
  aeolus list presets
  aeolus inspect runs

psx-bve, psx-gen, and psx-recompile remain available as compatibility
entry points for 'aeolus run bve', 'aeolus gen', and 'aeolus recompile'.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeolus",
        description="Aeolus - spectral dynamical cores on the sphere "
                    "(barotropic vorticity and rotating shallow water).",
        epilog=_TOP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    # aeolus run <solver>
    run_parser = commands.add_parser(
        "run", help="Run a solver.",
        description="Run a solver: the barotropic vorticity equation (bve) "
                    "or the rotating shallow-water equations (swe).")
    solvers = run_parser.add_subparsers(dest="solver", metavar="SOLVER",
                                        required=True)
    bve_parser = solvers.add_parser(
        "bve", help="Barotropic vorticity equation.",
        description="Run the barotropic vorticity equation on a planet.",
        epilog=_BVE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bve_arguments(bve_parser)
    bve_parser.set_defaults(_handler=_cmd_run_bve, _parser=bve_parser)

    swe_parser = solvers.add_parser(
        "swe", help="Rotating shallow-water equations.",
        description="Run the rotating shallow-water equations on a planet "
                    "(inviscid, with optional fixed bottom topography; "
                    "prognostics: vorticity, divergence, perturbation "
                    "thickness geopotential).",
        epilog=_SWE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_swe_arguments(swe_parser)
    swe_parser.set_defaults(_handler=_cmd_run_swe, _parser=swe_parser)

    pe_parser = solvers.add_parser(
        "pe", help="Dry primitive equations (hydrostatic, sigma coordinate).",
        description="Run the dry hydrostatic primitive equations on a planet "
                    "(vorticity/divergence/temperature/ln p_s prognostics, "
                    "fixed-step RK4; no forcing, diffusion, or semi-implicit "
                    "terms).",
        epilog=_PE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pe_arguments(pe_parser)
    pe_parser.set_defaults(_handler=_cmd_run_pe, _parser=pe_parser)

    # aeolus list <topic>
    list_parser = commands.add_parser(
        "list", help="List presets or scenarios.",
        description="List available presets or initial-condition scenarios.")
    topics = list_parser.add_subparsers(dest="topic", metavar="TOPIC",
                                        required=True)
    presets_parser = topics.add_parser(
        "presets", help="Named run configurations for 'aeolus run bve --preset'.")
    presets_parser.set_defaults(_handler=_cmd_list_presets)
    scenarios_parser = topics.add_parser(
        "scenarios", help="Initial-condition scenarios for 'aeolus run bve --scenario'.")
    scenarios_parser.set_defaults(_handler=_cmd_list_scenarios)

    # aeolus inspect RUN_PATH
    inspect_parser = commands.add_parser(
        "inspect", help="Summarize a run directory from its manifest.",
        description="Print a summary of a finished run from its manifest.json"
                    " / config.json. Never initializes CUDA.")
    inspect_parser.add_argument(
        "run_path",
        help="A run directory, an experiment directory, or a base directory "
             "containing latest_run.txt (e.g. 'runs').")
    inspect_parser.set_defaults(_handler=_cmd_inspect)

    # aeolus gen
    gen_parser = commands.add_parser(
        "gen", help="Generate a demo planet and save a summary plot.",
        description="Generate a demo planet and save a summary plot.")
    generate_planet.add_arguments(gen_parser)
    gen_parser.set_defaults(_handler=_cmd_gen)

    # aeolus recompile
    recompile_parser = commands.add_parser(
        "recompile",
        help="Clear the CuPy kernel cache and verify kernel compilation.",
        description="Clear CuPy's kernel cache and verify that the "
                    "spherical-harmonics kernel recompiles.")
    clear_cache.add_arguments(recompile_parser)
    recompile_parser.set_defaults(_handler=_cmd_recompile)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
