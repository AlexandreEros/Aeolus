"""aeolus — the public command-line interface for Aeolus.

Command tree::

    aeolus run bve [...]        run the barotropic vorticity solver
    aeolus list presets         named run configurations
    aeolus list scenarios       initial-condition scenarios
    aeolus inspect RUN_DIR      summarize a finished run from its manifest
    aeolus planet generate      demo planet + summary plot (psx-gen)
    aeolus cache rebuild        clear/verify the CuPy kernel cache (psx-recompile)

The ``psx-bve`` / ``psx-gen`` / ``psx-recompile`` commands are kept as
compatibility aliases and delegate here.

Design rules for this module:

- Import-light: parsing, ``--help``, ``list``, and ``inspect`` must never
  import CuPy or matplotlib. Heavy imports happen inside command handlers,
  after argument validation.
- Parser options default to ``None``; the documented defaults live in
  ``BVE_DEFAULTS`` and are applied during configuration resolution so that
  presets can be layered between defaults and explicit flags
  (defaults < preset < explicit flags).
- Resolved run semantics live in ``planetary_sandbox.run.bve.config``;
  this module owns only parsing, aliases, and presets.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from planetary_sandbox.cli import clear_cache, generate_planet

# ---------------------------------------------------------------------------
# Choices, defaults, presets
# ---------------------------------------------------------------------------

#: Initial-condition scenarios. Must match INITIAL_CONDITIONS in
#: run/bve/initial_conditions.py (kept as a plain tuple here because that
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

#: Documented defaults for `aeolus run bve`, identical to the historical
#: psx-bve argparse defaults. Parser options default to None; these are
#: applied in resolution so presets can sit between them and explicit flags.
BVE_DEFAULTS = {
    "lmax": 21,
    "grid": "geodesic",
    "resolution": 4,
    "nlat": 128,
    "nlon": 256,
    "day_hours": float("inf"),
    "radius_earth_units": 1.0,
    "duration_days": 1.0,
    "scenario": "two_vortices",
    "viscosity": 0.0,
    "product_quadrature": "fine",
    "out": "runs",
    "experiment": None,
    "overwrite": False,
}

#: Snapshot controls stay None by default; the 21600 s historical default
#: is applied by resolve_snapshot_interval() only when neither is given.
SNAPSHOT_DEFAULTS = {
    "n_snapshots": None,
    "snapshot_interval_seconds": None,
}

#: Named bundles of run-bve settings (the README-documented configurations).
#: Explicit flags always override preset values.
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
            "snapshot_interval_seconds": 21600.0,
            "product_quadrature": "fine",
            "viscosity": 0.0,
            "experiment": "validation-rh4",
        },
    },
    "two-vortices": {
        "description": "Small, fast two-vortex smoke run "
                       "(the README quickstart configuration).",
        "settings": {
            "scenario": "two_vortices",
            "lmax": 8,
            "resolution": 3,
            "nlat": 12,
            "nlon": 24,
            "duration_days": 0.02,
            "snapshot_interval_seconds": 864.0,
            "experiment": "quickstart",
        },
    },
}


# ---------------------------------------------------------------------------
# run bve: arguments and configuration resolution
# ---------------------------------------------------------------------------

_BVE_EXAMPLES = """\
examples:
  aeolus run bve                          geodesic grid, two_vortices, 1 day, 6 h snapshots
  aeolus run bve --preset rh4             documented RH4 validation configuration
  aeolus run bve --days 1 --n-snapshots 5
  aeolus run bve --backend gauss-latlon --nlat 12 --nlon 24 --lmax 8 --days 0.02 --n-snapshots 3
"""


def add_bve_arguments(parser: argparse.ArgumentParser) -> None:
    """All `run bve` options. Defaults are None (see module docstring)."""
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
        "--lmax", type=int, default=None,
        help="Maximum spherical harmonic degree [default: 21].")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Geodesic grid subdivision level [default: 4]. "
             "The (resolution=4, lmax=21) default keeps ~10 grid points per "
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
        help="Store N states, including both the initial and the final state, "
             "evenly spaced over the duration (N >= 2). Mutually exclusive "
             "with --snapshot-interval-seconds.")
    snapshots.add_argument(
        "--snapshot-interval-seconds", "--dt-snapshots",
        dest="snapshot_interval_seconds", type=float, metavar="SECONDS",
        default=None,
        help="Store a state every SECONDS of simulated time [default: 21600 "
             "(6 h)]. The initial state is always stored; the final state is "
             "stored only if the duration is a multiple of the interval. "
             "--dt-snapshots is a compatibility alias.")

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

    ``apply_defaults=True`` fills in the documented defaults directly on the
    parser (the historical psx-bve behavior; kept for the legacy
    ``planetary_sandbox.cli.bve.build_parser`` import surface). Snapshot
    controls stay None either way — their default is applied in resolution.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the barotropic vorticity equation on a planet.",
        epilog=_BVE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bve_arguments(parser)
    if apply_defaults:
        parser.set_defaults(**BVE_DEFAULTS)
    return parser


def resolve_bve_config(args: argparse.Namespace):
    """Layer defaults < preset < explicit flags into a BVERunConfig.

    Raises ValueError for invalid combinations (the caller maps that onto a
    parser error). No CuPy/matplotlib import happens here.
    """
    from planetary_sandbox.run.bve.config import (
        BVERunConfig, resolve_snapshot_interval)

    settings = dict(BVE_DEFAULTS, **SNAPSHOT_DEFAULTS)
    preset = dict(PRESETS[args.preset]["settings"]) if getattr(args, "preset", None) else {}
    explicit = {k: v for k, v in vars(args).items()
                if k in settings and v is not None}

    # The two snapshot controls are one mutually exclusive choice: an
    # explicit flag replaces whichever form the preset used.
    if "n_snapshots" in explicit or "snapshot_interval_seconds" in explicit:
        preset.pop("n_snapshots", None)
        preset.pop("snapshot_interval_seconds", None)

    settings.update(preset)
    settings.update(explicit)

    dt_snapshots = resolve_snapshot_interval(
        settings["duration_days"],
        n_snapshots=settings.pop("n_snapshots"),
        snapshot_interval_seconds=settings.pop("snapshot_interval_seconds"))

    if settings["grid"] == "gauss-latlon":
        settings["grid"] = "latlon"

    return BVERunConfig(dt_snapshots=dt_snapshots, **settings)


def _print_resolved_config(cfg, preset: Optional[str]) -> None:
    times = cfg.snapshot_times_seconds()
    lines = ["Resolved run configuration:"]
    if preset:
        lines.append(f"  preset              {preset}")
    lines.append(f"  backend/grid        {cfg.grid}")
    if cfg.grid == "geodesic":
        lines.append(f"  resolution          {cfg.resolution} (geodesic subdivision level)")
    else:
        lines.append(f"  nlat x nlon         {cfg.nlat} x {cfg.nlon} "
                     "(Gauss-Legendre latitudes x uniform longitudes)")
    day = "inf (non-rotating)" if cfg.day_hours == float("inf") else f"{cfg.day_hours:g} h"
    out = cfg.out if cfg.experiment is None else f"{cfg.out} (experiment: {cfg.experiment})"
    lines += [
        f"  l_max               {cfg.lmax}",
        f"  scenario            {cfg.scenario}",
        f"  day length          {day}",
        f"  radius              {cfg.radius_earth_units:g} Earth radii",
        f"  duration            {cfg.duration_days:g} days",
        f"  snapshot interval   {cfg.dt_snapshots:g} s "
        f"({len(times)} stored states incl. t=0)",
        f"  viscosity           {cfg.viscosity:g} m^2/s",
        f"  product quadrature  {cfg.product_quadrature}",
        f"  output base         {out}",
    ]
    if not cfg.includes_final_state:
        lines.append(
            "  note: the duration is not a multiple of the snapshot interval, "
            "so the final state will not be stored (use --n-snapshots to "
            "include it).")
    print("\n".join(lines))


def _cmd_run_bve(args: argparse.Namespace) -> int:
    try:
        cfg = resolve_bve_config(args)
    except ValueError as err:
        args._parser.error(str(err))
    _print_resolved_config(cfg, args.preset)
    # Heavy imports (CuPy, matplotlib) happen inside execute_run.
    from planetary_sandbox.cli import bve as bve_module
    return bve_module.execute_run(cfg)


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
    width = max(len(name) for name in SCENARIOS)
    for name in sorted(SCENARIOS):
        print(f"{name:<{width}}  {SCENARIOS[name]}")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def _cmd_inspect(args: argparse.Namespace) -> int:
    import json
    import pathlib

    target = pathlib.Path(args.run_dir)
    run_dir = target
    if not (run_dir / "manifest.json").exists() and (target / "latest_run.txt").exists():
        pointer = (target / "latest_run.txt").read_text(encoding="utf-8").strip()
        pointed = pathlib.Path(pointer)
        run_dir = pointed if pointed.is_absolute() else target / pointer

    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.exists() and not config_path.exists():
        print(f"error: no manifest.json or config.json found under: {run_dir}",
              file=sys.stderr)
        return 2

    print(f"Run directory: {run_dir}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config = manifest.get("run_config") or {}
    else:
        manifest = {}
        run_config = json.loads(config_path.read_text(encoding="utf-8"))

    def show(label: str, value) -> None:
        if value not in (None, "", {}):
            print(f"  {label:<18}{value}")

    show("run id", manifest.get("run_id") or run_config.get("run_id"))
    show("created (UTC)", manifest.get("created_utc"))
    show("experiment", manifest.get("experiment") or run_config.get("experiment"))

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
    if run_config.get("duration_days") is not None:
        show("duration", f"{run_config['duration_days']} days "
                         f"(snapshots every {run_config.get('dt_snapshots')} s)")
    show("viscosity", run_config.get("viscosity"))

    numerics = manifest.get("numerics") or {}
    show("backend", numerics.get("backend"))
    show("product sampling", numerics.get("product_sampling"))
    show("gpu", manifest.get("gpu"))
    versions = manifest.get("versions") or {}
    if versions:
        show("versions", ", ".join(f"{k} {v}" for k, v in versions.items() if v))
    return 0


# ---------------------------------------------------------------------------
# planet / cache
# ---------------------------------------------------------------------------

def _cmd_planet_generate(args: argparse.Namespace) -> int:
    return generate_planet.run(args)


def _cmd_cache_rebuild(args: argparse.Namespace) -> int:
    return clear_cache.run(args)


# ---------------------------------------------------------------------------
# Parser tree and entry point
# ---------------------------------------------------------------------------

_TOP_EXAMPLES = """\
examples:
  aeolus run bve --preset rh4
  aeolus run bve --backend gauss-latlon --scenario two_vortices --days 10
  aeolus run bve --days 1 --n-snapshots 5
  aeolus list presets
  aeolus inspect runs

psx-bve, psx-gen, and psx-recompile remain as compatibility aliases for
'aeolus run bve', 'aeolus planet generate', and 'aeolus cache rebuild'.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeolus",
        description="Aeolus - a spectral barotropic vorticity solver on the sphere.",
        epilog=_TOP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    # aeolus run <solver>
    run_parser = commands.add_parser(
        "run", help="Run a solver.",
        description="Run a solver. Currently the barotropic vorticity "
                    "equation (bve) is the only solver.")
    solvers = run_parser.add_subparsers(dest="solver", metavar="SOLVER",
                                        required=True)
    bve_parser = solvers.add_parser(
        "bve", help="Barotropic vorticity equation.",
        description="Run the barotropic vorticity equation on a planet.",
        epilog=_BVE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bve_arguments(bve_parser)
    bve_parser.set_defaults(_handler=_cmd_run_bve, _parser=bve_parser)

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

    # aeolus inspect RUN_DIR
    inspect_parser = commands.add_parser(
        "inspect", help="Summarize a run directory from its manifest.",
        description="Print a summary of a finished run from its manifest.json "
                    "/ config.json. Never initializes CUDA.")
    inspect_parser.add_argument(
        "run_dir",
        help="A run directory, or a base directory containing latest_run.txt "
             "(e.g. 'runs').")
    inspect_parser.set_defaults(_handler=_cmd_inspect)

    # aeolus planet generate
    planet_parser = commands.add_parser(
        "planet", help="Planet utilities.",
        description="Planet utilities.")
    planet_subparsers = planet_parser.add_subparsers(dest="action",
                                                     metavar="ACTION",
                                                     required=True)
    generate_parser = planet_subparsers.add_parser(
        "generate", help="Generate a demo planet and save a summary plot.",
        description="Generate a demo planet and save a summary plot.")
    generate_planet.add_arguments(generate_parser)
    generate_parser.set_defaults(_handler=_cmd_planet_generate)

    # aeolus cache rebuild
    cache_parser = commands.add_parser(
        "cache", help="CuPy kernel-cache utilities.",
        description="CuPy kernel-cache utilities.")
    cache_subparsers = cache_parser.add_subparsers(dest="action",
                                                   metavar="ACTION",
                                                   required=True)
    rebuild_parser = cache_subparsers.add_parser(
        "rebuild",
        help="Clear the CuPy kernel cache and verify kernel compilation.",
        description="Clear CuPy's kernel cache and verify that the "
                    "spherical-harmonics kernel recompiles.")
    clear_cache.add_arguments(rebuild_parser)
    rebuild_parser.set_defaults(_handler=_cmd_cache_rebuild)

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
