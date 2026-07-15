"""psx-bve compatibility entry point and the BVE run executor.

``aeolus run bve`` (planetary_sandbox.cli.main) is the canonical interface;
``psx-bve`` delegates to it unchanged. Heavy imports (CuPy, matplotlib)
happen inside :func:`execute_run`, so parsing and ``--help`` never
initialize CUDA.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planetary_sandbox.run.bve.config import BVERunConfig


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    """Ensure the runs *base* directory exists and is writable; fall back if not."""
    out_dir = pathlib.Path(requested_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_path = out_dir / ".write_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return out_dir, False
    except OSError:
        # Fall back to the system temp location, NOT dir="." — dropping the
        # temp dir into the repo root left persistent, sometimes ACL-locked
        # `psx-bve-*` directories behind (one of which broke bare `pytest`
        # collection). Keep run outputs out of the project tree entirely.
        fallback_root = pathlib.Path(tempfile.mkdtemp(prefix="psx-bve-"))
        fallback_out_dir = fallback_root / out_dir.name
        fallback_out_dir.mkdir(parents=True, exist_ok=True)
        return fallback_out_dir, True


def build_parser():
    """Legacy import surface: the full BVE parser with defaults applied.

    Equivalent to the historical psx-bve parser (same flags, same defaults),
    plus the aeolus additions (--preset, --n-snapshots, aliases).
    """
    from planetary_sandbox.cli.main import build_bve_parser
    return build_bve_parser(prog="psx-bve", apply_defaults=True)


def execute_run(cfg: "BVERunConfig") -> int:
    """Execute one resolved BVE run: run dir, provenance, then the solver."""
    from planetary_sandbox.run.bve.io import create_run_dir, write_run_manifest

    base_dir, used_fallback = _resolve_writable_base_dir(cfg.out)
    if used_fallback:
        banner = "=" * 72
        print(banner)
        print(f"WARNING: requested --out '{cfg.out}' is not writable.")
        print(f"         Run outputs will be written OUTSIDE the project tree to:")
        print(f"           {base_dir.resolve()}")
        print(banner)

    # Build the config dict *before* creating the run dir so the run ID
    # reflects exactly what will be written to disk.
    run_config = cfg.to_run_config_dict()
    run_config["out"] = str(base_dir)

    run_dir = create_run_dir(
        base_dir, run_config,
        experiment=cfg.experiment,
        overwrite=cfg.overwrite,
    )
    out_dir = run_dir.path
    run_dir.update_latest_pointer()
    print(f"Run directory: {out_dir}")
    if run_dir.reused:
        print("  (reused via --overwrite; existing files may be replaced)")

    # Heavy imports only now, after all user-error validation is done.
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.runner import run_bve
    from planetary_sandbox.run.bve.initial_conditions import make_ic

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=cfg.day_hours,
            radius_earth_units=cfg.radius_earth_units),
        grid_resolution=cfg.resolution,
        l_max=cfg.lmax,
        product_quadrature=cfg.product_quadrature,
        grid_type=cfg.grid,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
    )

    # Initial condition on grid, then transform -> spectral ζ_lm
    zeta0_grid = make_ic(cfg.scenario, planet)                  # (nlat, nlon) cupy or numpy ok
    zeta0_lm = planet.sh.transform(zeta0_grid)

    # Save run config + provenance manifest (git commit, versions, GPU, argv)
    run_config["run_id"] = run_dir.run_id
    run_config["experiment"] = cfg.experiment
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       numerics=planet.so.backend.describe(cfg.product_quadrature))

    run_bve(planet=planet,
            zeta0_lm=zeta0_lm,
            dt_snapshots=cfg.dt_snapshots,
            t_end_days=cfg.duration_days,
            out_dir=out_dir,
            viscosity=cfg.viscosity,
            scenario=cfg.scenario,
            figure_metadata=run_dir.figure_metadata(),
            snapshot_times=cfg.snapshot_times_seconds(),
            plots=cfg.plots)
    return 0


def main() -> int:
    """psx-bve == aeolus run bve, with the legacy snapshot default.

    The only behavioral difference from the canonical interface: when
    neither --n-snapshots nor --snapshot-interval-seconds is given, psx-bve
    keeps the historical 21600 s interval instead of the count default, so
    old invocations do not silently change behavior.
    """
    import sys
    from planetary_sandbox.cli.main import build_bve_parser, run_bve_command
    parser = build_bve_parser(prog="psx-bve")
    args = parser.parse_args(sys.argv[1:])
    return run_bve_command(args, parser, snapshot_default="interval")


if __name__ == "__main__":
    raise SystemExit(main())
