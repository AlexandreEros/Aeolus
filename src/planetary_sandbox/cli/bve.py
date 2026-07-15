"""psx-bve compatibility entry point and the BVE run executor.

``aeolus run bve`` (planetary_sandbox.cli.main) is the canonical interface;
``psx-bve`` delegates to it unchanged. Heavy imports (CuPy, matplotlib)
happen inside :func:`execute_run`, so parsing and ``--help`` never
initialize CUDA.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planetary_sandbox.run.bve.config import BVERunConfig


#: Generated artifacts wiped when --overwrite reuses a directory. Anything
#: not in this set (notably user-added files) is preserved. Diagnostics/
#: figures/ live in subdirectories, so directory names are enumerated too.
_OVERWRITE_ARTIFACTS: tuple[str, ...] = (
    "config.json", "manifest.json",
    "vorticity_coeffs.npy", "vorticity_grid.npy",
    "bve_summary.png",
    "diagnostics", "figures",
)


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    """Ensure the runs *base* directory exists and is writable; fall back if not.

    Base-directory creation itself is inside the guarded path: an OSError
    from ``mkdir`` triggers the documented temporary-directory fallback
    instead of escaping as an unhandled exception (Codex finding 6).
    """
    requested = pathlib.Path(requested_out)

    def _try_probe(target: pathlib.Path) -> bool:
        """Attempt to create ``target`` and write a probe file inside it."""
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        probe_path = target / ".write_probe"
        try:
            probe_path.write_text("ok", encoding="utf-8")
        except OSError:
            return False
        finally:
            # Robust cleanup: even if the probe write failed midway (e.g.
            # a partial file was created), remove it silently. Missing
            # file is fine; we only want to avoid leaving a stray probe.
            with contextlib.suppress(OSError):
                probe_path.unlink(missing_ok=True)
        return True

    if _try_probe(requested):
        return requested, False

    # Fall back to the system temp location, NOT dir="." — dropping the
    # temp dir into the repo root left persistent, sometimes ACL-locked
    # `psx-bve-*` directories behind (one of which broke bare `pytest`
    # collection). Keep run outputs out of the project tree entirely.
    fallback_root = pathlib.Path(tempfile.mkdtemp(prefix="psx-bve-"))
    fallback_out_dir = fallback_root / requested.name
    fallback_out_dir.mkdir(parents=True, exist_ok=True)
    return fallback_out_dir, True


def build_parser():
    """Legacy import surface: the full BVE parser with defaults applied.

    Equivalent to the historical psx-bve parser (same flags, same defaults),
    plus the aeolus additions (--preset, --n-snapshots, aliases).
    """
    from planetary_sandbox.cli.main import build_bve_parser
    return build_bve_parser(prog="psx-bve", apply_defaults=True)


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known generated outputs before a reused run overwrites them.

    Prevents stale artifacts from a previous configuration (different plot
    selection, different snapshot mode) from contradicting the new
    configuration once the run finishes.
    """
    for name in _OVERWRITE_ARTIFACTS:
        target = out_dir / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                target.unlink(missing_ok=True)
    # Per-snapshot panel PNGs use a scenario-dependent filename; sweep
    # any *.png that isn't a user-added file.
    for png in out_dir.glob("*.png"):
        with contextlib.suppress(OSError):
            png.unlink()


def execute_run(cfg: "BVERunConfig") -> int:
    """Execute one resolved BVE run: run dir, provenance, then the solver.

    Lifecycle:

    1. Resolve the output base directory (with the writability fallback).
    2. Create the run directory.
    3. Write config.json and manifest.json with status='running' *before*
       any numerical work.
    4. Import CuPy, build the planet, and dispatch the runner.
    5. On success, rewrite manifest.json with status='completed' and only
       then update latest_run.txt.
    6. On failure, rewrite manifest.json with status='failed' + a concise
       error record, do NOT update latest_run.txt, and re-raise so the
       harness sees the nonzero exit as before.
    """
    from planetary_sandbox.run.bve.io import (
        RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, RUN_STATUS_RUNNING,
        create_run_dir, failure_record, update_manifest_status,
        write_run_manifest)

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
    print(f"Run directory: {out_dir}")
    if run_dir.reused:
        print("  (reused via --overwrite; stale generated artifacts will be removed)")
        _clean_overwrite_artifacts(out_dir)

    # Write initial provenance so an interrupted or failing run leaves a
    # traceable capsule marked 'running' / 'failed', not a silent hole.
    run_config["run_id"] = run_dir.run_id
    run_config["experiment"] = cfg.experiment
    (out_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8")
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       status=RUN_STATUS_RUNNING)

    try:
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
        zeta0_grid = make_ic(cfg.scenario, planet)
        zeta0_lm = planet.sh.transform(zeta0_grid)

        # Rewrite manifest now that we know the backend/product-sampling
        # provenance. Status stays 'running' until the runner returns.
        write_run_manifest(out_dir, run_config,
                           run_id=run_dir.run_id, experiment=cfg.experiment,
                           numerics=planet.so.backend.describe(cfg.product_quadrature),
                           status=RUN_STATUS_RUNNING)

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
    except BaseException as exc:
        # Failed runs must not leave latest_run.txt pointing at an empty
        # capsule. Mark status='failed' and re-raise; the CLI harness maps
        # the exception to a nonzero exit as it did before.
        update_manifest_status(out_dir, RUN_STATUS_FAILED,
                               error=failure_record(exc))
        raise

    update_manifest_status(out_dir, RUN_STATUS_COMPLETED)
    run_dir.update_latest_pointer()
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
