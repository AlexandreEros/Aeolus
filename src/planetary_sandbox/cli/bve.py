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
import re
import shutil
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planetary_sandbox.run.bve.config import BVERunConfig


class OverwriteCleanupError(RuntimeError):
    """A known stale generated artifact could not be removed on --overwrite.

    Raised so the overwrite aborts *before* the new run is reported
    completed, rather than leaving a stale artifact from a prior
    configuration alongside fresh outputs.
    """


#: Generated *result* files removed when --overwrite reuses a directory.
#: config.json / manifest.json are deliberately excluded — they are handled
#: by the run lifecycle (overwritten in place, atomically), kept separate
#: from generated-result cleanup.
_GENERATED_RESULT_FILES: tuple[str, ...] = (
    "vorticity_coeffs.npy",   # saved spectral state
    "vorticity_grid.npy",     # saved plotting snapshots
    "bve_summary.png",        # the summary image
)

#: Generated *result* directories removed on --overwrite (diagnostics CSV /
#: spectra and the diagnostics figures directory).
_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
)

#: Per-snapshot panel image naming pattern produced by
#: ``VorticityViewer.plot_all_snapshots``:
#: ``{scenario}_t{t0:02.2f}h-{t1:02.2f}h-{dt:02.2f}h.png``. Matched narrowly
#: (numeric triplet) so a blanket ``*.png`` sweep never claims a user file
#: such as ``custom.png``. ``bve_summary.png`` also does not match.
_SNAPSHOT_PANEL_RE = re.compile(
    r".+_t\d+\.\d{2}h-\d+\.\d{2}h-\d+\.\d{2}h\.png\Z")


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


def _iter_generated_artifacts(out_dir: pathlib.Path):
    """Yield the known Aeolus-generated result artifacts present in ``out_dir``.

    Strictly scoped: named result files/directories plus per-snapshot panel
    PNGs matching the viewer's naming pattern. Never yields config.json /
    manifest.json (lifecycle-managed) or arbitrary user files, so unrelated
    content such as a root-level ``custom.png`` is preserved.
    """
    for name in _GENERATED_RESULT_DIRS:
        target = out_dir / name
        if target.is_dir():
            yield target
    for name in _GENERATED_RESULT_FILES:
        target = out_dir / name
        if target.exists():
            yield target
    for child in out_dir.iterdir():
        if child.is_file() and _SNAPSHOT_PANEL_RE.match(child.name):
            yield child


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known generated outputs before a reused run overwrites them.

    Only Aeolus-generated result artifacts are removed; arbitrary user files
    (e.g. a root-level ``custom.png``) are preserved. If any *known* stale
    artifact cannot be removed, every removable one is still attempted and
    then :class:`OverwriteCleanupError` is raised, so the overwrite aborts
    before the new run is reported completed rather than leaving a stale
    artifact contradicting the fresh configuration.
    """
    failures: list[str] = []
    for target in list(_iter_generated_artifacts(out_dir)):
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as err:
            failures.append(f"{target.name}: {err}")
    if failures:
        raise OverwriteCleanupError(
            "could not remove stale generated artifact(s) during --overwrite "
            f"of {out_dir}: " + "; ".join(failures))


def _execute_solver(cfg: "BVERunConfig", run_dir, run_config: dict) -> None:
    """Heavy numerical portion of a run: build the planet and drive the solver.

    Isolated from :func:`execute_run`'s provenance lifecycle so the lifecycle
    (run-dir reuse, atomic status writes, pointer publication) is testable
    without CUDA — tests replace this with a stub that succeeds or raises.
    Imports CuPy/matplotlib only here, after all user-error validation.
    """
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.io import RUN_STATUS_RUNNING, write_run_manifest
    from planetary_sandbox.run.bve.runner import run_bve
    from planetary_sandbox.run.bve.initial_conditions import make_ic

    out_dir = run_dir.path
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
            plots=cfg.plots,
            snapshot_mode=cfg.snapshot_mode)


def execute_run(cfg: "BVERunConfig") -> int:
    """Execute one resolved BVE run: run dir, provenance, then the solver.

    Lifecycle:

    1. Resolve the output base directory (with the writability fallback).
    2. Create the run directory.
    3. On --overwrite reuse: clear latest_run.txt if it points at this
       directory, then remove known stale generated artifacts — both
       *before* the manifest is replaced / status set back to 'running'.
    4. Write config.json and manifest.json with status='running' (atomically)
       *before* any numerical work.
    5. Build the planet and dispatch the runner.
    6. On success, rewrite manifest.json with status='completed' (raising if
       that cannot be persisted) and only then publish latest_run.txt.
    7. On failure, rewrite manifest.json with status='failed' + a concise
       error record, do NOT publish latest_run.txt, and re-raise so the
       harness sees the nonzero exit as before. A provenance-persistence
       failure on this path is surfaced, never swallowed.
    """
    from planetary_sandbox.run.bve.io import (
        RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, RUN_STATUS_RUNNING,
        RunProvenanceError, atomic_write_text, create_run_dir, failure_record,
        update_manifest_status, write_run_manifest)

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
    run_config["run_id"] = run_dir.run_id
    run_config["experiment"] = cfg.experiment
    if run_dir.reused:
        print("  (reused via --overwrite; stale generated artifacts will be removed)")
        # Invalidate the pointer first: if latest_run.txt references this
        # directory, a failure of the overwritten run must not leave the
        # pointer aimed at an incomplete capsule.
        run_dir.clear_latest_pointer_if_matches()
        # Transition the existing capsule away from 'completed' before any
        # destructive cleanup. If cleanup fails after partially removing
        # results, persist 'failed' + the cleanup error so the damaged capsule
        # can never continue to claim successful completion.
        try:
            update_manifest_status(out_dir, RUN_STATUS_RUNNING)
            _clean_overwrite_artifacts(out_dir)
        except BaseException as exc:
            try:
                update_manifest_status(out_dir, RUN_STATUS_FAILED,
                                       error=failure_record(exc))
            except RunProvenanceError as prov_err:
                raise prov_err from exc
            raise

    # Write initial provenance so an interrupted or failing run leaves a
    # traceable capsule marked 'running' / 'failed', not a silent hole.
    atomic_write_text(out_dir / "config.json", json.dumps(run_config, indent=2))
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       status=RUN_STATUS_RUNNING)

    try:
        _execute_solver(cfg, run_dir, run_config)
    except BaseException as exc:
        # Failed runs must not leave latest_run.txt pointing at an empty
        # capsule. Mark status='failed' and re-raise; the CLI harness maps
        # the exception to a nonzero exit as it did before. If persisting the
        # 'failed' status itself fails, surface that (chained to the run
        # failure) rather than swallowing the provenance error.
        try:
            update_manifest_status(out_dir, RUN_STATUS_FAILED,
                                   error=failure_record(exc))
        except RunProvenanceError as prov_err:
            raise prov_err from exc
        raise

    # Completion must be durably persisted before the pointer is published;
    # update_manifest_status raises if it cannot write the 'completed' status,
    # so latest_run.txt is never published for an unpersisted completion.
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
