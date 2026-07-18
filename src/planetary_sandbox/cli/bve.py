"""psx-bve compatibility entry point and the BVE run executor.

``aeolus run bve`` (planetary_sandbox.cli.main) is the canonical interface;
``psx-bve`` delegates to it unchanged. Heavy imports (CuPy, matplotlib)
happen inside :func:`execute_run`, so parsing and ``--help`` never
initialize CUDA.
"""
from __future__ import annotations

import pathlib
import re
import shutil
from typing import TYPE_CHECKING

from planetary_sandbox.cli.run_lifecycle import (execute_with_provenance,
                                                 resolve_writable_base_dir)

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
    "bve_snapshot_times.npy", # authoritative stored times (seconds)
    "bve_summary.png",        # the summary image
)

#: Generated *result* directories removed on --overwrite (diagnostics CSV /
#: spectra and the diagnostics figures directory).
_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
)

#: Snapshot image names: the legacy viewer montage followed by current
#: timeline frames (fixed-width seconds). Matched narrowly so a blanket PNG
#: sweep never claims a user file such as ``custom.png``; the legacy form is
#: retained only so --overwrite cleans capsules created by older versions.
_SNAPSHOT_PANEL_RE = re.compile(
    r"(?:.+_t\d+\.\d{2}h-\d+\.\d{2}h-\d+\.\d{2}h|"
    r".+_t\d{13}\.\d{9}s)\.png\Z")


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    """Ensure the runs *base* directory exists and is writable; fall back if not.

    Delegates to the shared lifecycle helper (Codex finding 6 lives there),
    keeping the historical psx-bve fallback prefix.
    """
    return resolve_writable_base_dir(requested_out, fallback_prefix="psx-bve-")


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
    PNGs matching a known snapshot naming pattern. Never yields config.json /
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

    The (carefully ordered) lifecycle lives in
    ``cli/run_lifecycle.execute_with_provenance``; this wrapper injects the
    BVE-specific pieces. The callables below resolve the module globals at
    call time, so tests that monkeypatch ``_execute_solver`` /
    ``_clean_overwrite_artifacts`` keep working unchanged.
    """
    return execute_with_provenance(
        cfg,
        solver=lambda c, run_dir, run_config: _execute_solver(
            c, run_dir, run_config),
        clean_artifacts=lambda out_dir: _clean_overwrite_artifacts(out_dir),
        resolve_base_dir=lambda out: _resolve_writable_base_dir(out))


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
