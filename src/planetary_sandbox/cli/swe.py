"""``aeolus run swe`` — the shallow-water run executor.

Heavy imports (CuPy, matplotlib) happen inside :func:`_execute_solver`, so
parsing and ``--help`` never initialize CUDA. The provenance lifecycle is
shared with the BVE command (``cli/run_lifecycle.py``).
"""
from __future__ import annotations

import pathlib
import re
import shutil
from typing import TYPE_CHECKING

from planetary_sandbox.cli.run_lifecycle import (execute_with_provenance,
                                                 resolve_writable_base_dir)

if TYPE_CHECKING:
    from planetary_sandbox.run.swe.config import SWERunConfig


#: Generated *result* files removed when --overwrite reuses a directory
#: (config.json / manifest.json are lifecycle-managed, never swept).
_GENERATED_RESULT_FILES: tuple[str, ...] = (
    "swe_coeffs.npy",
    "swe_snapshot_times.npy",
    "swe_summary.png",
)

_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
)

_SNAPSHOT_FRAME_RE = re.compile(r".+_t\d{13}\.\d{9}s\.png\Z")

#: Manifest notes describing the shallow-water solver.
SWE_MANIFEST_NOTES = {
    "equations": "rotating shallow-water equations (vorticity-divergence "
                 "form, perturbation geopotential; see "
                 "physics/shallow_water.py and docs/MATHEMATICAL_MODEL.md)",
    "timestep_policy": "state-adaptive CFL ceiling "
                       "0.5*cfl_length_scale/max(|u|+sqrt(Phi0+phi)) "
                       "recomputed from every accepted state; steps shorten "
                       "to land exactly on output times and t_end",
    "diagnostics": "see run/swe/diagnostics.py module docstring for definitions",
}


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    return resolve_writable_base_dir(requested_out,
                                     fallback_prefix="aeolus-swe-")


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known SWE-generated outputs before a reused run overwrites them.

    Strictly scoped to Aeolus-generated result artifacts, like the BVE
    variant; raises OverwriteCleanupError (via the shared type in cli/bve.py)
    if any known stale artifact cannot be removed.
    """
    from planetary_sandbox.cli.bve import OverwriteCleanupError

    failures: list[str] = []
    targets: list[pathlib.Path] = []
    for name in _GENERATED_RESULT_DIRS:
        target = out_dir / name
        if target.is_dir():
            targets.append(target)
    for name in _GENERATED_RESULT_FILES:
        target = out_dir / name
        if target.exists():
            targets.append(target)
    targets.extend(
        child for child in out_dir.iterdir()
        if child.is_file() and _SNAPSHOT_FRAME_RE.match(child.name))
    for target in targets:
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


def _execute_solver(cfg: "SWERunConfig", run_dir, run_config: dict) -> None:
    """Heavy numerical portion of a run: build planet + model, drive the solver."""
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.run.bve.io import (RUN_STATUS_RUNNING,
                                              write_run_manifest)
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic
    from planetary_sandbox.run.swe.runner import run_swe

    out_dir = run_dir.path
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=cfg.day_hours,
            radius_earth_units=cfg.radius_earth_units),
        grid_resolution=cfg.resolution,
        l_max=cfg.lmax,
        product_quadrature="fine",
        grid_type=cfg.grid,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
    )
    model = ShallowWaterModel(planet, gravity=cfg.gravity,
                              mean_depth=cfg.mean_depth_m)
    state0 = make_swe_ic(cfg.scenario, model)

    # Rewrite manifest now that we know the backend/product-sampling
    # provenance. Status stays 'running' until the runner returns.
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       numerics=planet.so.backend.describe("fine"),
                       status=RUN_STATUS_RUNNING, notes=SWE_MANIFEST_NOTES)

    run_swe(model=model,
            state0=state0,
            dt_snapshots=cfg.dt_snapshots,
            t_end_days=cfg.duration_days,
            out_dir=out_dir,
            figure_metadata=run_dir.figure_metadata(),
            snapshot_times=cfg.snapshot_times_seconds(),
            plots=cfg.plots,
            snapshot_mode=cfg.snapshot_mode,
            scenario=cfg.scenario)


def execute_run(cfg: "SWERunConfig") -> int:
    """Execute one resolved shallow-water run inside the shared lifecycle."""
    return execute_with_provenance(
        cfg,
        solver=lambda c, run_dir, run_config: _execute_solver(
            c, run_dir, run_config),
        clean_artifacts=lambda out_dir: _clean_overwrite_artifacts(out_dir),
        resolve_base_dir=lambda out: _resolve_writable_base_dir(out),
        notes=SWE_MANIFEST_NOTES)
