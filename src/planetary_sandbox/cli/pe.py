"""``aeolus run pe`` — the dry primitive-equation run executor.

Heavy imports (CuPy, matplotlib) happen inside :func:`_execute_solver`, so
parsing and ``--help`` never initialize CUDA. The provenance lifecycle is the
same one the BVE and SWE commands share (``cli/run_lifecycle.py``): the PE run
capsule uses the identical run-id, atomic-write, status, latest-pointer, and
manifest machinery — there is no weaker PE-specific persistence path.
"""
from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING

from planetary_sandbox.cli.run_lifecycle import (execute_with_provenance,
                                                 resolve_writable_base_dir)

if TYPE_CHECKING:
    from planetary_sandbox.run.pe.config import PERunConfig


#: Generated *result* files removed when --overwrite reuses a directory
#: (config.json / manifest.json are lifecycle-managed, never swept).
_GENERATED_RESULT_FILES: tuple[str, ...] = (
    "pe_coeffs.npy",
    "pe_snapshot_times.npy",
    "pe_summary.png",
)

_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
)

#: Manifest notes describing the dry primitive-equation solver and the exact
#: stored-array contract. The scientific run_config (nlev, sigma_interfaces,
#: dt_seconds, r_dry/cp_dry, IC parameters, snapshot_times, ...) is written
#: separately by the shared lifecycle; these notes document the invariants a
#: reader needs to interpret the capsule.
PE_MANIFEST_NOTES = {
    "equations": "dry hydrostatic primitive equations in vorticity-divergence "
                 "form on a sigma-coordinate vertical grid (see "
                 "physics/primitive_equations.py and "
                 "docs/PRIMITIVE_EQUATIONS_DESIGN.md)",
    "coefficient_ordering": "pe_coeffs.npy has shape "
                            "(n_snapshots, 3*nlev+1, l_max+1, l_max+1); axis 1 "
                            "is [zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, "
                            "ln_ps] top-to-bottom (K = nlev), and the trailing "
                            "two axes are the (degree, order) spherical-harmonic "
                            "coefficient block",
    "timestep_policy": "user-supplied FIXED timestep dt_seconds (no adaptive "
                       "CFL controller); individual steps are shortened only to "
                       "land exactly on requested output times and t_end. No "
                       "forcing, no hyperdiffusion/filters, no semi-implicit "
                       "terms, no adaptive timestepping",
    "diagnostics": "see run/pe/diagnostics.py module docstring; total_mass is "
                   "the integral of p_s dA (mass proxy), and no total-energy-"
                   "conservation claim is made",
}


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    return resolve_writable_base_dir(requested_out,
                                     fallback_prefix="aeolus-pe-")


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known PE-generated outputs before a reused run overwrites them.

    Strictly scoped to Aeolus-generated result artifacts, like the BVE/SWE
    variants; raises OverwriteCleanupError if any known stale artifact cannot
    be removed.
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


def _execute_solver(cfg: "PERunConfig", run_dir, run_config: dict) -> None:
    """Heavy numerical portion: build planet + PE model, drive the solver."""
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    from planetary_sandbox.run.bve.io import (RUN_STATUS_RUNNING,
                                              write_run_manifest)
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    from planetary_sandbox.run.pe.runner import run_pe

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
    sigma = SigmaGrid(cfg.sigma_interfaces_resolved())
    model = PrimitiveEquationsModel(planet, sigma, r_dry=cfg.r_dry,
                                    cp_dry=cfg.cp_dry)
    state0 = make_pe_ic(cfg.scenario, model, temperature=cfg.temperature,
                        surface_pressure=cfg.surface_pressure,
                        thermal_amplitude=cfg.thermal_amplitude)

    # Rewrite manifest now that we know the backend/product-sampling
    # provenance. Status stays 'running' until the runner returns.
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       numerics=planet.so.backend.describe("fine"),
                       status=RUN_STATUS_RUNNING, notes=PE_MANIFEST_NOTES)

    run_pe(model=model,
           state0=state0,
           dt_seconds=cfg.dt_seconds,
           t_end_days=cfg.duration_days,
           out_dir=out_dir,
           snapshot_times=cfg.snapshot_times_seconds(),
           snapshot_mode=cfg.snapshot_mode,
           dt_snapshots=cfg.dt_snapshots,
           figure_metadata=run_dir.figure_metadata(),
           plots=cfg.plots,
           scenario=cfg.scenario)


def execute_run(cfg: "PERunConfig") -> int:
    """Execute one resolved primitive-equation run inside the shared lifecycle."""
    return execute_with_provenance(
        cfg,
        solver=lambda c, run_dir, run_config: _execute_solver(
            c, run_dir, run_config),
        clean_artifacts=lambda out_dir: _clean_overwrite_artifacts(out_dir),
        resolve_base_dir=lambda out: _resolve_writable_base_dir(out),
        notes=PE_MANIFEST_NOTES)
