from __future__ import annotations

import numpy as np
import cupy as cp
import pathlib
from collections import deque
from typing import Sequence, Tuple

from planetary_sandbox.planet import Planet
from .barotropic_vorticity import BarotropicVorticity, BarotropicState
from .config import (PLOT_TYPES, SNAPSHOT_TIME_TOLERANCE_SECONDS,
                     interval_snapshot_times)
from .diagnostics import DiagnosticsRecorder, plot_diagnostics
from ...viz.vorticity_viewer import VorticityViewer

def run_bve(planet: Planet,
            zeta0_lm: cp.ndarray,
            dt_snapshots: float | None,
            t_end_days: float,
            out_dir: pathlib.Path,
            viscosity: float,
            scenario: str = "two_vortices",
            figure_metadata: dict | None = None,
            snapshot_times: Sequence[float] | None = None,
            plots: Sequence[str] | None = None) -> int:
    """Integrate the BVE and persist snapshots, diagnostics, and figures.

    ``snapshot_times`` is the authoritative output schedule (seconds). When
    omitted, it is derived from ``dt_snapshots`` with the historical interval
    semantics, so legacy callers are unaffected. ``plots`` selects which
    image products to render (subset of ``config.PLOT_TYPES``); None keeps
    the historical behavior of rendering everything. Field-snapshot
    persistence and per-step diagnostics are independent of ``plots``.
    """
    state = BarotropicState(coeffs=zeta0_lm)
    model = BarotropicVorticity(planet, scenario=scenario, viscosity=viscosity)

    t_end = t_end_days * 86400.0
    if snapshot_times is None:
        if dt_snapshots is None:
            raise ValueError("provide snapshot_times or dt_snapshots")
        snapshot_times = interval_snapshot_times(float(dt_snapshots), t_end)
    plots = tuple(PLOT_TYPES) if plots is None else tuple(plots)

    # CFL-based timestep from initial max speed and the geometry-owned
    # length scale (geodesic: min edge length; lat-lon: min meridional
    # spacing — see the geometry's cfl_length_scale docstring).
    C = 0.5 # CFL safety factor
    # GridGeometry guarantees cfl_length_scale (base returns None; geodesic
    # routes min_edge_length through it). None/0 falls through to the fixed
    # default below.
    length_scale = getattr(planet.grid, "cfl_length_scale", None)
    psi0_lm = planet.so.inv_laplacian(zeta0_lm)
    u0, v0 = planet.so.velocity_from_streamfunction(psi0_lm)
    max_speed = float(cp.max(cp.sqrt(u0**2 + v0**2)).item())
    if length_scale and max_speed > 0:
        dt_cfl = C * length_scale / max_speed
    else:
        dt_cfl = 600 # idk

    t = 0.0
    step = 0

    # Scalar diagnostics recorded every accepted step, straight from the
    # spectral state (not the plotting fields). Cheap; append-only.
    # Independent of the snapshot schedule and of plot selection.
    recorder = DiagnosticsRecorder(
        sh=planet.sh, so=planet.so, grid=planet.grid,
        radius=planet.params.radius,
        omega=planet.params.angular_velocity,
        out_dir=out_dir,
    )
    recorder.record(t, state.coeffs, dt=0.0, step=0)

    all_zeta_lm = []
    vorticity_grid_snapshot_list = []
    stored_times_hours = []

    tol = SNAPSHOT_TIME_TOLERANCE_SECONDS
    pending = deque(sorted(s for s in snapshot_times if s <= t_end + tol))

    while True:
        # Store every schedule entry the integrator has reached. Steps are
        # clipped to land exactly on schedule times, so this matches exactly
        # (the tolerance only absorbs float accumulation noise).
        while pending and t >= pending[0] - tol:
            pending.popleft()
            all_zeta_lm.append(cp.copy(state.coeffs))
            print(f"Time: {t/3600.0:8.2f} hrs | Step: {step} ")
            stored_times_hours.append(t / 3600.0)
            # Dump ζ on grid for plotting
            zeta_grid = planet.sh.inv_transform(state.coeffs)
            vorticity_grid_snapshot_list.append(cp.copy(zeta_grid))

        if t >= t_end - tol:
            break
        next_stop = pending[0] if pending else t_end
        dt_step = min(dt_cfl, next_stop - t, t_end - t)
        if dt_step <= 0:
            break
        state = rk4_step(model, state, t, dt_step)
        t += dt_step
        step += 1
        recorder.record(t, state.coeffs, dt=dt_step, step=step)

    recorder.close()

    # Field-snapshot persistence is independent of plot selection.
    all_zeta_lm = cp.array(all_zeta_lm)
    np.save(out_dir / "vorticity_coeffs.npy", cp.asnumpy(all_zeta_lm))

    all_vorticity_grid: np.ndarray = cp.array(vorticity_grid_snapshot_list)
    np.save(out_dir / "vorticity_grid.npy", all_vorticity_grid)

    # Image products, in the fixed order of PLOT_TYPES:
    # diagnostics -> snapshots -> summary.
    if "diagnostics" in plots:
        try:
            plot_diagnostics(out_dir, metadata=figure_metadata)
        except Exception as err:
            # Plotting must never take down a finished run; the CSV/npz survive.
            print(f"Diagnostics plotting failed (data preserved): {err}")

    wants_viewer = ("snapshots" in plots or "summary" in plots)
    if wants_viewer and len(vorticity_grid_snapshot_list) > 0:
        if stored_times_hours:
            snapshot_times_arr = np.array(stored_times_hours, dtype=np.float64)
        else:
            snapshot_times_arr = np.empty((0,), dtype=np.float64)

        viewer = VorticityViewer(planet,
                                 scenario=scenario,
                                 vorticity_snapshots=cp.asnumpy(all_vorticity_grid),
                                 times=snapshot_times_arr)

        if "snapshots" in plots:
            # Individual snapshot plots for debugging
            viewer.plot_all_snapshots(scenario=scenario, out_dir=out_dir,
                                      metadata=figure_metadata)

        if "summary" in plots:
            fig = viewer.plot_summary()
            fig.savefig(out_dir / "bve_summary.png", dpi=200,
                        metadata=figure_metadata)

    return 0
    


def rk4_step(model: BarotropicVorticity, y: BarotropicState, t: float, dt: float, forcing_coeffs=None) -> BarotropicState:
    k1 = model.tendency(y, forcing_coeffs)
    k2 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k1), forcing_coeffs)
    k3 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k2), forcing_coeffs)
    k4 = model.tendency(BarotropicState(y.coeffs + dt*k3), forcing_coeffs)
    return BarotropicState(y.coeffs + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
