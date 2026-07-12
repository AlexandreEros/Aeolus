from __future__ import annotations

import numpy as np
import cupy as cp
import pathlib
from typing import Tuple

from planetary_sandbox.planet import Planet
from .barotropic_vorticity import BarotropicVorticity, BarotropicState
from .diagnostics import DiagnosticsRecorder, plot_diagnostics
from ...viz.vorticity_viewer import VorticityViewer

def run_bve(planet: Planet,
            zeta0_lm: cp.ndarray,
            dt_snapshots: float,
            t_end_days: float,
            out_dir: pathlib.Path,
            viscosity: float,
            scenario: str = "two_vortices",
            figure_metadata: dict | None = None) -> int:
    state = BarotropicState(coeffs=zeta0_lm)
    model = BarotropicVorticity(planet, scenario=scenario, viscosity=viscosity)

    # CFL-based timestep from initial max speed and minimum edge length.
    C = 0.5 # CFL safety factor
    min_edge_length = getattr(planet.grid, "min_edge_length", None)
    psi0_lm = planet.so.inv_laplacian(zeta0_lm)
    u0, v0 = planet.so.velocity_from_streamfunction(psi0_lm)
    max_speed = float(cp.max(cp.sqrt(u0**2 + v0**2)).item())
    if min_edge_length and max_speed > 0:
        dt_cfl = C * min_edge_length / max_speed
    else:
        dt_cfl = 600 # idk

    t = 0.0
    t_end = t_end_days * 86400.0
    ovarall_step = 0

    # Scalar diagnostics recorded every accepted step, straight from the
    # spectral state (not the plotting fields). Cheap; append-only.
    recorder = DiagnosticsRecorder(
        sh=planet.sh, so=planet.so, grid=planet.grid,
        radius=planet.params.radius,
        omega=planet.params.angular_velocity,
        out_dir=out_dir,
    )
    recorder.record(t, state.coeffs, dt=0.0, step=0)

    all_zeta_lm = []
    vorticity_grid_snapshot_list = []

    snapshot_tol = 1e-6 * dt_snapshots
    time_to_snapshot = 0.0
    snapshot_times = []

    while t <= t_end + snapshot_tol:
        if time_to_snapshot <= snapshot_tol:
            all_zeta_lm.append(cp.copy(state.coeffs))
            print(f"Time: {t/3600.0:8.2f} hrs | Step: {ovarall_step} ")
            snapshot_times.append(t / 3600.0)

            # Dump ζ on grid for plotting
            zeta_grid = planet.sh.inv_transform(state.coeffs)
            vorticity_grid_snapshot_list.append(cp.copy(zeta_grid))
            
            # # Save as numpy array
            # cp.save(out_dir / f"zeta_{t/3600.0:04f}.npy", cp.asarray(zeta_grid))
            time_to_snapshot = dt_snapshots
        
        remaining = t_end - t
        if remaining <= snapshot_tol:
            break
        dt_step = min(dt_cfl, time_to_snapshot, remaining)
        if dt_step <= 0:
            break
        state = rk4_step(model, state, t, dt_step)
        t += dt_step
        time_to_snapshot = max(0.0, time_to_snapshot - dt_step)
        ovarall_step += 1
        recorder.record(t, state.coeffs, dt=dt_step, step=ovarall_step)

    recorder.close()
    try:
        plot_diagnostics(out_dir, metadata=figure_metadata)
    except Exception as err:
        # Plotting must never take down a finished run; the CSV/npz survive.
        print(f"Diagnostics plotting failed (data preserved): {err}")

    all_zeta_lm = cp.array(all_zeta_lm)
    np.save(out_dir / "vorticity_coeffs.npy", cp.asnumpy(all_zeta_lm))
    
    all_vorticity_grid: np.ndarray = cp.array(vorticity_grid_snapshot_list)
    np.save(out_dir / "vorticity_grid.npy", all_vorticity_grid)

    # Create viewer and plot summary

    if snapshot_times:
        snapshot_times_arr = np.array(snapshot_times, dtype=np.float64)
    else:
        snapshot_times_arr = np.empty((0,), dtype=np.float64)

    viewer = VorticityViewer(planet,
                             scenario=scenario,
                             vorticity_snapshots=cp.asnumpy(all_vorticity_grid),
                             times=snapshot_times_arr)

    # Generate individual snapshot plots for debugging
    viewer.plot_all_snapshots(scenario=scenario, out_dir=out_dir,
                              metadata=figure_metadata)

    # Generate summary plot
    fig = viewer.plot_summary()
    fig.savefig(out_dir / "bve_summary.png", dpi=200, metadata=figure_metadata)

    return 0
    


def rk4_step(model: BarotropicVorticity, y: BarotropicState, t: float, dt: float, forcing_coeffs=None) -> BarotropicState:
    k1 = model.tendency(y, forcing_coeffs)
    k2 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k1), forcing_coeffs)
    k3 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k2), forcing_coeffs)
    k4 = model.tendency(BarotropicState(y.coeffs + dt*k3), forcing_coeffs)
    return BarotropicState(y.coeffs + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
