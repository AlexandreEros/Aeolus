from __future__ import annotations

import numpy as np
import cupy as cp
import pathlib
from typing import Tuple

from planetary_sandbox.planet import Planet
from .barotropic_vorticity import BarotropicVorticity, BarotropicState
from ...viz.vorticity_viewer import VorticityViewer

def run_bve(*, planet, zeta0_lm: cp.ndarray, dt: float, t_end_days: float, out_dir: pathlib.Path, scenario: str = "two_vortices") -> int:
    state = BarotropicState(coeffs=zeta0_lm)
    model = BarotropicVorticity(planet, scenario=scenario)

    t = 0.0
    t_end = t_end_days * 86400.0
    step = 0
    ovarall_step = 0

    all_zeta_lm = []
    vorticity_grid_snapshot_list = []

    snap_every = max(1, int((6*3600) / dt))  # every 6 hours by default

    while t <= t_end:
        if step % snap_every == 0:
            all_zeta_lm.append(cp.copy(state.coeffs))
            print(f"Time: {t/3600.0:8.2f} hrs | Step: {ovarall_step} ")

            # Dump ζ on grid for plotting
            zeta_grid = planet.sh.inv_transform(state.coeffs)
            vorticity_grid_snapshot_list.append(cp.copy(zeta_grid))
            
            # # Save as numpy array
            # cp.save(out_dir / f"zeta_{t/3600.0:04f}.npy", cp.asarray(zeta_grid))
        
        state = rk4_step(model, state, t, dt)
        t += dt
        step += 1
        ovarall_step += 1
    
    all_zeta_lm = cp.array(all_zeta_lm)
    np.save(out_dir / "vorticity_coeffs.npy", cp.asnumpy(all_zeta_lm))
    
    all_vorticity_grid: np.ndarray = cp.array(vorticity_grid_snapshot_list)
    np.save(out_dir / "vorticity_grid.npy", all_vorticity_grid)

    # Create viewer and plot summary

    snapshot_times = [dt * snap_every * i / 3600 for i in range(all_vorticity_grid.shape[0])]

    viewer = VorticityViewer(planet,
                             scenario=scenario,
                             vorticity_snapshots=cp.asnumpy(all_vorticity_grid),
                             times=np.array(snapshot_times))

    # Generate individual snapshot plots for debugging
    viewer.plot_all_snapshots(scenario=scenario, out_dir=out_dir)

    # Generate summary plot
    fig = viewer.plot_summary()
    fig.savefig(out_dir / "bve_summary.png", dpi=200)

    return 0
    


def rk4_step(model: BarotropicVorticity, y: BarotropicState, t: float, dt: float, forcing_coeffs=None) -> BarotropicState:
    k1 = model.tendency(y, forcing_coeffs)
    k2 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k1), forcing_coeffs)
    k3 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k2), forcing_coeffs)
    k4 = model.tendency(BarotropicState(y.coeffs + dt*k3), forcing_coeffs)
    return BarotropicState(y.coeffs + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))

def rhs(planet: Planet,
        zeta_c: cp.ndarray, 
        forcing_coeffs: cp.ndarray,
        nu: float) -> cp.ndarray[Tuple[int, int], cp.complex128]:
    """
    Compute dζ/dt in spectral space

    Parameters
    ----------
    planet : Planet
        Planet object containing SH and SO
    zeta_c : cp.ndarray
        Current vorticity coefficients in spectral space for a planet
    forcing_coeffs : cp.ndarray, optional
        Forcing in spectral space
    nu : float
        Viscosity coefficient

    Returns
    -------
    zeta_new_coeffs : cp.ndarray
        dζ/dt in spectral space
    """

    sh = planet.sh
    so = planet.so
    grid = planet.grid
    f = 2.0 * planet.params.angular_velocity * sh.sin_phi  # (n_lat,)
    plNf = f[:, None] * cp.ones((grid.num_lon,))[None, :]  # (n_lat, n_lon)

    # Get stream function
    psi_c = so.inv_laplacian(zeta_c)

    # Get absolute vorticity η = ζ + f
    zeta_grid = sh.inv_transform(zeta_c)
    eta_grid = zeta_grid + plNf
    eta_c = sh.transform(eta_grid)

    # Compute advection: -J(ψ, η)
    jacobian_grid = so.jacobian_spectral(psi_c, eta_c, grid)
    advection_c = sh.transform(-jacobian_grid)

    # Compute diffusion: ν∇²ζ
    diffusion_c = nu * so.lap_eigs[:, None] * zeta_c

    # Total tendency
    dzeta_dt = advection_c + diffusion_c + forcing_coeffs

    return dzeta_dt