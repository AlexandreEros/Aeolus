"""Standalone audit of the R-5 mechanism (docs/KNOWN_RISKS.md R-5).

Measures, on a fixed representative state, how the production tendency's
grid-space construction of absolute vorticity

    eta = transform( inv_transform(zeta) + f_grid )

corrupts the state relative to the exact spectral construction

    eta = zeta + f_lm,    (f_lm)_{1,0} = 2*Omega*sqrt(4*pi/3), else 0,

and decomposes the error into (a) the zeta round-trip projection error and
(b) the f transform leakage. Also compares the instantaneous kinetic-energy
production rate dE/dt of the two eta constructions through the full tendency.

This is an investigation script, not a collected test (name lacks test_
prefix) and not part of per-timestep runtime diagnostics. Run:

    python tests/audit_r5_mechanism.py
"""
import numpy as np
import cupy as cp

from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.barotropic_vorticity import BarotropicVorticity, BarotropicState
from planetary_sandbox.run.bve.initial_conditions import make_ic
from planetary_sandbox.run.bve.diagnostics import _mode_power

import warnings
warnings.simplefilter("ignore")

RES, L_MAX, DAY_HOURS = 4, 21, 24.0


def energy_production_rate(zeta_lm, dzeta_dt_lm, R):
    """dE/dt = R^4 sum' Re(conj(zeta) * dzeta/dt) / (l(l+1)), with m-doubling."""
    l_max = zeta_lm.shape[0] - 1
    l_idx, m_idx = cp.indices(zeta_lm.shape)
    valid = (m_idx <= l_idx) & (l_idx >= 1)
    mult = cp.where(m_idx == 0, 1.0, 2.0) * valid
    ll1 = cp.where(l_idx >= 1, l_idx * (l_idx + 1.0), 1.0)
    integrand = mult * cp.real(cp.conj(zeta_lm) * dzeta_dt_lm) / ll1
    return float(R**4 * integrand.sum())


def main():
    params = PlanetaryParameters.from_earth_like(day_hours=DAY_HOURS)
    planet = Planet.generate(params=params, grid_resolution=RES, l_max=L_MAX)
    sh, so, grid = planet.sh, planet.so, planet.grid
    R, Omega = params.radius, params.angular_velocity
    model = BarotropicVorticity(planet, viscosity=0.0)

    zeta_lm = sh.transform(make_ic("two_vortices", planet))
    f_grid = 2.0 * Omega * cp.sin(cp.asarray(grid.point_latitudes))

    # Exact spectral f
    f_lm_exact = cp.zeros_like(zeta_lm)
    a10 = 2.0 * Omega * np.sqrt(4.0 * np.pi / 3.0)
    f_lm_exact[1, 0] = a10

    print(f"config: res={RES}, l_max={L_MAX}, rotating ({DAY_HOURS}h day)")
    print(f"|zeta| scale: max coeff {float(cp.abs(zeta_lm).max()):.3e}, "
          f"f a10 = {a10:.6e}  (f dwarfs zeta by ~{a10/float(cp.abs(zeta_lm).max()):.0f}x)\n")

    # ---- 1. transform(f_grid) vs exact ----------------------------------
    f_lm_t = sh.transform(f_grid)
    err10 = complex(f_lm_t[1, 0] - a10)
    leak = f_lm_t.copy(); leak[1, 0] = 0.0
    leak_norm = float(cp.sqrt(_mode_power(leak).sum()))
    f_recon_err = float(cp.max(cp.abs(sh.inv_transform(f_lm_t) - f_grid)) / cp.max(cp.abs(f_grid)))
    print("1. transform(f_grid) vs exact spectral f:")
    print(f"   a10 error:      {abs(err10):.3e}  (relative {abs(err10)/a10:.3e})")
    print(f"   leakage ||.||:  {leak_norm:.3e}  (relative to a10: {leak_norm/a10:.3e})")
    print(f"   grid recon err: {f_recon_err:.3e} (max, relative)\n")

    # ---- 2. eta error decomposition -------------------------------------
    eta_exact = zeta_lm + f_lm_exact
    zeta_rt = sh.transform(sh.inv_transform(zeta_lm))          # zeta round trip alone
    eta_old = sh.transform(sh.inv_transform(zeta_lm) + f_grid) # production construction

    def norm(x):
        return float(cp.sqrt(_mode_power(x).sum()))

    zeta_norm = norm(zeta_lm)
    e_rt_zeta = norm(zeta_rt - zeta_lm)
    e_f = norm(f_lm_t - f_lm_exact)
    e_comb = norm(eta_old - eta_exact)
    print("2. eta construction error (norms of coefficient perturbations):")
    print(f"   ||zeta||:                        {zeta_norm:.4e}")
    print(f"   zeta round-trip error:           {e_rt_zeta:.4e}  ({e_rt_zeta/zeta_norm:.2%} of zeta)")
    print(f"   f transform error:               {e_f:.4e}  ({e_f/zeta_norm:.2%} of zeta)")
    print(f"   combined (old eta - exact eta):  {e_comb:.4e}  ({e_comb/zeta_norm:.2%} of zeta)")
    print(f"   -> f-error / zeta-rt-error ratio: {e_f/max(e_rt_zeta,1e-300):.1f}x\n")

    # ---- 3. instantaneous dE/dt through the full tendency ----------------
    dz_old = model.tendency(BarotropicState(zeta_lm), None)   # production (old on parent)
    # tendency with exact spectral eta, everything else identical:
    psi_c = model.vorticity_to_streamfunction(BarotropicState(zeta_lm))
    J = so.jacobian_pseudospectral(psi_c, eta_exact)
    dz_new = sh.transform(-J)
    dz_new[0, :] = 0.0

    E = 0.5 * energy_production_rate(zeta_lm, zeta_lm, R)  # = E itself (reuse formula)
    r_old = energy_production_rate(zeta_lm, dz_old, R)
    r_new = energy_production_rate(zeta_lm, dz_new, R)
    day = 86400.0
    print("3. instantaneous energy production rate (should be ~0, inviscid):")
    print(f"   E = {E:.4e} m^4 s^-2")
    print(f"   production tendency:      dE/dt = {r_old:+.4e}  -> {r_old*day/E:+.3e} /day")
    print(f"   spectral-eta tendency:    dE/dt = {r_new:+.4e}  -> {r_new*day/E:+.3e} /day")


if __name__ == "__main__":
    main()
