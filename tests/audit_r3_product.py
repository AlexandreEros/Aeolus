"""Characterization audit for docs/KNOWN_RISKS.md R-3: which operation in the
pseudospectral nonlinear product introduces the dominant invariant defect?

NO production numerics are modified; all variants live in this script.

Factorized tendency variants (all use the post-R5 spectral eta = zeta + f_lm):

  A "current"    jacobian(dealias=True) -> grid -> transform(-J)
                 (production path: spectral truncation at cut=2L/3 inside the
                  jacobian, THEN an extra synthesis + re-analysis round trip)
  B "trunc-once" jacobian(dealias=False) -> transform(-J) -> spectral truncation
                 (same truncation, no extra round trip)     A-B = extra round trip
  C "no-trunc"   jacobian(dealias=False) -> transform(-J)   B-C = the truncation
  D "overres"    derivative fields synthesized on the res-(r+1) point set at the
                 SAME l_max; pointwise product there; analyzed with the fine
                 quadrature; truncated like B.              D-B = product-analysis
                                                            quadrature/aliasing

Case matrix:
  res4: A, B, C, D; plus variant A with the RH4 pattern tilted 30 and 60 deg
        (rotations preserve degree content l in {1,5} but change alignment to
        the icosahedral grid; tilted RH4 is NOT a rigid solution, so tilted
        drift magnitudes are indicative of orientation sensitivity only).
  res5: A, B, C (D would need a res-6 grid whose pure-Python construction is
        prohibitive on this machine; not required for attribution).

Metrics: E and Z_abs drift at 5 days (fixed dt from initial CFL, N identical
across variants at a given resolution); spurious-energy fraction (standard RH4
stays exactly in l in {1,5}; energy elsewhere is numerical corruption);
instantaneous discrete dE/dt and dZ_abs/dt at t=0 and at a 20-step evolved
state.

Run:
    python tests/audit_r3_product.py res4
    python tests/audit_r3_product.py res5
"""
from __future__ import annotations

import pathlib
import sys
import time as wallclock
import warnings

import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation

warnings.simplefilter("ignore")

from planetary_sandbox.numerics import (
    GeodesicGridGeometry,
    GeodesicSphericalHarmonics,
    SpectralOperators,
)
from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.barotropic_vorticity import BarotropicState, BarotropicVorticity
from planetary_sandbox.run.bve.runner import rk4_step
from planetary_sandbox.run.bve.diagnostics import spectral_diagnostics

L_MAX = 21
DAY_HOURS = 24.0
T_END_DAYS = 5.0
CUT = (2 * L_MAX) // 3          # match production truncation
OUT = pathlib.Path("runs") / "r3-product"


# ---------------------------------------------------------------------------
# ICs
# ---------------------------------------------------------------------------

def rh4_field(lat: cp.ndarray, lon: cp.ndarray) -> cp.ndarray:
    nu = K = 7.848e-6
    sinphi, cosphi = cp.sin(lat), cp.cos(lat)
    return 2.0 * nu * sinphi - 30.0 * K * sinphi * cosphi**4 * cp.sin(4.0 * lon)


def rh4_tilted(grid: GeodesicGridGeometry, tilt_deg: float) -> cp.ndarray:
    """RH4 pattern evaluated in a frame tilted about the y-axis."""
    pts = np.asarray(grid.points, dtype=np.float64)
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    q = Rotation.from_euler("y", -tilt_deg, degrees=True).apply(pts)
    lat = cp.asarray(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
    lon = cp.asarray(np.arctan2(q[:, 1], q[:, 0]))
    return rh4_field(lat, lon)


# ---------------------------------------------------------------------------
# Variant tendencies
# ---------------------------------------------------------------------------

class VariantModel:
    """rk4_step-compatible model with a factorized nonlinear product."""

    def __init__(self, planet: Planet, variant: str,
                 fine_sh=None, fine_so=None):
        self.variant = variant
        self.sh, self.so = planet.sh, planet.so
        self.fine_sh, self.fine_so = fine_sh, fine_so
        base = BarotropicVorticity(planet, viscosity=0.0)
        self.f_lm = base.f_lm
        self.laplacian_eig = base.laplacian_eig
        self._to_psi = base.vorticity_to_streamfunction

    def tendency(self, vrt_state: BarotropicState, forcing_coeffs) -> cp.ndarray:
        zeta_c = vrt_state.coeffs
        psi_c = self._to_psi(vrt_state)
        eta_c = zeta_c + self.f_lm

        v = self.variant
        if v == "A":       # production: internal truncation + extra round trip
            J = self.so.jacobian_pseudospectral(psi_c, eta_c, dealias=True)
            adv = self.sh.transform(-J)
        elif v in ("B", "C"):
            J = self.so.jacobian_pseudospectral(psi_c, eta_c, dealias=False)
            adv = self.sh.transform(-J)
            if v == "B":
                adv[CUT + 1:, :] = 0.0
        elif v == "D":     # product on the fine point set, fine quadrature
            J = self.fine_so.jacobian_pseudospectral(psi_c, eta_c, dealias=False)
            adv = self.fine_sh.transform(-J)
            adv[CUT + 1:, :] = 0.0
        else:
            raise ValueError(v)

        dz = adv
        dz[0, :] = 0.0
        return dz


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def production_rates(zeta_lm: cp.ndarray, dz: cp.ndarray, f_lm: cp.ndarray,
                     R: float) -> tuple[float, float]:
    """Instantaneous discrete dE/dt and dZ_abs/dt (both exactly 0 for the PDE)."""
    l_idx, m_idx = cp.indices(zeta_lm.shape)
    valid = m_idx <= l_idx
    mult = cp.where(m_idx == 0, 1.0, 2.0) * valid
    ll1 = cp.where(l_idx >= 1, l_idx * (l_idx + 1.0), 1.0)
    mask_l = (l_idx >= 1)
    dEdt = float(R**4 * (mult * mask_l * cp.real(cp.conj(zeta_lm) * dz) / ll1).sum())
    eta = zeta_lm + f_lm
    dZdt = float(R**2 * (mult * cp.real(cp.conj(eta) * dz)).sum())
    return dEdt, dZdt


def spurious_energy_fraction(d: dict) -> float:
    """Energy outside l in {1,5} / total (standard RH4 lives exactly in {1,5})."""
    E_l = d["energy_l"]
    total = E_l.sum()
    return float((total - E_l[1] - E_l[5]) / total) if total > 0 else 0.0


def run_case(planet: Planet, zeta0_grid: cp.ndarray, model: VariantModel,
             N: int, dt: float) -> dict:
    sh = planet.sh
    R, om = planet.params.radius, planet.params.angular_velocity
    state = BarotropicState(sh.transform(zeta0_grid))
    d0 = spectral_diagnostics(state.coeffs, R, om)
    t0 = wallclock.perf_counter()
    for _ in range(N):
        state = rk4_step(model, state, 0.0, dt)
    cp.cuda.Stream.null.synchronize()
    wall = wallclock.perf_counter() - t0
    d1 = spectral_diagnostics(state.coeffs, R, om)
    return dict(
        E_drift=(d1["energy"] - d0["energy"]) / abs(d0["energy"]),
        Z_drift=(d1["enstrophy_abs"] - d0["enstrophy_abs"]) / d0["enstrophy_abs"],
        spur0=spurious_energy_fraction(d0),
        spurT=spurious_energy_fraction(d1),
        wall=wall,
        final=state.coeffs,
    )


def main(part: str):
    res = 4 if part == "res4" else 5
    print(f"[audit_r3_product] part={part} (res={res}, l_max={L_MAX}, "
          f"T={T_END_DAYS} d, cut={CUT})")

    params = PlanetaryParameters.from_earth_like(day_hours=DAY_HOURS)
    planet = Planet.generate(params=params, grid_resolution=res, l_max=L_MAX)
    R, om = params.radius, params.angular_velocity
    sh, so, grid = planet.sh, planet.so, planet.grid

    # fine grid for variant D (res4 part only)
    fine_sh = fine_so = None
    variants = ["A", "B", "C"]
    if part == "res4":
        print("[audit_r3_product] building res-5 fine grid for variant D ...")
        grid5 = GeodesicGridGeometry(resolution=5, radius=R)
        fine_sh = GeodesicSphericalHarmonics(grid5, L_MAX, weights="voronoi")
        fine_so = SpectralOperators(fine_sh, R, grid5)
        variants.append("D")

    # fixed dt and N shared by ALL variants at this resolution
    zeta0 = rh4_field(cp.asarray(grid.point_latitudes),
                      cp.asarray(grid.point_longitudes))
    z0_lm = sh.transform(zeta0)
    psi0 = so.inv_laplacian(z0_lm)
    u0, v0 = so.velocity_from_streamfunction(psi0)
    dt_cfl = 0.5 * grid.min_edge_length / float(cp.max(cp.sqrt(u0**2 + v0**2)))
    T = T_END_DAYS * 86400.0
    N = max(1, int(round(T / dt_cfl)))
    dt = T / N
    print(f"[audit_r3_product] N={N} steps, dt={dt:.1f} s")

    lines = [f"# R-3 product attribution — part {part} "
             f"(res={res}, l_max={L_MAX}, cut={CUT}, N={N}, dt={dt:.1f} s, "
             f"T={T_END_DAYS} d, IC=rh4, nu=0, day={DAY_HOURS} h)", ""]

    # ---- instantaneous production at t0 and at an evolved state -----------
    model_A = VariantModel(planet, "A", fine_sh, fine_so)
    st = BarotropicState(cp.copy(z0_lm))
    for _ in range(20):
        st = rk4_step(model_A, st, 0.0, dt)
    z20_lm = st.coeffs
    E0 = spectral_diagnostics(z0_lm, R, om)["energy"]
    Zr0 = spectral_diagnostics(z0_lm, R, om)["enstrophy_rel"]

    lines.append("Instantaneous discrete production (per day, normalized):")
    lines.append(f"{'variant':>8} {'state':>8} {'dE/dt / E':>14} {'dZa/dt / Zrel':>14}")
    for v in variants:
        m = VariantModel(planet, v, fine_sh, fine_so)
        for tag, zlm in [("t0", z0_lm), ("20-step", z20_lm)]:
            dz = m.tendency(BarotropicState(zlm), None)
            dE, dZ = production_rates(zlm, dz, m.f_lm, R)
            lines.append(f"{v:>8} {tag:>8} {dE*86400/E0:>+14.3e} "
                         f"{dZ*86400/Zr0:>+14.3e}")
    lines.append("")

    # ---- 5-day integrations, all variants, same N/dt ----------------------
    lines.append("5-day integrations (standard orientation):")
    lines.append(f"{'variant':>8} {'E drift':>12} {'Zabs drift':>12} "
                 f"{'spurE(T)/E':>12} {'wall s':>8}")
    results = {}
    for v in variants:
        m = VariantModel(planet, v, fine_sh, fine_so)
        r = run_case(planet, zeta0, m, N, dt)
        results[v] = r
        lines.append(f"{v:>8} {r['E_drift']:>+12.4e} {r['Z_drift']:>+12.4e} "
                     f"{r['spurT']:>12.3e} {r['wall']:>8.1f}")
    lines.append("")

    # ---- orientation sensitivity (res4 only, variant A) --------------------
    if part == "res4":
        lines.append("Orientation sensitivity (variant A; tilted RH4 is not a "
                     "rigid solution — magnitudes indicative only):")
        lines.append(f"{'tilt':>6} {'E drift':>12} {'Zabs drift':>12} {'spurE(T)/E':>12}")
        for tilt in [0.0, 30.0, 60.0]:
            z_t = rh4_tilted(grid, tilt) if tilt else zeta0
            m = VariantModel(planet, "A", fine_sh, fine_so)
            r = run_case(planet, z_t, m, N, dt)
            lines.append(f"{tilt:>6.0f} {r['E_drift']:>+12.4e} "
                         f"{r['Z_drift']:>+12.4e} {r['spurT']:>12.3e}")
        lines.append("")

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_{part}.txt").write_text(summary, encoding="utf-8")
    np.savez(OUT / f"final_states_{part}.npz",
             **{f"zeta_{v}": cp.asnumpy(results[v]["final"]) for v in results})
    print(f"[audit_r3_product] wrote {OUT}/summary_{part}.txt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "res4")
