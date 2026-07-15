"""Backend parity: the same physics checks against geodesic AND lat-lon.

One parametrized fixture builds a comparable planet per backend
(geodesic res4 / lat-lon 32x64, both l_max=21, both 'fine' product
quadrature) and every test in this module runs on both. Covers: transform
round trip, spectral-operator correctness (solid-body Jacobian, J(a,a)=0,
zero-mean Jacobian, analytic velocity), RH4 1-day conservation, and
spectral-vs-grid diagnostics consistency.

Tolerances are set per backend where the numerics genuinely differ: the
lat-lon transform quadrature is exact (machine-precision round trips),
the geodesic one is a least-squares/Voronoi approximation.
"""
import numpy as np
import pytest

try:
    import cupy as cp

    _HAS_CUDA = cp.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/CuPy not available")

if _HAS_CUDA:
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.barotropic_vorticity import (
        BarotropicState,
        BarotropicVorticity,
    )
    from planetary_sandbox.run.bve.diagnostics import (
        grid_diagnostics,
        spectral_diagnostics,
    )
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import rk4_step

L_MAX = 21

# Per-backend accuracy envelopes (the lat-lon quadrature is exact; the
# geodesic one is approximate — see docs/KNOWN_RISKS.md R-2/R-3).
TOL = {
    "geodesic": {
        "roundtrip": 5e-2,
        "solid_body": 5e-3,
        "velocity": 5e-3,
        "diag_consistency": 5e-2,
    },
    "latlon": {
        "roundtrip": 1e-10,
        "solid_body": 1e-8,
        "velocity": 1e-8,
        "diag_consistency": 1e-9,
    },
}


@pytest.fixture(scope="module", params=["geodesic", "latlon"])
def backend_case(request):
    kind = request.param
    params = PlanetaryParameters.from_earth_like(day_hours=24.0)
    if kind == "geodesic":
        planet = Planet.generate(params=params, grid_resolution=4,
                                 l_max=L_MAX, product_quadrature="fine")
    else:
        planet = Planet.generate(params=params, grid_type="latlon",
                                 nlat=32, nlon=64,
                                 l_max=L_MAX, product_quadrature="fine")
    return kind, planet


def _flat(values):
    return values.ravel() if values.ndim > 1 else values


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_transform_roundtrip(backend_case):
    kind, planet = backend_case
    rng = np.random.default_rng(7)
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    for l in range(L_MAX + 1):
        a[l, 0] = 1e-5 * rng.standard_normal()
        for m in range(1, l + 1):
            a[l, m] = 1e-5 * (rng.standard_normal() + 1j * rng.standard_normal())
    back = planet.sh.transform(planet.sh.inv_transform(a))
    rel = float(cp.linalg.norm(back - a)) / float(cp.linalg.norm(a))
    assert rel < TOL[kind]["roundtrip"], f"[{kind}] round-trip {rel:.2e}"


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _solid_body_psi(planet, omega_sb=1e-5):
    R = planet.params.radius
    psi_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    psi_lm[1, 0] = -omega_sb * R**2 * np.sqrt(4.0 * np.pi / 3.0)
    return psi_lm


@pytest.mark.parametrize("l, m", [(3, 2), (8, 4), (12, 6)])
def test_jacobian_solid_body_advection(backend_case, l, m):
    """J(psi_sb, Y_l^m)_lm == i m w on every backend."""
    kind, planet = backend_case
    omega_sb = 1e-5
    psi_lm = _solid_body_psi(planet, omega_sb)
    q_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    q_lm[l, m] = 1.0
    j_lm = planet.so.jacobian_pseudospectral(psi_lm, q_lm, dealias=False,
                                             return_spectral=True)
    ratio = complex(j_lm[l, m] / (1j * m * omega_sb))
    assert abs(ratio - 1.0) < TOL[kind]["solid_body"], \
        f"[{kind}] (l={l},m={m}) ratio {ratio}"


def test_self_jacobian_vanishes(backend_case):
    kind, planet = backend_case
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    a[4, 2] = 1.0 + 0.3j
    a[7, 5] = -0.6 + 0.2j
    j = planet.so.jacobian_pseudospectral(a, a, dealias=False)
    assert float(cp.max(cp.abs(j))) < 1e-15


def test_jacobian_integral_zero(backend_case):
    kind, planet = backend_case
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    b = cp.zeros_like(a)
    a[4, 2] = 1.0 + 0.3j
    b[3, 1] = 0.7 - 0.2j
    j_lm = planet.so.jacobian_pseudospectral(a, b, dealias=False,
                                             return_spectral=True)
    assert abs(complex(j_lm[0, 0])) < 1e-11


def test_velocity_solid_body_analytic(backend_case):
    kind, planet = backend_case
    omega_sb = 1e-5
    R = planet.params.radius
    u, v = planet.so.velocity_from_streamfunction(_solid_body_psi(planet, omega_sb))
    lat = cp.asarray(planet.grid.point_latitudes)
    mask = cp.abs(lat) < np.deg2rad(80.0)
    u_expected = omega_sb * R * cp.cos(lat)
    rel = float(cp.max(cp.abs(_flat(u) - u_expected)[mask])) / (omega_sb * R)
    assert rel < TOL[kind]["velocity"], f"[{kind}] u error {rel:.2e}"
    assert float(cp.max(cp.abs(_flat(v))[mask])) / (omega_sb * R) < TOL[kind]["velocity"]


# ---------------------------------------------------------------------------
# RH4 conservation (1 day, CFL dt from the geometry-owned length scale)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rh4_run(backend_case):
    kind, planet = backend_case
    model = BarotropicVorticity(planet, scenario="rh4", viscosity=0.0)
    zeta_lm = planet.sh.transform(make_ic("rh4", planet))

    psi0 = planet.so.inv_laplacian(zeta_lm)
    u0, v0 = planet.so.velocity_from_streamfunction(psi0)
    max_speed = float(cp.max(cp.sqrt(u0**2 + v0**2)))
    length = planet.grid.cfl_length_scale
    assert length and length > 0
    dt_cfl = 0.5 * length / max_speed

    T = 1.0 * 86400.0
    N = max(1, int(round(T / dt_cfl)))
    dt = T / N

    state = BarotropicState(cp.copy(zeta_lm))
    for _ in range(N):
        state = rk4_step(model, state, 0.0, dt)
    return kind, planet, zeta_lm, state.coeffs


def test_rh4_energy_and_enstrophy_conservation(rh4_run):
    kind, planet, zeta0_lm, zeta1_lm = rh4_run
    R, om = planet.params.radius, planet.params.angular_velocity
    d0 = spectral_diagnostics(zeta0_lm, R, om)
    d1 = spectral_diagnostics(zeta1_lm, R, om)

    e_drift = (d1["energy"] - d0["energy"]) / abs(d0["energy"])
    z_drift = (d1["enstrophy_abs"] - d0["enstrophy_abs"]) / abs(d0["enstrophy_abs"])
    print(f"\n[{kind}] RH4 1-day drift: energy {e_drift:+.3e}, "
          f"abs enstrophy {z_drift:+.3e}")

    # Both backends must hold the invariants over a day; band from the
    # geodesic characterization (R-3: ~ -4.5e-4 over 5 days ~ 1e-4/day).
    assert abs(e_drift) < 3e-4, f"[{kind}] energy drift {e_drift:+.3e}"
    assert abs(z_drift) < 3e-3, f"[{kind}] abs-enstrophy drift {z_drift:+.3e}"


def test_rh4_circulation_conserved(rh4_run):
    kind, planet, zeta0_lm, zeta1_lm = rh4_run
    R, om = planet.params.radius, planet.params.angular_velocity
    g0 = spectral_diagnostics(zeta0_lm, R, om)["circulation"]
    g1 = spectral_diagnostics(zeta1_lm, R, om)["circulation"]
    scale = np.sqrt(2.0 * spectral_diagnostics(zeta0_lm, R, om)["energy"])
    assert abs(g1 - g0) / scale < 1e-12  # tendency[0,:] pinned to zero


# ---------------------------------------------------------------------------
# Diagnostics: spectral vs grid paths agree
# ---------------------------------------------------------------------------

def test_diagnostics_spectral_vs_grid_consistency(backend_case):
    kind, planet = backend_case
    R, om = planet.params.radius, planet.params.angular_velocity
    zeta_lm = planet.sh.transform(make_ic("rh4", planet))

    spec = spectral_diagnostics(zeta_lm, R, om)
    grid = grid_diagnostics(zeta_lm, planet.sh, R, om,
                            cp.asarray(planet.grid.point_latitudes))

    tol = TOL[kind]["diag_consistency"]
    for key in ("energy", "enstrophy_rel", "enstrophy_abs"):
        rel = abs(grid[key] - spec[key]) / abs(spec[key])
        assert rel < tol, f"[{kind}] {key}: spectral vs grid rel diff {rel:.2e}"
    # circulation of RH4 is ~0; compare on the enstrophy scale
    circ_scale = np.sqrt(2.0 * spec["enstrophy_rel"]) * R
    assert abs(grid["circulation"] - spec["circulation"]) / circ_scale < tol
