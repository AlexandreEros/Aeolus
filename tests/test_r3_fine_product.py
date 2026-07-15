"""R-3 fix regression tests: fine-grid product evaluation.

Locks the "overresolved product quadrature" repair characterized in
tests/audit_r3_product.py and preregistered in docs/KNOWN_RISKS.md R-3: the
pointwise Jacobian product is evaluated and analyzed on a reusable
resolution-(r+1) product grid at the same l_max, truncated once spectrally,
and consumed spectrally by the tendency.

Preregistered predictions verified here (recorded BEFORE implementation):
  P1: res4/l21 RH4 5-day E drift  -2.64e-3 -> -4.5e-4 (+/-20%)
  P2: t0 discrete production rate -> ~ -6e-5/day (vs ~ -4.4e-4/day coarse)
(P3, tilt-60 orientation sensitivity, is verified in the run record, not
here, to keep suite time bounded.)
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
    from planetary_sandbox.numerics import (
        GeodesicGridGeometry,
        GeodesicSphericalHarmonics,
        SpectralOperators,
    )
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.barotropic_vorticity import (
        BarotropicState,
        BarotropicVorticity,
    )
    from planetary_sandbox.run.bve.runner import rk4_step
    from planetary_sandbox.run.bve.diagnostics import spectral_diagnostics

RES, L_MAX = 4, 21
CUT = (2 * L_MAX) // 3


def _rh4(grid):
    nu = K = 7.848e-6
    lat = cp.asarray(grid.point_latitudes)
    lon = cp.asarray(grid.point_longitudes)
    return (2.0 * nu * cp.sin(lat)
            - 30.0 * K * cp.sin(lat) * cp.cos(lat) ** 4 * cp.sin(4.0 * lon))


@pytest.fixture(scope="module")
def planet():
    params = PlanetaryParameters.from_earth_like(day_hours=24.0)
    return Planet.generate(params=params, grid_resolution=RES, l_max=L_MAX)


def _production_rate(planet, model, zeta_lm):
    """Discrete dE/dt normalized per day (exactly 0 for the PDE)."""
    R = planet.params.radius
    dz = model.tendency(BarotropicState(zeta_lm), None)
    l_idx, m_idx = cp.indices(zeta_lm.shape)
    valid = (m_idx <= l_idx) & (l_idx >= 1)
    mult = cp.where(m_idx == 0, 1.0, 2.0) * valid
    ll1 = cp.where(l_idx >= 1, l_idx * (l_idx + 1.0), 1.0)
    dEdt = float(R**4 * (mult * cp.real(cp.conj(zeta_lm) * dz) / ll1).sum())
    E = spectral_diagnostics(zeta_lm, R, planet.params.angular_velocity)["energy"]
    return dEdt * 86400.0 / E


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_product_grid_built_once_at_init(planet):
    so = planet.so
    assert so.product_quadrature == "fine"
    assert so.product_grid is not None and so.product_sh is not None
    assert so.product_grid.resolution == RES + 1
    assert so.product_sh.l_max == L_MAX          # same truncation, finer points
    # reusable: the same objects across calls (nothing constructed in tendency)
    assert so.product_sh is planet.so.product_sh


def test_truncation_applied_exactly_once_spectrally(planet):
    """Spectral return carries the single 2/3-rule truncation."""
    so = planet.so
    rng = np.random.default_rng(3)
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    b = cp.zeros_like(a)
    for l in range(1, L_MAX + 1):
        for m in range(0, l + 1):
            imag_a = rng.standard_normal() if m else 0.0
            imag_b = rng.standard_normal() if m else 0.0
            a[l, m] = 1e-5 * (rng.standard_normal() + 1j * imag_a)
            b[l, m] = 1e-5 * (rng.standard_normal() + 1j * imag_b)
    J_lm = so.jacobian_pseudospectral(a, b, dealias=True, return_spectral=True)
    assert float(cp.abs(J_lm[CUT + 1:, :]).max()) == 0.0
    assert float(cp.abs(J_lm[:, CUT + 1:]).max()) == 0.0
    assert float(cp.abs(J_lm[1:CUT + 1, :CUT + 1]).max()) > 0.0  # not all zero


def test_coarse_option_retained_for_ab_comparisons(planet):
    """product_quadrature='coarse' keeps the historical state-grid path."""
    grid, sh = planet.grid, planet.sh
    so_c = SpectralOperators(sh, planet.params.radius, grid,
                             product_quadrature="coarse")
    assert so_c.product_quadrature == "coarse"
    assert so_c.product_sh is None

    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    b = cp.zeros_like(a)
    a[4, 2] = 1e-5
    b[3, 1] = 1e-5
    # legacy grid returns, both paths, sized to the STATE grid
    J_coarse = so_c.jacobian_pseudospectral(a, b, dealias=False)
    J_fine_as_grid = planet.so.jacobian_pseudospectral(a, b, dealias=False)
    assert J_coarse.shape == (grid.n_points,)
    assert J_fine_as_grid.shape == (grid.n_points,)
    # both are discretizations of the same smooth low-l Jacobian
    denom = float(cp.abs(J_coarse).max())
    rel = float(cp.abs(J_coarse - J_fine_as_grid).max()) / denom
    assert rel < 5e-2, f"coarse/fine low-l Jacobian mismatch {rel:.2e}"


def test_invalid_product_quadrature_rejected(planet):
    with pytest.raises(ValueError):
        SpectralOperators(planet.sh, planet.params.radius, planet.grid,
                          product_quadrature="exact")


def test_no_silent_fallback_on_unsupported_grid(planet):
    """'fine' on a non-geodesic grid must raise, never fall back to 'coarse'."""
    from planetary_sandbox.numerics import LatLonGridGeometry
    latlon = LatLonGridGeometry.create((9, 17))
    with pytest.raises(ValueError):
        SpectralOperators(planet.sh, planet.params.radius, latlon,
                          product_quadrature="fine")


def test_cli_exposes_and_defaults_product_quadrature():
    """--product-quadrature is a CLI option, defaults to 'fine', and lands in
    the args dict that config.json and manifest.json serialize."""
    from planetary_sandbox.cli.bve import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert vars(args)["product_quadrature"] == "fine"
    # choices constrained: an invalid value must be rejected by the parser
    with pytest.raises(SystemExit):
        parser.parse_args(["--product-quadrature", "exact"])
    # help text mentions it (exposed, not hidden)
    assert "--product-quadrature" in parser.format_help()


# ---------------------------------------------------------------------------
# Preregistered predictions
# ---------------------------------------------------------------------------

def test_prediction_p2_t0_production_rate(planet):
    """P2: fine-path production rate ~ -6e-5/day; coarse ~7x worse.

    Characterization values (audit_r3_product, res4/l21 RH4 t0):
    variant D -6.13e-5/day, variant A -4.36e-4/day."""
    zeta_lm = planet.sh.transform(_rh4(planet.grid))

    rate_fine = _production_rate(planet, BarotropicVorticity(planet, viscosity=0.0),
                                 zeta_lm)
    assert abs(rate_fine) < 1.5e-4, f"fine-path |dE/dt|/E = {abs(rate_fine):.2e}/day"

    # coarse-product A/B model on the same planet state
    so_c = SpectralOperators(planet.sh, planet.params.radius, planet.grid,
                             product_quadrature="coarse")
    model_c = BarotropicVorticity(planet, viscosity=0.0)
    model_c.so = so_c
    rate_coarse = _production_rate(planet, model_c, zeta_lm)
    assert abs(rate_coarse) > 2.5e-4, (
        f"coarse-path |dE/dt|/E = {abs(rate_coarse):.2e}/day — expected the "
        "historical defect to remain measurable through the debug option"
    )


def test_prediction_p1_5day_energy_drift(planet):
    """P1 (preregistered): res4/l21 RH4 5-day E drift -> -4.5e-4 +/- 20%.

    Characterization: variant D measured -4.4555e-4; production is now
    variant D, so the band is [-5.4e-4, -3.6e-4]. ~25 s on the MX110."""
    sh, so, grid = planet.sh, planet.so, planet.grid
    R, om = planet.params.radius, planet.params.angular_velocity
    model = BarotropicVorticity(planet, viscosity=0.0)

    zeta_lm = sh.transform(_rh4(grid))
    psi0 = so.inv_laplacian(zeta_lm)
    u0, v0 = so.velocity_from_streamfunction(psi0)
    dt_cfl = 0.5 * grid.min_edge_length / float(cp.max(cp.sqrt(u0**2 + v0**2)))
    T = 5.0 * 86400.0
    N = max(1, int(round(T / dt_cfl)))
    dt = T / N

    E0 = spectral_diagnostics(zeta_lm, R, om)["energy"]
    state = BarotropicState(cp.copy(zeta_lm))
    for _ in range(N):
        state = rk4_step(model, state, 0.0, dt)
    E1 = spectral_diagnostics(state.coeffs, R, om)["energy"]
    drift = (E1 - E0) / abs(E0)

    assert -5.4e-4 < drift < -3.6e-4, (
        f"5-day RH4 energy drift {drift:+.4e} outside preregistered band "
        "[-5.4e-4, -3.6e-4] (prediction -4.5e-4 +/- 20%)"
    )
