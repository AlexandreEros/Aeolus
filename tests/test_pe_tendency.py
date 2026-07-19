"""Primitive-equation TENDENCY milestone verification (CUDA-gated).

Covers the tendency-milestone requirements layered on top of the foundation
tests in test_primitive_equations.py (which stay untouched):

* Phase 1 — the private tendency-path reconstruction of every
  primitive-equation field on the backend PRODUCT sampling, its agreement
  with the state-grid diagnostics in the exact band-limited Gauss case, and
  the spectral-input (complex-coefficient) hydrostatic reconstruction;
* later phases append their sections below (vector curl/divergence
  analysis, thermodynamic + surface-pressure tendencies, momentum
  tendencies, and the full assembled tendency()).

Tolerancing policy: the Gauss-Legendre lat-lon backend has exact transforms
for band-limited fields, so its tolerances are round-off-level; the
geodesic backend's residuals are measured and asserted against documented
envelopes, never hidden.
"""
from __future__ import annotations

import math

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(),
                                reason="CUDA/CuPy not available")

T0 = 260.0
PS0 = 101325.0
SQRT4PI = math.sqrt(4.0 * math.pi)


def _make_planet(day_hours=24.0, grid_type="latlon", nlat=32, nlon=64,
                 l_max=15, resolution=3):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    return Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)


@pytest.fixture(scope="module")
def latlon_planet():
    return _make_planet()


@pytest.fixture(scope="module")
def geodesic_planet():
    # l_max = 10 at resolution 3 keeps the geodesic transform inside its
    # supported points-per-basis envelope (docs/KNOWN_RISKS.md R-2).
    return _make_planet(grid_type="geodesic", l_max=10)


def _make_model(planet, nlev=5, **kwargs):
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev), **kwargs)


def _rest_state(model):
    from planetary_sandbox.physics.primitive_equations import (
        isothermal_rest_state)
    return isothermal_rest_state(model.l_max, model.nlev,
                                 temperature=T0, surface_pressure=PS0)


def _band_limited_state(model, seed=3, l_band=None):
    """Random zeta/delta/T/ln_ps perturbations, band-limited to l <= l_band.

    The default band limit is the model's 2/3-truncation cut, so states
    survive one truncation exactly (the natural home of an RK4 state).
    """
    import numpy as np
    import cupy as cp
    if l_band is None:
        l_band = (2 * model.l_max) // 3
    state = _rest_state(model)
    rng = np.random.default_rng(seed)
    n = model.l_max + 1

    def _pert(amp, zero_monopole):
        re = rng.standard_normal((n, n))
        im = rng.standard_normal((n, n))
        pert = amp * (re + 1j * im)
        pert[:, 0] = np.real(pert[:, 0])      # m = 0 modes are real
        pert *= np.tril(np.ones((n, n)))      # m <= l only
        pert[l_band + 1:, :] = 0.0            # band limit
        if zero_monopole:
            pert[0, 0] = 0.0
        return cp.asarray(pert)

    for k in range(model.nlev):
        state.zeta[k] += _pert(2e-6, True)
        state.delta[k] += _pert(2e-6, True)
        state.temperature[k] += _pert(0.5, False)
    state.ln_ps[:] += _pert(1e-3, True)
    return state


# ---------------------------------------------------------------------------
# Phase 1 — spectral-input hydrostatics (linearity of the column operator)
# ---------------------------------------------------------------------------

def test_hydrostatic_on_spectral_coefficients_matches_grid_path(latlon_planet):
    """Hydrostatic reconstruction is linear in (T, Phi_s), so applying it
    directly to complex spectral coefficients must agree with grid
    reconstruction followed by reanalysis in the exact band-limited Gauss
    case (handoff Section 1.2: complex input was untested before this)."""
    import cupy as cp
    from planetary_sandbox.physics.sigma_coordinate import (
        hydrostatic_geopotential)

    n = latlon_planet.sh.l_max + 1
    phi_s_lm = cp.zeros((n, n), dtype=cp.complex128)
    phi_s_lm[0, 0] = 700.0 * SQRT4PI
    phi_s_lm[3, 2] = 25.0 - 10.0j
    model = _make_model(latlon_planet, surface_geopotential_lm=phi_s_lm)
    state = _band_limited_state(model, seed=5)

    # Spectral path: the column operator applied to complex coefficients.
    phi_full_lm, phi_below_lm = hydrostatic_geopotential(
        model.sigma, state.temperature, model.phi_surface_lm, model.r_dry)

    # Grid path: synthesize T and Phi_s, reconstruct, reanalyze.
    sh = latlon_planet.sh
    T_grid = cp.stack([sh.inv_transform(state.temperature[k]).real
                       for k in range(model.nlev)])
    phi_s_grid = sh.inv_transform(model.phi_surface_lm).real
    phi_full_g, phi_below_g = hydrostatic_geopotential(
        model.sigma, T_grid, phi_s_grid, model.r_dry)

    scale = float(cp.abs(phi_full_lm).max())
    assert scale > 0.0
    for k in range(model.nlev):
        re_full = sh.transform(phi_full_g[k])
        re_below = sh.transform(phi_below_g[k])
        assert float(cp.abs(re_full - phi_full_lm[k]).max()) < 1e-12 * scale
        assert float(cp.abs(re_below - phi_below_lm[k]).max()) < 1e-12 * scale


# ---------------------------------------------------------------------------
# Phase 1 — product-grid tendency-path reconstruction
# ---------------------------------------------------------------------------

def test_product_fields_of_rest_state_are_exactly_zero(latlon_planet):
    """Isothermal rest: every dynamic product-grid field is bitwise zero and
    the hydrostatic profile is the analytic isothermal column."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    f = model._tendency_product_fields(state.coeffs)

    for key in ("u", "v", "v_grad_lnps", "g_full", "dlnps_dt", "sigma_dot",
                "omega_over_p", "sigma_dot_dU", "sigma_dot_dV",
                "sigma_dot_dT"):
        assert float(cp.abs(f[key]).max()) == 0.0, key

    # Phi on the product sampling equals the analytic isothermal profile.
    for k in range(model.nlev):
        analytic = -model.r_dry * T0 * (
            math.log(model.sigma.interfaces[k + 1]) - model.sigma.alpha[k])
        assert float(cp.abs(f["phi_full"][k] - analytic).max()) \
            <= 1e-9 * max(abs(analytic), 1.0)


def test_product_fields_match_state_diagnostics_band_limited(latlon_planet):
    """On the Gauss backend the fine product sampling and the state sampling
    coincide analytically for the default fixture (the 3/2-rule grid is not
    larger than the 32x64 state grid), so the tendency-path product fields
    must reproduce the state-grid diagnostics to round-off."""
    import cupy as cp
    model = _make_model(latlon_planet)
    # Guard: this comparison is only pointwise-meaningful because the
    # product geometry has the same Gauss nodes as the state geometry.
    ps_geom = model._ps.geometry
    assert ps_geom is not None
    assert (ps_geom.nlat, ps_geom.nlon) == (model.grid.nlat, model.grid.nlon)

    state = _band_limited_state(model, seed=7)
    f = model._tendency_product_fields(state.coeffs)
    diag = model.continuity_diagnostics(state)
    u_s, v_s = model.wind_on_state_grid(state)
    T_s = model.temperature_on_state_grid(state)
    vt = model.vertical_transport_diagnostics(state, continuity=diag)
    ex = model.energy_exchange_diagnostics(state, continuity=diag)
    phi_state = model.geopotential_fields(state)

    def _close(a, b, key, rtol=1e-11):
        a2 = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1)
        b2 = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(-1)
        scale = max(float(cp.abs(b2).max()), 1e-30)
        assert float(cp.abs(a2 - b2).max()) < rtol * scale, key

    _close(f["u"], u_s, "u")
    _close(f["v"], v_s, "v")
    _close(f["temperature"], T_s, "temperature")
    _close(f["v_grad_lnps"], diag["v_grad_lnps"], "v_grad_lnps")
    _close(f["g_full"], diag["g_full"], "g_full")
    _close(f["dlnps_dt"], diag["dlnps_dt"], "dlnps_dt")
    _close(f["sigma_dot"], diag["sigma_dot"], "sigma_dot")
    _close(f["omega_over_p"], ex["omega_over_p"], "omega_over_p")
    _close(f["sigma_dot_dU"], vt["sigma_dot_dU"], "sigma_dot_dU")
    _close(f["sigma_dot_dV"], vt["sigma_dot_dV"], "sigma_dot_dV")
    _close(f["sigma_dot_dT"], vt["sigma_dot_dT"], "sigma_dot_dT")
    _close(f["phi_full"], phi_state["phi_full"], "phi_full")


def test_product_continuity_structural_and_closure(latlon_planet):
    """Structural sigma_dot impermeability and round-off layer closure hold
    for the PRODUCT-grid G exactly as they do on the state grid."""
    import cupy as cp
    from planetary_sandbox.physics.sigma_coordinate import (
        layer_mass_residual)
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=11)
    f = model._tendency_product_fields(state.coeffs)

    g_scale = float(cp.abs(f["g_full"]).max())
    assert g_scale > 0.0
    assert float(cp.abs(f["sigma_dot"][0]).max()) == 0.0
    assert float(cp.abs(f["sigma_dot"][-1]).max()) == 0.0
    assert float(cp.abs(f["sigma_dot"][1:-1]).max()) > 0.0
    residual = layer_mass_residual(model.sigma, f["g_full"])
    assert float(cp.abs(residual).max()) < 1e-12 * g_scale


def test_product_phi_matches_spectral_hydrostatics(latlon_planet):
    """Product-grid Phi (from product-grid T) equals the synthesis of the
    spectral hydrostatic Phi at the product points in the band-limited
    exact case — the guarantee that lets the delta equation use the exact
    spectral -lap(Phi) while the grid nonlinearities use product Phi."""
    import cupy as cp
    from planetary_sandbox.physics.sigma_coordinate import (
        hydrostatic_geopotential)
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=13)
    f = model._tendency_product_fields(state.coeffs)

    phi_full_lm, _ = hydrostatic_geopotential(
        model.sigma, state.temperature, model.phi_surface_lm, model.r_dry)
    sh_p = model._ps.sh
    scale = float(cp.abs(f["phi_full"]).max())
    for k in range(model.nlev):
        synth = sh_p.inv_transform(phi_full_lm[k]).real
        assert float(cp.abs(synth - f["phi_full"][k]).max()) < 1e-11 * scale


def test_product_fields_uniform_pressure_kill_mass_terms(latlon_planet):
    """Uniform ln_ps in genuine flow: grad(lnps), A, and the pressure parts
    vanish bitwise on the product grid; G reduces to delta."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=17)
    state.ln_ps[:] = 0.0
    state.ln_ps[0, 0] = math.log(PS0) * SQRT4PI
    f = model._tendency_product_fields(state.coeffs)

    assert float(cp.abs(f["lnps_lam"]).max()) == 0.0
    assert float(cp.abs(f["lnps_snt"]).max()) == 0.0
    assert float(cp.abs(f["v_grad_lnps"]).max()) == 0.0
    assert float(cp.abs(f["grad_lnps_u"]).max()) == 0.0
    assert float(cp.abs(f["grad_lnps_v"]).max()) == 0.0
    delta_g = cp.stack([model._ps.sh.inv_transform(state.delta[k]).real
                        for k in range(model.nlev)])
    assert float(cp.abs(f["g_full"] - delta_g).max()) == 0.0


# ---------------------------------------------------------------------------
# Phase 2a — scalar-round-trip curl/divergence REFERENCE pathway
# ---------------------------------------------------------------------------
#
# The reference analyzes the vector components as scalars, differentiates
# their projections spectrally, and assembles grid-space curl/div. Component
# fields of a band-limited vector field are NOT band-limited scalars (they
# carry spin-1 structure; e.g. solid-body u = u0*cos(lat) has an infinite
# zonal Legendre expansion), so this pathway has an inherent representation
# error even on the exact-quadrature Gauss backend. Tolerances below are
# therefore measured envelopes for low-degree potentials, not round-off;
# that inherent error is exactly why this is a reference, not production.

def _potential_coeffs(l_max, modes):
    """Complex (l_max+1)^2 coefficient array with the given (l, m, value)."""
    import cupy as cp
    out = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    for l, m, val in modes:
        out[l, m] = val
    return out


def _vector_from_potentials(planet, psi_lm, chi_lm):
    """F = k x grad(psi) + grad(chi) evaluated on the product sampling."""
    so = planet.so
    ps = so.backend.product_space(so.product_quadrature)
    sh_p = ps.sh
    R = so.R
    coslat = ps.coslat

    def _derivs(c_lm):
        lam = sh_p.inv_transform(so.d_lambda_coeffs(c_lm)).real
        snt = sh_p.inv_transform(so.sin_theta_d_theta_coeffs(c_lm)).real / R
        return lam, snt

    psi_lam, psi_snt = _derivs(psi_lm)
    chi_lam, chi_snt = _derivs(chi_lm)
    f_east = (psi_snt + chi_lam) / coslat
    f_north = (psi_lam - chi_snt) / coslat
    return f_east, f_north


def _exact_curl_div(planet, psi_lm, chi_lm):
    """curl(F) = lap(psi), div(F) = lap(chi) in spectral space."""
    import cupy as cp
    l = cp.arange(planet.sh.l_max + 1, dtype=cp.float64)
    lap = (-l * (l + 1.0) / planet.so.R**2)[:, None]
    return lap * psi_lm, lap * chi_lm


def _rel_err(got, want, scale=None):
    import cupy as cp
    if scale is None:
        scale = float(cp.abs(want).max())
    return float(cp.abs(got - want).max()) / scale


#: Measured Gauss-backend envelope for the round-trip reference on
#: low-degree (l <= 4) potentials at l_max = 15. Measured 2026-07: curl
#: recovery errors 3.1e-2 (rotational), 9.2e-2 curl-leakage (divergent),
#: 1.8e-1 (mixed); errors reach 4.2e-1 at l = l_max. The error is the
#: spin-1 representation tail of the scalar component analysis, not
#: quadrature — it does NOT vanish on the exact Gauss backend, which is
#: exactly why this pathway is a reference, not production.
ROUNDTRIP_LOW_DEGREE_RTOL = 0.2


def test_roundtrip_rotational_field_latlon(latlon_planet):
    """Pure rotational low-degree field: curl recovered within the measured
    envelope, divergence stays small on that same scale, output finite."""
    import cupy as cp
    l_max = latlon_planet.sh.l_max
    psi_lm = _potential_coeffs(l_max, [(2, 0, 3.0e7), (3, 2, 1.0e7 - 5.0e6j)])
    chi_lm = cp.zeros_like(psi_lm)
    f_east, f_north = _vector_from_potentials(latlon_planet, psi_lm, chi_lm)
    curl_lm, div_lm = latlon_planet.so.vector_curl_div_roundtrip(
        f_east, f_north, truncate=False)
    curl_exact, _ = _exact_curl_div(latlon_planet, psi_lm, chi_lm)

    assert bool(cp.isfinite(curl_lm).all()) and bool(cp.isfinite(div_lm).all())
    scale = float(cp.abs(curl_exact).max())
    assert _rel_err(curl_lm, curl_exact, scale) < ROUNDTRIP_LOW_DEGREE_RTOL
    assert float(cp.abs(div_lm).max()) / scale < ROUNDTRIP_LOW_DEGREE_RTOL


def test_roundtrip_divergent_field_latlon(latlon_planet):
    """Pure divergent low-degree field: div recovered, curl small."""
    import cupy as cp
    l_max = latlon_planet.sh.l_max
    chi_lm = _potential_coeffs(l_max, [(2, 1, 2.0e7 + 1.0e7j), (4, 0, 1.5e7)])
    psi_lm = cp.zeros_like(chi_lm)
    f_east, f_north = _vector_from_potentials(latlon_planet, psi_lm, chi_lm)
    curl_lm, div_lm = latlon_planet.so.vector_curl_div_roundtrip(
        f_east, f_north, truncate=False)
    _, div_exact = _exact_curl_div(latlon_planet, psi_lm, chi_lm)

    scale = float(cp.abs(div_exact).max())
    assert _rel_err(div_lm, div_exact, scale) < ROUNDTRIP_LOW_DEGREE_RTOL
    assert float(cp.abs(curl_lm).max()) / scale < ROUNDTRIP_LOW_DEGREE_RTOL


def test_roundtrip_mixed_field_latlon(latlon_planet):
    """Mixed field: both spectra recovered within the documented envelope."""
    l_max = latlon_planet.sh.l_max
    psi_lm = _potential_coeffs(l_max, [(3, 1, 2.0e7 - 1.0e7j)])
    chi_lm = _potential_coeffs(l_max, [(2, 2, 1.0e7 + 4.0e6j)])
    f_east, f_north = _vector_from_potentials(latlon_planet, psi_lm, chi_lm)
    curl_lm, div_lm = latlon_planet.so.vector_curl_div_roundtrip(
        f_east, f_north, truncate=False)
    curl_exact, div_exact = _exact_curl_div(latlon_planet, psi_lm, chi_lm)

    assert _rel_err(curl_lm, curl_exact) < ROUNDTRIP_LOW_DEGREE_RTOL
    assert _rel_err(div_lm, div_exact) < ROUNDTRIP_LOW_DEGREE_RTOL


def test_roundtrip_zero_field_is_bitwise_zero(latlon_planet):
    """A zero vector field analyzes to exactly zero spectra (the BVE-
    degeneracy prerequisite for any pathway)."""
    import cupy as cp
    ps = latlon_planet.so.backend.product_space(
        latlon_planet.so.product_quadrature)
    npts = ps.coslat.shape[0]
    zero = cp.zeros(npts, dtype=cp.float64)
    curl_lm, div_lm = latlon_planet.so.vector_curl_div_roundtrip(zero, zero)
    assert float(cp.abs(curl_lm).max()) == 0.0
    assert float(cp.abs(div_lm).max()) == 0.0


def test_product_fields_on_geodesic_are_finite_and_structural(geodesic_planet):
    """The same reconstruction runs on the geodesic backend: finite fields,
    structural sigma_dot zeros, round-off layer closure."""
    import cupy as cp
    from planetary_sandbox.physics.sigma_coordinate import (
        layer_mass_residual)
    model = _make_model(geodesic_planet)
    state = _band_limited_state(model, seed=19)
    f = model._tendency_product_fields(state.coeffs)

    for key in ("u", "v", "g_full", "dlnps_dt", "sigma_dot", "phi_full",
                "omega_over_p", "sigma_dot_dU", "sigma_dot_dV",
                "sigma_dot_dT"):
        assert bool(cp.isfinite(f[key]).all()), key
    assert float(cp.abs(f["sigma_dot"][0]).max()) == 0.0
    assert float(cp.abs(f["sigma_dot"][-1]).max()) == 0.0
    g_scale = float(cp.abs(f["g_full"]).max())
    assert g_scale > 0.0
    residual = layer_mass_residual(model.sigma, f["g_full"])
    assert float(cp.abs(residual).max()) < 1e-12 * g_scale
