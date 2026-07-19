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


# ---------------------------------------------------------------------------
# Phase 2b — weak-form (Bourke-style) vector spectral analysis: PRODUCTION
# ---------------------------------------------------------------------------
#
# (div F)_lm = (im/R) a_lm + (1/R)(C+_lm b_{l+1,m} + C-_lm b_{l-1,m})
# (curl F)_lm = (im/R) b_lm - (1/R)(C+_lm a_{l+1,m} + C-_lm a_{l-1,m})
# with a = analysis(F_u/cos(lat)), b = analysis(F_v/cos(lat)). The
# meridional coupling is the TRANSPOSE (adjoint) of the synthesis-side
# sin(theta) d/dtheta coupling. Only the integration by parts is a
# continuous step; every discrete operation afterward is pointwise-exact,
# so on the Gauss backend (exact product quadrature for these integrands)
# recovery of analytic curl/div spectra is round-off.

def test_adjoint_sin_theta_matches_inner_product(latlon_planet):
    """<S c, d> == <c, S^T d> for the meridional coupling operator: the
    adjoint identity that justifies moving the derivative onto the basis."""
    import numpy as np
    import cupy as cp
    so = latlon_planet.so
    n = so.l_max + 1
    rng = np.random.default_rng(31)
    mask = np.tril(np.ones((n, n)))
    c = cp.asarray((rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))) * mask)
    d = cp.asarray((rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))) * mask)

    lhs = complex(cp.sum(so.sin_theta_d_theta_coeffs(c) * cp.conj(d)))
    rhs = complex(cp.sum(c * cp.conj(so.adjoint_sin_theta_d_theta_coeffs(d))))
    scale = max(abs(lhs), abs(rhs), 1e-30)
    assert abs(lhs - rhs) < 1e-12 * scale


def _weak_form_errors(planet, psi_modes, chi_modes, truncate=False):
    import cupy as cp
    l_max = planet.sh.l_max
    psi_lm = _potential_coeffs(l_max, psi_modes)
    chi_lm = _potential_coeffs(l_max, chi_modes)
    f_east, f_north = _vector_from_potentials(planet, psi_lm, chi_lm)
    curl_lm, div_lm = planet.so.vector_curl_div_spectral(
        f_east, f_north, truncate=truncate)
    curl_exact, div_exact = _exact_curl_div(planet, psi_lm, chi_lm)
    if truncate:
        cut = (2 * l_max) // 3
        curl_exact = curl_exact.copy()
        div_exact = div_exact.copy()
        curl_exact[cut + 1:, :] = 0.0
        div_exact[cut + 1:, :] = 0.0
    scale = max(float(cp.abs(curl_exact).max()),
                float(cp.abs(div_exact).max()))
    curl_err = float(cp.abs(curl_lm - curl_exact).max()) / scale
    div_err = float(cp.abs(div_lm - div_exact).max()) / scale
    return curl_err, div_err, curl_lm, div_lm


def test_weak_form_rotational_field_latlon_is_exact(latlon_planet):
    """Pure rotational field (zonal + nonzonal, low and near-cut modes):
    curl recovered to round-off, divergence round-off on the same scale."""
    curl_err, div_err, _, _ = _weak_form_errors(
        latlon_planet,
        [(1, 0, 4.0e7), (3, 2, 1.0e7 - 5.0e6j), (10, 7, 6.0e6 + 2.0e6j)],
        [])
    assert curl_err < 1e-12
    assert div_err < 1e-12


def test_weak_form_divergent_field_latlon_is_exact(latlon_planet):
    """Pure divergent field: div recovered to round-off, curl round-off."""
    curl_err, div_err, _, _ = _weak_form_errors(
        latlon_planet,
        [],
        [(2, 0, 3.0e7), (4, 1, 2.0e7 + 1.0e7j), (10, 10, 5.0e6 - 1.0e6j)])
    assert curl_err < 1e-12
    assert div_err < 1e-12


def test_weak_form_mixed_field_latlon_full_band(latlon_planet):
    """Mixed rotational+divergent field with modes up to l_max - 1: both
    exact spectra recovered to round-off UNtruncated on the exact-
    quadrature Gauss product grid. (l_max - 1, not l_max, is the exact
    envelope: see test_weak_form_top_degree_is_wind_synthesis_limited.)"""
    l_max = latlon_planet.sh.l_max
    curl_err, div_err, _, _ = _weak_form_errors(
        latlon_planet,
        [(2, 1, 2.0e7 - 1.0e7j), (l_max - 1, 3, 4.0e6)],
        [(3, 0, 1.5e7), (l_max - 1, l_max - 1, 3.0e6 + 2.0e6j)])
    assert curl_err < 1e-12
    assert div_err < 1e-12


def test_weak_form_top_degree_is_wind_synthesis_limited(latlon_planet):
    """CHARACTERIZATION: potentials at l = l_max do NOT recover exactly —
    and the defect belongs to the repository's Helmholtz wind synthesis,
    not to the vector operator. sin_theta_d_theta_coeffs clips the
    degree-(l_max+1) component of the meridional derivative (its storage
    is (l_max+1)^2; C+ at l_max is zeroed), so the synthesized wind of an
    l_max-degree potential is already a truncated vector field before any
    curl/div analysis happens — a pre-existing convention shared with the
    BVE/SWE wind reconstruction. The operator faithfully analyzes the
    field it is GIVEN; measured deviation at l_max = 15 is a few percent
    up to ~40% depending on the mode (recorded in the design doc). This
    test pins the l_max-1 exactness boundary and documents that the
    l_max-mode deviation is bounded and confined to the same zonal
    wavenumber."""
    import cupy as cp
    l_max = latlon_planet.sh.l_max
    # Exact at l_max - 1 ...
    curl_err, _, _, _ = _weak_form_errors(
        latlon_planet, [(l_max - 1, 3, 1.0e7)], [])
    assert curl_err < 1e-12
    # ... measurably inexact at l_max, bounded, and m-confined. The
    # clipped wind field is no longer exactly nondivergent either (its
    # dropped degree-(l_max+1) meridional-derivative part carried both
    # rotational and divergent projections), so BOTH spectra deviate.
    curl_err, div_err, curl_lm, div_lm = _weak_form_errors(
        latlon_planet, [(l_max, 3, 1.0e7)], [])
    assert 1e-6 < curl_err < 0.5
    assert 1e-6 < div_err < 0.5
    for out in (curl_lm, div_lm):
        other_m = cp.abs(out).sum() - cp.abs(out[:, 3]).sum()
        assert float(other_m) < 1e-12 * float(cp.abs(out).max())


def test_weak_form_truncation_and_m_le_l_structure(latlon_planet):
    """The truncated call zeroes l > cut rows/cols and never produces
    invalid m > l content (checked untruncated too)."""
    import numpy as np
    import cupy as cp
    l_max = latlon_planet.sh.l_max
    cut = (2 * l_max) // 3
    for truncate in (False, True):
        _, _, curl_lm, div_lm = _weak_form_errors(
            latlon_planet,
            [(2, 1, 2.0e7), (cut, 4, 8.0e6)],
            [(3, 3, 1.0e7)],
            truncate=truncate)
        upper = cp.asarray(np.triu(np.ones((l_max + 1, l_max + 1)), k=1))
        assert float(cp.abs(curl_lm * upper).max()) == 0.0
        assert float(cp.abs(div_lm * upper).max()) == 0.0
        if truncate:
            assert float(cp.abs(curl_lm[cut + 1:, :]).max()) == 0.0
            assert float(cp.abs(div_lm[cut + 1:, :]).max()) == 0.0
            assert float(cp.abs(curl_lm[:, cut + 1:]).max()) == 0.0
            assert float(cp.abs(div_lm[:, cut + 1:]).max()) == 0.0


def test_weak_form_zero_field_is_bitwise_zero(latlon_planet):
    import cupy as cp
    ps = latlon_planet.so.backend.product_space(
        latlon_planet.so.product_quadrature)
    zero = cp.zeros(ps.coslat.shape[0], dtype=cp.float64)
    curl_lm, div_lm = latlon_planet.so.vector_curl_div_spectral(zero, zero)
    assert float(cp.abs(curl_lm).max()) == 0.0
    assert float(cp.abs(div_lm).max()) == 0.0


def test_weak_form_agrees_with_roundtrip_within_reference_envelope(
        latlon_planet):
    """Cross-validation of the two pathways: they must agree within the
    round-trip's own documented representation envelope (the round-trip is
    the one carrying the error; the weak form is exact here)."""
    import cupy as cp
    l_max = latlon_planet.sh.l_max
    psi_lm = _potential_coeffs(l_max, [(3, 1, 2.0e7 - 1.0e7j)])
    chi_lm = _potential_coeffs(l_max, [(2, 2, 1.0e7 + 4.0e6j)])
    f_east, f_north = _vector_from_potentials(latlon_planet, psi_lm, chi_lm)
    curl_w, div_w = latlon_planet.so.vector_curl_div_spectral(
        f_east, f_north, truncate=False)
    curl_r, div_r = latlon_planet.so.vector_curl_div_roundtrip(
        f_east, f_north, truncate=False)
    scale = max(float(cp.abs(curl_w).max()), float(cp.abs(div_w).max()))
    assert float(cp.abs(curl_w - curl_r).max()) / scale \
        < ROUNDTRIP_LOW_DEGREE_RTOL
    assert float(cp.abs(div_w - div_r).max()) / scale \
        < ROUNDTRIP_LOW_DEGREE_RTOL


#: Measured geodesic-backend envelope (res 3, l_max = 10, fine res-4
#: co-grid product quadrature), 2026-07: weak-form recovery errors are
#: 1.0e-3 .. 1.9e-2 relative across single modes l = 1..9 (worst at
#: l = 1), 3.3e-3 for the mixed case below; the scalar round-trip on the
#: same fields measures 4.6e-2 .. 3.0e-1 — one to two orders worse. The
#: envelope asserts ~6x headroom over the measured mixed-case value.
GEODESIC_WEAK_FORM_RTOL = 0.02


def test_weak_form_geodesic_measured_envelope(geodesic_planet):
    """Geodesic backend: the weak form's recovery error is the backend's
    quadrature error. Assert the measured envelope and that the weak form
    is not worse than the scalar round-trip reference on the same field."""
    import cupy as cp
    l_max = geodesic_planet.sh.l_max
    psi_lm = _potential_coeffs(l_max, [(2, 0, 3.0e7), (3, 2, 1.0e7 - 5.0e6j)])
    chi_lm = _potential_coeffs(l_max, [(2, 1, 2.0e7 + 1.0e7j), (4, 0, 1.5e7)])
    f_east, f_north = _vector_from_potentials(geodesic_planet, psi_lm, chi_lm)
    curl_exact, div_exact = _exact_curl_div(geodesic_planet, psi_lm, chi_lm)
    scale = max(float(cp.abs(curl_exact).max()),
                float(cp.abs(div_exact).max()))

    curl_w, div_w = geodesic_planet.so.vector_curl_div_spectral(
        f_east, f_north, truncate=False)
    assert bool(cp.isfinite(curl_w).all()) and bool(cp.isfinite(div_w).all())
    err_w = max(float(cp.abs(curl_w - curl_exact).max()),
                float(cp.abs(div_w - div_exact).max())) / scale
    assert err_w < GEODESIC_WEAK_FORM_RTOL

    curl_r, div_r = geodesic_planet.so.vector_curl_div_roundtrip(
        f_east, f_north, truncate=False)
    err_r = max(float(cp.abs(curl_r - curl_exact).max()),
                float(cp.abs(div_r - div_exact).max())) / scale
    # The production pathway must not be worse than the reference.
    assert err_w <= err_r * 1.5


# ---------------------------------------------------------------------------
# Phases 3 + 4 — thermodynamic and surface-pressure tendencies
# ---------------------------------------------------------------------------

def _thermo_mass(model, state):
    fields = model._tendency_product_fields(state.coeffs)
    t_dot, lnps_dot = model._thermo_mass_tendencies(state.coeffs, fields)
    return fields, t_dot, lnps_dot


def test_thermo_and_lnps_tendencies_zero_at_isothermal_rest(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    _, t_dot, lnps_dot = _thermo_mass(model, state)
    assert float(cp.abs(t_dot).max()) == 0.0
    assert float(cp.abs(lnps_dot).max()) == 0.0


def test_thermo_and_lnps_tendencies_zero_for_structured_T_at_rest(
        latlon_planet):
    """Zero flow + structured T: no advection, no conversion (omega/p = 0
    bitwise), no mass flux — the thermodynamic and ln p_s tendencies are
    exactly zero. (The T structure's pressure-gradient response lives in
    the delta tendency, not here.)"""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    for k in range(model.nlev):
        state.temperature[k, 0, 0] = (T0 - 8.0 * k) * SQRT4PI
        state.temperature[k, 3, 2] = 2.5 + 0.5 * k
    _, t_dot, lnps_dot = _thermo_mass(model, state)
    assert float(cp.abs(t_dot).max()) == 0.0
    assert float(cp.abs(lnps_dot).max()) == 0.0


def test_uniform_temperature_feels_only_conversion(latlon_planet):
    """Uniform T in genuine flow: horizontal and vertical temperature
    advection are bitwise zero, so the assembled T tendency equals the
    analyzed conversion term alone — bitwise, not approximately."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=23)
    state.temperature[:] = 0.0
    state.temperature[:, 0, 0] = T0 * SQRT4PI     # uniform T everywhere
    fields, t_dot, lnps_dot = _thermo_mass(model, state)

    assert float(cp.abs(fields["sigma_dot_dT"]).max()) == 0.0
    sh_p = model._ps.sh
    for k in range(model.nlev):
        conv_only = model._truncate(sh_p.transform(
            model.kappa * fields["temperature"][k]
            * fields["omega_over_p"][k]))
        assert float(cp.abs(t_dot[k] - conv_only).max()) == 0.0
    # Conversion is genuinely nonzero here (the test is not vacuous).
    assert float(cp.abs(t_dot).max()) > 0.0


def test_energy_exchange_identity_from_tendency_path_fields(latlon_planet):
    """The Simmons–Burridge exchange identity, both sides INDEPENDENTLY
    assembled in the test from the tendency-path product fields (no calls
    into column_energy_conversion / column_pressure_work): conversion
    equals column-local pressure work to round-off on the product grid."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=29)
    fields, _, _ = _thermo_mass(model, state)

    conv = None
    work = None
    for k in range(model.nlev):
        dsig = model.sigma.thickness[k]
        c_term = dsig * model.r_dry * fields["temperature"][k] \
            * fields["omega_over_p"][k]
        w_term = dsig * (model.r_dry * fields["temperature"][k]
                         * fields["v_grad_lnps"][k]
                         - (fields["phi_full"][k] - fields["phi_surface"])
                         * fields["g_full"][k])
        conv = c_term if conv is None else conv + c_term
        work = w_term if work is None else work + w_term
    scale = max(float(cp.abs(conv).max()), float(cp.abs(work).max()))
    assert scale > 0.0
    assert float(cp.abs(conv - work).max()) < 1e-12 * scale


def test_energy_exchange_identity_from_tendency_path_geodesic(
        geodesic_planet):
    import cupy as cp
    model = _make_model(geodesic_planet)
    state = _band_limited_state(model, seed=31)
    fields, _, _ = _thermo_mass(model, state)
    conv = None
    work = None
    for k in range(model.nlev):
        dsig = model.sigma.thickness[k]
        c_term = dsig * model.r_dry * fields["temperature"][k] \
            * fields["omega_over_p"][k]
        w_term = dsig * (model.r_dry * fields["temperature"][k]
                         * fields["v_grad_lnps"][k]
                         - (fields["phi_full"][k] - fields["phi_surface"])
                         * fields["g_full"][k])
        conv = c_term if conv is None else conv + c_term
        work = w_term if work is None else work + w_term
    scale = max(float(cp.abs(conv).max()), float(cp.abs(work).max()))
    assert scale > 0.0
    assert float(cp.abs(conv - work).max()) < 1e-12 * scale


def test_lnps_tendency_matches_independent_product_reference(latlon_planet):
    """Continuity consistency: the spectral ln p_s tendency equals the
    analysis of a reference -sum G dsigma assembled in the test from the
    reconstruction's primitive ingredients (u, v, lnps derivatives, delta
    fields), analyzed once and truncated once."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=37)
    fields, _, lnps_dot = _thermo_mass(model, state)

    g_ref = None
    dlnps_ref = None
    for k in range(model.nlev):
        adv = (fields["u"][k] * fields["lnps_lam"]
               - fields["v"][k] * fields["lnps_snt"]) / fields["coslat"]
        g_k = fields["delta"][k] + adv
        term = -model.sigma.thickness[k] * g_k
        dlnps_ref = term if dlnps_ref is None else dlnps_ref + term
    ref_lm = model._truncate(model._ps.sh.transform(dlnps_ref))
    scale = float(cp.abs(ref_lm).max())
    assert scale > 0.0
    assert float(cp.abs(lnps_dot - ref_lm).max()) < 1e-12 * scale


def test_lnps_tendency_band_limited_matches_state_diagnostic(latlon_planet):
    """Band-limited Gauss case: the product-path spectral tendency and the
    analysis of the state-grid diagnostic dlnps_dt agree to round-off
    (both quadratures are exact for this integrand)."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=41)
    _, _, lnps_dot = _thermo_mass(model, state)
    diag = model.continuity_diagnostics(state)
    ref_lm = model._truncate(model.sh.transform(diag["dlnps_dt"]))
    scale = float(cp.abs(ref_lm).max())
    assert scale > 0.0
    assert float(cp.abs(lnps_dot - ref_lm).max()) < 1e-11 * scale


def test_thermo_lnps_monopoles_evolve_and_boundaries_stay_pinned(
        latlon_planet):
    """The T and ln p_s tendency monopoles are NOT zeroed (global-mean T
    evolves through conversion; global-mean ln p_s through the mass
    divergence), and the structural sigma_dot boundary zeros survive the
    tendency-path evaluation."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=43)
    fields, t_dot, lnps_dot = _thermo_mass(model, state)
    assert float(cp.abs(lnps_dot[0, 0])) > 0.0
    assert max(float(cp.abs(t_dot[k, 0, 0])) for k in range(model.nlev)) > 0.0
    assert float(cp.abs(fields["sigma_dot"][0]).max()) == 0.0
    assert float(cp.abs(fields["sigma_dot"][-1]).max()) == 0.0


# ---------------------------------------------------------------------------
# Phase 5 — momentum (zeta / delta) tendencies
# ---------------------------------------------------------------------------

def _momentum(model, state):
    fields = model._tendency_product_fields(state.coeffs)
    zeta_dot, delta_dot = model._momentum_tendencies(state.coeffs, fields)
    return fields, zeta_dot, delta_dot


def test_momentum_tendencies_zero_at_isothermal_rest(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    _, zeta_dot, delta_dot = _momentum(model, state)
    assert float(cp.abs(zeta_dot).max()) == 0.0
    assert float(cp.abs(delta_dot).max()) == 0.0


def _bve_degeneracy_errors(planet, nlev=4):
    """Max relative error of the per-level PE zeta tendency against the
    BVE tendency for delta = 0, per-level horizontally uniform T, uniform
    ln p_s (sigma_dot = 0). Returns (err, scale, max |extra| rows)."""
    import cupy as cp
    from planetary_sandbox.physics.barotropic import (
        BarotropicState, BarotropicVorticity)
    from planetary_sandbox.run.bve.initial_conditions import make_ic

    model = _make_model(planet, nlev=nlev)
    zeta_lm = planet.sh.transform(make_ic("rh4", planet))
    zeta_lm[0, :] = 0.0

    state = _rest_state(model)
    for k in range(model.nlev):
        state.zeta[k] += zeta_lm
        state.temperature[k, 0, 0] = (T0 - 6.0 * k) * SQRT4PI

    bve = BarotropicVorticity(planet, viscosity=0.0)
    bve_dot = bve.tendency(BarotropicState(zeta_lm), None)

    _, zeta_dot, _ = _momentum(model, state)
    err = max(float(cp.abs(zeta_dot[k] - bve_dot).max())
              for k in range(model.nlev))
    scale = float(cp.abs(bve_dot).max())
    return err, scale


def test_bve_degeneracy_latlon(latlon_planet):
    """The acceptance invariant: with delta = 0, horizontally uniform T,
    uniform ln p_s (so sigma_dot = 0 and Z = 0 bitwise), the PE zeta
    tendency at every level reproduces BarotropicVorticity.tendency —
    only round-off (division association) separates the two paths."""
    err, scale = _bve_degeneracy_errors(latlon_planet)
    assert err <= 1e-12 * scale


def test_bve_degeneracy_geodesic(geodesic_planet):
    """Same pointwise path on the geodesic backend: the SWE precedent
    shows the two evaluations differ only by round-off there too."""
    err, scale = _bve_degeneracy_errors(geodesic_planet)
    assert err <= 1e-12 * scale


def test_zero_flow_structured_T_gives_hydrostatic_pgf_divergence(
        latlon_planet):
    """Zero flow + structured T + uniform ln p_s: the zeta tendency is
    bitwise zero and the delta tendency is exactly the diagonal spectral
    -lap(Phi) of the hydrostatic geopotential of that T field (E = 0, all
    nonlinear vectors bitwise zero) — the analytically expected
    pressure-gradient response, verified against an independent spectral
    hydrostatic reconstruction in the test."""
    import cupy as cp
    from planetary_sandbox.physics.sigma_coordinate import (
        hydrostatic_geopotential)
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    for k in range(model.nlev):
        state.temperature[k, 0, 0] = (T0 - 7.0 * k) * SQRT4PI
        state.temperature[k, 4, 2] = 3.0 - 1.0j
        state.temperature[k, 2, 0] = 1.5 * (k + 1)
    _, zeta_dot, delta_dot = _momentum(model, state)

    assert float(cp.abs(zeta_dot).max()) == 0.0

    phi_full_lm, _ = hydrostatic_geopotential(
        model.sigma, state.temperature, model.phi_surface_lm, model.r_dry)
    expected = -model.lap_eigs[:, None] * phi_full_lm
    scale = float(cp.abs(expected).max())
    assert scale > 0.0
    assert float(cp.abs(delta_dot - expected).max()) < 1e-12 * scale


def test_momentum_monopole_rows_are_bitwise_zero(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _band_limited_state(model, seed=47)
    _, zeta_dot, delta_dot = _momentum(model, state)
    assert float(cp.abs(zeta_dot[:, 0, :]).max()) == 0.0
    assert float(cp.abs(delta_dot[:, 0, :]).max()) == 0.0
    # Not vacuous: the tendencies themselves are nonzero.
    assert float(cp.abs(zeta_dot).max()) > 0.0
    assert float(cp.abs(delta_dot).max()) > 0.0


def test_momentum_tendencies_finite_on_geodesic(geodesic_planet):
    import cupy as cp
    model = _make_model(geodesic_planet)
    state = _band_limited_state(model, seed=53)
    _, zeta_dot, delta_dot = _momentum(model, state)
    assert bool(cp.isfinite(zeta_dot).all())
    assert bool(cp.isfinite(delta_dot).all())
    assert float(cp.abs(zeta_dot[:, 0, :]).max()) == 0.0
    assert float(cp.abs(delta_dot[:, 0, :]).max()) == 0.0


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
