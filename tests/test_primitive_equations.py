"""Primitive-equation foundation verification (model level; CUDA-gated).

Covers the foundation-milestone requirements of
docs/PRIMITIVE_EQUATIONS_DESIGN.md on real backends: state layout and
views, isothermal hydrostatic exactness, the exactly-resting atmosphere,
uniform-pressure zero-wind behavior, structural sigma_dot impermeability,
column mass closure for genuinely divergent flows, hard state-validation
failures, engine (RK4 array) compatibility, and geodesic/lat-lon backend
agreement of the analytic properties.

Most tests run on the Gauss-Legendre lat-lon backend (exact transforms for
band-limited fields, tight tolerances); the analytic invariants repeat on
the geodesic backend.
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
    return _make_planet(grid_type="geodesic")


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


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

def test_state_layout_and_views(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsState)

    l_max, nlev = latlon_planet.sh.l_max, 4
    state = PrimitiveEquationsState.zeros(l_max, nlev)
    assert state.coeffs.shape == (3 * nlev + 1, l_max + 1, l_max + 1)
    assert state.nlev == nlev
    assert state.coeffs.dtype == cp.complex128

    # Properties are views into the single stack (RK4 arithmetic relies on
    # the stack being the one true storage).
    state.zeta[2, 3, 1] = 1.5 + 0.5j
    state.ln_ps[0, 0] = 11.0
    assert complex(state.coeffs[2, 3, 1]) == 1.5 + 0.5j
    assert complex(state.coeffs[3 * nlev, 0, 0]) == 11.0

    rebuilt = PrimitiveEquationsState.from_fields(
        state.zeta, state.delta, state.temperature, state.ln_ps)
    assert bool((rebuilt.coeffs == state.coeffs).all())


def test_state_rejects_bad_shapes():
    import cupy as cp
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsState, PrimitiveEquationsStateError)

    for shape in [(3, 16, 16),      # 3K+1 cannot be 3
                  (12, 16, 16),     # 3K+1 cannot be 12
                  (13, 16, 8),      # non-square coefficient axes
                  (13, 16)]:        # not 3-D
        with pytest.raises(PrimitiveEquationsStateError):
            PrimitiveEquationsState(cp.zeros(shape, dtype=cp.complex128))


# ---------------------------------------------------------------------------
# Analytic hydrostatic isothermal column
# ---------------------------------------------------------------------------

def _check_isothermal_hydrostatics(planet, rtol):
    import cupy as cp
    model = _make_model(planet)
    state = _rest_state(model)
    fields = model.geopotential_fields(state)
    phi_below = fields["phi_below"]
    phi_full = fields["phi_full"]

    assert float(cp.abs(fields["phi_surface"]).max()) < 1e-9  # no topography
    for k in range(model.nlev):
        analytic = -model.r_dry * T0 * math.log(model.sigma.interfaces[k + 1])
        level = phi_below[k]
        # Horizontally uniform (constant-mode synthesis is exact).
        assert float(cp.abs(level - float(level.reshape(-1)[0])).max()) \
            <= 1e-9 * max(abs(analytic), 1.0)
        assert float(level.reshape(-1)[0]) == pytest.approx(
            analytic, rel=rtol, abs=1e-9)
        analytic_full = -model.r_dry * T0 * (
            math.log(model.sigma.interfaces[k + 1]) - model.sigma.alpha[k])
        assert float(phi_full[k].reshape(-1)[0]) == pytest.approx(
            analytic_full, rel=rtol)
    # Surface boundary condition is exact.
    assert float(cp.abs(phi_below[-1]).max()) < 1e-9


def test_isothermal_hydrostatic_column_latlon(latlon_planet):
    _check_isothermal_hydrostatics(latlon_planet, rtol=1e-12)


def test_isothermal_hydrostatic_column_geodesic(geodesic_planet):
    _check_isothermal_hydrostatics(geodesic_planet, rtol=1e-12)


def test_surface_geopotential_is_representable(latlon_planet):
    """Nonzero Phi_s propagates through the hydrostatic reconstruction."""
    import cupy as cp
    n = latlon_planet.sh.l_max + 1
    phi_s_lm = cp.zeros((n, n), dtype=cp.complex128)
    phi_s_lm[0, 0] = 500.0 * SQRT4PI  # constant 500 m^2/s^2 surface field
    model = _make_model(latlon_planet, surface_geopotential_lm=phi_s_lm)
    state = _rest_state(model)
    fields = model.geopotential_fields(state)
    assert float(fields["phi_below"][-1].reshape(-1)[0]) == pytest.approx(
        500.0, rel=1e-12)
    analytic_top = 500.0 - model.r_dry * T0 * math.log(
        model.sigma.interfaces[1])
    assert float(fields["phi_below"][0].reshape(-1)[0]) == pytest.approx(
        analytic_top, rel=1e-12)


# ---------------------------------------------------------------------------
# Exact resting atmosphere / uniform-pressure zero-wind behavior
# ---------------------------------------------------------------------------

def test_resting_atmosphere_continuity_is_exactly_zero(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    model.validate_state(state, context="isothermal rest")

    u, v = model.wind_on_state_grid(state)
    assert float(cp.abs(u).max()) == 0.0
    assert float(cp.abs(v).max()) == 0.0

    diag = model.continuity_diagnostics(state)
    assert float(cp.abs(diag["g_full"]).max()) == 0.0
    assert float(cp.abs(diag["dlnps_dt"]).max()) == 0.0
    assert float(cp.abs(diag["sigma_dot"]).max()) == 0.0
    assert diag["max_abs_layer_residual"] == 0.0


def test_uniform_pressure_zero_wind_with_structured_temperature(latlon_planet):
    """T structure alone must not create mass flux: G, dlnps, sigma_dot = 0."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    # Horizontally structured, vertically varying temperature.
    for k in range(model.nlev):
        state.temperature[k, 2, 2] = 3.0 * (k + 1)
        state.temperature[k, 0, 0] = (T0 - 10.0 * k) * SQRT4PI
    model.validate_state(state)

    diag = model.continuity_diagnostics(state)
    assert float(cp.abs(diag["g_full"]).max()) == 0.0
    assert float(cp.abs(diag["dlnps_dt"]).max()) == 0.0
    assert float(cp.abs(diag["sigma_dot"]).max()) == 0.0


# ---------------------------------------------------------------------------
# Column continuity for divergent flow
# ---------------------------------------------------------------------------

def _divergent_state(model, seed=3):
    """Band-limited random zeta/delta/ln_ps perturbations on isothermal rest."""
    import numpy as np
    import cupy as cp
    state = _rest_state(model)
    rng = np.random.default_rng(seed)
    n = model.l_max + 1
    for stack, amp in ((state.zeta, 2e-6), (state.delta, 2e-6)):
        for k in range(model.nlev):
            re = rng.standard_normal((n, n))
            im = rng.standard_normal((n, n))
            pert = amp * (re + 1j * im)
            pert[0, :] = 0.0                      # zero monopole row (l=0)
            pert[:, 0] = np.real(pert[:, 0])      # m=0 modes real
            mask = np.tril(np.ones((n, n)))       # m <= l only
            pert *= mask
            pert[11:, :] = 0.0                    # band-limit to l <= 10
            stack[k] += cp.asarray(pert)
    lnps_pert = 1e-3 * rng.standard_normal((n, n)) \
        + 1e-3j * rng.standard_normal((n, n))
    lnps_pert[:, 0] = np.real(lnps_pert[:, 0])
    lnps_pert *= np.tril(np.ones((n, n)))
    lnps_pert[0, 0] = 0.0
    lnps_pert[11:, :] = 0.0
    state.ln_ps[:] += cp.asarray(lnps_pert)
    return state


def test_divergent_flow_mass_closure(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _divergent_state(model)
    model.validate_state(state, context="divergent test state")

    diag = model.continuity_diagnostics(state)
    g_scale = float(cp.abs(diag["g_full"]).max())
    assert g_scale > 0.0
    # Layer mass budget closes to round-off relative to the integrand.
    assert diag["max_abs_layer_residual"] < 1e-12 * g_scale
    # Structural impermeability under real flow.
    assert float(cp.abs(diag["sigma_dot"][0]).max()) == 0.0
    assert float(cp.abs(diag["sigma_dot"][-1]).max()) == 0.0
    # Interior sigma_dot is genuinely nonzero (the test is not vacuous).
    assert float(cp.abs(diag["sigma_dot"][1:-1]).max()) > 0.0


def test_divergent_flow_mass_closure_geodesic(geodesic_planet):
    import cupy as cp
    model = _make_model(geodesic_planet)
    state = _divergent_state(model, seed=9)
    diag = model.continuity_diagnostics(state)
    g_scale = float(cp.abs(diag["g_full"]).max())
    assert diag["max_abs_layer_residual"] < 1e-12 * g_scale
    assert float(cp.abs(diag["sigma_dot"][0]).max()) == 0.0
    assert float(cp.abs(diag["sigma_dot"][-1]).max()) == 0.0


# ---------------------------------------------------------------------------
# Simmons–Burridge energy exchange on the sphere
# ---------------------------------------------------------------------------

def test_resting_atmosphere_exchanges_no_energy(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    ex = model.energy_exchange_diagnostics(state)
    assert float(cp.abs(ex["omega_over_p"]).max()) == 0.0
    assert float(cp.abs(ex["heating"]).max()) == 0.0
    assert float(cp.abs(ex["conversion"]).max()) == 0.0
    assert float(cp.abs(ex["work"]).max()) == 0.0
    assert ex["max_abs_energy_residual"] == 0.0


def _check_energy_identity(planet, seed):
    import cupy as cp
    model = _make_model(planet)
    state = _divergent_state(model, seed=seed)
    diag = model.continuity_diagnostics(state)
    ex = model.energy_exchange_diagnostics(state, continuity=diag)

    scale = max(float(cp.abs(ex["conversion"]).max()),
                float(cp.abs(ex["work"]).max()))
    assert scale > 0.0  # the test is not vacuous
    assert ex["max_abs_energy_residual"] < 1e-12 * scale

    # Heating is exactly kappa * T * (omega/p), pointwise.
    T = model.temperature_on_state_grid(state)
    err = cp.abs(ex["heating"] - model.kappa * T * ex["omega_over_p"])
    assert float(err.max()) == 0.0
    return model, state, ex


def test_energy_exchange_identity_latlon(latlon_planet):
    _check_energy_identity(latlon_planet, seed=3)


def test_energy_exchange_identity_geodesic(geodesic_planet):
    _check_energy_identity(geodesic_planet, seed=9)


def test_energy_identity_with_topography(latlon_planet):
    """Phi_s != 0 shifts Phi but the (Phi - Phi_s) identity still closes."""
    import cupy as cp
    n = latlon_planet.sh.l_max + 1
    phi_s_lm = cp.zeros((n, n), dtype=cp.complex128)
    phi_s_lm[0, 0] = 800.0 * SQRT4PI
    phi_s_lm[2, 1] = 40.0          # a little structure, not just a constant
    model = _make_model(latlon_planet, surface_geopotential_lm=phi_s_lm)
    state = _divergent_state(model, seed=13)
    ex = model.energy_exchange_diagnostics(state)
    scale = max(float(cp.abs(ex["conversion"]).max()),
                float(cp.abs(ex["work"]).max()))
    assert ex["max_abs_energy_residual"] < 1e-12 * scale


def test_uniform_pressure_divergent_flow_energy_bookkeeping(latlon_planet):
    """ln p_s uniform: A = 0 exactly, so omega/p is pure column integral
    and the conversion equals -sum Dsigma (Phi - Phi_s) G computed
    independently from the geopotential fields."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _divergent_state(model, seed=17)
    state.ln_ps[:] = 0.0
    state.ln_ps[0, 0] = math.log(PS0) * SQRT4PI
    diag = model.continuity_diagnostics(state)
    assert float(cp.abs(diag["v_grad_lnps"]).max()) == 0.0

    ex = model.energy_exchange_diagnostics(state, continuity=diag)
    phi = model.geopotential_fields(state)
    manual = None
    for k in range(model.nlev):
        term = -model.sigma.thickness[k] * (
            (phi["phi_full"][k] - phi["phi_surface"]) * diag["g_full"][k])
        manual = term if manual is None else manual + term
    scale = float(cp.abs(manual).max())
    assert scale > 0.0
    assert float(cp.abs(ex["work"] - manual).max()) < 1e-12 * scale
    assert float(cp.abs(ex["conversion"] - manual).max()) < 1e-12 * scale


# ---------------------------------------------------------------------------
# Lorenz-grid vertical transport on the sphere
# ---------------------------------------------------------------------------

def test_resting_atmosphere_has_zero_vertical_transport(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    vt = model.vertical_transport_diagnostics(state)
    for key in ("sigma_dot_dU", "sigma_dot_dV", "sigma_dot_dT",
                "ke_exchange_lhs", "ke_exchange_rhs"):
        assert float(cp.abs(vt[key]).max()) == 0.0, key
    assert vt["max_abs_ke_exchange_residual"] == 0.0


def test_constant_temperature_feels_no_vertical_transport(latlon_planet):
    """Divergent flow with vertically/horizontally uniform T: sigma_dot_dT
    is bitwise zero while momentum transport is genuinely nonzero."""
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _divergent_state(model, seed=21)   # T stays uniform T0
    vt = model.vertical_transport_diagnostics(state)
    assert float(cp.abs(vt["sigma_dot_dT"]).max()) == 0.0
    assert float(cp.abs(vt["sigma_dot_dU"]).max()) > 0.0
    assert float(cp.abs(vt["sigma_dot_dV"]).max()) > 0.0


def _check_ke_exchange_on_sphere(planet, seed):
    """KE exchange relation with the RIGHT side assembled in the test from
    winds and continuity fields only (no vertical-transport code)."""
    import cupy as cp
    model = _make_model(planet)
    state = _divergent_state(model, seed=seed)
    diag = model.continuity_diagnostics(state)
    vt = model.vertical_transport_diagnostics(state, continuity=diag)

    u, v = model.wind_on_state_grid(state)
    manual_rhs = None
    for k in range(model.nlev):
        term = model.sigma.thickness[k] * (u[k] ** 2 + v[k] ** 2) * (
            diag["g_full"][k] + diag["dlnps_dt"])
        manual_rhs = term if manual_rhs is None else manual_rhs + term

    scale = max(float(cp.abs(vt["ke_exchange_lhs"]).max()),
                float(cp.abs(manual_rhs).max()))
    assert scale > 0.0
    assert float(cp.abs(vt["ke_exchange_rhs"] - manual_rhs).max()) \
        < 1e-12 * scale
    assert vt["max_abs_ke_exchange_residual"] < 1e-12 * scale


def test_ke_exchange_relation_latlon(latlon_planet):
    _check_ke_exchange_on_sphere(latlon_planet, seed=25)


def test_ke_exchange_relation_geodesic(geodesic_planet):
    _check_ke_exchange_on_sphere(geodesic_planet, seed=27)


def test_structured_temperature_transport_is_finite_and_nonzero(latlon_planet):
    import cupy as cp
    model = _make_model(latlon_planet)
    state = _divergent_state(model, seed=29)
    for k in range(model.nlev):
        state.temperature[k, 0, 0] = (T0 - 8.0 * k) * SQRT4PI  # lapse
        state.temperature[k, 3, 2] = 2.0 + 0.5 * k             # structure
    vt = model.vertical_transport_diagnostics(state)
    dT = vt["sigma_dot_dT"]
    assert bool(cp.isfinite(dT).all())
    assert float(cp.abs(dT).max()) > 0.0
    # Zero-flow limit: the same T field at rest transports nothing.
    rest = _rest_state(model)
    rest.temperature[:] = state.temperature
    vt_rest = model.vertical_transport_diagnostics(rest)
    assert float(cp.abs(vt_rest["sigma_dot_dT"]).max()) == 0.0


# ---------------------------------------------------------------------------
# Characteristic speed
# ---------------------------------------------------------------------------

def test_characteristic_speed_of_rest_state_is_lamb_bound(latlon_planet):
    from planetary_sandbox.physics.primitive_equations import (
        GAMMA_DRY, R_DRY)
    model = _make_model(latlon_planet)
    state = _rest_state(model)
    expected = math.sqrt(GAMMA_DRY * R_DRY * T0)
    assert model.max_characteristic_speed(state) == pytest.approx(
        expected, rel=1e-9)
    # Hotter atmosphere -> faster waves (monotonic sanity).
    hot = _rest_state(model)
    hot.temperature[:, 0, 0] = (T0 + 40.0) * SQRT4PI
    assert model.max_characteristic_speed(hot) > expected


# ---------------------------------------------------------------------------
# Hard validation failures
# ---------------------------------------------------------------------------

def test_validation_rejects_invalid_states(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsState, PrimitiveEquationsStateError)
    model = _make_model(latlon_planet)

    # Negative temperature (uniform field, negative everywhere).
    bad = _rest_state(model)
    bad.temperature[1, 0, 0] = -50.0 * SQRT4PI
    with pytest.raises(PrimitiveEquationsStateError, match="temperature"):
        model.validate_state(bad)

    # Zero temperature is not strictly positive either.
    bad = _rest_state(model)
    bad.temperature[0, 0, 0] = 0.0
    with pytest.raises(PrimitiveEquationsStateError, match="temperature"):
        model.validate_state(bad)

    # NaN coefficient anywhere is fatal.
    bad = _rest_state(model)
    bad.coeffs[0, 3, 2] = math.nan
    with pytest.raises(PrimitiveEquationsStateError, match="NaN/Inf"):
        model.validate_state(bad)

    # Nonzero vorticity monopole (level-resolved message).
    bad = _rest_state(model)
    bad.zeta[2, 0, 0] = 1e-3
    with pytest.raises(PrimitiveEquationsStateError,
                       match="zeta monopole .* level 3"):
        model.validate_state(bad)

    # Nonzero divergence monopole.
    bad = _rest_state(model)
    bad.delta[0, 0, 0] = 1e-3
    with pytest.raises(PrimitiveEquationsStateError, match="delta monopole"):
        model.validate_state(bad)

    # ln(p_s) large enough that p_s = exp(ln p_s) overflows to Inf.
    bad = _rest_state(model)
    bad.ln_ps[0, 0] = 1000.0 * SQRT4PI
    with pytest.raises(PrimitiveEquationsStateError, match="surface pressure"):
        model.validate_state(bad)

    # Shape mismatch against the model (wrong level count).
    other = PrimitiveEquationsState.zeros(model.l_max, model.nlev + 1)
    with pytest.raises(PrimitiveEquationsStateError, match="shape"):
        model.validate_state(other)


def test_model_rejects_bad_parameters(latlon_planet):
    import cupy as cp
    n = latlon_planet.sh.l_max + 1
    with pytest.raises(ValueError):
        _make_model(latlon_planet, r_dry=-1.0)
    with pytest.raises(ValueError):
        _make_model(latlon_planet, cp_dry=100.0)  # cp_dry <= r_dry
    with pytest.raises(ValueError):
        _make_model(latlon_planet,
                    surface_geopotential_lm=cp.zeros((n, n - 1),
                                                     dtype=cp.complex128))
    bad_phi = cp.zeros((n, n), dtype=cp.complex128)
    bad_phi[1, 1] = math.inf
    with pytest.raises(ValueError):
        _make_model(latlon_planet, surface_geopotential_lm=bad_phi)


# ---------------------------------------------------------------------------
# Engine compatibility (the state is one RK4-able array)
# ---------------------------------------------------------------------------

def test_state_stack_plugs_into_rk4_step_array(latlon_planet):
    """rk4_step_array advances the PE stack like any coefficient array.

    Uses a linear decay tendency (NOT model physics — the model has no
    tendency yet, by design); verifies the classical RK4 amplification
    factor is applied to every row of the (3K+1, n, n) stack.
    """
    import cupy as cp
    from planetary_sandbox.run.engine import rk4_step_array
    model = _make_model(latlon_planet)
    state = _divergent_state(model)

    lam, dt = -1.0 / 900.0, 60.0
    y1 = rk4_step_array(lambda y: lam * y, state.coeffs, 0.0, dt)
    x = lam * dt
    factor = 1.0 + x + x**2 / 2.0 + x**3 / 6.0 + x**4 / 24.0
    assert float(cp.abs(y1 - factor * state.coeffs).max()) < 1e-14 * float(
        cp.abs(state.coeffs).max())


# ---------------------------------------------------------------------------
# Backend agreement of grid-independent scalars
# ---------------------------------------------------------------------------

def test_backends_agree_on_column_scalars(latlon_planet, geodesic_planet):
    """Column physics must not depend on the horizontal backend."""
    m_lat = _make_model(latlon_planet)
    m_geo = _make_model(geodesic_planet)
    s_lat = _rest_state(m_lat)
    s_geo = _rest_state(m_geo)
    c_lat = m_lat.max_characteristic_speed(s_lat)
    c_geo = m_geo.max_characteristic_speed(s_geo)
    assert c_lat == pytest.approx(c_geo, rel=1e-9)
