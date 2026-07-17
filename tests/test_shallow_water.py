"""Shallow-water core verification (model level, no runner).

Covers the spec's model-level requirements: resting atmosphere, the BVE
limit, Helmholtz velocity reconstruction, the linear gravity-wave dispersion
relation, exact monopole (mass) conservation, hyperdiffusion monopole
safety, and hard state-validation failures.

Most tests run on the Gauss-Legendre lat-lon backend, whose transforms are
exact for band-limited fields, so tolerances can be tight; one BVE-limit
check repeats on the geodesic backend with a looser tolerance.
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

EARTH_RADIUS = 6.371e6


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
def nonrotating_planet():
    return _make_planet(day_hours=math.inf)


# ---------------------------------------------------------------------------
# Resting atmosphere
# ---------------------------------------------------------------------------

def test_resting_state_all_tendencies_zero(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    model = ShallowWaterModel(latlon_planet, mean_depth=1000.0)
    state = ShallowWaterState.zeros(latlon_planet.sh.l_max)
    out = model.tendency(state.coeffs)
    assert float(cp.abs(out).max()) == 0.0


# ---------------------------------------------------------------------------
# BVE limit
# ---------------------------------------------------------------------------

def _bve_limit_max_error(planet):
    """Return (max |SW - BVE| zeta tendency, max |BVE|) for delta=phi=0."""
    import cupy as cp
    from planetary_sandbox.physics.barotropic import (
        BarotropicState, BarotropicVorticity)
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)
    from planetary_sandbox.run.bve.initial_conditions import make_ic

    zeta_lm = planet.sh.transform(make_ic("rh4", planet))
    zeta_lm[0, :] = 0.0  # monopole-clean state (the transform is inexact)

    bve = BarotropicVorticity(planet, viscosity=0.0)
    bve_dot = bve.tendency(BarotropicState(zeta_lm), None)

    sw = ShallowWaterModel(planet, mean_depth=1000.0)
    zeros = cp.zeros_like(zeta_lm)
    sw_dot = sw.tendency(
        ShallowWaterState.from_fields(zeta_lm, zeros, zeros).coeffs)

    err = float(cp.abs(sw_dot[0] - bve_dot).max())
    scale = float(cp.abs(bve_dot).max())
    # delta and phi must not be dragged along by advection alone: their
    # tendencies contain only the (linear-balance) divergence response.
    assert float(cp.abs(sw_dot[2]).max()) == 0.0  # phi: -Phi0*0 - div(0*u)
    return err, scale


def test_bve_limit_matches_bve_tendency_latlon(latlon_planet):
    err, scale = _bve_limit_max_error(latlon_planet)
    # Same pointwise expression evaluated with an extra cos(lat) division;
    # only round-off separates the two paths.
    assert err <= 1e-12 * scale


def test_bve_limit_matches_bve_tendency_geodesic():
    planet = _make_planet(grid_type="geodesic", resolution=3, l_max=10)
    err, scale = _bve_limit_max_error(planet)
    assert err <= 1e-12 * scale


# ---------------------------------------------------------------------------
# Helmholtz velocity reconstruction
# ---------------------------------------------------------------------------

def test_streamfunction_only_state_reconstructs_solid_body(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    planet = latlon_planet
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    a = planet.params.radius
    u0 = 40.0
    l_max = planet.sh.l_max

    # psi = -a*u0*sin(lat) -> u = u0*cos(lat), v = 0, zeta = 2*u0*sin(lat)/a.
    psi_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    psi_lm[1, 0] = -a * u0 * math.sqrt(4.0 * math.pi / 3.0)
    zeta_lm = model.lap_eigs[:, None] * psi_lm

    state = ShallowWaterState.from_fields(
        zeta_lm, cp.zeros_like(zeta_lm), cp.zeros_like(zeta_lm))
    u, v = model.wind_on_state_grid(state)

    lat = cp.asarray(planet.grid.point_latitudes)
    assert float(cp.abs(u - u0 * cp.cos(lat)).max()) <= 1e-10 * u0
    assert float(cp.abs(v).max()) <= 1e-10 * u0

    # The Helmholtz solve itself must return the exact input psi.
    psi_back, chi_back = model.helmholtz(state)
    assert float(cp.abs(psi_back - psi_lm).max()) <= 1e-12 * float(
        cp.abs(psi_lm).max())
    assert float(cp.abs(chi_back).max()) == 0.0


def test_velocity_potential_only_state_reconstructs_meridional_flow(
        latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    planet = latlon_planet
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    a = planet.params.radius
    c0 = a * 10.0  # chi amplitude giving ~10 m/s meridional flow
    l_max = planet.sh.l_max

    # chi = c0*sin(lat) -> u = 0, v = (c0/a)*cos(lat), delta = -2*c0*sin(lat)/a^2.
    chi_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    chi_lm[1, 0] = c0 * math.sqrt(4.0 * math.pi / 3.0)
    delta_lm = model.lap_eigs[:, None] * chi_lm

    state = ShallowWaterState.from_fields(
        cp.zeros_like(delta_lm), delta_lm, cp.zeros_like(delta_lm))
    u, v = model.wind_on_state_grid(state)

    lat = cp.asarray(planet.grid.point_latitudes)
    v_ref = (c0 / a) * cp.cos(lat)
    vmax = float(cp.abs(v_ref).max())
    assert float(cp.abs(v - v_ref).max()) <= 1e-10 * vmax
    assert float(cp.abs(u).max()) <= 1e-10 * vmax


def _div_curl_of_wind(planet, u, v):
    """Numerical divergence and vertical curl of a grid wind field.

    div(u, v) = (1/(R cos)) du/dlambda + (1/(R cos)) d(v cos)/dlat-form,
    computed with the repository's spectral derivative operators; the curl
    is div applied to the rotated field (v, -u).
    """
    import cupy as cp
    sh, so, R = planet.sh, planet.so, planet.params.radius
    coslat = cp.cos(cp.asarray(planet.grid.point_latitudes))

    def div(A, B):
        dA = sh.inv_transform(so.d_lambda_coeffs(sh.transform(A))).real
        sV = sh.inv_transform(so.sin_theta_d_theta_coeffs(
            sh.transform(B * coslat))).real / R
        return dA / coslat - sV / coslat**2

    return div(u, v), div(v, -u)


def test_reconstructed_wind_recovers_vorticity_and_divergence(latlon_planet):
    """Closure check on analytically generated psi-only and chi-only states.

    For the solid-body-type flows (u or v proportional to cos(lat)) the
    fields u*cos(lat), v*cos(lat) are exactly band-limited, so the discrete
    div/curl of the reconstructed wind must reproduce the state's vorticity
    and divergence to round-off (rather than to scalar-truncation level).
    """
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    planet = latlon_planet
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    a = planet.params.radius
    l_max = planet.sh.l_max
    lat = cp.asarray(planet.grid.point_latitudes)

    # --- streamfunction-only: psi = -a*u0*sin(lat) -> zeta = 2*u0*sin/a ---
    u0 = 40.0
    psi_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    psi_lm[1, 0] = -a * u0 * math.sqrt(4.0 * math.pi / 3.0)
    zeta_lm = model.lap_eigs[:, None] * psi_lm
    state = ShallowWaterState.from_fields(
        zeta_lm, cp.zeros_like(zeta_lm), cp.zeros_like(zeta_lm))
    u, v = model.wind_on_state_grid(state)
    div_num, curl_num = _div_curl_of_wind(planet, u, v)
    zeta_ref = (2.0 * u0 / a) * cp.sin(lat)
    zeta_scale = 2.0 * u0 / a
    assert float(cp.abs(curl_num - zeta_ref).max()) <= 1e-10 * zeta_scale
    assert float(cp.abs(div_num).max()) <= 1e-10 * zeta_scale

    # --- velocity-potential-only: chi = a*v0*sin(lat) -> delta = -2*v0*sin/a
    v0 = 10.0
    chi_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    chi_lm[1, 0] = a * v0 * math.sqrt(4.0 * math.pi / 3.0)
    delta_lm = model.lap_eigs[:, None] * chi_lm
    state = ShallowWaterState.from_fields(
        cp.zeros_like(delta_lm), delta_lm, cp.zeros_like(delta_lm))
    u, v = model.wind_on_state_grid(state)
    div_num, curl_num = _div_curl_of_wind(planet, u, v)
    delta_ref = (-2.0 * v0 / a) * cp.sin(lat)
    delta_scale = 2.0 * v0 / a
    assert float(cp.abs(div_num - delta_ref).max()) <= 1e-10 * delta_scale
    assert float(cp.abs(curl_num).max()) <= 1e-10 * delta_scale


# ---------------------------------------------------------------------------
# Linear gravity wave dispersion
# ---------------------------------------------------------------------------

def test_gravity_wave_frequency_matches_dispersion_relation(
        nonrotating_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)
    from planetary_sandbox.run.engine import rk4_step_array

    planet = nonrotating_planet
    assert planet.params.angular_velocity == 0.0

    mean_depth = 1000.0
    model = ShallowWaterModel(planet, mean_depth=mean_depth)
    a = planet.params.radius
    l, m = 4, 2
    omega_exact = math.sqrt(model.phi0 * l * (l + 1)) / a

    l_max = planet.sh.l_max
    phi_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    amp = 1e-4 * model.phi0  # small amplitude: linear regime
    phi_lm[l, m] = amp
    state = ShallowWaterState.from_fields(
        cp.zeros_like(phi_lm), cp.zeros_like(phi_lm), phi_lm)

    # Starting from rest, phi_lm(t) = amp*cos(omega t): its first zero
    # crossing is at t = pi/(2*omega).
    dt = 300.0
    y = state.coeffs
    t = 0.0
    prev_val, prev_t = float(y[2, l, m].real), 0.0
    t_zero = None
    max_steps = int(2.0 * math.pi / omega_exact / dt) + 10
    for _ in range(max_steps):
        y = rk4_step_array(model.tendency, y, t, dt)
        t += dt
        val = float(y[2, l, m].real)
        if prev_val > 0.0 >= val:
            # Linear interpolation of the crossing time.
            t_zero = prev_t + dt * prev_val / (prev_val - val)
            break
        prev_val, prev_t = val, t
    assert t_zero is not None, "no zero crossing found — wave did not oscillate"

    omega_num = math.pi / (2.0 * t_zero)
    assert omega_num == pytest.approx(omega_exact, rel=1e-3)


# ---------------------------------------------------------------------------
# Mass conservation and hyperdiffusion monopole safety
# ---------------------------------------------------------------------------

def test_monopoles_conserved_exactly_through_nonlinear_steps(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)
    from planetary_sandbox.run.engine import rk4_step_array

    planet = latlon_planet
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    l_max = planet.sh.l_max

    zeta_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    phi_lm = cp.zeros_like(zeta_lm)
    zeta_lm[3, 2] = 2e-5 * (1.0 + 0.5j)
    zeta_lm[1, 0] = 1e-5
    phi_lm[2, 1] = 0.05 * model.phi0 * (0.7 - 0.2j)  # strongly nonlinear
    state = ShallowWaterState.from_fields(
        zeta_lm, cp.zeros_like(zeta_lm), phi_lm)

    y = state.coeffs
    for _ in range(20):
        y = rk4_step_array(model.tendency, y, 0.0, 120.0)

    # l=0 tendency rows are pinned, and RK4 is a linear combination of
    # tendencies, so the monopoles are conserved to the last bit.
    assert float(cp.abs(y[:, 0, :]).max()) == 0.0
    model.validate_state(ShallowWaterState(y), context="after 20 RK4 steps")


def test_hyperdiffusion_damps_but_never_touches_monopoles(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    planet = latlon_planet
    l_max = planet.sh.l_max
    zeta_lm = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    zeta_lm[5, 3] = 1e-5
    state = ShallowWaterState.from_fields(
        zeta_lm, cp.zeros_like(zeta_lm), cp.zeros_like(zeta_lm))

    base = ShallowWaterModel(planet, mean_depth=1000.0)
    damped = ShallowWaterModel(planet, mean_depth=1000.0,
                               hyperdiffusion_nu4=1e16)
    dot0 = base.tendency(state.coeffs)
    dot1 = damped.tendency(state.coeffs)

    diff = dot1 - dot0
    lam = 5 * 6 / planet.params.radius**2
    expected = -1e16 * lam**2 * zeta_lm[5, 3]
    assert complex(diff[0, 5, 3]) == pytest.approx(complex(expected), rel=1e-12)
    assert float(cp.abs(diff[:, 0, :]).max()) == 0.0  # monopoles untouched


# ---------------------------------------------------------------------------
# State validation failures
# ---------------------------------------------------------------------------

def test_validate_state_rejects_nan(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState, ShallowWaterStateError)

    model = ShallowWaterModel(latlon_planet, mean_depth=1000.0)
    state = ShallowWaterState.zeros(latlon_planet.sh.l_max)
    state.coeffs[0, 2, 1] = cp.nan
    with pytest.raises(ShallowWaterStateError, match="NaN"):
        model.validate_state(state)


def test_validate_state_rejects_nonzero_monopole(latlon_planet):
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState, ShallowWaterStateError)

    model = ShallowWaterModel(latlon_planet, mean_depth=1000.0)
    state = ShallowWaterState.zeros(latlon_planet.sh.l_max)
    state.coeffs[2, 0, 0] = 1.0  # mean phi belongs in Phi0, not the state
    with pytest.raises(ShallowWaterStateError, match="phi monopole"):
        model.validate_state(state)


def test_validate_state_rejects_negative_fluid_depth(latlon_planet):
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState, ShallowWaterStateError)

    model = ShallowWaterModel(latlon_planet, mean_depth=1000.0)
    state = ShallowWaterState.zeros(latlon_planet.sh.l_max)
    # phi = A*sin(lat) with amplitude far beyond Phi0: depth collapses.
    state.coeffs[2, 1, 0] = 5.0 * model.phi0
    with pytest.raises(ShallowWaterStateError, match="strictly positive"):
        model.validate_state(state)


def test_validate_state_checks_positivity_on_product_sampling():
    """Audit finding 1: depth can collapse on the (finer) product grid while
    every state-grid point stays positive; validation must scan both."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState, ShallowWaterStateError)

    planet = _make_planet(grid_type="geodesic", resolution=3, l_max=10)
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    fine_sh = planet.so.product_sh
    assert fine_sh is not None and fine_sh is not planet.sh

    # Find a high-degree mode whose minimum is sampled noticeably deeper on
    # the fine product grid than on the coarse state grid.
    l_max = planet.sh.l_max
    best = None
    for l, m in [(10, 1), (10, 2), (9, 1), (9, 2), (10, 3)]:
        mode = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
        mode[l, m] = 1.0
        min_state = float(planet.sh.inv_transform(mode).real.min())
        min_prod = float(fine_sh.inv_transform(mode).real.min())
        ratio = min_prod / min_state  # both negative; > 1 means deeper on fine
        if best is None or ratio > best[0]:
            best = (ratio, l, m, min_state, min_prod)
    ratio, l, m, min_state, min_prod = best
    assert ratio > 1.02, (
        f"no mode found with a deeper product-grid minimum (best {best})")

    # Amplitude between the two collapse thresholds: positive everywhere on
    # the state grid, negative somewhere on the product grid.
    amp = model.phi0 / (0.5 * (abs(min_state) + abs(min_prod)))
    state = ShallowWaterState.zeros(l_max)
    state.coeffs[2, l, m] = amp
    assert model.phi0 + amp * min_state > 0.0   # state grid alone looks fine
    assert model.phi0 + amp * min_prod < 0.0    # but the products see collapse

    with pytest.raises(ShallowWaterStateError, match="strictly positive"):
        model.validate_state(state)

    # The extrema helper reports the envelope the validator (and the CFL
    # characteristic speed) use.
    lo, hi = model.total_geopotential_extrema(state)
    assert lo < 0.0 < hi


def test_rk4_stage_validation_catches_transient_depth_collapse(
        nonrotating_planet):
    """Audit finding 2: a too-large gravity-wave step drives intermediate RK4
    stages through negative depth while the accepted state looks valid; the
    stage validator must fail explicitly."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState, ShallowWaterStateError)
    from planetary_sandbox.run.engine import rk4_step_array

    planet = nonrotating_planet
    model = ShallowWaterModel(planet, mean_depth=1000.0)
    l, m = 4, 2
    l_max = planet.sh.l_max

    # Amplitude: field minimum at -0.5*Phi0 (valid initial state), stepped
    # with dt*omega = 3.2 so the y + dt*k3 stage state scales the mode by
    # ~(1 - (omega*dt)^2) = -9.24: far past depth collapse.
    unit = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    unit[l, m] = 1.0
    unit_min = float(planet.sh.inv_transform(unit).real.min())
    amp = 0.5 * model.phi0 / abs(unit_min)
    state = ShallowWaterState.zeros(l_max)
    state.coeffs[2, l, m] = amp
    model.validate_state(state)  # the initial state itself is valid

    omega = (model.phi0 * l * (l + 1)) ** 0.5 / planet.params.radius
    dt = 3.2 / omega

    def validator(y_stage):
        model.validate_state(ShallowWaterState(y_stage), context="RK4 stage")

    with pytest.raises(ShallowWaterStateError, match="strictly positive"):
        rk4_step_array(model.tendency, state.coeffs, 0.0, dt,
                       stage_validator=validator)


def test_model_rejects_bad_parameters(latlon_planet):
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel

    with pytest.raises(ValueError, match="mean_depth"):
        ShallowWaterModel(latlon_planet, mean_depth=0.0)
    with pytest.raises(ValueError, match="gravity"):
        ShallowWaterModel(latlon_planet, gravity=-9.8, mean_depth=1000.0)
    with pytest.raises(ValueError, match="hyperdiffusion"):
        ShallowWaterModel(latlon_planet, mean_depth=1000.0,
                          hyperdiffusion_nu4=-1.0)
