"""Williamson et al. (1992) test case 2: steady nonlinear zonal geostrophic flow.

The exact steady solution (flow orientation alpha = 0):

    u    = u0 * cos(lat),          u0 = 2*pi*a / (12 days)
    v    = 0
    g*h  = g*h0 - C * sin^2(lat),  C  = a*Omega*u0 + u0^2/2,  g*h0 = 2.94e4

In this repository's prognostics: zeta = (2*u0/a)*sin(lat) is the pure (1,0)
mode, delta = 0, and the PERTURBATION geopotential is the zero-mean part
phi' = C*(1/3 - sin^2 lat) = -(2C/3)*P2(sin lat), a pure (2,0) mode; the
global mean Phi0 = g*h0 - C/3 is carried by the model constant (mean depth
H = Phi0/g), not by the state.

Every field in play is band-limited at degree <= 2, so on the Gauss-Legendre
lat-lon backend the discrete solution is steady to round-off: the measured
1-day drift is ~1e-15 (relative) and energy/mass drift is exactly zero. The
geodesic backend's transform is inexact; its measured residuals are ~1e-5.
Tolerances below are those measurements with two orders of magnitude of
headroom — they detect any formulation error, which would show up at O(1).
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

GRAVITY = 9.80616                      # m/s^2 (Williamson et al. 1992)
GH0 = 2.94e4                           # m^2/s^2
OMEGA = 7.29212e-5                     # s^-1
DAY_HOURS = 2.0 * math.pi / OMEGA / 3600.0   # sidereal day giving Omega


def _make_planet(grid_type="latlon", nlat=32, nlon=64, l_max=15,
                 resolution=3):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    return Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=DAY_HOURS),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)


def make_williamson2(planet):
    """Return (model, state, refs) for the Williamson-2 steady solution."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterModel, ShallowWaterState)

    a = float(planet.params.radius)
    omega = float(planet.params.angular_velocity)
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    C = a * omega * u0 + 0.5 * u0 * u0
    mean_depth = (GH0 - C / 3.0) / GRAVITY   # Phi0 = mean(g*h) = g*h0 - C/3

    model = ShallowWaterModel(planet, gravity=GRAVITY, mean_depth=mean_depth)

    n = planet.sh.l_max + 1
    zeta_lm = cp.zeros((n, n), dtype=cp.complex128)
    phi_lm = cp.zeros_like(zeta_lm)
    zeta_lm[1, 0] = (2.0 * u0 / a) * math.sqrt(4.0 * math.pi / 3.0)
    phi_lm[2, 0] = -(4.0 * C / 3.0) * math.sqrt(math.pi / 5.0)

    state = ShallowWaterState.from_fields(
        zeta_lm, cp.zeros_like(zeta_lm), phi_lm)
    refs = {"u0": u0, "C": C, "a": a, "omega": omega}
    return model, state, refs


def _tendency_scales(refs, phi0):
    """Characteristic magnitudes of the individual tendency terms.

    Residuals are normalized against these (the sizes of the terms that must
    cancel), so the assertions measure genuine cancellation quality.
    """
    u0, a, omega = refs["u0"], refs["a"], refs["omega"]
    eta_amp = 2.0 * u0 / a + 2.0 * omega
    return {
        "zeta": (2.0 * u0 / a) * (u0 / a),   # advective scale
        "delta": 2.0 * u0 * eta_amp / a,     # size of curl / lap(K+phi) terms
        "phi": phi0 * (2.0 * u0 / a),        # Phi0*delta scale
    }


# ---------------------------------------------------------------------------
# Initial tendencies
# ---------------------------------------------------------------------------

def test_w2_initial_tendencies_vanish_latlon():
    import cupy as cp

    planet = _make_planet()
    model, state, refs = make_williamson2(planet)
    model.validate_state(state, context="Williamson-2 initial state")

    dot = model.tendency(state.coeffs)
    scales = _tendency_scales(refs, model.phi0)
    # Measured residuals: zeta/phi exactly 0.0, delta ~1e-14 of its term
    # scale (pure float cancellation; every field is band-limited, so the
    # Gauss transform is exact).
    assert float(cp.abs(dot[0]).max()) <= 1e-13 * scales["zeta"]
    assert float(cp.abs(dot[1]).max()) <= 1e-12 * scales["delta"]
    assert float(cp.abs(dot[2]).max()) <= 1e-13 * scales["phi"]


def test_w2_initial_tendencies_small_geodesic():
    import cupy as cp

    planet = _make_planet(grid_type="geodesic", resolution=3, l_max=10)
    model, state, refs = make_williamson2(planet)

    dot = model.tendency(state.coeffs)
    scales = _tendency_scales(refs, model.phi0)
    # Measured: delta residual ~6.6e-5 of the term scale (geodesic transform
    # quadrature error), zeta/phi exactly zero (zonal flow: no lambda
    # dependence anywhere).
    assert float(cp.abs(dot[0]).max()) <= 1e-12 * scales["zeta"]
    assert float(cp.abs(dot[1]).max()) <= 5e-3 * scales["delta"]
    assert float(cp.abs(dot[2]).max()) <= 1e-12 * scales["phi"]


# ---------------------------------------------------------------------------
# Short-term stability and agreement with the analytic solution
# ---------------------------------------------------------------------------

def _integrate_fixed_cfl(planet, model, state, days):
    """RK4-integrate for `days` at the initial advective+gravity-wave CFL dt."""
    import math as _math
    from planetary_sandbox.run.engine import (advective_cfl_timestep,
                                              rk4_step_array)

    length_scale = getattr(planet.grid, "cfl_length_scale", None)
    dt = advective_cfl_timestep(
        length_scale, model.max_characteristic_speed(state))
    n_steps = int(_math.ceil(days * 86400.0 / dt))
    dt = days * 86400.0 / n_steps
    y = state.coeffs.copy()
    for i in range(n_steps):
        y = rk4_step_array(model.tendency, y, i * dt, dt)
    return y, dt, n_steps


def _grid_height_error(planet, model, y, y0):
    """Williamson-style normalized l2 and linf error of the height field."""
    import cupy as cp
    w = cp.asarray(planet.sh.weights)
    h = model.phi0 + planet.sh.inv_transform(y[2]).real
    h_ref = model.phi0 + planet.sh.inv_transform(y0[2]).real
    l2 = float(cp.sqrt(cp.sum(w * (h - h_ref) ** 2) / cp.sum(w * h_ref**2)))
    linf = float(cp.abs(h - h_ref).max() / cp.abs(h_ref).max())
    return l2, linf


def _total_energy_mass(planet, model, y):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    fields = model.characteristic_fields(ShallowWaterState(y))
    w = cp.asarray(planet.sh.weights) * planet.params.radius**2
    phi_t = fields["phi_total"]
    ke = 0.5 * phi_t * (fields["u"] ** 2 + fields["v"] ** 2)
    pe = 0.5 * phi_t**2
    return float(cp.sum(w * (ke + pe))), float(cp.sum(w * phi_t))


def test_w2_one_day_steady_latlon():
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState

    planet = _make_planet()
    model, state, refs = make_williamson2(planet)
    y0 = state.coeffs.copy()
    E0, M0 = _total_energy_mass(planet, model, y0)

    y, dt, n_steps = _integrate_fixed_cfl(planet, model, state, days=1.0)
    assert n_steps >= 50  # the CFL ceiling actually constrained the run
    model.validate_state(ShallowWaterState(y), context="after 1 day")

    # Steady to round-off (measured ~1e-15 relative after 60 steps).
    err_zeta = float(cp.linalg.norm(y[0] - y0[0]) / cp.linalg.norm(y0[0]))
    err_phi = float(cp.linalg.norm(y[2] - y0[2]) / cp.linalg.norm(y0[2]))
    growth_delta = float(cp.linalg.norm(y[1]) / cp.linalg.norm(y0[0]))
    assert err_zeta <= 1e-12
    assert err_phi <= 1e-12
    assert growth_delta <= 1e-12

    l2, linf = _grid_height_error(planet, model, y, y0)
    assert l2 <= 1e-12 and linf <= 1e-12

    # Energy and mass conservation (measured drift exactly 0.0 here).
    E1, M1 = _total_energy_mass(planet, model, y)
    assert abs(E1 - E0) <= 1e-12 * E0
    assert abs(M1 - M0) <= 1e-13 * M0
    # Mass conservation is also exact at the state level: the phi monopole
    # never moves.
    assert complex(y[2, 0, 0]) == 0j


def test_w2_six_hours_stable_geodesic():
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState

    planet = _make_planet(grid_type="geodesic", resolution=3, l_max=10)
    model, state, refs = make_williamson2(planet)
    y0 = state.coeffs.copy()
    E0, M0 = _total_energy_mass(planet, model, y0)

    y, dt, n_steps = _integrate_fixed_cfl(planet, model, state, days=0.25)
    model.validate_state(ShallowWaterState(y), context="after 6 hours")

    # Measured after 6 h: err_phi 1.3e-5, err_zeta 5.9e-5, dE/E 1.6e-9
    # (geodesic transform error, not a formulation error, which would be O(1)).
    err_zeta = float(cp.linalg.norm(y[0] - y0[0]) / cp.linalg.norm(y0[0]))
    err_phi = float(cp.linalg.norm(y[2] - y0[2]) / cp.linalg.norm(y0[2]))
    assert err_zeta <= 5e-3
    assert err_phi <= 2e-3
    E1, M1 = _total_energy_mass(planet, model, y)
    assert abs(E1 - E0) <= 1e-7 * E0
    assert abs(M1 - M0) <= 1e-9 * M0
    assert complex(y[2, 0, 0]) == 0j
