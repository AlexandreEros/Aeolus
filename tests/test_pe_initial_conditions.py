"""Primitive-equation initial-condition presets (CUDA-gated).

Two presets only (the runner milestone scope): ``isothermal_rest`` exercises
the model's exact-rest property; ``thermal_wave`` adds a single deterministic
degree-2 temperature perturbation so the model has something smooth to
evolve. Both are built spectrally (no grid round trip), so they are exactly
monopole-free in vorticity/divergence and exactly band-limited.
"""
from __future__ import annotations

import math

import pytest

from planetary_sandbox.run.pe.initial_conditions import (
    PE_INITIAL_CONDITIONS, THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER, make_pe_ic)

T0 = 260.0
PS0 = 101325.0
AMP = 1.0
SQRT4PI = math.sqrt(4.0 * math.pi)


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(),
                                reason="CUDA/CuPy not available")


def _make_model(grid_type="latlon", nlat=32, nlon=64, l_max=15, resolution=3,
                nlev=5):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


@pytest.fixture(scope="module")
def latlon_model():
    return _make_model()


@pytest.fixture(scope="module")
def geodesic_model():
    # l_max = 10 at resolution 3 keeps the geodesic transform inside its
    # supported points-per-basis envelope (docs/KNOWN_RISKS.md R-2).
    return _make_model(grid_type="geodesic", l_max=10)


def _both(latlon_model, geodesic_model):
    return (latlon_model, geodesic_model)


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

def test_registry_has_exactly_the_two_presets():
    assert set(PE_INITIAL_CONDITIONS) == {"isothermal_rest", "thermal_wave"}


def test_unknown_preset_raises(latlon_model):
    with pytest.raises(ValueError):
        make_pe_ic("held_suarez", latlon_model,
                   temperature=T0, surface_pressure=PS0)


# ---------------------------------------------------------------------------
# Shared shape / validity properties (both presets, both backends)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["isothermal_rest", "thermal_wave"])
def test_shape_dtype_and_validity(scenario, latlon_model, geodesic_model):
    import cupy as cp
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic(scenario, model, temperature=T0,
                           surface_pressure=PS0, thermal_amplitude=AMP)
        n = model.l_max + 1
        assert state.coeffs.shape == (3 * model.nlev + 1, n, n)
        assert state.coeffs.dtype == cp.complex128
        assert bool(cp.isfinite(state.coeffs).all())
        # Model's hard validation must accept the initial state unmodified.
        model.validate_state(state, context=f"{scenario} initial condition")


@pytest.mark.parametrize("scenario", ["isothermal_rest", "thermal_wave"])
def test_vorticity_and_divergence_are_exactly_zero(scenario, latlon_model,
                                                   geodesic_model):
    import cupy as cp
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic(scenario, model, temperature=T0,
                           surface_pressure=PS0, thermal_amplitude=AMP)
        assert not bool(cp.any(state.zeta))
        assert not bool(cp.any(state.delta))


@pytest.mark.parametrize("scenario", ["isothermal_rest", "thermal_wave"])
def test_surface_pressure_is_uniform_and_positive(scenario, latlon_model,
                                                  geodesic_model):
    import cupy as cp
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic(scenario, model, temperature=T0,
                           surface_pressure=PS0, thermal_amplitude=AMP)
        lnps = state.ln_ps
        # Uniform field: only the (0,0) monopole is nonzero.
        assert cp.isclose(lnps[0, 0], math.log(PS0) * SQRT4PI)
        off = lnps.copy()
        off[0, 0] = 0.0
        assert not bool(cp.any(off))


@pytest.mark.parametrize("scenario", ["isothermal_rest", "thermal_wave"])
def test_deterministic_construction(scenario, latlon_model):
    import cupy as cp
    a = make_pe_ic(scenario, latlon_model, temperature=T0,
                   surface_pressure=PS0, thermal_amplitude=AMP)
    b = make_pe_ic(scenario, latlon_model, temperature=T0,
                   surface_pressure=PS0, thermal_amplitude=AMP)
    assert bool((a.coeffs == b.coeffs).all())


# ---------------------------------------------------------------------------
# isothermal_rest: exact uniform mode + exact-rest tendency
# ---------------------------------------------------------------------------

def test_isothermal_rest_temperature_is_pure_monopole(latlon_model,
                                                      geodesic_model):
    import cupy as cp
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic("isothermal_rest", model, temperature=T0,
                           surface_pressure=PS0)
        for k in range(model.nlev):
            field = state.temperature[k]
            assert cp.isclose(field[0, 0], T0 * SQRT4PI)
            off = field.copy()
            off[0, 0] = 0.0
            assert not bool(cp.any(off))


def test_isothermal_rest_has_exactly_zero_tendency(latlon_model):
    import cupy as cp
    model = latlon_model
    state = make_pe_ic("isothermal_rest", model, temperature=T0,
                       surface_pressure=PS0)
    tend = model.tendency(state.coeffs)
    assert not bool(cp.any(tend))


# ---------------------------------------------------------------------------
# thermal_wave: a single deterministic degree-2 perturbation
# ---------------------------------------------------------------------------

def test_thermal_wave_adds_only_the_degree_two_mode(latlon_model,
                                                    geodesic_model):
    import cupy as cp
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic("thermal_wave", model, temperature=T0,
                           surface_pressure=PS0, thermal_amplitude=AMP)
        for k in range(model.nlev):
            field = state.temperature[k]
            # Base monopole unchanged.
            assert cp.isclose(field[0, 0], T0 * SQRT4PI)
            # The chosen degree-2 mode carries the perturbation.
            assert cp.isclose(field[THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER],
                              AMP)
            # And nothing else off the monopole/perturbation is populated.
            off = field.copy()
            off[0, 0] = 0.0
            off[THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER] = 0.0
            assert not bool(cp.any(off))


def test_thermal_wave_differs_from_rest(latlon_model):
    import cupy as cp
    model = latlon_model
    rest = make_pe_ic("isothermal_rest", model, temperature=T0,
                      surface_pressure=PS0)
    wave = make_pe_ic("thermal_wave", model, temperature=T0,
                      surface_pressure=PS0, thermal_amplitude=AMP)
    assert not bool((rest.coeffs == wave.coeffs).all())
    # The perturbation drives a genuinely nonzero tendency.
    assert bool(cp.any(model.tendency(wave.coeffs)))


def test_thermal_wave_zero_amplitude_is_rest(latlon_model):
    import cupy as cp
    model = latlon_model
    rest = make_pe_ic("isothermal_rest", model, temperature=T0,
                      surface_pressure=PS0)
    wave0 = make_pe_ic("thermal_wave", model, temperature=T0,
                       surface_pressure=PS0, thermal_amplitude=0.0)
    assert bool((rest.coeffs == wave0.coeffs).all())


def test_thermal_wave_keeps_temperature_positive(latlon_model, geodesic_model):
    for model in _both(latlon_model, geodesic_model):
        state = make_pe_ic("thermal_wave", model, temperature=T0,
                           surface_pressure=PS0, thermal_amplitude=AMP)
        t_min, _ = model.temperature_extrema(state)
        assert t_min > 0.0
