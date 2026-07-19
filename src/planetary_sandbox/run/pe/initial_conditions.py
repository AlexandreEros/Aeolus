"""Initial conditions for the dry primitive-equation core.

Two presets only (the runner milestone scope). Both are built spectrally on
top of :func:`~planetary_sandbox.physics.primitive_equations.isothermal_rest_state`,
so vorticity and divergence are exactly zero, surface pressure is exactly
uniform, and the fields are exactly band-limited (no grid round trip).

``isothermal_rest``
    Exactly resting, horizontally uniform isothermal atmosphere: zeta = delta
    = 0, T_k = ``temperature`` at every level, p_s = ``surface_pressure``
    everywhere. This is the model's exact-rest state (its tendency is exactly
    zero), used to verify that the runner preserves rest bit-for-bit.

``thermal_wave``
    The resting isothermal state plus a single deterministic degree-2
    temperature perturbation. The perturbation is one real spherical-harmonic
    coefficient placed on the (l, m) = (2, 2) sectoral mode at *every* full
    level (a vertically uniform profile), following the repository's
    real-field coefficient convention (a single real coefficient at
    (l, m>0) synthesizes a valid longitude-varying real field, exactly as the
    shallow-water ``gravity_wave`` preset does). ``thermal_amplitude`` is that
    coefficient's value in kelvin; ~1 K keeps the perturbed temperature
    positive everywhere. Surface pressure stays uniform and the initial winds
    stay zero, so the state is deliberately *unbalanced* — it exists to show
    the PE model launches a smooth, finite response, not a balanced flow.
"""
from __future__ import annotations

from planetary_sandbox.physics.primitive_equations import (
    PrimitiveEquationsModel, PrimitiveEquationsState, isothermal_rest_state)

#: The thermal-wave perturbation lives on this single real spherical-harmonic
#: mode (degree 2, order 2): a sectoral, longitude-varying, low-degree mode
#: well below any usable l_max.
THERMAL_WAVE_DEGREE = 2
THERMAL_WAVE_ORDER = 2


def _isothermal_rest(model: PrimitiveEquationsModel, *, temperature: float,
                     surface_pressure: float,
                     thermal_amplitude: float = 0.0) -> PrimitiveEquationsState:
    del thermal_amplitude  # unused: rest has no perturbation
    return isothermal_rest_state(model.l_max, model.nlev,
                                 temperature=temperature,
                                 surface_pressure=surface_pressure)


def _thermal_wave(model: PrimitiveEquationsModel, *, temperature: float,
                  surface_pressure: float,
                  thermal_amplitude: float) -> PrimitiveEquationsState:
    if model.l_max < THERMAL_WAVE_DEGREE:
        raise ValueError(
            f"thermal_wave needs l_max >= {THERMAL_WAVE_DEGREE}, got "
            f"{model.l_max}")
    state = isothermal_rest_state(model.l_max, model.nlev,
                                  temperature=temperature,
                                  surface_pressure=surface_pressure)
    # One real coefficient on the (2, 2) mode at every full level. The base
    # monopole (T0) and the perturbation are the only nonzero temperature
    # coefficients; zeta/delta/ln_ps are untouched (rest, uniform p_s).
    state.temperature[:, THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER] = \
        float(thermal_amplitude)
    return state


PE_INITIAL_CONDITIONS = {
    "isothermal_rest": _isothermal_rest,
    "thermal_wave": _thermal_wave,
}


def make_pe_ic(name: str, model: PrimitiveEquationsModel, *,
               temperature: float, surface_pressure: float,
               thermal_amplitude: float = 0.0) -> PrimitiveEquationsState:
    """Construct a named primitive-equation initial state.

    ``temperature`` (K) and ``surface_pressure`` (Pa) set the resting base
    state; ``thermal_amplitude`` (K) is the degree-2 perturbation coefficient,
    used only by ``thermal_wave``.
    """
    if name not in PE_INITIAL_CONDITIONS:
        raise ValueError(
            f"Unknown pe initial condition: {name}. "
            f"Available: {sorted(PE_INITIAL_CONDITIONS)}")
    return PE_INITIAL_CONDITIONS[name](
        model, temperature=temperature, surface_pressure=surface_pressure,
        thermal_amplitude=thermal_amplitude)
