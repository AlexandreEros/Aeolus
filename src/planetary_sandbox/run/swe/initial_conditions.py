"""Initial conditions for the shallow-water core.

Each scenario returns a :class:`ShallowWaterState` built *spectrally* (no
grid round trip), so the states are exactly monopole-free and exactly
band-limited. The available scenarios are the minimal set required by the
first shallow-water milestone (see run/swe/config.SWE_SCENARIOS).
"""
from __future__ import annotations

import math

import cupy as cp

from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                     ShallowWaterState)


def _rest(model: ShallowWaterModel) -> ShallowWaterState:
    return ShallowWaterState.zeros(model.l_max)


def _gravity_wave(model: ShallowWaterModel) -> ShallowWaterState:
    """Small-amplitude Y_4^2 perturbation of the geopotential, fluid at rest.

    On a non-rotating planet this oscillates at
    omega^2 = Phi0 * l(l+1) / a^2 (the verified dispersion relation); on a
    rotating one it is simply a small gravity-wave demo state.
    """
    l, m = 4, 2
    if model.l_max < l:
        raise ValueError(
            f"gravity_wave scenario needs l_max >= {l}, got {model.l_max}")
    state = ShallowWaterState.zeros(model.l_max)
    state.coeffs[2, l, m] = 1e-3 * model.phi0
    return state


def _williamson2(model: ShallowWaterModel) -> ShallowWaterState:
    """Williamson et al. (1992) case 2 (alpha = 0), an exact steady solution.

    u = u0*cos(lat) with u0 = 2*pi*a/(12 days); the balanced total
    geopotential is Phi0 + C*(1/3 - sin^2 lat) with C = a*Omega*u0 + u0^2/2.
    Valid for any configured mean depth as long as the total geopotential
    stays positive (validated); the canonical g*h0 = 2.94e4 configuration
    corresponds to mean depth (2.94e4 - C/3)/g.
    """
    a = model.R
    omega = model.Omega
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    C = a * omega * u0 + 0.5 * u0 * u0

    state = ShallowWaterState.zeros(model.l_max)
    # zeta = (2*u0/a) sin(lat): pure (1,0); sin(lat) = sqrt(4*pi/3) Y_1^0.
    state.coeffs[0, 1, 0] = (2.0 * u0 / a) * math.sqrt(4.0 * math.pi / 3.0)
    # phi' = C*(1/3 - sin^2 lat) = -(2C/3) P2: pure (2,0) with
    # P2 = sqrt(4*pi/5) Y_2^0.
    state.coeffs[2, 2, 0] = -(4.0 * C / 3.0) * math.sqrt(math.pi / 5.0)
    model.validate_state(state, context="williamson2 initial condition")
    return state


SWE_INITIAL_CONDITIONS = {
    "rest": _rest,
    "gravity_wave": _gravity_wave,
    "williamson2": _williamson2,
}


def make_swe_ic(name: str, model: ShallowWaterModel) -> ShallowWaterState:
    if name not in SWE_INITIAL_CONDITIONS:
        raise ValueError(
            f"Unknown swe initial condition: {name}. "
            f"Available: {sorted(SWE_INITIAL_CONDITIONS)}")
    return SWE_INITIAL_CONDITIONS[name](model)
