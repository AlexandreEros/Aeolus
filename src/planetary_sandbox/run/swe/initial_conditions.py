"""Initial conditions for the shallow-water core.

Each scenario returns a :class:`ShallowWaterState` built *spectrally* (no
grid round trip), so the states are exactly monopole-free and exactly
band-limited. The available scenarios are the minimal set required by the
first shallow-water milestone (see run/swe/config.SWE_SCENARIOS).

Topography-aware construction
-----------------------------
The ``rest``, ``gravity_wave``, and ``williamson2`` scenarios specify a
velocity field and a FREE-SURFACE geopotential anomaly ``phi_fs'`` (both
monopole-free); the prognostic thickness perturbation is then

    phi = phi_fs' - phi_s'

The ``williamson5`` scenario is the deliberate, benchmark-defining
exception: it prescribes the THICKNESS field directly and never
compensates for terrain (see ``_williamson5``).

where ``phi_s'`` is the mean-removed surface geopotential of the model's
fixed bottom topography (exactly zero for a flat bottom, so every flat-
bottom state is bit-for-bit identical to the historical construction).
This makes each scenario well-defined over terrain: ``rest`` is the exact
lake-at-rest state (constant free surface, zero velocity), and the flowing
scenarios launch their historical wind/free-surface pair over the mountain
(for ``williamson2`` that is a mountain-flow experiment, not the flat-
bottom steady solution). The global-mean thickness is exactly the model's
``mean_depth`` in every case.

Every scenario validates its state before returning, so terrain that
protrudes through the fluid layer fails here — before integration — with
the model's explicit thickness-collapse diagnosis.
"""
from __future__ import annotations

import math

import cupy as cp

from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                     ShallowWaterState)
from .config import W5_U0_MS


def _rest(model: ShallowWaterModel) -> ShallowWaterState:
    """Exact resting state: zero velocity, constant free surface.

    Over a flat bottom this is the all-zero state; over topography the
    thickness perturbation is -phi_s', giving spatially varying thickness
    under a spatially constant free-surface geopotential.
    """
    state = ShallowWaterState.zeros(model.l_max)
    if model.has_topography:  # keep the flat state exactly all-(+0.0)
        state.coeffs[2] = -model.phi_s_anom_lm
    model.validate_state(state, context="rest initial condition")
    return state


def _gravity_wave(model: ShallowWaterModel) -> ShallowWaterState:
    """Small-amplitude Y_4^2 free-surface perturbation, fluid at rest.

    On a non-rotating planet with a flat bottom this oscillates at
    omega^2 = Phi0 * l(l+1) / a^2 (the verified dispersion relation); over
    topography it is the same free-surface bump launched on the lake-at-
    rest state.
    """
    l, m = 4, 2
    if model.l_max < l:
        raise ValueError(
            f"gravity_wave scenario needs l_max >= {l}, got {model.l_max}")
    state = ShallowWaterState.zeros(model.l_max)
    if model.has_topography:  # keep the flat state bit-identical to history
        state.coeffs[2] = -model.phi_s_anom_lm
    state.coeffs[2, l, m] += 1e-3 * model.phi0
    model.validate_state(state, context="gravity_wave initial condition")
    return state


def _williamson2(model: ShallowWaterModel) -> ShallowWaterState:
    """Williamson et al. (1992) case 2 (alpha = 0) wind/free-surface pair.

    u = u0*cos(lat) with u0 = 2*pi*a/(12 days); the balanced free-surface
    geopotential anomaly is C*(1/3 - sin^2 lat) with C = a*Omega*u0 +
    u0^2/2. Over a flat bottom this is the exact steady solution for any
    configured mean depth as long as the fluid thickness stays positive
    (validated); the canonical g*h0 = 2.94e4 configuration corresponds to
    mean depth (2.94e4 - C/3)/g. Over non-flat topography the same wind and
    free surface are launched above the terrain — a smooth mountain-flow
    experiment (NOT the steady solution, and NOT Williamson case 5, whose
    mountain is conical).
    """
    a = model.R
    omega = model.Omega
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    C = a * omega * u0 + 0.5 * u0 * u0

    state = ShallowWaterState.zeros(model.l_max)
    # zeta = (2*u0/a) sin(lat): pure (1,0); sin(lat) = sqrt(4*pi/3) Y_1^0.
    state.coeffs[0, 1, 0] = (2.0 * u0 / a) * math.sqrt(4.0 * math.pi / 3.0)
    # phi_fs' = C*(1/3 - sin^2 lat) = -(2C/3) P2: pure (2,0) with
    # P2 = sqrt(4*pi/5) Y_2^0.
    if model.has_topography:  # keep the flat state bit-identical to history
        state.coeffs[2] = -model.phi_s_anom_lm
    state.coeffs[2, 2, 0] += -(4.0 * C / 3.0) * math.sqrt(math.pi / 5.0)
    model.validate_state(state, context="williamson2 initial condition")
    return state


def _williamson5(model: ShallowWaterModel) -> ShallowWaterState:
    """Williamson et al. (1992) case 5: zonal flow over an isolated mountain.

    The wind/THICKNESS pair is the Williamson-2 shape with u0 = 20 m/s:

        u = u0*cos(lat),  v = 0,
        h = h0 - (C/g) sin^2(lat),   C = a*Omega*u0 + u0^2/2

    built exactly in spectral space: zeta = (2*u0/a) sin(lat) is the pure
    (1,0) mode, delta = 0, and the mean-zero THICKNESS perturbation is
    phi = C*(1/3 - sin^2 lat), a pure (2,0) mode. The canonical mean depth
    carried by the model is H = h0 - C/(3g) (resolved by the config layer).

    THE DEFINING W5 CONVENTION — deliberately different from the
    terrain-aware ``williamson2`` scenario above: the thickness field is
    NOT compensated by the surface-geopotential anomaly (no
    ``phi -= phi_s'`` term). The mountain contributes separately through
    the fixed phi_s, so the initial FREE SURFACE ``Phi0 + phi + phi_s`` is
    raised over the cone; that raised surface is exactly the canonical
    topographic forcing of test case 5. The state is therefore identical
    whether or not the model carries terrain (a tested invariant), and a
    regression test fails if this construction is ever changed to
    ``phi = phi_balanced - phi_s'``.
    """
    a = model.R
    omega = model.Omega
    u0 = W5_U0_MS
    C = a * omega * u0 + 0.5 * u0 * u0

    state = ShallowWaterState.zeros(model.l_max)
    # zeta = (2*u0/a) sin(lat): pure (1,0); sin(lat) = sqrt(4*pi/3) Y_1^0.
    state.coeffs[0, 1, 0] = (2.0 * u0 / a) * math.sqrt(4.0 * math.pi / 3.0)
    # phi = C*(1/3 - sin^2 lat) = -(2C/3) P2: pure (2,0) with
    # P2 = sqrt(4*pi/5) Y_2^0. NO terrain compensation (docstring).
    state.coeffs[2, 2, 0] = -(4.0 * C / 3.0) * math.sqrt(math.pi / 5.0)
    model.validate_state(state, context="williamson5 initial condition")
    return state


SWE_INITIAL_CONDITIONS = {
    "rest": _rest,
    "gravity_wave": _gravity_wave,
    "williamson2": _williamson2,
    "williamson5": _williamson5,
}


def make_swe_ic(name: str, model: ShallowWaterModel) -> ShallowWaterState:
    if name not in SWE_INITIAL_CONDITIONS:
        raise ValueError(
            f"Unknown swe initial condition: {name}. "
            f"Available: {sorted(SWE_INITIAL_CONDITIONS)}")
    return SWE_INITIAL_CONDITIONS[name](model)
