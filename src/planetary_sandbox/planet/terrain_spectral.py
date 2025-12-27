from __future__ import annotations

import numpy as np
import cupy as cp

from dataclasses import dataclass

# Import as TYPE_CHECKING to avoid circular imports at runtime if needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from planetary_sandbox.numerics.spherical_harmonics import LatLonSphericalHarmonics
    from planetary_sandbox.numerics.optimized_geodesic_sh import GeodesicSphericalHarmonics


@dataclass
class SpectralTerrainParams:
    """
    Parameters for spherical-harmonic (spectral) terrain generation.
    """
    rms_elevation: float = 2000.0   # target RMS height (e.g. meters)
    spectral_exponent: float = 3.5  # alpha in P_l ~ (l+1)^(-alpha)
    seed: int = 1                   # RNG seed for reproducibility
    l_min: int = 1                  # ignore l < l_min (no huge DC/tilt modes)
    # TODO: Add more knobs later (continent masks, plate stuff, etc.)


def generate_spectral_terrain_gpu(
    sph: GeodesicSphericalHarmonics,
    params: SpectralTerrainParams,
) -> cp.ndarray:
    """
    Generate a random terrain field on the sphere using spherical harmonics,
    entirely on the GPU, then return it as a numpy array.

    Assumes a *real-valued* SH basis with coefficients stored as a real
    (l_max+1, l_max+1) array where for each degree l, orders m=0..l are valid
    and m>l are ignored/zero.

    Parameters
    ----------
    sph : GeodesicSphericalHarmonics
        Spherical harmonics engine instance (already initialized with l_max).
    params : SpectralTerrainParams
        Controls spectral slope, amplitude, RNG seed, etc.

    Returns
    -------
    coeffs : cp.ndarray, shape (l_max+1, l_max+1)
        The generated spectral coefficients on GPU.
    """
    l_max = sph.l_max
    alpha = params.spectral_exponent

    # Create random coefficients on GPU
    # Shape (l_max+1, l_max+1) with real coefficients; we zero out m>l.
    if params.seed is not None:
        # CuPy's global RNG doesn't support seeding *per-call* nicely,
        # so we use RandomState to keep it deterministic when requested.
        rs = cp.random.RandomState(params.seed)
        coeff_init_mags = rs.standard_normal((l_max + 1, l_max + 1), dtype=cp.float64)
    else: coeff_init_mags = cp.random.standard_normal((l_max + 1, l_max + 1), dtype=cp.float64)
    coeff_phases = cp.random.uniform(0.0, 2.0 * cp.pi, (l_max + 1, l_max + 1), dtype=cp.float64)
    coeffs = coeff_init_mags * cp.exp(1j * coeff_phases)

    # Power spectrum: P_l ~ (l+1)^(-alpha)
    l_vals = cp.arange(l_max + 1, dtype=cp.float64)
    # Avoid zero at l=0; we can damp that separately
    amp_l = (l_vals + 1.0) ** (-alpha / 2.0)  # amplitude so that power ~ amp^2

    # Apply radial dependence
    coeffs *= amp_l[:, None]

    # Optionally suppress very low degrees (DC / global tilt) by hand
    if params.l_min > 0:
        coeffs[:params.l_min, :] = 0.0

    # Enforce "only m <= l is valid"
    l_idx, m_idx = cp.indices((l_max + 1, l_max + 1))
    coeffs = cp.where(m_idx <= l_idx, coeffs, 0.0)

    # --- Reconstruct field on GPU using the existing method ---
    # Convert grids to GPU arrays
    # lambda_gpu = cp.asarray(lambda_grid, dtype=cp.float64)
    # phi_gpu = cp.asarray(phi_grid, dtype=cp.float64)

    height_gpu = sph.inv_transform(coeffs)

    # --- Normalize and scale on GPU ---
    mean = cp.mean(height_gpu)
    height_gpu = height_gpu - mean
    std = cp.std(height_gpu)
    if std > 0 and params.rms_elevation is not None:
        coeffs = coeffs * (params.rms_elevation / std)

    # Ensure zero-mean if low degrees are allowed.
    if params.l_min <= 0:
        coeffs[0, 0] = 0.0

    return coeffs


def diffuse_spectral_coeffs(
    coeffs: cp.ndarray,
    kappa: float,
    dt: float,
) -> cp.ndarray:
    """
    Apply one step of spectral diffusion (∂h/∂t = κ ∇² h) to SH coefficients.

    In spherical harmonics, ∇²Y_lm = -l(l+1) Y_lm, so each mode decays as:
        a_lm(t + dt) = a_lm(t) * exp(-κ l(l+1) dt)

    Parameters
    ----------
    coeffs : cp.ndarray, shape (l_max+1, l_max+1)
        Spectral coefficients (real basis, m>=0 layout).
    kappa : float
        Diffusivity coefficient (controls the rate of smoothing).
    dt : float
        Time step.

    Returns
    -------
    cp.ndarray
        New coefficients after diffusion (same shape).
    """
    l_max = coeffs.shape[0] - 1
    l_vals = cp.arange(l_max + 1, dtype=cp.float64)
    decay = cp.exp(-kappa * l_vals * (l_vals + 1.0) * dt)
    decay_mat = decay[:, None]  # broadcast over m

    return coeffs * decay_mat
