from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp

# for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..numerics import GeodesicSphericalHarmonics
    from ..numerics.spectral_operators import SpectralOperators


@dataclass
class TectonicParams:
    """
    Controls how strongly tectonics modify the terrain.
    """
    dt: float = 1.0                 # "tectonic time step"
    kappa_height: float = 1e-6      # spectral diffusion (erosion) for height
    kappa_strain: float = 5e-6      # diffusion/smoothing for strain field
    noise_strength: float = 0.3     # how much new mountain noise to inject
    l_cut_noise: int = 8            # only add noise for l >= l_cut_noise
    gamma_activity: float = 2.0     # nonlinearity on activity (S_norm ** gamma)
    renormalize_height: bool = True # optionally keep RMS of h roughly fixed


def tectonic_update_step(
    sph: GeodesicSphericalHarmonics,
    so: SpectralOperators,
    h_lm: cp.ndarray,
    S_lm: cp.ndarray,
    params: TectonicParams,
    rng: Optional[cp.random.RandomState] = None,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    One 'tectonic' update:
        * Diffuse height and strain spectrally (erosion + smoothing of belts)
        * Generate high-l noise
        * Go to grid, gate noise by strain field, update height
        * Return new spectral coeffs for height and strain

    Parameters
    ----------
    sph : GeodesicSphericalHarmonics
        The SH engine (must support inv_transform/transform using CuPy arrays).
    so : SpectralOperators
        Spectral operators on the sphere (for Laplacian, gradients, etc.).
    h_lm : cp.ndarray, shape (L+1, L+1)
        Spectral coefficients for elevation (real basis, m >= 0).
    S_lm : cp.ndarray, shape (L+1, L+1)
        Spectral coefficients for 'tectonic strain / activity'.
    params : TectonicParams
        Controls the relative strength of erosion and noise.
    rng : cp.random.RandomState, optional
        RNG for reproducible noise; if None, cp.random is used.

    Returns
    -------
    h_lm_new : cp.ndarray
        Updated elevation coefficients.
    S_lm_new : cp.ndarray
        Updated strain coefficients.
    """
    dt = params.dt
    assert S_lm.shape == h_lm.shape

    # Spectral diffusion (erosion + strain smoothing)
    L = so.lap_eigs  # spherical Laplacian eigenvalues
    # L is negative (-l(l+1)/R^2), so this decays higher l when kappa > 0.
    decay_h = cp.exp(params.kappa_height * L * so.R**2 * dt)
    decay_S = cp.exp(params.kappa_strain * L * so.R**2 * dt)

    decay_h_mat = decay_h[:, None]
    decay_S_mat = decay_S[:, None]

    h_lm = h_lm * decay_h_mat
    S_lm = S_lm * decay_S_mat

    # Generate high-l noise coeffs
    if rng is None:
        noise_lm = cp.random.standard_normal(h_lm.shape, dtype=cp.float64)
    else:
        noise_lm = rng.standard_normal(h_lm.shape, dtype=cp.float64)

    # Only keep m <= l (triangular part)
    l_idx, m_idx = cp.indices(h_lm.shape)
    noise_lm = cp.where(m_idx <= l_idx, noise_lm, 0.0)

    # Zero out low degrees so it's purely small-scale structure
    l_cut = params.l_cut_noise
    if l_cut > 0:
        noise_lm[:l_cut, :] = 0.0

    # Go to grid: h, strain S, and high-freq noise N ----
    h = sph.inv_transform(h_lm)
    S = sph.inv_transform(S_lm)
    N = sph.inv_transform(noise_lm)

    # Build an 'activity' field from strain
    # Here we use a simple normalized |S|; we could alternatively use ∇²h or whatever.
    S_abs = cp.abs(S)
    S_min = S_abs.min()
    S_max = S_abs.max()
    denom = S_max - S_min + 1e-12
    S_norm = (S_abs - S_min) / denom  # [0,1]

    # Sharpen belts: values near 1 become much stronger
    # A = S_norm ** params.gamma_activity  # still in [0,1]
    L_h = so.laplacian_field(h_lm)
    L_abs = cp.abs(L_h)
    L_norm = (L_abs - L_abs.min()) / (L_abs.max() - L_abs.min() + 1e-12)
    A = ((S_norm + L_norm) / 2) ** params.gamma_activity


    # Gate high-frequency noise by activity and update h
    # Normalize N to unit std so noise_strength is somewhat scale-free
    N = N - N.mean()
    std_N = N.std()
    if std_N > 0:
        N = N / std_N

    delta_h = A * N  # mountains only where activity is high

    h_new = h + params.noise_strength * delta_h

    if params.renormalize_height:
        # Optional: keep RMS of h roughly constant over time
        h_new = h_new - h_new.mean()
        std_h_new = h_new.std()
        std_h_old = h.std()
        if std_h_new > 0 and std_h_old > 0:
            h_new = h_new * (std_h_old / std_h_new)

    # Back to spectral
    h_lm_new = sph.transform(h_new)
    # For now S_lm only evolved via spectral diffusion; if S is ever modified
    # on the grid, also transform it back here.
    S_lm_new = S_lm

    return h_lm_new, S_lm_new
