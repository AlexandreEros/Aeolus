"""Spectral self-convergence comparison for shallow-water runs.

Pure numpy (no CuPy): these helpers operate on saved ``swe_coeffs.npy``
coefficient arrays after the fact, so they run on any host. They let a
coarser run (e.g. l_max = 42) be compared against a finer one (l_max = 63)
on the same physical grid, over the shared (l, m) triangle both resolve,
which is the meaningful self-convergence measure when the truncations differ.

The spherical-power convention matches the model's m>=0 real-transform
storage (see ``run/bve/diagnostics._mode_power``): the m = 0 column is real
and counts once; every m > 0 mode stands in for its +/-m conjugate pair and
counts double; entries above the diagonal (m > l) are not modes and are
ignored.
"""
from __future__ import annotations

import numpy as np


def mode_power(coeffs: np.ndarray) -> np.ndarray:
    """Per-(l, m) contribution to the spherical integral ``|f|^2 dOmega``.

    ``coeffs`` is an ``(n_l, n_m)`` complex array in the m>=0 layout. Returns
    a real array of the same shape: ``Re^2`` for m = 0, ``2|.|^2`` for m > 0,
    and zero wherever ``m > l``.
    """
    n_l, n_m = coeffs.shape[-2], coeffs.shape[-1]
    l_idx, m_idx = np.indices((n_l, n_m))
    valid = m_idx <= l_idx
    power = np.where(m_idx == 0, coeffs.real ** 2, 2.0 * np.abs(coeffs) ** 2)
    return np.where(valid, power, 0.0)


def _spherical_sq_norm(coeffs: np.ndarray) -> float:
    """Sum of :func:`mode_power` — the (radius-free) spherical L2 norm squared."""
    return float(mode_power(coeffs).sum())


def common_subset_relative_l2(field: np.ndarray,
                              reference: np.ndarray) -> float:
    """Relative spherical-L2 difference of two coeff fields on their shared corner.

    Both arguments are ``(n_l, n_m)`` complex coeff arrays, possibly at
    different truncations. The comparison is restricted to the shared
    ``(min l_max, min n_m)`` corner — the modes both fields resolve — so a
    finer field's extra high-degree content never enters. The result is
    ``||field - reference|| / ||reference||`` in the spherical L2 norm; the
    radius factor cancels.
    """
    n_l = min(field.shape[-2], reference.shape[-2])
    n_m = min(field.shape[-1], reference.shape[-1])
    f = field[..., :n_l, :n_m]
    r = reference[..., :n_l, :n_m]
    return float(np.sqrt(_spherical_sq_norm(f - r) / _spherical_sq_norm(r)))


def relative_difference(value: float, reference: float) -> float:
    """``|value - reference| / |reference|`` — scalar-invariant agreement."""
    return abs(value - reference) / abs(reference)
