"""Spectral self-convergence comparator for shallow-water runs (CPU-only).

These lock the numpy comparator used to check that two W5 runs at different
truncations (e.g. l_max 42 vs 63, same 96x192 grid) agree on their shared
(l, m) triangle within the measured self-convergence envelope. Pure numpy:
no CuPy, so they run without a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from planetary_sandbox.run.swe.convergence import (
    common_subset_relative_l2,
    mode_power,
    relative_difference,
)


# ---------------------------------------------------------------------------
# mode_power: the m>=0 real-transform spherical-power convention
# ---------------------------------------------------------------------------

def test_mode_power_m0_mode_weight_is_one():
    c = np.zeros((3, 3), dtype=complex)
    c[1, 0] = 5.0                      # a purely real (l=1, m=0) mode
    assert mode_power(c).sum() == pytest.approx(25.0)


def test_mode_power_mpositive_mode_weight_is_two():
    """m>0 modes represent the +/-m conjugate pair, so they count double."""
    c = np.zeros((3, 3), dtype=complex)
    c[1, 1] = 3.0 + 4.0j              # |.|^2 = 25
    assert mode_power(c).sum() == pytest.approx(50.0)


def test_mode_power_ignores_invalid_upper_triangle():
    """Storage above the diagonal (m > l) is not a real spherical mode and
    must never contribute, even if a caller leaves garbage there."""
    c = np.zeros((3, 3), dtype=complex)
    c[0, 1] = 999.0 + 1.0j            # m=1 > l=0: invalid
    c[1, 2] = -42.0                   # m=2 > l=1: invalid
    assert mode_power(c).sum() == 0.0


# ---------------------------------------------------------------------------
# common_subset_relative_l2: relative spherical-L2 on the shared (l,m) corner
# ---------------------------------------------------------------------------

def test_identical_fields_have_zero_difference():
    rng = np.random.default_rng(0)
    a = _random_triangular(rng, 6)
    assert common_subset_relative_l2(a, a) == pytest.approx(0.0)


def test_extra_high_degree_content_outside_common_corner_is_ignored():
    """A finer field that matches the coarse one on l <= L_coarse and only
    adds content at higher degrees must compare EQUAL: the comparison lives
    on the shared corner, not the finer field's extra modes."""
    rng = np.random.default_rng(1)
    coarse = _random_triangular(rng, 4)          # l_max = 3  -> shape (4,4)
    fine = np.zeros((8, 8), dtype=complex)        # l_max = 7
    fine[:4, :4] = coarse
    fine[6, 5] = 100.0 + 7.0j                     # extra high-l content only
    assert common_subset_relative_l2(coarse, fine) == pytest.approx(0.0)
    assert common_subset_relative_l2(fine, coarse) == pytest.approx(0.0)


def test_relative_difference_scales_correctly():
    rng = np.random.default_rng(2)
    a = _random_triangular(rng, 5)
    # ||a - 2a|| / ||a|| = 1 exactly.
    assert common_subset_relative_l2(2.0 * a, a) == pytest.approx(1.0)


def test_difference_only_in_shared_corner_is_measured():
    """A perturbation inside the shared corner is measured relative to the
    reference (second argument) norm, with the m>0 double weighting."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 10.0                    # reference power 100
    b = a.copy()
    b[1, 1] = 1.0                     # add an m=1 mode: extra power 2*1 = 2
    # ||b - a||^2 = 2 (m=1 weight), ||a||^2 = 100 -> sqrt(2)/10.
    assert common_subset_relative_l2(b, a) == pytest.approx(np.sqrt(2.0) / 10.0)


# ---------------------------------------------------------------------------
# relative_difference: scalar invariant agreement (e.g. potential enstrophy)
# ---------------------------------------------------------------------------

def test_relative_difference_of_scalars():
    assert relative_difference(1.02, 1.0) == pytest.approx(0.02)
    assert relative_difference(1.0, 1.0) == 0.0
    # Sign-independent magnitude, relative to the reference (second arg).
    assert relative_difference(-3.0, -2.0) == pytest.approx(0.5)


def _random_triangular(rng, n_l):
    """A random (n_l, n_l) complex coeff array with a valid m<=l triangle:
    m=0 real, m>0 complex, upper triangle exactly zero."""
    c = (rng.standard_normal((n_l, n_l))
         + 1j * rng.standard_normal((n_l, n_l)))
    l_idx, m_idx = np.indices((n_l, n_l))
    c[m_idx > l_idx] = 0.0
    c[m_idx == 0] = c[m_idx == 0].real   # m=0 modes are real
    return c
