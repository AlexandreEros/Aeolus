"""Correctness tests for the spectral differential operators.

These lock the behaviour that regressed as KNOWN_RISKS.md R-1: the
pseudospectral Jacobian used by the barotropic-vorticity tendency must equal the
true spherical Jacobian J(a, b) = u_a . grad b, not -cosφ * J.

The reference is analytic: for solid-body rotation psi = -w R^2 sin(lat)
(u = w R cos(lat), v = 0), advecting a single spherical-harmonic mode Y_l^m
gives  u . grad Y_l^m = w * dY/dlambda = i m w Y_l^m, so the (l, m) coefficient
of J(psi, Y_l^m) must equal i m w.
"""
import warnings

import numpy as np
import pytest

try:
    import cupy as cp

    _HAS_CUDA = cp.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/CuPy not available")

if _HAS_CUDA:
    from planetary_sandbox.numerics import (
        GeodesicGridGeometry,
        GeodesicSphericalHarmonics,
        SpectralOperators,
    )

R = 6.371e6
OMEGA = 1e-5  # solid-body angular rate [1/s]
L_MAX = 15


@pytest.fixture(scope="module")
def operators():
    grid = GeodesicGridGeometry(resolution=4, radius=R)
    sh = GeodesicSphericalHarmonics(grid, L_MAX, weights="voronoi")
    so = SpectralOperators(sh, R, grid)
    return grid, sh, so


def _solid_body_streamfunction():
    psi_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    # sin(lat) = sqrt(4pi/3) * Y_1^0  ->  psi = -w R^2 sin(lat)
    psi_lm[1, 0] = -OMEGA * R**2 * np.sqrt(4.0 * np.pi / 3.0)
    return psi_lm


@pytest.mark.parametrize("l, m", [(3, 2), (5, 3), (8, 4), (12, 6)])
def test_jacobian_matches_solid_body_advection(operators, l, m):
    """J(psi_sb, Y_l^m)_lm == i m w  (R-1 regression lock)."""
    _, sh, so = operators
    psi_lm = _solid_body_streamfunction()

    q_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    q_lm[l, m] = 1.0

    j_grid = so.jacobian_pseudospectral(psi_lm, q_lm, dealias=False)
    assert bool(cp.all(cp.isfinite(j_grid))), "Jacobian produced non-finite values"

    j_lm = sh.transform(j_grid)
    expected = 1j * m * OMEGA
    ratio = complex(j_lm[l, m] / expected)

    # Sign and magnitude must be correct (the R-1 bug gave ~ -0.8).
    assert abs(ratio - 1.0) < 5e-3, f"(l={l}, m={m}) ratio to analytic = {ratio}"


def test_jacobian_agrees_with_velocity_form(operators):
    """The two advection paths must now agree closely."""
    _, sh, so = operators
    psi_lm = _solid_body_streamfunction()
    q_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    q_lm[6, 3] = 1.0 - 0.4j

    j_lm = sh.transform(so.jacobian_pseudospectral(psi_lm, q_lm, dealias=False))
    adv_lm = sh.transform(
        so.advect_scalar_by_streamfunction(psi_lm, q_lm, dealias=False)
    )
    denom = float(cp.max(cp.abs(adv_lm))) + 1e-30
    rel = float(cp.max(cp.abs(j_lm - adv_lm))) / denom
    assert rel < 1e-3, f"jacobian vs velocity-form relative diff = {rel:.2e}"


def test_self_jacobian_vanishes(operators):
    """J(a, a) == 0 identically."""
    _, _, so = operators
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    a[4, 2] = 1.0 + 0.3j
    a[7, 5] = -0.6 + 0.2j
    j = so.jacobian_pseudospectral(a, a, dealias=False)
    assert float(cp.max(cp.abs(j))) < 1e-18


def test_jacobian_integral_is_zero(operators):
    """integral over the sphere of J(a, b) == 0  (so Y_0^0 coefficient ~ 0)."""
    _, sh, so = operators
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    b = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    a[4, 2] = 1.0 + 0.3j
    b[3, 1] = 0.7 - 0.2j
    j_lm = sh.transform(so.jacobian_pseudospectral(a, b, dealias=False))
    assert abs(complex(j_lm[0, 0])) < 1e-12


def test_underresolution_warns():
    """The transform must warn when the grid cannot support l_max (R-2)."""
    grid = GeodesicGridGeometry(resolution=4, radius=1.0)  # 2562 points
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GeodesicSphericalHarmonics(grid, l_max=45, weights="voronoi")
    assert any("under-resolved" in str(w.message) for w in caught), \
        "expected an under-resolution warning for l_max=45 at resolution 4"


def test_adequate_resolution_does_not_warn():
    grid = GeodesicGridGeometry(resolution=4, radius=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GeodesicSphericalHarmonics(grid, l_max=15, weights="voronoi")
    assert not any("under-resolved" in str(w.message) for w in caught)
