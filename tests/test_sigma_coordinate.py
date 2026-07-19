"""Sigma-coordinate vertical grid and column operators (CPU, NumPy).

Covers the vertical half of the primitive-equation foundation
(docs/PRIMITIVE_EQUATIONS_DESIGN.md Sections 3–6, 9) without any GPU:
grid-metadata validation, Simmons–Burridge hydrostatic exactness for
isothermal columns, structural top/bottom impermeability, discrete column
mass closure, and the NumPy/CuPy backend-independence contract (the CuPy
parity test alone is CUDA-gated).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from planetary_sandbox.physics.sigma_coordinate import (
    SigmaGrid, SigmaGridError, column_energy_conversion,
    column_mass_tendency, column_pressure_work, energy_exchange,
    hydrostatic_geopotential, interface_sigma_dot, layer_mass_residual,
    omega_over_p)

R_DRY = 287.04

#: A deliberately nonuniform 6-layer grid (top-heavy stretching).
NONUNIFORM = SigmaGrid((0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.0))


# ---------------------------------------------------------------------------
# Grid metadata validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interfaces", [
    (0.0,),                          # fewer than 2 interfaces
    (0.1, 0.5, 1.0),                 # top not exactly 0
    (0.0, 0.5, 0.999),               # bottom not exactly 1
    (0.0, 0.6, 0.4, 1.0),            # not increasing
    (0.0, 0.5, 0.5, 1.0),            # duplicate (not strictly increasing)
    (0.0, math.nan, 1.0),            # non-finite
    (0.0, math.inf, 1.0),            # non-finite
    (0.0, "mid", 1.0),               # non-numeric
])
def test_grid_rejects_invalid_interfaces(interfaces):
    with pytest.raises(SigmaGridError):
        SigmaGrid(interfaces)


@pytest.mark.parametrize("nlev", [0, -3, 2.5, True])
def test_uniform_rejects_bad_level_count(nlev):
    with pytest.raises(SigmaGridError):
        SigmaGrid.uniform(nlev)


def test_uniform_grid_metadata():
    grid = SigmaGrid.uniform(5)
    assert grid.nlev == 5
    assert grid.interfaces[0] == 0.0 and grid.interfaces[-1] == 1.0
    assert math.isclose(sum(grid.thickness), 1.0, rel_tol=1e-15)
    for k in range(5):
        assert math.isclose(
            grid.full_levels[k],
            0.5 * (grid.interfaces[k] + grid.interfaces[k + 1]),
            rel_tol=1e-15)
    assert grid.interfaces_array().dtype == np.float64
    assert grid.thickness_array().shape == (5,)
    assert grid.full_levels_array().shape == (5,)


def test_grid_is_immutable():
    grid = SigmaGrid.uniform(3)
    with pytest.raises(Exception):
        grid.interfaces = (0.0, 1.0)


def test_simmons_burridge_coefficients():
    grid = NONUNIFORM
    # Top layer: alpha_1 = ln 2, and the log ratio is +inf but never used.
    assert grid.alpha[0] == math.log(2.0)
    assert grid.interface_log_ratios[0] == math.inf
    s = grid.interfaces
    for k in range(1, grid.nlev):
        ratio = math.log(s[k + 1] / s[k])
        assert math.isclose(grid.interface_log_ratios[k], ratio,
                            rel_tol=1e-15)
        expected = 1.0 - (s[k] / (s[k + 1] - s[k])) * ratio
        assert math.isclose(grid.alpha[k], expected, rel_tol=1e-14)
        # alpha in (0, 1) for interior layers of any valid grid.
        assert 0.0 < grid.alpha[k] < 1.0


# ---------------------------------------------------------------------------
# Hydrostatic geopotential (analytic isothermal column)
# ---------------------------------------------------------------------------

def test_isothermal_interface_geopotential_is_analytic():
    """SB interface recursion telescopes to Phi_s - R T0 ln(sigma) exactly."""
    grid = NONUNIFORM
    T0, phi_s = 250.0, 1234.5
    T = np.full((grid.nlev,), T0)
    phi_full, phi_below = hydrostatic_geopotential(grid, T, phi_s, R_DRY)

    assert phi_below[-1] == phi_s  # surface boundary condition, exact
    for k in range(grid.nlev):
        analytic = phi_s - R_DRY * T0 * math.log(grid.interfaces[k + 1])
        assert phi_below[k] == pytest.approx(analytic, rel=1e-13)
        # Full level = analytic profile at ln(sigma_eff) = ln(s_{k+1/2}) - alpha_k.
        analytic_full = phi_s - R_DRY * T0 * (
            math.log(grid.interfaces[k + 1]) - grid.alpha[k])
        assert phi_full[k] == pytest.approx(analytic_full, rel=1e-13)

    # Top layer effective level is the arithmetic-mean full level exactly:
    # sigma_eff = sigma_{3/2} * exp(-ln 2) = sigma_{3/2}/2 = sigma_1.
    top_analytic = phi_s - R_DRY * T0 * math.log(grid.full_levels[0])
    assert phi_full[0] == pytest.approx(top_analytic, rel=1e-13)


def test_hydrostatic_broadcasts_over_columns():
    """Trailing dims are untouched columns; phi_surface broadcasts."""
    grid = SigmaGrid.uniform(4)
    rng = np.random.default_rng(7)
    T = 220.0 + 60.0 * rng.random((grid.nlev, 3, 5))
    phi_s = 100.0 * rng.random((3, 5))
    phi_full, phi_below = hydrostatic_geopotential(grid, T, phi_s, R_DRY)
    assert phi_full.shape == (grid.nlev, 3, 5)
    assert np.array_equal(phi_below[-1], phi_s)
    # Column independence: recomputing one column alone matches.
    pf1, pb1 = hydrostatic_geopotential(grid, T[:, 1, 2], phi_s[1, 2], R_DRY)
    np.testing.assert_allclose(phi_full[:, 1, 2], pf1, rtol=1e-15)
    np.testing.assert_allclose(phi_below[:, 1, 2], pb1, rtol=1e-15)
    # Warmer columns sit higher: monotonic in T at every level.
    hot, _ = hydrostatic_geopotential(grid, T + 50.0, phi_s, R_DRY)
    assert np.all(hot[:-1] > phi_full[:-1])


def test_hydrostatic_rejects_bad_inputs():
    grid = SigmaGrid.uniform(4)
    with pytest.raises(ValueError):
        hydrostatic_geopotential(grid, np.ones((3, 2)), 0.0, R_DRY)  # nlev
    with pytest.raises(ValueError):
        hydrostatic_geopotential(grid, np.ones((4,)), 0.0, -1.0)
    with pytest.raises(ValueError):
        hydrostatic_geopotential(grid, np.ones((4,)), 0.0, math.nan)


# ---------------------------------------------------------------------------
# Discrete column continuity
# ---------------------------------------------------------------------------

def test_sigma_dot_boundaries_are_exactly_zero():
    """Top structurally zero; bottom a bitwise cancellation — not approx."""
    grid = NONUNIFORM
    rng = np.random.default_rng(11)
    G = rng.standard_normal((grid.nlev, 40)) * 1e-5
    sdot = interface_sigma_dot(grid, G)
    assert sdot.shape == (grid.nlev + 1, 40)
    assert np.all(sdot[0] == 0.0)
    assert np.all(sdot[-1] == 0.0)


def test_uniform_g_gives_zero_sigma_dot():
    """G independent of level: partial sums telescope, sigma_dot ~ 0."""
    grid = NONUNIFORM
    c = 3.7e-6
    G = np.full((grid.nlev, 8), c)
    sdot = interface_sigma_dot(grid, G)
    assert np.abs(sdot).max() < 1e-18  # pure round-off of sums of ~1e-6
    dlnps = column_mass_tendency(grid, G)
    np.testing.assert_allclose(dlnps, -c, rtol=1e-13)


def test_single_layer_source_hand_formula():
    """G nonzero in one layer only: sigma_dot matches the closed form."""
    grid = SigmaGrid.uniform(5)
    j = 2  # 0-based forced layer
    G = np.zeros((grid.nlev,))
    G[j] = 1.0e-5
    w = G[j] * grid.thickness[j]
    sdot = interface_sigma_dot(grid, G)
    for k in range(grid.nlev + 1):
        below = w if k >= j + 1 else 0.0   # sum_{i<=k} over 1-based layers
        expected = grid.interfaces[k] * w - below
        assert sdot[k] == pytest.approx(expected, abs=1e-21)


def test_layer_mass_closure_is_round_off():
    grid = NONUNIFORM
    rng = np.random.default_rng(23)
    G = rng.standard_normal((grid.nlev, 100)) * 1e-5
    residual = layer_mass_residual(grid, G)
    scale = np.abs(G).max()
    assert residual.shape == G.shape
    assert np.abs(residual).max() < 1e-13 * scale


def test_column_mass_tendency_matches_dot_product():
    grid = NONUNIFORM
    rng = np.random.default_rng(31)
    G = rng.standard_normal((grid.nlev, 17))
    expected = -np.tensordot(grid.thickness_array(), G, axes=(0, 0))
    np.testing.assert_allclose(column_mass_tendency(grid, G), expected,
                               rtol=1e-14)


def test_continuity_rejects_wrong_level_axis():
    grid = SigmaGrid.uniform(4)
    with pytest.raises(ValueError):
        interface_sigma_dot(grid, np.ones((3, 2)))
    with pytest.raises(ValueError):
        column_mass_tendency(grid, np.ones((5,)))
    with pytest.raises(ValueError):
        layer_mass_residual(grid, np.ones((2, 2)))


# ---------------------------------------------------------------------------
# Simmons–Burridge omega/p and the discrete energy-exchange identity
# ---------------------------------------------------------------------------

def test_omega_over_p_is_finite_and_exact_for_uniform_g():
    """G independent of level: (omega/p)_k = A_k - c exactly below the top
    layer (the alpha/beta terms telescope to the continuous value); the top
    layer gives the known SB approximation A_1 - c*ln(2). No Inf/NaN from
    the formally infinite top-layer beta."""
    grid = NONUNIFORM
    c = 4.0e-6
    G = np.full((grid.nlev, 7), c)
    A = np.zeros((grid.nlev, 7))
    wp = omega_over_p(grid, G, A)
    assert np.all(np.isfinite(wp))
    assert wp[0] == pytest.approx(-c * math.log(2.0), rel=1e-14)
    for k in range(1, grid.nlev):
        np.testing.assert_allclose(wp[k], -c, rtol=1e-12)


def test_omega_over_p_reduces_to_advection_without_mass_flux():
    grid = SigmaGrid.uniform(5)
    rng = np.random.default_rng(41)
    A = rng.standard_normal((grid.nlev, 6)) * 1e-6
    wp = omega_over_p(grid, np.zeros_like(A), A)
    np.testing.assert_array_equal(wp, A)


def test_omega_over_p_rejects_wrong_shapes():
    grid = SigmaGrid.uniform(4)
    with pytest.raises(ValueError):
        omega_over_p(grid, np.ones((3, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        omega_over_p(grid, np.ones((4, 2)), np.ones((3, 2)))


def test_energy_exchange_identity_closes_to_round_off():
    """(E_d): conversion == column-local pressure work, per column."""
    grid = NONUNIFORM
    rng = np.random.default_rng(43)
    T = 210.0 + 80.0 * rng.random((grid.nlev, 50))
    G = rng.standard_normal((grid.nlev, 50)) * 1e-5
    A = rng.standard_normal((grid.nlev, 50)) * 1e-6
    phi_s = 500.0 * rng.random((50,))
    out = energy_exchange(grid, T, phi_s, G, A, R_DRY)
    scale = max(np.abs(out["conversion"]).max(), np.abs(out["work"]).max())
    assert scale > 0.0
    assert np.abs(out["residual"]).max() < 1e-12 * scale


def test_energy_exchange_identity_single_layer():
    """K = 1 degenerate column: both sides equal R T (A - ln2 G)."""
    grid = SigmaGrid.uniform(1)
    T = np.array([260.0])
    G = np.array([2.0e-6])
    A = np.array([5.0e-7])
    out = energy_exchange(grid, T, 0.0, G, A, R_DRY)
    expected = R_DRY * T[0] * (A[0] - math.log(2.0) * G[0])
    assert out["conversion"] == pytest.approx(expected, rel=1e-13)
    assert out["work"] == pytest.approx(expected, rel=1e-13)
    assert abs(out["residual"]) < 1e-12 * abs(expected)


def test_energy_exchange_isothermal_uniform_hand_value():
    """Closed form: T = T0, G = c, A = a uniform =>
    conversion = R T0 [(a - c) + c * Dsigma_1 * (1 - ln 2)]."""
    grid = NONUNIFORM
    T0, c, a = 250.0, 3.0e-6, 1.0e-6
    ncol = 4
    T = np.full((grid.nlev, ncol), T0)
    G = np.full((grid.nlev, ncol), c)
    A = np.full((grid.nlev, ncol), a)
    out = energy_exchange(grid, T, 777.0, G, A, R_DRY)
    expected = R_DRY * T0 * ((a - c)
                             + c * grid.thickness[0] * (1.0 - math.log(2.0)))
    np.testing.assert_allclose(out["conversion"], expected, rtol=1e-12)
    np.testing.assert_allclose(out["work"], expected, rtol=1e-12)


def test_pressure_work_requires_consistent_geopotential():
    """Feeding a NON-Simmons-Burridge Phi must break the identity — the
    closure is a property of the consistent pair, not of any Phi."""
    grid = NONUNIFORM
    rng = np.random.default_rng(47)
    T = 210.0 + 80.0 * rng.random((grid.nlev, 10))
    G = rng.standard_normal((grid.nlev, 10)) * 1e-5
    A = np.zeros((grid.nlev, 10))
    phi_full, _ = hydrostatic_geopotential(grid, T, 0.0, R_DRY)
    conversion = column_energy_conversion(grid, T, G, A, R_DRY)
    good = column_pressure_work(grid, T, phi_full, 0.0, G, A, R_DRY)
    scale = np.abs(conversion).max()
    assert np.abs(conversion - good).max() < 1e-12 * scale
    # Perturb Phi at one level: the identity must visibly fail.
    bad_phi = phi_full.copy()
    bad_phi[2] *= 1.01
    bad = column_pressure_work(grid, T, bad_phi, 0.0, G, A, R_DRY)
    assert np.abs(conversion - bad).max() > 1e-6 * scale


def test_resting_column_exchanges_no_energy():
    grid = SigmaGrid.uniform(6)
    T = np.full((grid.nlev, 3), 260.0)
    zeros = np.zeros((grid.nlev, 3))
    out = energy_exchange(grid, T, 0.0, zeros, zeros, R_DRY)
    assert np.all(out["conversion"] == 0.0)
    assert np.all(out["work"] == 0.0)
    assert np.all(out["residual"] == 0.0)


# ---------------------------------------------------------------------------
# Backend-independent array semantics (CuPy parity; CUDA-gated)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_column_operators_are_backend_independent():
    """Identical results and preserved array family for CuPy inputs."""
    import cupy as cp

    grid = NONUNIFORM
    rng = np.random.default_rng(5)
    G_np = rng.standard_normal((grid.nlev, 30)) * 1e-5
    T_np = 220.0 + 60.0 * rng.random((grid.nlev, 30))
    phis_np = 100.0 * rng.random((30,))
    G_cp, T_cp, phis_cp = (cp.asarray(a) for a in (G_np, T_np, phis_np))

    for op, args_np, args_cp in [
        (column_mass_tendency, (G_np,), (G_cp,)),
        (interface_sigma_dot, (G_np,), (G_cp,)),
        (layer_mass_residual, (G_np,), (G_cp,)),
    ]:
        out_np = op(grid, *args_np)
        out_cp = op(grid, *args_cp)
        assert isinstance(out_cp, cp.ndarray), op.__name__
        np.testing.assert_allclose(cp.asnumpy(out_cp), out_np, rtol=0,
                                   atol=1e-20)

    pf_np, pb_np = hydrostatic_geopotential(grid, T_np, phis_np, R_DRY)
    pf_cp, pb_cp = hydrostatic_geopotential(grid, T_cp, phis_cp, R_DRY)
    assert isinstance(pf_cp, cp.ndarray) and isinstance(pb_cp, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(pf_cp), pf_np, rtol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(pb_cp), pb_np, rtol=1e-15)

    # Structural zeros must survive the backend change bitwise.
    sdot_cp = interface_sigma_dot(grid, G_cp)
    assert bool((sdot_cp[0] == 0.0).all())
    assert bool((sdot_cp[-1] == 0.0).all())

    # Energy operators: same parity contract.
    A_np = rng.standard_normal(G_np.shape) * 1e-6
    A_cp = cp.asarray(A_np)
    wp_np = omega_over_p(grid, G_np, A_np)
    wp_cp = omega_over_p(grid, G_cp, A_cp)
    assert isinstance(wp_cp, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(wp_cp), wp_np, rtol=1e-14)
    ex_np = energy_exchange(grid, T_np, phis_np, G_np, A_np, R_DRY)
    ex_cp = energy_exchange(grid, T_cp, phis_cp, G_cp, A_cp, R_DRY)
    scale = float(np.abs(ex_np["conversion"]).max())
    for key in ("conversion", "work"):
        assert isinstance(ex_cp[key], cp.ndarray), key
        np.testing.assert_allclose(cp.asnumpy(ex_cp[key]), ex_np[key],
                                   rtol=1e-12, atol=1e-15 * scale)
    # The identity must close on the GPU independently of the CPU result.
    assert float(cp.abs(ex_cp["residual"]).max()) < 1e-12 * scale
