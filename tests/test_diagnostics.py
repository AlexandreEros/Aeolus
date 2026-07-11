"""Tests for run diagnostics (run/bve/diagnostics.py).

Two independent code paths compute the same invariants:
  - `spectral_diagnostics`: pure degree sums on the coefficients;
  - `grid_diagnostics`:     synthesis + quadrature-weight integration.
Their agreement is bounded by the transform round-trip accuracy, so this
doubles as a transform-consistency regression test. Closed-form checks pin
the definitions themselves (a wrong formula in BOTH paths would still agree).
"""
import csv

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
    from planetary_sandbox.run.bve.diagnostics import (
        CSV_COLUMNS,
        DiagnosticsRecorder,
        grid_diagnostics,
        spectral_diagnostics,
    )

R = 6.371e6
OMEGA = 7.292e-5
L_MAX = 10  # keep products (degree 2*L_MAX) well inside the res-4 quadrature envelope


@pytest.fixture(scope="module")
def setup():
    grid = GeodesicGridGeometry(resolution=4, radius=R)
    sh = GeodesicSphericalHarmonics(grid, L_MAX, weights="voronoi")
    so = SpectralOperators(sh, R, grid)
    return grid, sh, so


def _multimode_state():
    rng = np.random.default_rng(7)
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    for l in range(1, L_MAX + 1):
        for m in range(0, l + 1):
            amp = 1e-5 * (l + 1.0) ** -1.5
            # m=0 coefficients of a real field are real; the synthesis discards
            # Im(a_l0), so generating one would just contaminate roundtrips.
            imag = rng.standard_normal() if m > 0 else 0.0
            zeta_lm[l, m] = amp * (rng.standard_normal() + 1j * imag)
    return zeta_lm


def test_single_mode_closed_forms():
    """Z, E, Gamma for a single (l, m) mode against pencil-and-paper values."""
    l, m, a = 5, 3, 2.5e-5
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    zeta_lm[l, m] = a

    d = spectral_diagnostics(zeta_lm, R, omega=0.0)
    # integral zeta^2 dOmega = 2|a|^2 for a single m>0 mode of a real field
    Z_exact = 0.5 * R**2 * 2.0 * a**2
    E_exact = 0.5 * R**4 * 2.0 * a**2 / (l * (l + 1.0))
    assert d["circulation"] == 0.0
    assert abs(d["enstrophy_rel"] / Z_exact - 1.0) < 1e-12
    assert abs(d["energy"] / E_exact - 1.0) < 1e-12
    # all enstrophy sits at degree l
    assert abs(d["enstrophy_l"][l] / Z_exact - 1.0) < 1e-12
    assert float(np.delete(d["enstrophy_l"], l).sum()) < 1e-30


def test_circulation_closed_form():
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    zeta_lm[0, 0] = 3e-6
    d = spectral_diagnostics(zeta_lm, R, omega=0.0)
    assert abs(d["circulation"] / (R**2 * np.sqrt(4 * np.pi) * 3e-6) - 1.0) < 1e-12


def test_absolute_enstrophy_includes_planetary_background():
    """With zeta = 0, Z_abs = 1/2 integral f^2 dA = (8*pi/3) R^2 Omega^2.

    (integral of sin^2(lat) over the unit sphere is 4*pi/3, and f = 2*Omega*sin(lat).)
    """
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    d = spectral_diagnostics(zeta_lm, R, omega=OMEGA)
    Z_f_exact = (8.0 * np.pi / 3.0) * R**2 * OMEGA**2
    assert abs(d["enstrophy_abs"] / Z_f_exact - 1.0) < 1e-12
    assert d["enstrophy_rel"] == 0.0


def test_padding_slots_are_ignored():
    """Garbage written into m > l storage slots must not leak into diagnostics."""
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    zeta_lm[5, 3] = 1e-5
    clean = spectral_diagnostics(zeta_lm, R, omega=OMEGA)
    zeta_lm[2, 7] = 123.0 + 456.0j  # invalid: m > l
    dirty = spectral_diagnostics(zeta_lm, R, omega=OMEGA)
    for key in ("circulation", "energy", "enstrophy_rel", "enstrophy_abs"):
        assert clean[key] == dirty[key]


def test_spectral_vs_grid_agreement(setup):
    """Independent spectral and quadrature paths agree within transform accuracy."""
    grid, sh, _ = setup
    zeta_lm = _multimode_state()
    spec = spectral_diagnostics(zeta_lm, R, omega=OMEGA)
    quad = grid_diagnostics(zeta_lm, sh, R, omega=OMEGA, latitudes=grid.latitudes)

    for key in ("energy", "enstrophy_rel", "enstrophy_abs"):
        rel = abs(quad[key] - spec[key]) / abs(spec[key])
        assert rel < 5e-3, f"{key}: spectral vs grid relative diff = {rel:.2e}"
    # circulation of this state is 0 spectrally; the grid value is pure
    # quadrature error. Normalize by R*sqrt(2 Z) so the ratio is dimensionless
    # (Gamma is m^2/s; sqrt(2Z) is m/s).
    scale = R * np.sqrt(2.0 * spec["enstrophy_rel"])
    assert abs(quad["circulation"]) / scale < 5e-3


def test_recorder_writes_csv_and_spectra(setup, tmp_path):
    grid, sh, so = setup
    rec = DiagnosticsRecorder(sh, so, grid, R, OMEGA, tmp_path,
                              spectra_every=1, roundtrip_every=1)
    zeta_lm = _multimode_state()
    for i in range(3):
        row = rec.record(t=i * 100.0, zeta_lm=zeta_lm, dt=100.0 if i else 0.0, step=i)
        assert np.isfinite(row["energy"]) and row["energy"] > 0
        assert np.isfinite(row["roundtrip_residual"])
    rec.close()

    csv_path = tmp_path / "diagnostics" / "timeseries.csv"
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert list(rows[0].keys()) == CSV_COLUMNS
    times = [float(r["time_s"]) for r in rows]
    assert times == sorted(times)
    # unchanged state -> identical invariants in every row
    assert len({r["enstrophy_abs"] for r in rows}) == 1

    with np.load(tmp_path / "diagnostics" / "spectra.npz") as z:
        assert z["enstrophy_l"].shape == (3, L_MAX + 1)

    # round-trip residual should sit at the transform envelope, not O(1)
    assert float(rows[0]["roundtrip_residual"]) < 1e-2
