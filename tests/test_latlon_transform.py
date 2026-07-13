"""Corrected Gauss-Legendre lat-lon geometry and SH transform.

The legacy `LatLonSphericalHarmonics` integrates with Simpson panels over a
longitude axis built with ``endpoint=False`` — the closing panel [2pi-dl, 2pi)
is simply missing, biasing every analysis low by ~1/nlon — and uses
equiangular-latitude Simpson quadrature including the poles. The corrected
transform uses Gauss-Legendre nodes/weights in latitude and uniform-weight
periodic longitudes (exact for trigonometric polynomials), making analysis of
band-limited fields exact to machine precision instead of merely approximate.
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
        GaussLatLonGridGeometry,
        GaussLatLonSphericalHarmonics,
        LatLonSphericalHarmonics,
    )

L_MAX = 10
NLAT, NLON = 24, 48  # >= (l_max+1, 2*l_max+1): analysis exact for l <= l_max
RADIUS = 6.371e6


@pytest.fixture(scope="module")
def geometry():
    return GaussLatLonGridGeometry(NLAT, NLON, radius=RADIUS)


@pytest.fixture(scope="module")
def transform(geometry):
    return GaussLatLonSphericalHarmonics(geometry, L_MAX)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_quadrature_weights_sum_to_4pi(geometry):
    total = float(cp.sum(cp.asarray(geometry.solid_angle_weights)))
    assert total == pytest.approx(4.0 * np.pi, rel=1e-13)


def test_geometry_shapes_and_ranges(geometry):
    assert geometry.grid_shape == (NLAT, NLON)
    assert geometry.n_points == NLAT * NLON
    assert geometry.point_latitudes.shape == (NLAT * NLON,)
    assert geometry.point_longitudes.shape == (NLAT * NLON,)
    lat = cp.asnumpy(cp.asarray(geometry.latitudes))
    assert lat.shape == (NLAT,)
    # GL nodes exclude the poles and are strictly monotonic
    assert np.all(np.abs(lat) < np.pi / 2)
    assert np.all(np.diff(lat) < 0) or np.all(np.diff(lat) > 0)
    lon = cp.asnumpy(cp.asarray(geometry.longitudes))
    assert lon.shape == (NLON,)
    assert lon[0] == 0.0 and lon[-1] < 2.0 * np.pi  # periodic, no duplicate seam


def test_cell_areas_cover_sphere(geometry):
    total = float(cp.sum(cp.asarray(geometry.cell_areas)))
    assert total == pytest.approx(4.0 * np.pi * RADIUS**2, rel=1e-12)


def test_cfl_length_scale_is_meridional_spacing(geometry):
    """Min meridional node spacing: the zonal direction imposes no tighter
    pseudospectral constraint (zonal wavenumber capped by l_max, so resolved
    zonal wavelength per point is latitude-independent)."""
    colat = np.pi / 2.0 - cp.asnumpy(cp.asarray(geometry.latitudes))
    expected = RADIUS * float(np.min(np.abs(np.diff(colat))))
    assert geometry.cfl_length_scale == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Transform correctness
# ---------------------------------------------------------------------------

def _random_bandlimited_coeffs(l_max, rng, amp=1e-5):
    """Coefficients of a real field: a_l0 real, m>0 complex, m>l zero."""
    a = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    for l in range(l_max + 1):
        a[l, 0] = amp * rng.standard_normal()
        for m in range(1, l + 1):
            a[l, m] = amp * (rng.standard_normal() + 1j * rng.standard_normal())
    return a


def test_roundtrip_machine_precision(transform):
    """analyze(synthesize(a)) == a to ~machine precision (exact quadrature)."""
    rng = np.random.default_rng(11)
    a = _random_bandlimited_coeffs(L_MAX, rng)
    values = transform.inv_transform(a)
    back = transform.transform(values)
    err = float(cp.max(cp.abs(back - a)))
    scale = float(cp.max(cp.abs(a)))
    assert err / scale < 1e-10, f"round-trip error {err/scale:.2e}"


def test_analysis_of_single_mode_is_delta(transform):
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    a[3, 2] = 1.0 - 0.5j
    values = transform.inv_transform(a)
    back = transform.transform(values)
    off = back.copy()
    off[3, 2] = 0.0
    assert abs(complex(back[3, 2]) - (1.0 - 0.5j)) < 1e-11
    assert float(cp.max(cp.abs(off))) < 1e-11


def test_constant_field_analysis_exact_vs_legacy_panel_bias(geometry, transform):
    """f == 1 must give a_00 = sqrt(4pi) exactly; the legacy Simpson transform
    on an endpoint=False longitude axis is biased low by ~1/nlon."""
    ones = cp.ones(geometry.n_points, dtype=cp.float64)
    a00 = complex(transform.transform(ones)[0, 0])
    exact = np.sqrt(4.0 * np.pi)
    assert abs(a00 - exact) / exact < 1e-13

    # Legacy transform on a comparable equiangular grid: the missing closing
    # longitude panel shows up as a relative bias of order 1/nlon.
    legacy = LatLonSphericalHarmonics(
        L_MAX,
        lon_grid=np.linspace(0.0, 2.0 * np.pi, NLON, endpoint=False),
        colat_grid=np.linspace(0.0, np.pi, NLAT + 1),
    )
    ones2d = cp.ones((NLAT + 1, NLON), dtype=cp.float64)
    a00_legacy = complex(legacy.transform(ones2d)[0, 0])
    legacy_rel_err = abs(a00_legacy - exact) / exact
    assert legacy_rel_err > 1e-3, (
        "expected the legacy longitude-panel bias to be measurable "
        f"(got {legacy_rel_err:.2e}) — has the legacy path been fixed?"
    )


def test_underresolved_grid_warns():
    geom = GaussLatLonGridGeometry(8, 16, radius=1.0)  # too small for l_max=10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GaussLatLonSphericalHarmonics(geom, 10)
    assert any("under-resolved" in str(w.message) for w in caught)


def test_adequate_grid_does_not_warn():
    geom = GaussLatLonGridGeometry(NLAT, NLON, radius=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GaussLatLonSphericalHarmonics(geom, L_MAX)
    assert not any("under-resolved" in str(w.message) for w in caught)
