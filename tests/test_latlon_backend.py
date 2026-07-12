"""Contract tests for LatLonBackend through the SphericalGridBackend seam.

Mirrors tests/test_spherical_backend.py: modes are explicit ('coarse',
'fine'), unsupported modes raise (no silent fallback), product spaces are
cached, and 'fine' is the lat-lon backend's own choice — a denser
Gauss-Legendre grid sized by the 3/2 rule so quadratic products are
integrated EXACTLY (unlike the geodesic co-grid, which is only an
overresolved approximation).
"""
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
        GeodesicGridGeometry,
        GeodesicSphericalHarmonics,
        LatLonBackend,
        LatLonGridGeometry,
        PointSetBackend,
        SpectralOperators,
        make_backend,
    )

L_MAX = 5
NLAT, NLON = 12, 24  # comfortably >= (l_max+1, 2*l_max+1)


@pytest.fixture(scope="module")
def latlon():
    grid = GaussLatLonGridGeometry(NLAT, NLON, radius=1.0)
    sh = GaussLatLonSphericalHarmonics(grid, L_MAX)
    return grid, sh


def test_make_backend_infers_gauss_latlon(latlon):
    grid, sh = latlon
    assert isinstance(make_backend(grid, sh), LatLonBackend)
    # the legacy equiangular geometry stays a coarse-only point set
    legacy = LatLonGridGeometry.create((9, 17))
    assert isinstance(make_backend(legacy, sh), PointSetBackend)


def test_supported_modes_explicit(latlon):
    grid, sh = latlon
    assert LatLonBackend(grid, sh).supported_product_quadratures() == (
        "coarse", "fine")


def test_unsupported_mode_raises_no_silent_fallback(latlon):
    grid, sh = latlon
    with pytest.raises(ValueError, match="no silent fallback"):
        LatLonBackend(grid, sh).product_space("exact")


def test_rejects_foreign_geometry(latlon):
    _, sh = latlon
    geo = GeodesicGridGeometry(resolution=2, radius=1.0)
    with pytest.raises(ValueError):
        LatLonBackend(geo, sh)


def test_product_spaces_cached(latlon):
    grid, sh = latlon
    backend = LatLonBackend(grid, sh)
    assert backend.product_space("fine") is backend.product_space("fine")
    assert backend.product_space("coarse") is backend.product_space("coarse")


def test_coarse_product_space_is_state_sampling(latlon):
    grid, sh = latlon
    ps = LatLonBackend(grid, sh).product_space("coarse")
    assert ps.sh is sh
    assert ps.geometry is None
    assert ps.coslat.shape == (grid.n_points,)
    assert float(cp.min(ps.coslat)) >= 1e-8
    assert "state" in ps.label


def test_fine_product_space_is_three_halves_rule(latlon):
    """Fine mode: GL product grid sized so degree-2*l_max products are
    analyzed exactly against degree-l_max harmonics."""
    grid, sh = latlon
    ps = LatLonBackend(grid, sh).product_space("fine")
    assert ps.geometry is not None
    # exactness: 2*nlat - 1 >= 3*l_max  and  nlon >= 3*l_max + 1
    assert 2 * ps.geometry.nlat - 1 >= 3 * L_MAX
    assert ps.geometry.nlon >= 3 * L_MAX + 1
    assert ps.sh.l_max == L_MAX
    assert ps.coslat.shape == (ps.geometry.n_points,)
    assert "latlon" in ps.label and "3/2" in ps.label


def test_fine_product_analysis_is_exact(latlon):
    """Pointwise product of two band-limited fields, analyzed on the fine
    grid, must equal the analysis on a much denser reference grid."""
    grid, sh = latlon
    ps = LatLonBackend(grid, sh).product_space("fine")

    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    b = cp.zeros_like(a)
    a[2, 1] = 1.0 - 0.3j
    a[4, 0] = 0.7
    b[3, 2] = 0.5 + 0.2j
    b[5, 4] = -0.4 + 0.1j

    prod_fine = ps.sh.inv_transform(a) * ps.sh.inv_transform(b)
    coeffs_fine = ps.sh.transform(prod_fine)

    ref_grid = GaussLatLonGridGeometry(4 * L_MAX, 8 * L_MAX, radius=1.0)
    ref_sh = GaussLatLonSphericalHarmonics(ref_grid, L_MAX)
    prod_ref = ref_sh.inv_transform(a) * ref_sh.inv_transform(b)
    coeffs_ref = ref_sh.transform(prod_ref)

    err = float(cp.max(cp.abs(coeffs_fine - coeffs_ref)))
    assert err < 1e-11, f"fine product analysis not exact: {err:.2e}"


def test_spectral_operators_fine_mode_on_latlon(latlon):
    """The seam works end-to-end: SpectralOperators accepts the lat-lon
    backend in 'fine' mode and the Jacobian contract holds (J(a,a)=0)."""
    grid, sh = latlon
    backend = LatLonBackend(grid, sh)
    so = SpectralOperators(sh, 1.0, grid, product_quadrature="fine",
                           backend=backend)
    assert so.backend is backend
    a = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    a[3, 2] = 1.0 + 0.3j
    j = so.jacobian_pseudospectral(a, a, dealias=False)
    assert float(cp.max(cp.abs(j))) < 1e-15
