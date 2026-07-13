"""Contract tests for the SphericalGridBackend seam (feat/grid-abstraction).

The backend pairs a grid geometry with an SH transform and owns product-
quadrature policy. These tests pin the interface contract so future backends
(e.g. lat-lon) conform without touching operator code:
  - 'coarse' is mandatory; other modes are backend-defined;
  - unsupported modes raise ValueError (no silent fallback);
  - product spaces are built once and cached (never per tendency call);
  - nothing in the interface assumes a geodesic co-grid.
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
        GeodesicBackend,
        GeodesicGridGeometry,
        GeodesicSphericalHarmonics,
        LatLonGridGeometry,
        PointSetBackend,
        SpectralOperators,
        make_backend,
    )

L_MAX = 5  # tiny: contract tests, not accuracy tests


@pytest.fixture(scope="module")
def geo():
    grid = GeodesicGridGeometry(resolution=3, radius=1.0)
    sh = GeodesicSphericalHarmonics(grid, L_MAX, weights="voronoi")
    return grid, sh


def test_make_backend_infers_family(geo):
    grid, sh = geo
    assert isinstance(make_backend(grid, sh), GeodesicBackend)
    latlon = LatLonGridGeometry.create((9, 17))
    assert isinstance(make_backend(latlon, sh), PointSetBackend)


def test_supported_modes(geo):
    grid, sh = geo
    assert make_backend(grid, sh).supported_product_quadratures() == ("coarse", "fine")
    latlon = LatLonGridGeometry.create((9, 17))
    assert make_backend(latlon, sh).supported_product_quadratures() == ("coarse",)


def test_unsupported_mode_raises_no_silent_fallback(geo):
    grid, sh = geo
    backend = make_backend(grid, sh)
    with pytest.raises(ValueError, match="no silent fallback"):
        backend.product_space("exact")
    latlon_backend = make_backend(LatLonGridGeometry.create((9, 17)), sh)
    with pytest.raises(ValueError, match="no silent fallback"):
        latlon_backend.product_space("fine")


def test_product_space_cached(geo):
    grid, sh = geo
    backend = GeodesicBackend(grid, sh)
    assert backend.product_space("fine") is backend.product_space("fine")
    assert backend.product_space("coarse") is backend.product_space("coarse")


def test_coarse_product_space_is_state_sampling(geo):
    grid, sh = geo
    ps = GeodesicBackend(grid, sh).product_space("coarse")
    assert ps.sh is sh                      # the state transform itself
    assert ps.geometry is None              # no distinct product geometry
    assert ps.coslat.shape == (grid.n_points,)
    assert float(cp.min(ps.coslat)) >= 1e-8  # clamped
    assert "state" in ps.label


def test_fine_product_space_geodesic_choice(geo):
    """The res-(r+1) co-grid is the GEODESIC backend's choice, exposed only
    through ProductSpace fields — operator code never assumes it."""
    grid, sh = geo
    ps = GeodesicBackend(grid, sh).product_space("fine")
    assert ps.geometry is not None
    assert ps.geometry.resolution == grid.resolution + 1
    assert ps.sh.l_max == L_MAX             # same truncation, finer sampling
    assert ps.coslat.shape == (ps.geometry.n_points,)
    # transform pair usable in both directions on the product sampling
    coeffs = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    coeffs[2, 1] = 1.0
    values = ps.sh.inv_transform(coeffs)
    assert values.shape == (ps.geometry.n_points,)
    back = ps.sh.transform(values)
    assert abs(complex(back[2, 1]) - 1.0) < 1e-2


def test_geodesic_backend_rejects_foreign_geometry(geo):
    _, sh = geo
    with pytest.raises(ValueError):
        GeodesicBackend(LatLonGridGeometry.create((9, 17)), sh)


def test_spectral_operators_accepts_explicit_backend(geo):
    grid, sh = geo
    backend = GeodesicBackend(grid, sh)
    so = SpectralOperators(sh, 1.0, grid, product_quadrature="fine",
                           backend=backend)
    assert so.backend is backend
    # back-compat attributes preserved for existing tests/diagnostics
    assert so.product_grid is backend.product_space("fine").geometry
    assert so.product_sh is backend.product_space("fine").sh
    so_c = SpectralOperators(sh, 1.0, grid, product_quadrature="coarse",
                             backend=backend)
    assert so_c.product_grid is None and so_c.product_sh is None
