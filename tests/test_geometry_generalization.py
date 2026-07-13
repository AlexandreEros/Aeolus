"""Geometry-generalization tests (pre-lat-lon groundwork).

Velocity reconstruction and CFL length-scale handling must not assume the
geodesic grid family: `velocity_from_streamfunction` may use only the
GridGeometry interface (per-point latitudes), and the CFL length scale is a
geometry-owned property (`cfl_length_scale`) instead of a hard-coded
`min_edge_length` attribute lookup.
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
        GeodesicGridGeometry,
        GeodesicSphericalHarmonics,
        PointSetSphericalHarmonics,
        SpectralOperators,
    )
    from planetary_sandbox.numerics.grid_base import GridGeometry

R = 6.371e6
OMEGA = 1e-5
L_MAX = 10


class _PointSetOnlyGeometry(GridGeometry if _HAS_CUDA else object):
    """Minimal geometry: per-point angles only — no cartesian `.points`,
    no adjacency, no `min_edge_length`. Anything reachable from operator
    code beyond the GridGeometry interface breaks this on purpose."""

    def __init__(self, lat, lon):
        self._lat = cp.asarray(lat, dtype=cp.float64)
        self._lon = cp.asarray(lon, dtype=cp.float64)
        self.coslat = cp.cos(self._lat)

    @property
    def latitudes(self):
        return self._lat

    @property
    def longitudes(self):
        return self._lon

    @property
    def point_latitudes(self):
        return self._lat

    @property
    def point_longitudes(self):
        return self._lon

    @property
    def n_points(self):
        return int(self._lat.size)

    def points_latlon(self):
        return np.column_stack(
            [cp.asnumpy(self._lon), cp.asnumpy(self._lat)])


@pytest.fixture(scope="module")
def geodesic():
    grid = GeodesicGridGeometry(resolution=4, radius=R)
    sh = GeodesicSphericalHarmonics(grid, L_MAX, weights="voronoi")
    so = SpectralOperators(sh, R, grid)
    return grid, sh, so


def _solid_body_psi():
    psi_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    psi_lm[1, 0] = -OMEGA * R**2 * np.sqrt(4.0 * np.pi / 3.0)
    return psi_lm


def test_velocity_solid_body_analytic_geodesic(geodesic):
    """Regression pin: u = w R cos(lat), v = 0 for solid-body psi."""
    grid, _, so = geodesic
    u, v = so.velocity_from_streamfunction(_solid_body_psi())
    u_expected = OMEGA * R * cp.cos(cp.asarray(grid.point_latitudes))
    # away from the pole clamp region
    mask = cp.abs(cp.asarray(grid.point_latitudes)) < np.deg2rad(80.0)
    rel = float(cp.max(cp.abs(u - u_expected)[mask])) / (OMEGA * R)
    assert rel < 5e-3, f"solid-body u mismatch: {rel:.2e}"
    assert float(cp.max(cp.abs(v[mask]))) / (OMEGA * R) < 5e-3


def test_velocity_works_without_cartesian_points(geodesic):
    """velocity_from_streamfunction must rely only on the GridGeometry
    interface: same points, stub geometry without `.points` -> same (u, v)."""
    grid, sh, so = geodesic
    stub = _PointSetOnlyGeometry(grid.point_latitudes, grid.point_longitudes)
    sh_stub = PointSetSphericalHarmonics(
        stub.point_latitudes, stub.point_longitudes, L_MAX,
        weights=sh.weights)
    so_stub = SpectralOperators(sh_stub, R, stub)  # PointSetBackend, coarse

    psi_lm = _solid_body_psi()
    u_ref, v_ref = so.velocity_from_streamfunction(psi_lm)
    u_stub, v_stub = so_stub.velocity_from_streamfunction(psi_lm)

    scale = float(cp.max(cp.abs(u_ref))) + 1e-30
    assert float(cp.max(cp.abs(u_stub - u_ref))) / scale < 1e-10
    assert float(cp.max(cp.abs(v_stub - v_ref))) / scale < 1e-10


def test_cfl_length_scale_geodesic_equals_min_edge(geodesic):
    """Geodesic geometry: cfl_length_scale is exactly the historical
    min_edge_length (behavior-preserving)."""
    grid, _, _ = geodesic
    assert grid.cfl_length_scale == pytest.approx(grid.min_edge_length)


def test_cfl_length_scale_default_none_for_bare_geometry(geodesic):
    """Geometries that define no length scale report None (callers then use
    their explicit fallback, never a silent wrong number)."""
    grid, _, _ = geodesic
    stub = _PointSetOnlyGeometry(grid.point_latitudes, grid.point_longitudes)
    assert stub.cfl_length_scale is None
