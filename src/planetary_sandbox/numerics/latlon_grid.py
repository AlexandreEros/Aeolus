"""Corrected structured lat-lon grid: Gauss-Legendre latitudes, uniform
periodic longitudes, and an exact-quadrature SH transform.

Why "corrected": the legacy pairing (`grid.LatLonGridGeometry` +
`spherical_harmonics.LatLonSphericalHarmonics`) integrates with Simpson
panels over a longitude axis built with ``endpoint=False`` — the closing
panel [2pi-dl, 2pi) is missing entirely, biasing every analysis low by
~1/nlon — and uses equiangular-latitude Simpson quadrature with explicit
pole rows (where sin(colat)=0 wastes samples and the panel rule is only
O(h^4) accurate). This module replaces (not wraps) that scheme:

* latitude nodes/weights are Gauss-Legendre in sin(lat): exact for
  polynomial integrands up to degree 2*nlat-1,
* longitudes are uniform with equal weights 2*pi/nlon: exact for
  trigonometric polynomials of zonal degree < nlon (the correct quadrature
  for a periodic axis — no panels, no seam),

so analysis of a band-limited field of degree L against Y_lm (degree
l <= l_max) is EXACT whenever  L + l_max <= 2*nlat - 1  and
L + l_max < nlon. For the state grid (L = l_max) that is
nlat >= l_max + 1, nlon >= 2*l_max + 1; a product grid sized by the
3/2 rule (nlat >= ceil((3*l_max+1)/2), nlon >= 3*l_max + 1) integrates
quadratic products exactly.

The transform itself is `PointSetSphericalHarmonics` (the same GPU matrix
path the geodesic backend uses) fed with the tensor-product quadrature
weights; fields are flat (n_points,) arrays in lat-major order, matching the
unstructured convention used throughout the operators.
"""
from __future__ import annotations

import warnings

import numpy as np
import cupy as cp

from .grid_base import GridGeometry
from .fast_geodesic_sh import PointSetSphericalHarmonics

__all__ = ["GaussLatLonGridGeometry", "GaussLatLonSphericalHarmonics"]


class GaussLatLonGridGeometry(GridGeometry):
    """Structured Gauss-Legendre x uniform-longitude grid on the sphere.

    Latitude rows run north -> south (colatitude ascending); per-point
    arrays are the C-order flattening of the (nlat, nlon) grid.
    """

    def __init__(self, nlat: int, nlon: int, radius: float = 1.0):
        if nlat < 2 or nlon < 4:
            raise ValueError(
                f"GaussLatLonGridGeometry needs nlat >= 2 and nlon >= 4, "
                f"got ({nlat}, {nlon})")
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.radius = float(radius)

        # Gauss-Legendre nodes x_j = cos(colat_j) and weights on [-1, 1].
        x, w_gl = np.polynomial.legendre.leggauss(self.nlat)
        order = np.argsort(-x)              # x descending == colat ascending
        x, w_gl = x[order], w_gl[order]
        colat = np.arccos(x)                # (nlat,), in (0, pi): no poles
        self._latitudes_np = np.pi / 2.0 - colat
        self._longitudes_np = np.linspace(0.0, 2.0 * np.pi, self.nlon,
                                          endpoint=False)
        self._gl_weights = w_gl             # sum == 2 exactly

        self._latitudes = cp.asarray(self._latitudes_np)
        self._longitudes = cp.asarray(self._longitudes_np)

        # Per-point arrays: lat-major C-order flattening.
        self._point_latitudes = cp.asarray(
            np.repeat(self._latitudes_np, self.nlon))
        self._point_longitudes = cp.asarray(
            np.tile(self._longitudes_np, self.nlat))

        self.coslat = cp.cos(self._point_latitudes)
        self.sinlat = cp.sin(self._point_latitudes)

        # Tensor-product solid-angle weights: w_j * (2*pi/nlon), sum == 4*pi.
        w_point = np.repeat(w_gl, self.nlon) * (2.0 * np.pi / self.nlon)
        self.solid_angle_weights = cp.asarray(w_point)

    # -- GridGeometry interface ---------------------------------------------

    @property
    def latitudes(self) -> cp.ndarray:
        return self._latitudes

    @property
    def longitudes(self) -> cp.ndarray:
        return self._longitudes

    @property
    def point_latitudes(self) -> cp.ndarray:
        return self._point_latitudes

    @property
    def point_longitudes(self) -> cp.ndarray:
        return self._point_longitudes

    @property
    def n_points(self) -> int:
        return self.nlat * self.nlon

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)

    def points_latlon(self) -> np.ndarray:
        return np.column_stack([
            cp.asnumpy(self._point_longitudes),
            cp.asnumpy(self._point_latitudes),
        ])

    # -- quadrature / metric ---------------------------------------------------

    @property
    def cell_areas(self) -> cp.ndarray:
        """Physical per-point areas (solid angle x R^2); same role as the
        geodesic grid's Voronoi cell areas (viz weighting, diagnostics)."""
        return self.solid_angle_weights * self.radius**2

    @property
    def cfl_length_scale(self) -> float:
        """Minimum meridional node spacing, R * min(dcolat).

        The zonal direction imposes no tighter constraint for a spectral
        transform method: the zonal wavenumber of any representable field is
        capped by l_max, so the resolved zonal wavelength *per grid point*
        (2*pi*R*cos(lat)/m divided by nlon points at that latitude) is
        latitude-independent — pole-ward metric shrinkage of dx is matched by
        the shrinkage of the fastest representable zonal feature. GL colatitude
        nodes are near-uniform (theta_j ~ (j - 1/4)*pi/(nlat + 1/2)), so this
        scale is ~ pi*R/nlat, not pathologically small at the poles.
        """
        colat = np.pi / 2.0 - self._latitudes_np
        return self.radius * float(np.min(np.abs(np.diff(colat))))


class GaussLatLonSphericalHarmonics:
    """SH analysis/synthesis on a GaussLatLonGridGeometry.

    Thin pairing of the grid's exact tensor-product quadrature with the
    GPU matrix transform (`PointSetSphericalHarmonics`). Fields are flat
    (n_points,) arrays in the geometry's lat-major point order; 2D
    (nlat, nlon) inputs to `transform` are accepted and flattened.
    """

    def __init__(self, grid: GaussLatLonGridGeometry, l_max: int):
        if not isinstance(grid, GaussLatLonGridGeometry):
            raise ValueError(
                f"GaussLatLonSphericalHarmonics requires a "
                f"GaussLatLonGridGeometry, got {type(grid).__name__}")
        self.grid = grid
        self.l_max = int(l_max)

        # Exactness envelope for analysis of degree-l_max fields.
        if grid.nlat < l_max + 1 or grid.nlon < 2 * l_max + 1:
            warnings.warn(
                f"Gauss lat-lon SH is under-resolved: ({grid.nlat}, "
                f"{grid.nlon}) grid for l_max={l_max}. Analysis is exact "
                f"only for nlat >= {l_max + 1} and nlon >= {2 * l_max + 1}; "
                f"below that, round trips leak energy across modes.",
                stacklevel=2,
            )

        self.sh = PointSetSphericalHarmonics(
            grid.point_latitudes,
            grid.point_longitudes,
            self.l_max,
            weights=grid.solid_angle_weights,
        )
        self.weights = self.sh.weights

    def transform(self, values) -> cp.ndarray:
        """Analysis: field values (flat or (nlat, nlon)) -> coefficients."""
        return self.sh.transform(values)

    def inv_transform(self, coeffs) -> cp.ndarray:
        """Synthesis: coefficients -> flat (n_points,) field."""
        return self.sh.inv_transform(coeffs)

    def inverse_transform(self, coeffs):
        """Alias for inv_transform."""
        return self.inv_transform(coeffs)

    def __getattr__(self, name):
        return getattr(self.sh, name)
