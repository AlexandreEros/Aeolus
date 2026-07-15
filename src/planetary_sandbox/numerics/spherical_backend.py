"""Backend seam separating grid geometry from spherical-harmonic transforms.

A *backend* pairs one grid geometry (point locations, metric data) with one
spherical-harmonic transform (analysis/synthesis on those points) and is the
sole authority on where nonlinear (pseudospectral) products may be evaluated.
Operator code (`SpectralOperators`) talks only to this interface and contains
no knowledge of any particular grid family.

Product-quadrature contract
---------------------------
`product_space(mode)` returns a :class:`ProductSpace` — the sampling on which
pointwise products are formed and analyzed back into the shared
`(l_max+1, l_max+1)` coefficient layout. Modes are **backend-defined**:

* ``"coarse"`` is mandatory for every backend and means "the state sampling
  itself" (the historical behavior).
* ``"fine"`` is optional and means "an overresolved product sampling of this
  backend's choosing". The geodesic backend uses a resolution-(r+1) co-grid;
  a structured lat-lon backend could instead use a denser Gauss-Legendre grid
  sized by the 3/2 rule, a cubed-sphere backend a refined panel set, etc.
  Nothing in the interface assumes a geodesic co-grid — only that the
  returned `ProductSpace` can synthesize coefficient fields at its points and
  analyze point values back to coefficients with its own quadrature.

Unsupported modes must raise ``ValueError``; there is **no silent fallback**
(docs/KNOWN_RISKS.md R-3 provenance requirement). Product spaces are constructed
once per backend (at first request, cached) — never per tendency call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cupy as cp

from .geodesic_grid import GeodesicGridGeometry
from .grid_base import GridGeometry
from .latlon_grid import GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics
from .optimized_geodesic_sh import GeodesicSphericalHarmonics


@dataclass(frozen=True)
class ProductSpace:
    """A sampling on which pointwise products are evaluated and analyzed.

    Attributes
    ----------
    sh : object
        Transform provider for this sampling: ``inv_transform(coeffs)``
        synthesizes a coefficient field at the product points and
        ``transform(values)`` analyzes point values into the shared
        coefficient layout using this sampling's quadrature.
    coslat : cp.ndarray
        cos(latitude) per product point, clamped away from zero (used for
        the metric factors of pointwise operators such as the Jacobian).
    geometry : GridGeometry | None
        The geometry of the product sampling, when one exists as a distinct
        object (None when the product space *is* the state sampling and no
        separate geometry was built).
    label : str
        Human-readable provenance (e.g. ``"geodesic-res5-voronoi"``), for
        manifests and error messages.
    """

    sh: Any
    coslat: cp.ndarray
    geometry: GridGeometry | None
    label: str


_COSLAT_FLOOR = 1e-8  # matches the historical divide-by-zero guard


class SphericalGridBackend(ABC):
    """Pairing of a grid geometry with an SH transform, plus product spaces."""

    def __init__(self, geometry: GridGeometry, sh: Any):
        self.geometry = geometry
        self.sh = sh
        self.l_max = sh.l_max
        self._product_spaces: dict[str, ProductSpace] = {}

    # -- interface ---------------------------------------------------------

    @abstractmethod
    def supported_product_quadratures(self) -> tuple[str, ...]:
        """Modes accepted by product_space(); must include 'coarse'."""

    def product_space(self, mode: str) -> ProductSpace:
        """Return the (cached) product space for `mode`; raise on unsupported."""
        if mode not in self.supported_product_quadratures():
            raise ValueError(
                f"product_quadrature must be one of "
                f"{self.supported_product_quadratures()}, got {mode!r} "
                f"(backend {type(self).__name__}; no silent fallback)"
            )
        if mode not in self._product_spaces:
            self._product_spaces[mode] = self._build_product_space(mode)
        return self._product_spaces[mode]

    @abstractmethod
    def _build_product_space(self, mode: str) -> ProductSpace:
        """Construct the product space for a supported `mode` (called once)."""

    def describe(self, product_quadrature: str) -> dict:
        """JSON-serializable numerics provenance for run manifests.

        Records the backend family, grid family, state sampling, the product
        sampling actually used for `product_quadrature`, and the transform
        type — enough to know which numerics produced a run's outputs.
        """
        return {
            "backend": type(self).__name__,
            "grid": type(self.geometry).__name__,
            "state_sampling": self.product_space("coarse").label,
            "product_quadrature": product_quadrature,
            "product_sampling": self.product_space(product_quadrature).label,
            "transform": type(self.sh).__name__,
            "l_max": int(self.l_max),
        }

    # -- shared helpers ------------------------------------------------------

    def _coarse_coslat(self) -> cp.ndarray:
        coslat = getattr(self.geometry, "coslat", None)
        if coslat is None:
            coslat = cp.cos(cp.asarray(self.geometry.point_latitudes))
        return cp.maximum(cp.asarray(coslat), _COSLAT_FLOOR)


class GeodesicBackend(SphericalGridBackend):
    """Icosahedral-geodesic backend.

    'fine' builds one resolution-(r+1) geodesic co-grid with Voronoi
    quadrature at the same l_max ("overresolved product quadrature",
    docs/KNOWN_RISKS.md R-3) — a backend-specific choice, not part of the
    interface contract.
    """

    def __init__(self, geometry: GeodesicGridGeometry, sh: Any):
        if not isinstance(geometry, GeodesicGridGeometry):
            raise ValueError(
                f"GeodesicBackend requires a GeodesicGridGeometry, got "
                f"{type(geometry).__name__}")
        super().__init__(geometry, sh)

    def supported_product_quadratures(self) -> tuple[str, ...]:
        return ("coarse", "fine")

    def _build_product_space(self, mode: str) -> ProductSpace:
        if mode == "coarse":
            return ProductSpace(
                sh=self.sh,
                coslat=self._coarse_coslat(),
                geometry=None,
                label=f"geodesic-res{self.geometry.resolution}-state",
            )
        # mode == "fine"
        fine_geometry = GeodesicGridGeometry(
            self.geometry.resolution + 1, self.geometry.radius)
        fine_sh = GeodesicSphericalHarmonics(
            fine_geometry, self.l_max, weights="voronoi")
        return ProductSpace(
            sh=fine_sh,
            coslat=cp.maximum(cp.asarray(fine_geometry.coslat), _COSLAT_FLOOR),
            geometry=fine_geometry,
            label=f"geodesic-res{fine_geometry.resolution}-voronoi",
        )


class LatLonBackend(SphericalGridBackend):
    """Structured Gauss-Legendre lat-lon backend.

    'fine' builds one denser Gauss-Legendre product grid sized by the 3/2
    rule (nlat >= ceil((3*l_max+1)/2), nlon >= 3*l_max+1), on which the
    pointwise product of two degree-l_max fields is analyzed EXACTLY against
    degree-l_max harmonics — true dealiasing quadrature, not merely the
    overresolved approximation the geodesic co-grid provides.
    """

    def __init__(self, geometry: GaussLatLonGridGeometry, sh: Any):
        if not isinstance(geometry, GaussLatLonGridGeometry):
            raise ValueError(
                f"LatLonBackend requires a GaussLatLonGridGeometry, got "
                f"{type(geometry).__name__}")
        super().__init__(geometry, sh)

    def supported_product_quadratures(self) -> tuple[str, ...]:
        return ("coarse", "fine")

    def _build_product_space(self, mode: str) -> ProductSpace:
        g = self.geometry
        if mode == "coarse":
            return ProductSpace(
                sh=self.sh,
                coslat=self._coarse_coslat(),
                geometry=None,
                label=f"latlon-gauss-{g.nlat}x{g.nlon}-state",
            )
        # mode == "fine": 3/2-rule product grid — exact for quadratic products.
        nlat_f = max((3 * self.l_max) // 2 + 1, g.nlat)
        nlon_f = max(3 * self.l_max + 1, g.nlon)
        fine_geometry = GaussLatLonGridGeometry(nlat_f, nlon_f, g.radius)
        fine_sh = GaussLatLonSphericalHarmonics(fine_geometry, self.l_max)
        return ProductSpace(
            sh=fine_sh,
            coslat=cp.maximum(cp.asarray(fine_geometry.coslat), _COSLAT_FLOOR),
            geometry=fine_geometry,
            label=(f"latlon-gauss-{nlat_f}x{nlon_f}-3/2rule"),
        )


class PointSetBackend(SphericalGridBackend):
    """Fallback backend for arbitrary point sets / non-geodesic geometries.

    Supports only 'coarse'. Grid families that have a principled overresolved
    product sampling should get their own backend (e.g. a future lat-lon
    backend using a denser Gauss-Legendre product grid) rather than extending
    this one.
    """

    def supported_product_quadratures(self) -> tuple[str, ...]:
        return ("coarse",)

    def _build_product_space(self, mode: str) -> ProductSpace:
        return ProductSpace(
            sh=self.sh,
            coslat=self._coarse_coslat(),
            geometry=None,
            label="point-set-state",
        )


def make_backend(geometry: GridGeometry, sh: Any) -> SphericalGridBackend:
    """Infer the appropriate backend for a geometry/transform pairing."""
    if isinstance(geometry, GeodesicGridGeometry):
        return GeodesicBackend(geometry, sh)
    if isinstance(geometry, GaussLatLonGridGeometry):
        return LatLonBackend(geometry, sh)
    return PointSetBackend(geometry, sh)
