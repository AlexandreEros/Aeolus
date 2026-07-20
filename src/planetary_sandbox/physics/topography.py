"""Immutable, band-limited bottom topography on the sphere.

This module is the minimal topography foundation shared by spherical
spectral cores. It deliberately distinguishes, everywhere, between

* surface **elevation** ``h_s`` in metres (geometry, gravity-independent),
* surface **geopotential** ``phi_s = g * h_s`` in m^2/s^2 (dynamics).

A :class:`Topography` stores the spectral coefficients of the surface
*elevation* at exactly the model truncation ``l_max`` in the repository's
standard dense ``(l_max+1, l_max+1)`` complex layout, resident on the GPU.
Because the field is held spectrally, it can be synthesized exactly on any
sampling a solver evaluates on (state grid, fine product grid) with the
model's own transforms — no interpolation, no per-step host-device traffic.

:meth:`Topography.surface_geopotential_lm` returns ``g * elevation_lm``,
which is byte-layout compatible with the ``surface_geopotential_lm``
argument the primitive-equations core already reserves
(``physics/primitive_equations.py``). No PE coupling exists yet; this only
means a later PE integration is a data change, not a schema change.

Presets
-------
``flat``
    Canonical default: all coefficients exactly zero. A model given a flat
    topography must behave bit-for-bit like one given no topography.

``mountain``
    One smooth isolated Gaussian mountain
    ``h_s(x) = h0 * exp(-(d/sigma)^2)`` where ``d`` is the great-circle
    angular distance from the center and ``sigma`` the e-folding width.
    The mountain is **defined analytically on the construction grid and
    projected onto the truncation by the backend's own analysis
    transform** (``sh.transform``); it is not constructed directly in
    spectral space. The projection is validated: the quadrature-weighted
    relative L2 residual between the analytic field and its band-limited
    synthesis must not exceed :data:`MAX_PROJECTION_RESIDUAL`, otherwise
    construction fails loudly (a too-narrow mountain cannot be silently
    smoothed into something else). The projected terrain necessarily
    carries Gibbs-type ripples; the residual bound quantifies them.

Immutability is by convention (CuPy arrays cannot be write-protected):
accessors return defensive copies of the coefficient array, and nothing in
this module ever mutates a stored array after construction.
"""
from __future__ import annotations

import math

import cupy as cp

#: Presets available to configuration layers (kept in sync with
#: run/swe/config.SWE_TOPOGRAPHIES, which must stay import-light).
#: ``williamson5_cone`` is benchmark-owned: it is constructed only by the
#: ``williamson5`` shallow-water scenario and is deliberately NOT part of
#: the user-facing SWE/PE ``--topography`` vocabulary.
TOPOGRAPHY_PRESETS = ("flat", "mountain", "williamson5_cone")

#: Maximum quadrature-weighted relative L2 residual accepted between the
#: analytic mountain and its band-limited projection. Above this the terrain
#: is not faithfully representable at the model truncation and construction
#: fails (widen the mountain or raise l_max) rather than silently aliasing.
MAX_PROJECTION_RESIDUAL = 0.2

#: Physical sanity cap on the mountain amplitude (m). Far above any depth
#: the shallow-water core can carry; rejects unit mistakes (e.g. km vs m).
MAX_MOUNTAIN_HEIGHT_M = 1.0e5

# ---------------------------------------------------------------------------
# Williamson et al. (1992) test case 5: the canonical conical mountain.
# ---------------------------------------------------------------------------
#: Canonical cone parameters (Williamson et al. 1992, JCP 102, case 5):
#: peak height hs0 (m), support radius R0 (rad), center (pi/6, 3*pi/2).
W5_CONE_HEIGHT_M = 2000.0
W5_CONE_RADIUS_RAD = math.pi / 9.0
W5_CONE_LAT_RAD = math.pi / 6.0
W5_CONE_LON_RAD = 3.0 * math.pi / 2.0

#: Benchmark-specific projection gate for the canonical cone. The cone is
#: C^0 with a summit cusp and a slope discontinuity at its base, so it is
#: NOT band-limited and its projection residual is a property of the
#: benchmark to REPORT, not a defect to hide. Measured quadrature-weighted
#: relative L2 residuals: Gauss-Legendre 0.0895 (l_max=15) -> 0.0121
#: (l_max=63); geodesic res4/l_max=21 0.0643. The gate only rejects
#: qualitatively degraded terrain (measured: geodesic res3/l_max=10 gives
#: 0.3276 — visibly no longer the canonical cone). It is deliberately
#: separate from the Gaussian preset's MAX_PROJECTION_RESIDUAL = 0.2,
#: which remains unchanged.
W5_CONE_MAX_PROJECTION_RESIDUAL = 0.25


class TopographyError(ValueError):
    """A topography definition is invalid or not representable as configured."""


def _require(name: str, value, *, lo: float | None = None,
             hi: float | None = None, positive: bool = False) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise TopographyError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(f):
        raise TopographyError(f"{name} must be finite, got {value}")
    if positive and not f > 0.0:
        raise TopographyError(f"{name} must be > 0, got {f}")
    if lo is not None and f < lo:
        raise TopographyError(f"{name} must be >= {lo}, got {f}")
    if hi is not None and f > hi:
        raise TopographyError(f"{name} must be <= {hi}, got {f}")
    return f


def williamson5_cone_elevation(lat: cp.ndarray, lon: cp.ndarray, *,
                               height_m: float = W5_CONE_HEIGHT_M
                               ) -> cp.ndarray:
    """Analytic Williamson-5 cone elevation (m) at (lat, lon) radians.

    The published definition (Williamson et al. 1992, case 5) uses the
    COORDINATE-PLANE angular distance — NOT great-circle distance —

        r = min(R0, sqrt(dlambda^2 + (lat - lat_c)^2))
        h_s = hs0 * (1 - r / R0)

    with the longitude difference ``dlambda`` wrapped continuously into
    [-pi, pi] around the center, so the cone is well-defined on every
    longitude branch. The result has compact support (exactly zero for
    r >= R0), a summit cusp, and a slope discontinuity at the base; these
    are properties of the analytic benchmark, deliberately preserved.
    """
    lat = cp.asarray(lat, dtype=cp.float64)
    lon = cp.asarray(lon, dtype=cp.float64)
    dlam = (lon - W5_CONE_LON_RAD + math.pi) % (2.0 * math.pi) - math.pi
    r = cp.minimum(W5_CONE_RADIUS_RAD,
                   cp.sqrt(dlam * dlam + (lat - W5_CONE_LAT_RAD) ** 2))
    return float(height_m) * (1.0 - r / W5_CONE_RADIUS_RAD)


def _analyze_elevation(sh, analytic: cp.ndarray, *,
                       l_cut: int | None = None
                       ) -> tuple[cp.ndarray, cp.ndarray, float]:
    """Project an analytic state-grid elevation field onto the truncation.

    Shared by the Gaussian-mountain and Williamson-5-cone constructors:
    analyze once with the backend's own transform, optionally zero every
    coefficient above ``l_cut``, synthesize once, and measure the
    quadrature-weighted relative L2 residual between the analytic field and
    its band-limited synthesis. Returns ``(coeffs, synthesized, residual)``;
    the caller applies its own preset-specific residual policy.
    """
    coeffs = cp.asarray(sh.transform(analytic), dtype=cp.complex128)
    if not bool(cp.isfinite(coeffs).all()):
        raise TopographyError(
            "elevation projection produced non-finite coefficients")
    if l_cut is not None:
        coeffs[int(l_cut) + 1:, :] = 0.0
        coeffs[:, int(l_cut) + 1:] = 0.0
    synthesized = sh.inv_transform(coeffs).real
    w = cp.asarray(sh.weights, dtype=cp.float64)
    norm = float(cp.sqrt(cp.sum(w * analytic**2)))
    residual = float(cp.sqrt(cp.sum(w * (synthesized - analytic) ** 2)))
    rel = residual / norm if norm > 0.0 else math.inf
    return coeffs, synthesized, rel


class Topography:
    """Fixed (time-independent) bottom topography held spectrally on the GPU.

    Build instances with :meth:`flat` or :meth:`mountain`; the constructor
    only validates and stores already-projected coefficients.

    Parameters
    ----------
    elevation_lm : cp.ndarray
        Surface-elevation spectral coefficients (metres), dense complex
        ``(l_max+1, l_max+1)`` layout at the model truncation.
    preset : str
        Preset name (member of :data:`TOPOGRAPHY_PRESETS`), for provenance.
    parameters : dict
        Resolved preset parameters (plain floats), for provenance/summaries.
    """

    def __init__(self, elevation_lm: cp.ndarray, *, preset: str,
                 parameters: dict | None = None):
        if preset not in TOPOGRAPHY_PRESETS:
            raise TopographyError(
                f"unknown topography preset {preset!r}; "
                f"choose from {TOPOGRAPHY_PRESETS}")
        arr = cp.asarray(elevation_lm, dtype=cp.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            raise TopographyError(
                "elevation_lm must be a square (l_max+1, l_max+1) array "
                f"with l_max >= 1, got shape {tuple(arr.shape)}")
        if not bool(cp.isfinite(arr).all()):
            raise TopographyError("elevation_lm contains NaN/Inf coefficients")
        self._elevation_lm = arr.copy()
        self._preset = str(preset)
        self._parameters = dict(parameters or {})
        self._l_max = int(arr.shape[0]) - 1
        # A preset tagged flat must be exactly zero terrain, and vice versa
        # nothing enforces a "mountain" to be nonzero (a zero-height mountain
        # is rejected earlier by parameter validation).
        self._is_flat = not bool(cp.any(arr))
        if preset == "flat" and not self._is_flat:
            raise TopographyError(
                "preset 'flat' requires all elevation coefficients to be zero")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def l_max(self) -> int:
        return self._l_max

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def parameters(self) -> dict:
        return dict(self._parameters)

    @property
    def is_flat(self) -> bool:
        """True iff every elevation coefficient is exactly zero."""
        return self._is_flat

    @property
    def elevation_lm(self) -> cp.ndarray:
        """Surface-elevation coefficients (metres); defensive copy."""
        return self._elevation_lm.copy()

    @property
    def mean_elevation_m(self) -> float:
        """Global-mean surface elevation (m), from the monopole exactly."""
        return float(self._elevation_lm[0, 0].real) / math.sqrt(4.0 * math.pi)

    def describe(self) -> str:
        """One-line human-readable summary for run summaries / inspect."""
        if self._preset == "mountain":
            p = self._parameters
            return (f"mountain (h={p.get('height_m'):g} m at "
                    f"lat {p.get('lat_deg'):g}°, "
                    f"lon {p.get('lon_deg'):g}°, "
                    f"width {p.get('width_deg'):g}°)")
        if self._preset == "williamson5_cone":
            p = self._parameters
            return (f"Williamson-5 cone (hs0={p.get('height_m'):g} m at "
                    f"lat 30°, lon -90°, R0=pi/9; projection residual "
                    f"{p.get('projection_residual'):.3f})")
        return "flat"

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------

    def surface_geopotential_lm(self, gravity: float) -> cp.ndarray:
        """Surface-geopotential coefficients ``phi_s = g * h_s`` (m^2/s^2).

        The returned layout matches the ``surface_geopotential_lm`` input the
        primitive-equations core reserves, so the same object can later feed
        both cores. Fresh array; the stored elevation is never aliased.
        """
        g = _require("gravity", gravity, positive=True)
        return g * self._elevation_lm

    def elevation_on(self, sh) -> cp.ndarray:
        """Synthesize the band-limited surface elevation (m) on ``sh``'s grid.

        ``sh`` is any of the model's transform objects (state or product
        sampling); synthesis is the exact basis evaluation at those points.
        """
        return sh.inv_transform(self._elevation_lm).real

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def flat(cls, l_max: int) -> "Topography":
        """The canonical flat bottom: all coefficients exactly zero."""
        if int(l_max) < 1:
            raise TopographyError(f"l_max must be >= 1, got {l_max}")
        n = int(l_max) + 1
        return cls(cp.zeros((n, n), dtype=cp.complex128), preset="flat",
                   parameters={})

    @classmethod
    def mountain(cls, planet, *, height_m: float, lat_deg: float,
                 lon_deg: float, width_deg: float,
                 l_cut: int | None = None) -> "Topography":
        """One smooth isolated Gaussian mountain, projected to the truncation.

        ``h_s(x) = height_m * exp(-(d/sigma)^2)`` with ``d`` the great-circle
        angular distance from ``(lat_deg, lon_deg)`` and ``sigma =
        width_deg`` (e-folding half-width) in radians. Evaluated analytically
        at the planet's state-grid points and analyzed once with the
        backend's own transform, so the stored field is band-limited at
        exactly the model truncation and identical for every later
        synthesis. Construction fails if the projection residual exceeds
        :data:`MAX_PROJECTION_RESIDUAL` (terrain too narrow for l_max).

        ``l_cut`` (optional) additionally zeroes every coefficient with
        degree or order above ``l_cut`` BEFORE the residual gate, so the
        gate validates the terrain actually used. The primitive-equation
        core requires terrain band-limited at its dealiased product
        truncation (2*l_max/3): its full-T pressure-gradient force reaches
        the tendency through the dealiased nonlinear pathway while
        -lap(Phi) is an exact diagonal spectral term, so terrain content
        above that cut can never cancel and would act as a permanent
        spurious momentum forcing (measured: the uncancelled per-degree
        residual equals |lap * Phi_s| exactly for l > cut). The
        shallow-water core has no such asymmetry and does not pass
        ``l_cut``.
        """
        height = _require("mountain height_m", height_m, positive=True,
                          hi=MAX_MOUNTAIN_HEIGHT_M)
        lat0 = _require("mountain lat_deg", lat_deg, lo=-90.0, hi=90.0)
        lon0 = _require("mountain lon_deg", lon_deg, lo=-360.0, hi=360.0)
        width = _require("mountain width_deg", width_deg, positive=True,
                         hi=90.0)
        if l_cut is not None and not 1 <= int(l_cut) <= planet.sh.l_max:
            raise TopographyError(
                f"l_cut must be in [1, l_max={planet.sh.l_max}], got {l_cut}")

        sh = planet.sh
        grid = planet.grid
        lat = cp.asarray(grid.point_latitudes, dtype=cp.float64)
        lon = cp.asarray(grid.point_longitudes, dtype=cp.float64)
        lat0_r = math.radians(lat0)
        lon0_r = math.radians(lon0)
        sigma = math.radians(width)

        # Great-circle angular distance from the center, clipped against
        # round-off before arccos.
        cosd = (math.sin(lat0_r) * cp.sin(lat)
                + math.cos(lat0_r) * cp.cos(lat) * cp.cos(lon - lon0_r))
        d = cp.arccos(cp.clip(cosd, -1.0, 1.0))
        analytic = height * cp.exp(-(d / sigma) ** 2)

        # Quantified band-limitedness: quadrature-weighted relative L2
        # residual between the analytic field and its truncated synthesis.
        coeffs, _, rel = _analyze_elevation(sh, analytic, l_cut=l_cut)
        effective_l = sh.l_max if l_cut is None else int(l_cut)
        if not (rel <= MAX_PROJECTION_RESIDUAL):
            raise TopographyError(
                f"mountain (height {height:g} m, width {width:g} deg) is not "
                f"representable at degree {effective_l}"
                + (f" (the dealiased cut of l_max={sh.l_max})"
                   if l_cut is not None else "")
                + f": relative projection residual {rel:.3f} exceeds "
                f"{MAX_PROJECTION_RESIDUAL}. "
                "Widen the mountain or raise the spectral resolution.")

        parameters = {
            "height_m": height, "lat_deg": lat0, "lon_deg": lon0,
            "width_deg": width, "projection_residual": rel,
        }
        if l_cut is not None:
            parameters["l_cut"] = int(l_cut)
        return cls(coeffs, preset="mountain", parameters=parameters)

    @classmethod
    def williamson5_cone(cls, planet, *,
                         height_m: float = W5_CONE_HEIGHT_M) -> "Topography":
        """The canonical Williamson et al. (1992) case-5 conical mountain.

        Geometry is fixed at the published values (hs0 = 2000 m, R0 = pi/9,
        center (pi/6, 3*pi/2), coordinate-plane distance, wrapped longitude
        — see :func:`williamson5_cone_elevation`); only the peak height is
        overridable, for characterization runs (a zero-height experiment
        uses :meth:`flat` instead).

        Projection policy: the analytic cone is evaluated at the planet's
        state-grid points and analyzed ONCE with the backend's own
        transform at the full model truncation (the shallow-water core's
        linear topographic term is spectrally symmetric, so no dealiasing
        cut is applied — unlike the PE ``l_cut`` pathway). The cone is not
        band-limited: the constructor MEASURES and RECORDS its projection
        error (relative L2 residual, synthesized elevation extrema on the
        state sampling, peak undershoot) in ``parameters`` instead of
        demanding smoothness, and rejects only a qualitatively degraded
        representation (see :data:`W5_CONE_MAX_PROJECTION_RESIDUAL`).
        Because the analysis uses each backend's own quadrature, the stored
        coefficients are backend-dependent at the ~1e-2 relative level
        (measured at l_max=21); this is characterized in the test suite
        rather than hidden behind a shared construction.
        """
        height = _require("cone height_m", height_m, positive=True,
                          hi=MAX_MOUNTAIN_HEIGHT_M)
        sh = planet.sh
        grid = planet.grid
        lat = cp.asarray(grid.point_latitudes, dtype=cp.float64)
        lon = cp.asarray(grid.point_longitudes, dtype=cp.float64)
        analytic = williamson5_cone_elevation(lat, lon, height_m=height)

        coeffs, synthesized, rel = _analyze_elevation(sh, analytic)
        if not (rel <= W5_CONE_MAX_PROJECTION_RESIDUAL):
            raise TopographyError(
                f"the Williamson-5 cone is not representable at the model "
                f"truncation l_max={sh.l_max} on this grid: relative "
                f"projection residual {rel:.3f} exceeds "
                f"{W5_CONE_MAX_PROJECTION_RESIDUAL} (the projected terrain "
                "is no longer qualitatively the canonical cone). Raise the "
                "spectral resolution.")

        elev_min = float(synthesized.min())
        elev_max = float(synthesized.max())
        parameters = {
            "height_m": height,
            "radius_rad": W5_CONE_RADIUS_RAD,
            "lat_center_deg": 30.0,
            "lon_center_deg": -90.0,
            "projection_residual": rel,
            "elevation_min_m": elev_min,
            "elevation_max_m": elev_max,
            "peak_error_m": height - elev_max,
        }
        return cls(coeffs, preset="williamson5_cone", parameters=parameters)
