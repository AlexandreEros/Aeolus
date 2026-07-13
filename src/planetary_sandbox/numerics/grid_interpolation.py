import numpy as np
import cupy as cp
from typing import Literal
from scipy.interpolate import griddata

from .geodesic_grid import GeodesicGridGeometry
from .grid import LatLonGridGeometry


def _wrap_longitudes(longitudes: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    span = target_max - target_min
    if not np.isfinite(span) or span <= 0:
        return longitudes

    two_pi = 2.0 * np.pi
    if np.isclose(span, two_pi, rtol=1e-3, atol=1e-6) or span > two_pi * 0.9:
        return ((longitudes - target_min) % two_pi) + target_min

    return longitudes


def geodesic_to_latlon_grid(
    values: np.ndarray,
    geodesic_grid: GeodesicGridGeometry,
    latlon_grid: LatLonGridGeometry,
    method: Literal["nearest", "linear", "cubic"] = "linear",
    fill_value: float | None = np.nan,
) -> np.ndarray:
    """
    Interpolate values defined on a geodesic grid onto a structured lat/lon grid.
    """
    values_arr = np.asarray(values)
    if values_arr.shape[0] != geodesic_grid.n_points:
        raise ValueError("values must have length n_points from geodesic_grid")
    if values_arr.ndim > 2:
        raise ValueError("values must be 1D or 2D (n_points, n_fields)")

    # Per-point coordinates: identical to .longitudes/.latitudes for geodesic
    # grids, and the flattened axes for structured source grids.
    lon_src = cp.asnumpy(cp.asarray(geodesic_grid.point_longitudes))
    lat_src = cp.asnumpy(cp.asarray(geodesic_grid.point_latitudes))
    lon_tgt = cp.asnumpy(latlon_grid.lon_grid)
    lat_tgt = cp.asnumpy(latlon_grid.lat_grid)

    lon_src = _wrap_longitudes(lon_src, float(lon_tgt.min()), float(lon_tgt.max()))

    points = np.column_stack([lon_src, lat_src])
    target_points = np.column_stack([lon_tgt.ravel(), lat_tgt.ravel()])

    out_flat = griddata(
        points=points,
        values=values_arr,
        xi=target_points,
        method=method,
        fill_value=fill_value,
    )

    if values_arr.ndim == 1:
        return out_flat.reshape(lat_tgt.shape)

    return out_flat.reshape(lat_tgt.shape + (values_arr.shape[1],))


def latlon_to_geodesic_grid(
    values: np.ndarray,
    latlon_grid: LatLonGridGeometry,
    geodesic_grid: GeodesicGridGeometry,
    method: Literal["nearest", "linear", "cubic"] = "linear",
    fill_value: float | None = np.nan,
) -> np.ndarray:
    """
    Interpolate values defined on a structured lat/lon grid onto a geodesic grid.
    """
    values_arr = np.asarray(values)
    if values_arr.shape[:2] != latlon_grid.lat_grid.shape:
        raise ValueError("values must match latlon_grid.lat_grid shape")
    if values_arr.ndim > 3:
        raise ValueError("values must be 2D or 3D (n_lat, n_lon, n_fields)")

    lon_src = cp.asnumpy(latlon_grid.lon_grid)
    lat_src = cp.asnumpy(latlon_grid.lat_grid)
    lon_tgt = cp.asnumpy(geodesic_grid.longitudes)
    lat_tgt = cp.asnumpy(geodesic_grid.latitudes)

    lon_tgt = _wrap_longitudes(lon_tgt, float(lon_src.min()), float(lon_src.max()))

    points = np.column_stack([lon_src.ravel(), lat_src.ravel()])
    values_flat = values_arr.reshape(points.shape[0], -1)
    target_points = np.column_stack([lon_tgt, lat_tgt])

    out = griddata(
        points,
        values_flat,
        target_points,
        method=method,
        fill_value=fill_value,
    )

    if values_arr.ndim == 2:
        return out.reshape(geodesic_grid.n_points)

    return out.reshape(geodesic_grid.n_points, values_arr.shape[2])
