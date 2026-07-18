"""Adapters from simulation samplings to the repository's map-view grid."""
from __future__ import annotations

import numpy as np

from ..numerics import LatLonGridGeometry, geodesic_to_latlon_grid
from .fields import ScalarGridField


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def map_to_uniform_latlon(values, source_grid, *,
                          target_grid: LatLonGridGeometry | None = None
                          ) -> tuple[LatLonGridGeometry, np.ndarray]:
    """Map any repository state sampling to the standard 91x181 view grid."""
    target = target_grid or LatLonGridGeometry.create((91, 181))
    values = _host(values)
    if values.ndim == 2 and values.shape == target.lat_grid.shape:
        return target, values
    mapped = geodesic_to_latlon_grid(
        values, source_grid, target, method="linear")
    if np.isnan(mapped).any():
        nearest = geodesic_to_latlon_grid(
            values, source_grid, target, method="nearest")
        mapped = np.where(np.isnan(mapped), nearest, mapped)
    return target, mapped


def scalar_field_on_uniform_latlon(values, source_grid, *, name: str,
                                   units: str,
                                   times: np.ndarray | None = None
                                   ) -> ScalarGridField:
    """Build a physical-grid field on the established uniform map sampling."""
    target, mapped = map_to_uniform_latlon(values, source_grid)
    return ScalarGridField(
        mapped, _host(target.latitudes), _host(target.longitudes),
        name=name, units=units, times=times)
