"""Backend-independent scientific field representations.

The repository stores planetary grids north-to-south in latitude and with a
periodic, endpoint-exclusive longitude axis in ``[0, 2*pi)``.  The types in
this module make that convention explicit and contain no rendering objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _array(value, *, name: str) -> np.ndarray:
    """Return a NumPy array without importing a device-array implementation."""
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} must be a numeric array")
    return array


def _times(value, *, expected: int) -> np.ndarray | None:
    if value is None:
        return None
    times = _array(value, name="times").astype(np.float64, copy=False)
    if times.ndim != 1 or times.size != expected:
        raise ValueError(
            f"times must be one-dimensional with length {expected}, got "
            f"shape {times.shape}")
    if not np.isfinite(times).all():
        raise ValueError("times must contain only finite values")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    return times


def _normalize_index(index: int, size: int) -> int:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise TypeError(f"time index must be an integer, got {index!r}")
    normalized = int(index)
    if normalized < 0:
        normalized += size
    if normalized < 0 or normalized >= size:
        raise IndexError(f"time index {index} is out of range for {size} state(s)")
    return normalized


@dataclass(frozen=True)
class ScalarGridField:
    """A scalar sampled on the planetary latitude-longitude grid.

    ``values`` has shape ``(lat, lon)`` or ``(time, lat, lon)``.  Latitude is
    stored north-to-south; longitude is increasing, periodic, and excludes the
    duplicate ``2*pi`` seam, matching the simulation-grid convention.
    """

    values: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    name: str
    units: str
    times: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _array(self.values, name="values")
        latitudes = _array(self.latitudes, name="latitudes").astype(
            np.float64, copy=False)
        longitudes = _array(self.longitudes, name="longitudes").astype(
            np.float64, copy=False)

        if values.ndim not in (2, 3):
            raise ValueError(
                "scalar-grid values must have shape (lat, lon) or "
                f"(time, lat, lon), got {values.shape}")
        if latitudes.ndim != 1 or longitudes.ndim != 1:
            raise ValueError("latitude and longitude coordinates must be 1D")
        if latitudes.size == 0 or longitudes.size == 0:
            raise ValueError("latitude and longitude coordinates must be nonempty")
        if values.shape[-2:] != (latitudes.size, longitudes.size):
            raise ValueError(
                f"values end in {values.shape[-2:]}, but coordinates imply "
                f"({latitudes.size}, {longitudes.size})")
        if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
            raise ValueError("latitude and longitude coordinates must be finite")
        tol = 32.0 * np.finfo(np.float64).eps
        if np.any(latitudes < -np.pi / 2.0 - tol) or np.any(
                latitudes > np.pi / 2.0 + tol):
            raise ValueError("latitudes must lie in [-pi/2, pi/2]")
        if latitudes.size > 1 and not np.all(np.diff(latitudes) < 0.0):
            raise ValueError("latitudes must be strictly north-to-south")
        if np.any(longitudes < -tol) or np.any(longitudes >= 2.0 * np.pi - tol):
            raise ValueError(
                "longitudes must use the endpoint-exclusive [0, 2*pi) convention")
        if longitudes.size > 1 and not np.all(np.diff(longitudes) > 0.0):
            raise ValueError("longitudes must be strictly increasing")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("field name must be a nonempty string")
        if not isinstance(self.units, str):
            raise TypeError("field units must be a string")

        state_count = values.shape[0] if values.ndim == 3 else 1
        times = _times(self.times, expected=state_count)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "latitudes", latitudes)
        object.__setattr__(self, "longitudes", longitudes)
        object.__setattr__(self, "times", times)

    @property
    def state_count(self) -> int:
        return self.values.shape[0] if self.values.ndim == 3 else 1

    def values_at(self, time_index: int = 0) -> np.ndarray:
        index = _normalize_index(time_index, self.state_count)
        return self.values[index] if self.values.ndim == 3 else self.values

    def select_time(self, time_index: int) -> "ScalarGridField":
        index = _normalize_index(time_index, self.state_count)
        selected_times = None if self.times is None else self.times[index:index + 1]
        return ScalarGridField(
            self.values_at(index), self.latitudes, self.longitudes,
            self.name, self.units, selected_times)


@dataclass(frozen=True)
class SphericalHarmonicField:
    """Complex coefficients in the repository's unpacked ``(l, m>=0)`` layout.

    Coefficients have shape ``(degree, order)`` or
    ``(time, degree, order)``.  Both coefficient axes have length
    ``l_max + 1``; only positions ``m <= l`` are valid.
    """

    coefficients: np.ndarray
    name: str
    units: str
    times: np.ndarray | None = None
    normalization: str = "orthonormal"
    layout: str = "unpacked-l-m-nonnegative"
    longitude_origin_radians: float = 0.0

    def __post_init__(self) -> None:
        coefficients = _array(self.coefficients, name="coefficients")
        if coefficients.ndim not in (2, 3):
            raise ValueError(
                "spectral coefficients must have shape (l, m) or "
                f"(time, l, m), got {coefficients.shape}")
        if coefficients.shape[-2] != coefficients.shape[-1]:
            raise ValueError(
                "unpacked coefficient storage must have equal degree/order "
                f"axis lengths, got {coefficients.shape[-2:]}")
        if coefficients.shape[-1] == 0:
            raise ValueError("coefficient axes must be nonempty")
        if not np.iscomplexobj(coefficients):
            raise TypeError("spherical-harmonic coefficients must be complex-valued")
        if self.layout != "unpacked-l-m-nonnegative":
            raise ValueError(f"unsupported coefficient layout {self.layout!r}")
        if not isinstance(self.normalization, str) or not self.normalization.strip():
            raise ValueError("spherical-harmonic normalization must be nonempty")
        if not np.isfinite(self.longitude_origin_radians):
            raise ValueError("longitude origin must be finite")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("field name must be a nonempty string")
        if not isinstance(self.units, str):
            raise TypeError("field units must be a string")

        state_count = coefficients.shape[0] if coefficients.ndim == 3 else 1
        times = _times(self.times, expected=state_count)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "times", times)

    @property
    def l_max(self) -> int:
        return self.coefficients.shape[-1] - 1

    @property
    def state_count(self) -> int:
        return self.coefficients.shape[0] if self.coefficients.ndim == 3 else 1

    @property
    def valid_mask(self) -> np.ndarray:
        degree = np.arange(self.l_max + 1)[:, None]
        order = np.arange(self.l_max + 1)[None, :]
        return order <= degree

    def coefficients_at(self, time_index: int = 0) -> np.ndarray:
        index = _normalize_index(time_index, self.state_count)
        if self.coefficients.ndim == 3:
            return self.coefficients[index]
        return self.coefficients

    def select_time(self, time_index: int) -> "SphericalHarmonicField":
        index = _normalize_index(time_index, self.state_count)
        selected_times = None if self.times is None else self.times[index:index + 1]
        return SphericalHarmonicField(
            self.coefficients_at(index), self.name, self.units,
            selected_times, self.normalization, self.layout,
            self.longitude_origin_radians)
