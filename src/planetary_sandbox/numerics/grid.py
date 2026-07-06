import numpy as np
from typing import Tuple

from .grid_base import GridGeometry

class LatLonGridGeometry(GridGeometry):
    """Grid information for the planet surface."""

    def __init__(
        self,
        num_lat: int,
        num_lon: int,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> None:
        self.num_lat = num_lat
        self.num_lon = num_lon
        self._latitudes = latitudes
        self._longitudes = longitudes
        self._lat_grid = lat_grid
        self._lon_grid = lon_grid

    # Colatitude (for spherical harmonics)
    @property
    def latitudes(self) -> np.ndarray:
        return self._latitudes

    @latitudes.setter
    def latitudes(self, value: np.ndarray) -> None:
        self._latitudes = value

    @property
    def longitudes(self) -> np.ndarray:
        return self._longitudes

    @longitudes.setter
    def longitudes(self, value: np.ndarray) -> None:
        self._longitudes = value

    @property
    def lat_grid(self) -> np.ndarray:
        return self._lat_grid

    @lat_grid.setter
    def lat_grid(self, value: np.ndarray) -> None:
        self._lat_grid = value

    @property
    def lon_grid(self) -> np.ndarray:
        return self._lon_grid

    @lon_grid.setter
    def lon_grid(self, value: np.ndarray) -> None:
        self._lon_grid = value

    @property
    def colatitudes(self):
        return np.pi/2 - self.latitudes

    @property
    def colat_grid(self):
      return np.pi/2 - self.lat_grid

    @property
    def point_latitudes(self) -> np.ndarray:
        return self.lat_grid.ravel()

    @property
    def point_longitudes(self) -> np.ndarray:
        return self.lon_grid.ravel()

    @property
    def n_points(self) -> int:
        return int(self.num_lat * self.num_lon)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.lat_grid.shape

    def points_latlon(self) -> np.ndarray:
        return np.column_stack([self.lon_grid.ravel(), self.lat_grid.ravel()])

    @classmethod
    def create(cls, grid_resolution: Tuple[int, int],
              lon_range: Tuple[float, float] = (0, 2*np.pi)) -> 'LatLonGridGeometry':

        num_lat, num_lon = grid_resolution

        # For Simpson integration, num_lat and num_lon must be odd
        num_lat = num_lat + 1 - (num_lat % 2)
        num_lon = num_lon + 1 - (num_lon % 2)

        # Colatitude from north pole (0) to south pole (pi)
        colatitudes = np.linspace(0, np.pi, num_lat)
        # Latitude derived from colatitude
        latitudes = np.pi/2 - colatitudes
        longitudes = np.linspace(*lon_range, endpoint=False, num=num_lon)

        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        return cls(
            num_lat=num_lat,
            num_lon=num_lon,
            latitudes=latitudes,
            longitudes=longitudes,
            lat_grid=lat_grid,
            lon_grid=lon_grid
        )
