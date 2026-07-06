from abc import ABC, abstractmethod
import cupy as cp


class GridGeometry(ABC):
    """
    Abstract base for grid geometries on a sphere.
    Subclasses must provide latitudes/longitudes and point-count access.
    For structured grids, latitudes/longitudes may be 1D axes; use
    point_latitudes/point_longitudes for per-point arrays.
    """

    @property
    @abstractmethod
    def latitudes(self) -> cp.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def longitudes(self) -> cp.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def point_latitudes(self) -> cp.ndarray:
        """
        Return per-point latitudes (length n_points).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def point_longitudes(self) -> cp.ndarray:
        """
        Return per-point longitudes (length n_points).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_points(self) -> int:
        raise NotImplementedError

    @property
    def colatitudes(self) -> cp.ndarray:
        return cp.pi / 2.0 - cp.asarray(self.latitudes)

    @property
    def grid_shape(self) -> tuple[int, int] | None:
        return None

    @property
    def is_structured(self) -> bool:
        return self.grid_shape is not None

    @abstractmethod
    def points_latlon(self) -> cp.ndarray:
        """
        Return an (n_points, 2) array of [lon, lat] pairs.
        """
        raise NotImplementedError
