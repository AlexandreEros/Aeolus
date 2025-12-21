from abc import ABC, abstractmethod
import cupy as cp


class GridGeometry(ABC):
    """
    Abstract base for grid geometries on a sphere.
    Subclasses must provide latitudes/longitudes and point-count access.
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
    def n_points(self) -> int:
        raise NotImplementedError

    @property
    def colatitudes(self) -> cp.ndarray:
        return cp.pi / 2.0 - cp.asarray(self.latitudes)

    @abstractmethod
    def points_latlon(self) -> cp.ndarray:
        """
        Return an (n_points, 2) array of [lon, lat] pairs.
        """
        raise NotImplementedError
