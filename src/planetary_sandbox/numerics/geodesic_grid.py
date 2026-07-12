import numpy as np
import cupy as cp
from scipy import sparse
from scipy.spatial.transform import Rotation

from .cartesian_to_spherical import cartesian_to_spherical
from .grid_base import GridGeometry

normalize = lambda vec: vec / np.linalg.norm(vec, axis=-1)[..., None]

class GeodesicGridGeometry(GridGeometry):
# Create a basic geodesic grid (icosahedron-based)

    # Create the vertices of an icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices: list[list[float]] = (np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]) / np.hypot(1, phi)       ).tolist()

    # Define the icosahedron faces (triangles)
    ico_faces: list[list[int]] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]


    def __init__(self, resolution: int = 0, radius: float = 1.0):
        try:
            self.resolution = resolution
            self.radius = radius

            self.ico_vertices_array = np.array(self.ico_vertices)
            anti_singularity_rotation = Rotation.from_rotvec(np.array([1e-2, 0, 0]))
            self.ico_vertices_array = anti_singularity_rotation.apply(self.ico_vertices_array)
            self.ico_vertices = self.ico_vertices_array.tolist()

            self.mesh = self.geodesic_subdivide()
            self.points = self.radius * self.mesh[0]
            self.faces = self.mesh[1]

            self.n_points = len(self.points)
            self.n_faces = len(self.faces)

            self.min_edge_length = None
            self.adjacency_matrix = self.build_adjacency_matrix()


            self.spherical_coordinates = cp.array(cartesian_to_spherical(self.points))
            self.radial_distances = cp.ascontiguousarray(self.spherical_coordinates[:,0])
            self.longitudes = cp.ascontiguousarray(self.spherical_coordinates[:,1])
            self.latitudes = cp.ascontiguousarray(self.spherical_coordinates[:,2])

            self.sinlat = cp.sin(self.latitudes)
            self.coslat = cp.cos(self.latitudes)
            self.sincolat = cp.sin(cp.pi/2 - self.latitudes)

            self.volume = (4.0/3.0) * np.pi * (self.radius**3) / self.n_points
            self.reference_radius = self.radius
            # TODO: Implement oblateness later

            self._cell_areas = None

        except Exception as err:
            raise Exception(f"Error in the constructor of `GeodesicGrid`:\n{err}")

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
    def n_points(self) -> int:
        return self._n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        self._n_points = int(value)

    @property
    def point_latitudes(self) -> np.ndarray:
        return self.latitudes

    @property
    def point_longitudes(self) -> np.ndarray:
        return self.longitudes


    def geodesic_subdivide(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Subdivide each triangle in the faces list to increase the resolution.
        """
        vertices = [np.array(vertex) for vertex in self.ico_vertices]
        faces = self.ico_faces.copy()

        for _ in range(self.resolution):
            try:
                new_faces = []
                edge_to_midpoint = {}

                # Subdivide each triangle into four smaller triangles
                for tri in faces:
                    try:
                        v1, v2, v3 = tri
                        vm12 = self._get_midpoint_index(vertices, edge_to_midpoint, v1, v2)
                        vm23 = self._get_midpoint_index(vertices, edge_to_midpoint, v2, v3)
                        vm31 = self._get_midpoint_index(vertices, edge_to_midpoint, v3, v1)

                        # Create four new triangles
                        new_faces.extend([
                            [v1, vm12, vm31],
                            [v2, vm23, vm12],
                            [v3, vm31, vm23],
                            [vm12, vm23, vm31]
                        ])
                    except ValueError as err:
                        raise ValueError(f"Error subdividing triangle {tri}: {err}")

                # Replace old faces with new ones
                faces = new_faces

            except Exception as e:
                print(f"An error occurred during geodesic subdivision at depth {_}: {e}")
                raise

        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        return vertices, faces


    @staticmethod
    def _get_midpoint_index(vertices: list[np.ndarray], edge_to_midpoint: dict[tuple[int, int], int],
                            a: int, b: int) -> int:
        edge = (a, b) if a < b else (b, a)
        if edge in edge_to_midpoint:
            return edge_to_midpoint[edge]

        midpoint = normalize((vertices[a] + vertices[b]) / 2.0)
        vertices.append(midpoint)
        idx = len(vertices) - 1
        edge_to_midpoint[edge] = idx
        return idx


    def build_adjacency_matrix(self) -> sparse.coo_matrix:
        """
        Build an adjacency matrix for the geodesic grid using the inverse of the distance between vertices as weights.
        """

        row_indices = []
        col_indices = []
        data = []
        min_edge_length = None

        for face in self.faces:
            # Each face is a tuple of three vertex indices
            v1, v2, v3 = face

            for (a, b) in [(v1, v2), (v2, v3), (v3, v1)]:
                dist = np.linalg.norm(self.points[a] - self.points[b])
                if dist > 0 and (min_edge_length is None or dist < min_edge_length):
                    min_edge_length = dist
                weight = 1.0 / dist if dist > 0 else 0

                row_indices.extend([a, b])
                col_indices.extend([b, a])
                data.extend([weight, weight])

        if min_edge_length is None:
            min_edge_length = 0.0
        self.min_edge_length = float(min_edge_length)

        return sparse.coo_matrix((data, (row_indices, col_indices)), shape=(self.n_points, self.n_points))

    def points_latlon(self) -> np.ndarray:
        return np.column_stack([self.longitudes, self.latitudes])

    @property
    def cfl_length_scale(self) -> float | None:
        """The historical CFL scale for geodesic grids: minimum edge length."""
        return self.min_edge_length

    @property
    def cell_areas(self) -> np.ndarray:
        """
        Compute quadrature weights for spherical harmonic integration.
        Each weight represents the area of the Voronoi cell around each point.

        Returns
        -------
        np.ndarray
            Array of quadrature weights, one per grid point.
        """
        if self._cell_areas is None:
            self._cell_areas = self._compute_voronoi_areas()
        return self._cell_areas

    def _compute_voronoi_areas(self) -> np.ndarray:
        """
        Compute the area of the Voronoi cell around each vertex.
        For a geodesic grid, we approximate this using the dual mesh approach:
        each triangle contributes 1/3 of its area to each of its vertices.

        Returns
        -------
        np.ndarray
            Array of areas, one per grid point.
        """
        areas = np.zeros(self.n_points, dtype=np.float64)

        for face in self.faces:
            # Get the three vertices of this triangle
            v1, v2, v3 = face
            p1, p2, p3 = self.points[v1], self.points[v2], self.points[v3]

            # Compute the area of the spherical triangle using the proper formula
            # Normalize points to unit sphere for area calculation
            a = p1 / np.linalg.norm(p1)
            b = p2 / np.linalg.norm(p2)
            c = p3 / np.linalg.norm(p3)

            # Use the formula: A = E * R^2, where E is the spherical excess
            # Spherical excess using L'Huilier's theorem
            # E = 4 * arctan(sqrt(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2)))
            # where a, b, c are the side lengths and s = (a+b+c)/2

            # Calculate side lengths (great circle distances on unit sphere)
            side_a = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            side_b = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
            side_c = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

            s = (side_a + side_b + side_c) / 2.0

            # L'Huilier's formula for spherical excess
            tan_E_4 = np.sqrt(
                np.tan(s / 2.0) *
                np.tan((s - side_a) / 2.0) *
                np.tan((s - side_b) / 2.0) *
                np.tan((s - side_c) / 2.0)
            )

            spherical_excess = 4.0 * np.arctan(tan_E_4)

            # Area on the actual sphere (scaled by radius^2)
            triangle_area = spherical_excess * (self.radius ** 2)

            # Each vertex gets 1/3 of the triangle's area
            areas[v1] += triangle_area / 3.0
            areas[v2] += triangle_area / 3.0
            areas[v3] += triangle_area / 3.0

        return areas
