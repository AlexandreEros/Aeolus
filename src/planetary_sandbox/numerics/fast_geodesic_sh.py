import cupy as cp
from .cuda.cuda_utils import raw_module_from_cuda


class PointSetSphericalHarmonics:
    """
    Optimized spherical harmonics using matrix multiplication for arbitrary point sets.
    Works on any sampling pattern; pass quadrature weights for non-uniform grids.

    Uses complex spherical harmonics convention with m >= 0 basis functions.
    """
    def __init__(self, latitudes, longitudes, l_max, weights=None):
        if not cp.is_available():
            raise ImportError("CuPy is required.")

        self.l_max = l_max
        self.n_points = len(latitudes)

        # Prepare contiguous buffers for the raw kernel indexing.
        self.latitudes = cp.ascontiguousarray(cp.asarray(latitudes, dtype=cp.float64))
        self.longitudes = cp.ascontiguousarray(cp.asarray(longitudes, dtype=cp.float64))

        assert self.longitudes.shape == self.latitudes.shape, \
            "Latitude and Longitude arrays must have the same shape."
        self.grid_shape = self.latitudes.shape
        assert len(self.grid_shape) == 1, "Input lat/lon must be 1D arrays."

        # Compatibility attributes
        self.phi_grid_gpu = cp.pi/2.0 - self.latitudes
        self.lambda_grid_gpu = self.longitudes

        if weights is None:
            # Default to uniform area approximation (equal-area sampling only).
            self.weights = cp.full(self.n_points, 4.0 * cp.pi / self.n_points, dtype=cp.float64)
        else:
            self.weights = cp.ascontiguousarray(cp.asarray(weights, dtype=cp.float64))

        # Compile Kernel
        self.module = raw_module_from_cuda("sh_matrix")
        self.kernel = self.module.get_function('generate_sph_harm_basis')

        # Generate Basis Matrix Y
        # Shape: (n_points, n_basis)
        self.n_basis = (l_max + 1) * (l_max + 2) // 2
        self.Y_matrix = cp.zeros((self.n_points, self.n_basis), dtype=cp.complex128)

        block_size = 256
        grid_size = (self.n_points + block_size - 1) // block_size

        self.kernel((grid_size,), (block_size,),
                    (self.latitudes, self.longitudes, self.Y_matrix, self.n_points, l_max))

        self.Y_matrix_real = self.Y_matrix.real

        # Indices mapper for (l, m) -> linear index
        self.l_indices = []
        self.m_indices = []
        for l in range(l_max + 1):
            for m in range(0, l + 1):
                self.l_indices.append(l)
                self.m_indices.append(m)
        self.l_indices = cp.array(self.l_indices, dtype=cp.int32)
        self.m_indices = cp.array(self.m_indices, dtype=cp.int32)

    def transform(self, values: cp.ndarray) -> cp.ndarray:
        """
        Forward Transform: f -> coeffs (complex spherical harmonics)

        Vectorized: a = Y^H @ (f * w)
        """
        if not isinstance(values, cp.ndarray):
            values = cp.asarray(values)

        if values.ndim > 1:
            values = values.ravel()

        weighted_vals = values * self.weights
        coeffs_flat = cp.dot(self.Y_matrix.conj().T, weighted_vals)

        # Unpack to 2D grid (l, m)
        coeffs_grid = cp.zeros((self.l_max + 1, self.l_max + 1), dtype=cp.complex128)
        coeffs_grid[self.l_indices, self.m_indices] = coeffs_flat

        return coeffs_grid

    def inv_transform(self, coeffs_grid: cp.ndarray) -> cp.ndarray:
        """
        Inverse Transform: coeffs -> f (real field from m >= 0 complex coefficients)

        For a real field, coefficients satisfy conjugate symmetry.
        Reconstruct using the m >= 0 layout:
            f = 2 * Re(Y @ a) - Re(Y @ a_m0)
        """
        coeffs_flat = coeffs_grid[self.l_indices, self.m_indices]
        term_sum = cp.dot(self.Y_matrix, coeffs_flat)
        mask_m0 = (self.m_indices == 0)
        coeffs_only_m0 = coeffs_flat * mask_m0
        term_m0 = cp.dot(self.Y_matrix, coeffs_only_m0)
        result = 2.0 * term_sum.real - term_m0.real

        if self.grid_shape is not None:
            return result.reshape(self.grid_shape)

        return result

    def inverse_transform(self, coeffs):
        """Alias for inv_transform."""
        return self.inv_transform(coeffs)


# Backwards-compatible alias (deprecated).
GeodesicSphericalHarmonics = PointSetSphericalHarmonics
