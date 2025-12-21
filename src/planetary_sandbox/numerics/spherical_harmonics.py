import numpy as np
import cupy as cp

from .integration import simpson_2d
from .cuda.cuda_utils import raw_module_from_cuda

class LatLonSphericalHarmonics:
    """GPU-accelerated spherical harmonics using custom CUDA kernels (Iterative)."""

    def __init__(self, l_max: int, lon_grid: np.ndarray | None = None, colat_grid: np.ndarray | None = None):
        if not cp.is_available():
            raise ImportError("CuPy is required for CUDA implementation")

        self.l_max = l_max

        # Compile kernels
        self.legendre_module = raw_module_from_cuda("legendre")
        self.legendre_kernel = self.legendre_module.get_function('compute_legendre')

        self.sph_harm_module = raw_module_from_cuda("sph_harm")
        self.sph_harm_kernel = self.sph_harm_module.get_function('compute_sph_harm')

        # Cache for factorials
        self._factorial_cache = cp.ones((2 * self.l_max + 1,), dtype=cp.float64)
        for i in range(1, len(self._factorial_cache)):
            self._factorial_cache[i] = i * self._factorial_cache[i - 1]

        # Normalization factors N_{l,m}
        self._l_range = cp.arange(0, self.l_max + 1, dtype=int)
        self._m_range = cp.concatenate([cp.arange(self.l_max + 1), cp.arange(-self.l_max, 0)])
        _L, _M = cp.meshgrid(self._l_range, self._m_range, indexing='ij')
        self.lm_grid = cp.stack([_L, _M], axis=-1)

        self.normalization_factors = cp.apply_along_axis(
            lambda lm: self._normalization_factor(int(lm[0]), int(lm[1])),
            -1, self.lm_grid
        )

        self._grid_prepared = False
        if lon_grid is not None or colat_grid is not None:
            if lon_grid is None or colat_grid is None:
                raise ValueError("lon_grid and colat_grid must be provided together.")
            self.set_grid(lon_grid, colat_grid)

    def set_grid(self, lon_grid: np.ndarray, colat_grid: np.ndarray) -> None:
        lon = cp.asarray(lon_grid, dtype=cp.float64)
        colat = cp.asarray(colat_grid, dtype=cp.float64)

        if lon.ndim == 2 or colat.ndim == 2:
            if lon.ndim != 2 or colat.ndim != 2 or lon.shape != colat.shape:
                raise ValueError("lon_grid and colat_grid must have the same 2D shape.")

            lon_1d = lon[0, :]
            colat_1d = colat[:, 0]

            if lon.shape[0] > 1 and not cp.allclose(lon, lon_1d[None, :]):
                raise ValueError("lon_grid rows must be identical for meshgrid inputs.")
            if colat.shape[1] > 1 and not cp.allclose(colat, colat_1d[:, None]):
                raise ValueError("colat_grid columns must be identical for meshgrid inputs.")

            lon = lon_1d
            colat = colat_1d
        elif lon.ndim != 1 or colat.ndim != 1:
            raise ValueError("lon_grid and colat_grid must be 1D arrays or 2D meshgrids.")

        self.lambda_grid_gpu = cp.ascontiguousarray(lon)
        self.phi_grid_gpu = cp.ascontiguousarray(colat)
        self.n_lambda = self.lambda_grid_gpu.size
        self.n_phi = self.phi_grid_gpu.size
        self.sin_phi = cp.sin(self.phi_grid_gpu)
        cos_phi = cp.cos(self.phi_grid_gpu)

        # Precompute P_l^m
        P = cp.zeros((self.l_max + 1, self.l_max + 1, self.n_phi), dtype=cp.float64)
        for l in range(self.l_max + 1):
            for m in range(0, l + 1):
                P[l, m, :] = self.compute_legendre(cos_phi, l, m)
        self.P_cache = P

        # Precompute E_m
        m_vals = cp.arange(0, self.l_max + 1, dtype=cp.int32)
        self.E_cache = cp.exp(1j * m_vals[:, None] * self.lambda_grid_gpu[None, :])
        self._grid_prepared = True


    def _factorial(self, n):
        return self._factorial_cache[n]

    def _normalization_factor(self, l, m):
        m_abs = abs(int(m))
        val = ((2 * l + 1) / (4.0 * cp.pi) * self._factorial(l - m_abs) / self._factorial(l + m_abs))
        return cp.sqrt(val)

    def compute_legendre(self, x, l, m):
        n_points = x.size
        P = cp.zeros(n_points, dtype=cp.float64)
        block_size = 256
        grid_size = (n_points + block_size - 1) // block_size
        self.legendre_kernel((grid_size,), (block_size,), (x, P, n_points, l, m))
        return P

    def transform(self, f_values):
        if not self._grid_prepared: raise ValueError("Grid not prepared")
        f_values_gpu = cp.asarray(f_values)
        sin_phi = self.sin_phi[:, None]
        coeffs = cp.zeros((self.l_max + 1, self.l_max + 1), dtype=cp.complex128)

        for l in range(self.l_max + 1):
            for m in range(0, l + 1):
                P_lm = self.P_cache[l, m, :]
                E_m = self.E_cache[m, :]
                N_lm = self.normalization_factors[l, m].real
                
                # FIX: Kernel already applies Condon-Shortley phase (-1)^m.
                # Do NOT apply it again here.
                N_eff = N_lm 
                
                Y_lm = N_eff * P_lm[:, None] * E_m[None, :]
                
                integrand = f_values_gpu * cp.conj(Y_lm) * sin_phi
                coeffs[l, m] = simpson_2d(integrand, self.lambda_grid_gpu, self.phi_grid_gpu)
        return coeffs

    def inv_transform(self, coeffs):
        """
        Inverse Transform from Laplace coefficients to spatial grid

        f = sum( a_lm * Y_lm )

        Parameters
        ----------
        coeffs_grid : cp.ndarray, shape (l_max+1, l_max+1)
            Spectral coefficients (real basis, m>=0 layout)
        Returns
        -------
        f_recon : cp.ndarray, shape (n_lat, n_lon)
            Reconstructed field values on the original grid (2D array of shape (n_lat, n_lon)).
        """
        if not self._grid_prepared: raise ValueError("Grid not prepared")
        f_recon = cp.zeros((self.n_phi, self.n_lambda), dtype=cp.float64)
        for l in range(coeffs.shape[0]):
            # m=0
            P_l0 = self.P_cache[l, 0, :]
            N_l0 = self.normalization_factors[l, 0].real
            Y_l0 = N_l0 * P_l0[:, None]
            f_recon += cp.real(coeffs[l, 0] * Y_l0)
            # m>0
            for m in range(1, coeffs.shape[1]):
                P_lm = self.P_cache[l, m, :]
                E_m = self.E_cache[m, :]
                N_lm = self.normalization_factors[l, m].real
                
                # FIX: Kernel already applies Condon-Shortley phase (-1)^m.
                # Do NOT apply it again here.
                N_eff = N_lm 
                
                Y_lm = N_eff * P_lm[:, None] * E_m[None, :]
                f_recon += 2.0 * cp.real(coeffs[l, m] * Y_lm)
        return f_recon
