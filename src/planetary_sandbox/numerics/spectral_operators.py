import cupy as cp

from .differential_operators_spherical import DifferentialOperatorsSpherical
from .geodesic_grid import GeodesicGridGeometry
# from .spherical_harmonics import LatLonSphericalHarmonics as SphericalHarmonics
from .fast_geodesic_sh import PointSetSphericalHarmonics

# class SpectralOperators:
#     def __init__(self, sh, radius: float):
#         self.sh: PointSetSphericalHarmonics = sh
#         self.R = radius
        
#         self.l_max = sh.l_max
#         self._precompute_zonal_derivative()
#         self._precompute_laplacian()
#         self._precompute_meridional_operator()

#     def _precompute_zonal_derivative(self):
#         m = cp.arange(0, self.l_max + 1, dtype=cp.float64)[None, :]
#         self.im_m = 1j * m

#     def _precompute_laplacian(self):
#         l = cp.arange(0, self.l_max + 1, dtype=cp.float64)
#         self.lap_eigs = -l * (l + 1.0) / (self.R**2)

#     def _precompute_meridional_operator(self):
#         L = cp.arange(0, self.l_max + 1, dtype=cp.float64)[:, None]
#         M = cp.arange(0, self.l_max + 1, dtype=cp.float64)[None, :]
#         Lb = cp.broadcast_to(L, (self.l_max + 1, self.l_max + 1))
#         Mb = cp.broadcast_to(M, (self.l_max + 1, self.l_max + 1))
#         valid = Mb <= Lb
#         num_plus = (Lb + 1.0)**2 - Mb**2
#         den_plus = (2*Lb + 1.0) * (2*Lb + 3.0)
#         C_plus = Lb * cp.sqrt(cp.maximum(num_plus, 0.0) / cp.maximum(den_plus, 1.0))
#         num_minus = Lb**2 - Mb**2
#         den_minus = (2*Lb - 1.0) * (2*Lb + 1.0)
#         C_minus = -(Lb + 1.0) * cp.sqrt(cp.maximum(num_minus, 0.0) / cp.maximum(den_minus, 1.0))
#         C_plus = cp.where(valid, C_plus, 0.0)
#         C_minus = cp.where(valid, C_minus, 0.0)
#         C_plus[self.l_max, :] = 0.0
#         C_minus[0, :] = 0.0
#         self.C_plus = C_plus.astype(cp.complex128)
#         self.C_minus = C_minus.astype(cp.complex128)

#     def laplacian_coeffs(self, coeffs): return self.lap_eigs[:, None] * coeffs if coeffs.ndim == 2 else self.lap_eigs * coeffs
#     def d_lambda_coeffs(self, coeffs): return (self.im_m * coeffs) / self.R
#     def sin_theta_d_theta_coeffs(self, coeffs):
#         g = cp.zeros_like(coeffs, dtype=cp.complex128)
#         g[1:, :] += self.C_plus[:-1, :] * coeffs[:-1, :]
#         g[:-1, :] += self.C_minus[1:, :] * coeffs[1:, :]
#         return g

class SpectralOperators:
    """
    Spectral differential operators on the sphere, built on top of a
    SphericalHarmonics instance.

    All operators act on spherical-harmonic coefficients a_{l,m} stored
    in a 2D array of shape (l_max+1, l_max+1), where:
        - axis 0 = degree l = 0..l_max
        - axis 1 = order  m = 0..l_max  (entries with m > l are ignored/zero)
    """

    def __init__(self, sh: PointSetSphericalHarmonics, radius: float):
        self.sh = sh
        self.R = float(radius)
        self.l_max = sh.l_max
        self._diff_ops = None
        self._diff_ops_grid_id = None

        # --- Eigenvalues for diagonal spectral operators ---
        l = cp.arange(0, self.l_max + 1, dtype=cp.float64)              # (l,)
        m = cp.arange(0, self.l_max + 1, dtype=cp.float64)[None, :]     # (1, m)

        # ∇² Y_l^m = -l(l+1)/R² * Y_l^m
        self.lap_eigs = -l * (l + 1.0) / self.R**2   # l=0 stays exactly 0.0
        
        # ∂Y_l^m/∂λ = i m / R * Y_l^m  (zonal derivative on sphere of radius R)
        self._im_m_over_R = 1j * m / self.R   # (1, m)
        self._im_m_over_R[0, 0] = 1.0  # avoid division by zero for m=0
        # (will zero out later as needed)

        # Coefficients for sinθ ∂/∂θ coupling l ↔ l±1
        self._C_plus, self._C_minus = self._precompute_sin_theta_d_theta_coeffs()

        # Precompute grid spacings and cos(lat) for Jacobian metric term
        # Assumes sh.lat_grid and sh.num_lat / sh.num_lon exist.
        lat = cp.asarray(self.sh.latitudes, dtype=cp.float64)
        self._dphi = float(lat[1] - lat[0])
        self._dlambda = float(self.sh.longitudes[1] - self.sh.longitudes[0])

        # Metric factor: cos φ (since dA = R² cosφ dφ dλ on the sphere)
        cosphi = cp.cos(lat)
        self._cosphi = cp.maximum(cosphi, 1e-10)  # avoid division by zero at poles

    # ------------------------------------------------------------------
    # Internal: meridional operator coefficients
    # ------------------------------------------------------------------
    def _precompute_sin_theta_d_theta_coeffs(self):
        """
        Precompute C_plus[l,m], C_minus[l,m] such that

            sinθ ∂Y_l^m/∂θ
              = C_plus[l,m]  Y_{l+1}^m
              + C_minus[l,m] Y_{l-1}^m

        for a real / complex Y_l^m basis with 0 ≤ m ≤ l.
        """
        L = cp.arange(0, self.l_max + 1, dtype=cp.float64)[:, None]     # (l,1)
        M = cp.arange(0, self.l_max + 1, dtype=cp.float64)[None, :]     # (1,m)

        Lb = cp.broadcast_to(L, (self.l_max + 1, self.l_max + 1))       # (l,m)
        Mb = cp.broadcast_to(M, (self.l_max + 1, self.l_max + 1))

        # Only m ≤ l are valid Y_l^m modes
        valid = Mb <= Lb

        # C_plus for coupling to l+1
        num_plus = (Lb + 1.0)**2 - Mb**2
        den_plus = (2.0 * Lb + 1.0) * (2.0 * Lb + 3.0)
        C_plus = Lb * cp.sqrt(cp.maximum(num_plus, 0.0) /
                              cp.maximum(den_plus, 1.0))

        # C_minus for coupling to l-1
        num_minus = Lb**2 - Mb**2
        den_minus = (2.0 * Lb - 1.0) * (2.0 * Lb + 1.0)
        C_minus = -(Lb + 1.0) * cp.sqrt(
            cp.maximum(num_minus, 0.0) /
            cp.maximum(den_minus, 1.0)
        )

        # Zero out invalid modes
        C_plus = cp.where(valid, C_plus, 0.0)
        C_minus = cp.where(valid, C_minus, 0.0)

        # Enforce boundaries: no l+1 term at top, no l-1 term at bottom
        C_plus[self.l_max, :] = 0.0   # no l_max+1
        C_minus[0, :] = 0.0           # no l=-1

        return C_plus.astype(cp.complex128), C_minus.astype(cp.complex128)

    # ------------------------------------------------------------------
    # Public spectral operators
    # ------------------------------------------------------------------
    def laplacian_coeffs(self, coeffs: cp.ndarray) -> cp.ndarray:
        """
        Apply the spherical Laplacian ∇² to a scalar field in spectral space.
        """
        lap_eigs = self.lap_eigs[:, None]  # (l,1)
        return lap_eigs * coeffs

    def d_lambda_coeffs(self, coeffs: cp.ndarray) -> cp.ndarray:
        """
        Zonal derivative ∂ψ/∂λ in spectral space.
        """
        im_m_over_R = self._im_m_over_R  # (1,m)
        im_m_over_R[0, 0] = 0.0  # zero out m=0 mode to avoid spurious values
        return self._im_m_over_R * coeffs

    def sin_theta_d_theta_coeffs(self, coeffs: cp.ndarray) -> cp.ndarray:
        """
        Meridional derivative in spectral space, multiplied by sin(θ):
        (g = sinθ ∂ψ/∂θ, where θ is colatitude)
        """
        g = cp.zeros_like(coeffs, dtype=cp.complex128)

        # l -> l+1 contribution (C_plus)
        g[1:, :] += self._C_plus[:-1, :] * coeffs[:-1, :]

        # l -> l-1 contribution (C_minus)
        g[:-1, :] += self._C_minus[1:, :] * coeffs[1:, :]

        return g
    
    # ------------------------------------------------------------------
    # Public inverse spectral operators
    # ------------------------------------------------------------------
    def inv_laplacian(self, coeffs: cp.ndarray) -> cp.ndarray:
        """
        Apply the inverse spherical Laplacian ∇⁻² to a scalar field in spectral space.
        NOTE: the l=0 mode is arbitrarily set to 1 / R**2 to avoid division by zero.

        Parameters
        ----------
        coeffs : cp.ndarray
            Scalar field in spectral space.

        Returns
        -------
        cp.ndarray
            Inverse Laplacian applied to the input coefficients.
        """
        inv_eigs = self.lap_eigs.copy()
        inv_eigs[0] = 1.0 / self.R**2   # arbitrary, only affects l=0
        return coeffs / inv_eigs[:, None]
    
    def inv_d_lambda(self, coeffs: cp.ndarray) -> cp.ndarray:
        """
        Inverse zonal derivative ∂⁻¹/∂λ in spectral space.

        Parameters
        ----------
        coeffs : cp.ndarray
            Scalar field in spectral space.

        Returns
        -------
        cp.ndarray
            Inverse zonal derivative applied to the input coefficients.
        """
        inv_im_m_over_R = 1.0 / self._im_m_over_R   # (1,m)
        inv_im_m_over_R[0, 0] = 0.0  # zero out m=0 mode to avoid spurious values
        return inv_im_m_over_R * coeffs
    

    # Convenience wrappers to go back to grid
    def laplacian_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.laplacian_coeffs(coeffs))

    def d_lambda_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.d_lambda_coeffs(coeffs))

    def sin_theta_d_theta_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.sin_theta_d_theta_coeffs(coeffs))

    # ------------------------------------------------------------------
    # Jacobians (nonlinear, live in grid space)
    # ------------------------------------------------------------------

    def jacobian_spectral(self, a_coeffs, b_coeffs, grid) -> cp.ndarray:
        """
        Compute Jacobian J(A, B) in grid space.

        For geodesic grids, use least-squares spatial derivatives on the point set.
        For lat-lon grids, use an Arakawa energy+enstrophy-conserving stencil
        in (theta, lambda) coordinates.

        A, B are given in spectral space and transformed to grid space first.

        Parameters
        ----------
        a_coeffs : cp.ndarray
            Scalar field A in spectral space
        b_coeffs : cp.ndarray
            Scalar field B in spectral space
        grid : GridGeometry
            Grid geometry for the planet

        Returns
        -------
        jacobian : cp.ndarray
            Jacobian J(A, B) in grid space
        """

        # Transform to grid space (real fields)
        A = self.sh.inv_transform(a_coeffs).real
        B = self.sh.inv_transform(b_coeffs).real

        if isinstance(grid, GeodesicGridGeometry) or hasattr(grid, "adjacency_matrix"):
            diff_ops = self._get_differential_ops(grid)
            grad_a = diff_ops.calculate_gradient(A)
            grad_b = diff_ops.calculate_gradient(B)

            dA_dx = grad_a[..., 0]
            dA_dy = grad_a[..., 1]
            dB_dx = grad_b[..., 0]
            dB_dy = grad_b[..., 1]
            return dA_dx * dB_dy - dA_dy * dB_dx

        # Grid spacing in theta, lambda (radians)
        dtheta = cp.pi / (grid.num_lat - 1)
        dlambda = 2.0 * cp.pi / grid.num_lon

        # Plane Arakawa Jacobian in (theta, lambda)
        J_plane = self._arakawa_J(A, B, dtheta, dlambda)

        # Metric factor: theta = colatitude, so sin(theta) = cos(phi)
        sin_theta = cp.sin(cp.asarray(grid.colat_grid))
        sin_theta = cp.maximum(sin_theta, 1e-10)

        jacobian = J_plane / (self.R**2 * sin_theta)

        return jacobian

    def _get_differential_ops(self, grid: GeodesicGridGeometry) -> DifferentialOperatorsSpherical:
        grid_id = id(grid)
        if self._diff_ops is None or self._diff_ops_grid_id != grid_id:
            self._diff_ops = DifferentialOperatorsSpherical.from_geodesic_grid(grid)
            self._diff_ops_grid_id = grid_id
        return self._diff_ops

    def _arakawa_J(self, A: cp.ndarray, B: cp.ndarray,
                   dtheta: float, dlambda: float) -> cp.ndarray:
        """
        Arakawa energy+enstrophy-conserving Jacobian on a uniform (θ, λ) grid:

            J(A,B) ≈ (J1 + J2 + J3) / 3

        This is the "plane" Jacobian in coordinate space; metric factors
        (1 / (R^2 cosφ)) are handled outside this function.
        A, B shape: (num_lat, num_lon)
        Periodic in λ; θ is treated with simple wrap (OK-ish near tiny polar caps).
        """

        # Short names
        dx = dlambda
        dy = dtheta

        # Shifts in latitude (axis=0, θ) and longitude (axis=1, λ)
        A_n = cp.roll(A, -1, axis=0)  # i+1
        A_s = cp.roll(A,  1, axis=0)  # i-1
        A_e = cp.roll(A, -1, axis=1)  # j+1
        A_w = cp.roll(A,  1, axis=1)  # j-1

        B_n = cp.roll(B, -1, axis=0)
        B_s = cp.roll(B,  1, axis=0)
        B_e = cp.roll(B, -1, axis=1)
        B_w = cp.roll(B,  1, axis=1)

        # Diagonals
        A_ne = cp.roll(A_n, -1, axis=1)
        A_nw = cp.roll(A_n,  1, axis=1)
        A_se = cp.roll(A_s, -1, axis=1)
        A_sw = cp.roll(A_s,  1, axis=1)

        B_ne = cp.roll(B_n, -1, axis=1)
        B_nw = cp.roll(B_n,  1, axis=1)
        B_se = cp.roll(B_s, -1, axis=1)
        B_sw = cp.roll(B_s,  1, axis=1)

        # --- Arakawa's three forms ---

        # J1: advective form
        J1 = ((A_e - A_w) * (B_n - B_s) -
              (A_n - A_s) * (B_e - B_w)) / (4.0 * dx * dy)

        # J2: flux form
        J2 = (A_n * (B_ne - B_nw) -
              A_s * (B_se - B_sw) -
              A_e * (B_ne - B_se) +
              A_w * (B_nw - B_sw)) / (4.0 * dx * dy)

        # J3: vorticity form
        J3 = (B_n * (A_ne - A_nw) -
              B_s * (A_se - A_sw) -
              B_e * (A_ne - A_se) +
              B_w * (A_nw - A_sw)) / (4.0 * dx * dy)

        return (J1 + J2 + J3) / 3.0
