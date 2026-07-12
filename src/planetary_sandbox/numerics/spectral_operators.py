import cupy as cp

from .differential_operators_spherical import DifferentialOperatorsSpherical
from .geodesic_grid import GeodesicGridGeometry
# from .spherical_harmonics import LatLonSphericalHarmonics as SphericalHarmonics
from .optimized_geodesic_sh import GeodesicSphericalHarmonics

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

    def __init__(self, sh: GeodesicSphericalHarmonics, radius: float, grid: GeodesicGridGeometry,
                 product_quadrature: str = "coarse", backend=None):
        """
        Parameters
        ----------
        product_quadrature : str
            Where pseudospectral (pointwise) products are evaluated and
            analyzed; the mode set is defined by the backend
            (`backend.supported_product_quadratures()`). "coarse" (default,
            supported by every backend): on the state sampling itself — the
            historical behavior; its quadrature cannot integrate the
            degree-~2·l_max product content, and the resulting aliasing lands
            in the retained band (KNOWN_RISKS.md R-3). "fine" (where
            supported): a backend-chosen overresolved product sampling —
            the geodesic backend uses a resolution-(r+1) co-grid built once
            at initialization ("overresolved product quadrature"). This is
            NOT exact dealiasing; it is a quadrature upgrade whose measured
            effect is a ~6× smaller invariant-production defect at
            res 4 / l_max 21. Unsupported modes raise ValueError (no silent
            fallback).
        backend : SphericalGridBackend, optional
            The geometry/transform pairing that owns product-space policy.
            Inferred from `grid` when omitted (GeodesicBackend for geodesic
            geometries, coarse-only PointSetBackend otherwise).
        """
        from .spherical_backend import make_backend

        self.sh = sh
        self.R = float(radius)
        self.l_max = sh.l_max
        self.grid = grid
        self._diff_ops = None
        self._diff_ops_grid_id = None

        self.backend = backend if backend is not None else make_backend(grid, sh)
        self.product_quadrature = product_quadrature
        # Built once here (never inside the tendency); raises on unsupported
        # modes.
        self._product_space = self.backend.product_space(product_quadrature)
        # Back-compat attributes (used by tests and diagnostics tooling):
        # populated only when a distinct product sampling exists.
        self.product_grid = self._product_space.geometry
        self.product_sh = (self._product_space.sh
                           if self._product_space.geometry is not None else None)

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

        self._dphi = None
        self._dlambda = None
        self._cosphi = None
        # self._init_latlon_metrics()

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

    @staticmethod
    def _is_geodesic_grid(grid) -> bool:
        return grid is not None and (isinstance(grid, GeodesicGridGeometry) or hasattr(grid, "adjacency_matrix"))

    @staticmethod
    def _is_latlon_grid(grid) -> bool:
        return grid is not None and hasattr(grid, "num_lat") and hasattr(grid, "num_lon")

    # def _init_latlon_metrics(self) -> None:
    #     if self._is_geodesic_grid(self.grid):
    #         return

    #     lat = None
    #     lon = None
    #     if self._is_latlon_grid(self.grid):
    #         lat = cp.asarray(self.grid.latitudes, dtype=cp.float64)
    #         lon = cp.asarray(self.grid.longitudes, dtype=cp.float64)
    #     elif hasattr(self.sh, "latitudes") and hasattr(self.sh, "longitudes"):
    #         lat = cp.asarray(self.sh.latitudes, dtype=cp.float64)
    #         lon = cp.asarray(self.sh.longitudes, dtype=cp.float64)

    #     if lat is None or lon is None or lat.size == 0 or lon.size == 0:
    #         return

    #     if lat.size > 1:
    #         self._dphi = float(lat[1] - lat[0])
    #     if lon.size > 1:
    #         self._dlambda = float(lon[1] - lon[0])

    #     cosphi = cp.cos(lat)
    #     self._cosphi = cp.maximum(cosphi, 1e-10)

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
        inv = cp.zeros_like(self._im_m_over_R)
        inv[:, 1:] = 1.0 / self._im_m_over_R[:, 1:]
        return inv * coeffs

    

    # Convenience wrappers to go back to grid
    def laplacian_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.laplacian_coeffs(coeffs))

    def d_lambda_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.d_lambda_coeffs(coeffs))

    def sin_theta_d_theta_field(self, coeffs: cp.ndarray) -> cp.ndarray:
        return self.sh.inv_transform(self.sin_theta_d_theta_coeffs(coeffs))


    def _get_differential_ops(self, grid) -> DifferentialOperatorsSpherical:
        grid_id = id(grid)
        if self._diff_ops is None or self._diff_ops_grid_id != grid_id:
            self._diff_ops = DifferentialOperatorsSpherical.from_geodesic_grid(grid)
            self._diff_ops_grid_id = grid_id
        return self._diff_ops
    

    # ------------------------------------------------------------------
    # Jacobians (nonlinear, live in grid space)
    # ------------------------------------------------------------------


    def velocity_from_streamfunction(self, psi_lm: cp.ndarray):
        """
        Return (u, v) on the SH evaluation grid (u=eastward, v=northward).
        """
        # cosφ from the geometry interface only (no grid-family assumptions;
        # cos(lat) >= 0, identical to the old sqrt(x^2+y^2)/r for unit points).
        lat = cp.asarray(self.grid.point_latitudes, cp.float64)
        coslat = cp.cos(lat)
        coslat_safe = cp.where(cp.abs(coslat) < 1e-6, cp.nan, coslat)

        # spectral derivatives -> grid
        psi_lam_over_R = self.sh.inv_transform(self.d_lambda_coeffs(psi_lm)).real
        gpsi = self.sh.inv_transform(self.sin_theta_d_theta_coeffs(psi_lm)).real

        # u (east) and v (north) in m/s (still has 1/cosφ)
        u = gpsi / (self.R * coslat_safe)
        v = psi_lam_over_R / coslat_safe

        u = cp.nan_to_num(u, nan=0.0)
        v = cp.nan_to_num(v, nan=0.0)

        return u, v

    

    def grad_from_scalar(self, q_lm: cp.ndarray):
        """
        Return the two components of ∇q in physical units:
          dq_dx = (1/(R cosφ)) q_λ
          dq_dy = (1/R) q_φ
        on the SH evaluation grid.
        """

        # (1/R) q_λ
        q_lam_over_R = self.sh.inv_transform(self.d_lambda_coeffs(q_lm)).real

        gq = self.sh.inv_transform(self.sin_theta_d_theta_coeffs(q_lm)).real
        q_phi_over_R = -gq / (self.R * self.grid.coslat)

        dq_dx = q_lam_over_R / self.grid.coslat
        dq_dy = q_phi_over_R
        return dq_dx, dq_dy
    

    def advect_scalar_by_streamfunction(self, 
                                        psi_lm: cp.ndarray, 
                                        q_lm: cp.ndarray, 
                                        dealias: bool = True,
                                        return_spectral: bool = False):
        """
        Compute u·∇q on the SH evaluation grid (no latlon conversion).
        Optionally dealias by filtering after transforming the product.
        """
        u, v = self.velocity_from_streamfunction(psi_lm)
        dq_dx, dq_dy = self.grad_from_scalar(q_lm)

        adv_grid = u * dq_dx + v * dq_dy

        if not (dealias or return_spectral):
            return adv_grid

        adv_lm = self.sh.transform(adv_grid)

        if dealias:
            L = self.l_max
            cut = (2 * L) // 3
            adv_lm[cut+1:, :] = 0.0
            adv_lm[:, cut+1:] = 0.0

        return adv_lm if return_spectral else self.sh.inv_transform(adv_lm).real
    


    def jacobian_pseudospectral(self, a_lm: cp.ndarray, b_lm: cp.ndarray,
                                dealias: bool = True,
                                return_spectral: bool = False) -> cp.ndarray:
        """
        Pseudospectral spherical Jacobian.

            J(a, b) = (1/(R^2 cosφ)) (a_λ b_φ - a_φ b_λ) = u_a · ∇b,
            with u_a = k × ∇a.

        The available derivative fields are
            a_lam   = (1/R) a_λ
            a_sinth = (1/R) sinθ a_θ = -(1/R) cosφ a_φ   (θ = π/2 - φ colatitude)
        so, eliminating the physical φ-derivatives,
            J(a, b) = (a_sinth b_lam - a_lam b_sinth) / cos²φ.

        Product evaluation sampling (see __init__ `product_quadrature`):
        the backend's ProductSpace decides where the derivative coefficient
        fields are evaluated, multiplied pointwise, and analyzed back into
        the same (l_max+1, l_max+1) coefficient layout. With the geodesic
        backend, "fine" is a resolution-(r+1) co-grid ("overresolved product
        quadrature" — a quadrature upgrade, not exact dealiasing); "coarse"
        is the state sampling itself (historical behavior).

        Parameters
        ----------
        dealias : bool
            Apply the 2/3-rule spectral truncation to the analyzed product
            (exactly once). The historical name is kept; the operation is a
            truncation.
        return_spectral : bool
            If True, return the truncated coefficients directly — no
            synthesis/re-analysis round trip. If False (legacy), return a
            field on the *state* grid: for dealias=True this reproduces the
            historical truncate-then-synthesize behavior; combined with an
            external `sh.transform`, that path reproduces the pre-fix
            production tendency exactly (kept for A/B comparisons).
        """
        ps = self._product_space
        sh_p = ps.sh
        coslat = ps.coslat

        # Spectral -> product-grid derivative fields (direct basis evaluation
        # at the product points; no interpolation from the state grid).
        a_lam   = sh_p.inv_transform(self.d_lambda_coeffs(a_lm)).real               # (1/R) A_λ
        b_lam   = sh_p.inv_transform(self.d_lambda_coeffs(b_lm)).real               # (1/R) B_λ
        a_sinth = sh_p.inv_transform(self.sin_theta_d_theta_coeffs(a_lm)).real / self.R  # (1/R) sinθ A_θ
        b_sinth = sh_p.inv_transform(self.sin_theta_d_theta_coeffs(b_lm)).real / self.R  # (1/R) sinθ B_θ

        J_grid = (a_sinth * b_lam - a_lam * b_sinth) / coslat**2

        if not (dealias or return_spectral):
            if sh_p is self.sh:
                return J_grid
            # Fine-path callers asking for a grid field get it on the STATE
            # grid (callers' arrays are state-grid sized); one analysis +
            # synthesis is unavoidable here.
            return self.sh.inv_transform(sh_p.transform(J_grid)).real

        J_lm = sh_p.transform(J_grid)

        if dealias:
            # Spectral truncation (2/3 rule), applied exactly once.
            cut = (2 * self.l_max) // 3
            J_lm[cut + 1:, :] = 0.0
            J_lm[:, cut + 1:] = 0.0

        if return_spectral:
            return J_lm
        return self.sh.inv_transform(J_lm).real
