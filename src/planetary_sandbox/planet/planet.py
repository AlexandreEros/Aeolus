import numpy as np
import cupy as cp

from typing import Optional, Tuple
from ..numerics import SpectralOperators, GeodesicSphericalHarmonics, GridGeometryBase, GeodesicGridGeometry
from .planetary_parameters import PlanetaryParameters
from .elevation_data import ElevationData
from .terrain_spectral import generate_spectral_terrain_gpu, SpectralTerrainParams
from .tectonics import TectonicParams, tectonic_update_step

class Planet:
    """
    A procedurally generated planet with realistic physical properties.

    This class separates concerns:
    - Physical parameters (mass, rotation, etc.)
    - Grid geometry
    - Elevation data
    - Noise generation
    """

    def __init__(self,
                 params: PlanetaryParameters,
                 grid: GeodesicGridGeometry,
                 elevation: ElevationData,
                 sh: GeodesicSphericalHarmonics,
                 so: SpectralOperators):

        self.params = params
        self.grid = grid
        self.elevation = elevation
        self.sh = sh
        self.so = so

    def __repr__(self):
        return (f"Planet(M={self.params.mass/5.972e24:.2f}M⊕, "
                f"R={self.params.equatorial_radius/1e3:.0f}km, "
                f"T={self.params.sidereal_day/3600:.1f}h, "
                f"f={self.params.oblateness:.5f}, "
                f"grid=GridGeometry(n_points={self.grid.n_points})")

    # @classmethod
    # def generate(cls,
    #              params: PlanetaryParameters = PlanetaryParameters.from_earth_like(),
    #             #  grid_resolution: Tuple[int, int] = (512, 1024),
    #              grid_resolution: int = 3,
    #              l_max: int = 15,
    #              seed: int = 1) -> 'Planet':
    #     """
    #     Generate a complete planet with procedural terrain.

    #     Parameters:
    #     -----------
    #     params : PlanetaryParameters
    #         Physical parameters
    #     grid_resolution : int
    #         Resolution of the geodesic grid (default: 3)
    #     noise_params : NoiseParameters, optional
    #         Noise generation parameters
    #     compute_sh : bool
    #         Whether to compute spherical harmonic decomposition
    #     l_max : int, optional
    #         Maximum degree for SH (default: num_lat - 1)
    #     """

    #     # Create grid
    #     grid = GridGeometry(resolution=grid_resolution, radius=params.equatorial_radius)
    #     # grid = GridGeometry.create(grid_resolution)
    #     # num_lat, num_lon = grid.num_lat, grid.num_lon


    #     # Create spherical harmonics and decompose
    #     print("Computing spherical harmonic decomposition on GPU...")

    #     sh = SphericalHarmonics(grid, l_max=l_max)
    #     # sh.set_grid(grid.longitudes, grid.colatitudes)

    #     so = SpectralOperators(sh, params.radius)


    #     # if noise_params is None:
    #     #     noise_params = NoiseParameters()
    #     # # Generate terrain using noise
    #     # print("Generating terrain...")
    #     # surface_height = cls._generate_terrain_noise(
    #     #     params, grid, noise_params
    #     # )
    #     noise_params = SpectralTerrainParams(
    #         rms_elevation=4000.0,
    #         spectral_exponent=3.5,
    #         seed=seed,
    #         l_min=1,
    #     )
    #     # height, height_coeffs = generate_spectral_terrain_gpu(
    #     #     sph=sh,
    #     #     lambda_grid=grid.lon_grid,
    #     #     phi_grid=grid.lat_grid,
    #     #     params=noise_params
    #     # )
    #     height, height_coeffs = generate_spectral_terrain_gpu(
    #         sph=sh,
    #         grid=grid,
    #         params=noise_params
    #     )

    #     # Calculate radial distances (includes oblateness)
    #     radial_distance = cls._calculate_radial_distances(
    #         params, grid, grid.radial_distances
    #     )


    #     tect_params = TectonicParams(
    #         dt=2.0,
    #         kappa_height=1e-5,
    #         kappa_strain=5e-6,
    #         noise_strength=5.0,
    #         l_cut_noise=8,
    #         gamma_activity=4.0,
    #     )
    #     strain_coeffs = cp.zeros_like(height_coeffs)
    #     for step in range(30):
    #         height_coeffs, strain_coeffs = tectonic_update_step(
    #             sph=sh,
    #             so=so,
    #             h_lm=height_coeffs,
    #             S_lm=strain_coeffs,
    #             params=tect_params
    #         )

    #     # Final terrain back to CPU grid:
    #     height = sh.inv_transform(height_coeffs)

    #     # Create elevation data structure
    #     elevation = ElevationData(
    #         surface_height=height,
    #         radial_distance=radial_distance,
    #         sh_coeffs=height_coeffs,
    #         power_spectrum = cls._compute_power_spectrum(
    #             height_coeffs,
    #             sh.l_max
    #         )
    #     )

    #     so = SpectralOperators(sh, params.radius)

    #     pln = cls(params, grid, elevation, sh, so)

    #     print(f"Spherical harmonics computed up to degree {sh.l_max}")
    #     print(f"Theoretical oblateness = {params.oblateness:.6f}")
    #     print(f"SH-derived oblateness = {elevation.oblateness_from_sh:.6f}")

    #     return pln
    
    @classmethod
    def generate(cls,
                 params: PlanetaryParameters,
                 grid_resolution: int = 3,
                 terrain_params: Optional[SpectralTerrainParams] = None,
                 tectonic_params: Optional[TectonicParams] = None,
                 l_max: int = 15,
                 product_quadrature: str = "fine") -> 'Planet':
        """
        product_quadrature : {"fine", "coarse"}
            Passed to SpectralOperators. "fine" (default) evaluates nonlinear
            products on a reusable resolution-(grid_resolution+1) product grid
            ("overresolved product quadrature", KNOWN_RISKS.md R-3);
            "coarse" retains the historical state-grid product path for
            A/B comparisons.
        """
        if terrain_params is None:
            terrain_params = SpectralTerrainParams(
                rms_elevation=params.radius * 0.001,  # 0.1% of radius
            )
        if tectonic_params is None:
            tectonic_params = TectonicParams()

        # Create grid
        grid = GeodesicGridGeometry(grid_resolution, params.radius)

        sh = GeodesicSphericalHarmonics(grid, l_max)
        so = SpectralOperators(sh, params.radius, grid,
                               product_quadrature=product_quadrature)

        height_coeffs = generate_spectral_terrain_gpu(
            sph=sh,
            params=terrain_params
        )

        tect_params = TectonicParams(
            dt=2.0,
            kappa_height=1e-5,
            kappa_strain=5e-6,
            noise_strength=5.0,
            l_cut_noise=8,
            gamma_activity=4.0,
        )
        # strain_coeffs = cp.zeros_like(height_coeffs)
        # for step in range(20):
        #     height_coeffs, strain_coeffs = tectonic_update_step(
        #         sph=sh,
        #         so=so,
        #         h_lm=height_coeffs,
        #         S_lm=strain_coeffs,
        #         params=tect_params
        #     )

        # Final terrain back to CPU grid:
        height = sh.inv_transform(height_coeffs)

        # Calculate radial distances (includes oblateness)
        radial_distance = cls._calculate_radial_distances(
            params, grid, height
        )

        # Create elevation data structure
        elevation = ElevationData(
            surface_height=height,
            radial_distance=radial_distance,
            sh_coeffs=height_coeffs,
            power_spectrum = cls._compute_power_spectrum(
                height_coeffs,
                sh.l_max
            )
        )

        # Calculate radial distances
        radial_distance = cls._calculate_radial_distances(params, grid, height)

        print("Computing spherical harmonic decomposition (Fast Matrix Method)...")

        # Weights shape must match flattened grid
        # sin_colat = cp.sin(grid.colatitudes)
        # Broadcast to 2D grid (num_lat, num_lon)
        # weights_2d = sin_colat[:, None] * dlat * dlon
        # # Normalize to sum to 4pi? Actually SH orthogonality expects exact integration.
        # # Riemann sum approximation is decent.
        # weights_flat = weights_2d.ravel()

        pln = cls(params, grid, elevation, sh, so)

        print(f"Spherical harmonics computed up to degree {sh.l_max}")

        return pln
    

    @staticmethod
    def _calculate_radial_distances(params: PlanetaryParameters,
                                    grid: GridGeometryBase,
                                    surface_height: np.ndarray) -> np.ndarray:
        """
        Calculate distance from planet center including oblateness and terrain.
        """
        R_eq = params.equatorial_radius
        f = params.oblateness

        # Reference oblate spheroid radius at each latitude
        # r(lat) = R_eq * sqrt(cos²(lat) + (1-f)² * sin²(lat))
        cos_lat = np.cos(grid.latitudes)
        sin_lat = np.sin(grid.latitudes)

        reference_radius = R_eq * np.sqrt(
            cos_lat**2 + (1 - f)**2 * sin_lat**2
            )

        # Total radial distance
        radial_distance = reference_radius + surface_height

        return radial_distance

    @staticmethod
    def _compute_power_spectrum(coeffs: cp.ndarray, l_max: int) -> np.ndarray:
        """Compute power spectrum from spherical harmonic coefficients
        stored for m >= 0 only."""
        if coeffs is None:
            return np.array([])

        n_l, n_m = coeffs.shape
        l_max = min(l_max, n_l - 1)

        # m = 0 term
        power = cp.real(coeffs[:, 0] * cp.conj(coeffs[:, 0]))

        # m > 0 terms: 2 * sum |a_l^m|^2
        if n_m > 1:
            pos_m = coeffs[:, 1:]
            power += 2.0 * cp.real((pos_m * cp.conj(pos_m)).sum(axis=1))

        # Bring back to CPU and truncate to requested l_max
        return cp.asnumpy(power[:l_max + 1])


    def reconstruct_surface(self,
                            l_max_filt: Optional[int] = None,
                            m_max_filt: Optional[int] = None
                            ) -> np.ndarray:
        """
        Reconstruct function from spherical harmonic coefficients.
        """
        if self.sh is None or self.elevation.sh_coeffs is None:
            raise ValueError("Spherical harmonics not computed")

        coeffs = self.elevation.sh_coeffs

        coeffs = coeffs.copy()
        if l_max_filt is not None:
            # Keep only degrees 0..l_max_filt
            l_max_filt = min(l_max_filt, coeffs.shape[0] - 1)
            coeffs[l_max_filt + 1:, :] = 0.0

        if m_max_filt is not None:
            # Keep only orders 0...m_max_filt
            m_max_filt = min(m_max_filt, coeffs.shape[1] - 1)
            coeffs[:, m_max_filt + 1:] = 0.0

        reconstructed = self.sh.inv_transform(coeffs)

        return reconstructed
