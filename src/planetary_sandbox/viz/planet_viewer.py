import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import hsv_to_rgb

from ..planet import Planet
from ..numerics import (
    LatLonSphericalHarmonics,
    SpectralOperators,
    GridGeometryBase,
    LatLonGridGeometry,
    geodesic_to_latlon_grid,
)

class PlanetViewer:
    def __init__(self, planet: Planet):
        self.planet = planet
        self.grid: GridGeometryBase = planet.grid
        # Precompute triangulation for geodesic grids; None for structured lat/lon
        self._tri = None
        if hasattr(self.grid, "faces"):
            lon = cp.asnumpy(self.grid.longitudes)
            lat = cp.asnumpy(self.grid.latitudes)
            faces = cp.asnumpy(getattr(self.grid, "faces"))
            if lon.ndim == 1 and lat.ndim == 1 and faces.ndim == 2:
                self._tri = mtri.Triangulation(
                    lon * 180 / np.pi,
                    lat * 180 / np.pi,
                    triangles=faces
                )


    def plot_summary(self):
        """Create a comprehensive visualization of the planet."""


        view_grid = None
        if hasattr(self.grid, "faces"):
            view_grid = LatLonGridGeometry.create((91, 181))

        def _map_to_view(values):
            if view_grid is None:
                return self.grid, values
            if isinstance(values, cp.ndarray):
                values = cp.asnumpy(values)
            mapped = geodesic_to_latlon_grid(values, self.grid, view_grid, method="linear")
            return view_grid, mapped

        fig = plt.figure(figsize=(16, 10))

        print(f"{self.planet.grid=}")
        print(f"{hasattr(self.planet.grid, 'faces')=}")

        # 1. Surface elevation map
        ax1 = plt.subplot(3, 3, 1)
        surface_grid, surface_vals = _map_to_view(self.planet.elevation.surface_height)
        self.plot_scalar(q=surface_vals, # / 1000,
                         grid=surface_grid,
                         title=f"Surface Elevation (km)",
                         cmap='terrain',
                         ax=ax1)

        # 2. Power spectrum
        if self.planet.elevation.power_spectrum is not None:
            # Ensure power_spectrum is a numpy array for matplotlib
            power_spectrum_np = self.planet.elevation.power_spectrum
            if isinstance(power_spectrum_np, cp.ndarray):
                power_spectrum_np = cp.asnumpy(power_spectrum_np)

            ax2 = plt.subplot(3, 3, 2)
            degrees = np.arange(len(power_spectrum_np)) # Use the NumPy version here
            ax2.loglog(degrees[1:], power_spectrum_np[1:]) # Use the NumPy version here
            ax2.set_xlabel('Spherical Harmonic Degree l')
            ax2.set_ylabel('Power')
            ax2.set_title('Elevation Power Spectrum')
            ax2.grid(True, alpha=0.3)

        # 3. Radial distance (shows oblateness)
        ax3 = plt.subplot(3, 3, 3)
        radial_deviation = (self.planet.elevation.radial_distance -
                           self.planet.params.radius)
        radial_grid, radial_vals = _map_to_view(radial_deviation / 1000)
        self.plot_scalar(radial_vals,
                         grid=radial_grid,
                         title="Radial Distance Deviation (km)", 
                         ax=ax3)

        # 4. Low-degree reconstruction
        l_max_filt = 8
        m_max_filt = 2
        if self.planet.elevation.sh_coeffs is not None:
            ax4 = plt.subplot(3, 3, 4)
            low_deg_surface = self.planet.reconstruct_surface(l_max_filt=l_max_filt,
                                                              m_max_filt=m_max_filt)
            low_grid, low_vals = _map_to_view(low_deg_surface / 1000)
            self.plot_scalar(low_vals,
                            title=f"Reconstructed (l≤{l_max_filt}, m≤{m_max_filt})",
                            grid=low_grid,
                            cmap='terrain',
                            ax=ax4)

        # 5. Statistics text
        ax5 = plt.subplot(3, 3, 5)
        ax5.axis('off')
        stats_text = f"""
        Planet Statistics:
        ------------------
        Mass: {self.planet.params.mass/5.972e24:.3f} Mearth
        Equatorial Radius: {self.planet.params.equatorial_radius/1e3:.0f} km
        Polar Radius: {self.planet.params.polar_radius/1e3:.0f} km
        Sidereal Day: {self.planet.params.sidereal_day/3600:.2f} hours
        Angular Velocity: {self.planet.params.angular_velocity*1e3:.3f} mrad/s

        Oblateness (theoretical): {self.planet.params.oblateness:.6f}
        J2 (from SH): {self.planet.elevation.oblateness_from_sh:.6e}

        Elevation Range: [{self.planet.elevation.surface_height.min()/1000:.1f},
                          {self.planet.elevation.surface_height.max()/1000:.1f}] km

        Grid points: {getattr(self.grid, 'n_vertices', 'N/A')}
        Grid faces: {getattr(self.grid, 'n_faces', 'N/A')}
        SH Degrees: {self.planet.elevation.max_degree}
        """
        ax5.text(0.1, 0.5, stats_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center')

        # 6. Low-degree reconstruction of radial distance
        l_max_filt = 2
        m_max_filt = 1

        ax6 = plt.subplot(3, 3, 6)

        radial_deviation = (self.planet.elevation.radial_distance -
                           self.planet.params.radius)
        sh_coeffs_rad_dev = self.planet.sh.transform(radial_deviation)
        low_deg_radial_deviation = self.planet.sh.inv_transform(
            sh_coeffs_rad_dev[:l_max_filt+1, :m_max_filt+1])

        low_radial_grid, low_radial_vals = _map_to_view(low_deg_radial_deviation / 1000)
        self.plot_scalar(low_radial_vals,
                         low_radial_grid,
                         title=f"Radial Deviation (km) (l≤{l_max_filt}, m≤{m_max_filt})",
                         ax=ax6)

        # 7. Coefficient magnitude visualization
        ax7 = plt.subplot(3, 3, 7)
        self.plot_coefficient_complex_visualization(self.planet.elevation.sh_coeffs, ax7, fig)

        # 9. More coefficient magnitude visualization
        ax9 = plt.subplot(3, 3, 9)
        self.plot_coefficient_complex_visualization(sh_coeffs_rad_dev, ax9, fig)


        plt.tight_layout()
        return fig


    @staticmethod
    def plot_coefficient_complex_visualization(coeffs_complex, ax1, fig):
        """
        Visualize SH coefficients with magnitude as value and phase as hue.
        """

        max_l = min(coeffs_complex.shape[0]-1, 30)

        coeffs_complex = cp.roll(coeffs_complex, 0, axis=0).get()

        magnitude = np.abs(coeffs_complex)
        totalpower = np.sum(coeffs_complex * coeffs_complex.conj())
        cum_pwr_covered = np.cumsum(np.sum(magnitude ** 2, axis=1)) / totalpower
        l_99 = np.where(cum_pwr_covered >= 0.99)[0][0]
        coeffs_complex = coeffs_complex[:l_99+1, :l_99+1]
        max_l = l_99

        # Compute magnitude and phase
        magnitude = np.abs(coeffs_complex)
        phase = np.angle(-coeffs_complex)  # Returns [-π, π]

        # Normalize magnitude for brightness (log scale)
        # mag_normalized = np.log10(magnitude + 1e-12)
        mag_normalized = magnitude / magnitude.max()

        # Create HSV image
        # H: phase mapped to [0, 1] (full color wheel)
        # S: constant at 1.0 (fully saturated) where magnitude > threshold, 0 otherwise
        # V: normalized magnitude

        hue = (phase + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
        saturation = mag_normalized
        value = np.ones(hue.shape) # np.where(magnitude > 1e-10, 0.9, 0.0)  # Desaturate near-zero coeffs

        # Stack into HSV image
        hsv_image = np.dstack([hue, saturation, value])

        # Convert to RGB
        rgb_image = hsv_to_rgb(hsv_image)

        # # Create figure
        # fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        # Plot 1: Complex coefficient visualization
        im1 = ax1.imshow(rgb_image, aspect='equal', origin='lower',
                         extent=[0, l_99, 0, l_99])
                        #extent=[-max_l, max_l, 0, max_l])
        ax1.set_xlabel('Order m', fontsize=12)
        ax1.set_ylabel('Degree l', fontsize=12)
        ax1.set_title('SH Coefficients: Sat=Phase, Brightness=|C_lm|', fontsize=14)
        ax1.grid(True, alpha=0.2, color='white', linewidth=0.5)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_scalar(q: np.ndarray |  cp.ndarray,
                    grid: GridGeometryBase,
                    title="Field", cmap="RdBu_r", ax=None):
        
        # Precompute triangulation for geodesic grids; None for structured lat/lon
        triangulation = None
        _tri = q.ndim==1
        if hasattr(grid, "faces"):
            lon = cp.asnumpy(grid.longitudes)
            lat = cp.asnumpy(grid.latitudes)
            faces = np.asarray(getattr(grid, "faces"))
            if lon.ndim == 1 and lat.ndim == 1 and faces.ndim == 2:
                triangulation = mtri.Triangulation(
                                            lon * 180 / np.pi,
                                            lat * 180 / np.pi,
                                            triangles=faces
                )

        if isinstance(q, cp.ndarray):
            q = cp.asnumpy(q)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # If using the geodesic grid with per-vertex data, render via triangulation
        if triangulation is not None and q.ndim == 1 and q.size == triangulation.x.size:
            im = ax.tripcolor(triangulation, q, shading='gouraud', cmap=cmap)
        
        else:
            # Assume structured grid
            im = ax.imshow(np.flip(q, axis=0),
                        extent=(grid.longitudes[0] * 180/np.pi,
                                grid.longitudes[-1] * 180/np.pi,
                                -grid.latitudes[-1] * 180/np.pi,
                                -grid.latitudes[0] * 180/np.pi),
                                cmap=cmap, origin='lower')

        ax.set_aspect('equal')            
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05, aspect=30)
        
        return ax
