from __future__ import annotations

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from ..viz.maps import plot_velocity_streamlines
from ..planet import Planet
from ..numerics import LatLonGridGeometry, geodesic_to_latlon_grid

class VorticityViewer:
    @staticmethod
    def _build_summary_text(duration_str, steps_str, circ0, circ1, ke0, ke1,
                            max_z0, max_z1, rms_z0, rms_z1, max_speed0, max_speed1):
        decay_pct = (1 - ke1 / ke0) * 100 if ke0 != 0 else 0.0
        return (
            "\n"
            "Simulation Overview\n"
            "-------------------\n"
            f"Duration: {duration_str}\n"
            f"Snapshots: {steps_str}\n"
            "\n"
            "Total Circulation (Gamma):\n"
            f"Initial: {circ0:+.2e} m^2/s\n"
            f"Final:   {circ1:+.2e} m^2/s\n"
            "\n"
            "Total Kinetic Energy (K):\n"
            f"Initial: {ke0:.2e} J/kg\n"
            f"Final:   {ke1:.2e} J/kg\n"
            f"Decay:   {decay_pct:.1f}%\n"
            "\n"
            "Vorticity (zeta):\n"
            f"Initial Max: {max_z0:.2e} s^-1\n"
            f"Final Max:   {max_z1:.2e} s^-1\n"
            "\n"
            "RMS Vorticity (Enstrophy):\n"
            f"Initial: {rms_z0:.2e}\n"
            f"Final:   {rms_z1:.2e}\n"
            "\n"
            "Max Flow Speed:\n"
            f"Initial: {max_speed0:.2f} m/s\n"
            f"Final:   {max_speed1:.2f} m/s\n"
        )

    def __init__(self, 
                 planet: Planet,
                 scenario: str,
                 vorticity_snapshots: np.ndarray, 
                 times: np.ndarray = np.empty((0,), dtype=np.float32)):
        """
        Visualizer for Barotropic Vorticity Equation results.

        Parameters
        ----------
        planet : Planet 
            Planet object containing grid and spectral operators
        scenario : str
            Initial condition scenario name
        vorticity_snapshots: np.ndarray
            Time series of vorticity snapshots for conservation checks
        times: np.ndarray, optional
            Corresponding times for the snapshots in hours
        """
        self.planet = planet
        self.grid = planet.grid
        self._view_grid = None
        if not isinstance(self.grid, LatLonGridGeometry):
            # Non-equiangular grids (geodesic, Gauss-Legendre lat-lon) render
            # via interpolation onto a uniform view grid — imshow/streamplot
            # need equally spaced axes. Fields stay flat (n_points,).
            self._view_grid = LatLonGridGeometry.create((91, 181))

        assert isinstance(vorticity_snapshots, np.ndarray), "Snapshots must be a numpy array."
        assert isinstance(times, np.ndarray), "Times must be a numpy array."
        if vorticity_snapshots.size == 0:
            self.snapshots = vorticity_snapshots
            self.zeta_init = vorticity_snapshots
            self.zeta_final = vorticity_snapshots
            self.times = times
            return
        if self._view_grid is None:
            assert vorticity_snapshots.ndim == 3, "Snapshots must be a 3D array (time, lat, lon) for structured grids."
        else:
            assert vorticity_snapshots.ndim in (2, 3), "Snapshots must be 3D (time, lat, lon) or 2D (time, n_points) for geodesic grids."

        self.snapshots = vorticity_snapshots
        self.zeta_init = vorticity_snapshots[0]
        self.zeta_final = vorticity_snapshots[-1]
        self.times = times

        # If the snapshots carry a duplicate 2pi longitude column, drop it to keep
        # the data consistent with the periodic grid used by the spectral transforms.
        expected_nlon = getattr(self.planet.sh, "num_lon", None)
        if self.zeta_init.ndim == 2 and expected_nlon is not None and self.zeta_init.shape[1] == expected_nlon + 1:
            if np.allclose(self.zeta_init[:, 0], self.zeta_init[:, -1], atol=1e-12):
                self.snapshots = self.snapshots[..., :-1]
                self.zeta_init = self.zeta_init[:, :-1]
                self.zeta_final = self.zeta_final[:, :-1]

        # Compute streamfunctions and velocities
        # Assume zeta are on grid
        zeta_init_coeffs = planet.sh.transform(self.zeta_init)
        zeta_final_coeffs = planet.sh.transform(self.zeta_final)

        psi_init_coeffs = planet.so.inv_laplacian(zeta_init_coeffs)
        psi_final_coeffs = planet.so.inv_laplacian(zeta_final_coeffs)

        self.psi_init = planet.sh.inv_transform(psi_init_coeffs)
        self.psi_final = planet.sh.inv_transform(psi_final_coeffs)

        # Compute velocities
        # self.vel_init = bve.streamfunction_to_velocity(psi_init_coeffs)
        # self.vel_final = bve.streamfunction_to_velocity(psi_final_coeffs)
        self.vel_init = self.planet.so.velocity_from_streamfunction(psi_init_coeffs)
        self.vel_final = self.planet.so.velocity_from_streamfunction(psi_final_coeffs)


    def _map_scalar_to_view(self, values: np.ndarray):
        if self._view_grid is None:
            return self.grid, values
        if values.ndim == 2 and values.shape == self._view_grid.lat_grid.shape:
            return self._view_grid, values
        if isinstance(values, cp.ndarray):
            values = cp.asnumpy(values)
        mapped = geodesic_to_latlon_grid(values, self.grid, self._view_grid, method="linear")
        if np.isnan(mapped).any():
            mapped_nearest = geodesic_to_latlon_grid(values, self.grid, self._view_grid, method="nearest")
            mapped = np.where(np.isnan(mapped), mapped_nearest, mapped)
        return self._view_grid, mapped

    def _map_vector_to_view(self, u: np.ndarray, v: np.ndarray):
        if self._view_grid is None:
            return self.grid, (u, v)
        if isinstance(u, cp.ndarray):
            u = cp.asnumpy(u)
        if isinstance(v, cp.ndarray):
            v = cp.asnumpy(v)
        if u.ndim == 2 and u.shape == self._view_grid.lat_grid.shape:
            return self._view_grid, (u, v)
        u_grid = geodesic_to_latlon_grid(u, self.grid, self._view_grid, method="linear")
        v_grid = geodesic_to_latlon_grid(v, self.grid, self._view_grid, method="linear")
        if np.isnan(u_grid).any() or np.isnan(v_grid).any():
            u_nearest = geodesic_to_latlon_grid(u, self.grid, self._view_grid, method="nearest")
            v_nearest = geodesic_to_latlon_grid(v, self.grid, self._view_grid, method="nearest")
            u_grid = np.where(np.isnan(u_grid), u_nearest, u_grid)
            v_grid = np.where(np.isnan(v_grid), v_nearest, v_grid)
        return self._view_grid, (u_grid, v_grid)


    def plot_all_snapshots(self, scenario="snapshots", out_dir=None, metadata=None):
        """
        Create individual plots for each snapshot to diagnose time evolution.

        Parameters
        ----------
        out_dir : pathlib.Path, optional
            Directory to save snapshot plots. If None, uses current directory.
        """
        import pathlib
        if out_dir is None:
            out_dir = pathlib.Path(".")

        from ..run.bve.barotropic_vorticity import BarotropicVorticity
        bve = BarotropicVorticity(self.planet)

        nsnap = len(self.snapshots)

        print(f"\nGenerating plots for {nsnap} snapshots...")

        # Create figure with nsnap * 4 subplots
        fig, axes = plt.subplots(
            nsnap,
            4,
            figsize=(24, 6 * nsnap),
            gridspec_kw={"width_ratios": [1, 1, 1, 0.9]},
        )
        if nsnap == 1:
            axes = axes[None, :]

        sample_snap = self.snapshots[0]
        if isinstance(sample_snap, cp.ndarray):
            sample_snap = cp.asnumpy(sample_snap)

        R = self.planet.params.equatorial_radius
        if sample_snap.ndim == 2 and hasattr(self.planet.grid, "latitudes"):
            lats = self.planet.grid.latitudes
            d_lat = np.abs(self.planet.grid.latitudes[1] - self.planet.grid.latitudes[0])
            d_lon = np.abs(self.planet.grid.longitudes[1] - self.planet.grid.longitudes[0])
            weights = np.cos(lats) * d_lat * d_lon * R**2
        elif sample_snap.ndim == 1 and hasattr(self.planet.grid, "cell_areas"):
            weights = cp.asnumpy(self.planet.grid.cell_areas)
        else:
            raise ValueError("Grid shapes do not match.")

        for idx, zeta_snap in enumerate(self.snapshots):
            time_hrs = self.times[idx] if idx < len(self.times) else 0.0

            # Convert to numpy if needed
            if isinstance(zeta_snap, cp.ndarray):
                zeta_snap = cp.asnumpy(zeta_snap)

            # Compute streamfunction and velocity for this snapshot
            zeta_coeffs = self.planet.sh.transform(zeta_snap)
            psi_coeffs = self.planet.so.inv_laplacian(zeta_coeffs)
            psi_grid = self.planet.sh.inv_transform(psi_coeffs)
            # u, v = bve.streamfunction_to_velocity(psi_coeffs)
            u, v = self.planet.so.velocity_from_streamfunction(psi_coeffs)

            # Convert to numpy
            if isinstance(psi_grid, cp.ndarray):
                psi_grid = cp.asnumpy(psi_grid)
            if isinstance(u, cp.ndarray):
                u = cp.asnumpy(u)
            if isinstance(v, cp.ndarray):
                v = cp.asnumpy(v)

            circ = np.sum(zeta_snap * weights)
            ke = 0.5 * np.sum((u**2 + v**2) * weights)
            max_z = np.max(np.abs(zeta_snap))
            rms_z = np.sqrt(np.mean(zeta_snap**2))
            max_speed = np.max(np.sqrt(u**2 + v**2))

            stats_str = (
                "Snapshot Stats\n"
                "----------------\n"
                f"Time: {time_hrs:.1f} h\n"
                f"Circulation: {circ:+.2e} m^2/s\n"
                f"Kinetic Energy: {ke:.2e} J/kg\n"
                f"Max |vorticity|: {max_z:.2e} 1/s\n"
                f"RMS vorticity: {rms_z:.2e}\n"
                f"Max speed: {max_speed:.2f} m/s\n"
            )

            view_grid, zeta_plot = self._map_scalar_to_view(zeta_snap)
            _, psi_plot = self._map_scalar_to_view(psi_grid)
            _, (u_plot, v_plot) = self._map_vector_to_view(u, v)

            # Vorticity
            im0 = axes[idx, 0].imshow(np.flip(zeta_plot, axis=0),
                                 extent=(0, 360, -89, 89),
                                 cmap='RdBu_r',
                                 aspect='equal',
                                 origin='lower')
            axes[idx, 0].set_title(f"Vorticity @ t={time_hrs:.1f}h")
            axes[idx, 0].set_xlabel("Longitude (deg)")
            axes[idx, 0].set_ylabel("Latitude (deg)")
            plt.colorbar(im0, ax=axes[idx, 0], orientation='horizontal', pad=0.1, fraction=0.05, label='s^-1')

            # Streamfunction
            im1 = axes[idx, 1].imshow(np.flip(psi_plot, axis=0),
                                 extent=(0, 360, -89, 89),
                                 cmap='viridis',
                                 aspect='equal',
                                 origin='lower')
            axes[idx, 1].set_title(f"Streamfunction @ t={time_hrs:.1f}h")
            axes[idx, 1].set_xlabel("Longitude (deg)")
            axes[idx, 1].set_ylabel("Latitude (deg)")
            plt.colorbar(im1, ax=axes[idx, 1], orientation='horizontal', pad=0.1, fraction=0.05, label='m^2/s')

            # Velocity streamlines
            axes[idx, 2] = plot_velocity_streamlines((u_plot, v_plot), self.planet, ax=axes[idx, 2],
                                                      title=f"Flow @ t={time_hrs:.1f}h", grid=view_grid)

            ax_stats = axes[idx, 3]
            ax_stats.axis('off')
            ax_stats.text(0.02, 0.5, stats_str, ha='left', va='center',
                          fontfamily='monospace', fontsize=10)

            plt.tight_layout()

        # Save figure
        dt = self.times[1] - self.times[0] if nsnap > 1 else 0.0
        filename = out_dir / f"{scenario}_t{self.times[0]:02.2f}h-{self.times[-1]:02.2f}h-{dt:02.2f}h.png"
        fig.savefig(filename, dpi=200, bbox_inches='tight', metadata=metadata)
        print(f"  Saved: {filename.name}")
        plt.close(fig)

        print("All snapshot plots generated.\n")



    def plot_summary(self):
        """
        Visual summary of the Barotropic Vorticity simulation.
        """
        # Handle Cupy inputs
        if isinstance(self.zeta_init, cp.ndarray): zeta_init = cp.asnumpy(self.zeta_init)
        else: zeta_init = self.zeta_init
        if isinstance(self.zeta_final, cp.ndarray): zeta_final = cp.asnumpy(self.zeta_final)
        else: zeta_final = self.zeta_final
        if isinstance(self.psi_init, cp.ndarray): psi_init = cp.asnumpy(self.psi_init)
        else: psi_init = self.psi_init
        if isinstance(self.psi_final, cp.ndarray): psi_final = cp.asnumpy(self.psi_final)
        else: psi_final = self.psi_final

        view_grid, zeta_init_plot = self._map_scalar_to_view(zeta_init)
        _, zeta_final_plot = self._map_scalar_to_view(zeta_final)
        _, psi_init_plot = self._map_scalar_to_view(psi_init)
        _, psi_final_plot = self._map_scalar_to_view(psi_final)

        u0, v0 = self.vel_init
        u1, v1 = self.vel_final
        if hasattr(u0, 'get'): u0, v0 = u0.get(), v0.get()
        if hasattr(u1, 'get'): u1, v1 = u1.get(), v1.get()

        _, (u0_plot, v0_plot) = self._map_vector_to_view(u0, v0)
        _, (u1_plot, v1_plot) = self._map_vector_to_view(u1, v1)

        fig = plt.figure(figsize=(20, 18))

        # Grid for plots (3 rows, 3 columns)
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 0.6, 1])

        # --- 1. Initial Vorticity (Top Left) ---
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np.flip(zeta_init_plot, axis=0),
                         extent=(0, 360, -89, 89),
                         cmap='RdBu_r',
                         aspect='equal',
                         origin='lower')
        ax1.set_title("Initial Vorticity")
        ax1.set_xlabel("Longitude (deg)")
        ax1.set_ylabel("Latitude (deg)")
        plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, fraction=0.05, aspect=30, label='s^-1')

        # --- 2. Final Vorticity (Top Right) ---
        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(np.flip(zeta_final_plot, axis=0),
                         extent=(0, 360, -89, 89),
                         cmap='RdBu_r',
                         aspect='equal',
                         origin='lower')
        ax2.set_title("Final Vorticity")
        ax2.set_xlabel("Longitude (deg)")
        ax2.set_ylabel("Latitude (deg)")
        plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, fraction=0.05, aspect=30, label='s^-1')

        # --- 3. Initial Flow (Middle Left) ---
        ax3 = fig.add_subplot(gs[1, 0])
        plot_velocity_streamlines((u0_plot, v0_plot), self.planet, ax=ax3, title="Initial Flow", grid=view_grid)

        # --- 4. Final Flow (Middle Right) ---
        ax4 = fig.add_subplot(gs[1, 2])
        plot_velocity_streamlines((u1_plot, v1_plot), self.planet, ax=ax4, title="Final Flow", grid=view_grid)

        # --- 5. Initial Streamfunction (Bottom Left) ---
        ax5 = fig.add_subplot(gs[2, 0])
        im5 = ax5.imshow(np.flip(psi_init_plot, axis=0),
                         extent=(0, 360, -89, 89),
                         cmap='viridis',
                         aspect='equal',
                         origin='lower')
        ax5.set_title("Initial Streamfunction")
        ax5.set_xlabel("Longitude (deg)")
        ax5.set_ylabel("Latitude (deg)")
        plt.colorbar(im5, ax=ax5, orientation='horizontal', pad=0.1, fraction=0.05, aspect=30, label='m^2/s')

        # --- 6. Final Streamfunction (Bottom Right) ---
        ax6 = fig.add_subplot(gs[2, 2])
        im6 = ax6.imshow(np.flip(psi_final_plot, axis=0),
                         extent=(0, 360, -89, 89),
                         cmap='viridis',
                         aspect='equal',
                         origin='lower')
        ax6.set_title("Final Streamfunction")
        ax6.set_xlabel("Longitude (deg)")
        ax6.set_ylabel("Latitude (deg)")
        plt.colorbar(im6, ax=ax6, orientation='horizontal', pad=0.1, fraction=0.05, aspect=30, label='m^2/s')

        # --- 7. Stats (Top Middle) ---
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis('off')

        R = self.planet.params.equatorial_radius
        if zeta_init.ndim == 2 and hasattr(self.planet.grid, "lat_grid"):
            lats = self.planet.grid.latitudes
            d_lat = np.abs(self.planet.grid.latitudes[1] - self.planet.grid.latitudes[0])
            d_lon = np.abs(self.planet.grid.longitudes[1] - self.planet.grid.longitudes[0])
            weights = np.cos(lats) * d_lat * d_lon * R**2
        elif zeta_init.ndim == 1 and hasattr(self.planet.grid, "cell_areas"):
            weights = cp.asnumpy(self.planet.grid.cell_areas)
        else:
            raise ValueError("Grid shapes do not match.")

        circ0 = np.sum(zeta_init * weights)
        circ1 = np.sum(zeta_final * weights)
        ke0 = 0.5 * np.sum((u0**2 + v0**2) * weights)
        ke1 = 0.5 * np.sum((u1**2 + v1**2) * weights)

        max_z0 = np.max(np.abs(zeta_init))
        max_z1 = np.max(np.abs(zeta_final))
        rms_z0 = np.sqrt(np.mean(zeta_init**2))
        rms_z1 = np.sqrt(np.mean(zeta_final**2))
        max_speed0 = np.max(np.sqrt(u0**2 + v0**2))
        max_speed1 = np.max(np.sqrt(u1**2 + v1**2))

        duration_str = f"{self.times[-1]:.1f} hours" if self.times is not None and len(self.times) > 0 else "N/A"
        steps_str = f"{len(self.times)}" if self.times is not None else "N/A"

        stats_str = self._build_summary_text(
            duration_str=duration_str,
            steps_str=steps_str,
            circ0=circ0,
            circ1=circ1,
            ke0=ke0,
            ke1=ke1,
            max_z0=max_z0,
            max_z1=max_z1,
            rms_z0=rms_z0,
            rms_z1=rms_z1,
            max_speed0=max_speed0,
            max_speed1=max_speed1,
        )
        ax_text.text(0.5, 0.5, stats_str, ha='center', va='center',
                     fontfamily='monospace', fontsize=11)

        # --- 8. Time Series Plot (Middle Middle) ---
        if self.snapshots is not None and self.times is not None:
            ax_plot = fig.add_subplot(gs[1, 1])
            snapshots = self.snapshots
            if isinstance(snapshots, cp.ndarray):
                snapshots = cp.asnumpy(snapshots)
            if snapshots.ndim == 3:
                enstrophy = np.sqrt(np.mean(snapshots**2, axis=(1, 2)))
            else:
                enstrophy = np.sqrt(np.mean(snapshots**2, axis=1))
            enstrophy_norm = enstrophy / enstrophy[0]

            ax_plot.plot(self.times, enstrophy_norm, 'k-', linewidth=1.5, label='RMS zeta (norm)')
            ax_plot.set_xlabel('Time (hours)')
            ax_plot.set_ylabel('Normalized Magnitude')
            ax_plot.set_title('Conservation Check')
            ax_plot.grid(True, alpha=0.3, linestyle='--')
            ax_plot.legend(loc='best', fontsize='small')
            y_margin = max(0.001, np.max(np.abs(enstrophy_norm - 1.0)) * 1.2)
            ax_plot.set_ylim(1.0 - y_margin, 1.0 + y_margin)

        plt.tight_layout()

        print("VorticityViewer: Summary plot generated.")
        print(stats_str)

        return fig
