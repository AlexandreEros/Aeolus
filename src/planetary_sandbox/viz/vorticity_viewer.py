from __future__ import annotations

import numpy as np
import cupy as cp

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
                 times: np.ndarray = np.empty((0,), dtype=np.float32),
                 initial_field: np.ndarray | None = None):
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
        initial_field : np.ndarray, optional
            Genuine ``t=0`` vorticity field on the state grid. When given,
            the summary plot compares it against the last stored snapshot
            even if the initial state was not persisted (N=1 case);
            without it, the first stored snapshot is treated as initial.
        """
        self.planet = planet
        self.grid = planet.grid
        self._view_grid = None
        if not isinstance(self.grid, LatLonGridGeometry):
            # Non-equiangular grids (geodesic, Gauss-Legendre lat-lon) render
            # via interpolation onto a uniform view grid â€” imshow/streamplot
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
        self.zeta_init = (initial_field if initial_field is not None
                          else vorticity_snapshots[0])
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
        import matplotlib.pyplot as plt
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

    def summary_spec(self):
        """Return the backend-neutral specification for the BVE summary."""
        from ..run.bve.visualization import build_bve_summary_spec
        return build_bve_summary_spec(self)
