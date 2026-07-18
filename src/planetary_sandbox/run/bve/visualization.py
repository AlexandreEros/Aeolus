"""BVE-specific presentation decisions expressed as generic specifications."""
from __future__ import annotations

import numpy as np

from planetary_sandbox.viz.fields import ScalarGridField
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.specs import (
    FigureSpec, LinePanelSpec, LineSeriesSpec, PanelPlacement, ScalarMapSpec,
    StreamlineMapSpec, TextPanelSpec)


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def build_bve_summary_spec(viewer) -> tuple[FigureSpec, str]:
    """Preserve the historical BVE summary content without backend calls."""
    zeta_init = _host(viewer.zeta_init)
    zeta_final = _host(viewer.zeta_final)
    psi_init = _host(viewer.psi_init)
    psi_final = _host(viewer.psi_final)

    view_grid, zeta_init_plot = viewer._map_scalar_to_view(zeta_init)
    _, zeta_final_plot = viewer._map_scalar_to_view(zeta_final)
    _, psi_init_plot = viewer._map_scalar_to_view(psi_init)
    _, psi_final_plot = viewer._map_scalar_to_view(psi_final)

    u0, v0 = (_host(component) for component in viewer.vel_init)
    u1, v1 = (_host(component) for component in viewer.vel_final)
    _, (u0_plot, v0_plot) = viewer._map_vector_to_view(u0, v0)
    _, (u1_plot, v1_plot) = viewer._map_vector_to_view(u1, v1)

    latitudes = _host(view_grid.latitudes)
    longitudes = _host(view_grid.longitudes)

    def scalar(values, name, units):
        return ScalarGridField(
            _host(values), latitudes, longitudes, name=name, units=units)

    automatic = NormalizationPolicy.automatic()
    panels = [
        PanelPlacement(ScalarMapSpec(
            scalar(zeta_init_plot, "relative vorticity", "s^-1"),
            "Initial Vorticity", normalization=automatic,
            color_policy="RdBu_r"), 0, 0),
        PanelPlacement(ScalarMapSpec(
            scalar(zeta_final_plot, "relative vorticity", "s^-1"),
            "Final Vorticity", normalization=automatic,
            color_policy="RdBu_r"), 0, 2),
        PanelPlacement(StreamlineMapSpec(
            latitudes, longitudes, u0_plot, v0_plot,
            radius=viewer.planet.params.equatorial_radius,
            title="Initial Flow"), 1, 0),
        PanelPlacement(StreamlineMapSpec(
            latitudes, longitudes, u1_plot, v1_plot,
            radius=viewer.planet.params.equatorial_radius,
            title="Final Flow"), 1, 2),
        PanelPlacement(ScalarMapSpec(
            scalar(psi_init_plot, "streamfunction", "m^2/s"),
            "Initial Streamfunction", normalization=automatic,
            color_policy="viridis"), 2, 0),
        PanelPlacement(ScalarMapSpec(
            scalar(psi_final_plot, "streamfunction", "m^2/s"),
            "Final Streamfunction", normalization=automatic,
            color_policy="viridis"), 2, 2),
    ]

    radius = viewer.planet.params.equatorial_radius
    if zeta_init.ndim == 2 and hasattr(viewer.planet.grid, "lat_grid"):
        lats = _host(viewer.planet.grid.latitudes)
        d_lat = abs(lats[1] - lats[0])
        lons = _host(viewer.planet.grid.longitudes)
        d_lon = abs(lons[1] - lons[0])
        weights = np.cos(lats)[:, None] * d_lat * d_lon * radius**2
    elif zeta_init.ndim == 1 and hasattr(viewer.planet.grid, "cell_areas"):
        weights = _host(viewer.planet.grid.cell_areas)
    else:
        raise ValueError("Grid shapes do not match.")

    circ0 = np.sum(zeta_init * weights)
    circ1 = np.sum(zeta_final * weights)
    ke0 = 0.5 * np.sum((u0**2 + v0**2) * weights)
    ke1 = 0.5 * np.sum((u1**2 + v1**2) * weights)
    stats = viewer._build_summary_text(
        duration_str=(f"{viewer.times[-1]:.1f} hours"
                      if viewer.times is not None and len(viewer.times) else "N/A"),
        steps_str=(f"{len(viewer.times)}" if viewer.times is not None else "N/A"),
        circ0=circ0, circ1=circ1, ke0=ke0, ke1=ke1,
        max_z0=np.max(np.abs(zeta_init)),
        max_z1=np.max(np.abs(zeta_final)),
        rms_z0=np.sqrt(np.mean(zeta_init**2)),
        rms_z1=np.sqrt(np.mean(zeta_final**2)),
        max_speed0=np.max(np.sqrt(u0**2 + v0**2)),
        max_speed1=np.max(np.sqrt(u1**2 + v1**2)))
    panels.append(PanelPlacement(TextPanelSpec(stats), 0, 1))

    snapshots = _host(viewer.snapshots)
    if snapshots.ndim == 3:
        enstrophy = np.sqrt(np.mean(snapshots**2, axis=(1, 2)))
    else:
        enstrophy = np.sqrt(np.mean(snapshots**2, axis=1))
    if enstrophy[0] == 0.0:
        enstrophy_normalized = np.ones_like(enstrophy)
    else:
        enstrophy_normalized = enstrophy / enstrophy[0]
    y_margin = max(
        0.001, float(np.max(np.abs(enstrophy_normalized - 1.0))) * 1.2)
    panels.append(PanelPlacement(LinePanelSpec(
        series=(LineSeriesSpec(
            _host(viewer.times), enstrophy_normalized,
            label="RMS zeta (norm)"),),
        title="Conservation Check", x_label="Time (hours)",
        y_label="Normalized Magnitude",
        y_limits=(1.0 - y_margin, 1.0 + y_margin)), 1, 1))

    return FigureSpec(
        panels=tuple(panels), rows=3, columns=3,
        size_inches=(20.0, 18.0), dpi=200,
        width_ratios=(1.0, 0.6, 1.0)), stats
