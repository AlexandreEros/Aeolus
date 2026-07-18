"""BVE-specific presentation decisions expressed as generic specifications."""
from __future__ import annotations

import pathlib

import numpy as np

from planetary_sandbox.viz.fields import (ScalarGridField,
                                           SphericalHarmonicField)
from planetary_sandbox.viz.grid_adapter import map_to_uniform_latlon
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.renderers import get_default_renderer
from planetary_sandbox.viz.specs import (
    FigureSpec, LinePanelSpec, LineSeriesSpec, PanelGroupSpec, PanelPlacement,
    ScalarMapSpec, SpectralCoefficientMapSpec, StreamlineMapSpec,
    TextPanelSpec)
from planetary_sandbox.viz.timeline import (FigureFrame, FigureTimeline,
                                             render_snapshot_product)


BVE_SNAPSHOT_TIMES_FILENAME = "bve_snapshot_times.npy"
_SPECTRAL_NORMALIZATION = "orthonormal-complex-m>=0-real-field"


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _backend_array(owner, values):
    """Use device coefficients only for the repository's GPU operators."""
    if type(owner).__module__.startswith("planetary_sandbox."):
        import cupy as cp
        return cp.asarray(values)
    return values


def _map_vector_to_view(planet, u, v):
    view_grid, mapped_u = map_to_uniform_latlon(_host(u), planet.grid)
    _, mapped_v = map_to_uniform_latlon(
        _host(v), planet.grid, target_grid=view_grid)
    return view_grid, mapped_u, mapped_v


def build_bve_snapshot_timeline_from_data(
        planet, vorticity_grids, times_seconds, *, scenario: str,
        coefficients=None) -> FigureTimeline:
    """Compose BVE snapshot panels from an already-loaded artifact payload."""
    grids = _host(vorticity_grids)
    times = _host(times_seconds).astype(np.float64, copy=False)
    if grids.ndim not in (2, 3) or grids.shape[0] == 0:
        raise ValueError(
            "BVE snapshot grids must have shape (time, points) or "
            "(time, lat, lon) with at least one state")
    if times.shape != (grids.shape[0],):
        raise ValueError("BVE snapshot times must match the grid time axis")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("BVE snapshot times must be finite and nonnegative")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("BVE snapshot times must be strictly increasing")

    if coefficients is None:
        coefficient_states = [
            _host(planet.sh.transform(_backend_array(planet.sh, grid)))
            for grid in grids]
    else:
        coefficient_array = _host(coefficients)
        if (coefficient_array.ndim != 3 or
                coefficient_array.shape[0] != grids.shape[0]):
            raise ValueError(
                "BVE coefficients must have shape (time, l, m) matching grids")
        coefficient_states = coefficient_array

    radius = planet.params.equatorial_radius
    frames: list[FigureFrame] = []
    for index, (time_seconds, zeta_grid, zeta_lm) in enumerate(
            zip(times, grids, coefficient_states)):
        zeta_grid = np.asarray(zeta_grid).real
        device_zeta_lm = _backend_array(planet.so, zeta_lm)
        psi_lm = planet.so.inv_laplacian(device_zeta_lm)
        psi_grid = _host(planet.sh.inv_transform(psi_lm)).real
        u_grid, v_grid = planet.so.velocity_from_streamfunction(psi_lm)
        u_grid, v_grid = _host(u_grid).real, _host(v_grid).real

        view_grid, zeta_view = map_to_uniform_latlon(zeta_grid, planet.grid)
        _, psi_view = map_to_uniform_latlon(
            psi_grid, planet.grid, target_grid=view_grid)
        _, u_view, v_view = _map_vector_to_view(planet, u_grid, v_grid)
        latitudes = _host(view_grid.latitudes)
        longitudes = _host(view_grid.longitudes)
        selected_time = times[index:index + 1]

        zeta_field = ScalarGridField(
            zeta_view, latitudes, longitudes,
            name="relative vorticity", units="s^-1", times=selected_time)
        psi_field = ScalarGridField(
            psi_view, latitudes, longitudes,
            name="streamfunction", units="m^2/s", times=selected_time)

        time_hours = time_seconds / 3600.0

        panels = (
            PanelPlacement(ScalarMapSpec(
                zeta_field, f"Relative vorticity @ t={time_hours:.2f} h",
                normalization=NormalizationPolicy.symmetric(),
                color_policy="signed", normalization_group="bve-vorticity"),
                0, 0),
            PanelPlacement(ScalarMapSpec(
                psi_field, f"Streamfunction @ t={time_hours:.2f} h",
                normalization=NormalizationPolicy.automatic(),
                color_policy="viridis",
                normalization_group="bve-streamfunction"), 0, 1),
            PanelPlacement(StreamlineMapSpec(
                latitudes, longitudes, u_view, v_view, radius=radius,
                title=f"Velocity streamlines @ t={time_hours:.2f} h",
                normalization_group="bve-speed"), 0, 2),
        )
        frames.append(FigureFrame(
            time_seconds, FigureSpec(
                panels=panels, rows=1, columns=3,
                size_inches=(18.0, 6.0), dpi=200,
                panel_groups=(
                    PanelGroupSpec(
                        "Prognostic state", 0, 0, column_span=1),
                    PanelGroupSpec(
                        "Diagnostic fields", 0, 1, column_span=2,
                        separator_before=True)))))

    return FigureTimeline(tuple(frames), filename_prefix=scenario)


def build_bve_snapshot_timeline(
        planet, out_dir: pathlib.Path | str, *, scenario: str
        ) -> FigureTimeline:
    """Load the persisted BVE state and compose its snapshot timeline."""
    out_dir = pathlib.Path(out_dir)
    coefficients = np.load(out_dir / "vorticity_coeffs.npy")
    grids = np.load(out_dir / "vorticity_grid.npy")
    times = np.load(out_dir / BVE_SNAPSHOT_TIMES_FILENAME)
    return build_bve_snapshot_timeline_from_data(
        planet, grids, times, scenario=scenario, coefficients=coefficients)


def build_bve_spectral_snapshot_timeline_from_data(
        coefficients, times_seconds, *, scenario: str) -> FigureTimeline:
    """Compose BVE coefficient-space frames from persisted artifacts."""
    coefficients = _host(coefficients)
    times = _host(times_seconds).astype(np.float64, copy=False)
    if coefficients.ndim != 3 or coefficients.shape[0] == 0:
        raise ValueError(
            "BVE coefficients must have shape (time, l, m) with at least "
            "one state")
    if times.shape != (coefficients.shape[0],):
        raise ValueError("BVE snapshot times must match the coefficient time axis")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("BVE snapshot times must be finite and nonnegative")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("BVE snapshot times must be strictly increasing")

    field = SphericalHarmonicField(
        coefficients, "relative vorticity", "s^-1", times,
        normalization=_SPECTRAL_NORMALIZATION)
    frames = tuple(
        FigureFrame(time_seconds, FigureSpec(
            panels=(PanelPlacement(SpectralCoefficientMapSpec(
                field,
                f"Vorticity coefficients @ t={time_seconds / 3600.0:.2f} h",
                time_index=index,
                normalization=NormalizationPolicy.logarithmic_magnitude(),
                color_policy="magnitude",
                normalization_group="bve-vorticity-coefficients"), 0, 0),),
            rows=1, columns=1, size_inches=(8.0, 6.0), dpi=200))
        for index, time_seconds in enumerate(times))
    return FigureTimeline(frames, filename_prefix=f"{scenario}-spectral")


def build_bve_spectral_snapshot_timeline(
        planet, out_dir: pathlib.Path | str, *, scenario: str
        ) -> FigureTimeline:
    """Load persisted BVE coefficients and compose their timeline."""
    del planet  # Kept for a symmetric public adapter signature.
    out_dir = pathlib.Path(out_dir)
    coefficients = np.load(out_dir / "vorticity_coeffs.npy")
    times = np.load(out_dir / BVE_SNAPSHOT_TIMES_FILENAME)
    return build_bve_spectral_snapshot_timeline_from_data(
        coefficients, times, scenario=scenario)


def build_bve_snapshot_timelines(
        planet, out_dir: pathlib.Path | str, *, scenario: str
        ) -> dict[str, FigureTimeline]:
    """Build physical and spectral BVE views from one persisted payload."""
    out_dir = pathlib.Path(out_dir)
    coefficients = np.load(out_dir / "vorticity_coeffs.npy")
    grids = np.load(out_dir / "vorticity_grid.npy")
    times = np.load(out_dir / BVE_SNAPSHOT_TIMES_FILENAME)
    return {
        "physical": build_bve_snapshot_timeline_from_data(
            planet, grids, times, scenario=scenario,
            coefficients=coefficients),
        "spectral": build_bve_spectral_snapshot_timeline_from_data(
            coefficients, times, scenario=scenario),
    }


def render_bve_snapshots(
        planet, out_dir: pathlib.Path | str, *, scenario: str,
        metadata: dict | None = None, renderer=None
        ) -> dict[str, tuple[pathlib.Path, ...]]:
    """Atomically render the complete physical/spectral BVE product."""
    out_dir = pathlib.Path(out_dir)
    timelines = build_bve_snapshot_timelines(
        planet, out_dir, scenario=scenario)
    return render_snapshot_product(
        timelines, out_dir, renderer=renderer or get_default_renderer(),
        metadata=metadata)


# Descriptive aliases kept at the adapter boundary.
build_bve_timeline = build_bve_snapshot_timeline
render_bve_snapshot_timeline = render_bve_snapshots


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
