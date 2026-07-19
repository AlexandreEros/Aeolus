"""SWE-specific field extraction and declarative panel composition."""
from __future__ import annotations

import pathlib

import numpy as np

from planetary_sandbox.physics.shallow_water import (DELTA, PHI, ZETA,
                                                       ShallowWaterState)
from planetary_sandbox.viz.fields import (ScalarGridField,
                                           SphericalHarmonicField)
from planetary_sandbox.viz.grid_adapter import map_to_uniform_latlon
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.renderers import get_default_renderer
from planetary_sandbox.viz.specs import (
    FigureSpec, PanelGroupSpec, PanelPlacement, ScalarMapSpec,
    SpectralCoefficientMapSpec, StreamlineMapSpec)
from planetary_sandbox.viz.timeline import (FigureFrame, FigureTimeline,
                                             render_snapshot_product)


SWE_SUMMARY_FILENAME = "swe_summary.png"
SWE_SNAPSHOT_TIMES_FILENAME = "swe_snapshot_times.npy"
_SPECTRAL_NORMALIZATION = "orthonormal-complex-m>=0-real-field"


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _backend_array(owner, values):
    """Use device state arrays only for the repository's SWE model."""
    if type(owner).__module__.startswith("planetary_sandbox."):
        import cupy as cp
        return cp.asarray(values)
    return values


def _load_swe_fields(
        out_dir: pathlib.Path | str
        ) -> tuple[tuple[SphericalHarmonicField, ...], np.ndarray]:
    """Validate and expose the authoritative persisted SWE state."""
    out_dir = pathlib.Path(out_dir)
    coefficients = np.load(out_dir / "swe_coeffs.npy")
    times = np.load(out_dir / SWE_SNAPSHOT_TIMES_FILENAME)
    if coefficients.ndim != 4 or coefficients.shape[1] != 3:
        raise ValueError(
            "swe_coeffs.npy must have shape (time, 3, l, m), got "
            f"{coefficients.shape}")
    if times.shape != (coefficients.shape[0],):
        raise ValueError(
            "swe_snapshot_times.npy must match the coefficient time axis")
    if coefficients.shape[0] == 0:
        raise ValueError("an SWE visualization requires persisted states")

    return (
        SphericalHarmonicField(
            coefficients[:, ZETA], "relative vorticity", "s^-1", times,
            normalization=_SPECTRAL_NORMALIZATION),
        SphericalHarmonicField(
            coefficients[:, DELTA], "horizontal divergence", "s^-1", times,
            normalization=_SPECTRAL_NORMALIZATION),
        SphericalHarmonicField(
            coefficients[:, PHI], "perturbation geopotential", "m^2 s^-2",
            times, normalization=_SPECTRAL_NORMALIZATION),
    ), times


def _synthesize(model, field: SphericalHarmonicField,
                time_index: int) -> np.ndarray:
    selected = field.coefficients_at(time_index)
    # Repository transforms index with CuPy arrays.  Keep that execution
    # detail at this persisted-artifact adapter; viz fields/specs stay NumPy.
    if type(model.sh).__module__.startswith("planetary_sandbox."):
        import cupy as cp
        selected = cp.asarray(selected)
    return _host(model.sh.inv_transform(selected)).real


def _map_scalar_series(values, model, *, name: str, units: str,
                       times: np.ndarray) -> ScalarGridField:
    mapped = []
    view_grid = None
    for values_at_time in values:
        view_grid, view_values = map_to_uniform_latlon(
            values_at_time, model.grid, target_grid=view_grid)
        mapped.append(view_values)
    assert view_grid is not None
    return ScalarGridField(
        np.stack(mapped), _host(view_grid.latitudes),
        _host(view_grid.longitudes), name=name, units=units, times=times)


def _extract_swe_scalar_fields(
        model, spectral_fields: tuple[SphericalHarmonicField, ...],
        times: np.ndarray) -> tuple[ScalarGridField, ...]:
    """Synthesize all physical SWE fields once for summary/timeline reuse."""
    zeta = [_synthesize(model, spectral_fields[0], i)
            for i in range(times.size)]
    delta = [_synthesize(model, spectral_fields[1], i)
             for i in range(times.size)]
    # phi is the thickness perturbation relative to Phi0=gH; phi/g is the
    # corresponding layer-thickness anomaly relative to H.
    thickness = [_synthesize(model, spectral_fields[2], i) / model.gravity
                 for i in range(times.size)]
    return (
        _map_scalar_series(
            zeta, model, name="relative vorticity", units="s^-1",
            times=times),
        _map_scalar_series(
            delta, model, name="horizontal divergence", units="s^-1",
            times=times),
        _map_scalar_series(
            thickness, model,
            name="layer-thickness anomaly h' derived as Phi'/g", units="m",
            times=times),
    )


def _extract_swe_topography_fields(
        model, spectral_fields: tuple[SphericalHarmonicField, ...],
        times: np.ndarray) -> tuple[ScalarGridField, ScalarGridField] | None:
    """Free-surface anomaly and terrain elevation for a non-flat bottom.

    Returns ``None`` for a flat bottom (the historical figures are rendered
    unchanged). The free-surface anomaly is eta' = (phi + phi_s')/g in
    metres (phi_s' the mean-removed surface geopotential); the terrain
    panel is the band-limited surface elevation h_s in metres, constant in
    time (replicated per frame so run-wide normalization is trivially
    comparable across time).
    """
    if not getattr(model, "has_topography", False):
        return None
    phi_s_anom = _host(
        model.sh.inv_transform(model.phi_s_anom_lm).real)
    eta = [(_synthesize(model, spectral_fields[2], i) + phi_s_anom)
           / model.gravity for i in range(times.size)]
    elevation = _host(model.topography.elevation_on(model.sh))
    terrain = [elevation for _ in range(times.size)]
    return (
        _map_scalar_series(
            eta, model,
            name="free-surface anomaly eta' = (Phi' + Phi_s')/g",
            units="m", times=times),
        _map_scalar_series(
            terrain, model,
            name="surface elevation h_s (band-limited)", units="m",
            times=times),
    )


def _extract_swe_winds(
        model, spectral_fields: tuple[SphericalHarmonicField, ...]
        ) -> tuple[np.ndarray, np.ndarray,
                   tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Derive instantaneous state-grid winds from each persisted SWE state."""
    winds = []
    view_grid = None
    for index in range(spectral_fields[0].state_count):
        coefficients = np.stack(
            [field.coefficients_at(index) for field in spectral_fields])
        state = ShallowWaterState(_backend_array(model, coefficients))
        u_grid, v_grid = model.wind_on_state_grid(state)
        view_grid, u_view = map_to_uniform_latlon(
            _host(u_grid), model.grid, target_grid=view_grid)
        _, v_view = map_to_uniform_latlon(
            _host(v_grid), model.grid, target_grid=view_grid)
        winds.append((u_view, v_view))
    assert view_grid is not None
    return (_host(view_grid.latitudes), _host(view_grid.longitudes),
            tuple(winds))


_SWE_PHYSICAL_TITLES = (
    "Relative vorticity",
    "Horizontal divergence",
    "Layer-thickness anomaly h' = Phi'/g",
)
_SWE_PHYSICAL_NORMALIZATION_GROUPS = (
    "swe-relative-vorticity",
    "swe-horizontal-divergence",
    "swe-thickness-anomaly",
)
_SWE_SPECTRAL_TITLES = (
    "Relative vorticity",
    "Horizontal divergence",
    "Perturbation geopotential",
)
_SWE_SPECTRAL_NORMALIZATION_GROUPS = (
    "swe-relative-vorticity",
    "swe-horizontal-divergence",
    "swe-perturbation-geopotential",
)

#: Extra panels rendered only for a non-flat bottom. The free-surface
#: anomaly is a signed dynamic field (symmetric run-wide normalization);
#: the terrain panel is static shading in its own sequential colors so it
#: can never be confused with — or recolor — the dynamic fields.
_SWE_TOPO_TITLES = (
    "Free-surface anomaly eta' = (Phi' + Phi_s')/g",
    "Surface elevation h_s (band-limited, static)",
)
_SWE_TOPO_NORMALIZATION_GROUPS = (
    "swe-free-surface-anomaly",
    "swe-terrain-elevation",
)


def _topography_panels(topo_fields, *, row: int, time_index: int,
                       title_suffix: str) -> list[PanelPlacement]:
    eta_field, terrain_field = topo_fields
    return [
        PanelPlacement(ScalarMapSpec(
            eta_field, _SWE_TOPO_TITLES[0] + title_suffix,
            time_index=time_index,
            normalization=NormalizationPolicy.symmetric(),
            color_policy="signed",
            normalization_group=_SWE_TOPO_NORMALIZATION_GROUPS[0]), row, 0),
        PanelPlacement(ScalarMapSpec(
            terrain_field, _SWE_TOPO_TITLES[1], time_index=time_index,
            normalization=NormalizationPolicy.automatic(),
            color_policy="viridis",
            normalization_group=_SWE_TOPO_NORMALIZATION_GROUPS[1]), row, 1),
    ]


def _build_swe_scalar_figure(fields: tuple[ScalarGridField, ...], *,
                             time_index: int,
                             title_suffix: str = "",
                             topo_fields=None) -> FigureSpec:
    panels = [
        PanelPlacement(ScalarMapSpec(
            field, title + title_suffix, time_index=time_index,
            normalization=NormalizationPolicy.symmetric(),
            color_policy="signed", normalization_group=group), 0, column)
        for column, (field, title, group) in enumerate(zip(
            fields, _SWE_PHYSICAL_TITLES,
            _SWE_PHYSICAL_NORMALIZATION_GROUPS))]
    if topo_fields is None:
        return FigureSpec(
            panels=tuple(panels), rows=1, columns=3,
            size_inches=(18.0, 6.0), dpi=200)
    panels += _topography_panels(topo_fields, row=1, time_index=time_index,
                                 title_suffix=title_suffix)
    return FigureSpec(
        panels=tuple(panels), rows=2, columns=3,
        size_inches=(18.0, 12.0), dpi=200,
        panel_groups=(
            PanelGroupSpec("Prognostic state", 0, 0, column_span=3),
            PanelGroupSpec("Topography & free surface", 1, 0,
                           column_span=2)))


def _build_swe_physical_figure(
        fields: tuple[ScalarGridField, ...], *, time_index: int,
        latitudes: np.ndarray, longitudes: np.ndarray,
        wind: tuple[np.ndarray, np.ndarray], radius: float,
        title_suffix: str = "", topo_fields=None) -> FigureSpec:
    panels = [
        PanelPlacement(ScalarMapSpec(
            field, title + title_suffix, time_index=time_index,
            normalization=NormalizationPolicy.symmetric(),
            color_policy="signed", normalization_group=group), 0, column)
        for column, (field, title, group) in enumerate(zip(
            fields, _SWE_PHYSICAL_TITLES,
            _SWE_PHYSICAL_NORMALIZATION_GROUPS))]
    panels.append(PanelPlacement(StreamlineMapSpec(
        latitudes, longitudes, wind[0], wind[1], radius=radius,
        title="Velocity streamlines" + title_suffix,
        normalization_group="swe-speed"), 0, 3))
    groups = [
        PanelGroupSpec("Prognostic state", 0, 0, column_span=3),
        PanelGroupSpec("Diagnostic fields", 0, 3, separator_before=True)]
    rows = 1
    size = (24.0, 6.0)
    if topo_fields is not None:
        panels += _topography_panels(
            topo_fields, row=1, time_index=time_index,
            title_suffix=title_suffix)
        groups.append(PanelGroupSpec("Topography & free surface", 1, 0,
                                     column_span=2))
        rows = 2
        size = (24.0, 12.0)
    return FigureSpec(
        panels=tuple(panels), rows=rows, columns=4,
        size_inches=size, dpi=200, panel_groups=tuple(groups))


def _build_swe_spectral_figure(
        fields: tuple[SphericalHarmonicField, ...], *, time_index: int,
        title_suffix: str = "") -> FigureSpec:
    panels = tuple(
        PanelPlacement(SpectralCoefficientMapSpec(
            field, title + " coefficients" + title_suffix,
            time_index=time_index,
            normalization=NormalizationPolicy.logarithmic_magnitude(),
            encoding="phase-magnitude",
            color_policy="magnitude",
            normalization_group=f"{group}-coefficients"), 0, column)
        for column, (field, title, group) in enumerate(zip(
            fields, _SWE_SPECTRAL_TITLES,
            _SWE_SPECTRAL_NORMALIZATION_GROUPS)))
    return FigureSpec(
        panels=panels, rows=1, columns=3,
        size_inches=(18.0, 6.0), dpi=200)


def _build_swe_physical_timeline(
        model, spectral_fields: tuple[SphericalHarmonicField, ...],
        times: np.ndarray, *, scenario: str) -> FigureTimeline:
    fields = _extract_swe_scalar_fields(model, spectral_fields, times)
    topo_fields = _extract_swe_topography_fields(
        model, spectral_fields, times)
    latitudes, longitudes, winds = _extract_swe_winds(
        model, spectral_fields)
    frames = tuple(
        FigureFrame(time_seconds, _build_swe_physical_figure(
            fields, time_index=index, latitudes=latitudes,
            longitudes=longitudes, wind=winds[index], radius=model.R,
            title_suffix=f" @ t={time_seconds / 3600.0:.2f} h",
            topo_fields=topo_fields))
        for index, time_seconds in enumerate(times))
    return FigureTimeline(frames, filename_prefix=scenario)


def _build_swe_spectral_timeline(
        spectral_fields: tuple[SphericalHarmonicField, ...],
        times: np.ndarray, *, scenario: str) -> FigureTimeline:
    frames = tuple(
        FigureFrame(time_seconds, _build_swe_spectral_figure(
            spectral_fields, time_index=index,
            title_suffix=f" @ t={time_seconds / 3600.0:.2f} h"))
        for index, time_seconds in enumerate(times))
    return FigureTimeline(frames, filename_prefix=f"{scenario}-spectral")


def build_swe_summary_spec(model, out_dir: pathlib.Path | str, *,
                           time_index: int = -1) -> FigureSpec:
    """Load persisted SWE coefficients and describe one selected state."""
    spectral_fields, times = _load_swe_fields(out_dir)
    fields = _extract_swe_scalar_fields(model, spectral_fields, times)
    selected_fields = tuple(field.select_time(time_index) for field in fields)
    topo_fields = _extract_swe_topography_fields(
        model, spectral_fields, times)
    if topo_fields is not None:
        topo_fields = tuple(
            field.select_time(time_index) for field in topo_fields)
    return _build_swe_scalar_figure(selected_fields, time_index=0,
                                    topo_fields=topo_fields)


def build_swe_snapshot_timeline(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe"
        ) -> FigureTimeline:
    """Build every SWE snapshot frame from the persisted coefficient capsule."""
    spectral_fields, times = _load_swe_fields(out_dir)
    return _build_swe_physical_timeline(
        model, spectral_fields, times, scenario=scenario)


def build_swe_spectral_snapshot_timeline(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe"
        ) -> FigureTimeline:
    """Build SWE coefficient-space frames at the persisted snapshot times."""
    del model  # Kept for a symmetric public adapter signature.
    spectral_fields, times = _load_swe_fields(out_dir)
    return _build_swe_spectral_timeline(
        spectral_fields, times, scenario=scenario)


def build_swe_snapshot_timelines(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe"
        ) -> dict[str, FigureTimeline]:
    """Build the physical and spectral views from one persisted payload."""
    spectral_fields, times = _load_swe_fields(out_dir)
    return {
        "physical": _build_swe_physical_timeline(
            model, spectral_fields, times, scenario=scenario),
        "spectral": _build_swe_spectral_timeline(
            spectral_fields, times, scenario=scenario),
    }


def render_swe_snapshots(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe",
        metadata: dict | None = None, renderer=None
        ) -> dict[str, tuple[pathlib.Path, ...]]:
    """Atomically render the complete physical/spectral SWE product."""
    out_dir = pathlib.Path(out_dir)
    timelines = build_swe_snapshot_timelines(
        model, out_dir, scenario=scenario)
    return render_snapshot_product(
        timelines, out_dir, renderer=renderer or get_default_renderer(),
        metadata=metadata)


def render_swe_summary(model, out_dir: pathlib.Path | str, *,
                       time_index: int = -1,
                       metadata: dict | None = None,
                       renderer=None) -> pathlib.Path:
    """Atomically render an SWE summary; failures deliberately propagate."""
    out_dir = pathlib.Path(out_dir)
    specification = build_swe_summary_spec(
        model, out_dir, time_index=time_index)
    backend = renderer or get_default_renderer()
    return backend.render_figure(
        specification, out_dir / SWE_SUMMARY_FILENAME, metadata=metadata)


build_swe_timeline = build_swe_snapshot_timeline
render_swe_snapshot_timeline = render_swe_snapshots


__all__ = [
    "SWE_SNAPSHOT_TIMES_FILENAME",
    "SWE_SUMMARY_FILENAME",
    "build_swe_snapshot_timelines",
    "build_swe_snapshot_timeline",
    "build_swe_spectral_snapshot_timeline",
    "build_swe_summary_spec",
    "build_swe_timeline",
    "render_swe_snapshot_timeline",
    "render_swe_snapshots",
    "render_swe_summary",
]
