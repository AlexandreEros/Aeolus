"""SWE-specific field extraction and declarative panel composition."""
from __future__ import annotations

import pathlib

import numpy as np

from planetary_sandbox.physics.shallow_water import DELTA, PHI, ZETA
from planetary_sandbox.viz.fields import (ScalarGridField,
                                           SphericalHarmonicField)
from planetary_sandbox.viz.grid_adapter import map_to_uniform_latlon
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.renderers import get_default_renderer
from planetary_sandbox.viz.specs import (FigureSpec, PanelPlacement,
                                         ScalarMapSpec)
from planetary_sandbox.viz.timeline import (FigureFrame, FigureTimeline,
                                             render_figure_timeline)


SWE_SUMMARY_FILENAME = "swe_summary.png"
SWE_SNAPSHOT_TIMES_FILENAME = "swe_snapshot_times.npy"
_SPECTRAL_NORMALIZATION = "orthonormal-complex-m>=0-real-field"


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


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
    # phi is the perturbation relative to Phi0=gH; phi/g is the corresponding
    # layer-thickness anomaly relative to H.
    thickness = [_synthesize(model, spectral_fields[2], i) / model.gravity
                 for i in range(times.size)]
    return (
        _map_scalar_series(
            thickness, model, name="layer thickness anomaly", units="m",
            times=times),
        _map_scalar_series(
            zeta, model, name="relative vorticity", units="s^-1",
            times=times),
        _map_scalar_series(
            delta, model, name="horizontal divergence", units="s^-1",
            times=times),
    )


_SWE_TITLES = (
    "Layer thickness anomaly",
    "Relative vorticity",
    "Horizontal divergence",
)
_SWE_NORMALIZATION_GROUPS = (
    "swe-thickness-anomaly",
    "swe-relative-vorticity",
    "swe-horizontal-divergence",
)


def _build_swe_figure(fields: tuple[ScalarGridField, ...], *,
                      time_index: int, title_suffix: str = "") -> FigureSpec:
    panels = tuple(
        PanelPlacement(ScalarMapSpec(
            field, title + title_suffix, time_index=time_index,
            normalization=NormalizationPolicy.symmetric(),
            color_policy="signed", normalization_group=group), 0, column)
        for column, (field, title, group) in enumerate(zip(
            fields, _SWE_TITLES, _SWE_NORMALIZATION_GROUPS)))
    return FigureSpec(
        panels=panels, rows=1, columns=3,
        size_inches=(18.0, 6.0), dpi=200)


def build_swe_summary_spec(model, out_dir: pathlib.Path | str, *,
                           time_index: int = -1) -> FigureSpec:
    """Load persisted SWE coefficients and describe one selected state."""
    spectral_fields, times = _load_swe_fields(out_dir)
    fields = _extract_swe_scalar_fields(model, spectral_fields, times)
    selected_fields = tuple(field.select_time(time_index) for field in fields)
    return _build_swe_figure(selected_fields, time_index=0)


def build_swe_snapshot_timeline(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe"
        ) -> FigureTimeline:
    """Build every SWE snapshot frame from the persisted coefficient capsule."""
    spectral_fields, times = _load_swe_fields(out_dir)
    fields = _extract_swe_scalar_fields(model, spectral_fields, times)
    frames = tuple(
        FigureFrame(time_seconds, _build_swe_figure(
            fields, time_index=index,
            title_suffix=f" @ t={time_seconds / 3600.0:.2f} h"))
        for index, time_seconds in enumerate(times))
    return FigureTimeline(frames, filename_prefix=scenario)


def render_swe_snapshots(
        model, out_dir: pathlib.Path | str, *, scenario: str = "swe",
        metadata: dict | None = None, renderer=None
        ) -> tuple[pathlib.Path, ...]:
    """Render all SWE persisted states through the shared timeline path."""
    out_dir = pathlib.Path(out_dir)
    timeline = build_swe_snapshot_timeline(
        model, out_dir, scenario=scenario)
    return render_figure_timeline(
        timeline, out_dir, renderer=renderer or get_default_renderer(),
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
    "build_swe_snapshot_timeline",
    "build_swe_summary_spec",
    "build_swe_timeline",
    "render_swe_snapshot_timeline",
    "render_swe_snapshots",
    "render_swe_summary",
]
