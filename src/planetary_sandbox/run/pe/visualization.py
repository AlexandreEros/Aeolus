"""Primitive-equation summary artifact (minimal, honest first-runner view).

For one selectable full sigma level (the middle level by default), four
panels of one stored state (the final state by default):

1. relative vorticity (s^-1),
2. horizontal divergence (s^-1),
3. temperature anomaly relative to the horizontal mean (K),
4. surface-pressure ln p_s anomaly relative to the horizontal mean.

The horizontal mean is exactly the (0,0) spherical-harmonic monopole, so the
anomalies are formed by zeroing that coefficient before synthesis. This is a
single-state snapshot on the model's own grid, mapped to a uniform lat-lon
view for display; it makes NO claim of being a statistically equilibrated
climate. Works for both backends (it uses the model's transform + the shared
lat-lon view adapter, exactly like the SWE summary).
"""
from __future__ import annotations

import pathlib

import numpy as np

from planetary_sandbox.viz.fields import ScalarGridField
from planetary_sandbox.viz.grid_adapter import map_to_uniform_latlon
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.renderers import get_default_renderer
from planetary_sandbox.viz.specs import FigureSpec, PanelPlacement, ScalarMapSpec

PE_SUMMARY_FILENAME = "pe_summary.png"
PE_COEFFS_FILENAME = "pe_coeffs.npy"
PE_SNAPSHOT_TIMES_FILENAME = "pe_snapshot_times.npy"


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _load_pe_coeffs(out_dir: pathlib.Path | str,
                    nlev: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the persisted PE coefficient stack and times."""
    out_dir = pathlib.Path(out_dir)
    coeffs = np.load(out_dir / PE_COEFFS_FILENAME)
    times = np.load(out_dir / PE_SNAPSHOT_TIMES_FILENAME)
    if coeffs.ndim != 4 or coeffs.shape[1] != 3 * nlev + 1:
        raise ValueError(
            f"pe_coeffs.npy must have shape (time, 3*nlev+1, l, m) with "
            f"nlev={nlev}, got {coeffs.shape}")
    if coeffs.shape[2] != coeffs.shape[3]:
        raise ValueError(
            f"pe_coeffs.npy trailing axes must be square (l, m), got "
            f"{coeffs.shape[2:]}")
    if times.shape != (coeffs.shape[0],):
        raise ValueError(
            "pe_snapshot_times.npy must match the coefficient time axis")
    if coeffs.shape[0] == 0:
        raise ValueError("a PE visualization requires persisted states")
    return coeffs, times


def _synthesize(model, coeffs_2d: np.ndarray, *,
                subtract_mean: bool) -> np.ndarray:
    """Synthesize one grid field, optionally as an anomaly about its mean."""
    if type(model.sh).__module__.startswith("planetary_sandbox."):
        import cupy as cp
        selected = cp.asarray(coeffs_2d)
    else:
        selected = np.asarray(coeffs_2d)
    if subtract_mean:
        selected = selected.copy()
        selected[0, 0] = 0.0  # the (0,0) monopole IS the horizontal mean
    return _host(model.sh.inv_transform(selected)).real


def _view_field(model, grid_values: np.ndarray, *, name: str, units: str,
                target_grid=None):
    view_grid, view_values = map_to_uniform_latlon(
        grid_values, model.grid, target_grid=target_grid)
    return view_grid, ScalarGridField(
        view_values[None], _host(view_grid.latitudes),
        _host(view_grid.longitudes), name=name, units=units)


def build_pe_summary_spec(model, out_dir: pathlib.Path | str, *,
                          level: int | None = None,
                          time_index: int = -1) -> FigureSpec:
    """Describe the four-panel PE summary for one level and one stored state."""
    K = model.nlev
    coeffs, _ = _load_pe_coeffs(out_dir, K)
    level = K // 2 if level is None else int(level)
    if not 0 <= level < K:
        raise ValueError(f"level must be in [0, {K}), got {level}")
    sigma_full = model.sigma.full_levels[level]
    suffix = f" (level {level + 1}/{K}, sigma={sigma_full:.3f})"

    state = coeffs[time_index]
    zeta = _synthesize(model, state[level], subtract_mean=False)
    delta = _synthesize(model, state[K + level], subtract_mean=False)
    t_anom = _synthesize(model, state[2 * K + level], subtract_mean=True)
    lnps_anom = _synthesize(model, state[3 * K], subtract_mean=True)

    view_grid = None
    panel_data = [
        (zeta, "Relative vorticity" + suffix, "s^-1", "pe-vorticity"),
        (delta, "Horizontal divergence" + suffix, "s^-1", "pe-divergence"),
        (t_anom, "Temperature anomaly (vs horizontal mean)" + suffix, "K",
         "pe-temperature-anomaly"),
        (lnps_anom, "ln p_s anomaly (vs horizontal mean)", "ln(Pa)",
         "pe-lnps-anomaly"),
    ]
    panels = []
    for column, (values, title, units, group) in enumerate(panel_data):
        view_grid, field = _view_field(
            model, values, name=title, units=units, target_grid=view_grid)
        panels.append(PanelPlacement(ScalarMapSpec(
            field, title, time_index=0,
            normalization=NormalizationPolicy.symmetric(),
            color_policy="signed", normalization_group=group), 0, column))
    return FigureSpec(panels=tuple(panels), rows=1, columns=4,
                      size_inches=(24.0, 6.0), dpi=200)


def render_pe_summary(model, out_dir: pathlib.Path | str, *,
                      level: int | None = None, time_index: int = -1,
                      metadata: dict | None = None,
                      renderer=None) -> pathlib.Path:
    """Render the PE summary PNG; failures deliberately propagate."""
    out_dir = pathlib.Path(out_dir)
    spec = build_pe_summary_spec(model, out_dir, level=level,
                                 time_index=time_index)
    backend = renderer or get_default_renderer()
    return backend.render_figure(
        spec, out_dir / PE_SUMMARY_FILENAME, metadata=metadata)


__all__ = [
    "PE_COEFFS_FILENAME",
    "PE_SNAPSHOT_TIMES_FILENAME",
    "PE_SUMMARY_FILENAME",
    "build_pe_summary_spec",
    "render_pe_summary",
]
