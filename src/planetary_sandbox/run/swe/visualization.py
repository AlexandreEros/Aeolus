"""SWE-specific final-state presentation from persisted run artifacts."""
from __future__ import annotations

import pathlib

import numpy as np

from planetary_sandbox.physics.shallow_water import DELTA, PHI, ZETA
from planetary_sandbox.viz.fields import SphericalHarmonicField
from planetary_sandbox.viz.grid_adapter import scalar_field_on_uniform_latlon
from planetary_sandbox.viz.normalization import NormalizationPolicy
from planetary_sandbox.viz.renderers import get_default_renderer
from planetary_sandbox.viz.specs import (FigureSpec, PanelPlacement,
                                         ScalarMapSpec)


SWE_SUMMARY_FILENAME = "swe_summary.png"
_SPECTRAL_NORMALIZATION = "orthonormal-complex-m>=0-real-field"


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def build_swe_summary_spec(model, out_dir: pathlib.Path | str, *,
                           time_index: int = -1) -> FigureSpec:
    """Load persisted SWE coefficients and describe the selected-state summary."""
    out_dir = pathlib.Path(out_dir)
    coefficients = np.load(out_dir / "swe_coeffs.npy")
    times = np.load(out_dir / "swe_snapshot_times.npy")
    if coefficients.ndim != 4 or coefficients.shape[1] != 3:
        raise ValueError(
            "swe_coeffs.npy must have shape (time, 3, l, m), got "
            f"{coefficients.shape}")
    if times.shape != (coefficients.shape[0],):
        raise ValueError(
            "swe_snapshot_times.npy must match the coefficient time axis")
    if coefficients.shape[0] == 0:
        raise ValueError("an SWE summary requires at least one persisted state")

    spectral_fields = (
        SphericalHarmonicField(
            coefficients[:, ZETA], "relative vorticity", "s^-1", times,
            normalization=_SPECTRAL_NORMALIZATION),
        SphericalHarmonicField(
            coefficients[:, DELTA], "horizontal divergence", "s^-1", times,
            normalization=_SPECTRAL_NORMALIZATION),
        SphericalHarmonicField(
            coefficients[:, PHI], "perturbation geopotential", "m^2 s^-2",
            times, normalization=_SPECTRAL_NORMALIZATION),
    )

    selected_time = spectral_fields[0].select_time(time_index).times

    def synthesize(field: SphericalHarmonicField) -> np.ndarray:
        selected = field.coefficients_at(time_index)
        # Repository transforms index with CuPy arrays and therefore require a
        # device coefficient array.  Keep that execution detail here at the
        # persisted-artifact adapter; field/spec objects remain NumPy-only.
        if type(model.sh).__module__.startswith("planetary_sandbox."):
            import cupy as cp
            selected = cp.asarray(selected)
        return _host(model.sh.inv_transform(selected)).real

    zeta_grid = synthesize(spectral_fields[0])
    delta_grid = synthesize(spectral_fields[1])
    # phi is the persisted perturbation relative to Phi0=gH.  Dividing by g
    # gives the corresponding layer-thickness anomaly relative to H.
    thickness_anomaly = synthesize(spectral_fields[2]) / model.gravity

    fields = (
        scalar_field_on_uniform_latlon(
            thickness_anomaly, model.grid, name="layer thickness anomaly",
            units="m", times=selected_time),
        scalar_field_on_uniform_latlon(
            zeta_grid, model.grid, name="relative vorticity", units="s^-1",
            times=selected_time),
        scalar_field_on_uniform_latlon(
            delta_grid, model.grid, name="horizontal divergence", units="s^-1",
            times=selected_time),
    )
    titles = (
        "Layer thickness anomaly",
        "Relative vorticity",
        "Horizontal divergence",
    )
    panels = tuple(
        PanelPlacement(ScalarMapSpec(
            field, title, normalization=NormalizationPolicy.symmetric(),
            color_policy="signed"), 0, column)
        for column, (field, title) in enumerate(zip(fields, titles)))
    return FigureSpec(
        panels=panels, rows=1, columns=3,
        size_inches=(18.0, 6.0), dpi=200)


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
