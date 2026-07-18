"""Matplotlib implementation of the small visualization backend protocol."""
from __future__ import annotations

import contextlib
import os
import pathlib
import uuid

import numpy as np

from .normalization import NormalizationKind
from .specs import (FigureSpec, LinePanelSpec, PanelPlacement, ScalarMapSpec,
                    SpectralCoefficientMapSpec, StreamlineMapSpec,
                    TextPanelSpec)


_SEMANTIC_COLORS = {
    "signed": "RdBu_r",
    "magnitude": "viridis",
    "sequential": "viridis",
}


class MatplotlibRenderer:
    """Render visualization specifications with Matplotlib's non-GUI backend."""

    def __init__(self) -> None:
        import matplotlib
        matplotlib.use("Agg")

    def render_scalar_map(self, specification: ScalarMapSpec,
                          output_path: pathlib.Path | str, *,
                          metadata: dict | None = None,
                          dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(12.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_spectral_coefficient_map(
            self, specification: SpectralCoefficientMapSpec,
            output_path: pathlib.Path | str, *, metadata: dict | None = None,
            dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(8.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_figure(self, specification: FigureSpec,
                      output_path: pathlib.Path | str, *,
                      metadata: dict | None = None) -> pathlib.Path:
        import matplotlib.pyplot as plt

        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=specification.size_inches)
        try:
            grid = figure.add_gridspec(
                specification.rows, specification.columns,
                width_ratios=specification.width_ratios,
                height_ratios=specification.height_ratios)
            for placement in specification.panels:
                axes = figure.add_subplot(grid[
                    placement.row:placement.row + placement.row_span,
                    placement.column:placement.column + placement.column_span])
                self._render_panel(figure, axes, placement.panel)
            if specification.tight_layout:
                figure.tight_layout()
            self._save_atomic(
                figure, output, dpi=specification.dpi, metadata=metadata)
        finally:
            plt.close(figure)
        return output

    def _render_panel(self, figure, axes, panel) -> None:
        if isinstance(panel, ScalarMapSpec):
            self._render_scalar_panel(figure, axes, panel)
        elif isinstance(panel, SpectralCoefficientMapSpec):
            self._render_spectral_panel(figure, axes, panel)
        elif isinstance(panel, StreamlineMapSpec):
            self._render_streamline_panel(figure, axes, panel)
        elif isinstance(panel, TextPanelSpec):
            axes.axis("off")
            axes.text(
                0.5, 0.5, panel.text, ha=panel.horizontal_alignment,
                va="center", fontfamily=panel.font_family,
                fontsize=panel.font_size)
        elif isinstance(panel, LinePanelSpec):
            self._render_line_panel(axes, panel)
        else:  # pragma: no cover - guarded by the specification union
            raise TypeError(f"unsupported panel specification {type(panel).__name__}")

    @staticmethod
    def _mpl_normalization(policy, values):
        from matplotlib.colors import LogNorm, Normalize

        resolved = policy.resolve(values)
        if resolved.kind is NormalizationKind.LOG_MAGNITUDE:
            return resolved, LogNorm(vmin=resolved.vmin, vmax=resolved.vmax)
        return resolved, Normalize(vmin=resolved.vmin, vmax=resolved.vmax)

    @staticmethod
    def _color_map(identifier: str):
        import matplotlib.pyplot as plt

        name = _SEMANTIC_COLORS.get(identifier, identifier)
        return plt.get_cmap(name).copy()

    def _render_scalar_panel(self, figure, axes, spec: ScalarMapSpec) -> None:
        if spec.view != "equirectangular":
            raise NotImplementedError(
                f"the Matplotlib backend does not support map view {spec.view!r}")
        if spec.central_longitude not in (None, 0, 0.0):
            raise NotImplementedError(
                "the initial Matplotlib backend supports central_longitude=0 only")
        values = np.asarray(spec.field.values_at(spec.time_index))
        _, norm = self._mpl_normalization(spec.normalization, values)
        cmap = self._color_map(spec.color_policy)

        # Source rows are north-to-south.  imshow with origin='lower' expects
        # south-to-north, so reverse exactly once at the renderer boundary.
        south_to_north = np.flip(values, axis=0)
        lon = np.rad2deg(spec.field.longitudes)
        lat = np.rad2deg(spec.field.latitudes)
        lon_step = (lon[1] - lon[0]) if lon.size > 1 else 360.0
        image = axes.imshow(
            south_to_north,
            extent=(float(lon[0]), float(lon[-1] + lon_step),
                    float(lat[-1]), float(lat[0])),
            cmap=cmap, norm=norm, aspect="equal", origin="lower")
        axes.set_title(spec.title)
        axes.set_xlabel("Longitude (deg)")
        axes.set_ylabel("Latitude (deg)")
        figure.colorbar(
            image, ax=axes, orientation="horizontal", pad=0.1,
            fraction=0.05, aspect=30, label=spec.display_units)

    def _render_spectral_panel(
            self, figure, axes, spec: SpectralCoefficientMapSpec) -> None:
        coefficients = spec.field.coefficients_at(spec.time_index)
        magnitude = np.abs(coefficients)
        valid = spec.field.valid_mask
        resolved, norm = self._mpl_normalization(
            spec.normalization, magnitude[valid])
        display = np.where(valid, np.maximum(magnitude, resolved.vmin), np.nan)
        display = np.ma.masked_where(~valid, display)
        cmap = self._color_map(spec.color_policy)
        cmap.set_bad("#d9d9d9")
        image = axes.imshow(
            display, origin="lower", interpolation="nearest", aspect="auto",
            extent=(-0.5, spec.field.l_max + 0.5,
                    -0.5, spec.field.l_max + 0.5),
            cmap=cmap, norm=norm)
        axes.set_title(spec.title)
        axes.set_xlabel("Spherical-harmonic order m")
        axes.set_ylabel("Spherical-harmonic degree l")
        unit_suffix = f" [{spec.display_units}]" if spec.display_units else ""
        figure.colorbar(
            image, ax=axes, orientation="vertical",
            label=f"Coefficient magnitude{unit_suffix}")

    def _render_streamline_panel(
            self, figure, axes, spec: StreamlineMapSpec) -> None:
        latitudes = spec.latitudes[::-1]
        u_grid = spec.zonal_velocity[::-1, :]
        v_grid = spec.meridional_velocity[::-1, :]
        longitudes = spec.longitudes

        stride = 1
        if latitudes.size > 300 or longitudes.size > 500:
            stride = 2
        if latitudes.size > 600 or longitudes.size > 1000:
            stride = 4
        if stride > 1:
            latitudes = latitudes[::stride]
            longitudes = longitudes[::stride]
            u_grid = u_grid[::stride, ::stride]
            v_grid = v_grid[::stride, ::stride]

        cos_lat = np.maximum(np.cos(latitudes), 1.0e-4)
        u_angular = u_grid / (spec.radius * cos_lat[:, None])
        v_angular = v_grid / spec.radius
        speed = np.sqrt(u_grid * u_grid + v_grid * v_grid)
        stream = axes.streamplot(
            np.rad2deg(longitudes), np.rad2deg(latitudes),
            u_angular, v_angular, color=speed,
            cmap=_SEMANTIC_COLORS.get(spec.color_policy, spec.color_policy),
            density=spec.density, linewidth=1, arrowsize=1.2)
        figure.colorbar(
            stream.lines, ax=axes, label=f"Flow Speed ({spec.units})",
            orientation="horizontal", pad=0.1, fraction=0.05, aspect=30)
        axes.set_xlim(np.rad2deg(longitudes).min(), np.rad2deg(longitudes).max())
        axes.set_ylim(np.rad2deg(latitudes).min(), np.rad2deg(latitudes).max())
        axes.set_xlabel("Longitude (deg)")
        axes.set_ylabel("Latitude (deg)")
        axes.set_title(spec.title)
        axes.set_aspect("equal")

    @staticmethod
    def _render_line_panel(axes, spec: LinePanelSpec) -> None:
        for series in spec.series:
            axes.plot(
                series.x, series.y, color=series.color,
                linestyle=series.line_style, linewidth=series.line_width,
                label=series.label)
        axes.set_xlabel(spec.x_label)
        axes.set_ylabel(spec.y_label)
        axes.set_title(spec.title)
        if spec.show_grid:
            axes.grid(True, alpha=0.3, linestyle="--")
        if spec.show_legend:
            axes.legend(loc="best", fontsize="small")
        if spec.y_limits is not None:
            axes.set_ylim(*spec.y_limits)

    @staticmethod
    def _save_atomic(figure, output: pathlib.Path, *, dpi: int,
                     metadata: dict | None) -> None:
        """Save to a same-directory temporary sibling, then atomically replace."""
        if not output.suffix:
            raise ValueError("render output path must have a file extension")
        temporary = output.with_name(
            f".{output.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}{output.suffix}")
        try:
            figure.savefig(temporary, dpi=dpi, metadata=metadata)
            # Windows' FlushFileBuffers requires a writable handle.
            with open(temporary, "rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        except BaseException:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
