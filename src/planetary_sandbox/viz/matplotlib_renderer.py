"""Matplotlib implementation of the small visualization backend protocol."""
from __future__ import annotations

import contextlib
import os
import pathlib
import uuid

import numpy as np

from .complex_encoding import phase_magnitude_hsv
from .normalization import NormalizationKind
from .specs import (FigureSpec, LinePanelSpec, PanelPlacement, ScalarMapSpec,
                    SpectralCoefficientMapSpec, SpectralEncoding,
                    StreamlineMapSpec,
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

    def render_streamline_map(
            self, specification: StreamlineMapSpec,
            output_path: pathlib.Path | str, *, metadata: dict | None = None,
            dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(12.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_figure(self, specification: FigureSpec,
                      output_path: pathlib.Path | str, *,
                      metadata: dict | None = None) -> pathlib.Path:
        import matplotlib.pyplot as plt

        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=specification.size_inches)
        try:
            header_rows = sorted({group.row
                                  for group in specification.panel_groups})
            source_height_ratios = (
                specification.height_ratios or
                (1.0,) * specification.rows)
            mean_row_ratio = sum(source_height_ratios) / len(
                source_height_ratios)
            expanded_height_ratios = []
            for row, ratio in enumerate(source_height_ratios):
                if row in header_rows:
                    expanded_height_ratios.append(0.08 * mean_row_ratio)
                expanded_height_ratios.append(ratio)

            grid = figure.add_gridspec(
                specification.rows + len(header_rows), specification.columns,
                width_ratios=specification.width_ratios,
                height_ratios=expanded_height_ratios)
            rendered_groups = []
            for group in specification.panel_groups:
                header_row = group.row + sum(
                    candidate < group.row for candidate in header_rows)
                axes = figure.add_subplot(grid[
                    header_row,
                    group.column:group.column + group.column_span])
                axes.axis("off")
                axes.text(
                    0.5, 0.45, group.title, ha="center", va="center",
                    fontsize=11.0, fontweight="semibold", color="#333333")
                rendered_groups.append((group, axes))

            rendered_panels = []
            phase_panels = []
            for placement in specification.panels:
                first_row = placement.row + sum(
                    candidate <= placement.row for candidate in header_rows)
                last_source_row = placement.row + placement.row_span - 1
                last_row = last_source_row + sum(
                    candidate <= last_source_row for candidate in header_rows)
                axes = figure.add_subplot(grid[
                    first_row:last_row + 1,
                    placement.column:placement.column + placement.column_span])
                self._render_panel(figure, axes, placement.panel)
                rendered_panels.append((placement, axes))
                if (isinstance(placement.panel, SpectralCoefficientMapSpec) and
                        placement.panel.encoding is
                        SpectralEncoding.PHASE_MAGNITUDE):
                    phase_panels.append(placement.panel)
            if specification.tight_layout:
                phase_margin = min(
                    0.14, 0.7 / figure.get_figheight()) if phase_panels else 0.0
                figure.tight_layout(
                    rect=(0.0, phase_margin, 1.0, 1.0)
                    if phase_panels else None)
            self._render_panel_groups(
                figure, specification, rendered_panels, rendered_groups)
            if phase_panels:
                self._render_phase_legend(figure, phase_panels)
            self._save_atomic(
                figure, output, dpi=specification.dpi, metadata=metadata)
        finally:
            plt.close(figure)
        return output

    @staticmethod
    def _render_panel_groups(
            figure, specification, rendered_panels, rendered_groups) -> None:
        """Draw requested generic inter-group dividers after layout."""
        if not specification.panel_groups:
            return

        from matplotlib.lines import Line2D
        from matplotlib.transforms import Bbox

        # Layout must be settled before axes positions can anchor separators.
        figure.canvas.draw()
        bounds = []
        heading_axes = dict(rendered_groups)
        for group in specification.panel_groups:
            axes = [axes for placement, axes in rendered_panels
                    if group.contains(placement)]
            if not axes:  # FigureSpec validation normally makes this impossible.
                continue
            if group in heading_axes:
                axes.append(heading_axes[group])
            box = Bbox.union([axes.get_position() for axes in axes])
            bounds.append((group, box))

        for group, box in bounds:
            if not group.separator_before:
                continue
            previous = [
                (other, other_box) for other, other_box in bounds
                if (other.row < group.row + group.row_span and
                    group.row < other.row + other.row_span and
                    other.column + other.column_span <= group.column)]
            if previous:
                _, previous_box = max(
                    previous, key=lambda member: member[0].column +
                    member[0].column_span)
                x = (previous_box.x1 + box.x0) / 2.0
                y0 = min(previous_box.y0, box.y0)
                y1 = max(previous_box.y1, box.y1)
            else:
                x = box.x0 - 0.012
                y0, y1 = box.y0, box.y1
            figure.add_artist(Line2D(
                (x, x), (y0, y1), transform=figure.transFigure,
                color="#666666", linewidth=0.8, alpha=0.4,
                solid_capstyle="round"))

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

        if spec.encoding is SpectralEncoding.PHASE_MAGNITUDE:
            from matplotlib.colors import hsv_to_rgb

            hsv = phase_magnitude_hsv(
                coefficients, spec.normalization, valid_mask=valid,
                phase_offset_radians=spec.phase_offset_radians,
                magnitude_floor_db=spec.magnitude_floor_db)
            display = hsv_to_rgb(hsv)
            display[~valid] = (0.85, 0.85, 0.85)
            axes.imshow(
                display, origin="lower", interpolation="nearest",
                aspect="auto",
                extent=(-0.5, spec.field.l_max + 0.5,
                        -0.5, spec.field.l_max + 0.5))
            axes.set_facecolor("#f2f2f2")
            self._label_spectral_axes(axes, spec)
            return

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
        self._label_spectral_axes(axes, spec)
        unit_suffix = f" [{spec.display_units}]" if spec.display_units else ""
        figure.colorbar(
            image, ax=axes, orientation="vertical",
            label=f"Coefficient magnitude{unit_suffix}")

    @staticmethod
    def _label_spectral_axes(axes, spec: SpectralCoefficientMapSpec) -> None:
        axes.set_title(spec.title)
        axes.set_xlabel("Spherical-harmonic order m")
        axes.set_ylabel("Spherical-harmonic degree l")

    @staticmethod
    def _render_phase_legend(figure, panels) -> None:
        """Add one compact fixed-domain cyclic phase legend per figure."""
        from matplotlib.colors import hsv_to_rgb

        offsets = {panel.phase_offset_radians for panel in panels}
        if len(offsets) != 1:
            raise ValueError(
                "phase-magnitude panels in one figure must share a phase offset")
        offset = offsets.pop()
        phase = np.linspace(-np.pi, np.pi, 512, endpoint=False)[None, :]
        hue = ((phase + offset + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi)
        hsv = np.stack(
            (hue, np.ones_like(hue), np.ones_like(hue)), axis=-1)
        legend_width = min(0.5, 3.0 / figure.get_figwidth())
        legend_height = min(0.04, 0.16 / figure.get_figheight())
        legend_bottom = min(0.04, 0.24 / figure.get_figheight())
        legend = figure.add_axes((
            0.5 - legend_width / 2.0, legend_bottom,
            legend_width, legend_height))
        legend.imshow(
            hsv_to_rgb(hsv), aspect="auto", origin="lower",
            extent=(-np.pi, np.pi, 0.0, 1.0), interpolation="nearest")
        legend.set_yticks(())
        legend.set_xticks((-np.pi, 0.0, np.pi), ("−π", "0", "π"))
        legend.tick_params(axis="x", labelsize=8, length=2, pad=1)
        label = panels[0].encoding_label
        if offset:
            label += f"; palette phase offset = {offset:.6g} rad"
        legend.set_xlabel(label, fontsize=8, labelpad=1)
        for spine in legend.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.5)

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
        _, norm = self._mpl_normalization(spec.normalization, speed)
        stream = axes.streamplot(
            np.rad2deg(longitudes), np.rad2deg(latitudes),
            u_angular, v_angular, color=speed,
            cmap=_SEMANTIC_COLORS.get(spec.color_policy, spec.color_policy),
            norm=norm, density=spec.density, linewidth=1, arrowsize=1.2)
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
