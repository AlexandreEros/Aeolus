"""Declarative visualization specifications with no backend objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TypeAlias

import numpy as np

from .fields import ScalarGridField, SphericalHarmonicField
from .normalization import NormalizationPolicy


class SpectralEncoding(str, Enum):
    """Supported visual encodings for complex spectral coefficients."""

    PHASE_MAGNITUDE = "phase-magnitude"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class ScalarMapSpec:
    field: ScalarGridField
    title: str
    time_index: int = 0
    units: str | None = None
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    view: str = "equirectangular"
    central_longitude: float | None = None
    color_policy: str = "viridis"
    normalization_group: str | None = None

    def __post_init__(self) -> None:
        self.field.values_at(self.time_index)
        if not self.title:
            raise ValueError("map title must be nonempty")
        if not isinstance(self.view, str) or not self.view:
            raise ValueError("map view identifier must be nonempty")
        if self.central_longitude is not None and not np.isfinite(
                self.central_longitude):
            raise ValueError("central longitude must be finite")
        if not self.color_policy:
            raise ValueError("color policy must be nonempty")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")

    @property
    def display_units(self) -> str:
        return self.field.units if self.units is None else self.units


@dataclass(frozen=True)
class SpectralCoefficientMapSpec:
    """A fixed-range triangular coefficient map with explicit complex encoding.

    The canonical phase-magnitude encoding uses the argument of the stored
    coefficient with no implicit sign flip. Hue spans the fixed ``[-pi, pi)``
    phase domain; timeline-relative amplitude dB controls saturation.
    """

    field: SphericalHarmonicField
    title: str
    time_index: int = 0
    units: str | None = None
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.logarithmic_magnitude)
    color_policy: str = "viridis"
    normalization_group: str | None = None
    encoding: SpectralEncoding | str = SpectralEncoding.PHASE_MAGNITUDE
    phase_offset_radians: float = 0.0
    magnitude_floor_db: float = -60.0

    def __post_init__(self) -> None:
        self.field.coefficients_at(self.time_index)
        if not self.title:
            raise ValueError("coefficient-map title must be nonempty")
        try:
            encoding = SpectralEncoding(self.encoding)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"unsupported spectral encoding {self.encoding!r}") from err
        if not math.isfinite(self.phase_offset_radians):
            raise ValueError("spectral phase offset must be finite")
        if (not math.isfinite(self.magnitude_floor_db) or
                self.magnitude_floor_db >= 0.0):
            raise ValueError("spectral magnitude dB floor must be finite and negative")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")
        object.__setattr__(self, "encoding", encoding)

    @property
    def display_units(self) -> str:
        return self.field.units if self.units is None else self.units

    @property
    def encoding_label(self) -> str:
        if self.encoding is SpectralEncoding.PHASE_MAGNITUDE:
            return (
                "Hue = phase; saturation = relative magnitude "
                f"[{self.magnitude_floor_db:g}, 0] dB")
        return "Color = |C_lm|"

    @property
    def convention_metadata(self) -> dict[str, object]:
        """Basis and palette conventions needed to interpret coefficient color."""
        return {
            "encoding": self.encoding.value,
            "phase_definition": "arg(C_lm)",
            "phase_domain": "[-pi, pi)",
            "phase_offset_radians": float(self.phase_offset_radians),
            "magnitude_floor_db": float(self.magnitude_floor_db),
            "coefficient_normalization": self.field.normalization,
            "coefficient_layout": self.field.layout,
            "longitude_origin_radians": float(
                self.field.longitude_origin_radians),
            "magnitude_mapping": (
                "amplitude dB relative to timeline maximum, mapped to saturation"),
        }


@dataclass(frozen=True)
class StreamlineMapSpec:
    latitudes: np.ndarray
    longitudes: np.ndarray
    zonal_velocity: np.ndarray
    meridional_velocity: np.ndarray
    radius: float
    title: str
    units: str = "m/s"
    density: float = 1.5
    color_policy: str = "viridis"
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    normalization_group: str | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitudes)
        lon = np.asarray(self.longitudes)
        u = np.asarray(self.zonal_velocity)
        v = np.asarray(self.meridional_velocity)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("streamline coordinates must be one-dimensional")
        if u.shape != v.shape or u.shape != (lat.size, lon.size):
            raise ValueError("streamline velocities must have shape (lat, lon)")
        if lat.size > 1 and not np.all(np.diff(lat) < 0.0):
            raise ValueError("streamline latitudes must be north-to-south")
        if lon.size > 1 and not np.all(np.diff(lon) > 0.0):
            raise ValueError("streamline longitudes must be increasing")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("streamline radius must be finite and positive")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")
        object.__setattr__(self, "latitudes", lat)
        object.__setattr__(self, "longitudes", lon)
        object.__setattr__(self, "zonal_velocity", u)
        object.__setattr__(self, "meridional_velocity", v)


@dataclass(frozen=True)
class TextPanelSpec:
    text: str
    font_family: str = "monospace"
    font_size: float = 11.0
    horizontal_alignment: str = "center"


@dataclass(frozen=True)
class LineSeriesSpec:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: str = "k"
    line_style: str = "-"
    line_width: float = 1.5

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("line-series x and y must be equal-length 1D arrays")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True)
class LinePanelSpec:
    series: tuple[LineSeriesSpec, ...]
    title: str
    x_label: str
    y_label: str
    y_limits: tuple[float, float] | None = None
    show_grid: bool = True
    show_legend: bool = True


PanelSpec: TypeAlias = (
    ScalarMapSpec | SpectralCoefficientMapSpec | StreamlineMapSpec |
    TextPanelSpec | LinePanelSpec)


@dataclass(frozen=True)
class PanelPlacement:
    panel: PanelSpec
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1


@dataclass(frozen=True)
class PanelGroupSpec:
    """A labeled rectangular group of scientifically related panels.

    The group describes layout and presentation only. Scientific role names
    and the decision about which fields belong together remain adapter-owned.
    """

    title: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    separator_before: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("panel-group title must be a nonempty string")
        if (not isinstance(self.separator_before, bool) or
                any(not isinstance(value, int) or isinstance(value, bool)
                    for value in (self.row, self.column, self.row_span,
                                  self.column_span))):
            raise TypeError("panel-group placement values must be integers")
        if (self.row < 0 or self.column < 0 or self.row_span < 1 or
                self.column_span < 1):
            raise ValueError("panel-group placement must be positive")

    def contains(self, placement: PanelPlacement) -> bool:
        """Whether a panel placement lies completely inside this group."""
        return (
            placement.row >= self.row and
            placement.column >= self.column and
            placement.row + placement.row_span <= self.row + self.row_span and
            placement.column + placement.column_span <=
            self.column + self.column_span)


# Concise public spelling for callers that do not need the Spec suffix.
PanelGroup = PanelGroupSpec


@dataclass(frozen=True)
class FigureSpec:
    panels: tuple[PanelPlacement, ...]
    rows: int
    columns: int
    size_inches: tuple[float, float]
    dpi: int = 200
    width_ratios: tuple[float, ...] | None = None
    height_ratios: tuple[float, ...] | None = None
    tight_layout: bool = True
    panel_groups: tuple[PanelGroupSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("figure layout dimensions must be positive")
        if self.dpi < 1:
            raise ValueError("figure dpi must be positive")
        if self.width_ratios is not None and len(self.width_ratios) != self.columns:
            raise ValueError("width-ratio count must equal figure columns")
        if self.height_ratios is not None and len(self.height_ratios) != self.rows:
            raise ValueError("height-ratio count must equal figure rows")
        groups = tuple(self.panel_groups)
        occupied: set[tuple[int, int]] = set()
        for group in groups:
            if not isinstance(group, PanelGroupSpec):
                raise TypeError("figure panel_groups must contain PanelGroupSpec")
            if (group.row + group.row_span > self.rows or
                    group.column + group.column_span > self.columns):
                raise ValueError(
                    f"panel group is outside figure grid: {group}")
            cells = {
                (row, column)
                for row in range(group.row, group.row + group.row_span)
                for column in range(
                    group.column, group.column + group.column_span)}
            if occupied.intersection(cells):
                raise ValueError("panel groups must not overlap")
            occupied.update(cells)
        for placement in self.panels:
            if (placement.row < 0 or placement.column < 0 or
                    placement.row_span < 1 or placement.column_span < 1 or
                    placement.row + placement.row_span > self.rows or
                    placement.column + placement.column_span > self.columns):
                raise ValueError(f"panel placement is outside figure grid: {placement}")
        for group in groups:
            if not any(group.contains(placement) for placement in self.panels):
                raise ValueError(
                    f"panel group contains no panels: {group.title!r}")
        object.__setattr__(self, "panel_groups", groups)
