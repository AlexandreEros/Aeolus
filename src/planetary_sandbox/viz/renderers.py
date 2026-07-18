"""Minimal visualization-backend boundary."""
from __future__ import annotations

import pathlib
from typing import Protocol, runtime_checkable

from .specs import FigureSpec, ScalarMapSpec, SpectralCoefficientMapSpec


@runtime_checkable
class Renderer(Protocol):
    """Render declarative specifications without exposing backend objects."""

    def render_scalar_map(self, specification: ScalarMapSpec,
                          output_path: pathlib.Path | str, *,
                          metadata: dict | None = None,
                          dpi: int = 200) -> pathlib.Path:
        ...

    def render_spectral_coefficient_map(
            self, specification: SpectralCoefficientMapSpec,
            output_path: pathlib.Path | str, *, metadata: dict | None = None,
            dpi: int = 200) -> pathlib.Path:
        ...

    def render_figure(self, specification: FigureSpec,
                      output_path: pathlib.Path | str, *,
                      metadata: dict | None = None) -> pathlib.Path:
        ...


def get_default_renderer() -> Renderer:
    """Return the configured initial backend without exposing it to models."""
    from .matplotlib_renderer import MatplotlibRenderer
    return MatplotlibRenderer()
