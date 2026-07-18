"""Backend-independent visualization data/specifications and renderers."""

from .fields import ScalarGridField, SphericalHarmonicField
from .normalization import (NormalizationKind, NormalizationPolicy,
                            ResolvedNormalization)
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, ScalarMapSpec, SpectralCoefficientMapSpec)
from .timeline import (FigureFrame, FigureTimeline, TimelineFrame,
                       render_figure_timeline, render_timeline)

__all__ = [
    "FigureSpec",
    "FigureFrame",
    "FigureTimeline",
    "NormalizationKind",
    "NormalizationPolicy",
    "PlanetViewer",
    "Renderer",
    "ResolvedNormalization",
    "ScalarGridField",
    "ScalarMapSpec",
    "SpectralCoefficientMapSpec",
    "SphericalHarmonicField",
    "TimelineFrame",
    "get_default_renderer",
    "render_figure_timeline",
    "render_timeline",
]


def __getattr__(name: str):
    """Keep the legacy PlanetViewer export lazy (it imports Matplotlib)."""
    if name == "PlanetViewer":
        from .planet_viewer import PlanetViewer
        return PlanetViewer
    raise AttributeError(name)
