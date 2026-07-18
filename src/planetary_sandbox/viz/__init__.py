"""Backend-independent visualization data/specifications and renderers."""

from .fields import ScalarGridField, SphericalHarmonicField
from .normalization import (NormalizationKind, NormalizationPolicy,
                            ResolvedNormalization)
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, PanelGroup, PanelGroupSpec, ScalarMapSpec,
                    SpectralCoefficientMapSpec, StreamlineMapSpec)
from .timeline import (FigureFrame, FigureTimeline, TimelineFrame,
                       build_timeline_overview, render_figure_timeline,
                       render_snapshot_product, render_timeline,
                       select_representative_frame_indices)

__all__ = [
    "FigureSpec",
    "FigureFrame",
    "FigureTimeline",
    "NormalizationKind",
    "NormalizationPolicy",
    "PanelGroup",
    "PanelGroupSpec",
    "PlanetViewer",
    "Renderer",
    "ResolvedNormalization",
    "ScalarGridField",
    "ScalarMapSpec",
    "SpectralCoefficientMapSpec",
    "StreamlineMapSpec",
    "SphericalHarmonicField",
    "TimelineFrame",
    "build_timeline_overview",
    "get_default_renderer",
    "render_figure_timeline",
    "render_snapshot_product",
    "render_timeline",
    "select_representative_frame_indices",
]


def __getattr__(name: str):
    """Keep the legacy PlanetViewer export lazy (it imports Matplotlib)."""
    if name == "PlanetViewer":
        from .planet_viewer import PlanetViewer
        return PlanetViewer
    raise AttributeError(name)
