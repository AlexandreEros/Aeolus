"""Backend-independent visualization data/specifications and renderers."""

from .complex_encoding import (PHASE_DOMAIN, normalized_magnitude_strength,
                               phase_hue, phase_magnitude_hsv,
                               relative_magnitude_db, wrapped_phase)
from .fields import ScalarGridField, SphericalHarmonicField
from .normalization import (NormalizationKind, NormalizationPolicy,
                            ResolvedNormalization)
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, PanelGroup, PanelGroupSpec, ScalarMapSpec,
                    SpectralCoefficientMapSpec, SpectralEncoding,
                    StreamlineMapSpec)
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
    "PHASE_DOMAIN",
    "PanelGroup",
    "PanelGroupSpec",
    "PlanetViewer",
    "Renderer",
    "ResolvedNormalization",
    "ScalarGridField",
    "ScalarMapSpec",
    "SpectralCoefficientMapSpec",
    "SpectralEncoding",
    "StreamlineMapSpec",
    "SphericalHarmonicField",
    "TimelineFrame",
    "build_timeline_overview",
    "get_default_renderer",
    "render_figure_timeline",
    "render_snapshot_product",
    "render_timeline",
    "select_representative_frame_indices",
    "normalized_magnitude_strength",
    "phase_hue",
    "phase_magnitude_hsv",
    "relative_magnitude_db",
    "wrapped_phase",
]


def __getattr__(name: str):
    """Keep the legacy PlanetViewer export lazy (it imports Matplotlib)."""
    if name == "PlanetViewer":
        from .planet_viewer import PlanetViewer
        return PlanetViewer
    raise AttributeError(name)
