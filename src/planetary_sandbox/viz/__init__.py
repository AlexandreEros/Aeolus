"""Backend-independent visualization data/specifications and renderers."""

from .fields import ScalarGridField, SphericalHarmonicField
from .normalization import (NormalizationKind, NormalizationPolicy,
                            ResolvedNormalization)
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, ScalarMapSpec, SpectralCoefficientMapSpec)

__all__ = [
    "FigureSpec",
    "NormalizationKind",
    "NormalizationPolicy",
    "PlanetViewer",
    "Renderer",
    "ResolvedNormalization",
    "ScalarGridField",
    "ScalarMapSpec",
    "SpectralCoefficientMapSpec",
    "SphericalHarmonicField",
    "get_default_renderer",
]


def __getattr__(name: str):
    """Keep the legacy PlanetViewer export lazy (it imports Matplotlib)."""
    if name == "PlanetViewer":
        from .planet_viewer import PlanetViewer
        return PlanetViewer
    raise AttributeError(name)
