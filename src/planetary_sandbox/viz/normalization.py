"""Backend-independent color-normalization policies."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class NormalizationKind(str, Enum):
    AUTO_LINEAR = "auto-linear"
    SYMMETRIC = "symmetric"
    FIXED = "fixed"
    LOG_MAGNITUDE = "log-magnitude"


@dataclass(frozen=True)
class ResolvedNormalization:
    kind: NormalizationKind
    vmin: float
    vmax: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.vmin) and np.isfinite(self.vmax)):
            raise ValueError("normalization limits must be finite")
        if not self.vmin < self.vmax:
            raise ValueError("normalization requires vmin < vmax")
        if self.kind is NormalizationKind.LOG_MAGNITUDE and self.vmin <= 0.0:
            raise ValueError("logarithmic normalization requires vmin > 0")


@dataclass(frozen=True)
class NormalizationPolicy:
    kind: NormalizationKind = NormalizationKind.AUTO_LINEAR
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self) -> None:
        if self.kind is NormalizationKind.FIXED:
            if self.vmin is None or self.vmax is None:
                raise ValueError("fixed normalization requires vmin and vmax")
            ResolvedNormalization(self.kind, float(self.vmin), float(self.vmax))
        else:
            if (self.vmin is None) != (self.vmax is None):
                raise ValueError("normalization limits must be supplied together")
            if self.vmin is not None:
                ResolvedNormalization(
                    self.kind, float(self.vmin), float(self.vmax))

    @classmethod
    def automatic(cls) -> "NormalizationPolicy":
        return cls(NormalizationKind.AUTO_LINEAR)

    @classmethod
    def symmetric(cls) -> "NormalizationPolicy":
        return cls(NormalizationKind.SYMMETRIC)

    @classmethod
    def fixed(cls, vmin: float, vmax: float) -> "NormalizationPolicy":
        return cls(NormalizationKind.FIXED, float(vmin), float(vmax))

    @classmethod
    def logarithmic_magnitude(
            cls, vmin: float | None = None,
            vmax: float | None = None) -> "NormalizationPolicy":
        """Log magnitude scaling, optionally with reusable fixed limits."""
        return cls(NormalizationKind.LOG_MAGNITUDE, vmin, vmax)

    @classmethod
    def from_resolved(
            cls, normalization: ResolvedNormalization
            ) -> "NormalizationPolicy":
        """Keep semantic scaling while freezing already-resolved limits."""
        return cls(normalization.kind, normalization.vmin, normalization.vmax)

    def resolve(self, values) -> ResolvedNormalization:
        """Resolve numeric limits from any state or full time sequence.

        Passing an entire time sequence resolves one pair of limits that can
        subsequently be frozen with :meth:`from_resolved` for every frame.
        """
        data = np.asarray(values)
        if self.vmin is not None:
            return ResolvedNormalization(
                self.kind, float(self.vmin), float(self.vmax))

        if self.kind is NormalizationKind.LOG_MAGNITUDE:
            magnitude = np.abs(data)
            finite_positive = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
            if finite_positive.size == 0:
                return ResolvedNormalization(self.kind, 1.0e-12, 1.0)
            lo = float(np.min(finite_positive))
            hi = float(np.max(finite_positive))
            if lo == hi or _nearly_equal_logarithmically(lo, hi):
                lower = lo / 10.0
                upper = hi
                if lower <= 0.0:
                    lower = lo
                if not lower < upper:
                    upper = float(np.nextafter(lower, np.inf))
                return ResolvedNormalization(self.kind, lower, upper)
            return ResolvedNormalization(self.kind, lo, hi)

        real = np.real(data)
        finite = real[np.isfinite(real)]
        if finite.size == 0:
            raise ValueError("cannot normalize data without finite values")

        if self.kind is NormalizationKind.SYMMETRIC:
            bound = float(np.max(np.abs(finite)))
            if bound <= 1.0e-12:
                bound = 1.0
            return ResolvedNormalization(self.kind, -bound, bound)

        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if _nearly_equal(lo, hi):
            midpoint = 0.5 * (lo + hi)
            padding = max(abs(midpoint) * 1.0e-6, 1.0e-12)
            lo, hi = midpoint - padding, midpoint + padding
        return ResolvedNormalization(self.kind, lo, hi)


def _nearly_equal(lo: float, hi: float) -> bool:
    scale = max(1.0, abs(lo), abs(hi))
    return abs(hi - lo) <= 1.0e-12 * scale


def _nearly_equal_logarithmically(lo: float, hi: float) -> bool:
    """Detect a numerically degenerate positive interval in log space."""
    log_lo = float(np.log(lo))
    log_hi = float(np.log(hi))
    scale = max(1.0, abs(log_lo), abs(log_hi))
    return abs(log_hi - log_lo) <= np.finfo(np.float64).eps * scale
