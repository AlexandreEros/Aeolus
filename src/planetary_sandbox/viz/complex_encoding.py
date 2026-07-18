"""Backend-neutral phase and magnitude encoding for complex fields."""
from __future__ import annotations

import math

import numpy as np

from .normalization import NormalizationPolicy


PHASE_DOMAIN = (-math.pi, math.pi)


def wrapped_phase(coefficients, *, phase_offset_radians: float = 0.0
                  ) -> np.ndarray:
    """Return ``arg(C) + offset`` wrapped into the fixed ``[-pi, pi)`` domain."""
    if not math.isfinite(phase_offset_radians):
        raise ValueError("phase offset must be finite")
    phase = np.angle(np.asarray(coefficients)) + phase_offset_radians
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def phase_hue(coefficients, *, phase_offset_radians: float = 0.0
              ) -> np.ndarray:
    """Map fixed-domain complex phase to cyclic hue in ``[0, 1)``."""
    phase = wrapped_phase(
        coefficients, phase_offset_radians=phase_offset_radians)
    return (phase + math.pi) / (2.0 * math.pi)


def normalized_magnitude_strength(
        magnitudes, normalization: NormalizationPolicy, *,
        reference_values=None, floor_db: float = -60.0) -> np.ndarray:
    """Map relative magnitude in decibels to saturation strength.

    The resolved normalization's maximum is the reference amplitude. Zero,
    non-finite, and below-floor magnitudes deterministically map to zero.
    """
    db = relative_magnitude_db(
        magnitudes, normalization, reference_values=reference_values)
    if not math.isfinite(floor_db) or floor_db >= 0.0:
        raise ValueError("magnitude dB floor must be finite and negative")
    return np.clip((db - floor_db) / -floor_db, 0.0, 1.0)


def relative_magnitude_db(
        magnitudes, normalization: NormalizationPolicy, *,
        reference_values=None) -> np.ndarray:
    """Return amplitude decibels relative to the resolved maximum."""
    magnitude = np.abs(np.asarray(magnitudes, dtype=np.float64))
    values = magnitude if reference_values is None else np.abs(
        np.asarray(reference_values, dtype=np.float64))
    if normalization.vmax is not None:
        reference = float(normalization.vmax)
    else:
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        reference = (float(np.max(finite_positive))
                     if finite_positive.size else 1.0)
    db = np.full_like(magnitude, -np.inf, dtype=np.float64)
    positive = np.isfinite(magnitude) & (magnitude > 0.0)
    db[positive] = 20.0 * (
        np.log10(magnitude[positive]) - math.log10(reference))
    return db


def phase_magnitude_hsv(
        coefficients, normalization: NormalizationPolicy, *,
        valid_mask: np.ndarray | None = None,
        phase_offset_radians: float = 0.0,
        magnitude_floor_db: float = -60.0) -> np.ndarray:
    """Encode HSV with phase hue and relative-amplitude-dB saturation.

    ``valid_mask`` affects normalization resolution only. Mask presentation is
    deliberately left to the renderer so invalid coefficient slots remain
    distinct from valid zero coefficients.
    """
    values = np.asarray(coefficients)
    magnitude = np.abs(values)
    reference = magnitude if valid_mask is None else magnitude[
        np.asarray(valid_mask, dtype=bool)]
    hue = phase_hue(
        values, phase_offset_radians=phase_offset_radians)
    saturation = normalized_magnitude_strength(
        magnitude, normalization, reference_values=reference,
        floor_db=magnitude_floor_db)
    return np.stack((hue, saturation, np.ones_like(hue)), axis=-1)


__all__ = [
    "PHASE_DOMAIN",
    "normalized_magnitude_strength",
    "phase_hue",
    "phase_magnitude_hsv",
    "relative_magnitude_db",
    "wrapped_phase",
]
