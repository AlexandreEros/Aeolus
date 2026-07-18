"""Backend-neutral, timestamped figure sequences.

A timeline contains only declarative :class:`~planetary_sandbox.viz.specs.FigureSpec`
objects.  It resolves color limits across the complete sequence before any
backend is invoked, then renders into a staging directory and publishes the
complete set.  A failed encoder therefore neither exposes a partial sequence
nor replaces a previously complete frame.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import os
import pathlib
import re
import shutil
import tempfile

import numpy as np

from .normalization import NormalizationPolicy
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, ScalarMapSpec,
                    SpectralCoefficientMapSpec, StreamlineMapSpec)


_TIME_DECIMAL_PLACES = 9
_TIME_INTEGER_PLACES = 13
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class FigureFrame:
    """One figure at an authoritative simulation time in seconds."""

    time_seconds: float
    specification: FigureSpec

    def __post_init__(self) -> None:
        time = float(self.time_seconds)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("figure-frame time must be finite and nonnegative")
        object.__setattr__(self, "time_seconds", time)


# A convenient spelling for callers that think in terms of frames first.
TimelineFrame = FigureFrame


@dataclass(frozen=True)
class FigureTimeline:
    """An ordered sequence of figures with shared normalization semantics.

    Panels are matched across frames by an explicit ``normalization_group``
    when present, otherwise by their type and grid placement.  Each group
    receives limits resolved from every frame in the group.  The policy kind
    (symmetric, logarithmic, and so on) is retained after its limits freeze.
    """

    frames: tuple[FigureFrame, ...]
    filename_prefix: str

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("a figure timeline requires at least one frame")
        if not isinstance(self.filename_prefix, str) or not _PREFIX_RE.fullmatch(
                self.filename_prefix):
            raise ValueError(
                "timeline filename prefix must contain only letters, digits, "
                "dots, underscores, and hyphens")
        times = np.asarray([frame.time_seconds for frame in frames])
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("figure-frame times must be strictly increasing")
        names = [self.filename_for(frame) for frame in frames]
        if len(set(names)) != len(names):
            raise ValueError(
                "figure-frame times are too close for deterministic filename "
                f"precision ({_TIME_DECIMAL_PLACES} decimal places)")
        object.__setattr__(self, "frames", frames)

    @classmethod
    def from_figures(cls, times_seconds, specifications, *,
                     filename_prefix: str) -> "FigureTimeline":
        """Build a timeline from parallel time and figure sequences."""
        times = tuple(times_seconds)
        figures = tuple(specifications)
        if len(times) != len(figures):
            raise ValueError("timeline times and figures must have equal length")
        return cls(tuple(FigureFrame(t, figure)
                         for t, figure in zip(times, figures)),
                   filename_prefix)

    @property
    def times_seconds(self) -> np.ndarray:
        return np.asarray([frame.time_seconds for frame in self.frames],
                          dtype=np.float64)

    @property
    def specifications(self) -> tuple[FigureSpec, ...]:
        return tuple(frame.specification for frame in self.frames)

    def filename_for(self, frame_or_index: FigureFrame | int) -> str:
        frame = (self.frames[frame_or_index]
                 if isinstance(frame_or_index, int) else frame_or_index)
        width = _TIME_INTEGER_PLACES + 1 + _TIME_DECIMAL_PLACES
        token = f"{frame.time_seconds:0{width}.{_TIME_DECIMAL_PLACES}f}"
        return f"{self.filename_prefix}_t{token}s.png"

    def resolve_normalizations(self) -> "FigureTimeline":
        """Return a copy with every cross-frame normalization frozen."""
        grouped: dict[tuple, list[tuple[int, int, object, np.ndarray]]] = {}
        for frame_index, frame in enumerate(self.frames):
            for panel_index, placement in enumerate(frame.specification.panels):
                panel = placement.panel
                values = _normalizable_values(panel)
                if values is None:
                    continue
                group = getattr(panel, "normalization_group", None)
                key = (("named", group) if group is not None else
                       ("placement", type(panel), placement.row,
                        placement.column, placement.row_span,
                        placement.column_span))
                grouped.setdefault(key, []).append(
                    (frame_index, panel_index, panel, values))

        replacements: dict[tuple[int, int], object] = {}
        for key, members in grouped.items():
            policies = [member[2].normalization for member in members]
            signature = {(policy.kind, policy.vmin, policy.vmax)
                         for policy in policies}
            if len(signature) != 1:
                raise ValueError(
                    f"normalization group {key!r} uses inconsistent policies")
            combined = np.concatenate(
                [np.asarray(member[3]).reshape(-1) for member in members])
            resolved = policies[0].resolve(combined)
            frozen = NormalizationPolicy.from_resolved(resolved)
            for frame_index, panel_index, panel, _ in members:
                replacements[(frame_index, panel_index)] = replace(
                    panel, normalization=frozen)

        frames = []
        for frame_index, frame in enumerate(self.frames):
            placements = tuple(
                replace(placement, panel=replacements.get(
                    (frame_index, panel_index), placement.panel))
                for panel_index, placement in enumerate(
                    frame.specification.panels))
            frames.append(replace(
                frame, specification=replace(
                    frame.specification, panels=placements)))
        return replace(self, frames=tuple(frames))

    def matches_filename(self, filename: str) -> bool:
        """Whether ``filename`` belongs to this timeline's generated set."""
        width = _TIME_INTEGER_PLACES + 1 + _TIME_DECIMAL_PLACES
        pattern = (rf"{re.escape(self.filename_prefix)}_t"
                   rf"\d{{{_TIME_INTEGER_PLACES}}}\.\d{{{_TIME_DECIMAL_PLACES}}}"
                   rf"s\.png\Z")
        # Keep the width calculation beside the pattern as a format invariant.
        assert width == 23
        return re.fullmatch(pattern, filename) is not None


def _normalizable_values(panel) -> np.ndarray | None:
    if isinstance(panel, ScalarMapSpec):
        return np.asarray(panel.field.values_at(panel.time_index))
    if isinstance(panel, SpectralCoefficientMapSpec):
        coefficients = panel.field.coefficients_at(panel.time_index)
        return np.abs(coefficients[panel.field.valid_mask])
    if isinstance(panel, StreamlineMapSpec):
        return np.sqrt(panel.zonal_velocity**2 + panel.meridional_velocity**2)
    return None


def render_figure_timeline(
        timeline: FigureTimeline, output_dir: pathlib.Path | str, *,
        renderer: Renderer | None = None, metadata: dict | None = None
        ) -> tuple[pathlib.Path, ...]:
    """Render and transactionally publish all frames in ``timeline``.

    Rendering happens entirely in a same-filesystem staging directory.  Only
    after every frame exists are old generated frames backed up and the new
    set moved into place.  Publication errors are rolled back best-effort;
    rendering errors leave the prior complete sequence untouched.
    """
    backend = renderer or get_default_renderer()
    resolved = timeline.resolve_normalizations()
    destination_dir = pathlib.Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(
        prefix=f".{timeline.filename_prefix}.timeline-",
        dir=destination_dir))
    staged: list[pathlib.Path] = []
    destinations = tuple(
        destination_dir / resolved.filename_for(frame)
        for frame in resolved.frames)
    try:
        for frame in resolved.frames:
            target = stage / resolved.filename_for(frame)
            backend.render_figure(
                frame.specification, target, metadata=metadata)
            if not target.is_file():
                raise RuntimeError(
                    f"renderer did not create requested timeline frame {target.name}")
            staged.append(target)

        previous = tuple(
            child for child in destination_dir.iterdir()
            if child.is_file() and resolved.matches_filename(child.name))
        backup_dir = stage / ".previous"
        backup_dir.mkdir()
        backed_up: list[tuple[pathlib.Path, pathlib.Path]] = []
        published: list[pathlib.Path] = []
        try:
            for old in previous:
                backup = backup_dir / old.name
                os.replace(old, backup)
                backed_up.append((backup, old))
            for source, destination in zip(staged, destinations):
                os.replace(source, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                with contextlib.suppress(OSError):
                    destination.unlink()
            for backup, original in reversed(backed_up):
                with contextlib.suppress(OSError):
                    os.replace(backup, original)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return destinations


# Short alias for callers that already establish the figure nature in context.
render_timeline = render_figure_timeline


__all__ = [
    "FigureFrame",
    "FigureTimeline",
    "TimelineFrame",
    "render_figure_timeline",
    "render_timeline",
]
