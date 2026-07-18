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
from typing import Mapping

import numpy as np

from .normalization import NormalizationPolicy
from .renderers import Renderer, get_default_renderer
from .specs import (FigureSpec, PanelPlacement, ScalarMapSpec,
                    SpectralCoefficientMapSpec, StreamlineMapSpec,
                    TextPanelSpec)


_TIME_DECIMAL_PLACES = 9
_TIME_INTEGER_PLACES = 13
_PRODUCT_TIME_INTEGER_PLACES = 6
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_REPRESENTATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


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
    filename_prefix: str | None

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if (self.filename_prefix is not None and
                (not isinstance(self.filename_prefix, str) or
                 not _PREFIX_RE.fullmatch(self.filename_prefix))):
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
                     filename_prefix: str | None) -> "FigureTimeline":
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
        if self.filename_prefix is None:
            fixed = f"{frame.time_seconds:.{_TIME_DECIMAL_PLACES}f}"
            integer, fraction = fixed.split(".")
            token = integer.zfill(_PRODUCT_TIME_INTEGER_PLACES)
            if int(fraction):
                token += f".{fraction}"
            return f"t{token}s.png"

        width = _TIME_INTEGER_PLACES + 1 + _TIME_DECIMAL_PLACES
        token = f"{frame.time_seconds:0{width}.{_TIME_DECIMAL_PLACES}f}"
        return f"{self.filename_prefix}_t{token}s.png"

    def representative_indices(self, *, max_frames: int = 5) -> tuple[int, ...]:
        """Indices of the physical-time representatives for an overview."""
        return select_representative_frame_indices(
            self.times_seconds, max_frames=max_frames)

    def overview_specification(self, *,
                               max_frames: int = 5) -> FigureSpec | None:
        """Vertically assemble representative, fully normalized frames.

        Normalizations are first resolved from the complete timeline.  The
        representative subset therefore never acquires its own, narrower
        color scale.
        """
        if not self.frames:
            return None
        resolved = self.resolve_normalizations()
        return _assemble_overview(
            resolved, resolved.representative_indices(max_frames=max_frames))

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
        if self.filename_prefix is None:
            return re.fullmatch(
                rf"t\d{{{_PRODUCT_TIME_INTEGER_PLACES},}}"
                rf"(?:\.\d{{{_TIME_DECIMAL_PLACES}}})?s\.png\Z",
                filename) is not None
        width = _TIME_INTEGER_PLACES + 1 + _TIME_DECIMAL_PLACES
        pattern = (rf"{re.escape(self.filename_prefix)}_t"
                   rf"\d{{{_TIME_INTEGER_PLACES}}}\.\d{{{_TIME_DECIMAL_PLACES}}}"
                   rf"s\.png\Z")
        # Keep the width calculation beside the pattern as a format invariant.
        assert width == 23
        return re.fullmatch(pattern, filename) is not None


def select_representative_frame_indices(
        times_seconds, *, max_frames: int = 5) -> tuple[int, ...]:
    """Select up to ``max_frames`` approximately uniformly in physical time.

    For a long, irregular schedule this minimizes total distance from evenly
    spaced target times while requiring distinct, chronological snapshots.
    The endpoints are fixed, so the first and last stored states are always
    represented.
    """
    if (not isinstance(max_frames, int) or isinstance(max_frames, bool) or
            max_frames < 2):
        raise ValueError("max_frames must be an integer of at least two")
    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("snapshot times must be one-dimensional")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("snapshot times must be finite and nonnegative")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("snapshot times must be strictly increasing")
    count = times.size
    if count <= max_frames:
        return tuple(range(count))

    targets = np.linspace(times[0], times[-1], max_frames)
    costs = np.full((max_frames, count), np.inf)
    previous = np.full((max_frames, count), -1, dtype=np.int64)
    costs[0, 0] = 0.0

    # Dynamic programming gives the nearest *distinct chronological* set,
    # including sensible behavior when several targets share one nearest
    # snapshot in a strongly clustered irregular schedule.
    for target_index in range(1, max_frames - 1):
        first = target_index
        last = count - (max_frames - target_index)
        best_predecessor = -1
        best_cost = np.inf
        for snapshot_index in range(first, last + 1):
            candidate = snapshot_index - 1
            candidate_cost = costs[target_index - 1, candidate]
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_predecessor = candidate
            if np.isfinite(best_cost):
                costs[target_index, snapshot_index] = (
                    best_cost +
                    abs(times[snapshot_index] - targets[target_index]))
                previous[target_index, snapshot_index] = best_predecessor

    final_predecessor = int(np.argmin(costs[max_frames - 2, :count - 1]))
    costs[max_frames - 1, count - 1] = costs[
        max_frames - 2, final_predecessor]
    previous[max_frames - 1, count - 1] = final_predecessor

    selected = [count - 1]
    for target_index in range(max_frames - 1, 0, -1):
        predecessor = int(previous[target_index, selected[-1]])
        if predecessor < 0:  # pragma: no cover - guarded by count > max_frames
            raise RuntimeError("could not select representative timeline frames")
        selected.append(predecessor)
    return tuple(reversed(selected))


def _physical_time_label(time_seconds: float) -> str:
    seconds = np.format_float_positional(
        time_seconds, precision=_TIME_DECIMAL_PLACES, trim="-")
    if time_seconds >= 3600.0:
        hours = np.format_float_positional(
            time_seconds / 3600.0, precision=6, trim="-")
        return f"Physical time: {seconds} s ({hours} h)"
    return f"Physical time: {seconds} s"


def _assemble_overview(
        resolved: FigureTimeline, indices: tuple[int, ...]) -> FigureSpec:
    """Stack already-resolved model frames without changing their contents."""
    if not indices:
        raise ValueError("a timeline overview requires at least one frame")
    figures = [resolved.frames[index].specification for index in indices]
    columns = figures[0].columns
    width_ratios = figures[0].width_ratios
    if any(figure.columns != columns or figure.width_ratios != width_ratios
           for figure in figures[1:]):
        raise ValueError(
            "timeline overview frames must use a consistent column layout")

    label_height = 0.35
    placements = []
    panel_groups = []
    height_ratios: list[float] = []
    row_offset = 0
    for index, figure in zip(indices, figures):
        placements.append(PanelPlacement(
            TextPanelSpec(
                _physical_time_label(resolved.frames[index].time_seconds),
                font_family="sans-serif", font_size=12.0),
            row_offset, 0, column_span=columns))
        height_ratios.append(label_height)
        row_offset += 1

        source_ratios = figure.height_ratios or (1.0,) * figure.rows
        ratio_total = float(sum(source_ratios))
        height_ratios.extend(
            figure.size_inches[1] * ratio / ratio_total
            for ratio in source_ratios)
        placements.extend(
            replace(placement, row=placement.row + row_offset)
            for placement in figure.panels)
        panel_groups.extend(
            replace(group, row=group.row + row_offset)
            for group in figure.panel_groups)
        row_offset += figure.rows

    return FigureSpec(
        panels=tuple(placements), rows=row_offset, columns=columns,
        size_inches=(max(figure.size_inches[0] for figure in figures),
                     sum(figure.size_inches[1] for figure in figures) +
                     label_height * len(figures)),
        dpi=max(figure.dpi for figure in figures),
        width_ratios=width_ratios, height_ratios=tuple(height_ratios),
        panel_groups=tuple(panel_groups))


def build_timeline_overview(
        timeline: FigureTimeline, *, max_frames: int = 5) -> FigureSpec | None:
    """Build a generic representative overview from a complete timeline."""
    return timeline.overview_specification(max_frames=max_frames)


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
        renderer: Renderer | None = None, metadata: dict | None = None,
        overview_filename: str | None = "timeline.png"
        ) -> tuple[pathlib.Path, ...]:
    """Render and transactionally publish all frames and their overview.

    Rendering happens entirely in a same-filesystem staging directory.  Only
    after every frame and the representative overview exist are old generated
    images backed up and the new set moved into place. Publication errors are
    rolled back best-effort; rendering errors leave the prior complete
    sequence untouched. The returned paths remain the complete-frame paths;
    the overview has the stable name ``timeline.png`` by default.
    """
    if (overview_filename is not None and
            (pathlib.Path(overview_filename).name != overview_filename or
             not overview_filename.lower().endswith(".png"))):
        raise ValueError("overview filename must be a plain PNG filename")
    backend = renderer or get_default_renderer()
    resolved = timeline.resolve_normalizations()
    destination_dir = pathlib.Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(
        prefix=f".{timeline.filename_prefix or 'frames'}.timeline-",
        dir=destination_dir))
    staged: list[pathlib.Path] = []
    destinations = tuple(
        destination_dir / resolved.filename_for(frame)
        for frame in resolved.frames)
    overview_destination = (
        None if overview_filename is None else
        destination_dir / overview_filename)
    try:
        for frame in resolved.frames:
            target = stage / resolved.filename_for(frame)
            backend.render_figure(
                frame.specification, target, metadata=metadata)
            if not target.is_file():
                raise RuntimeError(
                    f"renderer did not create requested timeline frame {target.name}")
            staged.append(target)

        if overview_filename is not None and resolved.frames:
            overview_target = stage / overview_filename
            backend.render_figure(
                _assemble_overview(
                    resolved, resolved.representative_indices(max_frames=5)),
                overview_target, metadata=metadata)
            if not overview_target.is_file():
                raise RuntimeError(
                    "renderer did not create requested timeline overview "
                    f"{overview_target.name}")
            staged.append(overview_target)

        previous = tuple(
            child for child in destination_dir.iterdir()
            if child.is_file() and (
                resolved.matches_filename(child.name) or
                (overview_filename is not None and
                 child.name == overview_filename)))
        backup_dir = stage / ".previous"
        backup_dir.mkdir()
        backed_up: list[tuple[pathlib.Path, pathlib.Path]] = []
        published: list[pathlib.Path] = []
        try:
            for old in previous:
                backup = backup_dir / old.name
                os.replace(old, backup)
                backed_up.append((backup, old))
            artifact_destinations = list(destinations)
            if overview_destination is not None and resolved.frames:
                artifact_destinations.append(overview_destination)
            for source, destination in zip(staged, artifact_destinations):
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


def render_snapshot_product(
        timelines: Mapping[str, FigureTimeline],
        run_capsule: pathlib.Path | str, *, renderer: Renderer | None = None,
        metadata: dict | None = None,
        directory_name: str = "snapshots"
        ) -> dict[str, tuple[pathlib.Path, ...]]:
    """Render and atomically publish a multi-representation snapshot product.

    All representations must describe exactly the same authoritative stored
    times. They are fully rendered below one same-filesystem staging
    directory before the complete ``snapshots/`` directory replaces its
    predecessor, so a failed physical or spectral render cannot expose a
    mixed-generation product.
    """
    if not timelines:
        raise ValueError("a snapshot product requires an enabled representation")
    if (pathlib.Path(directory_name).name != directory_name or
            not _REPRESENTATION_RE.fullmatch(directory_name)):
        raise ValueError("snapshot product directory name must be a plain name")

    representations = dict(timelines)
    for name in representations:
        if not isinstance(name, str) or not _REPRESENTATION_RE.fullmatch(name):
            raise ValueError(
                "snapshot representation names may contain only letters, "
                "digits, dots, underscores, and hyphens")
    first_timeline = next(iter(representations.values()))
    authoritative_times = first_timeline.times_seconds
    for name, timeline in representations.items():
        if not np.array_equal(timeline.times_seconds, authoritative_times):
            raise ValueError(
                f"snapshot representation {name!r} does not use the "
                "authoritative persisted times")

    backend = renderer or get_default_renderer()
    capsule = pathlib.Path(run_capsule)
    capsule.mkdir(parents=True, exist_ok=True)
    destination = capsule / directory_name
    if destination.exists() and not destination.is_dir():
        raise ValueError(
            f"snapshot product destination is not a directory: {destination}")
    stage_root = pathlib.Path(tempfile.mkdtemp(
        prefix=f".{directory_name}.product-", dir=capsule))
    staged_product = stage_root / directory_name
    staged_product.mkdir()
    published_paths: dict[str, tuple[pathlib.Path, ...]] = {}

    try:
        for name, timeline in representations.items():
            # Representation directories make a scenario prefix redundant.
            # The prefix-free form also yields the product contract's compact
            # t000000s.png naming while retaining nanosecond precision when a
            # persisted time is fractional.
            product_timeline = replace(timeline, filename_prefix=None)
            render_figure_timeline(
                product_timeline, staged_product / name, renderer=backend,
                metadata=metadata)
            frame_paths = tuple(
                destination / name / product_timeline.filename_for(frame)
                for frame in product_timeline.frames)
            published_paths[name] = frame_paths + (
                (destination / name / "timeline.png",)
                if product_timeline.frames else ())

        previous = stage_root / f".{directory_name}.previous"
        moved_previous = False
        try:
            if destination.exists():
                os.replace(destination, previous)
                moved_previous = True
            os.replace(staged_product, destination)
        except BaseException:
            if moved_previous:
                with contextlib.suppress(OSError):
                    os.replace(previous, destination)
            raise
        if moved_previous:
            shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return published_paths


# Short alias for callers that already establish the figure nature in context.
render_timeline = render_figure_timeline


__all__ = [
    "build_timeline_overview",
    "FigureFrame",
    "FigureTimeline",
    "TimelineFrame",
    "render_figure_timeline",
    "render_snapshot_product",
    "render_timeline",
    "select_representative_frame_indices",
]
