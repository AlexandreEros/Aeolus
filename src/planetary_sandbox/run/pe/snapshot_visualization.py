"""Per-snapshot primitive-equation visualization (upper vs. lower atmosphere).

For every stored primitive-equation state this renders one scientifically
honest figure that places the horizontal structure at two representative model
sigma levels side by side, so upper- and lower-atmospheric structure can be
compared at a glance WITHOUT pretending to show a full time-varying 3-D
atmosphere. Each figure carries, per selected level, the physical-space
relative vorticity, horizontal divergence, and temperature anomaly (relative to
that level's area-weighted horizontal mean), plus a single surface-pressure
anomaly panel (surface pressure has no vertical index).

The frames are published exactly like the BVE/SWE snapshot products: through
the shared :func:`planetary_sandbox.viz.timeline.render_snapshot_product`, so a
PE capsule grows a capsule-root ``snapshots/`` directory with one representation
subdirectory (``physical``) of time-named frames (``t000000s.png`` ...) and a
representative ``timeline.png`` overview, staged and swapped into place
atomically. PE has a single combined physical-space figure per snapshot, so it
contributes one ``physical`` representation (no ``spectral`` view exists yet).

Design notes:

* it reuses the model's own spherical-harmonic synthesis and the shared lat-lon
  view adapter (via :mod:`planetary_sandbox.run.pe.visualization` — no second
  synthesis implementation), the shared declarative
  :mod:`planetary_sandbox.viz.specs` panels, the timeline's cross-frame
  normalization, and the default Matplotlib renderer's atomic write;
* color scaling is symmetric about zero and resolved ONCE per variable across
  the whole run: upper and lower panels of a variable share a single
  ``normalization_group`` name, so the timeline resolves one shared symmetric
  limit spanning all stored times AND both selected levels. Exactly-zero fields
  fall back to a small valid symmetric extent (the shared
  :class:`~planetary_sandbox.viz.normalization.NormalizationPolicy` behavior),
  so exact-rest never yields an invalid normalization;
* per snapshot only the two selected levels are synthesized (never the whole
  column), matching how the SWE product reconstructs its physical frames;
* sigma is a dimensionless vertical coordinate (p/p_s); nothing here describes
  it as geometric altitude, and the figures make no climate/convection/3-D
  claim.

This module changes no solver, integration, initial-condition, diagnostics, or
persistence behavior; it only reads a persisted run capsule. Top-level imports
stay light (numpy + the vertical-grid metadata) so :func:`select_snapshot_levels`
is usable without CUDA; the synthesis/rendering helpers are imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import pathlib

import numpy as np

from planetary_sandbox.physics.sigma_coordinate import SigmaGrid


# The per-snapshot product is a capsule-root ``snapshots/`` directory with one
# representation subdirectory, exactly like the BVE/SWE capsules.
PE_SNAPSHOTS_DIRNAME = "snapshots"
PE_SNAPSHOTS_REPRESENTATION = "physical"

# Default representative full levels (dimensionless sigma = p/p_s).
DEFAULT_UPPER_SIGMA = 0.25
DEFAULT_LOWER_SIGMA = 0.75


# ---------------------------------------------------------------------------
# Representative full-level selection (pure vertical-grid logic, no CUDA)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectedLevels:
    """Two distinct representative model full levels and their sigma values."""

    upper_index: int
    lower_index: int
    upper_sigma: float
    lower_sigma: float


def _nearest_full_level(full_levels: tuple[float, ...], target: float,
                        *, exclude: int | None = None) -> int:
    """Index of the full level whose sigma is nearest ``target``.

    ``exclude`` removes one already-taken index from consideration so the two
    selected levels are guaranteed distinct on any (nonuniform) grid.
    """
    best_index = -1
    best_distance = math.inf
    for index, sigma in enumerate(full_levels):
        if index == exclude:
            continue
        distance = abs(sigma - target)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def select_snapshot_levels(sigma: SigmaGrid, *,
                           upper_index: int | None = None,
                           lower_index: int | None = None,
                           upper_sigma_target: float = DEFAULT_UPPER_SIGMA,
                           lower_sigma_target: float = DEFAULT_LOWER_SIGMA
                           ) -> SelectedLevels:
    """Choose an upper (~0.25) and a lower (~0.75) representative full level.

    Uses the grid's ACTUAL full-level sigma coordinates (works for nonuniform
    grids); the two indices are always distinct. Optional explicit
    ``upper_index`` / ``lower_index`` (or explicit sigma targets) make the
    selection configurable for later callers. Fails clearly if the column has
    fewer than two full levels.
    """
    full_levels = sigma.full_levels
    if len(full_levels) < 2:
        raise ValueError(
            "PE snapshot visualization needs at least two full levels to "
            f"contrast upper and lower atmosphere, got {len(full_levels)}")

    def _resolve(explicit: int | None, target: float,
                 *, exclude: int | None) -> int:
        if explicit is not None:
            if not isinstance(explicit, int) or isinstance(explicit, bool):
                raise ValueError(f"level index must be an int, got {explicit!r}")
            if not 0 <= explicit < len(full_levels):
                raise ValueError(
                    f"level index {explicit} out of range "
                    f"[0, {len(full_levels)})")
            return explicit
        return _nearest_full_level(full_levels, target, exclude=exclude)

    upper = _resolve(upper_index, upper_sigma_target, exclude=None)
    lower = _resolve(lower_index, lower_sigma_target, exclude=upper)
    if upper == lower:
        raise ValueError(
            f"upper and lower snapshot levels must be distinct, both resolved "
            f"to index {upper}")
    return SelectedLevels(
        upper_index=upper, lower_index=lower,
        upper_sigma=float(full_levels[upper]),
        lower_sigma=float(full_levels[lower]))


# ---------------------------------------------------------------------------
# Snapshot field preparation (data, separate from Matplotlib layout)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PESnapshotFields:
    """State-grid physical fields for one stored PE snapshot.

    All arrays are host (NumPy) fields on the model's own horizontal sampling
    (flat, per-point). Vorticity/divergence are the physical fields (s^-1);
    the temperature and surface-pressure fields are anomalies relative to
    their own area-weighted horizontal means (K and hPa). Rendering maps these
    to the shared lat-lon view; keeping the numeric fields here lets the
    preparation be tested without touching an image.
    """

    index: int
    total: int
    time_seconds: float
    levels: SelectedLevels
    zeta_upper: np.ndarray
    zeta_lower: np.ndarray
    delta_upper: np.ndarray
    delta_lower: np.ndarray
    t_anom_upper: np.ndarray
    t_anom_lower: np.ndarray
    ps_anom: np.ndarray


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _area_weighted_mean(model, field_host) -> float:
    """Area-weighted horizontal mean using the transform's quadrature weights.

    ``model.sh.weights`` are the analysis weights aligned point-for-point with
    ``model.sh.inv_transform`` output, so this is the exact spherical mean of a
    band-limited field (the same weighting the run diagnostics use for the mass
    integral).
    """
    weights = _host(model.sh.weights)
    field = np.asarray(field_host)
    return float(np.sum(weights * field) / np.sum(weights))


def _surface_pressure_anomaly(model, lnps_coeffs_2d) -> np.ndarray:
    """Physical surface-pressure anomaly (hPa) about its area-weighted mean.

    Recovers p_s = exp(ln p_s) on the state grid (so the nonlinearity is
    honored — the ln p_s monopole is NOT the mean of p_s), subtracts the
    area-weighted horizontal mean, and converts Pa -> hPa for a readable scale.
    """
    from .visualization import _synthesize
    lnps = _synthesize(model, lnps_coeffs_2d, subtract_mean=False)
    ps = np.exp(lnps)
    mean = _area_weighted_mean(model, ps)
    return (ps - mean) / 100.0


def prepare_pe_snapshot_fields(model, coeffs_2d, *, index: int, total: int,
                               time_seconds: float,
                               levels: SelectedLevels) -> PESnapshotFields:
    """Synthesize the state-grid fields for ONE snapshot's coefficient stack.

    ``coeffs_2d`` is the ``(3*nlev+1, l_max+1, l_max+1)`` spectral stack for a
    single stored time. Temperature anomalies are formed by zeroing the (0,0)
    monopole before synthesis (that monopole IS the exact area-weighted mean
    for the orthonormal basis); surface pressure is reconstructed and
    de-meaned physically. Only the two selected levels are transferred and
    synthesized — never the whole column.
    """
    from .visualization import _synthesize
    K = model.nlev
    up, lo = levels.upper_index, levels.lower_index
    return PESnapshotFields(
        index=int(index), total=int(total), time_seconds=float(time_seconds),
        levels=levels,
        zeta_upper=_synthesize(model, coeffs_2d[up], subtract_mean=False),
        zeta_lower=_synthesize(model, coeffs_2d[lo], subtract_mean=False),
        delta_upper=_synthesize(model, coeffs_2d[K + up], subtract_mean=False),
        delta_lower=_synthesize(model, coeffs_2d[K + lo], subtract_mean=False),
        t_anom_upper=_synthesize(model, coeffs_2d[2 * K + up],
                                 subtract_mean=True),
        t_anom_lower=_synthesize(model, coeffs_2d[2 * K + lo],
                                 subtract_mean=True),
        ps_anom=_surface_pressure_anomaly(model, coeffs_2d[3 * K]),
    )


# ---------------------------------------------------------------------------
# Declarative figure layout (Matplotlib-free)
# ---------------------------------------------------------------------------

def _readable_time(seconds: float) -> str:
    if seconds >= 86400.0:
        return f"{seconds / 86400.0:.3f} days"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.3f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.3f} min"
    return f"{seconds:.3f} s"


def _backend_label(model) -> str:
    name = type(model.grid).__name__
    lowered = name.lower()
    if "geodesic" in lowered:
        return "geodesic"
    if "latlon" in lowered or "lat_lon" in lowered:
        return "Gauss lat-lon"
    return name


def _snapshot_header(fields: PESnapshotFields, *, scenario: str | None,
                     backend_label: str | None, run_id: str | None,
                     l_max: int, terrain_note: str | None = None) -> str:
    levels = fields.levels
    identity = []
    if run_id:
        identity.append(f"run {run_id}")
    if scenario:
        identity.append(str(scenario))
    if backend_label:
        identity.append(f"{backend_label} grid")
    identity.append(f"l_max={l_max}")
    line1 = "Primitive-equation snapshot  |  " + ", ".join(identity)
    line2 = (f"snapshot {fields.index + 1}/{fields.total}    "
             f"t = {fields.time_seconds:.6g} s ({_readable_time(fields.time_seconds)})")
    line3 = ("model full levels (dimensionless sigma = p/p_s): "
             f"upper sigma={levels.upper_sigma:.4f}, "
             f"lower sigma={levels.lower_sigma:.4f}")
    lines = [line1, line2, line3]
    if terrain_note:
        lines.append(terrain_note)
    return "\n".join(lines)


def _terrain_note(model) -> str | None:
    """One header line of terrain context, or None for a flat surface.

    Synthesized from the model's own resolved Phi_s so the note describes
    exactly the terrain the dynamics used; flat runs produce byte-identical
    headers to the pre-topography product.
    """
    import cupy as cp
    if not bool(cp.any(model.phi_surface_lm)):
        return None
    phi = model.sh.inv_transform(model.phi_surface_lm).real
    return (f"prescribed surface geopotential Phi_s in "
            f"[{float(phi.min()):.4g}, {float(phi.max()):.4g}] m^2/s^2 "
            "(balanced response visible in the surface-pressure anomaly)")


def build_pe_snapshot_figure(model, fields: PESnapshotFields, *,
                             scenario: str | None = None,
                             backend_label: str | None = None,
                             run_id: str | None = None,
                             terrain_note: str | None = None,
                             target_grid=None):
    """Describe one snapshot figure (upper/lower maps + surface pressure).

    Returns ``(view_grid, FigureSpec)``; pass the returned ``view_grid`` back
    as ``target_grid`` on the next snapshot so the shared 91x181 view sampling
    is reused. Layout (3 rows x 4 columns):

        row 0            header text (spans all columns)
        row 1  z upper   d upper   T' upper   |  p_s'
        row 2  z lower   d lower   T' lower   |  (p_s' spans rows 1-2)

    Every signed field uses a symmetric normalization tagged with a per-variable
    ``normalization_group``; the upper and lower panels of a variable share one
    group name, so when the timeline resolves normalizations they receive ONE
    shared symmetric scale spanning all stored times and both levels.
    """
    from .visualization import _view_field
    from planetary_sandbox.viz.normalization import NormalizationPolicy
    from planetary_sandbox.viz.specs import (FigureSpec, PanelPlacement,
                                             ScalarMapSpec, TextPanelSpec)

    levels = fields.levels
    su = f"{levels.upper_sigma:.3f}"
    sl = f"{levels.lower_sigma:.3f}"

    # (values, title, units, variable-group, row, col, row_span)
    layout = [
        (fields.zeta_upper, f"Relative vorticity (upper sigma={su})", "s^-1",
         "vorticity", 1, 0, 1),
        (fields.delta_upper, f"Horizontal divergence (upper sigma={su})",
         "s^-1", "divergence", 1, 1, 1),
        (fields.t_anom_upper, f"Temperature anomaly (upper sigma={su})", "K",
         "temperature", 1, 2, 1),
        (fields.zeta_lower, f"Relative vorticity (lower sigma={sl})", "s^-1",
         "vorticity", 2, 0, 1),
        (fields.delta_lower, f"Horizontal divergence (lower sigma={sl})",
         "s^-1", "divergence", 2, 1, 1),
        (fields.t_anom_lower, f"Temperature anomaly (lower sigma={sl})", "K",
         "temperature", 2, 2, 1),
        (fields.ps_anom, "Surface-pressure anomaly (vs horizontal mean)",
         "hPa", "surface_pressure", 1, 3, 2),
    ]

    view_grid = target_grid
    panels = []
    for values, title, units, group, row, column, row_span in layout:
        view_grid, field = _view_field(
            model, values, name=title, units=units, target_grid=view_grid)
        panels.append(PanelPlacement(
            ScalarMapSpec(field, title, time_index=0,
                          normalization=NormalizationPolicy.symmetric(),
                          color_policy="signed",
                          normalization_group=f"pe-snapshot-{group}"),
            row, column, row_span=row_span))

    header = _snapshot_header(fields, scenario=scenario,
                              backend_label=backend_label, run_id=run_id,
                              l_max=model.l_max, terrain_note=terrain_note)
    panels.append(PanelPlacement(
        TextPanelSpec(header, font_family="sans-serif", font_size=11.0),
        0, 0, column_span=4))

    spec = FigureSpec(
        panels=tuple(panels), rows=3, columns=4,
        size_inches=(26.0, 13.0), dpi=200,
        height_ratios=(0.18, 1.0, 1.0))
    return view_grid, spec


# ---------------------------------------------------------------------------
# Timeline assembly and rendering (shared BVE/SWE snapshot-product path)
# ---------------------------------------------------------------------------

def build_pe_snapshot_timeline(model, out_dir: pathlib.Path | str, *,
                               scenario: str | None = None,
                               run_id: str | None = None,
                               upper_index: int | None = None,
                               lower_index: int | None = None):
    """Build the physical PE snapshot timeline from a persisted run capsule.

    One :class:`~planetary_sandbox.viz.timeline.FigureFrame` per stored time.
    Each snapshot's two selected levels are synthesized independently, so no
    complete time x level x lat x lon dataset is stacked on the device.
    """
    from .visualization import _load_pe_coeffs
    from planetary_sandbox.viz.timeline import FigureFrame, FigureTimeline

    coeffs, times = _load_pe_coeffs(out_dir, model.nlev)
    levels = select_snapshot_levels(model.sigma, upper_index=upper_index,
                                    lower_index=lower_index)
    backend_label = _backend_label(model)
    terrain_note = _terrain_note(model)
    total = int(coeffs.shape[0])

    view_grid = None
    frames = []
    for index in range(total):
        fields = prepare_pe_snapshot_fields(
            model, coeffs[index], index=index, total=total,
            time_seconds=float(times[index]), levels=levels)
        view_grid, spec = build_pe_snapshot_figure(
            model, fields, scenario=scenario, backend_label=backend_label,
            run_id=run_id, terrain_note=terrain_note,
            target_grid=view_grid)
        frames.append(FigureFrame(float(times[index]), spec))
    return FigureTimeline(tuple(frames), filename_prefix=scenario or "pe")


def render_pe_snapshots(model, out_dir: pathlib.Path | str, *,
                        scenario: str | None = None,
                        metadata: dict | None = None,
                        renderer=None) -> dict[str, tuple[pathlib.Path, ...]]:
    """Render the physical PE snapshot product into ``<capsule>/snapshots/``.

    Publishes exactly like the BVE/SWE snapshot products: a capsule-root
    ``snapshots/physical/`` directory of time-named frames plus a
    representative ``timeline.png`` overview, staged and swapped atomically so a
    failure never exposes a partial or mixed-generation set. Failures propagate
    (a required scientific artifact, like the single-level ``pe_summary.png``).
    """
    from planetary_sandbox.viz.renderers import get_default_renderer
    from planetary_sandbox.viz.timeline import render_snapshot_product

    run_id = (metadata or {}).get("RunId") or None
    timeline = build_pe_snapshot_timeline(model, out_dir, scenario=scenario,
                                          run_id=run_id)
    return render_snapshot_product(
        {PE_SNAPSHOTS_REPRESENTATION: timeline}, out_dir,
        renderer=renderer or get_default_renderer(), metadata=metadata,
        directory_name=PE_SNAPSHOTS_DIRNAME)


__all__ = [
    "DEFAULT_LOWER_SIGMA",
    "DEFAULT_UPPER_SIGMA",
    "PE_SNAPSHOTS_DIRNAME",
    "PE_SNAPSHOTS_REPRESENTATION",
    "PESnapshotFields",
    "SelectedLevels",
    "build_pe_snapshot_figure",
    "build_pe_snapshot_timeline",
    "prepare_pe_snapshot_fields",
    "render_pe_snapshots",
    "select_snapshot_levels",
]
