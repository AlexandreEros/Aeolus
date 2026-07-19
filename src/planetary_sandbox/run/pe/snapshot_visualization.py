"""Per-snapshot primitive-equation visualization (upper vs. lower atmosphere).

For every stored primitive-equation state this module renders one scientifically
honest figure that places the horizontal structure at two representative model
sigma levels side by side, so upper- and lower-atmospheric structure can be
compared at a glance WITHOUT pretending to show a full time-varying 3-D
atmosphere. Each figure carries, per selected level, the physical-space
relative vorticity, horizontal divergence, and temperature anomaly (relative to
that level's area-weighted horizontal mean), plus a single surface-pressure
anomaly panel (surface pressure has no vertical index).

Design constraints honored here (see the branch brief):

* it reuses the model's own spherical-harmonic synthesis and the shared lat-lon
  view adapter (via :mod:`planetary_sandbox.run.pe.visualization` — no second
  synthesis implementation), the shared declarative
  :mod:`planetary_sandbox.viz.specs` panels, and the default Matplotlib
  renderer's atomic write;
* color scaling is resolved ONCE per variable across the whole run (all stored
  times and both selected levels) and is symmetric about zero; exactly-zero
  fields fall back to a small valid symmetric extent rather than an invalid
  normalization;
* memory stays bounded: a first pass scans every snapshot to fix the shared
  color limits, a second pass renders one snapshot at a time — a complete
  time x level x lat x lon dataset is never stacked on the device, and large
  device temporaries are released between snapshots;
* sigma is a dimensionless vertical coordinate (p/p_s); nothing here describes
  it as geometric altitude, and the figures make no climate/convection/3-D
  claim.

This module changes no solver, integration, initial-condition, diagnostics, or
persistence behavior; it only reads a persisted run capsule. Top-level imports
stay light (numpy + the vertical-grid metadata) so :func:`select_snapshot_levels`
is usable without CUDA; the synthesis/rendering helpers are imported lazily.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
import os
import pathlib
import shutil
import tempfile

import numpy as np

from planetary_sandbox.physics.sigma_coordinate import SigmaGrid


# The per-snapshot figures live inside the run capsule's figures/ directory,
# beside the diagnostics plots and the single-level pe_summary.png.
PE_FIGURES_DIRNAME = "figures"
PE_SNAPSHOTS_DIRNAME = "pe_snapshots"

# Default representative full levels (dimensionless sigma = p/p_s).
DEFAULT_UPPER_SIGMA = 0.25
DEFAULT_LOWER_SIGMA = 0.75

# Below this a signed field is treated as identically zero, so the symmetric
# color scale falls back to a small valid extent (mirrors the repository's
# NormalizationPolicy.symmetric fallback so exact-rest never yields an invalid
# normalization).
_ZERO_FIELD_TOL = 1.0e-12
_ZERO_FIELD_FALLBACK = 1.0


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
# Shared run-wide symmetric color limits (bounded-memory scan)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PERunColorLimits:
    """Run-wide max |field| per variable (over all times and both levels).

    Stored as the raw maxima; :meth:`symmetric_limits` applies the exact-zero
    fallback so a resting run still yields a valid symmetric normalization.
    """

    vorticity: float
    divergence: float
    temperature: float
    surface_pressure: float

    def bound(self, name: str) -> float:
        raw = getattr(self, name)
        return raw if raw > _ZERO_FIELD_TOL else _ZERO_FIELD_FALLBACK

    def symmetric_limits(self, name: str) -> tuple[float, float]:
        bound = self.bound(name)
        return (-bound, bound)


def compute_pe_run_color_limits(model, out_dir: pathlib.Path | str,
                                levels: SelectedLevels) -> PERunColorLimits:
    """Scan every stored snapshot to fix one symmetric scale per variable.

    Bounded memory: the coefficient stack is memory-mapped and each snapshot's
    two selected levels are synthesized, reduced to a running max, and released
    before the next snapshot. Nothing accumulates a per-time field stack.
    """
    from .visualization import _load_pe_coeffs, _synthesize
    K = model.nlev
    coeffs, _times = _load_pe_coeffs(out_dir, K, mmap_mode="r")
    max_v = max_d = max_t = max_p = 0.0
    for index in range(coeffs.shape[0]):
        snapshot = coeffs[index]
        for level in (levels.upper_index, levels.lower_index):
            zeta = _synthesize(model, snapshot[level], subtract_mean=False)
            max_v = max(max_v, float(np.max(np.abs(zeta))))
            delta = _synthesize(model, snapshot[K + level],
                                subtract_mean=False)
            max_d = max(max_d, float(np.max(np.abs(delta))))
            t_anom = _synthesize(model, snapshot[2 * K + level],
                                 subtract_mean=True)
            max_t = max(max_t, float(np.max(np.abs(t_anom))))
            del zeta, delta, t_anom
        ps_anom = _surface_pressure_anomaly(model, snapshot[3 * K])
        max_p = max(max_p, float(np.max(np.abs(ps_anom))))
        del ps_anom
        _free_device_memory()
    return PERunColorLimits(vorticity=max_v, divergence=max_d,
                            temperature=max_t, surface_pressure=max_p)


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


def _frozen_symmetric(bound: float):
    from planetary_sandbox.viz.normalization import (NormalizationKind,
                                                     NormalizationPolicy)
    return NormalizationPolicy(NormalizationKind.SYMMETRIC, -bound, bound)


def _snapshot_header(fields: PESnapshotFields, *, scenario: str | None,
                     backend_label: str | None, run_id: str | None,
                     l_max: int) -> str:
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
    return "\n".join((line1, line2, line3))


def build_pe_snapshot_figure(model, fields: PESnapshotFields,
                             limits: PERunColorLimits, *,
                             scenario: str | None = None,
                             backend_label: str | None = None,
                             run_id: str | None = None,
                             target_grid=None):
    """Describe one snapshot figure (upper/lower maps + surface pressure).

    Returns ``(view_grid, FigureSpec)``; pass the returned ``view_grid`` back
    as ``target_grid`` on the next snapshot so the shared 91x181 view sampling
    is reused. Layout (3 rows x 4 columns):

        row 0            header text (spans all columns)
        row 1  z upper   d upper   T' upper   |  p_s'
        row 2  z lower   d lower   T' lower   |  (p_s' spans rows 1-2)

    Every signed field uses the run-wide frozen symmetric scale, so upper and
    lower panels of a variable share limits and all snapshots are comparable.
    """
    from .visualization import _view_field
    from planetary_sandbox.viz.specs import (FigureSpec, PanelPlacement,
                                             ScalarMapSpec, TextPanelSpec)

    levels = fields.levels
    su = f"{levels.upper_sigma:.3f}"
    sl = f"{levels.lower_sigma:.3f}"

    # (values, title, units, variable-name, color-limit key, row, col, span)
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
    for values, title, units, key, row, column, row_span in layout:
        view_grid, field = _view_field(
            model, values, name=title, units=units, target_grid=view_grid)
        bound = limits.bound(key)
        panels.append(PanelPlacement(
            ScalarMapSpec(field, title, time_index=0,
                          normalization=_frozen_symmetric(bound),
                          color_policy="signed",
                          normalization_group=f"pe-snapshot-{key}"),
            row, column, row_span=row_span))

    header = _snapshot_header(fields, scenario=scenario,
                              backend_label=backend_label, run_id=run_id,
                              l_max=model.l_max)
    panels.append(PanelPlacement(
        TextPanelSpec(header, font_family="sans-serif", font_size=11.0),
        0, 0, column_span=4))

    spec = FigureSpec(
        panels=tuple(panels), rows=3, columns=4,
        size_inches=(26.0, 13.0), dpi=200,
        height_ratios=(0.18, 1.0, 1.0))
    return view_grid, spec


# ---------------------------------------------------------------------------
# Rendering (bounded-memory two-pass, atomic directory publish)
# ---------------------------------------------------------------------------

def _free_device_memory() -> None:
    """Release pooled device memory between snapshots (small-GPU friendly)."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _backend_label(model) -> str:
    name = type(model.grid).__name__
    lowered = name.lower()
    if "geodesic" in lowered:
        return "geodesic"
    if "latlon" in lowered or "lat_lon" in lowered:
        return "Gauss lat-lon"
    return name


def render_all_pe_snapshots(model, out_dir: pathlib.Path | str, *,
                            scenario: str | None = None,
                            metadata: dict | None = None,
                            renderer=None,
                            upper_index: int | None = None,
                            lower_index: int | None = None
                            ) -> tuple[pathlib.Path, ...]:
    """Render one figure per stored snapshot into ``figures/pe_snapshots/``.

    Two bounded-memory passes: pass one fixes the shared per-variable symmetric
    color limits; pass two synthesizes and renders one snapshot at a time. The
    complete set is staged on the same filesystem and the ``pe_snapshots``
    directory is swapped into place atomically, so a failure never leaves a
    partial or mixed-generation set and never touches unrelated figures.
    Failures deliberately propagate (a required scientific artifact, like the
    single-level ``pe_summary.png``).
    """
    from .visualization import _load_pe_coeffs
    from planetary_sandbox.viz.renderers import get_default_renderer

    out_dir = pathlib.Path(out_dir)
    K = model.nlev
    levels = select_snapshot_levels(model.sigma, upper_index=upper_index,
                                    lower_index=lower_index)
    coeffs, times = _load_pe_coeffs(out_dir, K, mmap_mode="r")
    total = int(coeffs.shape[0])

    limits = compute_pe_run_color_limits(model, out_dir, levels)
    backend = renderer or get_default_renderer()
    backend_label = _backend_label(model)
    run_id = (metadata or {}).get("RunId") or None
    width = max(4, len(str(total - 1)))

    figures_dir = out_dir / PE_FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    destination = figures_dir / PE_SNAPSHOTS_DIRNAME
    if destination.exists() and not destination.is_dir():
        raise ValueError(
            f"PE snapshot destination is not a directory: {destination}")

    stage_root = pathlib.Path(tempfile.mkdtemp(
        prefix=f".{PE_SNAPSHOTS_DIRNAME}.product-", dir=figures_dir))
    staged = stage_root / PE_SNAPSHOTS_DIRNAME
    staged.mkdir()
    published: list[pathlib.Path] = []
    try:
        view_grid = None
        for index in range(total):
            fields = prepare_pe_snapshot_fields(
                model, coeffs[index], index=index, total=total,
                time_seconds=float(times[index]), levels=levels)
            view_grid, spec = build_pe_snapshot_figure(
                model, fields, limits, scenario=scenario,
                backend_label=backend_label, run_id=run_id,
                target_grid=view_grid)
            name = f"snapshot_{index:0{width}d}.png"
            backend.render_figure(spec, staged / name, metadata=metadata)
            if not (staged / name).is_file():
                raise RuntimeError(
                    f"renderer did not create requested snapshot {name}")
            published.append(destination / name)
            del fields, spec
            _free_device_memory()

        previous = stage_root / f".{PE_SNAPSHOTS_DIRNAME}.previous"
        moved_previous = False
        try:
            if destination.exists():
                os.replace(destination, previous)
                moved_previous = True
            os.replace(staged, destination)
        except BaseException:
            if moved_previous:
                with contextlib.suppress(OSError):
                    os.replace(previous, destination)
            raise
        if moved_previous:
            shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return tuple(published)


__all__ = [
    "DEFAULT_LOWER_SIGMA",
    "DEFAULT_UPPER_SIGMA",
    "PE_FIGURES_DIRNAME",
    "PE_SNAPSHOTS_DIRNAME",
    "PERunColorLimits",
    "PESnapshotFields",
    "SelectedLevels",
    "build_pe_snapshot_figure",
    "compute_pe_run_color_limits",
    "prepare_pe_snapshot_fields",
    "render_all_pe_snapshots",
    "select_snapshot_levels",
]
