"""Per-snapshot PE visualization: field preparation, shared normalization, and
the capsule-root ``snapshots/physical`` product (CUDA-gated).

These exercise the numerical preparation and declarative layout WITHOUT
inspecting pixels: area-weighted anomalies, physical p_s reconstruction, the
run-wide symmetric normalization resolved across all times and both levels by
the shared timeline, panel/axes metadata, frame filenames, and nonzero PNG
output on both horizontal backends.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

T0 = 260.0
PS0 = 101325.0
AMP = 1.0


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(),
                                reason="CUDA/CuPy not available")


def _make_model(grid_type="latlon", nlat=32, nlon=64, l_max=12, resolution=3,
                nlev=6):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


def _run_capsule(model, out_dir, scenario="thermal_wave", n_snapshots=3,
                 t_end_s=900.0):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    from planetary_sandbox.run.pe.runner import run_pe
    state = make_pe_ic(scenario, model, temperature=T0, surface_pressure=PS0,
                       thermal_amplitude=AMP)
    times = count_snapshot_times(n_snapshots, t_end_s)
    dt_snap = t_end_s / (n_snapshots - 1) if n_snapshots >= 2 else None
    run_pe(model, state, dt_seconds=300.0, t_end_days=t_end_s / 86400.0,
           out_dir=pathlib.Path(out_dir), snapshot_times=times,
           snapshot_mode="count", dt_snapshots=dt_snap,
           plots=("diagnostics",), scenario=scenario)
    return times


@pytest.fixture(scope="module")
def latlon_model():
    return _make_model()


def _group_limits(timeline, group):
    """Distinct (vmin, vmax) pairs a group resolves to across every frame."""
    resolved = timeline.resolve_normalizations()
    seen = set()
    for frame in resolved.frames:
        for placement in frame.specification.panels:
            panel = placement.panel
            if getattr(panel, "normalization_group", None) == group:
                seen.add((panel.normalization.vmin, panel.normalization.vmax))
    return seen


_GROUPS = ("pe-snapshot-vorticity", "pe-snapshot-divergence",
           "pe-snapshot-temperature", "pe-snapshot-surface_pressure")


# ---------------------------------------------------------------------------
# Field preparation
# ---------------------------------------------------------------------------

def test_prepared_fields_use_the_selected_levels(latlon_model, tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        prepare_pe_snapshot_fields, select_snapshot_levels)
    _run_capsule(latlon_model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    times = np.load(tmp_path / "pe_snapshot_times.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[-1], index=len(times) - 1, total=len(times),
        time_seconds=float(times[-1]), levels=levels)
    assert fields.levels == levels
    assert fields.index == len(times) - 1
    assert fields.total == len(times)


def test_temperature_anomaly_has_zero_area_weighted_mean(latlon_model,
                                                         tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        _area_weighted_mean, prepare_pe_snapshot_fields, select_snapshot_levels)
    _run_capsule(latlon_model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[-1], index=0, total=1, time_seconds=0.0,
        levels=levels)
    for anomaly in (fields.t_anom_upper, fields.t_anom_lower):
        assert abs(_area_weighted_mean(latlon_model, anomaly)) < 1e-6
        assert np.all(np.isfinite(anomaly))


def test_surface_pressure_reconstruction_and_zero_mean(latlon_model, tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        _area_weighted_mean, prepare_pe_snapshot_fields, select_snapshot_levels)
    _run_capsule(latlon_model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[-1], index=0, total=1, time_seconds=0.0,
        levels=levels)
    # Anomaly recovered from p_s = exp(ln p_s); its area-weighted mean is zero.
    assert abs(_area_weighted_mean(latlon_model, fields.ps_anom)) < 1e-6
    assert np.all(np.isfinite(fields.ps_anom))


def test_exact_rest_fields_are_all_zero(latlon_model, tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        prepare_pe_snapshot_fields, select_snapshot_levels)
    _run_capsule(latlon_model, tmp_path, scenario="isothermal_rest")
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[-1], index=0, total=1, time_seconds=0.0,
        levels=levels)
    for values in (fields.zeta_upper, fields.zeta_lower, fields.delta_upper,
                   fields.delta_lower, fields.t_anom_upper, fields.t_anom_lower,
                   fields.ps_anom):
        assert np.allclose(values, 0.0, atol=1e-9)


def test_fields_finite_on_geodesic_backend(tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        prepare_pe_snapshot_fields, select_snapshot_levels)
    model = _make_model(grid_type="geodesic", l_max=10)
    _run_capsule(model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(model.sigma)
    fields = prepare_pe_snapshot_fields(
        model, coeffs[-1], index=0, total=1, time_seconds=0.0, levels=levels)
    for values in (fields.zeta_upper, fields.delta_lower, fields.t_anom_upper,
                   fields.ps_anom):
        assert np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# Shared symmetric normalization (resolved by the timeline across times/levels)
# ---------------------------------------------------------------------------

def test_timeline_shares_one_symmetric_scale_per_variable(latlon_model,
                                                          tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        build_pe_snapshot_timeline)
    _run_capsule(latlon_model, tmp_path, n_snapshots=3)
    timeline = build_pe_snapshot_timeline(latlon_model, tmp_path,
                                          scenario="thermal_wave")
    for group in _GROUPS:
        seen = _group_limits(timeline, group)
        # Exactly one (vmin, vmax), symmetric, positive: shared across all
        # stored times AND both selected levels (both carry this group name).
        assert len(seen) == 1
        (vmin, vmax), = seen
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert vmin == pytest.approx(-vmax) and vmax > 0.0


def test_zero_field_run_yields_valid_fallback_limits(latlon_model, tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        build_pe_snapshot_timeline)
    _run_capsule(latlon_model, tmp_path, scenario="isothermal_rest",
                 n_snapshots=2)
    timeline = build_pe_snapshot_timeline(latlon_model, tmp_path,
                                          scenario="isothermal_rest")
    for group in _GROUPS:
        seen = _group_limits(timeline, group)
        assert len(seen) == 1
        (vmin, vmax), = seen
        assert np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax


def test_late_snapshot_controls_the_shared_run_scale(latlon_model, tmp_path):
    """A late high-amplitude snapshot must set the shared vorticity scale."""
    from planetary_sandbox.viz.timeline import FigureTimeline
    from planetary_sandbox.run.pe.snapshot_visualization import (
        build_pe_snapshot_timeline, select_snapshot_levels)
    K = latlon_model.nlev
    n = latlon_model.l_max + 1
    levels = select_snapshot_levels(latlon_model.sigma)
    coeffs = np.zeros((2, 3 * K + 1, n, n), dtype=np.complex128)
    coeffs[0, levels.upper_index, 3, 1] = 1.0e-5   # small early
    coeffs[1, levels.upper_index, 3, 1] = 7.0e-4   # large late
    np.save(tmp_path / "pe_coeffs.npy", coeffs)
    np.save(tmp_path / "pe_snapshot_times.npy",
            np.asarray([0.0, 900.0], dtype=np.float64))

    timeline = build_pe_snapshot_timeline(latlon_model, tmp_path)
    group = "pe-snapshot-vorticity"
    (full,) = _group_limits(timeline, group)
    (early,) = _group_limits(
        FigureTimeline((timeline.frames[0],), "pe"), group)
    (late,) = _group_limits(
        FigureTimeline((timeline.frames[1],), "pe"), group)
    assert full == late
    assert full[1] > early[1]


# ---------------------------------------------------------------------------
# Figure layout / metadata (no pixels)
# ---------------------------------------------------------------------------

def test_snapshot_figure_layout_and_titles(latlon_model, tmp_path):
    from planetary_sandbox.viz.specs import ScalarMapSpec, TextPanelSpec
    from planetary_sandbox.run.pe.snapshot_visualization import (
        build_pe_snapshot_figure, prepare_pe_snapshot_fields,
        select_snapshot_levels)
    times = _run_capsule(latlon_model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[-1], index=len(times) - 1, total=len(times),
        time_seconds=float(times[-1]), levels=levels)
    _, spec = build_pe_snapshot_figure(
        latlon_model, fields, scenario="thermal_wave",
        backend_label="Gauss lat-lon", run_id="run-xyz")

    map_panels = [p.panel for p in spec.panels
                  if isinstance(p.panel, ScalarMapSpec)]
    text_panels = [p.panel for p in spec.panels
                   if isinstance(p.panel, TextPanelSpec)]
    # Six atmospheric maps + one surface-pressure map = 7; one header text.
    assert len(map_panels) == 7
    assert len(text_panels) == 1

    titles = [panel.title for panel in map_panels]
    pressure_titles = [t for t in titles if "surface-pressure" in t.lower()]
    assert len(pressure_titles) == 1  # plotted once, not per level

    joined = " | ".join(titles)
    assert f"{levels.upper_sigma:.3f}" in joined
    assert f"{levels.lower_sigma:.3f}" in joined

    header = text_panels[0].text
    assert f"{len(times)}" in header
    assert str(int(times[-1])) in header  # simulation time in seconds
    assert "altitude" not in header.lower()

    # Upper and lower vorticity panels carry ONE shared normalization group,
    # so the timeline resolves them to a single scale.
    vort = [panel for panel in map_panels
            if panel.title.lower().startswith("relative vorticity")]
    assert len(vort) == 2
    assert {panel.normalization_group for panel in vort} == {
        "pe-snapshot-vorticity"}


def test_surface_pressure_panel_spans_both_map_rows(latlon_model, tmp_path):
    from planetary_sandbox.viz.specs import ScalarMapSpec
    from planetary_sandbox.run.pe.snapshot_visualization import (
        build_pe_snapshot_figure, prepare_pe_snapshot_fields,
        select_snapshot_levels)
    times = _run_capsule(latlon_model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    levels = select_snapshot_levels(latlon_model.sigma)
    fields = prepare_pe_snapshot_fields(
        latlon_model, coeffs[0], index=0, total=len(times),
        time_seconds=float(times[0]), levels=levels)
    _, spec = build_pe_snapshot_figure(latlon_model, fields)
    pressure = [p for p in spec.panels
                if isinstance(p.panel, ScalarMapSpec)
                and "surface-pressure" in p.panel.title.lower()]
    assert len(pressure) == 1
    assert pressure[0].row_span == 2


# ---------------------------------------------------------------------------
# Rendering the capsule-root snapshots/physical product
# ---------------------------------------------------------------------------

def _frame_names(snap_dir):
    return sorted(p.name for p in snap_dir.glob("t*s.png"))


def test_render_writes_physical_product_at_capsule_root(latlon_model,
                                                        tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        PE_SNAPSHOTS_DIRNAME, PE_SNAPSHOTS_REPRESENTATION, render_pe_snapshots)
    times = _run_capsule(latlon_model, tmp_path, n_snapshots=3)
    outputs = render_pe_snapshots(latlon_model, tmp_path,
                                  scenario="thermal_wave")
    assert set(outputs) == {PE_SNAPSHOTS_REPRESENTATION}
    # snapshots/ is at the capsule ROOT (not under figures/), like BVE/SWE.
    assert not (tmp_path / "figures" / "pe_snapshots").exists()
    snap_dir = tmp_path / PE_SNAPSHOTS_DIRNAME / PE_SNAPSHOTS_REPRESENTATION
    frames = _frame_names(snap_dir)
    assert frames == [f"t{int(t):06d}s.png" for t in times]
    assert (snap_dir / "timeline.png").exists()
    for name in frames + ["timeline.png"]:
        assert (snap_dir / name).stat().st_size > 0


def test_render_preserves_stored_data_and_summary(latlon_model, tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        render_pe_snapshots)
    from planetary_sandbox.run.pe.visualization import render_pe_summary
    _run_capsule(latlon_model, tmp_path, n_snapshots=3)
    coeffs_before = np.load(tmp_path / "pe_coeffs.npy")
    times_before = np.load(tmp_path / "pe_snapshot_times.npy")
    render_pe_summary(latlon_model, tmp_path)
    summary_before = (tmp_path / "pe_summary.png").read_bytes()

    render_pe_snapshots(latlon_model, tmp_path, scenario="thermal_wave")

    assert np.array_equal(np.load(tmp_path / "pe_coeffs.npy"), coeffs_before)
    assert np.array_equal(np.load(tmp_path / "pe_snapshot_times.npy"),
                          times_before)
    assert (tmp_path / "pe_summary.png").read_bytes() == summary_before


def test_render_on_geodesic_and_exact_rest(tmp_path):
    from planetary_sandbox.run.pe.snapshot_visualization import (
        PE_SNAPSHOTS_DIRNAME, PE_SNAPSHOTS_REPRESENTATION, render_pe_snapshots)
    model = _make_model(grid_type="geodesic", l_max=10)
    times = _run_capsule(model, tmp_path, scenario="isothermal_rest",
                         n_snapshots=2)
    render_pe_snapshots(model, tmp_path, scenario="isothermal_rest")
    snap_dir = tmp_path / PE_SNAPSHOTS_DIRNAME / PE_SNAPSHOTS_REPRESENTATION
    assert len(_frame_names(snap_dir)) == len(times)
    assert (snap_dir / "timeline.png").exists()


def test_runner_auto_generates_snapshots_with_summary_plot(latlon_model,
                                                           tmp_path):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    from planetary_sandbox.run.pe.runner import run_pe
    from planetary_sandbox.run.pe.snapshot_visualization import (
        PE_SNAPSHOTS_DIRNAME, PE_SNAPSHOTS_REPRESENTATION)
    state = make_pe_ic("thermal_wave", latlon_model, temperature=T0,
                       surface_pressure=PS0, thermal_amplitude=AMP)
    times = count_snapshot_times(3, 900.0)
    run_pe(latlon_model, state, dt_seconds=300.0, t_end_days=900.0 / 86400.0,
           out_dir=tmp_path, snapshot_times=times, snapshot_mode="count",
           dt_snapshots=450.0, plots=("summary",), scenario="thermal_wave")
    snap_dir = tmp_path / PE_SNAPSHOTS_DIRNAME / PE_SNAPSHOTS_REPRESENTATION
    assert len(_frame_names(snap_dir)) == len(times)
    assert (snap_dir / "timeline.png").exists()
    assert (tmp_path / "pe_summary.png").exists()
