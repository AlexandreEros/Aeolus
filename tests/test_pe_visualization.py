"""Primitive-equation summary visualization (CUDA-gated).

A minimal, honest summary artifact for a selectable middle sigma level:
relative vorticity, divergence, temperature anomaly (relative to the
horizontal mean), and ln p_s anomaly. No climate-equilibrium claim.
"""
from __future__ import annotations

import pathlib

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
                nlev=5):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


def _run_capsule(model, out_dir):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    from planetary_sandbox.run.pe.runner import run_pe
    state = make_pe_ic("thermal_wave", model, temperature=T0,
                       surface_pressure=PS0, thermal_amplitude=AMP)
    times = count_snapshot_times(2, 900.0)
    run_pe(model, state, dt_seconds=300.0, t_end_days=900.0 / 86400.0,
           out_dir=pathlib.Path(out_dir), snapshot_times=times,
           snapshot_mode="count", dt_snapshots=900.0, plots=("diagnostics",),
           scenario="thermal_wave")


def test_summary_spec_has_four_labelled_panels(tmp_path):
    from planetary_sandbox.run.pe.visualization import build_pe_summary_spec
    model = _make_model()
    _run_capsule(model, tmp_path)
    spec = build_pe_summary_spec(model, tmp_path)
    assert len(spec.panels) == 4
    titles = [p.panel.title for p in spec.panels]
    joined = " | ".join(titles).lower()
    assert "vorticity" in joined
    assert "divergence" in joined
    assert "temperature" in joined and "anomaly" in joined
    assert "p_s" in joined or "pressure" in joined
    # The selected sigma level is visible in the panel labels.
    assert "sigma" in joined or "level" in joined


def test_render_summary_writes_nonempty_png(tmp_path):
    from planetary_sandbox.run.pe.visualization import (PE_SUMMARY_FILENAME,
                                                       render_pe_summary)
    model = _make_model()
    _run_capsule(model, tmp_path)
    out = render_pe_summary(model, tmp_path)
    assert out == pathlib.Path(tmp_path) / PE_SUMMARY_FILENAME
    assert out.exists() and out.stat().st_size > 0


def test_render_summary_works_on_geodesic_backend(tmp_path):
    from planetary_sandbox.run.pe.visualization import render_pe_summary
    model = _make_model(grid_type="geodesic", l_max=10)
    _run_capsule(model, tmp_path)
    out = render_pe_summary(model, tmp_path)
    assert out.exists() and out.stat().st_size > 0
