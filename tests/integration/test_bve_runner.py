"""GPU integration coverage for explicit BVE snapshot schedules."""
from __future__ import annotations

import csv

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_runner_consumes_explicit_schedule_exactly(tmp_path):
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.config import count_snapshot_times
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    t_end_days = 0.02
    schedule = count_snapshot_times(3, t_end_days * 86400.0)
    rc = run_bve(
        planet=planet, zeta0_lm=zeta0_lm,
        dt_snapshots=None, t_end_days=t_end_days,
        out_dir=tmp_path, viscosity=0.0, scenario="rh4",
        snapshot_times=schedule, plots=())

    assert rc == 0
    coeffs = np.load(tmp_path / "vorticity_coeffs.npy")
    assert coeffs.shape[0] == 3
    assert np.isfinite(coeffs).all()
    assert (tmp_path / "diagnostics" / "timeseries.csv").exists()
    assert not (tmp_path / "bve_summary.png").exists()
    assert not list(tmp_path.glob("*.png"))
    assert not (tmp_path / "figures").exists()
    with open(tmp_path / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        last = list(csv.DictReader(fh))[-1]
    assert float(last["time_s"]) == pytest.approx(
        t_end_days * 86400.0, abs=1e-6)
