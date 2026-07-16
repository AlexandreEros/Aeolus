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


def _final_diag_time(run_dir):
    with open(run_dir / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        return float(list(csv.DictReader(fh))[-1]["time_s"])


# A deliberately non-aligned end time (seconds): fractional, sub-microsecond.
_MISALIGNED_T_END_S = 600.0000003
_MISALIGNED_DAYS = _MISALIGNED_T_END_S / 86400.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_count_n1_ends_exactly_at_misaligned_t_end(tmp_path):
    """Blocker 1: N=1 stores the final state at exactly the requested t_end."""
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.config import count_snapshot_times
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    rc = run_bve(
        planet=planet, zeta0_lm=zeta0_lm, dt_snapshots=None,
        t_end_days=_MISALIGNED_DAYS, out_dir=tmp_path, viscosity=0.0,
        scenario="rh4", snapshot_times=count_snapshot_times(1, _MISALIGNED_T_END_S),
        plots=(), snapshot_mode="count")

    assert rc == 0
    coeffs = np.load(tmp_path / "vorticity_coeffs.npy")
    assert coeffs.shape[0] == 1                       # only the final state
    # Exact — not merely within a pre-target tolerance.
    assert _final_diag_time(tmp_path) == _MISALIGNED_T_END_S


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_count_n0_ends_exactly_at_misaligned_t_end(tmp_path):
    """Blocker 1: N=0 stores nothing but diagnostics still end exactly at t_end."""
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.config import count_snapshot_times
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    rc = run_bve(
        planet=planet, zeta0_lm=zeta0_lm, dt_snapshots=None,
        t_end_days=_MISALIGNED_DAYS, out_dir=tmp_path, viscosity=0.0,
        scenario="rh4", snapshot_times=count_snapshot_times(0, _MISALIGNED_T_END_S),
        plots=(), snapshot_mode="count")

    assert rc == 0
    coeffs = np.load(tmp_path / "vorticity_coeffs.npy")
    assert coeffs.shape[0] == 0
    assert _final_diag_time(tmp_path) == _MISALIGNED_T_END_S


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_legacy_interval_vs_count_stopping_semantics(tmp_path):
    """Blocker 2: interval mode omits the misaligned final state; count keeps it.

    Exercises the two runner execution paths end-to-end on the same misaligned
    duration, confirming the legacy interval stopping tolerance is honored
    (final state NOT stored) while count mode lands exactly on t_end.
    """
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.config import count_snapshot_times
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    t_end_days = 0.05                       # 4320 s
    dt_snapshots = 1800.0                   # 4320 is NOT a multiple of 1800
    t_end_s = t_end_days * 86400.0

    interval_dir = tmp_path / "interval"
    interval_dir.mkdir()
    run_bve(planet=planet, zeta0_lm=zeta0_lm, dt_snapshots=dt_snapshots,
            t_end_days=t_end_days, out_dir=interval_dir, viscosity=0.0,
            scenario="rh4", plots=(), snapshot_mode="interval")
    interval_times = np.load(interval_dir / "vorticity_grid.npy")
    # Historical semantics: t=0, 1800, 3600 stored; 4320 (final) NOT stored.
    assert interval_times.shape[0] == 3
    assert _final_diag_time(interval_dir) == pytest.approx(t_end_s, abs=1e-6)

    count_dir = tmp_path / "count"
    count_dir.mkdir()
    run_bve(planet=planet, zeta0_lm=zeta0_lm, dt_snapshots=None,
            t_end_days=t_end_days, out_dir=count_dir, viscosity=0.0,
            scenario="rh4", snapshot_times=count_snapshot_times(3, t_end_s),
            plots=(), snapshot_mode="count")
    count_coeffs = np.load(count_dir / "vorticity_coeffs.npy")
    assert count_coeffs.shape[0] == 3
    assert _final_diag_time(count_dir) == t_end_s      # exact final landing
