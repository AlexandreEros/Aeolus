"""Fixed-step primitive-equation runner (CUDA-gated).

Focused unit coverage of ``run.pe.runner.run_pe``: fixed-step RK4 with stage
and accepted-state validation, strict event-time storage, immediate host
transfer, and clean failure on an invalid state. The full both-backend
exact-rest / smooth-evolution acceptance battery lives in
tests/test_pe_acceptance.py.
"""
from __future__ import annotations

import math
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


def _make_model(l_max=12, nlev=4, day_hours=24.0):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_type="latlon", nlat=32, nlon=64, l_max=l_max, grid_resolution=3)
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


@pytest.fixture(scope="module")
def model():
    return _make_model()


def _ic(model, scenario, amplitude=0.0):
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    return make_pe_ic(scenario, model, temperature=T0, surface_pressure=PS0,
                      thermal_amplitude=amplitude)


def _run(model, out_dir, scenario, *, dt_seconds=300.0, t_end_s=1200.0,
         n_snapshots=3, amplitude=AMP, plots=("diagnostics",)):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.runner import run_pe
    times = count_snapshot_times(n_snapshots, t_end_s)
    dt_snap = t_end_s / (n_snapshots - 1) if n_snapshots >= 2 else None
    run_pe(model, _ic(model, scenario, amplitude),
           dt_seconds=dt_seconds, t_end_days=t_end_s / 86400.0,
           out_dir=pathlib.Path(out_dir), snapshot_times=times,
           snapshot_mode="count", dt_snapshots=dt_snap, plots=plots,
           scenario=scenario)
    return times


# ---------------------------------------------------------------------------
# Storage: shape, ordering axis, exact stored times
# ---------------------------------------------------------------------------

def test_stored_array_shape_and_times(model, tmp_path):
    times = _run(model, tmp_path, "thermal_wave")
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    stored = np.load(tmp_path / "pe_snapshot_times.npy")
    n = model.l_max + 1
    # (n_snapshots, 3K+1, l_max+1, l_max+1): the 3K+1 axis carries the
    # [zeta..., delta..., T..., ln_ps] row ordering.
    assert coeffs.shape == (len(times), 3 * model.nlev + 1, n, n)
    assert np.array_equal(stored, np.asarray(times, dtype=np.float64))


def test_diagnostics_rows_are_initial_plus_one_per_step(model, tmp_path):
    _run(model, tmp_path, "thermal_wave", dt_seconds=300.0, t_end_s=1200.0)
    csv = pathlib.Path(tmp_path) / "diagnostics" / "timeseries.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    # header + initial row (t=0) + 4 accepted steps (1200/300).
    assert len(lines) == 1 + 1 + 4


# ---------------------------------------------------------------------------
# Exact rest: stored snapshots bitwise identical to the initial state
# ---------------------------------------------------------------------------

def test_isothermal_rest_is_preserved_bitwise(model, tmp_path):
    import cupy as cp
    _run(model, tmp_path, "isothermal_rest", amplitude=0.0)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    initial = cp.asnumpy(_ic(model, "isothermal_rest").coeffs)
    for i in range(coeffs.shape[0]):
        assert np.array_equal(coeffs[i], initial), (
            f"stored rest snapshot {i} drifted from the initial state")


def test_isothermal_rest_diagnostics_are_quiescent(model, tmp_path):
    _run(model, tmp_path, "isothermal_rest", amplitude=0.0)
    csv = pathlib.Path(tmp_path) / "diagnostics" / "timeseries.csv"
    data = np.atleast_1d(
        np.genfromtxt(csv, delimiter=",", names=True))
    assert np.all(data["max_wind_ms"] == 0.0)
    assert np.all(data["max_abs_zeta"] == 0.0)
    assert np.all(data["max_abs_delta"] == 0.0)
    assert np.all(data["mass_rel_drift"] == 0.0)


# ---------------------------------------------------------------------------
# Smooth evolution: finite, valid, at least one field changes
# ---------------------------------------------------------------------------

def test_thermal_wave_evolves_measurably_and_stays_valid(model, tmp_path):
    times = _run(model, tmp_path, "thermal_wave", dt_seconds=300.0,
                 t_end_s=1200.0)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    assert np.all(np.isfinite(coeffs))
    first, last = coeffs[0], coeffs[-1]
    # Something changed between the first and last stored state.
    assert not np.array_equal(first, last)
    # Specifically, divergence (initially exactly zero) has been launched.
    K = model.nlev
    delta_last = last[K:2 * K]
    assert np.abs(delta_last).max() > 0.0
    # Temperature stayed positive throughout (finite ln p_s too).
    data = np.atleast_1d(np.genfromtxt(
        pathlib.Path(tmp_path) / "diagnostics" / "timeseries.csv",
        delimiter=",", names=True))
    assert np.all(data["t_min"] > 0.0)
    assert np.all(data["ps_min"] > 0.0)


# ---------------------------------------------------------------------------
# Clean failure on an invalid state
# ---------------------------------------------------------------------------

def test_invalid_initial_state_fails_loudly(model, tmp_path):
    import cupy as cp
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsState, PrimitiveEquationsStateError)
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.runner import run_pe

    bad = _ic(model, "isothermal_rest")
    # Force a nonpositive temperature monopole: an invalid state.
    bad.temperature[:, 0, 0] = 0.0
    times = count_snapshot_times(2, 600.0)
    with pytest.raises(PrimitiveEquationsStateError):
        run_pe(model, PrimitiveEquationsState(bad.coeffs), dt_seconds=300.0,
               t_end_days=600.0 / 86400.0, out_dir=pathlib.Path(tmp_path),
               snapshot_times=times, snapshot_mode="count",
               dt_snapshots=600.0, plots=())
