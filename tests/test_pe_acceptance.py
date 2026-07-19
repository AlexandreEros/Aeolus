"""End-to-end acceptance battery for the primitive-equation runner (CUDA).

The two required acceptance runs, exercised on BOTH horizontal backends
through the real ``run_pe`` path (every RK4 stage and every accepted state is
validated by the runner itself):

* exact rest -- ``isothermal_rest`` must be preserved bit-for-bit across many
  fixed steps, with exactly quiescent diagnostics and zero mass drift;
* smooth evolution -- ``thermal_wave`` must stay finite/valid for a few
  conservative fixed steps while at least one prognostic field changes
  measurably and temperature / surface pressure stay positive.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

T0 = 260.0
PS0 = 101325.0
AMP = 1.0
DT = 300.0


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(),
                                reason="CUDA/CuPy not available")

_BACKENDS = {
    "latlon": dict(grid_type="latlon", nlat=32, nlon=64, l_max=12,
                   resolution=3),
    "geodesic": dict(grid_type="geodesic", nlat=32, nlon=64, l_max=10,
                     resolution=3),
}


def _make_model(spec, nlev=5):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type=spec["grid_type"], nlat=spec["nlat"], nlon=spec["nlon"],
        l_max=spec["l_max"], grid_resolution=spec["resolution"])
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


@pytest.fixture(scope="module", params=list(_BACKENDS))
def model(request):
    return _make_model(_BACKENDS[request.param])


def _run(model, out_dir, scenario, *, t_end_s, n_snapshots, amplitude=0.0):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    from planetary_sandbox.run.pe.runner import run_pe
    state = make_pe_ic(scenario, model, temperature=T0, surface_pressure=PS0,
                       thermal_amplitude=amplitude)
    times = count_snapshot_times(n_snapshots, t_end_s)
    dt_snap = t_end_s / (n_snapshots - 1) if n_snapshots >= 2 else None
    run_pe(model, state, dt_seconds=DT, t_end_days=t_end_s / 86400.0,
           out_dir=pathlib.Path(out_dir), snapshot_times=times,
           snapshot_mode="count", dt_snapshots=dt_snap,
           plots=("diagnostics",), scenario=scenario)
    return times


def _diag(out_dir):
    return np.atleast_1d(np.genfromtxt(
        pathlib.Path(out_dir) / "diagnostics" / "timeseries.csv",
        delimiter=",", names=True))


# ---------------------------------------------------------------------------
# Exact rest (both backends)
# ---------------------------------------------------------------------------

def test_exact_rest_is_bitwise_preserved(model, tmp_path):
    import cupy as cp
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    # Five fixed steps, all snapshot times exact multiples of dt.
    times = _run(model, tmp_path, "isothermal_rest", t_end_s=5 * DT,
                 n_snapshots=6)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    stored = np.load(tmp_path / "pe_snapshot_times.npy")
    assert np.array_equal(stored, np.asarray(times))

    initial = cp.asnumpy(make_pe_ic("isothermal_rest", model, temperature=T0,
                                    surface_pressure=PS0).coeffs)
    for i in range(coeffs.shape[0]):
        assert np.array_equal(coeffs[i], initial), (
            f"rest snapshot {i} drifted from the initial state")


def test_exact_rest_diagnostics_are_quiescent(model, tmp_path):
    _run(model, tmp_path, "isothermal_rest", t_end_s=5 * DT, n_snapshots=6)
    data = _diag(tmp_path)
    assert np.all(data["max_wind_ms"] == 0.0)
    assert np.all(data["max_abs_zeta"] == 0.0)
    assert np.all(data["max_abs_delta"] == 0.0)
    assert np.all(data["mass_rel_drift"] == 0.0)
    # Temperature and surface pressure exactly unchanged.
    assert np.all(data["t_min"] == data["t_min"][0])
    assert np.all(data["t_max"] == data["t_max"][0])
    assert np.all(data["ps_min"] == data["ps_min"][0])
    assert np.all(data["ps_max"] == data["ps_max"][0])


# ---------------------------------------------------------------------------
# Smooth evolving run (both backends)
# ---------------------------------------------------------------------------

def test_thermal_wave_evolves_and_stays_valid(model, tmp_path):
    times = _run(model, tmp_path, "thermal_wave", t_end_s=4 * DT,
                 n_snapshots=3, amplitude=AMP)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    stored = np.load(tmp_path / "pe_snapshot_times.npy")

    assert np.array_equal(stored, np.asarray(times))
    assert coeffs.shape == (len(times), 3 * model.nlev + 1,
                            model.l_max + 1, model.l_max + 1)
    assert np.all(np.isfinite(coeffs))
    # At least one prognostic field changed measurably.
    assert not np.array_equal(coeffs[0], coeffs[-1])
    K = model.nlev
    assert np.abs(coeffs[-1][K:2 * K]).max() > 0.0  # divergence launched

    data = _diag(tmp_path)
    assert np.all(np.isfinite(data["t_min"])) and np.all(data["t_min"] > 0.0)
    assert np.all(data["ps_min"] > 0.0)
    for col in data.dtype.names:
        values = data[col]
        if col == "courant":
            # The initial row (dt=0) has no Courant number by design (NaN
            # sentinel, matching the SWE convention); every stepped row must
            # be finite.
            values = values[data["dt_s"] > 0]
        assert np.all(np.isfinite(values)), f"non-finite diagnostic {col}"
