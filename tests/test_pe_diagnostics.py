"""Run-scoped diagnostics for the dry primitive-equation core (CUDA-gated)."""
from __future__ import annotations

import math
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


def _make_model(l_max=12, nlev=5, day_hours=24.0):
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


def _rest(model):
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    return make_pe_ic("isothermal_rest", model, temperature=T0,
                      surface_pressure=PS0)


def _wave(model):
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    return make_pe_ic("thermal_wave", model, temperature=T0,
                      surface_pressure=PS0, thermal_amplitude=AMP)


def test_columns_present_and_csv_written(model, tmp_path):
    from planetary_sandbox.run.pe.diagnostics import (PE_CSV_COLUMNS,
                                                      PEDiagnosticsRecorder)
    rec = PEDiagnosticsRecorder(model, tmp_path)
    row = rec.record(0.0, _rest(model), dt=0.0, step=0)
    rec.close()
    for col in PE_CSV_COLUMNS:
        assert col in row
    csv_path = pathlib.Path(tmp_path) / "diagnostics" / "timeseries.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(PE_CSV_COLUMNS)


def test_rest_state_is_quiescent(model, tmp_path):
    from planetary_sandbox.run.pe.diagnostics import PEDiagnosticsRecorder
    rec = PEDiagnosticsRecorder(model, tmp_path)
    row = rec.record(0.0, _rest(model), dt=0.0, step=0)
    rec.close()
    assert row["max_wind_ms"] == 0.0
    assert row["max_abs_zeta"] == 0.0
    assert row["max_abs_delta"] == 0.0
    assert row["mass_rel_drift"] == 0.0
    assert math.isclose(row["t_min"], T0, rel_tol=1e-9)
    assert math.isclose(row["t_max"], T0, rel_tol=1e-9)
    assert math.isclose(row["ps_min"], PS0, rel_tol=1e-9)
    assert math.isclose(row["ps_max"], PS0, rel_tol=1e-9)


def test_mass_drift_zero_when_state_unchanged(model, tmp_path):
    from planetary_sandbox.run.pe.diagnostics import PEDiagnosticsRecorder
    rec = PEDiagnosticsRecorder(model, tmp_path)
    rec.record(0.0, _rest(model), dt=0.0, step=0)
    row2 = rec.record(300.0, _rest(model), dt=300.0, step=1)
    rec.close()
    # Same state reported again: mass drift stays exactly zero.
    assert row2["mass_rel_drift"] == 0.0


def test_courant_matches_validated_characteristic_speed(model, tmp_path):
    from planetary_sandbox.run.pe.diagnostics import PEDiagnosticsRecorder
    rec = PEDiagnosticsRecorder(model, tmp_path)
    state = _rest(model)
    row = rec.record(0.0, state, dt=600.0, step=1)
    rec.close()
    expected_speed = model.max_characteristic_speed(state)
    assert math.isclose(row["max_char_speed_ms"], expected_speed, rel_tol=1e-9)
    length = model.grid.cfl_length_scale
    assert math.isclose(row["courant"], expected_speed * 600.0 / length,
                        rel_tol=1e-9)


def test_thermal_wave_diagnostics_are_finite_and_positive(model, tmp_path):
    from planetary_sandbox.run.pe.diagnostics import PEDiagnosticsRecorder
    rec = PEDiagnosticsRecorder(model, tmp_path)
    row = rec.record(0.0, _wave(model), dt=300.0, step=0)
    rec.close()
    assert all(math.isfinite(v) for v in row.values())
    assert row["t_min"] > 0.0
    assert row["ps_min"] > 0.0
