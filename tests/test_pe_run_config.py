"""Resolved primitive-equation run configuration (CPU / import-light).

These tests never import CuPy: like the BVE/SWE config tests they exercise
only ``run.pe.config``, which must resolve, validate, and hash a run's
scientific identity without touching CUDA. The PE-specific additions over
the SWE config are the *fixed integration timestep* (``dt_seconds``), the
vertical grid (``nlev`` / explicit sigma interfaces), the configurable dry
gas constants, and the initial-condition thermodynamic parameters — every
one of which must participate in the scientific-configuration hash.
"""
from __future__ import annotations

import math

import pytest

from planetary_sandbox.run.pe.config import (PE_PLOT_TYPES, PE_SCENARIOS,
                                             PERunConfig)


def _resolve(**explicit) -> PERunConfig:
    return PERunConfig.resolve(explicit)


# ---------------------------------------------------------------------------
# Defaults and resolution
# ---------------------------------------------------------------------------

def test_defaults_are_a_small_safe_demo():
    cfg = _resolve()
    # A tiny, explicit, stable demonstration — not an expensive run.
    assert cfg.scenario in PE_SCENARIOS
    assert cfg.nlev >= 1
    assert cfg.dt_seconds > 0
    assert cfg.duration_days > 0
    # Fixed integration step must not exceed the whole simulated duration.
    assert cfg.dt_seconds <= cfg.duration_days * 86400.0


def test_explicit_values_override_defaults():
    cfg = _resolve(lmax=12, nlev=6, dt_seconds=250.0, temperature=250.0,
                   surface_pressure=90000.0, scenario="thermal_wave")
    assert cfg.lmax == 12
    assert cfg.nlev == 6
    assert cfg.dt_seconds == 250.0
    assert cfg.temperature == 250.0
    assert cfg.surface_pressure == 90000.0
    assert cfg.scenario == "thermal_wave"


def test_uniform_sigma_interfaces_from_nlev():
    cfg = _resolve(nlev=4)
    assert cfg.sigma_interfaces_resolved() == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_explicit_sigma_interfaces_override_nlev():
    interfaces = (0.0, 0.1, 0.4, 1.0)
    cfg = _resolve(sigma_interfaces=interfaces)
    assert cfg.sigma_interfaces_resolved() == interfaces
    assert cfg.nlev == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_nonpositive_timestep():
    with pytest.raises(ValueError):
        _resolve(dt_seconds=0.0)
    with pytest.raises(ValueError):
        _resolve(dt_seconds=-10.0)


def test_rejects_timestep_larger_than_duration():
    with pytest.raises(ValueError):
        _resolve(dt_seconds=10_000.0, duration_days=0.01)  # 864 s duration


def test_rejects_nonpositive_temperature_and_pressure():
    with pytest.raises(ValueError):
        _resolve(temperature=0.0)
    with pytest.raises(ValueError):
        _resolve(surface_pressure=-1.0)


def test_rejects_cp_not_greater_than_r():
    with pytest.raises(ValueError):
        _resolve(r_dry=1004.0, cp_dry=1004.0)


def test_rejects_bad_sigma_interfaces():
    with pytest.raises(ValueError):
        _resolve(sigma_interfaces=(0.1, 0.5, 1.0))   # top not 0.0
    with pytest.raises(ValueError):
        _resolve(sigma_interfaces=(0.0, 0.5, 0.9))   # bottom not 1.0
    with pytest.raises(ValueError):
        _resolve(sigma_interfaces=(0.0, 0.5, 0.5, 1.0))  # not increasing


def test_rejects_unknown_scenario():
    with pytest.raises(ValueError):
        _resolve(scenario="held_suarez")


def test_thermal_wave_requires_lmax_at_least_two():
    with pytest.raises(ValueError):
        _resolve(scenario="thermal_wave", lmax=1)


def test_latlon_backend_dimension_minimums():
    with pytest.raises(ValueError):
        _resolve(grid="latlon", nlat=1, nlon=64)


# ---------------------------------------------------------------------------
# Scientific-identity hash sensitivity
# ---------------------------------------------------------------------------

def _sci(cfg: PERunConfig) -> dict:
    return cfg.scientific_config_dict()


def test_hash_omits_locational_control_keys():
    a = _resolve(out="runs")
    b = _resolve(out="/tmp/elsewhere", experiment="x", overwrite=True)
    assert _sci(a) == _sci(b)


@pytest.mark.parametrize("field,value", [
    ("dt_seconds", 123.0),
    ("nlev", 6),
    ("temperature", 271.0),
    ("surface_pressure", 95000.0),
    ("r_dry", 287.5),
    ("cp_dry", 1005.0),
    ("thermal_amplitude", 2.0),
    ("lmax", 12),
])
def test_scientific_identity_changes_with_pe_parameters(field, value):
    base = _resolve(scenario="thermal_wave")
    changed = _resolve(scenario="thermal_wave", **{field: value})
    assert _sci(base) != _sci(changed), (
        f"scientific config must depend on {field}")


def test_scientific_identity_changes_with_sigma_interfaces():
    base = _resolve(sigma_interfaces=(0.0, 0.5, 1.0))
    changed = _resolve(sigma_interfaces=(0.0, 0.3, 1.0))
    assert _sci(base) != _sci(changed)


# ---------------------------------------------------------------------------
# Config-dict schema (consumed by the runner and provenance)
# ---------------------------------------------------------------------------

def test_run_config_dict_carries_pe_metadata():
    cfg = _resolve(nlev=5, dt_seconds=300.0, scenario="thermal_wave")
    d = cfg.to_run_config_dict()
    assert d["solver"] == "pe"
    assert d["nlev"] == 5
    assert d["dt_seconds"] == 300.0
    assert d["sigma_interfaces"] == list(cfg.sigma_interfaces_resolved())
    assert d["r_dry"] == cfg.r_dry
    assert d["cp_dry"] == cfg.cp_dry
    assert d["scenario"] == "thermal_wave"
    assert d["temperature"] == cfg.temperature
    assert d["surface_pressure"] == cfg.surface_pressure
    # Snapshot schedule is authoritative and in seconds.
    assert d["snapshot_times"][0] == 0.0
    assert math.isclose(d["snapshot_times"][-1],
                        cfg.duration_days * 86400.0)


def test_summary_lines_are_plain_strings():
    lines = _resolve().summary_lines()
    assert lines and all(isinstance(s, str) for s in lines)
    assert any("pe" in s for s in lines)


def test_plot_types_exposed():
    assert "summary" in PE_PLOT_TYPES
    assert "diagnostics" in PE_PLOT_TYPES
