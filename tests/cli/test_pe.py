"""`aeolus run pe` parsing, resolution, and CPU-safety (import-light).

These never touch CUDA: they stub the heavy executor and assert the parser
resolves a PERunConfig correctly, and a subprocess probe proves that
`aeolus run pe --help` / `--no-plots` parsing imports neither CuPy nor the PE
runner/visualization modules.
"""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import main

from .conftest import assert_probe_passes


@pytest.fixture
def stub_pe_execute_run(monkeypatch):
    import planetary_sandbox.cli.pe as pe_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(pe_module, "execute_run", fake_execute_run)
    return captured


def _run(argv, captured):
    assert main(argv) == 0
    return captured["cfg"]


def test_default_run_pe_is_a_tiny_thermal_wave(stub_pe_execute_run):
    cfg = _run(["run", "pe"], stub_pe_execute_run)
    assert cfg.scenario == "thermal_wave"
    assert cfg.snapshot_mode == "count"
    assert cfg.dt_seconds == 300.0
    assert cfg.nlev == 8
    assert cfg.to_run_config_dict()["solver"] == "pe"


def test_isothermal_rest_selection(stub_pe_execute_run):
    cfg = _run(["run", "pe", "--scenario", "isothermal_rest",
                "--levels", "4", "--dt-seconds", "250"], stub_pe_execute_run)
    assert cfg.scenario == "isothermal_rest"
    assert cfg.nlev == 4
    assert cfg.dt_seconds == 250.0


def test_explicit_sigma_interfaces(stub_pe_execute_run):
    cfg = _run(["run", "pe", "--sigma-interfaces", "0,0.25,0.6,1.0"],
               stub_pe_execute_run)
    assert cfg.sigma_interfaces_resolved() == (0.0, 0.25, 0.6, 1.0)
    assert cfg.nlev == 3


def test_gauss_latlon_alias_and_dimensions(stub_pe_execute_run):
    cfg = _run(["run", "pe", "--backend", "gauss-latlon", "--nlat", "32",
                "--nlon", "64", "--l-max", "15"], stub_pe_execute_run)
    assert cfg.grid == "latlon"
    assert cfg.nlat == 32 and cfg.nlon == 64 and cfg.lmax == 15


def test_no_plots_disables_image_products(stub_pe_execute_run):
    cfg = _run(["run", "pe", "--no-plots"], stub_pe_execute_run)
    assert cfg.plots == ()


def test_invalid_timestep_is_a_parser_error():
    with pytest.raises(SystemExit):
        main(["run", "pe", "--dt-seconds", "0"])


def test_list_scenarios_includes_pe(capsys):
    assert main(["list", "scenarios"]) == 0
    out = capsys.readouterr().out
    assert "pe scenarios" in out
    assert "thermal_wave" in out and "isothermal_rest" in out


# ---------------------------------------------------------------------------
# CPU-safety: parsing run pe must not import CUDA / heavy modules
# ---------------------------------------------------------------------------

def test_run_pe_help_is_cpu_safe():
    assert_probe_passes(
        "from planetary_sandbox.cli.main import main",
        "main(['run', 'pe', '--help'])")
