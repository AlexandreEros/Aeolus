"""Preset resolution and explicit-value precedence."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import PRESETS
from planetary_sandbox.run.bve.config import BASE_DEFAULTS, SECONDS_PER_DAY

from .conftest import (
    ADDITIVE_CONFIG_KEYS,
    LEGACY_CONFIG_KEYS,
    run_aeolus_stubbed,
)


def test_preset_value_beats_ordinary_default(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--preset", "two-vortices-quick"], stub_execute_run)
    assert cfg.lmax == 8
    assert cfg.duration_days == 0.02
    assert cfg.dt_snapshots == 864.0
    assert cfg.snapshot_mode == "interval"
    assert cfg.viscosity == 0.0
    assert cfg.out == "runs"


def test_explicit_flag_beats_preset(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--preset", "two-vortices-quick", "--l-max", "10"],
        stub_execute_run)
    assert cfg.lmax == 10
    assert cfg.scenario == "two_vortices"


def test_ordinary_default_without_preset(stub_execute_run):
    cfg = run_aeolus_stubbed(["run", "bve"], stub_execute_run)
    assert cfg.lmax == BASE_DEFAULTS["lmax"] == 21


def test_explicit_snapshot_count_silences_preset_interval(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--preset", "two-vortices-quick",
         "--n-snapshots", "5"],
        stub_execute_run)
    assert cfg.snapshot_mode == "count"
    assert cfg.dt_snapshots == pytest.approx(0.02 * SECONDS_PER_DAY / 4)


def test_preset_rh4_matches_documented_configuration(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--preset", "rh4"], stub_execute_run)
    assert cfg.scenario == "rh4"
    assert cfg.lmax == 21 and cfg.resolution == 4
    assert cfg.day_hours == 24.0
    assert cfg.dt_snapshots == 21600.0
    assert cfg.experiment == "validation-rh4"


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves(name, stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--preset", name], stub_execute_run)
    assert set(cfg.to_run_config_dict()) == LEGACY_CONFIG_KEYS | ADDITIVE_CONFIG_KEYS
