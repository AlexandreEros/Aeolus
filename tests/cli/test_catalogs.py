"""Preset/scenario listing and scenario-registry parity."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import PRESETS, SCENARIOS, main


def test_list_presets_output(capsys):
    assert main(["list", "presets"]) == 0
    out = capsys.readouterr().out
    for name, entry in PRESETS.items():
        assert name in out
        assert entry["description"].split()[0] in out


def test_list_scenarios_output(capsys):
    assert main(["list", "scenarios"]) == 0
    out = capsys.readouterr().out
    for name in SCENARIOS:
        assert name in out


def test_scenario_choices_match_initial_conditions():
    pytest.importorskip("cupy")
    from planetary_sandbox.run.bve.initial_conditions import INITIAL_CONDITIONS
    assert set(SCENARIOS) == set(INITIAL_CONDITIONS)
