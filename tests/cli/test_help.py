"""CPU-safety of top-level and subcommand help, list, and inspect --help.

Every help/list path runs in a fresh interpreter and asserts CuPy,
matplotlib, and the numerical modules never landed in ``sys.modules``.
"""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import main

from .conftest import assert_probe_passes

AEOLUS_HELP_INVOCATIONS = [
    ["--help"],
    ["run", "--help"],
    ["run", "bve", "--help"],
    ["list", "--help"],
    ["list", "presets"],
    ["list", "scenarios"],
    ["inspect", "--help"],
    ["gen", "--help"],
    ["recompile", "--help"],
]


@pytest.mark.parametrize("argv", AEOLUS_HELP_INVOCATIONS,
                         ids=lambda a: " ".join(a))
def test_aeolus_help_and_list_are_cpu_safe(argv):
    assert_probe_passes("from planetary_sandbox.cli.main import main",
                        f"main({argv!r})")


@pytest.mark.parametrize("module,name", [
    ("planetary_sandbox.cli.bve", "psx-bve"),
    ("planetary_sandbox.cli.generate_planet", "psx-gen"),
    ("planetary_sandbox.cli.clear_cache", "psx-recompile"),
])
def test_psx_help_is_cpu_safe(module, name):
    assert_probe_passes(
        f"import {module} as m; sys.argv = [{name!r}, '--help']",
        "m.main()")


def test_bare_aeolus_prints_help(capsys):
    assert main([]) == 0
    assert "aeolus" in capsys.readouterr().out
