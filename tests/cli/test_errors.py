"""CLI usage errors and exit-code contracts."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import main


@pytest.mark.parametrize("argv", [
    ["run", "bve", "--backend", "cubed-sphere"],
    ["run", "bve", "--grid", "cubed-sphere"],
    ["run", "bve", "--scenario", "nonexistent"],
    ["run", "bve", "--preset", "nonexistent"],
    ["run", "bve", "--n-snapshots", "5", "--snapshot-interval-seconds", "60"],
    ["run", "bve", "--n-snapshots", "5", "--dt-snapshots", "60"],
    ["run", "bve", "--n-snapshots", "-1"],
    ["run", "bve", "--days", "-1"],
    ["run", "bve", "--plot", "nonexistent"],
    ["run", "bve", "--plot", "summary", "--no-plots"],
    ["run", "bve", "--n-snapshots", "0", "--plot", "summary"],
    ["run", "bve", "--n-snapshots", "0", "--plot", "snapshots"],
    ["run"],
    ["list"],
    ["list", "moons"],
], ids=lambda a: " ".join(a))
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
