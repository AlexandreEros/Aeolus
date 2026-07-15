"""Snapshot schedule construction and count/interval mode contracts."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import build_parser
from planetary_sandbox.run.bve.config import (
    SECONDS_PER_DAY,
    BVERunConfig,
    count_snapshot_times,
    interval_snapshot_times,
)

from .conftest import run_aeolus_stubbed


def test_snapshot_controls_parse_to_none():
    args = build_parser().parse_args(["run", "bve"])
    assert args.n_snapshots is None
    assert args.dt_snapshots is None


def test_count_schedule_n0_is_empty():
    assert count_snapshot_times(0, 86400.0) == []


def test_count_schedule_n1_is_final_state_only():
    assert count_snapshot_times(1, 86400.0) == [86400.0]


def test_count_schedule_n2_is_endpoints():
    assert count_snapshot_times(2, 86400.0) == [0.0, 86400.0]


def test_count_schedule_n5_matches_historical_default_run():
    assert count_snapshot_times(5, 86400.0) == [
        0.0, 21600.0, 43200.0, 64800.0, 86400.0]


@pytest.mark.parametrize("n", [3, 7, 24, 100])
def test_count_schedule_arbitrary_n(n):
    t_end = 2.5 * SECONDS_PER_DAY
    times = count_snapshot_times(n, t_end)
    assert len(times) == n
    assert times[0] == 0.0 and times[-1] == t_end
    spacing = t_end / (n - 1)
    for i, t in enumerate(times):
        assert t == pytest.approx(i * spacing, abs=1e-9)


def test_interval_schedule_preserves_legacy_semantics():
    assert interval_snapshot_times(864.0, 1728.0) == [0.0, 864.0, 1728.0]
    times = interval_snapshot_times(21600.0, 1.1 * SECONDS_PER_DAY)
    assert times[0] == 0.0
    assert times[-1] == 86400.0


def test_count_and_interval_mutually_exclusive_in_resolution():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BVERunConfig.resolve({"n_snapshots": 5, "dt_snapshots": 60.0})


def test_negative_count_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        BVERunConfig.resolve({"n_snapshots": -2})


def test_nonpositive_interval_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        BVERunConfig.resolve({"dt_snapshots": 0.0})


def test_n0_and_n1_modes(stub_execute_run):
    cfg0 = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "0"], stub_execute_run)
    assert cfg0.snapshot_times_seconds() == []
    assert cfg0.dt_snapshots is None
    assert cfg0.to_run_config_dict()["dt_snapshots"] is None

    cfg1 = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "1"], stub_execute_run)
    assert cfg1.snapshot_times_seconds() == [SECONDS_PER_DAY]
    assert cfg1.dt_snapshots is None
