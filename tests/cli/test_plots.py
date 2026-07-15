"""Plot-selection resolution independently of state persistence."""
from __future__ import annotations

from planetary_sandbox.run.bve.config import PLOT_TYPES, SECONDS_PER_DAY

from .conftest import run_aeolus_stubbed


def test_default_plots_reproduce_current_behavior(stub_execute_run):
    cfg = run_aeolus_stubbed(["run", "bve"], stub_execute_run)
    assert cfg.plots == PLOT_TYPES


def test_default_plots_degrade_when_no_snapshots(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "0"], stub_execute_run)
    assert cfg.plots == ("diagnostics",)
    assert cfg.snapshot_times_seconds() == []


def test_single_plot_selection(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--plot", "summary"], stub_execute_run)
    assert cfg.plots == ("summary",)


def test_multiple_plots_any_order_deterministic(stub_execute_run):
    a = run_aeolus_stubbed(
        ["run", "bve", "--plot", "summary", "--plot", "diagnostics"],
        stub_execute_run)
    b = run_aeolus_stubbed(
        ["run", "bve", "--plot", "diagnostics", "--plot", "summary"],
        stub_execute_run)
    assert a.plots == b.plots == ("diagnostics", "summary")


def test_duplicate_plots_deduplicated(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--plot", "summary", "--plot", "summary"],
        stub_execute_run)
    assert cfg.plots == ("summary",)


def test_plot_all_expands_to_every_product(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--plot", "all"], stub_execute_run)
    assert cfg.plots == PLOT_TYPES


def test_no_plots_suppresses_images_not_snapshots(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "20", "--no-plots"],
        stub_execute_run)
    assert cfg.plots == ()
    assert len(cfg.snapshot_times_seconds()) == 20


def test_n1_with_summary_plot_is_valid(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "1", "--plot", "summary"],
        stub_execute_run)
    assert cfg.plots == ("summary",)
    assert cfg.snapshot_times_seconds() == [SECONDS_PER_DAY]


def test_plots_recorded_in_provenance(stub_execute_run):
    cfg = run_aeolus_stubbed(
        ["run", "bve", "--no-plots"], stub_execute_run)
    d = cfg.to_run_config_dict()
    assert d["plots"] == []
    assert d["snapshot_mode"] == "count"
    assert d["snapshot_times"] == cfg.snapshot_times_seconds()
