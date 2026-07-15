"""Legacy run-ID stability and canonical scientific-config hashes."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from planetary_sandbox.run.bve.io import make_run_id

from .conftest import run_aeolus_stubbed, run_psx_bve_stubbed


_NOW = datetime(2026, 7, 15, 0, 0, 0, tzinfo=timezone.utc)


def _run_id_for(argv, captured, monkeypatch, legacy=False):
    cfg = (run_psx_bve_stubbed(argv, captured, monkeypatch) if legacy
           else run_aeolus_stubbed(["run", "bve", *argv], captured))
    return make_run_id(cfg.to_run_config_dict(), now=_NOW, commit=None)


def test_legacy_interval_run_id_form_is_stable(
        monkeypatch, stub_execute_run):
    run_id = _run_id_for(
        ["--scenario", "rh4", "--day-hours", "24",
         "--dt-snapshots", "21600"],
        stub_execute_run, monkeypatch, legacy=True)
    assert run_id == "20260715T000000Z_rh4_rot24h_r4_l21_dt6h"


def test_default_legacy_invocation_run_id_unchanged(
        monkeypatch, stub_execute_run):
    run_id = _run_id_for(
        [], stub_execute_run, monkeypatch, legacy=True)
    assert run_id == "20260715T000000Z_two-vortices_norot_r4_l21_dt6h"


def test_count_mode_run_id_uses_uniform_interval_tag(
        monkeypatch, stub_execute_run):
    run_id = _run_id_for(
        ["--n-snapshots", "5"], stub_execute_run, monkeypatch)
    parts = run_id.split("_")
    assert parts[-2] == "dt6h"
    assert len(parts[-1]) == 8
    assert all(c in "0123456789abcdef" for c in parts[-1])


@pytest.mark.parametrize("n,tag", [(0, "snap0"), (1, "snap1")])
def test_snap_tags_for_intervalless_counts(
        monkeypatch, stub_execute_run, n, tag):
    run_id = _run_id_for(
        ["--n-snapshots", str(n)], stub_execute_run, monkeypatch)
    parts = run_id.split("_")
    assert parts[-2] == tag
    assert len(parts[-1]) == 8
    assert all(c in "0123456789abcdef" for c in parts[-1])


def test_scientific_config_hash_distinguishes_runs(
        monkeypatch, stub_execute_run):
    base = _run_id_for(
        ["--n-snapshots", "5"], stub_execute_run, monkeypatch)
    diff_lmax = _run_id_for(
        ["--n-snapshots", "5", "--l-max", "8"],
        stub_execute_run, monkeypatch)
    diff_out = _run_id_for(
        ["--n-snapshots", "5", "--out", "elsewhere"],
        stub_execute_run, monkeypatch)
    diff_experiment = _run_id_for(
        ["--n-snapshots", "5", "--experiment", "x"],
        stub_execute_run, monkeypatch)
    assert base != diff_lmax
    assert base == diff_out
    assert base == diff_experiment
