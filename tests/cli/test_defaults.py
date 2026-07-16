"""Canonical/legacy defaults, aliases, and parser compatibility surfaces."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import build_parser
from planetary_sandbox.run.bve.config import (
    DEFAULT_N_SNAPSHOTS,
    DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
)

from .conftest import (
    ADDITIVE_CONFIG_KEYS,
    LEGACY_CONFIG_KEYS,
    run_aeolus_stubbed,
    run_psx_bve_stubbed,
)


def test_canonical_default_is_count_mode_n5(stub_execute_run):
    cfg = run_aeolus_stubbed(["run", "bve"], stub_execute_run)
    assert cfg.snapshot_mode == "count"
    assert cfg.n_snapshots == DEFAULT_N_SNAPSHOTS == 5
    assert cfg.dt_snapshots == 21600.0
    assert cfg.snapshot_times_seconds() == [
        0.0, 21600.0, 43200.0, 64800.0, 86400.0]


def test_legacy_psx_bve_default_is_interval_mode(
        monkeypatch, stub_execute_run):
    cfg = run_psx_bve_stubbed([], stub_execute_run, monkeypatch)
    assert cfg.snapshot_mode == "interval"
    assert cfg.n_snapshots is None
    assert cfg.dt_snapshots == DEFAULT_SNAPSHOT_INTERVAL_SECONDS == 21600.0


def test_legacy_default_does_not_scale_with_duration(
        monkeypatch, stub_execute_run):
    legacy = run_psx_bve_stubbed(
        ["--duration-days", "2"], stub_execute_run, monkeypatch)
    assert len(legacy.snapshot_times_seconds()) == 9
    canonical = run_aeolus_stubbed(
        ["run", "bve", "--days", "2"], stub_execute_run)
    assert len(canonical.snapshot_times_seconds()) == 5
    assert canonical.dt_snapshots == 43200.0


def test_defaults_match_legacy_psx_bve_values(stub_execute_run):
    cfg = run_aeolus_stubbed(["run", "bve"], stub_execute_run)
    d = cfg.to_run_config_dict()
    assert set(d) == LEGACY_CONFIG_KEYS | ADDITIVE_CONFIG_KEYS
    assert d["lmax"] == 21
    assert d["grid"] == "geodesic"
    assert d["resolution"] == 4
    assert d["nlat"] == 128 and d["nlon"] == 256
    assert d["day_hours"] == float("inf")
    assert d["duration_days"] == 1.0
    assert d["scenario"] == "two_vortices"
    assert d["viscosity"] == 0.0
    assert d["product_quadrature"] == "fine"
    assert d["out"] == "runs"
    assert d["experiment"] is None
    assert d["overwrite"] is False


@pytest.mark.parametrize("argv,attr,value", [
    (["--backend", "latlon"], "grid", "latlon"),
    (["--grid", "latlon"], "grid", "latlon"),
    (["--backend", "gauss-latlon"], "grid", "latlon"),
    (["--l-max", "8"], "lmax", 8),
    (["--lmax", "8"], "lmax", 8),
    (["--days", "10"], "duration_days", 10.0),
    (["--duration-days", "10"], "duration_days", 10.0),
    (["--snapshot-interval-seconds", "864"], "dt_snapshots", 864.0),
    (["--dt-snapshots", "864"], "dt_snapshots", 864.0),
    (["--radius-earth-units", "2"], "radius_earth_units", 2.0),
], ids=lambda x: " ".join(x) if isinstance(x, list) else str(x))
def test_canonical_and_legacy_spellings(
        stub_execute_run, argv, attr, value):
    cfg = run_aeolus_stubbed(["run", "bve", *argv], stub_execute_run)
    assert getattr(cfg, attr) == value


def test_frozen_argparse_dest_names():
    args = build_parser().parse_args(["run", "bve"])
    for dest in LEGACY_CONFIG_KEYS:
        assert hasattr(args, dest), f"missing frozen dest {dest}"


def test_legacy_build_parser_keeps_defaults():
    from planetary_sandbox.cli.bve import build_parser as legacy_parser

    d = vars(legacy_parser().parse_args([]))
    assert d["grid"] == "geodesic"
    assert d["lmax"] == 21
    assert d["resolution"] == 4
    assert d["day_hours"] == float("inf")
    assert d["scenario"] == "two_vortices"
    assert d["n_snapshots"] is None
    # The legacy default-applying parser surface restores the historical
    # psx-bve dt_snapshots default (21600 s), matching the pre-aeolus parser
    # byte-for-byte; the canonical aeolus parser leaves it None so count-mode
    # N=5 remains the resolved default (asserted below).
    assert d["dt_snapshots"] == DEFAULT_SNAPSHOT_INTERVAL_SECONDS == 21600.0


def test_canonical_bve_parser_leaves_dt_snapshots_none():
    """Canonical aeolus parser must NOT apply the legacy interval default."""
    from planetary_sandbox.cli.main import build_bve_parser

    args = build_bve_parser().parse_args([])
    assert args.dt_snapshots is None
    assert args.n_snapshots is None


def test_legacy_build_parser_accepts_historical_flags():
    from planetary_sandbox.cli.bve import build_parser as legacy_parser

    args = legacy_parser().parse_args([
        "--grid", "latlon", "--nlat", "12", "--nlon", "24", "--lmax", "8",
        "--scenario", "two_vortices", "--duration-days", "0.02",
        "--dt-snapshots", "864", "--out", "runs",
        "--experiment", "quickstart-latlon"])
    assert args.grid == "latlon"
    assert args.duration_days == 0.02
    assert args.dt_snapshots == 864.0
