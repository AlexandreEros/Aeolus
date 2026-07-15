"""CPU-safe tests for the aeolus CLI (parsing, resolution, dispatch, aliases).

The heavy solver path is mocked at the ``execute_run`` seam; subprocess
tests additionally assert that help, list, and inspect commands never
import CuPy or matplotlib. One guarded GPU test at the bottom exercises the
runner's explicit snapshot schedule end-to-end.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import types

import pytest

from planetary_sandbox.cli.main import PRESETS, SCENARIOS, build_parser, main
from planetary_sandbox.run.bve.config import (
    BASE_DEFAULTS, DEFAULT_N_SNAPSHOTS, DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
    PLOT_TYPES, SECONDS_PER_DAY, BVERunConfig, count_snapshot_times,
    interval_snapshot_times)
from planetary_sandbox.run.bve.io import make_run_id

# The historical psx-bve `vars(args)` key set: make_run_id and the on-disk
# config.json schema depend on it. Frozen.
LEGACY_CONFIG_KEYS = {
    "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
    "radius_earth_units", "duration_days", "dt_snapshots", "scenario",
    "viscosity", "product_quadrature", "out", "experiment", "overwrite",
}
#: The documented additive provenance keys (config.py module docstring).
ADDITIVE_CONFIG_KEYS = {"snapshot_mode", "n_snapshots", "snapshot_times", "plots"}


def run_bve_stubbed(monkeypatch, argv):
    """Run `aeolus <argv>` with the solver mocked; return the resolved config."""
    import planetary_sandbox.cli.bve as bve_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(bve_module, "execute_run", fake_execute_run)
    assert main(argv) == 0
    return captured["cfg"]


def psx_bve_stubbed(monkeypatch, argv):
    """Run `psx-bve <argv>` with the solver mocked; return the config."""
    import planetary_sandbox.cli.bve as bve_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(bve_module, "execute_run", fake_execute_run)
    monkeypatch.setattr(sys, "argv", ["psx-bve", *argv])
    assert bve_module.main() == 0
    return captured["cfg"]


# ---------------------------------------------------------------------------
# Help / CPU-safety (subprocess: fresh interpreter, no CUDA init)
# ---------------------------------------------------------------------------

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

_HEAVY_MODULES = ("cupy", "cupyx", "matplotlib",
                  "planetary_sandbox.planet.planet",
                  "planetary_sandbox.run.bve.runner",
                  "planetary_sandbox.viz")


def _cpu_safety_probe(import_stmt: str, call_stmt: str) -> str:
    return (
        "import sys\n"
        f"{import_stmt}\n"
        "try:\n"
        f"    code = {call_stmt}\n"
        "except SystemExit as exc:\n"
        "    code = 0 if exc.code in (0, None) else exc.code\n"
        "assert code == 0, f'exit code {code}'\n"
        f"banned = [m for m in {_HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, f'heavy modules imported: {banned}'\n"
    )


def _assert_probe_passes(import_stmt: str, call_stmt: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _cpu_safety_probe(import_stmt, call_stmt)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("argv", AEOLUS_HELP_INVOCATIONS,
                         ids=lambda a: " ".join(a))
def test_aeolus_help_and_list_are_cpu_safe(argv):
    _assert_probe_passes("from planetary_sandbox.cli.main import main",
                         f"main({argv!r})")


@pytest.mark.parametrize("module,name", [
    ("planetary_sandbox.cli.bve", "psx-bve"),
    ("planetary_sandbox.cli.generate_planet", "psx-gen"),
    ("planetary_sandbox.cli.clear_cache", "psx-recompile"),
])
def test_psx_help_is_cpu_safe(module, name):
    _assert_probe_passes(
        f"import {module} as m; sys.argv = [{name!r}, '--help']",
        "m.main()")


def test_bare_aeolus_prints_help(capsys):
    assert main([]) == 0
    assert "aeolus" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Invalid choices and usage errors
# ---------------------------------------------------------------------------

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
    ["run"],           # missing solver
    ["list"],          # missing topic
    ["list", "moons"],
], ids=lambda a: " ".join(a))
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Snapshot schedules (pure configuration logic)
# ---------------------------------------------------------------------------

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
    assert len(times) == n                     # exact count
    assert times[0] == 0.0 and times[-1] == t_end  # both endpoints, exact
    spacing = t_end / (n - 1)
    for i, t in enumerate(times):
        assert t == pytest.approx(i * spacing, abs=1e-9)  # evenly spaced


def test_interval_schedule_preserves_legacy_semantics():
    # Aligned: endpoints included.
    assert interval_snapshot_times(864.0, 1728.0) == [0.0, 864.0, 1728.0]
    # Misaligned: final state NOT stored (historical psx-bve behavior).
    times = interval_snapshot_times(21600.0, 1.1 * SECONDS_PER_DAY)
    assert times[0] == 0.0
    assert times[-1] == 86400.0  # not t_end = 95040 s


def test_count_and_interval_mutually_exclusive_in_resolution():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BVERunConfig.resolve({"n_snapshots": 5, "dt_snapshots": 60.0})


def test_negative_count_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        BVERunConfig.resolve({"n_snapshots": -2})


def test_nonpositive_interval_rejected():
    with pytest.raises(ValueError, match="positive"):
        BVERunConfig.resolve({"dt_snapshots": 0.0})


# ---------------------------------------------------------------------------
# Canonical versus legacy defaults, dispatch, aliases
# ---------------------------------------------------------------------------

def test_canonical_default_is_count_mode_n5(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve"])
    assert cfg.snapshot_mode == "count"
    assert cfg.n_snapshots == DEFAULT_N_SNAPSHOTS == 5
    # For the default one-day run this reproduces the historical states
    # and a meaningful uniform interval.
    assert cfg.dt_snapshots == 21600.0
    assert cfg.snapshot_times_seconds() == [0.0, 21600.0, 43200.0, 64800.0, 86400.0]


def test_legacy_psx_bve_default_is_interval_mode(monkeypatch):
    cfg = psx_bve_stubbed(monkeypatch, [])
    assert cfg.snapshot_mode == "interval"
    assert cfg.n_snapshots is None
    assert cfg.dt_snapshots == DEFAULT_SNAPSHOT_INTERVAL_SECONDS == 21600.0


def test_legacy_default_does_not_scale_with_duration(monkeypatch):
    # psx-bve --duration-days 2: still one snapshot every 6 h (9 states);
    # aeolus --days 2: still 5 states, spaced 12 h.
    legacy = psx_bve_stubbed(monkeypatch, ["--duration-days", "2"])
    assert len(legacy.snapshot_times_seconds()) == 9
    canonical = run_bve_stubbed(monkeypatch, ["run", "bve", "--days", "2"])
    assert len(canonical.snapshot_times_seconds()) == 5
    assert canonical.dt_snapshots == 43200.0


def test_defaults_match_legacy_psx_bve_values(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve"])
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
    (["--backend", "gauss-latlon"], "grid", "latlon"),  # normalized
    (["--l-max", "8"], "lmax", 8),
    (["--lmax", "8"], "lmax", 8),
    (["--days", "10"], "duration_days", 10.0),
    (["--duration-days", "10"], "duration_days", 10.0),
    (["--snapshot-interval-seconds", "864"], "dt_snapshots", 864.0),
    (["--dt-snapshots", "864"], "dt_snapshots", 864.0),
    (["--radius-earth-units", "2"], "radius_earth_units", 2.0),
], ids=lambda x: " ".join(x) if isinstance(x, list) else str(x))
def test_canonical_and_legacy_spellings(monkeypatch, argv, attr, value):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", *argv])
    assert getattr(cfg, attr) == value


def test_frozen_argparse_dest_names():
    args = build_parser().parse_args(["run", "bve"])
    for dest in LEGACY_CONFIG_KEYS:
        assert hasattr(args, dest), f"missing frozen dest {dest}"


def test_n0_and_n1_modes(monkeypatch):
    cfg0 = run_bve_stubbed(monkeypatch, ["run", "bve", "--n-snapshots", "0"])
    assert cfg0.snapshot_times_seconds() == []
    assert cfg0.dt_snapshots is None
    assert cfg0.to_run_config_dict()["dt_snapshots"] is None

    cfg1 = run_bve_stubbed(monkeypatch, ["run", "bve", "--n-snapshots", "1"])
    assert cfg1.snapshot_times_seconds() == [SECONDS_PER_DAY]
    assert cfg1.dt_snapshots is None


# ---------------------------------------------------------------------------
# Preset precedence: explicit flag > preset value > ordinary default
# ---------------------------------------------------------------------------

def test_preset_value_beats_ordinary_default(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--preset", "two-vortices-quick"])
    assert cfg.lmax == 8          # preset, not the ordinary default 21
    assert cfg.duration_days == 0.02
    assert cfg.dt_snapshots == 864.0
    assert cfg.snapshot_mode == "interval"
    # A value the preset does not set falls through to the ordinary default.
    assert cfg.viscosity == 0.0
    assert cfg.out == "runs"


def test_explicit_flag_beats_preset(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--preset", "two-vortices-quick", "--l-max", "10"])
    assert cfg.lmax == 10
    assert cfg.scenario == "two_vortices"  # rest of preset still applies


def test_ordinary_default_without_preset(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve"])
    assert cfg.lmax == BASE_DEFAULTS["lmax"] == 21


def test_explicit_snapshot_count_silences_preset_interval(monkeypatch):
    # The preset uses interval mode; an explicit count replaces it entirely.
    cfg = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--preset", "two-vortices-quick", "--n-snapshots", "5"])
    assert cfg.snapshot_mode == "count"
    assert cfg.dt_snapshots == pytest.approx(0.02 * SECONDS_PER_DAY / 4)


def test_preset_rh4_matches_documented_configuration(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--preset", "rh4"])
    assert cfg.scenario == "rh4"
    assert cfg.lmax == 21 and cfg.resolution == 4
    assert cfg.day_hours == 24.0
    assert cfg.dt_snapshots == 21600.0
    assert cfg.experiment == "validation-rh4"


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves(name, monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--preset", name])
    assert set(cfg.to_run_config_dict()) == LEGACY_CONFIG_KEYS | ADDITIVE_CONFIG_KEYS


# ---------------------------------------------------------------------------
# Plot selection
# ---------------------------------------------------------------------------

def test_default_plots_reproduce_current_behavior(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve"])
    assert cfg.plots == PLOT_TYPES  # diagnostics, snapshots, summary


def test_default_plots_degrade_when_no_snapshots(monkeypatch):
    # No explicit plot request + empty schedule: only schedule-independent
    # products; numerical diagnostics recording itself is unaffected.
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--n-snapshots", "0"])
    assert cfg.plots == ("diagnostics",)
    assert cfg.snapshot_times_seconds() == []


def test_single_plot_selection(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--plot", "summary"])
    assert cfg.plots == ("summary",)


def test_multiple_plots_any_order_deterministic(monkeypatch):
    a = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--plot", "summary", "--plot", "diagnostics"])
    b = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--plot", "diagnostics", "--plot", "summary"])
    assert a.plots == b.plots == ("diagnostics", "summary")


def test_duplicate_plots_deduplicated(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--plot", "summary", "--plot", "summary"])
    assert cfg.plots == ("summary",)


def test_plot_all_expands_to_every_product(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--plot", "all"])
    assert cfg.plots == PLOT_TYPES


def test_no_plots_suppresses_images_not_snapshots(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--n-snapshots", "20", "--no-plots"])
    assert cfg.plots == ()
    assert len(cfg.snapshot_times_seconds()) == 20  # persistence unaffected


def test_n1_with_summary_plot_is_valid(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--n-snapshots", "1", "--plot", "summary"])
    assert cfg.plots == ("summary",)
    assert cfg.snapshot_times_seconds() == [SECONDS_PER_DAY]


def test_plots_recorded_in_provenance(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--no-plots"])
    d = cfg.to_run_config_dict()
    assert d["plots"] == []
    assert d["snapshot_mode"] == "count"
    assert d["snapshot_times"] == cfg.snapshot_times_seconds()


# ---------------------------------------------------------------------------
# Run-id stability and new snapshot tags
# ---------------------------------------------------------------------------

from datetime import datetime, timezone  # noqa: E402

_NOW = datetime(2026, 7, 15, 0, 0, 0, tzinfo=timezone.utc)


def _run_id_for(monkeypatch, argv, legacy=False):
    cfg = (psx_bve_stubbed(monkeypatch, argv) if legacy
           else run_bve_stubbed(monkeypatch, ["run", "bve", *argv]))
    return make_run_id(cfg.to_run_config_dict(), now=_NOW, commit=None)


def test_legacy_interval_run_id_form_is_stable(monkeypatch):
    # A historical psx-bve invocation must keep its historical run id.
    run_id = _run_id_for(
        monkeypatch,
        ["--scenario", "rh4", "--day-hours", "24", "--dt-snapshots", "21600"],
        legacy=True)
    assert run_id == "20260715T000000Z_rh4_rot24h_r4_l21_dt6h"


def test_default_legacy_invocation_run_id_unchanged(monkeypatch):
    run_id = _run_id_for(monkeypatch, [], legacy=True)
    assert run_id == "20260715T000000Z_two-vortices_norot_r4_l21_dt6h"


def test_count_mode_run_id_uses_uniform_interval_tag(monkeypatch):
    run_id = _run_id_for(monkeypatch, ["--n-snapshots", "5"])
    assert run_id.endswith("_dt6h")


@pytest.mark.parametrize("n,tag", [(0, "snap0"), (1, "snap1")])
def test_snap_tags_for_intervalless_counts(monkeypatch, n, tag):
    run_id = _run_id_for(monkeypatch, ["--n-snapshots", str(n)])
    assert run_id.endswith(f"_{tag}")


# ---------------------------------------------------------------------------
# Backward-compatible import surfaces
# ---------------------------------------------------------------------------

def test_legacy_build_parser_keeps_defaults():
    # test_provenance.py::test_cli_exposes_grid_choice depends on this.
    from planetary_sandbox.cli.bve import build_parser

    d = vars(build_parser().parse_args([]))
    assert d["grid"] == "geodesic"
    assert d["lmax"] == 21
    assert d["resolution"] == 4
    assert d["day_hours"] == float("inf")
    assert d["scenario"] == "two_vortices"
    # Snapshot controls stay None at parser level by design.
    assert d["n_snapshots"] is None
    assert d["dt_snapshots"] is None


def test_legacy_build_parser_accepts_historical_flags():
    from planetary_sandbox.cli.bve import build_parser

    args = build_parser().parse_args([
        "--grid", "latlon", "--nlat", "12", "--nlon", "24", "--lmax", "8",
        "--scenario", "two_vortices", "--duration-days", "0.02",
        "--dt-snapshots", "864", "--out", "runs",
        "--experiment", "quickstart-latlon"])
    assert args.grid == "latlon"
    assert args.duration_days == 0.02
    assert args.dt_snapshots == 864.0


# ---------------------------------------------------------------------------
# list / inspect
# ---------------------------------------------------------------------------

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


def _write_manifest(run_dir, run_id):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": "2026-07-15T00:00:00+00:00",
        "run_id": run_id,
        "experiment": "validation-rh4",
        "run_config": {
            "lmax": 21, "grid": "geodesic", "resolution": 4,
            "duration_days": 1.0, "dt_snapshots": 21600.0,
            "scenario": "rh4", "viscosity": 0.0,
            "snapshot_mode": "count", "n_snapshots": 5,
            "snapshot_times": [0.0, 21600.0, 43200.0, 64800.0, 86400.0],
            "plots": ["diagnostics", "summary"],
        },
        "numerics": {"backend": "GeodesicBackend",
                     "product_sampling": "geodesic-r5-product"},
        "git": {"commit": "abcdef0123456789", "branch": "main", "dirty": False},
        "versions": {"python": "3.12.12"},
        "gpu": "NVIDIA GeForce MX110",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest),
                                           encoding="utf-8")
    (run_dir / "bve_summary.png").write_bytes(b"png")
    return run_id


def test_inspect_run_directory(tmp_path, capsys):
    _write_manifest(tmp_path / "run1", "runid-alpha")
    assert main(["inspect", str(tmp_path / "run1")]) == 0
    out = capsys.readouterr().out
    assert "runid-alpha" in out
    assert "rh4" in out
    assert "GeodesicBackend" in out
    assert "count mode, N=5" in out
    assert "diagnostics, summary" in out
    assert "bve_summary.png" in out


def test_inspect_follows_latest_run_pointer(tmp_path, capsys):
    _write_manifest(tmp_path / "exp" / "run2", "runid-beta")
    (tmp_path / "latest_run.txt").write_text("exp/run2\n", encoding="utf-8")
    assert main(["inspect", str(tmp_path)]) == 0
    assert "runid-beta" in capsys.readouterr().out


def test_inspect_experiment_directory_picks_newest(tmp_path, capsys):
    exp = tmp_path / "validation-rh4"
    _write_manifest(exp / "20260714T000000Z_rh4", "runid-old")
    _write_manifest(exp / "20260715T000000Z_rh4", "runid-new")
    assert main(["inspect", str(exp)]) == 0
    out = capsys.readouterr().out
    assert "runid-new" in out
    assert "runid-old" not in out
    assert "newest of 2 runs" in out


def test_inspect_missing_manifest_fails(tmp_path, capsys):
    assert main(["inspect", str(tmp_path)]) == 2
    assert "no run found" in capsys.readouterr().err


def test_inspect_malformed_manifest_fails(tmp_path, capsys):
    run = tmp_path / "bad"
    run.mkdir()
    (run / "manifest.json").write_text("{not json", encoding="utf-8")
    assert main(["inspect", str(run)]) == 2
    assert "malformed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# gen (psx-gen) fixes
# ---------------------------------------------------------------------------

def test_gen_grid_resolution_is_an_int_not_a_list():
    from planetary_sandbox.cli.generate_planet import build_parser as gen_parser

    args = gen_parser().parse_args(["--grid-resolution", "4"])
    assert args.grid_resolution == 4  # historical nargs=1 bug returned [4]


def test_gen_legacy_radius_spelling_still_accepted():
    from planetary_sandbox.cli.generate_planet import build_parser as gen_parser

    args = gen_parser().parse_args(["--eq_radius-earth_units", "2.0"])
    assert args.radius_earth_units == 2.0
    args = gen_parser().parse_args(["--radius-earth-units", "2.0"])
    assert args.radius_earth_units == 2.0


def test_gen_creates_output_directory(tmp_path, monkeypatch):
    from planetary_sandbox.cli.generate_planet import resolve_output_path

    monkeypatch.chdir(tmp_path)
    out_path = resolve_output_path("planet_summary.png")
    assert out_path == pathlib.Path("out") / "planet_summary.png"
    assert out_path.parent.is_dir()  # historical crash: out/ never created

    absolute = resolve_output_path(str(tmp_path / "deep" / "dir" / "x.png"))
    assert absolute.parent.is_dir()


def test_gen_dispatch_via_mocked_run(monkeypatch):
    import planetary_sandbox.cli.generate_planet as gen_module

    captured = {}
    monkeypatch.setattr(gen_module, "run",
                        lambda args: captured.setdefault("args", args) and 0 or 0)
    assert main(["gen", "--l-max", "9", "--grid-resolution", "2"]) == 0
    assert captured["args"].l_max == 9
    assert captured["args"].grid_resolution == 2


# ---------------------------------------------------------------------------
# recompile (psx-recompile) fixes
# ---------------------------------------------------------------------------

def _fake_cupy():
    pool = types.SimpleNamespace(free_all_blocks=lambda: None)
    return types.SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool)


def test_recompile_clears_cache_with_ascii_output(tmp_path, monkeypatch, capsys):
    from planetary_sandbox.cli import clear_cache

    cache_dir = tmp_path / ".cupy" / "kernel_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "kernel.cubin").write_bytes(b"x")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())

    rc = clear_cache.run(clear_cache.build_parser().parse_args(["--skip-verify"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert not cache_dir.exists()
    assert "[ok]" in out
    out.encode("cp1252")  # must not raise on legacy Windows code pages
    assert out.isascii()


def test_recompile_friendly_error_without_cupy(monkeypatch, capsys):
    from planetary_sandbox.cli import clear_cache

    monkeypatch.setitem(sys.modules, "cupy", None)  # forces ImportError
    rc = clear_cache.run(clear_cache.build_parser().parse_args([]))
    out = capsys.readouterr().out
    assert rc == 1
    assert "CuPy is unavailable" in out


# ---------------------------------------------------------------------------
# Parity with the CUDA-side scenario registry (skipped without CuPy)
# ---------------------------------------------------------------------------

def test_scenario_choices_match_initial_conditions():
    pytest.importorskip("cupy")
    from planetary_sandbox.run.bve.initial_conditions import INITIAL_CONDITIONS
    assert set(SCENARIOS) == set(INITIAL_CONDITIONS)


# ---------------------------------------------------------------------------
# Runner: explicit schedule end-to-end (GPU; skipped without CUDA)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_runner_consumes_explicit_schedule_exactly(tmp_path):
    """Count-mode schedule: exact snapshot count, endpoints stored, steps
    clipped to boundaries, plots suppressed while diagnostics survive."""
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.config import count_snapshot_times
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    t_end_days = 0.02
    schedule = count_snapshot_times(3, t_end_days * 86400.0)
    rc = run_bve(planet=planet, zeta0_lm=zeta0_lm,
                 dt_snapshots=None, t_end_days=t_end_days,
                 out_dir=tmp_path, viscosity=0.0, scenario="rh4",
                 snapshot_times=schedule, plots=())

    assert rc == 0
    coeffs = np.load(tmp_path / "vorticity_coeffs.npy")
    assert coeffs.shape[0] == 3          # exact count incl. both endpoints
    assert np.isfinite(coeffs).all()
    # Numerical diagnostics written; no image products.
    assert (tmp_path / "diagnostics" / "timeseries.csv").exists()
    assert not (tmp_path / "bve_summary.png").exists()
    assert not list(tmp_path.glob("*.png"))
    assert not (tmp_path / "figures").exists()
    # The integrator landed exactly on t_end (last diagnostics row).
    import csv
    with open(tmp_path / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        last = list(csv.DictReader(fh))[-1]
    assert float(last["time_s"]) == pytest.approx(t_end_days * 86400.0, abs=1e-6)
