"""CPU-safe tests for the aeolus CLI (parsing, resolution, dispatch, aliases).

None of these tests require CUDA. The heavy solver path is stubbed at the
``execute_run`` seam; the subprocess tests additionally assert that help,
list, and inspect commands never import CuPy or matplotlib.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from planetary_sandbox.cli.main import (
    BVE_DEFAULTS, PRESETS, SCENARIOS, build_parser, main)
from planetary_sandbox.run.bve.config import (
    DEFAULT_SNAPSHOT_INTERVAL_SECONDS, BVERunConfig, resolve_snapshot_interval)

# The historical psx-bve `vars(args)` key set: make_run_id and the on-disk
# config.json schema depend on it. Frozen for this CLI change.
LEGACY_CONFIG_KEYS = {
    "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
    "radius_earth_units", "duration_days", "dt_snapshots", "scenario",
    "viscosity", "product_quadrature", "out", "experiment", "overwrite",
}


def run_bve_stubbed(monkeypatch, argv):
    """Run `aeolus <argv>` with the heavy solver stubbed; return the config."""
    import planetary_sandbox.cli.bve as bve_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(bve_module, "execute_run", fake_execute_run)
    assert main(argv) == 0
    return captured["cfg"]


# ---------------------------------------------------------------------------
# Help / CPU-safety (subprocess: fresh interpreter, no CUDA init)
# ---------------------------------------------------------------------------

HELP_INVOCATIONS = [
    ["--help"],
    ["run", "--help"],
    ["run", "bve", "--help"],
    ["list", "--help"],
    ["list", "presets"],
    ["list", "scenarios"],
    ["inspect", "--help"],
    ["planet", "--help"],
    ["planet", "generate", "--help"],
    ["cache", "--help"],
    ["cache", "rebuild", "--help"],
]


def _cpu_safety_probe(argv: list[str]) -> str:
    return (
        "import sys\n"
        "from planetary_sandbox.cli.main import main\n"
        f"argv = {argv!r}\n"
        "try:\n"
        "    code = main(argv)\n"
        "except SystemExit as exc:\n"
        "    code = 0 if exc.code in (0, None) else exc.code\n"
        "assert code == 0, f'exit code {code} for {argv}'\n"
        "banned = [m for m in ('cupy', 'cupyx', 'matplotlib') if m in sys.modules]\n"
        "assert not banned, f'heavy modules imported for {argv}: {banned}'\n"
    )


@pytest.mark.parametrize("argv", HELP_INVOCATIONS, ids=lambda a: " ".join(a))
def test_help_and_list_are_cpu_safe(argv):
    result = subprocess.run(
        [sys.executable, "-c", _cpu_safety_probe(argv)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


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
    ["run", "bve", "--n-snapshots", "1"],
    ["run", "bve", "--days", "-1"],
    ["run"],           # missing solver
    ["list"],          # missing topic
    ["list", "moons"],
], ids=lambda a: " ".join(a))
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Snapshot resolution (pure configuration logic)
# ---------------------------------------------------------------------------

def test_snapshot_default_applied_only_in_resolution():
    parser = build_parser()
    args = parser.parse_args(["run", "bve"])
    assert args.n_snapshots is None
    assert args.snapshot_interval_seconds is None
    assert resolve_snapshot_interval(1.0) == DEFAULT_SNAPSHOT_INTERVAL_SECONDS


def test_n_snapshots_resolves_to_interval():
    assert resolve_snapshot_interval(1.0, n_snapshots=5) == 21600.0
    assert resolve_snapshot_interval(0.02, n_snapshots=3) == 864.0


def test_snapshot_controls_mutually_exclusive_in_resolution():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_snapshot_interval(1.0, n_snapshots=5,
                                  snapshot_interval_seconds=60.0)


@pytest.mark.parametrize("n", [1, 0, -3])
def test_n_snapshots_must_be_at_least_2(n):
    with pytest.raises(ValueError, match=">= 2"):
        resolve_snapshot_interval(1.0, n_snapshots=n)


def test_interval_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        resolve_snapshot_interval(1.0, snapshot_interval_seconds=0.0)


def test_count_based_snapshots_include_both_endpoints():
    cfg = BVERunConfig(duration_days=1.0, dt_snapshots=21600.0)
    times = cfg.snapshot_times_seconds()
    assert times == [0.0, 21600.0, 43200.0, 64800.0, 86400.0]
    assert cfg.includes_final_state


def test_misaligned_interval_drops_final_state():
    cfg = BVERunConfig(duration_days=1.1, dt_snapshots=21600.0)
    times = cfg.snapshot_times_seconds()
    assert times[0] == 0.0
    assert times[-1] == 86400.0  # last stored state, not t_end = 95040 s
    assert not cfg.includes_final_state


# ---------------------------------------------------------------------------
# Configuration resolution and dispatch (stubbed solver)
# ---------------------------------------------------------------------------

def test_defaults_match_legacy_psx_bve(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve"])
    d = cfg.to_run_config_dict()
    assert set(d) == LEGACY_CONFIG_KEYS
    assert d["lmax"] == 21
    assert d["grid"] == "geodesic"
    assert d["resolution"] == 4
    assert d["nlat"] == 128 and d["nlon"] == 256
    assert d["day_hours"] == float("inf")
    assert d["duration_days"] == 1.0
    assert d["dt_snapshots"] == 21600.0
    assert d["scenario"] == "two_vortices"
    assert d["viscosity"] == 0.0
    assert d["product_quadrature"] == "fine"
    assert d["out"] == "runs"
    assert d["experiment"] is None
    assert d["overwrite"] is False


def test_n_snapshots_resolved_into_dt_snapshots_field(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--days", "1", "--n-snapshots", "5"])
    assert cfg.dt_snapshots == 21600.0
    assert cfg.includes_final_state
    assert set(cfg.to_run_config_dict()) == LEGACY_CONFIG_KEYS


def test_canonical_interval_and_legacy_alias(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--snapshot-interval-seconds", "864"])
    assert cfg.dt_snapshots == 864.0
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--dt-snapshots", "864"])
    assert cfg.dt_snapshots == 864.0


def test_backend_gauss_latlon_normalizes_to_latlon(monkeypatch):
    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--backend", "gauss-latlon"])
    assert cfg.grid == "latlon"
    assert cfg.to_run_config_dict()["grid"] == "latlon"


def test_days_alias_sets_duration(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--days", "10"])
    assert cfg.duration_days == 10.0


def test_preset_applies_and_explicit_flags_override(monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--preset", "rh4"])
    assert cfg.scenario == "rh4"
    assert cfg.lmax == 21
    assert cfg.day_hours == 24.0
    assert cfg.dt_snapshots == 21600.0
    assert cfg.experiment == "validation-rh4"

    cfg = run_bve_stubbed(
        monkeypatch, ["run", "bve", "--preset", "rh4", "--lmax", "8"])
    assert cfg.lmax == 8
    assert cfg.scenario == "rh4"


def test_explicit_snapshot_count_silences_preset_interval(monkeypatch):
    # two-vortices preset sets snapshot_interval_seconds=864 over 0.02 days;
    # an explicit --n-snapshots must win over the preset's interval.
    cfg = run_bve_stubbed(
        monkeypatch,
        ["run", "bve", "--preset", "two-vortices", "--n-snapshots", "5"])
    assert cfg.dt_snapshots == pytest.approx(0.02 * 86400.0 / 4)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves(name, monkeypatch):
    cfg = run_bve_stubbed(monkeypatch, ["run", "bve", "--preset", name])
    assert set(cfg.to_run_config_dict()) == LEGACY_CONFIG_KEYS


def test_preset_settings_use_known_keys():
    valid = set(BVE_DEFAULTS) | {"n_snapshots", "snapshot_interval_seconds"}
    for name, entry in PRESETS.items():
        unknown = set(entry["settings"]) - valid
        assert not unknown, f"preset {name} sets unknown keys: {unknown}"
        scenario = entry["settings"].get("scenario")
        if scenario is not None:
            assert scenario in SCENARIOS


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def test_psx_bve_delegates_to_aeolus_run_bve(monkeypatch):
    import planetary_sandbox.cli.bve as bve_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(bve_module, "execute_run", fake_execute_run)
    monkeypatch.setattr(sys, "argv", ["psx-bve", "--n-snapshots", "3"])
    assert bve_module.main() == 0
    assert captured["cfg"].dt_snapshots == pytest.approx(86400.0 / 2)


def test_legacy_build_parser_keeps_defaults():
    # test_provenance.py::test_cli_exposes_grid_choice depends on this
    # surface; assert the fuller legacy contract here.
    from planetary_sandbox.cli.bve import build_parser

    args = build_parser().parse_args([])
    d = vars(args)
    assert d["grid"] == "geodesic"
    assert d["lmax"] == 21
    assert d["resolution"] == 4
    assert d["day_hours"] == float("inf")
    assert d["scenario"] == "two_vortices"
    # Snapshot controls stay None at parser level by design.
    assert d["n_snapshots"] is None
    assert d["snapshot_interval_seconds"] is None


def test_legacy_build_parser_accepts_historical_flags():
    from planetary_sandbox.cli.bve import build_parser

    args = build_parser().parse_args([
        "--grid", "latlon", "--nlat", "12", "--nlon", "24", "--lmax", "8",
        "--scenario", "two_vortices", "--duration-days", "0.02",
        "--dt-snapshots", "864", "--out", "runs",
        "--experiment", "quickstart-latlon"])
    assert args.grid == "latlon"
    assert args.duration_days == 0.02
    assert args.snapshot_interval_seconds == 864.0


# ---------------------------------------------------------------------------
# list / inspect
# ---------------------------------------------------------------------------

def test_list_presets_output(capsys):
    assert main(["list", "presets"]) == 0
    out = capsys.readouterr().out
    for name in PRESETS:
        assert name in out


def test_list_scenarios_output(capsys):
    assert main(["list", "scenarios"]) == 0
    out = capsys.readouterr().out
    for name in SCENARIOS:
        assert name in out


def _write_manifest(run_dir, run_id="20260715T000000Z_rh4_rot24h_r4_l21_dt6h"):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": "2026-07-15T00:00:00+00:00",
        "run_id": run_id,
        "experiment": "validation-rh4",
        "run_config": {
            "lmax": 21, "grid": "geodesic", "resolution": 4,
            "duration_days": 1.0, "dt_snapshots": 21600.0,
            "scenario": "rh4", "viscosity": 0.0,
        },
        "numerics": {"backend": "GeodesicBackend",
                     "product_sampling": "geodesic-r5-product"},
        "git": {"commit": "abcdef0123456789", "branch": "main", "dirty": False},
        "versions": {"python": "3.12.12"},
        "gpu": "GeForce MX110",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest),
                                           encoding="utf-8")
    return run_id


def test_inspect_run_directory(tmp_path, capsys):
    run_id = _write_manifest(tmp_path / "run1")
    assert main(["inspect", str(tmp_path / "run1")]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "rh4" in out
    assert "GeodesicBackend" in out


def test_inspect_follows_latest_run_pointer(tmp_path, capsys):
    run_id = _write_manifest(tmp_path / "exp" / "run2")
    (tmp_path / "latest_run.txt").write_text("exp/run2\n", encoding="utf-8")
    assert main(["inspect", str(tmp_path)]) == 0
    assert run_id in capsys.readouterr().out


def test_inspect_missing_manifest_fails(tmp_path, capsys):
    assert main(["inspect", str(tmp_path)]) == 2
    assert "manifest.json" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Parity with the CUDA-side scenario registry (skipped without CuPy)
# ---------------------------------------------------------------------------

def test_scenario_choices_match_initial_conditions():
    pytest.importorskip("cupy")
    from planetary_sandbox.run.bve.initial_conditions import INITIAL_CONDITIONS
    assert set(SCENARIOS) == set(INITIAL_CONDITIONS)
