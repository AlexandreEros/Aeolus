"""Run-capsule target resolution and inspect output/error contracts."""
from __future__ import annotations

import json

from planetary_sandbox.cli.main import main


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
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
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
