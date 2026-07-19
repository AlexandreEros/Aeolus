"""PE run-capsule persistence and provenance.

The PE command reuses the shared run-capsule machinery (run.bve.io): the same
run-id / scientific-hash construction, atomic writes, strict status and
latest-pointer rules, and manifest validation the BVE/SWE runners use. These
tests split into a CPU-only group (run-id sensitivity, overwrite, malformed-
manifest rejection -- no CUDA) and one CUDA-gated end-to-end capsule check.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from planetary_sandbox.run.bve.io import (RUN_STATUS_COMPLETED,
                                          RunProvenanceError, create_run_dir,
                                          make_run_id, update_manifest_status)
from planetary_sandbox.run.pe.config import PERunConfig

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
COMMIT = "abcdef12"


def _rid(**explicit) -> str:
    cfg = PERunConfig.resolve({"scenario": "thermal_wave", **explicit})
    return make_run_id(cfg.to_run_config_dict(), now=NOW, commit=COMMIT)


# ---------------------------------------------------------------------------
# Scientific-ID sensitivity (CPU)
# ---------------------------------------------------------------------------

def test_run_id_carries_a_scientific_hash():
    # A non-BVE solver always gets the 4-byte scientific hash appended.
    rid = _rid()
    assert rid.startswith("20260719T120000Z_thermal-wave")
    assert rid.endswith(f"_{COMMIT}")


@pytest.mark.parametrize("field,value", [
    ("dt_seconds", 137.0),
    ("nlev", 6),
    ("temperature", 271.0),
    ("surface_pressure", 95000.0),
    ("thermal_amplitude", 2.0),
    ("r_dry", 287.5),
    ("cp_dry", 1005.0),
])
def test_run_id_changes_with_pe_science(field, value):
    assert _rid() != _rid(**{field: value}), (
        f"PE run id must change when {field} changes")


def test_run_id_changes_with_sigma_grid():
    a = _rid(sigma_interfaces=(0.0, 0.5, 1.0))
    b = _rid(sigma_interfaces=(0.0, 0.3, 1.0))
    assert a != b


def test_run_id_stable_across_output_location():
    assert _rid(out="runs") == _rid(out="/tmp/x", experiment="e", overwrite=True)


# ---------------------------------------------------------------------------
# Duplicate / overwrite behavior (CPU)
# ---------------------------------------------------------------------------

def test_duplicate_run_dir_requires_overwrite(tmp_path):
    cfg = PERunConfig.resolve({"scenario": "thermal_wave"}).to_run_config_dict()
    d1 = create_run_dir(tmp_path, cfg, now=NOW, commit=COMMIT)
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, cfg, now=NOW, commit=COMMIT)
    d2 = create_run_dir(tmp_path, cfg, now=NOW, commit=COMMIT, overwrite=True)
    assert d2.reused and d2.path == d1.path


# ---------------------------------------------------------------------------
# Manifest validation rejects malformed metadata (CPU)
# ---------------------------------------------------------------------------

def test_manifest_status_update_rejects_malformed_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{ this is not json",
                                            encoding="utf-8")
    with pytest.raises(RunProvenanceError):
        update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)


def test_manifest_status_update_rejects_non_object_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RunProvenanceError):
        update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)


# ---------------------------------------------------------------------------
# End-to-end capsule (CUDA-gated)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


def _tiny_cfg(tmp_path, **overrides):
    settings = {
        "grid": "latlon", "nlat": 32, "nlon": 64, "lmax": 8, "nlev": 3,
        "scenario": "thermal_wave", "dt_seconds": 300.0,
        "duration_days": 600.0 / 86400.0, "n_snapshots": 2,
        "out": str(tmp_path / "runs"),
    }
    settings.update(overrides)
    return PERunConfig.resolve(settings)


def _load_capsule(base):
    import json
    import pathlib
    base = pathlib.Path(base)
    pointer = (base / "latest_run.txt").read_text(encoding="utf-8").strip()
    run_dir = base / pointer
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return run_dir, manifest


@cuda
def test_end_to_end_capsule_is_complete_and_loadable(tmp_path):
    import pathlib
    from planetary_sandbox.cli import pe as pe_module

    cfg = _tiny_cfg(tmp_path)
    assert pe_module.execute_run(cfg) == 0

    base = pathlib.Path(cfg.out).resolve()
    run_dir, manifest = _load_capsule(base)

    assert manifest["status"] == RUN_STATUS_COMPLETED
    rc = manifest["run_config"]
    assert rc["solver"] == "pe"
    assert rc["nlev"] == 3
    assert rc["dt_seconds"] == 300.0
    assert rc["sigma_interfaces"] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert rc["r_dry"] == cfg.r_dry and rc["cp_dry"] == cfg.cp_dry
    assert rc["scenario"] == "thermal_wave"
    assert rc["temperature"] == cfg.temperature
    assert rc["surface_pressure"] == cfg.surface_pressure
    assert rc["snapshot_times"] == [0.0, 600.0]
    # Backend/product-sampling provenance was recorded.
    assert manifest["numerics"] and manifest["numerics"].get("backend")
    # PE-specific manifest notes document the equation set and ordering.
    assert "primitive" in manifest["notes"]["equations"].lower()
    assert "ln_ps" in manifest["notes"]["coefficient_ordering"]

    # Stored arrays load with the documented shape and exact times.
    coeffs = np.load(run_dir / "pe_coeffs.npy")
    times = np.load(run_dir / "pe_snapshot_times.npy")
    assert coeffs.shape == (2, 3 * 3 + 1, 9, 9)
    assert np.array_equal(times, np.array([0.0, 600.0]))
    # The summary artifact is produced and nonempty.
    summary = run_dir / "pe_summary.png"
    assert summary.exists() and summary.stat().st_size > 0
