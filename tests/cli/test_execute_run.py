"""execute_run lifecycle: atomic provenance, overwrite pointer/cleanup safety.

These exercise the *installed* run seam (execute_run) on CPU by stubbing the
heavy numerical portion (`_execute_solver`) and, where a reused directory is
needed, the run-directory factory. No CUDA required.
"""
from __future__ import annotations

import json
import types

import pytest

import planetary_sandbox.cli.bve as bve
import planetary_sandbox.run.bve.io as io
from planetary_sandbox.run.bve.config import BVERunConfig
from planetary_sandbox.run.bve.io import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RunDirectory,
    RunProvenanceError,
    update_manifest_status,
    write_run_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path, **overrides):
    return BVERunConfig.resolve({"out": str(tmp_path / "runs"), **overrides})


def _install_reused_run_dir(monkeypatch, base, run_id="RID"):
    """Force execute_run to reuse a fixed, already-existing run directory."""
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    path = base / run_id
    path.mkdir(parents=True, exist_ok=True)
    rd = RunDirectory(path=path, run_id=run_id, base=base,
                      experiment=None, commit="deadbeef", reused=True)
    write_run_manifest(path, {"x": 1}, run_id=run_id,
                       status=RUN_STATUS_COMPLETED)

    def fake_create_run_dir(base_dir, config, *, experiment=None,
                            overwrite=False, now=None, commit=None):
        return rd

    monkeypatch.setattr(io, "create_run_dir", fake_create_run_dir)
    return rd


def _read_status(run_path):
    return json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))["status"]


# ---------------------------------------------------------------------------
# atomic_write_text / update_manifest_status (Blocker 3 units)
# ---------------------------------------------------------------------------

def test_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    target = tmp_path / "manifest.json"
    io.atomic_write_text(target, "one")
    io.atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    # No leftover temp siblings.
    assert list(tmp_path.glob(".manifest.json.tmp*")) == []


def test_update_manifest_status_raises_on_missing(tmp_path):
    with pytest.raises(RunProvenanceError, match="missing or unreadable"):
        update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)


def test_update_manifest_status_raises_on_malformed(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RunProvenanceError, match="malformed"):
        update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)


def test_update_manifest_status_raises_on_non_object(tmp_path):
    (tmp_path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RunProvenanceError, match="not a JSON object"):
        update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)


def test_update_manifest_status_writes_atomically(tmp_path):
    write_run_manifest(tmp_path, {"x": 1}, run_id="RID",
                       status=RUN_STATUS_RUNNING)
    update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == RUN_STATUS_COMPLETED
    assert "updated_utc" in m
    assert list(tmp_path.glob(".manifest.json.tmp*")) == []


@pytest.mark.parametrize("manifest_case", [
    "missing", "malformed", "failed", "mismatched-run-id",
])
def test_latest_pointer_publish_requires_matching_completed_manifest(
        tmp_path, manifest_case):
    base = tmp_path / "runs"
    path = base / "RID"
    path.mkdir(parents=True)
    rd = RunDirectory(path=path, run_id="RID", base=base)

    if manifest_case == "malformed":
        (path / "manifest.json").write_text("{oops", encoding="utf-8")
    elif manifest_case == "failed":
        write_run_manifest(path, {}, run_id="RID", status=RUN_STATUS_FAILED)
    elif manifest_case == "mismatched-run-id":
        write_run_manifest(path, {}, run_id="OTHER",
                           status=RUN_STATUS_COMPLETED)

    with pytest.raises(RunProvenanceError, match="cannot publish latest pointer"):
        rd.update_latest_pointer()
    assert not (base / "latest_run.txt").exists()


# ---------------------------------------------------------------------------
# Blocker 2: the installed interval seam actually reaches the runner
# ---------------------------------------------------------------------------

def test_execute_run_forwards_snapshot_mode_to_runner(monkeypatch, tmp_path):
    """psx-bve interval config must arrive at run_bve as snapshot_mode='interval'."""
    import planetary_sandbox.planet as planet_mod
    import planetary_sandbox.run.bve.initial_conditions as ic_mod
    import planetary_sandbox.run.bve.runner as runner_mod

    captured = {}
    fake_planet = types.SimpleNamespace(
        sh=types.SimpleNamespace(transform=lambda x: x),
        so=types.SimpleNamespace(
            backend=types.SimpleNamespace(describe=lambda q: {"backend": "fake"})),
    )

    class FakePlanet:
        @staticmethod
        def generate(**kwargs):
            return fake_planet

    monkeypatch.setattr(planet_mod, "Planet", FakePlanet)
    monkeypatch.setattr(ic_mod, "make_ic", lambda scenario, planet: "zeta0")
    monkeypatch.setattr(runner_mod, "run_bve",
                        lambda **kw: captured.update(kw) or 0)

    cfg = BVERunConfig.resolve(
        {"out": str(tmp_path / "runs"), "duration_days": 1.1},
        snapshot_default="interval")
    assert cfg.snapshot_mode == "interval"
    assert bve.execute_run(cfg) == 0
    assert captured["snapshot_mode"] == "interval"
    assert captured["dt_snapshots"] == 21600.0


def test_execute_run_forwards_count_mode_to_runner(monkeypatch, tmp_path):
    import planetary_sandbox.planet as planet_mod
    import planetary_sandbox.run.bve.initial_conditions as ic_mod
    import planetary_sandbox.run.bve.runner as runner_mod

    captured = {}
    fake_planet = types.SimpleNamespace(
        sh=types.SimpleNamespace(transform=lambda x: x),
        so=types.SimpleNamespace(
            backend=types.SimpleNamespace(describe=lambda q: {})),
    )
    monkeypatch.setattr(planet_mod, "Planet",
                        types.SimpleNamespace(generate=lambda **k: fake_planet))
    monkeypatch.setattr(ic_mod, "make_ic", lambda scenario, planet: "zeta0")
    monkeypatch.setattr(runner_mod, "run_bve",
                        lambda **kw: captured.update(kw) or 0)

    cfg = _cfg(tmp_path)  # canonical: count mode N=5
    assert cfg.snapshot_mode == "count"
    assert bve.execute_run(cfg) == 0
    assert captured["snapshot_mode"] == "count"


# ---------------------------------------------------------------------------
# Blocker 3: overwrite pointer lifecycle
# ---------------------------------------------------------------------------

def test_overwrite_of_latest_then_failure_clears_pointer(monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)
    # Pre-existing completed capsule referenced by latest_run.txt.
    write_run_manifest(rd.path, {"x": 1}, run_id=rd.run_id,
                       status=RUN_STATUS_RUNNING)
    update_manifest_status(rd.path, RUN_STATUS_COMPLETED)
    rd.update_latest_pointer()
    pointer = base.resolve() / "latest_run.txt"
    assert pointer.exists()

    # The overwritten run fails during the solver.
    monkeypatch.setattr(bve, "_execute_solver",
                        lambda cfg, run_dir, run_config: (_ for _ in ()).throw(
                            RuntimeError("boom")))

    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(RuntimeError, match="boom"):
        bve.execute_run(cfg)

    # Pointer must NOT reference the failed capsule; it was cleared up-front
    # and never republished.
    assert not pointer.exists()
    assert _read_status(rd.path) == RUN_STATUS_FAILED


@pytest.mark.parametrize("failure_at", ["read", "unlink"])
def test_overwrite_pointer_clear_io_failure_aborts_before_invalidation(
        monkeypatch, tmp_path, failure_at):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)
    write_run_manifest(rd.path, {"x": 1}, run_id=rd.run_id,
                       status=RUN_STATUS_COMPLETED)
    rd.update_latest_pointer()
    pointer = base.resolve() / "latest_run.txt"

    path_cls = type(pointer)
    real_read_text = path_cls.read_text
    real_unlink = path_cls.unlink

    def guarded_read_text(self, *args, **kwargs):
        if failure_at == "read" and self == pointer:
            raise PermissionError("locked pointer read")
        return real_read_text(self, *args, **kwargs)

    def guarded_unlink(self, *args, **kwargs):
        if failure_at == "unlink" and self == pointer:
            raise PermissionError("locked pointer unlink")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(path_cls, "read_text", guarded_read_text)
    monkeypatch.setattr(path_cls, "unlink", guarded_unlink)
    solver_called = False

    def solver_must_not_run(cfg, run_dir, run_config):
        nonlocal solver_called
        solver_called = True

    monkeypatch.setattr(bve, "_execute_solver", solver_must_not_run)

    with pytest.raises(RunProvenanceError, match="latest pointer"):
        bve.execute_run(_cfg(tmp_path, overwrite=True))

    assert solver_called is False
    assert pointer.exists()
    assert _read_status(rd.path) == RUN_STATUS_COMPLETED


def test_overwrite_success_pointer_references_completed_manifest(
        monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)
    write_run_manifest(rd.path, {"x": 1}, run_id=rd.run_id,
                       status=RUN_STATUS_RUNNING)
    update_manifest_status(rd.path, RUN_STATUS_COMPLETED)
    rd.update_latest_pointer()

    monkeypatch.setattr(bve, "_execute_solver",
                        lambda cfg, run_dir, run_config: None)

    cfg = _cfg(tmp_path, overwrite=True)
    assert bve.execute_run(cfg) == 0

    pointer = base.resolve() / "latest_run.txt"
    assert pointer.exists()
    pointed = (base.resolve() / pointer.read_text(encoding="utf-8").strip()).resolve()
    assert pointed == rd.path.resolve()
    assert _read_status(rd.path) == RUN_STATUS_COMPLETED


def test_completion_with_missing_manifest_raises_and_no_pointer(
        monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)

    def solver_drops_manifest(cfg, run_dir, run_config):
        (run_dir.path / "manifest.json").unlink()

    monkeypatch.setattr(bve, "_execute_solver", solver_drops_manifest)

    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(RunProvenanceError, match="missing or unreadable"):
        bve.execute_run(cfg)
    assert not (base.resolve() / "latest_run.txt").exists()


def test_completion_with_malformed_manifest_raises_and_no_pointer(
        monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)

    def solver_corrupts_manifest(cfg, run_dir, run_config):
        (run_dir.path / "manifest.json").write_text("{oops", encoding="utf-8")

    monkeypatch.setattr(bve, "_execute_solver", solver_corrupts_manifest)

    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(RunProvenanceError, match="malformed"):
        bve.execute_run(cfg)
    assert not (base.resolve() / "latest_run.txt").exists()


def test_pointer_publish_failure_is_not_swallowed(monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)
    monkeypatch.setattr(bve, "_execute_solver",
                        lambda cfg, run_dir, run_config: None)

    def boom(self):
        raise OSError("cannot publish pointer")

    monkeypatch.setattr(RunDirectory, "update_latest_pointer", boom)

    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(OSError, match="cannot publish pointer"):
        bve.execute_run(cfg)
    # Status was durably completed before the (failed) publish attempt.
    assert _read_status(rd.path) == RUN_STATUS_COMPLETED


def test_failed_run_marks_status_and_withholds_pointer(monkeypatch, tmp_path):
    """Non-overwrite failure path: failed status, no pointer published."""
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)  # reuse=True is harmless here
    monkeypatch.setattr(bve, "_execute_solver",
                        lambda cfg, run_dir, run_config: (_ for _ in ()).throw(
                            ValueError("solver exploded")))
    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(ValueError, match="solver exploded"):
        bve.execute_run(cfg)
    assert _read_status(rd.path) == RUN_STATUS_FAILED
    assert not (base.resolve() / "latest_run.txt").exists()


# ---------------------------------------------------------------------------
# Blocker 4: narrow overwrite cleanup
# ---------------------------------------------------------------------------

def _make_generated_outputs(run_path, scenario="two_vortices"):
    """Populate a run dir with a realistic '--plot all' output set."""
    (run_path / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_path / "diagnostics" / "timeseries.csv").write_text("t\n0\n", encoding="utf-8")
    (run_path / "figures").mkdir(exist_ok=True)
    (run_path / "figures" / "energy.png").write_bytes(b"png")
    (run_path / "vorticity_coeffs.npy").write_bytes(b"npy")
    (run_path / "vorticity_grid.npy").write_bytes(b"npy")
    (run_path / "bve_snapshot_times.npy").write_bytes(b"npy")
    (run_path / "bve_summary.png").write_bytes(b"png")
    (run_path / f"{scenario}_t0.00h-24.00h-6.00h.png").write_bytes(b"png")
    (run_path / f"{scenario}_t0000000000000.000000000s.png").write_bytes(b"png")


def test_clean_removes_known_generated_artifacts(tmp_path):
    _make_generated_outputs(tmp_path)
    bve._clean_overwrite_artifacts(tmp_path)
    for name in ("diagnostics", "figures", "vorticity_coeffs.npy",
                 "vorticity_grid.npy", "bve_snapshot_times.npy",
                 "bve_summary.png"):
        assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob("*_t*h-*h-*h.png"))
    assert not list(tmp_path.glob("*_t*s.png"))


def test_clean_preserves_user_files_including_custom_png(tmp_path):
    _make_generated_outputs(tmp_path)
    (tmp_path / "custom.png").write_bytes(b"user")
    (tmp_path / "notes.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    bve._clean_overwrite_artifacts(tmp_path)

    # User files survive.
    assert (tmp_path / "custom.png").read_bytes() == b"user"
    assert (tmp_path / "notes.txt").exists()
    # Config/manifest lifecycle files are NOT part of generated-result cleanup.
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_plot_all_to_no_plots_leaves_no_stale_images(tmp_path):
    """Changing --plot all -> --no-plots must leave no stale generated images."""
    _make_generated_outputs(tmp_path)
    (tmp_path / "custom.png").write_bytes(b"user")
    bve._clean_overwrite_artifacts(tmp_path)
    stale = [p.name for p in tmp_path.glob("*.png") if p.name != "custom.png"]
    assert stale == []
    assert (tmp_path / "custom.png").exists()


def test_clean_raises_when_stale_artifact_cannot_be_removed(
        monkeypatch, tmp_path):
    _make_generated_outputs(tmp_path)

    real_rmtree = bve.shutil.rmtree

    def stubborn_rmtree(path, *a, **k):
        if str(path).endswith("diagnostics"):
            raise OSError("directory in use")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(bve.shutil, "rmtree", stubborn_rmtree)
    with pytest.raises(bve.OverwriteCleanupError, match="diagnostics"):
        bve._clean_overwrite_artifacts(tmp_path)
    # Everything else that COULD be removed still was (best-effort), and the
    # undeletable one is reported.
    assert not (tmp_path / "bve_summary.png").exists()


def test_overwrite_cleanup_failure_aborts_before_completion(
        monkeypatch, tmp_path):
    base = tmp_path / "runs"
    rd = _install_reused_run_dir(monkeypatch, base)
    _make_generated_outputs(rd.path)
    # Prior completed run referenced by the pointer.
    write_run_manifest(rd.path, {"x": 1}, run_id=rd.run_id,
                       status=RUN_STATUS_RUNNING)
    update_manifest_status(rd.path, RUN_STATUS_COMPLETED)
    rd.update_latest_pointer()

    real_rmtree = bve.shutil.rmtree
    monkeypatch.setattr(bve.shutil, "rmtree",
                        lambda p, *a, **k: (_ for _ in ()).throw(
                            OSError("locked")) if str(p).endswith("diagnostics")
                        else real_rmtree(p, *a, **k))

    # The solver must never run if cleanup fails.
    def solver_should_not_run(cfg, run_dir, run_config):
        raise AssertionError("solver ran despite cleanup failure")

    monkeypatch.setattr(bve, "_execute_solver", solver_should_not_run)

    cfg = _cfg(tmp_path, overwrite=True)
    with pytest.raises(bve.OverwriteCleanupError):
        bve.execute_run(cfg)

    # Pointer was cleared up-front and not republished. The capsule was
    # transitioned away from completed before cleanup and records the cleanup
    # failure even though some generated outputs were already removed.
    assert not (base.resolve() / "latest_run.txt").exists()
    manifest = json.loads(
        (rd.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == RUN_STATUS_FAILED
    assert manifest["error"]["type"] == "OverwriteCleanupError"
