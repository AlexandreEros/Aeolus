"""Run identity / immutability tests (pure Python, no GPU required)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from planetary_sandbox.run.bve.io import (
    RunDirectory,
    create_run_dir,
    make_run_id,
)

BASE_CONFIG = {
    "scenario": "two_vortices",
    "day_hours": 24.0,
    "resolution": 4,
    "lmax": 21,
    "dt_snapshots": 21600.0,   # 6 h
}

T0 = datetime(2026, 7, 11, 21, 36, 34, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

def test_run_id_is_deterministic():
    a = make_run_id(BASE_CONFIG, now=T0, commit="a724f60")
    b = make_run_id(BASE_CONFIG, now=T0, commit="a724f60")
    assert a == b


def test_run_id_encodes_all_required_fields():
    rid = make_run_id(BASE_CONFIG, now=T0, commit="a724f60")
    assert "20260711T213634Z" in rid   # UTC timestamp
    assert "two-vortices" in rid       # scenario (sanitized)
    assert "rot24h" in rid             # rotation state
    assert "r4" in rid                 # resolution
    assert "l21" in rid                # l_max
    assert "dt6h" in rid               # timestep (dt_snapshots)
    assert "a724f60" in rid            # short commit


def test_run_id_encodes_non_rotating_state():
    cfg = dict(BASE_CONFIG, day_hours=math.inf)
    rid = make_run_id(cfg, now=T0, commit="abc12345")
    assert "norot" in rid
    assert "rot" not in rid.replace("norot", "")


def test_run_id_changes_when_any_field_changes():
    base = make_run_id(BASE_CONFIG, now=T0, commit="a724f60")
    variations = [
        dict(BASE_CONFIG, scenario="rh4"),
        dict(BASE_CONFIG, day_hours=12.0),
        dict(BASE_CONFIG, resolution=5),
        dict(BASE_CONFIG, lmax=42),
        dict(BASE_CONFIG, dt_snapshots=3600.0),
    ]
    for v in variations:
        assert make_run_id(v, now=T0, commit="a724f60") != base

    # timestamp change
    later = T0 + timedelta(seconds=1)
    assert make_run_id(BASE_CONFIG, now=later, commit="a724f60") != base
    # commit change
    assert make_run_id(BASE_CONFIG, now=T0, commit="deadbeef") != base


def test_run_id_is_filesystem_safe(tmp_path):
    """Weird scenario names must not escape the run dir or break the FS."""
    cfg = dict(BASE_CONFIG, scenario="weird/../name with spaces")
    rid = make_run_id(cfg, now=T0, commit="a724f60")
    for bad in ("/", "\\", "..", " "):
        assert bad not in rid
    # and it actually works as a directory name
    (tmp_path / rid).mkdir()
    assert (tmp_path / rid).is_dir()


def test_run_id_is_timezone_normalized():
    """Passing a naive datetime is treated as UTC; TZ-aware inputs are normalized."""
    naive = datetime(2026, 7, 11, 21, 36, 34)  # no tzinfo
    aware_other_tz = T0.astimezone(timezone(timedelta(hours=-4)))
    assert (
        make_run_id(BASE_CONFIG, now=naive, commit="c")
        == make_run_id(BASE_CONFIG, now=T0, commit="c")
        == make_run_id(BASE_CONFIG, now=aware_other_tz, commit="c")
    )


# ---------------------------------------------------------------------------
# create_run_dir
# ---------------------------------------------------------------------------

def test_create_run_dir_writes_under_base(tmp_path):
    rd = create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")
    assert isinstance(rd, RunDirectory)
    assert rd.path.parent == tmp_path.resolve()
    assert rd.path.is_dir()
    assert rd.reused is False


def test_experiment_groups_runs_in_subdir(tmp_path):
    rd = create_run_dir(tmp_path, BASE_CONFIG, experiment="baseline",
                        now=T0, commit="a724f60")
    assert rd.path.parent.name == "baseline"
    assert rd.path.parent.parent == tmp_path.resolve()
    assert rd.experiment == "baseline"


def test_collision_refuses_without_overwrite(tmp_path):
    create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")


def test_overwrite_reuses_directory_without_wiping(tmp_path):
    rd1 = create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")
    (rd1.path / "existing.txt").write_text("kept", encoding="utf-8")

    rd2 = create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60",
                         overwrite=True)
    assert rd2.path == rd1.path
    assert rd2.reused is True
    # explicit reuse must NOT wipe the directory silently
    assert (rd2.path / "existing.txt").read_text(encoding="utf-8") == "kept"


def test_two_consecutive_runs_preserve_outputs(tmp_path):
    """The required test: two runs → two directories, both intact afterwards."""
    rd1 = create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")
    (rd1.path / "diagnostics").mkdir()
    (rd1.path / "diagnostics" / "timeseries.csv").write_text("run 1 data\n",
                                                             encoding="utf-8")
    rd1.update_latest_pointer()

    rd2 = create_run_dir(tmp_path, BASE_CONFIG,
                         now=T0 + timedelta(seconds=1), commit="a724f60")
    (rd2.path / "diagnostics").mkdir()
    (rd2.path / "diagnostics" / "timeseries.csv").write_text("run 2 data\n",
                                                             encoding="utf-8")
    rd2.update_latest_pointer()

    # both directories present and distinct
    assert rd1.path != rd2.path
    assert rd1.path.is_dir() and rd2.path.is_dir()

    # neither run's output was clobbered
    assert (rd1.path / "diagnostics" / "timeseries.csv").read_text(
        encoding="utf-8") == "run 1 data\n"
    assert (rd2.path / "diagnostics" / "timeseries.csv").read_text(
        encoding="utf-8") == "run 2 data\n"

    # latest_run.txt points to the newest run only
    pointer = tmp_path / "latest_run.txt"
    assert pointer.exists()
    pointed = (tmp_path / pointer.read_text(encoding="utf-8").strip()).resolve()
    assert pointed == rd2.path


def test_latest_pointer_survives_experiment_grouping(tmp_path):
    """`latest_run.txt` still lives at the base and stores a relative path."""
    rd = create_run_dir(tmp_path, BASE_CONFIG, experiment="baseline",
                        now=T0, commit="a724f60")
    rd.update_latest_pointer()
    pointer = tmp_path / "latest_run.txt"
    assert pointer.exists()
    content = pointer.read_text(encoding="utf-8").strip()
    # Should be the *relative* path (base was the parent), rooted at experiment
    assert content.startswith("baseline")
    resolved = (tmp_path / content).resolve()
    assert resolved == rd.path


# ---------------------------------------------------------------------------
# RunDirectory.figure_metadata
# ---------------------------------------------------------------------------

def test_figure_metadata_contains_all_required_fields(tmp_path):
    rd = create_run_dir(tmp_path, BASE_CONFIG, experiment="baseline",
                        now=T0, commit="a724f60")
    meta = rd.figure_metadata(source="diagnostics/timeseries.csv")
    for key in ("RunId", "Experiment", "Commit", "RunPath", "Source", "Software"):
        assert key in meta
    assert meta["RunId"] == rd.run_id
    assert meta["Experiment"] == "baseline"
    assert meta["Commit"] == "a724f60"
    assert meta["Source"] == "diagnostics/timeseries.csv"
    # RunPath is *relative* to base — that's what makes it portable
    assert meta["RunPath"] == f"baseline/{rd.run_id}" or \
           meta["RunPath"] == f"baseline\\{rd.run_id}"  # Windows


def test_figure_metadata_omits_source_when_not_given(tmp_path):
    rd = create_run_dir(tmp_path, BASE_CONFIG, now=T0, commit="a724f60")
    assert "Source" not in rd.figure_metadata()
