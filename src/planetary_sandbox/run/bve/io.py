"""Run identity and provenance.

Every run writes to a **unique, immutable** directory under a base ``runs/``
folder. A run's identity is derived from:

    UTC timestamp | scenario | rotation state | resolution | l_max | timestep | short commit

so different runs of the same command produce different directories, and
re-running the *same* command in the same second requires an explicit
``--overwrite``. This keeps every ``.csv`` / ``.npz`` / ``.png`` traceable to
exactly one invocation, prevents silent output clobbering across baselines,
and lets the diagnostics files be trusted as evidence rather than being
mistaken for a stale earlier run's output.

Public API:
    make_run_id(config, *, now, commit)  -> str
    create_run_dir(base, config, ...)    -> RunDirectory
    RunDirectory.figure_metadata()       -> dict for matplotlib savefig
    RunDirectory.update_latest_pointer() -> writes latest_run.txt
    write_run_manifest(dir, config, ...) -> manifest.json with full provenance
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10,
            cwd=pathlib.Path(__file__).resolve().parents[4],
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _short_commit() -> str | None:
    """Short git commit hash of the current repo, or None outside a repo."""
    return _git(["rev-parse", "--short=8", "HEAD"])


# ---------------------------------------------------------------------------
# Run-ID construction
# ---------------------------------------------------------------------------

_FS_SAFE = "-"  # not "." — a lone "." or ".." must never appear in a run-id token


def _sanitize(name: str) -> str:
    """Filesystem-safe kebab-ish token; collapses runs of separators.

    All non-alphanumeric characters (including ``.``, ``/``, ``\\``) collapse
    to ``-``, so user-supplied scenario/experiment names can't inject path
    traversal (``..``) or hidden files (leading ``.``).
    """
    out = "".join(c if c.isalnum() or c in _FS_SAFE else "-" for c in str(name).lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _rotation_tag(day_hours: float) -> str:
    """`norot` for f=0, otherwise `rot{h}h` with `.` -> `p` in decimals."""
    if not math.isfinite(day_hours):
        return "norot"
    if float(day_hours) == int(day_hours):
        return f"rot{int(day_hours)}h"
    return f"rot{day_hours:.2f}h".replace(".", "p")


def _dt_tag(dt_seconds: float) -> str:
    """`dtNh`/`dtNm`/`dtNs` — the shortest exact form of the timestep."""
    dt = float(dt_seconds)
    if dt <= 0:
        return "dt0"
    if dt >= 3600.0 and math.isclose(dt % 3600.0, 0.0, abs_tol=1e-9):
        return f"dt{int(round(dt / 3600.0))}h"
    if dt >= 60.0 and math.isclose(dt % 60.0, 0.0, abs_tol=1e-9):
        return f"dt{int(round(dt / 60.0))}m"
    return f"dt{dt:g}s".replace(".", "p")


def _snapshot_tag(config: dict) -> str:
    """Snapshot token of the run id.

    Runs with a uniform snapshot interval (legacy interval mode, and count
    mode with N >= 2) keep the historical `dtNh`/`dtNm`/`dtNs` form. Count
    modes without a meaningful interval (`dt_snapshots is None`, i.e. N=0
    or N=1) get a deterministic `snapN` tag instead.
    """
    dt = config.get("dt_snapshots")
    if dt is not None:
        return _dt_tag(float(dt))
    return f"snap{int(config.get('n_snapshots') or 0)}"


def _config_hash(config: dict) -> str:
    """Short deterministic hash of a run's scientific configuration.

    Purely locational/control values (``out``, ``experiment``,
    ``overwrite``) and derived-artifact values (``plots``) are excluded so
    two runs of the same science produce the same hash regardless of
    output location or plot selection.
    """
    scrubbed = {k: v for k, v in config.items()
                if k not in ("out", "experiment", "overwrite", "plots")}
    blob = json.dumps(scrubbed, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=4).hexdigest()


def make_run_id(
    config: dict,
    *,
    now: Optional[datetime] = None,
    commit: Optional[str] = None,
) -> str:
    """Deterministic run identifier from `config` + wall clock + commit.

    Required config keys: ``scenario, day_hours, resolution, lmax,
    dt_snapshots`` — where ``dt_snapshots`` may be None for count-based
    snapshot modes without a uniform interval (N=0/N=1), in which case
    ``n_snapshots`` supplies the `snapN` tag instead.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ts = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    parts = [
        ts,
        _sanitize(str(config.get("scenario", "run"))),
        _rotation_tag(float(config.get("day_hours", math.inf))),
        f"r{int(config['resolution'])}",
        f"l{int(config['lmax'])}",
        _snapshot_tag(config),
    ]
    # Legacy interval-mode runs preserve their historical run-id format
    # exactly (dtNh token, no scientific hash) so downstream tooling that
    # matched on the old shape keeps working. New count-mode / N=0-or-1
    # runs get a 4-byte hash of the scientific config so distinct
    # backends, dimensions, duration, viscosity, quadrature, or snapshot
    # schedule do not collide at the same timestamp.
    if config.get("snapshot_mode") == "count":
        parts.append(_config_hash(config))
    if commit:
        parts.append(_sanitize(commit))
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

@dataclass
class RunDirectory:
    """A concrete, immutable-by-convention run output directory."""

    path: pathlib.Path
    run_id: str
    base: pathlib.Path
    experiment: Optional[str] = None
    commit: Optional[str] = None
    reused: bool = field(default=False)

    def figure_metadata(self, source: Optional[str] = None) -> dict:
        """Metadata dict for ``matplotlib.figure.Figure.savefig(metadata=...)``.

        ``source`` is a run-relative path to the data file the figure was
        rendered from (e.g. ``diagnostics/timeseries.csv``); omit for figures
        that don't have a single source file.
        """
        try:
            rel = self.path.relative_to(self.base)
        except ValueError:
            rel = self.path
        meta = {
            "Software": "planetary-sandbox",
            "RunId": self.run_id,
            "Experiment": self.experiment or "",
            "Commit": self.commit or "",
            "RunPath": str(rel),
            "Creation Time": datetime.now(timezone.utc).isoformat(),
        }
        if source:
            meta["Source"] = source
        return meta

    def update_latest_pointer(self) -> pathlib.Path:
        """Write/refresh ``{base}/latest_run.txt`` -> path of this run.

        The pointer is a plain text file (relative path if possible) so shell
        scripts can do ``$(cat runs/latest_run.txt)``.
        """
        pointer = self.base / "latest_run.txt"
        try:
            rel = self.path.relative_to(self.base)
            content = str(rel)
        except ValueError:
            content = str(self.path)
        pointer.write_text(content + "\n", encoding="utf-8")
        return pointer


def create_run_dir(
    base_dir: pathlib.Path | str,
    config: dict,
    *,
    experiment: Optional[str] = None,
    overwrite: bool = False,
    now: Optional[datetime] = None,
    commit: Optional[str] = None,
) -> RunDirectory:
    """Create a unique run directory and return a :class:`RunDirectory`.

    Layout::

        <base_dir>/
        ├── latest_run.txt              (written by update_latest_pointer())
        ├── <run_id>/                   ← if experiment is None
        └── <experiment>/<run_id>/      ← otherwise

    Collisions raise :class:`FileExistsError` unless ``overwrite=True``.
    """
    base = pathlib.Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    if commit is None:
        commit = _short_commit()

    run_id = make_run_id(config, now=now, commit=commit)
    parent = base / _sanitize(experiment) if experiment else base
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / run_id

    reused = False
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {path}\n"
                "Pass overwrite=True (or --overwrite on the CLI) to reuse it, "
                "or vary the config / wait a second so the timestamp advances."
            )
        reused = True
    else:
        path.mkdir(parents=True)

    return RunDirectory(
        path=path,
        run_id=run_id,
        base=base,
        experiment=experiment,
        commit=commit,
        reused=reused,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

#: Run lifecycle states recorded in manifest.json["status"].
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


def write_run_manifest(
    out_dir: pathlib.Path,
    run_config: dict,
    *,
    run_id: Optional[str] = None,
    experiment: Optional[str] = None,
    numerics: Optional[dict] = None,
    status: str = RUN_STATUS_RUNNING,
    error: Optional[dict] = None,
) -> pathlib.Path:
    """Write a ``manifest.json`` capturing everything needed to reproduce.

    ``status`` records the run lifecycle: 'running' is written before
    execution, then 'completed' or 'failed' overwrites the file when the
    run finishes. ``error`` holds ``{type, message}`` for failed runs so
    an operator can see at a glance why a capsule is incomplete.
    """
    versions = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "cupy", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None

    gpu = None
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        gpu = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    except Exception:
        pass

    git_status = _git(["status", "--porcelain"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": experiment,
        "status": status,
        "argv": sys.argv,
        "run_config": run_config,
        # Backend/state-sampling/product-sampling/transform provenance
        # (SphericalGridBackend.describe()); None for legacy callers.
        "numerics": numerics,
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(git_status) if git_status is not None else None,
        },
        "versions": versions,
        "gpu": gpu,
        "notes": {
            "equations": "barotropic vorticity equation on a rotating sphere (see docs/MATHEMATICAL_MODEL.md)",
            "timestep_policy": "fixed dt ceiling from initial CFL; individual steps may be shortened to land exactly on output times and t_end (docs/KNOWN_RISKS.md R-4)",
            "diagnostics": "see diagnostics.py module docstring for definitions",
        },
    }
    if error is not None:
        manifest["error"] = error

    path = pathlib.Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def update_manifest_status(out_dir: pathlib.Path, status: str,
                           error: Optional[dict] = None) -> None:
    """Rewrite an existing manifest.json's status and optional error block.

    Preserves all other manifest fields (numerics, versions, git). No-op if
    the manifest is missing or unreadable — the caller has already surfaced
    the underlying error.
    """
    path = pathlib.Path(out_dir) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    manifest["status"] = status
    if error is not None:
        manifest["error"] = error
    elif status == RUN_STATUS_COMPLETED and "error" in manifest:
        del manifest["error"]
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def failure_record(exc: BaseException) -> dict:
    """Concise error record for the manifest."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }
