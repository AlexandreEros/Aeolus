"""Run provenance: write a manifest.json so every output directory is a
self-contained, reproducible experiment record (branch, commit, dirty flag,
command line, package versions, GPU)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10,
            cwd=pathlib.Path(__file__).resolve().parents[4],
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def write_run_manifest(out_dir: pathlib.Path, run_config: dict) -> pathlib.Path:
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

    status = _git(["status", "--porcelain"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "run_config": run_config,
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(status) if status is not None else None,
        },
        "versions": versions,
        "gpu": gpu,
        "notes": {
            "equations": "barotropic vorticity equation on a rotating sphere (see MATHEMATICAL_MODEL.md)",
            "timestep_policy": "fixed dt from initial CFL (KNOWN_RISKS.md R-4)",
            "diagnostics": "see diagnostics.py module docstring for definitions",
        },
    }

    path = pathlib.Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
