"""Solver-independent run lifecycle: base dir, provenance, status, pointer.

Extracted from ``cli/bve.py::execute_run`` so the BVE and shallow-water
commands share one implementation of the (carefully ordered) run lifecycle:

1. Resolve the output base directory (with the writability fallback).
2. Create the run directory.
3. On --overwrite reuse: clear latest_run.txt if it points at this
   directory, then remove known stale generated artifacts — both *before*
   the manifest is replaced / status set back to 'running'.
4. Write config.json and manifest.json with status='running' (atomically)
   *before* any numerical work.
5. Dispatch the solver callback.
6. On success, rewrite manifest.json with status='completed' (raising if
   that cannot be persisted) and only then publish latest_run.txt.
7. On failure, rewrite manifest.json with status='failed' + a concise error
   record, do NOT publish latest_run.txt, and re-raise. A provenance-
   persistence failure on this path is surfaced, never swallowed.

The solver-specific pieces are injected as callables: ``solver(cfg,
run_dir, run_config)`` runs the heavy numerics, ``clean_artifacts(out_dir)``
removes that solver's stale generated outputs on --overwrite reuse.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
from typing import Callable, Optional


def resolve_writable_base_dir(requested_out: str,
                              fallback_prefix: str = "psx-bve-"
                              ) -> tuple[pathlib.Path, bool]:
    """Ensure the runs *base* directory exists and is writable; fall back if not.

    Base-directory creation itself is inside the guarded path: an OSError
    from ``mkdir`` triggers the documented temporary-directory fallback
    instead of escaping as an unhandled exception.
    """
    requested = pathlib.Path(requested_out)

    def _try_probe(target: pathlib.Path) -> bool:
        """Attempt to create ``target`` and write a probe file inside it."""
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        probe_path = target / ".write_probe"
        try:
            probe_path.write_text("ok", encoding="utf-8")
        except OSError:
            return False
        finally:
            # Robust cleanup: even if the probe write failed midway (e.g.
            # a partial file was created), remove it silently. Missing
            # file is fine; we only want to avoid leaving a stray probe.
            with contextlib.suppress(OSError):
                probe_path.unlink(missing_ok=True)
        return True

    if _try_probe(requested):
        return requested, False

    # Fall back to the system temp location, NOT dir="." — dropping the
    # temp dir into the repo root left persistent, sometimes ACL-locked
    # directories behind (one of which broke bare `pytest` collection).
    # Keep run outputs out of the project tree entirely.
    fallback_root = pathlib.Path(tempfile.mkdtemp(prefix=fallback_prefix))
    fallback_out_dir = fallback_root / requested.name
    fallback_out_dir.mkdir(parents=True, exist_ok=True)
    return fallback_out_dir, True


def execute_with_provenance(cfg, *,
                            solver: Callable,
                            clean_artifacts: Callable[[pathlib.Path], None],
                            resolve_base_dir: Callable,
                            notes: Optional[dict] = None) -> int:
    """Execute one resolved run inside the full provenance lifecycle.

    ``cfg`` must expose ``out``, ``experiment``, ``overwrite``, and
    ``to_run_config_dict()``. ``notes`` overrides the manifest's descriptive
    notes block (None keeps the historical BVE notes).
    """
    from planetary_sandbox.run.bve.io import (
        RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, RUN_STATUS_RUNNING,
        RunProvenanceError, atomic_write_text, create_run_dir, failure_record,
        update_manifest_status, write_run_manifest)

    base_dir, used_fallback = resolve_base_dir(cfg.out)
    if used_fallback:
        banner = "=" * 72
        print(banner)
        print(f"WARNING: requested --out '{cfg.out}' is not writable.")
        print(f"         Run outputs will be written OUTSIDE the project tree to:")
        print(f"           {base_dir.resolve()}")
        print(banner)

    # Build the config dict *before* creating the run dir so the run ID
    # reflects exactly what will be written to disk.
    run_config = cfg.to_run_config_dict()
    run_config["out"] = str(base_dir)

    run_dir = create_run_dir(
        base_dir, run_config,
        experiment=cfg.experiment,
        overwrite=cfg.overwrite,
    )
    out_dir = run_dir.path
    print(f"Run directory: {out_dir}")
    run_config["run_id"] = run_dir.run_id
    run_config["experiment"] = cfg.experiment
    if run_dir.reused:
        print("  (reused via --overwrite; stale generated artifacts will be removed)")
        # Invalidate the pointer first: if latest_run.txt references this
        # directory, a failure of the overwritten run must not leave the
        # pointer aimed at an incomplete capsule.
        run_dir.clear_latest_pointer_if_matches()
        # Transition the existing capsule away from 'completed' before any
        # destructive cleanup. If cleanup fails after partially removing
        # results, persist 'failed' + the cleanup error so the damaged capsule
        # can never continue to claim successful completion.
        try:
            update_manifest_status(out_dir, RUN_STATUS_RUNNING)
            clean_artifacts(out_dir)
        except BaseException as exc:
            try:
                update_manifest_status(out_dir, RUN_STATUS_FAILED,
                                       error=failure_record(exc))
            except RunProvenanceError as prov_err:
                raise prov_err from exc
            raise

    # Write initial provenance so an interrupted or failing run leaves a
    # traceable capsule marked 'running' / 'failed', not a silent hole.
    atomic_write_text(out_dir / "config.json", json.dumps(run_config, indent=2))
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       status=RUN_STATUS_RUNNING, notes=notes)

    try:
        solver(cfg, run_dir, run_config)
    except BaseException as exc:
        # Failed runs must not leave latest_run.txt pointing at an empty
        # capsule. Mark status='failed' and re-raise; the CLI harness maps
        # the exception to a nonzero exit as it did before. If persisting the
        # 'failed' status itself fails, surface that (chained to the run
        # failure) rather than swallowing the provenance error.
        try:
            update_manifest_status(out_dir, RUN_STATUS_FAILED,
                                   error=failure_record(exc))
        except RunProvenanceError as prov_err:
            raise prov_err from exc
        raise

    # Completion must be durably persisted before the pointer is published;
    # update_manifest_status raises if it cannot write the 'completed' status,
    # so latest_run.txt is never published for an unpersisted completion.
    update_manifest_status(out_dir, RUN_STATUS_COMPLETED)
    run_dir.update_latest_pointer()
    return 0
