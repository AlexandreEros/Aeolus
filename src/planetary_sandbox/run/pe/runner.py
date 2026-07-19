"""Dry primitive-equation run driver: integrate (fixed step), validate, persist.

Mirrors ``run/swe/runner.py`` on top of the shared integration engine
(``run.engine``) with ONE deliberate difference: the timestep is a
user-supplied **fixed** value, not the state-adaptive advective-CFL ceiling
the BVE/SWE runners use. This first runnable PE experiment makes no CFL
controller claim — it drives the existing :class:`IntegrationScheduler` (count
/ legacy-interval semantics unchanged) with a *constant* ``dt_seconds`` ceiling,
so every accepted step is exactly ``dt_seconds`` except where the scheduler
clips it to land exactly on a requested output time or ``t_end`` (no silent
final-time overshoot).

Validation is identical in spirit to the SWE runner: every RK4 intermediate
stage AND every accepted state is checked by ``model.validate_state`` (NaN/Inf,
per-level zeta/delta monopoles, strictly positive temperature, finite surface
pressure). Any violation aborts the run with an explicit
``PrimitiveEquationsStateError`` — the CLI lifecycle then marks the capsule
'failed'. Nothing here clips values, replaces NaNs, forces positivity, or
shrinks the perturbation to survive an unstable step.

Stored artifacts:

* ``pe_coeffs.npy``          (n_snapshots, 3K+1, l_max+1, l_max+1) spectral
                             states. Axis 1 carries the prognostic row
                             ordering ``[zeta_1..zeta_K, delta_1..delta_K,
                             T_1..T_K, ln p_s]`` (K = nlev); the trailing
                             (l_max+1, l_max+1) is the per-row spherical-
                             harmonic coefficient block in its natural
                             (degree, order) layout.
* ``pe_snapshot_times.npy``  stored times in seconds (exactly the requested
                             schedule).
* ``diagnostics/timeseries.csv``  per-step scalar diagnostics.
* ``figures/``               rendered when 'diagnostics' in ``plots``.
* ``pe_summary.png``         rendered when 'summary' in ``plots``.
* ``snapshots/physical/``    per-snapshot upper/lower figures + timeline.png,
                             rendered alongside the summary (same BVE/SWE
                             capsule-root snapshot-product layout).
"""
from __future__ import annotations

import math
import pathlib
from typing import Sequence

import numpy as np
import cupy as cp

from planetary_sandbox.physics.primitive_equations import (
    PrimitiveEquationsModel, PrimitiveEquationsState)
from ..engine import IntegrationScheduler, rk4_step_array
from .config import PE_PLOT_TYPES
from .diagnostics import PEDiagnosticsRecorder, plot_pe_diagnostics


def run_pe(model: PrimitiveEquationsModel,
           state0: PrimitiveEquationsState,
           *,
           dt_seconds: float,
           t_end_days: float,
           out_dir: pathlib.Path,
           snapshot_times: Sequence[float],
           snapshot_mode: str = "count",
           dt_snapshots: float | None = None,
           figure_metadata: dict | None = None,
           plots: Sequence[str] | None = None,
           scenario: str = "pe") -> int:
    """Integrate the dry primitive equations with a fixed step and persist."""
    t_end = t_end_days * 86400.0
    if not (math.isfinite(dt_seconds) and dt_seconds > 0):
        raise ValueError(f"dt_seconds must be finite and > 0, got {dt_seconds}")
    if snapshot_mode not in ("count", "interval"):
        raise ValueError(f"unknown snapshot_mode: {snapshot_mode!r}")
    plots = tuple(PE_PLOT_TYPES) if plots is None else tuple(plots)

    model.validate_state(state0, context="initial state")
    state = PrimitiveEquationsState(cp.array(state0.coeffs, copy=True))

    recorder = PEDiagnosticsRecorder(model, out_dir)
    last_row = recorder.record(0.0, state, dt=0.0, step=0)

    step = 0
    snapshots: list[np.ndarray] = []
    stored_times: list[float] = []

    def on_store(event_time: float) -> None:
        # Transfer to host immediately: stacking many separate device arrays
        # at the end of the run (cp.stack) hits a pathologically slow CuPy
        # path on small GPUs, stalling final persistence for minutes.
        snapshots.append(cp.asnumpy(state.coeffs))
        stored_times.append(event_time)
        print(f"Time: {event_time / 3600.0:8.3f} hrs | Step: {step} ")

    def validate_stage(y_stage: cp.ndarray) -> None:
        # RK4 intermediate stages must be physically valid too: an invalid
        # stage would otherwise feed the next tendency and could launder
        # itself back into an apparently valid accepted state.
        model.validate_state(PrimitiveEquationsState(y_stage),
                             context=f"in an RK4 stage after step {step}")

    def do_step(t_before: float, t_after: float, dt_step: float,
                step_index: int) -> None:
        nonlocal state, step, last_row
        state = PrimitiveEquationsState(
            rk4_step_array(model.tendency, state.coeffs, t_before, dt_step,
                           stage_validator=validate_stage))
        step = step_index
        # Hard validation after every accepted step: NaN/Inf, monopole
        # conservation, positive temperature, finite p_s — loud, not silent.
        model.validate_state(
            state, context=f"at t={t_after:g} s (step {step_index})")
        last_row = recorder.record(t_after, state, dt=dt_step, step=step_index)

    # Fixed-step driver: the scheduler is asked for one event at a time with a
    # CONSTANT ceiling, so it never adapts the step; count mode still clips the
    # step to land exactly on each output time and t_end (no overshoot).
    scheduler = IntegrationScheduler(
        t_end, mode=snapshot_mode,
        snapshot_times=snapshot_times, dt_snapshots=dt_snapshots)
    try:
        t = 0.0
        step_index = 0
        while True:
            event = scheduler.next_event(dt_seconds)
            if event is None:
                break
            kind, dt_step, event_time = event
            if kind == "store":
                on_store(event_time)
            else:  # step
                step_index += 1
                do_step(t, event_time, dt_step, step_index)
                t = event_time
    finally:
        recorder.close()

    if snapshots:
        coeffs_stack = np.stack(snapshots)
    else:
        coeffs_stack = np.empty((0, *state.coeffs.shape),
                                dtype=state.coeffs.dtype)
    np.save(out_dir / "pe_coeffs.npy", coeffs_stack)
    np.save(out_dir / "pe_snapshot_times.npy",
            np.asarray(stored_times, dtype=np.float64))

    # Final run summary: stored-snapshot count and the last diagnostic row, so
    # the CLI reports what actually happened without re-reading the capsule.
    print(f"Stored {len(stored_times)} snapshot(s) over "
          f"{step_index} fixed {dt_seconds:g} s step(s).")
    print(f"Final diagnostics @ t={last_row['time_s'] / 3600.0:.3f} h: "
          f"T[min,max]=[{last_row['t_min']:.2f}, {last_row['t_max']:.2f}] K, "
          f"p_s[min,max]=[{last_row['ps_min']:.1f}, {last_row['ps_max']:.1f}] Pa, "
          f"max|V|={last_row['max_wind_ms']:.3g} m/s, "
          f"mass drift={last_row['mass_rel_drift']:.2e}")

    if "diagnostics" in plots:
        try:
            plot_pe_diagnostics(out_dir, metadata=figure_metadata)
        except Exception as err:
            # Plotting must never take down a finished run; the CSV survives.
            print(f"Diagnostics plotting failed (data preserved): {err}")

    if "summary" in plots:
        # Part of the selected run product: a failure propagates so the shared
        # lifecycle marks the run failed and never publishes it as complete.
        # The single-level summary and the per-snapshot upper/lower figures are
        # rendered together from the just-persisted coefficient stack.
        from .visualization import render_pe_summary
        from .snapshot_visualization import render_pe_snapshots
        render_pe_summary(model, out_dir, metadata=figure_metadata)
        render_pe_snapshots(model, out_dir, metadata=figure_metadata,
                            scenario=scenario)

    return 0
