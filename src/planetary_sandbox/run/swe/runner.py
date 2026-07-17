"""Shallow-water run driver: integrate, validate, persist, plot.

Mirrors ``run/bve/runner.py`` on top of the shared integration engine
(``run.engine``): the scheduler owns the step/store contract, the ceiling is
the state-adaptive advective+gravity-wave CFL (recomputed from every
accepted state's ``max_char_speed_ms``), and diagnostics rows are recorded
after every accepted step. After each accepted step the state is validated
(NaN/Inf, monopoles, positive fluid depth) and any violation aborts the run
with an explicit exception — the CLI lifecycle then marks the capsule
'failed'.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np
import cupy as cp

from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                     ShallowWaterState)
from ..engine import (IntegrationScheduler, advective_cfl_timestep,
                      integrate, rk4_step_array, validate_snapshot_schedule)
from .config import SWE_PLOT_TYPES
from .diagnostics import SWEDiagnosticsRecorder, plot_swe_diagnostics


def run_swe(model: ShallowWaterModel,
            state0: ShallowWaterState,
            dt_snapshots: float | None,
            t_end_days: float,
            out_dir: pathlib.Path,
            figure_metadata: dict | None = None,
            snapshot_times: Sequence[float] | None = None,
            plots: Sequence[str] | None = None,
            snapshot_mode: str | None = None) -> int:
    """Integrate the shallow-water equations and persist outputs.

    Snapshot semantics are identical to ``run_bve`` (count mode with exact
    target-time landing, or the legacy interval contract). Stored artifacts:

    * ``swe_coeffs.npy``          (n_snapshots, 3, l_max+1, l_max+1) spectral
                                  states, stacked [zeta, delta, phi]
    * ``swe_snapshot_times.npy``  stored times in seconds
    * ``diagnostics/timeseries.csv``  per-step scalar diagnostics
    * ``figures/``                rendered when 'diagnostics' in ``plots``
    """
    planet = model.planet
    t_end = t_end_days * 86400.0
    if snapshot_mode is None:
        snapshot_mode = "interval" if snapshot_times is None else "count"
    if snapshot_mode not in ("count", "interval"):
        raise ValueError(f"unknown snapshot_mode: {snapshot_mode!r}")
    if snapshot_mode == "count":
        if snapshot_times is None:
            raise ValueError("count mode requires an explicit snapshot_times schedule")
        snapshot_times = validate_snapshot_schedule(snapshot_times, t_end)
    else:
        if dt_snapshots is None:
            raise ValueError("interval mode requires dt_snapshots")
    plots = tuple(SWE_PLOT_TYPES) if plots is None else tuple(plots)

    model.validate_state(state0, context="initial state")
    state = ShallowWaterState(cp.array(state0.coeffs, copy=True))

    length_scale = getattr(planet.grid, "cfl_length_scale", None)

    recorder = SWEDiagnosticsRecorder(model, out_dir)
    initial_row = recorder.record(0.0, state, dt=0.0, step=0)
    dt_cfl = advective_cfl_timestep(length_scale,
                                    initial_row["max_char_speed_ms"])

    step = 0
    snapshots: list[np.ndarray] = []
    stored_times: list[float] = []

    def on_store(event_time: float) -> None:
        # Transfer to host immediately: stacking many separate device arrays
        # at the end of the run (cp.stack) hits a pathologically slow CuPy
        # path on small GPUs, stalling final persistence for minutes.
        snapshots.append(cp.asnumpy(state.coeffs))
        stored_times.append(event_time)
        print(f"Time: {event_time/3600.0:8.2f} hrs | Step: {step} ")

    def validate_stage(y_stage: cp.ndarray) -> None:
        # RK4 intermediate stages must be physically valid too: an invalid
        # stage would otherwise feed the next tendency and could launder
        # itself back into an apparently valid accepted state.
        model.validate_state(ShallowWaterState(y_stage),
                             context=f"in an RK4 stage after step {step}")

    def on_step(t_before: float, t_after: float, dt_step: float,
                step_index: int) -> float:
        nonlocal state, step
        state = ShallowWaterState(
            rk4_step_array(model.tendency, state.coeffs, t_before, dt_step,
                           stage_validator=validate_stage))
        step = step_index
        # Hard validation after every accepted step: NaN/Inf, monopole
        # conservation, and positive fluid depth fail loudly, not silently.
        model.validate_state(
            state, context=f"at t={t_after:g} s (step {step_index})")
        row = recorder.record(t_after, state, dt=dt_step, step=step_index)
        return row["max_char_speed_ms"]

    scheduler = IntegrationScheduler(
        t_end, mode=snapshot_mode,
        snapshot_times=snapshot_times, dt_snapshots=dt_snapshots)
    try:
        integrate(scheduler, dt_cfl, length_scale,
                  on_step=on_step, on_store=on_store)
    finally:
        recorder.close()

    if snapshots:
        coeffs_stack = np.stack(snapshots)
    else:
        coeffs_stack = np.empty((0, *state.coeffs.shape),
                                dtype=state.coeffs.dtype)
    np.save(out_dir / "swe_coeffs.npy", coeffs_stack)
    np.save(out_dir / "swe_snapshot_times.npy",
            np.asarray(stored_times, dtype=np.float64))

    if "diagnostics" in plots:
        try:
            plot_swe_diagnostics(out_dir, metadata=figure_metadata)
        except Exception as err:
            # Plotting must never take down a finished run; the CSV survives.
            print(f"Diagnostics plotting failed (data preserved): {err}")

    return 0
