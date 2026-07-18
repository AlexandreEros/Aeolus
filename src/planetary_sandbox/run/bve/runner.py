from __future__ import annotations

import numpy as np
import cupy as cp
import pathlib
from typing import Sequence

from planetary_sandbox.planet import Planet
from .barotropic_vorticity import BarotropicVorticity, BarotropicState
from .config import (PLOT_TYPES, IntegrationScheduler, advective_cfl_timestep,
                     validate_snapshot_schedule)
# The physics-agnostic driver loop lives in the shared engine; `_integrate`
# is kept as this module's historical name for it (tests import it here).
from ..engine import integrate as _integrate
from .diagnostics import DiagnosticsRecorder, plot_diagnostics
from ...viz.vorticity_viewer import VorticityViewer
from ...viz.renderers import get_default_renderer


def _empty_coeffs_stack(zeta0_lm: cp.ndarray) -> np.ndarray:
    """(0, l_max+1, l_max+1) stack matching a real snapshot's dtype/shape."""
    return np.empty((0, *zeta0_lm.shape), dtype=zeta0_lm.dtype)


def _empty_grid_stack(zeta0_grid: cp.ndarray) -> np.ndarray:
    """(0, *grid_shape) stack matching a real grid snapshot's dtype/shape."""
    return np.empty((0, *zeta0_grid.shape), dtype=zeta0_grid.dtype)


def run_bve(planet: Planet,
            zeta0_lm: cp.ndarray,
            dt_snapshots: float | None,
            t_end_days: float,
            out_dir: pathlib.Path,
            viscosity: float,
            scenario: str = "two_vortices",
            figure_metadata: dict | None = None,
            snapshot_times: Sequence[float] | None = None,
            plots: Sequence[str] | None = None,
            snapshot_mode: str | None = None) -> int:
    """Integrate the BVE and persist snapshots, diagnostics, and figures.

    ``snapshot_mode`` selects the execution semantics **explicitly** at the
    runner boundary (the caller states which contract it wants; the runner
    never infers the legacy path from ``snapshot_times is None``):

    * ``"count"`` — the authoritative ``snapshot_times`` schedule is consumed
      with **exact target-time** semantics: every scheduled time (including
      ``t_end`` for N>=1) is landed on exactly, so the stored snapshot times
      and the final diagnostic time equal the requested targets. Guaranteed
      even for deliberately non-aligned durations.
    * ``"interval"`` — historical interval-boundary semantics reconstructed
      from ``dt_snapshots`` (stored at t=0 and each boundary; the final state
      only when the duration is a multiple of the interval; legacy
      ``1e-6 * dt_snapshots`` stopping tolerance). Preserved bit-for-bit.

    When ``snapshot_mode`` is None it is inferred for backward compatibility
    with direct callers (``"count"`` if an explicit ``snapshot_times`` is
    given, else ``"interval"``); the installed ``execute_run`` always passes
    it explicitly. ``plots`` selects which image products to render (subset
    of ``config.PLOT_TYPES``); None renders everything. Field-snapshot
    persistence and per-step numerical diagnostics are independent of
    ``plots``.
    """
    state = BarotropicState(coeffs=zeta0_lm)
    model = BarotropicVorticity(planet, scenario=scenario, viscosity=viscosity)

    t_end = t_end_days * 86400.0
    if snapshot_mode is None:
        snapshot_mode = "interval" if snapshot_times is None else "count"
    if snapshot_mode not in ("count", "interval"):
        raise ValueError(f"unknown snapshot_mode: {snapshot_mode!r}")

    if snapshot_mode == "count":
        if snapshot_times is None:
            raise ValueError("count mode requires an explicit snapshot_times schedule")
        snapshot_times = validate_snapshot_schedule(snapshot_times, t_end)
    else:  # interval
        if dt_snapshots is None:
            raise ValueError("interval mode requires dt_snapshots")
    plots = tuple(PLOT_TYPES) if plots is None else tuple(plots)

    # Geometry-owned CFL length scale (geodesic: min edge length; lat-lon: min
    # meridional spacing — see the geometry's cfl_length_scale docstring).
    # GridGeometry guarantees the attribute (base returns None; geodesic routes
    # min_edge_length through it). None/0 -> the fixed fallback in the helper.
    length_scale = getattr(planet.grid, "cfl_length_scale", None)

    # Grid-space initial vorticity — captured for provenance so the
    # summary plot can compare against the genuine initial field even
    # when only the final state is stored (N=1 case).
    zeta_initial_grid = planet.sh.inv_transform(state.coeffs)

    # Scalar diagnostics recorded every accepted step, straight from the
    # spectral state (not the plotting fields). Cheap; append-only.
    # Independent of the snapshot schedule and of plot selection.
    recorder = DiagnosticsRecorder(
        sh=planet.sh, so=planet.so, grid=planet.grid,
        radius=planet.params.radius,
        omega=planet.params.angular_velocity,
        out_dir=out_dir,
    )
    # The initial diagnostics row already reconstructs velocity and reports
    # max_speed_ms; reuse it for the *first* advective CFL ceiling instead of
    # inverting the streamfunction a second time here.
    initial_row = recorder.record(0.0, state.coeffs, dt=0.0, step=0)
    dt_cfl = advective_cfl_timestep(length_scale, initial_row["max_speed_ms"])

    step = 0
    all_zeta_lm: list[cp.ndarray] = []
    vorticity_grid_snapshot_list: list[cp.ndarray] = []
    stored_times_hours: list[float] = []

    def on_store(event_time: float) -> None:
        # Transfer to host immediately: stacking many separate device arrays
        # at the end of the run (cp.stack) hits a pathologically slow CuPy
        # path on small GPUs, stalling final persistence for minutes.
        all_zeta_lm.append(cp.asnumpy(state.coeffs))
        print(f"Time: {event_time/3600.0:8.2f} hrs | Step: {step} ")
        # Record the scheduled time (exact target in count mode), not a drifted
        # accumulator, so the stored snapshot time is authoritative.
        stored_times_hours.append(event_time / 3600.0)
        # Dump ζ on grid for plotting.
        zeta_grid = planet.sh.inv_transform(state.coeffs)
        vorticity_grid_snapshot_list.append(cp.asnumpy(zeta_grid))

    def on_step(t_before: float, t_after: float, dt_step: float,
                step_index: int) -> float:
        nonlocal state, step
        state = rk4_step(model, state, t_before, dt_step)
        # t_after is the scheduler's exact post-step time; in count mode a
        # landing step snaps it to the target, so the final diagnostic time
        # equals t_end exactly rather than a drifted value.
        step = step_index
        # record() performs the ONE velocity reconstruction of the step: it
        # both writes the diagnostic row and returns max_speed_ms, which drives
        # the next advective CFL ceiling. No second reconstruction is done.
        row = recorder.record(t_after, state.coeffs, dt=dt_step,
                              step=step_index)
        return row["max_speed_ms"]

    # State-adaptive advective CFL stepping: the scheduler is stepped one event
    # at a time with the *current* ceiling, and after every accepted RK4 step
    # the ceiling is recomputed from that state's max speed. The exact-target
    # (count) and legacy-interval snapshot/stopping contracts live in the
    # scheduler, keeping them CPU-testable.
    scheduler = IntegrationScheduler(
        t_end, mode=snapshot_mode,
        snapshot_times=snapshot_times, dt_snapshots=dt_snapshots)
    _integrate(scheduler, dt_cfl, length_scale,
               on_step=on_step, on_store=on_store)

    recorder.close()

    # Field-snapshot persistence is independent of plot selection.
    # For empty schedules, keep the same array contract as ordinary runs
    # (leading time axis, matching per-snapshot shape and dtype) so a
    # downstream `np.load(...)[i]` continues to work.
    if all_zeta_lm:
        coeffs_stack = np.stack(all_zeta_lm)
        grid_stack = np.stack(vorticity_grid_snapshot_list)
    else:
        coeffs_stack = _empty_coeffs_stack(zeta0_lm)
        grid_stack = _empty_grid_stack(zeta_initial_grid)
    np.save(out_dir / "vorticity_coeffs.npy", coeffs_stack)
    np.save(out_dir / "vorticity_grid.npy", grid_stack)

    # Image products, in the fixed order of PLOT_TYPES:
    # diagnostics -> snapshots -> summary. The initial grid field is
    # always supplied so the summary can render an honest initial-vs-final
    # comparison even when only the final state is stored (N=1).
    if "diagnostics" in plots:
        try:
            plot_diagnostics(out_dir, metadata=figure_metadata)
        except Exception as err:
            # Plotting must never take down a finished run; the CSV/npz survive.
            print(f"Diagnostics plotting failed (data preserved): {err}")

    wants_viewer = ("snapshots" in plots or "summary" in plots)
    if wants_viewer and grid_stack.shape[0] > 0:
        snapshot_times_arr = (
            np.array(stored_times_hours, dtype=np.float64)
            if stored_times_hours else np.empty((0,), dtype=np.float64))

        viewer = VorticityViewer(
            planet,
            scenario=scenario,
            vorticity_snapshots=grid_stack,
            times=snapshot_times_arr,
            initial_field=cp.asnumpy(zeta_initial_grid))

        if "snapshots" in plots:
            # Individual snapshot plots for debugging
            viewer.plot_all_snapshots(scenario=scenario, out_dir=out_dir,
                                      metadata=figure_metadata)

        if "summary" in plots:
            summary, stats = viewer.summary_spec()
            get_default_renderer().render_figure(
                summary, out_dir / "bve_summary.png",
                metadata=figure_metadata)
            print("VorticityViewer: Summary plot generated.")
            print(stats)

    return 0



def rk4_step(model: BarotropicVorticity, y: BarotropicState, t: float, dt: float, forcing_coeffs=None) -> BarotropicState:
    k1 = model.tendency(y, forcing_coeffs)
    k2 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k1), forcing_coeffs)
    k3 = model.tendency(BarotropicState(y.coeffs + 0.5*dt*k2), forcing_coeffs)
    k4 = model.tendency(BarotropicState(y.coeffs + dt*k3), forcing_coeffs)
    return BarotropicState(y.coeffs + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
