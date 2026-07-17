"""Model-independent integration engine.

This module owns everything about *when* the solver steps and stores, and
nothing about *what* it integrates:

* snapshot schedule construction (count / legacy-interval semantics),
* the state-adaptive advective CFL ceiling policy,
* the incremental step/store scheduler (:class:`IntegrationScheduler`),
* the timestep driver loop (:func:`integrate`),
* a generic RK4 step for spectral-coefficient states.

It was extracted verbatim from ``run/bve/config.py`` and ``run/bve/runner.py``
so the barotropic-vorticity and shallow-water cores share one integration
engine; ``run.bve.config`` re-exports every moved name, so the historical
import surface is unchanged. The module is import-light (stdlib only, except
the optional CuPy import inside :func:`rk4_step_array`), so schedules and the
scheduler remain CPU-testable.

Physics models plug in through two callbacks (see :func:`integrate`):
``on_step`` advances one accepted step and returns the state's maximum
characteristic speed (m/s) — for the BVE that is max |u|; for shallow water
max(|u| + sqrt(Phi0 + phi)) — and ``on_store`` persists a snapshot. The
engine never inspects the state itself, which keeps the timestep controller
model-independent.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

SECONDS_PER_DAY = 86400.0


def _require_finite_number(name: str, value) -> float:
    """Reject NaN, +/-inf, and non-numeric values with a clear message."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {value}")
    return f


def _require_finite_positive(name: str, value) -> float:
    f = _require_finite_number(name, value)
    if not f > 0:
        raise ValueError(f"{name} must be > 0, got {f}")
    return f


def _require_finite_nonneg(name: str, value) -> float:
    f = _require_finite_number(name, value)
    if f < 0:
        raise ValueError(f"{name} must be >= 0, got {f}")
    return f


def interval_snapshot_times(dt_snapshots: float, t_end: float) -> list[float]:
    """Legacy interval-mode schedule: t=0 and every boundary up to t_end.

    Mirrors the historical runner countdown: the final state appears only
    when the duration is a multiple of the interval (within the historical
    ``1e-6 * dt`` tolerance).
    """
    tol = 1e-6 * dt_snapshots
    times: list[float] = []
    k = 0
    while k * dt_snapshots <= t_end + tol:
        times.append(min(k * dt_snapshots, t_end))
        k += 1
    return times


def count_snapshot_times(n_snapshots: int, t_end: float) -> list[float]:
    """Count-mode schedule: N evenly spaced states including both endpoints.

    N=0 -> []; N=1 -> [t_end]; N>=2 -> [0, ..., t_end] with exact endpoints.
    """
    if n_snapshots == 0:
        return []
    if n_snapshots == 1:
        return [t_end]
    spacing = t_end / (n_snapshots - 1)
    times = [i * spacing for i in range(n_snapshots)]
    times[-1] = t_end  # exact, no accumulated float error
    return times


#: Frozen CFL safety factor for the advective condition (dt = C * L / |u|).
#: Deliberately not a CLI/config field: this feature controls only the
#: advective CFL number, and exposing it would widen the run-identity schema.
CFL_NUMBER = 0.5

#: Fallback advective timestep (seconds) when the CFL condition cannot be
#: formed (no geometry length scale, or an exactly motionless state).
DEFAULT_CFL_FALLBACK_SECONDS = 600.0


def advective_cfl_timestep(length_scale: Optional[float], max_speed: float, *,
                           cfl_number: float = CFL_NUMBER,
                           fallback: float = DEFAULT_CFL_FALLBACK_SECONDS
                           ) -> float:
    """State-independent advective CFL ceiling ``cfl_number * L / max_speed``.

    This is the *only* place the advective ceiling is computed. The runner
    feeds it a fresh ``max_speed`` after every accepted step, so the ceiling
    tracks the evolving flow — genuine state-adaptive advective CFL stepping
    (see docs/KNOWN_RISKS.md R-4). It controls *only* the advective condition:
    RK4 stability for an explicit ``ν∇²`` (diffusion) term is deliberately not
    governed here. The controller is model-independent: the physics model
    supplies its own characteristic-speed estimate (max |u| for the BVE,
    max(|u| + sqrt(Phi0 + phi)) for shallow water).

    * A positive finite ``length_scale`` and positive finite ``max_speed``
      return ``cfl_number * length_scale / max_speed``.
    * A missing/zero ``length_scale`` or an exactly zero ``max_speed`` returns
      ``fallback`` (the historical 600 s ceiling).
    * NaN, infinity, a negative speed, a negative length scale, or a resulting
      non-finite / non-positive timestep are rejected with a clear exception.
    """
    speed = _require_finite_nonneg("max_speed", max_speed)
    if length_scale is None:
        return fallback
    scale = _require_finite_nonneg("length_scale", length_scale)
    if scale == 0.0 or speed == 0.0:
        return fallback
    dt = cfl_number * scale / speed
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"advective CFL timestep is not finite and positive "
            f"(got {dt} from length_scale={scale}, max_speed={speed}, "
            f"cfl_number={cfl_number})")
    return dt


def _count_step(t: float, next_stop: float,
                dt_cfl: float) -> tuple[float, float]:
    """Return ``(dt, t_after)`` for one exact-count planner step.

    Every representably positive residual is integrated. If a CFL-limited
    step is too small to advance the floating-point clock, abort explicitly
    rather than silently replacing it with the (potentially much larger)
    remaining gap and violating the CFL ceiling.
    """
    gap = next_stop - t
    if gap <= 0.0:
        raise RuntimeError(
            f"count planner expected a positive residual, got {gap} "
            f"at t={t} for target={next_stop}")
    dt = min(dt_cfl, gap)
    if dt == gap:
        return dt, next_stop
    t_after = t + dt
    if t_after <= t:
        raise FloatingPointError(
            f"count planner step stagnated at t={t}: dt_cfl={dt_cfl} "
            f"cannot advance time toward target={next_stop}")
    if t_after > next_stop:
        raise FloatingPointError(
            f"count planner step overshot target={next_stop}: "
            f"t={t}, dt={dt}, t_after={t_after}")
    return dt, t_after


def _count_events(t_end: float, targets: Sequence[float]):
    """Incremental generator of the exact-target count-mode step/store events.

    A coroutine: each ``yield`` emits one event and receives (via ``send``) the
    ``dt_cfl`` ceiling to use for the *next* step it computes. The ceiling for
    a step is therefore whatever the driver supplies on the call that returns
    that step — which is how the runner makes the flow speed of the newly
    accepted state govern the next accepted step (state-adaptive advective
    CFL). A store event ignores the ceiling supplied with it.

    Contract (unchanged from the historical fixed-plan version):

    * A target with a representably *positive* residual (``target - t > 0``) is
      never treated as reached; its residual is integrated, however small.
    * Steps toward a target are clipped to land exactly on it; the resulting
      ``t`` is snapped to the target to kill accumulated float drift, so the
      stored snapshot time and the final diagnostic time equal the requested
      target exactly.
    * No zero-length steps; a sub-ULP ceiling that cannot advance the clock
      raises ``FloatingPointError`` (via :func:`_count_step`) rather than
      silently violating the ceiling.

    Each event is ``("store", 0.0, time)`` or ``("step", dt, t_after)``.
    """
    t = 0.0
    i = 0
    n = len(targets)
    dt_cfl = yield  # prime: the first send supplies the ceiling for event #1
    while True:
        # Store only targets actually reached. There is deliberately no
        # pre-target tolerance: even a one-ULP positive residual gets its own
        # positive integration step before the snapshot is recorded.
        while i < n:
            target = float(targets[i])
            if target > t:
                break
            dt_cfl = yield ("store", 0.0, target)
            i += 1
        if i >= n and t >= t_end:
            return
        next_stop = float(targets[i]) if i < n else t_end
        dt, t = _count_step(t, next_stop, dt_cfl)
        dt_cfl = yield ("step", dt, t)


def _interval_events(t_end: float, dt_snapshots: float):
    """Incremental generator of the historical interval-mode events.

    Mirrors the original psx-bve countdown exactly: store at t=0 and every
    interval boundary, decrement a ``time_to_snapshot`` countdown, and stop
    once the remaining time falls within ``1e-6 * dt_snapshots``. The final
    state is stored only when the duration is a multiple of the interval — the
    intentional legacy stopping behavior. Stored times are the actual
    accumulated ``t`` values (with their historical float drift).

    Like :func:`_count_events` it is a coroutine: each accepted step uses the
    ``dt_cfl`` sent on the call that returns it, so the accepted-step sequence
    now varies with the evolving flow speed while the snapshot/stopping
    semantics stay bit-for-bit identical.
    """
    t = 0.0
    snapshot_tol = 1e-6 * dt_snapshots
    time_to_snapshot = 0.0
    dt_cfl = yield  # prime
    while t <= t_end + snapshot_tol:
        if time_to_snapshot <= snapshot_tol:
            dt_cfl = yield ("store", 0.0, t)
            time_to_snapshot = dt_snapshots
        remaining = t_end - t
        if remaining <= snapshot_tol:
            break
        dt_step = min(dt_cfl, time_to_snapshot, remaining)
        if dt_step <= 0:
            break
        t += dt_step
        time_to_snapshot = max(0.0, time_to_snapshot - dt_step)
        dt_cfl = yield ("step", dt_step, t)


class IntegrationScheduler:
    """Stateful step/store scheduler that owns only time + snapshot bookkeeping.

    The runner requests one event at a time, supplying the *current* advective
    CFL ceiling each call::

        scheduler = IntegrationScheduler(t_end, mode=snapshot_mode,
                                         snapshot_times=snapshot_times,
                                         dt_snapshots=dt_snapshots)
        while True:
            event = scheduler.next_event(dt_cfl)
            if event is None:
                break
            ...

    This is the seam that makes state-adaptive stepping possible: because the
    ceiling is consumed one event at a time (rather than baked into a
    precomputed plan), a ceiling recomputed from each newly accepted state
    governs the very next accepted step. The scheduler is independent of CuPy
    and model physics, so the exact-count and legacy-interval contracts remain
    CPU-testable.

    ``mode`` is explicit ("count" or "interval"); the runner never infers the
    legacy path from ``snapshot_times is None``.
    """

    def __init__(self, t_end: float, *, mode: str,
                 snapshot_times: Optional[Sequence[float]] = None,
                 dt_snapshots: Optional[float] = None):
        self.mode = mode
        if mode == "count":
            self._gen = _count_events(t_end, list(snapshot_times or []))
        elif mode == "interval":
            if dt_snapshots is None:
                raise ValueError("interval mode requires dt_snapshots")
            self._gen = _interval_events(t_end, float(dt_snapshots))
        else:
            raise ValueError(f"unknown integration mode: {mode!r}")
        next(self._gen)  # advance to the priming yield
        self._done = False

    def next_event(self, dt_cfl: float) -> Optional[tuple]:
        """Return the next ``("store"/"step", ...)`` event, or ``None`` at end.

        ``dt_cfl`` is the current positive finite advective ceiling; a returned
        ``step`` event is sized against it (clipped shorter only to land on the
        next snapshot target or ``t_end``). A returned ``store`` event does not
        consume the ceiling.
        """
        if dt_cfl <= 0 or not math.isfinite(dt_cfl):
            raise ValueError(f"dt_cfl must be finite and positive, got {dt_cfl}")
        if self._done:
            return None
        try:
            return self._gen.send(dt_cfl)
        except StopIteration:
            self._done = True
            return None


def integration_plan(t_end: float, dt_cfl: float, *, mode: str,
                     snapshot_times: Optional[Sequence[float]] = None,
                     dt_snapshots: Optional[float] = None) -> list[tuple]:
    """Materialize the whole step/store sequence for a *constant* ceiling.

    Compatibility / test helper: it drives :class:`IntegrationScheduler` with a
    single fixed ``dt_cfl`` and collects every event, reproducing the historical
    fixed-ceiling plan exactly. The live runner no longer builds a full plan —
    it steps the scheduler one event at a time with a ceiling recomputed from
    each accepted state — but this helper keeps the deterministic, physics-free
    contract unit-testable on CPU.
    """
    if dt_cfl <= 0 or not math.isfinite(dt_cfl):
        raise ValueError(f"dt_cfl must be finite and positive, got {dt_cfl}")
    scheduler = IntegrationScheduler(
        t_end, mode=mode, snapshot_times=snapshot_times,
        dt_snapshots=dt_snapshots)
    events: list[tuple] = []
    while True:
        event = scheduler.next_event(dt_cfl)
        if event is None:
            break
        events.append(event)
    return events


def scheduler_tolerance(t_end: float, times: Sequence[float]) -> float:
    """Scale/gap-aware tolerance for matching schedule entries at runtime.

    Small enough that two distinct entries are never coalesced (a fraction
    of the smallest positive inter-entry gap), and small enough that a
    short simulation is not entirely consumed (a fraction of t_end), while
    staying well above floating-point noise (>= 1e-12 s).
    """
    tol = max(1e-12, 1e-9 * t_end) if t_end > 0 else 1e-12
    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if gaps:
            tol = min(tol, 0.25 * min(gaps))
    return tol


def validate_snapshot_schedule(times: Sequence[float], t_end: float) -> list[float]:
    """Return a clean schedule; raise ValueError on any anomaly.

    Entries must be finite, strictly increasing, non-duplicated, and lie in
    [0, t_end] (allowing a scale-aware slack against float noise on the
    endpoints; interior duplicates are always rejected).
    """
    if t_end <= 0 or not math.isfinite(t_end):
        raise ValueError(f"t_end must be finite and positive, got {t_end}")
    cleaned: list[float] = []
    slack = max(1e-9, 1e-9 * t_end)
    for i, t in enumerate(times):
        if not math.isfinite(t):
            raise ValueError(
                f"snapshot_times[{i}] = {t} is not finite")
        if t < -slack or t > t_end + slack:
            raise ValueError(
                f"snapshot_times[{i}] = {t} is outside [0, {t_end}]")
        clipped = max(0.0, min(t_end, float(t)))
        if cleaned and not clipped > cleaned[-1]:
            raise ValueError(
                "snapshot_times must be strictly increasing without "
                f"duplicates; got {times!r}")
        cleaned.append(clipped)
    return cleaned


def integrate(scheduler: IntegrationScheduler, dt_cfl: float,
              length_scale: Optional[float], *, on_step, on_store) -> tuple:
    """Drive an IntegrationScheduler with state-adaptive advective-CFL stepping.

    Physics-agnostic control seam (kept separate so it is CPU-testable with
    stubbed callbacks): the scheduler is asked for one event at a time using
    the *current* ceiling ``dt_cfl``; after each accepted step the ceiling is
    recomputed from that step's max characteristic speed, so the newly
    accepted state governs the next accepted step.

    ``on_step(t_before, t_after, dt_step, step)`` advances one accepted RK step
    and returns the post-step maximum characteristic speed (m/s);
    ``on_store(event_time)`` persists a snapshot (and does not affect the
    ceiling). Returns the final ``(t, step)`` reached.
    """
    t = 0.0
    step = 0
    while True:
        event = scheduler.next_event(dt_cfl)
        if event is None:
            break
        kind, dt_step, event_time = event
        if kind == "store":
            on_store(event_time)
        else:  # step
            step += 1
            max_speed = on_step(t, event_time, dt_step, step)
            t = event_time
            dt_cfl = advective_cfl_timestep(length_scale, max_speed)
    return t, step


def rk4_step_array(tendency, y, t: float, dt: float, *,
                   stage_validator=None):
    """Classical RK4 step for a state stored as one array of coefficients.

    ``tendency(y)`` must return an array of the same shape. Used by the
    shallow-water core (state = stacked (3, l_max+1, l_max+1) coefficients);
    the BVE keeps its historical ``rk4_step`` in ``run/bve/runner.py`` so its
    numerics are guaranteed byte-identical.

    ``stage_validator(y_stage)``, when given, is called on each INTERMEDIATE
    stage state (y + dt/2*k1, y + dt/2*k2, y + dt*k3) before its tendency is
    evaluated, and should raise on a physically invalid state. Without it, a
    stage can pass through an invalid region (e.g. negative fluid depth) and
    still return an apparently valid final state — the failure must be
    explicit, not laundered through the final linear combination.
    """
    def _stage(y_stage):
        if stage_validator is not None:
            stage_validator(y_stage)
        return y_stage

    k1 = tendency(y)
    k2 = tendency(_stage(y + 0.5 * dt * k1))
    k3 = tendency(_stage(y + 0.5 * dt * k2))
    k4 = tendency(_stage(y + dt * k3))
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
