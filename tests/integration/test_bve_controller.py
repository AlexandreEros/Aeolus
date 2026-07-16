"""Controller-wiring tests for the state-adaptive advective-CFL loop.

These exercise the runner's scheduling/control seam (``_integrate``) with the
*real* scheduler and *real* CFL helper but purely stubbed physics, so they run
on CPU and prove — without a long numerical run — that the max speed returned
after accepted step ``n`` determines the CFL ceiling used to size step
``n+1``. See docs/KNOWN_RISKS.md R-4.
"""
from __future__ import annotations

from planetary_sandbox.run.bve.config import (
    IntegrationScheduler,
    advective_cfl_timestep,
)
from planetary_sandbox.run.bve.runner import _integrate


def test_speed_after_step_n_sets_ceiling_for_step_n_plus_1():
    length_scale = 1000.0                       # 0.5 * L / speed = 500 / speed
    t_end = 260.0
    # Post-step max speeds returned by the stubbed physics for steps 1, 2, 3,
    # then constant. Each drives the NEXT step's ceiling: 500/speed.
    speeds = [10.0, 50.0, 25.0]

    def speed_for(step):                        # 1-based step index
        return speeds[step - 1] if step - 1 < len(speeds) else speeds[-1]

    dt_steps: list[float] = []

    def on_step(t_before, t_after, dt_step, step):
        dt_steps.append(dt_step)
        return speed_for(step)

    stores: list[float] = []

    scheduler = IntegrationScheduler(
        t_end, mode="count", snapshot_times=[t_end])   # N=1: final only

    # First ceiling comes from the initial state (speed 5 -> 500/5 = 100).
    initial_dt_cfl = advective_cfl_timestep(length_scale, 5.0)
    assert initial_dt_cfl == 100.0

    final_t, n_steps = _integrate(
        scheduler, initial_dt_cfl, length_scale,
        on_step=on_step, on_store=stores.append)

    # Step 1 used the initial ceiling; each later step used the ceiling derived
    # from the PREVIOUS accepted step's returned speed.
    assert dt_steps[0] == 100.0                 # from initial speed 5
    assert dt_steps[1] == advective_cfl_timestep(length_scale, 10.0)  # 50
    assert dt_steps[2] == advective_cfl_timestep(length_scale, 50.0)  # 10
    assert dt_steps[3] == advective_cfl_timestep(length_scale, 25.0)  # 20
    # Every step honors its ceiling; the final one is clipped to land on t_end.
    assert final_t == t_end
    assert stores == [t_end]
    assert sum(dt_steps) == t_end               # exact accumulation to t_end


def test_integrate_recomputes_ceiling_only_after_steps_not_stores():
    """A store event does not advance time or change the ceiling."""
    length_scale = 500.0
    t_end = 100.0
    scheduler = IntegrationScheduler(
        t_end, mode="count", snapshot_times=[0.0, t_end])  # store at 0 and end

    ceilings_seen: list[float] = []

    def on_step(t_before, t_after, dt_step, step):
        ceilings_seen.append(dt_step)
        return 25.0                             # -> next ceiling 0.5*500/25 = 10

    stores: list[float] = []
    _integrate(scheduler, 10.0, length_scale,
               on_step=on_step, on_store=stores.append)

    # The store at t=0 comes first and must not have triggered a step.
    assert stores[0] == 0.0
    # Every step is sized at 10 (the ceiling from the constant 25 m/s speed),
    # clipped only on the final landing step.
    assert ceilings_seen[0] == 10.0
    assert stores[-1] == t_end
