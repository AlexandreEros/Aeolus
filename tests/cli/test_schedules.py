"""Snapshot schedule construction and count/interval mode contracts."""
from __future__ import annotations

import math

import pytest

from planetary_sandbox.cli.main import build_parser
from planetary_sandbox.run.bve.config import (
    SECONDS_PER_DAY,
    BVERunConfig,
    IntegrationScheduler,
    _count_step,
    advective_cfl_timestep,
    count_snapshot_times,
    integration_plan,
    interval_snapshot_times,
)

from .conftest import run_aeolus_stubbed


# ---------------------------------------------------------------------------
# integration_plan: the constant-ceiling compatibility/test helper. The live
# runner steps IntegrationScheduler one event at a time with a ceiling
# recomputed from each accepted state (see the scheduler tests below); this
# helper drives the scheduler with a single fixed dt_cfl, so its whole plan is
# deterministic and physics-free on CPU — preserving the historical contract.
# ---------------------------------------------------------------------------

def _plan_stores(plan):
    return [t for kind, _dt, t in plan if kind == "store"]


def _plan_final_time(plan):
    steps = [t for kind, _dt, t in plan if kind == "step"]
    return steps[-1] if steps else 0.0


def _reference_interval_plan(t_end, dt_snapshots, dt_cfl):
    """Verbatim transcription of main's runner bookkeeping (no physics).

    Used to prove the installed legacy interval path reproduces the historical
    accepted-step sequence and stopping tolerance bit-for-bit.
    """
    t = 0.0
    snapshot_tol = 1e-6 * dt_snapshots
    time_to_snapshot = 0.0
    stores, steps = [], []
    while t <= t_end + snapshot_tol:
        if time_to_snapshot <= snapshot_tol:
            stores.append(t)
            time_to_snapshot = dt_snapshots
        remaining = t_end - t
        if remaining <= snapshot_tol:
            break
        dt_step = min(dt_cfl, time_to_snapshot, remaining)
        if dt_step <= 0:
            break
        t += dt_step
        time_to_snapshot = max(0.0, time_to_snapshot - dt_step)
        steps.append(dt_step)
    return stores, steps, t


# A deliberately non-aligned duration: not a multiple of any snapshot spacing,
# and with a sub-microsecond fractional part that a coarse tolerance would eat.
_MISALIGNED_T_END = 600.0000003


def test_count_n1_stores_final_state_at_exact_t_end():
    plan = integration_plan(
        _MISALIGNED_T_END, 250.0, mode="count",
        snapshot_times=count_snapshot_times(1, _MISALIGNED_T_END))
    assert _plan_stores(plan) == [_MISALIGNED_T_END]        # exact, not 600.0
    assert _plan_final_time(plan) == _MISALIGNED_T_END      # diagnostics exact


def test_count_n0_diagnostics_end_at_exact_t_end():
    plan = integration_plan(
        _MISALIGNED_T_END, 250.0, mode="count",
        snapshot_times=count_snapshot_times(0, _MISALIGNED_T_END))
    assert _plan_stores(plan) == []
    assert _plan_final_time(plan) == _MISALIGNED_T_END


def test_count_explicit_intermediate_not_stored_early():
    """A sub-microsecond pre-target residual must be integrated, not eaten."""
    residual = 1e-7
    target = 300.0 + residual
    schedule = [0.0, target, _MISALIGNED_T_END]
    plan = integration_plan(
        _MISALIGNED_T_END, 250.0, mode="count", snapshot_times=schedule)
    stores = _plan_stores(plan)
    assert stores == schedule                    # every target hit exactly
    assert target in stores                       # not clamped to 300.0
    assert _plan_final_time(plan) == _MISALIGNED_T_END


def test_count_large_duration_no_early_termination():
    """A huge valid duration must not grow a tolerance that ends the run early."""
    t_end = 8.64e8  # 10,000 days in seconds
    plan = integration_plan(
        t_end, t_end, mode="count",  # single CFL step spanning the run
        snapshot_times=count_snapshot_times(1, t_end))
    assert _plan_stores(plan) == [t_end]
    assert _plan_final_time(plan) == t_end


def test_count_multistep_lands_on_targets_exactly():
    t_end = _MISALIGNED_T_END
    schedule = count_snapshot_times(4, t_end)  # [0, t/3, 2t/3, t]
    plan = integration_plan(t_end, 71.0, mode="count", snapshot_times=schedule)
    # Stored times equal the requested schedule exactly, and the diagnostics
    # end exactly at t_end despite the odd CFL step and misaligned duration.
    assert _plan_stores(plan) == schedule
    assert _plan_final_time(plan) == t_end
    # No zero-length steps.
    assert all(dt > 0 for kind, dt, _ in plan if kind == "step")


def test_count_never_treats_positive_residual_as_reached():
    """A one-ULP intermediate residual gets a step before its store event."""
    target = math.nextafter(1.0, math.inf)
    plan = integration_plan(
        2.0, 1.0, mode="count", snapshot_times=[target, 2.0])
    assert plan[:3] == [
        ("step", 1.0, 1.0),
        ("step", target - 1.0, target),
        ("store", 0.0, target),
    ]
    assert _plan_final_time(plan) == 2.0


@pytest.mark.parametrize("n_snapshots", [0, 1])
def test_count_tiny_positive_duration_is_integrated(n_snapshots):
    """No absolute ULP floor may consume an entire short simulation."""
    t_end = 1e-16
    schedule = count_snapshot_times(n_snapshots, t_end)
    plan = integration_plan(
        t_end, 1.0, mode="count", snapshot_times=schedule)
    assert ("step", t_end, t_end) in plan
    assert _plan_final_time(plan) == t_end
    assert _plan_stores(plan) == ([] if n_snapshots == 0 else [t_end])


def test_count_step_stagnation_aborts_instead_of_violating_cfl():
    """A sub-ULP CFL step must not be replaced by the whole target gap."""
    t = 1.0
    dt_cfl = math.ulp(t) / 2.0
    with pytest.raises(FloatingPointError, match="stagnated"):
        _count_step(t, 2.0, dt_cfl)


@pytest.mark.parametrize("t_end,dt,cfl", [
    (86400.0, 21600.0, 600.0),        # aligned: final state stored
    (1.1 * SECONDS_PER_DAY, 21600.0, 600.0),   # misaligned: final NOT stored
    (0.02 * SECONDS_PER_DAY, 864.0, 137.0),    # quickstart-ish, odd cfl
    (_MISALIGNED_T_END, 200.0, 250.0),         # sub-us misaligned
    (5 * SECONDS_PER_DAY, 21600.0, 611.7),
])
def test_interval_plan_matches_main_bit_for_bit(t_end, dt, cfl):
    plan = integration_plan(t_end, cfl, mode="interval", dt_snapshots=dt)
    ref_stores, ref_steps, ref_final = _reference_interval_plan(t_end, dt, cfl)
    assert _plan_stores(plan) == ref_stores
    assert [d for kind, d, _ in plan if kind == "step"] == ref_steps
    assert _plan_final_time(plan) == ref_final


def test_interval_misaligned_omits_final_state():
    """Legacy stopping tolerance: duration not a multiple => final state absent."""
    t_end = 1.1 * SECONDS_PER_DAY
    plan = integration_plan(t_end, 600.0, mode="interval", dt_snapshots=21600.0)
    stores = _plan_stores(plan)
    assert all(abs(s - t_end) > 1e-6 for s in stores)  # t_end not stored
    # ...whereas count mode over the same duration ends exactly at t_end.
    count_plan = integration_plan(
        t_end, 600.0, mode="count",
        snapshot_times=count_snapshot_times(5, t_end))
    assert _plan_final_time(count_plan) == t_end


# ---------------------------------------------------------------------------
# advective_cfl_timestep: the state-independent CFL arithmetic. This is the
# only place the ceiling is computed; the runner feeds it a fresh max speed
# after every accepted step (state-adaptive advective CFL), so its validation
# and fallback contract is exercised directly here.
# ---------------------------------------------------------------------------

def test_cfl_ordinary_positive_speed():
    # 0.5 * 1000 / 20 = 25.0
    assert advective_cfl_timestep(1000.0, 20.0) == 25.0


def test_cfl_faster_speed_gives_smaller_timestep():
    slow = advective_cfl_timestep(1000.0, 10.0)
    fast = advective_cfl_timestep(1000.0, 40.0)
    assert fast < slow
    assert fast == pytest.approx(0.5 * 1000.0 / 40.0)


def test_cfl_slower_speed_gives_larger_timestep():
    base = advective_cfl_timestep(1000.0, 20.0)
    slower = advective_cfl_timestep(1000.0, 5.0)
    assert slower > base


def test_cfl_zero_speed_uses_fallback():
    assert advective_cfl_timestep(1000.0, 0.0) == 600.0
    assert advective_cfl_timestep(1000.0, 0.0, fallback=42.0) == 42.0


def test_cfl_missing_or_zero_length_scale_uses_fallback():
    assert advective_cfl_timestep(None, 20.0) == 600.0
    assert advective_cfl_timestep(0.0, 20.0) == 600.0


def test_cfl_number_defaults_to_half():
    # The frozen CFL safety factor: 0.5 * L / speed.
    assert advective_cfl_timestep(800.0, 20.0) == 0.5 * 800.0 / 20.0


@pytest.mark.parametrize("bad_speed", [
    float("nan"), float("inf"), -1.0, -0.0001])
def test_cfl_rejects_invalid_speed(bad_speed):
    with pytest.raises((ValueError, ArithmeticError)):
        advective_cfl_timestep(1000.0, bad_speed)


@pytest.mark.parametrize("bad_length", [
    float("nan"), float("inf"), -1.0])
def test_cfl_rejects_invalid_length_scale(bad_length):
    with pytest.raises((ValueError, ArithmeticError)):
        advective_cfl_timestep(bad_length, 20.0)


def test_cfl_rejects_nonfinite_result_from_overflow():
    # A finite length scale over a tiny speed that overflows to +inf must be
    # rejected, not returned as an infinite timestep.
    tiny = 5e-324  # smallest positive subnormal
    with pytest.raises((ValueError, ArithmeticError)):
        advective_cfl_timestep(1e308, tiny)


# ---------------------------------------------------------------------------
# IntegrationScheduler: the *incremental* seam. Unlike integration_plan (which
# freezes one ceiling for the whole run), the scheduler is handed the current
# dt_cfl on every next_event call, so a ceiling recomputed from the evolving
# flow speed governs the very next accepted step. These tests fail against a
# fixed-plan implementation.
# ---------------------------------------------------------------------------

def test_scheduler_count_uses_current_ceiling_for_next_step():
    """Consecutive next_event calls with different ceilings size each step."""
    t_end = 1000.0
    scheduler = IntegrationScheduler(
        t_end, mode="count", snapshot_times=[0.0, t_end])

    assert scheduler.next_event(100.0) == ("store", 0.0, 0.0)   # ceiling unused
    assert scheduler.next_event(300.0) == ("step", 300.0, 300.0)   # uses 300
    assert scheduler.next_event(400.0) == ("step", 400.0, 700.0)   # uses 400
    # The final step is clipped to land exactly on t_end (< the 1000 ceiling).
    assert scheduler.next_event(1000.0) == ("step", 300.0, 1000.0)
    assert scheduler.next_event(500.0) == ("store", 0.0, 1000.0)
    assert scheduler.next_event(500.0) is None


def test_scheduler_count_snapshot_times_stay_exact():
    """Varying the ceiling must not perturb the stored (exact) snapshot times."""
    t_end = _MISALIGNED_T_END
    schedule = count_snapshot_times(4, t_end)  # [0, t/3, 2t/3, t]
    scheduler = IntegrationScheduler(
        t_end, mode="count", snapshot_times=schedule)
    stores = []
    final_t = 0.0
    ceilings = iter([50.0, 71.0, 123.0, 40.0, 200.0, 99.0])
    while True:
        event = scheduler.next_event(next(ceilings, 250.0))
        if event is None:
            break
        kind, _dt, t = event
        if kind == "store":
            stores.append(t)
        else:
            final_t = t
    assert stores == schedule           # exact targets despite varying ceilings
    assert final_t == t_end             # diagnostics end exactly at t_end


def test_scheduler_interval_step_sizes_respond_to_ceiling():
    """Interval steps track the supplied ceiling; stores stay on boundaries."""
    t_end = 10000.0
    dt_snapshots = 4000.0               # 10000 is NOT a multiple: final omitted
    scheduler = IntegrationScheduler(
        t_end, mode="interval", dt_snapshots=dt_snapshots)

    ceilings = [500.0, 1000.0, 1500.0, 2000.0]
    stores, steps = [], []
    i = 0
    while True:
        cfl = ceilings[i] if i < len(ceilings) else 2000.0
        i += 1
        event = scheduler.next_event(cfl)
        if event is None:
            break
        kind, dt, t = event
        if kind == "store":
            stores.append(t)
        else:
            steps.append(dt)

    # Store at t=0 and each interval boundary; the misaligned final NOT stored.
    assert 0.0 in stores
    assert 4000.0 in stores
    assert 8000.0 in stores
    assert all(abs(s - t_end) > 1e-6 for s in stores)
    # The first two steps are ceiling-limited (below the snapshot countdown),
    # so they equal exactly the ceilings supplied for those calls.
    assert steps[0] == 1000.0          # second call's ceiling (first was store)
    assert steps[1] == 1500.0          # third call's ceiling


def test_scheduler_matches_integration_plan_for_constant_ceiling():
    """Driving the scheduler with a constant ceiling reproduces the plan."""
    for t_end, cfl, schedule in [
        (_MISALIGNED_T_END, 71.0, count_snapshot_times(4, _MISALIGNED_T_END)),
        (600.0, 137.0, count_snapshot_times(3, 600.0)),
    ]:
        scheduler = IntegrationScheduler(
            t_end, mode="count", snapshot_times=schedule)
        events = []
        while True:
            event = scheduler.next_event(cfl)
            if event is None:
                break
            events.append(event)
        assert events == integration_plan(
            t_end, cfl, mode="count", snapshot_times=schedule)


def test_scheduler_rejects_nonpositive_ceiling():
    scheduler = IntegrationScheduler(
        100.0, mode="count", snapshot_times=[100.0])
    with pytest.raises(ValueError, match="dt_cfl"):
        scheduler.next_event(0.0)
    with pytest.raises(ValueError, match="dt_cfl"):
        scheduler.next_event(float("inf"))


def test_scheduler_stagnation_still_aborts():
    """A sub-ULP ceiling that cannot advance time still fails explicitly."""
    scheduler = IntegrationScheduler(2.0, mode="count", snapshot_times=[2.0])
    # First step from t=1 would need to advance, but a sub-ULP ceiling at the
    # relevant magnitude stagnates. Drive to t=1 first with a real step.
    assert scheduler.next_event(1.0) == ("step", 1.0, 1.0)
    with pytest.raises(FloatingPointError, match="stagnated"):
        scheduler.next_event(math.ulp(1.0) / 2.0)


def test_integration_plan_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown integration mode"):
        integration_plan(100.0, 10.0, mode="bogus")


def test_integration_plan_interval_requires_dt():
    with pytest.raises(ValueError, match="interval mode requires dt_snapshots"):
        integration_plan(100.0, 10.0, mode="interval")


def test_snapshot_controls_parse_to_none():
    args = build_parser().parse_args(["run", "bve"])
    assert args.n_snapshots is None
    assert args.dt_snapshots is None


def test_count_schedule_n0_is_empty():
    assert count_snapshot_times(0, 86400.0) == []


def test_count_schedule_n1_is_final_state_only():
    assert count_snapshot_times(1, 86400.0) == [86400.0]


def test_count_schedule_n2_is_endpoints():
    assert count_snapshot_times(2, 86400.0) == [0.0, 86400.0]


def test_count_schedule_n5_matches_historical_default_run():
    assert count_snapshot_times(5, 86400.0) == [
        0.0, 21600.0, 43200.0, 64800.0, 86400.0]


@pytest.mark.parametrize("n", [3, 7, 24, 100])
def test_count_schedule_arbitrary_n(n):
    t_end = 2.5 * SECONDS_PER_DAY
    times = count_snapshot_times(n, t_end)
    assert len(times) == n
    assert times[0] == 0.0 and times[-1] == t_end
    spacing = t_end / (n - 1)
    for i, t in enumerate(times):
        assert t == pytest.approx(i * spacing, abs=1e-9)


def test_interval_schedule_preserves_legacy_semantics():
    assert interval_snapshot_times(864.0, 1728.0) == [0.0, 864.0, 1728.0]
    times = interval_snapshot_times(21600.0, 1.1 * SECONDS_PER_DAY)
    assert times[0] == 0.0
    assert times[-1] == 86400.0


def test_count_and_interval_mutually_exclusive_in_resolution():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BVERunConfig.resolve({"n_snapshots": 5, "dt_snapshots": 60.0})


def test_negative_count_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        BVERunConfig.resolve({"n_snapshots": -2})


def test_nonpositive_interval_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        BVERunConfig.resolve({"dt_snapshots": 0.0})


def test_n0_and_n1_modes(stub_execute_run):
    cfg0 = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "0"], stub_execute_run)
    assert cfg0.snapshot_times_seconds() == []
    assert cfg0.dt_snapshots is None
    assert cfg0.to_run_config_dict()["dt_snapshots"] is None

    cfg1 = run_aeolus_stubbed(
        ["run", "bve", "--n-snapshots", "1"], stub_execute_run)
    assert cfg1.snapshot_times_seconds() == [SECONDS_PER_DAY]
    assert cfg1.dt_snapshots is None
