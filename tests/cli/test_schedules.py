"""Snapshot schedule construction and count/interval mode contracts."""
from __future__ import annotations

import pytest

from planetary_sandbox.cli.main import build_parser
from planetary_sandbox.run.bve.config import (
    SECONDS_PER_DAY,
    BVERunConfig,
    count_snapshot_times,
    integration_plan,
    interval_snapshot_times,
)

from .conftest import run_aeolus_stubbed


# ---------------------------------------------------------------------------
# integration_plan: the physics-free step/store seam the runner replays.
# dt_cfl is a fixed ceiling, so the whole plan is deterministic on CPU.
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
    """Directly exercise the anti-early-stop contract at a tiny residual."""
    t_end = 1000.0
    # dt_cfl overshoots the target grossly; the planner must still land on it.
    plan = integration_plan(
        t_end, 999.9999, mode="count", snapshot_times=[t_end])
    assert _plan_stores(plan) == [t_end]
    assert _plan_final_time(plan) == t_end


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
