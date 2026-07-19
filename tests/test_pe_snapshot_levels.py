"""Representative full-level selection for PE snapshot visualization.

Pure vertical-grid logic (no CUDA): choose an upper level near sigma=0.25 and
a lower level near sigma=0.75 from a SigmaGrid's actual full-level sigma
coordinates, working for uniform and nonuniform grids, always distinct, and
failing clearly for a single-level column.
"""
from __future__ import annotations

import pytest

from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
from planetary_sandbox.run.pe.snapshot_visualization import (
    select_snapshot_levels)


def test_uniform_grid_hits_quarter_and_three_quarter_levels():
    # nlev=10 uniform: full levels are 0.05, 0.15, 0.25, ..., 0.95, so
    # sigma=0.25 and sigma=0.75 are hit exactly by distinct levels.
    levels = select_snapshot_levels(SigmaGrid.uniform(10))
    assert levels.upper_index == 2
    assert levels.lower_index == 7
    assert levels.upper_sigma == pytest.approx(0.25)
    assert levels.lower_sigma == pytest.approx(0.75)
    assert levels.upper_index != levels.lower_index


def test_nonuniform_grid_selects_nearest_full_levels():
    # full levels: 0.1, 0.25, 0.45, 0.8 (nonuniform).
    grid = SigmaGrid((0.0, 0.2, 0.3, 0.6, 1.0))
    levels = select_snapshot_levels(grid)
    assert levels.upper_index == 1
    assert levels.upper_sigma == pytest.approx(0.25)
    assert levels.lower_index == 3
    assert levels.lower_sigma == pytest.approx(0.8)


def test_nearest_behavior_when_target_sits_between_levels():
    # full levels: 0.2, 0.45, 0.75; 0.25 is nearest to 0.2 (idx 0).
    grid = SigmaGrid((0.0, 0.4, 0.5, 1.0))
    levels = select_snapshot_levels(grid)
    assert levels.upper_index == 0
    assert levels.upper_sigma == pytest.approx(0.2)
    assert levels.lower_index == 2
    assert levels.lower_sigma == pytest.approx(0.75)


def test_upper_and_lower_indices_are_always_distinct():
    # A two-level column: the nearest to both targets must not collapse.
    levels = select_snapshot_levels(SigmaGrid.uniform(2))
    assert levels.upper_index != levels.lower_index
    assert {levels.upper_index, levels.lower_index} == {0, 1}


def test_single_level_column_fails_clearly():
    with pytest.raises(ValueError, match="at least two full levels"):
        select_snapshot_levels(SigmaGrid.uniform(1))


def test_explicit_index_override():
    grid = SigmaGrid.uniform(8)
    levels = select_snapshot_levels(grid, upper_index=0, lower_index=7)
    assert levels.upper_index == 0
    assert levels.lower_index == 7
    assert levels.upper_sigma == pytest.approx(grid.full_levels[0])
    assert levels.lower_sigma == pytest.approx(grid.full_levels[7])


def test_explicit_equal_indices_fail():
    with pytest.raises(ValueError, match="distinct"):
        select_snapshot_levels(SigmaGrid.uniform(8), upper_index=3,
                               lower_index=3)


def test_explicit_out_of_range_index_fails():
    with pytest.raises(ValueError):
        select_snapshot_levels(SigmaGrid.uniform(4), upper_index=9)


def test_explicit_sigma_targets_override_defaults():
    grid = SigmaGrid.uniform(10)  # full levels 0.05..0.95
    levels = select_snapshot_levels(
        grid, upper_sigma_target=0.05, lower_sigma_target=0.95)
    assert levels.upper_index == 0
    assert levels.lower_index == 9
