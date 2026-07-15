"""Resolved BVE run configuration.

This module is deliberately independent of argparse, CLI aliases, and
presets: it represents *resolved run semantics only*. The CLI layer
(``planetary_sandbox.cli``) parses flags, applies presets, and maps
user-facing spellings onto these fields.

It is also import-light on purpose (stdlib only), so help/list/inspect
commands and configuration validation never touch CuPy or matplotlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SECONDS_PER_DAY = 86400.0

#: Historical psx-bve snapshot cadence (seconds). Applied only when the
#: caller supplies neither a snapshot count nor an interval; the CLI
#: parser itself defaults both controls to None.
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 21600.0

GRID_TYPES = ("geodesic", "latlon")
PRODUCT_QUADRATURES = ("fine", "coarse")

#: Tolerance factor the runner uses when matching snapshot boundaries
#: (see runner.py: ``snapshot_tol = 1e-6 * dt_snapshots``).
_SNAPSHOT_TOL_FACTOR = 1e-6


def resolve_snapshot_interval(
    duration_days: float,
    n_snapshots: Optional[int] = None,
    snapshot_interval_seconds: Optional[float] = None,
) -> float:
    """Resolve snapshot controls into a single interval in seconds.

    At most one of ``n_snapshots`` / ``snapshot_interval_seconds`` may be
    given; with neither, the historical default of 21600 s (6 h) applies.

    ``n_snapshots`` means N stored states *including both the initial and
    the final state*, evenly spaced over the run, so the interval is
    ``duration / (N - 1)`` and N must be >= 2. Because the runner already
    clips integration steps to snapshot boundaries, this guarantees both
    endpoints are stored without changing the integration loop.
    """
    if not duration_days > 0:
        raise ValueError(f"duration must be positive, got {duration_days} days")
    if n_snapshots is not None and snapshot_interval_seconds is not None:
        raise ValueError(
            "snapshot count and snapshot interval are mutually exclusive; "
            "provide at most one")
    if n_snapshots is not None:
        if n_snapshots < 2:
            raise ValueError(
                "snapshot count must be >= 2 (the initial and final states), "
                f"got {n_snapshots}")
        return duration_days * SECONDS_PER_DAY / (n_snapshots - 1)
    if snapshot_interval_seconds is not None:
        if not snapshot_interval_seconds > 0:
            raise ValueError(
                f"snapshot interval must be positive, got {snapshot_interval_seconds} s")
        return float(snapshot_interval_seconds)
    return DEFAULT_SNAPSHOT_INTERVAL_SECONDS


@dataclass(frozen=True)
class BVERunConfig:
    """Fully resolved configuration for one BVE run.

    The field names and the ``to_run_config_dict()`` key set are frozen:
    they feed ``make_run_id`` (io.py) and the on-disk ``config.json``
    schema, which must stay identical to the historical ``vars(args)``
    of psx-bve.
    """

    lmax: int = 21
    grid: str = "geodesic"
    resolution: int = 4
    nlat: int = 128
    nlon: int = 256
    day_hours: float = math.inf
    radius_earth_units: float = 1.0
    duration_days: float = 1.0
    dt_snapshots: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS
    scenario: str = "two_vortices"
    viscosity: float = 0.0
    product_quadrature: str = "fine"
    out: str = "runs"
    experiment: Optional[str] = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.grid not in GRID_TYPES:
            raise ValueError(f"grid must be one of {GRID_TYPES}, got {self.grid!r}")
        if self.product_quadrature not in PRODUCT_QUADRATURES:
            raise ValueError(
                f"product_quadrature must be one of {PRODUCT_QUADRATURES}, "
                f"got {self.product_quadrature!r}")
        if self.lmax < 1:
            raise ValueError(f"lmax must be >= 1, got {self.lmax}")
        if self.resolution < 0:
            raise ValueError(f"resolution must be >= 0, got {self.resolution}")
        if self.nlat < 1 or self.nlon < 1:
            raise ValueError(f"nlat/nlon must be positive, got {self.nlat}x{self.nlon}")
        if not self.duration_days > 0:
            raise ValueError(f"duration must be positive, got {self.duration_days} days")
        if not self.dt_snapshots > 0:
            raise ValueError(f"dt_snapshots must be positive, got {self.dt_snapshots} s")
        if self.viscosity < 0:
            raise ValueError(f"viscosity must be >= 0, got {self.viscosity}")

    def to_run_config_dict(self) -> dict:
        """The historical psx-bve config dict (exact key set, frozen)."""
        return {
            "lmax": self.lmax,
            "grid": self.grid,
            "resolution": self.resolution,
            "nlat": self.nlat,
            "nlon": self.nlon,
            "day_hours": self.day_hours,
            "radius_earth_units": self.radius_earth_units,
            "duration_days": self.duration_days,
            "dt_snapshots": self.dt_snapshots,
            "scenario": self.scenario,
            "viscosity": self.viscosity,
            "product_quadrature": self.product_quadrature,
            "out": self.out,
            "experiment": self.experiment,
            "overwrite": self.overwrite,
        }

    def snapshot_times_seconds(self) -> list[float]:
        """Snapshot times the runner will store, mirroring its loop arithmetic.

        The runner stores t=0, then every ``dt_snapshots`` seconds (steps are
        clipped to land exactly on those boundaries), stopping at the run end.
        The final state is stored only when the duration is a multiple of the
        interval — which ``n_snapshots``-derived intervals guarantee.
        """
        t_end = self.duration_days * SECONDS_PER_DAY
        tol = _SNAPSHOT_TOL_FACTOR * self.dt_snapshots
        times: list[float] = []
        k = 0
        while k * self.dt_snapshots <= t_end + tol:
            times.append(min(k * self.dt_snapshots, t_end))
            k += 1
        return times

    @property
    def includes_final_state(self) -> bool:
        times = self.snapshot_times_seconds()
        t_end = self.duration_days * SECONDS_PER_DAY
        return bool(times) and abs(times[-1] - t_end) <= _SNAPSHOT_TOL_FACTOR * self.dt_snapshots
