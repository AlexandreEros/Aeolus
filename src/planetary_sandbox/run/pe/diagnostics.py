"""Run-scoped diagnostics for the dry primitive-equation core.

Same architecture as the BVE/SWE diagnostics: data production (an append-only
CSV written straight from the dynamical state during the run) is decoupled
from plotting. The recorded quantities are deliberately the honest first-runner
set — no total-energy conservation claim (the PE core has no independently
tested global discrete energy diagnostic yet), and no CFL *controller* (this
runner takes a user-supplied fixed timestep). Definitions:

* ``t_min`` / ``t_max``         temperature extrema over every model sampling
                                (state and product grids), via the model's
                                own ``temperature_extrema``
* ``ps_min`` / ``ps_max``       surface-pressure extrema on the state grid (Pa)
* ``max_wind_ms``               max |V| over the state grid (m/s)
* ``max_abs_zeta``              max |zeta| over the state grid, all levels (s^-1)
* ``max_abs_delta``             max |delta| over the state grid, all levels (s^-1)
* ``max_char_speed_ms``         the model's validated characteristic-speed
                                estimate max|V| + sqrt(gamma R_d T_max)
                                (design doc Section 10)
* ``courant``                   a DIAGNOSTIC Courant number
                                max_char_speed * dt / cfl_length_scale,
                                derived from that validated helper — it does
                                NOT control the timestep (this runner uses a
                                fixed user-supplied dt)
* ``total_mass``                proportional to total atmospheric mass:
                                integral of p_s dA over the sphere by state-grid
                                quadrature (rho/g factor omitted). A resting
                                atmosphere has an exactly uniform p_s, which
                                the quadrature integrates exactly, so the
                                relative drift of a preserved state is exactly
                                zero on both backends
* ``mass_rel_drift``            (total_mass - total_mass_0) / total_mass_0,
                                relative to the first recorded state
"""
from __future__ import annotations

import csv
import pathlib

import cupy as cp

from planetary_sandbox.physics.primitive_equations import (
    PrimitiveEquationsModel, PrimitiveEquationsState)

PE_CSV_COLUMNS = [
    "time_s",
    "dt_s",
    "step",
    "t_min",
    "t_max",
    "ps_min",
    "ps_max",
    "max_wind_ms",
    "max_abs_zeta",
    "max_abs_delta",
    "max_char_speed_ms",
    "courant",
    "total_mass",
    "mass_rel_drift",
]


class PEDiagnosticsRecorder:
    """Append-only diagnostics writer, called after every accepted step.

    ``record()`` reconstructs the diagnostic fields once per call and returns
    the row. The first recorded state fixes the mass baseline against which
    every later row's relative mass drift is measured.
    """

    def __init__(self, model: PrimitiveEquationsModel, out_dir: pathlib.Path):
        self.model = model
        self.R = model.R
        self.length_scale = getattr(model.grid, "cfl_length_scale", None)
        weights = cp.asarray(model.sh.weights)
        self._area_weights = weights * self.R**2
        self._mass0: float | None = None

        self.dir = pathlib.Path(out_dir) / "diagnostics"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.dir / "timeseries.csv", "w", newline="",
                              encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=PE_CSV_COLUMNS)
        self._writer.writeheader()

    def record(self, t: float, state: PrimitiveEquationsState, dt: float,
               step: int) -> dict:
        model = self.model

        t_min, t_max = model.temperature_extrema(state)
        ps = model.surface_pressure_on_state_grid(state)
        u, v = model.wind_on_state_grid(state)
        wind_speed = cp.sqrt(u * u + v * v)

        zeta_grid = cp.stack([model.sh.inv_transform(state.zeta[k]).real
                              for k in range(model.nlev)])
        delta_grid = cp.stack([model.sh.inv_transform(state.delta[k]).real
                               for k in range(model.nlev)])

        max_char = model.max_characteristic_speed(state)
        courant = (max_char * dt / self.length_scale
                   if (self.length_scale and dt > 0) else float("nan"))

        # Total mass proxy: integral of p_s dA by state-grid quadrature.
        total_mass = float(cp.sum(self._area_weights * ps))
        if self._mass0 is None:
            self._mass0 = total_mass
        mass_rel_drift = ((total_mass - self._mass0) / self._mass0
                          if self._mass0 else 0.0)

        row = {
            "time_s": t,
            "dt_s": dt,
            "step": step,
            "t_min": t_min,
            "t_max": t_max,
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "max_wind_ms": float(wind_speed.max()),
            "max_abs_zeta": float(cp.abs(zeta_grid).max()),
            "max_abs_delta": float(cp.abs(delta_grid).max()),
            "max_char_speed_ms": max_char,
            "courant": courant,
            "total_mass": total_mass,
            "mass_rel_drift": mass_rel_drift,
        }
        self._writer.writerow(row)
        self._csv_file.flush()
        return row

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Post-processing (separate from data production; safe to rerun offline)
# ---------------------------------------------------------------------------

def plot_pe_diagnostics(out_dir: pathlib.Path,
                        metadata: dict | None = None) -> list[pathlib.Path]:
    """Render the standard primitive-equation figures from a run's CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = pathlib.Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    base_meta = dict(metadata) if metadata else {}

    def _save(fig, path: pathlib.Path) -> None:
        meta = dict(base_meta)
        meta["Source"] = "diagnostics/timeseries.csv"
        fig.savefig(path, dpi=150, bbox_inches="tight", metadata=meta)
        plt.close(fig)
        written.append(path)

    data = np.genfromtxt(diag_dir / "timeseries.csv", delimiter=",", names=True)
    data = np.atleast_1d(data)
    t_hours = data["time_s"] / 3600.0

    # 1. Mass drift and the diagnostic Courant number.
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_hours, data["mass_rel_drift"], "C0-", label="relative mass drift")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_xlabel("time [hours]")
    ax1.set_ylabel("relative mass drift", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(t_hours, data["courant"], "C1-", label="diagnostic Courant")
    ax2.set_ylabel("Courant (diagnostic)", color="C1")
    ax1.set_title("Mass drift and diagnostic Courant (fixed timestep)")
    _save(fig, fig_dir / "mass_and_courant.png")

    # 2. Temperature / surface-pressure envelope and flow amplitude.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t_hours, data["t_min"], label="min T")
    ax1.plot(t_hours, data["t_max"], label="max T")
    ax1.set_ylabel("temperature [K]")
    ax1.set_title("Temperature envelope (must stay > 0)")
    ax1.legend()
    ax2.plot(t_hours, data["max_wind_ms"], "C2-", label="max |V|")
    ax2.plot(t_hours, data["max_char_speed_ms"], "C3-",
             label="max |V| + sqrt(gamma R T)")
    ax2.set_xlabel("time [hours]")
    ax2.set_ylabel("speed [m/s]")
    ax2.legend()
    _save(fig, fig_dir / "thermo_and_speeds.png")

    return written
