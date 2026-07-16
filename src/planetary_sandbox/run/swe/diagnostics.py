"""Run-scoped diagnostics for the shallow-water core.

Same architecture as the BVE diagnostics: data production (append-only CSV
written during the run, straight from the dynamical state) is decoupled from
plotting (which reads the CSV afterwards). Definitions:

* ``max_wind_ms``        max |u| over the state grid (m/s)
* ``max_char_speed_ms``  max(|u| + sqrt(Phi0 + phi)) — the model's
                         characteristic speed; this (not sqrt(phi)) drives
                         the adaptive CFL ceiling
* ``cfl``                max_char_speed * dt / cfl_length_scale
* ``phi_total_min/max``  extrema of the total geopotential Phi0 + phi
* ``total_mass``         integral of (Phi0 + phi) dA — proportional to layer
                         mass (rho/g factor omitted); conserved exactly
                         because the phi monopole is pinned
* ``total_energy``       integral of [Phi*|u|^2/2 + Phi^2/2] dA with
                         Phi = Phi0 + phi (g factor omitted) — the shallow-
                         water total energy, monitored for drift
* ``zeta_l2/delta_l2/phi_l2``  L2(sphere) norms sqrt(integral X^2 dA),
                         computed spectrally from the coefficient power
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import cupy as cp

from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                     ShallowWaterState)
from ..bve.diagnostics import _mode_power

SWE_CSV_COLUMNS = [
    "time_s",
    "dt_s",
    "step",
    "max_wind_ms",
    "max_char_speed_ms",
    "cfl",
    "phi_total_min",
    "phi_total_max",
    "total_mass",
    "total_energy",
    "zeta_l2",
    "delta_l2",
    "phi_l2",
]


def _l2_norm(coeffs: cp.ndarray, radius: float) -> float:
    """sqrt(integral X^2 dA) from spectral power (R^2 * sum of mode power)."""
    return float(cp.sqrt(radius**2 * _mode_power(coeffs).sum()))


class SWEDiagnosticsRecorder:
    """Append-only diagnostics writer, called after every accepted step.

    ``record()`` performs the single velocity/geopotential reconstruction of
    the step and returns the row; the runner reuses ``max_char_speed_ms`` to
    drive the adaptive CFL ceiling (no second reconstruction).
    """

    def __init__(self, model: ShallowWaterModel, out_dir: pathlib.Path):
        self.model = model
        self.R = model.R
        self.length_scale = getattr(model.grid, "cfl_length_scale", None)
        weights = cp.asarray(model.sh.weights)
        self._area_weights = weights * self.R**2

        self.dir = pathlib.Path(out_dir) / "diagnostics"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.dir / "timeseries.csv", "w", newline="",
                              encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file,
                                      fieldnames=SWE_CSV_COLUMNS)
        self._writer.writeheader()

    def record(self, t: float, state: ShallowWaterState, dt: float,
               step: int) -> dict:
        model = self.model
        fields = model.characteristic_fields(state)
        phi_total = fields["phi_total"]
        w = self._area_weights

        max_wind = float(fields["wind_speed"].max())
        max_char = float(fields["char_speed"].max())
        cfl = (max_char * dt / self.length_scale
               if (self.length_scale and dt > 0) else np.nan)

        kinetic = 0.5 * phi_total * (fields["u"] ** 2 + fields["v"] ** 2)
        potential = 0.5 * phi_total**2

        row = {
            "time_s": t,
            "dt_s": dt,
            "step": step,
            "max_wind_ms": max_wind,
            "max_char_speed_ms": max_char,
            "cfl": cfl,
            "phi_total_min": float(phi_total.min()),
            "phi_total_max": float(phi_total.max()),
            "total_mass": float(cp.sum(w * phi_total)),
            "total_energy": float(cp.sum(w * (kinetic + potential))),
            "zeta_l2": _l2_norm(state.coeffs[0], self.R),
            "delta_l2": _l2_norm(state.coeffs[1], self.R),
            "phi_l2": _l2_norm(state.coeffs[2], self.R),
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

def plot_swe_diagnostics(out_dir: pathlib.Path,
                         metadata: dict | None = None) -> list[pathlib.Path]:
    """Render the standard shallow-water figures from a run's diagnostics CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    t_days = data["time_s"] / 86400.0

    def rel_drift(x):
        x0 = x[0]
        return (x - x0) / abs(x0) if x0 != 0 else x - x0

    # 1. Conservation: total energy and total mass relative drift.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_days, rel_drift(data["total_energy"]), label="total energy")
    ax.plot(t_days, rel_drift(data["total_mass"]), label="total mass")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("relative drift")
    ax.set_title("Shallow-water invariant drift (should be ~0)")
    ax.legend()
    _save(fig, fig_dir / "invariant_drift.png")

    # 2. Speeds and CFL history.
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_days, data["max_wind_ms"], "C0-", label="max wind")
    ax1.plot(t_days, data["max_char_speed_ms"], "C2-",
             label="max |u| + sqrt(Phi)")
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel("speed [m/s]")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(t_days, data["cfl"], "C1-", label="CFL")
    ax2.set_ylabel("CFL number", color="C1")
    ax1.set_title("Characteristic speeds and CFL (state-adaptive dt)")
    _save(fig, fig_dir / "cfl_history.png")

    # 3. Total geopotential envelope and prognostic norms.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t_days, data["phi_total_min"], label="min(Phi0 + phi)")
    ax1.plot(t_days, data["phi_total_max"], label="max(Phi0 + phi)")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("geopotential [m$^2$/s$^2$]")
    ax1.set_title("Total geopotential envelope (must stay > 0)")
    ax1.legend()
    for name in ("zeta_l2", "delta_l2", "phi_l2"):
        norm = data[name]
        ax2.semilogy(t_days, np.maximum(norm, 1e-300), label=name)
    ax2.set_xlabel("time [days]")
    ax2.set_ylabel("L2 norm")
    ax2.legend()
    _save(fig, fig_dir / "state_norms.png")

    return written
