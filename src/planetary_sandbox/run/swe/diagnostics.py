"""Run-scoped diagnostics for the shallow-water core.

Same architecture as the BVE diagnostics: data production (append-only CSV
written during the run, straight from the dynamical state) is decoupled from
plotting (which reads the CSV afterwards). Definitions:

* ``max_wind_ms``        max |u| over the state grid (m/s)
* ``max_char_speed_ms``  max|u| + sqrt(max(Phi0 + phi)) with the total-
                         geopotential maximum taken over every model
                         sampling (state and product grids) — the model's
                         characteristic speed; this (not sqrt(phi)) drives
                         the adaptive CFL ceiling
* ``cfl``                max_char_speed * dt / cfl_length_scale
* ``phi_total_min/max``  extrema of the total geopotential Phi0 + phi over
                         every model sampling (state and product grids)
* ``total_mass``         integral of (Phi0 + phi) dA — proportional to layer
                         mass (rho/g factor omitted; Phi0 + phi is the
                         layer-THICKNESS geopotential, so bottom topography
                         never enters this integral); computed SPECTRALLY as
                         R^2*(4*pi*Phi0 + sqrt(4*pi)*Re(phi_00)) so the
                         reported value reflects the conserved quantity
                         itself (a grid quadrature on the geodesic backend
                         would leak higher modes into the integral and show
                         spurious drift)
* ``total_energy``       integral of [Phi*|u|^2/2 + Phi^2/2 + Phi*phi_s] dA
                         with Phi = Phi0 + phi the thickness geopotential
                         and phi_s the fixed surface geopotential (g factor
                         omitted; phi_s = 0 for a flat bottom, recovering
                         the historical definition bit-for-bit) — the
                         topographic shallow-water total energy, monitored
                         for drift
* ``h_min_m``            minimum fluid thickness (m): phi_total_min/gravity,
                         over every model sampling — the positivity margin
* ``eta_min_m/eta_max_m``  extrema (m) of the free-surface elevation anomaly
                         (phi + phi_s')/g on the state grid, where phi_s'
                         is the mean-removed surface geopotential (equal to
                         the thickness anomaly phi/g for a flat bottom)
* ``terrain_max_m``      maximum surface elevation (m) over every model
                         sampling; constant per run, 0 for a flat bottom
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
    "h_min_m",
    "eta_min_m",
    "eta_max_m",
    "terrain_max_m",
    "zeta_l2",
    "delta_l2",
    "phi_l2",
]


def _l2_norm(coeffs: cp.ndarray, radius: float) -> float:
    """sqrt(integral X^2 dA) from spectral power (R^2 * sum of mode power)."""
    return float(cp.sqrt(radius**2 * _mode_power(coeffs).sum()))


def potential_enstrophy(model: ShallowWaterModel,
                        state: ShallowWaterState) -> float:
    """Potential enstrophy  Z = integral (zeta + f)^2 / (2 h) dA.

    This is the correct shallow-water invariant for VARIABLE fluid
    thickness and bottom topography: the potential vorticity
    q = (zeta + f)/h is materially conserved, and its mass-weighted square
    integral  Z = integral h q^2/2 dA = integral (zeta+f)^2/(2h) dA  is
    conserved by the continuous equations. It is NOT the plain integral of
    zeta^2 (which is not an invariant of the SWE). Topography never enters
    directly: h = (Phi0 + phi)/gravity is the FLUID thickness, and the
    terrain influences Z only through the dynamics.

    Evaluated with the state-grid quadrature (exact on the Gauss-Legendre
    backend for band-limited integrands up to quadrature order; approximate
    on the geodesic backend, whose measured drift envelope is therefore
    looser). Deliberately a standalone helper, not a CSV column: the
    per-step diagnostics schema is frozen for historical byte-compatibility.

    Units: s^-2 m^2 / m * m^2 -> the (rho, g)-free convention matching the
    other diagnostics (an overall constant does not affect drift ratios).
    """
    zeta_g = model.sh.inv_transform(state.coeffs[0]).real
    f_g = 2.0 * model.Omega * cp.sin(
        cp.asarray(model.grid.point_latitudes, dtype=cp.float64))
    h_g = (model.phi0 + model.sh.inv_transform(state.coeffs[2]).real
           ) / model.gravity
    if not bool(cp.all(h_g > 0.0)):
        raise ValueError(
            "potential enstrophy is undefined for non-positive fluid "
            "thickness (state-grid min h = %g m)" % float(h_g.min()))
    w = cp.asarray(model.sh.weights, dtype=cp.float64) * model.R**2
    return float(cp.sum(w * (zeta_g + f_g) ** 2 / (2.0 * h_g)))


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

        # Fixed-topography constants, read once (never per step): the
        # state-grid surface geopotential (device array cached by the
        # model), its exact spectral mean, and the terrain maximum.
        self._phi_s_state = model.surface_geopotential_on_state_grid()
        self._mean_phi_s = (
            float(model.phi_s_lm[0, 0].real) / np.sqrt(4.0 * np.pi)
            if self._phi_s_state is not None else 0.0)
        self._terrain_max_m = model.surface_elevation_extrema[1]

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
        phi_min, phi_max = fields["phi_total_extrema"]
        # Same envelope/definition as the model's CFL estimate: max wind plus
        # the gravity-wave speed of the total-geopotential maximum over every
        # model sampling (state and product grids).
        max_char = max_wind + float(np.sqrt(max(phi_max, 0.0)))
        cfl = (max_char * dt / self.length_scale
               if (self.length_scale and dt > 0) else np.nan)

        kinetic = 0.5 * phi_total * (fields["u"] ** 2 + fields["v"] ** 2)
        potential = 0.5 * phi_total**2

        # Free-surface anomaly and topographic potential energy. The flat
        # branch performs no extra arithmetic, keeping historical rows
        # bit-identical; all topography constants were cached in __init__
        # (no per-step transfers).
        gravity = model.gravity
        if self._phi_s_state is not None:
            # Potential energy over terrain: Phi^2/2 + Phi*phi_s (docstring).
            potential = potential + phi_total * self._phi_s_state
            eta_anom = (phi_total - model.phi0
                        + (self._phi_s_state - self._mean_phi_s)) / gravity
        else:
            eta_anom = (phi_total - model.phi0) / gravity

        # Spectral mass: the conserved quantity itself, exact by monopole
        # pinning (see module docstring). Topography is not fluid and never
        # enters this integral.
        total_mass = self.R**2 * (
            4.0 * np.pi * self.model.phi0
            + np.sqrt(4.0 * np.pi) * float(state.coeffs[2][0, 0].real))

        row = {
            "time_s": t,
            "dt_s": dt,
            "step": step,
            "max_wind_ms": max_wind,
            "max_char_speed_ms": max_char,
            "cfl": cfl,
            "phi_total_min": phi_min,
            "phi_total_max": phi_max,
            "total_mass": total_mass,
            "total_energy": float(cp.sum(w * (kinetic + potential))),
            "h_min_m": phi_min / gravity,
            "eta_min_m": float(eta_anom.min()),
            "eta_max_m": float(eta_anom.max()),
            "terrain_max_m": self._terrain_max_m,
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

    # 4. Fluid thickness and free-surface envelope in metres (only when the
    # columns exist, so pre-topography CSVs remain plottable).
    names = data.dtype.names or ()
    if {"h_min_m", "eta_min_m", "eta_max_m", "terrain_max_m"} <= set(names):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_days, data["h_min_m"], "C0-", label="min fluid thickness")
        ax.plot(t_days, data["eta_min_m"], "C1--",
                label="min free-surface anomaly")
        ax.plot(t_days, data["eta_max_m"], "C1-",
                label="max free-surface anomaly")
        terrain_max = float(np.max(data["terrain_max_m"]))
        if terrain_max > 0.0:
            ax.axhline(terrain_max, color="k", lw=0.8, ls=":",
                       label=f"max terrain height ({terrain_max:g} m)")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("time [days]")
        ax.set_ylabel("elevation / thickness [m]")
        ax.set_title("Fluid thickness and free-surface envelope")
        ax.legend()
        _save(fig, fig_dir / "thickness_surface.png")

    return written
