"""Run-scoped solver diagnostics for the BVE core.

Computes conservation and spectral-health diagnostics *directly from the
dynamical state* (spectral coefficients + the transform's own quadrature
weights) — never from the interpolated plotting fields (see KNOWN_RISKS.md
R-14). Data production is decoupled from visualization: `DiagnosticsRecorder`
appends numbers to ``diagnostics/timeseries.csv`` and ``diagnostics/spectra.npz``
during the run; `plot_diagnostics` reads those files afterwards.

Spectral conventions (see MATHEMATICAL_MODEL.md §2): orthonormal complex
spherical harmonics, coefficients stored for m >= 0 in a dense
(l_max+1, l_max+1) array, real fields implied. For a real field f with
coefficients a_lm the area integrals reduce to degree sums:

    integral f^2 dA   = R^2 * sum_l P_l,      P_l = Re(a_l0)^2 + 2*sum_{m>0}|a_lm|^2
    circulation       = R^2 * sqrt(4*pi) * Re a_00
    energy E = -1/2 integral psi*zeta dA
             = 1/2 * R^4 * sum_{l>=1} P_l(zeta) / (l(l+1))
    enstrophy Z = 1/2 integral zeta^2 dA;  on rotating runs the conserved
    quantity is the ABSOLUTE enstrophy Z_abs with q = zeta + f
    (f contributes only to the (1,0) coefficient: 2*Omega*sqrt(4*pi/3)).

Only Re(a_l0) enters P_l because the synthesis discards Im(a_l0); entries
with m > l are masked (they are storage padding, not data).
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import cupy as cp

CSV_COLUMNS = [
    "time_s",
    "dt_s",
    "step",
    "max_speed_ms",
    "cfl",
    "circulation",
    "energy",
    "enstrophy_rel",
    "enstrophy_abs",
    "zeta_max",
    "zeta_rms",
    "energy_l1",
    "high_l_enstrophy_frac",
    "roundtrip_residual",
]


def _mode_power(coeffs: cp.ndarray) -> cp.ndarray:
    """Per-(l,m) contribution to integral |f|^2 dOmega for the m>=0 layout."""
    n_l, n_m = coeffs.shape
    l_idx, m_idx = cp.indices((n_l, n_m))
    valid = m_idx <= l_idx
    power = cp.where(m_idx == 0, coeffs.real**2, 2.0 * cp.abs(coeffs) ** 2)
    return cp.where(valid, power, 0.0)


def spectral_diagnostics(zeta_lm: cp.ndarray, radius: float, omega: float) -> dict:
    """Invariants and degree spectra computed purely in spectral space.

    Returns a dict with scalars (circulation, energy, enstrophy_rel,
    enstrophy_abs, energy_l1, high_l_enstrophy_frac) and per-degree arrays
    (energy_l, enstrophy_l) as numpy arrays.
    """
    R = float(radius)
    l_max = zeta_lm.shape[0] - 1
    l = cp.arange(l_max + 1, dtype=cp.float64)

    P_rel = _mode_power(zeta_lm)                      # (l, m) power of zeta
    Z_l = 0.5 * R**2 * P_rel.sum(axis=1)              # enstrophy per degree

    inv_llp1 = cp.zeros(l_max + 1, dtype=cp.float64)
    inv_llp1[1:] = 1.0 / (l[1:] * (l[1:] + 1.0))
    E_l = 0.5 * R**4 * inv_llp1 * P_rel.sum(axis=1)   # energy per degree

    # Absolute vorticity q = zeta + f; f is a pure (1,0) mode.
    q_lm = zeta_lm.copy()
    q_lm[1, 0] = q_lm[1, 0] + 2.0 * omega * np.sqrt(4.0 * np.pi / 3.0)
    Z_abs = float(0.5 * R**2 * _mode_power(q_lm).sum())

    Z_total = float(Z_l.sum())
    cut = (2 * l_max) // 3
    high_frac = float(Z_l[cut + 1:].sum()) / Z_total if Z_total > 0 else 0.0

    return {
        "circulation": float(R**2 * np.sqrt(4.0 * np.pi) * zeta_lm[0, 0].real),
        "energy": float(E_l.sum()),
        "enstrophy_rel": Z_total,
        "enstrophy_abs": Z_abs,
        "energy_l1": float(E_l[1]),
        "high_l_enstrophy_frac": high_frac,
        "energy_l": cp.asnumpy(E_l),
        "enstrophy_l": cp.asnumpy(Z_l),
    }


def grid_diagnostics(zeta_lm: cp.ndarray, sh, radius: float, omega: float,
                     latitudes: cp.ndarray) -> dict:
    """Same invariants from the synthesized grid field + quadrature weights.

    Independent of `spectral_diagnostics` (different code path), so the pair
    doubles as a transform-consistency regression check.
    """
    R = float(radius)
    w_area = cp.asarray(sh.weights) * R**2
    zeta = sh.inv_transform(zeta_lm).real
    f = 2.0 * omega * cp.sin(cp.asarray(latitudes))

    l_max = zeta_lm.shape[0] - 1
    l = cp.arange(l_max + 1, dtype=cp.float64)
    psi_lm = cp.zeros_like(zeta_lm)
    lam = -l * (l + 1.0) / R**2
    psi_lm[1:, :] = zeta_lm[1:, :] / lam[1:, None]
    psi = sh.inv_transform(psi_lm).real

    return {
        "circulation": float(cp.sum(zeta * w_area)),
        "energy": float(-0.5 * cp.sum(psi * zeta * w_area)),
        "enstrophy_rel": float(0.5 * cp.sum(zeta**2 * w_area)),
        "enstrophy_abs": float(0.5 * cp.sum((zeta + f) ** 2 * w_area)),
    }


class DiagnosticsRecorder:
    """Append-only diagnostics writer, called after every accepted step.

    Writes ``diagnostics/timeseries.csv`` row-by-row (flushed, so a crashed
    run keeps its history) and degree spectra to ``diagnostics/spectra.npz``
    on `close()`.
    """

    def __init__(self, sh, so, grid, radius: float, omega: float,
                 out_dir: pathlib.Path,
                 spectra_every: int = 10,
                 roundtrip_every: int = 10):
        self.sh = sh
        self.so = so
        self.grid = grid
        self.R = float(radius)
        self.omega = float(omega)
        self.spectra_every = int(spectra_every)
        self.roundtrip_every = int(roundtrip_every)
        # Geometry-owned CFL length scale (geodesic routes min_edge_length
        # through cfl_length_scale); attribute name kept for CSV/plot continuity.
        self.min_edge = getattr(grid, "cfl_length_scale", None)

        self.dir = pathlib.Path(out_dir) / "diagnostics"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.dir / "timeseries.csv", "w", newline="",
                              encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

        self._n_records = 0
        self._spectra_times: list[float] = []
        self._spectra_E: list[np.ndarray] = []
        self._spectra_Z: list[np.ndarray] = []

    def record(self, t: float, zeta_lm: cp.ndarray, dt: float, step: int) -> dict:
        spec = spectral_diagnostics(zeta_lm, self.R, self.omega)

        # Grid-side quantities that need a synthesis anyway.
        zeta = self.sh.inv_transform(zeta_lm).real
        w = cp.asarray(self.sh.weights)
        zeta_max = float(cp.max(cp.abs(zeta)))
        zeta_rms = float(cp.sqrt(cp.sum(w * zeta**2) / cp.sum(w)))

        psi_lm = self.so.inv_laplacian(zeta_lm)
        u, v = self.so.velocity_from_streamfunction(psi_lm)
        max_speed = float(cp.max(cp.sqrt(u**2 + v**2)))
        cfl = (max_speed * dt / self.min_edge) if (self.min_edge and dt > 0) else np.nan

        # Transform-consistency residual ||T(S(a)) - a|| / ||a||. Note this also
        # flags non-representable state content (e.g. Im(a_l0), which synthesis
        # discards) — a large value means transform loss OR a corrupted state.
        if self.roundtrip_every > 0 and self._n_records % self.roundtrip_every == 0:
            back = self.sh.transform(zeta)
            denom = float(cp.linalg.norm(zeta_lm)) + 1e-300
            roundtrip = float(cp.linalg.norm(back - zeta_lm)) / denom
        else:
            roundtrip = np.nan

        row = {
            "time_s": t,
            "dt_s": dt,
            "step": step,
            "max_speed_ms": max_speed,
            "cfl": cfl,
            "circulation": spec["circulation"],
            "energy": spec["energy"],
            "enstrophy_rel": spec["enstrophy_rel"],
            "enstrophy_abs": spec["enstrophy_abs"],
            "zeta_max": zeta_max,
            "zeta_rms": zeta_rms,
            "energy_l1": spec["energy_l1"],
            "high_l_enstrophy_frac": spec["high_l_enstrophy_frac"],
            "roundtrip_residual": roundtrip,
        }
        self._writer.writerow(row)
        self._csv_file.flush()

        if self.spectra_every > 0 and self._n_records % self.spectra_every == 0:
            self._spectra_times.append(t)
            self._spectra_E.append(spec["energy_l"])
            self._spectra_Z.append(spec["enstrophy_l"])

        self._n_records += 1
        return row

    def close(self) -> None:
        if self._spectra_times:
            np.savez(
                self.dir / "spectra.npz",
                times=np.asarray(self._spectra_times),
                energy_l=np.asarray(self._spectra_E),
                enstrophy_l=np.asarray(self._spectra_Z),
            )
        if not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Post-processing (separate from data production; safe to rerun offline)
# ---------------------------------------------------------------------------

def plot_diagnostics(out_dir: pathlib.Path,
                     metadata: dict | None = None) -> list[pathlib.Path]:
    """Render standard figures from a run's diagnostics files.

    ``metadata`` is stamped into each PNG's tEXt chunks (matplotlib PNG
    backend); per-figure ``Source`` is filled with the run-relative path of
    the data file that fed the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    base_meta = dict(metadata) if metadata else {}

    def _save(fig, path: pathlib.Path, source: str | None = None) -> None:
        meta = dict(base_meta)
        if source:
            meta["Source"] = source
        fig.savefig(path, dpi=150, bbox_inches="tight", metadata=meta or None)

    data = np.genfromtxt(diag_dir / "timeseries.csv", delimiter=",", names=True)
    data = np.atleast_1d(data)
    t_days = data["time_s"] / 86400.0

    def rel_drift(x):
        x0 = x[0]
        return (x - x0) / abs(x0) if x0 != 0 else x - x0

    # 1. Invariant drift
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_days, rel_drift(data["energy"]), label="energy (rel. drift)")
    ax.plot(t_days, rel_drift(data["enstrophy_abs"]),
            label="absolute enstrophy (rel. drift)")
    # Circulation starts near zero, so relative drift is meaningless; normalize
    # the absolute error by the characteristic circulation scale sqrt(2 E0)
    # (dimensionally m^2/s, like Gamma itself).
    scale = np.sqrt(2.0 * data["energy"][0])
    ax.plot(t_days, (data["circulation"] - data["circulation"][0]) / max(scale, 1e-300),
            label="circulation error / sqrt(2 E0)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("relative drift")
    ax.set_title("Inviscid invariant drift (should be ~0)")
    ax.legend()
    p = fig_dir / "invariant_drift.png"
    _save(fig, p, source="diagnostics/timeseries.csv")
    plt.close(fig)
    written.append(p)

    # 2. CFL / max speed history
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_days, data["max_speed_ms"], "C0-", label="max speed")
    ax1.set_xlabel("time [days]")
    ax1.set_ylabel("max speed [m/s]", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(t_days, data["cfl"], "C1-", label="CFL")
    ax2.set_ylabel("CFL number", color="C1")
    ax1.set_title("Flow speed and CFL (dt is fixed: R-4)")
    p = fig_dir / "cfl_history.png"
    _save(fig, p, source="diagnostics/timeseries.csv")
    plt.close(fig)
    written.append(p)

    # 3. Spectral health: l=1 energy, high-l enstrophy fraction, roundtrip
    fig, ax = plt.subplots(figsize=(8, 4))
    e1 = data["energy_l1"]
    ax.semilogy(t_days, np.maximum(e1 / max(e1[0], 1e-300), 1e-16),
                label="l=1 energy (norm.)")
    ax.semilogy(t_days, np.maximum(data["high_l_enstrophy_frac"], 1e-16),
                label="high-l enstrophy fraction")
    rt = data["roundtrip_residual"]
    mask = np.isfinite(rt)
    if mask.any():
        ax.semilogy(t_days[mask], np.maximum(rt[mask], 1e-16), "o--",
                    ms=3, label="transform round-trip residual")
    ax.set_xlabel("time [days]")
    ax.set_title("Spectral health")
    ax.legend()
    p = fig_dir / "spectral_health.png"
    _save(fig, p, source="diagnostics/timeseries.csv")
    plt.close(fig)
    written.append(p)

    # 4. Degree spectra at selected times
    spectra_path = diag_dir / "spectra.npz"
    if spectra_path.exists():
        with np.load(spectra_path) as z:
            times, Z_l = z["times"], z["enstrophy_l"]
        picks = sorted({0, len(times) // 2, len(times) - 1})
        fig, ax = plt.subplots(figsize=(8, 5))
        l_axis = np.arange(Z_l.shape[1])
        for i in picks:
            ax.loglog(l_axis[1:], np.maximum(Z_l[i][1:], 1e-300),
                      label=f"t = {times[i]/86400:.2f} d")
        ax.set_xlabel("degree l")
        ax.set_ylabel("enstrophy Z_l")
        ax.set_title("Enstrophy degree spectra (watch for pile-up near l_max)")
        ax.legend()
        p = fig_dir / "spectra.png"
        _save(fig, p, source="diagnostics/spectra.npz")
        plt.close(fig)
        written.append(p)

    return written
