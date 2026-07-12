"""Characterization audit for KNOWN_RISKS.md R-4: fixed-dt convergence sweep.

This is NOT a fix and does NOT implement adaptive timestepping. It measures
observed temporal order p, invariant histories, trajectory differences, and
error floor to attribute the residual post-R-5 drift between R-4 (temporal
error) and R-3 (dealiasing/product analysis error).

Setup:
- Rotating (24h) Earth-like planet, res=4, l_max=21, viscosity=0.
- IC: Rossby-Haurwitz wave (wavenumber 4). The nondivergent BVE admits this
  as a rigid westward rotation, so every scalar invariant is an exact
  constant for the continuous evolution; measured drift is pure numerical.
- Horizon: T_END_DAYS=5 (~17% of the RH4 period).
- Three fixed dt values: dt, dt/2, dt/4, each dt_i = T / N_i so that all
  three trajectories share the same sample times {0, T/N, 2T/N, ..., T}
  (aligned per protocol -- no snapshot-clipping artifacts).

Reads:  --- (no CLI args; hard-wired for reproducibility)
Writes:
    runs/r4-convergence/dt_scale_{1p00,0p50,0p25}/diagnostics/timeseries.csv
    runs/r4-convergence/aligned.npz          (per-step arrays, aligned)
    runs/r4-convergence/summary.txt          (human-readable summary)

Run:
    python tests/audit_r4_convergence.py
"""
from __future__ import annotations

import json
import pathlib
import time as wallclock
import warnings

import numpy as np
import cupy as cp

warnings.simplefilter("ignore")

from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.barotropic_vorticity import BarotropicState, BarotropicVorticity
from planetary_sandbox.run.bve.initial_conditions import make_ic
from planetary_sandbox.run.bve.runner import rk4_step
from planetary_sandbox.run.bve.diagnostics import spectral_diagnostics

RES, L_MAX = 4, 21
DAY_HOURS = 24.0
T_END_DAYS = 5.0
DT_SCALES = [1.0, 0.5, 0.25]
IC = "rh4"


def build_planet() -> Planet:
    params = PlanetaryParameters.from_earth_like(day_hours=DAY_HOURS)
    return Planet.generate(params=params, grid_resolution=RES, l_max=L_MAX)


def base_cfl_dt(planet: Planet) -> float:
    sh, so, grid = planet.sh, planet.so, planet.grid
    zeta = sh.transform(make_ic(IC, planet))
    psi = so.inv_laplacian(zeta)
    u, v = so.velocity_from_streamfunction(psi)
    max_speed = float(cp.max(cp.sqrt(u**2 + v**2)))
    return 0.5 * grid.min_edge_length / max_speed


def sweep(planet: Planet, dt_base: float, dt_scale: float, T: float,
          N: int | None = None) -> dict:
    """Run T with fixed dt = dt_scale * dt_base, adjusting to exact N*dt = T.
    Samples EVERY step; returns full ζ_lm history + invariants.

    Pass N explicitly to lock the step count (used to force N_finer = k*N_ref
    so aligned sample times are exact).
    """
    sh = planet.sh
    R, omega = planet.params.radius, planet.params.angular_velocity

    if N is None:
        dt_nominal = dt_scale * dt_base
        N = max(1, int(round(T / dt_nominal)))
    dt = T / N

    state = BarotropicState(sh.transform(make_ic(IC, planet)))
    model = BarotropicVorticity(planet, viscosity=0.0)

    d0 = spectral_diagnostics(state.coeffs, R, omega)
    times = [0.0]
    E = [d0["energy"]]
    Za = [d0["enstrophy_abs"]]
    circ = [d0["circulation"]]
    E_l1 = [d0["energy_l"][1]]
    zeta_hist = [cp.asnumpy(state.coeffs)]

    t0 = wallclock.perf_counter()
    for i in range(N):
        state = rk4_step(model, state, 0.0, dt)
        d = spectral_diagnostics(state.coeffs, R, omega)
        times.append((i + 1) * dt)
        E.append(d["energy"])
        Za.append(d["enstrophy_abs"])
        circ.append(d["circulation"])
        E_l1.append(d["energy_l"][1])
        zeta_hist.append(cp.asnumpy(state.coeffs))
    cp.cuda.Stream.null.synchronize()
    wall = wallclock.perf_counter() - t0

    return dict(
        t=np.asarray(times),
        E=np.asarray(E),
        Zabs=np.asarray(Za),
        circulation=np.asarray(circ),
        E_l1=np.asarray(E_l1),
        zeta=np.stack(zeta_hist),
        N=N,
        dt=dt,
        wall=wall,
    )


def align(results: dict, N_ref: int) -> dict:
    """Down-sample each sweep to the coarsest sample times (multiples of dt)."""
    aligned = {}
    for scale, r in results.items():
        stride = r["N"] // N_ref
        assert stride * N_ref == r["N"], f"N={r['N']} not divisible by N_ref={N_ref}"
        aligned[scale] = {
            "t": r["t"][:: stride],
            "E": r["E"][:: stride],
            "Zabs": r["Zabs"][:: stride],
            "circulation": r["circulation"][:: stride],
            "E_l1": r["E_l1"][:: stride],
            "zeta": r["zeta"][:: stride],
        }
    # sanity: all three share the exact same sample times
    t_ref = aligned[1.0]["t"]
    for scale in results:
        assert np.allclose(aligned[scale]["t"], t_ref, rtol=0, atol=1e-6), \
            f"snapshot times mis-aligned at scale={scale}"
    return aligned


def convergence_metrics(aligned: dict) -> dict:
    """Trajectory L2 differences vs the finest run, and observed order."""
    ref = aligned[0.25]["zeta"]
    n_samples = ref.shape[0]

    diff_1 = np.array([np.linalg.norm(aligned[1.0]["zeta"][k] - ref[k])
                       for k in range(n_samples)])
    diff_h = np.array([np.linalg.norm(aligned[0.5]["zeta"][k] - ref[k])
                       for k in range(n_samples)])
    zref_norm = np.array([np.linalg.norm(ref[k]) for k in range(n_samples)])

    # Observed order p from the final-time difference ratio.
    # Under Richardson with error(dt) = C dt^p:
    #   d1 = ||dt run   - dt/4 run|| ~ C dt^p (1 - 4^-p)
    #   dh = ||dt/2 run - dt/4 run|| ~ C (dt/2)^p (1 - 2^-p)
    # so R = d1/dh = 2^p (1 - 4^-p)/(1 - 2^-p) = 2^p (1 + 2^-p) = 2^p + 1,
    # giving p = log2(R - 1). (Using log2(R) overestimates p by
    # log2(1 + 2^-p) ~ 0.09 at p = 4.)
    r1, r2 = diff_1[-1], diff_h[-1]
    if r2 > 0 and r1 > r2 and (r1 / r2) > 1.0:
        p_obs = float(np.log2(r1 / r2 - 1.0))
    else:
        p_obs = float("nan")

    return dict(diff_1=diff_1, diff_h=diff_h, zref_norm=zref_norm, p_obs=p_obs)


def summarize(planet: Planet, dt_base: float, results: dict, aligned: dict,
              conv: dict, T: float) -> str:
    lines = []
    lines.append(f"# R-4 vs R-3 attribution — dt-convergence characterization\n")
    lines.append(f"config: res={RES}, l_max={L_MAX}, day_hours={DAY_HOURS}, "
                 f"IC={IC}, viscosity=0")
    lines.append(f"horizon: T = {T/86400:.1f} days = {T:.0f} s "
                 f"({T*planet.params.angular_velocity/(2*np.pi):.3f} planet rotations)")
    lines.append(f"base CFL dt (from initial state) = {dt_base:.1f} s")
    lines.append("")

    N_ref = results[1.0]["N"]
    lines.append(f"{'dt_scale':>10} {'N steps':>8} {'dt (s)':>10} "
                 f"{'wall (s)':>10} {'ms/step':>9}")
    for s in DT_SCALES:
        r = results[s]
        lines.append(f"{s:>10.2f} {r['N']:>8d} {r['dt']:>10.1f} "
                     f"{r['wall']:>10.2f} {r['wall']*1000/r['N']:>9.1f}")
    lines.append("")

    # Invariant drift at t = 0, T/4, T/2, 3T/4, T
    ks = [0, N_ref // 4, N_ref // 2, 3 * N_ref // 4, N_ref]
    lines.append("Energy drift (E(t)-E(0))/|E(0)| at aligned sample times:")
    lines.append(f"{'t/days':>7} " + " ".join(f"{'dt*'+f'{s:.2f}':>15}" for s in DT_SCALES))
    for k in ks:
        row = f"{aligned[1.0]['t'][k]/86400:>7.2f}"
        for s in DT_SCALES:
            E0 = aligned[s]["E"][0]
            row += f" {(aligned[s]['E'][k]-E0)/abs(E0):>+15.4e}"
        lines.append(row)
    lines.append("")

    lines.append("Absolute-enstrophy drift (Zabs(t)-Zabs(0))/Zabs(0):")
    lines.append(f"{'t/days':>7} " + " ".join(f"{'dt*'+f'{s:.2f}':>15}" for s in DT_SCALES))
    for k in ks:
        row = f"{aligned[1.0]['t'][k]/86400:>7.2f}"
        for s in DT_SCALES:
            Z0 = aligned[s]["Zabs"][0]
            row += f" {(aligned[s]['Zabs'][k]-Z0)/Z0:>+15.4e}"
        lines.append(row)
    lines.append("")

    lines.append("l=1 energy at final time (rotating solid-body content):")
    for s in DT_SCALES:
        E10 = aligned[s]["E_l1"][0]
        E1e = aligned[s]["E_l1"][-1]
        lines.append(f"  dt*{s:.2f}:  E_l1(0)={E10:.4e}  E_l1(T)={E1e:.4e}  "
                     f"ratio={E1e/E10:.6f}")
    lines.append("")

    lines.append("Trajectory L2 (coefficient norm) differences at t=T:")
    lines.append(f"  ||zeta(dt)   - zeta(dt/4)||_2 = {conv['diff_1'][-1]:.4e} "
                 f"({conv['diff_1'][-1]/conv['zref_norm'][-1]:.3e} rel to ||zeta_ref||)")
    lines.append(f"  ||zeta(dt/2) - zeta(dt/4)||_2 = {conv['diff_h'][-1]:.4e} "
                 f"({conv['diff_h'][-1]/conv['zref_norm'][-1]:.3e} rel)")
    lines.append(f"  ratio R (dt / dt-half) = {conv['diff_1'][-1]/conv['diff_h'][-1]:.3f}")
    lines.append(f"  observed order p = log2(R - 1) = {conv['p_obs']:.3f}")
    lines.append("")

    # Verdict — two independent tests:
    #   Test A (temporal order): p >= 3.5 means RK4 is behaving; p < 3.5 means
    #     something OTHER than dt is corrupting the trajectory faster than RK4
    #     is producing it.
    #   Test B (conservation floor): does the invariant drift shrink meaningfully
    #     when dt is halved? If yes, further dt reduction (R-4) helps; if no,
    #     dt is not the constraint and R-3 territory is.
    # Order alone doesn't recommend an action -- p ~ 4 means "RK4 is fine",
    # which could equally mean "temporal error is small" or "temporal error is
    # the binding constraint". Test B disambiguates.
    lines.append("Verdict")
    lines.append("-" * 60)
    p = conv["p_obs"]
    lines.append(f"Test A (temporal order): observed p = {p:.2f}  "
                 f"({'RK4 ideal ~4 achieved' if p >= 3.5 else 'sub-ideal'})")

    eE = [(aligned[s]["E"][-1] - aligned[s]["E"][0]) / abs(aligned[s]["E"][0])
          for s in DT_SCALES]
    eZ = [(aligned[s]["Zabs"][-1] - aligned[s]["Zabs"][0]) / aligned[s]["Zabs"][0]
          for s in DT_SCALES]

    def shrink_ratio(a, b):
        return abs(a) / max(abs(b), 1e-300)

    rE1 = shrink_ratio(eE[0], eE[1])
    rE2 = shrink_ratio(eE[1], eE[2])
    rZ1 = shrink_ratio(eZ[0], eZ[1])
    rZ2 = shrink_ratio(eZ[1], eZ[2])
    lines.append(f"Test B (conservation floor):")
    lines.append(f"  final |E drift|  dt/dt/2/dt/4:  "
                 f"{abs(eE[0]):.3e}  {abs(eE[1]):.3e}  {abs(eE[2]):.3e}   "
                 f"shrink {rE1:.2f}x, {rE2:.2f}x")
    lines.append(f"  final |Za drift| dt/dt/2/dt/4:  "
                 f"{abs(eZ[0]):.3e}  {abs(eZ[1]):.3e}  {abs(eZ[2]):.3e}   "
                 f"shrink {rZ1:.2f}x, {rZ2:.2f}x")

    # traj diff vs invariant drift ratio: what does dt actually move?
    traj_rel = conv["diff_1"][-1] / max(conv["zref_norm"][-1], 1e-30)
    lines.append(f"Trajectory sensitivity to dt: ||dt run - dt/4 run||/||ref|| "
                 f"= {traj_rel:.2e}")
    lines.append(f"vs invariant drift ~ {abs(eE[-1]):.2e} -> ratio "
                 f"{abs(eE[-1])/max(traj_rel,1e-300):.1f}x  "
                 f"(temporal error is {'small vs' if abs(eE[-1]) > 10*traj_rel else 'comparable to'} "
                 "the conservation error)")

    lines.append("")
    # Combined recommendation
    plateau = max(rE1, rE2, rZ1, rZ2) < 1.5   # halving dt didn't change drift
    if plateau and p >= 3.5:
        lines.append("RECOMMENDATION: fix R-3 next.")
        lines.append("RK4 achieves its nominal order (temporal integration is fine), but")
        lines.append("halving dt does NOT reduce the invariant drift -- the drift is at")
        lines.append("its spatial-error floor. Further dt shrinkage or adaptive stepping")
        lines.append("(R-4) would not improve conservation; the binding constraint is the")
        lines.append("pseudospectral product's truncation-only 'dealiasing' (R-3), whose")
        lines.append("aliasing does not shrink with dt.")
    elif not plateau and p >= 3.5:
        lines.append("RECOMMENDATION: fix R-4 next.")
        lines.append("Invariant drift shrinks with dt AND RK4 order is ideal, so the")
        lines.append("temporal error is the current bottleneck. Recomputing dt from the")
        lines.append("evolving state and/or shrinking the CFL safety factor would improve")
        lines.append("conservation directly.")
    elif p < 3.5 and plateau:
        lines.append("RECOMMENDATION: fix R-3 next (with caveat).")
        lines.append("Neither reducing dt nor RK4 is behaving nominally -- likely the")
        lines.append("spatial error is dominating hard enough to distort the temporal")
        lines.append("convergence too. Fix R-3 first, then re-characterize.")
    else:
        lines.append("RECOMMENDATION: fix R-4 next.")
        lines.append("Order p < 4 AND invariant drift shrinks with dt: the time integrator")
        lines.append("is contributing meaningful error, possibly because the CFL fixed-from-")
        lines.append("initial-state dt is too large in some regime.")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"[audit_r4_convergence] rotating res={RES} l_max={L_MAX} IC={IC} "
          f"horizon={T_END_DAYS} days")
    planet = build_planet()
    dt_base = base_cfl_dt(planet)
    T = T_END_DAYS * 86400.0
    print(f"[audit_r4_convergence] base CFL dt = {dt_base:.1f} s")

    # Pick N for the coarsest sweep from CFL; force finer sweeps to exact
    # multiples so the sample times align without any rounding drift.
    coarsest = max(DT_SCALES)
    N_ref = max(1, int(round(T / (coarsest * dt_base))))
    N_by_scale = {s: N_ref * int(round(coarsest / s)) for s in DT_SCALES}

    results = {}
    for s in DT_SCALES:
        print(f"[audit_r4_convergence] sweep dt_scale={s} (N={N_by_scale[s]}) ...",
              flush=True)
        results[s] = sweep(planet, dt_base, s, T, N=N_by_scale[s])
        print(f"    N_steps={results[s]['N']}, dt={results[s]['dt']:.2f} s, "
              f"wall={results[s]['wall']:.2f} s")

    # Divisibility is guaranteed by construction; keep the assertion as a
    # sanity check.
    for s in DT_SCALES:
        assert results[s]["N"] % N_ref == 0, \
            f"N({s})={results[s]['N']} not a multiple of N_ref={N_ref}"

    aligned = align(results, N_ref)
    conv = convergence_metrics(aligned)

    summary = summarize(planet, dt_base, results, aligned, conv, T)
    print("\n" + summary)

    out = pathlib.Path("runs") / "r4-convergence"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(summary, encoding="utf-8")

    np.savez(
        out / "aligned.npz",
        t=aligned[1.0]["t"],
        E_dt=aligned[1.0]["E"], E_dth=aligned[0.5]["E"], E_dt4=aligned[0.25]["E"],
        Z_dt=aligned[1.0]["Zabs"], Z_dth=aligned[0.5]["Zabs"], Z_dt4=aligned[0.25]["Zabs"],
        El1_dt=aligned[1.0]["E_l1"], El1_dth=aligned[0.5]["E_l1"], El1_dt4=aligned[0.25]["E_l1"],
        diff_1_ref=conv["diff_1"], diff_h_ref=conv["diff_h"], zref_norm=conv["zref_norm"],
        dt_base=dt_base,
        dts=np.array([results[s]["dt"] for s in DT_SCALES]),
        N_steps=np.array([results[s]["N"] for s in DT_SCALES]),
        wall=np.array([results[s]["wall"] for s in DT_SCALES]),
    )
    print(f"\n[audit_r4_convergence] wrote {out/'summary.txt'} and {out/'aligned.npz'}")


if __name__ == "__main__":
    main()
