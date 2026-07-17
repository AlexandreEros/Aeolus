# The Rotating Shallow-Water Core

This document describes the flat-bottom, inviscid rotating shallow-water
dynamical core added in `physics/shallow_water.py` + `run/swe/`. It coexists
with the barotropic-vorticity (BVE) core: both share the spherical-harmonic
transforms, the pseudo-spectral product machinery, the model-independent
integration engine (`run/engine.py`), and the run/provenance system.

## Prognostic variables

| Variable | Meaning | Units |
|---|---|---|
| `ζ`  | relative vorticity (`k·∇×u`) | s⁻¹ |
| `δ`  | horizontal divergence (`∇·u`) | s⁻¹ |
| `φ`  | **perturbation** geopotential | m² s⁻² |

All three are held as spectral coefficients in the repository's standard
dense `(l_max+1, l_max+1)` complex layout (orthonormal complex spherical
harmonics, m ≥ 0, real fields implied), stacked into one
`(3, l_max+1, l_max+1)` array (`ShallowWaterState`).

## Perturbation geopotential

The total geopotential is

```
Φ = Φ₀ + φ,      Φ₀ = g·H  (constant resting geopotential)
```

with gravity `g` and mean fluid depth `H` fixed model parameters. The global
mean of `Φ` is represented **entirely by `Φ₀`**: the prognostic `φ` has zero
monopole (`φ₀₀ = 0`), which is enforced at initialization and preserved
exactly because every tendency's l = 0 row is hard-zeroed (the same
mechanism that pins circulation in the BVE). Consequently total layer mass
`∮ Φ dA` is conserved to the last bit. The total geopotential must remain
strictly positive; a state where `min(Φ₀ + φ) ≤ 0` fails validation
explicitly (see below).

## Governing equations (vector-invariant form)

With `η = ζ + f`, `f = 2Ω sin φ_lat` (held spectrally as the exact (1,0)
mode `2Ω√(4π/3)`), and `K = |u|²/2`:

```
∂ζ/∂t = −∇·(η u)
∂δ/∂t =  k·∇×(η u) − ∇²(K + φ)
∂φ/∂t = −Φ₀ δ − ∇·(φ u)
```

This is the standard spectral transform formulation (Williamson et al. 1992;
Hack & Jakob 1992), restricted to a flat bottom and no forcing/dissipation.
An optional scale-selective hyperdiffusion `−ν₄∇⁴` (constructor parameter,
default 0) may be applied to all three prognostics; its eigenvalue
`−ν₄ (l(l+1)/a²)²` is exactly zero at l = 0, so it can never modify the
conserved monopoles.

## Velocity reconstruction (Helmholtz decomposition)

```
u = k×∇ψ + ∇χ,     ∇²ψ = ζ,     ∇²χ = δ
```

solved diagonally in spectral space with the repository's inverse-Laplacian
convention (`∇²Y_l^m = −l(l+1)/a² · Y_l^m`):

```
ψ_lm = − a²/(l(l+1)) · ζ_lm,     χ_lm = − a²/(l(l+1)) · δ_lm
```

(l = 0 modes zeroed). On the grid, with the spectral derivative fields
`q_lam = (1/a) ∂q/∂λ` and `q_snt = (1/a) sinθ ∂q/∂θ = −(cos φ_lat/a) ∂q/∂φ_lat`:

```
u (east)  = (ψ_snt + χ_lam) / cos φ_lat
v (north) = (ψ_lam − χ_snt) / cos φ_lat
```

identical in convention to the BVE's `velocity_from_streamfunction`.

## Discretization of the nonlinear terms

The nonlinear products (`η u`, `φ u`, `K`) are evaluated **pseudo-spectrally
on the backend's product space** — exactly the machinery the BVE Jacobian
uses (geodesic: the resolution-(r+1) "overresolved product quadrature"
co-grid; Gauss lat-lon: the 3/2-rule product grid, exact for quadratic
products). The flux divergence/curl are formed via the pointwise-expanded
identities

```
∇·(q u)     = u·∇q + q δ
k·∇×(q u)   = q ζ + (∇q × u)·k
```

with `u·∇q = (u·q_lam − v·q_snt)/cos` and `(∇q × u)·k = (v·q_lam + u·q_snt)/cos`
— the same metric handling as `jacobian_pseudospectral`, avoiding a second
transform round trip. Each nonlinear product is analyzed **once** on the
product grid and truncated **once** with the 2/3 rule; the linear terms
(`−Φ₀δ`, `−∇²φ`, and the Laplacian of the analyzed `K`) are exact diagonal
spectral operations.

In the limit `δ = 0, φ = 0` the ζ tendency reduces *pointwise* to the BVE's
`−J(ψ, η)` expression, so the shallow-water core degenerates to the existing
BVE core identically (verified to 1e−12 on both backends).

## State validation

After every accepted step the state is validated; violations raise
`ShallowWaterStateError` (the run capsule is then marked `failed`):

- any NaN/Inf coefficient;
- a nonzero ζ, δ, or φ monopole (relative tolerance 1e−10);
- non-positive total geopotential `min(Φ₀ + φ) ≤ 0` (collapsed fluid depth),
  where the minimum is taken over **every sampling the model evaluates on**
  — the state grid *and* the product grid, since a high-degree φ mode can be
  positive at every state point yet negative where the nonlinear products
  are actually formed.

The runner also validates the **RK4 intermediate stages** (`y + Δt/2·k₁`,
`y + Δt/2·k₂`, `y + Δt·k₃`) before evaluating their tendencies, so a stage
that passes through an invalid region (e.g. negative depth at a too-large
timestep) fails explicitly instead of laundering itself back into an
apparently valid accepted state. Floating-point time stagnation (a CFL step
too small to advance the clock) raises `FloatingPointError` from the shared
scheduler.

## Adaptive timestep (CFL)

The timestep controller is unchanged and model-independent
(`run/engine.py::advective_cfl_timestep`, ceiling `0.5·L/s_max` with the
geometry-owned length scale `L`); only the characteristic-speed estimate is
model-specific. The shallow-water model supplies

```
s_max = max|u| + sqrt( max(Φ₀ + φ) )
```

— the advective speed plus the **total-geopotential** gravity-wave speed
(never `sqrt(φ)` of the perturbation), with the geopotential maximum taken
over the same state+product-grid envelope the validator checks (the
sum-of-maxima form is conservative). The ceiling is recomputed from every
accepted state, exactly as for the BVE.

## Initial conditions

| Scenario | Description |
|---|---|
| `rest` | `ζ = δ = φ = 0`; all tendencies are exactly zero. |
| `gravity_wave` | Small-amplitude `Y₄²` geopotential perturbation at rest; on a non-rotating planet it oscillates at `ω² = Φ₀ l(l+1)/a²`. |
| `williamson2` | Williamson et al. (1992) case 2 (α = 0): steady nonlinear zonal geostrophic flow, `u = u₀ cos φ_lat`, `u₀ = 2πa/(12 days)`, balanced `φ = C(1/3 − sin²φ_lat)`, `C = aΩu₀ + u₀²/2`. Exact steady solution for any positive mean depth; the canonical `g·h₀ = 2.94×10⁴ m²/s²` case corresponds to mean depth `(2.94×10⁴ − C/3)/g`. |

All scenarios are built spectrally (no grid round trip), so they are exactly
monopole-free and band-limited.

## Minimal CLI example

```powershell
# Default: Williamson-2 steady flow, geodesic res 4, l_max 21, 1 day,
# 5 stored states, diagnostics figures.
aeolus run swe

# Gauss lat-lon backend, gravity-wave test on a non-rotating planet:
aeolus run swe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15 `
               --scenario gravity_wave --day-hours inf --mean-depth 1000
```

Options: `--gravity`, `--mean-depth`, `--day-hours`, `--radius-earth-units`,
`--l-max`, `--resolution` / `--nlat`+`--nlon`, `--days`, `--n-snapshots` /
`--snapshot-interval-seconds`, `--scenario`, `--no-plots`, `--out`,
`--experiment`, `--overwrite`. Run capsules carry the same provenance as BVE
runs (`config.json`, `manifest.json` with status lifecycle,
`latest_run.txt`); stored artifacts are `swe_coeffs.npy`
(`(N, 3, l_max+1, l_max+1)` spectral snapshots), `swe_snapshot_times.npy`,
`diagnostics/timeseries.csv`, and `figures/`.

## Diagnostics

Per accepted step (`diagnostics/timeseries.csv`): time, dt, step, max wind
speed, max characteristic speed, CFL number, min/max total geopotential
(over the state+product-grid envelope), total mass `∮(Φ₀+φ) dA` (computed
**spectrally** as `a²[4πΦ₀ + √(4π)·Re φ₀₀]`, i.e. the conserved quantity
itself, exact by monopole pinning), total energy `∮[Φ|u|²/2 + Φ²/2] dA`
(grid quadrature), and spectral L2 norms of ζ, δ, φ.

## Verification status (tests)

- Resting atmosphere: all tendencies exactly zero.
- BVE limit: ζ tendency agrees with `BarotropicVorticity.tendency` to 1e−12
  on both backends.
- Helmholtz reconstruction: analytic ψ-only / χ-only states reproduce the
  analytic wind to 1e−10 and close the div/curl loop to round-off.
- Linear gravity wave: measured frequency matches `ω² = Φ₀ l(l+1)/a²` to
  1e−3 (RK4, small amplitude).
- Mass: monopoles conserved to the last bit through nonlinear RK4 steps.
- Williamson case 2: initial tendencies vanish (machine precision on the
  Gauss lat-lon backend; ~7e−5 of the term scale on geodesic = transform
  quadrature error); 1-day lat-lon integration steady to ~1e−15 with exactly
  zero energy/mass drift; geodesic 6 h run bounded by transform error.

## Current limitations

- Flat bottom only: no topography, forcing, radiative relaxation, or
  multilayer/primitive-equation structure.
- No semi-implicit gravity-wave treatment: explicit RK4 with the gravity-wave
  CFL, so deep layers cost proportionally smaller timesteps.
- Williamson cases 1 and 3–7 are not implemented in this milestone.
- The geodesic backend's transform is inexact (see docs/MATHEMATICAL_MODEL.md
  §4.1); steady-state residuals there are set by quadrature error, not by
  the formulation.
