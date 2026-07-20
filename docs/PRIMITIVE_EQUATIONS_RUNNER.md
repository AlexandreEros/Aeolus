# The first runnable dry primitive-equation experiment

This documents the *runner* milestone: the smallest scientifically honest
end-to-end path that constructs a valid dry hydrostatic primitive-equation
initial state, evolves it with fixed-step RK4, validates every intermediate
stage and every accepted state, persists coefficient snapshots with full
provenance, exposes the run through the CLI, and renders a basic summary.

It is an **ignition path for the primitive-equation dynamical core**, not a
climate model. The nonlinear tendency core it drives is documented in
[PRIMITIVE_EQUATIONS_DESIGN.md](PRIMITIVE_EQUATIONS_DESIGN.md) and
[PRIMITIVE_EQUATIONS_TENDENCY_HANDOFF.md](PRIMITIVE_EQUATIONS_TENDENCY_HANDOFF.md).

## What this runner deliberately excludes

No forcing (Held–Suarez, radiative), no moist physics, no hyperdiffusion or
filters, no semi-implicit stepping, no adaptive timestepping, no
arbitrary-expression initial conditions, and no long climate integrations. There is **no CFL controller** and **no total-energy
conservation claim**. The runner never adds damping, clips values, replaces
NaNs, or forces positivity to make an unstable run appear to survive — an
invalid state aborts the run loudly.

## Initial-condition presets

All presets are built spectrally on top of the model's exact-rest state, so
vorticity and divergence are exactly zero and the fields are exactly
band-limited (no grid round trip). The base state is set by
`--temperature T` (K) and `--surface-pressure p_s` (Pa).

### `isothermal_rest`

`zeta = delta = 0`; `T_k = T` at every full level; `p_s` uniform. A constant
field is the pure `(0,0)` spherical-harmonic mode with coefficient value
`c * sqrt(4*pi)`. This is the model's exact-rest state: its tendency is exactly
zero, so the runner must preserve it bit-for-bit.

### `thermal_wave`

The resting isothermal state plus a single deterministic degree-2 temperature
perturbation. The perturbation is one real coefficient placed on the
`(l, m) = (2, 2)` sectoral spherical-harmonic mode at **every** full level (a
vertically uniform profile), following the repository's real-field coefficient
convention (a single real coefficient at `(l, m>0)` synthesizes a valid
longitude-varying real field, exactly as the shallow-water `gravity_wave`
preset does). `--thermal-amplitude` (K, default `1`) is that coefficient's
value; ~1 K keeps the perturbed temperature positive everywhere. Surface
pressure stays uniform and the initial winds stay zero, so the state is
deliberately *unbalanced* — it exists to show the model launches a smooth,
finite response (a nonzero divergence field), not a balanced flow.

### `orographic_isothermal_rest`

An analytically balanced resting isothermal atmosphere over the configured
surface topography — the first exact benchmark of the PE terrain coupling.
With `T_k = T` at every level, `zeta = delta = 0`, and

```
ln(p_s) = ln(p_ref) - Phi_s / (R_d T)
```

(`p_ref` = `--surface-pressure`, the surface pressure where `Phi_s = 0`),
the state is horizontally and hydrostatically balanced because
`grad(Phi_s) + R_d T grad(ln p_s) = 0` pointwise. The relation is applied
directly to the spectral coefficients of the **exact resolved `Phi_s` the
model integrates with** (no grid round trip, no independent terrain
reconstruction), so the pressure-gradient terms of the momentum tendency
cancel analytically. Zero terrain reduces bitwise to `isothermal_rest`.

The isothermal restriction belongs to this benchmark, not to the PE
topography support: a resting atmosphere with one horizontally uniform
non-isothermal `T(sigma)` profile over terrain is generally **not**
balanced, because sigma surfaces cut across pressure surfaces wherever
`p_s` varies. A future exact non-isothermal construction must define a
pressure-coordinate reference profile `T_ref(p)`, hydrostatically derive
`Phi_ref(p)`, solve `Phi_ref(p_s) = Phi_s` for the local surface pressure,
and evaluate `T_k = T_ref(sigma_k p_s(lambda, phi))`.

## Surface topography

`--topography mountain` (with the same `--mountain-*` parameters as
`aeolus run swe`) prescribes one band-limited Gaussian mountain; the flat
default is exactly zero `Phi_s`, bit-for-bit the historical model.
Terrain is fixed model data, independent of the chosen scenario: any PE
scenario runs over it (`orographic_isothermal_rest` is merely the first
analytically exact one). How it enters the dynamics:

* the stored terrain is the surface **elevation** `h_s` (m), band-limited
  spectral coefficients (`physics/topography.py`, shared with the SWE);
* the PE core consumes the surface **geopotential**
  `Phi_s = g h_s` (m²/s²), with `g` set by `--gravity` (default 9.80616;
  a *terrain* parameter here — nothing else in the dry sigma core uses g);
* `Phi_s` anchors the Simmons–Burridge hydrostatic recursion, so every
  column's geopotential — and hence the pressure-gradient force — feels
  the terrain. The vertical coordinate stays `sigma = p/p_s`; nothing is
  re-gridded;
* every resolved terrain parameter (including `gravity`) is part of the
  persisted `run_config` and the scientific-config hash, so a terrain run
  can never collide with a flat run. Flat runs emit exactly the
  historical config schema (old run ids remain valid; manifests without a
  `topography` key are unambiguously flat).

**Dealiased terrain truncation.** PE terrain is band-limited at the
model's product-truncation cut `l = floor(2 l_max / 3)`, not at `l_max`.
The full-T pressure-gradient force `R_d T grad(ln p_s)` reaches the
tendency through the dealiased (2/3-truncated) nonlinear pathway, while
`-lap(Phi)` is an exact diagonal spectral term; terrain content above the
cut therefore *cannot* be balanced and would act as a permanent spurious
momentum forcing (measured: for full-`l_max` terrain the uncancelled
per-degree residual at rest equals `|lap * Phi_s|` exactly for every `l`
above the cut). The projection-residual gate validates the terrain
actually used and fails loudly when a mountain is too narrow for the cut
— at the default `l_max = 10` (cut 6) the default 20° mountain is
rejected; widen it (≥ ~25°) or raise `--l-max`. The model itself rejects
any supplied `Phi_s` with content above the cut. The SWE core has no such
asymmetry (its `Phi_s` enters only spectrally) and keeps full-`l_max`
terrain; this is a real, documented difference between the two solvers.

**What the orographic-rest benchmark proves.** On the Gauss–Legendre
lat-lon backend the balance closes to floating-point roundoff: the initial
tendency is ~1e-15 relative to the pressure-gradient scale, and five real
RK4 steps preserve the state to ~1e-21 in the momentum coefficients with
exactly zero mass drift and bitwise-unchanged temperature. On the geodesic
backend the residual is the backend's measured weak-form quadrature
envelope (~2.6e-3 relative; within the documented
`GEODESIC_WEAK_FORM_RTOL = 0.02`), which integrates over five steps into a
~1e-2 m/s spurious divergent flow and ~1e-7-relative T / ln p_s responses
— a property of the geodesic quadrature, not of the coupling. It is an
**equilibrium test, not a climate simulation**: it proves the terrain
plumbing, the hydrostatic anchoring, and the discrete pressure-gradient
cancellation, and proves nothing about flows over terrain, stability of
large-amplitude dynamics, or long integrations.

## Timestep policy

A **user-supplied fixed timestep** `--dt-seconds` (default `300`). The runner
drives the shared `IntegrationScheduler` with a *constant* ceiling, so every
accepted step is exactly `dt_seconds` except where the scheduler shortens it to
land exactly on a requested output time or `t_end` (no silent overshoot). A
diagnostic Courant number is recorded (derived from the model's validated
`max_characteristic_speed` helper and the grid `cfl_length_scale`) but it does
**not** control the step.

## CLI

```powershell
# Tiny default demonstration (thermal_wave, coarse geodesic, fixed 300 s step)
aeolus run pe

# Verify the exact-rest property
aeolus run pe --scenario isothermal_rest

# Gauss–Legendre lat-lon backend
aeolus run pe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15

# Explicit control of levels, step, duration, and storage
aeolus run pe --levels 12 --dt-seconds 200 --days 0.05 --n-snapshots 4

# Explicit (non-uniform) sigma interfaces
aeolus run pe --sigma-interfaces 0,0.25,0.6,1.0 --temperature 250

# Balanced resting atmosphere over a Gaussian mountain (exact equilibrium
# benchmark; needs l-max high enough for the mountain at the dealiased cut)
aeolus run pe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15 `
  --scenario orographic_isothermal_rest --topography mountain `
  --mountain-height-m 1500 --mountain-width-deg 25
```

`aeolus run pe --help` lists every option. The command prints the run
directory, backend/resolution, level count, timestep and duration, the stored
snapshot count, and the final diagnostic summary; `aeolus inspect runs` (or a
run directory) summarizes a finished capsule from its manifest.

## Stored capsule

The capsule uses the same run-directory / manifest / latest-pointer machinery
as the BVE and SWE runners (unique timestamped run id with a scientific-config
hash, atomic writes, `running → completed/failed` status). Stored artifacts:

| file | contents |
|------|----------|
| `pe_coeffs.npy` | `(n_snapshots, 3*nlev+1, l_max+1, l_max+1)` complex spectral states |
| `pe_snapshot_times.npy` | stored times in seconds (exactly the requested schedule) |
| `diagnostics/timeseries.csv` | per-step scalar diagnostics |
| `figures/` | diagnostic figures (when `diagnostics` in `--plot`) |
| `pe_summary.png` | four-panel summary (when `summary` in `--plot`) |
| `config.json`, `manifest.json` | resolved config + full provenance |

The coefficient stack's axis 1 carries the prognostic row ordering

```
[zeta_1 ... zeta_K,  delta_1 ... delta_K,  T_1 ... T_K,  ln p_s]   (K = nlev)
```

top to bottom; the trailing two axes are the `(degree, order)` spherical-
harmonic coefficient block. The manifest's `run_config` records the equation
set (`solver = "pe"`), `nlev`, the sigma interfaces, `r_dry`/`cp_dry`, the
backend and resolution, `l_max`, the planet parameters, the initial-condition
preset and its parameters, the fixed `dt_seconds` and duration, and the exact
`snapshot_times`; the scientific-config hash in the run id is sensitive to the
sigma grid, timestep, level count, and initial-condition parameters.

## Diagnostics

Per accepted step (`diagnostics/timeseries.csv`): simulation time, step index,
timestep, temperature min/max, surface-pressure min/max, max horizontal wind
speed, max `|zeta|`, max `|delta|`, the model's validated characteristic speed,
the diagnostic Courant number, a total-mass proxy (`integral of p_s dA`), and
the relative mass drift from the initial state. A resting atmosphere has an
exactly uniform `p_s`, which the state-grid quadrature integrates exactly, so
the relative mass drift of a preserved rest state is exactly zero on both
backends. No total-energy conservation is reported.

## Summary visualization

`pe_summary.png` shows, for one selectable full sigma level (the middle level
by default) of one stored state (the final state by default): relative
vorticity, horizontal divergence, temperature anomaly (relative to the
horizontal mean), and the `ln p_s` anomaly. The anomalies are formed by zeroing
the `(0,0)` monopole (which is the horizontal mean) before synthesis. It works
on both backends and makes no claim of representing a statistically
equilibrated climate.

## Verified results

* **Exact rest** — on both the geodesic and Gauss–Legendre lat-lon backends,
  `isothermal_rest` integrated over multiple fixed steps stores snapshots that
  are bitwise identical to the initial state; diagnostics show zero wind,
  vorticity, divergence, and mass drift, with temperature and surface pressure
  exactly unchanged.
* **Smooth evolution** — on both backends, `thermal_wave` integrated over a few
  conservative fixed steps stays finite and valid at every RK4 stage and every
  accepted state, launches a nonzero divergence field, and keeps temperature
  and surface pressure positive. No damping is required.
* **Orographic balance** — `orographic_isothermal_rest` over a band-limited
  Gaussian mountain is preserved through real fixed RK4 steps to roundoff on
  the Gauss–Legendre backend (exactly zero mass drift, bitwise-unchanged
  temperature) and to the measured weak-form quadrature envelope on the
  geodesic backend (see “Surface topography” above; measured values live in
  `tests/test_pe_orographic_rest.py`).

## Limitations

The fixed timestep is the user's responsibility: the runner reports a
diagnostic Courant number but never adapts the step. The default demonstration
is deliberately tiny and short — it is an ignition/smoke path, not a validated
long integration. Total-energy conservation is not diagnosed, because the PE
core has no independently tested global discrete energy diagnostic yet.
