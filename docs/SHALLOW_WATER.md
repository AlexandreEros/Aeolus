# The Rotating Shallow-Water Core

This document describes the inviscid rotating shallow-water dynamical core
added in `physics/shallow_water.py` + `run/swe/`, including its optional
fixed bottom topography (`physics/topography.py`). It coexists with the
barotropic-vorticity (BVE) core: both share the spherical-harmonic
transforms, the pseudo-spectral product machinery, the model-independent
integration engine (`run/engine.py`), and the run/provenance system.

## Prognostic variables and vertical geometry

| Variable | Meaning | Units |
|---|---|---|
| `ζ`  | relative vorticity (`k·∇×u`) | s⁻¹ |
| `δ`  | horizontal divergence (`∇·u`) | s⁻¹ |
| `φ`  | **perturbation thickness** geopotential | m² s⁻² |

All three are held as spectral coefficients in the repository's standard
dense `(l_max+1, l_max+1)` complex layout (orthonormal complex spherical
harmonics, m ≥ 0, real fields implied), stacked into one
`(3, l_max+1, l_max+1)` array (`ShallowWaterState`).

The fixed vertical geometry distinguishes three quantities (each with an
elevation form in metres and a geopotential form in m²/s²):

| Quantity | Elevation (m) | Geopotential (m² s⁻²) |
|---|---|---|
| bottom / surface topography (fixed) | `h_s` | `φ_s = g·h_s` |
| fluid thickness (prognostic) | `h` | `Φ = g·h = Φ₀ + φ` |
| free surface | `η_fs = h_s + h` | `φ_s + Φ` |

## Perturbation thickness geopotential

The layer-thickness geopotential is

```
Φ = g·h = Φ₀ + φ,      Φ₀ = g·H  (constant mean-thickness geopotential)
```

with gravity `g` and global-mean fluid thickness `H` fixed model
parameters. The global mean of `Φ` is represented **entirely by `Φ₀`**: the
prognostic `φ` has zero monopole (`φ₀₀ = 0`), which is enforced at
initialization and preserved exactly because every tendency's l = 0 row is
hard-zeroed (the same mechanism that pins circulation in the BVE).
Consequently total layer mass `∮ Φ dA` is conserved to the last bit —
bottom topography is not fluid and never enters this integral. The
thickness must remain strictly positive; a state where `min(Φ₀ + φ) ≤ 0`
fails validation explicitly (see below).

Over a flat bottom the thickness geopotential and the free-surface
geopotential coincide, which is why the pre-topography code (and its
persisted artifacts) never had to distinguish them; the stored prognostic's
meaning is unchanged.

## Governing equations (vector-invariant form)

With `η = ζ + f`, `f = 2Ω sin φ_lat` (held spectrally as the exact (1,0)
mode `2Ω√(4π/3)`), `K = |u|²/2`, and fixed surface geopotential `φ_s`
(zero for a flat bottom):

```
∂ζ/∂t = −∇·(η u)
∂δ/∂t =  k·∇×(η u) − ∇²(K + φ + φ_s)
∂φ/∂t = −Φ₀ δ − ∇·(φ u)
```

This is the standard spectral transform formulation (Williamson et al. 1992;
Hack & Jakob 1992) with fixed bottom topography and no
forcing/dissipation. Topography enters **only** the divergence tendency:
the pressure force is the gradient of the free-surface geopotential
`Φ + φ_s` (curl-free, so the ζ equation is untouched), while continuity
transports the fluid thickness. Since `φ_s` is time-independent and
band-limited, `−∇²φ_s` is a constant, exact, diagonal spectral term that is
precomputed once — the flat-bottom code path is bit-for-bit unchanged.

An optional scale-selective hyperdiffusion `−ν₄∇⁴` (constructor parameter,
default 0, not exposed on the CLI) may be applied to all three prognostics;
its eigenvalue `−ν₄ (l(l+1)/a²)²` is exactly zero at l = 0, so it can never
modify the conserved monopoles. With a non-flat bottom the φ component
damps the **free-surface anomaly** `φ + φ_s'` (φ_s' = mean-removed φ_s)
rather than the thickness perturbation alone, so damping relaxes toward
the lake-at-rest state instead of eroding it; over a flat bottom the two
definitions coincide exactly.

## Bottom topography representation

`physics/topography.py` provides an immutable `Topography` holding the
spectral coefficients of the surface **elevation** `h_s` (metres) at
exactly the model truncation, resident on the GPU for the model's
lifetime. The model derives `φ_s = g·h_s` and everything the per-step
tendency needs once at construction; integration performs **no
host-device topography transfers** and no per-step synthesis.

Two deterministic analytic presets exist:

- `flat` — all coefficients exactly zero (canonical default). A model
  given a flat topography is bit-for-bit identical to one given none.
- `mountain` — one smooth isolated Gaussian mountain
  `h_s(x) = h₀ · exp(−(d/σ)²)` (`d` = great-circle distance from the
  center, `σ` = e-folding width). It is **defined analytically on the
  state grid and projected onto the truncation by the backend's own
  analysis transform** (not constructed directly in spectral space). The
  projection is validated: the quadrature-weighted relative L2 residual
  between the analytic field and its band-limited synthesis must not
  exceed 0.2, otherwise construction fails with instructions to widen the
  mountain or raise `l_max`. The projected terrain necessarily carries
  Gibbs-type ripples (including small negative overshoots around the
  mountain); the residual bound quantifies them, and the stored spectral
  field — not the analytic formula — **is** the model's terrain,
  identically on every sampling (state and product grids) via exact basis
  evaluation. Nothing about the terrain bypasses the model's dealiasing:
  φ_s enters only linearly, where no truncation is needed.

Because the terrain is deterministic given the resolved configuration and
backend, runs persist **no terrain arrays**: the topography is
reconstructed from `config.json`/`manifest.json` (whose topography keys
participate in the scientific hash). This keeps snapshots small and makes
the configuration the single source of truth.

### Resting balance (lake at rest)

The resting state over terrain is zero velocity with a spatially constant
free surface: `ζ = δ = 0`, `φ = −φ_s'`. Then `φ + φ_s` is spatially
uniform (purely l = 0), and in spectral space the entire divergence
tendency is annihilated **exactly**: the l = 0 Laplacian eigenvalue is
exactly zero, every other coefficient of `∇²(φ + φ_s)` cancels bitwise,
and every tendency's l = 0 row is hard-zeroed. All other tendency terms
vanish pointwise because `u = 0` and `δ = 0` on the grid. The discrete
lake-at-rest state is therefore preserved exactly (verified to `== 0.0`
per tendency evaluation and through RK4 steps), not merely to a loose
tolerance — this holds because `φ_s` is band-limited at the truncation, so
`φ = −φ_s'` is exactly representable. The construction keeps the
global-mean thickness exactly `H` (`--mean-depth` retains its meaning) and
puts the constant free surface at elevation `H + mean(h_s)`.

### Positivity over terrain

The positivity constraint is on the **actual fluid thickness**
`h = (Φ₀ + φ)/g` — never on the free surface or on whichever variable is
stored — and is checked over every sampling the model evaluates on (state
and product grids), at the initial condition, after every accepted step,
and at every RK4 intermediate stage. A mountain that protrudes through
the fluid layer therefore fails at initial-condition validation with an
explicit diagnosis (terrain summary, max surface elevation vs mean
thickness) rather than producing negative depth, NaNs, or silent clipping.
There is no clipping or damping anywhere in the topographic path.

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
- non-positive fluid thickness `min(Φ₀ + φ) ≤ 0` (collapsed fluid depth;
  `Φ₀ + φ` is the thickness geopotential with or without topography),
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

— the advective speed plus the gravity-wave speed of the **local fluid
thickness** (`Φ₀ + φ` is the thickness geopotential, so `sqrt(max(Φ₀+φ))`
is `sqrt(g·h)` at the deepest point of the layer — never `sqrt(φ)` of the
perturbation and never the free-surface geopotential), with the maximum
taken over the same state+product-grid envelope the validator checks (the
sum-of-maxima form is conservative). Topography needs no separate CFL
term: it only redistributes thickness, and the thickness envelope already
reflects that (e.g. the lake-at-rest state over a mountain has its wave
speed set by the deepest fluid, exactly as it should). The ceiling is
recomputed from every accepted state, exactly as for the BVE.

## Initial conditions

The `rest`, `gravity_wave`, and `williamson2` scenarios specify a velocity
field and a **free-surface** geopotential anomaly `φ_fs'`; the prognostic
thickness perturbation is `φ = φ_fs' − φ_s'`. For a flat bottom
(`φ_s' = 0`) this reproduces the historical states bit-for-bit; over
terrain each of these scenarios is well-defined relative to the
lake-at-rest state. `williamson5` is the deliberate exception: it
prescribes the **thickness** field directly and never compensates for
terrain (see its section below).

| Scenario | Description |
|---|---|
| `rest` | Zero velocity, constant free surface. Flat bottom: `ζ = δ = φ = 0`; over terrain: `φ = −φ_s'` (the exact lake-at-rest state). All tendencies are exactly zero either way. |
| `gravity_wave` | Small-amplitude `Y₄²` **free-surface** perturbation at rest; on a non-rotating flat-bottom planet it oscillates at `ω² = Φ₀ l(l+1)/a²`. |
| `williamson2` | Williamson et al. (1992) case 2 (α = 0) wind/free-surface pair: `u = u₀ cos φ_lat`, `u₀ = 2πa/(12 days)`, `φ_fs' = C(1/3 − sin²φ_lat)`, `C = aΩu₀ + u₀²/2`. Over a flat bottom: the exact steady solution for any positive mean depth (canonical `g·h₀ = 2.94×10⁴ m²/s²` ↔ mean depth `(2.94×10⁴ − C/3)/g`). Over a mountain: the same wind and free surface launched above the terrain — a smooth mountain-flow experiment (NOT steady, and NOT Williamson case 5, whose mountain is conical and whose `u₀` is 20 m/s). |
| `williamson5` | Williamson et al. (1992) case 5: the W2-shaped wind/**thickness** pair with `u₀ = 20 m/s`, `h₀ = 5960 m` over the canonical conical mountain (`hs0 = 2000 m`, `R0 = π/9`, center 30 N / −90 E). Owns its terrain; canonical constants resolved automatically. See the dedicated section below. |

All scenarios are built spectrally (no grid round trip), so they are exactly
monopole-free and band-limited, and each validates its state before
returning (protruding terrain fails here, before integration).

## Williamson test case 5 (`--scenario williamson5`)

First-class implementation of Williamson et al. (1992) case 5: zonal flow
impinging on the canonical isolated **conical** mountain. The scenario owns
its terrain (pairing it with `--topography` is rejected) and resolves the
canonical constants automatically:

| Quantity | Canonical value |
|---|---|
| radius `a` | `6.37122e6 m` (exact, perfect sphere — `PlanetaryParameters.ideal_sphere`) |
| rotation `Omega` | `7.292e-5 s^-1` (exact; `day_hours = 2*pi/Omega/3600`) |
| gravity `g` | `9.80616 m/s^2` |
| wind `u0` | `20 m/s` |
| reference depth `h0` | `5960 m` |
| mean depth `H = h0 - C/(3g)` | `5637.3529003537915 m`, `C = a*Omega*u0 + u0^2/2` |
| cone | `hs0 = 2000 m`, `R0 = pi/9`, center `(30N, -90E)` |

The initial state is the W2-shaped pair with `u0 = 20`:
`zeta = (2u0/a) sin(lat)` (pure `(1,0)`), `delta = 0`,
`phi = C(1/3 - sin^2 lat)` (pure `(2,0)`), built exactly in spectral space.
**Defining convention:** `phi` is the *thickness* perturbation and is NOT
compensated by the terrain (`phi = phi_fs' - phi_s'` is the `williamson2`
scenario's construction, not W5). The initial free surface
`Phi0 + phi + phi_s` is therefore raised over the mountain — that raised
surface is the canonical topographic forcing, and at `t = 0` the entire
tendency is the exact spectral term `-laplacian(phi_s)` in the divergence
equation (a tested invariant, with an `hs0 = 0` null experiment).

The cone uses the published **coordinate-plane** angular distance
`r = min(R0, sqrt(dlambda^2 + dlat^2))` with wrapped longitude — not
great-circle distance, and not a Gaussian. It is not band-limited: the
constructor (`Topography.williamson5_cone`) analyzes the analytic cone once
on the backend's state sampling at the full truncation and **records** the
measured projection residual, elevation extrema, and peak undershoot in its
provenance instead of demanding smoothness. Measured residuals: GL 0.0895
(`l_max=15`) → 0.0249 (`l_max=42`) → 0.0121 (`l_max=63`); geodesic
res4/`l_max=21` 0.0643. A benchmark-specific gate (0.25) rejects only
qualitatively degraded terrain (e.g. geodesic res3/`l_max=10` at 0.328);
the Gaussian preset's 0.2 gate is untouched. Because each backend analyzes
with its own quadrature, the stored cone coefficients are backend-dependent
at the ~1e-2 relative level (measured at `l_max=21`) — characterized in the
tests, not hidden.

Explicit `--mean-depth`, `--gravity`, `--day-hours`, `--radius-earth-units`
overrides are honored but the run is then labeled `NONCANONICAL
(W5-derived)` in the summary, manifest note, and the hashed
`w5_canonical` config key. Every W5-defining choice (cone geometry, `u0`,
projection policy, canonicality) is emitted additively into the config
dict and scientific hash; flat/Gaussian/W2 identities are unchanged.

Canonical 15-day benchmark (Gauss–Legendre primary path, day-0/5/10/15
snapshots):

```powershell
aeolus run swe --scenario williamson5 --backend gauss-latlon `
               --nlat 64 --nlon 128 --l-max 42 --days 15 --n-snapshots 4
```

The run is inviscid (no hidden damping; the SWE CLI exposes no
hyperdiffusion). Validation layers: exact spectral setup tests, projection
characterization, short-run conservation envelopes (measured: 6 h GL
`l_max=21` energy drift `+7.1e-6`, dt-independent ⇒ truncation-limited;
geodesic 3 h `+2.5e-5`), and the 15-day acceptance evidence in
`tests/test_williamson5.py` / the run capsule. Potential enstrophy
`Z = ∫ (zeta+f)^2/(2h) dA` is available as
`run.swe.diagnostics.potential_enstrophy` (the correct variable-thickness
invariant; deliberately not a CSV column so historical CSVs stay
byte-identical). No trusted machine-readable external reference field is
bundled: setup exactness and self-convergence are validated, an external
numerical error norm remains future work.

## Minimal CLI example

```powershell
# Default: Williamson-2 steady flow, geodesic res 4, l_max 21, 1 day,
# 5 stored states, a normalized snapshot timeline, diagnostics, and a summary.
aeolus run swe

# Gauss lat-lon backend, gravity-wave test on a non-rotating planet:
aeolus run swe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15 `
               --scenario gravity_wave --day-hours inf --mean-depth 1000

# Mountain-flow demonstration: the Williamson-2 zonal jet impinging on a
# 2000 m Gaussian mountain (default position lat 30, lon 90, width 20 deg).
# Deterministic, inviscid, positive-depth; fits comfortably on the MX110.
aeolus run swe --topography mountain --mean-depth 5960 --days 2
```

Options: `--gravity`, `--mean-depth`, `--day-hours`, `--radius-earth-units`,
`--l-max`, `--resolution` / `--nlat`+`--nlon`, `--days`, `--n-snapshots` /
`--snapshot-interval-seconds`, `--scenario`, `--topography flat|mountain`
with `--mountain-height-m`, `--mountain-lat-deg`, `--mountain-lon-deg`,
`--mountain-width-deg` (defaults 2000 m, 30°, 90°, 20°; only valid with
`--topography mountain`), repeatable `--plot TYPE`, `--no-plots`, `--out`,
`--experiment`, `--overwrite`. Flat topography is the default, keeps the
historical config schema (no topography keys are emitted), and therefore
preserves every existing flat-bottom run identity; non-flat terrain
parameters participate fully in the scientific hash and are shown by
`aeolus inspect`. Run capsules carry the same provenance as BVE
runs (`config.json`, `manifest.json` with status lifecycle,
`latest_run.txt`); stored artifacts are `swe_coeffs.npy`
(`(N, 3, l_max+1, l_max+1)` spectral snapshots), `swe_snapshot_times.npy`,
`diagnostics/timeseries.csv`, `figures/`, `snapshots/`, and
`swe_summary.png`. The snapshot product contains physical and coefficient
frames at every persisted time plus a representative `timeline.png` for each
view. The prognostic physical group shows relative vorticity, horizontal
divergence, and the layer-thickness anomaly explicitly labeled
`h' = Phi'/g`; its diagnostic group shows velocity streamlines reconstructed
from that persisted state's Helmholtz decomposition. Spectral frames show the
persisted relative-vorticity, horizontal-divergence, and perturbation-
geopotential coefficient magnitudes. Each field keeps one normalization
across the complete timeline.
For a non-flat bottom, physical frames and the summary gain a
"Topography & free surface" row with two panels: the **free-surface
anomaly** `η' = (φ + φ_s')/g` (signed, symmetric run-wide normalization —
the dynamic field) and the static band-limited **surface elevation**
`h_s` (own sequential colors, so terrain never recolors or obscures the
dynamic fields; both panels in metres). The prognostic thickness-anomaly
panel `h' = Φ'/g` is unchanged, so thickness and free surface are shown
as explicitly distinct fields. The flat case renders exactly the
historical figures.
A selected-product rendering failure marks the capsule failed and prevents
publication through `latest_run.txt`.

## Diagnostics

Per accepted step (`diagnostics/timeseries.csv`): time, dt, step, max wind
speed, max characteristic speed, CFL number, min/max thickness geopotential
(over the state+product-grid envelope), total mass `∮(Φ₀+φ) dA` (computed
**spectrally** as `a²[4πΦ₀ + √(4π)·Re φ₀₀]`, i.e. the conserved quantity
itself, exact by monopole pinning; terrain never enters), total energy
`∮[Φ|u|²/2 + Φ²/2 + Φ·φ_s] dA` (grid quadrature; the `Φ·φ_s` term is the
topographic potential energy and is identically absent for a flat bottom),
minimum fluid thickness `h_min_m`, free-surface anomaly extrema
`eta_min_m`/`eta_max_m`, maximum terrain height `terrain_max_m` (constant
per run, 0 when flat), and spectral L2 norms of ζ, δ, φ. The rendered
figures add a thickness/free-surface envelope plot (metres) alongside the
historical invariant-drift, CFL, and norm figures.

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
- Topography (tests/test_swe_topography.py): flat-bottom bit-identity
  (model with flat topography == model without, tendencies and states);
  exact lake-at-rest preservation over a mountain on both backends
  (tendency `== 0`, state and mass unchanged through RK4 steps); uniform
  bottom+free-surface offset invariance; mountain parameter/projection
  validation incl. protruding-terrain failure; exact mass conservation in
  a mountain-flow run; 4th-order timestep convergence of a mountain-flow
  case; no host-device topography transfers in the tendency loop.

## Current limitations

- Topography is SWE-only: the primitive-equation core is untouched by this
  feature. `Topography.surface_geopotential_lm(g)` deliberately matches
  the `surface_geopotential_lm` input the PE constructor already reserves
  (same dense spectral layout, m²/s²), so a later PE coupling is a data
  handoff — but no such coupling exists or is claimed yet, and the PE
  sigma-coordinate pressure-gradient error over terrain
  (docs/PRIMITIVE_EQUATIONS_DESIGN.md §8) must be re-examined first.
- Terrain is limited to the two analytic presets (flat, one Gaussian
  mountain); no raster/NetCDF/real-Earth data, no interpolation, no
  time-dependent terrain.
- The projected mountain carries Gibbs-type ripples (bounded by the 0.2
  projection-residual gate); narrow mountains at low `l_max` are rejected
  rather than smoothed.
- No semi-implicit gravity-wave treatment: explicit RK4 with the gravity-wave
  CFL, so deep layers cost proportionally smaller timesteps.
- Williamson cases 1 and 3–7 are not implemented; in particular the
  mountain-flow demonstration is NOT Williamson case 5 (different mountain
  shape, different jet speed, different mean depth unless configured).
- The geodesic backend's transform is inexact (see docs/MATHEMATICAL_MODEL.md
  §4.1); steady-state residuals there are set by quadrature error, not by
  the formulation.
