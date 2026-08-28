# Semantic audit — MRI–JMA Williamson Test Case 5 vs Aeolus

**Date:** 2026-07-29 · **Branch:** `feat/w5-mri-reference-clean` @ `580c566a`
**Question:** Does Aeolus's `williamson5` scenario describe the same physical
initial-value problem as the MRI–JMA (Yoshimura) Williamson Test Case 5
reference archive?

**Verdict: NO — the initial conditions describe two different physical
problems.** Winds, planet constants, and cone geometry are identical, but the
mass field differs: MRI's initial *free surface* is the Williamson case-2
field, while Aeolus's initial *layer thickness* is the Williamson case-2
field. Near the mountain the two initial states differ by up to 2000 m in
both free-surface height and fluid depth. Per the task contract, the
scientific validation stops at the day-zero check; Notebook B encodes the
refusal. Details and required future correction below.

---

## 1. What MRI's archived `h` represents

**Source dataset:** `Williamson5/N959_1920x960/sh` from
`https://climate.mri-jma.go.jp/pub/archives/Yoshimura_DFS_SW_Testcase/`
(SH reference model, N = 958 triangular truncation, 1920×1080 — actually
1920×960 Gaussian grid, Δt = 600 s, 16 daily outputs, days 0–15).
`data.nc` sha256 `09470d41e0f4…`, `README.txt` sha256 `cb925d76…`.

**Archive README** (Yoshimura, 2022-01-15) says only:

> `data.nc, data.ncctl: Simulated lon-lat data. Variables: h (height),
> u (zonal wind), v (meridional wind), vor (vorticity), div (divergence).`

"height" is ambiguous, so the identity was established from the publication
and from the data itself:

1. **Publication** — Yoshimura (2022), *Improved double Fourier series on a
   sphere and its application to a semi-implicit semi-Lagrangian shallow
   water model*, GMD 15, 2561–2597, doi:10.5194/gmd-15-2561-2022. The
   shallow-water system is written (Eqs. 96–97) with the momentum equation
   containing **−g∇h** and the continuity equation advancing
   **d(h − h_s)/dt = −(h − h_s)∇·v**. Therefore `h` is the **free-surface
   height** and `h − h_s` is the fluid-layer depth; `h_s` is the surface
   (mountain) height. Case-5 figures ("Predicted height") plot this same `h`.

2. **Measured, day 0** (this audit, from `data.nc` itself): the archived
   `h(t=0)` equals the analytic Williamson case-2 free surface
   `η(θ) = h0 − (C/g)·sin²θ`, with `C = aΩu0 + u0²/2`, at **every** one of
   the 1920×960 grid points to `max|Δ| = 2.4e-4 m` (float32 rounding).
   There is **no mountain signature** in the archived day-0 `h` — which is
   exactly the canonical case-5 initial free surface, and rules out `h`
   being the fluid depth (the depth would show a 2000 m dip at the cone).

3. **Measured, mass diagnostic**: the run's `STDOUT` reports
   `Global mean of mass = 5619.92593916377`. The area-weighted mean of
   `η − h_s` with the **coordinate-plane** cone (see §2) is **5619.9255 m**
   (match to 4×10⁻⁴ m). The mean of `η` alone is 5637.35 m and the mean of
   `η − h_s(great-circle)` is 5617.17 m — both excluded. This confirms the
   model's internal mass variable is `h − h_s` with the Williamson
   coordinate-plane cone, i.e. `h` is free surface.

**Conclusion:**

```
MRI archived h        = free_surface_height                 [m]
MRI fluid-layer depth = h − h_s   (not archived; derived)   [m]
MRI topography h_s    = Williamson case-5 cone (not archived; analytic)
```

## 2. MRI / Williamson definitions

Williamson, Drake, Hack, Jakob & Swarztrauber (1992), *A standard test set
for numerical approximations to the shallow water equations in spherical
geometry*, J. Comput. Phys. 102, 211–224 — test case 5: the wind and height
fields are those of case 2 (solid-body zonal flow with the balanced height
field) with `u0 = 20 m/s`, `h0 = 5960 m`, and the conical mountain

```
h_s = hs0 · (1 − r/R),   hs0 = 2000 m,  R = π/9
r   = min(R, sqrt((λ − λc)² + (θ − θc)²)),  (λc, θc) = (3π/2, π/6)
```

with `r` measured in the **coordinate plane** (λ, θ), longitude difference
wrapped — not great-circle distance. (Aeolus's own implementation documents
this: `src/planetary_sandbox/physics/topography.py:113-135`; and the MRI
STDOUT mass diagnostic above independently confirms MRI used exactly this
cone.) Constants (shared by Williamson, MRI, and Aeolus, verified against
`src/planetary_sandbox/run/swe/config.py:82-95`):

```
a = 6.37122e6 m,  Ω = 7.292e-5 s⁻¹,  g = 9.80616 m/s²,
u0 = 20 m/s,  h0 = 5960 m,  C = aΩu0 + u0²/2  (C/g ≈ 968.04 m)
```

In case 5 the "height field as in case 2" is the **free surface**:
`η(θ) = h0 − (C/g) sin²θ`; the initial depth is `η − h_s` (a mountain-shaped
depression in the fluid layer); the initial winds are `u = u0·cosθ, v = 0`.
The MRI archive matches this exactly at day 0 (§1, items 2–3).

## 3. What Aeolus stores in `phi`, `Phi0`, `phi_s`

From `src/planetary_sandbox/physics/shallow_water.py:9-20` (module
docstring, verified against the tendency code at lines 369-401 and the
validators at lines 430-441, 494-540):

```
phi   PERTURBATION thickness geopotential (m² s⁻²)  — prognostic
Phi   = g·h = Phi0 + phi     "layer-THICKNESS geopotential"
Phi0  = g·H                  constant mean-thickness geopotential
phi_s = g·h_s                fixed surface geopotential (topography)
free-surface geopotential = Phi0 + phi + phi_s     (line 20)
```

`h` in Aeolus's docstring is the **fluid-layer thickness** (depth), not the
free surface. `phi_s` is held spectrally, band-limited at the model
truncation (`physics/topography.py`, `Topography.williamson5_cone`,
lines 364-419: analytic cone analyzed on the state grid at full truncation;
measured projection residual ≈ 2.5 % rel. L2 at l_max = 42).

## 4. Physical fields from an Aeolus state

Verified from source (not assumed):

```
layer_depth          = (Phi0 + phi) / g
    shallow_water.py:11-12 ("Phi = g*h = Phi0 + phi"), 430-432
    (thickness-positivity check on Phi0+phi), and
    run/swe/diagnostics.py:106-107 ("h = (Phi0 + phi)/gravity is the
    FLUID thickness")

free_surface_height  = (Phi0 + phi + phi_s) / g
    shallow_water.py:20 ("free-surface geopotential = Phi0 + phi + phi_s");
    anomaly form used by run/swe/visualization.py:138-139

topography_height    = phi_s / g
    shallow_water.py:18 ("surface geopotential phi_s = g*h_s");
    grid values via model.surface_geopotential_on_state_grid()
    (shallow_water.py:327-334)
```

These match the identities stated in the task prompt. Caveat: Aeolus's
`phi_s` is the **band-limited** cone, so Aeolus's `topography_height`
differs pointwise from the analytic cone by the truncation residual
(ringing near the cusp; rel. L2 ≈ 2.5 % at l42, ≈ 1.2 % at l63 — measured
values recorded in the run manifests). This is a representation difference,
distinct from the initial-condition mismatch below.

## 5. Initial-condition comparison — MISMATCH

Aeolus `_williamson5` (`src/planetary_sandbox/run/swe/initial_conditions.py:109-145`)
builds, in spectral space:

```
zeta = (2u0/a)·sinθ,  delta = 0,
phi  = C·(1/3 − sin²θ)          # pure (2,0) mode, NO terrain term
H    = h0 − C/(3g) = 5637.353 m # config.py:91, run manifests
```

i.e. the **layer thickness** is `h0 − (C/g)sin²θ` — the case-2 field — and
the docstring states explicitly (lines 122-131): *"the thickness field is
NOT compensated by the surface-geopotential anomaly … the initial FREE
SURFACE Phi0 + phi + phi_s is raised over the cone; that raised surface is
exactly the canonical topographic forcing of test case 5."*

That claim is **incorrect** with respect to Williamson (1992) and to the
MRI reference:

| quantity at t = 0 | MRI / Williamson (1992) | Aeolus `williamson5` |
|---|---|---|
| free-surface height | `η(θ)` — smooth zonal, no cone signature (measured, §1.2) | `η(θ) + h_s` — bulges up to ≈ +2000 m over the cone |
| layer depth | `η(θ) − h_s` — cone-shaped depression | `η(θ)` — smooth zonal |
| global-mean depth | 5619.926 m (measured, §1.3) | 5637.353 m (`W5_MEAN_DEPTH_M`) |
| winds | `u = u0·cosθ, v = 0` (measured ≤ 1e-6 m/s) | identical (pure (1,0) ζ mode) |
| planet, g, cone geometry | §2 constants | identical (`config.py:82-95`) |

The two models therefore integrate **different initial-value problems**.
The difference is not a relabeling: no assignment of Aeolus's fields to
MRI's fields makes the states equal (free surfaces differ by `h_s`, depths
differ by `h_s`, means differ by 17.4 m). Day-zero comparison in Notebook B
fails the physical contract, and the 15-day scientific comparison is
refused, as required.

### Required future correction

> **Status (2026-07-29): IMPLEMENTED** on `fix/williamson5-initial-condition`
> (uncommitted at audit-update time): `_williamson5` now subtracts `phi_s'`,
> `W5_MEAN_DEPTH_M` absorbs the cone's exact closed-form spherical mean
> (17.42696 m, giving H = 5619.92594 m), the pinning regression test was
> replaced by its inverse, and the day-zero contract passes on both grids
> (free-surface max |Δ| = 0.011 m T42 / 0.0065 m T63). The items below are
> retained as the original specification.

To make `williamson5` integrate the canonical IVP:

1. Initial thickness perturbation must subtract the terrain anomaly:
   `phi = C·(1/3 − sin²θ) − phi_s'` (the same construction `_williamson2`
   already uses at `initial_conditions.py:102-104`), **and**
2. the resolved mean depth must absorb the cone's spherical mean:
   `H = h0 − C/(3g) − h̄_s` with
   `h̄_s = phi_s[0,0]/(g·√(4π)) ≈ 17.43 m` (because the model pins the
   prognostic monopole to zero and carries the mean in `Phi0`), giving
   `H ≈ 5619.93 m` — which then matches MRI's measured mean mass.
3. The regression test that pins the current convention (see the
   `_williamson5` docstring, lines 129-131) and the W5 acceptance
   envelopes/capsules must be re-derived; `w5_canonical` provenance labels
   and docs must be corrected.

### Unresolved ambiguity

- Whether MRI applies its `h_s` on-grid analytically or spectrally
  truncated inside the model is not stated in the archive; the mean-mass
  identity (4×10⁻⁴ m) is consistent with the analytic cone. This does not
  affect the reference package (the archived `h` is used as-is; the
  package's `topography_height` is the analytic Williamson cone, and the
  equation `layer_depth = free_surface_height − topography_height` is the
  Williamson definition itself).
- Aeolus's band-limited terrain means that, even after the IC fix, Aeolus
  depth vs MRI depth will contain a static, cone-local representation
  difference of the order of the projection residual. Notebook B reports
  this separately from trajectory error.

## Appendix — symbol mapping from the original equations

Transcribed first-hand from **ORNL/TM-11895** (Williamson, Drake, Hack,
Jakob, Swarztrauber, *A Standard Test Set for Numerical Approximations to
the Shallow Water Equations in Spherical Geometry*, Oak Ridge National
Laboratory, published August 1991; OSTI 5232139,
`https://www.osti.gov/servlets/purl/5232139`) — the companion
technical-report version of the JCP paper (J. Comput. Phys. 102, 211–224,
1992), with identical section structure, equations, and equation numbering.
Page/equation numbers below are the TM's. One substantive difference is
flagged at item 4.

### 1. Prognostic mass variable — the fluid depth `h*`

p. 3, Eq. (2) (flux form) and p. 4, Eq. (9) (advective form):

```
∂h*/∂t + ∇·(h* v) = 0                                   (2)
dh*/dt + h* ∇·v  = 0                                    (9)
```

### 2. Momentum pressure-gradient term with topography — `−g∇h`, h the free surface

p. 3, Eq. (1) (flux form) and p. 4, Eq. (8) (advective form):

```
∂(h*v)/∂t + ∇·(v h*v) = −f k̂ × h*v − g h* ∇h           (1)
dv/dt = −f k̂ × v − g ∇h                                 (8)
```

The gradient acts on `h = h* + h_s`, so topography enters the momentum
equation only through `∇h = ∇h* + ∇h_s`.

### 3. Continuity equation

Eq. (2)/(9) above: mass continuity advances the **depth** `h*`; the
free surface `h` never appears in the continuity equation.

### 4. Test Case 5 initial-height equation

§3.5, p. 18: *"It consists of zonal flow as in case 2 impinging on a
mountain. The wind and height field are as in case 2, with α = 0, but the
mean height is changed to h₀ = 5400 m. The surface or mountain height is
given by"*

```
h_s = h_s0 (1 − r/R)                                     (134)
"where h_s0 = 2000 m, R = π/9 and r² = min[R², (λ−λc)² + (θ−θc)²].
 The center is taken as λc = −π/2 and θc = π/6."
```

The case-2 height field referenced is §3.2, p. 13, Eq. (95):

```
g h = g h₀ − (a Ω u₀ + u₀²/2)(−cos λ cos θ sin α + sin θ cos α)²   (95)
```

introduced by the sentence *"The analytic h field is given by"*; with
α = 0 this is `g h = g h₀ − (a Ω u₀ + u₀²/2) sin²θ`, and the winds
(Eqs. 90–91, p. 13) reduce to `u = u₀ cos θ, v = 0`.

> **Version note:** the TM prints `h₀ = 5400 m` for case 5; the published
> JCP (1992) version reads `h₀ = 5960 m`, the value used by every reference
> implementation. The MRI archive confirms 5960 m is in force there
> (`NORM_LMAX_INIT = 5959.997` in STDOUT; archived day-0 max
> h = 5959.9976 m). Aeolus also uses 5960 m (`W5_H0_M`,
> `run/swe/config.py:89`), so this discrepancy does not affect any
> conclusion below.

### 5. Textual definitions of `h` and `h_s`

p. 3, immediately after Eq. (2):

> *"where h\* is the depth of the fluid and h is the height of the free
> surface above a reference sphere (sea level). If h_s denotes the height
> of the underlying mountains, h = h\* + h_s."*

p. 18 (case 5): *"The surface or mountain height is given by …"* (Eq. 134).

### 6. Symbol-mapping table

| Physical quantity | Williamson (1992) symbol | Yoshimura (2022) symbol | MRI archive field | Aeolus representation |
|---|---|---|---|---|
| fluid depth | `h*` (Eqs. 1–2, 8–9, p. 3–4) | `h − h_s` (continuity, Eq. 97) | `h − h_s` — **derived**, not archived | `(Phi0 + phi)/g` |
| free-surface elevation | `h = h* + h_s` (p. 3) | `h` (momentum `−g∇h`, Eq. 96) | archived `h` ("height (m)") | `(Phi0 + phi + phi_s)/g` |
| bottom elevation | `h_s` (Eq. 134) | `h_s` | not archived — analytic cone (Eq. 134) | `phi_s/g` (band-limited cone) |
| depth geopotential | `g h*` | `g(h − h_s)` | `g·(h − h_s)` (derived) | `Phi = Phi0 + phi` (prognostic `phi` + pinned mean) |
| free-surface geopotential | `g h` (Eq. 95 defines it) | `g h` | `g·h` | `Phi0 + phi + phi_s` |

### A. Does Williamson's case-5 initial analytic `h` denote fluid depth or free-surface elevation?

**Free-surface elevation.** The case-5 text prescribes "the wind and height
field … as in case 2"; the case-2 height field is the analytic `h` of
Eq. (95); and §2 (p. 3) defines `h` unambiguously as *"the height of the
free surface above a reference sphere"*, with the depth written `h*` and
related by `h = h* + h_s`. Nothing in §3.5 redefines `h`. The initial
**depth** is therefore `h* = h − h_s`: the case-2 surface with a
cone-shaped bite taken out of the fluid layer.

### B. After translating into Yoshimura's formulation, should Yoshimura's initial archived `h` contain +h_s, omit h_s, or subtract h_s?

**Omit h_s** — i.e. the archived `h` is exactly the case-2 field of
Eq. (95), with no topographic term of either sign. The translation is
symbol-for-symbol: Yoshimura's momentum equation (Eq. 96) carries `−g∇h`
like Williamson's Eq. (8), and his continuity (Eq. 97) advances `h − h_s`
like Williamson's Eq. (9) advances `h*`; hence Yoshimura's `h` **is**
Williamson's free-surface `h`, and Yoshimura's `h − h_s` **is**
Williamson's `h*`. Since Williamson's case-5 initial `h` is the unmodified
case-2 field, Yoshimura's initial (and archived) `h` must be that same
field — the `−h_s` belongs to the *depth* `h − h_s`, and a `+h_s` would
belong to no canonical field at all. (The archive is consistent with this
translation: measured day-0 `h` equals Eq. (95) with α = 0 to 2.4e-4 m —
corroboration, not the basis of the conclusion.)

### C. Is the mismatch evidence that Aeolus is noncanonical, that the MRI run is noncanonical, or only that they solve different IVPs?

**Aeolus is the noncanonical one** (and consequently the two codes also
solve different IVPs). Judged purely against the printed definitions:

- Williamson (1992) prescribes: initial free surface `h` = Eq. (95) with
  α = 0; initial depth `h* = h − h_s` (§2 definition + §3.5).
- The MRI initial state satisfies exactly this prescription under the
  symbol translation of (B) — its free surface is Eq. (95) and its mass
  variable is `h − h_s` with the Eq. (134) cone.
- Aeolus (`run/swe/initial_conditions.py:_williamson5`) sets its **depth**
  `(Phi0 + phi)/g` equal to the Eq. (95) field and lets the free surface be
  Eq. (95) + `h_s`. Under the same translation this assigns Williamson's
  `h`-field formula to Williamson's `h*` — a different physical initial
  state (free surface raised over the cone; no bite in the fluid layer;
  global-mean depth 5637.35 m instead of `mean(h) − mean(h_s)` =
  5619.93 m).

So the asymmetry is decidable from the primary text alone: MRI conforms to
the published test definition; Aeolus's `williamson5` docstring claim of
canonicity is contradicted by Williamson §2 + §3.5. The correction
(separate branch) is specified in §5 above.

## Evidence inventory

- Local cached source (hashes verified against `source_inventory.json`):
  `mri-w5-reference-preparation-v1-fixed-run/source-cache/Williamson5_N959_1920x960_sh/`
  (`data.nc`, `README.txt`, `STDOUT`, `data.ncctl`, `weight_lat.nc`).
- Day-0 measurements: this audit's script (re-run inside Notebook A as
  permanent assertions): `h(0) ≡ η` to 2.4e-4 m; `u(0) ≡ u0cosθ` to 1e-6;
  `|v(0)| ≤ 3.4e-16`; weighted mean `h(0)` = 5637.3524 = mean η to 1e-4 m;
  STDOUT mass 5619.9259 = mean(η − h_s_coord-cone) to 4e-4 m.
- Aeolus sources cited: `physics/shallow_water.py:9-20,327-334,430-441`;
  `physics/topography.py:113-135,364-419`;
  `run/swe/initial_conditions.py:90-145`; `run/swe/config.py:82-95,320-337`;
  `run/swe/diagnostics.py:81-113`; `cli/swe.py:37-55,134-198`;
  `numerics/latlon_grid.py:59-67` (Gaussian-grid convention: leggauss
  nodes ordered north→south, longitudes uniform from 0, no endpoint).
