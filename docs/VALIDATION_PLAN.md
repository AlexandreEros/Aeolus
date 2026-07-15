# Validation Plan

Audit snapshot: branch `geodesic-grid-refactor`, commit `2a30a18`, 2026-07-11.
Companions: [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[KNOWN_RISKS.md](KNOWN_RISKS.md) (R-n references).

**Status declaration — read first.** As of this audit, *no physics benchmark has been
implemented or run in this repository*. The existing pytest suite (3 tests) exercises
interpolation and transform plumbing with loose thresholds (R-8). The audit itself ran
diagnostic probes (transform round trips, solid-body advection, invariant drift) whose
numbers appear below as *diagnostics*, not benchmarks. Nothing in this plan may be
described as "passing" until the corresponding test exists, runs, and meets its
pre-declared criteria. No new physical features (3-D dynamics, radiation, clouds,
hydrology, …) are in scope; the shallow-water extension in Stage B is the only equation-set
change, chosen because it is the smallest extension that unlocks the two recognized
community benchmarks (Williamson TC2, Galewsky).

**Hard prerequisites** (block Stage A sign-off):

- **P-1**: fix the tendency Jacobian (R-1). The velocity-form advection operator already in
  the codebase was verified correct during the audit and is the recommended replacement.
- **P-2**: enforce a transform operating envelope (R-2), provisionally
  `n_points ≥ 6 · (l_max+1)(l_max+2)/2`, to be replaced by the measured envelope from A-1.
- **P-3**: add a CPU-importable reference path for the SH basis (NumPy/SciPy), both to
  enable CI and to make A-9 (CPU/GPU agreement) meaningful (R-15).

---

## Stage A — validate the existing barotropic-vorticity core

Order matters: A-1 through A-4 are pure-operator tests (no time stepping) and gate the
rest. Effort estimates assume one developer familiar with the code; "S" ≤ ½ day,
"M" ≈ 1–2 days, "L" ≈ 3–5 days.

### A-1. Forward/inverse transform round trips

| | |
|---|---|
| Failure detected | wrong normalization/phase; quadrature-weight errors; under-resolution (Gram ≠ I); layout bugs (m>l slots, conjugate symmetry) |
| Reference | exact: analysis∘synthesis = identity on band-limited data; synthesis of known coefficients vs `scipy.special.sph_harm_y` evaluated on the same points (CPU reference, P-3) |
| Metrics | per-mode diagonal error `|a′_lm/a_lm − 1|`; leakage `‖a′ − a‖₂/‖a‖₂`; Gram diagnostics (diag spread, max off-diag); grid→spectral→grid residual for random band-limited fields |
| Sweeps | resolution ∈ {3,4,5,(6 if GPU allows)} × l_max ∈ {10,15,20,30,45,60} × weights ∈ {voronoi, optimize, uniform} |
| Pass criteria (preliminary) | basis matrix vs SciPy: max abs diff < 1e−12. Round trip: diag error < 1e−6 and leakage < 1e−5 for all l ≤ l_max whenever pts/basis ≥ 6; document the measured envelope as the *authoritative* config table |
| Effort | S–M (the audit probe scripts are 80 % of it) |

Audit diagnostic (not a benchmark): voronoi weights give ~1e−3 accuracy at 9.5 pts/basis
and fail badly at 2.4 pts/basis (see MATHEMATICAL_MODEL §4.1). Note the 1e−6 target likely
requires "optimize" weights or more points — if the envelope at voronoi accuracy saturates
near 1e−3, record that ceiling and carry it into Stage B expectations honestly.

### A-2. Single-mode reconstruction and spectral-operator identities

| | |
|---|---|
| Failure detected | errors in Laplacian eigenvalues, ∂λ, sinθ∂θ recurrence, velocity/gradient assembly, sign conventions |
| Reference | closed forms: `∇²Y_lm = −l(l+1)/R² Y_lm`; `∂λY_lm = imY_lm`; `sinθ∂θY_lm = lε_{l+1,m}Y_{l+1,m} − (l+1)ε_{lm}Y_{l−1,m}`; `u,v` from ψ = Y_lm; gradient of cos/sin test fields; ∇·(k×∇ψ) = 0 |
| Metrics | max relative error on the grid vs closed form, per (l,m) over the retained band |
| Sweeps | same grid/l_max matrix as A-1 |
| Pass criteria | < 1e−10 in spectral space (operators are exact algebra); grid-space comparisons bounded by the A-1 round-trip envelope |
| Effort | S |

### A-3. Analytic gradient/vorticity/Laplacian consistency (grid space)

| | |
|---|---|
| Failure detected | metric-term errors (cos φ factors), pole handling, geodesic-specific artifacts |
| Reference | analytic fields, e.g. f = sin φ, cos φ sin λ, Gaussian caps; verify ζ(k×∇ψ) = ∇²ψ pointwise; check error concentration vs latitude (pole rings) |
| Metrics | area-weighted L2 and L∞ error maps; error-vs-latitude profile |
| Sweeps | resolution 3–5 |
| Pass criteria | no systematic pole-adjacent blow-up beyond the global error level; convergence with resolution at the rate the transform envelope predicts |
| Effort | S |

### A-4. Nonlinear product / Jacobian structural tests

| | |
|---|---|
| Failure detected | exactly the class of bug in R-1: sign, metric factors, missing antisymmetry; broken integral invariants of J |
| Reference | identities: `J(a,a) = 0`; `J(a,b) = −J(b,a)`; `∫J(a,b)dΩ = 0`; `∫a·J(a,b)dΩ = 0`; solid-body advection `J(ψ_sb, Y_lm) = imω·Y_lm` (audit probe); comparison of `jacobian` vs `advect_scalar_by_streamfunction` paths |
| Metrics | relative amplitude and phase error per mode; identity residuals normalized by ‖a‖‖b‖ |
| Sweeps | modes (l,m) across the band × resolutions |
| Pass criteria | solid-body ratio = 1 within 1e−3 (voronoi envelope) for l ≤ 2/3·l_max; identity residuals < 1e−6 (weighted) |
| Effort | S (audit probe exists; convert to pytest with assertions) |

### A-5. Exact travelling-wave solutions of the nonlinear BVE — the recognized analytic benchmark

Two applicable, published, exact solutions of the *nondivergent* BVE (this is precisely the
equation implemented, so both are legitimate — no custom "visually interesting" runs count):

1. **Single-harmonic Rossby–Haurwitz mode.** For ζ₀ ∝ Re Y_lm, ψ ∝ ζ (single l) makes
   J(ψ,ζ) = 0, and the mode propagates *unchanged* with angular phase speed
   `ω_phase = −2Ω m / (l(l+1))` (westward). Classical linear/nonlinear Rossby wave result
   (Haurwitz 1940; standard textbook material, e.g. Holton).
2. **Wavenumber-4 Rossby–Haurwitz wave** (already present as the `rh4` initial condition,
   audit-verified against the closed-form ζ). For the nondivergent BVE the full RH wave
   (zonal l = 1 component + l = 5, m = 4 wave component) translates rigidly at
   `ν = [R(3+R)ω − 2Ω] / [(1+R)(2+R)]` (Haurwitz 1940; quoted for the nondivergent case in
   Williamson et al. 1992, test case 6 discussion). With ω = K = 7.848e−6 s⁻¹, R = 4,
   Ω = 7.292e−5 s⁻¹ this evaluates to ≈ +2.46e−6 rad s⁻¹ (≈ 12.2°/day eastward) — re-derive
   and verify this constant against the cited sources at implementation time before using
   it as truth.

| | |
|---|---|
| Failure detected | advection sign/magnitude errors (R-1 would fail this instantly: reversed propagation), dispersion errors, spurious deformation of the wave |
| Reference | exact solutions above (published closed forms) |
| Metrics | phase-speed error (track argmax of the m-component phase vs time); shape error: normalized ℓ2(ζ − ζ_exact(t)) at t = 1, 5, 30 days; amplitude decay |
| Sweeps | l_max ∈ {15, 30, 42} at admissible resolutions × dt ∈ {dt₀, dt₀/2, dt₀/4} |
| Pass criteria (preliminary) | phase speed within 0.5 % of analytic after 5 days at mid resolution; ℓ2 shape error < 1e−3 at 5 days; both improving with resolution |
| Effort | M |

### A-6. Spatial and temporal convergence

| | |
|---|---|
| Failure detected | order-of-accuracy loss (wrong operator implementations often still "look right" at one resolution); dt-order bugs in RK4 usage |
| Reference | self-convergence (Richardson) on a smooth IC (e.g. `rh4` + small perturbation) against a fine-dt, high-res control; RK4 slope check with operators frozen |
| Metrics | observed order p from error(dt), error(l_max) sequences |
| Sweeps | dt halvings ×4; l_max ∈ {15, 21, 30, 42} |
| Pass criteria | temporal order 4.0 ± 0.2 until floor; spatial error controlled by the A-1 envelope (spectral: error drops until quadrature floor) |
| Effort | M |

### A-7. Conservation of invariants (inviscid)

| | |
|---|---|
| Failure detected | non-conservative advection, aliasing, transform loss (R-5); wrong l = 0 handling |
| Reference | exact invariants of the inviscid BVE: circulation ∮ζdA, kinetic energy −½∮ψζdA, enstrophy ½∮ζ²dA |
| Metrics | relative drift per simulated day; drift vs dt (should scale as dt⁴ if time-limited) and vs resolution (quadrature-limited floor) |
| Sweeps | as A-6 |
| Pass criteria (preliminary) | energy & enstrophy drift < 1e−6/day at mid resolution with correct Jacobian and envelope-compliant l_max; document the achievable floor if the quadrature (not the scheme) limits it |
| Effort | S (audit probe exists) — plus wiring drift monitors into the runner (M) |

Audit diagnostic: current code drifts at ~1e−2 over 6 days (res 4/l_max 20) and ~2.5 % per
12 h at the CLI default — both dominated by R-1/R-2/R-5, so this test primarily certifies
their fixes.

### A-8. Rotation / longitude-invariance and aliasing diagnostics

| | |
|---|---|
| Failure detected | grid imprinting (icosahedral m = 5 symmetry leaking into solutions), longitude-origin dependence, aliasing pile-up at high l ("spectral blocking") |
| Reference | invariance: evolve IC, and separately evolve the same IC rotated by random Euler angles, rotate the result back (rotation of spectral coefficients via Wigner-D, or regenerate the IC analytically in rotated coordinates); the two must agree. Aliasing: product of two band-limited fields analyzed on the geodesic grid vs the exact product coefficients computed with a dense Gauss–Legendre reference (CPU) |
| Metrics | ℓ2 difference between rotated/unrotated runs; spurious-energy fraction above 2/3-cutoff before truncation; energy-spectrum tail slope over time; projection of ζ onto the grid's icosahedral symmetry modes (m ≡ 0 mod 5 fingerprint) |
| Sweeps | ≥ 3 random rotations; resolutions 4–5 |
| Pass criteria (preliminary) | rotation mismatch within 3× the A-1 round-trip floor; no monotonic growth of the m ≡ 0 (mod 5) fingerprint above background in a 30-day turbulent run |
| Effort | M–L (Wigner-D rotation or analytic-rotated ICs; GL reference grid) |

### A-9. CPU/GPU numerical agreement

| | |
|---|---|
| Failure detected | CUDA kernel bugs (recurrence, phase, lgamma normalization), non-determinism, precision loss |
| Reference | NumPy/SciPy implementation of the basis matrix and (slow) transforms on identical points (P-3) |
| Metrics | max abs difference of basis matrices; coefficient differences for random fields; tendency difference for one RK4 step |
| Sweeps | l_max ∈ {5, 15, 45}; a fixed random point set + both grids |
| Pass criteria | basis < 1e−12; one-step tendency < 1e−10 relative |
| Effort | M (writing the reference is P-3 anyway) |

**Stage A exit criteria**: A-1…A-7 all green in CI at declared thresholds; A-8/A-9 measured
and documented (thresholds may be resolution-limited); KNOWN_RISKS R-1…R-8 closed or
formally accepted; the operating-envelope table published in the README.

---

## Stage B — rotating shallow-water equations + Williamson test case 2

### B-0. Design: smallest coherent extension

Extend to the rotating shallow-water equations (SWE) in **vorticity–divergence–geopotential
form** (Bourke 1972; Hack & Jakob 1992, NCAR TN-343 — the standard spectral-transform SWE
formulation):

```
∂ζ/∂t = −∇·[(ζ+f) V]
∂δ/∂t =  k·∇×[(ζ+f) V] − ∇²( Φ + ½|V|² )
∂Φ/∂t = −∇·(Φ V)                     (Φ = g h; optionally split Φ = Φ̄ + Φ′)
```

Prognostics: `ζ_lm, δ_lm, Φ_lm` — same dense (l,m) layout, same transforms. Velocities:
`ψ = ∇⁻²ζ, χ = ∇⁻²δ, V = k×∇ψ + ∇χ` — every operator already exists
(`inv_laplacian`, `d_lambda_coeffs`, `sin_theta_d_theta_coeffs`); the flux divergence/curl
terms are pseudospectral products followed by analysis, exactly like the BVE Jacobian.
Time stepping: the existing RK4 (explicit); note the gravity-wave CFL
`dt ≲ Δx/√(gh₀)` (√(gh₀) ≈ 171 m/s for TC2, ≈ 313 m/s for Galewsky) is ~5–10× stricter than
the advective CFL — accept the cost; a semi-implicit scheme is explicitly *out of scope*
for the minimal extension. BVE remains available as the Φ→∞ degenerate configuration.
Nothing else is added — no 3-D, no physics parameterizations.

Effort: L (model class ≈ 200 lines + ICs + tests), assuming Stage A infrastructure exists.

### B-1. Williamson et al. (1992) test case 2 — the principal analytic correctness benchmark

Steady geostrophically balanced zonal flow; the exact solution is the initial condition,
so any deviation is pure numerical error.

**Canonical parameters** (Williamson, Drake, Hack, Jakob & Swarztrauber, *J. Comput. Phys.*
102 (1992) 211–224 — copy values from the paper at implementation time; the ones below are
the standard set):
`a = 6.37122e6 m`, `Ω = 7.292e−5 s⁻¹`, `g = 9.80616 m s⁻²`, `u₀ = 2πa/(12 days) ≈ 38.61 m/s`,
`gh₀ = 2.94e4 m²/s²`, flow-orientation angle `α ∈ {0, 0.05, π/2 − 0.05, π/2}`; integrate
5 days (standard reporting time; 12-day runs optional).

Protocol, per the audit requirements:

- **State comparison**: h and (u, v) vs the analytic steady state at t = 5 days.
- **Error norms**: Williamson-normalized `ℓ1, ℓ2, ℓ∞` of h (and of the velocity magnitude),
  computed with the grid quadrature weights:
  `ℓ2(h) = sqrt(I[(h−h_T)²]) / sqrt(I[h_T²])`, etc.
- **Sweeps**: resolutions (l_max, grid) ∈ envelope-admissible set, e.g.
  {(21, res4), (31, res5), (42, res5)} × dt ∈ {dt₀, dt₀/2, dt₀/4}.
- **Space/time error separation**: at each resolution, halve dt until ℓ2 stops changing
  (temporal floor) — report the dt-converged value as spatial error; report the dt-slope
  (should be ≈ 4) as temporal error.
- **Conservation**: relative drift of mass `I[h]`, total energy
  `I[½h|V|² + ½g(h² − h_s²)]` (h_s = 0 here), and potential enstrophy
  `I[(ζ+f)²/(2h)]` over 5 days.
- **Rotated configurations**: all four α values; α ≠ 0 runs the flow obliquely across the
  grid's icosahedral symmetry axes and doubles as a grid-imprinting test. (Supported by
  construction — the geodesic grid has no pole alignment to break.)
- **Failure detected**: any residual error in the coupled operator set (divergence/flux
  forms, Coriolis treatment, geopotential gradient), balance-destroying transform loss,
  grid imprinting, conservation defects.
- **Reference**: analytic (exact steady solution) — parameter source: the paper.
- **Pass criteria — declared now, before any run**:
  1. ℓ2(h) at 5 days, dt-converged, decreases monotonically with resolution;
  2. ℓ2(h) ≤ 1e−4 at the (42, res5) configuration — *provisional*: if Stage A-1 shows the
     voronoi-weight quadrature floor is ≳1e−3, this criterion must be re-baselined to
     ≤ 3× that measured floor and the limitation stated openly;
  3. mass drift ≤ 1e−10 (relative), energy and potential-enstrophy drift ≤ 1e−6 over 5 days;
  4. no visible wavenumber-5 imprint in the h-error map above the ℓ∞ level;
  5. α-rotated runs within 2× of the α = 0 error norms.
- **Runtime**: report wall-clock per simulated day (GPU; CPU reference too slow to matter —
  state as N/A unless P-3 grows a full CPU model).
- Effort: M once B-0 lands.

**Decision gate after B-1**: if criterion 2 fails at the re-baselined threshold, stop and
fix the transform (better weights / more points per basis / Gauss–Legendre co-grid for
products) before Stage C. Williamson TC2 is the arbiter of "the dynamical core is correct";
Galewsky must not proceed on a core that fails TC2.

### B-2 (optional, cheap): Williamson TC6 (RH wave in SWE)

Runs the `rh4` machinery in SWE form with h₀ = 8000 m-class parameters from the same paper;
no closed-form truth (the SWE RH wave is not exact) but abundant published reference values
of phase speed and error norms. Secondary priority; do only after TC2 passes.

---

## Stage C — Galewsky–Scott–Polvani barotropically unstable jet

The portfolio-facing nonlinear benchmark. **Blocked until B-1 passes its declared criteria.**

**Canonical setup** (Galewsky, Scott & Polvani, *Tellus* 56A (2004) 429–440 — copy exact
constants from the paper at implementation time): `a = 6.37122e6`, `Ω = 7.292e−5`,
`g = 9.80616`; zonal jet `u(φ) = (u_max/e_n)·exp(1/((φ−φ₀)(φ−φ₁)))` for φ₀ < φ < φ₁, else 0,
with `u_max = 80 m/s`, `φ₀ = π/7`, `φ₁ = π/2 − φ₀`, `e_n = exp(−4/(φ₁−φ₀)²)`; h obtained by
numerically integrating gradient-wind balance with global-mean depth 10 km; perturbation
`h′ = ĥ cosφ · exp(−(λ/α_p)²) · exp(−((φ₂−φ)/β_p)²)` with `ĥ = 120 m, α_p = 1/3, β_p = 1/15,
φ₂ = π/4`; integrate 144 h. Dissipation: use the explicit diffusion operator and coefficient
specified in the paper for the reference solution (∇⁴ hyperdiffusion — take the exact
coefficient from the paper §3 rather than folklore; adding a ∇⁴ option to the SWE core is
part of this stage and is the only permitted "new feature").

Protocol, per the audit requirements:

1. **Balance verification first**: run the *unperturbed* jet for 5 days; ℓ∞(h − h(0)) and
   max|v| must stay at the discretization floor measured in B-1 (declare: ℓ2(h) drift
   ≤ 5× the TC2 error at the same resolution). If the unperturbed jet destabilizes on its
   own, the balance integral or the resolution is wrong — stop.
2. **Day-6 diagnostics**: relative vorticity ζ at t = 144 h (the standard figure), plus h′.
3. **Quantitative reference, not visual similarity**: compare against an identified
   high-resolution reference — the paper's own T341 spectral solution (figures/data), and a
   independently generated reference from an established open code (e.g. a pyshtools/shtns
   SWE script, Dedalus sphere example, or SpeedyWeather.jl) run at ≥ 4× the test
   resolution with matched diffusion. Interpolate both to a common Gauss grid; report
   normalized ℓ2(ζ) and ℓ2(h) at day 6, and the latitude of maximum eddy activity.
4. **Sensitivity studies**: resolution sweep (e.g. l_max 42/63/85 as memory allows — note
   the dense-Y memory ceiling, ARCHITECTURE §4), dt sweep at fixed resolution, and
   dissipation sweep (×½, ×1, ×2 the reference coefficient), reporting how the day-6 ζ
   field and its spectrum respond.
5. **Conservation**: mass, total energy, potential-enstrophy drift over 6 days at each
   resolution.
6. **Spectra & imprinting**: ζ (or PV) spectra at days 4/5/6; check for pile-up at the
   truncation (spectral blocking) and for icosahedral (m ≡ 0 mod 5) fingerprints.
7. **Interpretation discipline**: state explicitly which differences are expected —
   the instability is a chaotic amplifier of truncation-scale noise, so *pointwise*
   day-6 agreement degrades below ~T85; the deliverables distinguish (a) qualitative
   reproduction (eddy positions/count at matched resolution), from (b) quantitative
   convergence (norms decreasing toward the reference with resolution). Claim only what
   the norms support.
- **Failure detected**: nonlinear cascade errors, aliasing/blocking, dissipation-operator
  bugs, imbalance in ICs, grid imprinting under a demanding flow.
- **Pass criteria — declared now**: (i) unperturbed-jet balance criterion above;
  (ii) day-6 ζ ℓ2 difference vs the high-res reference decreasing monotonically with
  resolution; (iii) at the highest feasible resolution, eddy count and phase in the
  standard longitude window match the reference visually *and* the zonal-mean zonal wind
  profile matches within 5 %; (iv) conservation drift within 10× of TC2 levels;
  (v) all sensitivity results documented.
- Effort: L (balance integral S, reference generation M, sweeps + analysis M).

---

## Benchmark table

Every row must eventually link to a script under `tests/` or `benchmarks/` and a results
artifact. **Pass status is "not run" everywhere by definition of this audit.** Audit
diagnostics are quoted in footnotes only; they are not benchmark results.

| Test | Equations exercised | Reference solution | Parameter source | Resolutions / timesteps | Error metrics | Conservation reported | CPU runtime | GPU runtime | Pass | Caveats |
|---|---|---|---|---|---|---|---|---|---|---|
| A-1 round trips | transforms only | identity; SciPy basis | — | res 3–6 × l_max 10–60 × 3 weight modes | per-mode diag/leak, Gram | — | TBD (P-3) | TBD | **not run**¹ | envelope defines all later configs |
| A-2 operator identities | spectral operators | closed forms | — | as A-1 | max rel. error | — | TBD | TBD | **not run** | |
| A-3 grid-space calculus | operators + synthesis | analytic fields | — | res 3–5 | wtd L2/L∞, lat profile | — | TBD | TBD | **not run** | pole rings |
| A-4 Jacobian structure | nonlinear term | identities; solid-body | — | modes × res | ratio, residuals | — | TBD | TBD | **not run**² | gates R-1 fix |
| A-5 RH mode / RH4 wave | full BVE | exact (Haurwitz 1940) | Williamson 1992 TC6 params | l_max 15/30/42 × 3 dt | phase speed, ℓ2 shape | C, E, Z drift | TBD | TBD | **not run** | verify phase-speed constant vs sources |
| A-6 convergence | full BVE | self (Richardson) | — | 4 dt halvings × 4 l_max | observed order | — | TBD | TBD | **not run** | |
| A-7 invariants | full BVE, ν=0 | exact invariants | — | as A-6 | drift/day | C, E, Z | TBD | TBD | **not run**³ | floor set by quadrature |
| A-8 rotation/aliasing | full BVE | self + GL reference | — | ≥3 rotations, res 4–5 | ℓ2 mismatch, spectra, mod-5 proj | — | TBD | TBD | **not run** | needs Wigner-D or analytic rotated ICs |
| A-9 CPU/GPU | transforms, 1 step | NumPy/SciPy port | — | l_max 5/15/45 | max abs diff | — | TBD | TBD | **not run** | requires P-3 |
| B-1 Williamson TC2 (×4 α) | full SWE | analytic steady state | Williamson et al. 1992 | ≥3 (l_max,res) × 3 dt | Williamson ℓ1/ℓ2/ℓ∞ (h, V) | mass, energy, pot. enstrophy | N/A | TBD | **not run** | criteria may re-baseline to measured quadrature floor |
| B-2 Williamson TC6 | full SWE | published values | Williamson et al. 1992 | 2 res | phase speed, norms | as B-1 | N/A | TBD | **not run** | optional |
| C-1 Galewsky balanced jet | full SWE | initial state | Galewsky et al. 2004 | ≥2 res | drift norms | as B-1 | N/A | TBD | **not run** | prerequisite to C-2 |
| C-2 Galewsky day-6 | full SWE | paper T341 + independent high-res run | Galewsky et al. 2004 | l_max 42/63/85 × dt × ν sweeps | day-6 ℓ2(ζ,h), spectra, u profile | as B-1 | N/A | TBD | **not run** | chaotic sensitivity; claim per §C.7 |

¹ Audit diagnostic: voronoi round trip ≈1e−3 at ≥9 pts/basis; 7 %/30 % at 2.4 pts/basis (CLI default).
² Audit diagnostic: current Jacobian fails (ratios −0.76…−0.81 vs +1); velocity form passes.
³ Audit diagnostic: current code drifts ~1–2 %/6 days (res4/l20), ~2.5 %/12 h (defaults).

Do **not** proceed to Held–Suarez, Jablonowski–Williamson, or any 3-D test: the repository
contains no 3-D primitive-equation core, validated or otherwise.

---

## Smallest set of changes that makes this repository credible public evidence

Ordered; items 1–6 are the minimum. "Credible" means: a stranger can clone the repo, run
one command, and see quantified, referenced correctness evidence — with limitations stated.

1. **Fix R-1** (use the verified velocity-form advection) and **R-2** (enforce the
   envelope, change CLI defaults to an admissible pair, e.g. `--lmax 28 --resolution 4`).
   Without this, publishing the repo is negative evidence.
2. **P-3 CPU reference basis + CI**: a NumPy/SciPy basis implementation behind the same
   interface, `pytest` markers separating `gpu`/`cpu`, and a GitHub Actions workflow
   running the CPU math tests on every push. This single change converts all math tests
   into publicly verifiable claims.
3. **Convert A-1, A-2, A-4, A-7 into asserting pytest tests** with the declared
   thresholds (replacing the current no-assert / 60 %-tolerance tests) and delete or fix
   `step_leapfrog` (R-6). These four cover transforms, operators, the nonlinear term, and
   conservation — the minimum spanning set for "the solver solves its equation".
4. **A-5 Rossby–Haurwitz benchmark** with a results table (phase-speed and shape error vs
   resolution) committed under `benchmarks/` — the recognized analytic BVE result, and the
   headline demonstration until Stage B exists.
5. **README rewrite**: equations actually solved, SH conventions, the operating-envelope
   table, how to run tests/benchmarks, current limitations (GPU-only, quadrature floor,
   no SWE yet), and links to these four documents. State plainly what has and has not been
   validated — the honesty is itself the evidence of competence.
6. **Repository hygiene** (R-16/R-17): delete dead engines/`.bak`/empty modules or move
   them to `experimental/`, fix the mislabeled diagnostics (R-14), and declare real
   dependencies in `pyproject.toml`.
7. *(Stretch, high leverage)* Stage B + Williamson TC2 with the full error-norm table:
   this is the smallest step that turns the repo from "correct operators" into "a
   dynamical core validated against the community-standard benchmark."

Everything else in this plan is staged behind these.
