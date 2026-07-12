# Known Risks, Discrepancies, and Likely Bugs

Audit snapshot: branch `geodesic-grid-refactor`, commit `2a30a18`, 2026-07-11.
Ranked by severity. **[measured]** = verified numerically during the audit on this machine
(Python 3.12, CuPy 13.4, GeForce MX110); reproduction probes are described inline so they
can be re-run. Nothing in this list has been fixed yet — this is the audit record.

Severity scale: **S1** invalidates results · **S2** materially degrades results ·
**S3** correctness/robustness hazard · **S4** hygiene/documentation.

> **Fix log (branch `fix/s1-jacobian-and-envelope`).** Both S1 risks are fixed and
> locked with asserting tests (`tests/test_spectral_operators.py`):
> - **R-1 fixed** — `jacobian_pseudospectral` now computes the true spherical Jacobian;
>   solid-body advection of single modes recovers `i m ω` to <5e−3 (was −0.76…−0.81).
> - **R-2 fixed** — `GeodesicSphericalHarmonics` warns when under-resolved
>   (`n_points < 6·n_basis`); CLI default changed from (l_max 45, res 4) to (l_max 21, res 4).
>
> The S2/S3/S4 risks below are **not** addressed on this branch. Note that correcting R-1
> makes the enstrophy cascade physical, so it now exposes **R-3/R-5** as the binding
> constraint on inviscid conservation (measured: energy drift is round-trip-loss-limited and
> falls ~14× from res 4→5).
>
> **Correction (post-fix re-measurement).** Earlier drift numbers quoted *relative*
> enstrophy ½∮ζ²dA, which is **not** an invariant of the rotating BVE — the materially
> conserved quantity is *absolute* enstrophy ½∮(ζ+f)²dA. Re-measured on the same rotating
> case (res 4 / l_max 20, ν = 0, ≈6 days): relative enstrophy changed +6.7 %, but **95 % of
> that is physical exchange with planetary vorticity**; the genuine numerical error,
> measured on absolute enstrophy, is ΔZ_abs ≈ +30 vs an eddy enstrophy of 9.65e3
> (**≈0.3 %**). Energy drift (−1.6 %) is unaffected by this correction and remains the
> dominant conservation defect (R-5). Diagnostics must therefore always track
> Z_abs = ½∮(ζ+f)²dA on rotating runs; Z_rel alone misattributes physics as error.

---

## S1 — Invalidates simulation results

### R-1. The Jacobian used by the BVE tendency computes −cos φ·J(ψ,η) instead of J(ψ,η)  — ✅ FIXED (branch `fix/s1-jacobian-and-envelope`)

`SpectralOperators.jacobian_pseudospectral`
([spectral_operators.py:344](src/planetary_sandbox/numerics/spectral_operators.py)) forms

```
J_code = (a_λ·(sinθ b_θ) − (sinθ a_θ)·b_λ) / (R² cos φ)
```

Since `sinθ ∂/∂θ = −cos φ ∂/∂φ`, this equals `−cos φ · J(a,b)`: the correct expression needs
division by cos²φ and the opposite sign (equivalently: `(sinθa_θ·b_λ − a_λ·sinθb_θ)/(R²cos²φ)`).
`BarotropicVorticity.tendency` then applies `transform(−J_code)`, so the advection term
integrated is **`+cos φ · J(ψ, ζ+f)` instead of `−J(ψ, ζ+f)`** — reversed sign with a
spurious latitude-dependent amplitude.

**[measured]** Solid-body streamfunction advecting single modes Y₃², Y₅³, Y₈⁴: ratio of
computed to analytic `(u·∇q)_lm` = **−0.805, −0.791, −0.761** (should be +1.000). The
velocity-form operator `advect_scalar_by_streamfunction` on the same inputs gives
**+1.0000, +1.0000, +1.0009** — i.e. a correct advection operator already exists in the
codebase but is not the one used.

Consequences: Rossby-wave propagation direction is reversed; vortex mutual advection is
reversed and latitude-weighted; **every figure in `out/` and every qualitative conclusion
drawn from runs to date does not depict solutions of the barotropic vorticity equation.**

Remediation (Stage A prerequisite): make `tendency` use the velocity form
(`advect_scalar_by_streamfunction(ψ_lm, η_lm)`, verified correct), or fix the Jacobian to
`(sinθa_θ·b_λ − a_λ·sinθb_θ)/(R² cos²φ)`, then lock with the solid-body and single-mode
Rossby tests (VALIDATION_PLAN A-4, A-5).

**Fix applied**: `jacobian_pseudospectral` corrected in-place to
`(a_sinth·b_lam − a_lam·b_sinth)/cos²φ`. **[measured, post-fix]** solid-body advection of
Y₃², Y₅³, Y₈⁴, Y₁₂⁶ now recovers the analytic `i m ω` to within 5e−3 (worst 2.9e−3 at l=12,
a quadrature effect that shrinks with resolution); it agrees with the independent
`advect_scalar_by_streamfunction` path to <1e−3; `J(a,a)=0` to machine zero; `∮J(a,b)dΩ≈0`.
Locked by `tests/test_spectral_operators.py`.

### R-2. Default configuration (l_max=45 on a resolution-4 grid) is unsupported by the transform  — ✅ FIXED (branch `fix/s1-jacobian-and-envelope`)

The CLI default pairs 1081 basis functions with 2562 points (2.4 points per basis
function) using Voronoi weights. **[measured]** analysis∘synthesis round trip: diagonal
error up to **7.1 %** and spurious-coefficient leakage RMS up to **30 %** at l = 45;
Gram-matrix off-diagonals reach 0.099. At res 5 (9.5 pts/basis) the same l_max is fine
(≤0.1 % / ≤1 %). Because the tendency round-trips the state through the transform on every
evaluation (see R-5), the defaults produce massive spurious dissipation:
**[measured]** 2.5 % kinetic-energy loss in 12 simulated hours with ν = 0.

Remediation: enforce/validate an operating envelope (e.g. `n_points ≥ 6·n_basis`, i.e.
l_max ≤ ~27 at res 4, ~57 at res 5) at construction time; change the CLI defaults; document
the envelope from the A-1 test matrix.

**Fix applied**: `GeodesicSphericalHarmonics` now emits a `UserWarning` (with the largest
safe l_max) whenever `n_points < 6·n_basis`; the `psx-bve` default changed from
(l_max 45, res 4) → (l_max 21, res 4), i.e. ~10 pts/basis. The 6× threshold is a soft guard,
not a hard error, so existing workflows are not broken. The authoritative envelope table is
still owed by VALIDATION_PLAN A-1. Locked by `tests/test_spectral_operators.py`
(warns at l_max 45, silent at l_max 15).

---

## S2 — Materially degrades results

### R-3. "Dealiasing" is truncation-only and does not dealias

The 2/3-rule zeroing in `jacobian_pseudospectral`/`advect_scalar_by_streamfunction` is
applied *after* analyzing the product on the same point set. With inexact quadrature there
is no exact-integration grid, so aliased energy folds into retained modes before the
truncation; the m-truncation line is redundant (m ≤ l already). Aliasing error is
unquantified. (VALIDATION_PLAN A-8 defines the diagnostic.)

### R-4. Time step is fixed from the initial state; "adaptive time-stepping" claim is wrong

`run_bve` computes `dt = 0.5·min_edge/max|u₀|` once (commit 8666138 claims adaptivity).
**[measured]** max speed grew 25.4 → 37.5 m/s in a 12 h default run with dt unchanged.
Also the CFL length scale should follow the spectral resolution (2πR/l_max), not the mesh
edge, and RK4 stability for the ν∇² term is never checked. No instability was observed in
short audit runs, but the margin is uncontrolled.

### R-5. Inviscid invariants drift at O(1 %)/day; state round-trips through a lossy transform every tendency call  — ✅ FIXED (branch `fix/r5-spectral-absolute-vorticity`)

`tendency` synthesizes ζ to the grid, adds f, and re-analyzes — 1 lossy round trip per
evaluation, 4 per RK4 step — although `f_lm` is a constant (a single (1,0) coefficient)
that could be added in spectral space. **[measured]** ν = 0, res 4 / l_max 20, 60 steps
(≈6 days): energy −2.4 %, enstrophy −1.4 % (circulation pinned to ~0 by construction).
For a correctly dealiased spectral BVE these should be at time-integration error levels.
No invariant is monitored during runs.

**Fix applied** (`89985aa`): η = ζ + f_lm built in spectral space; f_lm is the exact
(1,0) coefficient `2Ω·sqrt(4π/3)`; the state is never synthesized/re-analyzed in the
tendency. Mechanism quantified (`tests/audit_r5_mechanism.py`): the transform recovers
f's a₁₀ to machine precision but **leaks ~0.85 % of f across other degrees — ~12 % of
‖ζ‖ injected into η per call** (f ≈ 48×‖ζ‖), 12.4× the ζ round-trip error.
**[measured, 10-day rotating baseline res4/l21]** absolute-enstrophy drift
−7.9e−3 → **−3.6e−4** (22×); l=1 energy loss −18.8 % → −5.7 %; 0.5-day energy drift
+1.1e−2 → +3.5e−5 (~300×); total 10-day energy drift −6.4 % → **−3.8 %**; wall time
−21 % (8 fwd + 20 inv transforms per RK4 step, was 12 + 24). Locked by
`tests/test_r5_spectral_eta.py` (verified failing on parent `0b6c135`).

**Attribution corrections recorded for honesty:** (i) the earlier claim that the f
round trip explained the *entire* rotating energy loss was wrong — the non-rotating
"control" behind it is a quasi-steady state (axisymmetric vortices ~120° apart;
tendency ~1e−13), i.e. it has no dynamics to lose energy; (ii) the falsifiable
prediction "~0.01 % rotating drift after R-5" is **refuted** — the remaining −3.8 %/10 d
is not the η construction (instantaneous dE/dt is a null for both constructions since
∮ψJ(ψ,η)dA = 0 for any η; the damage was trajectory-level). Leading suspects for the
remainder: truncation-only dealiasing (R-3) acting on the β-driven cascade, product-
analysis quadrature error, and time-step size (R-4) — attribution is the next
fixed-dt/resolution sweep, not yet established.

### R-6. `step_leapfrog` is dead code that crashes if called

It passes raw coefficient arrays to `vorticity_to_streamfunction`, which asserts
`isinstance(vrt_state, BarotropicState)` → `AssertionError` on first use. Only `rk4_step`
is actually used. Delete or fix.

### R-7. Legacy lat–lon transform drops the last longitude panel (systematic 1/n_lon bias)

`LatLonSphericalHarmonics` + `simpson_2d`: longitudes are built with `endpoint=False`
(period missing its final panel) while Simpson's rule integrates the open interval; the
even-n fallback weights in `simpson_2d` are also incorrect. **[measured]** for f = 1 on a
33×65 grid: a₀₀ = 3.4904 vs √(4π) = 3.5449 — a deficit of exactly 1/65 — plus spurious
coefficients up to 3e−2. This engine is the *reference* in `consistency_test.py` and
`test_spherical_harmonics_compare.py`, so those comparisons are anchored to a biased
baseline (hence the 60 % tolerance, see R-8). The geodesic engine is *more* accurate than
the "truth" it is compared against.

### R-8. The test suite cannot catch regressions

- `test_orthogonality.py` has **no assertions** (prints diagnostics, returns a tuple —
  pytest even warns about the non-None return).
- `test_spherical_harmonics_compare.py` asserts relative L2 < **0.6** on low-degree
  coefficients only.
- `test_grid_interpolation.py` asserts RMSE < 0.2 for an O(1) field.
- `consistency_test.py` executes at import (module-level code), is not collected, and its
  0.1 "success threshold" is a print, not an assert.
- Nothing tests operators, the Jacobian, the model tendency, or conservation — which is how
  R-1 survived. All 3 collected tests pass **[measured]** while the dynamical core is wrong.

---

## S3 — Correctness/robustness hazards

### R-9. CLI default `--day-hours inf` silently yields a non-rotating planet

`PlanetaryParameters` accepts it (Ω = 0, oblateness 0), so the flagship "barotropic
vorticity on a rotating sphere" demo runs with f = 0 unless the user passes `--day-hours`.
Make rotation explicit, or default to 24 h.

### R-10. Diffusion uses the mutated l = 0 eigenvalue

`BarotropicVorticity.__init__` overwrites `laplacian_eig[0] = +1/R²`; the diffusion term
`ν·λ_l·ζ_lm` therefore *amplifies* the l = 0 mode. Currently masked by the hard-zeroing of
the l = 0 tendency row — a silent coupling between two hacks. Keep the eigenvalue array
physical and special-case the inversion instead. Related: `SpectralOperators.inv_laplacian`
and `vorticity_to_streamfunction` treat the l = 0 mode differently (1/R² vs zeroed).

### R-11. `d_lambda_coeffs` mutates shared precomputed state on every call

`self._im_m_over_R` is initialized with `[0,0] = 1.0` "to avoid division by zero", then the
*getter* sets `[0,0] = 0.0` in place each call. Anything reading the buffer before the
first call sees the wrong value; the pattern is also a thread-safety hazard. Precompute the
final values once.

### R-12. Unclamped `1/cos φ` divisions rely on a hidden 0.01-rad grid rotation

`grad_from_scalar` divides by `grid.coslat` with no floor (velocity path floors at 1e−6,
Jacobian at 1e−8). Nothing prevents a future grid (or a user-supplied point set with a point
at a pole) from producing Inf/NaN. The icosahedron "anti-singularity" rotation is the only
protection, and it is undocumented.

### R-13. Optimized-weight cache key omits radius and grid orientation

`sh_weights_res{r}_lmax{L}.pkl` verifies `n_points` and basis-id only. Weights computed for
one radius are valid solid-angle weights for another (they are normalized), so this is
currently benign, but any change to the anti-singularity rotation or subdivision scheme
would silently reuse stale caches. Include a content hash of the point set.

### R-14. Diagnostics mislabeled and inconsistent

`VorticityViewer`: "Total Kinetic Energy (K) … J/kg" is actually ∫½|u|²dA (m⁴ s⁻², per unit
density); "RMS Vorticity (Enstrophy)" is RMS ζ, not enstrophy ½∫ζ²dA; area weights use
`equatorial_radius` while the dynamics use the volumetric mean `params.radius`; plots
render on a −89…89° remapped lat-lon grid (pole rows extrapolated by `griddata`).

### R-15. GPU-only, Windows-TDR-exposed execution

No CPU fallback exists (module-scope `import cupy` everywhere), so: no CI without a GPU
runner, no CPU/GPU cross-validation, and on Windows display GPUs long kernels are killed by
the WDDM watchdog — **[observed]** `cudaErrorLaunchTimeout` during an audit Gram-matrix
computation at res 5. Pin of `cupy-cuda11x` also ties the project to CUDA 11.

---

## S4 — Hygiene and documentation

### R-16. Latent/duplicated numerical machinery

`DifferentialOperatorsSpherical` (local least-squares gradients + a **graph Laplacian
`D − W`, which is not a consistent surface-Laplacian discretization**) is constructed
lazily but never called by any live path. `sh_matrix_real.cu` is never loaded.
`LatLonSphericalHarmonics` survives only as a (biased, R-7) test baseline. Either quarantine
these clearly as experimental or delete them.

### R-17. Assorted hygiene

- Empty modules: `run/bve/config.py`, `run/bve/io.py`, `physics/gravity.py`,
  `viz/spectra.py`, `viz/__init__.py`.
- `.bak` files inside the package (`sh_matrixcu.bak`, `fast_geodesic_shpy.bak`).
- Large commented-out blocks (old `Planet.generate`, old `SpectralOperators`).
- `Planet.generate` computes `radial_distance` twice; runs terrain synthesis even for BVE
  runs; default terrain RMS is 0.1 % of the radius (6.4 km) — decorative but odd.
- `_random_low_l` writes coefficients into invalid m > l slots (harmlessly ignored) and
  non-zero imaginary parts on m = 0 (silently dropped by synthesis).
- `pyproject.toml` declares `dependencies = []` ("dependency hell" comment) while the
  package hard-requires numpy/scipy/cupy/matplotlib — `pip install planetary-sandbox`
  would produce a broken install; `requirements.txt` is the real manifest.
- Typo `ovarall_step` (runner); docstrings claiming "real basis" where the basis is
  complex (`terrain_spectral`, `LatLonSphericalHarmonics.inv_transform`); stale
  colatitude comment on the Coriolis setup.
- README documents only setup and two CLI examples; no statement of equations, status, or
  limitations.

---

## What was *checked and found sound* (for balance)

- Basis normalization/orthogonality at adequate sampling (Gram ≈ I to ~1e−3 with Voronoi
  weights at ≥9 pts/basis; machine-exact recovery of Y₀⁰, Y₁⁰ coefficients).
- Laplacian / inverse-Laplacian eigenvalue treatment (l ≥ 1), zonal derivative, the
  `sinθ ∂θ` recurrence (re-derived analytically and spot-checked), velocity-from-ψ and
  scalar-gradient operators (machine precision on solid-body fields **[measured]**),
  the velocity-form advection operator, circulation pinning, RK4 arithmetic,
  RH4 initial-condition formula (matches the closed form for ζ = ∇²ψ/… with (R²+3R+2) = 30
  at R = 4), Condon–Shortley handling after the recorded double-phase fix.
