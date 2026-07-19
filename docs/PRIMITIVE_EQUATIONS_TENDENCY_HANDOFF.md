# Dry Primitive-Equation Tendency — Implementation Blueprint (Handoff)

> **STATUS UPDATE (tendency milestone, 2026-07,
> `feat/primitive-equations-tendency-core`): the tendency described here
> is IMPLEMENTED.** Every item this document marked PROPOSED for the
> tendency milestone is now built and tested: the product-grid
> reconstruction (`_tendency_product_fields`), spectral hydrostatics on
> complex coefficients (tested), the Section-1.4 curl/divergence decision
> (pathway (a), the weak-form vector analysis, chosen as production on
> BOTH backends on measured evidence — design doc Section 8a; pathway (b)
> retained as a cross-validation reference), the thermodynamic and
> ln p_s tendencies, the momentum assembly, and the public gated
> `tendency()` with the rest/BVE/RK4 battery (tests/test_pe_tendency.py).
> Item 3 of Section 2 (normal modes) and item 4 (drift measurement)
> remain open, as do the runner/CLI items of Section 3 (out of scope by
> design). The term-by-term map below is retained for its derivations.

Audience: the agent implementing the nonlinear tendency on top of
`feat/dry-primitive-equations-foundation`. Read
`docs/PRIMITIVE_EQUATIONS_DESIGN.md` first; this document maps each term of
each prognostic equation to what already EXISTS (proven, tested) versus
what must still be BUILT (proposed, unproven). Do not blur that line in
commit messages or docstrings.

Status legend used below:

* **PROVEN** — implemented on the branch with an exact discrete identity or
  analytic reference test.
* **EXISTS** — implemented and tested, no identity claim needed.
* **PROPOSED** — designed here, not implemented, no test exists.

Everything lives in:

* `physics/sigma_coordinate.py` — backend-agnostic column operators
  (NumPy/CuPy via array methods; CuPy never imported).
* `physics/primitive_equations.py` — state, validation, model diagnostics
  (CuPy; per-level use of the SpectralOperators / product-space seams).
* `physics/shallow_water.py` — the pattern to copy for the pseudo-spectral
  horizontal machinery (frozen; do not refactor it into shared helpers).

---

## 1. Term-by-term map

Notation as in the design doc: `G_k = delta_k + A_k`,
`A_k = V_k . grad(ln p_s)`, `P_k = cumsum(G Dsigma)`, `E = |V|^2/2`.

### 1.1 `d(ln p_s)/dt = -Sum_j G_j Dsigma_j`

* **PROVEN**: `column_mass_tendency` (with `interface_sigma_dot` and the
  layer-budget residual). Model wrapper: `continuity_diagnostics` returns
  `g_full`, `v_grad_lnps`, `dlnps_dt`, `sigma_dot`.
* **STILL REQUIRED (PROPOSED)**: the tendency must be produced as SPECTRAL
  coefficients of d(ln p_s)/dt. Decide: analyze the grid field
  `dlnps_dt` on the PRODUCT sampling (G is a product of fields, degree
  ~2*l_max) and truncate once with the 2/3 rule — same policy as every SWE
  nonlinear product. Note `continuity_diagnostics` currently evaluates on
  the STATE grid (diagnostic); the tendency path must rebuild G on the
  product grid.

### 1.2 Hydrostatic `Phi` (diagnostic input to the delta equation)

* **PROVEN**: `hydrostatic_geopotential` (exact isothermal interface
  telescoping); model wrapper `geopotential_fields` (state grid).
* **STILL REQUIRED (PROPOSED)**: `Phi` evaluated on the PRODUCT grid for
  grid-space products, and/or analyzed to spectral coefficients for the
  exact `-lap(...)` term. Because hydrostatic reconstruction is LINEAR in
  (T, Phi_s), it commutes with the spectral transforms level-wise — it can
  be applied directly to spectral T coefficients (complex input is
  supported by the column operators; tested only for real inputs, so add a
  spectral-input test before relying on it).

### 1.3 Vertical transport (all prognostic equations)

* **PROVEN**: `vertical_advection` (centered; identity SBP with the
  KE/variance exchange diagonal), `vertical_flux_divergence` (identities
  A, B, C), `interface_mean`. Model wrapper:
  `vertical_transport_diagnostics` (u, v, T transport + KE closure on
  both backends).
* Usage: thermodynamic and momentum equations subtract
  `vertical_advection(...)` of T and of the grid winds u, v. Constant-T
  and at-rest limits are bitwise zero (tested).
* **STILL REQUIRED (PROPOSED)**: evaluation on the PRODUCT grid: winds,
  sigma_dot, and T must all be synthesized at product points (the
  operators are sampling-agnostic — they act on the level axis).

### 1.4 Momentum -> vorticity/divergence tendencies

The momentum RHS vector at level k, on the product grid:

    N_k = eta_k (k x V_k)  +  W_k  +  R_d T'_k grad(ln p_s)
    with  W_k = (V_adv(u)_k, V_adv(v)_k)      [component-wise, PROVEN]
          eta_k = zeta_k + f
    dV_k/dt = -N_k - grad(Phi_k + E_k + R_d T_ref_k ln p_s)

Then d(zeta_k)/dt = -k.curl(N_k), d(delta_k)/dt = -div(N_k)
- lap(Phi_k + E_k + R_d T_ref_k ln p_s) (the lap term is an exact diagonal
spectral operation — same as `-lap(K + phi)` in the SWE).

* **EXISTS**: the eta k x V part in ISOLATION has closed-form curl/div
  expansions the SWE already uses (`div(eta V) = u.grad(eta) + eta*delta`,
  `k.curl(eta V) = eta*zeta + (grad(eta) x u).k`) — copy that pattern
  verbatim for this term.
* **PROPOSED — the ONE genuinely open discretization decision**: the
  vertical-transport and `R_d T' grad(ln p_s)` parts of `N_k` have no such
  pointwise expansion. Two candidate pathways; neither is implemented or
  tested; DO NOT pick silently — record the choice and its test in the
  design doc:

  (a) **Vector spectral analysis** (Bourke 1974): compute the full
      `(F_u, F_v) = -N` on the product grid, then obtain the zeta/delta
      tendencies via the standard vector-harmonic analysis
      (curl/div moved onto the basis functions by integration by parts:
      one analysis of `F_u/cos`, `F_v/cos` pairs with the
      `d_lambda` / `sin_theta_d_theta` operator matrices applied in
      coefficient space). Requires a new, carefully tested analysis
      routine on BOTH backends; the geodesic backend's least-squares
      transform makes the integration-by-parts step nontrivial — validate
      against pathway (b) on the lat-lon backend first.

  (b) **Scalar round trip**: analyze `F_u`, `F_v` separately as scalars,
      synthesize their spectral derivatives, and form
      `div F = (1/(R cos)) dF_u/dlambda + ...` pointwise, then analyze
      once more. Costs one extra transform round trip and differentiates
      non-smooth (pole-multivalued) component fields — acceptable as a
      REFERENCE implementation to validate (a), not recommended for
      production.

  Acceptance test for either: with delta = 0, T horizontally uniform,
  ln p_s uniform, the zeta tendency must reduce to the BVE Jacobian
  exactly (the SWE's tested degeneracy invariant); additionally the
  sigma_dot = 0 limit must reproduce the SWE-form terms level-wise.

### 1.5 Thermodynamic equation

    dT_k/dt = -[V_k . grad(T)]_k        (horizontal, pseudo-spectral)
              - V_adv(T)_k              [PROVEN]
              + kappa T_k (omega/p)_k   [PROVEN: omega_over_p, heating in
                                         energy_exchange_diagnostics]

* **EXISTS**: horizontal advection pattern = SWE `u.grad(q)` metric
  identity (`(u q_lam - v q_snt)/cos`), per level on the product grid.
* **PROVEN**: the heating term's column identity (E_d) against the PGF
  work (`energy_exchange`, Section 7b) — keep `omega_over_p` fed by the
  SAME G used by continuity, or the identity (and the test) breaks.
* Combine: evaluate all three on the product grid, analyze ONCE, truncate
  ONCE (2/3 rule).

### 1.6 Truncation / dealiasing policy

Copy the SWE policy exactly: each nonlinear product analyzed once on the
backend product space (`so.backend.product_space(so.product_quadrature)`,
"fine" frozen in run configs), truncated once at `cut = (2*l_max)//3`;
linear terms (`-lap(...)`, `-Phi0*delta` analogues such as
`-R_d T_ref lap(ln p_s)` if the T_ref split is adopted) stay exact
diagonal spectral operations.

### 1.7 Monopole handling

* zeta, delta tendencies: hard-zero the l = 0 row per level (as SWE/BVE) —
  conserves per-level circulation and integrated divergence exactly;
  `validate_state` already enforces this on states.
* T tendency: DO NOT zero the monopole (global-mean T evolves through
  the heating term). ln p_s tendency: DO NOT zero (global-mean ln p_s
  evolves; total mass drift is the monitored diagnostic, design doc
  Section 9).

### 1.8 RK4-stage validation

`run.engine.rk4_step_array(..., stage_validator=...)` with
`model.validate_state` wrapped exactly as `run/swe/runner.py::validate_stage`
— intermediate stages must satisfy the same hard constraints (finiteness,
per-level monopoles, T > 0 envelope, finite exp(ln p_s)).

## 2. Invariants and first validation tests (in order)

1. Resting isothermal atmosphere: every tendency coefficient exactly zero
   (the foundation already proves every INGREDIENT is zero; the assembled
   tendency must be too).
2. BVE degeneracy (Section 1.4 acceptance test), lat-lon backend, tight
   tolerance; geodesic looser (R-2 envelope).
3. Linearized gravity/Lamb-wave frequencies for small perturbations about
   isothermal rest vs. the analytic normal modes of the K-level discrete
   system (build the vertical-structure matrix from the SAME alpha/beta
   metadata, not from continuous formulas).
4. Invariant drift in short nonlinear runs: per-level circulation and
   integrated divergence at round-off; total dry mass drift small and
   REPORTED (not asserted zero — ln p_s formulation); total dry energy
   `Integral (c_p T + Phi_s + E) dm` drift small at SWE-comparable levels.
   Only after (4) may any energy-conservation claim be made, and only as
   measured drift numbers.
5. Held–Suarez is OUT OF SCOPE until all of the above pass.

## 3. Recommended commit sequence (narrow, reviewable)

1. Product-grid continuity: G, d(ln p_s)/dt, sigma_dot on the product
   sampling + spectral ln p_s tendency; test against the state-grid
   diagnostics in the band-limited (exact lat-lon) case.
2. The vector curl/div pathway DECISION: implement (b) as reference, then
   (a) if chosen, with the cross-validation test; lat-lon first.
3. Thermodynamic tendency (horizontal + PROVEN vertical + PROVEN heating);
   resting/uniform-T zero tests.
4. Momentum (zeta/delta) tendency assembly; BVE-degeneracy test.
5. Full `tendency()` wiring into `rk4_step_array` + stage validation;
   resting-atmosphere multi-step bit-stability test.
6. Normal-mode test (invariant 3) and short-run drift report (invariant 4).
7. `PERunConfig` / runner / diagnostics recorder / `aeolus run pe` CLI —
   mirror `run/swe/*` and the shared `cli/run_lifecycle.py`; sigma
   interfaces join the run-identity config dict.

## 4. Unresolved items (explicitly NOT decided here)

* Curl/divergence pathway (a) vs (b) — Section 1.4.
* `T_ref(sigma)` profile values and whether the semi-implicit split is
  worth doing before the first explicit validation runs.
* Vertical-structure normal-mode tolerance levels on the geodesic backend.
* Hyperdiffusion coefficient for PE (SWE machinery applies; value TBD).
* Whether `continuity_diagnostics` (state grid) and the product-grid
  tendency path should share code or stay deliberately separate (the SWE
  precedent: diagnostics and tendency paths may differ in sampling but
  must agree in the band-limited exact case — test it).
