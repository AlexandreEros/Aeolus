# Dry Hydrostatic Primitive Equations — Design (Foundation Milestone)

Status: **foundation only**. This document specifies the formulation for the
future dry hydrostatic primitive-equation (PE) core and the exact scope of
the first milestone: vertical-grid metadata, the spectral state
representation, state validation, hydrostatic geopotential reconstruction,
the discrete column continuity operator (surface-pressure tendency and
interface sigma-velocity), and the column mass-closure diagnostics.

**No prognostic tendency is implemented in this milestone.** The model class
deliberately has no `tendency()` method; nothing silently returns zero.
Every discretization stated here as "deferred" is a documented decision
point, not an implemented default.

Notation: `a` planetary radius, `Omega` rotation rate, `f = 2*Omega*sin(lat)`,
`k` local vertical unit vector, `V = (u, v)` horizontal velocity (eastward,
northward), `grad`/`div`/`curl`/`lap` the horizontal (spherical-surface)
operators at constant `sigma`, `d/dt` local time derivative.

---

## 1. Continuous governing equations and sign conventions

Vertical coordinate: **sigma**, `sigma = p / p_s`, `sigma = 0` at the model
top, `sigma = 1` at the surface. `sigma` increases downward; `sigma_dot =
D(sigma)/Dt` is positive for downward motion.

Prognostic form is **vorticity–divergence** (vector-invariant), matching the
shallow-water core's conventions exactly. With

    zeta  = k . curl(V)          relative vorticity
    delta = div(V)               horizontal divergence
    eta   = zeta + f             absolute vorticity
    E     = (u^2 + v^2) / 2      horizontal kinetic energy per unit mass
    G     = delta + V . grad(ln p_s)      ( = div(p_s V) / p_s )

the momentum equation

    dV/dt = -eta k x V - sigma_dot dV/dsigma - grad(Phi + E) - R_d T grad(ln p_s)

yields, after applying `k.curl` and `div`,

    d(zeta)/dt  = -div(eta V) - k . curl( sigma_dot dV/dsigma + R_d T grad(ln p_s) )

    d(delta)/dt =  k . curl(eta V) - div( sigma_dot dV/dsigma + R_d T grad(ln p_s) )
                   - lap(Phi + E)

Thermodynamic equation (dry, adiabatic; `kappa = R_d / c_p`):

    dT/dt = -V . grad(T) - sigma_dot dT/dsigma + kappa T (omega / p)

    omega / p = sigma_dot / sigma + d(ln p_s)/dt + V . grad(ln p_s)

Mass continuity, integrated over the column (boundary conditions
`sigma_dot = 0` at `sigma = 0` and `sigma = 1`):

    d(ln p_s)/dt = - Integral_0^1 G dsigma

    sigma_dot(sigma) = sigma * Integral_0^1 G dsigma' - Integral_0^sigma G dsigma'

Hydrostatic balance (diagnostic; at fixed horizontal position `ln p =
ln sigma + ln p_s`, so `d(ln p) = d(ln sigma)` within a column):

    dPhi/d(ln sigma) = -R_d T ,        Phi(sigma = 1) = Phi_s

Sign conventions are those already used by the BVE and SWE cores:
`zeta > 0` counterclockwise seen from outside (northern-hemisphere cyclonic),
`delta > 0` for divergent flow, `f = 2*Omega*sin(lat)` is the exact spectral
(1,0) mode. In the pure-rotational, uniform-`ln p_s`, horizontally uniform-`T`
limit the `zeta` equation must degenerate pointwise to the BVE, exactly as
the SWE does (this is the acceptance invariant for the future tendency).

Physical constants (values fixed here so tests and docs cannot drift):

    R_d = 287.04  J kg^-1 K^-1      dry-air gas constant
    c_p = 1004.64 J kg^-1 K^-1      dry-air isobaric heat capacity
    kappa   = R_d / c_p           ~= 0.28572
    gamma_d = c_p / (c_p - R_d)   ~= 1.4      (used only for the CFL bound)

## 2. Prognostic and diagnostic variables, dimensions, units

Prognostic (spectral, complex orthonormal-SH coefficients, `m >= 0` storage,
layout identical to BVE/SWE: axis 0 = degree `l`, axis 1 = order `m`,
entries with `m > l` are zero):

| variable   | placement        | count | units      |
|------------|------------------|-------|------------|
| `zeta_k`   | full levels      | K     | s^-1       |
| `delta_k`  | full levels      | K     | s^-1       |
| `T_k`      | full levels      | K     | K (kelvin) |
| `ln p_s`   | surface (2-D)    | 1     | dimensionless (`p_s` in Pa) |

`T` is the **full** temperature (its `(0,0)` monopole is the global-mean
temperature, nonzero). The reference/perturbation split `T = T_ref(sigma) +
T'` is a *semi-implicit arrangement* deferred to the tendency milestone; it
changes which terms are linear, not the equations.

**`ln p_s`, not `p_s`, is prognostic.** Justification: (a) `p_s =
exp(ln p_s) > 0` by construction, so surface-pressure positivity can never
be violated by the time integration; (b) the pressure-gradient and
continuity terms need `grad(ln p_s)` directly; (c) it is the standard choice
of the spectral PE literature (Bourke 1974; Hoskins & Simmons 1975). The
known cost — the global mean of `exp(ln p_s)` (total dry mass) is not
conserved exactly by a spectral `ln p_s` equation — is accepted and
*monitored* (Section 9); a mass fixer is deferred.

Diagnostic:

| variable            | placement             | units       |
|---------------------|-----------------------|-------------|
| `Phi_k`             | full levels           | m^2 s^-2    |
| `Phi_{k+1/2}`       | interfaces below each layer | m^2 s^-2 |
| `sigma_dot_{k+1/2}` | interfaces            | s^-1        |
| `d(ln p_s)/dt`      | surface               | s^-1        |
| `u_k, v_k`          | full levels (grid)    | m s^-1      |

`Phi_s` (surface geopotential) is a fixed spectral field owned by the model
(default: all zeros — no topography). It is representable from day one so
adding topography later is a data change, not a schema change.

## 3. Vertical placement (Lorenz staggering)

`K` full levels indexed `k = 1..K` **top to bottom**; `K+1` interfaces
`k+1/2 = 1/2 .. K+1/2` with

    sigma_{1/2} = 0  (model top),   sigma_{K+1/2} = 1  (surface),
    sigma interfaces strictly increasing, all finite.

Layer thickness `Dsigma_k = sigma_{k+1/2} - sigma_{k-1/2} > 0`.

Full-level coordinate: `sigma_k = (sigma_{k-1/2} + sigma_{k+1/2}) / 2`
(arithmetic mean). Full-level pressure `p_k = sigma_k p_s`.

Lorenz staggering: `zeta, delta, T, Phi` (and later all their tendencies)
live at full levels; `sigma_dot` lives at interfaces. There is no prognostic
variable at interfaces. The known cost of the Lorenz grid — a computational
mode in the vertical temperature structure — is accepted for this core
(Section 12); the Charney–Phillips alternative was rejected because every
prognostic field staying at full levels keeps the state a single
`(3K+1, l_max+1, l_max+1)` array that plugs directly into the existing
`rk4_step_array` engine.

## 4. Hydrostatic integration and boundary condition

Simmons & Burridge (1981) discretization, specialized to sigma. Upward
recursion from the surface boundary condition `Phi_{K+1/2} = Phi_s`:

    Phi_{k-1/2} = Phi_{k+1/2} + R_d T_k ln(sigma_{k+1/2} / sigma_{k-1/2})     (k = K, ..., 2)

    Phi_k = Phi_{k+1/2} + alpha_k R_d T_k

    alpha_k = 1 - (sigma_{k-1/2} / Dsigma_k) ln(sigma_{k+1/2} / sigma_{k-1/2})   (k >= 2)
    alpha_1 = ln 2                                                (top layer, sigma_{1/2} = 0)

`Phi_{1/2}` (the geopotential of the `sigma = 0` interface) is **not
defined and never computed** — it is infinite for any atmosphere with
nonzero top-layer temperature, and nothing needs it: the PGF uses only
full-level `Phi_k`.

Exactness property used by the tests: for an isothermal column `T_k = T0`,
the interface recursion telescopes to

    Phi_{k+1/2} = Phi_s - R_d T0 ln(sigma_{k+1/2})     exactly (round-off only),

which is the analytic isothermal profile. Full-level values satisfy
`Phi_k = Phi_s - R_d T0 (ln sigma_{k+1/2} - alpha_k)`, i.e. the analytic
profile evaluated at the effective level `ln(sigma~_k) = ln(sigma_{k+1/2}) -
alpha_k`; for the top layer `sigma~_1 = sigma_{3/2}/2 = sigma_1` exactly.

## 5. Discrete continuity equation

With grid-point `G_k = delta_k + V_k . grad(ln p_s)` at full levels, the
vertical integral is the exact layer-thickness-weighted sum:

    d(ln p_s)/dt = - Sum_{j=1..K} G_j Dsigma_j

## 6. Recovery of the surface-pressure tendency and sigma_dot

Partial sums of the same quantity give the interface sigma-velocity:

    sigma_dot_{k+1/2} = sigma_{k+1/2} * Sum_{j=1..K} G_j Dsigma_j
                        - Sum_{j=1..k} G_j Dsigma_j          (k = 0..K)

Impermeability is **structural**, not enforced by clamping:

* top (`k = 0`): both terms are empty/zero, so `sigma_dot_{1/2} = 0`
  identically;
* bottom (`k = K`): the two sums are equal and cancel **exactly in floating
  point** — the implementation computes the bottom entry from the same
  cumulative sum object that appears in the first term, so the cancellation
  is bitwise, not approximate.

Discrete layer mass budget (the identity the closure diagnostics test):

    Dsigma_k * d(ln p_s)/dt + G_k Dsigma_k
        + (sigma_dot_{k+1/2} - sigma_dot_{k-1/2}) = 0     for every layer k.

This holds to round-off by construction; the diagnostic reports the maximum
absolute residual per column so any future refactoring that breaks the
telescoping is caught immediately.

## 7. Vertical-advection discretization and boundary behavior

### 7a. Vertical transport (RESOLVED; implemented)

Notation: `<X, Y> = Sum_k Dsigma_k X_k Y_k` is the sigma-mass-weighted
column inner product; `sigma_dot` boundary rows are structurally zero
(Section 6); `d(ln p_s)/dt = -Sum_j G_j Dsigma_j` from Section 5.

Two operators are defined at full levels, valid on any nonuniform grid.
Advective form (used for every prognostic transport term):

    V_adv(X)_k = [ sigma_dot_{k+1/2} (X_{k+1} - X_k)
                 + sigma_dot_{k-1/2} (X_k - X_{k-1}) ] / (2 Dsigma_k)

Flux form (conservation bookkeeping), with the ARITHMETIC-MEAN interface
value `Xhat_{k+1/2} = (X_k + X_{k+1}) / 2` — the 1/2:1/2 weights are not a
convenience but are FORCED by identity (A) below, even on nonuniform grids:

    V_flux(X)_k = [ sigma_dot_{k+1/2} Xhat_{k+1/2}
                  - sigma_dot_{k-1/2} Xhat_{k-1/2} ] / Dsigma_k

Boundary behavior: the terms multiplying `sigma_dot_{1/2}` and
`sigma_dot_{K+1/2}` are structurally absent (never evaluated), so no ghost
levels or extrapolated `X_0`, `X_{K+1}` values exist; for K = 1 both
operators are identically zero.

Exact discrete identities (each derived by index shifting / Abel summation
with the zero boundary sigma-velocities, and each enforced by tests):

  (A) Flux/advective compatibility, per level, ANY sigma_dot with zero
      boundary rows:

          V_flux(X)_k = V_adv(X)_k
                        + (X_k / Dsigma_k)(sigma_dot_{k+1/2} - sigma_dot_{k-1/2})

      — the discrete Leibniz rule d(sigma_dot X)/dsigma =
      sigma_dot dX/dsigma + X d(sigma_dot)/dsigma. It holds with the
      arithmetic interface mean ONLY (any other interface weighting breaks
      it, and with it every identity below).

  (B) Exact column conservation of the flux form:
      `Sum_k Dsigma_k V_flux(X)_k` telescopes to the boundary fluxes,
      which are structurally zero.

  (C) Constant-field compatibility with continuity: `V_adv(c) = 0`
      BITWISE (differences of equal values), and via the Section-6 layer
      mass budget

          V_flux(c)_k = -c (G_k + d(ln p_s)/dt) ,

      i.e. flux-form transport of a constant tracer reduces exactly to
      the discrete continuity equation — no spurious tracer source.

  (SBP) Mass-weighted summation-by-parts against the CONTINUITY-CONSISTENT
      sigma_dot (i.e. sigma_dot = interface_sigma_dot(G), Section 6):

          <X, V_adv(Y)> + <Y, V_adv(X)>
              = Sum_k Dsigma_k (X Y)_k (G_k + d(ln p_s)/dt) .        (SBP)

      Proof sketch: both advection sums shift onto interior interfaces,
      giving Sum sigma_dot_{k+1/2} [(XY)_{k+1} - (XY)_k]; Abel summation
      moves the difference onto sigma_dot, and the layer mass budget
      (Section 6) converts sigma-dot differences into
      -Dsigma_k (G_k + d ln p_s/dt).

      The diagonal X = Y is the KINETIC-ENERGY / SCALAR-VARIANCE EXCHANGE
      RELATION:

          2 <X, V_adv(X)> = Sum_k Dsigma_k X_k^2 (G_k + d(ln p_s)/dt) ,

      the discrete analogue of X sigma_dot dX/dsigma =
      (sigma_dot / 2) dX^2/dsigma. Its meaning: centered vertical
      advection produces quadratic content (KE for X in {u, v}, variance
      for T) ONLY through the mass-convergence factor that the p_s
      weighting of the full flux-form equations hands to continuity — it
      cannot create or destroy variance on its own.

Which equation uses which form:

1. Generic scalar and TEMPERATURE: `-V_adv(T)` in the advective-form
   thermodynamic equation. (C) guarantees an isothermal atmosphere feels
   no vertical transport; the diagonal (SBP) governs T-variance.
2. MOMENTUM: `-V_adv(u)`, `-V_adv(v)` applied COMPONENT-WISE to the
   reconstructed grid winds. The diagonal (SBP) with X = u and X = v is
   the KE-consistency statement. Note: u and v are multivalued at the
   poles; this is safe because V_adv acts level-wise at fixed horizontal
   position (no horizontal coupling), and the components enter the
   spectral equations only through the curl/divergence of the assembled
   nonlinear vector (Section 1), never as scalar spectral fields.
3. CURL/DIVERGENCE representation: the vector
   `W_k = (V_adv(u)_k, V_adv(v)_k)` joins the other nonlinear vector
   terms (eta k x V, R_d T' grad ln p_s) BEFORE the curl/divergence is
   taken; the spectral pathway for a general grid vector is a tendency-
   milestone decision (see the handoff document) and is deliberately NOT
   chosen here.

What this does NOT claim: no statement is made about total-energy
conservation of the full assembled tendency (horizontal terms, the
omega/p heating, and the PGF must be combined and tested first —
Section 7b provides the exchange identity, this section the transport
identities; the assembly proof is future work).

### 7b. Energy-conversion term `kappa T omega/p` (RESOLVED; implemented)

Notation: `beta_k = ln(sigma_{k+1/2} / sigma_{k-1/2})` (the Section-4
interface log ratio; `beta_1` is formally infinite and never used),
`P_k = Sum_{j<=k} G_j Dsigma_j` the discrete partial column integral
(`P_0 = 0`), `A_k = V_k . grad(ln p_s)`.

Continuous derivation. Using continuity, `omega/p` reduces to the pure
column form

    omega/p = V . grad(ln p_s) - (1/sigma) Integral_0^sigma G dsigma' .

Hydrostatic balance `dPhi/dsigma = -R_d T / sigma` and integration by parts
give the column-local ENERGY-EXCHANGE IDENTITY (valid at every instant,
before any horizontal integration; the `p_s` mass factor is common to all
terms):

    Integral_0^1 R_d T (omega/p) dsigma
        = Integral_0^1 [ R_d T A - (Phi - Phi_s) G ] dsigma .        (E)

Globally mass-weighted, `-Integral (Phi - Phi_s) G p_s dsigma dA` turns
into `Integral p_s V . grad(Phi)` plus the `Phi_s dp_s/dt` surface
potential-energy term, so (E) is exactly the kinetic <-> total-potential
energy exchange bookkeeping: heating input to `c_p T` equals the column-
local part of the work extracted from kinetic energy by the PGF.

Discrete requirement: (E) must hold EXACTLY (round-off only) with `Phi_k`
the Section-4 Simmons–Burridge geopotential. Substituting
`Phi_k - Phi_s = R_d (Sum_{j>k} beta_j T_j + alpha_k T_k)` into the
discrete right side and swapping the double sum
(`Sum_k Dsigma_k G_k Sum_{j>k} beta_j T_j = Sum_j beta_j T_j P_{j-1}`)
forces the unique choice

    (omega/p)_k = A_k - (beta_k / Dsigma_k) P_{k-1} - alpha_k G_k ,     (W)

i.e. the Simmons & Burridge (1981) energy-conserving form. The `k = 1`
beta-term is absent (`P_0 = 0`), so the infinite `beta_1` is never
multiplied by anything — same structural rule as the hydrostatic recursion.
With (W) the discrete identity

    Sum_k Dsigma_k R_d T_k (omega/p)_k
        = Sum_k Dsigma_k [ R_d T_k A_k - (Phi_k - Phi_s) G_k ]        (E_d)

holds per column to round-off, and is enforced by tests via the
`energy_exchange` residual diagnostics.

Properties (tested):

* For level-independent `G_k = c`, (W) gives `(omega/p)_k = A_k - c`
  EXACTLY for every `k >= 2` (the alpha/beta terms telescope) — the
  continuous value; the top layer gives `A_1 - c ln 2`, the known SB
  top-layer approximation (alpha_1 = ln 2 is fixed by hydrostatic/energy
  consistency, not by pointwise accuracy at k = 1).
* `G == 0` implies `(omega/p)_k = A_k`; a resting atmosphere gives
  exactly zero conversion and zero pressure work.

The heating term entering the thermodynamic equation is
`(kappa T omega/p)_k = kappa * T_k * (omega/p)_k` (K/s), with the SAME
`(omega/p)_k`; this is what makes the future tendency's PGF work and
heating compensate discretely.

## 8. Horizontal pressure-gradient formulation

The PGF in sigma coordinates is the two-term form

    PGF = - grad(Phi) - R_d T grad(ln p_s).

In the divergence equation the potential part appears as `-lap(Phi + E)`
(exact diagonal spectral operation, like `-lap(K + phi)` in the SWE), and
the `R_d T grad(ln p_s)` part joins the nonlinear vector evaluated
pseudo-spectrally on the backend's product grid.

Intended semi-implicit-ready split (deferred): `T = T_ref(sigma) + T'` moves
`R_d T_ref grad(ln p_s)` into the linear part, giving
`-lap(Phi + E + R_d T_ref ln p_s)` with only `R_d T' grad(ln p_s)` treated
pseudo-spectrally.

Known risk (accepted, mitigated by starting with `Phi_s = 0`): over steep
topography the two PGF terms are large and opposing, and their truncation
errors do not cancel (the classic sigma-coordinate PGF error). This is a
non-issue while `Phi_s = 0`; it must be re-examined before topography is
enabled (Section 12).

## 9. Expected invariants and validation rules

Hard validation (raise `PrimitiveEquationsStateError`; checked on the
initial state and, once a tendency exists, after every accepted step and on
every intermediate RK stage, exactly like the SWE):

1. finiteness of every spectral coefficient (NaN/Inf anywhere is fatal);
2. array layout: shape `(3K+1, l_max+1, l_max+1)`, complex;
3. `zeta_k` monopole = 0 for every level (global circulation of a
   single-valued velocity field is identically zero);
4. `delta_k` monopole = 0 for every level (global integral of a divergence);
   both monopole checks are relative to the field norm with the SWE's
   `1e-10` tolerance;
5. `T > 0` strictly, on **every sampling the model evaluates on** (state
   grid and product grid — the SWE positivity-envelope precedent);
6. `ln p_s` synthesizes to finite grid values and `p_s = exp(ln p_s)` is
   finite (positivity is automatic) on the same samplings.

Monitored invariants (diagnostics, not hard failures):

* total dry mass  `M ∝ <p_s> = area-mean of exp(ln p_s)` — quadrature mean,
  expected to drift at spectral-truncation level (the accepted `ln p_s`
  cost; drift is the monitored quantity);
* column mass closure — the Section 6 layer-budget residual, expected at
  round-off always;
* column energy-exchange closure — the Section 7b identity (E_d) residual
  (conversion minus column-local pressure work), expected at round-off
  always;
* vertical-transport closure — the Section 7a (SBP) residual and the
  flux-form column sum (B), expected at round-off always;
* global dry total energy `Integral (c_p T + Phi_s + E) dm` and global
  angular momentum — defined with the tendency milestone;
* `sigma_dot_{1/2} = sigma_dot_{K+1/2} = 0` exactly (tested, and structural).

## 10. Characteristic-speed / CFL strategy

The engine's model-independent controller
(`run.engine.advective_cfl_timestep`) consumes one scalar per accepted
step. The PE model will supply

    c_max = max_k max_grid |V_k|  +  sqrt( gamma_d * R_d * T_max )

with `T_max` the temperature maximum over the state and product samplings.
`sqrt(gamma_d R_d T)` is an upper bound on the Lamb/external gravity-wave
speed (~347 m/s at 300 K), which is the fastest signal of the explicit
hydrostatic system; the sum-of-maxima form is deliberately conservative,
matching the SWE's `max|u| + sqrt(max Phi)` philosophy. Consequence
(accepted): explicit integration is gravity-wave-limited (dt roughly 3–4x
smaller than an SWE run at equal resolution); semi-implicit treatment of
the linear gravity-wave terms is the documented escape hatch, deferred.

## 11. Integration with the engine and run-capsule architecture

* **State = one complex array** `(3K+1, l_max+1, l_max+1)`, rows ordered
  `[zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, ln p_s]` (top to bottom).
  RK4 stage arithmetic is then the plain array expression
  `run.engine.rk4_step_array` already implements, including
  `stage_validator` hooks.
* **Scheduler/CFL**: unchanged. The runner (future) mirrors
  `run/swe/runner.py`: `IntegrationScheduler` + `integrate` with
  `on_step` returning `c_max` from Section 10.
* **Run capsules**: a future `PERunConfig` mirrors `SWERunConfig`
  (`solver: "pe"` in `to_run_config_dict()`, plus `nlev`,
  `sigma_interfaces`, `t_ref_profile` when it exists); `make_run_id`,
  `config.json`/`manifest.json`, and the shared `cli/run_lifecycle.py`
  machinery are reused as-is. The sigma-interface list is part of run
  identity (it changes the science), exactly like `snapshot_times`.
* **CLI**: `aeolus run pe` is **not** added in this milestone — there is
  nothing to run without a tendency. Dispatch will follow the existing
  `run bve` / `run swe` pattern.
* **Backends**: all horizontal machinery is per-level reuse of the existing
  seams — per-level Helmholtz solves and derivative synthesis use
  `SpectralOperators` / `backend.product_space`, so both the geodesic and
  Gauss lat-lon backends work unchanged.

## 12. Deliberately deferred features and known numerical risks

Deferred (in intended order):

1. nonlinear tendency (separate commit series, only after this foundation's
   tests pass);
2. semi-implicit gravity-wave treatment and the `T_ref(sigma)` profile;
3. scale-selective hyperdiffusion (same `nabla^4` machinery as the SWE);
4. topography (`Phi_s != 0`) and the PGF-error re-examination;
5. Held–Suarez forcing, any moisture/radiation/convection/drag;
6. mass fixer for the `ln p_s` drift.

(The energy-conserving discrete `omega/p` and the vertical-transport
operators, deferred when this document was first written, are now RESOLVED
and implemented — Sections 7b and 7a. Still open within the tendency
milestone: the spectral curl/divergence pathway for assembled nonlinear
grid vectors — see docs/PRIMITIVE_EQUATIONS_TENDENCY_HANDOFF.md.)

Known numerical risks (accepted and recorded, not hidden):

* **Lorenz-grid computational mode** in the vertical `T` structure; visible
  as 2-grid-interval vertical noise in long runs; mitigations (vertical
  diffusion, Charney–Phillips) deferred.
* **Sigma-coordinate PGF error** over topography (Section 8).
* **`ln p_s` mass drift** (Sections 2, 9).
* **Explicit gravity-wave dt limit** (Section 10) — on the MX110-class GPU
  this bounds practical run lengths until the semi-implicit step exists.
* **Vertical resolution/truncation interaction** is unquantified: no
  convergence study exists yet relating `K` and `l_max`; the first
  linearized normal-mode tests must establish it.

## References

* Bourke, W. (1974). A multi-level spectral model. I. Formulation and
  hemispheric integrations. *Mon. Wea. Rev.*, 102, 687–701.
* Hoskins, B. J., & Simmons, A. J. (1975). A multi-layer spectral model and
  the semi-implicit method. *Quart. J. Roy. Meteor. Soc.*, 101, 637–655.
* Simmons, A. J., & Burridge, D. M. (1981). An energy and angular-momentum
  conserving vertical finite-difference scheme and hybrid vertical
  coordinates. *Mon. Wea. Rev.*, 109, 758–766.
