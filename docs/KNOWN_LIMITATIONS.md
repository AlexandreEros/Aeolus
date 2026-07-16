# Known Limitations

Aeolus is research software. This page states its current scope honestly and
lists what would have to change before it could claim more.

For the full audit trail — severity ratings, evidence, fix logs, and remaining
risks — see [KNOWN_RISKS.md](KNOWN_RISKS.md); for the forward program see
[VALIDATION_PLAN.md](VALIDATION_PLAN.md). Both began as audit snapshots, so
prefer their dated fix logs and the current asserting tests when an older status
paragraph conflicts with the implementation.

## Single-layer status

Aeolus solves the **non-divergent barotropic vorticity equation** and, since
the `feat/shallow-water` milestone, the **flat-bottom inviscid rotating
shallow-water equations** ([SHALLOW_WATER.md](SHALLOW_WATER.md)). It does
**not** solve the primitive equations or any general-circulation / weather
model, and it has no vertical structure, dynamically active topography,
forcing, moisture, or thermodynamics. Terrain generated elsewhere in the
package is decorative and does not enter the dynamics.

## CUDA / CuPy assumptions

- GPU/CuPy only: importing the active numerical stack and running the tests
  needs CUDA. There is no production CPU fallback or ordinary CPU CI path, and
  no independent CPU dynamical core for cross-checking.
- Dense transforms scale as `O(N L²)` work and approximately
  `16 × N × (L+1)(L+2)/2` bytes for each complex basis matrix. Fine geodesic
  products add a much larger res-`(r+1)` matrix; res-5 state / fine-res-6 cases
  can exceed small GPUs.
- Windows display GPUs can encounter WDDM/TDR timeouts on large kernels.
- The validated environment is CUDA 11.8 with `cupy-cuda11x==13.4.0` on an
  NVIDIA GeForce MX110. CUDA 12 and newer CuPy releases have not been validated
  by this repository's scientific benchmark suite.

## Geodesic quadrature limitations

- The geodesic transform and product quadrature remain approximate; the Gauss
  lat–lon backend is the stronger quadrature reference.
- Geodesic Voronoi quadrature is orientation-dependent. Its res-`(r+1)` product
  grid reduces aliasing but is not exact dealiasing.
- Under a matched band-limited round trip, the geodesic backend carries a
  `≈1×10⁻²` relative L2 residual versus `≈7×10⁻¹⁵` for Gauss — see
  [VALIDATION.md](VALIDATION.md).

## Nonlinear product aliasing / drift

- State/truncation mismatch: the prognostic state retains modes through `L=21`
  while each nonlinear tendency is cut at `l=14`. That is not a consistent
  Galerkin truncation of either band, so energy/enstrophy conservation is not
  guaranteed when nonlinear transfer reaches modes 15–21.
- The advective CFL ceiling is recomputed from every accepted state, so an
  accelerating flow tightens the step instead of eroding its advective margin
  (R-4). However, explicit diffusion (`ν∇²`) still has no separate stability
  check, so that margin remains uncontrolled.

## Reproducibility / hygiene gaps

- Run `logs/` and `states/` subdirectories are semantic targets rather than the
  current on-disk layout; viewer plots and state arrays remain at capsule root.
- `random_low_l` lacks recorded RNG seed provenance, and `step_leapfrog` is
  legacy/dead rather than a supported integration option.

## Before primitive equations (and remaining shallow-water gaps)

The shallow-water core is minimal: flat bottom, inviscid, explicit RK4 (no
semi-implicit gravity-wave treatment), and only Williamson case 2 among the
analytic benchmarks. Moving further requires, at minimum:

- A mathematically controlled reference on the geodesic backend, or a decision
  to build the next core on the Gauss lat–lon backend where quadrature is exact.
- A consistent Galerkin truncation (aligning the state band with the nonlinear
  tendency cut) so conservation is provable rather than merely measured.
- An independent CPU dynamical core (or equivalent cross-check) so results do
  not depend on a single GPU implementation.
- The analytic benchmarks laid out for the next stages —
  Williamson et al. (1992) shallow-water test cases and the
  Galewsky–Scott–Polvani unstable jet — in
  [VALIDATION_PLAN.md](VALIDATION_PLAN.md) (Stages B and C).
