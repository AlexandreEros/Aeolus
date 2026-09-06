# Aeolus

**A GPU-resident spectral laboratory for circulation on a rotating sphere.**

A thin layer of fluid on a spinning sphere does not stay smooth. Rotation,
curvature, and the poleward variation of the Coriolis parameter organize it:
energy collects at preferred scales, flow gathers into jets and long-lived
vortices, and disturbances travel as planetary-scale waves. Aeolus is a
numerical laboratory for watching that organization arise from the equations
themselves — from a discretized dynamical core and its spectral transforms,
not from structure imposed on the output.

That makes two kinds of question askable. **What patterns emerge** — which
dominant scales, which harmonic structure, which coherent structures a given
rotation rate, radius, and mean layer depth select. And **how organized flow is
redistributed across scales**: a conservative run can begin in essentially one
low-order mode and, through mode coupling alone, spread its energy over a broad
harmonic spectrum, raising the mean wavenumber and the effective number of
occupied modes, while mass, energy, and potential enstrophy stay controlled to
one part in 10⁵ or better. No explicit dissipation is applied; what broadens dramatically is the coarse-grained spectral description of the flow, while the principal invariants remain nearly unchanged. The figure below measuresexactly that. Aeolus is early, though: varying planetary parameters to see whenthe **spectral character** of a circulation changes — and eventually lettingsuch differences act on the transport of heat, mass, and momentum — needs
models it does not have yet. What exists today is the spectral machinery, the
conservation diagnostics, and the run provenance that would make those
comparisons worth believing.

![Aeolus Williamson Test Case 5 T63 shallow-water validation](docs/validation/williamson_5/overview.png)

*Aeolus solving Williamson Test Case 5 at T63 (96×192 Gauss–Legendre grid, ℓ ≤ 63): an initially axisymmetric zonal flow over an isolated conical mountain sheds a global, mountain-forced wave train over 15 simulated days, while layer mass stays bit-identical to day 0 and total energy and potential enstrophy drift by less than one part in 10⁵. The right-hand column tracks that reorganization spectrally, in the rotational/divergent modes the solver already carries: the flow starts as essentially one low-order mode (mean degree ⟨ℓ⟩ = 1, mean zonal wavenumber ⟨|m|⟩ = 0) and spreads to ⟨ℓ⟩ ≈ 2.7 and about 4.4 effective occupied modes by day 15 — the visual complexity is real dynamics, not a loss of the conserved quantities. [See the Williamson-5 validation evidence and reference comparison →](docs/validation/williamson5_mri_2026-07-30.md)*

Aeolus advances the non-divergent barotropic vorticity equation (BVE) and the
rotating shallow-water equations with spherical harmonics — with an early dry
hydrostatic primitive-equation core beginning to add vertical structure — and
can run the same models and operators on either an icosahedral geodesic point
set or a Gauss–Legendre latitude–longitude grid. It is research software for
people interested in spherical spectral methods, backend parity, conservation
diagnostics, and reproducible numerical experiments — **not** a general
circulation model.

![Two vortices evolving over ten days](docs/assets/two_vortices_evolution.png)

*A visibly evolving BVE run: two compact vortices stretch into filaments and
broader planetary-scale structure over ten days (geodesic res 4, `lmax=21`,
24 h rotation, inviscid). This is a qualitative dynamics showcase — the
controlled conservation evidence is in [docs/VALIDATION.md](docs/VALIDATION.md),
and the full 40-character configuration is in the tracked
[figure provenance](docs/assets/provenance.json).*

## What Aeolus is

- A rotating-sphere **barotropic-vorticity solver** with RK4 and optional
  Laplacian viscosity, prognostic in relative vorticity.
- An inviscid **rotating shallow-water solver** with optional fixed analytic
  bottom topography, prognostic in vorticity, divergence, and perturbation
  thickness geopotential, verified against the linear gravity-wave dispersion
  relation, Williamson test case 2, exact lake-at-rest balance over terrain
  (see [docs/SHALLOW_WATER.md](docs/SHALLOW_WATER.md)), and **Williamson test
  case 5** against an external high-resolution reference model
  ([report](docs/validation/williamson5_mri_2026-07-30.md)).
- An early **dry hydrostatic primitive-equation core** in sigma coordinates
  with a first runnable fixed-step experiment (`aeolus run pe`): exact rest,
  smooth evolution, and analytic orographic balance over fixed band-limited
  terrain are verified, but there is no forcing, moisture, hyperdiffusion,
  adaptive stepping, or energy-conservation claim
  ([docs/PRIMITIVE_EQUATIONS_RUNNER.md](docs/PRIMITIVE_EQUATIONS_RUNNER.md)).
- GPU spherical-harmonic analysis/synthesis in `float64`/`complex128` using
  CuPy, custom CUDA basis kernels, and dense GPU matrix products.
- **Two interchangeable grid backends** — icosahedral geodesic and
  Gauss–Legendre lat–lon — sharing one `(l,m)` coefficient layout, so a run can
  be reproduced on either grid to expose grid-orientation and quadrature errors.
- **Immutable run capsules** carrying command/configuration, Git and GPU
  provenance, per-step diagnostics, spectra, saved states, and plots.
- Rossby–Haurwitz wavenumber-4 (`rh4`) validation and backend-parity tests.

## What Aeolus is not

Aeolus is **not** a GCM or weather model. The primitive-equation core is an
ignition path, not a climate model: no forcing, moisture, hyperdiffusion,
semi-implicit or adaptive stepping, no CFL controller, no total-energy
conservation diagnostic, and nothing longer than short fixed-step
demonstrations. The BVE and shallow-water cores have no vertical structure at
all, and the shallow-water core is inviscid. Every solver takes only fixed
analytic bottom topography — `flat` or one Gaussian mountain, plus the
benchmark-owned cone that the `williamson5` scenario supplies — and the
separate terrain generated by `Planet.generate` remains decorative. See
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## Quick start

Requires Python 3.12, an NVIDIA CUDA-capable GPU, a compatible driver/toolkit,
and Git. Tested on Windows/PowerShell with Python 3.12.12, CuPy 13.4.0, and
CUDA 11.8.

```powershell
git clone https://github.com/AlexandreEros/Aeolus.git
cd Aeolus
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the known-good environment (`cupy-cuda11x==13.4.0` for
CUDA 11.x). For CUDA 12.x, replace that one pin with `cupy-cuda12x==13.4.0`;
install exactly one CuPy package. Confirm the GPU is visible:

```powershell
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
aeolus --help
```

`aeolus` is the canonical command-line interface. The `psx-bve`, `psx-gen`,
and `psx-recompile` commands remain available as compatibility entry points
(see below). Entry points are created at install time, so rerun
`pip install -e .` after pulling a change that touches them.

If PowerShell blocks venv activation, allow it for the current process only
(`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) and activate
again. See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) for the CUDA/
CuPy compatibility notes.

## Minimal run examples

Short two-vortex smoke run (the README quickstart configuration, packaged as
a preset):

```powershell
aeolus run bve --preset two-vortices-quick
```

The same quickstart on the Gauss lat–lon backend (the `12 × 24` state grid is
adequate for `l_max=8`; fine products are evaluated on the required `13 × 25`
grid):

```powershell
aeolus run bve --preset two-vortices-quick --backend gauss-latlon
```

One-day RH4 validation run at the production default envelope:

```powershell
aeolus run bve --preset rh4
```

which is shorthand for:

```powershell
aeolus run bve --backend geodesic --resolution 4 --l-max 21 --scenario rh4 --day-hours 24 --days 1 --snapshot-interval-seconds 21600 --product-quadrature fine --viscosity 0 --experiment validation-rh4
```

Explicit flags always override preset values. The CLI prints the resolved
configuration and the absolute run directory, and updates
`runs/latest_run.txt`. `aeolus run bve --help` is the complete, current
source of truth for options; `aeolus list presets` and
`aeolus list scenarios` enumerate the available presets and initial
conditions, and `aeolus inspect runs` summarizes the latest run capsule.

One-day shallow-water run of Williamson test case 2 (steady nonlinear zonal
geostrophic flow) with default settings, and the same on the Gauss backend:

```powershell
aeolus run swe
aeolus run swe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15
```

`aeolus run swe --help` lists the (deliberately minimal) shallow-water
options — gravity, mean depth, rotation, radius, resolution, duration, and
the snapshot schedule, plus optional fixed Gaussian-mountain topography; the
model and its verification are documented in
[docs/SHALLOW_WATER.md](docs/SHALLOW_WATER.md).

The first runnable dry **primitive-equation** experiment (hydrostatic,
sigma-coordinate, fixed-step RK4; no forcing, diffusion, or semi-implicit
terms) is exposed the same way:

```powershell
aeolus run pe                                  # tiny thermal_wave demo
aeolus run pe --scenario isothermal_rest       # verify the exact-rest property
aeolus run pe --backend gauss-latlon --nlat 32 --nlon 64 --l-max 15
```

`aeolus run pe --help` lists the options — backend/resolution, `--levels` or
explicit `--sigma-interfaces`, the dry gas constants, the initial-condition
preset and its temperature/pressure/amplitude, the **fixed** `--dt-seconds`
step, duration, and the snapshot schedule; the runner is documented in
[docs/PRIMITIVE_EQUATIONS_RUNNER.md](docs/PRIMITIVE_EQUATIONS_RUNNER.md).

### Snapshots and plots

Field-state storage and image generation are controlled independently:

- `--n-snapshots N` (canonical, default `5`) stores `N` evenly spaced states
  **including both `t=0` and `t_end`**. `N=0` stores no field snapshots;
  `N=1` stores only the final state. For the default one-day run, `N=5`
  reproduces the historical 0 h / 6 h / 12 h / 18 h / 24 h states.
- `--snapshot-interval-seconds S` (compatibility alias `--dt-snapshots`)
  keeps the historical interval semantics instead: `t=0` and every interval
  boundary are stored, and the final state is stored **only** when the
  duration is a multiple of the interval. The two controls are mutually
  exclusive. Legacy `psx-bve` invocations default to a 21600 s interval, so
  existing commands behave exactly as before.
- `--plot TYPE` (repeatable: `diagnostics`, `snapshots`, `summary`, or
  `all`) selects which image products to render; `--no-plots` renders none.
  The `snapshots` product is published as one staged directory containing
  `physical/` and `spectral/` frames plus a representative `timeline.png` in
  each representation. Both views use the persisted snapshot time axis and
  the same full-sequence normalization as their complete frame sets. Physical
  frames visibly separate prognostic state from instantaneous diagnostic
  fields; BVE diagnostics include streamfunction and velocity streamlines,
  while SWE derives velocity and `h' = Phi'/g` from each persisted state.
  Spectral coefficient frames use cyclic hue for phase and timeline-wide
  relative amplitude in decibels for saturation (default `[-60, 0] dB`),
  with valid zeros white and the invalid `m > l` triangle gray.
  Field snapshots (`vorticity_coeffs.npy` / `vorticity_grid.npy`), their
  authoritative `bve_snapshot_times.npy` time axis, and the
  per-step numerical diagnostics CSV are always written regardless of plot
  selection, so `--n-snapshots 20 --no-plots` saves twenty states without
  rendering a single figure, and `--n-snapshots 0` still yields a
  diagnostics-only run.

### Compatibility entry points

`psx-bve`, `psx-gen`, and `psx-recompile` delegate to the same
implementations as `aeolus run bve`, `aeolus gen`, and `aeolus recompile`.
Existing option spellings (`--lmax`, `--grid`, `--duration-days`,
`--dt-snapshots`) remain accepted everywhere as aliases of the canonical
names, and legacy interval-based invocations keep their historical run-id
format, `config.json` keys, and output behavior. `config.json` additionally
records `snapshot_mode`, `n_snapshots`, `snapshot_times`, and `plots`
(additive keys only).

## Example outputs

![Geodesic and Gauss lat–lon RH4 comparison](docs/assets/rh4_geodesic_vs_latlon.png)

*One-day RH4 comparison across both backends (commit `4a840226`, `L=21`, 24 h
rotation, inviscid). RH4 is a shape-preserving traveling wave: success means the
pattern translates at the analytic phase speed without deforming. Full figure
discussion and provenance are in [docs/VALIDATION.md](docs/VALIDATION.md).*

## Validation snapshot

> The **Gauss lat–lon backend is the stronger quadrature reference**; the
> **geodesic backend is experimental** — its Voronoi quadrature is approximate
> and orientation-dependent. The numbers below are measurements of the discrete
> solver, not analytic guarantees.

| Evidence | Result |
|---|---|
| RH4 geodesic, 5 days (res-4 state, res-5 fine product grid) | relative energy drift **−4.4555×10⁻⁴** |
| RH4 Gauss lat–lon, matched timestep (`32 × 64`) | energy drift **−1.34×10⁻¹⁰** |
| Transform round trip, `L=21` (geodesic vs Gauss) | relative L2 residual **1.04×10⁻²** vs **6.84×10⁻¹⁵** |
| **Williamson 5 vs MRI-JMA**, day-zero physical contract (T42 and T63) | **passed** — free surface `1.4×10⁻⁶` / `8.4×10⁻⁷` rel L2, winds `≈2.6×10⁻⁵ m/s` |
| **Williamson 5**, 15-day inviscid runs (T42 `64 × 128`, T63 `96 × 192`) | both **completed**; mass drift **0.0** (bit-identical), energy drift **+3.4×10⁻⁷** / **−7.9×10⁻⁷** |
| **Williamson 5**, day-15 free-surface difference vs MRI-JMA | wRMS **5.6 m** (T42) / **4.7 m** (T63) on a `~5620 m` layer |
| Test suite (Python 3.12.12, CuPy 13.4.0, MX110; GPU-guarded tests skipped without CUDA) | **626 passed, 1 skipped** in 318 s (the skip is the env-gated 15-day W5 acceptance run) |

The five-day geodesic energy number is locked by
`test_prediction_p1_5day_energy_drift`. Full tables, conservation diagnostics,
and orientation/rotation-equivalence tests are in
[docs/VALIDATION.md](docs/VALIDATION.md).

### Williamson test case 5

Aeolus integrates the corrected canonical Williamson-5 initial-value problem
(the case-2 height field prescribed as the **free surface**, over the canonical
conical mountain), passed the day-zero physical contract at both T42 and T63,
completed 15-day runs at both resolutions with excellent mass and energy
conservation, and was compared against the high-resolution MRI-JMA/Yoshimura
reference solution. All runs are at commit `668e6c9a` with a clean worktree, on
the Gauss lat–lon backend, inviscid and with no hyperdiffusion.

This is a **numerical-model intercomparison**, not a comparison against an
analytic truth solution — Williamson 5 has none. Aeolus is not claimed to match
truth, to reproduce MRI identically, or to demonstrate a formal convergence
order (two truncations cannot measure one). The large raw `layer_depth`
difference is dominated by Aeolus's band-limited cone versus the reference's
analytic cone, not by dynamics.

Full report, figures, provenance, and checksums:
[**docs/validation/williamson5_mri_2026-07-30.md**](docs/validation/williamson5_mri_2026-07-30.md).

## Documentation

- [docs/VALIDATION.md](docs/VALIDATION.md) — RH4 backend comparison, the
  Williamson-5 intercomparison summary, conservation diagnostics,
  geodesic-vs-Gauss discussion, rotation tests.
- [docs/validation/williamson5_mri_2026-07-30.md](docs/validation/williamson5_mri_2026-07-30.md)
  — the accepted Williamson test case 5 result against the MRI-JMA reference:
  day-zero contract, 15-day T42/T63 conservation, comparison tables, figures,
  and full provenance.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — package layout, backends,
  spectral transform flow, output capsules/provenance, adding a backend.
- [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) — current solver
  scope, CUDA/CuPy assumptions, quadrature limits.
- [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md) — equations and conventions.
- [docs/SHALLOW_WATER.md](docs/SHALLOW_WATER.md) — the rotating shallow-water
  core: prognostics, discretization, CFL, scenarios, verification status.
- [docs/PRIMITIVE_EQUATIONS_DESIGN.md](docs/PRIMITIVE_EQUATIONS_DESIGN.md) and
  [docs/PRIMITIVE_EQUATIONS_RUNNER.md](docs/PRIMITIVE_EQUATIONS_RUNNER.md) —
  the dry hydrostatic primitive-equation core and its first runnable
  fixed-step experiment (presets, capsule schema, diagnostics, results).
- Deeper audit records: [docs/KNOWN_RISKS.md](docs/KNOWN_RISKS.md),
  [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md),
  [docs/validation/](docs/validation/), and the current mermaid
  [class](docs/class-structure.md) / [call](docs/call-structure.md) diagrams.

## Current limitations

- Fixed analytic topography only, in every solver: no time-dependent or
  data-driven terrain, and no forcing anywhere. The BVE and shallow-water
  cores are single-layer; the dry primitive-equation core has vertical
  structure but no moisture and no validated long integrations.
- GPU/CuPy only: no production CPU fallback or CPU CI path.
- The advective CFL ceiling is recomputed from every accepted state (genuine
  state-adaptive advective stepping), but explicit-viscosity (`ν∇²`) stability
  is not controlled, and the state band extends above the nonlinear tendency
  cut, so conservation is measured, not guaranteed.
- The geodesic transform and product quadrature remain approximate; the Gauss
  backend is the stronger quadrature reference.

Full detail — with severity, evidence, and open items — is in
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md),
[docs/KNOWN_RISKS.md](docs/KNOWN_RISKS.md), and
[docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md).
