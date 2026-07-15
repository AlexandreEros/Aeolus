# Aeolus

**A GPU-resident spectral laboratory for two-dimensional flow on a sphere.**

Aeolus advances the non-divergent barotropic vorticity equation (BVE) with
spherical harmonics, and can run the same model and operators on either an
icosahedral geodesic point set or a Gauss–Legendre latitude–longitude grid. It
is research software for people interested in spherical spectral methods,
backend parity, conservation diagnostics, and reproducible numerical
experiments — **not** a general circulation model.

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
- GPU spherical-harmonic analysis/synthesis in `float64`/`complex128` using
  CuPy, custom CUDA basis kernels, and dense GPU matrix products.
- **Two interchangeable grid backends** — icosahedral geodesic and
  Gauss–Legendre lat–lon — sharing one `(l,m)` coefficient layout, so a run can
  be reproduced on either grid to expose grid-orientation and quadrature errors.
- **Immutable run capsules** carrying command/configuration, Git and GPU
  provenance, per-step diagnostics, spectra, saved states, and plots.
- Rossby–Haurwitz wavenumber-4 (`rh4`) validation and backend-parity tests.

## What Aeolus is not

Aeolus does **not** solve the shallow-water equations, primitive equations, or
any GCM / weather model. It has no vertical structure, divergence,
height/free-surface equation, topographic forcing, moisture, or thermodynamics.
Terrain generated elsewhere in the package is decorative and does not enter the
BVE. See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

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
psx-bve --help
```

If PowerShell blocks venv activation, allow it for the current process only
(`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) and activate
again. See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) for the CUDA/
CuPy compatibility notes.

## Minimal run examples

Short geodesic run:

```powershell
psx-bve --grid geodesic --resolution 3 --lmax 8 --scenario two_vortices --duration-days 0.02 --dt-snapshots 864 --out runs --experiment quickstart-geodesic
```

Short Gauss lat–lon run (the `12 × 24` state grid is adequate for `lmax=8`;
fine products are evaluated on the required `13 × 25` grid):

```powershell
psx-bve --grid latlon --nlat 12 --nlon 24 --lmax 8 --scenario two_vortices --duration-days 0.02 --dt-snapshots 864 --out runs --experiment quickstart-latlon
```

One-day RH4 validation run at the production default envelope:

```powershell
psx-bve --grid geodesic --resolution 4 --lmax 21 --scenario rh4 --day-hours 24 --duration-days 1 --dt-snapshots 21600 --product-quadrature fine --viscosity 0 --out runs --experiment validation-rh4
```

The CLI prints the absolute run directory and updates `runs/latest_run.txt`.
`psx-bve --help` is the complete, current source of truth for options.

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
| GPU test suite (Python 3.12.12, CuPy 13.4.0, MX110) | **105 passed**, one warning |

The five-day geodesic energy number is locked by
`test_prediction_p1_5day_energy_drift`. Full tables, conservation diagnostics,
and orientation/rotation-equivalence tests are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Documentation

- [docs/VALIDATION.md](docs/VALIDATION.md) — RH4 backend comparison,
  conservation diagnostics, geodesic-vs-Gauss discussion, rotation tests.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — package layout, backends,
  spectral transform flow, output capsules/provenance, adding a backend.
- [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) — BVE-only status,
  CUDA/CuPy assumptions, quadrature limits, path to shallow water.
- [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md) — equations and conventions.
- Deeper audit records: [docs/KNOWN_RISKS.md](docs/KNOWN_RISKS.md),
  [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md),
  [docs/validation/](docs/validation/), and the current mermaid
  [class](docs/class-structure.md) / [call](docs/call-structure.md) diagrams.

## Current limitations

- BVE only: no shallow-water or primitive-equation dynamics.
- GPU/CuPy only: no production CPU fallback or CPU CI path.
- The timestep is fixed from initial conditions; the state band extends above
  the nonlinear tendency cut, so conservation is measured, not guaranteed.
- The geodesic transform and product quadrature remain approximate; the Gauss
  backend is the stronger quadrature reference.

Full detail — with severity, evidence, and open items — is in
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md),
[docs/KNOWN_RISKS.md](docs/KNOWN_RISKS.md), and
[docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md).
