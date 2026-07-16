# Architecture

This document describes how Aeolus is put together today: package layout, the
two grid backends, the spectral transform flow, the run-capsule/provenance
model, and how to add or compare a backend.

> This file reflects the current backend/product-space design. The mermaid
> [class diagram](class-structure.md) and [call diagram](call-structure.md) are
> also current; equation conventions and normalizations live in
> [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md). Dated audit records
> ([KNOWN_RISKS.md](KNOWN_RISKS.md), [VALIDATION_PLAN.md](VALIDATION_PLAN.md))
> carry their own snapshot commits — prefer this file and the current tests
> where they disagree.

## Package layout

```text
src/planetary_sandbox/
├── numerics/        grids, transforms, backends, product spaces, operators
├── run/bve/         equation, RK4 runner, run config resolution, ICs, diagnostics, I/O
├── cli/             aeolus (main.py); psx-bve/psx-gen/psx-recompile compatibility entry points
├── planet/          planet assembly and decorative terrain
└── viz/             maps and run visualizations
tests/               asserting GPU tests plus standalone audit scripts
docs/                architecture, validation records, and tracked README assets
```

Numerical conventions matter more than style (there is no configured formatter
or linter yet): use SI units; keep live arrays on the GPU as CuPy arrays;
preserve the `m >= 0` coefficient convention; put grid-specific decisions behind
`GridGeometry`/`SphericalGridBackend`; reject unsupported product modes instead
of silently falling back; and add asserting backend-parity tests for numerical
changes.

## Spectral transform flow

The dependency direction is deliberately narrow:

```text
GridGeometry
    +
SphericalHarmonicTransform
    +
SphericalGridBackend
    +
ProductSpace
    →
SpectralOperators
    →
BarotropicVorticity
    →
Runner / Diagnostics
```

- `GridGeometry` owns points, areas/weights, shape, and the geometry-specific
  CFL length proxy.
- The transform maps between backend sampling and the shared dense `(l,m)`
  coefficient layout.
- `SphericalGridBackend` pairs geometry and transform, caches a `ProductSpace`,
  and is the sole authority on nonlinear-product sampling.
- `SpectralOperators` contains Laplacian, derivative recurrence, velocity, and
  pseudospectral Jacobian operations without branching on grid family.
- `BarotropicVorticity` owns the equation and exact spectral Coriolis mode; the
  runner owns RK4, snapshot storage, capsules, and diagnostics.
- `run/bve/config.py` (`BVERunConfig`) owns configuration resolution for the
  CLI: preset layering (explicit flag > preset > ordinary default), snapshot
  scheduling, and plot selection — all validated before CUDA initialization.
  The runner consumes the resolved explicit snapshot schedule and clips
  integration steps to land exactly on the requested output times.

The runner selects its execution semantics from an explicit `snapshot_mode`
passed at its boundary (never inferred from whether a schedule was supplied):

- **Count mode** (`--n-snapshots`, canonical) consumes the authoritative
  `snapshot_times` with **exact target-time** semantics. Every scheduled time,
  including `t_end` for `N >= 1`, is landed on exactly by clipping the final
  step and snapping off floating-point drift; a positive pre-target residual is
  always integrated, never absorbed by a tolerance. So the stored snapshot
  times and the final diagnostic time equal the requested targets exactly —
  guaranteed even for deliberately non-aligned durations such as
  `t_end = 600.0000003 s`.
- **Interval mode** (`--snapshot-interval-seconds` / legacy `psx-bve`)
  reproduces the historical countdown bit-for-bit: it stores `t = 0` and each
  interval boundary, and stops within the legacy `1e-6 * dt_snapshots`
  tolerance — so the final state is stored only when the duration is a multiple
  of the interval. This historical stopping behavior is preserved intentionally
  and is *not* replaced by the exact-`t_end` guarantee.

The prognostic variable is relative vorticity, represented by complex
spherical-harmonic coefficients for `m >= 0`:

```text
∂ζ/∂t + J(ψ, ζ + f) = ν∇²ζ + F
∇²ψ = ζ
f = 2Ω sin φ
u = k × ∇ψ
```

The CLI uses classical explicit RK4, exact spectral Laplacian eigenvalues,
and no forcing (`F=0`). The timestep policy is a **fixed CFL ceiling**
computed once from the initial velocity and the geometry-owned CFL length
scale; individual steps are shortened only to land exactly on a requested
snapshot time or on `t_end`, never lengthened. SI units and a perfect
spherical surface are used throughout. Equation conventions and
normalizations are detailed in
[MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md).

User-facing configuration is fully validated *before* CuPy is imported or
CUDA is initialized: the CLI parses with all defaults set to `None`, and
[`BVERunConfig.resolve`](../src/planetary_sandbox/run/bve/config.py) layers
explicit values over the selected preset over ordinary defaults, checks
finiteness/domains, and resolves the snapshot schedule and plot selection.
Ordinary user errors (invalid backend, negative viscosity, NaN duration,
misaligned `--plot summary --n-snapshots 0`) therefore fail with a clear
parser message instead of a mid-run CUDA traceback.

## Backends

Both backends produce the same coefficient layout and enter the BVE through
`SphericalGridBackend`. Keeping both exposes grid-orientation and quadrature
errors that a single implementation could hide.

- The **geodesic backend** exercises arbitrary point-set transforms on a
  quasi-uniform icosahedral mesh. It is the path toward geometry-independent
  spherical numerics and avoids the conceptual pole concentration of a
  structured grid.
- The **Gauss lat–lon backend** provides a mathematically controlled reference:
  tensor-product Gauss–Legendre × periodic-longitude quadrature gives
  floating-point-exact analysis for adequately band-limited fields and exact
  quadratic-product projection at the documented dimensions. It is also much
  faster and smaller at the present dense-transform resolutions.

The measured quality gap between the two is documented in
[VALIDATION.md](VALIDATION.md); the Gauss backend is the stronger quadrature
reference and the geodesic backend is experimental.

### Nonlinear product quadrature

Backend-owned `ProductSpace` sampling handles the nonlinear term. The
distinction between the backends is scientifically important:

- **Geodesic `fine`:** synthesize derivatives on a resolution-`(r+1)` geodesic
  co-grid, form the pointwise Jacobian, analyze with Voronoi weights, and then
  apply the spectral cut. This is an empirically useful **overresolved product
  quadrature**, not mathematically exact dealiasing.
- **Gauss lat–lon `fine`:** a product of two degree-`L` fields has degree at
  most `2L`; projecting it against harmonics through degree `L` requires
  integrating degree `3L`. The periodic longitude rule therefore needs
  `nlon >= 3L + 1`, while Gauss–Legendre exactness through degree
  `2*nlat - 1` requires `nlat >= ceil((3L + 1)/2)`. The code uses at least
  those sizes (or retains a larger state grid), making quadratic-product
  projection exact up to floating-point error for band-limited inputs.

For the state transform alone, the corresponding requirements are
`nlat >= L + 1` and `nlon >= 2L + 1`. The code warns rather than aborts when a
state grid is under-resolved.

## Output capsules and provenance

A current run capsule looks like this:

```text
runs/
├── latest_run.txt
└── <run-id>/
    ├── manifest.json
    ├── config.json
    ├── diagnostics/
    │   ├── timeseries.csv
    │   └── spectra.npz
    ├── figures/
    │   ├── invariant_drift.png
    │   ├── cfl_history.png
    │   ├── spectral_health.png
    │   └── spectra.png
    ├── vorticity_coeffs.npy
    ├── vorticity_grid.npy
    ├── bve_summary.png
    └── <scenario>_t<times>.png
```

The intended semantic categories are:

```text
runs/
└── <run-id>/
    ├── manifest.json
    ├── config.json
    ├── diagnostics/
    ├── states/       # currently the two vorticity .npy files at capsule root
    ├── figures/      # diagnostic figures; viewer figures are also at root
    └── logs/         # reserved; CLI stdout is not persisted yet
```

Which figures exist depends on the run's plot selection (`--plot` /
`--no-plots`): `figures/` comes from the `diagnostics` plot product, the
`<scenario>_t<times>.png` panel from `snapshots`, and `bve_summary.png` from
`summary`. The `.npy` state files and `diagnostics/` data are written
regardless of plot selection (and are empty-but-present when
`--n-snapshots 0` stores no field states).

`config.json` is the authoritative model/CLI configuration: the historical
psx-bve key set plus the additive keys `snapshot_mode`, `n_snapshots`,
`snapshot_times` (the resolved schedule in seconds), and `plots`.
`dt_snapshots` remains the uniform interval where one exists (interval mode,
or count mode with `N >= 2`) and is `null` for `N` in {0, 1}. `manifest.json`
adds the exact command, UTC creation time, run **status**
(`running` → `completed` or `failed`, with a concise error record on
failure), Git commit/branch/dirty flag, Python and library versions, GPU,
transform, state sampling, and actual product sampling.

Run ids encode UTC timestamp, scenario, rotation state, resolution, l_max,
and a snapshot tag (`dtNh` where a uniform interval exists, `snapN` for
count modes with `N` in {0, 1}). Legacy interval-mode runs keep the
historical run-id form byte-for-byte. New canonical (count-mode) runs also
append a short 8-hex-character deterministic hash of the resolved
scientific configuration (backend, dimensions, duration, viscosity,
quadrature, snapshot schedule); purely locational fields (`out`,
`experiment`, `overwrite`) and derived artifacts (`plots`) are excluded
from the hash. Two runs that differ only in output location share a hash;
two runs that differ scientifically are strongly disambiguated by it (the
digest is 32-bit, so a same-second collision between distinct scientific
configurations is very unlikely, not impossible). Figures embed
run metadata; `--overwrite` reuses a directory but first removes known
generated artifacts so a stale plot selection cannot survive alongside a
new run configuration.

The authoritative scientific diagnostics are `diagnostics/timeseries.csv` (one
flushed row per accepted step, including energy, relative and absolute
enstrophy, circulation, CFL, high-degree content, and periodic transform
residuals) and `diagnostics/spectra.npz` (degree spectra). The viewer summary is
useful for visual inspection but is not the scientific invariant record.
`vorticity_coeffs.npy` is the saved spectral state; `vorticity_grid.npy`
contains plotting snapshots.

A capsule is reproducible only to the extent recorded by its manifest. Runs from
a dirty tree require the uncommitted patch as well as the commit, and the
current `random_low_l` scenario does not record an RNG seed. Treat those as
explicit exceptions, not bitwise-reproducible experiments.

Diagnostic plots can be regenerated from the saved authoritative CSV/NPZ data
without rerunning the model:

```powershell
python -c "from pathlib import Path; from planetary_sandbox.run.bve.diagnostics import plot_diagnostics; r=Path('runs'); plot_diagnostics(r/(r/'latest_run.txt').read_text().strip())"
```

## How to add or compare a backend

1. Implement `GridGeometry` for coordinates, point count, areas/weights,
   optional structured shape, and a documented CFL length scale.
2. Provide a transform with `transform`, `inv_transform`, `weights`, and
   `l_max`, producing the shared dense coefficient layout.
3. Subclass `SphericalGridBackend`; define supported quadrature names and
   construct/cache each `ProductSpace` with a provenance label.
4. Register the pairing in `make_backend` and in `Planet.generate`/the CLI if it
   is a user-facing grid.
5. Run the same transform, Jacobian, velocity, diagnostics, RH4, provenance, and
   end-to-end tests used for both current backends.

### Tests

```powershell
pytest
```

The suite requires a working CUDA GPU. On a Windows machine whose global pytest
temp directory has stale ACLs, use a workspace-local directory:

```powershell
pytest --basetemp .pytest-tmp
```

`requirements-dev.txt` includes the runtime pins, an editable package install,
pytest, and ipykernel.

### Benchmarks and audits

These scripts are intentionally outside normal pytest collection and write their
artifacts beneath ignored `runs/` directories:

```powershell
python tests/audit_r3_product.py res4
python tests/audit_r3_product.py res5
python tests/audit_r4_convergence.py
python tests/audit_r5_mechanism.py
```

`audit_r3_product.py res5` and fine-product configurations can require much more
GPU memory than the default run. Read each script's header before running it.
The README figures can be reproduced from run capsules with
[`readme_figures.py`](readme_figures.py); their portable scientific provenance is
tracked separately from the ignored raw runs.
