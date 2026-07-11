# Architecture — As Implemented

Audit snapshot: branch `geodesic-grid-refactor`, commit `2a30a18`, 2026-07-11.
Companion documents: [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md) (equations and
conventions), [KNOWN_RISKS.md](KNOWN_RISKS.md) (findings, referenced as R-n),
[VALIDATION_PLAN.md](VALIDATION_PLAN.md).

---

## 1. Package layout

```
src/planetary_sandbox/
├── numerics/                      # spectral engine (GPU-only, CuPy)
│   ├── geodesic_grid.py           # icosahedral mesh, Voronoi weights, adjacency
│   ├── grid.py                    # legacy structured lat-lon grid
│   ├── grid_base.py               # abstract GridGeometry
│   ├── fast_geodesic_sh.py        # PointSetSphericalHarmonics (matrix transform)
│   ├── optimized_geodesic_sh.py   # GeodesicSphericalHarmonics (weights + cache)
│   ├── spherical_harmonics.py     # LatLonSphericalHarmonics (legacy, Simpson)
│   ├── spectral_operators.py      # SpectralOperators (∇², ∂λ, sinθ∂θ, J, u from ψ)
│   ├── differential_operators_spherical.py  # latent local FD operators (unused)
│   ├── grid_interpolation.py      # scipy griddata remapping (viz only)
│   ├── integration.py             # simpson_2d (legacy path only)
│   ├── compute_optimal_weights.py # standalone weight-optimization script
│   └── cuda/                      # RawModule kernels (.cu, compiled at runtime)
│       ├── legendre.cu            #   used by LatLonSphericalHarmonics
│       ├── sph_harm.cu            #   used by LatLonSphericalHarmonics
│       ├── sh_matrix.cu           #   used by PointSetSphericalHarmonics  ← operational
│       └── sh_matrix_real.cu      #   never loaded (dead)
├── planet/
│   ├── planet.py                  # Planet.generate(): grid + SH + operators + terrain
│   ├── planetary_parameters.py    # mass/radius/rotation → Ω, oblateness, mean radius
│   ├── terrain_spectral.py        # random power-law spectral terrain (decorative)
│   ├── tectonics.py               # spectral diffusion + gated noise (disabled in generate)
│   └── elevation_data.py          # container for terrain fields
├── run/bve/
│   ├── barotropic_vorticity.py    # BarotropicVorticity model + BarotropicState
│   ├── runner.py                  # run_bve() loop + rk4_step()
│   ├── initial_conditions.py      # scenario registry (vortex pairs, random, RH4)
│   ├── config.py, io.py           # EMPTY placeholder files
├── cli/
│   ├── bve.py                     # psx-bve entry point
│   ├── generate_planet.py         # psx-gen entry point
│   └── clear_cache.py             # psx-recompile (CuPy kernel cache clear)
├── viz/
│   ├── vorticity_viewer.py        # snapshot/summary plots + run statistics
│   ├── planet_viewer.py, maps.py  # planet/terrain plotting
│   └── spectra.py                 # EMPTY
└── physics/gravity.py             # EMPTY
tests/                             # 3 pytest tests + 3 manual diagnostic scripts
docs/                              # mermaid class/call diagrams (accurate, high level)
```

Empty modules (`config.py`, `io.py`, `gravity.py`, `spectra.py`) and `.bak` files
(`sh_matrixcu.bak`, `fast_geodesic_shpy.bak`) are scaffolding/leftovers (R-17).

## 2. Data flow of a BVE run

```
psx-bve CLI (argparse)
  → PlanetaryParameters.from_earth_like(day_hours, radius)     # Ω, R  (day-hours default: inf ⇒ Ω=0)
  → Planet.generate(params, grid_resolution, l_max)
      → GeodesicGridGeometry(res, R)          # points, faces, Voronoi areas, adjacency
      → GeodesicSphericalHarmonics(grid, l_max)   # weights="voronoi" default
          → PointSetSphericalHarmonics             # builds Y (n_points × n_basis) on GPU
      → SpectralOperators(sh, R, grid)         # eigenvalues, C± recurrence tables
      → generate_spectral_terrain_gpu(...)     # decorative; not used by BVE
  → make_ic(scenario, planet)                  # ζ₀ on grid points
  → planet.sh.transform(ζ₀)                    # ζ₀_lm
  → run_bve(...)
      loop: rk4_step → BarotropicVorticity.tendency
              ├─ ψ_lm = ζ_lm / λ_l
              ├─ ζ grid = inv_transform(ζ_lm); η = ζ + f; η_lm = transform(η)   # lossy round trip (R-5)
              ├─ J = SpectralOperators.jacobian_pseudospectral(ψ_lm, η_lm)      # DEFECTIVE (R-1)
              ├─ advection_lm = transform(−J); diffusion = ν λ_l ζ_lm
              └─ dζ_lm/dt (l=0 row zeroed)
      snapshots → .npy dumps + VorticityViewer plots (geodesic → lat-lon griddata remap)
```

The whole state (Y matrix, coefficients, grid fields) is GPU-resident; only plotting and
`.npy` saving copy to host.

## 3. Class responsibilities and coupling

- `Planet` is a facade bundling `params`, `grid`, `sh`, `so`, `elevation`. The BVE only
  needs `(sh, so, grid, params.radius, params.angular_velocity)` — terrain generation in
  `Planet.generate` is pure startup overhead for BVE runs (it even runs an SH synthesis and
  double-computes radial distances; R-17).
- `GeodesicSphericalHarmonics.__getattr__` transparently forwards unknown attributes to the
  wrapped `PointSetSphericalHarmonics` (e.g. `Y_matrix`, `l_indices`), which is convenient
  but makes the effective API implicit (and recurses infinitely if `self.sh` is missing).
- `SpectralOperators` holds a lazy `DifferentialOperatorsSpherical` factory
  (`_get_differential_ops`) that nothing calls; the local FD operator family and the graph
  Laplacian inside it are latent code (R-16).
- `BarotropicState` is a thin dataclass; `tendency` field unused. Mixed use of
  `BarotropicState` vs raw arrays across methods is what breaks `step_leapfrog` (R-6).

## 4. Execution environments

| Aspect | Status |
|---|---|
| CPU path | **None.** `import cupy` at module scope everywhere; package import fails without CUDA (R-15) |
| GPU precision | float64 / complex128 throughout |
| Kernels | 3 active `.cu` files compiled at runtime via `cp.RawModule` (NVRTC); cache cleared by `psx-recompile` |
| Transform cost | dense GEMV/GEMM: O(n_points · L²/2) per transform; ≈41 ms per RK4 step at res 4 / l_max 20 on MX110 **[measured]** |
| Memory ceiling | Y matrix = 16·n_points·n_basis bytes → res 5/l_max 45 ≈ 177 MB; res 6/l_max 63 ≈ 1.3 GB (exceeds small GPUs) |
| Windows TDR | long kernels on a display GPU can be killed by the WDDM watchdog — observed during the audit (`cudaErrorLaunchTimeout` on a res-5 Gram computation) (R-15) |
| Weight cache | `sh_weights_res{r}_lmax{L}.pkl` under `cache_dir`; keyed by (resolution, l_max, n_points, basis-id) but **not** radius (R-13) |

## 5. Testing and quality infrastructure

- `pytest` collects **3 tests** (grid interpolation round trip, basis orthogonality,
  lat-lon vs geodesic coefficient agreement). All pass **[measured]**, but: the
  orthogonality "test" contains no assertions (prints + returns a tuple); the coefficient
  agreement test accepts up to **60 % relative L2 error**; the interpolation test accepts
  RMSE 0.2 on an O(1) field (R-8).
- `consistency_test.py`, `debug_sh.py`, `speed_showdown.py` are manual scripts with
  module-level execution, not collected tests.
- No CI is possible as-is (GPU required to even import the package).
- No conservation/energy monitoring exists in the run loop; the viewer computes summary
  stats post-hoc with mislabeled units ("J/kg" for an area-integrated specific energy,
  "Enstrophy" for RMS vorticity) (R-14).

## 6. Notable architectural strengths

- Clean separation of grid geometry / transform / operators / model / runner / viz.
- The point-set transform design (arbitrary sampling + explicit quadrature weights) is a
  genuinely flexible abstraction: the same engine serves lat-lon and geodesic sampling, and
  the weight-optimization path shows awareness of the discrete-orthogonality problem.
- Spectral operator tables (Laplacian eigenvalues, ε recurrence) are precomputed once and
  correct (verified — see MATHEMATICAL_MODEL §5).
- Everything is float64 and deterministic (seeded RNG for terrain).

## 7. Main architectural liabilities

1. **GPU-only import graph** blocks CI, portability, and any CPU/GPU cross-check (R-15).
2. **Dense-matrix transform** is O(N·L²) with an O(N·L²) memory footprint — fine at res ≤ 5,
   a wall at res 6+; no separable/FFT path exists for the structured grid alternative.
3. **No enforced compatibility between `l_max` and grid resolution** — the CLI happily runs
   the default (45, res 4) combination that the transform cannot support (R-2).
4. **Dead/duplicated code** obscures the load-bearing 20 %: two grid geometries, three SH
   engines (one defective-legacy, one unused-real-basis kernel), latent FD operators, empty
   modules, commented-out blocks in `planet.py`/`spectral_operators.py` (R-16, R-17).
5. Run loop and model are entangled with plotting (runner constructs the viewer), and
   there is no machine-readable run manifest beyond `config.json` + raw `.npy` dumps.
