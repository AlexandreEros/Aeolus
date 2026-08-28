# Williamson test case 5 — Aeolus vs the MRI-JMA reference (2026-07-30)

**Status: accepted.** Aeolus integrates the corrected canonical Williamson-5
initial-value problem, passes the day-zero physical contract at T42 and T63,
completes 15 simulated days at both resolutions with excellent mass and energy
conservation, and tracks a published high-resolution reference model to
sub-metre free-surface RMS at day 0 and a few metres at day 15.

This is a **numerical-model intercomparison**, not a comparison against an
analytic truth solution: Williamson case 5 has no closed-form solution, so the
reference is another model's high-resolution output, itself carrying
discretization error. The numbers below bound the *difference between two
models*, not Aeolus's error.

---

## 1. Benchmark and reference

**Benchmark.** Williamson et al. (1992) test case 5 — zonal flow
`u = u0 cos(lat)`, `u0 = 20 m/s`, over the canonical isolated **conical**
mountain (`hs0 = 2000 m`, `R0 = pi/9`, centred 30°N / 90°W), with the case-2
height field prescribed as the **free surface**, `eta = h0 - (C/g) sin^2(lat)`,
`h0 = 5960 m`, `C = a*Omega*u0 + u0^2/2`. Aeolus's implementation, constants,
and the cone's coordinate-plane distance convention are documented in
[SHALLOW_WATER.md](../SHALLOW_WATER.md).

**Reference.** The Meteorological Research Institute / Japan Meteorological
Agency archive accompanying Yoshimura (2022), directory
`Williamson5/N959_1920x960/sh` (spherical-harmonic model output, 1920 × 960
Gaussian grid, `nmax = 958`), published at
`https://climate.mri-jma.go.jp/pub/archives/Yoshimura_DFS_SW_Testcase`.

- Yoshimura, H. (2022). *Improved double Fourier series on a sphere and its
  application to a semi-implicit semi-Lagrangian shallow water model.*
  Geosci. Model Dev. **15**, 2561–2597. doi:10.5194/gmd-15-2561-2022
- Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., Swarztrauber, P. N.
  (1992). *A standard test set for numerical approximations to the shallow
  water equations in spherical geometry.* J. Comput. Phys. **102**, 211–224.

The archived variable `h` was **established to be the free-surface height**
(not the layer depth) before any comparison was run: it matches the case-2
balanced surface at day 0 to `2.4e-4 m`, and the model's own logged global mean
mass equals `mean(h - h_s)` to `4e-4 m`. That determination — and the discovery
that Aeolus's pre-2026-07-29 initial condition prescribed the same field as
*thickness*, i.e. a physically different initial-value problem — is recorded in
[notebooks/W5_MRI_SEMANTIC_AUDIT.md](../../notebooks/W5_MRI_SEMANTIC_AUDIT.md).
The correction is commit
[`668e6c9a`](https://github.com/AlexandreEros/Aeolus/commit/668e6c9a5735b2d7200dcc121eb0f7c40450ae08),
*fix(swe): canonicalize Williamson-5 initial state*.

**Field contract.** Both sides are reduced to the same physical fields on the
same Gauss–Legendre grids before any metric is computed; nothing is inferred
from numerical agreement.

```
MRI:     free_surface_height = archived h
         layer_depth         = h - topography_height   (analytic cone)
Aeolus:  layer_depth         = (Phi0 + phi) / g
         free_surface_height = (Phi0 + phi + phi_s) / g
         topography_height   = phi_s / g               (band-limited cone)
```

Metrics are area-weighted with the Gauss–Legendre quadrature weights: `wRMS` is
the weighted root-mean-square difference, `rel L2` the weighted relative L2 norm
against the reference field, and `max abs` the pointwise maximum absolute
difference.

## 2. Aeolus configuration

Both runs use the **exact same clean commit** and differ only in resolution.

| | T42 | T63 |
|---|---|---|
| Git commit | `668e6c9a5735b2d7200dcc121eb0f7c40450ae08` | same |
| Worktree | clean (`git.dirty: false`) | clean |
| Backend | `gauss-latlon` (`GaussLatLonGridGeometry`) | same |
| State grid | `nlat = 64`, `nlon = 128` | `nlat = 96`, `nlon = 192` |
| Truncation | `l_max = 42` | `l_max = 63` |
| Duration | 15 days, snapshots at days 0/5/10/15 | same |
| Mean depth `H` | `5619.92594380121 m` | same |
| Product quadrature | `fine` (3/2 rule) | same |
| Dissipation | none — inviscid, no hyperdiffusion | same |
| Timestep | state-adaptive CFL ceiling; 2363 accepted steps | 3535 accepted steps |
| Cone projection residual | `0.025` | `0.012` |
| Run status | `completed` | `completed` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition | same |
| Environment | Python 3.12.13, NumPy 2.0.2, SciPy 1.16.3, CuPy 14.0.1 | same |

Command lines, verbatim from the manifests:

```bash
aeolus run swe --scenario williamson5 --backend gauss-latlon \
    --nlat 64 --nlon 128 --l-max 42 --days 15 --n-snapshots 4 --no-plots
```

```bash
aeolus run swe --scenario williamson5 --backend gauss-latlon \
    --nlat 96 --nlon 192 --l-max 63 --days 15 --n-snapshots 4 --no-plots
```

Full manifests:
[aeolus_t42_run_manifest.json](williamson5_mri_2026-07-30/aeolus_t42_run_manifest.json),
[aeolus_t63_run_manifest.json](williamson5_mri_2026-07-30/aeolus_t63_run_manifest.json).

## 3. Day-zero physical contract

Tolerances were **declared before** the comparison and gate the 15-day runs: the
notebook refuses to start an integration unless the contract passes.

| Quantity | Tolerance | T42 | T63 |
|---|---|---|---|
| wind `max abs` | `< 0.05 m/s` | `2.60e-05` | `2.72e-05` |
| height `max abs` | `< 5.0 m` | `0.0110` | `0.00652` |
| height `rel L2` | `< 5.0e-4` | `1.44e-06` | `8.37e-07` |

**Verdict: passed at both resolutions** (`contract_passed: true`), by three to
four orders of magnitude on every channel. Full record:
[day0_contract.json](williamson5_mri_2026-07-30/day0_contract.json).

The residual free-surface offset (`-0.011 m` at T42, `+0.0065 m` at T63) is the
terrain-monopole quadrature error of the band-limited cone, independently
predicted in `SHALLOW_WATER.md` as `-0.0086 m` at `l_max=42` and `+0.0041 m` at
`l_max=63` — same sign, same order.

The initial meridional wind agrees to `2e-17 m/s`: both models start from
`v = 0` to floating-point roundoff.

![Day-zero T42 comparison](../assets/validation/w5_mri/compare_t42_day00.png)

*Day 0, T42. Rows: free surface, layer depth, wind speed; columns: MRI, Aeolus,
signed difference. The free-surface and wind differences are at the `1e-2 m` and
`1e-5 m/s` level. The only visible structure is the ringing confined to the cone
in the layer-depth difference — Gibbs oscillation of the band-limited conical
terrain, not a dynamical discrepancy.*

## 4. Fifteen-day completion and conservation

Both runs reached `t = 1 296 000 s` (15 days) with `status: completed` and no
non-finite state. Diagnostics are computed at every accepted step; the values
below are derived from the per-step `diagnostics/timeseries.csv` in each capsule
([conservation_summary.json](williamson5_mri_2026-07-30/conservation_summary.json)).

| Diagnostic | T42 | T63 |
|---|---|---|
| Accepted steps | 2363 | 3535 |
| Relative **mass** drift (15 d) | `0.0` (bit-identical) | `0.0` (bit-identical) |
| Relative mass span over run | `0.0` | `0.0` |
| Relative **energy** drift (15 d) | `+3.36e-07` | `-7.92e-07` |
| Relative energy span over run | `4.56e-07` | `7.94e-07` |
| Timestep range | 109.8 – 587.6 s | 134.5 – 392.8 s |
| Maximum wind over run | `42.66 m/s` | `42.76 m/s` |
| Minimum layer depth over run | `3825.2 m` | `3761.5 m` |

Total mass is conserved to the last bit of the `float64` accumulator over
thousands of steps, and total energy drifts by under one part in `10^6` on an
**inviscid** run with no hyperdiffusion and no numerical damping of any kind.

## 5. Comparison against the reference

All values from
[comparison_metrics.csv](williamson5_mri_2026-07-30/comparison_metrics.csv).

| Day | Field | T42 wRMS | T42 rel L2 | T42 max abs | T63 wRMS | T63 rel L2 | T63 max abs |
|---|---|---|---|---|---|---|---|
| 0 | free surface `eta` [m] | 8.130e-03 | 1.440e-06 | 0.01099 | 4.727e-03 | 8.373e-07 | 6.520e-03 |
| 0 | zonal wind `u` [m/s] | 1.593e-05 | 9.753e-07 | 2.604e-05 | 1.600e-05 | 9.798e-07 | 2.724e-05 |
| 0 | meridional wind `v` [m/s] | 5.104e-19 | n/a | 2.012e-17 | 6.255e-19 | n/a | 4.462e-17 |
| 0 | raw depth `h*` [m] † | 3.293 | 5.851e-04 | 87.37 | 1.606 | 2.854e-04 | 36.49 |
| 5 | free surface `eta` [m] | 3.687 | 6.533e-04 | 27.56 | 3.669 | 6.501e-04 | 28.6 |
| 5 | zonal wind `u` [m/s] | 0.1475 | 9.213e-03 | 1.24 | 0.07969 | 4.976e-03 | 0.5748 |
| 5 | meridional wind `v` [m/s] | 0.1127 | 0.02839 | 1.005 | 0.0641 | 0.01615 | 0.406 |
| 5 | raw depth `h*` [m] † | 4.948 | 8.791e-04 | 88.67 | 4.007 | 7.119e-04 | 37.21 |
| 10 | free surface `eta` [m] | 3.967 | 7.029e-04 | 19.07 | 3.72 | 6.591e-04 | 18.74 |
| 10 | zonal wind `u` [m/s] | 0.2868 | 0.01815 | 1.478 | 0.1929 | 0.01221 | 0.9764 |
| 10 | meridional wind `v` [m/s] | 0.1911 | 0.03334 | 1.286 | 0.1079 | 0.01883 | 0.6569 |
| 10 | raw depth `h*` [m] † | 5.163 | 9.173e-04 | 88.3 | 4.054 | 7.203e-04 | 36.65 |
| 15 | free surface `eta` [m] | 5.571 | 9.873e-04 | 28.57 | 4.673 | 8.280e-04 | 27.77 |
| 15 | zonal wind `u` [m/s] | 0.444 | 0.02825 | 2.087 | 0.214 | 0.01362 | 1.051 |
| 15 | meridional wind `v` [m/s] | 0.3371 | 0.04747 | 1.626 | 0.1431 | 0.02015 | 0.7095 |
| 15 | raw depth `h*` [m] † | 6.476 | 1.151e-03 | 71.95 | 4.943 | 8.783e-04 | 32.89 |

† **The raw `layer_depth` row is terrain-representation-dominated and is not a
dynamical error measure.** It is retained for transparency; read § 6.

At day 15 the free-surface fields agree to a weighted RMS of **5.6 m (T42)** and
**4.7 m (T63)** on a `~5620 m` layer — a relative L2 of `9.9e-4` and `8.3e-4`.
The winds agree to a weighted RMS of `0.44 m/s` and `0.21 m/s` in `u` against
peak winds above `40 m/s`.

![Day-15 T63 comparison](../assets/validation/w5_mri/compare_t63_day15.png)

*Day 15, T63. After 15 days of inviscid nonlinear evolution the Rossby wave
train shed by the mountain is reproduced in phase and amplitude; the signed
differences are small-scale and an order of magnitude below the features
themselves.*

![Day-15 T42 comparison](../assets/validation/w5_mri/compare_t42_day15.png)

*Day 15, T42. The same structures at the coarser truncation, with visibly larger
small-scale residuals — consistent with the resolution sensitivity in § 6.*

![Day-zero T63 comparison](../assets/validation/w5_mri/compare_t63_day00.png)

*Day 0, T63. The terrain-representation ring is roughly halved relative to T42
(`87.4 m` → `36.5 m` peak), matching the drop in the cone projection residual
from `0.025` to `0.012`.*

## 6. Interpretation

**The large raw `layer_depth` maximum difference is terrain representation, not
dynamics.** Aeolus carries the cone as a **band-limited** spectral field; the
reference carries the **analytic** cone evaluated at grid nodes. Writing
`terr = topo_ref - topo_aeolus`, the depth difference decomposes exactly:

```
(h*_aeolus - h*_ref) - terr  ==  eta_aeolus - eta_ref
```

Verified on the canonical snapshot files, this identity holds to `9.1e-13 m` at
every stored day and both resolutions — floating-point roundoff. The static term
`terr` alone reaches `87.38 m` (T42) and `36.48 m` (T63), which is *precisely*
the day-0 raw `layer_depth` maximum (`87.37 m` / `36.49 m`), because at day 0 the
dynamical part is only `0.011 m` / `0.0065 m`. Measured cone peaks on the state
grid: analytic `1930.14 m` / `1922.33 m`; Aeolus band-limited `1842.76 m` /
`1885.84 m`, with Gibbs undershoot to `-19.56 m` / `-13.10 m`.

Consequently:

- **Use the free-surface height `eta` for the dynamical comparison.** The
  terrain-adjusted depth difference is *identically* the free-surface
  difference, so `eta` already is the terrain-adjusted measure — the day-zero
  contract's `layer_depth_minus_terrain_repr` row and its `free_surface_height`
  row agree to `1e-12 m`, as the identity requires.
- **The raw depth row stays in the table**, labelled, because hiding a metric
  that looks bad is worse than explaining it.

**This is an intercomparison, not an error against truth.** The reference is
MRI's own discrete solution at `nmax = 958`. Nothing here licenses the claim
that Aeolus "matches truth" or reproduces MRI identically; two models started
from the same initial state on the same benchmark stay close for 15 days.

**No convergence order is claimed.** Two resolutions cannot measure a
convergence rate, and the reference is not an exact solution. The T42 → T63
change is reported as a sensitivity, not a rate.

**Wind agreement improves with resolution more strongly than height agreement.**
At day 15, T42 → T63 reduces the weighted RMS difference by a factor of `2.07`
in `u` and `2.36` in `v`, but only `1.19` in the free surface. This is expected
rather than alarming: the height field is dominated by the large-scale balanced
structure, which is already well resolved at T42, while the winds carry the
fine-scale vorticity filaments that the extra truncation band actually resolves.
It is recorded here as an observation, not a defect, and it does not affect
acceptance.

**Non-blocking future work.** None of the following reopens acceptance:

- decomposing the day-15 difference into phase error, amplitude error, and
  small-scale content across additional resolutions;
- a dissipation-sensitivity study (the runs are deliberately inviscid, and the
  SWE CLI exposes no hyperdiffusion);
- a third truncation, which would be needed before any convergence-rate claim;
- comparing against an additional independent reference model.

## 7. Provenance

**Canonical result identity.** The only canonical W5 result is the T42/T63 pair
recorded in the manifests linked above, all reporting `status: completed`,
`git.commit: 668e6c9a5735b2d7200dcc121eb0f7c40450ae08`, and `git.dirty: false`.
Run IDs (the trailing field is the commit short hash):

```
20260730T011622Z_williamson5_rot23p93h_r4_l42_dt120h_a1f9d0a9_668e6c9a
20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a
```

**Notebooks.** Two notebooks reproduce the whole chain:

- [notebooks/mri_w5_reference_preparation.ipynb](../../notebooks/mri_w5_reference_preparation.ipynb)
  — CPU; downloads and hash-verifies the MRI archive, establishes the archived
  field semantics, and freezes the reference packages plus a manifest.
- [notebooks/w5_mri_validation_colab.ipynb](../../notebooks/w5_mri_validation_colab.ipynb)
  — GPU; clones Aeolus at the pinned commit and *asserts* the checkout produced
  it, verifies package hashes and the field contract, evaluates the day-zero
  gate, runs the 15-day integrations only if the gate passed, and postprocesses.
  It pins `AEOLUS_COMMIT = "668e6c9a5735b2d7200dcc121eb0f7c40450ae08"`.

*Honest note on notebook provenance.* The version of Notebook B committed at
`668e6c9a` itself still pinned the earlier commit `580c566a`; the pin was
updated in the working copy actually executed. The notebooks tracked here are
those **executed** versions — corrected pin, plus the browser upload/download
transport cells that carried the packages between the two Colab sessions — with
their bulky execution outputs stripped. They therefore differ from the
historical `668e6c9a` copies, and the historical copies are not described as the
executed ones. The authority for what ran is not the notebook in any case: each
run manifest independently records the executable's argv, resolved
configuration, backend and numerics, library versions, GPU model, and the
clean-worktree Git status at execution time.

**External archive.** The full generated evidence — both 15-day run capsules
with their per-step diagnostics CSVs and spectral coefficients, the smoke run,
the day-0/5/10/15 `npz` field packages, all eight comparison figures, and the
MRI reference packages — is mirrored on Drive and deliberately **not** in Git:

<https://drive.google.com/drive/folders/10RCkvpbkH69rlAp-4SGBT9J9fcn5it1H>

SHA-256 checksums for every external file, including the three upstream MRI
source files, are in
[CHECKSUMS.txt](williamson5_mri_2026-07-30/CHECKSUMS.txt). The reference and
result `npz` hashes there are the ones recorded independently at write time
inside
[mri_reference_manifest.json](williamson5_mri_2026-07-30/mri_reference_manifest.json)
and [aeolus_t42_COMPLETE.json](williamson5_mri_2026-07-30/aeolus_t42_COMPLETE.json)
/ [aeolus_t63_COMPLETE.json](williamson5_mri_2026-07-30/aeolus_t63_COMPLETE.json),
so the archive can be checked against Git without trusting either copy alone.

**Superseded material.** Every W5 trajectory, measured envelope, and acceptance
capsule produced before the 2026-07-29 initial-condition correction is
superseded, including the `archive/w5-mri-v2-invalid` branch (`42bcc38c`). Only
the two runs identified above are canonical.
