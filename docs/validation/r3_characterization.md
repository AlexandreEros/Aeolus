# R-3 characterization and fix record — fine-grid product evaluation

Tracked record of the R-3 investigation (KNOWN_RISKS.md R-3), kept in Git because the
raw run artifacts under `runs/` are Git-ignored. Every number below is reproducible from
the cited commit + script; run IDs embed their commit hashes.

## Commit trail

| commit | branch | content |
|---|---|---|
| `438f6bf` | `char/dt-convergence-r4` | fixed-dt sweep (dt, dt/2, dt/4): RK4 order nominal, conservation floor dt-independent → R-3 before R-4 |
| `9fec345` | `char/r3-product-aliasing` | corrected three-level order estimator, p = log₂(R−1); R = 16.958 → p = 3.996 |
| `c69c28b` | `char/r3-product-aliasing` | factorized-variant attribution (`tests/audit_r3_product.py`) + preregistered repair in KNOWN_RISKS |
| `0678ac1` | `fix/r3-fine-product-grid` | production fix + prediction verification + P3 falsification record |

Prerequisite context: R-1 Jacobian fix (`027bc14`), R-5 spectral-η fix (`89985aa`),
diagnostics/run-capsule infrastructure (`578a985`, `501bce4`).

## Configuration (all measurements unless stated)

Rotating Earth-like planet (`day_hours = 24`, Ω = 7.272e−5 s⁻¹), geodesic grid
resolution 4 (2562 points), l_max = 21, tendency truncation cut = ⌊2·21/3⌋ = 14,
viscosity 0, RK4 with fixed dt = 0.5·min_edge/max|u₀| (≈ 2204–2208 s for RH4),
horizon 5 days. IC: Rossby–Haurwitz wavenumber-4 (ν = K = 7.848e−6 s⁻¹) — an exact
rigid-rotation solution of the nondivergent BVE whose spectral content lies exactly in
l ∈ {1, 5}, so its initial products stay below the l = 14 cut and every invariant change
is numerical. Hardware: NVIDIA GeForce MX110, CuPy 13.4, float64.

## Factorized variants

| variant | product grid | truncation | return | isolates (pairwise) |
|---|---|---|---|---|
| A | state grid (res 4) | inside jacobian, then synthesize + re-analyze | grid | pre-fix production |
| B | state grid (res 4) | once, spectral | spectral | A−B = extra round trip |
| C | state grid (res 4) | none | spectral | B−C = the truncation |
| D | product grid (res 5, same l_max) | once, spectral | spectral | D−B = product-analysis quadrature |

## Characterization results (`c69c28b`, `tests/audit_r3_product.py`)

5-day energy / absolute-enstrophy drift, standard orientation:

| variant | res 4 E drift | res 4 Z_abs drift | res 5 E drift |
|---|---|---|---|
| A | −2.6423e−3 | −6.368e−4 | −7.203e−4 |
| B | −2.8356e−3 | −5.633e−4 | −3.777e−4 |
| C | −2.2926e−3 | −4.644e−4 | −4.869e−4 |
| D | **−4.4555e−4** | **−8.461e−5** | (needs res-6 co-grid; not run) |

Key closure: res-5-native B (−3.78e−4) ≈ res-4 D (−4.46e−4) — the conservation floor is
set by points-per-product-bandwidth. Instantaneous discrete production at t0
(A −4.36e−4/day vs D −6.13e−5/day) time-integrates to the observed 5-day drift.
Artifacts: `runs/r3-product/summary_res4.txt`, `summary_res5.txt`,
`final_states_{res4,res5}.npz` (regenerate with `python tests/audit_r3_product.py res4|res5`
at `c69c28b`+).

## Preregistered predictions (recorded in KNOWN_RISKS.md at `c69c28b`, before implementation)

| | prediction | outcome (`0678ac1`) |
|---|---|---|
| P1 | res4/l21 RH4 5-day E drift −2.64e−3 → −4.5e−4 ± 20 % | ✅ **−4.455e−4** (in band; locked by `test_prediction_p1_5day_energy_drift`) |
| P2 | t0 production rate → ≈ −6e−5/day | ✅ **−6.1e−5/day** (locked by `test_prediction_p2_t0_production_rate`) |
| P3 | tilt-60° drift magnitude shrinks ≥ 3× *from the fine quadrature* | ❌ **REFUTED as worded** — see falsification below |

## P3 falsification and corrected attribution

Decomposition on the fix branch (`0678ac1`), 5-day RH4 E drift by orientation
(tilt = rotation of the RH4 pattern about the y-axis; tilted RH4 is *not* a rigid
solution, so magnitudes are indicative of orientation sensitivity only):

| tilt | A coarse + round trip | B coarse + spectral | D fine + spectral (production) |
|---|---|---|---|
| 0° | −2.642e−3 | −2.836e−3 | **−4.455e−4** |
| 30° | −1.108e−2 | −1.026e−2 | −9.692e−3 |
| 60° | +1.746e−2 | +5.412e−4 | −9.256e−4 |

- **On-axis** (tilt 0): the fine quadrature does the work (B→D, 6.4×); the round trip is
  negligible. This is the regime the characterization measured — its "product quadrature
  is the dominant defect" claim was an on-axis truth.
- **Off-axis** (tilt 60): the extra synthesis/re-analysis round trip was a separate,
  larger, strongly orientation-dependent error; removing it (A→B) accounts for 32×, and
  the fine quadrature slightly *worsens* this non-solution flow. The characterization
  missed this because its tilt sweep ran variant A alone.
- P3 fails because it credited the fine quadrature with the tilt-60 gain. The fix as a
  whole (round-trip removal + fine quadrature, both shipped) improves every orientation
  — 6.4× / 1.1× / 19× — and is never harmful.

**Residual limitation (precise statement).** The tendency is truncated at l = 14 while
the prognostic state retains modes through l = 21, so the evolved system is not an
invariant-conserving Galerkin truncation of either the l ≤ 14 or the l ≤ 21 system.
Nonlinear transfer into the band l ∈ (14, 21] therefore carries no conservation
guarantee, and flows that drive such transfer (tilt-30 RH4 at ≈ −1e−2/5 d;
`two_vortices` filamentation, whose 5-step integrated |dE/E| rises from 1.7e−3 to
6.7e−3 under the more accurate fine quadrature) drift regardless of product-quadrature
quality. Closing this requires a consistent truncation/state treatment (evolve the state
at the cut, or scale-selective dissipation across it) — a separate design decision
outside R-3.

## Related supporting results

- dt-independence of the floor (`438f6bf`, `tests/audit_r4_convergence.py`): E drift
  −2.6423e−3 identical to 4 significant figures at dt, dt/2, dt/4; trajectory order
  p = 3.996; temporal error ≈ 2700× below the conservation floor. Artifacts:
  `runs/r4-convergence/{summary.txt, aligned.npz}` (with appended estimator-correction note).
- Provenance: `product_quadrature` is a CLI option (`--product-quadrature`, default
  `fine`), recorded in every run's `config.json` and `manifest.json` via the serialized
  args; no silent fallback exists (`ValueError` on unsupported grid/mode, locked by
  `test_no_silent_fallback_on_unsupported_grid`, `test_invalid_product_quadrature_rejected`,
  `test_cli_exposes_and_defaults_product_quadrature`).

## Run IDs (immutable capsules under `runs/`, Git-ignored; IDs embed commit)

| run ID | role |
|---|---|
| `20260712T023305Z_two-vortices_rot24h_r4_l21_dt10h_501bce43` | pre-R-5 rotating 10-day baseline |
| `20260712T032424Z_two-vortices_norot_r4_l21_dt10h_0b6c1351` | pre-R-5 non-rotating 10-day baseline |
| `20260712T033635Z_two-vortices_rot24h_r4_l21_dt10h_89985aaa` | post-R-5 rotating 10-day baseline |
| `20260712T033832Z_two-vortices_norot_r4_l21_dt10h_89985aaa` | post-R-5 non-rotating 10-day baseline |
| `r3-smoke/20260712T080551Z_rh4_rot24h_r4_l21_dt6h_0678ac18` | post-R-3 CLI end-to-end smoke |
| `runs/r4-convergence/` | dt-sweep analysis artifacts (not a run capsule) |
| `runs/r3-product/` | variant-attribution analysis artifacts (not a run capsule) |
