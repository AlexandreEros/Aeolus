# Lat-lon backend — mathematical correctness & backend-parity review

Review record for `feat/latlon-grid` (Gauss-Legendre lat-lon backend through the
`SphericalGridBackend` seam). Every number below is reproducible from the cited
commit + scripts; raw `runs/` artifacts are Git-ignored, so the measurements are
preserved here.

## Commit trail

| commit | content |
|---|---|
| `bbd82a7` | generalize velocity reconstruction + CFL length scale off geodesic-only APIs |
| `b47524d` | corrected Gauss-Legendre lat-lon geometry + exact SH transform |
| `4403a48` | `LatLonBackend` through the `SphericalGridBackend` seam |
| `09929b0` | Planet/CLI integration + numerics provenance in `manifest.json` |
| `06e784e` | backend-parity suite (same physics tests on both backends) |
| `7cdc186` | end-to-end lat-lon runs (viz via interpolated view grid) |
| `c5a3312` | cfl_length_scale regime clarification + distinct-fine-grid test |

## Configuration

Rotating Earth-like planet (`day_hours = 24`, Ω = 7.272e−5 s⁻¹), l_max = 21,
tendency truncation cut = ⌊2·21/3⌋ = 14, viscosity 0, RK4 with fixed
dt = 0.5·L(cfl)/max|u₀|, horizon 5 days, IC Rossby-Haurwitz wavenumber-4
(ν = K = 7.848e−6 s⁻¹). Backends compared: geodesic resolution 4 (2562 points)
vs Gauss-Legendre lat-lon 32×64 (2048 points), both `product_quadrature="fine"`.
GPU: NVIDIA MX110 (2 GB). Measurement scripts: `review_runs.py`,
`verify_quadrature_cfl.py` (session scratchpad).

---

## 1. CFL length scale is conservative for a spectral method

`GaussLatLonGridGeometry.cfl_length_scale` returns the minimum meridional node
spacing R·min(Δcolat). The physical *zonal* spacing a·cosφ·Δλ shrinks toward the
poles and is **not** the binding constraint, because the tendency is a global
spectral transform with no grid-local stencil (derivatives are spectral
multiplications; products are synthesize→multiply→analyze). Advective stability is
set by the max wavenumber l_max/R and max speed U, not grid spacing.

Measured (nlat=32, l_max=21, U₀≈99.1 m/s, R=6.371e6 m):

| quantity | value |
|---|---|
| meridional min spacing (used) | 6.107e5 m → native dt **3080 s** |
| physical zonal spacing at pole-most node (lat 85.76°) | 4.62e4 m (**13.2×** smaller) |
| naïve zonal-grid CFL would demand | dt = 233 s |
| spectral RK4 stability limit, dt ≤ 2.828/(U·l_max/R) | dt = 8655 s |
| **native/stability margin** | **2.81×** |
| meridional / (R·l_max⁻¹) | 2.01 |

The 5-day run is stable at 3080 s — 13× above the naïve zonal-grid CFL — confirming
the pole clustering is redundant sampling, not resolved structure. **Conservative,
correctly reasoned.** The proxy stays conservative in the resolved regime the
transform recommends (nlat ≥ l_max+1), which the transform *warns* about but does
not enforce; oversampling (nlat ≫ l_max) is safe but shrinks dt needlessly.
Nonblocking refinement: define it as ≈ πR/l_max to decouple dt from grid
oversampling.

## 2. Geodesic drift reconciles with the locked RH4 value

The locked value (`test_r3_fine_product.py::test_prediction_p1_5day_energy_drift`)
is the **5-day energy** drift. The review harness reproduces it exactly:

* geodesic native CFL → N=196, dt=2204.08 s, **E drift = −4.4555e−4** (lock band
  [−5.4e−4, −3.6e−4]; characterization −4.4555e−4). Exact to all printed digits.

Absolute enstrophy was never locked; the measured 5-day geodesic value is
**−8.46e−5** (strongly sublinear: −5.82e−5 at 1 day). An earlier hand-off note's
"~−3e−4" was a bad linear extrapolation and is superseded by this measurement.

## 3. Matched-dt vs native-CFL — the improvement is spatial, not temporal

| run | dt (s) | N | backend | E drift | absZ drift |
|---|---|---|---|---|---|
| (A) matched | 2204.08 | 196 | geodesic | −4.4555e−4 | −8.46e−5 |
| (A) matched | 2204.08 | 196 | lat-lon | **−1.34e−10** | −3.12e−11 |
| (B) native | 2204.08 | 196 | geodesic | −4.4555e−4 | −8.46e−5 |
| (B) native | 3085.71 | 140 | lat-lon | −7.20e−10 | −1.68e−10 |

At **identical dt and step count** lat-lon beats geodesic by ~6 orders
(−1.34e−10 vs −4.46e−4): the advantage is the exact GL product quadrature, not a
larger timestep. Lat-lon's residual grows from −1.34e−10 (matched) to −7.20e−10
(native, larger dt) — RK4 O(dt⁴) time-truncation — so lat-lon is time-error
dominated while geodesic is spatial-aliasing dominated. (Geodesic native = matched
here because its native dt is the smaller of the two.)

## 4. Fine GL product grid gives exact quadratic-product quadrature

Product of two degree-L fields has SH degree ≤ 2L; analyzed against Y_l^m (l ≤ L)
the integrand has degree ≤ 3L.

* **Longitude:** the uniform N_lon-point rule is exact for zonal wavenumbers
  |k| < N_lon; need N_lon > 3L ⟹ N_lon ≥ 3L+1. Code: `nlon_f = 3L+1 = 64`.
* **Latitude:** in x = cosθ the integrand is a genuine polynomial of degree ≤ 3L —
  the (1−x²)^{(|m₁|+|m₂|+|m|)/2} factor has integer power because |m₁|+|m₂|+|m| is
  always even when m₁+m₂=m. GL with N_lat nodes is exact to degree 2N_lat−1; need
  2N_lat−1 ≥ 3L ⟹ N_lat ≥ ⌈(3L+1)/2⌉. Code: `nlat_f = (3L)//2 + 1 = ⌈(3L+1)/2⌉
  = 32`, giving 2·32−1 = 63 = 3·21. **Meets with equality.**

Numeric confirmation at l_max=21: max|fine − 4×-dense reference| = **1.19e−12**. The
bound is tight, not padded: undersizing to nlat=30 (2·30−1=59 < 63) raises the error
to **2.54**. The distinct-fine-grid path (minimal state grid → strictly larger fine
grid) is regression-covered in
`test_latlon_backend.py::test_fine_grid_strictly_larger_than_minimal_state_grid`.

## 5. Wall time, per-step cost, peak GPU memory, provenance (native CFL, 5-day)

| | geodesic | lat-lon |
|---|---|---|
| N steps | 196 | 140 |
| wall | 22.58 s | **3.82 s** |
| per-step | 115.2 ms | **27.3 ms** (4.2×) |
| pool peak / growth | 112.5 / 42.4 MB | 86.8 / 16.7 MB |
| device footprint of run | 41.5 MB | 16.6 MB (of 2147 MB) |
| product sampling | geodesic-res5-voronoi (10242 pts) | latlon-gauss-32x64-3/2rule (2048 pts) |
| transform | GeodesicSphericalHarmonics | GaussLatLonSphericalHarmonics |

Provenance is recorded verbatim in `manifest.json["numerics"]`. For a state grid
that already satisfies the 3/2 rule (32×64 at l_max=21), the lat-lon `fine` grid
coincides with the state grid, so `coarse` and `fine` are numerically identical
(both exact) — this partly explains the per-step/memory advantage vs the geodesic
res-5 co-grid (4× the state points).

---

## Verdict

* **Merge blockers:** none. Transform exact where claimed, CFL conservative in
  every resolved config, operators backend-agnostic (20-test parity suite green),
  locked RH4 energy value reproduced exactly.
* **Nonblocking:** (1) redefine cfl_length_scale as a spectral scale ≈ πR/l_max;
  (2) fixed-dt-from-initial-CFL (pre-existing R-4) shared by both backends.
* **Recommendation: approve for merge.**
