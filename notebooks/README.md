# Williamson Test Case 5 — MRI-JMA vs Aeolus validation notebooks

Two small, self-contained Colab notebooks that answer one question:

> Does Aeolus reproduce the MRI-JMA Williamson test-case-5 solution when both
> models are compared on the **same physical fields** and the **same
> initial-value problem**?

**Current status: yes, within the documented envelopes.** The day-zero physical
contract passes at T42 and T63, both 15-day runs completed, and the accepted
result — with figures, conservation diagnostics, comparison tables, and full
provenance — is
[`docs/validation/williamson5_mri_2026-07-30.md`](../docs/validation/williamson5_mri_2026-07-30.md).

The path there matters. **`W5_MRI_SEMANTIC_AUDIT.md`** is the source-cited audit
that establishes what MRI's archived `h` is, what Aeolus's `phi / Phi0 / phi_s`
are, and whether the two initial conditions describe the same problem. Its
finding was that they did **not**: Aeolus prescribed the Williamson case-2 field
as layer *thickness*, while MRI and Williamson prescribe it as the *free
surface*. Notebook B's day-zero gate failed by design and refused the 15-day
runs until the initial condition was corrected — which it was, in commit
`668e6c9a`, *fix(swe): canonicalize Williamson-5 initial state*. Everything in
this directory is downstream of that correction; every pre-correction W5
trajectory is superseded.

## Workflow

```
Notebook A (CPU)                        Notebook B (GPU except postprocess)
mri_w5_reference_preparation.ipynb      w5_mri_validation_colab.ipynb
--------------------------------        -----------------------------------
download MRI archive (sha256)     -->   MODE="verify":
verify archived-field semantics           verify package hashes + schema
   (day-0 == case-2 surface,              verify field contract
    STDOUT mass identity)                 day-zero physical comparison
build physical fields on the              6-h smoke run, then STOP
   t42 (64x128) / t63 (96x192)          MODE="run_t42"/"run_t63"/"run_both":
   Gaussian grids                         15-day runs -- ONLY if the day-zero
freeze: npz per grid + manifest           contract passed (or explicit
   (hashes, units, equations)             override) -- then postprocess
zip the package for handoff             MODE="postprocess":
                                          figures + metrics from saved
                                          outputs (CPU; never re-runs)
```

The reference package moves from A to B through the browser: Notebook A writes
`mri-w5-reference.zip` and downloads it; Notebook B uploads and extracts it. No
Google Drive mount is required.

## Field contract (what is compared)

```
MRI:     free_surface_height = archived h
         layer_depth         = archived h - topography_height
Aeolus:  layer_depth         = (Phi0 + phi) / gravity
         free_surface_height = (Phi0 + phi + phi_s) / gravity
         topography_height   = phi_s / gravity   (band-limited cone)
topography_height (reference) = 2000 * (1 - min(r, pi/9)/(pi/9)),
         r = coordinate-plane distance to (30N, 270E), analytic, static
```

Both notebooks assert these identities; nothing is inferred from numerical
agreement between the two models. Because Aeolus carries a **band-limited** cone
and the reference an **analytic** one, the raw `layer_depth` difference includes
a large static terrain-representation term; removing it returns the free-surface
difference identically. See § 6 of the validation report.

## Usage

1. Run Notebook A once (any runtime; ~320 MB download from
   `climate.mri-jma.go.jp`). Output: `mri-w5-reference/` with two `.npz`
   packages, `manifest.json`, check figures, and `mri-w5-reference.zip`.
2. Open Notebook B on a **GPU runtime**, upload the ZIP when prompted, set
   `MODE = "verify"`, run all cells, and read the day-zero verdict. `verify`
   never starts a 15-day integration.
3. To reproduce the accepted result: set `MODE = "run_both"` (the mode these
   notebooks were last executed with); results and comparisons land in
   `w5-validation/`. `MODE = "postprocess"` re-renders comparisons without ever
   re-integrating; `FORCE_RERUN = False` reuses valid completed outputs.

Notebook B pins Aeolus to commit `668e6c9a5735b2d7200dcc121eb0f7c40450ae08` —
the commit that produced the canonical result — and asserts that the checkout
actually landed on it. Edit `AEOLUS_COMMIT` in the settings cell to validate
another commit. Notebook A never installs or imports Aeolus; Notebook B installs
only genuinely missing dependencies (CuPy on Colab) and never restarts the
kernel itself.

These are the versions that were executed, with their outputs stripped for
version control. They are not byte-identical to the copies committed at
`668e6c9a`, whose pin was still stale; the authoritative record of what ran is
the `manifest.json` inside each run capsule, which records argv, configuration,
environment, GPU, and clean-worktree status independently of any notebook.
