# Williamson Test Case 5 — MRI-JMA vs Aeolus validation notebooks

Two small, self-contained Colab notebooks that answer one question:

> Does Aeolus reproduce the MRI-JMA Williamson test-case-5 solution when both
> models are compared on the **same physical fields** and the **same
> initial-value problem**?

Start with **`W5_MRI_SEMANTIC_AUDIT.md`** — the source-cited audit that
establishes what MRI's archived `h` is, what Aeolus's `phi / Phi0 / phi_s`
are, and whether the two initial conditions describe the same problem.
**Current finding: they do not** (Aeolus prescribes the Williamson case-2
field as layer *thickness*; MRI/Williamson prescribe it as the *free
surface*), so Notebook B's day-zero gate fails by design and refuses the
15-day runs until the Aeolus initial condition is corrected on a separate
branch.

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
                                        MODE="postprocess":
                                          figures + metrics from saved
                                          outputs (CPU; never re-runs)
```

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
agreement between the two models.

## Usage

1. Run Notebook A once (any runtime; ~320 MB download from
   `climate.mri-jma.go.jp`). Output: `mri-w5-reference/` with two `.npz`
   packages, `manifest.json`, and check figures.
2. Open Notebook B on a **GPU runtime**, point `REFERENCE_DIR` at the
   Notebook A output (copy it to Drive or re-run A in the same session),
   leave `MODE = "verify"`, run all cells, and read the day-zero verdict.
3. Only if the contract passes (after the IC fix lands): set
   `MODE = "run_t42"` / `"run_t63"` / `"run_both"`; results and comparisons
   land in `w5-validation/`. `MODE = "postprocess"` re-renders comparisons
   without ever re-integrating; `FORCE_RERUN = False` reuses valid completed
   outputs.

Notebook B pins Aeolus to commit `580c566a…` (edit `AEOLUS_COMMIT` in the
settings cell to validate another commit). Notebook A never installs or
imports Aeolus; Notebook B installs only genuinely missing dependencies
(CuPy on Colab) and never restarts the kernel itself.
