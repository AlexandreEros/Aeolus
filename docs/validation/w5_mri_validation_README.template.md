# Williamson-5 MRI projected-reference validation

Aeolus T42 and T63 solutions were compared on their native Gaussian grids
with independently prepared T42 and T63 spectral projections of the MRI–JMA
N958 reference integration. Reference preparation used no Aeolus code, and
the frozen projected fields were consumed by hash without further
interpolation or spectral processing.

## Scientific scope

This validation deliberately keeps four questions separate:

1. **Day-zero convention verification** checks the initial free-surface
   height, eastward and northward velocity components, wind speed, units,
   signs, Gaussian-grid orientation/alignment, and mountain location before
   any forecast-error metrics are computed.
2. **MRI projected-reference agreement** reports resolved-scale
   free-surface-height and vector-velocity errors against the independent
   MRI→T42 and MRI→T63 projections at days 5, 10, and 15.
3. **Aeolus conservation and stability** reports intrinsic total-energy and
   potential-enstrophy drift, true minimum fluid-layer thickness, maximum
   wind speed, finite-state checks, step counts, and timestep statistics.
4. **T42–T63 self-convergence** compares saved Aeolus spectral states only on
   the common `l <= 42` triangle. The experiments differ in spectral
   truncation, native grid, resolved topography representation, and adaptive
   timestep history.

The MRI `height` field is an absolute free-surface height in metres. Aeolus
stores perturbation fluid-layer geopotential `phi` about `Phi0 = gH`, with
fixed surface geopotential `phi_s` separate from the prognostic state.
Therefore:

```text
Aeolus free-surface height = (Phi0 + phi + phi_s) / gravity
Aeolus fluid-layer thickness = (Phi0 + phi) / gravity
```

The second expression—not free-surface height—is used for positivity and
minimum-depth diagnostics.

## Interpretation limits

The projected-reference metrics are not direct errors against the untouched
N958 grid. No project-defined threshold is claimed to be an official
Williamson pass criterion. The frozen reference package is immutable input:
Notebook B performs no MRI spectral analysis, projection, velocity
reconstruction, interpolation, regridding, time repair, or artifact
regeneration.
