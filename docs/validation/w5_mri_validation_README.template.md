# Williamson-5 MRI projected-reference validation

Each completed Aeolus stage is compared on its native Gaussian grid with the
corresponding independently prepared spectral projection of the MRI–JMA N958
reference integration. Reference preparation used no Aeolus code, and the
frozen projected fields were consumed by hash without further interpolation
or spectral processing.

## Scientific scope

This validation deliberately keeps four questions separate:

1. **Day-zero convention verification** checks the initial layer depth,
   eastward and northward velocity components, wind speed, units,
   signs, Gaussian-grid orientation/alignment, and mountain location before
   any forecast-error metrics are computed.
2. **MRI projected-reference agreement** reports resolved-scale
   layer-depth and vector-velocity errors against the independent
   MRI→T42 and MRI→T63 projections at days 5, 10, and 15.
3. **Aeolus conservation and stability** reports intrinsic total-energy and
   potential-enstrophy drift, true minimum fluid-layer thickness, maximum
   wind speed, finite-state checks, step counts, and timestep statistics.
4. **T42–T63 self-convergence** compares saved Aeolus spectral states only on
   the common `l <= 42` triangle. The experiments differ in spectral
   truncation, native grid, resolved topography representation, and adaptive
   timestep history.

The MRI `height` field is layer depth (fluid-layer thickness) in metres.
Bottom topography is a separate physical field. Aeolus stores perturbation
fluid-layer geopotential `phi` about `Phi0 = gH`, with fixed surface
geopotential `phi_s` separate from the prognostic state. Therefore:

```text
Aeolus layer depth = (Phi0 + phi) / gravity
Aeolus free-surface elevation = (Phi0 + phi + phi_s) / gravity
```

The first expression is the mandatory MRI comparison field and is also used
for positivity and minimum-depth diagnostics. The second is a separate
free-surface diagnostic and is not used for MRI `height` norms.

## Interpretation limits

The projected-reference metrics are not direct errors against the untouched
N958 grid. No project-defined threshold is claimed to be an official
Williamson pass criterion. The frozen reference package is immutable input:
Notebook B performs no MRI spectral analysis, projection, velocity
reconstruction, interpolation, regridding, time repair, or artifact
regeneration.
