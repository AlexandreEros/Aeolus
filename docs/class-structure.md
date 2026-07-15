# Class Structure

Updated from the implementation on 2026-07-12. The diagrams separate the
load-bearing runtime classes from legacy and support classes so that the main
relationships remain readable.

## Core numerics and planet assembly

```mermaid
classDiagram
    direction LR

    class GridGeometry {
        <<abstract>>
        +latitudes
        +longitudes
        +point_latitudes
        +point_longitudes
        +n_points
        +colatitudes
        +grid_shape
        +cfl_length_scale
        +is_structured
        +points_latlon()
    }

    class GeodesicGridGeometry {
        +resolution
        +radius
        +points
        +faces
        +adjacency_matrix
        +min_edge_length
        +cell_areas
        +geodesic_subdivide()
        +build_adjacency_matrix()
    }

    class GaussLatLonGridGeometry {
        +nlat
        +nlon
        +radius
        +coslat
        +sinlat
        +solid_angle_weights
        +cell_areas
    }

    class PointSetSphericalHarmonics {
        +l_max
        +n_points
        +weights
        +Y_matrix
        +transform(values)
        +inv_transform(coeffs)
    }

    class GeodesicSphericalHarmonics {
        +grid
        +l_max
        +weights
        +transform(values)
        +inv_transform(coeffs)
        +inverse_transform(coeffs)
    }

    class GaussLatLonSphericalHarmonics {
        +grid
        +l_max
        +weights
        +transform(values)
        +inv_transform(coeffs)
        +inverse_transform(coeffs)
    }

    class ProductSpace {
        <<frozen dataclass>>
        +sh
        +coslat
        +geometry
        +label
    }

    class SphericalGridBackend {
        <<abstract>>
        +geometry
        +sh
        +l_max
        +supported_product_quadratures()
        +product_space(mode)
        +describe(product_quadrature)
    }

    class GeodesicBackend
    class LatLonBackend
    class PointSetBackend

    class SpectralOperators {
        +sh
        +grid
        +backend
        +product_quadrature
        +R
        +laplacian_coeffs(coeffs)
        +inv_laplacian(coeffs)
        +velocity_from_streamfunction(psi_lm)
        +grad_from_scalar(q_lm)
        +advect_scalar_by_streamfunction(psi_lm, q_lm)
        +jacobian_pseudospectral(a_lm, b_lm)
    }

    class PlanetaryParameters {
        <<dataclass>>
        +mass
        +equatorial_radius
        +sidereal_day
        +angular_velocity
        +oblateness
        +polar_radius
        +radius
        +density
        +from_earth_like()
        +from_si()
    }

    class ElevationData {
        <<dataclass>>
        +surface_height
        +radial_distance
        +sh_coeffs
        +power_spectrum
        +max_degree
        +get_j2()
        +oblateness_from_sh
    }

    class SpectralTerrainParams {
        <<dataclass>>
        +rms_elevation
        +spectral_exponent
        +seed
        +l_min
    }

    class TectonicParams {
        <<dataclass>>
        +dt
        +kappa_height
        +kappa_strain
        +noise_strength
        +l_cut_noise
        +gamma_activity
        +renormalize_height
    }

    class Planet {
        +params
        +grid
        +elevation
        +sh
        +so
        +generate(grid_type, product_quadrature)
        +reconstruct_surface()
    }

    GridGeometry <|-- GeodesicGridGeometry
    GridGeometry <|-- GaussLatLonGridGeometry

    GeodesicSphericalHarmonics *-- PointSetSphericalHarmonics : delegates to
    GaussLatLonSphericalHarmonics *-- PointSetSphericalHarmonics : delegates to
    GeodesicSphericalHarmonics --> GeodesicGridGeometry
    GaussLatLonSphericalHarmonics --> GaussLatLonGridGeometry

    SphericalGridBackend <|-- GeodesicBackend
    SphericalGridBackend <|-- LatLonBackend
    SphericalGridBackend <|-- PointSetBackend
    SphericalGridBackend *-- ProductSpace : caches
    GeodesicBackend --> GeodesicGridGeometry
    GeodesicBackend --> GeodesicSphericalHarmonics
    LatLonBackend --> GaussLatLonGridGeometry
    LatLonBackend --> GaussLatLonSphericalHarmonics

    SpectralOperators --> SphericalGridBackend : product policy
    SpectralOperators --> GridGeometry

    Planet *-- PlanetaryParameters
    Planet *-- GridGeometry
    Planet *-- ElevationData
    Planet *-- SpectralOperators
    Planet o-- GeodesicSphericalHarmonics
    Planet o-- GaussLatLonSphericalHarmonics
    Planet ..> SpectralTerrainParams : generate()
    Planet ..> TectonicParams : generate()
```

`Planet.generate()` selects one of two production families:

- `grid_type="geodesic"`: `GeodesicGridGeometry` +
  `GeodesicSphericalHarmonics` + `GeodesicBackend`.
- `grid_type="latlon"`: `GaussLatLonGridGeometry` +
  `GaussLatLonSphericalHarmonics` + `LatLonBackend`.

Both transform facades delegate their dense GPU analysis/synthesis work to
`PointSetSphericalHarmonics`. `SpectralOperators` asks the backend for a
cached `ProductSpace`; `coarse` uses the state sampling, while `fine` uses a
resolution-(r+1) geodesic grid or a 3/2-rule Gauss–Legendre grid.

## BVE runtime, diagnostics, and visualization

```mermaid
classDiagram
    direction LR

    class Planet

    class BarotropicState {
        <<dataclass>>
        +coeffs
        +tendency
    }

    class BarotropicVorticity {
        +planet
        +sh
        +so
        +grid
        +R
        +Omega
        +f
        +viscosity
        +vorticity_to_streamfunction(state)
        +streamfunction_to_vorticity(psi_coeffs)
        +tendency(state, forcing_coeffs)
        +step_leapfrog()
    }

    class DiagnosticsRecorder {
        +sh
        +so
        +grid
        +radius
        +omega
        +record(t, zeta_lm, dt, step)
        +close()
    }

    class RunDirectory {
        <<dataclass>>
        +path
        +run_id
        +base
        +experiment
        +commit
        +reused
        +figure_metadata(source)
        +update_latest_pointer()
    }

    class PlanetViewer {
        +planet
        +grid
        +plot_summary()
        +plot_scalar()
    }

    class VorticityViewer {
        +planet
        +grid
        +snapshots
        +times
        +plot_all_snapshots()
        +plot_summary()
    }

    BarotropicVorticity --> Planet : evolves
    BarotropicVorticity ..> BarotropicState : reads and creates
    DiagnosticsRecorder --> Planet : receives its numerics
    PlanetViewer --> Planet : visualizes
    VorticityViewer --> Planet : visualizes
    RunDirectory ..> PlanetViewer : figure metadata
    RunDirectory ..> VorticityViewer : figure metadata
```

`run_bve()` and `rk4_step()` are orchestration functions rather than classes.
They own the integration loop, pass `BarotropicState` through the model,
record every accepted step with `DiagnosticsRecorder`, save snapshots, and
construct `VorticityViewer`. The CLI creates a `RunDirectory` and writes the
run manifest/provenance around that flow.

## Secondary and legacy classes

```mermaid
classDiagram
    class GridGeometry {
        <<abstract>>
    }
    class LatLonGridGeometry {
        <<legacy equiangular grid>>
        +create(grid_resolution, lon_range)
    }
    class LatLonSphericalHarmonics {
        <<legacy Simpson transform>>
        +set_grid()
        +transform()
        +inv_transform()
    }
    class DifferentialOperatorsSpherical {
        <<local finite differences>>
        +from_geodesic_grid()
        +calculate_gradient()
        +calculate_divergence()
        +calculate_curl()
        +calculate_laplacian()
    }

    GridGeometry <|-- LatLonGridGeometry
```

The legacy equiangular grid/Simpson transform remain in the package for
comparison and compatibility but are not selected by `Planet.generate()`.
`DifferentialOperatorsSpherical` is retained as a local finite-difference
operator family; the active BVE path uses `SpectralOperators` instead.
