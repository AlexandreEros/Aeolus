# Call Structure

Updated from the implementation on 2026-07-12. Solid arrows are direct calls
or construction; dashed arrows show selected data passed into a later stage.

## Entry points and planet construction

```mermaid
flowchart LR
    subgraph cli["CLI entry points"]
        scripts["pyproject.toml scripts"]
        gen["psx-gen<br/>generate_planet.main()"]
        bve["psx-bve<br/>bve.main()"]
        cache["psx-recompile<br/>clear_cache.main()"]
    end

    subgraph assembly["Planet assembly"]
        params["PlanetaryParameters.from_earth_like()"]
        planet["Planet.generate()"]
        select{"grid_type"}
        geoGrid["GeodesicGridGeometry()"]
        geoSH["GeodesicSphericalHarmonics()"]
        geoBackend["GeodesicBackend()"]
        llGrid["GaussLatLonGridGeometry()"]
        llSH["GaussLatLonSphericalHarmonics()"]
        llBackend["LatLonBackend()"]
        pointSH["PointSetSphericalHarmonics()"]
        spectralOps["SpectralOperators()"]
        productSpace["backend.product_space()<br/>coarse or cached fine sampling"]
        terrain["generate_spectral_terrain_gpu()"]
        elevation["ElevationData()"]
    end

    subgraph outputs["Immediate consumers"]
        planetViewer["PlanetViewer.plot_summary()"]
        bveSetup["BVE setup"]
        png["out/&lt;output&gt;.png"]
    end

    scripts --> gen
    scripts --> bve
    scripts --> cache

    gen --> params
    gen --> planet
    bve --> params
    bve --> planet
    params -. "params" .-> planet

    planet --> select
    select -- "geodesic" --> geoGrid --> geoSH --> geoBackend
    select -- "latlon" --> llGrid --> llSH --> llBackend
    geoSH --> pointSH
    llSH --> pointSH
    geoBackend --> spectralOps
    llBackend --> spectralOps
    spectralOps --> productSpace
    planet --> terrain --> elevation
    pointSH --> cuda["CUDA sh_matrix kernel"]

    gen --> planetViewer --> png
    bve --> bveSetup
    cache --> cupyCache["clear CuPy kernel cache<br/>verify CUDA source"]
```

The two production grid paths converge at the same coefficient layout and
dense GPU point-set transform. `SpectralOperators` receives the selected
backend, which owns nonlinear-product sampling. A fine product space is built
once on first use and then cached.

## `psx-bve` setup and provenance

```mermaid
flowchart TD
    main["bve.main()"] --> parse["build_parser().parse_args()"]
    parse --> writable["_resolve_writable_base_dir()"]
    writable --> create["create_run_dir()"]
    create --> runDir["RunDirectory"]
    runDir --> latest["update_latest_pointer()"]

    main --> generate["Planet.generate()"]
    generate --> describe["planet.so.backend.describe()"]

    main --> ic["make_ic(scenario, planet)"]
    ic --> initialGrid["initial vorticity on state grid"]
    initialGrid --> transform["planet.sh.transform()"]
    transform --> initialLM["initial spectral state zeta_lm"]

    runDir --> config["write config.json"]
    describe --> manifest["write_run_manifest()"]
    runDir --> manifest
    initialLM --> run["run_bve()"]
    generate --> run
    runDir -. "path + figure metadata" .-> run
```

If the requested output base is not writable, setup moves the run beneath a
system temporary directory before creating its immutable run folder. The
manifest records the actual backend, grid, transform, product sampling,
`l_max`, environment, GPU, command, and Git provenance.

## BVE integration loop

```mermaid
flowchart TD
    run["run_bve()"] --> state["BarotropicState(zeta0_lm)"]
    run --> model["BarotropicVorticity(planet)"]
    model --> coriolis["construct grid f and exact spectral f_lm"]

    run --> psi0["SpectralOperators.inv_laplacian(zeta0_lm)"]
    psi0 --> velocity0["velocity_from_streamfunction()"]
    velocity0 --> cfl["CFL timestep from max speed<br/>and grid.cfl_length_scale"]

    run --> recorder["DiagnosticsRecorder()"]
    recorder --> record0["record initial state"]

    cfl --> loop{"t &lt;= t_end?"}
    loop -- "yes" --> snapshot{"snapshot due?"}
    snapshot -- "yes" --> synth["planet.sh.inv_transform(state.coeffs)"]
    synth --> memory["append vorticity grid + time"]
    snapshot --> dt["dt_step = min(CFL, snapshot, remaining)"]
    memory --> dt
    dt --> rk4["rk4_step(model, state, t, dt_step)"]

    rk4 --> k1["model.tendency(y)"]
    rk4 --> k2["model.tendency(y + dt*k1/2)"]
    rk4 --> k3["model.tendency(y + dt*k2/2)"]
    rk4 --> k4["model.tendency(y + dt*k3)"]
    k1 --> combine["weighted RK4 update"]
    k2 --> combine
    k3 --> combine
    k4 --> combine
    combine --> accepted["accepted BarotropicState"]
    accepted --> record["DiagnosticsRecorder.record()"]
    record --> loop

    loop -- "no" --> close["DiagnosticsRecorder.close()"]
    close --> diagPlot["plot_diagnostics()"]
    diagPlot --> save["save coefficient/grid snapshots"]
    save --> viewer["VorticityViewer()"]
    viewer --> snapshots["plot_all_snapshots()"]
    viewer --> summary["plot_summary()"]
```

## One tendency evaluation

```mermaid
flowchart LR
    input["BarotropicState.coeffs<br/>zeta_lm"] --> invert["vorticity_to_streamfunction()"]
    invert --> psi["psi_lm"]
    input --> eta["eta_lm = zeta_lm + f_lm"]

    psi --> jac["SpectralOperators.jacobian_pseudospectral()"]
    eta --> jac
    jac --> derivatives["spectral derivative recurrences"]
    derivatives --> product["synthesize derivatives on<br/>backend ProductSpace"]
    product --> multiply["pointwise spherical Jacobian"]
    multiply --> analyze["ProductSpace.sh.transform()"]
    analyze --> truncate["2/3 spectral truncation"]
    truncate --> advection["-J(psi, eta)_lm"]

    input --> diffusion["nu * Laplacian eigenvalue * zeta_lm"]
    forcing["forcing_lm or zero"] --> total["sum tendency"]
    advection --> total
    diffusion --> total
    total --> mean["zero l=0 row"]
    mean --> output["dzeta_lm / dt"]
```

Absolute vorticity is assembled directly in spectral space; the active RK4
path does not synthesize and re-analyze the state merely to add the Coriolis
term. The product is analyzed once on the backend-selected sampling and
returned spectrally to the integrator.

## Diagnostics and output products

```mermaid
flowchart LR
    accepted["accepted spectral state"] --> recorder["DiagnosticsRecorder.record()"]
    recorder --> spectral["spectral_diagnostics()"]
    recorder --> grid["synthesis + velocity + CFL<br/>and periodic round-trip check"]
    spectral --> csv["diagnostics/timeseries.csv<br/>flushed every step"]
    grid --> csv
    spectral --> spectra["diagnostics/spectra.npz<br/>written on close"]

    csv --> plots["plot_diagnostics()"]
    spectra --> plots
    plots --> figures["figures/*.png"]

    snapshots["saved run snapshots"] --> viewer["VorticityViewer"]
    viewer --> individual["per-snapshot figures"]
    viewer --> summary["bve_summary.png"]
```

Plotting failures after integration are caught so completed numerical data
remain available even when figure generation fails.
