# Call Structure

Updated from the implementation on 2026-07-15. Solid arrows are direct calls
or construction; dashed arrows show selected data passed into a later stage.

## Entry points and planet construction

`aeolus` (cli/main.py) is the canonical executable; its `run bve`, `gen`, and
`recompile` subcommands share implementations with the `psx-*` compatibility
entry points. `aeolus list` and `aeolus inspect` are pure-stdlib and never
reach the assembly stage below.

```mermaid
flowchart LR
    subgraph cli["CLI entry points"]
        scripts["pyproject.toml scripts"]
        aeolus["aeolus<br/>main.main()"]
        gen["psx-gen / aeolus gen<br/>generate_planet"]
        bve["psx-bve / aeolus run bve<br/>bve.execute_run()"]
        cache["psx-recompile / aeolus recompile<br/>clear_cache"]
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

    scripts --> aeolus
    aeolus --> gen
    aeolus --> bve
    aeolus --> cache
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

## `aeolus run bve` setup and provenance

Parsing and configuration resolution happen before any CuPy import:
`BVERunConfig.resolve()` (run/bve/config.py) layers explicit flags over the
selected preset over ordinary defaults, resolves the snapshot schedule and
plot selection, validates cross-field constraints, and prints the resolved
configuration. Only then does `execute_run` create the run directory and
import the numerical stack.

```mermaid
flowchart TD
    main["main._cmd_run_bve() / bve.main()"] --> parse["parse_args()"]
    parse --> resolve["BVERunConfig.resolve()<br/>presets, snapshot schedule, plots"]
    resolve --> writable["_resolve_writable_base_dir()<br/>mkdir + probe, temp-dir fallback"]
    writable --> create["create_run_dir()"]
    create --> runDir["RunDirectory (no pointer yet)"]
    runDir --> initcfg["write config.json + manifest.json<br/>status='running'"]

    initcfg --> generate["Planet.generate()"]
    generate --> describe["planet.so.backend.describe()"]

    describe --> ic["make_ic(scenario, planet)"]
    ic --> initialGrid["initial vorticity on state grid"]
    initialGrid --> transform["planet.sh.transform()"]
    transform --> initialLM["initial spectral state zeta_lm"]

    describe --> refresh["rewrite manifest.json<br/>with numerics provenance"]
    refresh --> run["run_bve(snapshot_times, plots)"]
    initialLM --> run
    runDir -. "path + figure metadata" .-> run
    run -- "success" --> ok["persist status='completed'<br/>validate manifest + publish latest pointer"]
    run -- "exception" --> fail["status='failed' + error record<br/>no new latest pointer"]
```

If the requested output base cannot be created *or* is not writable
(either the `mkdir` fails or a probe file cannot be written), setup moves
the run beneath a system temporary directory before creating its
immutable run folder — so neither an unwritable target nor a missing
parent escapes as an unhandled exception. The manifest records the
actual backend, grid, transform, product sampling, `l_max`, environment,
GPU, command, and Git provenance. A fresh failed run never publishes
`latest_run.txt`. Before overwriting the run currently referenced by that
pointer, Aeolus strictly clears the pointer and transitions the capsule away
from `completed`; cleanup or execution failure then persists `failed` and
leaves the pointer absent. Successful publication validates a matching
`status='completed'` manifest and atomically replaces `latest_run.txt`, so
shell scripts never receive a missing, malformed, running, or failed capsule.

## BVE integration loop

```mermaid
flowchart TD
    run["run_bve()"] --> state["BarotropicState(zeta0_lm)"]
    run --> model["BarotropicVorticity(planet)"]
    model --> coriolis["construct grid f and exact spectral f_lm"]

    run --> recorder["DiagnosticsRecorder()"]
    recorder --> record0["record initial state<br/>(returns max_speed_ms)"]
    record0 --> cfl0["advective_cfl_timestep()<br/>initial ceiling dt_cfl"]

    run --> schedule["explicit snapshot schedule<br/>(resolved by BVERunConfig)"]
    schedule --> sched["IntegrationScheduler.next_event(dt_cfl)"]
    cfl0 --> sched

    sched --> ev{"event kind?"}
    ev -- "store" --> synth["planet.sh.inv_transform(state.coeffs)"]
    synth --> memory["append vorticity grid + time"]
    memory --> sched
    ev -- "step<br/>dt_step ≤ dt_cfl,<br/>clipped to land on target" --> rk4["rk4_step(model, state, t, dt_step)"]

    rk4 --> k1["model.tendency(y)"]
    rk4 --> k2["model.tendency(y + dt*k1/2)"]
    rk4 --> k3["model.tendency(y + dt*k2/2)"]
    rk4 --> k4["model.tendency(y + dt*k3)"]
    k1 --> combine["weighted RK4 update"]
    k2 --> combine
    k3 --> combine
    k4 --> combine
    combine --> accepted["accepted BarotropicState"]
    accepted --> record["DiagnosticsRecorder.record()<br/>(returns max_speed_ms)"]
    record --> cfln["advective_cfl_timestep()<br/>recompute ceiling from new state"]
    cfln --> sched

    ev -- "none (done)" --> close["DiagnosticsRecorder.close()"]
    close --> save["save coefficient/grid/time snapshots<br/>(always, independent of plots)"]
    save --> plotsel{"plot selection"}
    plotsel -- "diagnostics" --> diagPlot["plot_diagnostics()"]
    plotsel -- "snapshots" --> adapter["BVE visualization adapter<br/>reload persisted arrays"]
    adapter --> timeline["FigureTimeline<br/>shared frame normalization"]
    timeline --> snapshots["transactionally publish<br/>time-named frames"]
    plotsel -- "summary" --> viewer["VorticityViewer()"]
    viewer --> summary["backend-neutral summary spec"]
```

Image products run in a fixed order (diagnostics, snapshots, summary) and
only when selected; `--no-plots` skips all of them while the `.npy` states
and diagnostics CSV are still written.

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

    snapshots["saved run snapshots"] --> adapter["BVE / SWE visualization adapter"]
    adapter --> timeline["FigureTimeline"]
    timeline --> individual["normalized time-named frames"]
    adapter --> summary["model summary spec"]
    summary --> summaryPng["bve_summary.png / swe_summary.png"]
```

Optional diagnostic plotting failures are caught so completed numerical data
remain available. Selected snapshot/summary failures propagate, allowing the
run lifecycle to mark the capsule failed and withhold latest-run publication.
