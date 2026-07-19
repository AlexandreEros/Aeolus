"""Topographic shallow-water verification.

Covers the topography-foundation spec end to end: representation validity
(band-limited Gaussian mountain, parameter rejection, projection-residual
gate), exact flat-bottom regression (bit-identity), the merge-blocking
resting-free-surface (lake-at-rest) test on both backends, uniform
bottom+free-surface offset invariance, exact mass conservation in a
mountain-flow run, RK4 timestep refinement over terrain, the no-transfer
regression for the integration loop, and the CLI/provenance/inspect
contracts (flat runs keep their historical config schema and hashes).

CPU-only configuration/CLI tests run everywhere; model/runner tests need
CUDA and use small Gauss lat-lon / geodesic configurations.
"""
from __future__ import annotations

import csv
import json
import math

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="CUDA/CuPy not available")

MOUNTAIN = dict(height_m=1500.0, lat_deg=25.0, lon_deg=60.0, width_deg=25.0)


# ---------------------------------------------------------------------------
# Configuration and provenance (CPU)
# ---------------------------------------------------------------------------

def test_flat_config_keeps_historical_schema_and_hash():
    """Flat runs must emit exactly the historical config dict (no topography
    keys), so old scientific hashes and run ids remain valid."""
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({})
    assert cfg.topography == "flat"
    d = cfg.to_run_config_dict()
    assert "topography" not in d
    assert not any(k.startswith("mountain_") for k in d)
    # The exact historical key set, frozen.
    assert set(d) == {
        "solver", "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
        "radius_earth_units", "duration_days", "gravity", "mean_depth_m",
        "dt_snapshots", "scenario", "product_quadrature", "out",
        "experiment", "overwrite", "snapshot_mode", "n_snapshots",
        "snapshot_times", "plots"}

    # Explicitly requesting the default is the same scientific identity.
    explicit_flat = SWERunConfig.resolve({"topography": "flat"})
    assert explicit_flat.scientific_config_dict() == cfg.scientific_config_dict()

    from datetime import datetime, timezone
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (make_run_id(d, now=now, commit="deadbeef")
            == make_run_id(explicit_flat.to_run_config_dict(), now=now,
                           commit="deadbeef"))


def test_mountain_config_resolution_and_identity():
    from datetime import datetime, timezone
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({"topography": "mountain"})
    assert cfg.mountain_height_m == 2000.0
    assert cfg.mountain_lat_deg == 30.0
    assert cfg.mountain_lon_deg == 90.0
    assert cfg.mountain_width_deg == 20.0
    d = cfg.to_run_config_dict()
    assert d["topography"] == "mountain"
    assert d["mountain_height_m"] == 2000.0

    # Terrain participates in the scientific identity: distinct hash and
    # run id from the flat configuration, and from other mountains.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    flat_id = make_run_id(SWERunConfig.resolve({}).to_run_config_dict(),
                          now=now, commit="deadbeef")
    mtn_id = make_run_id(d, now=now, commit="deadbeef")
    other = SWERunConfig.resolve({"topography": "mountain",
                                  "mountain_height_m": 500.0})
    other_id = make_run_id(other.to_run_config_dict(), now=now,
                           commit="deadbeef")
    assert len({flat_id, mtn_id, other_id}) == 3

    # Summary lines mention the terrain.
    assert any("mountain" in line for line in cfg.summary_lines())
    assert any("topography          flat" in line
               for line in SWERunConfig.resolve({}).summary_lines())


def test_config_rejects_invalid_topography_settings():
    from planetary_sandbox.run.swe.config import SWERunConfig

    with pytest.raises(ValueError, match="unknown topography"):
        SWERunConfig.resolve({"topography": "everest"})
    # Mountain parameters without the mountain preset are user errors.
    with pytest.raises(ValueError, match="require"):
        SWERunConfig.resolve({"mountain_height_m": 1000.0})
    with pytest.raises(ValueError, match="require"):
        SWERunConfig.resolve({"topography": "flat",
                              "mountain_width_deg": 10.0})
    for bad in ({"mountain_height_m": -5.0}, {"mountain_height_m": math.nan},
                {"mountain_width_deg": 0.0}, {"mountain_width_deg": 120.0},
                {"mountain_lat_deg": 100.0}, {"mountain_lon_deg": 500.0}):
        with pytest.raises(ValueError):
            SWERunConfig.resolve({"topography": "mountain", **bad})


def test_swe_cli_topography_parse_contracts(capsys):
    from planetary_sandbox.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--topography" in out and "--mountain-height-m" in out

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--topography", "everest"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--mountain-height-m", "1000"])
    assert exc.value.code == 2


def test_inspect_shows_topography(tmp_path, capsys):
    from planetary_sandbox.cli.main import main

    def write_manifest(run_dir, run_config):
        run_dir.mkdir(parents=True)
        manifest = {"created_utc": "2026-07-19T00:00:00+00:00",
                    "run_id": run_dir.name, "status": "completed",
                    "run_config": run_config}
        (run_dir / "manifest.json").write_text(json.dumps(manifest),
                                               encoding="utf-8")

    write_manifest(tmp_path / "mtn", {
        "solver": "swe", "lmax": 15, "grid": "latlon", "nlat": 32,
        "nlon": 64, "scenario": "williamson2", "topography": "mountain",
        "mountain_height_m": 2000.0, "mountain_lat_deg": 30.0,
        "mountain_lon_deg": 90.0, "mountain_width_deg": 20.0})
    assert main(["inspect", str(tmp_path / "mtn")]) == 0
    out = capsys.readouterr().out
    assert "mountain (h=2000.0 m" in out

    # Old flat-bottom manifests (no topography key) remain readable and are
    # reported as flat.
    write_manifest(tmp_path / "flat", {
        "solver": "swe", "lmax": 15, "grid": "latlon", "nlat": 32,
        "nlon": 64, "scenario": "williamson2"})
    assert main(["inspect", str(tmp_path / "flat")]) == 0
    out = capsys.readouterr().out
    assert "topography        flat" in out


# ---------------------------------------------------------------------------
# Model-level fixtures (GPU)
# ---------------------------------------------------------------------------

def _make_planet(grid_type="latlon", nlat=32, nlon=64, l_max=15,
                 resolution=3, day_hours=23.9345):
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    return Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_type=grid_type, nlat=nlat, nlon=nlon, l_max=l_max,
        grid_resolution=resolution)


@pytest.fixture(scope="module")
def latlon_planet():
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    return _make_planet()


@pytest.fixture(scope="module")
def geodesic_planet():
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    return _make_planet(grid_type="geodesic", resolution=3, l_max=10)


def _mountain_model(planet, mean_depth=3000.0, **model_kw):
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography
    topo = Topography.mountain(planet, **MOUNTAIN)
    return ShallowWaterModel(planet, mean_depth=mean_depth, topography=topo,
                             **model_kw)


def _nontrivial_state(l_max, phi0):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    zeta = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
    delta = cp.zeros_like(zeta)
    phi = cp.zeros_like(zeta)
    zeta[3, 2] = 2e-5 * (1.0 + 0.5j)
    zeta[1, 0] = 1e-5
    delta[4, 1] = 5e-7 * (0.3 + 1.0j)
    phi[2, 1] = 0.05 * phi0 * (0.7 - 0.2j)
    return ShallowWaterState.from_fields(zeta, delta, phi)


# ---------------------------------------------------------------------------
# 1. Flat-bottom regression (bit-identity)
# ---------------------------------------------------------------------------

@requires_cuda
def test_flat_topography_is_bit_identical_to_no_topography(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    m_none = ShallowWaterModel(latlon_planet, mean_depth=3000.0)
    m_flat = ShallowWaterModel(
        latlon_planet, mean_depth=3000.0,
        topography=Topography.flat(latlon_planet.sh.l_max))
    assert not m_flat.has_topography

    state = _nontrivial_state(latlon_planet.sh.l_max, m_none.phi0)
    t_none = m_none.tendency(state.coeffs)
    t_flat = m_flat.tendency(state.coeffs)
    # Bitwise identity, not merely numerical closeness.
    assert bool(cp.all(t_none.view(cp.float64) == t_flat.view(cp.float64)))

    # Scenario states are also bit-identical (including all-(+0.0) rest).
    for name in ("rest", "gravity_wave", "williamson2"):
        s_none = make_swe_ic(name, m_none)
        s_flat = make_swe_ic(name, m_flat)
        assert bool(cp.all(s_none.coeffs.view(cp.float64)
                           == s_flat.coeffs.view(cp.float64)))


@requires_cuda
def test_mountain_changes_only_the_divergence_tendency(latlon_planet):
    """The topographic term is exactly -laplacian(phi_s) in the delta row;
    zeta and phi tendencies are bitwise untouched."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel

    m_flat = ShallowWaterModel(latlon_planet, mean_depth=3000.0)
    m_mtn = _mountain_model(latlon_planet)
    state = _nontrivial_state(latlon_planet.sh.l_max, m_flat.phi0)

    t_flat = m_flat.tendency(state.coeffs)
    t_mtn = m_mtn.tendency(state.coeffs)
    assert bool(cp.array_equal(t_mtn[0], t_flat[0]))  # zeta untouched
    assert bool(cp.array_equal(t_mtn[2], t_flat[2]))  # phi untouched
    expected_delta = t_flat[1] - m_mtn.lap_eigs[:, None] * m_mtn.phi_s_lm
    expected_delta[0, :] = 0.0
    assert bool(cp.array_equal(t_mtn[1], expected_delta))


# ---------------------------------------------------------------------------
# 2. Resting free surface over topography (merge-blocking)
# ---------------------------------------------------------------------------

def _assert_lake_at_rest_preserved(planet):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    from planetary_sandbox.run.engine import rk4_step_array
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _mountain_model(planet)
    state = make_swe_ic("rest", model)

    # The state is nontrivial: spatially varying thickness under a constant
    # free surface, exactly monopole-free (mean thickness exactly H).
    assert float(cp.abs(state.coeffs[2]).max()) > 0.0
    assert complex(state.coeffs[2][0, 0]) == 0.0
    fs = state.coeffs[2] + model.phi_s_anom_lm  # free-surface anomaly
    assert float(cp.abs(fs).max()) == 0.0

    # Machine-exact zero tendency: phi + phi_s is purely l=0, where the
    # Laplacian eigenvalue is exactly 0 and tendency rows are pinned.
    dot = model.tendency(state.coeffs)
    assert float(cp.abs(dot).max()) == 0.0

    elevation_before = model.topography.elevation_lm  # defensive copy

    # Multiple RK4 steps: nothing moves, to the bit (values compare equal;
    # only +/-0.0 signs may differ).
    y0 = state.coeffs.copy()
    y = state.coeffs
    for _ in range(10):
        y = rk4_step_array(model.tendency, y, 0.0, 600.0)
        model.validate_state(ShallowWaterState(y), context="lake at rest")
    assert bool(cp.array_equal(y, y0))

    # Velocity exactly zero, free surface exactly constant, mass monopole
    # exactly conserved, thickness strictly positive, no gravity waves
    # (delta identically zero).
    u, v = model.wind_on_state_grid(ShallowWaterState(y))
    assert float(cp.abs(u).max()) == 0.0
    assert float(cp.abs(v).max()) == 0.0
    assert float(cp.abs(y[1]).max()) == 0.0
    assert complex(y[2][0, 0]) == 0.0
    lo, _ = model.total_geopotential_extrema(ShallowWaterState(y))
    assert lo > 0.0

    # Topography itself is unchanged by the integration.
    assert bool(cp.array_equal(model.topography.elevation_lm,
                               elevation_before))

    # Hyperdiffusion damps the free-surface anomaly, so it must also leave
    # the lake at rest exactly (flat-bottom hyperdiffusion is unchanged).
    damped = _mountain_model(planet, hyperdiffusion_nu4=1e16)
    dot_damped = damped.tendency(state.coeffs)
    assert float(cp.abs(dot_damped).max()) == 0.0


@requires_cuda
def test_lake_at_rest_is_exactly_preserved_latlon(latlon_planet):
    _assert_lake_at_rest_preserved(latlon_planet)


@requires_cuda
def test_lake_at_rest_is_exactly_preserved_geodesic(geodesic_planet):
    _assert_lake_at_rest_preserved(geodesic_planet)


# ---------------------------------------------------------------------------
# 3. Uniform offset invariance
# ---------------------------------------------------------------------------

@requires_cuda
def test_uniform_bottom_and_surface_offset_does_not_alter_dynamics(
        latlon_planet):
    """Raising bottom and free surface by the same constant (thickness
    unchanged) is dynamically invisible: the offset lives in the phi_s
    monopole, whose Laplacian eigenvalue is exactly zero."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography

    m_base = _mountain_model(latlon_planet)
    elev = m_base.topography.elevation_lm
    elev[0, 0] += 500.0 * math.sqrt(4.0 * math.pi)  # +500 m everywhere
    m_off = ShallowWaterModel(
        latlon_planet, mean_depth=3000.0,
        topography=Topography(elev, preset="mountain",
                              parameters=m_base.topography.parameters))

    state = _nontrivial_state(latlon_planet.sh.l_max, m_base.phi0)
    t_base = m_base.tendency(state.coeffs)
    t_off = m_off.tendency(state.coeffs)
    assert bool(cp.all(t_base == t_off))
    assert m_off.topography.mean_elevation_m == pytest.approx(
        m_base.topography.mean_elevation_m + 500.0)


# ---------------------------------------------------------------------------
# 4. Mountain validity
# ---------------------------------------------------------------------------

@requires_cuda
def test_mountain_terrain_is_finite_and_band_limited(latlon_planet,
                                                     geodesic_planet):
    import cupy as cp
    from planetary_sandbox.physics.topography import Topography

    for planet in (latlon_planet, geodesic_planet):
        topo = Topography.mountain(planet, **MOUNTAIN)
        coeffs = topo.elevation_lm
        assert coeffs.shape == (planet.sh.l_max + 1, planet.sh.l_max + 1)
        assert bool(cp.isfinite(coeffs).all())
        # The projection residual is quantified and within the gate.
        assert topo.parameters["projection_residual"] <= 0.2
        # The synthesized peak is close to the requested height (Gibbs
        # ripples bounded by the residual gate).
        elev = topo.elevation_on(planet.sh)
        assert float(elev.max()) == pytest.approx(MOUNTAIN["height_m"],
                                                  rel=0.1)
        assert topo.mean_elevation_m > 0.0
        assert not topo.is_flat


@requires_cuda
def test_mountain_rejects_invalid_parameters(latlon_planet):
    from planetary_sandbox.physics.topography import (Topography,
                                                      TopographyError)

    bad_params = [
        dict(MOUNTAIN, height_m=-100.0),
        dict(MOUNTAIN, height_m=0.0),
        dict(MOUNTAIN, height_m=math.nan),
        dict(MOUNTAIN, height_m=math.inf),
        dict(MOUNTAIN, height_m=1e7),
        dict(MOUNTAIN, width_deg=0.0),
        dict(MOUNTAIN, width_deg=-10.0),
        dict(MOUNTAIN, width_deg=200.0),
        dict(MOUNTAIN, lat_deg=100.0),
        dict(MOUNTAIN, lat_deg=math.nan),
        dict(MOUNTAIN, lon_deg=500.0),
    ]
    for params in bad_params:
        with pytest.raises(TopographyError):
            Topography.mountain(latlon_planet, **params)


@requires_cuda
def test_too_narrow_mountain_fails_the_projection_gate(latlon_planet):
    from planetary_sandbox.physics.topography import (Topography,
                                                      TopographyError)
    with pytest.raises(TopographyError, match="not representable"):
        Topography.mountain(latlon_planet, height_m=1500.0, lat_deg=25.0,
                            lon_deg=60.0, width_deg=3.0)


@requires_cuda
def test_protruding_mountain_fails_before_integration(latlon_planet):
    from planetary_sandbox.physics.shallow_water import (
        ShallowWaterStateError)
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _mountain_model(latlon_planet, mean_depth=200.0)
    with pytest.raises(ShallowWaterStateError,
                       match="protrude"):
        make_swe_ic("rest", model)


@requires_cuda
def test_topography_and_model_truncations_must_match(latlon_planet):
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography

    wrong = Topography.flat(latlon_planet.sh.l_max + 3)
    with pytest.raises(ValueError, match="truncation"):
        ShallowWaterModel(latlon_planet, mean_depth=3000.0, topography=wrong)


# ---------------------------------------------------------------------------
# 5. Mass conservation in a mountain-flow run
# ---------------------------------------------------------------------------

@requires_cuda
def test_mountain_flow_run_conserves_mass_exactly(tmp_path, latlon_planet):
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic
    from planetary_sandbox.run.swe.runner import run_swe

    model = _mountain_model(latlon_planet, mean_depth=5960.0)
    state0 = make_swe_ic("williamson2", model)
    t_end_days = 0.02
    rc = run_swe(model=model, state0=state0, dt_snapshots=None,
                 t_end_days=t_end_days, out_dir=tmp_path,
                 snapshot_times=count_snapshot_times(2, t_end_days * 86400.0),
                 plots=(), snapshot_mode="count")
    assert rc == 0

    with open(tmp_path / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 3
    # The spectral mass integral (thickness only — terrain is not fluid) is
    # bit-identical across every row: exact conservation by monopole pinning.
    assert len({r["total_mass"] for r in rows}) == 1
    # Topographic diagnostics: positive thickness margin, constant terrain
    # maximum, nontrivial free-surface anomaly range.
    assert all(float(r["h_min_m"]) > 0.0 for r in rows)
    assert len({r["terrain_max_m"] for r in rows}) == 1
    assert float(rows[0]["terrain_max_m"]) == pytest.approx(1500.0, rel=0.1)
    assert all(float(r["eta_max_m"]) > float(r["eta_min_m"]) for r in rows)


# ---------------------------------------------------------------------------
# 6. Timestep refinement over terrain
# ---------------------------------------------------------------------------

@requires_cuda
def test_rk4_timestep_refinement_for_mountain_flow(latlon_planet):
    """Fixed-step RK4 solutions of a smooth mountain-flow case converge at
    ~4th order (measured ratios ~16-17 per dt halving) before spatial error
    dominates."""
    import cupy as cp
    from planetary_sandbox.run.engine import rk4_step_array
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _mountain_model(latlon_planet, mean_depth=5960.0)
    state0 = make_swe_ic("williamson2", model)
    t_end = 3600.0

    def integrate(dt):
        y = state0.coeffs.copy()
        steps = int(round(t_end / dt))
        assert steps * dt == t_end
        for _ in range(steps):
            y = rk4_step_array(model.tendency, y, 0.0, dt)
        return y

    reference = integrate(56.25)
    errors = [float(cp.abs(integrate(dt) - reference).max())
              for dt in (900.0, 450.0, 225.0)]
    assert errors[0] > errors[1] > errors[2] > 0.0
    # RK4: each halving should shrink the error ~16x; allow slack for the
    # finite reference and round-off, but demand clearly 4th-order behavior.
    assert errors[0] / errors[1] > 8.0
    assert errors[1] / errors[2] > 8.0


# ---------------------------------------------------------------------------
# 7. No host-device topography transfers in the integration loop
# ---------------------------------------------------------------------------

@requires_cuda
def test_tendency_loop_performs_no_host_device_transfers(latlon_planet,
                                                         monkeypatch):
    """After construction, stepping must not move arrays across the PCIe
    bus: no numpy->device uploads and no device->host array downloads."""
    import cupy as cp
    import numpy as np
    from planetary_sandbox.run.engine import rk4_step_array
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _mountain_model(latlon_planet)
    assert isinstance(model.phi_s_lm, cp.ndarray)  # device-resident terrain
    assert isinstance(model.topography.elevation_lm, cp.ndarray)
    state = make_swe_ic("gravity_wave", model)

    uploads: list[type] = []
    downloads: list[type] = []
    real_asarray = cp.asarray
    real_asnumpy = cp.asnumpy

    def spy_asarray(a, *args, **kwargs):
        if isinstance(a, np.ndarray):
            uploads.append(type(a))
        return real_asarray(a, *args, **kwargs)

    def spy_asnumpy(a, *args, **kwargs):
        downloads.append(type(a))
        return real_asnumpy(a, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", spy_asarray)
    monkeypatch.setattr(cp, "asnumpy", spy_asnumpy)

    y = state.coeffs
    for _ in range(3):
        y = rk4_step_array(model.tendency, y, 0.0, 60.0)
    assert uploads == []
    assert downloads == []


# ---------------------------------------------------------------------------
# 8. CFL characteristic speed uses the local fluid thickness
# ---------------------------------------------------------------------------

@requires_cuda
def test_characteristic_speed_reflects_local_thickness(latlon_planet):
    """For the lake at rest over a mountain the deepest fluid (valley floor)
    sets the gravity-wave speed: strictly above sqrt(Phi0) because the
    valleys are deeper than the mean, and equal to sqrt(g*h_max)."""
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _mountain_model(latlon_planet)
    state = make_swe_ic("rest", model)
    _, phi_max = model.total_geopotential_extrema(state)
    speed = model.max_characteristic_speed(state)
    assert speed == pytest.approx(math.sqrt(phi_max))
    assert speed > math.sqrt(model.phi0)


# ---------------------------------------------------------------------------
# 9. Mountain-flow demonstration (deterministic, stable, nontrivial)
# ---------------------------------------------------------------------------

@requires_cuda
def test_mountain_flow_demo_is_nontrivial_and_stable(latlon_planet):
    """The documented demo (Williamson-2 jet over the default mountain,
    mean depth 5960 m): inviscid, no hidden damping, positive depth, and a
    clearly nontrivial divergence/vorticity response to the terrain."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                         ShallowWaterState)
    from planetary_sandbox.physics.topography import Topography
    from planetary_sandbox.run.engine import rk4_step_array
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    topo = Topography.mountain(latlon_planet, height_m=2000.0, lat_deg=30.0,
                               lon_deg=90.0, width_deg=20.0)
    model = ShallowWaterModel(latlon_planet, mean_depth=5960.0,
                              topography=topo)
    assert model.nu4 == 0.0  # inviscid: no hidden damping
    state = make_swe_ic("williamson2", model)
    y0 = state.coeffs.copy()

    y = state.coeffs
    for _ in range(72):  # 6 hours at dt = 300 s
        y = rk4_step_array(model.tendency, y, 0.0, 300.0)
        model.validate_state(ShallowWaterState(y), context="demo step")

    # Nontrivial: the mountain forces divergence (zero in the flat steady
    # state) and deforms the vorticity field...
    assert float(cp.abs(y[1]).max()) > 10.0 * float(cp.abs(y0[1]).max() + 1e-30)
    assert float(cp.abs(y - y0).max()) > 0.0
    # ...and stable: finite, positive thickness, exact monopoles.
    assert bool(cp.isfinite(y).all())
    lo, _ = model.total_geopotential_extrema(ShallowWaterState(y))
    assert lo > 0.0
    assert float(cp.abs(y[:, 0, :]).max()) == 0.0


# ---------------------------------------------------------------------------
# 10. CLI end-to-end with a mountain (provenance + visualization)
# ---------------------------------------------------------------------------

@requires_cuda
def test_swe_cli_mountain_end_to_end(tmp_path, capsys):
    import matplotlib.image as mpimg
    from planetary_sandbox.cli.main import main

    rc = main(["run", "swe", "--backend", "gauss-latlon", "--nlat", "32",
               "--nlon", "64", "--l-max", "15", "--days", "0.005",
               "--mean-depth", "5960",
               "--topography", "mountain", "--mountain-height-m", "1500",
               "--mountain-width-deg", "25",
               "--n-snapshots", "1", "--plot", "summary",
               "--out", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "topography          mountain (h=1500 m" in out

    pointer = (tmp_path / "runs" / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    run_dir = tmp_path / "runs" / pointer
    manifest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["run_config"]["topography"] == "mountain"
    assert manifest["run_config"]["mountain_height_m"] == 1500.0
    assert "topography" in manifest["notes"]["equations"]

    # The summary gains the topography row: 2 rows x 3 columns at
    # (18, 12) inches, 200 dpi -> 2400 x 3600 pixels.
    image = mpimg.imread(run_dir / "swe_summary.png")
    assert image.shape[:2] == (2400, 3600)

    # inspect reports the terrain.
    assert main(["inspect", str(run_dir)]) == 0
    assert "mountain (h=1500.0 m" in capsys.readouterr().out
