"""Orographic isothermal rest: the PE surface-topography coupling.

Covers the pe-orographic-rest spec end to end:

* configuration/provenance (CPU): the PE topography vocabulary mirrors the
  SWE one, flat runs keep their historical config schema (and therefore
  their scientific hashes and run ids), resolved terrain parameters --
  including the elevation-to-geopotential gravity -- participate fully in
  the scientific identity, invalid terrain settings are rejected loudly,
  and the CLI/inspect contracts hold;
* model wiring (GPU): the default PE model receives exactly zero Phi_s, a
  configured terrain field reaches PrimitiveEquationsModel unchanged (in
  the repository's real-field spherical-harmonic convention, units
  m^2/s^2), and invalid fields are rejected;
* analytic balance (GPU, both backends): the ``orographic_isothermal_rest``
  state has bitwise-zero vorticity and divergence, horizontally uniform
  temperature, a positive spatially varying surface pressure satisfying
  ln(p_s)' = -Phi_s'/(R_d T0) spectrally, and a model tendency that
  vanishes to the strongest justified per-backend tolerance (roundoff on
  the Gauss lat-lon backend; the measured weak-form quadrature envelope on
  the geodesic backend -- see the tolerance notes below);
* integrated preservation (GPU, both backends): the real fixed-step RK4
  runner preserves the state (no generated winds/vorticity/divergence,
  unchanged temperature and surface pressure, roundoff-level mass drift,
  finite diagnostics, the expected artifacts);
* degeneracy: zero terrain reduces bitwise to the existing
  ``isothermal_rest`` state.

CPU-only configuration/CLI tests run everywhere; model/runner tests need
CUDA and use small Gauss lat-lon / geodesic configurations.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="CUDA/CuPy not available")

T0 = 260.0
PS0 = 101325.0
GRAVITY = 9.80616
DT = 300.0
NLEV = 4

#: Same benchmark mountain the SWE topography suite uses (band-limited at
#: l_max >= 10 within the projection-residual gate).
MOUNTAIN = dict(height_m=1500.0, lat_deg=25.0, lon_deg=60.0, width_deg=25.0)

# ---------------------------------------------------------------------------
# Balance tolerances, relative to the spectral pressure-gradient scale
# max |lap_eigs * Phi_s_lm| (the magnitude of the two individually large
# terms whose cancellation is under test). PE terrain is band-limited at
# the model's dealiased product-truncation cut (2*l_max/3): the full-T
# pressure-gradient force reaches the tendency through the 2/3-truncated
# div/curl(Z) pathway while -lap(Phi) is exact-spectral, so content above
# the cut could never cancel (measured: for full-l_max terrain the
# per-degree residual above the cut equals |lap*Phi_s| exactly). With the
# cut enforced (2026-07 measurements, this configuration):
#
# * Gauss lat-lon: the weak-form vector analysis integrates these
#   integrands exactly; residual 8.3e-25 absolute = 1.3e-15 relative
#   (pure roundoff). Asserted at 1e-13 relative (~75x headroom).
# * Geodesic: the residual is the backend's measured weak-form quadrature
#   error: 1.7e-12 absolute = 2.6e-3 relative, inside the documented
#   envelope (tests/test_pe_tendency.py GEODESIC_WEAK_FORM_RTOL = 0.02).
#   Asserted at 1e-2 relative (~4x headroom, still below the envelope).
# ---------------------------------------------------------------------------
LATLON_BALANCE_RTOL = 1e-13
GEODESIC_BALANCE_RTOL = 1e-2

# Multi-step preservation tolerances (5 fixed 300 s RK4 steps), measured
# 2026-07 and asserted with headroom. Winds in m/s; coefficient drifts are
# absolute (coefficient units of the respective prognostic block).
#
# * Gauss lat-lon (measured): max wind 4.7e-15 m/s; drifts zeta 1.0e-21,
#   delta 1.2e-21, T 6.7e-17, ln_ps 8.7e-19; mass drift exactly 0.0;
#   temperature extrema bitwise unchanged.
# * Geodesic (measured): the envelope-level momentum residual integrates
#   into max wind 9.3e-3 m/s; drifts zeta 2.5e-9, delta 2.1e-9,
#   T 1.2e-4 (1.3e-7 relative to the 921.7 monopole; grid extrema move
#   5.1e-4 K), ln_ps 1.6e-6; mass drift 3.1e-10.
LATLON_MAX_WIND = 1e-12
GEODESIC_MAX_WIND = 0.03
LATLON_MASS_TOL = 1e-14
GEODESIC_MASS_TOL = 2e-9


# ---------------------------------------------------------------------------
# Configuration and provenance (CPU)
# ---------------------------------------------------------------------------

def test_pe_topography_vocabulary_mirrors_swe():
    from planetary_sandbox.run.pe.config import PE_TOPOGRAPHIES
    from planetary_sandbox.run.swe.config import SWE_TOPOGRAPHIES

    assert sorted(PE_TOPOGRAPHIES) == sorted(SWE_TOPOGRAPHIES)


def test_flat_pe_config_keeps_historical_schema_and_hash():
    """Flat PE runs must emit exactly the historical config dict (no
    topography keys), so existing scientific hashes and run ids remain
    valid for every configuration that does not request topography."""
    from datetime import datetime, timezone
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.pe.config import PERunConfig

    cfg = PERunConfig.resolve({})
    assert cfg.topography == "flat"
    d = cfg.to_run_config_dict()
    assert "topography" not in d
    assert "gravity" not in d
    assert not any(k.startswith("mountain_") for k in d)
    # The exact historical key set, frozen.
    assert set(d) == {
        "solver", "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
        "radius_earth_units", "nlev", "sigma_interfaces", "r_dry", "cp_dry",
        "duration_days", "dt_seconds", "scenario", "temperature",
        "surface_pressure", "thermal_amplitude", "product_quadrature",
        "dt_snapshots", "out", "experiment", "overwrite", "snapshot_mode",
        "n_snapshots", "snapshot_times", "plots"}

    # Explicitly requesting the default is the same scientific identity.
    explicit_flat = PERunConfig.resolve({"topography": "flat"})
    assert (explicit_flat.scientific_config_dict()
            == cfg.scientific_config_dict())

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (make_run_id(d, now=now, commit="deadbeef")
            == make_run_id(explicit_flat.to_run_config_dict(), now=now,
                           commit="deadbeef"))


def test_orographic_scenario_is_registered():
    from planetary_sandbox.run.pe.config import PE_SCENARIOS, PERunConfig

    assert "orographic_isothermal_rest" in PE_SCENARIOS
    cfg = PERunConfig.resolve({"scenario": "orographic_isothermal_rest",
                               "topography": "mountain"})
    assert cfg.scenario == "orographic_isothermal_rest"
    assert any("orographic_isothermal_rest" in line
               for line in cfg.summary_lines())
    # The scenario is terrain-independent: it also resolves with the flat
    # default (the degenerate zero-terrain benchmark).
    flat = PERunConfig.resolve({"scenario": "orographic_isothermal_rest"})
    assert flat.topography == "flat"


def test_mountain_pe_config_resolution_and_identity():
    from datetime import datetime, timezone
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.pe.config import PERunConfig

    cfg = PERunConfig.resolve({"topography": "mountain"})
    assert cfg.mountain_height_m == 2000.0
    assert cfg.mountain_lat_deg == 30.0
    assert cfg.mountain_lon_deg == 90.0
    assert cfg.mountain_width_deg == 20.0
    assert cfg.gravity == GRAVITY
    d = cfg.to_run_config_dict()
    assert d["topography"] == "mountain"
    assert d["mountain_height_m"] == 2000.0
    assert d["gravity"] == GRAVITY

    # Every terrain parameter participates in the scientific identity: a
    # terrain run never collides with an otherwise identical flat run, and
    # distinct mountains / gravities are distinct runs.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = {make_run_id(c.to_run_config_dict(), now=now, commit="deadbeef")
           for c in (
               PERunConfig.resolve({}),
               cfg,
               PERunConfig.resolve({"topography": "mountain",
                                    "mountain_height_m": 500.0}),
               PERunConfig.resolve({"topography": "mountain",
                                    "gravity": 3.71}),
           )}
    assert len(ids) == 4

    assert any("mountain" in line for line in cfg.summary_lines())
    assert any("topography          flat" in line
               for line in PERunConfig.resolve({}).summary_lines())


def test_pe_config_rejects_invalid_topography_settings():
    from planetary_sandbox.run.pe.config import PERunConfig

    with pytest.raises(ValueError, match="unknown topography"):
        PERunConfig.resolve({"topography": "everest"})
    # Terrain parameters without the mountain preset are user errors.
    with pytest.raises(ValueError, match="require"):
        PERunConfig.resolve({"mountain_height_m": 1000.0})
    with pytest.raises(ValueError, match="require"):
        PERunConfig.resolve({"gravity": 9.81})
    with pytest.raises(ValueError, match="require"):
        PERunConfig.resolve({"topography": "flat",
                             "mountain_width_deg": 10.0})
    for bad in ({"mountain_height_m": -5.0}, {"mountain_height_m": math.nan},
                {"mountain_height_m": 1e7},
                {"mountain_width_deg": 0.0}, {"mountain_width_deg": 120.0},
                {"mountain_lat_deg": 100.0}, {"mountain_lon_deg": 500.0},
                {"gravity": 0.0}, {"gravity": -9.8},
                {"gravity": math.inf}):
        with pytest.raises(ValueError):
            PERunConfig.resolve({"topography": "mountain", **bad})


def test_pe_cli_topography_parse_contracts(capsys):
    from planetary_sandbox.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "pe", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--topography" in out
    assert "--mountain-height-m" in out
    assert "--gravity" in out

    with pytest.raises(SystemExit) as exc:
        main(["run", "pe", "--topography", "everest"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["run", "pe", "--mountain-height-m", "1000"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["run", "pe", "--gravity", "9.81"])
    assert exc.value.code == 2


def test_inspect_shows_pe_topography(tmp_path, capsys):
    from planetary_sandbox.cli.main import main

    def write_manifest(run_dir, run_config):
        run_dir.mkdir(parents=True)
        manifest = {"created_utc": "2026-07-19T00:00:00+00:00",
                    "run_id": run_dir.name, "status": "completed",
                    "run_config": run_config}
        (run_dir / "manifest.json").write_text(json.dumps(manifest),
                                               encoding="utf-8")

    write_manifest(tmp_path / "mtn", {
        "solver": "pe", "lmax": 12, "grid": "latlon", "nlat": 32,
        "nlon": 64, "scenario": "orographic_isothermal_rest",
        "topography": "mountain", "mountain_height_m": 1500.0,
        "mountain_lat_deg": 25.0, "mountain_lon_deg": 60.0,
        "mountain_width_deg": 25.0, "gravity": 9.80616})
    assert main(["inspect", str(tmp_path / "mtn")]) == 0
    out = capsys.readouterr().out
    assert "mountain (h=1500.0 m" in out

    # Old flat PE manifests (no topography key) remain readable and are
    # reported as flat (zero surface geopotential).
    write_manifest(tmp_path / "flat", {
        "solver": "pe", "lmax": 10, "grid": "geodesic", "resolution": 3,
        "scenario": "isothermal_rest"})
    assert main(["inspect", str(tmp_path / "flat")]) == 0
    out = capsys.readouterr().out
    assert "topography        flat" in out


# ---------------------------------------------------------------------------
# Model-level fixtures (GPU)
# ---------------------------------------------------------------------------

def _make_planet(grid_type="latlon", nlat=32, nlon=64, l_max=15,
                 resolution=3, day_hours=24.0):
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


def _flat_model(planet, nlev=NLEV):
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev))


def _terrain_model(planet, nlev=NLEV):
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel, product_truncation_cut)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    from planetary_sandbox.physics.topography import Topography
    # PE terrain is band-limited at the dealiased product-truncation cut
    # (the same construction the CLI performs; see cli/pe.py).
    topo = Topography.mountain(
        planet, l_cut=product_truncation_cut(planet.sh.l_max), **MOUNTAIN)
    phi_s_lm = topo.surface_geopotential_lm(GRAVITY)
    model = PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev),
                                    surface_geopotential_lm=phi_s_lm)
    return model, topo, phi_s_lm


def _orographic_state(model):
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    return make_pe_ic("orographic_isothermal_rest", model,
                      temperature=T0, surface_pressure=PS0)


def _pgf_scale(model):
    """max |lap_eigs * Phi_s_lm|: the size of the two cancelling terms."""
    import cupy as cp
    return float(cp.abs(model.lap_eigs[:, None]
                        * model.phi_surface_lm).max())


# ---------------------------------------------------------------------------
# A. Model wiring (GPU)
# ---------------------------------------------------------------------------

@requires_cuda
def test_default_pe_model_receives_exactly_zero_phi_s(latlon_planet):
    import cupy as cp
    model = _flat_model(latlon_planet)
    assert model.phi_surface_lm.shape == (model.l_max + 1, model.l_max + 1)
    assert float(cp.abs(model.phi_surface_lm).max()) == 0.0


@requires_cuda
def test_terrain_reaches_model_unchanged(latlon_planet, geodesic_planet):
    import cupy as cp
    for planet in (latlon_planet, geodesic_planet):
        model, topo, phi_s_lm = _terrain_model(planet)
        n = planet.sh.l_max + 1
        assert model.phi_surface_lm.shape == (n, n)
        assert bool(cp.isfinite(model.phi_surface_lm).all())
        # Bitwise the same spectral field that was supplied.
        assert bool(cp.all(model.phi_surface_lm == phi_s_lm))
        # Units and convention: Phi_s = g * h_s coefficient-for-coefficient,
        # and the synthesized field peaks near g * height (band-limited).
        assert bool(cp.all(phi_s_lm == GRAVITY * topo.elevation_lm))
        phi_grid = planet.sh.inv_transform(model.phi_surface_lm).real
        peak = float(phi_grid.max())
        assert peak == pytest.approx(GRAVITY * MOUNTAIN["height_m"],
                                     rel=0.25)


@requires_cuda
def test_invalid_surface_geopotential_is_rejected(latlon_planet):
    import cupy as cp
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    n = latlon_planet.sh.l_max + 1
    with pytest.raises(ValueError, match="shape"):
        PrimitiveEquationsModel(
            latlon_planet, SigmaGrid.uniform(NLEV),
            surface_geopotential_lm=cp.zeros((n + 1, n + 1),
                                             dtype=cp.complex128))
    bad = cp.zeros((n, n), dtype=cp.complex128)
    bad[2, 1] = cp.nan
    with pytest.raises(ValueError, match="NaN"):
        PrimitiveEquationsModel(latlon_planet, SigmaGrid.uniform(NLEV),
                                surface_geopotential_lm=bad)


@requires_cuda
def test_super_cut_terrain_is_rejected_loudly(latlon_planet):
    """Terrain with spectral content above the dealiased product cut is
    scientifically unsupported by the full-T PE tendency (its
    pressure-gradient force could never cancel there) and must be
    rejected loudly, never silently accepted or truncated."""
    from planetary_sandbox.physics.primitive_equations import (
        PrimitiveEquationsModel)
    from planetary_sandbox.physics.sigma_coordinate import SigmaGrid
    from planetary_sandbox.physics.topography import Topography
    topo = Topography.mountain(latlon_planet, **MOUNTAIN)  # full l_max
    phi_full = topo.surface_geopotential_lm(GRAVITY)
    with pytest.raises(ValueError, match="dealiased product truncation"):
        PrimitiveEquationsModel(latlon_planet, SigmaGrid.uniform(NLEV),
                                surface_geopotential_lm=phi_full)


# ---------------------------------------------------------------------------
# B. Analytic balance (GPU, both backends)
# ---------------------------------------------------------------------------

def _assert_orographic_state_structure(model):
    import cupy as cp
    state = _orographic_state(model)
    K = model.nlev

    # Exactly resting.
    assert float(cp.abs(state.zeta).max()) == 0.0
    assert float(cp.abs(state.delta).max()) == 0.0

    # Horizontally uniform temperature at every level: the (0,0) monopole
    # is the only nonzero temperature coefficient.
    monopole = math.sqrt(4.0 * math.pi)
    t = state.temperature.copy()
    assert bool(cp.all(t[:, 0, 0] == T0 * monopole))
    t[:, 0, 0] = 0.0
    assert float(cp.abs(t).max()) == 0.0

    # The spectral balance relation: ln(p_s)' = -Phi_s' / (R_d T0), with
    # the (0,0) reference monopole ln(p_ref).
    expected = -model.phi_surface_lm / (model.r_dry * T0)
    expected = expected.copy()
    expected[0, 0] += math.log(PS0) * monopole
    diff = float(cp.abs(state.ln_ps - expected).max())
    scale = float(cp.abs(expected).max())
    assert diff <= 1e-15 * scale

    # Surface pressure is positive and spatially nonuniform over terrain.
    ps = model.surface_pressure_on_state_grid(state)
    assert float(ps.min()) > 0.0
    assert float(ps.max()) > 1.01 * float(ps.min())
    # The pressure deficit over the mountain matches the analytic
    # hypsometric factor exp(-Phi_s / (R_d T0)) pointwise.
    phi_grid = model.sh.inv_transform(model.phi_surface_lm).real
    ps_expected = PS0 * cp.exp(-phi_grid / (model.r_dry * T0))
    assert float(cp.abs(ps - ps_expected).max()) \
        <= 1e-12 * float(ps_expected.max())


@requires_cuda
def test_orographic_state_structure_latlon(latlon_planet):
    model, _, _ = _terrain_model(latlon_planet)
    _assert_orographic_state_structure(model)


@requires_cuda
def test_orographic_state_structure_geodesic(geodesic_planet):
    model, _, _ = _terrain_model(geodesic_planet)
    _assert_orographic_state_structure(model)


def _tendency_blocks(model, state):
    import cupy as cp
    K = model.nlev
    out = model.tendency(state.coeffs)
    return {
        "zeta": float(cp.abs(out[0:K]).max()),
        "delta": float(cp.abs(out[K:2 * K]).max()),
        "temperature": float(cp.abs(out[2 * K:3 * K]).max()),
        "ln_ps": float(cp.abs(out[3 * K]).max()),
    }


@requires_cuda
def test_orographic_tendency_is_roundoff_latlon(latlon_planet):
    """Gauss lat-lon: the weak-form analysis is exact for these integrands,
    so every block vanishes to roundoff relative to the size of the two
    cancelling pressure-gradient terms; the linear (T, ln p_s) blocks are
    bitwise zero because the state is exactly resting."""
    model, _, _ = _terrain_model(latlon_planet)
    state = _orographic_state(model)
    blocks = _tendency_blocks(model, state)
    scale = _pgf_scale(model)
    assert scale > 0.0
    assert blocks["temperature"] == 0.0
    assert blocks["ln_ps"] == 0.0
    assert blocks["zeta"] <= LATLON_BALANCE_RTOL * scale
    assert blocks["delta"] <= LATLON_BALANCE_RTOL * scale


@requires_cuda
def test_orographic_tendency_within_envelope_geodesic(geodesic_planet):
    """Geodesic: the residual is the backend's measured weak-form quadrature
    envelope (the same envelope every nonlinear geodesic PE evaluation
    carries); the linear (T, ln p_s) blocks are still bitwise zero."""
    model, _, _ = _terrain_model(geodesic_planet)
    state = _orographic_state(model)
    blocks = _tendency_blocks(model, state)
    scale = _pgf_scale(model)
    assert scale > 0.0
    assert blocks["temperature"] == 0.0
    assert blocks["ln_ps"] == 0.0
    assert blocks["zeta"] <= GEODESIC_BALANCE_RTOL * scale
    assert blocks["delta"] <= GEODESIC_BALANCE_RTOL * scale


@requires_cuda
def test_orographic_imbalance_localizes_to_momentum_blocks(latlon_planet):
    """Block-level attribution: the only terms that could fail to cancel
    live in the momentum (zeta/delta) pathway through curl/div of
    Z = R_d T grad(ln p_s); the thermodynamic and mass tendencies are
    built from bitwise-zero winds and G_k and must vanish bitwise."""
    import cupy as cp
    model, _, _ = _terrain_model(latlon_planet)
    state = _orographic_state(model)
    fields = model._tendency_product_fields(state.coeffs)
    # Winds and the continuity integrand are bitwise zero at rest.
    assert float(cp.abs(fields["u"]).max()) == 0.0
    assert float(cp.abs(fields["v"]).max()) == 0.0
    assert float(cp.abs(fields["g_full"]).max()) == 0.0
    assert float(cp.abs(fields["sigma_dot"]).max()) == 0.0
    t_dot, lnps_dot = model._thermo_mass_tendencies(state.coeffs, fields)
    assert float(cp.abs(t_dot).max()) == 0.0
    assert float(cp.abs(lnps_dot).max()) == 0.0


# ---------------------------------------------------------------------------
# C. Integrated preservation through the real runner (GPU, both backends)
# ---------------------------------------------------------------------------

def _run_orographic(model, out_dir, *, n_steps=5, n_snapshots=6):
    import pathlib
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.pe.runner import run_pe
    state = _orographic_state(model)
    t_end = n_steps * DT
    times = count_snapshot_times(n_snapshots, t_end)
    run_pe(model, state, dt_seconds=DT, t_end_days=t_end / 86400.0,
           out_dir=pathlib.Path(out_dir), snapshot_times=times,
           snapshot_mode="count",
           dt_snapshots=t_end / (n_snapshots - 1),
           plots=("diagnostics",), scenario="orographic_isothermal_rest")
    return times


def _block_drifts(coeffs, initial, nlev):
    K = nlev
    drift = np.abs(coeffs - initial[None])
    return {
        "zeta": float(drift[:, 0:K].max()),
        "delta": float(drift[:, K:2 * K].max()),
        "temperature": float(drift[:, 2 * K:3 * K].max()),
        "ln_ps": float(drift[:, 3 * K].max()),
    }


def _assert_preserved(model, tmp_path, *, wind_tol, zeta_tol, delta_tol,
                      t_tol, lnps_tol, mass_tol, t_extrema_tol):
    import cupy as cp
    times = _run_orographic(model, tmp_path)
    coeffs = np.load(tmp_path / "pe_coeffs.npy")
    stored = np.load(tmp_path / "pe_snapshot_times.npy")
    assert np.array_equal(stored, np.asarray(times))

    initial = cp.asnumpy(_orographic_state(model).coeffs)
    drifts = _block_drifts(coeffs, initial, model.nlev)
    assert drifts["zeta"] <= zeta_tol, drifts
    assert drifts["delta"] <= delta_tol, drifts
    assert drifts["temperature"] <= t_tol, drifts
    assert drifts["ln_ps"] <= lnps_tol, drifts

    data = np.atleast_1d(np.genfromtxt(
        tmp_path / "diagnostics" / "timeseries.csv",
        delimiter=",", names=True))
    assert np.all(data["max_wind_ms"] <= wind_tol)
    assert np.all(np.abs(data["mass_rel_drift"]) <= mass_tol)
    assert np.all(data["ps_min"] > 0.0)
    assert np.all(data["t_min"] > 0.0)
    for col in data.dtype.names:
        values = data[col]
        if col == "courant":
            values = values[data["dt_s"] > 0]
        assert np.all(np.isfinite(values)), f"non-finite diagnostic {col}"
    # Temperature extrema unchanged to the per-backend tolerance
    # (bitwise on the Gauss lat-lon backend).
    assert np.all(np.abs(data["t_min"] - data["t_min"][0])
                  <= t_extrema_tol)
    assert np.all(np.abs(data["t_max"] - data["t_max"][0])
                  <= t_extrema_tol)


@requires_cuda
def test_orographic_rest_preserved_by_runner_latlon(latlon_planet, tmp_path):
    """Five real fixed RK4 steps on the Gauss lat-lon backend: everything
    is preserved to floating-point roundoff (tolerances are measured
    roundoff levels with generous headroom, in absolute coefficient
    units); mass drift is exactly zero and the temperature extrema are
    bitwise unchanged."""
    model, _, _ = _terrain_model(latlon_planet)
    _assert_preserved(model, tmp_path,
                      wind_tol=LATLON_MAX_WIND,
                      zeta_tol=1e-16, delta_tol=1e-16,
                      t_tol=1e-13, lnps_tol=1e-15,
                      mass_tol=LATLON_MASS_TOL, t_extrema_tol=0.0)


@requires_cuda
def test_orographic_rest_preserved_by_runner_geodesic(geodesic_planet,
                                                      tmp_path):
    """Five real fixed RK4 steps on the geodesic backend: preservation is
    bounded by the backend's measured weak-form quadrature envelope (the
    envelope-level momentum residual integrates into a mm/s-scale
    spurious divergent flow and second-order T / ln p_s responses;
    tolerances are the measured values in the module header with 3-6x
    headroom)."""
    model, _, _ = _terrain_model(geodesic_planet)
    _assert_preserved(model, tmp_path,
                      wind_tol=GEODESIC_MAX_WIND,
                      zeta_tol=1e-8, delta_tol=1e-8,
                      t_tol=5e-4, lnps_tol=1e-5,
                      mass_tol=GEODESIC_MASS_TOL, t_extrema_tol=2e-3)


# ---------------------------------------------------------------------------
# D. Degenerate cases
# ---------------------------------------------------------------------------

@requires_cuda
def test_zero_terrain_reduces_to_isothermal_rest(latlon_planet,
                                                 geodesic_planet):
    """With zero Phi_s the orographic preset is bitwise the existing
    isothermal_rest state, so every existing exact-rest guarantee
    transfers unchanged."""
    import cupy as cp
    from planetary_sandbox.run.pe.initial_conditions import make_pe_ic
    for planet in (latlon_planet, geodesic_planet):
        model = _flat_model(planet)
        orographic = make_pe_ic("orographic_isothermal_rest", model,
                                temperature=T0, surface_pressure=PS0)
        rest = make_pe_ic("isothermal_rest", model,
                          temperature=T0, surface_pressure=PS0)
        assert bool(cp.all(orographic.coeffs == rest.coeffs))
        # And the exact-rest tendency stays exactly zero.
        assert float(cp.abs(model.tendency(orographic.coeffs)).max()) == 0.0


@requires_cuda
def test_snapshot_header_terrain_note(latlon_planet):
    """The per-snapshot figure header gains one terrain-context line for
    non-flat runs (derived from the model's own resolved Phi_s); flat
    runs keep the historical header byte-for-byte (note is None)."""
    from planetary_sandbox.run.pe.snapshot_visualization import _terrain_note
    model, _, _ = _terrain_model(latlon_planet)
    note = _terrain_note(model)
    assert note is not None and "Phi_s" in note
    assert _terrain_note(_flat_model(latlon_planet)) is None


# ---------------------------------------------------------------------------
# E. CLI end-to-end (provenance + artifacts)
# ---------------------------------------------------------------------------

@requires_cuda
def test_pe_cli_orographic_end_to_end(tmp_path, capsys):
    from planetary_sandbox.cli.main import main

    rc = main(["run", "pe", "--backend", "gauss-latlon", "--nlat", "32",
               "--nlon", "64", "--l-max", "15", "--levels", str(NLEV),
               "--scenario", "orographic_isothermal_rest",
               "--topography", "mountain",
               "--mountain-height-m", "1500", "--mountain-lat-deg", "25",
               "--mountain-lon-deg", "60", "--mountain-width-deg", "25",
               "--dt-seconds", "100", "--days", "0.005",
               "--n-snapshots", "2", "--plot", "summary",
               "--out", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mountain" in out

    pointer = (tmp_path / "runs" / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    run_dir = tmp_path / "runs" / pointer
    manifest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "completed"
    rc_cfg = manifest["run_config"]
    assert rc_cfg["topography"] == "mountain"
    assert rc_cfg["mountain_height_m"] == 1500.0
    assert rc_cfg["gravity"] == GRAVITY
    assert "topography" in json.dumps(manifest["notes"]).lower()

    assert (run_dir / "pe_coeffs.npy").exists()
    assert (run_dir / "pe_summary.png").exists()
    assert (run_dir / "snapshots" / "physical").is_dir()

    # inspect reports the terrain.
    assert main(["inspect", str(run_dir)]) == 0
    assert "mountain (h=1500.0 m" in capsys.readouterr().out
