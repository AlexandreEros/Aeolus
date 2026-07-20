"""Williamson et al. (1992) test case 5: zonal flow over an isolated mountain.

Benchmark authority: Williamson, Drake, Hack, Jakob & Swarztrauber (1992),
"A standard test set for numerical approximations to the shallow water
equations in spherical geometry", J. Comput. Phys. 102, 211-224, section
on test case 5.

Canonical constants (all SI):

    a      = 6.37122e6 m        planetary radius (perfect sphere)
    Omega  = 7.292e-5  s^-1     rotation rate
    g      = 9.80616   m/s^2    gravity
    u0     = 20        m/s      zonal wind amplitude
    h0     = 5960      m        reference FLUID-THICKNESS depth
    hs0    = 2000      m        mountain peak height
    R0     = pi/9      rad      cone support radius (coordinate-plane)
    lat_c  = pi/6      rad      cone center latitude  (30 N)
    lon_c  = 3*pi/2    rad      cone center longitude (270 E == -90 E)

The initial state is the Williamson-2-shaped wind/thickness pair with
u0 = 20 m/s:

    u    = u0 cos(lat),  v = 0
    h    = h0 - (C/g) sin^2(lat),      C = a*Omega*u0 + u0^2/2

h is the FLUID THICKNESS (depth), NOT the free-surface height: the mountain
raises the initial free surface  Phi0 + phi + phi_s  over the cone, which is
exactly the canonical topographic forcing.  The terrain-compensating
free-surface construction used by the `williamson2` scenario
(phi = phi_fs' - phi_s') is explicitly NOT Williamson 5; a direct
regression below fails if anyone ever "fixes" W5 to use it.

The cone uses COORDINATE-PLANE angular distance
r = min(R0, sqrt(dlambda^2 + dlat^2)) — not great-circle distance — and is
not band-limited; its measured projection residuals (quadrature-weighted
relative L2 between the analytic cone and its band-limited synthesis) are:

    Gauss-Legendre  l_max=15 (32x64):   0.0895
    Gauss-Legendre  l_max=21 (32x64):   0.0706
    Gauss-Legendre  l_max=31 (48x96):   0.0406
    Gauss-Legendre  l_max=42 (64x128):  0.0249
    Gauss-Legendre  l_max=63 (96x192):  0.0121
    geodesic res4   l_max=21:           0.0643
    geodesic res3   l_max=10:           0.3276  (rejected by the cone gate)

Tolerances below are those measurements with modest headroom, tight enough
that a silently substituted Gaussian / great-circle cone or a broken
projection fails immediately.
"""
from __future__ import annotations

import math
import os

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="CUDA/CuPy not available")

# Canonical constants, restated locally so a drive-by edit of the source
# constants cannot silently satisfy these tests.
A_CANON = 6.37122e6
OMEGA_CANON = 7.292e-5
GRAVITY = 9.80616
U0 = 20.0
H0 = 5960.0
HS0 = 2000.0
R0 = math.pi / 9.0
LATC = math.pi / 6.0
LONC = 3.0 * math.pi / 2.0
C_CANON = A_CANON * OMEGA_CANON * U0 + 0.5 * U0 * U0
MEAN_DEPTH_CANON = H0 - C_CANON / (3.0 * GRAVITY)
DAY_HOURS_CANON = 2.0 * math.pi / OMEGA_CANON / 3600.0


def _make_w5_planet(grid_type="latlon", nlat=32, nlon=64, l_max=21,
                    resolution=4):
    """Ideal-sphere planet with the exact canonical radius and rotation."""
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    params = PlanetaryParameters.ideal_sphere(
        radius_m=A_CANON, sidereal_day_s=DAY_HOURS_CANON * 3600.0)
    return Planet.generate(
        params=params, grid_type=grid_type, nlat=nlat, nlon=nlon,
        l_max=l_max, grid_resolution=resolution)


def _make_w5_model(planet, *, cone=True, mean_depth=MEAN_DEPTH_CANON):
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography
    topo = Topography.williamson5_cone(planet) if cone else None
    return ShallowWaterModel(planet, gravity=GRAVITY, mean_depth=mean_depth,
                             topography=topo)


# ===========================================================================
# Ideal-sphere planetary parameters (the canonical-radius seam)
# ===========================================================================

def test_ideal_sphere_parameters_are_exact():
    from planetary_sandbox.planet import PlanetaryParameters

    p = PlanetaryParameters.ideal_sphere(
        radius_m=A_CANON, sidereal_day_s=DAY_HOURS_CANON * 3600.0)
    # The benchmark sphere is perfect: no oblateness-derived shrinkage.
    assert p.radius == A_CANON
    assert p.equatorial_radius == A_CANON
    assert p.polar_radius == A_CANON
    assert p.oblateness == 0.0
    # Omega round-trips exactly through day_hours (verified float identity).
    assert p.angular_velocity == OMEGA_CANON


# ===========================================================================
# A. Canonical conical mountain: analytic definition
# ===========================================================================

@requires_cuda
def test_cone_analytic_center_peak_and_compact_support():
    import cupy as cp
    from planetary_sandbox.physics.topography import williamson5_cone_elevation

    lat = cp.asarray([LATC, LATC, LATC, 0.0, -LATC])
    lon = cp.asarray([LONC, LONC + 0.5 * R0, LONC + 1.5 * R0, LONC, LONC])
    hs = williamson5_cone_elevation(lat, lon)
    assert float(hs[0]) == HS0                      # exact peak at center
    assert float(hs[1]) == pytest.approx(HS0 * 0.5, abs=1e-9)
    assert float(hs[2]) == 0.0                      # compact support
    # (0, LONC): coordinate distance = LATC = pi/6 > R0 = pi/9 -> outside.
    assert float(hs[3]) == 0.0
    assert float(hs[4]) == 0.0


@requires_cuda
def test_cone_uses_coordinate_plane_distance_not_great_circle():
    """At the center latitude, a zonal offset dlambda has coordinate-plane
    distance |dlambda| but great-circle distance ~ |dlambda|*cos(lat_c).
    The canonical cone must follow the former exactly."""
    import cupy as cp
    from planetary_sandbox.physics.topography import williamson5_cone_elevation

    dl = 0.8 * R0
    hs = float(williamson5_cone_elevation(
        cp.asarray([LATC]), cp.asarray([LONC + dl]))[0])
    coordinate_value = HS0 * (1.0 - dl / R0)                 # = 0.2*hs0
    great_circle_d = math.acos(
        math.sin(LATC) ** 2 + math.cos(LATC) ** 2 * math.cos(dl))
    great_circle_value = HS0 * (1.0 - great_circle_d / R0)   # ~ 0.31*hs0
    assert hs == pytest.approx(coordinate_value, abs=1e-9)
    assert abs(hs - great_circle_value) > 0.05 * HS0


@requires_cuda
def test_cone_longitude_wrapping():
    import cupy as cp
    from planetary_sandbox.physics.topography import williamson5_cone_elevation

    lat = cp.asarray([LATC, LATC, LATC, LATC])
    # -pi/2 and 3*pi/2 and 7*pi/2 are the same meridian; a point slightly
    # "west" across the branch must match its unwrapped twin.
    lon = cp.asarray([-math.pi / 2.0, LONC, LONC + 2.0 * math.pi,
                      -math.pi / 2.0 - 0.5 * R0])
    hs = williamson5_cone_elevation(lat, lon)
    assert float(hs[0]) == HS0
    assert float(hs[1]) == HS0
    assert float(hs[2]) == HS0
    ref = float(williamson5_cone_elevation(
        cp.asarray([LATC]), cp.asarray([LONC - 0.5 * R0]))[0])
    assert float(hs[3]) == pytest.approx(ref, abs=1e-9)


@requires_cuda
def test_cone_is_not_a_gaussian():
    """A Gaussian is smooth and strictly positive everywhere; the canonical
    cone is exactly zero outside R0 and linear in r inside."""
    import cupy as cp
    from planetary_sandbox.physics.topography import williamson5_cone_elevation

    r_frac = cp.asarray([0.25, 0.5, 0.75])
    lat = LATC + r_frac * R0
    lon = cp.full(3, LONC)
    hs = williamson5_cone_elevation(lat, lon)
    # Exact linearity in r (a Gaussian fails this at O(1)).
    expected = HS0 * (1.0 - r_frac)
    assert float(cp.abs(hs - cp.asarray(expected)).max()) < 1e-9


# ===========================================================================
# B. Cone projection: measured characterization, per backend
# ===========================================================================

@requires_cuda
def test_cone_projection_latlon_measured_envelope():
    from planetary_sandbox.physics.topography import Topography

    planet = _make_w5_planet()          # GL 32x64, l_max=21
    topo = Topography.williamson5_cone(planet)
    assert topo.preset == "williamson5_cone"
    assert not topo.is_flat
    p = topo.parameters
    assert p["height_m"] == HS0
    assert p["radius_rad"] == pytest.approx(R0, abs=0.0)
    assert p["lat_center_deg"] == 30.0
    assert p["lon_center_deg"] == -90.0
    # Measured residual 0.0706; window catches any substituted terrain.
    assert 0.05 <= p["projection_residual"] <= 0.09
    # Gibbs ripples and cusp undershoot (measured min -28 m, max 1754 m).
    assert -60.0 <= p["elevation_min_m"] <= -5.0
    assert 1600.0 <= p["elevation_max_m"] <= 1900.0
    assert p["peak_error_m"] == pytest.approx(
        HS0 - p["elevation_max_m"], abs=1e-9)


@requires_cuda
def test_cone_projection_geodesic_measured_envelope():
    from planetary_sandbox.physics.topography import Topography

    planet = _make_w5_planet(grid_type="geodesic", resolution=4, l_max=21)
    topo = Topography.williamson5_cone(planet)
    # Measured residual 0.0643 on the geodesic res-4 transform.
    assert 0.04 <= topo.parameters["projection_residual"] <= 0.09


@requires_cuda
def test_cone_projection_converges_with_resolution():
    """The nonsmooth cone is not band-limited; its projection residual must
    fall monotonically as l_max rises (measured 0.0895 -> 0.0249)."""
    from planetary_sandbox.physics.topography import Topography

    residuals = []
    for l_max, nlat, nlon in ((15, 32, 64), (31, 48, 96), (42, 64, 128)):
        planet = _make_w5_planet(l_max=l_max, nlat=nlat, nlon=nlon)
        topo = Topography.williamson5_cone(planet)
        residuals.append(topo.parameters["projection_residual"])
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[2] <= 0.03


@requires_cuda
def test_cone_rejects_qualitatively_degraded_projection():
    """geodesic res3/l_max=10 measures residual 0.33 — no longer a faithful
    cone. The benchmark-specific gate (0.25) rejects it loudly."""
    from planetary_sandbox.physics.topography import (Topography,
                                                      TopographyError)

    planet = _make_w5_planet(grid_type="geodesic", resolution=3, l_max=10)
    with pytest.raises(TopographyError, match="not representable"):
        Topography.williamson5_cone(planet)


@requires_cuda
def test_cone_backend_projection_difference_is_measured():
    """The cone is analyzed independently per backend; at matched l_max=21
    the coefficient sets differ by ~1e-2 (measured 1.03e-2). Pin the order
    of magnitude so the backend dependence stays characterized."""
    import cupy as cp
    from planetary_sandbox.physics.topography import Topography

    t_lat = Topography.williamson5_cone(
        _make_w5_planet(l_max=21, nlat=48, nlon=96))
    t_geo = Topography.williamson5_cone(
        _make_w5_planet(grid_type="geodesic", resolution=4, l_max=21))
    a, b = t_lat.elevation_lm, t_geo.elevation_lm
    rel = float(cp.linalg.norm(a - b) / cp.linalg.norm(a))
    assert 1e-3 <= rel <= 5e-2


# ===========================================================================
# Configuration, canonical resolution, and provenance (CPU, import-light)
# ===========================================================================

def test_w5_config_resolves_canonical_values():
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({"scenario": "williamson5"})
    assert cfg.scenario == "williamson5"
    assert cfg.topography == "williamson5_cone"
    assert cfg.gravity == GRAVITY
    assert cfg.day_hours == DAY_HOURS_CANON
    assert cfg.radius_earth_units == 1.0
    # H = h0 - C/(3g), exactly (same float expression as the constants).
    assert cfg.mean_depth_m == MEAN_DEPTH_CANON
    assert cfg.w5_canonical()
    assert cfg.mountain_height_m is None


def test_w5_config_dict_carries_every_defining_choice():
    from planetary_sandbox.run.swe.config import SWERunConfig

    d = SWERunConfig.resolve({"scenario": "williamson5"}).to_run_config_dict()
    assert d["scenario"] == "williamson5"
    assert d["topography"] == "williamson5_cone"
    assert d["w5_u0_ms"] == U0
    assert d["w5_cone_height_m"] == HS0
    assert d["w5_cone_radius_rad"] == R0
    assert d["w5_cone_lat_deg"] == 30.0
    assert d["w5_cone_lon_deg"] == -90.0
    assert d["w5_canonical"] is True
    # The projection/truncation policy is part of the identity.
    assert d["w5_projection"] == "state-grid-analysis-full-truncation"
    # Gaussian-mountain keys must never appear on a W5 run.
    assert not any(k.startswith("mountain_") for k in d)


def test_w5_run_identity_is_distinct_and_canonicality_hashes():
    from datetime import datetime, timezone
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.swe.config import SWERunConfig

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def rid(cfg):
        return make_run_id(cfg.to_run_config_dict(), now=now,
                           commit="deadbeef")

    w5 = SWERunConfig.resolve({"scenario": "williamson5"})
    w2_flat = SWERunConfig.resolve({})
    w2_mtn = SWERunConfig.resolve({"topography": "mountain"})
    w5_derived = SWERunConfig.resolve({"scenario": "williamson5",
                                       "mean_depth_m": 3000.0})
    ids = {rid(w5), rid(w2_flat), rid(w2_mtn), rid(w5_derived)}
    assert len(ids) == 4
    assert not w5_derived.w5_canonical()
    assert w5_derived.to_run_config_dict()["w5_canonical"] is False


def test_w5_noncanonical_overrides_are_reported_not_silently_overridden():
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({"scenario": "williamson5",
                                "mean_depth_m": 3000.0,
                                "day_hours": 24.0})
    # Explicit values are honored...
    assert cfg.mean_depth_m == 3000.0
    assert cfg.day_hours == 24.0
    # ...and loudly labeled as W5-derived, not canonical.
    assert not cfg.w5_canonical()
    text = "\n".join(cfg.summary_lines())
    assert "NONCANONICAL" in text or "noncanonical" in text

    canonical_text = "\n".join(
        SWERunConfig.resolve({"scenario": "williamson5"}).summary_lines())
    assert "canonical" in canonical_text
    assert "Williamson" in canonical_text or "williamson5" in canonical_text


def test_w5_rejects_conflicting_terrain_settings():
    from planetary_sandbox.run.swe.config import SWERunConfig

    # W5 owns its terrain: any explicit topography selection conflicts.
    for topo in ("flat", "mountain"):
        with pytest.raises(ValueError, match="owns its terrain"):
            SWERunConfig.resolve({"scenario": "williamson5",
                                  "topography": topo})
    with pytest.raises(ValueError, match="owns its terrain"):
        SWERunConfig.resolve({"scenario": "williamson5",
                              "mountain_height_m": 1000.0})
    # The cone is benchmark-owned, not a user-facing preset.
    with pytest.raises(ValueError, match="benchmark-owned"):
        SWERunConfig.resolve({"topography": "williamson5_cone"})
    # And it cannot be paired with another scenario at the dataclass level.
    with pytest.raises(ValueError):
        SWERunConfig(scenario="williamson2", topography="williamson5_cone",
                     dt_snapshots=21600.0, snapshot_mode="count",
                     n_snapshots=5)


def test_w5_leaves_existing_identities_unchanged():
    """Flat and Gaussian-mountain config dicts must not grow W5 keys."""
    from planetary_sandbox.run.swe.config import SWERunConfig

    for explicit in ({}, {"topography": "mountain"}):
        d = SWERunConfig.resolve(explicit).to_run_config_dict()
        assert not any(k.startswith("w5_") for k in d)


@requires_cuda
def test_w5_config_constants_match_physics_cone():
    """The import-light config constants must stay in sync with the
    CuPy-importing physics module (duplicated deliberately)."""
    from planetary_sandbox.physics import topography as phys
    from planetary_sandbox.run.swe import config as swe_config

    assert swe_config.W5_CONE_HEIGHT_M == phys.W5_CONE_HEIGHT_M
    assert swe_config.W5_CONE_RADIUS_RAD == phys.W5_CONE_RADIUS_RAD
    assert swe_config.W5_U0_MS == U0
    assert swe_config.W5_RADIUS_M == A_CANON
    assert swe_config.W5_OMEGA == OMEGA_CANON
    assert swe_config.W5_GRAVITY == GRAVITY
    assert swe_config.W5_H0_M == H0
    assert swe_config.W5_MEAN_DEPTH_M == MEAN_DEPTH_CANON
    assert swe_config.W5_DAY_HOURS == DAY_HOURS_CANON


# ===========================================================================
# CLI executor: canonical planet construction, provenance, inspect
# ===========================================================================

def test_w5_executor_builds_exact_ideal_sphere_params():
    from planetary_sandbox.cli.swe import _w5_planet_params
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({"scenario": "williamson5"})
    params = _w5_planet_params(cfg)
    assert params.radius == A_CANON
    assert params.equatorial_radius == A_CANON
    assert params.oblateness == 0.0
    assert params.angular_velocity == OMEGA_CANON

    scaled = SWERunConfig.resolve({"scenario": "williamson5",
                                   "radius_earth_units": 2.0})
    assert _w5_planet_params(scaled).radius == 2.0 * A_CANON


@requires_cuda
def test_w5_cli_end_to_end_provenance_and_inspect(tmp_path, capsys):
    import json
    from planetary_sandbox.cli.main import main

    rc = main(["run", "swe", "--scenario", "williamson5",
               "--backend", "gauss-latlon", "--nlat", "32", "--nlon", "64",
               "--l-max", "21", "--days", "0.005", "--n-snapshots", "1",
               "--no-plots", "--out", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Williamson 5" in out and "canonical" in out

    pointer = (tmp_path / "runs" / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    assert "williamson5" in pointer
    run_dir = tmp_path / "runs" / pointer
    manifest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "completed"
    rcfg = manifest["run_config"]
    assert rcfg["scenario"] == "williamson5"
    assert rcfg["topography"] == "williamson5_cone"
    assert rcfg["w5_canonical"] is True
    assert rcfg["w5_cone_height_m"] == HS0
    # The benchmark note records canonicality AND the measured projection.
    note = manifest["notes"]["benchmark"]
    assert "Williamson" in note and "canonical" in note
    assert "residual" in note

    # `aeolus inspect` must not misreport the cone as a flat bottom.
    rc = main(["inspect", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Williamson-5 cone" in out


# ===========================================================================
# A. Exact initial-condition setup
# ===========================================================================

def test_w5_scenario_registry_in_sync():
    from planetary_sandbox.run.swe.config import SWE_SCENARIOS
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    from planetary_sandbox.run.swe.initial_conditions import (
        SWE_INITIAL_CONDITIONS)
    assert set(SWE_SCENARIOS) == set(SWE_INITIAL_CONDITIONS)
    assert "williamson5" in SWE_SCENARIOS


@requires_cuda
def test_w5_model_uses_exact_canonical_planet():
    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    assert model.R == A_CANON
    assert model.Omega == OMEGA_CANON
    assert model.gravity == GRAVITY
    assert model.mean_depth == MEAN_DEPTH_CANON
    assert model.phi0 == GRAVITY * MEAN_DEPTH_CANON


@requires_cuda
def test_w5_ic_spectral_construction_is_exact():
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    model = _make_w5_model(_make_w5_planet())
    state = make_swe_ic("williamson5", model)

    zeta, delta, phi = state.coeffs[0], state.coeffs[1], state.coeffs[2]
    # delta is exactly zero.
    assert float(cp.abs(delta).max()) == 0.0
    # zeta is the pure (1,0) mode: (2*u0/a)*sqrt(4*pi/3).
    expect_zeta = (2.0 * U0 / A_CANON) * math.sqrt(4.0 * math.pi / 3.0)
    assert complex(zeta[1, 0]) == pytest.approx(expect_zeta, rel=0, abs=0)
    z = zeta.copy()
    z[1, 0] = 0.0
    assert float(cp.abs(z).max()) == 0.0
    # phi is the pure (2,0) mode: -(4*C/3)*sqrt(pi/5), C from the exact
    # canonical constants (a and Omega are exact on the ideal sphere).
    expect_phi = -(4.0 * C_CANON / 3.0) * math.sqrt(math.pi / 5.0)
    assert complex(phi[2, 0]) == pytest.approx(expect_phi, rel=0, abs=0)
    p = phi.copy()
    p[2, 0] = 0.0
    assert float(cp.abs(p).max()) == 0.0
    # The phi monopole (mass anomaly) is exactly zero.
    assert complex(phi[0, 0]) == 0j


@requires_cuda
def test_w5_ic_reconstructs_canonical_wind_and_thickness():
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)

    lat = cp.asarray(planet.grid.point_latitudes, dtype=cp.float64)
    u, v = model.wind_on_state_grid(state)
    # u = u0*cos(lat), v = 0, to Gauss-Legendre transform accuracy.
    assert float(cp.abs(u - U0 * cp.cos(lat)).max()) < 1e-9
    assert float(cp.abs(v).max()) < 1e-9
    # Fluid thickness h = (Phi0 + phi)/g = h0 - (C/g) sin^2(lat).
    h = (model.phi0 + planet.sh.inv_transform(state.coeffs[2]).real
         ) / model.gravity
    h_ref = H0 - (C_CANON / GRAVITY) * cp.sin(lat) ** 2
    assert float(cp.abs(h - h_ref).max()) < 1e-9


@requires_cuda
def test_w5_ic_thickness_is_terrain_independent():
    """The SAME thickness/wind pair must come out whether or not the cone
    is present: W5 raises the free surface over the mountain rather than
    carving the mountain out of the fluid."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    with_cone = make_swe_ic("williamson5", _make_w5_model(planet))
    without = make_swe_ic("williamson5", _make_w5_model(planet, cone=False))
    assert bool(cp.all(with_cone.coeffs == without.coeffs))


@requires_cuda
def test_w5_regression_never_free_surface_compensated():
    """MUST FAIL if W5 is ever changed to the williamson2-style
    terrain-compensating construction phi = phi_balanced - phi_s'."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)
    phi = state.coeffs[2]

    # The compensated variant would inject the cone's rich spectrum into
    # phi. Locate the cone's largest off-(2,0) coefficient and demand the
    # state carries EXACTLY zero there...
    phi_s = model.phi_s_anom_lm.copy()
    phi_s[2, 0] = 0.0
    idx = int(cp.abs(phi_s).argmax())
    l_big, m_big = divmod(idx, phi_s.shape[1])
    assert float(cp.abs(phi_s[l_big, m_big])) > 0.0   # cone truly present
    assert complex(phi[l_big, m_big]) == 0j
    # ...and that phi differs from the compensated construction by exactly
    # the cone's surface-geopotential anomaly.
    compensated = phi - model.phi_s_anom_lm
    diff = float(cp.linalg.norm(phi - compensated))
    assert diff == pytest.approx(
        float(cp.linalg.norm(model.phi_s_anom_lm)), rel=1e-12)
    assert diff > 0.0


@requires_cuda
def test_w5_initial_free_surface_is_raised_over_the_mountain():
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)

    phi_grid = planet.sh.inv_transform(state.coeffs[2]).real
    surface = (model.phi0 + phi_grid
               + model.surface_geopotential_on_state_grid()) / model.gravity
    zonal_only = (model.phi0 + phi_grid) / model.gravity
    bump = surface - zonal_only
    # The free-surface bump IS the (band-limited) cone, peak ~1754 m here.
    assert float(bump.max()) > 1500.0
    assert float(bump.max()) < HS0


# ===========================================================================
# C. Short-run dynamics: the forcing originates from phi_s
# ===========================================================================

def _integrate_fixed_cfl(planet, model, state, days):
    """RK4-integrate for `days` at the initial advective+gravity-wave CFL."""
    from planetary_sandbox.run.engine import (advective_cfl_timestep,
                                              rk4_step_array)

    length_scale = getattr(planet.grid, "cfl_length_scale", None)
    dt = advective_cfl_timestep(
        length_scale, model.max_characteristic_speed(state))
    n_steps = int(math.ceil(days * 86400.0 / dt))
    dt = days * 86400.0 / n_steps
    y = state.coeffs.copy()
    for i in range(n_steps):
        y = rk4_step_array(model.tendency, y, i * dt, dt)
    return y, dt, n_steps


@requires_cuda
def test_w5_initial_forcing_originates_from_phi_s():
    """The initial tendency must be exactly the mountain term: the balanced
    wind/thickness pair alone is steady, so dot(delta) = -lap(phi_s) to
    cancellation accuracy and dot(zeta), dot(phi) remain ~0. With the cone
    removed (hs0 = 0), every tendency vanishes — the topographic response
    is entirely phi_s-driven, not an unbalanced construction."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)
    dot = model.tendency(state.coeffs)

    lap_phi_s = model.lap_eigs[:, None] * model.phi_s_lm
    forcing = float(cp.abs(lap_phi_s).max())
    assert forcing > 0.0
    # dot(delta) == -lap(phi_s) up to the balanced-pair cancellation floor
    # (measured ~1e-12 relative on Gauss-Legendre).
    residual = float(cp.abs(dot[1] + lap_phi_s).max()) / forcing
    assert residual < 1e-9
    # zeta/phi tendencies stay at the balanced floor: the mountain enters
    # ONLY the divergence equation at t=0 (measured ~1e-16 relative).
    assert float(cp.abs(dot[0]).max()) < 1e-9 * forcing
    assert float(cp.abs(dot[2]).max()) < 1e-9 * forcing

    # Null experiment: no cone -> no response (the state is steady).
    flat_model = _make_w5_model(planet, cone=False)
    dot_flat = flat_model.tendency(state.coeffs)
    assert float(cp.abs(dot_flat).max()) < 1e-9 * forcing


@requires_cuda
def test_w5_short_run_latlon_valid_and_conserving():
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    from planetary_sandbox.run.swe.diagnostics import potential_enstrophy
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)

    E0 = _total_energy(planet, model, state.coeffs)
    Z0 = potential_enstrophy(model, state)

    y, dt, n_steps = _integrate_fixed_cfl(planet, model, state, days=0.25)
    final = ShallowWaterState(y)
    model.validate_state(final, context="after 6 hours of W5")
    assert n_steps >= 10

    # Mass: the phi monopole is pinned exactly.
    assert complex(y[2, 0, 0]) == 0j
    # The mountain immediately produces nonzero divergence.
    assert float(cp.linalg.norm(y[1])) > 0.0
    # Energy and potential-enstrophy drift within the measured envelope.
    # Measured at 6 h, GL l_max=21: dE/E = +7.10e-6 (dt-INdependent:
    # +7.63e-6 at dt/2, i.e. truncation of the nonsmooth terrain
    # interaction, not time integration; falls to -5.2e-7 at l_max=31) and
    # dZ/Z = +3.86e-6. Tolerances carry ~7x/13x headroom.
    E1 = _total_energy(planet, model, y)
    Z1 = potential_enstrophy(model, final)
    assert abs(E1 - E0) <= 5e-5 * abs(E0)
    assert abs(Z1 - Z0) <= 5e-5 * abs(Z0)


@requires_cuda
def test_w5_short_run_geodesic_valid_and_conserving():
    """Geodesic backend characterization at its own measured envelope —
    NOT forced to meet Gauss-Legendre tolerances."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    from planetary_sandbox.run.swe.diagnostics import potential_enstrophy
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet(grid_type="geodesic", resolution=4, l_max=21)
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)

    E0 = _total_energy(planet, model, state.coeffs)
    Z0 = potential_enstrophy(model, state)
    y, dt, n_steps = _integrate_fixed_cfl(planet, model, state, days=0.125)
    final = ShallowWaterState(y)
    model.validate_state(final, context="after 3 hours of W5 (geodesic)")
    assert complex(y[2, 0, 0]) == 0j
    # Measured at 3 h, geodesic res4/l_max=21: dE/E = +2.46e-5,
    # dZ/Z = -5.52e-6 (the geodesic transform's inexact quadrature
    # dominates; ~3.5x the GL drift rate). Tolerances have ~8x/18x headroom.
    E1 = _total_energy(planet, model, y)
    Z1 = potential_enstrophy(model, final)
    assert abs(E1 - E0) <= 2e-4 * abs(E0)
    assert abs(Z1 - Z0) <= 1e-4 * abs(Z0)


def _total_energy(planet, model, y):
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    fields = model.characteristic_fields(ShallowWaterState(y))
    w = cp.asarray(planet.sh.weights) * planet.params.radius**2
    phi_t = fields["phi_total"]
    ke = 0.5 * phi_t * (fields["u"] ** 2 + fields["v"] ** 2)
    pe = 0.5 * phi_t**2
    phi_s = model.surface_geopotential_on_state_grid()
    if phi_s is not None:
        pe = pe + phi_t * phi_s
    return float(cp.sum(w * (ke + pe)))


# ===========================================================================
# Diagnostics: potential enstrophy
# ===========================================================================

@requires_cuda
def test_potential_enstrophy_matches_analytic_rest_value():
    """Z = integral (zeta+f)^2/(2h) dA. For a resting flat-bottom state,
    Z = 8*pi*Omega^2*R^2/(3H) exactly (integral of sin^2 = 4*pi/3)."""
    from planetary_sandbox.run.swe.diagnostics import potential_enstrophy
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet, cone=False)
    state = make_swe_ic("rest", model)
    expected = (8.0 * math.pi * OMEGA_CANON**2 * A_CANON**2
                / (3.0 * MEAN_DEPTH_CANON))
    assert potential_enstrophy(model, state) == pytest.approx(
        expected, rel=1e-12)


# ===========================================================================
# D. Fifteen-day canonical acceptance (env-gated: hours of GPU time)
# ===========================================================================

@requires_cuda
@pytest.mark.skipif(not os.environ.get("AEOLUS_W5_ACCEPTANCE"),
                    reason="15-day canonical W5 acceptance run (~3 h on the "
                           "MX110); set AEOLUS_W5_ACCEPTANCE=1 to enable")
def test_w5_fifteen_day_canonical_acceptance(tmp_path):
    """The canonical benchmark through the real CLI, verified against the
    measured 2026-07-20 acceptance envelopes (GL 64x128, l_max=42, RK4,
    inviscid): 2407 steps, mass bit-identical, dE/E = -2.135e-6,
    dZ/Z = +2.452e-5, day-15 h in [3759.6, 6196.0] m, max|u| 38.9 m/s.
    Tolerances carry ~10x headroom. Reference capsule:
    runs/w5-acceptance/20260720T050117Z_..._ac2c22de_c583365f."""
    import csv
    import numpy as np
    import cupy as cp
    from planetary_sandbox.cli.main import main
    from planetary_sandbox.physics.shallow_water import (ShallowWaterModel,
                                                         ShallowWaterState)
    from planetary_sandbox.physics.topography import Topography
    from planetary_sandbox.run.swe.diagnostics import potential_enstrophy

    rc = main(["run", "swe", "--scenario", "williamson5",
               "--backend", "gauss-latlon", "--nlat", "64", "--nlon", "128",
               "--l-max", "42", "--days", "15", "--n-snapshots", "4",
               "--no-plots", "--out", str(tmp_path / "runs")])
    assert rc == 0
    pointer = (tmp_path / "runs" / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    run_dir = tmp_path / "runs" / pointer

    times = np.load(run_dir / "swe_snapshot_times.npy")
    assert times.tolist() == [0.0, 5.0 * 86400.0, 10.0 * 86400.0,
                              15.0 * 86400.0]
    coeffs = np.load(run_dir / "swe_coeffs.npy")

    with open(run_dir / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len({r["total_mass"] for r in rows}) == 1     # bit-identical
    e = [float(r["total_energy"]) for r in rows]
    assert abs(e[-1] - e[0]) <= 2e-5 * abs(e[0])         # measured -2.1e-6
    h_min = min(float(r["h_min_m"]) for r in rows)
    assert h_min > 3000.0                                 # measured 3759.6
    assert max(float(r["max_wind_ms"]) for r in rows) < 60.0

    planet = _make_w5_planet(l_max=42, nlat=64, nlon=128)
    model = ShallowWaterModel(planet, gravity=GRAVITY,
                              mean_depth=MEAN_DEPTH_CANON,
                              topography=Topography.williamson5_cone(planet))
    z0 = potential_enstrophy(model, ShallowWaterState(cp.asarray(coeffs[0])))
    z1 = potential_enstrophy(model, ShallowWaterState(cp.asarray(coeffs[-1])))
    assert abs(z1 - z0) <= 2.5e-4 * abs(z0)              # measured +2.5e-5
    model.validate_state(ShallowWaterState(cp.asarray(coeffs[-1])),
                         context="day-15 acceptance state")


@requires_cuda
def test_gaussian_mountain_gate_unchanged():
    """The W5 cone policy must not touch the Gaussian preset's 0.2 gate."""
    from planetary_sandbox.physics.topography import (
        MAX_PROJECTION_RESIDUAL, Topography, TopographyError)

    assert MAX_PROJECTION_RESIDUAL == 0.2
    planet = _make_w5_planet()
    with pytest.raises(TopographyError, match="not representable"):
        Topography.mountain(planet, height_m=2000.0, lat_deg=30.0,
                            lon_deg=-90.0, width_deg=1.0)
