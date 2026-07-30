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
    h0     = 5960      m        reference FREE-SURFACE height
    hs0    = 2000      m        mountain peak height
    R0     = pi/9      rad      cone support radius (coordinate-plane)
    lat_c  = pi/6      rad      cone center latitude  (30 N)
    lon_c  = 3*pi/2    rad      cone center longitude (270 E == -90 E)

The initial state is the Williamson-2-shaped wind/FREE-SURFACE pair with
u0 = 20 m/s (Williamson et al. 1992: Sect. 2 defines h = h* + h_s with h
the free surface and h* the depth; Sect. 3.5 takes "the wind and height
field ... as in case 2, with alpha = 0"):

    u    = u0 cos(lat),  v = 0
    eta  = h0 - (C/g) sin^2(lat),      C = a*Omega*u0 + u0^2/2
    h*   = eta - h_s                   (cone-shaped depression in the layer)

In model variables: phi = C*(1/3 - sin^2 lat) - phi_s' (the same
terrain-compensating construction as every other scenario) and
H = h0 - C/(3g) - mean(h_s), so the free surface Phi0 + phi + phi_s
reproduces eta exactly and the depth carries the canonical bite. A direct
regression below fails if anyone ever reverts W5 to the pre-2026-07-29
uncompensated-thickness convention (which raised the free surface over the
cone — a physically different initial-value problem from the published
test case and the MRI-JMA reference; see
notebooks/W5_MRI_SEMANTIC_AUDIT.md).

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
DAY_HOURS_CANON = 2.0 * math.pi / OMEGA_CANON / 3600.0


def _cone_mean_height_local() -> float:
    """Exact spherical mean of the analytic cone, restated locally.

    hbar = (hs0 cos(lat_c)/2) * int_0^R0 (1 - r/R0) J0(r) r dr, the Bessel
    integral summed as the alternating series
    sum_k (-1)^k R0^(2k+2) / (4^k (k!)^2 (2k+2)(2k+3)). Deliberately the
    same float expression as run/swe/config.py so the equality assertions
    below are bitwise; an independent numerical-quadrature cross-check is
    a separate test.
    """
    r0sq = R0 * R0
    total = 0.0
    k = 0
    power = r0sq
    factorial = 1.0
    while True:
        term = power / ((4.0 ** k) * factorial * factorial
                        * (2 * k + 2) * (2 * k + 3))
        new_total = total + (term if k % 2 == 0 else -term)
        if new_total == total:
            break
        total = new_total
        k += 1
        power *= r0sq
        factorial *= k
    return 0.5 * HS0 * math.cos(math.radians(30.0)) * total


CONE_MEAN_CANON = _cone_mean_height_local()
#: Canonical mean depth H = mean(eta - h_s) = h0 - C/(3g) - mean(h_s).
MEAN_DEPTH_CANON = H0 - C_CANON / (3.0 * GRAVITY) - CONE_MEAN_CANON


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
    # H = h0 - C/(3g) - mean(h_s), exactly (same float expression as the
    # local constants; the cone mean is the closed-form Bessel series).
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
    assert swe_config.W5_CONE_MEAN_HEIGHT_M == CONE_MEAN_CANON
    assert swe_config.W5_MEAN_DEPTH_M == MEAN_DEPTH_CANON
    assert swe_config.W5_DAY_HOURS == DAY_HOURS_CANON


def test_w5_cone_mean_height_closed_form_matches_quadrature():
    """The Bessel-series cone mean must agree with a brute-force spherical
    quadrature of the analytic cone (independent derivation check), and
    with the MRI-JMA reference model's logged initial global mean of mass
    (5619.92593916377 m for depth = eta - h_s; STDOUT of
    Williamson5/N959_1920x960/sh)."""
    import numpy as np

    lat = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 8001)
    lon = np.linspace(0.0, 2.0 * math.pi, 2000, endpoint=False)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    dlam = np.mod(lon2 - LONC + math.pi, 2.0 * math.pi) - math.pi
    r = np.minimum(R0, np.hypot(dlam, lat2 - LATC))
    hs = HS0 * (1.0 - r / R0)
    w = np.cos(lat2)
    numerical = float((hs * w).sum() / w.sum())
    assert CONE_MEAN_CANON == pytest.approx(numerical, abs=5e-5)
    assert MEAN_DEPTH_CANON == pytest.approx(5619.92593916377, abs=1e-4)


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
    # phi = [case-2 free-surface (2,0) mode] - phi_s'. Adding phi_s' back
    # must leave the pure (2,0) mode -(4*C/3)*sqrt(pi/5): every other
    # entry cancels exactly (x + (-x) == 0 in IEEE), and the (2,0) entry
    # matches to one rounding step of the compensation arithmetic.
    expect_phi = -(4.0 * C_CANON / 3.0) * math.sqrt(math.pi / 5.0)
    fs_anomaly = phi + model.phi_s_anom_lm
    assert complex(fs_anomaly[2, 0]) == pytest.approx(expect_phi, rel=1e-12)
    f = fs_anomaly.copy()
    f[2, 0] = 0.0
    assert float(cp.abs(f).max()) == 0.0
    # The phi monopole (mass anomaly) is exactly zero: phi_s' is
    # mean-removed, so the compensation never touches (0,0).
    assert complex(phi[0, 0]) == 0j


@requires_cuda
def test_w5_ic_reconstructs_canonical_wind_free_surface_and_depth():
    """Direct canonical-field tests: winds, free-surface height, and layer
    depth against the analytic Williamson case-5 prescriptions."""
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

    phi_grid = planet.sh.inv_transform(state.coeffs[2]).real
    phi_s_grid = model.surface_geopotential_on_state_grid()
    depth = (model.phi0 + phi_grid) / model.gravity
    surface = depth + phi_s_grid / model.gravity
    eta_ref = H0 - (C_CANON / GRAVITY) * cp.sin(lat) ** 2

    # Free surface == analytic eta up to the terrain-monopole quadrature
    # residual (a CONSTANT offset: the discrete cone mean minus the exact
    # Bessel-series mean; measured +0.0908 m on this 32x64/l21 grid,
    # -0.0086 m at 64x128/l42, +0.0041 m at 96x192/l63). The spatially
    # varying parts cancel spectrally, so the deviation must also be
    # constant to transform accuracy.
    dev = surface - eta_ref
    assert float(cp.abs(dev).max()) < 0.15
    assert float((dev - dev.mean()).max()) < 1e-9

    # Layer depth == eta - h_s at the model's terrain representation
    # (band-limited cone), same constant offset.
    hs_band_limited = phi_s_grid / model.gravity
    dev_depth = depth - (eta_ref - hs_band_limited)
    assert float(cp.abs(dev_depth).max()) < 0.15

    # Against the ANALYTIC cone the depth differs additionally by the
    # band-limiting of the terrain itself (Gibbs ringing near the cusp;
    # projection residual 0.0706 at this truncation) — bounded, documented.
    lon = cp.asarray(planet.grid.point_longitudes, dtype=cp.float64)
    from planetary_sandbox.physics.topography import williamson5_cone_elevation
    hs_analytic = williamson5_cone_elevation(lat, lon)
    band_limit_err = float(cp.abs(depth - (eta_ref - hs_analytic)).max())
    assert band_limit_err < 500.0          # peak undershoot ~246 m at l21
    # The canonical cone-shaped BITE is present in the layer:
    # depth is ~hs0 shallower at the mountain than the zonal profile.
    bite = eta_ref - depth
    assert float(bite.max()) > 1500.0
    assert float(bite.max()) < HS0 * 1.05  # allow small Gibbs overshoot


@requires_cuda
def test_w5_ic_is_terrain_aware():
    """With the cone, phi differs from the flat construction by exactly
    -phi_s' (winds identical): the depth carries the mountain bite. The
    terrain-less model (defensive path) degenerates to the flat case-2
    pair."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    with_cone = make_swe_ic("williamson5", model)
    without = make_swe_ic("williamson5", _make_w5_model(planet, cone=False))
    assert bool(cp.all(with_cone.coeffs[0] == without.coeffs[0]))
    assert bool(cp.all(with_cone.coeffs[1] == without.coeffs[1]))
    diff = with_cone.coeffs[2] - without.coeffs[2]
    # Off-(2,0) entries cancel exactly; the (2,0) entry carries one
    # rounding step of the compensation arithmetic (measured 5.1e-13
    # against coefficients of order 1e4).
    assert float(cp.abs(diff + model.phi_s_anom_lm).max()) < 1e-8


@requires_cuda
def test_w5_regression_ic_is_free_surface_compensated():
    """MUST FAIL if W5 is ever reverted to the pre-2026-07-29 convention
    that prescribed the case-2 field as the THICKNESS (no -phi_s' term),
    i.e. a free surface raised over the mountain. That construction is a
    physically different initial-value problem from Williamson (1992)
    Sect. 2 + 3.5 and from the MRI-JMA reference trajectories (semantic
    audit: notebooks/W5_MRI_SEMANTIC_AUDIT.md)."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)
    phi = state.coeffs[2]

    # The cone's largest off-(2,0) coefficient must appear in phi with the
    # OPPOSITE sign (phi = case-2 mode - phi_s'): the uncompensated
    # convention carried exactly zero there.
    phi_s = model.phi_s_anom_lm.copy()
    phi_s[2, 0] = 0.0
    idx = int(cp.abs(phi_s).argmax())
    l_big, m_big = divmod(idx, phi_s.shape[1])
    assert float(cp.abs(phi_s[l_big, m_big])) > 0.0   # cone truly present
    assert complex(phi[l_big, m_big]) == complex(-phi_s[l_big, m_big])
    # Globally: phi + phi_s' is the pure case-2 (2,0) mode, nothing else.
    fs_anomaly = phi + model.phi_s_anom_lm
    f = fs_anomaly.copy()
    f[2, 0] = 0.0
    assert float(cp.abs(f).max()) == 0.0
    assert float(cp.abs(fs_anomaly[2, 0])) > 0.0


@requires_cuda
def test_w5_initial_free_surface_is_zonal_not_raised():
    """The canonical initial free surface shows NO mountain signature: the
    mountain lives entirely in the layer depth. (The pre-correction
    convention raised the surface ~1754 m over the cone at this
    truncation; a reader seeing a bump here is looking at the reverted,
    noncanonical construction.)"""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)

    phi_grid = planet.sh.inv_transform(state.coeffs[2]).real
    surface = (model.phi0 + phi_grid
               + model.surface_geopotential_on_state_grid()) / model.gravity
    # Zonally symmetric: max longitude spread at fixed latitude ~ 0.
    nlat, nlon = 32, 64
    surf2d = surface.reshape(nlat, nlon)
    zonal_spread = float((surf2d.max(axis=1) - surf2d.min(axis=1)).max())
    assert zonal_spread < 1e-9
    # And the depth — not the surface — carries the ~2000 m structure.
    depth2d = ((model.phi0 + phi_grid) / model.gravity).reshape(nlat, nlon)
    depth_spread = float((depth2d.max(axis=1) - depth2d.min(axis=1)).max())
    assert depth_spread > 1500.0


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
def test_w5_initial_forcing_is_depth_advection_over_the_cone():
    """Canonical case 5 at t = 0: the wind/free-surface pair is the
    balanced case-2 state, so the momentum side is quiescent —
    dot(zeta) ~ 0 and dot(delta) ~ 0 (the -lap(phi_s) mountain term is
    cancelled by the compensated phi: phi + phi_s is zonal). The entire
    initial response is zonal ADVECTION of the cone-shaped depth anomaly:

        dot(phi) = -div(phi u) = -(u0/a) d(phi)/dlambda
                 = -(u0/a) * (i m) * phi_lm

    spectrally exact up to the model's per-product 2/3 dealiasing rule
    (nonlinear products are truncated at l <= 2*l_max//3, so the terrain
    modes above the cut do not advect at t = 0).

    With the cone absent (flat model, flat-built state) every tendency
    vanishes: the response is entirely terrain-driven."""
    import cupy as cp
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic

    planet = _make_w5_planet()
    model = _make_w5_model(planet)
    state = make_swe_ic("williamson5", model)
    dot = model.tendency(state.coeffs)

    m_index = cp.arange(state.coeffs.shape[-1])[None, :]
    expected_phi_dot = -(U0 / A_CANON) * (1j * m_index) * state.coeffs[2]
    cut = (2 * 21) // 3          # the model's 2/3 product-truncation rule
    expected_phi_dot[cut + 1:, :] = 0.0
    expected_phi_dot[:, cut + 1:] = 0.0
    forcing = float(cp.abs(expected_phi_dot).max())
    assert forcing > 0.0
    # dot(phi) matches the analytic advection spectrum to product-grid
    # analysis accuracy.
    residual = float(cp.abs(dot[2] - expected_phi_dot).max()) / forcing
    assert residual < 1e-9
    # The momentum side stays at the balanced-cancellation floor, measured
    # against the mountain term -lap(phi_s) that the compensated phi must
    # cancel out of the divergence tendency (per-equation scale).
    cancel_scale = float(cp.abs(model.lap_eigs[:, None]
                                * model.phi_s_lm).max())
    assert cancel_scale > 0.0
    assert float(cp.abs(dot[0]).max()) < 1e-9 * cancel_scale
    assert float(cp.abs(dot[1]).max()) < 1e-9 * cancel_scale

    # Null experiment: no cone -> no response (flat case-2 is steady).
    flat_model = _make_w5_model(planet, cone=False)
    flat_state = make_swe_ic("williamson5", flat_model)
    dot_flat = flat_model.tendency(flat_state.coeffs)
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
    # Measured at 6 h, GL l_max=21, canonical (2026-07-29) IC:
    # dE/E = -4.59e-9 and dZ/Z = -1.03e-6 — far below the pre-correction
    # IC's +7.1e-6/+3.9e-6 (the canonical state starts as balanced
    # advection, not a 2 km raised-surface gravity-wave burst). Tolerances
    # kept at the historical envelope (now >48x headroom).
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
    # Measured at 3 h, geodesic res4/l_max=21, canonical (2026-07-29) IC:
    # dE/E = +1.07e-7, dZ/Z = -8.16e-8 — far below the pre-correction
    # IC's +2.5e-5/-5.5e-6. Tolerances kept at the historical envelope
    # (now >1000x headroom).
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
    """The canonical benchmark through the real CLI (GL 64x128, l_max=42,
    RK4, inviscid), gated by loose structural envelopes only.

    STALE-ENVELOPE WARNING (2026-07-29): the initial condition was
    corrected to the canonical Williamson free-surface prescription (see
    the module docstring). The 2026-07-20 measured acceptance values —
    2407 steps, dE/E = -2.135e-6, dZ/Z = +2.452e-5, day-15 h in
    [3759.6, 6196.0] m, max|u| 38.9 m/s — and the capsules in
    runs/w5-acceptance/*_c583365f were produced with the SUPERSEDED
    uncompensated-thickness IC and are NOT valid references for the
    corrected benchmark. The loose gates below (mass bit-identity,
    |dE/E| <= 2e-5, h_min > 3000 m, max wind < 60 m/s, |dZ/Z| <= 2.5e-4)
    are expected to hold for the canonical IC as well, but a fresh 15-day
    measurement pass must replace this docstring's envelope numbers and
    the acceptance capsules before the gates are tightened again."""
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
