"""Backend selection and numerics provenance.

Every run must record WHICH numerics produced it: backend family, state
sampling, product sampling (the ProductSpace label actually used), and
transform type — in manifest.json under "numerics".
"""
import json

import pytest

try:
    import cupy as cp

    _HAS_CUDA = cp.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/CuPy not available")

REQUIRED_NUMERICS_KEYS = {
    "backend", "grid", "state_sampling", "product_quadrature",
    "product_sampling", "transform", "l_max",
}


# ---------------------------------------------------------------------------
# backend.describe()
# ---------------------------------------------------------------------------

@requires_cuda
def test_geodesic_backend_describe():
    from planetary_sandbox.numerics import (
        GeodesicBackend, GeodesicGridGeometry, GeodesicSphericalHarmonics)
    grid = GeodesicGridGeometry(resolution=3, radius=1.0)
    sh = GeodesicSphericalHarmonics(grid, 5, weights="voronoi")
    desc = GeodesicBackend(grid, sh).describe("fine")
    assert REQUIRED_NUMERICS_KEYS <= set(desc)
    assert desc["backend"] == "GeodesicBackend"
    assert desc["grid"] == "GeodesicGridGeometry"
    assert desc["product_quadrature"] == "fine"
    assert "geodesic" in desc["product_sampling"]
    assert "state" in desc["state_sampling"]
    assert desc["l_max"] == 5
    json.dumps(desc)  # must be JSON-serializable as-is


@requires_cuda
def test_latlon_backend_describe():
    from planetary_sandbox.numerics import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics, LatLonBackend)
    grid = GaussLatLonGridGeometry(12, 24, radius=1.0)
    sh = GaussLatLonSphericalHarmonics(grid, 5)
    desc = LatLonBackend(grid, sh).describe("fine")
    assert REQUIRED_NUMERICS_KEYS <= set(desc)
    assert desc["backend"] == "LatLonBackend"
    assert "latlon" in desc["product_sampling"]
    json.dumps(desc)


# ---------------------------------------------------------------------------
# Planet.generate grid selection
# ---------------------------------------------------------------------------

@requires_cuda
def test_planet_generate_latlon_backend():
    from planetary_sandbox.numerics import LatLonBackend
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    assert isinstance(planet.so.backend, LatLonBackend)
    assert planet.grid.grid_shape == (16, 32)
    # transform pair usable
    coeffs = cp.zeros((8, 8), dtype=cp.complex128)
    coeffs[2, 1] = 1e-5
    values = planet.sh.inv_transform(coeffs)
    back = planet.sh.transform(values)
    assert abs(complex(back[2, 1]) - 1e-5) < 1e-16


@requires_cuda
def test_planet_generate_unknown_grid_type_raises():
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    with pytest.raises(ValueError, match="grid_type"):
        Planet.generate(params=PlanetaryParameters.from_earth_like(),
                        grid_type="cubed-sphere", l_max=5)


# ---------------------------------------------------------------------------
# CLI + manifest (pure Python)
# ---------------------------------------------------------------------------

def test_cli_exposes_grid_choice():
    from planetary_sandbox.cli.bve import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert vars(args)["grid"] == "geodesic"
    assert vars(parser.parse_args(["--grid", "latlon"]))["grid"] == "latlon"
    with pytest.raises(SystemExit):
        parser.parse_args(["--grid", "cubed-sphere"])


def test_manifest_records_numerics_section(tmp_path):
    from planetary_sandbox.run.bve.io import write_run_manifest
    numerics = {
        "backend": "LatLonBackend",
        "grid": "GaussLatLonGridGeometry",
        "state_sampling": "latlon-gauss-64x128-state",
        "product_quadrature": "fine",
        "product_sampling": "latlon-gauss-32x64-3/2rule",
        "transform": "GaussLatLonSphericalHarmonics",
        "l_max": 21,
    }
    write_run_manifest(tmp_path, {"scenario": "rh4"}, run_id="x",
                       experiment=None, numerics=numerics)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["numerics"] == numerics
