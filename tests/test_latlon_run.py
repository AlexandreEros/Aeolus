"""End-to-end BVE run on the lat-lon backend (runner + diagnostics + viz)."""
import numpy as np
import pytest

try:
    import cupy as cp

    _HAS_CUDA = cp.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/CuPy not available")

if _HAS_CUDA:
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import run_bve


def test_run_bve_completes_on_latlon(tmp_path):
    """A short lat-lon run must produce the full artifact set: coefficients,
    grid snapshots, diagnostics CSV, and the summary/snapshot figures."""
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=24.0),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    zeta0_lm = planet.sh.transform(make_ic("rh4", planet))

    rc = run_bve(planet=planet, zeta0_lm=zeta0_lm,
                 dt_snapshots=1800.0, t_end_days=0.02,
                 out_dir=tmp_path, viscosity=0.0, scenario="rh4")

    assert rc == 0
    assert (tmp_path / "vorticity_coeffs.npy").exists()
    assert (tmp_path / "vorticity_grid.npy").exists()
    assert (tmp_path / "diagnostics" / "timeseries.csv").exists()
    assert (tmp_path / "bve_summary.png").exists()
    # coefficients stay finite
    coeffs = np.load(tmp_path / "vorticity_coeffs.npy")
    assert np.isfinite(coeffs).all()
