import unittest
import numpy as np
import cupy as cp
from pathlib import Path

from planetary_sandbox.numerics import LatLonGridGeometry, GeodesicGridGeometry
from planetary_sandbox.numerics.spherical_harmonics import LatLonSphericalHarmonics
from planetary_sandbox.numerics import GeodesicSphericalHarmonics


def _has_cuda_cupy() -> bool:
    try:
        import cupy as cp  # noqa: F401
        _ = cp.zeros((1,))
    except Exception:
        return False
    return True


def test_scalar(lat, lon):
    return (cp.sin(lat)
            + 0.5 * cp.cos(2.0 * lon) * cp.cos(lat)
            + 0.2 * cp.sin(3.0 * lon) * cp.sin(lat) * cp.cos(lat)
        )

class TestSphericalHarmonicsAgreement(unittest.TestCase):
    @unittest.skipUnless(_has_cuda_cupy(), "CUDA/CuPy not available")
    def test_latlon_vs_geodesic_coefficients(self):
        l_max = 15

        latlon_grid = LatLonGridGeometry.create((33, 65))
        geodesic_grid = GeodesicGridGeometry(resolution=4, radius=1.0)

        lat = cp.array(latlon_grid.lat_grid)
        lon = cp.array(latlon_grid.lon_grid)
        values_latlon = test_scalar(lat, lon)

        latlon_sh = LatLonSphericalHarmonics(l_max)
        # set_grid expects (longitudes, colatitudes), not latitudes
        latlon_sh.set_grid(latlon_grid.longitudes, latlon_grid.colatitudes)
        coeffs_latlon = latlon_sh.transform(values_latlon)

        geo_lat = geodesic_grid.latitudes
        geo_lon = geodesic_grid.longitudes
        values_geo = test_scalar(geo_lat, geo_lon)

        # Use the weight-aware geodesic wrapper with cached quadrature weights.
        cache_dir = Path(__file__).parent / ".sh_cache"
        geo_sh = GeodesicSphericalHarmonics(geodesic_grid, l_max, cache_dir=cache_dir)
        coeffs_geo = geo_sh.transform(values_geo)


        coeffs_latlon_np = cp.asnumpy(coeffs_latlon)
        coeffs_geo_np = cp.asnumpy(coeffs_geo)

        low_l = 3
        latlon_slice = coeffs_latlon_np[: low_l + 1, : low_l + 1]
        geo_slice = coeffs_geo_np[: low_l + 1, : low_l + 1]

        diff = geo_slice - latlon_slice
        denom = np.linalg.norm(latlon_slice) + 1e-12
        rel_l2 = float(np.linalg.norm(diff) / denom)

        print(f"Relative L2 error (l<= {low_l}): {rel_l2:.3f}")
        if rel_l2 >= 0.6:
            try:
                import matplotlib.pyplot as plt
                from planetary_sandbox.viz.planet_viewer import PlanetViewer

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                PlanetViewer.plot_scalar(values_latlon, latlon_grid,
                                        title="LatLon Grid Scalar Field",
                                        ax=axes[0,0])
                PlanetViewer.plot_coefficient_complex_visualization(coeffs_latlon, axes[1,0], fig)
                axes[1,0].set_title("LatLon SH Coeffs")

                PlanetViewer.plot_scalar(values_geo, geodesic_grid,
                                        title="Geodesic Grid Scalar Field",
                                        ax=axes[0,1])
                PlanetViewer.plot_coefficient_complex_visualization(coeffs_geo, axes[1,1], fig)
                axes[1,1].set_title("Geodesic SH Coeffs")
                fig.suptitle("SH Coefficient Comparison", fontsize=12)

                fig.tight_layout()
                fig.savefig("tests/sh_coeffs_compare.png", dpi=150)
                plt.close(fig)
                print("Saved coefficient comparison to tests/sh_coeffs_compare.png")
                
            except Exception as err:
                print(f"Failed to render coefficient comparison: {err}")
        self.assertLess(rel_l2, 0.6)

if __name__ == "__main__":
    unittest.main()
