import unittest
import numpy as np

from planetary_sandbox.numerics import LatLonGridGeometry, GeodesicGridGeometry
from planetary_sandbox.numerics.grid_interpolation import (
    latlon_to_geodesic_grid,
    geodesic_to_latlon_grid,
)


class TestGridInterpolation(unittest.TestCase):
    def test_latlon_geodesic_roundtrip_error(self):
        latlon_grid = LatLonGridGeometry.create((33, 65))
        geodesic_grid = GeodesicGridGeometry(resolution=3, radius=1.0)

        lat = latlon_grid.lat_grid
        lon = latlon_grid.lon_grid
        values = np.sin(lat) + 0.3 * np.cos(2.0 * lon) * np.cos(lat)

        geodesic_values = latlon_to_geodesic_grid(
            values,
            latlon_grid,
            geodesic_grid,
            method="linear",
        )

        back_on_latlon = geodesic_to_latlon_grid(
            geodesic_values,
            geodesic_grid,
            latlon_grid,
            method="linear",
        )

        mask = np.isfinite(back_on_latlon)
        self.assertTrue(mask.any(), "Interpolation produced no finite values")

        error = back_on_latlon[mask] - values[mask]
        rmse = float(np.sqrt(np.mean(error ** 2)))
        max_abs = float(np.max(np.abs(error)))

        print(f"latlon->geodesic->latlon RMSE: {rmse:.6f}, max abs: {max_abs:.6f}")
        self.assertLess(rmse, 0.2)


if __name__ == "__main__":
    unittest.main()
