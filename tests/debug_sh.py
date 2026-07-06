"""Debug script to compare lat-lon SH against the geodesic wrapper."""
import numpy as np
import cupy as cp

from planetary_sandbox.numerics import GeodesicGridGeometry, LatLonGridGeometry
from planetary_sandbox.numerics import GeodesicSphericalHarmonics
from planetary_sandbox.numerics.spherical_harmonics import LatLonSphericalHarmonics


def test_y10(lat, lon):
    """This should give coefficient a_10 = sqrt(4*pi/3) and all others = 0."""
    del lon
    return cp.sin(lat)


def test_y00(lat, lon):
    """Constant function: should give only a_00 = sqrt(4*pi)."""
    del lon
    return cp.ones_like(lat)


def main():
    l_max = 5

    latlon_grid = LatLonGridGeometry.create((33, 65))
    geodesic_grid = GeodesicGridGeometry(resolution=3, radius=1.0)

    print("=" * 70)
    print("Testing Y_0^0 (constant function)")
    print("=" * 70)

    lat_ll = cp.array(latlon_grid.lat_grid)
    lon_ll = cp.array(latlon_grid.lon_grid)
    values_ll = test_y00(lat_ll, lon_ll)

    latlon_sh = LatLonSphericalHarmonics(l_max, latlon_grid.longitudes, latlon_grid.colatitudes)
    coeffs_ll = cp.asnumpy(latlon_sh.transform(values_ll))

    lat_geo = geodesic_grid.latitudes
    lon_geo = geodesic_grid.longitudes
    values_geo = test_y00(lat_geo, lon_geo)

    weights_geo = geodesic_grid.cell_areas
    print(f"\nGeodesic grid: {geodesic_grid.n_points} points")
    print(f"Weight sum: {np.sum(weights_geo):.6f}")
    print(f"Expected (4*pi*R^2): {4 * np.pi * geodesic_grid.radius**2:.6f}")
    print(f"Weight range: [{np.min(weights_geo):.6e}, {np.max(weights_geo):.6e}]")
    print(f"Weight std/mean: {np.std(weights_geo)/np.mean(weights_geo):.4f}")

    geo_sh = GeodesicSphericalHarmonics(geodesic_grid, l_max, weights="voronoi")
    coeffs_geo = cp.asnumpy(geo_sh.transform(values_geo))

    print(f"\nExpected a_00 = {np.sqrt(4*np.pi):.8f}")
    print(f"LatLon   a_00 = {coeffs_ll[0,0].real:.8f}")
    print(f"Geodesic a_00 = {coeffs_geo[0,0].real:.8f}")

    print("\nNon-zero coefficients (should only be a_00):")
    print(f"LatLon:   {np.sum(np.abs(coeffs_ll) > 1e-6)} coefficients > 1e-6")
    print(f"Geodesic: {np.sum(np.abs(coeffs_geo) > 1e-6)} coefficients > 1e-6")

    abs_coeffs = np.abs(coeffs_geo)
    abs_coeffs[0, 0] = 0
    idx = np.unravel_index(np.argmax(abs_coeffs), abs_coeffs.shape)
    print("\nLargest spurious geodesic coefficient:")
    print(f"  a_{idx[0]},{idx[1]} = {coeffs_geo[idx[0], idx[1]]:.8e}")

    print("\n" + "=" * 70)
    print("Testing Y_1^0 (sin(lat) function)")
    print("=" * 70)

    values_ll = test_y10(lat_ll, lon_ll)
    coeffs_ll = cp.asnumpy(latlon_sh.transform(values_ll))

    values_geo = test_y10(lat_geo, lon_geo)
    coeffs_geo = cp.asnumpy(geo_sh.transform(values_geo))

    expected_10 = np.sqrt(4 * np.pi / 3)

    print(f"\nExpected a_10 = {expected_10:.8f}")
    print(f"LatLon   a_10 = {coeffs_ll[1,0].real:.8f}")
    print(f"Geodesic a_10 = {coeffs_geo[1,0].real:.8f}")

    print("\nAll other coefficients should be ~0:")
    for l in range(min(3, l_max + 1)):
        for m in range(l + 1):
            if l == 1 and m == 0:
                continue
            ll_val = coeffs_ll[l, m]
            geo_val = coeffs_geo[l, m]
            if abs(ll_val) > 1e-6 or abs(geo_val) > 1e-6:
                print(f"  a_{l},{m}: LatLon={ll_val:.6e}, Geodesic={geo_val:.6e}")


if __name__ == "__main__":
    main()
