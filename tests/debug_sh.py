"""Debug script to check spherical harmonics basis and integration."""
import numpy as np
import cupy as cp

from planetary_sandbox.numerics import LatLonGridGeometry, GeodesicGridGeometry
from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
from planetary_sandbox.numerics.spherical_harmonics import LatLonSphericalHarmonics


# Test on a simple function: Y_1^0 (just sin(theta) = cos(phi) where phi is colatitude)
def test_y10(lat, lon):
    """This should give coefficient a_10 = sqrt(4pi/3) and all others = 0"""
    return cp.sin(lat)  # sin(lat) = cos(colat)


def test_y00(lat, lon):
    """Constant function - should give only a_00 = sqrt(4pi)"""
    return cp.ones_like(lat)


def main():
    l_max = 5

    # Create grids
    latlon_grid = LatLonGridGeometry.create((33, 65))
    geodesic_grid = GeodesicGridGeometry(resolution=3, radius=1.0)

    print("=" * 70)
    print("Testing Y_0^0 (constant function)")
    print("=" * 70)

    # LatLon test
    lat_ll = cp.array(latlon_grid.lat_grid)
    lon_ll = cp.array(latlon_grid.lon_grid)
    values_ll = test_y00(lat_ll, lon_ll)

    latlon_sh = LatLonSphericalHarmonics(l_max, latlon_grid.longitudes, latlon_grid.colatitudes)
    coeffs_ll = cp.asnumpy(latlon_sh.transform(values_ll))

    # Geodesic test
    lat_geo = geodesic_grid.latitudes
    lon_geo = geodesic_grid.longitudes
    values_geo = test_y00(lat_geo, lon_geo)

    weights_geo = geodesic_grid.cell_areas
    print(f"\nGeodesic grid: {geodesic_grid.n_points} points")
    print(f"Weight sum: {np.sum(weights_geo):.6f}")
    print(f"Expected (4πR²): {4 * np.pi * geodesic_grid.radius**2:.6f}")
    print(f"Weight range: [{np.min(weights_geo):.6e}, {np.max(weights_geo):.6e}]")
    print(f"Weight std/mean: {np.std(weights_geo)/np.mean(weights_geo):.4f}")

    geo_sh = PointSetSphericalHarmonics(lat_geo, lon_geo, l_max, weights=weights_geo)
    coeffs_geo = cp.asnumpy(geo_sh.transform(values_geo))

    print(f"\nExpected a_00 = {np.sqrt(4*np.pi):.8f}")
    print(f"LatLon   a_00 = {coeffs_ll[0,0].real:.8f}")
    print(f"Geodesic a_00 = {coeffs_geo[0,0].real:.8f}")

    print(f"\nNon-zero coefficients (should only be a_00):")
    print(f"LatLon:   {np.sum(np.abs(coeffs_ll) > 1e-6)} coefficients > 1e-6")
    print(f"Geodesic: {np.sum(np.abs(coeffs_geo) > 1e-6)} coefficients > 1e-6")

    # Show largest spurious coefficients for geodesic
    abs_coeffs = np.abs(coeffs_geo)
    abs_coeffs[0, 0] = 0  # Exclude the expected one
    idx = np.unravel_index(np.argmax(abs_coeffs), abs_coeffs.shape)
    print(f"\nLargest spurious geodesic coefficient:")
    print(f"  a_{idx[0]},{idx[1]} = {coeffs_geo[idx[0], idx[1]]:.8e}")

    print("\n" + "=" * 70)
    print("Testing Y_1^0 (sin(lat) function)")
    print("=" * 70)

    # LatLon test
    values_ll = test_y10(lat_ll, lon_ll)
    coeffs_ll = cp.asnumpy(latlon_sh.transform(values_ll))

    # Geodesic test
    values_geo = test_y10(lat_geo, lon_geo)
    coeffs_geo = cp.asnumpy(geo_sh.transform(values_geo))

    # Y_1^0 = sqrt(3/(4pi)) * cos(theta) where theta is colatitude
    # cos(theta) = sin(lat)
    # So coefficient should be sqrt(4pi/3)
    expected_10 = np.sqrt(4*np.pi/3)

    print(f"\nExpected a_10 = {expected_10:.8f}")
    print(f"LatLon   a_10 = {coeffs_ll[1,0].real:.8f}")
    print(f"Geodesic a_10 = {coeffs_geo[1,0].real:.8f}")

    print(f"\nAll other coefficients should be ~0:")
    for l in range(min(3, l_max+1)):
        for m in range(l+1):
            if l == 1 and m == 0:
                continue
            ll_val = coeffs_ll[l, m]
            geo_val = coeffs_geo[l, m]
            if abs(ll_val) > 1e-6 or abs(geo_val) > 1e-6:
                print(f"  a_{l},{m}: LatLon={ll_val:.6e}, Geodesic={geo_val:.6e}")


if __name__ == "__main__":
    main()
