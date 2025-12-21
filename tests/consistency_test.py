import cupy as cp
import numpy as np

from planetary_sandbox.numerics import (
    LatLonSphericalHarmonics,
    PointSetSphericalHarmonics
)

n_lat = 45 # Odd number
n_lon = 90 # Even/Odd doesn't strictly matter for FastSH, but good for simpson
l_max = 30

print("--- Consistency & Convergence Test ---\n")


# Create LatLon Grid
phi_1d = np.linspace(0, np.pi, n_lat)
lat_1d = np.pi/2 - phi_1d  # Decreasing latitude
lon_1d = np.linspace(0, 2*np.pi, n_lon)
lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
sh_old = LatLonSphericalHarmonics(l_max=l_max, lon_grid=lon_grid, colat_grid=cp.pi/2-lat_grid)

# Create geodesic grid
dlat = np.pi / (n_lat - 1)
dlon = 2 * np.pi / n_lon
weights = np.cos(lat_grid) * dlat * dlon 
fast_sh = PointSetSphericalHarmonics(lat_grid.ravel(), lon_grid.ravel(), l_max, weights=weights.ravel())
cp.cuda.Stream.null.synchronize()

# 1. Define a known 'Ground Truth' in spectral space
# We'll use a simple band-limited function composed of a few modes.
c_truth = cp.zeros((l_max + 1, l_max + 1), dtype=cp.complex128)
c_truth[2, 0] = 1.0          # Zonal harmonic (l=2, m=0)
c_truth[5, 3] = 0.5 - 0.5j   # Tesseral harmonic (l=5, m=3)

print("1. Generating synthetic field from known coefficients (using Old method inverse)...")
# We use the Old method for synthesis to avoid circular dependency, 
# assuming its inverse transform is correct (it's a standard formula).
field_synthetic_latlon = sh_old.inv_transform(c_truth)

# 2. Analyze using both methods
print("2. Performing Forward Transform with both engines...")

# Old Method (Iterative + Simpson's Rule)
c_recovered_old = sh_old.transform(field_synthetic_latlon)

# New Method (Matrix + Riemann Sum)
# Note: Input must be flattened if the instance was initialized with flattened arrays
field_synthetic_geodesic = fast_sh.inv_transform(c_truth) # Synthesize field on geodesic grid
c_recovered_new = fast_sh.transform(field_synthetic_geodesic)

# 3. Compare Results
diff_between_methods = cp.max(cp.abs(c_recovered_old - c_recovered_new))
print(f"\nMax Discrepancy (Old vs New): {diff_between_methods:.2e}")

# 4. Compare against Ground Truth
# Errors arise from discrete integration approximations (Simpson vs Riemann)
err_old = cp.max(cp.abs(c_recovered_old - c_truth))
err_new = cp.max(cp.abs(c_recovered_new - c_truth))

print(f"Recovery Error (Old - Simpson):   {err_old:.2e}")
print(f"Recovery Error (New - Riemann):   {err_new:.2e}")

# Success Threshold
if diff_between_methods < 0.1:
    print("\n>> SUCCESS: Both methods converge to the same solution (within quadrature error).")
else:
    print("\n>> WARNING: Methods diverge significantly.")
    import matplotlib.pyplot as plt
    from planetary_sandbox.viz.planet_viewer import PlanetViewer

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PlanetViewer.plot_coefficient_complex_visualization(c_recovered_old, axes[0], fig)
    axes[0].set_title("LatLon SH Coeffs")
    PlanetViewer.plot_coefficient_complex_visualization(c_recovered_new, axes[1], fig)
    axes[1].set_title("Geodesic SH Coeffs")
    fig.suptitle("SH Coefficient Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig("tests/sh_coeffs_consistency_test.png", dpi=150)
    plt.close(fig)
    print("Saved coefficient comparison to tests/sh_coeffs_consistency_test.png")
