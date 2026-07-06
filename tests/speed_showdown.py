import time
import numpy as np
import cupy as cp

from planetary_sandbox.numerics import (
    LatLonSphericalHarmonics, 
    PointSetSphericalHarmonics,
)

print("--- Speed Showdown ---\n")

# 1. Setup comparable grids
# We'll use a structured grid for fairness as the Old method requires it.
n_lat = 45 # Odd number
n_lon = 90 # Even/Odd doesn't strictly matter for FastSH, but good for simpson
l_max = 30
n_points = n_lat * n_lon

print(f"Grid Size: {n_lat}x{n_lon} ({n_points} points)")
print(f"L_max: {l_max}\n")

# Create Grid
# FIX: Ensure colatitude (phi) is strictly INCREASING (0 to pi)
# so that integration step dy is positive. 
# Otherwise Simpson's rule produces negative area.
phi_1d = np.linspace(0, np.pi, n_lat)
lat_1d = np.pi/2 - phi_1d  # Decreasing latitude
lon_1d = np.linspace(0, 2*np.pi, n_lon)

lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

# Weights for New Method (explicit area)
# dlat for weight calculation (scalar magnitude)
dlat = np.pi / (n_lat - 1)
dlon = 2 * np.pi / n_lon
weights = np.cos(lat_grid) * dlat * dlon 

# Dummy Data
data_np = np.sin(lat_grid) * np.cos(lon_grid)
data_cp = cp.asarray(data_np)

# ==========================================
# 2. Benchmark Old Method (Iterative)
# ==========================================
print("Benchmarking Old Method (SphericalHarmonics - Iterative)...")
sh_old = LatLonSphericalHarmonics(l_max=l_max)
# Pass grid explicitly. phi_1d is strictly increasing.
sh_old.set_grid(lon_1d, phi_1d)

# Warmup
_ = sh_old.transform(data_cp)
cp.cuda.Stream.null.synchronize()

t0 = time.time()
coeffs_old = sh_old.transform(data_cp)
cp.cuda.Stream.null.synchronize()
t_old = time.time() - t0
print(f"Old Method Time: {t_old*1000:.2f} ms")


# ==========================================
# 3. Benchmark New Method (Matrix)
# ==========================================
print("\nBenchmarking New Method (PointSetSphericalHarmonics - Matrix)...")

# Initialization (One-time cost)
t0_init = time.time()
pointset_sh = PointSetSphericalHarmonics(
    lat_grid.ravel(),
    lon_grid.ravel(),
    l_max=l_max,
    weights=weights.ravel(),
)
cp.cuda.Stream.null.synchronize()
print(f"Initialization Time: {(time.time()-t0_init)*1000:.2f} ms")

# Warmup
_ = pointset_sh.transform(data_cp)
cp.cuda.Stream.null.synchronize()

t0 = time.time()
coeffs_new = pointset_sh.transform(data_cp)
cp.cuda.Stream.null.synchronize()
t_new = time.time() - t0
print(f"New Method Time: {t_new*1000:.2f} ms")


# ==========================================
# 4. Result
# ==========================================
speedup = t_old / t_new
print(f"\nSpeedup Factor: {speedup:.1f}x")

# Check accuracy agreement
diff = cp.abs(coeffs_old - coeffs_new)
print(f"Max Difference in Coeffs: {cp.max(diff):.6e}")
