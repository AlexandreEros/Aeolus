#!/usr/bin/env python3
"""
Test spherical harmonic basis orthogonality.

For a proper quadrature scheme with weights w_i:
    Σ_i w_i · Y_l^m(θ_i, φ_i) · Y_l'^m'(θ_i, φ_i) ≈ δ_ll' δ_mm'

This checks if your geodesic grid + Voronoi weights give orthogonality.
"""
import numpy as np
import cupy as cp
from planetary_sandbox.numerics import GeodesicGridGeometry

def test_basis_orthogonality(l_max=5, resolution=4):
    """Test if Y^T W Y ≈ I where W = diag(weights)"""
    
    grid = GeodesicGridGeometry(resolution=resolution, radius=1.0)
    
    # Get quadrature weights
    weights = cp.array(grid.cell_areas)
    
    # Build Y matrix using your CURRENT (buggy) implementation
    from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
    
    sh = PointSetSphericalHarmonics(
        grid.latitudes, 
        grid.longitudes, 
        l_max, 
        weights=weights
    )
    
    # Y is (n_points, n_basis)
    Y = sh.Y_matrix
    n_points, n_basis = Y.shape
    
    print(f"Grid: {n_points} points, {n_basis} basis functions (l_max={l_max})")
    print(f"Total sphere area: {weights.sum():.6f} (should be 4π = {4*np.pi:.6f})")
    
    # Compute Gram matrix: G = Y^H W Y
    # For orthonormal basis: G should be identity
    W = cp.diag(weights)
    G = Y.conj().T @ W @ Y
    
    # Check diagonal (should be 1)
    diag = cp.diag(G)
    print(f"\nDiagonal of Gram matrix:")
    print(f"  Mean: {diag.mean():.6f} (should be 1.0)")
    print(f"  Std:  {diag.std():.6f} (should be ~0)")
    print(f"  Min:  {diag.min():.6f}")
    print(f"  Max:  {diag.max():.6f}")
    
    # Check off-diagonal (should be 0)
    I = cp.eye(n_basis)
    off_diag = cp.abs(G - I * diag[:, None])
    off_diag_vals = off_diag[cp.where(~cp.eye(n_basis, dtype=bool))]
    
    print(f"\nOff-diagonal of Gram matrix (should be ~0):")
    print(f"  Mean: {off_diag_vals.mean():.6e}")
    print(f"  Max:  {off_diag_vals.max():.6e}")
    
    # Compute condition number
    eigvals = cp.linalg.eigvalsh(G.real)  # Hermitian, so eigenvalues are real
    cond = float(eigvals.max() / eigvals.min())
    print(f"\nCondition number of Gram matrix: {cond:.2e}")
    print(f"  (should be close to 1.0 for orthonormal basis)")
    
    # Show first few basis functions
    print(f"\nFirst few basis functions (should have norm ≈ 1):")
    for i in range(min(10, n_basis)):
        l = sh.l_indices[i]
        m = sh.m_indices[i]
        norm = cp.sqrt(cp.sum(weights * cp.abs(Y[:, i])**2))
        print(f"  Y_{l}^{m}: norm = {norm:.6f}")
    
    return G, diag, off_diag_vals, eigvals


if __name__ == "__main__":
    print("Testing basis orthogonality with CURRENT implementation:")
    print("=" * 70)
    G, diag, off_diag, eigvals = test_basis_orthogonality(l_max=5, resolution=4)
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    if diag.std() > 0.01:
        print("❌ Diagonal is not constant → basis not properly normalized")
    else:
        print("✓ Diagonal is roughly constant")
    
    if off_diag.max() > 0.1:
        print("❌ Large off-diagonal elements → basis not orthogonal")
        print("   This explains the spectral leakage you're seeing!")
    else:
        print("✓ Off-diagonal elements are small")
    
    if abs(diag.mean() - 1.0) > 0.01:
        print("❌ Diagonal mean != 1 → normalization is wrong")
    else:
        print("✓ Diagonal mean ≈ 1")
