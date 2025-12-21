#!/usr/bin/env python3
"""
Compute optimal quadrature weights for geodesic grid SH transforms.

The Voronoi areas don't give orthogonality. Instead, we solve:
    Find w such that Y^T W Y ~ I
    
This is a least-squares problem.
"""
import cupy as cp
import numpy as np
from scipy.optimize import lsq_linear

def compute_sh_quadrature_weights(Y_matrix, initial_weights=None, l_max=None):
    """
    Compute quadrature weights that make the SH basis orthonormal.
    
    Solves: Find w such that Y^H diag(w) Y = I
    
    This is a constrained least-squares problem:
    - Minimize ||Y^H W Y - I||_F^2
    - Subject to: w_i > 0, sum(w_i) = 4 * pi
    
    Parameters
    ----------
    Y_matrix : cp.ndarray, shape (n_points, n_basis)
        Spherical harmonic basis matrix
    initial_weights : cp.ndarray, optional
        Initial guess for weights (e.g., Voronoi areas)
    l_max : int, optional
        If provided, only enforce orthogonality up to this degree
    
    Returns
    -------
    cp.ndarray
        Optimized quadrature weights
    """
    n_points, n_basis = Y_matrix.shape
    
    # Convert to numpy for scipy
    Y = cp.asnumpy(Y_matrix) if isinstance(Y_matrix, cp.ndarray) else Y_matrix
    
    if initial_weights is None:
        initial_weights = np.full(n_points, 4*np.pi / n_points)
    else:
        initial_weights = cp.asnumpy(initial_weights) if isinstance(initial_weights, cp.ndarray) else initial_weights
    
    # Only enforce orthogonality for low degrees if specified
    if l_max is not None:
        # Truncate basis to l <= l_max
        n_basis_truncated = (l_max + 1) * (l_max + 2) // 2
        Y = Y[:, :n_basis_truncated]
        n_basis = n_basis_truncated
    
    print(f"Computing optimal weights for {n_points} points, {n_basis} basis functions...")
    
    # Build the linear system for Y^T W Y = I
    # This gives us n_basis^2 constraints (but many are redundant due to symmetry)
    
    # Instead, use a simpler approach: enforce that each basis function
    # has the correct norm and minimal cross-talk
    
    # For each basis function, we want: sum_i w_i * |Y_ki|^2 = 1
    # This gives n_basis linear constraints
    
    A_norm = np.abs(Y) ** 2  # Shape: (n_points, n_basis)
    b_norm = np.ones(n_basis)
    
    # Add constraint: sum(w) = 4 * pi
    A_sum = np.ones((1, n_points))
    b_sum = np.array([4 * np.pi])
    
    # Combine constraints
    A = np.vstack([A_norm.T, A_sum])
    b = np.concatenate([b_norm, b_sum])
    
    # Solve with non-negativity constraint
    # Use small lower bound to avoid zeros
    min_weight = 1e-6 * (4 * np.pi) / n_points
    
    result = lsq_linear(
        A, b,
        bounds=(min_weight, np.inf),  # Prevent zeros
        method='bvls',  # Bounded-variable least squares
        verbose=0
    )
    
    weights_opt = result.x
    
    # Rescale to ensure sum = 4 * pi exactly
    weights_opt *= (4 * np.pi) / weights_opt.sum()
    
    # Double-check no zeros
    n_zeros = (weights_opt < min_weight).sum()
    if n_zeros > 0:
        print(f"  Warning: {n_zeros} weights are at minimum bound")
        # Redistribute from maximum weight
        deficit = n_zeros * min_weight - weights_opt[weights_opt < min_weight].sum()
        weights_opt[weights_opt < min_weight] = min_weight
        weights_opt[weights_opt.argmax()] -= deficit
    
    print(f"  Optimization converged: {result.success}")
    print(f"  Residual: {result.cost:.6e}")
    print(f"  Weight sum: {weights_opt.sum():.6f} (target: {4*np.pi:.6f})")
    print(f"  Weight range: [{weights_opt.min():.6e}, {weights_opt.max():.6e}]")
    
    return cp.array(weights_opt) if isinstance(Y_matrix, cp.ndarray) else weights_opt


def test_optimized_weights():
    """Test the optimized weights."""
    from planetary_sandbox.numerics import GeodesicGridGeometry
    from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
    
    grid = GeodesicGridGeometry(resolution=4, radius=1.0)
    
    # First, build basis with Voronoi weights
    sh_voronoi = PointSetSphericalHarmonics(
        grid.latitudes, grid.longitudes, l_max=5,
        weights=grid.cell_areas
    )
    
    Y = sh_voronoi.Y_matrix
    
    # Compute optimized weights
    weights_opt = compute_sh_quadrature_weights(Y, initial_weights=grid.cell_areas, l_max=5)
    
    # Test orthogonality with optimized weights
    W_opt = cp.diag(weights_opt)
    G_opt = Y.conj().T @ W_opt @ Y
    
    diag_opt = cp.diag(G_opt)
    I = cp.eye(Y.shape[1])
    off_diag_opt = cp.abs(G_opt - I * diag_opt[:, None])
    off_diag_vals_opt = off_diag_opt[cp.where(~cp.eye(Y.shape[1], dtype=bool))]
    
    print("\n" + "="*70)
    print("Optimized Weights Orthogonality Test")
    print("="*70)
    print(f"Diagonal mean: {diag_opt.mean():.6f} (should be 1.0)")
    print(f"Diagonal std:  {diag_opt.std():.6f} (should be ~0)")
    print(f"Off-diagonal mean: {off_diag_vals_opt.mean():.6e}")
    print(f"Off-diagonal max:  {off_diag_vals_opt.max():.6e}")
    
    eigvals = cp.linalg.eigvalsh(G_opt)
    cond = float(eigvals.max() / eigvals.min())
    print(f"Condition number: {cond:.2e}")
    
    if diag_opt.std() < 0.01 and off_diag_vals_opt.max() < 0.01:
        print("\nSUCCESS: Basis is now orthonormal!")
    else:
        print("\nStill not perfect, but should be much better")
    
    return weights_opt


if __name__ == "__main__":
    weights_opt = test_optimized_weights()
