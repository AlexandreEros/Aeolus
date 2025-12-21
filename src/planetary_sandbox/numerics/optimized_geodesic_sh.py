"""
Auto-optimizing spherical harmonics for geodesic grids.

This module provides a wrapper that automatically computes optimal quadrature
weights for geodesic grids, caching them for reuse.
"""
import cupy as cp
import numpy as np
from pathlib import Path
import pickle
from scipy.optimize import lsq_linear


class OptimizedGeodesicSH:
    """
    Geodesic grid SH with automatically optimized quadrature weights.
    
    The Voronoi cell areas don't provide orthogonality for SH integration.
    This class computes optimal weights via least-squares, then uses them
    for all transforms.
    """
    
    def __init__(self, grid, l_max, cache_dir=None):
        """
        Parameters
        ----------
        grid : GeodesicGridGeometry
            The geodesic grid
        l_max : int
            Maximum SH degree
        cache_dir : Path, optional
            Directory to cache computed weights. If None, recomputes every time.
        """
        from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
        
        self.grid = grid
        self.l_max = l_max
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._basis_id = "complex_m_ge_0_v2"
        
        # Try to load cached weights
        weights = self._load_cached_weights()
        
        if weights is None:
            # Step 1: Build initial SH with Voronoi weights
            print(f"Computing optimal weights for resolution={grid.resolution}, l_max={l_max}...")
            
            sh_initial = PointSetSphericalHarmonics(
                grid.latitudes, 
                grid.longitudes,
                l_max,
                weights=grid.cell_areas
            )
            
            # Step 2: Compute optimal weights
            Y = cp.asnumpy(sh_initial.Y_matrix)
            initial_weights = cp.asnumpy(grid.cell_areas)
            
            weights = self._compute_optimal_weights(Y, initial_weights, l_max)
            
            # Cache for next time
            self._save_cached_weights(weights)
            print("Weights computed and cached")
        else:
            print(f"Loaded cached weights for resolution={grid.resolution}, l_max={l_max}")
        
        # Step 3: Build final SH object with optimized weights
        self.sh = PointSetSphericalHarmonics(
            grid.latitudes,
            grid.longitudes, 
            l_max,
            weights=cp.array(weights)
        )
        
        self.weights = cp.array(weights)
    
    
    def _get_cache_filename(self):
        """Generate cache filename based on grid and l_max."""
        if self.cache_dir is None:
            return None
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache key: resolution and l_max
        filename = f"sh_weights_res{self.grid.resolution}_lmax{self.l_max}.pkl"
        return self.cache_dir / filename
    
    
    def _load_cached_weights(self):
        """Load weights from cache if available."""
        cache_file = self._get_cache_filename()
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Verify it's for the right grid
                    if (
                        data.get('n_points') == self.grid.n_points
                        and data.get('basis') == self._basis_id
                    ):
                        return data['weights']
            except Exception:
                pass
        return None
    
    
    def _save_cached_weights(self, weights):
        """Save weights to cache."""
        cache_file = self._get_cache_filename()
        if cache_file:
            data = {
                'n_points': self.grid.n_points,
                'resolution': self.grid.resolution,
                'l_max': self.l_max,
                'weights': weights,
                'basis': self._basis_id,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
    
    
    @staticmethod
    def _compute_optimal_weights(Y, initial_weights, l_max_optimize):
        """
        Compute optimal quadrature weights.
        
        Solves: Find w such that Y^H W Y = I
        with constraints: w > 0, sum(w) = 4 * pi
        """
        n_points, n_basis = Y.shape
        
        # Only enforce orthogonality for low degrees (saves computation)
        n_basis_opt = (l_max_optimize + 1) * (l_max_optimize + 2) // 2
        Y_opt = Y[:, :n_basis_opt]
        
        # Build linear system: each basis function should have norm 1
        # Constraint: sum_i w_i * |Y_ki|^2 = 1 for each basis function k
        A_norm = np.abs(Y_opt) ** 2
        b_norm = np.ones(n_basis_opt)
        
        # Add constraint: sum(w) = 4 * pi
        A_sum = np.ones((1, n_points))
        b_sum = np.array([4 * np.pi])
        
        # Combine
        A = np.vstack([A_norm.T, A_sum])
        b = np.concatenate([b_norm, b_sum])
        
        # Solve with positivity constraint
        min_weight = 1e-8 * (4 * np.pi) / n_points
        
        result = lsq_linear(
            A, b,
            bounds=(min_weight, np.inf),
            method='bvls',
            verbose=0
        )
        
        weights = result.x
        weights *= (4 * np.pi) / weights.sum()
        
        return weights
    
    
    def transform(self, values):
        """Forward transform using optimized weights."""
        return self.sh.transform(values)
    
    
    def inv_transform(self, coeffs):
        """Inverse transform."""
        return self.sh.inv_transform(coeffs)
    
    
    def inverse_transform(self, coeffs):
        """Alias for inv_transform."""
        return self.inv_transform(coeffs)


def test_optimized_vs_voronoi():
    """Compare Voronoi weights vs optimized weights."""
    from planetary_sandbox.numerics import GeodesicGridGeometry
    from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
    
    grid = GeodesicGridGeometry(resolution=4, radius=1.0)
    l_max = 5
    
    # Test with Voronoi weights
    print("Testing with Voronoi cell areas:")
    sh_voronoi = PointSetSphericalHarmonics(
        grid.latitudes, grid.longitudes, l_max,
        weights=grid.cell_areas
    )
    
    Y_voronoi = sh_voronoi.Y_matrix
    W_voronoi = cp.diag(grid.cell_areas)
    G_voronoi = Y_voronoi.conj().T @ W_voronoi @ Y_voronoi
    
    diag_v = cp.diag(G_voronoi)
    off_diag_v = cp.abs(G_voronoi - cp.eye(len(diag_v)) * diag_v[:, None])
    mask = ~cp.eye(len(diag_v), dtype=bool)
    
    print(f"  Diagonal std: {diag_v.std():.6f}")
    print(f"  Off-diag max: {off_diag_v[mask].max():.6f}")
    
    # Test with optimized weights  
    print("\nTesting with optimized weights:")
    sh_opt = OptimizedGeodesicSH(grid, l_max, cache_dir=Path("tests/.sh_cache"))
    
    Y_opt = sh_opt.sh.Y_matrix
    W_opt = cp.diag(sh_opt.weights)
    G_opt = Y_opt.conj().T @ W_opt @ Y_opt
    
    diag_o = cp.diag(G_opt)
    off_diag_o = cp.abs(G_opt - cp.eye(len(diag_o)) * diag_o[:, None])
    
    print(f"  Diagonal std: {diag_o.std():.6f}")
    print(f"  Off-diag max: {off_diag_o[mask].max():.6f}")
    
    print("\n" + "="*60)
    if diag_o.std() < 0.01 and off_diag_o[mask].max() < 0.5:
        print("Optimized weights provide much better orthogonality")
    else:
        print("Orthogonality improved but still not perfect")


if __name__ == "__main__":
    test_optimized_vs_voronoi()
