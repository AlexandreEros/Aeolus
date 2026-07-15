"""
Auto-optimizing spherical harmonics for geodesic grids.

This module provides a wrapper that automatically computes optimal quadrature
weights for geodesic grids, caching them for reuse.
"""
import cupy as cp
import numpy as np
import warnings
from pathlib import Path
import pickle
from scipy.optimize import lsq_linear

from .geodesic_grid import GeodesicGridGeometry
from .fast_geodesic_sh import PointSetSphericalHarmonics

__all__ = ["GeodesicSphericalHarmonics", "OptimizedGeodesicSH"]

# Minimum grid points per spherical-harmonic basis function for the discrete
# (non-exact) quadrature to keep analysis/synthesis round trips well conditioned.
# Empirically (audit 2026-07): ~2.4 pts/basis gives O(10%) leakage, ~9.5 gives
# O(1%). See docs/KNOWN_RISKS.md R-2 and docs/VALIDATION_PLAN.md A-1.
MIN_POINTS_PER_BASIS = 6.0


def _warn_if_underresolved(n_points: int, l_max: int) -> None:
    n_basis = (l_max + 1) * (l_max + 2) // 2
    ratio = n_points / n_basis
    if ratio < MIN_POINTS_PER_BASIS:
        l_safe = int((np.sqrt(1.0 + 8.0 * n_points / MIN_POINTS_PER_BASIS) - 3.0) / 2.0)
        warnings.warn(
            f"Geodesic SH is under-resolved: {n_points} points for l_max={l_max} "
            f"({n_basis} basis functions, {ratio:.1f} pts/basis < "
            f"{MIN_POINTS_PER_BASIS:.0f}). Analysis/synthesis round trips will leak "
            f"energy across modes and the solver will lose invariants spuriously. "
            f"Use l_max <= {l_safe} at this resolution, or a finer grid. "
            f"See docs/KNOWN_RISKS.md R-2.",
            stacklevel=3,
        )


class GeodesicSphericalHarmonics:
    """
    Weight-aware geodesic-grid SH wrapper around PointSetSphericalHarmonics.

    By default, weights are optimized (least-squares) and cached for reuse.
    """

    def __init__(self, grid: GeodesicGridGeometry, l_max=None, cache_dir=None,
                 weights="voronoi", latitudes=None, longitudes=None):
        """
        Parameters
        ----------
        grid : GeodesicGridGeometry, optional
            Geodesic grid geometry. Required for optimized weights.
        l_max : int
            Maximum SH degree
        cache_dir : Path, optional
            Directory to cache computed weights. If None, recomputes every time.
        weights : {"optimize", "voronoi", "uniform"} or array-like, optional
            Weight mode when a grid is provided. For point sets without a grid,
            provide an explicit weights array or leave as None for uniform.
        latitudes, longitudes : array-like, optional
            Point-set sampling (used when no grid is provided).
        """

        self.grid = grid
        if l_max is None:
            raise ValueError("l_max is required.")
        self.l_max = l_max
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._basis_id = "complex_m_ge_0_v2"

        if grid is None:
            if latitudes is None or longitudes is None:
                raise ValueError("Provide either grid or latitudes/longitudes.")
            _warn_if_underresolved(len(latitudes), l_max)
            if isinstance(weights, str) and weights in ("optimize", "voronoi"):
                raise ValueError("weights='optimize' or 'voronoi' requires a grid.")
            if weights is None or (isinstance(weights, str) and weights == "uniform"):
                weights_arr = None
            else:
                weights_arr = weights
            self.sh = PointSetSphericalHarmonics(
                latitudes,
                longitudes,
                l_max,
                weights=weights_arr
            )
            self.weights = self.sh.weights
            return

        if latitudes is not None or longitudes is not None:
            raise ValueError("Provide either grid or latitudes/longitudes, not both.")

        _warn_if_underresolved(grid.n_points, l_max)

        weights_arr = None
        if isinstance(weights, str):
            weights_mode = weights.lower()
            if weights_mode == "optimize":
                # Try to load cached weights
                weights_arr = self._load_cached_weights()

                if weights_arr is None:
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

                    weights_arr = self._compute_optimal_weights(Y, initial_weights, l_max)

                    # Cache for next time
                    self._save_cached_weights(weights_arr)
                    print("Weights computed and cached")
                else:
                    print(f"Loaded cached weights for resolution={grid.resolution}, l_max={l_max}")
            elif weights_mode == "voronoi":
                w = grid.cell_areas
                # If cell_areas are physical areas (m^2), convert to solid angle:
                w = w / (self.grid.radius**2)
                # Always normalize so sum(w) == 4π (removes any residual scaling drift):
                w = w * (4*cp.pi) / cp.sum(w)
                weights_arr = w
            elif weights_mode == "uniform":
                weights_arr = None
            else:
                raise ValueError(f"Unknown weights mode: {weights}")
        elif weights is None:
            weights_arr = None
        else:
            weights_arr = weights

        # Build final SH object with chosen weights
        self.sh = PointSetSphericalHarmonics(
            grid.latitudes,
            grid.longitudes,
            l_max,
            weights=cp.array(weights_arr) if weights_arr is not None else None
        )

        self.weights = self.sh.weights
    
    
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
        """Forward transform using configured weights."""
        return self.sh.transform(values)
    
    
    def inv_transform(self, coeffs):
        """Inverse transform."""
        return self.sh.inv_transform(coeffs)
    
    
    def inverse_transform(self, coeffs):
        """Alias for inv_transform."""
        return self.inv_transform(coeffs)

    def __getattr__(self, name):
        return getattr(self.sh, name)


def test_optimized_vs_voronoi(resolution=4, radius=1.0, l_max=15):
    """Compare Voronoi weights vs optimized weights."""
    from planetary_sandbox.numerics import GeodesicGridGeometry
    from planetary_sandbox.numerics.fast_geodesic_sh import PointSetSphericalHarmonics
    
    grid = GeodesicGridGeometry(resolution=resolution, radius=radius)
    
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
    sh_opt = GeodesicSphericalHarmonics(grid, l_max, cache_dir=Path("tests/.sh_cache"))
    
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

# Backwards-compatible alias
OptimizedGeodesicSH = GeodesicSphericalHarmonics
