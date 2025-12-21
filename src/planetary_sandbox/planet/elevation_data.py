import numpy as np
import cupy as cp
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class ElevationData:
    """
    Container for different elevation representations.

    Elevations are stored as deviations from reference surfaces.
    """
    # Raw noise-based topography
    surface_height: Optional[np.ndarray] = field(default=None, repr=False)  # meters above mean radius

    # Geometric elevations
    radial_distance: Optional[np.ndarray] = field(default=None, repr=False)  # distance from center

    # Spherical harmonic representations (stored as coefficients)
    sh_coeffs: Optional[cp.ndarray[Tuple[int, int], cp.complex128]] = field(default=None, repr=False)

    # Power spectra
    power_spectrum: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def max_degree(self) -> int:
        """Maximum spherical harmonic degree available."""
        if self.sh_coeffs is None: return 0
        else: return self.sh_coeffs.shape[0]

    def get_j2(self) -> float:
        """
        Extract J₂ (dynamical form factor) from spherical harmonics.
        J₂ = -√5 * C₂₀ (normalized)
        """
        if self.sh_coeffs is None:
            raise ValueError("Spherical harmonic coefficients are not available.")
        c20 = self.sh_coeffs[2,0]
        return -np.sqrt(5) * c20.real

    @property
    def oblateness_from_sh(self) -> float:
        """
        Estimate geometric oblateness from C₂₀ coefficient.
        For small oblateness: f ≈ -√5 * C₂₀ = J₂
        """
        return self.get_j2()
    