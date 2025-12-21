from dataclasses import dataclass
import numpy as np
from scipy import constants

@dataclass
class PlanetaryParameters:
    """Core physical parameters that define a planet."""
    mass: float  # kg
    equatorial_radius: float  # meters
    sidereal_day: float  # seconds

    def __post_init__(self):
        """Calculate derived quantities."""
        self.angular_velocity = 2 * np.pi / self.sidereal_day
        self.oblateness = self._calculate_oblateness()
        self.polar_radius = self.equatorial_radius * (1 - self.oblateness)
        self.volume = 4/3 * np.pi * self.equatorial_radius**3 * (1 - self.oblateness)
        self.radius = np.cbrt(self.volume / (4/3 * np.pi))
        self.density = self.mass / self.volume


    def _calculate_oblateness(self) -> float:
        """
        Calculate oblateness from rotation using first-order approximation.

        For a uniform density sphere (crude but decent first approximation):
        f ≈ (ω²R³)/(2GM) = (ω²R)/(2g) where g = GM/R²

        For more accuracy, use Darwin-de Sitter formula or full hydrostatic equilibrium.
        """
        R = self.equatorial_radius
        M = self.mass
        omega = self.angular_velocity

        # First order approximation (uniform density)
        f = (omega**2 * R**3) / (2 * constants.G * M)

        # Clamp to reasonable values
        return np.clip(f, 0.0, 0.5)

    @classmethod
    def from_earth_like(cls,
                       mass_earth_units: float = 1.0,
                       radius_earth_units: float = 1.0,
                       day_hours: float = 24.0) -> 'PlanetaryParameters':
        """Convenient constructor using Earth units."""
        EARTH_MASS = 5.972e24  # kg
        EARTH_RADIUS = 6.371e6  # m

        return cls(
            mass=mass_earth_units * EARTH_MASS,
            equatorial_radius=radius_earth_units * EARTH_RADIUS,
            sidereal_day=day_hours * 3600.0
        )

    @classmethod
    def from_si(cls, mass_kg, eqt_radius_m, day_s) -> 'PlanetaryParameters':
        return cls(
            mass=mass_kg,
            equatorial_radius=eqt_radius_m,
            sidereal_day=day_s
        )
    