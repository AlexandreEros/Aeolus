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

    @classmethod
    def ideal_sphere(cls, radius_m: float, sidereal_day_s: float,
                     mass_kg: float = 5.972e24) -> 'PlanetaryParameters':
        """A perfect sphere whose dynamical radius is EXACTLY ``radius_m``.

        Benchmark suites (Williamson et al. 1992) prescribe the planetary
        radius as an exact constant on a perfect sphere. The ordinary
        constructors derive a volumetric mean radius through the rotational
        oblateness model, which shrinks the dynamical radius by ~0.06% for
        Earth-like rotation — a silent noncanonical substitution for a
        benchmark. This constructor overrides the derived shape quantities
        so the sphere is genuinely spherical: ``radius``,
        ``equatorial_radius`` and ``polar_radius`` all equal ``radius_m``
        and ``oblateness`` is exactly zero. ``angular_velocity`` remains
        ``2*pi/sidereal_day_s`` from ``__post_init__``.
        """
        p = cls(mass=mass_kg, equatorial_radius=float(radius_m),
                sidereal_day=float(sidereal_day_s))
        p.oblateness = 0.0
        p.polar_radius = float(radius_m)
        p.radius = float(radius_m)
        p.volume = 4.0 / 3.0 * np.pi * float(radius_m) ** 3
        p.density = p.mass / p.volume
        return p
