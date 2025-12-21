import argparse

from planetary_sandbox.planet.terrain_spectral import SpectralTerrainParams
from planetary_sandbox.planet.tectonics import TectonicParams
from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.viz import PlanetViewer

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a demo planet and save a summary plot."
    )
    parser.add_argument(
        "--day-hours",
        type=float,
        default=24.0,
        help="Length of the sidereal day in hours (default: 24.0)",
    )
    parser.add_argument(
        "--mass-earth-masses",
        type=float,
        default=1.0,
        help="Planet mass in Earth masses (default: 1.0)",
    )
    parser.add_argument(
        "--eq_radius-earth_units",
        type=float,
        default=1.0,
        help="Equatorial radius in multiples of the Earth's (default: 1.0)",
    )

    parser.add_argument(
        "--l-max",
        type=int,
        default=15,
        help="Maximum spherical harmonic degree (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for terrain generation (default: None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="planet_summary.png",
        help="Output PNG file for the summary plot.",
    )

    # parser.add_argument(
    #     "--grid-resolution",
    #     type=int,
    #     nargs=2,
    #     default=(128, 256),
    #     help="Grid resolution as num_lat num_lon (default: 256 512)"
    # )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        nargs=1,
        default=3,
        help="Geodesic grid resolution (default: 3)"
    )

    args = parser.parse_args()

    # Build planetary parameters
    params = PlanetaryParameters.from_earth_like(day_hours=args.day_hours, 
                                                mass_earth_units=args.mass_earth_masses,
                                                radius_earth_units=args.eq_radius_earth_units
                                                )

    # Factory method to generate planet
    planet = Planet.generate(
        params=params,
        grid_resolution=args.grid_resolution,
        terrain_params=SpectralTerrainParams(
            seed=args.seed
        ),
        tectonic_params=TectonicParams(),
        l_max=args.l_max
    )

    # Plot
    viewer = PlanetViewer(planet)
    fig = viewer.plot_summary()

    fig.savefig(f"out/{args.output}", dpi=200)
    print(f"Saved planet summary to out/{args.output}")


if __name__ == "__main__":
    main()
