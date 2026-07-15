"""Generate a demo planet and save a summary plot.

``aeolus planet generate`` is the canonical interface; ``psx-gen`` is a
compatibility alias. Heavy imports (CuPy, matplotlib) happen after
argument parsing.
"""
from __future__ import annotations

import argparse
import pathlib


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--day-hours", type=float, default=24.0,
        help="Length of the sidereal day in hours (default: 24.0).")
    parser.add_argument(
        "--mass-earth-masses", type=float, default=1.0,
        help="Planet mass in Earth masses (default: 1.0).")
    parser.add_argument(
        "--radius-earth-units", "--eq_radius-earth_units",
        dest="radius_earth_units", type=float, default=1.0,
        help="Equatorial radius in Earth radii (default: 1.0). "
             "--eq_radius-earth_units is a compatibility alias.")
    parser.add_argument(
        "--l-max", type=int, default=15,
        help="Maximum spherical harmonic degree (default: 15).")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for terrain generation (default: None).")
    parser.add_argument(
        "--output", type=str, default="planet_summary.png",
        help="Output PNG file; relative paths are written under out/ "
             "(default: planet_summary.png).")
    parser.add_argument(
        "--grid-resolution", type=int, default=3,
        help="Geodesic grid subdivision level (default: 3).")


def build_parser(prog: str = "psx-gen") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate a demo planet and save a summary plot.")
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    # Heavy imports (CuPy via Planet, matplotlib via viz) after parsing.
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.planet.terrain_spectral import SpectralTerrainParams
    from planetary_sandbox.planet.tectonics import TectonicParams
    from planetary_sandbox.viz import PlanetViewer

    params = PlanetaryParameters.from_earth_like(
        day_hours=args.day_hours,
        mass_earth_units=args.mass_earth_masses,
        radius_earth_units=args.radius_earth_units,
    )

    planet = Planet.generate(
        params=params,
        grid_resolution=args.grid_resolution,
        terrain_params=SpectralTerrainParams(seed=args.seed),
        tectonic_params=TectonicParams(),
        l_max=args.l_max,
    )

    viewer = PlanetViewer(planet)
    fig = viewer.plot_summary()

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = pathlib.Path("out") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved planet summary to {out_path}")
    return 0


def main() -> int:
    """psx-gen == aeolus planet generate."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
