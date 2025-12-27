from __future__ import annotations

import pathlib
import json
import numpy as np

from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.runner import run_bve
from planetary_sandbox.run.bve.initial_conditions import make_ic, INITIAL_CONDITIONS
from planetary_sandbox.viz.vorticity_viewer import VorticityViewer

def main():
    import argparse
    parser = argparse.ArgumentParser("psx-bve",
        description="Run barotropic vorticity equation on a planet.",
        usage="psx-bve [options]"
    )

    parser.add_argument("--lmax", type=int, default=45)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--nlat", type=int, default=128)
    parser.add_argument("--nlon", type=int, default=256)
    parser.add_argument("--day-hours", type=float, default=np.inf)
    parser.add_argument("--radius-earth-units", type=float, default=1.0)
    parser.add_argument("--duration-days", type=float, default=1.0)
    parser.add_argument("--dt-snapshots", type=float, default=6*3600.0)
    parser.add_argument("--scenario", type=str, default="two_vortices",
                   choices=list(INITIAL_CONDITIONS.keys()))
    parser.add_argument("--viscosity", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="out/bve_run")

    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=args.day_hours,
            radius_earth_units=args.radius_earth_units),
        grid_resolution=args.resolution,
        l_max=args.lmax,
    )

    # Initial condition on grid, then transform -> spectral ζ_lm
    zeta0_grid = make_ic(args.scenario, planet)                  # (nlat, nlon) cupy or numpy ok
    zeta0_lm = planet.sh.transform(zeta0_grid)

    # Save run config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    run_bve(planet=planet,
            zeta0_lm=zeta0_lm,
            dt_snapshots=args.dt_snapshots,
            t_end_days=args.duration_days,
            out_dir=out_dir,
            viscosity=args.viscosity,
            scenario=args.scenario)
                         

if __name__ == "__main__":
    main()
