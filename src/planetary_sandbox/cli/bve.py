from __future__ import annotations

import json
import pathlib
import tempfile
import numpy as np

from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.runner import run_bve
from planetary_sandbox.run.bve.initial_conditions import make_ic, INITIAL_CONDITIONS
from planetary_sandbox.viz.vorticity_viewer import VorticityViewer


def _resolve_writable_out_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    out_dir = pathlib.Path(requested_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_path = out_dir / ".write_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return out_dir, False
    except OSError:
        fallback_root = pathlib.Path(tempfile.mkdtemp(prefix="psx-bve-", dir="."))
        fallback_out_dir = fallback_root / out_dir.name
        fallback_out_dir.mkdir(parents=True, exist_ok=True)
        return fallback_out_dir, True


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

    out_dir, used_fallback = _resolve_writable_out_dir(args.out)
    if used_fallback:
        print(f"Requested output path '{args.out}' is not writable. Using '{out_dir}' instead.")

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
    run_config = vars(args).copy()
    run_config["out"] = str(out_dir)
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    run_bve(planet=planet,
            zeta0_lm=zeta0_lm,
            dt_snapshots=args.dt_snapshots,
            t_end_days=args.duration_days,
            out_dir=out_dir,
            viscosity=args.viscosity,
            scenario=args.scenario)
                         

if __name__ == "__main__":
    main()
