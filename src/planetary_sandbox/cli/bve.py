from __future__ import annotations

import json
import pathlib
import tempfile
import numpy as np

from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.runner import run_bve
from planetary_sandbox.run.bve.initial_conditions import make_ic, INITIAL_CONDITIONS
from planetary_sandbox.run.bve.io import create_run_dir, write_run_manifest
from planetary_sandbox.viz.vorticity_viewer import VorticityViewer


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    """Ensure the runs *base* directory exists and is writable; fall back if not."""
    out_dir = pathlib.Path(requested_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_path = out_dir / ".write_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return out_dir, False
    except OSError:
        # Fall back to the system temp location, NOT dir="." — dropping the
        # temp dir into the repo root left persistent, sometimes ACL-locked
        # `psx-bve-*` directories behind (one of which broke bare `pytest`
        # collection). Keep run outputs out of the project tree entirely.
        fallback_root = pathlib.Path(tempfile.mkdtemp(prefix="psx-bve-"))
        fallback_out_dir = fallback_root / out_dir.name
        fallback_out_dir.mkdir(parents=True, exist_ok=True)
        return fallback_out_dir, True


def build_parser():
    import argparse
    parser = argparse.ArgumentParser("psx-bve",
        description="Run barotropic vorticity equation on a planet.",
        usage="psx-bve [options]"
    )

    # Default (l_max=21, resolution=4) keeps ~10 grid points per SH basis
    # function, within the transform's usable envelope (see KNOWN_RISKS.md R-2).
    parser.add_argument("--lmax", type=int, default=21)
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
    parser.add_argument("--product-quadrature", type=str, default="fine",
                        choices=["fine", "coarse"],
                        help="Where nonlinear (pseudospectral) products are "
                             "evaluated and analyzed. 'fine' (default): on a "
                             "reusable resolution-(r+1) product grid "
                             "('overresolved product quadrature', "
                             "KNOWN_RISKS.md R-3). 'coarse': the historical "
                             "state-grid path, kept for A/B comparisons. "
                             "There is no silent fallback: an unsupported "
                             "combination raises at startup.")
    parser.add_argument("--out", type=str, default="runs",
                        help="Base directory for run outputs. Each run creates a "
                             "unique subdirectory under this (or under "
                             "<out>/<experiment>/ if --experiment is given).")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Optional grouping name; runs go to <out>/<experiment>/<run_id>/.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reuse an existing run directory if the auto-generated "
                             "run ID collides (same command in the same second). "
                             "Off by default to keep runs immutable.")
    return parser


def main():
    args = build_parser().parse_args()

    base_dir, used_fallback = _resolve_writable_base_dir(args.out)
    if used_fallback:
        banner = "=" * 72
        print(banner)
        print(f"WARNING: requested --out '{args.out}' is not writable.")
        print(f"         Run outputs will be written OUTSIDE the project tree to:")
        print(f"           {base_dir.resolve()}")
        print(banner)

    # Build the config dict *before* creating the run dir so the run ID reflects
    # exactly what will be written to disk.
    run_config = vars(args).copy()
    run_config["out"] = str(base_dir)

    run_dir = create_run_dir(
        base_dir, run_config,
        experiment=args.experiment,
        overwrite=args.overwrite,
    )
    out_dir = run_dir.path
    run_dir.update_latest_pointer()
    print(f"Run directory: {out_dir}")
    if run_dir.reused:
        print("  (reused via --overwrite; existing files may be replaced)")

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=args.day_hours,
            radius_earth_units=args.radius_earth_units),
        grid_resolution=args.resolution,
        l_max=args.lmax,
        product_quadrature=args.product_quadrature,
    )

    # Initial condition on grid, then transform -> spectral ζ_lm
    zeta0_grid = make_ic(args.scenario, planet)                  # (nlat, nlon) cupy or numpy ok
    zeta0_lm = planet.sh.transform(zeta0_grid)

    # Save run config + provenance manifest (git commit, versions, GPU, argv)
    run_config["run_id"] = run_dir.run_id
    run_config["experiment"] = args.experiment
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=args.experiment)

    run_bve(planet=planet,
            zeta0_lm=zeta0_lm,
            dt_snapshots=args.dt_snapshots,
            t_end_days=args.duration_days,
            out_dir=out_dir,
            viscosity=args.viscosity,
            scenario=args.scenario,
            figure_metadata=run_dir.figure_metadata())


if __name__ == "__main__":
    main()
