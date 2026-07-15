"""Reproduce the scientific figures embedded in README.md.

The source run capsules remain under the ignored ``runs/`` tree.  The rendered
PNGs and a compact provenance record are written to tracked ``docs/assets/``.

Examples (from the repository root)::

    python docs/readme_figures.py rotation-runs
    python docs/readme_figures.py render \
        --dynamic-run runs/readme-dynamic/<run-id> \
        --rh4-run runs/validation-rh4/<run-id> \
        --geodesic-run runs/validation-rh4/<run-id> \
        --latlon-run runs/validation-rh4-latlon/<run-id> \
        --rotating-rotation-root runs/readme-rotations-rot24h
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from planetary_sandbox.numerics.geodesic_grid import GeodesicGridGeometry
from planetary_sandbox.numerics.latlon_grid import GaussLatLonGridGeometry
from planetary_sandbox.planet import Planet, PlanetaryParameters
from planetary_sandbox.run.bve.io import create_run_dir, write_run_manifest
from planetary_sandbox.run.bve.runner import run_bve


ROOT = pathlib.Path(__file__).resolve().parents[1]
ROTATION_CONFIG = {
    "grid": "geodesic",
    "resolution": 4,
    "nlat": 128,
    "nlon": 256,
    "lmax": 21,
    "day_hours": math.inf,
    "radius_earth_units": 1.0,
    "duration_days": 1.0,
    "dt_snapshots": 43200.0,
    "viscosity": 0.0,
    "product_quadrature": "fine",
}
ROTATIONS = {
    "north-pole": (90.0 - 1e-6, 0.0),
    "equator-0e": (0.0, 0.0),
    "equator-90e": (0.0, 90.0),
}


def _antipodal_vortices(planet: Planet, lat_deg: float, lon_deg: float) -> cp.ndarray:
    """Equal-and-opposite 10-degree Gaussian vortices at antipodal centers."""
    # Build the one-time IC on the host.  This avoids compiling a chain of
    # elementwise CuPy kernels solely for three documentation runs; the actual
    # transform and time integration remain GPU-resident.
    lat = cp.asnumpy(cp.asarray(planet.grid.point_latitudes))
    lon = cp.asnumpy(cp.asarray(planet.grid.point_longitudes))
    lat0 = np.deg2rad(lat_deg)
    lon0 = np.deg2rad(lon_deg)
    sigma = np.deg2rad(10.0)

    def gaussian(center_lat, center_lon):
        cosd = (np.sin(lat) * np.sin(center_lat)
                + np.cos(lat) * np.cos(center_lat) * np.cos(lon - center_lon))
        distance = np.arccos(np.clip(cosd, -1.0, 1.0))
        return np.exp(-(distance * distance) / (2.0 * sigma * sigma))

    return cp.asarray(5e-5 * gaussian(lat0, lon0)
                      - 5e-5 * gaussian(-lat0, lon0 + np.pi))


def make_rotation_runs(out: pathlib.Path, *, day_hours: float,
                       duration_days: float, dt_snapshots: float) -> None:
    out = out.resolve()
    run_config = dict(ROTATION_CONFIG)
    run_config.update({
        "day_hours": day_hours,
        "duration_days": duration_days,
        "dt_snapshots": dt_snapshots,
    })
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_resolution=run_config["resolution"],
        l_max=run_config["lmax"],
        product_quadrature=run_config["product_quadrature"],
        grid_type=run_config["grid"],
    )
    for label, (lat, lon) in ROTATIONS.items():
        scenario = f"two-vortices-{label}"
        config = dict(run_config)
        config.update({
            "scenario": scenario,
            "out": str(out),
            "experiment": None,
            "overwrite": False,
            "custom_initial_condition": {
                "kind": "antipodal_gaussian_vortex_pair",
                "positive_center_lat_deg": lat,
                "positive_center_lon_deg": lon,
                "negative_center": "antipode",
                "amplitude_s-1": 5e-5,
                "sigma_deg": 10.0,
            },
        })
        capsule = create_run_dir(out, config)
        config["run_id"] = capsule.run_id
        (capsule.path / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")
        write_run_manifest(
            capsule.path, config, run_id=capsule.run_id,
            numerics=planet.so.backend.describe(config["product_quadrature"]),
        )
        capsule.update_latest_pointer()
        zeta0_lm = planet.sh.transform(_antipodal_vortices(planet, lat, lon))
        run_bve(
            planet=planet,
            zeta0_lm=zeta0_lm,
            dt_snapshots=config["dt_snapshots"],
            t_end_days=config["duration_days"],
            out_dir=capsule.path,
            viscosity=config["viscosity"],
            scenario=scenario,
            figure_metadata=capsule.figure_metadata(),
        )
        print(f"rotation run: {capsule.path}")


def _manifest(run: pathlib.Path) -> dict:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def _geometry(config: dict):
    if config["grid"] == "geodesic":
        return GeodesicGridGeometry(config["resolution"], radius=1.0)
    return GaussLatLonGridGeometry(config["nlat"], config["nlon"], radius=1.0)


def _map_to_view(run: pathlib.Path, field: np.ndarray,
                 lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    config = _manifest(run)["run_config"]
    geometry = _geometry(config)
    lon = np.mod(cp.asnumpy(cp.asarray(geometry.point_longitudes)), 2.0 * np.pi)
    lat = cp.asnumpy(cp.asarray(geometry.point_latitudes))
    values = np.asarray(field).reshape(-1)
    # Periodic copies prevent a seam at 0/360 degrees.
    points = np.column_stack([
        np.concatenate([lon - 2.0 * np.pi, lon, lon + 2.0 * np.pi]),
        np.tile(lat, 3),
    ])
    mapped = griddata(points, np.tile(values, 3),
                      (lon_grid, lat_grid), method="linear")
    if np.isnan(mapped).any():
        nearest = griddata(points, np.tile(values, 3),
                           (lon_grid, lat_grid), method="nearest")
        mapped = np.where(np.isnan(mapped), nearest, mapped)
    return mapped


def _view_mesh() -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(0.0, 2.0 * np.pi, 361)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)
    return np.meshgrid(lon, lat)


def _drifts(run: pathlib.Path) -> tuple[float, float]:
    with (run / "diagnostics" / "timeseries.csv").open(
            newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    e0, e1 = float(rows[0]["energy"]), float(rows[-1]["energy"])
    z0, z1 = float(rows[0]["enstrophy_abs"]), float(rows[-1]["enstrophy_abs"])
    return (e1 / e0 - 1.0), (z1 / z0 - 1.0)


def _plot_backend_comparison(geodesic: pathlib.Path, latlon: pathlib.Path,
                             target: pathlib.Path) -> None:
    lon, lat = _view_mesh()
    geo = np.load(geodesic / "vorticity_grid.npy")
    ll = np.load(latlon / "vorticity_grid.npy")
    fields = {
        "Geodesic, initial": _map_to_view(geodesic, geo[0], lon, lat),
        "Geodesic, 1 day": _map_to_view(geodesic, geo[-1], lon, lat),
        "Gauss lat–lon, initial": _map_to_view(latlon, ll[0], lon, lat),
        "Gauss lat–lon, 1 day": _map_to_view(latlon, ll[-1], lon, lat),
    }
    vmax = max(np.max(np.abs(v)) for v in fields.values())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True,
                             sharex=True, sharey=True)
    for row, prefix in enumerate(("Geodesic", "Gauss lat–lon")):
        a = fields[f"{prefix}, initial"]
        b = fields[f"{prefix}, 1 day"]
        run = geodesic if row == 0 else latlon
        e, z = _drifts(run)
        for col, (title, data) in enumerate((("initial", a), ("after 1 day", b))):
            im = axes[row, col].pcolormesh(
                np.rad2deg(lon), np.rad2deg(lat), data,
                shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].set_title(f"{prefix}: {title}")
        axes[row, 1].text(
            0.02, 0.04, f"ΔE/E={e:+.2e}   ΔZabs/Zabs={z:+.2e}",
            transform=axes[row, 1].transAxes, fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    for ax in axes.flat:
        ax.set(xlabel="longitude (deg)", ylabel="latitude (deg)",
               xlim=(0, 360), ylim=(-90, 90))
    fig.colorbar(im, ax=axes, label="relative vorticity (s⁻¹)")
    fig.suptitle("RH4 backend comparison — identical physics, backend-native fixed timesteps")
    fig.savefig(target, dpi=180, metadata={"Software": "planetary-sandbox"})
    plt.close(fig)


def _plot_vortex_evolution(run: pathlib.Path, target: pathlib.Path) -> None:
    """Show actual evolution, rather than an invariant-dominated summary."""
    lon, lat = _view_mesh()
    snapshots = np.load(run / "vorticity_grid.npy")
    # The tracked hero run has daily snapshots over ten days.  Express the
    # selection as fractions so the renderer remains useful for similar runs.
    indices = [0, round(0.2 * (len(snapshots) - 1)),
               round(0.5 * (len(snapshots) - 1)), len(snapshots) - 1]
    config = _manifest(run)["run_config"]
    duration_days = float(config["duration_days"])
    mapped = [_map_to_view(run, snapshots[index], lon, lat) for index in indices]
    vmax = max(float(np.max(np.abs(field))) for field in mapped)
    e, z = _drifts(run)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, index, field in zip(axes.flat, indices, mapped):
        day = duration_days * index / (len(snapshots) - 1)
        im = ax.pcolormesh(np.rad2deg(lon), np.rad2deg(lat), field,
                           shading="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
        ax.set_title(f"day {day:g}")
        ax.set(xlabel="longitude (deg)", ylabel="latitude (deg)",
               xlim=(0, 360), ylim=(-90, 90))
    axes[-1, -1].text(
        0.02, 0.04,
        f"{duration_days:g}-day ΔE/E={e:+.2%}   ΔZabs/Zabs={z:+.2e}",
        transform=axes[-1, -1].transAxes, fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.colorbar(im, ax=axes, label="relative vorticity (s⁻¹)")
    fig.suptitle("Two vortices evolving on a rotating sphere")
    fig.savefig(target, dpi=180, metadata={"Software": "planetary-sandbox"})
    plt.close(fig)


def _rotation_run_map(root: pathlib.Path) -> dict[str, pathlib.Path]:
    result = {}
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ic = manifest["run_config"].get("custom_initial_condition", {})
        lon = ic.get("positive_center_lon_deg")
        lat = ic.get("positive_center_lat_deg")
        if lat is None:
            continue
        if abs(lat) > 89.0:
            result["north pole"] = manifest_path.parent
        elif abs(lon) < 1e-9:
            result["0°N, 0°E"] = manifest_path.parent
        elif abs(lon - 90.0) < 1e-9:
            result["0°N, 90°E"] = manifest_path.parent
    return result


def _plot_rotations(runs: dict[str, pathlib.Path], target: pathlib.Path) -> None:
    order = ["north pole", "0°N, 0°E", "0°N, 90°E"]
    lon, lat = _view_mesh()
    snapshots = {name: np.load(runs[name] / "vorticity_grid.npy") for name in order}
    mapped = {(name, i): _map_to_view(runs[name], snapshots[name][i], lon, lat)
              for name in order for i in (0, -1)}
    vmax = max(np.max(np.abs(v)) for v in mapped.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True,
                             sharex=True, sharey=True)
    for col, name in enumerate(order):
        e, z = _drifts(runs[name])
        for row, (index, time_label) in enumerate(((0, "initial"), (-1, "after 1 day"))):
            im = axes[row, col].pcolormesh(
                np.rad2deg(lon), np.rad2deg(lat), mapped[(name, index)],
                shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].set_title(
                f"+ vortex at {name}\n{time_label}")
            if row:
                axes[row, col].text(
                    0.02, 0.04, f"ΔE/E={e:+.1e}   ΔZabs/Zabs={z:+.1e}",
                    transform=axes[row, col].transAxes, fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8,
                          "edgecolor": "none"})
            axes[row, col].set(xlabel="longitude (deg)", ylabel="latitude (deg)",
                               xlim=(0, 360), ylim=(-90, 90))
    fig.colorbar(im, ax=axes, label="relative vorticity (s⁻¹)")
    fig.suptitle("Rotation-equivalent antipodal vortex pairs on the geodesic backend")
    fig.savefig(target, dpi=180, metadata={"Software": "planetary-sandbox"})
    plt.close(fig)


def _plot_rotating_streamlines(runs: dict[str, pathlib.Path],
                               target: pathlib.Path) -> None:
    """Compare diagnosed wind as planetary rotation breaks full SO(3) symmetry."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    order = ["north pole", "0°N, 0°E", "0°N, 90°E"]
    first_config = _manifest(runs[order[0]])["run_config"]
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=float(first_config["day_hours"])),
        grid_resolution=int(first_config["resolution"]),
        l_max=int(first_config["lmax"]),
        product_quadrature=first_config["product_quadrature"],
        grid_type=first_config["grid"],
    )

    lon_axis = np.linspace(0.0, 2.0 * np.pi, 289)
    lat_axis = np.deg2rad(np.linspace(-87.0, 87.0, 145))
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    winds: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_speed = 0.0
    for name in order:
        coeffs = np.load(runs[name] / "vorticity_coeffs.npy")
        for index in (0, -1):
            zeta_lm = cp.asarray(coeffs[index])
            psi_lm = planet.so.inv_laplacian(zeta_lm)
            u, v = planet.so.velocity_from_streamfunction(psi_lm)
            u_view = _map_to_view(runs[name], cp.asnumpy(u), lon, lat)
            v_view = _map_to_view(runs[name], cp.asnumpy(v), lon, lat)
            speed = np.sqrt(u_view**2 + v_view**2)
            winds[(name, index)] = (u_view, v_view, speed)
            max_speed = max(max_speed, float(np.nanmax(speed)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True,
                             sharex=True, sharey=True)
    radius = float(planet.params.radius)
    coslat = np.maximum(np.cos(lat), 0.02)
    norm = Normalize(vmin=0.0, vmax=max_speed)
    duration = float(first_config["duration_days"])
    for col, name in enumerate(order):
        e, z = _drifts(runs[name])
        for row, (index, label) in enumerate(((0, "initial"),
                                               (-1, f"after {duration:g} days"))):
            ax = axes[row, col]
            u, v, speed = winds[(name, index)]
            # Convert physical east/north components to angular map rates.
            u_map = u / (radius * coslat)
            v_map = v / radius
            linewidth = 0.45 + 1.5 * speed / max_speed
            ax.streamplot(
                np.rad2deg(lon_axis), np.rad2deg(lat_axis), u_map, v_map,
                color=speed, cmap="viridis", norm=norm, density=1.25,
                linewidth=linewidth, arrowsize=0.8, integration_direction="both")
            ax.set_title(f"+ vortex at {name}\n{label}")
            ax.set(xlabel="longitude (deg)", ylabel="latitude (deg)",
                   xlim=(0, 360), ylim=(-87, 87))
            if row:
                ax.text(
                    0.02, 0.04, f"ΔE/E={e:+.2%}   ΔZabs/Zabs={z:+.1e}",
                    transform=ax.transAxes, fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8,
                          "edgecolor": "none"})
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes,
                 label="wind speed (m s⁻¹)")
    fig.suptitle(
        "Wind evolution on a 24-hour rotating sphere — the rotation axis breaks orientation symmetry")
    fig.savefig(target, dpi=180, metadata={"Software": "planetary-sandbox"})
    plt.close(fig)


def _provenance_entry(run: pathlib.Path) -> dict:
    manifest = _manifest(run)
    return _json_safe({
        "run_id": manifest["run_id"],
        "commit": manifest["git"]["commit"],
        "dirty": manifest["git"]["dirty"],
        "config": manifest["run_config"],
        "numerics": manifest.get("numerics"),
    })


def _json_safe(value):
    """Return strict-JSON data (Python's manifests may contain +/-Infinity)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def render(args) -> None:
    assets = args.assets.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    rotation_runs = _rotation_run_map(args.rotation_root.resolve())
    if set(rotation_runs) != {"north pole", "0°N, 0°E", "0°N, 90°E"}:
        raise RuntimeError("rotation run set is incomplete; run the rotation-runs command")
    rotating_runs = _rotation_run_map(args.rotating_rotation_root.resolve())
    if set(rotating_runs) != {"north pole", "0°N, 0°E", "0°N, 90°E"}:
        raise RuntimeError(
            "rotating rotation run set is incomplete; run rotation-runs with --day-hours 24")

    rh4_target = assets / "rh4_simulation_summary.png"
    shutil.copy2(args.rh4_run.resolve() / "bve_summary.png", rh4_target)
    _plot_vortex_evolution(
        args.dynamic_run.resolve(), assets / "two_vortices_evolution.png")
    _plot_backend_comparison(
        args.geodesic_run.resolve(), args.latlon_run.resolve(),
        assets / "rh4_geodesic_vs_latlon.png")
    _plot_rotations(rotation_runs, assets / "two_vortices_rotation_comparison.png")
    _plot_rotating_streamlines(
        rotating_runs, assets / "two_vortices_rotating_streamlines.png")

    provenance = {
        "two_vortices_evolution.png": [
            _provenance_entry(args.dynamic_run.resolve())],
        "rh4_simulation_summary.png": [_provenance_entry(args.rh4_run.resolve())],
        "rh4_geodesic_vs_latlon.png": [
            _provenance_entry(args.geodesic_run.resolve()),
            _provenance_entry(args.latlon_run.resolve()),
        ],
        "two_vortices_rotation_comparison.png": [
            _provenance_entry(rotation_runs[name])
            for name in ("north pole", "0°N, 0°E", "0°N, 90°E")
        ],
        "two_vortices_rotating_streamlines.png": [
            _provenance_entry(rotating_runs[name])
            for name in ("north pole", "0°N, 0°E", "0°N, 90°E")
        ],
    }
    (assets / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote README assets and provenance to {assets}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rotations = commands.add_parser("rotation-runs")
    rotations.add_argument("--out", type=pathlib.Path,
                           default=ROOT / "runs" / "readme-rotations")
    rotations.add_argument("--day-hours", type=float, default=math.inf)
    rotations.add_argument("--duration-days", type=float, default=1.0)
    rotations.add_argument("--dt-snapshots", type=float, default=43200.0)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--dynamic-run", type=pathlib.Path, required=True)
    render_parser.add_argument("--rh4-run", type=pathlib.Path, required=True)
    render_parser.add_argument("--geodesic-run", type=pathlib.Path, required=True)
    render_parser.add_argument("--latlon-run", type=pathlib.Path, required=True)
    render_parser.add_argument("--rotation-root", type=pathlib.Path,
                               default=ROOT / "runs" / "readme-rotations")
    render_parser.add_argument("--rotating-rotation-root", type=pathlib.Path,
                               required=True)
    render_parser.add_argument("--assets", type=pathlib.Path,
                               default=ROOT / "docs" / "assets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "rotation-runs":
        make_rotation_runs(
            args.out, day_hours=args.day_hours,
            duration_days=args.duration_days,
            dt_snapshots=args.dt_snapshots)
    else:
        render(args)


if __name__ == "__main__":
    main()
