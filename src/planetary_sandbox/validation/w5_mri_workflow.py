"""Production workflow for W5 agreement with frozen MRI projections.

The only reference-side operations in this module are hash verification,
strict NPZ loading, exact index permutation, comparison, and plotting.  All
Aeolus fields come from persisted canonical run capsules and existing model
APIs.  No reference-side transform, interpolation, or reconstruction exists
here.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .w5_mri import (
    AEOLUS_HEIGHT_AUDIT,
    ARTIFACT_SPECS,
    NORM_DEFINITIONS,
    QUADRATURE_DEFINITION,
    ArtifactSpec,
    FrozenReferencePackage,
    GridPermutation,
    ReferenceContractError,
    atomic_write_json,
    atomic_write_text,
    canonical_signature,
    construct_aeolus_heights,
    derive_day0_tolerances,
    determine_grid_permutation,
    evaluate_day0_contract,
    hash_file_inventory,
    reference_identity,
    require_day0_contract,
    scalar_error_metrics,
    sha256_file,
    validate_stage_signature,
    vector_error_metrics,
    verify_reference_package,
)


DEFAULT_REPOSITORY_URL = "https://github.com/AlexandreEros/Aeolus.git"
WORKFLOW_SCHEMA_VERSION = "aeolus-w5-mri-validation-v1"
REQUIRED_SNAPSHOT_SECONDS = np.asarray(
    [0.0, 5.0 * 86400.0, 10.0 * 86400.0, 15.0 * 86400.0],
    dtype=np.float64,
)
FIELD_NAMES = ("zeta", "delta", "phi")


@dataclasses.dataclass(frozen=True)
class WorkflowConfig:
    reference_dir: pathlib.Path
    output_root: pathlib.Path
    repository_root: pathlib.Path
    repository_url: str = DEFAULT_REPOSITORY_URL
    git_ref: str = "feat/w5-mri-validation"
    run_t42: bool = True
    run_t63: bool = True
    force_rerun: bool = False


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git(repository: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repository, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def _source_tree_fingerprint(repository: pathlib.Path) -> dict[str, Any]:
    """Fingerprint executable project sources, including untracked new files."""
    candidates = sorted(
        path for path in (repository / "src").rglob("*")
        if path.is_file() and path.suffix in {".py", ".cu"})
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        candidate = repository / name
        if candidate.is_file():
            candidates.append(candidate)
    digest = hashlib.sha256()
    for path in sorted(candidates):
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(candidates),
        "scope": "src/**/*.{py,cu}, pyproject.toml, requirements*.txt",
    }


def repository_identity(repository: pathlib.Path,
                        *, configured_url: str,
                        git_ref: str) -> dict[str, Any]:
    commit = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    porcelain = _git(repository, "status", "--porcelain")
    try:
        origin = _git(repository, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        origin = None
    return {
        "configured_repository_url": configured_url,
        "origin_url": origin,
        "configured_git_ref": git_ref,
        "active_branch": branch or None,
        "commit_sha": commit,
        "dirty": bool(porcelain),
        "dirty_tree_status": porcelain.splitlines(),
        "executable_source_tree": _source_tree_fingerprint(repository),
    }


def _resolved_config_dict(spec: ArtifactSpec, *,
                          output_placeholder: str = "<stage>/aeolus_capsule"
                          ) -> dict[str, Any]:
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({
        "scenario": "williamson5",
        "grid": "latlon",
        "nlat": spec.nlat,
        "nlon": spec.nlon,
        "lmax": spec.truncation,
        "duration_days": 15.0,
        "n_snapshots": 4,
        "no_plots": True,
        "out": output_placeholder,
    })
    result = cfg.to_run_config_dict()
    if result["snapshot_times"] != REQUIRED_SNAPSHOT_SECONDS.tolist():
        raise RuntimeError("canonical W5 config did not resolve exact target times")
    return result


def _stage_signature_payload(
        spec: ArtifactSpec, package: FrozenReferencePackage,
        repository: Mapping[str, Any],
        tolerances: Mapping[str, Any],
        ) -> dict[str, Any]:
    return {
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "reference_hashes": dict(package.hashes),
        "reference_schema_version": package.manifest["schema_version"],
        "aeolus_commit_sha": repository["commit_sha"],
        "aeolus_dirty": repository["dirty"],
        "aeolus_executable_source_tree": repository["executable_source_tree"],
        "resolved_run_config": _resolved_config_dict(spec),
        "artifact": dataclasses.asdict(spec),
        "required_snapshot_seconds": REQUIRED_SNAPSHOT_SECONDS.tolist(),
        "canonical_numerical_policy": {
            "integrator": "existing Aeolus RK4 timestepper",
            "adaptive_cfl": "existing state-adaptive Aeolus CFL machinery",
            "horizontal_diffusion": "none",
            "hyperdiffusion_nu4_m4_s-1": 0.0,
            "product_quadrature": "fine",
            "topography_projection":
                "state-grid-analysis-full-truncation",
        },
        "height_audit": AEOLUS_HEIGHT_AUDIT,
        "day0_tolerances": tolerances,
        "workflow_firewall": {
            "reference_spectral_analysis": False,
            "reference_projection": False,
            "reference_velocity_reconstruction": False,
            "interpolation_or_regridding": False,
        },
    }


def _atomic_copy_verified(source: pathlib.Path, destination: pathlib.Path,
                          expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_hash:
            raise ReferenceContractError(
                f"archived reference file has unexpected hash and will not "
                f"be overwritten: {destination}")
        return
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        actual = sha256_file(temporary)
        if actual != expected_hash:
            raise ReferenceContractError(
                f"reference copy verification failed for {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def archive_reference_package(package: FrozenReferencePackage,
                              output_root: pathlib.Path
                              ) -> FrozenReferencePackage:
    """Copy the small immutable package once, then consume the verified copy."""
    destination = output_root / "reference"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, digest in package.hashes.items():
        _atomic_copy_verified(
            package.directory / filename, destination / filename, digest)
    archived = verify_reference_package(destination)
    atomic_write_json(
        destination / "reference_identity.json", reference_identity(archived))
    return archived


def _archive_existing_stage(stage: pathlib.Path) -> pathlib.Path:
    """Preserve a force-rerun predecessor instead of deleting it."""
    root = stage.parent.resolve()
    resolved = stage.resolve()
    if resolved.parent != root:
        raise RuntimeError(f"refusing to move stage outside output root: {stage}")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = stage.with_name(
        f"{stage.name}.superseded-{timestamp}-{uuid.uuid4().hex[:8]}")
    os.replace(stage, backup)
    return backup


def _completed_capsule(candidate: pathlib.Path,
                       base: pathlib.Path) -> pathlib.Path | None:
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as err:
        raise RuntimeError(
            f"capsule candidate escapes its base directory: {candidate}") from err
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return candidate if manifest.get("status") == "completed" else None


def _safe_capsule_from_pointer(base: pathlib.Path) -> pathlib.Path | None:
    pointer_path = base / "latest_run.txt"
    if pointer_path.is_file():
        relative = pointer_path.read_text(encoding="utf-8").strip()
        if relative:
            completed = _completed_capsule(base / relative, base)
            if completed is not None:
                return completed
    # A disconnect can occur after the lifecycle atomically marks a capsule
    # completed but before latest_run.txt is published.  Reuse that capsule
    # instead of repeating an expensive 15-day integration.
    candidates = sorted(
        (path.parent for path in base.glob("*/manifest.json")),
        key=lambda path: path.name, reverse=True)
    for candidate in candidates:
        completed = _completed_capsule(candidate, base)
        if completed is not None:
            return completed
    return None


def _run_canonical_capsule(spec: ArtifactSpec, stage: pathlib.Path,
                           *, resume: bool) -> pathlib.Path:
    """Run through the real CLI; resume a completed capsule when available."""
    from planetary_sandbox.cli.main import main

    base = stage / "aeolus_capsule"
    base.mkdir(parents=True, exist_ok=True)
    completed = _safe_capsule_from_pointer(base)
    if completed is not None:
        print(f"Reusing completed canonical Aeolus capsule: {completed}",
              flush=True)
        return completed

    args = [
        "run", "swe",
        "--scenario", "williamson5",
        "--backend", "gauss-latlon",
        "--nlat", str(spec.nlat),
        "--nlon", str(spec.nlon),
        "--l-max", str(spec.truncation),
        "--days", "15",
        "--n-snapshots", "4",
        "--no-plots",
        "--out", str(base),
    ]
    # An interrupted same-signature stage may contain a lifecycle-managed
    # failed/running capsule.  The CLI's scoped overwrite path replaces only
    # its known generated outputs and preserves capsule provenance.
    if resume and any(base.iterdir()):
        args.append("--overwrite")
    print(
        f"Running canonical W5 T{spec.truncation} on "
        f"{spec.nlat}x{spec.nlon} (separate resumable stage)", flush=True)
    rc = main(args)
    if rc != 0:
        raise RuntimeError(
            f"canonical Aeolus W5 stage failed for T{spec.truncation} "
            f"(CLI return code {rc})")
    completed = _safe_capsule_from_pointer(base)
    if completed is None:
        raise RuntimeError(
            f"CLI returned success without a completed capsule under {base}")
    return completed


def _canonical_model(spec: ArtifactSpec):
    """Reconstruct the persisted run's model through existing Aeolus APIs."""
    from planetary_sandbox.cli.swe import _w5_planet_params
    from planetary_sandbox.planet import Planet
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.physics.topography import Topography
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({
        "scenario": "williamson5",
        "grid": "latlon",
        "nlat": spec.nlat,
        "nlon": spec.nlon,
        "lmax": spec.truncation,
        "duration_days": 15.0,
        "n_snapshots": 4,
        "no_plots": True,
    })
    planet = Planet.generate(
        params=_w5_planet_params(cfg),
        grid_resolution=cfg.resolution,
        l_max=cfg.lmax,
        product_quadrature="fine",
        grid_type=cfg.grid,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
    )
    model = ShallowWaterModel(
        planet, gravity=cfg.gravity, mean_depth=cfg.mean_depth_m,
        topography=Topography.williamson5_cone(planet))
    return cfg, model


def _load_capsule_state(capsule: pathlib.Path,
                        spec: ArtifactSpec) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(capsule / "swe_snapshot_times.npy", allow_pickle=False)
    coefficients = np.load(capsule / "swe_coeffs.npy", allow_pickle=False)
    expected_coeff_shape = (
        4, 3, spec.truncation + 1, spec.truncation + 1)
    if times.dtype != np.float64 or not np.array_equal(
            times, REQUIRED_SNAPSHOT_SECONDS):
        raise RuntimeError(
            f"Aeolus capsule snapshot times are not exactly "
            f"{REQUIRED_SNAPSHOT_SECONDS.tolist()}")
    if coefficients.shape != expected_coeff_shape:
        raise RuntimeError(
            f"Aeolus coefficient shape mismatch: expected "
            f"{expected_coeff_shape}, got {coefficients.shape}")
    if coefficients.dtype != np.complex128:
        raise RuntimeError(
            f"Aeolus coefficients must be complex128, got {coefficients.dtype}")
    if not np.isfinite(coefficients.real).all() or not np.isfinite(
            coefficients.imag).all():
        raise RuntimeError("Aeolus snapshot coefficients contain non-finite data")
    return times, coefficients


def _extract_native_fields(model, coefficients: np.ndarray,
                           spec: ArtifactSpec
                           ) -> tuple[dict[str, np.ndarray], np.ndarray,
                                      np.ndarray]:
    """Synthesize Aeolus fields only through canonical model/backend APIs."""
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState

    native_latitude = cp.asnumpy(model.grid.latitudes).astype(
        np.float64, copy=False)
    native_longitude = cp.asnumpy(model.grid.longitudes).astype(
        np.float64, copy=False)
    phi_s_device = model.surface_geopotential_on_state_grid()
    if phi_s_device is None:
        phi_s = np.zeros((spec.nlat, spec.nlon), dtype=np.float64)
    else:
        phi_s = cp.asnumpy(phi_s_device).reshape(
            spec.nlat, spec.nlon).astype(np.float64, copy=False)

    height: list[np.ndarray] = []
    thickness: list[np.ndarray] = []
    u_values: list[np.ndarray] = []
    v_values: list[np.ndarray] = []
    for coefficient_block in coefficients:
        state = ShallowWaterState(cp.asarray(coefficient_block))
        phi = cp.asnumpy(
            model.sh.inv_transform(state.phi).real).reshape(
                spec.nlat, spec.nlon)
        heights = construct_aeolus_heights(
            phi, base_geopotential=model.phi0,
            surface_geopotential=phi_s, gravity=model.gravity)
        u_device, v_device = model.wind_on_state_grid(state)
        height.append(heights["free_surface_height_m"])
        thickness.append(heights["fluid_layer_thickness_m"])
        u_values.append(cp.asnumpy(u_device).reshape(spec.nlat, spec.nlon))
        v_values.append(cp.asnumpy(v_device).reshape(spec.nlat, spec.nlon))
    fields = {
        "height": np.stack(height).astype(np.float64, copy=False),
        "fluid_layer_thickness": np.stack(thickness).astype(
            np.float64, copy=False),
        "u": np.stack(u_values).astype(np.float64, copy=False),
        "v": np.stack(v_values).astype(np.float64, copy=False),
    }
    for name, values in fields.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Aeolus extracted {name} contains non-finite data")
    return fields, native_latitude, native_longitude


def _align_fields(fields: Mapping[str, np.ndarray],
                  permutation: GridPermutation) -> dict[str, np.ndarray]:
    return {name: permutation.apply(values)
            for name, values in fields.items()}


def _reference_convention_checks(
        package: FrozenReferencePackage, model,
        permutation: GridPermutation) -> dict[str, bool]:
    manifest = package.manifest
    constants = manifest.get("analytic_day0_constants", {})
    mapping = manifest.get("source_variable_mapping", {})
    topography = model.topography.parameters

    def exact(name: str, value: float) -> bool:
        try:
            return math.isclose(
                float(constants[name]), float(value),
                rel_tol=0.0, abs_tol=5.0e-13 * max(1.0, abs(float(value))))
        except (KeyError, TypeError, ValueError):
            return False

    return {
        "reference_height_is_free_surface_metres": (
            manifest.get("height_interpretation", {}).get("resolved_as")
            == "free_surface_height"
            and mapping.get("height", {}).get("units") == "m"),
        "u_is_eastward_m_s-1": (
            mapping.get("zonal_velocity", {}).get("positive_direction")
            == "eastward"
            and mapping.get("zonal_velocity", {}).get("units") == "m s-1"),
        "v_is_northward_m_s-1": (
            mapping.get("meridional_velocity", {}).get("positive_direction")
            == "northward"
            and mapping.get("meridional_velocity", {}).get("units") == "m s-1"),
        "gravity_matches": exact("gravity_m_s-2", model.gravity),
        "sphere_radius_matches": exact("sphere_radius_m", model.R),
        "rotation_rate_matches": exact("rotation_rate_s-1", model.Omega),
        "mountain_height_matches": exact("mountain_height_m", 2000.0),
        "mountain_latitude_matches": exact(
            "mountain_lat_rad", np.pi / 6.0),
        "mountain_longitude_matches": exact(
            "mountain_lon_rad", 3.0 * np.pi / 2.0),
        "mountain_radius_matches": exact(
            "mountain_radius_rad", np.pi / 9.0),
        "aeolus_topography_preset_is_williamson5_cone": (
            model.topography.preset == "williamson5_cone"),
        "aeolus_mountain_height_matches": math.isclose(
            float(topography.get("height_m", math.nan)), 2000.0,
            rel_tol=0.0, abs_tol=1.0e-12),
        "aeolus_mountain_latitude_matches": math.isclose(
            float(topography.get("lat_center_deg", math.nan)), 30.0,
            rel_tol=0.0, abs_tol=1.0e-12),
        "aeolus_mountain_longitude_matches": math.isclose(
            float(topography.get("lon_center_deg", math.nan)), -90.0,
            rel_tol=0.0, abs_tol=1.0e-12),
        "aeolus_mountain_radius_matches": math.isclose(
            float(topography.get("radius_rad", math.nan)), np.pi / 9.0,
            rel_tol=0.0, abs_tol=1.0e-15),
        "initial_wind_matches": exact("u0_m_s-1", 20.0),
        "latitude_orientation_resolved_by_exact_indexing": isinstance(
            permutation.latitude_reversed, bool),
        "longitude_alignment_resolved_by_exact_indexing": isinstance(
            permutation.longitude_roll, int),
        "no_interpolation_or_regridding": True,
    }


def _comparison_record(time_days: float, fields: Mapping[str, np.ndarray],
                       reference: Mapping[str, np.ndarray], index: int,
                       weights: np.ndarray, dlon: float) -> dict[str, Any]:
    role = ("day-zero convention verification" if index == 0
            else "MRI projected-reference agreement")
    return {
        "time_days": time_days,
        "comparison_role": role,
        "reference_label": None,  # filled with the truncation-specific label
        "height": scalar_error_metrics(
            fields["height"][index], reference["height"][index],
            weights, dlon),
        "velocity": vector_error_metrics(
            fields["u"][index], fields["v"][index],
            reference["u"][index], reference["v"][index],
            weights, dlon),
    }


def _metric_value(metric: Mapping[str, Any]) -> float | None:
    return metric.get("value") if metric.get("status") == "defined" else None


def _flatten_comparison_record(record: Mapping[str, Any]) -> dict[str, Any]:
    height = record["height"]
    velocity = record["velocity"]
    row: dict[str, Any] = {
        "time_days": record["time_days"],
        "comparison_role": record["comparison_role"],
        "reference_label": record["reference_label"],
    }
    for norm in ("L1", "L2", "Linf"):
        item = height["normalized"][norm]
        row[f"height_{norm}"] = _metric_value(item)
        row[f"height_{norm}_status"] = item["status"]
        item = velocity["normalized"][norm]
        row[f"velocity_vector_{norm}"] = _metric_value(item)
        row[f"velocity_vector_{norm}_status"] = item["status"]
    for prefix, values in (
            ("height", height["absolute"]),
            ("velocity_vector", velocity["absolute_vector"]),
            ("u_component", velocity["absolute_u_component"]),
            ("v_component", velocity["absolute_v_component"])):
        for name, value in values.items():
            row[f"{prefix}_{name}"] = value
    return row


def _csv_text(rows: Sequence[Mapping[str, Any]],
              fieldnames: Sequence[str] | None = None) -> str:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    fields = list(fieldnames or rows[0].keys())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: "" if value is None else value
            for key, value in row.items()
        })
    return output.getvalue()


def _read_diagnostic_rows(capsule: pathlib.Path) -> list[dict[str, str]]:
    with (capsule / "diagnostics" / "timeseries.csv").open(
            newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise RuntimeError("Aeolus diagnostics contain no accepted steps")
    return rows


def _conservation_products(
        capsule: pathlib.Path, model, coefficients: np.ndarray,
        stage: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp
    from planetary_sandbox.physics.shallow_water import ShallowWaterState
    from planetary_sandbox.run.swe.diagnostics import potential_enstrophy

    raw_rows = _read_diagnostic_rows(capsule)
    required = {
        "time_s", "dt_s", "step", "max_wind_ms", "total_energy", "h_min_m"}
    if not required.issubset(raw_rows[0]):
        raise RuntimeError("Aeolus diagnostics are missing required columns")

    numeric = []
    for raw in raw_rows:
        row = {key: float(raw[key]) for key in required - {"step"}}
        row["step"] = int(raw["step"])
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("Aeolus diagnostics contain non-finite values")
        numeric.append(row)
    energy0 = numeric[0]["total_energy"]
    if energy0 == 0.0:
        raise RuntimeError("initial total energy is zero")

    enstrophy = [
        potential_enstrophy(
            model, ShallowWaterState(cp.asarray(coefficient_block)))
        for coefficient_block in coefficients
    ]
    if not np.isfinite(enstrophy).all() or enstrophy[0] == 0.0:
        raise RuntimeError("potential enstrophy diagnostics are invalid")
    enstrophy_by_time = {
        float(time): float(value)
        for time, value in zip(REQUIRED_SNAPSHOT_SECONDS, enstrophy)
    }
    timeseries = []
    for row in numeric:
        time_s = row["time_s"]
        z = enstrophy_by_time.get(time_s)
        timeseries.append({
            "time_s": time_s,
            "time_days": time_s / 86400.0,
            "step": row["step"],
            "dt_s": row["dt_s"],
            "total_energy": row["total_energy"],
            "relative_total_energy_drift": (
                (row["total_energy"] - energy0) / abs(energy0)),
            "potential_enstrophy": z,
            "relative_potential_enstrophy_drift": (
                None if z is None else (z - enstrophy[0]) / abs(enstrophy[0])),
            "minimum_fluid_layer_thickness_m": row["h_min_m"],
            "maximum_wind_speed_m_s-1": row["max_wind_ms"],
        })
    step_dt = np.asarray(
        [row["dt_s"] for row in numeric if row["step"] > 0],
        dtype=np.float64)
    energy_drift = np.asarray(
        [row["relative_total_energy_drift"] for row in timeseries])
    enstrophy_drift = (
        (np.asarray(enstrophy) - enstrophy[0]) / abs(enstrophy[0]))
    summary = {
        "relative_total_energy_drift_final": float(energy_drift[-1]),
        "maximum_absolute_relative_total_energy_drift": float(
            np.max(np.abs(energy_drift))),
        "relative_potential_enstrophy_drift_final": float(
            enstrophy_drift[-1]),
        "maximum_absolute_relative_potential_enstrophy_drift_at_snapshots":
            float(np.max(np.abs(enstrophy_drift))),
        "minimum_fluid_layer_thickness_m": float(
            min(row["h_min_m"] for row in numeric)),
        "maximum_wind_speed_m_s-1": float(
            max(row["max_wind_ms"] for row in numeric)),
        "non_finite_state_detected": False,
        "total_step_count": int(numeric[-1]["step"]),
        "minimum_timestep_s": float(step_dt.min()),
        "maximum_timestep_s": float(step_dt.max()),
        "mean_timestep_s": float(step_dt.mean()),
        "median_timestep_s": float(np.median(step_dt)),
        "exact_snapshot_times_s": REQUIRED_SNAPSHOT_SECONDS.tolist(),
        "potential_enstrophy_sampling": (
            "computed with existing Aeolus potential_enstrophy() at the four "
            "persisted exact snapshots; all other diagnostics are per accepted "
            "step from the canonical recorder"),
        "official_williamson_pass_criteria": False,
    }
    atomic_write_text(
        stage / "conservation_metrics.csv", _csv_text(timeseries))
    atomic_write_json(stage / "conservation_summary.json", summary)
    return summary, timeseries


def _finite_limits(values: Sequence[np.ndarray], *,
                   symmetric: bool = False,
                   nonnegative: bool = False) -> tuple[float, float]:
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    if symmetric:
        bound = max(abs(minimum), abs(maximum), np.finfo(float).eps)
        return -bound, bound
    if nonnegative:
        return 0.0, max(maximum, np.finfo(float).eps)
    if minimum == maximum:
        padding = max(abs(minimum), 1.0) * 1.0e-12
        return minimum - padding, maximum + padding
    return minimum, maximum


def _save_figure_atomic(figure, target: pathlib.Path, **savefig_kwargs) -> None:
    """Render completely beside the target, fsync, then publish by replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}")
    try:
        figure.savefig(temporary, **savefig_kwargs)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _render_comparison_maps(
        stage: pathlib.Path, spec: ArtifactSpec,
        fields: Mapping[str, np.ndarray],
        reference: Mapping[str, np.ndarray],
        latitude: np.ndarray, longitude: np.ndarray,
        *, indices: Sequence[int] = (0, 1, 2, 3),
        ) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    plot_dir = stage / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    lon_deg = np.rad2deg(longitude)
    lat_deg = np.rad2deg(latitude)
    label = f"MRI\u2192T{spec.truncation}"
    limits_manifest: dict[str, Any] = {}

    days = (0, 5, 10, 15)
    for index in indices:
        day = days[index]
        ref_height = reference["height"][index]
        aeolus_height = fields["height"][index]
        height_error = aeolus_height - ref_height
        ref_speed = np.hypot(reference["u"][index], reference["v"][index])
        aeolus_speed = np.hypot(fields["u"][index], fields["v"][index])
        velocity_error = np.hypot(
            fields["u"][index] - reference["u"][index],
            fields["v"][index] - reference["v"][index])

        height_limits = _finite_limits([ref_height, aeolus_height])
        height_error_limits = _finite_limits([height_error], symmetric=True)
        speed_limits = _finite_limits(
            [ref_speed, aeolus_speed], nonnegative=True)
        velocity_error_limits = _finite_limits(
            [velocity_error], nonnegative=True)
        height_levels = np.linspace(*height_limits, 21)
        height_error_levels = np.linspace(*height_error_limits, 21)
        speed_levels = np.linspace(*speed_limits, 21)
        velocity_error_levels = np.linspace(*velocity_error_limits, 21)
        limits_manifest[f"day_{day}"] = {
            "height_m": {
                "limits": list(height_limits),
                "levels": height_levels.tolist(),
            },
            "signed_height_error_m": {
                "limits": list(height_error_limits),
                "levels": height_error_levels.tolist(),
                "zero_centered_symmetric": True,
            },
            "wind_speed_m_s-1": {
                "limits": list(speed_limits),
                "levels": speed_levels.tolist(),
            },
            "vector_velocity_error_m_s-1": {
                "limits": list(velocity_error_limits),
                "levels": velocity_error_levels.tolist(),
            },
            "display": (
                "native Gaussian cells via pcolormesh(shading='nearest'); "
                "no image interpolation or regridding"),
        }

        panels = (
            (ref_height, f"{label} projected free-surface height",
             "viridis", height_levels, "m"),
            (aeolus_height, "Aeolus free-surface height",
             "viridis", height_levels, "m"),
            (height_error, "Aeolus \u2212 projected-reference height",
             "RdBu_r", height_error_levels, "m"),
            (ref_speed, f"{label} projected wind speed",
             "magma", speed_levels, "m s$^{-1}$"),
            (aeolus_speed, "Aeolus wind speed",
             "magma", speed_levels, "m s$^{-1}$"),
            (velocity_error, "Vector-velocity-error magnitude",
             "inferno", velocity_error_levels, "m s$^{-1}$"),
        )
        fig, axes = plt.subplots(
            2, 3, figsize=(16.5, 8.5), constrained_layout=True,
            sharex=True, sharey=True)
        for axis, (values, title, cmap, levels, unit) in zip(
                axes.flat, panels):
            norm = colors.BoundaryNorm(levels, ncolors=256, clip=True)
            image = axis.pcolormesh(
                lon_deg, lat_deg, values, shading="nearest",
                cmap=cmap, norm=norm, rasterized=True)
            axis.set_title(title)
            axis.set_xlabel("longitude [degrees east]")
            axis.set_ylabel("latitude [degrees north]")
            axis.set_xlim(float(lon_deg[0]), float(lon_deg[-1]))
            axis.set_ylim(float(lat_deg[-1]), float(lat_deg[0]))
            colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
            colorbar.set_label(unit)
        role = ("initial-state contract check" if day == 0
                else "resolved-scale MRI projected-reference agreement")
        fig.suptitle(
            f"Williamson 5 T{spec.truncation} ({spec.nlat}\u00d7{spec.nlon}) "
            f"day {day}: {role}", fontsize=14)
        stem = plot_dir / f"projected_reference_day{day:02d}"
        _save_figure_atomic(fig, stem.with_suffix(".png"), dpi=300)
        _save_figure_atomic(fig, stem.with_suffix(".pdf"))
        plt.close(fig)

    return limits_manifest


def _render_error_timeseries(stage: pathlib.Path,
                             comparison_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = [_flatten_comparison_record(row) for row in comparison_rows]
    days = np.asarray([row["time_days"] for row in flat])
    plot_dir = stage / "plots"

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True)
    for prefix, label, linestyle in (
            ("height", "free-surface height", "-"),
            ("velocity_vector", "vector velocity", "--")):
        for norm in ("L1", "L2", "Linf"):
            values = [
                np.nan if row[f"{prefix}_{norm}"] is None
                else row[f"{prefix}_{norm}"] for row in flat]
            axes[0].semilogy(
                days, values, marker="o", linestyle=linestyle,
                label=f"{label} {norm}")
    axes[0].set_title("Normalized projected-reference errors")
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("normalized error")
    axes[0].set_xticks(days)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for key, label in (
            ("height_weighted_rms_error", "height [m]"),
            ("velocity_vector_weighted_rms_error", "vector velocity [m/s]"),
            ("u_component_weighted_rms_error", "u component [m/s]"),
            ("v_component_weighted_rms_error", "v component [m/s]")):
        axes[1].semilogy(
            days, [row[key] for row in flat], marker="o", label=label)
    axes[1].set_title("Absolute weighted-RMS errors")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel("weighted RMS")
    axes[1].set_xticks(days)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    _save_figure_atomic(fig, plot_dir / "error_timeseries.png", dpi=300)
    _save_figure_atomic(fig, plot_dir / "error_timeseries.pdf")
    plt.close(fig)

    forecast = [row for row in flat if row["time_days"] > 0.0]
    columns = [
        "day", "height L2", "velocity L2", "height WRMS [m]",
        "velocity WRMS [m/s]"]
    cells = [[
        f"{row['time_days']:g}",
        f"{row['height_L2']:.4e}",
        f"{row['velocity_vector_L2']:.4e}",
        f"{row['height_weighted_rms_error']:.4e}",
        f"{row['velocity_vector_weighted_rms_error']:.4e}",
    ] for row in forecast]
    fig, axis = plt.subplots(figsize=(11, 2.2), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    axis.set_title("MRI projected-reference agreement (forecast times)")
    _save_figure_atomic(
        fig, plot_dir / "projected_reference_metric_table.png", dpi=300)
    _save_figure_atomic(
        fig, plot_dir / "projected_reference_metric_table.pdf")
    plt.close(fig)


def _render_conservation_timeseries(
        stage: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.asarray([row["time_days"] for row in rows])
    energy = np.asarray([row["relative_total_energy_drift"] for row in rows])
    z_rows = [row for row in rows
              if row["relative_potential_enstrophy_drift"] is not None]
    z_days = np.asarray([row["time_days"] for row in z_rows])
    z_drift = np.asarray(
        [row["relative_potential_enstrophy_drift"] for row in z_rows])
    thickness = np.asarray(
        [row["minimum_fluid_layer_thickness_m"] for row in rows])
    speed = np.asarray([row["maximum_wind_speed_m_s-1"] for row in rows])

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(days, energy)
    axes[0, 0].set_title("Relative total-energy drift")
    axes[0, 0].set_ylabel("(E \u2212 E\u2080) / |E\u2080|")
    axes[0, 1].plot(z_days, z_drift, marker="o")
    axes[0, 1].set_title("Relative potential-enstrophy drift")
    axes[0, 1].set_ylabel("(Z \u2212 Z\u2080) / |Z\u2080|")
    axes[1, 0].plot(days, thickness)
    axes[1, 0].set_title("Minimum true fluid-layer thickness")
    axes[1, 0].set_ylabel("m")
    axes[1, 1].plot(days, speed)
    axes[1, 1].set_title("Maximum wind speed")
    axes[1, 1].set_ylabel("m s$^{-1}$")
    for axis in axes.flat:
        axis.set_xlabel("time [days]")
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        "Intrinsic Aeolus conservation and stability diagnostics "
        "(not projected-reference errors)")
    plot_dir = stage / "plots"
    _save_figure_atomic(
        fig, plot_dir / "conservation_stability.png", dpi=300)
    _save_figure_atomic(
        fig, plot_dir / "conservation_stability.pdf")
    plt.close(fig)


def _validate_capsule_manifest(capsule: pathlib.Path,
                               spec: ArtifactSpec) -> Mapping[str, Any]:
    manifest = json.loads(
        (capsule / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("Aeolus capsule is not marked completed")
    actual = manifest.get("run_config", {})
    expected = _resolved_config_dict(spec)
    ignored = {"out", "overwrite"}
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if key not in ignored and actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Aeolus capsule does not match the resolved canonical run "
            f"configuration: {mismatches}")
    return manifest


def execute_stage(
        *, spec: ArtifactSpec, package: FrozenReferencePackage,
        output_root: pathlib.Path, repository: Mapping[str, Any],
        tolerances: Mapping[str, Any], force_rerun: bool,
        ) -> dict[str, Any]:
    stage = output_root / spec.stage_name
    signature_payload = _stage_signature_payload(
        spec, package, repository, tolerances)
    signature = canonical_signature(signature_payload)
    if force_rerun and stage.exists():
        backup = _archive_existing_stage(stage)
        print(f"Preserved prior stage at {backup}", flush=True)
    state = validate_stage_signature(stage, signature)
    if state == "complete":
        print(f"Reusing verified completed stage: {stage}", flush=True)
        return json.loads(
            (stage / "COMPLETE.json").read_text(encoding="utf-8"))

    stage.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage / "STAGE_SIGNATURE.json", {
        "signature": signature,
        "payload": signature_payload,
        "created_or_verified_at_utc": _utc_now(),
    })
    capsule = _run_canonical_capsule(
        spec, stage, resume=(state == "incomplete"))
    capsule_manifest = _validate_capsule_manifest(capsule, spec)
    times, coefficients = _load_capsule_state(capsule, spec)
    _, model = _canonical_model(spec)
    native_fields, native_lat, native_lon = _extract_native_fields(
        model, coefficients, spec)
    reference_artifact = package.artifacts[spec.filename]
    reference = reference_artifact.arrays
    permutation = determine_grid_permutation(
        native_lat, native_lon,
        reference["latitude"], reference["longitude"],
        native_field_shape=native_fields["height"].shape[-2:])
    fields = _align_fields(native_fields, permutation)
    atomic_write_json(
        stage / "coordinate_permutation.json", permutation.to_manifest())

    dlon = 2.0 * np.pi / spec.nlon
    convention_checks = _reference_convention_checks(
        package, model, permutation)
    day0 = evaluate_day0_contract(
        height=fields["height"][0],
        u=fields["u"][0],
        v=fields["v"][0],
        height_reference=reference["height"][0],
        u_reference=reference["u"][0],
        v_reference=reference["v"][0],
        latitude_weights=reference["latitude_weights"],
        longitude_spacing=dlon,
        tolerances=tolerances,
        convention_checks=convention_checks,
    )
    day0.update({
        "reference_label": f"MRI\u2192T{spec.truncation} projection",
        "aeolus_height_formula":
            AEOLUS_HEIGHT_AUDIT["free_surface_height_formula"],
        "aeolus_fluid_layer_thickness_formula":
            AEOLUS_HEIGHT_AUDIT["fluid_layer_thickness_formula"],
        "units_and_components": {
            "height": "metres; absolute free-surface height",
            "u": "m/s; eastward-positive",
            "v": "m/s; northward-positive",
            "wind_speed": "m/s; sqrt(u^2+v^2)",
        },
        "coordinate_permutation": permutation.to_manifest(),
        "mountain_location": {
            "latitude_rad": np.pi / 6.0,
            "longitude_rad": 3.0 * np.pi / 2.0,
        },
    })
    atomic_write_json(stage / "day0_contract_metrics.json", day0)
    day0_plot_limits = _render_comparison_maps(
        stage, spec, fields, reference,
        reference["latitude"], reference["longitude"], indices=(0,))
    atomic_write_json(stage / "plot_limits.json", day0_plot_limits)
    if not day0["passed"]:
        print(json.dumps(
            day0["likely_diagnostic_causes_if_failed"], indent=2), flush=True)
    require_day0_contract(day0)

    # Forecast metrics are deliberately not evaluated until the day-zero
    # contract above has passed and been persisted.
    comparison_records = []
    for index, time_days in enumerate(reference["time_days"]):
        record = _comparison_record(
            float(time_days), fields, reference, index,
            reference["latitude_weights"], dlon)
        record["reference_label"] = (
            f"MRI\u2192T{spec.truncation} projected reference")
        comparison_records.append(record)
    atomic_write_json(stage / "projected_reference_metrics.json", {
        "comparison": (
            "MRI projected-reference agreement on the Aeolus native "
            "Gaussian grid"),
        "reference_is_untouched_n958": False,
        "records": comparison_records,
        "norm_definitions": NORM_DEFINITIONS,
        "quadrature": QUADRATURE_DEFINITION,
    })
    flat_records = [
        _flatten_comparison_record(record)
        for record in comparison_records
    ]
    atomic_write_text(
        stage / "projected_reference_metrics.csv",
        _csv_text(flat_records))

    conservation_summary, conservation_rows = _conservation_products(
        capsule, model, coefficients, stage)
    forecast_plot_limits = _render_comparison_maps(
        stage, spec, fields, reference,
        reference["latitude"], reference["longitude"], indices=(1, 2, 3))
    plot_limits = {**day0_plot_limits, **forecast_plot_limits}
    atomic_write_json(stage / "plot_limits.json", plot_limits)
    _render_error_timeseries(stage, comparison_records)
    _render_conservation_timeseries(stage, conservation_rows)

    capsule_relative = capsule.relative_to(stage).as_posix()
    completion_without_hashes = {
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "completed_at_utc": _utc_now(),
        "stage_signature": signature,
        "stage": spec.stage_name,
        "truncation": spec.truncation,
        "grid": [spec.nlat, spec.nlon],
        "reference_artifact": spec.filename,
        "reference_sha256": reference_artifact.sha256,
        "aeolus_capsule_relative_path": capsule_relative,
        "aeolus_run_id": capsule_manifest.get("run_id"),
        "exact_snapshot_times_s": times.tolist(),
        "coordinate_permutation": permutation.to_manifest(),
        "day0_contract_passed": True,
        "conservation_summary": conservation_summary,
        "plot_limits_recorded": plot_limits,
    }
    output_hashes = hash_file_inventory(stage)
    completion = {
        **completion_without_hashes,
        "output_hashes": output_hashes,
    }
    # The sentinel is written atomically and last.  A disconnect before this
    # point leaves a same-signature incomplete stage that can reuse its
    # lifecycle-completed Aeolus capsule.
    atomic_write_json(stage / "COMPLETE.json", completion)
    return completion


def _load_stage_comparison(stage: pathlib.Path) -> Mapping[str, Any]:
    return json.loads(
        (stage / "projected_reference_metrics.json").read_text(
            encoding="utf-8"))


def _load_completed_capsule(stage: pathlib.Path) -> pathlib.Path:
    complete = json.loads(
        (stage / "COMPLETE.json").read_text(encoding="utf-8"))
    capsule = (stage / complete["aeolus_capsule_relative_path"]).resolve()
    capsule.relative_to(stage.resolve())
    return capsule


def _safe_common_subset_l2(field: np.ndarray,
                           reference: np.ndarray) -> dict[str, Any]:
    from planetary_sandbox.run.swe.convergence import (
        common_subset_relative_l2)

    n_l = min(field.shape[-2], reference.shape[-2])
    n_m = min(field.shape[-1], reference.shape[-1])
    shared_reference = reference[..., :n_l, :n_m]
    if not np.any(shared_reference):
        return {
            "value": None,
            "status": "undefined_zero_reference_denominator",
            "denominator": "shared-triangle reference spectral L2 norm",
        }
    value = common_subset_relative_l2(field, reference)
    if not math.isfinite(value):
        raise RuntimeError("self-convergence metric became non-finite")
    return {
        "value": value,
        "status": "defined",
        "denominator": "shared-triangle reference spectral L2 norm",
    }


def generate_self_convergence(output_root: pathlib.Path) -> dict[str, Any]:
    coarse_stage = output_root / ARTIFACT_SPECS["t42_64x128.npz"].stage_name
    fine_stage = output_root / ARTIFACT_SPECS["t63_96x192.npz"].stage_name
    coarse_capsule = _load_completed_capsule(coarse_stage)
    fine_capsule = _load_completed_capsule(fine_stage)
    coarse_times = np.load(
        coarse_capsule / "swe_snapshot_times.npy", allow_pickle=False)
    fine_times = np.load(
        fine_capsule / "swe_snapshot_times.npy", allow_pickle=False)
    if not np.array_equal(coarse_times, fine_times) or not np.array_equal(
            coarse_times, REQUIRED_SNAPSHOT_SECONDS):
        raise RuntimeError("self-convergence snapshot schedules differ")
    coarse = np.load(coarse_capsule / "swe_coeffs.npy", allow_pickle=False)
    fine = np.load(fine_capsule / "swe_coeffs.npy", allow_pickle=False)

    records = []
    for index, time_s in enumerate(coarse_times):
        metrics = {
            name: _safe_common_subset_l2(
                coarse[index, field_index], fine[index, field_index])
            for field_index, name in enumerate(FIELD_NAMES)
        }
        records.append({
            "time_s": float(time_s),
            "time_days": float(time_s / 86400.0),
            "shared_triangle": "l <= 42",
            "metrics": metrics,
        })
    result = {
        "comparison": "T42\u2013T63 Aeolus self-convergence",
        "method": (
            "existing Aeolus common_subset_relative_l2 on matching persisted "
            "spectral states, restricted to the shared l<=42 triangle"),
        "experiments_differ_in": [
            "spectral truncation",
            "native Gaussian grid",
            "resolved topography representation",
            "adaptive timestep history",
        ],
        "not_a_projected_reference_metric": True,
        "records": records,
    }
    target = output_root / "self_convergence"
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "metrics.json", result)
    rows = []
    for record in records:
        row = {
            "time_s": record["time_s"],
            "time_days": record["time_days"],
            "shared_triangle": record["shared_triangle"],
        }
        for name, metric in record["metrics"].items():
            row[f"{name}_relative_l2"] = metric["value"]
            row[f"{name}_relative_l2_status"] = metric["status"]
        rows.append(row)
    atomic_write_text(target / "metrics.csv", _csv_text(rows))
    _render_self_convergence(target, rows)
    return result


def _render_self_convergence(
        target: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = target / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    days = [row["time_days"] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in FIELD_NAMES:
        values = [
            np.nan if row[f"{name}_relative_l2"] is None
            else row[f"{name}_relative_l2"] for row in rows]
        axis.semilogy(days, values, marker="o", label=name)
    axis.set_xlabel("time [days]")
    axis.set_ylabel("common-subset relative spectral L2")
    axis.set_xticks(days)
    axis.set_title("Aeolus T42\u2013T63 self-convergence on shared l\u226442")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _save_figure_atomic(
        fig, plot_dir / "self_convergence.png", dpi=300)
    _save_figure_atomic(
        fig, plot_dir / "self_convergence.pdf")
    plt.close(fig)


def _summary_rows(output_root: pathlib.Path,
                  specs: Sequence[ArtifactSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        payload = _load_stage_comparison(output_root / spec.stage_name)
        for record in payload["records"]:
            if record["time_days"] == 0.0:
                continue
            flat = _flatten_comparison_record(record)
            rows.append({
                "truncation": f"T{spec.truncation}",
                "grid": f"{spec.nlat}x{spec.nlon}",
                "time_days": flat["time_days"],
                "height_L1": flat["height_L1"],
                "height_L2": flat["height_L2"],
                "height_Linf": flat["height_Linf"],
                "velocity_vector_L1": flat["velocity_vector_L1"],
                "velocity_vector_L2": flat["velocity_vector_L2"],
                "velocity_vector_Linf": flat["velocity_vector_Linf"],
                "height_weighted_rms_error_m":
                    flat["height_weighted_rms_error"],
                "velocity_weighted_rms_error_m_s-1":
                    flat["velocity_vector_weighted_rms_error"],
            })
    return rows


_README_FALLBACK = """# Williamson-5 MRI projected-reference validation

Aeolus T42 and T63 solutions were compared on their native Gaussian grids
with independently prepared T42 and T63 spectral projections of the MRI\u2013JMA
N958 reference integration. Reference preparation used no Aeolus code, and
the frozen projected fields were consumed by hash without further
interpolation or spectral processing.

## Four distinct validation views

1. **Day-zero convention verification** checks initial conditions, units,
   velocity conventions, grid alignment, mountain location, and the audited
   free-surface formula before forecast errors are evaluated.
2. **MRI projected-reference agreement** reports resolved-scale scalar-height
   and vector-velocity errors at days 5, 10, and 15.
3. **Aeolus conservation and stability** reports intrinsic energy,
   potential-enstrophy, thickness, wind, finite-state, step, and timestep
   diagnostics separately from reference agreement.
4. **T42\u2013T63 self-convergence** compares persisted Aeolus spectral states on
   the common `l <= 42` triangle. The experiments differ in truncation, native
   grid, resolved topography, and timestep history.

No project-defined threshold is claimed to be an official Williamson pass
criterion. These results are not direct errors against the untouched N958
field.
"""


def _readme_template(repository_root: pathlib.Path) -> str:
    candidate = (
        repository_root / "docs" / "validation"
        / "w5_mri_validation_README.template.md")
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return _README_FALLBACK


def generate_summary(output_root: pathlib.Path,
                     repository_root: pathlib.Path,
                     specs: Sequence[ArtifactSpec]) -> list[dict[str, Any]]:
    summary = output_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows(output_root, specs)
    atomic_write_text(summary / "metrics_table.csv", _csv_text(rows))
    readme = _readme_template(repository_root)
    readme += (
        "\n## Artifacts\n\n"
        "- `metrics_table.csv`: compact day-5/day-10/day-15 agreement table.\n"
        "- Per-stage directories: day-zero gate, projected-reference metrics, "
        "conservation diagnostics, native-grid figures, and Aeolus capsule.\n"
        "- `self_convergence/`: shared-triangle Aeolus-only comparison when "
        "both primary stages completed.\n"
        "- `validation_manifest.json`: strict-JSON provenance and hash index.\n"
    )
    atomic_write_text(summary / "README.md", readme)
    return rows


DIAGNOSTIC_DEFINITIONS = {
    "relative_total_energy_drift": "(E(t)-E(0))/abs(E(0))",
    "relative_potential_enstrophy_drift": "(Z(t)-Z(0))/abs(Z(0))",
    "minimum_fluid_layer_thickness": (
        "minimum of (Phi0+phi)/g over every model sampling, from canonical "
        "per-step Aeolus diagnostics"),
    "maximum_wind_speed": (
        "maximum sqrt(u^2+v^2) on the Aeolus state grid, per accepted step"),
    "timestep_statistics": (
        "min/max/mean/median accepted dt; the initial dt=0 diagnostic row "
        "is excluded"),
    "non_finite_state_detection": (
        "canonical runner validates every RK4 stage and accepted state; "
        "persisted states and consumed diagnostic columns are rechecked"),
}


def _build_validation_manifest(
        *, config: WorkflowConfig, repository: Mapping[str, Any],
        package: FrozenReferencePackage, specs: Sequence[ArtifactSpec],
        tolerances: Mapping[str, Any], self_convergence: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
    stages = {}
    for spec in specs:
        stage = config.output_root / spec.stage_name
        complete = json.loads(
            (stage / "COMPLETE.json").read_text(encoding="utf-8"))
        day0 = json.loads(
            (stage / "day0_contract_metrics.json").read_text(
                encoding="utf-8"))
        manifest = json.loads(
            (_load_completed_capsule(stage) / "manifest.json").read_text(
                encoding="utf-8"))
        stages[spec.stage_name] = {
            "resolved_run_config": manifest.get("run_config"),
            "aeolus_run_id": manifest.get("run_id"),
            "aeolus_capsule_git": manifest.get("git"),
            "aeolus_capsule_numerics": manifest.get("numerics"),
            "aeolus_capsule_notes": manifest.get("notes"),
            "canonical_model_defaults": {
                "horizontal_diffusion": "none",
                "hyperdiffusion_nu4_m4_s-1": 0.0,
                "integrator": "existing Aeolus RK4 timestepper",
                "adaptive_cfl": "existing Aeolus adaptive-CFL machinery",
            },
            "coordinate_permutation": complete["coordinate_permutation"],
            "day0_gate": {
                "passed": day0["passed"],
                "tolerances": day0["tolerances"],
                "checks": day0["checks"],
            },
            "exact_snapshot_times_s": complete["exact_snapshot_times_s"],
            "reference_artifact": complete["reference_artifact"],
            "conservation_summary": complete["conservation_summary"],
        }
    manifest = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "aeolus_repository": {
            "url": config.repository_url,
            "git_ref": config.git_ref,
            **repository,
        },
        "reference": {
            "schema_version": package.manifest["schema_version"],
            "hashes": dict(package.hashes),
            "source_identity": package.manifest.get("dataset_identity"),
            "truncations_and_grids": package.manifest.get("targets"),
            "height_interpretation":
                package.manifest.get("height_interpretation"),
            "archived_relative_path": "reference/",
            "immutable_input": True,
        },
        "aeolus_height_convention_audit": AEOLUS_HEIGHT_AUDIT,
        "day0_gate_tolerances": tolerances,
        "norm_definitions": NORM_DEFINITIONS,
        "quadrature_convention": QUADRATURE_DEFINITION,
        "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
        "stages": stages,
        "self_convergence": self_convergence,
        "scientific_language": {
            "reference_comparison": "MRI projected-reference agreement",
            "intrinsic_diagnostics": "Aeolus conservation and stability",
            "cross_resolution": "T42\u2013T63 self-convergence",
            "initial_gate": "day-zero convention verification",
        },
        "reference_firewall": {
            "original_mri_data_nc_read": False,
            "skyborn_or_spherepack_installed_or_imported": False,
            "mri_spectral_analysis_or_projection": False,
            "mri_velocity_reconstruction": False,
            "interpolation_or_regridding": False,
            "reference_artifacts_altered_or_regenerated": False,
            "missing_reference_times_added": False,
        },
        "warnings_and_caveats": [
            "Agreement metrics are against independent MRI\u2192T42 and "
            "MRI\u2192T63 low-pass projected references, not the untouched "
            "N958 physical grid.",
            "Project-defined diagnostics are not official Williamson pass "
            "criteria.",
            "T42 and T63 experiments differ in spectral truncation, native "
            "grid, resolved topography representation, and timestep history.",
            "No explicit horizontal diffusion was added; the canonical "
            "inviscid W5 configuration was used unchanged.",
            *package.manifest.get("warnings_and_scientific_caveats", []),
        ],
    }
    manifest["output_file_hashes"] = hash_file_inventory(
        config.output_root, exclude=("validation_manifest.json",))
    return manifest


def run_validation_workflow(config: WorkflowConfig) -> dict[str, Any]:
    """Verify reference, execute/reuse independent stages, and summarize."""
    config = dataclasses.replace(
        config,
        reference_dir=pathlib.Path(config.reference_dir),
        output_root=pathlib.Path(config.output_root),
        repository_root=pathlib.Path(config.repository_root),
    )
    repository = repository_identity(
        config.repository_root, configured_url=config.repository_url,
        git_ref=config.git_ref)
    print(
        f"Aeolus commit: {repository['commit_sha']} | "
        f"dirty tree: {repository['dirty']}", flush=True)

    # Verification happens before output values are loaded or any model run.
    source_package = verify_reference_package(config.reference_dir)
    print("Frozen reference hashes and schema verified.", flush=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    package = archive_reference_package(source_package, config.output_root)
    tolerances = derive_day0_tolerances(package.manifest)
    atomic_write_json(
        config.output_root / "day0_tolerances_predeclared.json", tolerances)

    requested_specs = []
    if config.run_t42:
        requested_specs.append(ARTIFACT_SPECS["t42_64x128.npz"])
    if config.run_t63:
        requested_specs.append(ARTIFACT_SPECS["t63_96x192.npz"])
    if not requested_specs:
        raise ValueError("at least one of run_t42 or run_t63 must be enabled")

    completions = {}
    for spec in requested_specs:
        completions[spec.stage_name] = execute_stage(
            spec=spec, package=package, output_root=config.output_root,
            repository=repository, tolerances=tolerances,
            force_rerun=config.force_rerun)

    convergence = None
    both = {
        ARTIFACT_SPECS["t42_64x128.npz"].stage_name,
        ARTIFACT_SPECS["t63_96x192.npz"].stage_name,
    }
    completed_names = {
        name for name in both
        if (config.output_root / name / "COMPLETE.json").is_file()
    }
    if completed_names == both:
        convergence = generate_self_convergence(config.output_root)

    generate_summary(
        config.output_root, config.repository_root, requested_specs)
    manifest = _build_validation_manifest(
        config=config, repository=repository, package=package,
        specs=requested_specs, tolerances=tolerances,
        self_convergence=convergence)
    manifest_path = atomic_write_json(
        config.output_root / "validation_manifest.json", manifest)
    final_hash = sha256_file(manifest_path)
    print(f"Validation root: {config.output_root}", flush=True)
    print(
        f"Validation manifest: {manifest_path} (sha256={final_hash})",
        flush=True)
    for path in sorted(config.output_root.rglob("COMPLETE.json")):
        print(
            f"Stage sentinel: {path} (sha256={sha256_file(path)})",
            flush=True)
    return {
        "validation_root": str(config.output_root),
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": final_hash,
        "stage_completions": completions,
    }


__all__ = [
    "DEFAULT_REPOSITORY_URL",
    "WorkflowConfig",
    "execute_stage",
    "generate_self_convergence",
    "repository_identity",
    "run_validation_workflow",
]
