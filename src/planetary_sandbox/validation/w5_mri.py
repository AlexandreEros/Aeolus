"""Immutable-reference contracts and metrics for W5 MRI validation.

This module is intentionally NumPy-only.  It never imports CuPy, opens the
original MRI NetCDF, or performs spectral analysis, projection, wind
reconstruction, interpolation, or regridding.  The workflow layer may pass
Aeolus fields produced by the canonical model APIs into these helpers.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


REFERENCE_SCHEMA_VERSION = "mri-w5-reference-v1"
REQUIRED_REFERENCE_KEYS = frozenset({
    "time_days",
    "height",
    "u",
    "v",
    "latitude",
    "longitude",
    "latitude_weights",
})
REQUIRED_TIMES_DAYS = np.asarray([0.0, 5.0, 10.0, 15.0],
                                 dtype=np.float64)
REFERENCE_HASHES = {
    "manifest.json":
        "e294512a13ce89173eae2cda629bdb3f6d9c3322f0ecfa9293c9951589669c3d",
    "t42_64x128.npz":
        "fa320414507a7ed7f859c5061bd9782b07dda0b504ca5f68d57608360fb175d7",
    "t63_96x192.npz":
        "0072a566d0e8360277ef3da659dbffbc078d9fc3e6a7304c749d6d001f7a2dae",
}

# These tolerances validate only coordinates/quadrature metadata.  They do not
# relax the field comparison.  The longitude tolerance is inherited from the
# reference-preparation manifest; the weight sum tolerance is likewise the
# preparation contract.
COORDINATE_ATOL_RAD = 5.0e-12
WEIGHT_SUM_ATOL = 5.0e-13


class W5MRIValidationError(RuntimeError):
    """Base class for validation-only workflow failures."""


class ReferenceContractError(W5MRIValidationError):
    """A frozen reference file failed its immutable-input contract."""


class GridContractError(W5MRIValidationError):
    """The Aeolus native grid cannot be matched by a permitted permutation."""


class DayZeroContractError(W5MRIValidationError):
    """Initial-condition or convention agreement failed."""


class StageSignatureError(W5MRIValidationError):
    """A resumable stage exists but belongs to a different run signature."""


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    truncation: int
    nlat: int
    nlon: int

    @property
    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "time_days": (4,),
            "height": (4, self.nlat, self.nlon),
            "u": (4, self.nlat, self.nlon),
            "v": (4, self.nlat, self.nlon),
            "latitude": (self.nlat,),
            "longitude": (self.nlon,),
            "latitude_weights": (self.nlat,),
        }

    @property
    def stage_name(self) -> str:
        return f"t{self.truncation}_{self.nlat}x{self.nlon}"


ARTIFACT_SPECS = {
    "t42_64x128.npz": ArtifactSpec("t42_64x128.npz", 42, 64, 128),
    "t63_96x192.npz": ArtifactSpec("t63_96x192.npz", 63, 96, 192),
}


@dataclasses.dataclass(frozen=True)
class ReferenceArtifact:
    spec: ArtifactSpec
    arrays: Mapping[str, np.ndarray]
    sha256: str


@dataclasses.dataclass(frozen=True)
class FrozenReferencePackage:
    directory: pathlib.Path
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, ReferenceArtifact]
    hashes: Mapping[str, str]


def sha256_file(path: os.PathLike[str] | str,
                *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without interpreting file contents."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def load_strict_json(path: os.PathLike[str] | str) -> Any:
    """Load RFC-compatible JSON and reject NaN/Infinity tokens."""
    try:
        return json.loads(
            pathlib.Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as err:
        raise ReferenceContractError(
            f"invalid strict JSON in {path}: {err}") from err


def strict_jsonable(value: Any) -> Any:
    """Convert common scientific scalars while rejecting non-finite values."""
    if dataclasses.is_dataclass(value):
        return strict_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): strict_jsonable(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return strict_jsonable(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("strict JSON cannot represent NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported strict-JSON value: {type(value).__name__}")


def strict_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(strict_jsonable(value), indent=indent, allow_nan=False,
                      sort_keys=True) + "\n"


def atomic_write_text(path: os.PathLike[str] | str, text: str) -> pathlib.Path:
    """Replace one text file atomically after a fully flushed temporary write."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: os.PathLike[str] | str,
                      value: Any) -> pathlib.Path:
    return atomic_write_text(path, strict_json_dumps(value))


def validate_reference_manifest(
        manifest: Mapping[str, Any],
        *, expected_hashes: Mapping[str, str] = REFERENCE_HASHES) -> None:
    """Validate the frozen manifest contract without following recorded paths."""
    if not isinstance(manifest, Mapping):
        raise ReferenceContractError("manifest.json must contain a JSON object")
    if manifest.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ReferenceContractError(
            "reference schema_version mismatch: expected "
            f"{REFERENCE_SCHEMA_VERSION!r}, got "
            f"{manifest.get('schema_version')!r}")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or (
            validation.get("all_reference_preparation_checks_passed")
            is not True):
        raise ReferenceContractError(
            "reference preparation checks were not explicitly recorded as "
            "passed")
    height = manifest.get("height_interpretation")
    if not isinstance(height, Mapping) or (
            height.get("resolved_as") != "free_surface_height"):
        raise ReferenceContractError(
            "reference height must be resolved_as='free_surface_height'")

    final_artifacts = manifest.get("final_artifacts")
    targets = manifest.get("targets")
    if not isinstance(final_artifacts, Mapping) or not isinstance(
            targets, Mapping):
        raise ReferenceContractError(
            "manifest must contain final_artifacts and targets objects")
    for filename, spec in ARTIFACT_SPECS.items():
        artifact_entry = final_artifacts.get(filename)
        target_entry = targets.get(filename)
        if not isinstance(artifact_entry, Mapping) or not isinstance(
                target_entry, Mapping):
            raise ReferenceContractError(
                f"manifest is missing contract metadata for {filename}")
        if artifact_entry.get("sha256") != expected_hashes[filename]:
            raise ReferenceContractError(
                f"manifest hash metadata mismatch for {filename}")
        expected_target = (spec.truncation, spec.nlat, spec.nlon)
        actual_target = (
            target_entry.get("truncation"),
            target_entry.get("nlat"),
            target_entry.get("nlon"),
        )
        if actual_target != expected_target:
            raise ReferenceContractError(
                f"manifest target mismatch for {filename}: expected "
                f"{expected_target}, got {actual_target}")


def validate_reference_artifact(path: os.PathLike[str] | str,
                                spec: ArtifactSpec,
                                digest: str = "<not-computed>"
                                ) -> ReferenceArtifact:
    """Load and validate one frozen NPZ without repairing any mismatch."""
    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_keys = frozenset(archive.files)
            if actual_keys != REQUIRED_REFERENCE_KEYS:
                missing = sorted(REQUIRED_REFERENCE_KEYS - actual_keys)
                extra = sorted(actual_keys - REQUIRED_REFERENCE_KEYS)
                raise ReferenceContractError(
                    f"{spec.filename} keys mismatch; missing={missing}, "
                    f"extra={extra}")
            arrays = {key: np.array(archive[key], copy=True)
                      for key in REQUIRED_REFERENCE_KEYS}
    except ReferenceContractError:
        raise
    except Exception as err:
        raise ReferenceContractError(
            f"could not read {spec.filename} as an immutable NPZ: {err}") from err

    for key, expected_shape in spec.expected_shapes.items():
        array = arrays[key]
        if array.shape != expected_shape:
            raise ReferenceContractError(
                f"{spec.filename}:{key} shape mismatch: expected "
                f"{expected_shape}, got {array.shape}")
        if array.dtype != np.dtype(np.float64):
            raise ReferenceContractError(
                f"{spec.filename}:{key} must be float64, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ReferenceContractError(
                f"{spec.filename}:{key} contains non-finite values")

    if not np.array_equal(arrays["time_days"], REQUIRED_TIMES_DAYS):
        raise ReferenceContractError(
            f"{spec.filename}:time_days must equal [0, 5, 10, 15] exactly")

    latitude = arrays["latitude"]
    if not np.all(np.diff(latitude) < 0.0):
        raise ReferenceContractError(
            f"{spec.filename}:latitude must be strictly north-to-south")
    if not np.all((-0.5 * np.pi < latitude)
                  & (latitude < 0.5 * np.pi)):
        raise ReferenceContractError(
            f"{spec.filename}:Gaussian latitudes must lie strictly between "
            "the poles")

    longitude = arrays["longitude"]
    if not np.all((0.0 <= longitude) & (longitude < 2.0 * np.pi)):
        raise ReferenceContractError(
            f"{spec.filename}:longitude must lie in [0, 2*pi)")
    if not np.all(np.diff(longitude) > 0.0):
        raise ReferenceContractError(
            f"{spec.filename}:longitude must be strictly increasing")
    periodic_steps = np.diff(np.r_[longitude, longitude[0] + 2.0 * np.pi])
    nominal_step = 2.0 * np.pi / spec.nlon
    if not np.allclose(periodic_steps, nominal_step, rtol=0.0,
                       atol=COORDINATE_ATOL_RAD):
        raise ReferenceContractError(
            f"{spec.filename}:longitude is not uniformly periodic")
    if math.isclose(float(longitude[-1] - longitude[0]), 2.0 * np.pi,
                    rel_tol=0.0, abs_tol=COORDINATE_ATOL_RAD):
        raise ReferenceContractError(
            f"{spec.filename}:duplicate periodic longitude endpoint")

    weights = arrays["latitude_weights"]
    if not np.all(weights > 0.0):
        raise ReferenceContractError(
            f"{spec.filename}:latitude_weights must be positive")
    if not math.isclose(float(weights.sum()), 2.0, rel_tol=0.0,
                        abs_tol=WEIGHT_SUM_ATOL):
        raise ReferenceContractError(
            f"{spec.filename}:latitude_weights must sum to 2")
    for array in arrays.values():
        array.flags.writeable = False
    return ReferenceArtifact(spec=spec, arrays=arrays, sha256=digest)


def verify_reference_package(
        reference_dir: os.PathLike[str] | str,
        *, expected_hashes: Mapping[str, str] = REFERENCE_HASHES,
        ) -> FrozenReferencePackage:
    """Hash all three files first, then validate manifest and NPZ contents.

    File discovery is strictly relative to ``reference_dir``.  Absolute paths
    recorded inside the preparation manifest are provenance only and are
    never consulted here.
    """
    directory = pathlib.Path(reference_dir)
    expected_names = frozenset(REFERENCE_HASHES)
    if frozenset(expected_hashes) != expected_names:
        raise ValueError(
            f"expected_hashes must define exactly {sorted(expected_names)}")

    hashes: dict[str, str] = {}
    for filename in REFERENCE_HASHES:
        path = directory / filename
        if not path.is_file():
            raise ReferenceContractError(
                f"required frozen reference file is missing: {path}")
        hashes[filename] = sha256_file(path)
        if hashes[filename] != expected_hashes[filename]:
            raise ReferenceContractError(
                f"SHA-256 mismatch for {filename}: expected "
                f"{expected_hashes[filename]}, got {hashes[filename]}")

    manifest = load_strict_json(directory / "manifest.json")
    validate_reference_manifest(manifest, expected_hashes=expected_hashes)
    artifacts = {
        filename: validate_reference_artifact(
            directory / filename, spec, hashes[filename])
        for filename, spec in ARTIFACT_SPECS.items()
    }
    return FrozenReferencePackage(
        directory=directory, manifest=manifest, artifacts=artifacts,
        hashes=hashes)


def reference_identity(package: FrozenReferencePackage) -> dict[str, Any]:
    """Small auditable identity record without preparation-run file paths."""
    manifest = package.manifest
    return {
        "schema_version": manifest["schema_version"],
        "hashes": dict(package.hashes),
        "dataset_identity": manifest.get("dataset_identity"),
        "run_signature": manifest.get("run_signature"),
        "height_interpretation": manifest.get("height_interpretation"),
        "required_times": manifest.get("required_times"),
        "targets": manifest.get("targets"),
        "validation_all_reference_preparation_checks_passed": True,
        "configured_reference_dir_recorded_for_provenance_only":
            str(package.directory),
        "artifact_discovery": "all files resolved relative to REFERENCE_DIR",
    }


def _angular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


@dataclasses.dataclass(frozen=True)
class GridPermutation:
    """Exact reversible operations that map Aeolus fields to reference order."""
    latitude_reversed: bool = False
    longitude_roll: int = 0
    axes_transposed: bool = False

    def apply(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values)
        if result.ndim < 2:
            raise GridContractError("grid fields must have at least two axes")
        if self.axes_transposed:
            result = np.swapaxes(result, -2, -1)
        if self.latitude_reversed:
            result = np.flip(result, axis=-2)
        if self.longitude_roll:
            result = np.roll(result, self.longitude_roll, axis=-1)
        return result

    def to_manifest(self) -> dict[str, Any]:
        return {
            "axes_transposed": self.axes_transposed,
            "latitude_reversed": self.latitude_reversed,
            "longitude_roll": self.longitude_roll,
            "operations_are_exact_reversible_index_permutations": True,
            "interpolation_or_regridding": False,
        }


def determine_grid_permutation(
        native_latitude: Sequence[float],
        native_longitude: Sequence[float],
        reference_latitude: Sequence[float],
        reference_longitude: Sequence[float],
        *, native_field_shape: tuple[int, int] | None = None,
        atol: float = COORDINATE_ATOL_RAD,
        ) -> GridPermutation:
    """Match coordinates using only reversal, cyclic roll, and transpose."""
    native_lat = np.asarray(native_latitude, dtype=np.float64)
    native_lon = np.asarray(native_longitude, dtype=np.float64)
    ref_lat = np.asarray(reference_latitude, dtype=np.float64)
    ref_lon = np.asarray(reference_longitude, dtype=np.float64)
    if native_lat.ndim != 1 or native_lon.ndim != 1:
        raise GridContractError("Aeolus native coordinates must be 1-D axes")
    if native_lat.shape != ref_lat.shape or native_lon.shape != ref_lon.shape:
        raise GridContractError(
            "Aeolus and reference coordinate-axis lengths differ")

    if np.allclose(native_lat, ref_lat, rtol=0.0, atol=atol):
        latitude_reversed = False
    elif np.allclose(native_lat[::-1], ref_lat, rtol=0.0, atol=atol):
        latitude_reversed = True
    else:
        raise GridContractError(
            "latitude axes do not match directly or by exact reversal")

    longitude_roll: int | None = None
    for start in np.flatnonzero(
            _angular_error(native_lon, ref_lon[0]) <= atol):
        roll = -int(start)
        candidate = np.roll(native_lon, roll)
        if np.all(_angular_error(candidate, ref_lon) <= atol):
            longitude_roll = roll
            break
    if longitude_roll is None:
        raise GridContractError(
            "longitude axes do not match by a cyclic roll")

    transposed = False
    if native_field_shape is not None:
        expected = (ref_lat.size, ref_lon.size)
        if tuple(native_field_shape) == expected:
            transposed = False
        elif tuple(native_field_shape) == expected[::-1]:
            transposed = True
        else:
            raise GridContractError(
                f"Aeolus field shape {native_field_shape} is neither "
                f"{expected} nor its documented transpose {expected[::-1]}")

    permutation = GridPermutation(
        latitude_reversed=latitude_reversed,
        longitude_roll=longitude_roll,
        axes_transposed=transposed,
    )
    aligned_lat = native_lat[::-1] if latitude_reversed else native_lat
    aligned_lon = np.roll(native_lon, longitude_roll)
    if not np.allclose(aligned_lat, ref_lat, rtol=0.0, atol=atol) or not (
            np.all(_angular_error(aligned_lon, ref_lon) <= atol)):
        raise GridContractError("coordinate equality failed after permutation")
    return permutation


def construct_aeolus_heights(
        perturbation_geopotential: np.ndarray,
        *, base_geopotential: float,
        surface_geopotential: np.ndarray | float,
        gravity: float,
        ) -> dict[str, np.ndarray]:
    """Apply Aeolus's audited W5 state convention without changing the state.

    ``phi`` is perturbation *fluid-layer* geopotential and ``Phi0`` is the
    base fluid-layer geopotential.  Fixed terrain ``phi_s`` is separate:

    ``fluid thickness = (Phi0 + phi) / g``
    ``free-surface height = (Phi0 + phi + phi_s) / g``
    """
    phi = np.asarray(perturbation_geopotential, dtype=np.float64)
    phi_s = np.asarray(surface_geopotential, dtype=np.float64)
    if not math.isfinite(base_geopotential):
        raise ValueError("base_geopotential must be finite")
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(phi).all() or not np.isfinite(phi_s).all():
        raise ValueError("height-construction inputs must be finite")
    thickness = (base_geopotential + phi) / gravity
    free_surface = (base_geopotential + phi + phi_s) / gravity
    return {
        "fluid_layer_thickness_m": thickness,
        "free_surface_height_m": free_surface,
    }


def _validate_quadrature_field(
        values: np.ndarray,
        latitude_weights: Sequence[float],
        longitude_spacing: float,
        ) -> tuple[np.ndarray, np.ndarray, float]:
    field = np.asarray(values, dtype=np.float64)
    weights = np.asarray(latitude_weights, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"quadrature field must be 2-D, got {field.shape}")
    if weights.shape != (field.shape[0],):
        raise ValueError("latitude weights do not match the field")
    if not np.isfinite(field).all() or not np.isfinite(weights).all():
        raise ValueError("quadrature inputs must be finite")
    if not np.all(weights > 0.0):
        raise ValueError("latitude weights must be positive")
    if not math.isfinite(longitude_spacing) or longitude_spacing <= 0.0:
        raise ValueError("longitude spacing must be finite and positive")
    return field, weights, float(longitude_spacing)


def weighted_integral(values: np.ndarray,
                      latitude_weights: Sequence[float],
                      longitude_spacing: float) -> float:
    """Explicit sum_j sum_k f[j,k] w[j] delta_lambda."""
    field, weights, dlon = _validate_quadrature_field(
        values, latitude_weights, longitude_spacing)
    return float(np.sum(field * weights[:, None]) * dlon)


def _absolute_error_metrics(error_magnitude: np.ndarray,
                            latitude_weights: Sequence[float],
                            longitude_spacing: float) -> dict[str, float]:
    error = np.asarray(error_magnitude, dtype=np.float64)
    sphere_area = 4.0 * np.pi
    return {
        "weighted_mean_absolute_error": (
            weighted_integral(np.abs(error), latitude_weights,
                              longitude_spacing) / sphere_area),
        "weighted_rms_error": math.sqrt(
            weighted_integral(error * error, latitude_weights,
                              longitude_spacing) / sphere_area),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def _normalized(numerator: float, denominator: float,
                *, denominator_name: str, square_root: bool = False
                ) -> dict[str, Any]:
    if denominator == 0.0:
        return {
            "value": None,
            "status": "undefined_zero_reference_denominator",
            "denominator": denominator_name,
        }
    value = numerator / denominator
    if square_root:
        value = math.sqrt(value)
    if not math.isfinite(value):
        raise ValueError("normalized metric unexpectedly became non-finite")
    return {
        "value": float(value),
        "status": "defined",
        "denominator": denominator_name,
    }


def scalar_error_metrics(field: np.ndarray,
                         reference: np.ndarray,
                         latitude_weights: Sequence[float],
                         longitude_spacing: float) -> dict[str, Any]:
    """Williamson normalized scalar norms plus absolute area-weighted errors."""
    field = np.asarray(field, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if field.shape != reference.shape:
        raise ValueError("field and reference shapes differ")
    error = field - reference
    l1_num = weighted_integral(
        np.abs(error), latitude_weights, longitude_spacing)
    l1_den = weighted_integral(
        np.abs(reference), latitude_weights, longitude_spacing)
    l2_num = weighted_integral(
        error * error, latitude_weights, longitude_spacing)
    l2_den = weighted_integral(
        reference * reference, latitude_weights, longitude_spacing)
    linf_num = float(np.max(np.abs(error)))
    linf_den = float(np.max(np.abs(reference)))
    return {
        "normalized": {
            "L1": _normalized(
                l1_num, l1_den, denominator_name="integral(abs(reference))"),
            "L2": _normalized(
                l2_num, l2_den, square_root=True,
                denominator_name="sqrt(integral(reference^2))"),
            "Linf": _normalized(
                linf_num, linf_den, denominator_name="max(abs(reference))"),
        },
        "absolute": _absolute_error_metrics(
            error, latitude_weights, longitude_spacing),
    }


def vector_error_metrics(u: np.ndarray, v: np.ndarray,
                         u_reference: np.ndarray, v_reference: np.ndarray,
                         latitude_weights: Sequence[float],
                         longitude_spacing: float) -> dict[str, Any]:
    """Williamson vector norms and vector/component absolute diagnostics."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ur = np.asarray(u_reference, dtype=np.float64)
    vr = np.asarray(v_reference, dtype=np.float64)
    if not (u.shape == v.shape == ur.shape == vr.shape):
        raise ValueError("all vector components must share one shape")
    du = u - ur
    dv = v - vr
    error = np.hypot(du, dv)
    reference_speed = np.hypot(ur, vr)
    l1_num = weighted_integral(
        error, latitude_weights, longitude_spacing)
    l1_den = weighted_integral(
        reference_speed, latitude_weights, longitude_spacing)
    l2_num = weighted_integral(
        error * error, latitude_weights, longitude_spacing)
    l2_den = weighted_integral(
        reference_speed * reference_speed,
        latitude_weights, longitude_spacing)
    return {
        "normalized": {
            "L1": _normalized(
                l1_num, l1_den,
                denominator_name="integral(reference_speed)"),
            "L2": _normalized(
                l2_num, l2_den, square_root=True,
                denominator_name="sqrt(integral(reference_speed^2))"),
            "Linf": _normalized(
                float(np.max(error)), float(np.max(reference_speed)),
                denominator_name="max(reference_speed)"),
        },
        "absolute_vector": _absolute_error_metrics(
            error, latitude_weights, longitude_spacing),
        "absolute_u_component": _absolute_error_metrics(
            du, latitude_weights, longitude_spacing),
        "absolute_v_component": _absolute_error_metrics(
            dv, latitude_weights, longitude_spacing),
    }


def derive_day0_tolerances(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Predeclare conservative gates from Notebook A floors and float64 noise."""
    try:
        declared = manifest["validation"]["tolerances_declared_before_results"]
        analytic = declared["analytic_day0_max_abs"]
        scalar_roundtrip = declared["scalar_roundtrip_max_abs"]
        height_floor = max(float(analytic["h_m"]),
                           float(scalar_roundtrip["h"]))
        velocity_floor = max(
            float(analytic["u_m_s-1"]),
            float(analytic["v_m_s-1"]),
            float(declared["velocity_reconstruction_max_abs_m_s-1"]),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise ReferenceContractError(
            "manifest lacks the Notebook A validation floors needed to "
            "predeclare day-zero gates") from err
    double_floor = 256.0 * np.finfo(np.float64).eps
    return {
        "declared_before_forecast_metrics": True,
        "height_m": {
            "maximum_absolute_error": 2.0 * height_floor,
            "weighted_mean_absolute_error": height_floor,
            "weighted_rms_error": height_floor,
        },
        "velocity_component_m_s-1": {
            "maximum_absolute_error": 2.5 * velocity_floor,
            "weighted_mean_absolute_error": velocity_floor,
            "weighted_rms_error": velocity_floor,
        },
        "wind_speed_m_s-1": {
            "maximum_absolute_error": 2.5 * velocity_floor,
            "weighted_mean_absolute_error": velocity_floor,
            "weighted_rms_error": velocity_floor,
        },
        "derivation": {
            "notebook_a_height_floor_m": height_floor,
            "notebook_a_velocity_floor_m_s-1": velocity_floor,
            "aeolus_precision": "float64 fields / complex128 coefficients",
            "float64_relative_roundoff_floor": double_floor,
            "policy": (
                "height uses 2x Notebook A's maximum declared scalar floor "
                "for max error and 1x for weighted errors; velocity uses "
                "2.5x Notebook A's reconstruction floor for max error and "
                "1x for weighted errors. These are convention gates, not "
                "forecast-accuracy criteria, and are never auto-relaxed."
            ),
        },
    }


def _absolute_field_check(field: np.ndarray, reference: np.ndarray,
                          weights: np.ndarray, dlon: float) -> dict[str, float]:
    return _absolute_error_metrics(
        np.asarray(field) - np.asarray(reference), weights, dlon)


def evaluate_day0_contract(
        *, height: np.ndarray, u: np.ndarray, v: np.ndarray,
        height_reference: np.ndarray, u_reference: np.ndarray,
        v_reference: np.ndarray, latitude_weights: np.ndarray,
        longitude_spacing: float, tolerances: Mapping[str, Any],
        convention_checks: Mapping[str, bool] | None = None,
        ) -> dict[str, Any]:
    """Evaluate the initial-condition/convention gate without auto-relaxation."""
    fields = {
        "free_surface_height": _absolute_field_check(
            height, height_reference, latitude_weights, longitude_spacing),
        "eastward_velocity_u": _absolute_field_check(
            u, u_reference, latitude_weights, longitude_spacing),
        "northward_velocity_v": _absolute_field_check(
            v, v_reference, latitude_weights, longitude_spacing),
        "wind_speed": _absolute_field_check(
            np.hypot(u, v), np.hypot(u_reference, v_reference),
            latitude_weights, longitude_spacing),
    }
    tolerance_groups = {
        "free_surface_height": tolerances["height_m"],
        "eastward_velocity_u": tolerances["velocity_component_m_s-1"],
        "northward_velocity_v": tolerances["velocity_component_m_s-1"],
        "wind_speed": tolerances["wind_speed_m_s-1"],
    }
    checks: list[dict[str, Any]] = []
    for field_name, metrics in fields.items():
        for metric_name, value in metrics.items():
            limit = float(tolerance_groups[field_name][metric_name])
            checks.append({
                "field": field_name,
                "metric": metric_name,
                "value": value,
                "tolerance": limit,
                "passed": bool(value <= limit),
            })
    conventions = dict(convention_checks or {})
    for name, passed in conventions.items():
        checks.append({
            "field": "convention",
            "metric": name,
            "value": bool(passed),
            "tolerance": True,
            "passed": bool(passed),
        })
    passed = all(check["passed"] for check in checks)
    likely_causes: list[str] = []
    if not passed:
        likely_causes = [
            "free-surface versus layer-thickness mismatch",
            "missing or double-counted topography",
            "missing base geopotential",
            "latitude reversal",
            "longitude roll",
            "velocity sign or component-convention mismatch",
            "unit mismatch",
        ]
    return {
        "contract": "day-zero convention verification",
        "passed": passed,
        "tolerances": tolerances,
        "field_metrics": fields,
        "convention_checks": conventions,
        "checks": checks,
        "likely_diagnostic_causes_if_failed": likely_causes,
    }


def require_day0_contract(result: Mapping[str, Any]) -> None:
    if result.get("passed") is not True:
        causes = "; ".join(
            result.get("likely_diagnostic_causes_if_failed", []))
        raise DayZeroContractError(
            "day-zero initial-condition/convention gate failed; no day "
            f"5/10/15 reference metrics may be reported. Likely causes: {causes}")


def canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        strict_jsonable(payload), allow_nan=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_stage_signature(stage_dir: os.PathLike[str] | str,
                             expected_signature: str) -> str:
    """Return ``absent``, ``incomplete``, or ``complete``; reject stale data."""
    stage = pathlib.Path(stage_dir)
    if not stage.exists():
        return "absent"
    signature_path = stage / "STAGE_SIGNATURE.json"
    if not signature_path.is_file():
        if any(stage.iterdir()):
            raise StageSignatureError(
                f"existing stage has no signature and will not be reused: "
                f"{stage}")
        return "absent"
    record = load_strict_json(signature_path)
    actual = record.get("signature") if isinstance(record, Mapping) else None
    if actual != expected_signature:
        raise StageSignatureError(
            f"stale stage signature at {stage}: expected "
            f"{expected_signature}, got {actual}")
    complete_path = stage / "COMPLETE.json"
    if not complete_path.is_file():
        return "incomplete"
    complete = load_strict_json(complete_path)
    if not isinstance(complete, Mapping) or (
            complete.get("stage_signature") != expected_signature):
        raise StageSignatureError(
            f"completion sentinel does not match stage signature at {stage}")
    output_hashes = complete.get("output_hashes")
    if not isinstance(output_hashes, Mapping):
        raise StageSignatureError(
            f"completion sentinel has no output hash inventory at {stage}")
    for relative_name, expected_hash in output_hashes.items():
        candidate = stage / relative_name
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise StageSignatureError(
                f"completed stage output failed integrity verification: "
                f"{candidate}")
    return "complete"


def hash_file_inventory(root: os.PathLike[str] | str, *,
                        exclude: Sequence[str] = ("COMPLETE.json",)
                        ) -> dict[str, str]:
    directory = pathlib.Path(root)
    excluded = set(exclude)
    result = {}
    for path in sorted(item for item in directory.rglob("*")
                       if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative not in excluded:
            result[relative] = sha256_file(path)
    return result


NORM_DEFINITIONS = {
    "scalar_height": {
        "L1": "integral(abs(h-h_ref)) / integral(abs(h_ref))",
        "L2": "sqrt(integral((h-h_ref)^2) / integral(h_ref^2))",
        "Linf": "max(abs(h-h_ref)) / max(abs(h_ref))",
    },
    "vector_velocity": {
        "error_magnitude": "sqrt((u-u_ref)^2 + (v-v_ref)^2)",
        "reference_magnitude": "sqrt(u_ref^2 + v_ref^2)",
        "L1": "integral(error_magnitude) / integral(reference_magnitude)",
        "L2": (
            "sqrt(integral(error_magnitude^2) / "
            "integral(reference_magnitude^2))"),
        "Linf": "max(error_magnitude) / max(reference_magnitude)",
    },
    "undefined_normalization": (
        "strict JSON null with status="
        "'undefined_zero_reference_denominator'; never NaN or infinity"),
}

QUADRATURE_DEFINITION = {
    "formula": "sum_lat sum_lon f(lat,lon) * latitude_weight(lat) * delta_lon",
    "latitude_weights": "frozen reference Gauss-Legendre weights; sum=2",
    "longitude_spacing": "2*pi/nlon, included explicitly",
    "sphere_solid_angle": "4*pi",
    "radius_factor": (
        "not included because it cancels in normalized ratios and weighted "
        "means; all compared absolute field units are unchanged"),
}

AEOLUS_HEIGHT_AUDIT = {
    "prognostic_field": (
        "phi: perturbation fluid-layer geopotential about Phi0=g*H"),
    "base_field": "Phi0: mean fluid-layer geopotential g*H",
    "topography_field": (
        "phi_s: fixed surface geopotential, stored separately from the "
        "prognostic state"),
    "free_surface_height_formula": "(Phi0 + phi + phi_s) / gravity",
    "fluid_layer_thickness_formula": "(Phi0 + phi) / gravity",
    "positivity_diagnostic_field": "true fluid-layer thickness",
    "source_evidence": [
        "src/planetary_sandbox/physics/shallow_water.py module equations and "
        "ShallowWaterModel.characteristic_fields",
        "src/planetary_sandbox/run/swe/initial_conditions.py::_williamson5",
        "src/planetary_sandbox/run/swe/diagnostics.py h_min_m definition",
        "tests/test_williamson5.py free-surface construction regression",
    ],
}
