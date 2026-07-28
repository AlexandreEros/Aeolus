"""CPU-only contracts for Notebook B's immutable-reference workflow."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from planetary_sandbox.validation.w5_mri import (
    ARTIFACT_SPECS,
    DayZeroContractError,
    GridContractError,
    ReferenceContractError,
    StageSignatureError,
    atomic_write_json,
    construct_aeolus_heights,
    determine_grid_permutation,
    evaluate_day0_contract,
    require_day0_contract,
    scalar_error_metrics,
    sha256_file,
    validate_reference_artifact,
    validate_reference_manifest,
    validate_stage_signature,
    vector_error_metrics,
    verify_reference_package,
)


def _valid_manifest():
    hashes = {
        name: f"hash-for-{name}" for name in (
            "manifest.json", "t42_64x128.npz", "t63_96x192.npz")}
    manifest = {
        "schema_version": "mri-w5-reference-v1",
        "validation": {
            "all_reference_preparation_checks_passed": True,
        },
        "height_interpretation": {
            "resolved_as": "free_surface_height",
        },
        "final_artifacts": {},
        "targets": {},
    }
    for filename, spec in ARTIFACT_SPECS.items():
        manifest["final_artifacts"][filename] = {
            "sha256": hashes[filename]}
        manifest["targets"][filename] = {
            "truncation": spec.truncation,
            "nlat": spec.nlat,
            "nlon": spec.nlon,
        }
    return manifest, hashes


def _valid_arrays(spec):
    x, weights = np.polynomial.legendre.leggauss(spec.nlat)
    order = np.argsort(-x)
    latitude = np.arcsin(x[order]).astype(np.float64)
    weights = weights[order].astype(np.float64)
    longitude = np.linspace(
        0.0, 2.0 * np.pi, spec.nlon, endpoint=False, dtype=np.float64)
    shape = (4, spec.nlat, spec.nlon)
    return {
        "time_days": np.asarray([0.0, 5.0, 10.0, 15.0], dtype=np.float64),
        "height": np.ones(shape, dtype=np.float64),
        "u": np.ones(shape, dtype=np.float64),
        "v": np.zeros(shape, dtype=np.float64),
        "latitude": latitude,
        "longitude": longitude,
        "latitude_weights": weights,
    }


def _write_npz(path, arrays):
    np.savez(path, **arrays)
    return path


def test_sha256_verification(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_package_hash_mismatch_fails_before_content_use(tmp_path):
    for name in ("manifest.json", "t42_64x128.npz", "t63_96x192.npz"):
        (tmp_path / name).write_bytes(b"immutable payload")
    expected = {
        "manifest.json": "0" * 64,
        "t42_64x128.npz": "1" * 64,
        "t63_96x192.npz": "2" * 64,
    }
    with pytest.raises(ReferenceContractError, match="SHA-256 mismatch"):
        verify_reference_package(tmp_path, expected_hashes=expected)


def test_schema_rejection():
    manifest, hashes = _valid_manifest()
    manifest["schema_version"] = "future-schema"
    with pytest.raises(ReferenceContractError, match="schema_version"):
        validate_reference_manifest(manifest, expected_hashes=hashes)


def test_manifest_rejects_failed_preparation_and_height_meaning():
    manifest, hashes = _valid_manifest()
    manifest["validation"]["all_reference_preparation_checks_passed"] = False
    with pytest.raises(ReferenceContractError, match="not explicitly"):
        validate_reference_manifest(manifest, expected_hashes=hashes)
    manifest, hashes = _valid_manifest()
    manifest["height_interpretation"]["resolved_as"] = "layer_thickness"
    with pytest.raises(ReferenceContractError, match="free_surface_height"):
        validate_reference_manifest(manifest, expected_hashes=hashes)


def test_artifact_key_and_shape_validation(tmp_path):
    spec = ARTIFACT_SPECS["t42_64x128.npz"]
    arrays = _valid_arrays(spec)
    arrays.pop("v")
    path = _write_npz(tmp_path / "missing-key.npz", arrays)
    with pytest.raises(ReferenceContractError, match="keys mismatch"):
        validate_reference_artifact(path, spec)

    arrays = _valid_arrays(spec)
    arrays["height"] = arrays["height"][:, :-1]
    path = _write_npz(tmp_path / "bad-shape.npz", arrays)
    with pytest.raises(ReferenceContractError, match="shape mismatch"):
        validate_reference_artifact(path, spec)


def test_valid_artifact_is_loaded_read_only(tmp_path):
    spec = ARTIFACT_SPECS["t42_64x128.npz"]
    artifact = validate_reference_artifact(
        _write_npz(tmp_path / "valid.npz", _valid_arrays(spec)), spec)
    assert set(artifact.arrays) == {
        "time_days", "height", "u", "v", "latitude", "longitude",
        "latitude_weights"}
    assert all(not values.flags.writeable
               for values in artifact.arrays.values())


def test_artifact_rejects_nonfinite_and_wrong_dtype(tmp_path):
    spec = ARTIFACT_SPECS["t42_64x128.npz"]
    arrays = _valid_arrays(spec)
    arrays["u"][0, 0, 0] = np.inf
    path = _write_npz(tmp_path / "nonfinite.npz", arrays)
    with pytest.raises(ReferenceContractError, match="non-finite"):
        validate_reference_artifact(path, spec)

    arrays = _valid_arrays(spec)
    arrays["v"] = arrays["v"].astype(np.float32)
    path = _write_npz(tmp_path / "float32.npz", arrays)
    with pytest.raises(ReferenceContractError, match="float64"):
        validate_reference_artifact(path, spec)


def test_target_time_validation(tmp_path):
    spec = ARTIFACT_SPECS["t42_64x128.npz"]
    arrays = _valid_arrays(spec)
    arrays["time_days"][-1] = 14.999999999
    path = _write_npz(tmp_path / "bad-time.npz", arrays)
    with pytest.raises(ReferenceContractError, match="\\[0, 5, 10, 15\\]"):
        validate_reference_artifact(path, spec)


def test_latitude_reversal_detection_and_application():
    reference_lat = np.asarray([0.8, 0.2, -0.2, -0.8])
    native_lat = reference_lat[::-1]
    longitude = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    permutation = determine_grid_permutation(
        native_lat, longitude, reference_lat, longitude,
        native_field_shape=(4, 4))
    assert permutation.latitude_reversed
    field = np.repeat(native_lat[:, None], 4, axis=1)
    assert np.array_equal(permutation.apply(field)[:, 0], reference_lat)


def test_cyclic_longitude_roll_detection_and_application():
    reference_lon = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    native_lon = np.roll(reference_lon, 2)
    latitude = np.asarray([0.5, -0.5])
    permutation = determine_grid_permutation(
        latitude, native_lon, latitude, reference_lon,
        native_field_shape=(2, 4))
    assert permutation.longitude_roll == -2
    field = np.repeat(native_lon[None, :], 2, axis=0)
    assert np.array_equal(permutation.apply(field)[0], reference_lon)


def test_documented_axis_transpose_is_exact_and_reversible():
    latitude = np.asarray([0.5, -0.5])
    longitude = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    native_lat_lon_field = np.arange(8).reshape(2, 4)
    native_lon_lat_field = native_lat_lon_field.T
    permutation = determine_grid_permutation(
        latitude, longitude, latitude, longitude,
        native_field_shape=native_lon_lat_field.shape)
    assert permutation.axes_transposed
    assert np.array_equal(
        permutation.apply(native_lon_lat_field), native_lat_lon_field)


def test_grid_rejects_nonmatching_coordinates():
    latitude = np.asarray([0.5, -0.5])
    longitude = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    with pytest.raises(GridContractError, match="latitude"):
        determine_grid_permutation(
            latitude + np.asarray([0.0, 0.1]), longitude,
            latitude, longitude)
    with pytest.raises(GridContractError, match="longitude"):
        determine_grid_permutation(
            latitude, longitude + np.asarray([0.0, 0.0, 0.1, 0.0]),
            latitude, longitude)


def test_weighted_scalar_norms_exact_equality_and_constant_offset():
    weights = np.asarray([1.0, 1.0])
    reference = np.ones((2, 2))
    equal = scalar_error_metrics(reference, reference, weights, np.pi)
    for metric in equal["normalized"].values():
        assert metric["status"] == "defined"
        assert metric["value"] == 0.0

    offset = scalar_error_metrics(reference + 2.0, reference, weights, np.pi)
    assert offset["normalized"]["L1"]["value"] == pytest.approx(2.0)
    assert offset["normalized"]["L2"]["value"] == pytest.approx(2.0)
    assert offset["normalized"]["Linf"]["value"] == pytest.approx(2.0)
    assert offset["absolute"]["weighted_mean_absolute_error"] == pytest.approx(
        2.0)
    assert offset["absolute"]["weighted_rms_error"] == pytest.approx(2.0)
    assert offset["absolute"]["maximum_absolute_error"] == pytest.approx(2.0)


def test_weighted_scalar_single_point_and_latitude_dependent_errors():
    weights = np.asarray([1.0, 1.0])
    reference = np.ones((2, 2))
    field = reference.copy()
    field[0, 0] += 2.0
    single = scalar_error_metrics(field, reference, weights, np.pi)
    assert single["normalized"]["L1"]["value"] == pytest.approx(0.5)
    assert single["normalized"]["L2"]["value"] == pytest.approx(1.0)
    assert single["normalized"]["Linf"]["value"] == pytest.approx(2.0)

    latitude_weights = np.asarray([0.5, 1.5])
    error = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    dependent = scalar_error_metrics(
        reference + error, reference, latitude_weights, np.pi)
    assert dependent["absolute"][
        "weighted_mean_absolute_error"] == pytest.approx(2.5)
    assert dependent["absolute"]["weighted_rms_error"] == pytest.approx(
        math.sqrt(7.0))


def test_weighted_vector_norms_one_component_only():
    weights = np.asarray([1.0, 1.0])
    u_reference = np.ones((2, 2))
    v_reference = np.zeros((2, 2))
    result = vector_error_metrics(
        u_reference + 2.0, v_reference,
        u_reference, v_reference, weights, np.pi)
    assert result["normalized"]["L1"]["value"] == pytest.approx(2.0)
    assert result["normalized"]["L2"]["value"] == pytest.approx(2.0)
    assert result["normalized"]["Linf"]["value"] == pytest.approx(2.0)
    assert result["absolute_u_component"][
        "weighted_rms_error"] == pytest.approx(2.0)
    assert result["absolute_v_component"][
        "maximum_absolute_error"] == 0.0


def test_undefined_normalizations_use_null_status_not_nonfinite():
    zeros = np.zeros((2, 2))
    weights = np.asarray([1.0, 1.0])
    scalar = scalar_error_metrics(
        np.ones((2, 2)), zeros, weights, np.pi)
    vector = vector_error_metrics(
        np.ones((2, 2)), zeros, zeros, zeros, weights, np.pi)
    for collection in (scalar["normalized"], vector["normalized"]):
        for metric in collection.values():
            assert metric["value"] is None
            assert metric["status"] == "undefined_zero_reference_denominator"
    json.dumps({"scalar": scalar, "vector": vector}, allow_nan=False)


def test_free_surface_height_uses_audited_aeolus_convention():
    phi = np.asarray([[10.0, -10.0]])
    phi_s = np.asarray([[5.0, 20.0]])
    result = construct_aeolus_heights(
        phi, base_geopotential=100.0,
        surface_geopotential=phi_s, gravity=10.0)
    assert np.array_equal(
        result["fluid_layer_thickness_m"], np.asarray([[11.0, 9.0]]))
    assert np.array_equal(
        result["free_surface_height_m"], np.asarray([[11.5, 11.0]]))


def test_day_zero_contract_failure_stops_forecast_metrics():
    zeros = np.zeros((2, 2))
    tolerances = {
        "height_m": {
            "maximum_absolute_error": 0.1,
            "weighted_mean_absolute_error": 0.1,
            "weighted_rms_error": 0.1,
        },
        "velocity_component_m_s-1": {
            "maximum_absolute_error": 0.1,
            "weighted_mean_absolute_error": 0.1,
            "weighted_rms_error": 0.1,
        },
        "wind_speed_m_s-1": {
            "maximum_absolute_error": 0.1,
            "weighted_mean_absolute_error": 0.1,
            "weighted_rms_error": 0.1,
        },
    }
    result = evaluate_day0_contract(
        height=zeros + 1.0, u=zeros, v=zeros,
        height_reference=zeros, u_reference=zeros, v_reference=zeros,
        latitude_weights=np.asarray([1.0, 1.0]),
        longitude_spacing=np.pi, tolerances=tolerances,
        convention_checks={"units_match": True})
    assert not result["passed"]
    with pytest.raises(
            DayZeroContractError, match="no day 5/10/15 reference metrics"):
        require_day0_contract(result)


def test_stale_stage_signature_rejection(tmp_path):
    stage = tmp_path / "t42_64x128"
    stage.mkdir()
    atomic_write_json(stage / "STAGE_SIGNATURE.json", {
        "signature": "old-signature"})
    with pytest.raises(StageSignatureError, match="stale stage signature"):
        validate_stage_signature(stage, "new-signature")
