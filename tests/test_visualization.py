"""Focused tests for backend-neutral fields, specs, and Matplotlib rendering."""
from __future__ import annotations

import json

import matplotlib.image as mpimg
import numpy as np
import pytest

from planetary_sandbox.viz.fields import (ScalarGridField,
                                           SphericalHarmonicField)
from planetary_sandbox.viz.matplotlib_renderer import MatplotlibRenderer
from planetary_sandbox.viz.normalization import (NormalizationKind,
                                                  NormalizationPolicy)
from planetary_sandbox.viz.specs import (ScalarMapSpec,
                                         SpectralCoefficientMapSpec)


LATITUDES = np.array([np.pi / 2.0, 0.0, -np.pi / 2.0])
LONGITUDES = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])


def test_scalar_grid_field_validation_and_time_selection():
    values = np.arange(24.0).reshape(2, 3, 4)
    field = ScalarGridField(
        values, LATITUDES, LONGITUDES, "temperature", "K",
        times=np.array([0.0, 60.0]))

    assert field.state_count == 2
    np.testing.assert_array_equal(field.values_at(-1), values[1])
    selected = field.select_time(1)
    assert selected.values.shape == (3, 4)
    assert selected.times.tolist() == [60.0]

    with pytest.raises(ValueError, match="coordinates imply"):
        ScalarGridField(values, LATITUDES[:2], LONGITUDES, "bad", "K")
    with pytest.raises(ValueError, match="north-to-south"):
        ScalarGridField(values[0], LATITUDES[::-1], LONGITUDES, "bad", "K")
    with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
        ScalarGridField(
            values[0, :, :3], LATITUDES,
            np.array([0.0, np.pi, 2.0 * np.pi]), "bad", "K")
    with pytest.raises(IndexError, match="out of range"):
        field.select_time(2)


def test_spherical_harmonic_field_validation_and_time_selection():
    coefficients = np.zeros((2, 4, 4), dtype=np.complex128)
    coefficients[1, 3, 2] = 2.0 + 3.0j
    field = SphericalHarmonicField(
        coefficients, "vorticity", "s^-1", times=np.array([0.0, 10.0]),
        normalization="orthonormal-complex-m>=0-real-field")

    assert field.l_max == 3 and field.state_count == 2
    assert field.valid_mask[3, 2]
    assert not field.valid_mask[2, 3]
    selected = field.select_time(-1)
    assert selected.coefficients.shape == (4, 4)
    assert selected.coefficients[3, 2] == 2.0 + 3.0j
    assert selected.times.tolist() == [10.0]

    with pytest.raises(TypeError, match="complex-valued"):
        SphericalHarmonicField(np.zeros((4, 4)), "bad", "1")
    with pytest.raises(ValueError, match="equal degree/order"):
        SphericalHarmonicField(
            np.zeros((3, 4), dtype=np.complex128), "bad", "1")


def test_normalization_signed_constant_nearly_constant_fixed_and_logarithmic():
    signed = NormalizationPolicy.symmetric().resolve(np.array([-2.0, 1.0]))
    assert (signed.vmin, signed.vmax) == (-2.0, 2.0)

    zero = NormalizationPolicy.automatic().resolve(np.zeros((2, 2)))
    assert zero.vmin < 0.0 < zero.vmax
    constant = NormalizationPolicy.automatic().resolve(np.full((2, 2), 5.0))
    assert constant.vmin < 5.0 < constant.vmax
    nearly = NormalizationPolicy.automatic().resolve(
        np.array([1.0, 1.0 + 1.0e-14]))
    assert nearly.vmin < 1.0 < nearly.vmax

    fixed = NormalizationPolicy.fixed(-7.0, 9.0).resolve(np.arange(3))
    assert fixed.kind is NormalizationKind.FIXED
    assert (fixed.vmin, fixed.vmax) == (-7.0, 9.0)

    log = NormalizationPolicy.logarithmic_magnitude().resolve(
        np.array([0.0, 1.0e-6, 2.0]))
    assert log.vmin == pytest.approx(1.0e-6)
    assert log.vmax == pytest.approx(2.0)
    shared_log = NormalizationPolicy.logarithmic_magnitude(
        1.0e-5, 3.0).resolve(np.array([1.0e-2]))
    assert (shared_log.vmin, shared_log.vmax) == (1.0e-5, 3.0)
    log_zero = NormalizationPolicy.logarithmic_magnitude().resolve(
        np.zeros(4))
    assert 0.0 < log_zero.vmin < log_zero.vmax


def test_matplotlib_scalar_map_render_and_latitude_orientation(
        tmp_path, monkeypatch):
    # Northern values occupy source row 0; the renderer must reverse once so
    # the southern row is the first imshow row with origin='lower'.
    values = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
        [-10.0, -10.0, -10.0, -10.0],
    ])
    field = ScalarGridField(
        values, LATITUDES, LONGITUDES, "signed field", "1")
    spec = ScalarMapSpec(
        field, "Signed field", normalization=NormalizationPolicy.symmetric(),
        color_policy="signed")

    from matplotlib.axes import Axes
    original = Axes.imshow
    captured = []

    def recording_imshow(self, data, *args, **kwargs):
        captured.append(np.asarray(data))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    output = MatplotlibRenderer().render_scalar_map(
        spec, tmp_path / "scalar.png", dpi=100)

    assert output.stat().st_size > 0
    assert captured[0][0, 0] == -10.0
    assert captured[0][-1, 0] == 10.0
    image = mpimg.imread(output)
    assert image.shape[:2] == (600, 1200)
    assert not list(tmp_path.glob(".*.tmp-*.png"))


def test_spectral_coefficient_diagnostic_masks_invalid_triangle(
        tmp_path, monkeypatch):
    coefficients = np.zeros((4, 4), dtype=np.complex128)
    coefficients[1, 0] = 1.0
    coefficients[3, 2] = 1.0e-4j
    field = SphericalHarmonicField(coefficients, "height", "m")
    spec = SpectralCoefficientMapSpec(field, "Height coefficients")

    from matplotlib.axes import Axes
    original = Axes.imshow
    captured = []

    def recording_imshow(self, data, *args, **kwargs):
        captured.append(data)
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    output = MatplotlibRenderer().render_spectral_coefficient_map(
        spec, tmp_path / "coefficients.png", dpi=100)

    assert output.stat().st_size > 0
    mask = np.ma.getmaskarray(captured[0])
    assert mask[0, 1]  # m > l is invalid
    assert not mask[3, 2]


def test_atomic_renderer_preserves_previous_image_and_removes_partial_temp(
        tmp_path, monkeypatch):
    field = ScalarGridField(
        np.zeros((3, 4)), LATITUDES, LONGITUDES, "constant", "1")
    specification = ScalarMapSpec(field, "Constant")
    output = tmp_path / "summary.png"
    output.write_bytes(b"previous-complete-image")

    from matplotlib.figure import Figure

    def fail_after_partial_write(self, path, *args, **kwargs):
        path.write_bytes(b"partial")
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(Figure, "savefig", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="encoder failure"):
        MatplotlibRenderer().render_scalar_map(specification, output)

    assert output.read_bytes() == b"previous-complete-image"
    assert not list(tmp_path.glob(".*.tmp-*.png"))


class _FakeTransform:
    def inv_transform(self, coefficients):
        marker = int(round(float(np.real(coefficients[1, 0]))))
        latitude_profile = np.linspace(1.0, -1.0, 91)[:, None]
        longitude_profile = np.cos(np.linspace(0.0, 2.0 * np.pi, 181,
                                               endpoint=False))[None, :]
        return marker * latitude_profile * longitude_profile


class _FakeSWEModel:
    sh = _FakeTransform()
    grid = object()  # values already have the established view-grid shape
    gravity = 10.0


def _write_fake_swe_artifacts(path):
    coefficients = np.zeros((2, 3, 3, 3), dtype=np.complex128)
    coefficients[-1, 0, 1, 0] = 1.0
    coefficients[-1, 1, 1, 0] = 2.0
    coefficients[-1, 2, 1, 0] = 3.0
    np.save(path / "swe_coeffs.npy", coefficients)
    np.save(path / "swe_snapshot_times.npy", np.array([0.0, 3600.0]))


def test_swe_summary_titles_units_shape_and_nonempty_image(tmp_path):
    from planetary_sandbox.run.swe.visualization import (
        build_swe_summary_spec, render_swe_summary)

    _write_fake_swe_artifacts(tmp_path)
    specification = build_swe_summary_spec(_FakeSWEModel(), tmp_path)
    scalar_specs = [placement.panel for placement in specification.panels]
    assert [panel.title for panel in scalar_specs] == [
        "Layer thickness anomaly", "Relative vorticity",
        "Horizontal divergence"]
    assert [panel.display_units for panel in scalar_specs] == [
        "m", "s^-1", "s^-1"]
    assert all(panel.normalization.kind is NormalizationKind.SYMMETRIC
               for panel in scalar_specs)
    # phi marker 3 is converted to thickness using g=10.
    assert np.max(np.abs(scalar_specs[0].field.values)) == pytest.approx(0.3)
    assert scalar_specs[0].field.latitudes[0] > scalar_specs[0].field.latitudes[-1]
    initial_specification = build_swe_summary_spec(
        _FakeSWEModel(), tmp_path, time_index=0)
    assert all(np.allclose(placement.panel.field.values, 0.0)
               for placement in initial_specification.panels)

    output = render_swe_summary(_FakeSWEModel(), tmp_path)
    image = mpimg.imread(output)
    assert output.stat().st_size > 0
    assert image.shape[:2] == (1200, 3600)


def test_swe_visualization_failure_prevents_completion_and_publication(
        tmp_path, monkeypatch):
    from planetary_sandbox.cli import swe
    from planetary_sandbox.run.swe.config import SWERunConfig
    from planetary_sandbox.run.swe.visualization import render_swe_summary

    class FailingRenderer:
        def render_figure(self, specification, output_path, *, metadata=None):
            raise RuntimeError("synthetic SWE visualization failure")

    def solver(cfg, run_dir, run_config):
        _write_fake_swe_artifacts(run_dir.path)
        render_swe_summary(
            _FakeSWEModel(), run_dir.path, renderer=FailingRenderer())

    monkeypatch.setattr(swe, "_execute_solver", solver)
    cfg = SWERunConfig.resolve({"out": str(tmp_path), "n_snapshots": 2})
    with pytest.raises(RuntimeError, match="visualization failure"):
        swe.execute_run(cfg)

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert not (tmp_path / "latest_run.txt").exists()
    assert not (run_dirs[0] / "swe_summary.png").exists()
