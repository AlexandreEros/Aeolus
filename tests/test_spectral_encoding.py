"""Contracts for phase-aware complex spherical-harmonic rendering."""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.colors import hsv_to_rgb

from planetary_sandbox.viz import (FigureFrame, FigureTimeline,
                                    NormalizationKind, NormalizationPolicy,
                                    PHASE_DOMAIN, SphericalHarmonicField,
                                    SpectralCoefficientMapSpec,
                                    SpectralEncoding, phase_hue,
                                    phase_magnitude_hsv,
                                    relative_magnitude_db)
from planetary_sandbox.viz.matplotlib_renderer import MatplotlibRenderer
from planetary_sandbox.viz.specs import FigureSpec, PanelPlacement


def _fixed_policy() -> NormalizationPolicy:
    return NormalizationPolicy.logarithmic_magnitude(1.0e-6, 1.0)


def _spectral_figure(field, time_index=0, *, normalization=None,
                     encoding="phase-magnitude") -> FigureSpec:
    return FigureSpec(
        panels=(PanelPlacement(SpectralCoefficientMapSpec(
            field, "Complex coefficients", time_index=time_index,
            normalization=normalization or _fixed_policy(),
            normalization_group="coefficients", encoding=encoding), 0, 0),),
        rows=1, columns=1, size_inches=(4.0, 3.0), dpi=50)


def test_phase_and_magnitude_have_independent_backend_neutral_channels():
    coefficients = np.array([
        [1.0 + 0.0j, 1.0j],
        [1.0e-2 + 0.0j, 0.0j],
    ])
    hsv = phase_magnitude_hsv(coefficients, _fixed_policy())
    rgb = hsv_to_rgb(hsv)

    # Equal magnitudes with different arguments retain different cyclic hues.
    assert hsv[0, 0, 0] != hsv[0, 1, 0]
    assert hsv[0, 0, 1] == hsv[0, 1, 1]
    assert not np.allclose(rgb[0, 0], rgb[0, 1])
    # Equal phases with different magnitudes change saturation, not hue/value.
    assert hsv[0, 0, 0] == hsv[1, 0, 0]
    assert hsv[0, 0, 1] > hsv[1, 0, 1] > 0.0
    assert hsv[0, 0, 2] == hsv[1, 0, 2] == 1.0
    # A valid exact zero has zero saturation and is therefore white.
    np.testing.assert_allclose(rgb[1, 1], (1.0, 1.0, 1.0))


def test_relative_amplitude_decibels_map_to_the_default_saturation_floor():
    magnitudes = np.array([1.0, 0.1, 0.01, 0.001])
    db = relative_magnitude_db(magnitudes, _fixed_policy())
    hsv = phase_magnitude_hsv(
        magnitudes.astype(np.complex128), _fixed_policy())

    np.testing.assert_allclose(db, (0.0, -20.0, -40.0, -60.0))
    np.testing.assert_allclose(hsv[..., 1], (1.0, 2 / 3, 1 / 3, 0.0))


def test_phase_domain_is_fixed_and_wraps_cyclically_without_legacy_sign_flip():
    epsilon = 1.0e-8
    coefficients = np.exp(1j * np.array([
        -np.pi + epsilon, np.pi - epsilon, 0.0]))
    hues = phase_hue(coefficients)
    colors = hsv_to_rgb(np.stack(
        (hues, np.ones_like(hues), np.ones_like(hues)), axis=-1))

    assert PHASE_DOMAIN == (-np.pi, np.pi)
    assert hues[0] < 1.0e-6
    assert hues[1] > 1.0 - 1.0e-6
    np.testing.assert_allclose(colors[0], colors[1], atol=1.0e-7)
    # arg(+1) is zero, hence the midpoint hue; angle(-C) would map elsewhere.
    assert hues[2] == 0.5


def test_all_zero_and_extremely_small_fields_have_safe_log_fallbacks(
        tmp_path):
    zero = np.zeros((4, 4), dtype=np.complex128)
    field = SphericalHarmonicField(zero, "zero coefficients", "1")
    policy = NormalizationPolicy.logarithmic_magnitude()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hsv = phase_magnitude_hsv(zero, policy, valid_mask=field.valid_mask)
        output = MatplotlibRenderer().render_spectral_coefficient_map(
            SpectralCoefficientMapSpec(field, "All-zero coefficients"),
            tmp_path / "all-zero.png", dpi=50)

    assert output.stat().st_size > 0
    assert caught == []
    np.testing.assert_allclose(hsv[..., 1], 0.0)
    np.testing.assert_allclose(hsv_to_rgb(hsv), 1.0)

    one_nonzero = zero.copy()
    one_nonzero[3, 2] = 1.0e-30j
    one_hsv = phase_magnitude_hsv(
        one_nonzero, policy, valid_mask=field.valid_mask)
    assert np.isfinite(one_hsv).all()
    assert one_hsv[3, 2, 1] == 1.0

    smallest = np.nextafter(0.0, 1.0)
    resolved = policy.resolve(np.array([0.0, smallest]))
    assert resolved.kind is NormalizationKind.LOG_MAGNITUDE
    assert 0.0 < resolved.vmin < resolved.vmax
    tiny_hsv = phase_magnitude_hsv(
        np.array([[smallest + 0.0j]]), policy)
    assert np.isfinite(tiny_hsv).all()


def test_renderer_distinguishes_invalid_triangle_from_valid_zero(
        tmp_path, monkeypatch):
    coefficients = np.zeros((4, 4), dtype=np.complex128)
    coefficients[3, 2] = np.exp(0.25j)
    coefficients[0, 1] = 1.0e9  # Invalid and excluded from the reference.
    field = SphericalHarmonicField(coefficients, "coefficients", "1")

    from matplotlib.axes import Axes
    original = Axes.imshow
    captured = []
    legend_images = []
    labels = []
    original_set_xlabel = Axes.set_xlabel

    def recording_imshow(self, data, *args, **kwargs):
        array = np.asarray(data)
        if array.shape == (4, 4, 3):
            captured.append((array.copy(), kwargs.copy()))
        if array.shape == (1, 512, 3):
            legend_images.append(array.copy())
        return original(self, data, *args, **kwargs)

    def recording_set_xlabel(self, label, *args, **kwargs):
        labels.append(label)
        return original_set_xlabel(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    monkeypatch.setattr(Axes, "set_xlabel", recording_set_xlabel)
    MatplotlibRenderer().render_spectral_coefficient_map(
        SpectralCoefficientMapSpec(
            field, "Triangular coefficients"),
        tmp_path / "triangle.png", dpi=50)

    image, arguments = captured[0]
    np.testing.assert_allclose(image[0, 0], (1.0, 1.0, 1.0))
    np.testing.assert_allclose(image[0, 1], (0.85, 0.85, 0.85))
    assert not np.allclose(image[3, 2], image[0, 0])
    assert arguments["extent"] == (-0.5, 3.5, -0.5, 3.5)
    assert len(legend_images) == 1
    assert "Hue = phase; saturation = relative magnitude [-60, 0] dB" in labels


def test_timeline_and_overview_share_magnitude_normalization_and_show_decay():
    coefficients = np.zeros((2, 5, 5), dtype=np.complex128)
    coefficients[0, 4, 4] = np.exp(0.5j)
    coefficients[1, 4, 4] = 1.0e-6 * np.exp(0.5j)
    field = SphericalHarmonicField(
        coefficients, "decaying coefficients", "1", times=np.array([0.0, 9.0]))
    timeline = FigureTimeline((
        FigureFrame(0.0, _spectral_figure(
            field, 0, normalization=NormalizationPolicy.logarithmic_magnitude())),
        FigureFrame(9.0, _spectral_figure(
            field, 1, normalization=NormalizationPolicy.logarithmic_magnitude())),
    ), filename_prefix="spectral")

    resolved = timeline.resolve_normalizations()
    frame_panels = [frame.specification.panels[0].panel
                    for frame in resolved.frames]
    limits = {(panel.normalization.vmin, panel.normalization.vmax)
              for panel in frame_panels}
    assert len(limits) == 1
    vmin, vmax = next(iter(limits))
    assert np.isclose(vmin, 1.0e-6) and vmax == 1.0

    overview = timeline.overview_specification()
    overview_panels = [placement.panel for placement in overview.panels
                       if isinstance(placement.panel,
                                     SpectralCoefficientMapSpec)]
    assert {(panel.normalization.vmin, panel.normalization.vmax)
            for panel in overview_panels} == limits

    early = phase_magnitude_hsv(
        coefficients[0], frame_panels[0].normalization,
        valid_mask=field.valid_mask)
    late = phase_magnitude_hsv(
        coefficients[1], frame_panels[1].normalization,
        valid_mask=field.valid_mask)
    assert early[4, 4, 1] == 1.0
    assert late[4, 4, 1] == 0.0
    per_frame_late = phase_magnitude_hsv(
        coefficients[1], NormalizationPolicy.logarithmic_magnitude(),
        valid_mask=field.valid_mask)
    assert per_frame_late[4, 4, 1] == 1.0


def test_renderer_keeps_complete_degree_order_range_without_power_cropping(
        tmp_path, monkeypatch):
    coefficients = np.zeros((2, 6, 6), dtype=np.complex128)
    coefficients[0, 0, 0] = 1.0
    coefficients[0, 5, 5] = 1.0e-2j
    coefficients[1, 0, 0] = 1.0
    coefficients[1, 5, 5] = -1.0e-2j
    field = SphericalHarmonicField(coefficients, "full range", "1")

    from matplotlib.axes import Axes
    original = Axes.imshow
    captured = []

    def recording_imshow(self, data, *args, **kwargs):
        array = np.asarray(data)
        if array.shape == (6, 6, 3):
            captured.append((array.copy(), kwargs["extent"]))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    renderer = MatplotlibRenderer()
    for index in range(2):
        renderer.render_spectral_coefficient_map(
            SpectralCoefficientMapSpec(
                field, f"Frame {index}", time_index=index,
                normalization=_fixed_policy()),
            tmp_path / f"frame-{index}.png", dpi=50)

    assert [image.shape for image, _ in captured] == [(6, 6, 3), (6, 6, 3)]
    assert {extent for _, extent in captured} == {(-0.5, 5.5, -0.5, 5.5)}
    assert all(not np.allclose(image[5, 5], (1.0, 1.0, 1.0))
               for image, _ in captured)


def test_magnitude_mode_and_convention_metadata_remain_explicit(tmp_path):
    coefficients = np.zeros((3, 3), dtype=np.complex128)
    coefficients[2, 1] = 1.0j
    field = SphericalHarmonicField(
        coefficients, "coefficients", "s^-1",
        normalization="orthonormal-complex-m>=0-real-field",
        longitude_origin_radians=0.25)
    phase_spec = SpectralCoefficientMapSpec(field, "Phase coefficients")
    magnitude_spec = SpectralCoefficientMapSpec(
        field, "Magnitude coefficients", encoding="magnitude")

    assert phase_spec.encoding is SpectralEncoding.PHASE_MAGNITUDE
    assert phase_spec.encoding_label == (
        "Hue = phase; saturation = relative magnitude [-60, 0] dB")
    assert phase_spec.convention_metadata == {
        "encoding": "phase-magnitude",
        "phase_definition": "arg(C_lm)",
        "phase_domain": "[-pi, pi)",
        "phase_offset_radians": 0.0,
        "magnitude_floor_db": -60.0,
        "coefficient_normalization": "orthonormal-complex-m>=0-real-field",
        "coefficient_layout": "unpacked-l-m-nonnegative",
        "longitude_origin_radians": 0.25,
        "magnitude_mapping": (
            "amplitude dB relative to timeline maximum, mapped to saturation"),
    }
    assert magnitude_spec.encoding is SpectralEncoding.MAGNITUDE
    assert magnitude_spec.encoding_label == "Color = |C_lm|"
    output = MatplotlibRenderer().render_spectral_coefficient_map(
        magnitude_spec, tmp_path / "magnitude.png", dpi=50)
    assert output.stat().st_size > 0
