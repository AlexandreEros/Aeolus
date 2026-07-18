"""Focused tests for backend-neutral fields, specs, and Matplotlib rendering."""
from __future__ import annotations

import json
import pathlib

import matplotlib.image as mpimg
import numpy as np
import pytest

from planetary_sandbox.viz.fields import (ScalarGridField,
                                           SphericalHarmonicField)
from planetary_sandbox.viz.matplotlib_renderer import MatplotlibRenderer
from planetary_sandbox.viz.normalization import (NormalizationKind,
                                                  NormalizationPolicy)
from planetary_sandbox.viz.specs import (ScalarMapSpec,
                                         SpectralCoefficientMapSpec,
                                         FigureSpec, PanelGroupSpec,
                                         PanelPlacement, StreamlineMapSpec,
                                         TextPanelSpec)
from planetary_sandbox.viz.timeline import (FigureFrame, FigureTimeline,
                                             render_figure_timeline,
                                             render_snapshot_product,
                                             select_representative_frame_indices)


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


def test_streamline_map_is_a_direct_first_class_renderable(tmp_path):
    specification = StreamlineMapSpec(
        LATITUDES, LONGITUDES,
        np.ones((3, 4)), np.zeros((3, 4)),
        radius=1.0, title="Velocity")

    output = MatplotlibRenderer().render_streamline_map(
        specification, tmp_path / "streamlines.png", dpi=50)

    assert output.stat().st_size > 0
    assert mpimg.imread(output).shape[:2] == (300, 600)


def test_matplotlib_renders_generic_panel_group_headings_and_separator(
        tmp_path, monkeypatch):
    field = ScalarGridField(
        np.ones((3, 4)), LATITUDES, LONGITUDES, "field", "1")
    specification = FigureSpec(
        panels=(
            PanelPlacement(ScalarMapSpec(field, "State"), 0, 0),
            PanelPlacement(ScalarMapSpec(field, "Derived"), 0, 1),
        ),
        rows=1, columns=2, size_inches=(6.0, 3.0), dpi=50,
        panel_groups=(
            PanelGroupSpec("First role", 0, 0),
            PanelGroupSpec(
                "Second role", 0, 1, separator_before=True)))

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    original_text = Axes.text
    original_add_artist = Figure.add_artist
    headings = []
    separators = []

    def recording_text(self, x, y, text, *args, **kwargs):
        if text in ("First role", "Second role"):
            headings.append(text)
        return original_text(self, x, y, text, *args, **kwargs)

    def recording_add_artist(self, artist, *args, **kwargs):
        if isinstance(artist, Line2D):
            separators.append(artist)
        return original_add_artist(self, artist, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", recording_text)
    monkeypatch.setattr(Figure, "add_artist", recording_add_artist)
    output = MatplotlibRenderer().render_figure(
        specification, tmp_path / "groups.png")

    assert output.stat().st_size > 0
    assert headings == ["First role", "Second role"]
    assert len(separators) == 1
    assert separators[0].get_alpha() < 0.5


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


def _scalar_frame(values, time_seconds):
    field = ScalarGridField(
        np.asarray(values), LATITUDES, LONGITUDES, "shared", "1")
    panel = ScalarMapSpec(
        field, f"t={time_seconds}",
        normalization=NormalizationPolicy.symmetric(),
        color_policy="signed", normalization_group="shared-field")
    return FigureFrame(time_seconds, FigureSpec(
        panels=(PanelPlacement(panel, 0, 0),), rows=1, columns=1,
        size_inches=(2.0, 1.0), dpi=50))


def test_figure_timeline_resolves_shared_limits_and_time_filenames():
    timeline = FigureTimeline((
        _scalar_frame(np.full((3, 4), -2.0), 0.0),
        _scalar_frame(np.full((3, 4), 7.0), 3600.0),
    ), filename_prefix="experiment")

    resolved = timeline.resolve_normalizations()
    policies = [frame.specification.panels[0].panel.normalization
                for frame in resolved.frames]
    assert all(policy.kind is NormalizationKind.SYMMETRIC
               for policy in policies)
    assert {(policy.vmin, policy.vmax) for policy in policies} == {(-7.0, 7.0)}
    assert [timeline.filename_for(i) for i in range(2)] == [
        "experiment_t0000000000000.000000000s.png",
        "experiment_t0000000003600.000000000s.png",
    ]


def test_figure_timeline_render_failure_keeps_prior_complete_set(tmp_path):
    timeline = FigureTimeline((
        _scalar_frame(np.zeros((3, 4)), 0.0),
        _scalar_frame(np.ones((3, 4)), 1.0),
    ), filename_prefix="transaction")
    previous = []
    for index in range(2):
        path = tmp_path / timeline.filename_for(index)
        path.write_bytes(f"old-{index}".encode())
        previous.append(path)

    class FailsOnSecondFrame:
        calls = 0

        def render_figure(self, specification, output_path, *, metadata=None):
            self.calls += 1
            output_path.write_bytes(b"new")
            if self.calls == 2:
                raise RuntimeError("synthetic frame failure")
            return output_path

    with pytest.raises(RuntimeError, match="frame failure"):
        render_figure_timeline(
            timeline, tmp_path, renderer=FailsOnSecondFrame())

    assert [path.read_bytes() for path in previous] == [b"old-0", b"old-1"]
    assert not list(tmp_path.glob(".transaction.timeline-*"))


def test_representative_frame_selection_handles_empty_short_and_irregular():
    assert select_representative_frame_indices([]) == ()
    for count in range(1, 6):
        assert select_representative_frame_indices(np.arange(count)) == tuple(
            range(count))

    # Targets are [0, 25.25, 50.5, 75.75, 101].  Several targets are much
    # closer to one cluster than the other; the selected result stays
    # chronological and distinct while retaining both endpoints.
    irregular = np.array([0.0, 10.0, 11.0, 12.0, 100.0, 101.0])
    assert select_representative_frame_indices(irregular) == (0, 2, 3, 4, 5)


def test_timeline_overview_uses_full_sequence_normalization():
    # Six frames force a five-frame overview. Frame 3 is not selected, but its
    # outlier must still determine every overview panel's frozen limits.
    values = [1.0, 2.0, 3.0, 99.0, 4.0, 5.0]
    timeline = FigureTimeline(tuple(
        _scalar_frame(np.full((3, 4), value), float(index))
        for index, value in enumerate(values)), filename_prefix="overview")
    assert timeline.representative_indices() == (0, 1, 2, 4, 5)

    overview = timeline.overview_specification()
    scalar_panels = [placement.panel for placement in overview.panels
                     if isinstance(placement.panel, ScalarMapSpec)]
    labels = [placement.panel for placement in overview.panels
              if isinstance(placement.panel, TextPanelSpec)]
    assert len(scalar_panels) == len(labels) == 5
    assert {(panel.normalization.vmin, panel.normalization.vmax)
            for panel in scalar_panels} == {(-99.0, 99.0)}
    assert labels[0].text == "Physical time: 0 s"
    assert labels[-1].text == "Physical time: 5 s"


def test_mixed_panel_timeline_normalizes_all_first_class_panel_families():
    scalar = ScalarGridField(
        np.full((3, 4), 2.0), LATITUDES, LONGITUDES, "scalar", "1")
    coefficients = np.zeros((3, 3), dtype=np.complex128)
    coefficients[2, 1] = 8.0j
    spectral = SphericalHarmonicField(coefficients, "spectral", "1")
    figure = FigureSpec(
        panels=(
            PanelPlacement(ScalarMapSpec(
                scalar, "Scalar", normalization=NormalizationPolicy.symmetric(),
                normalization_group="mixed-scalar"), 0, 0),
            PanelPlacement(StreamlineMapSpec(
                LATITUDES, LONGITUDES,
                np.full((3, 4), 3.0), np.full((3, 4), 4.0),
                radius=1.0, title="Flow",
                normalization_group="mixed-flow"), 0, 1),
            PanelPlacement(SpectralCoefficientMapSpec(
                spectral, "Coefficients",
                normalization_group="mixed-spectral"), 0, 2),
        ), rows=1, columns=3, size_inches=(6.0, 2.0), dpi=50)
    timeline = FigureTimeline(
        (FigureFrame(0.0, figure),), filename_prefix="mixed")

    resolved_panels = [placement.panel for placement in
                       timeline.resolve_normalizations().frames[0].specification.panels]
    assert (resolved_panels[0].normalization.vmin,
            resolved_panels[0].normalization.vmax) == (-2.0, 2.0)
    assert (resolved_panels[1].normalization.vmin < 5.0 <
            resolved_panels[1].normalization.vmax)
    assert (resolved_panels[2].normalization.vmin < 8.0 <
            resolved_panels[2].normalization.vmax)
    assert (resolved_panels[2].normalization.kind is
            NormalizationKind.LOG_MAGNITUDE)


class _ByteRenderer:
    def __init__(self, *, fail_on_representation=None):
        self.fail_on_representation = fail_on_representation

    def render_figure(self, specification, output_path, *, metadata=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"new")
        if (self.fail_on_representation is not None and
                self.fail_on_representation in output_path.parts):
            raise RuntimeError("synthetic representation failure")
        return output_path


def _two_representation_timelines():
    frames = (
        _scalar_frame(np.zeros((3, 4)), 0.0),
        _scalar_frame(np.ones((3, 4)), 3600.0),
    )
    return {
        "physical": FigureTimeline(frames, filename_prefix="physical-model"),
        "spectral": FigureTimeline(frames, filename_prefix="spectral-model"),
    }


def test_snapshot_product_layout_and_rerender_removes_stale_frames(tmp_path):
    stale = tmp_path / "snapshots" / "physical"
    stale.mkdir(parents=True)
    (stale / "t000123s.png").write_bytes(b"stale")
    (stale / "timeline.png").write_bytes(b"stale")

    outputs = render_snapshot_product(
        _two_representation_timelines(), tmp_path, renderer=_ByteRenderer())

    for representation in ("physical", "spectral"):
        directory = tmp_path / "snapshots" / representation
        assert {path.name for path in directory.iterdir()} == {
            "t000000s.png", "t003600s.png", "timeline.png"}
        assert all(path.suffix == ".png" for path in directory.iterdir())
        assert outputs[representation][-1] == directory / "timeline.png"
    assert not list(tmp_path.glob(".snapshots.product-*"))


def test_snapshot_product_failure_preserves_previous_complete_product(tmp_path):
    for representation in ("physical", "spectral"):
        directory = tmp_path / "snapshots" / representation
        directory.mkdir(parents=True)
        (directory / "timeline.png").write_bytes(
            f"old-{representation}".encode())
        (directory / "t000000s.png").write_bytes(
            f"old-frame-{representation}".encode())

    with pytest.raises(RuntimeError, match="representation failure"):
        render_snapshot_product(
            _two_representation_timelines(), tmp_path,
            renderer=_ByteRenderer(fail_on_representation="spectral"))

    assert (tmp_path / "snapshots" / "physical" / "timeline.png").read_bytes() == (
        b"old-physical")
    assert (tmp_path / "snapshots" / "spectral" / "timeline.png").read_bytes() == (
        b"old-spectral")
    assert not list(tmp_path.glob(".snapshots.product-*"))


def test_snapshot_product_publication_failure_rolls_back_directory(
        tmp_path, monkeypatch):
    old = tmp_path / "snapshots" / "physical" / "timeline.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old-product")

    from planetary_sandbox.viz import timeline as timeline_module
    real_replace = timeline_module.os.replace

    def fail_final_directory_publish(source, destination):
        source = pathlib.Path(source)
        if (source.name == "snapshots" and
                source.parent.name.startswith(".snapshots.product-")):
            raise OSError("synthetic directory publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        timeline_module.os, "replace", fail_final_directory_publish)
    with pytest.raises(OSError, match="directory publication failure"):
        render_snapshot_product(
            _two_representation_timelines(), tmp_path,
            renderer=_ByteRenderer())

    assert old.read_bytes() == b"old-product"
    assert not list(tmp_path.glob(".snapshots.product-*"))


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
    R = 1.0

    @staticmethod
    def wind_on_state_grid(state):
        u_marker = float(np.real(state.coeffs[0, 1, 0]))
        v_marker = float(np.real(state.coeffs[1, 1, 0]))
        shape = (91, 181)
        return np.full(shape, u_marker), np.full(shape, v_marker)


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
        "Relative vorticity", "Horizontal divergence",
        "Layer-thickness anomaly h' = Phi'/g"]
    assert [panel.display_units for panel in scalar_specs] == [
        "s^-1", "s^-1", "m"]
    assert all(panel.normalization.kind is NormalizationKind.SYMMETRIC
               for panel in scalar_specs)
    # phi marker 3 is converted to thickness using g=10.
    assert np.max(np.abs(scalar_specs[2].field.values)) == pytest.approx(0.3)
    assert scalar_specs[0].field.latitudes[0] > scalar_specs[0].field.latitudes[-1]
    initial_specification = build_swe_summary_spec(
        _FakeSWEModel(), tmp_path, time_index=0)
    assert all(np.allclose(placement.panel.field.values, 0.0)
               for placement in initial_specification.panels)

    output = render_swe_summary(_FakeSWEModel(), tmp_path)
    image = mpimg.imread(output)
    assert output.stat().st_size > 0
    assert image.shape[:2] == (1200, 3600)


def test_swe_snapshot_timeline_uses_persisted_times_and_shared_limits(tmp_path):
    from planetary_sandbox.run.swe.visualization import (
        build_swe_snapshot_timeline, build_swe_snapshot_timelines,
        render_swe_snapshots)

    _write_fake_swe_artifacts(tmp_path)
    timeline = build_swe_snapshot_timeline(
        _FakeSWEModel(), tmp_path, scenario="gravity_wave")
    assert timeline.times_seconds.tolist() == [0.0, 3600.0]
    assert timeline.filename_for(1) == (
        "gravity_wave_t0000000003600.000000000s.png")

    resolved = timeline.resolve_normalizations()
    thickness_policies = [
        frame.specification.panels[2].panel.normalization
        for frame in resolved.frames]
    assert {(policy.vmin, policy.vmax) for policy in thickness_policies} == {
        (-0.3, 0.3)}
    physical_specification = timeline.frames[0].specification
    assert [group.title for group in physical_specification.panel_groups] == [
        "Prognostic state", "Diagnostic fields"]
    assert isinstance(
        physical_specification.panels[-1].panel, StreamlineMapSpec)
    assert "Phi'/g" in physical_specification.panels[2].panel.title

    timelines = build_swe_snapshot_timelines(
        _FakeSWEModel(), tmp_path, scenario="gravity_wave")
    assert tuple(timelines) == ("physical", "spectral")
    np.testing.assert_array_equal(
        timelines["physical"].times_seconds,
        timelines["spectral"].times_seconds)
    assert all(isinstance(placement.panel, SpectralCoefficientMapSpec)
               for placement in
               timelines["spectral"].frames[0].specification.panels)

    render_swe_snapshots(
        _FakeSWEModel(), tmp_path, scenario="gravity_wave",
        renderer=_ByteRenderer())
    assert (tmp_path / "snapshots" / "physical" / "timeline.png").exists()
    assert (tmp_path / "snapshots" / "spectral" / "timeline.png").exists()


class _FakeBVETransform:
    def inv_transform(self, coefficients):
        marker = float(np.real(coefficients[1, 0]))
        return np.full((91, 181), marker)


class _FakeBVEOperators:
    def __init__(self, transform):
        self.transform = transform

    @staticmethod
    def inv_laplacian(coefficients):
        return coefficients * 2.0

    def velocity_from_streamfunction(self, coefficients):
        values = self.transform.inv_transform(coefficients)
        return values, -values


class _FakeBVEGrid:
    cell_areas = np.ones((91, 181))


class _FakeBVEParams:
    equatorial_radius = 1.0


class _FakeBVEPlanet:
    sh = _FakeBVETransform()
    so = _FakeBVEOperators(sh)
    grid = _FakeBVEGrid()
    params = _FakeBVEParams()


def test_bve_snapshot_timeline_reloads_persisted_artifacts(tmp_path):
    from planetary_sandbox.run.bve.visualization import (
        BVE_SNAPSHOT_TIMES_FILENAME, build_bve_snapshot_timeline,
        build_bve_snapshot_timelines, render_bve_snapshots)

    coefficients = np.zeros((2, 3, 3), dtype=np.complex128)
    coefficients[0, 1, 0] = 1.0
    coefficients[1, 1, 0] = 3.0
    np.save(tmp_path / "vorticity_coeffs.npy", coefficients)
    np.save(tmp_path / "vorticity_grid.npy", np.stack((
        np.full((91, 181), 1.0), np.full((91, 181), 3.0))))
    np.save(tmp_path / BVE_SNAPSHOT_TIMES_FILENAME,
            np.array([0.0, 12.5]))

    timeline = build_bve_snapshot_timeline(
        _FakeBVEPlanet(), tmp_path, scenario="rh4")
    assert timeline.times_seconds.tolist() == [0.0, 12.5]
    assert all(len(frame.specification.panels) == 3
               for frame in timeline.frames)
    assert [group.title for group in
            timeline.frames[0].specification.panel_groups] == [
                "Prognostic state", "Diagnostic fields"]
    assert isinstance(
        timeline.frames[0].specification.panels[-1].panel,
        StreamlineMapSpec)
    policies = [frame.specification.panels[0].panel.normalization
                for frame in timeline.resolve_normalizations().frames]
    assert {(policy.vmin, policy.vmax) for policy in policies} == {(-3.0, 3.0)}

    timelines = build_bve_snapshot_timelines(
        _FakeBVEPlanet(), tmp_path, scenario="rh4")
    np.testing.assert_array_equal(
        timelines["physical"].times_seconds,
        timelines["spectral"].times_seconds)
    assert isinstance(
        timelines["spectral"].frames[0].specification.panels[0].panel,
        SpectralCoefficientMapSpec)

    render_bve_snapshots(
        _FakeBVEPlanet(), tmp_path, scenario="rh4", renderer=_ByteRenderer())
    assert (tmp_path / "snapshots" / "physical" / "t000000s.png").exists()
    assert (tmp_path / "snapshots" / "spectral" /
            "t000012.500000000s.png").exists()


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
