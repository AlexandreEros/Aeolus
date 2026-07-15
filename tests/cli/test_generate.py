"""aeolus gen / psx-gen parsing, paths, and dispatch."""
from __future__ import annotations

import pathlib

from planetary_sandbox.cli.main import main


def test_gen_grid_resolution_is_an_int_not_a_list():
    from planetary_sandbox.cli.generate_planet import build_parser

    args = build_parser().parse_args(["--grid-resolution", "4"])
    assert args.grid_resolution == 4


def test_gen_legacy_radius_spelling_still_accepted():
    from planetary_sandbox.cli.generate_planet import build_parser

    args = build_parser().parse_args(["--eq_radius-earth_units", "2.0"])
    assert args.radius_earth_units == 2.0
    args = build_parser().parse_args(["--radius-earth-units", "2.0"])
    assert args.radius_earth_units == 2.0


def test_gen_creates_output_directory(tmp_path, monkeypatch):
    from planetary_sandbox.cli.generate_planet import resolve_output_path

    monkeypatch.chdir(tmp_path)
    out_path = resolve_output_path("planet_summary.png")
    assert out_path == pathlib.Path("out") / "planet_summary.png"
    assert out_path.parent.is_dir()

    absolute = resolve_output_path(str(tmp_path / "deep" / "dir" / "x.png"))
    assert absolute.parent.is_dir()


def test_gen_dispatch_via_mocked_run(monkeypatch):
    import planetary_sandbox.cli.generate_planet as gen_module

    captured = {}
    monkeypatch.setattr(
        gen_module, "run",
        lambda args: captured.setdefault("args", args) and 0 or 0)
    assert main(["gen", "--l-max", "9", "--grid-resolution", "2"]) == 0
    assert captured["args"].l_max == 9
    assert captured["args"].grid_resolution == 2
