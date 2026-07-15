"""aeolus recompile / psx-recompile behavior without real CUDA work."""
from __future__ import annotations

import pathlib
import sys
import types


def _fake_cupy():
    pool = types.SimpleNamespace(free_all_blocks=lambda: None)
    return types.SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool)


def test_recompile_clears_cache_with_ascii_output(
        tmp_path, monkeypatch, capsys):
    from planetary_sandbox.cli import clear_cache

    cache_dir = tmp_path / ".cupy" / "kernel_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "kernel.cubin").write_bytes(b"x")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())

    rc = clear_cache.run(clear_cache.build_parser().parse_args(["--skip-verify"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert not cache_dir.exists()
    assert "[ok]" in out
    out.encode("cp1252")
    assert out.isascii()


def test_recompile_friendly_error_without_cupy(monkeypatch, capsys):
    from planetary_sandbox.cli import clear_cache

    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = clear_cache.run(clear_cache.build_parser().parse_args([]))
    out = capsys.readouterr().out
    assert rc == 1
    assert "CuPy is unavailable" in out
