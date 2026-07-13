"""Fallback-directory behavior for the psx-bve CLI (pure Python, no GPU)."""
from __future__ import annotations

import pathlib

import planetary_sandbox.cli.bve as bve_cli

# Repo root: .../PlanetarySandbox/src/planetary_sandbox/cli/bve.py -> parents[3].
REPO_ROOT = pathlib.Path(bve_cli.__file__).resolve().parents[3]


def _force_probe_failure(monkeypatch):
    """Make the writability probe raise, as an unwritable --out would."""
    orig_write_text = pathlib.Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self.name == ".write_probe":
            raise OSError("simulated read-only filesystem")
        return orig_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", failing_write_text)


def test_fallback_lands_outside_repo_when_out_unwritable(tmp_path, monkeypatch):
    """An unwritable --out must not litter the repository root."""
    _force_probe_failure(monkeypatch)

    base_dir, used_fallback = bve_cli._resolve_writable_base_dir(str(tmp_path / "runs"))

    assert used_fallback is True
    resolved = base_dir.resolve()
    assert resolved.is_dir()
    # The whole point of the fix: the chosen fallback is NOT inside the project tree.
    assert REPO_ROOT != resolved
    assert REPO_ROOT not in resolved.parents
