"""Shallow-water config resolution, runner, and CLI integration tests.

Config-resolution tests are pure CPU (import-light modules only); the
runner/CLI tests need CUDA and use a small Gauss lat-lon configuration.
"""
from __future__ import annotations

import csv
import json
import math

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config resolution (CPU)
# ---------------------------------------------------------------------------

def test_swe_config_defaults():
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({})
    assert cfg.scenario == "williamson2"
    assert cfg.grid == "geodesic"
    assert cfg.snapshot_mode == "count" and cfg.n_snapshots == 5
    assert cfg.gravity == pytest.approx(9.80616)
    assert cfg.mean_depth_m == 3000.0
    assert cfg.day_hours == pytest.approx(23.9345)
    assert cfg.plots == ("diagnostics",)
    times = cfg.snapshot_times_seconds()
    assert len(times) == 5 and times[0] == 0.0 and times[-1] == 86400.0


def test_swe_config_rejects_bad_values():
    from planetary_sandbox.run.swe.config import SWERunConfig

    with pytest.raises(ValueError, match="gravity"):
        SWERunConfig.resolve({"gravity": -1.0})
    with pytest.raises(ValueError, match="mean_depth_m"):
        SWERunConfig.resolve({"mean_depth_m": 0.0})
    with pytest.raises(ValueError, match="scenario"):
        SWERunConfig.resolve({"scenario": "topography"})
    with pytest.raises(ValueError, match="mutually exclusive"):
        SWERunConfig.resolve({"n_snapshots": 3, "dt_snapshots": 60.0})
    with pytest.raises(ValueError, match="duration_days"):
        SWERunConfig.resolve({"duration_days": math.nan})
    with pytest.raises(ValueError, match="unknown explicit"):
        SWERunConfig.resolve({"viscosity": 1.0})


def test_swe_config_dict_feeds_run_id():
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({"grid": "gauss-latlon", "nlat": 16,
                                "nlon": 32, "lmax": 7})
    assert cfg.grid == "latlon"  # alias normalized
    d = cfg.to_run_config_dict()
    assert d["solver"] == "swe"
    run_id = make_run_id(d, commit="deadbeef")
    assert "williamson2" in run_id and run_id.endswith("deadbeef")

    # Locational values must not change the scientific identity.
    d2 = SWERunConfig.resolve({"grid": "gauss-latlon", "nlat": 16,
                               "nlon": 32, "lmax": 7,
                               "out": "elsewhere"}).scientific_config_dict()
    assert d2 == cfg.scientific_config_dict()


def test_swe_cli_help_and_parse_errors(capsys):
    from planetary_sandbox.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--mean-depth" in out and "--gravity" in out

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--scenario", "nonsense"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["run", "swe", "--gravity", "-9.8"])
    assert exc.value.code == 2


def test_swe_interval_run_ids_disambiguate_physics():
    """Audit finding 4: interval-mode SWE runs must not share the hashless
    legacy BVE id format — configs differing only in physics (e.g. gravity)
    must get distinct ids."""
    from datetime import datetime, timezone
    from planetary_sandbox.run.bve.io import make_run_id
    from planetary_sandbox.run.swe.config import SWERunConfig

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = []
    for gravity in (9.80616, 3.71):
        cfg = SWERunConfig.resolve({"gravity": gravity,
                                    "dt_snapshots": 3600.0})
        assert cfg.snapshot_mode == "interval"
        ids.append(make_run_id(cfg.to_run_config_dict(), now=now,
                               commit="deadbeef"))
    assert ids[0] != ids[1]

    # The legacy BVE interval id format is preserved exactly (no hash token).
    bve_interval = {"scenario": "rh4", "day_hours": 24.0, "resolution": 4,
                    "lmax": 21, "dt_snapshots": 21600.0,
                    "snapshot_mode": "interval"}
    parts = make_run_id(bve_interval, now=now, commit="deadbeef").split("_")
    assert parts == ["20260101T000000Z", "rh4", "rot24h", "r4", "l21",
                     "dt6h", "deadbeef"]


# ---------------------------------------------------------------------------
# Runner integration (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_swe_runner_end_to_end(tmp_path):
    import numpy as np
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic
    from planetary_sandbox.run.swe.runner import run_swe

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=23.9345),
        grid_type="latlon", nlat=16, nlon=32, l_max=7)
    model = ShallowWaterModel(planet, mean_depth=2500.0)
    state0 = make_swe_ic("williamson2", model)

    # N=11 deliberately: >10 stored states previously stalled final
    # persistence for minutes via cp.stack on small GPUs (audit finding 3);
    # snapshots are now transferred to host individually.
    t_end_days = 0.02
    schedule = count_snapshot_times(11, t_end_days * 86400.0)
    rc = run_swe(model=model, state0=state0, dt_snapshots=None,
                 t_end_days=t_end_days, out_dir=tmp_path,
                 snapshot_times=schedule, plots=(), snapshot_mode="count")
    assert rc == 0

    coeffs = np.load(tmp_path / "swe_coeffs.npy")
    times = np.load(tmp_path / "swe_snapshot_times.npy")
    assert coeffs.shape == (11, 3, 8, 8)
    assert np.isfinite(coeffs).all()
    assert times.tolist() == schedule
    assert not list(tmp_path.glob("*.png")) and not (tmp_path / "figures").exists()

    with open(tmp_path / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[-1]["time_s"]) == t_end_days * 86400.0  # exact landing
    # Mass exactly conserved; energy drift at round-off for the steady state.
    mass = [float(r["total_mass"]) for r in rows]
    energy = [float(r["total_energy"]) for r in rows]
    assert max(abs(m - mass[0]) for m in mass) <= 1e-12 * abs(mass[0])
    assert max(abs(e - energy[0]) for e in energy) <= 1e-10 * abs(energy[0])
    # Characteristic speed includes the gravity-wave speed sqrt(Phi0+phi),
    # so it must exceed sqrt(Phi0)*0.9 even though the wind is ~40 m/s.
    assert float(rows[0]["max_char_speed_ms"]) > 0.9 * math.sqrt(model.phi0)
    assert float(rows[0]["phi_total_min"]) > 0.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_swe_geodesic_mass_diagnostic_exactly_conserved(tmp_path):
    """Audit finding 5: total_mass is the spectrally computed conserved
    quantity, so it must be exactly constant even on the geodesic backend
    (whose grid quadrature would show spurious ~1e-7 drift)."""
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.physics.shallow_water import ShallowWaterModel
    from planetary_sandbox.run.engine import count_snapshot_times
    from planetary_sandbox.run.swe.initial_conditions import make_swe_ic
    from planetary_sandbox.run.swe.runner import run_swe

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=23.9345),
        grid_type="geodesic", grid_resolution=3, l_max=10)
    model = ShallowWaterModel(planet, mean_depth=2500.0)
    state0 = make_swe_ic("williamson2", model)

    t_end_days = 0.01
    rc = run_swe(model=model, state0=state0, dt_snapshots=None,
                 t_end_days=t_end_days, out_dir=tmp_path,
                 snapshot_times=count_snapshot_times(2, t_end_days * 86400.0),
                 plots=(), snapshot_mode="count")
    assert rc == 0
    with open(tmp_path / "diagnostics" / "timeseries.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 2
    masses = {r["total_mass"] for r in rows}
    assert len(masses) == 1  # bit-identical CSV values, not merely close


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_swe_cli_end_to_end(tmp_path, capsys):
    from planetary_sandbox.cli.main import main

    rc = main(["run", "swe", "--backend", "gauss-latlon", "--nlat", "16",
               "--nlon", "32", "--l-max", "7", "--days", "0.01",
               "--n-snapshots", "1", "--no-plots",
               "--out", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "solver              swe" in out

    pointer = (tmp_path / "runs" / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    run_dir = tmp_path / "runs" / pointer
    manifest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["run_config"]["solver"] == "swe"
    assert "shallow-water" in manifest["notes"]["equations"]
    assert manifest["numerics"]["product_quadrature"] == "fine"
    assert (run_dir / "swe_coeffs.npy").exists()
    assert (run_dir / "diagnostics" / "timeseries.csv").exists()
    assert not (run_dir / "figures").exists()
