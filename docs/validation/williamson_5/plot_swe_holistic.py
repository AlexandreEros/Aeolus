"""One static, vertically ordered W5 figure from saved physical snapshots.

Three saved artifacts of the single canonical T63 run feed this figure; none
of them is recomputed here and the simulation is never re-run.

* ``aeolus_w5_t63.npz``            physical snapshot fields (u, v, layer
                                   depth, band-limited terrain, Gauss
                                   quadrature weights, potential enstrophy)
* ``runs/.../swe_coeffs.npy``      the spectral state at the same four
                                   snapshot times, stacked [zeta, delta, phi]
* ``runs/.../diagnostics/``        the per-step conservation record, read
  ``timeseries.csv``               only at the exact snapshot times

Diagnostic definitions
----------------------
**Area-mean speed.** Area-weighted with the Gauss-Legendre latitude weights.
A plain grid mean over a Gauss lat-lon grid over-weights the polar rows and
under-reports a tropically peaked jet; for the day-0 solid-body flow the
area-weighted value reproduces the analytic ``u0*pi/4`` exactly.

**Conservation drift.** Relative drift from day 0 of the run's own conserved
quantities: layer mass and topographic total energy from the diagnostics CSV
(definitions in ``run/swe/diagnostics.py``), and potential enstrophy
``Z = integral (zeta + f)^2 / (2h) dA`` from the snapshot package.

**Spectral complexity.** The horizontal velocity is decomposed the way the
solver already carries it -- rotational (streamfunction) plus divergent
(velocity potential) parts, i.e. the vector-spherical-harmonic split -- never
by treating raw lat-lon u and v as two independent scalar fields. With
``psi_lm = -R^2 zeta_lm / (l(l+1))`` and ``chi_lm = -R^2 delta_lm/(l(l+1))``,
and the two parts orthogonal in the energy integral over a closed surface,
the modal kinetic energy is

    E(l,m) = (R^4 / 2) * [P_zeta(l,m) + P_delta(l,m)] / (l(l+1)),   l >= 1
    E(0,0) = 0                                (no velocity in the l=0 mode)

where ``P`` is the repository's own ``_mode_power`` convention for the m>=0
storage layout: ``|c|^2`` for m=0 and ``2|c|^2`` for m>0, so each stored mode
already carries the power of its full +/-m conjugate pair. E(l,m) is
nonnegative by construction, and ``sum E`` equals the grid quadrature of
``0.5*integral |u|^2 dA`` -- asserted below to 1e-11 relative, which pins the
normalization, the m-doubling and the run identity of all three artifacts in
one check. The distribution is then

    p(l,m) = E(l,m) / sum E(l,m)          (nonnegative, sums to 1)

and the three reported measures are

    <l>   = sum l * p(l,m)                power-weighted mean degree
    <|m|> = sum |m| * p(l,m)              power-weighted mean zonal wavenumber
    S     = -sum p ln p                   Shannon entropy of p, zero-power
                                          modes contributing exactly 0
    N_eff = exp(S)                        effective number of occupied modes

``S`` is the Shannon entropy of a modal energy distribution. It is not a
thermodynamic entropy and not a measure of "information content". The figure
reports the more legible ``N_eff``; ``S`` is printed to stdout. The same
definition is applied to all four snapshots, and R cancels in p, so the three
measures are independent of the planetary radius.
"""
from pathlib import Path
import csv
import hashlib
root = Path(__file__).resolve().parent

source = root / "aeolus_w5_t63.npz"
capsule = (root / "runs" / "t63" / "w5-mri" /
           "20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a")
coeff_source = capsule / "swe_coeffs.npy"
coeff_times_source = capsule / "swe_snapshot_times.npy"
conservation_source = capsule / "diagnostics" / "timeseries.csv"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, Normalize, hsv_to_rgb,
                               rgb_to_hsv, to_rgba)
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

# Canonical W5 constants of the run being plotted (manifest run_config).
RADIUS_M = 6.37122e6
SECONDS_PER_DAY = 86400.0

for required in (source, coeff_source, coeff_times_source, conservation_source):
    if not required.is_file():
        raise FileNotFoundError(
            f"{required} is missing. This figure reports only quantities that "
            "can be computed rigorously from the saved canonical T63 run; it "
            "does not approximate or reconstruct absent artifacts.")

source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
coeff_hash = hashlib.sha256(coeff_source.read_bytes()).hexdigest()
conservation_hash = hashlib.sha256(conservation_source.read_bytes()).hexdigest()
with np.load(source, allow_pickle=False) as saved:
    fields = {key: saved[key].copy() for key in saved.files}

np.testing.assert_array_equal(fields["time"], [0, 5, 10, 15])
lat = fields["latitude"][::-1]
lon = fields["longitude"]
terrain = fields["topography_height_bandlimited"][::-1]
# The saved export defines layer_depth=(Phi0+phi)/g and terrain=phi_s/g.
height = fields["layer_depth"][:, ::-1] + terrain[None, :, :]
np.testing.assert_allclose(height, fields["free_surface_height"][:, ::-1],
                           rtol=0, atol=1e-10)
assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
for key in ("u", "v", "layer_depth", "free_surface_height"):
    assert fields[key].shape == (4, lat.size, lon.size)
    assert np.isfinite(fields[key]).all()
assert np.isfinite(terrain).all()

# Normalized area weights for the Gauss lat-lon state grid: the latitude
# quadrature weights sum to 2 and every longitude carries the same share.
area_weights = (fields["gl_weights"][:, None]
                / (fields["gl_weights"].sum() * lon.size))
np.testing.assert_allclose(area_weights.sum() * lon.size, 1.0, rtol=0,
                           atol=1e-14)

# ---------------------------------------------------------------------------
# Spectral complexity from the saved coefficients (see module docstring).
# ---------------------------------------------------------------------------
coeffs = np.load(coeff_source, allow_pickle=False)
coeff_times = np.load(coeff_times_source, allow_pickle=False)
if coeffs.ndim != 4 or coeffs.shape[1] != 3:
    raise ValueError("swe_coeffs.npy must have shape (time, 3, l, m), got "
                     f"{coeffs.shape}")
np.testing.assert_allclose(coeff_times / SECONDS_PER_DAY, fields["time"],
                           rtol=0, atol=1e-9)
ZETA, DELTA, PHI = 0, 1, 2
l_max = coeffs.shape[2] - 1
degree = np.arange(l_max + 1, dtype=np.float64)
order = np.arange(l_max + 1, dtype=np.float64)
degree_index, order_index = np.indices(coeffs.shape[2:])
retained = order_index <= degree_index
# 1/(l(l+1)) with the l=0 row left at zero: the l=0 mode carries no velocity,
# so it contributes exactly zero energy rather than dividing by zero.
inverse_laplacian = np.zeros(l_max + 1, dtype=np.float64)
inverse_laplacian[1:] = 1.0 / (degree[1:] * (degree[1:] + 1.0))


def mode_power(single_time_coeffs):
    """Per-(l,m) contribution to integral |f|^2 dOmega, m>=0 storage layout."""
    power = np.where(order_index == 0, single_time_coeffs.real ** 2,
                     2.0 * np.abs(single_time_coeffs) ** 2)
    return np.where(retained, power, 0.0)


mean_degree = np.empty(4)
mean_order = np.empty(4)
spectral_entropy = np.empty(4)
effective_modes = np.empty(4)
for index in range(4):
    modal_energy = (0.5 * RADIUS_M ** 4 * inverse_laplacian[:, None]
                    * (mode_power(coeffs[index, ZETA])
                       + mode_power(coeffs[index, DELTA])))
    assert np.all(modal_energy >= 0.0) and np.isfinite(modal_energy).all()
    # The spectral kinetic energy must equal the grid quadrature of the saved
    # velocity: one check that pins normalization, m-doubling, the rotational/
    # divergent split, and that all three artifacts describe the same run.
    grid_kinetic_energy = 0.5 * RADIUS_M ** 2 * 4.0 * np.pi * np.sum(
        area_weights * (fields["u"][index] ** 2 + fields["v"][index] ** 2))
    np.testing.assert_allclose(modal_energy.sum(), grid_kinetic_energy,
                               rtol=1e-11, atol=0)

    probability = modal_energy / modal_energy.sum()
    np.testing.assert_allclose(probability.sum(), 1.0, rtol=0, atol=1e-13)
    mean_degree[index] = (degree[:, None] * probability).sum()
    mean_order[index] = (order[None, :] * probability).sum()
    # Zero-power modes contribute exactly zero to -sum p ln p; excluding them
    # avoids 0*log(0) rather than approximating it.  The +0.0 turns the -0.0
    # of a single fully occupied mode into a plain 0.0.
    occupied = probability[probability > 0.0]
    spectral_entropy[index] = float(-(occupied * np.log(occupied)).sum()) + 0.0
    assert spectral_entropy[index] >= 0.0
    effective_modes[index] = np.exp(spectral_entropy[index])

# ---------------------------------------------------------------------------
# Conservation drift, read at the exact snapshot times of the same run.
# ---------------------------------------------------------------------------
with conservation_source.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
step_time = np.array([float(row["time_s"]) for row in rows])
snapshot_rows = []
for target in coeff_times:
    exact = np.flatnonzero(step_time == target)
    if exact.size == 0:
        raise ValueError(f"no diagnostics row lands exactly on t={target} s")
    snapshot_rows.append(rows[int(exact[0])])
total_mass = np.array([float(row["total_mass"]) for row in snapshot_rows])
total_energy = np.array([float(row["total_energy"]) for row in snapshot_rows])
enstrophy = fields["potential_enstrophy"]
mass_drift = total_mass / total_mass[0] - 1.0
energy_drift = total_energy / total_energy[0] - 1.0
enstrophy_drift = enstrophy / enstrophy[0] - 1.0
# The layer-mass integral is the l=0 coefficient of the thickness geopotential,
# which the spectral divergence operator never touches; assert the exactness
# that lets the figure report a mass drift of identically zero.
assert np.all(coeffs[:, PHI, 0, 0] == coeffs[0, PHI, 0, 0])
assert np.all(mass_drift == 0.0)
worst_drift = max(np.abs(mass_drift).max(), np.abs(energy_drift).max(),
                  np.abs(enstrophy_drift).max())

# Close the periodic longitude seam; resample Gaussian latitudes for streamplot.
lon_closed = np.r_[lon, 360.0]
view_lat = np.linspace(lat[0], lat[-1], 181)
view_lon = np.linspace(0, 360, 361)
xx, yy = np.meshgrid(view_lon, view_lat)
points = np.stack((yy, xx), axis=-1)
terrain_closed = np.column_stack((terrain, terrain[:, 0]))

# Fixed encodings across all four snapshots.  Restrict terrain to its land
# portion so 0--2000 m never enters the blue ocean/depth segment.
terrain_colors = plt.get_cmap("terrain")(np.linspace(0.25, 0.72, 256))
terrain_colors[:, :3] *= 0.35
terrain_cmap = LinearSegmentedColormap.from_list(
    "w5_elevated_terrain", terrain_colors)
contour_cmap = LinearSegmentedColormap.from_list(
    "w5_height_contours", ["#44bbfb7a", "#0000af"])
terrain_norm = Normalize(-20, 2000)
dataset_speed = np.hypot(fields["u"], fields["v"])
speed_max = dataset_speed.max()
speed_norm = Normalize(0, speed_max)
speed_start = np.array([*terrain_colors[0, :3] / 0.35, 0.2])
speed_end = np.array(to_rgba("#ff4511ff"))
start_hsv = rgb_to_hsv(speed_start[:3])
end_hsv = rgb_to_hsv(speed_end[:3])
hue_delta = (end_hsv[0] - start_hsv[0] + 0.5) % 1.0 - 0.5
speed_positions = np.linspace(0.0, 1.0, 256)
speed_hsv = np.column_stack([
    (start_hsv[0] + hue_delta * speed_positions) % 1.0,
    np.linspace(start_hsv[1], end_hsv[1], speed_positions.size),
    np.linspace(start_hsv[2], end_hsv[2], speed_positions.size),
])
speed_colors = np.column_stack([
    hsv_to_rgb(speed_hsv),
    np.linspace(speed_start[3], speed_end[3], speed_positions.size),
])
speed_cmap = LinearSegmentedColormap.from_list(
    "w5_speed_green_to_plasma_orange",
    speed_colors)
height_norm = Normalize(5000, 6000)
height_levels = np.arange(5000, 6001, 100)

# What the reader can verify in each panel; no claim beyond what is drawn.
SNAPSHOT_CAPTIONS = (
    "initial state · undisturbed zonal flow",
    "wave train building downstream of the mountain",
    "meanders have reached every longitude",
    "developed wave train in both hemispheres",
)


def drift_text(value, is_reference):
    if is_reference:
        return "reference"
    if value == 0.0:
        return "0.00e+00"
    return f"{value:+.2e}"


def diagnostic_section(axis, cursor, title, subtitle):
    """Small caption-weight heading; returns the next line's y position."""
    axis.text(0.06, cursor, title, fontsize=8.5, fontweight="bold",
              color="#25313a", va="top")
    axis.text(0.06, cursor - 0.042, subtitle, fontsize=7.5, color="#8894a0",
              va="top")
    return cursor - 0.100


def diagnostic_entry(axis, cursor, label, value):
    """One label/value line, value right-aligned to the column edge."""
    axis.text(0.06, cursor, label, fontsize=8.0, color="#5c6870", va="center")
    axis.text(1.0, cursor, value, fontsize=8.6, color="#25313a", va="center",
              ha="right", fontfamily="DejaVu Sans Mono")
    return cursor - 0.062


plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "axes.linewidth": 0.65,
                     "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(15, 21))
grid = fig.add_gridspec(4, 3, width_ratios=(2.15, 9.0, 2.35),
                        left=0.055, right=0.97, bottom=0.045, top=0.92,
                        wspace=0.08, hspace=0.31)
fig.suptitle("Williamson Test Case 5 — Flow Over an Isolated Mountain",
             fontsize=18, y=0.972)
fig.text(0.5, 0.947,
         "Aeolus shallow-water solver at T63 (ℓ ≤ 63 on a 96×192 Gauss grid) · "
         "free-surface height and velocity at four saved snapshots",
         ha="center", fontsize=10, color="#45505a")

# A single, quiet guide remains to the left of every map.
guide = fig.add_subplot(grid[:, 0])
guide.axis("off")
guide.text(0.0, 0.995, "MAP ENCODINGS", transform=guide.transAxes,
           fontsize=10, fontweight="bold", color="#25313a", va="top")
guide.text(0.0, 0.977, "Identical limits in all four snapshots",
           transform=guide.transAxes, fontsize=8.5, color="#5c6870", va="top")


def guide_bar(bounds, mapper, ticks, title, detail, background=None):
    axis = guide.inset_axes(bounds)
    if background is not None:
        axis.set_facecolor(background)
    colorbar = fig.colorbar(mapper, cax=axis, orientation="vertical", ticks=ticks)
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    center_x = bounds[0] + bounds[2] / 2
    guide.text(center_x, bounds[1] + bounds[3] + 0.006, title,
               transform=guide.transAxes, fontsize=9.5, fontweight="bold",
               color="#25313a", ha="center", va="bottom")
    guide.text(center_x, bounds[1] - 0.006, detail,
               transform=guide.transAxes, fontsize=8.5, color="#45505a",
               ha="center", va="top", linespacing=1.35)


guide_bar([0.10, 0.845, 0.15, 0.105],
          ScalarMappable(norm=speed_norm, cmap=speed_cmap),
          [0, 15, 30, speed_max], "Wind speed",
          "Dashed streamlines; colour and\nopacity give the speed in m s⁻¹,\n"
          f"0 to the run maximum {speed_max:.1f}.\nOpacity rises 0.20 → 1.00.",
          background=terrain_cmap(terrain_norm(0)))
guide_bar([0.10, 0.680, 0.15, 0.105],
          ScalarMappable(norm=height_norm, cmap=contour_cmap),
          [5000, 5500, 6000], "Free-surface height",
          "Solid contours every 100 m,\ncoloured by height in metres.\n"
          "Height of the water surface\nabove the reference sphere.")
guide_bar([0.10, 0.515, 0.15, 0.105],
          ScalarMappable(norm=terrain_norm, cmap=terrain_cmap),
          [0, 1000, 2000], "Terrain elevation",
          "The Williamson-5 cone: 2000 m\nnominal peak, "
          f"{terrain.max():.0f} m after\nspectral truncation to ℓ ≤ {l_max}.")

guide.text(0.0, 0.455, "READING THE DIAGNOSTICS",
           transform=guide.transAxes, fontsize=9.5, fontweight="bold",
           color="#25313a", va="top")
guide.text(0.0, 0.435,
           "Free-surface height\n"
           "  H = (Φ₀ + φ + φₛ)/g, the layer\n"
           "  depth plus terrain elevation.\n"
           "  It stays positive in every\n"
           "  snapshot, so no zero contour\n"
           "  is drawn.\n"
           "\n"
           "Area-mean speed is weighted by\n"
           "  the Gauss latitude weights,\n"
           "  not a plain grid average.",
           transform=guide.transAxes, fontsize=7.8, color="#45505a",
           va="top", linespacing=1.5)
guide.text(0.0, 0.330,
           "Conservation drift is measured\n"
           "  against day 0. Layer mass and\n"
           "  total energy come from the\n"
           "  run's per-step record;\n"
           "  potential enstrophy is\n"
           "  Z = ∫(ζ+f)²/2h dA.",
           transform=guide.transAxes, fontsize=7.8, color="#45505a",
           va="top", linespacing=1.5)
guide.text(0.0, 0.258,
           "Spectral complexity splits the\n"
           "  kinetic energy into rotational\n"
           "  and divergent harmonic modes,\n"
           "  never into raw u and v:\n"
           "  E(ℓ,m) ∝ (|ζ|²+|δ|²)/ℓ(ℓ+1),\n"
           "  normalised to p = E/ΣE.\n"
           "  ⟨ℓ⟩ and ⟨|m|⟩ are p-weighted\n"
           "  mean degree and zonal wave-\n"
           "  number; the effective mode\n"
           "  count is Nₑ = exp(S), where\n"
           "  S = −Σ p ln p is a Shannon\n"
           "  entropy of that distribution\n"
           "  — not a thermodynamic one.",
           transform=guide.transAxes, fontsize=7.8, color="#45505a",
           va="top", linespacing=1.5)

guide.text(0.0, 0.115, "WHAT TO LOOK FOR", transform=guide.transAxes,
           fontsize=9.5, fontweight="bold", color="#25313a", va="top")
guide.text(0.0, 0.095,
           "Day 0 is one single mode, ℓ = 1\n"
           "and m = 0: the undisturbed zonal\n"
           "jet, holding every last bit of\n"
           f"the kinetic energy, so Nₑ = {effective_modes[0]:.2f}.\n"
           "Flow over the mountain spreads it\n"
           f"across ⟨ℓ⟩ = {mean_degree[-1]:.1f} and about "
           f"{effective_modes[-1]:.1f}\neffective modes by day 15, while\n"
           "mass, energy and potential\n"
           f"enstrophy hold to {worst_drift:.0e} or less.",
           transform=guide.transAxes, fontsize=7.8, color="#45505a",
           va="top", linespacing=1.5)

for index in range(4):
    ax = fig.add_subplot(grid[index, 1])
    diagnostics = fig.add_subplot(grid[index, 2])
    ax.pcolormesh(lon_closed, lat, terrain_closed, cmap=terrain_cmap,
                  norm=terrain_norm, shading="auto", rasterized=True, zorder=0)

    winds = []
    for key in ("u", "v"):
        field = fields[key][index, ::-1]
        periodic = np.column_stack((field, field[:, 0]))
        winds.append(RegularGridInterpolator((lat, lon_closed), periodic)(points))
    u, v = winds
    speed = np.hypot(u, v)
    # Existing viz/maps.py geometry: dlon/dt=u/(R*cos(lat)), dlat/dt=v/R.
    cos_lat = np.maximum(np.cos(np.deg2rad(view_lat)), 1e-4)
    streams = ax.streamplot(view_lon, view_lat, u / cos_lat[:, None], v,
                            color=speed, cmap=speed_cmap, norm=speed_norm,
                            density=0.97, linewidth=0.60, arrowsize=0.78,
                            zorder=2)
    streams.lines.set_linestyle((0, (3.0, 2.2)))

    h_closed = np.column_stack((height[index], height[index, :, 0]))
    ax.contour(lon_closed, lat, h_closed, levels=height_levels,
               cmap=contour_cmap, norm=height_norm, linewidths=1.20,
               linestyles="solid", zorder=3)
    ax.set(title=f"Day {int(fields['time'][index])}   ·   {SNAPSHOT_CAPTIONS[index]}",
           xlabel="Longitude (°E)", ylabel="Latitude (°)",
           xlim=(0, 360), ylim=(-90, 90),
           xticks=np.arange(0, 361, 60), yticks=np.arange(-90, 91, 30))
    ax.set_aspect("equal")
    ax.tick_params(width=0.6, length=3)

    field_speed = np.hypot(fields["u"][index], fields["v"][index])
    area_mean_speed = (area_weights * field_speed).sum()
    h_min, h_max = height[index].min(), height[index].max()
    diagnostics.set_xlim(0, 1)
    diagnostics.set_ylim(0, 1)
    diagnostics.axis("off")
    diagnostics.axvline(0.0, 0.06, 0.94, color="#d3d9dd", linewidth=0.8)

    is_reference = index == 0
    cursor = diagnostic_section(diagnostics, 0.945,
                                "FLOW AND SURFACE", "this snapshot")
    cursor = diagnostic_entry(diagnostics, cursor, "Area-mean speed",
                              f"{area_mean_speed:.2f} m s⁻¹")
    cursor = diagnostic_entry(diagnostics, cursor, "Peak speed",
                              f"{field_speed.max():.2f} m s⁻¹")
    cursor = diagnostic_entry(diagnostics, cursor, "Free-surface height",
                              f"{h_min:.0f}–{h_max:.0f} m")

    cursor = diagnostic_section(diagnostics, cursor - 0.055,
                                "CONSERVATION", "relative drift from day 0")
    cursor = diagnostic_entry(diagnostics, cursor, "Layer mass",
                              drift_text(mass_drift[index], is_reference))
    cursor = diagnostic_entry(diagnostics, cursor, "Total energy",
                              drift_text(energy_drift[index], is_reference))
    cursor = diagnostic_entry(diagnostics, cursor, "Potential enstrophy",
                              drift_text(enstrophy_drift[index], is_reference))

    cursor = diagnostic_section(diagnostics, cursor - 0.055,
                                "SPECTRAL COMPLEXITY", "kinetic energy by ℓ and |m|")
    cursor = diagnostic_entry(diagnostics, cursor, "Mean degree ⟨ℓ⟩",
                              f"{mean_degree[index]:.3f}")
    cursor = diagnostic_entry(diagnostics, cursor, "Mean zonal wavenumber",
                              f"{mean_order[index]:.3f}")
    cursor = diagnostic_entry(diagnostics, cursor, "Effective modes Nₑ",
                              f"{effective_modes[index]:.2f}")

output = root / "overview.png"
fig.savefig(output, dpi=300, facecolor="white",
            metadata={"SourceSHA256": source_hash,
                      "CoefficientsSHA256": coeff_hash,
                      "DiagnosticsSHA256": conservation_hash})
plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
assert hashlib.sha256(coeff_source.read_bytes()).hexdigest() == coeff_hash
assert (hashlib.sha256(conservation_source.read_bytes()).hexdigest()
        == conservation_hash)
with Image.open(output) as image:
    image.verify()
with Image.open(output) as image:
    assert image.size == (4500, 6300)
    assert abs(image.info["dpi"][0] - 300) < 0.01
    print(f"Saved: {output}\nSize: {image.size}; DPI: {image.info['dpi']}")
print(f"Total-height range: {height.min():.3f} to {height.max():.3f} m")
print(f"\n{'Day':>4}  {'mass':>10} {'energy':>10} {'enstrophy':>10}"
      f"  {'<l>':>7} {'<|m|>':>7} {'S':>7} {'N_eff':>7}")
for index in range(4):
    print(f"{fields['time'][index]:4.0f}  {mass_drift[index]:10.2e} "
          f"{energy_drift[index]:10.2e} {enstrophy_drift[index]:10.2e}"
          f"  {mean_degree[index]:7.3f} {mean_order[index]:7.3f} "
          f"{spectral_entropy[index]:7.4f} {effective_modes[index]:7.3f}")
print("Reconstruction verified; source archives unchanged.")
