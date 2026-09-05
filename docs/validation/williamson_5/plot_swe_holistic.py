"""One static, vertically ordered W5 figure from saved physical snapshots."""
from pathlib import Path
import hashlib
root = Path(__file__).resolve().parent

source = root / "aeolus_w5_t63.npz"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import (LinearSegmentedColormap, Normalize, hsv_to_rgb,
                               rgb_to_hsv, to_rgba)
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
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

def human_time(days: float) -> str:
    total_minutes = round(days * 24 * 60)
    day, remainder = divmod(total_minutes, 24 * 60)
    hour, minute = divmod(remainder, 60)
    return f"{day} days, {hour} hours, {minute} minutes"


plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "axes.linewidth": 0.65,
                     "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(15, 21))
grid = fig.add_gridspec(4, 3, width_ratios=(2.15, 9.0, 2.35),
                        left=0.055, right=0.97, bottom=0.045, top=0.92,
                        wspace=0.08, hspace=0.31)
fig.suptitle("Williamson test case 5", fontsize=18, y=0.972)
fig.text(0.5, 0.947,
         "Saved T63 shallow-water snapshots · total free-surface height and velocity",
         ha="center", fontsize=10, color="#45505a")

# A single, quiet guide remains to the left of every map.
guide = fig.add_subplot(grid[:, 0])
guide.axis("off")
guide.text(0.0, 0.988, "MAP ENCODINGS", transform=guide.transAxes,
           fontsize=10, fontweight="bold", color="#25313a", va="top")
guide.text(0.0, 0.947, "Fixed limits for all snapshots", transform=guide.transAxes,
           fontsize=8.5, color="#5c6870", va="top")


def guide_bar(bounds, mapper, ticks, title, detail, background=None):
    axis = guide.inset_axes(bounds)
    if background is not None:
        axis.set_facecolor(background)
    colorbar = fig.colorbar(mapper, cax=axis, orientation="vertical", ticks=ticks)
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    center_x = bounds[0] + bounds[2] / 2
    guide.text(center_x, bounds[1] + bounds[3] + 0.018, title,
               transform=guide.transAxes, fontsize=9.5, fontweight="bold",
               color="#25313a", ha="center", va="bottom")
    guide.text(center_x, bounds[1] - 0.018, detail,
               transform=guide.transAxes, fontsize=8.5, color="#45505a",
               ha="center", va="top", linespacing=1.35)


guide_bar([0.10, 0.735, 0.15, 0.12], ScalarMappable(norm=speed_norm, cmap=speed_cmap),
          [0, 15, 30, speed_max], "Streamlines",
          "Dashed; colour and opacity\nshow speed (m s⁻¹).\nα increases 0.10 → 1.00.",
          background=terrain_cmap(terrain_norm(0)))
guide_bar([0.10, 0.475, 0.15, 0.12], ScalarMappable(norm=height_norm, cmap=contour_cmap),
          [5000, 5500, 6000], "Free-surface contours", "Solid; colour shows total\nheight H (m). 100 m spacing.")
guide_bar([0.10, 0.215, 0.15, 0.12], ScalarMappable(norm=terrain_norm, cmap=terrain_cmap),
          [0, 1000, 2000], "Terrain relief", "Land-only terrain map\nfor 0–2000 m relief.")
guide.text(0.0, 0.095,
           "H = (Φ₀ + φ + φₛ)/g\n\nH remains positive in every\nsaved snapshot; no zero\nreference contour applies.",
           transform=guide.transAxes, fontsize=8.5, color="#45505a",
           va="top", linespacing=1.4)

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
    ax.set(title=f"Day {int(fields['time'][index])}   ·   {human_time(fields['time'][index])}",
           xlabel="Longitude (°E)", ylabel="Latitude (°)",
           xlim=(0, 360), ylim=(-90, 90),
           xticks=np.arange(0, 361, 60), yticks=np.arange(-90, 91, 30))
    ax.set_aspect("equal")
    ax.tick_params(width=0.6, length=3)

    field_speed = np.hypot(fields["u"][index], fields["v"][index])
    h_min, h_max = height[index].min(), height[index].max()
    diagnostics.set_xlim(0, 1)
    diagnostics.set_ylim(0, 1)
    diagnostics.axis("off")
    diagnostics.axvline(0.03, 0.08, 0.92, color="#d3d9dd", linewidth=0.8)
    diagnostics.text(0.13, 0.90, "SNAPSHOT DIAGNOSTICS",
                     fontsize=9, fontweight="bold", color="#25313a", va="top")
    diagnostics.text(0.13, 0.72, "Mean speed", fontsize=8.5, color="#5c6870")
    diagnostics.text(0.13, 0.64, f"{field_speed.mean():5.2f} m s⁻¹",
                     fontsize=11, color="#25313a", fontfamily="DejaVu Sans Mono")
    diagnostics.text(0.13, 0.49, "Peak speed", fontsize=8.5, color="#5c6870")
    diagnostics.text(0.13, 0.41, f"{field_speed.max():5.2f} m s⁻¹",
                     fontsize=11, color="#25313a", fontfamily="DejaVu Sans Mono")
    diagnostics.text(0.13, 0.26, "Total H range", fontsize=8.5, color="#5c6870")
    diagnostics.text(0.13, 0.18, f"{h_min:4.0f}–{h_max:4.0f} m",
                     fontsize=11, color="#25313a", fontfamily="DejaVu Sans Mono")
    diagnostics.text(0.13, 0.07, f"Potential enstrophy  {fields['potential_enstrophy'][index]:.4f}",
                     fontsize=8.5, color="#45505a")

output = root / "overview.png"
fig.savefig(output, dpi=300, facecolor="white",
            metadata={"SourceSHA256": source_hash})
plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
with Image.open(output) as image:
    image.verify()
with Image.open(output) as image:
    assert image.size == (4500, 6300)
    assert abs(image.info["dpi"][0] - 300) < 0.01
    print(f"Saved: {output}\nSize: {image.size}; DPI: {image.info['dpi']}")
print(f"Total-height range: {height.min():.3f} to {height.max():.3f} m")
print("Reconstruction verified; source archive unchanged.")
