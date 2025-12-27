import numpy as np
from typing import Tuple
from ..planet import Planet

def plot_velocity_streamlines(U: Tuple[np.ndarray, np.ndarray],
                              planet: Planet,
                              ax = None,
                              density: float = 1.5,
                              title: str = "Global Flow Streamlines",
                              grid = None):
    """
    Displays a matplotlib streamplot of the velocity field U on the planet.

    Parameters
    ----------
    U : tuple of (u_lambda, u_phi)
        Zonal (eastward) and Meridional (northward) velocity components [m/s].
    planet : Planet
        Planet object containing grid information.
    grid : optional
        Grid to use for plotting. Defaults to planet.grid.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    density : float
        Density of streamlines.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    # Handle inputs
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    u_grid, v_grid = U

    # Ensure CPU numpy arrays
    if cp is not None and isinstance(u_grid, cp.ndarray):
        u_grid = cp.asnumpy(u_grid)
    if cp is not None and isinstance(v_grid, cp.ndarray):
        v_grid = cp.asnumpy(v_grid)

    # Get grid coordinates (radians)
    if grid is None:
        grid = planet.grid

    lons = grid.longitudes  # Typically 0 to 2pi
    lats = grid.latitudes   # Typically pi/2 to -pi/2 (decreasing)

    # Streamplot requires strictly increasing x and y
    # Check if latitudes are decreasing (common in generated grids)
    if cp is not None and isinstance(lons, cp.ndarray):
        lons = cp.asnumpy(lons)
    if cp is not None and isinstance(lats, cp.ndarray):
        lats = cp.asnumpy(lats)

    if lats[0] > lats[-1]:
        # Flip everything along latitude axis (axis 0)
        lats = np.flip(lats)
        u_grid = np.flip(u_grid, axis=0)
        v_grid = np.flip(v_grid, axis=0)

    # Downsample if grid is too large for matplotlib streamplot (optional optimization)
    stride = 1
    if lats.size > 300 or lons.size > 500:
        stride = 2
    if lats.size > 600 or lons.size > 1000:
        stride = 4

    if stride > 1:
        lats = lats[::stride]
        lons = lons[::stride]
        u_grid = u_grid[::stride, ::stride]
        v_grid = v_grid[::stride, ::stride]

    # Convert to degrees for plotting
    lon_deg = np.rad2deg(lons)
    lat_deg = np.rad2deg(lats)

    # Convert linear velocity (m/s) to angular rates for geometrically correct
    # streamlines in the equirectangular projection (lat/lon grid).
    # dx_map = d(lon), dy_map = d(lat)
    # d(lon)/dt = u / (R * cos(lat))
    # d(lat)/dt = v / R

    R = planet.params.equatorial_radius
    cos_lat = np.cos(lats)
    cos_lat = np.maximum(cos_lat, 1e-4) # Avoid division by zero at poles

    # u_ang and v_ang represent the "velocity" in the map's coordinate space (radians/s)
    u_ang = u_grid / (R * cos_lat[:, np.newaxis])
    v_ang = v_grid / R

    # Calculate physical speed for coloring (magnitude of original vector)
    speed = np.sqrt(u_grid**2 + v_grid**2)

    # Plot
    st = ax.streamplot(lon_deg, lat_deg, u_ang, v_ang,
                       color=speed, cmap='viridis',
                       density=density, linewidth=1, arrowsize=1.2)

    plt.colorbar(st.lines, ax=ax, label='Flow Speed (m/s)',
                 orientation='horizontal', pad=0.1, fraction=0.05, aspect=30)

    ax.set_xlim(lon_deg.min(), lon_deg.max())
    ax.set_ylim(lat_deg.min(), lat_deg.max())
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)

    # Set aspect ratio to auto or specific value if desired.
    # For global maps, equal steps in lat/lon is often used in basic plots,
    # but physics is distorted.
    ax.set_aspect('equal')

    return ax
