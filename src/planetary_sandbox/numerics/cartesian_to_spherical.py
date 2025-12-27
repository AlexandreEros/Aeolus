import numpy as np

def cartesian_to_spherical(cartesian_vertex_array: np.ndarray):
    """
    Convert Cartesian coordinates (x, y, z) to Spherical coordinates (r, lambda, phi).

    Returns:
        spherical_array: Array of shape (..., 3) containing:
            - r: Radius
            - longitude: Azimuthal angle in [-pi, pi]
            - latitude: Polar elevation angle in [-pi/2, pi/2] (geocentric latitude)
    """
    # Use slicing instead of transpose to preserve leading dimensions for shape (..., 3)
    x = cartesian_vertex_array[..., 0]
    y = cartesian_vertex_array[..., 1]
    z = cartesian_vertex_array[..., 2]

    # Radius
    xy_dist = np.hypot(x, y)
    radius = np.hypot(xy_dist, z)

    # Longitude (lambda)
    longitude = np.arctan2(y, x)

    # Latitude (phi)
    # arctan2(z, xy_dist) is more stable than arcsin(z/radius) near poles
    latitude = np.arctan2(z, xy_dist)

    spherical_array = np.stack([radius, longitude, latitude], axis=-1)

    # Ensure output shape matches input shape
    assert(cartesian_vertex_array.shape == spherical_array.shape)
    return np.ascontiguousarray(spherical_array)
