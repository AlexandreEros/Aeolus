from __future__ import annotations

import cupy as cp

from ...planet.planet import Planet

def _grid_point_latlon(planet: Planet) -> tuple[cp.ndarray, cp.ndarray]:
    lat = cp.asarray(planet.grid.point_latitudes)
    lon = cp.asarray(planet.grid.point_longitudes)
    return lat, lon

def _reshape_if_structured(planet: Planet, values: cp.ndarray) -> cp.ndarray:
    shape = getattr(planet.grid, "grid_shape", None)
    if shape is None:
        return values
    expected = int(shape[0] * shape[1])
    if values.size != expected:
        raise ValueError("values size does not match grid shape")
    return values.reshape(shape)

def _gaussian_on_sphere(lat, lon, lat0, lon0, sigma):
    # great-circle distance (cheap-ish)
    dlon = lon - lon0
    cosd = cp.sin(lat)*cp.sin(lat0) + cp.cos(lat)*cp.cos(lat0)*cp.cos(dlon)
    cosd = cp.clip(cosd, -1.0, 1.0)
    d = cp.arccos(cosd)
    return cp.exp(-(d*d)/(2*sigma*sigma))

def _two_vortices(planet: Planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon,  33*cp.pi/180,  cp.pi/2, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, -33*cp.pi/180, -cp.pi/2, s)
    return _reshape_if_structured(planet, z)

def _inverted_vortices(planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon,  33*cp.pi/180,  cp.pi/2, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, -33*cp.pi/180, -cp.pi/2, s)
    return _reshape_if_structured(planet, z)

def _polar_vortices(planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon,  cp.pi/2-1e-6,  0, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, -cp.pi/2+1e-6, cp.pi, s)
    return _reshape_if_structured(planet, z)

def _inverted_polar_vortices(planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon,  cp.pi/2-1e-6,  0, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, -cp.pi/2+1e-6, cp.pi, s)
    return _reshape_if_structured(planet, z)

def _equatorial_vortices(planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon, 0,  cp.pi/2, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, 0, -cp.pi/2, s)
    return _reshape_if_structured(planet, z)

def _inverted_equatorial_vortices(planet):
    lat, lon = _grid_point_latlon(planet)
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon, 0,  cp.pi/2, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, 0, -cp.pi/2, s)
    return _reshape_if_structured(planet, z)

def _random_low_l(planet):
    # easiest: random spectral coeffs for l <= L0, then inv_transform
    L0 = min(10, planet.sh.l_max)
    a = cp.zeros((planet.sh.l_max+1, planet.sh.l_max+1), dtype=cp.complex128)
    a[:L0+1, :L0+1] = (cp.random.standard_normal((L0+1, L0+1))
                      + 1j*cp.random.standard_normal((L0+1, L0+1))) * 1e-6
    return planet.sh.inv_transform(a)

def _rh4(planet):
    # Rossby–Haurwitz wave (wavenumber 4), returned as relative vorticity ζ(φ, λ).
    # Streamfunction (Williamson et al. / Thuburn & Li):
    #   ψ = -a^2*nu*sinφ + a^2*K*cos^4φ*sinφ*sin(4λ)
    # with nu = K = 7.848e-6 s^-1
    #
    # Using ζ = ∇²ψ/a² gives the closed form:
    #   ζ = 2*nu*sinφ - 30*K*sinφ*cos^4φ*sin(4λ)

    lat, lon = _grid_point_latlon(planet)
    nu = 7.848e-6  # s^-1
    K  = 7.848e-6  # s^-1

    sinphi = cp.sin(lat)
    cosphi = cp.cos(lat)

    zeta = 2.0 * nu * sinphi
    zeta += -30.0 * K * sinphi * (cosphi**4) * cp.sin(4.0 * lon)
    return _reshape_if_structured(planet, zeta)

INITIAL_CONDITIONS = {
    "two_vortices": _two_vortices,
    "inverted_vortices": _inverted_vortices,
    "polar_vortices": _polar_vortices,
    "inverted_polar_vortices": _inverted_polar_vortices,
    "equatorial_vortices": _equatorial_vortices,
    "inverted_equatorial_vortices": _inverted_equatorial_vortices,
    "random_low_l": _random_low_l,
    "rh4": _rh4,
}

def make_ic(name: str, planet):
    if name not in INITIAL_CONDITIONS:
        raise ValueError(f"Unknown initial condition: {name}. Available: {list(INITIAL_CONDITIONS.keys())}")
    return INITIAL_CONDITIONS[name](planet)

