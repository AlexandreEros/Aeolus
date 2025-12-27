from __future__ import annotations

import cupy as cp

from ...planet.planet import Planet

def _gaussian_on_sphere(lat, lon, lat0, lon0, sigma):
    # great-circle distance (cheap-ish)
    dlon = lon - lon0
    cosd = cp.sin(lat)*cp.sin(lat0) + cp.cos(lat)*cp.cos(lat0)*cp.cos(dlon)
    cosd = cp.clip(cosd, -1.0, 1.0)
    d = cp.arccos(cosd)
    return cp.exp(-(d*d)/(2*sigma*sigma))

def _two_vortices(planet: Planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon,  33*cp.pi/180,  cp.pi/2, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, -33*cp.pi/180, -cp.pi/2, s)
    return z

def _inverted_vortices(planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon,  33*cp.pi/180,  cp.pi/2, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, -33*cp.pi/180, -cp.pi/2, s)
    return z

def _polar_vortices(planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon,  cp.pi/2-1e-6,  0, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, -cp.pi/2+1e-6, cp.pi, s)
    return z

def _inverted_polar_vortices(planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon,  cp.pi/2-1e-6,  0, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, -cp.pi/2+1e-6, cp.pi, s)
    return z

def _equatorial_vortices(planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  +5e-5 * _gaussian_on_sphere(lat, lon, 0,  cp.pi/2, s)
    z += -5e-5 * _gaussian_on_sphere(lat, lon, 0, -cp.pi/2, s)
    return z

def _inverted_equatorial_vortices(planet):
    lat = planet.grid.latitudes
    lon = planet.grid.longitudes
    s = 10.0 * cp.pi/180.0
    z =  -5e-5 * _gaussian_on_sphere(lat, lon, 0,  cp.pi/2, s)
    z += +5e-5 * _gaussian_on_sphere(lat, lon, 0, -cp.pi/2, s)
    return z

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

    lat = cp.asarray(planet.grid.lat_grid)
    lon = cp.asarray(planet.grid.lon_grid)
    nu = 7.848e-6  # s^-1
    K  = 7.848e-6  # s^-1

    sinphi = cp.sin(lat)
    cosphi = cp.cos(lat)

    zeta = 2.0 * nu * sinphi
    zeta += -30.0 * K * sinphi * (cosphi**4) * cp.sin(4.0 * lon)
    return zeta

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

