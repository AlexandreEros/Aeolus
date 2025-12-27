from __future__ import annotations
from dataclasses import dataclass

import cupy as cp

from planetary_sandbox.planet import Planet

@dataclass
class BarotropicState:
    coeffs: cp.ndarray
    tendency: cp.ndarray = None


class BarotropicVorticity:
    """
    Barotropic vorticity equation solver on a rotating sphere.

    Solves: ∂ζ/∂t + J(ψ, ζ + f) = ν∇²ζ + F
    """

    def __init__(self,
                 planet: Planet,
                 scenario: str = "two_vortices",
                 viscosity: float = 0.0):
        """
        Parameters
        ----------
        planet : Planet
            Planet object with grid already set
        scenario : str
            Initial condition scenario name
        viscosity : float
            Kinematic viscosity [m²/s] (optional damping)
        """
        self.planet = planet
        self.nu = viscosity

        self.sh = planet.sh
        self.so = planet.so
        self.grid = planet.grid
        self.R = planet.params.radius
        self.Omega = planet.params.angular_velocity

        # Precompute planetary vorticity f = 2Ω sin(φ)
        # φ is latitude, but we have colatitude θ
        # sin(φ) = sin(π/2 - θ) = cos(θ)
        self.f = 2 * self.Omega * cp.sin(cp.array(self.grid.latitudes))  # Coriolis parameter

        # Precompute Laplacian eigenvalues
        l_vals = cp.arange(self.sh.l_max + 1, dtype=cp.float64)
        self.laplacian_eig = -l_vals * (l_vals + 1) / self.R**2
        self.laplacian_eig[0] = 1.0 / self.R**2 # Avoid division by zero (l=0 mode is constant)

    def vorticity_to_streamfunction(self, 
                                    vrt_state: BarotropicState
                                    ) -> cp.ndarray:
        """
        Invert Laplacian: ∇²ψ = ζ  →  ψ = ∇⁻²ζ

        In spectral space: ψ_lm = ζ_lm / λ_l  where λ_l = -l(l+1)/R²

        Parameters
        ----------
        zeta : cp.ndarray
            Coefficients of vorticity in spectral space

        Returns
        -------
        psi_coeffs : BarotropicState
            Coefficients of streamfunction in spectral space
        """
        assert isinstance(vrt_state, BarotropicState)
        zeta_coeffs = vrt_state.coeffs
        psi_coeffs = cp.zeros_like(zeta_coeffs)

        # Divide by eigenvalues (skip l=0, keep it zero)
        psi_coeffs[1:, :] = zeta_coeffs[1:, :] / self.laplacian_eig[1:, None]

        return psi_coeffs

    def streamfunction_to_vorticity(self, psi_coeffs) -> BarotropicState:
        """
        Laplacian: ∇²ζ = ψ

        In spectral space: ζ_lm = ∇²ψ_lm = -l(l+1)/R² * ψ_lm

        Parameters
        ----------
        psi_coeffs : cp.ndarray
            Coefficients of streamfunction in spectral space

        Returns
        -------
        zeta_coeffs : BarotropicState
            Coefficients of vorticity in spectral space
        """

        zeta_coeffs = cp.zeros_like(psi_coeffs)

        # Multiply by eigenvalues (skip l=0)
        zeta_coeffs[1:, :] = psi_coeffs[1:, :] * self.laplacian_eig[1:, None]

        return BarotropicState(zeta_coeffs)


    def tendency(self, vrt_state: BarotropicState, forcing_coeffs) -> cp.ndarray:
        """
        Compute dζ/dt in spectral space

        Parameters
        ----------
        zeta : cp.ndarray
            Current vorticity coefficients
        forcing_coeffs : cp.ndarray, optional
            Forcing in spectral space

        Returns
        -------
        zeta_new_coeffs : cp.ndarray
            dζ/dt in spectral space
        """

        zeta_c = vrt_state.coeffs

        if forcing_coeffs is None:
            forcing_coeffs = cp.zeros_like(zeta_c)

        # Get stream function
        psi_c = self.vorticity_to_streamfunction(vrt_state)

        # Get absolute vorticity η = ζ + f
        zeta_grid = self.sh.inv_transform(zeta_c)
        eta_grid = zeta_grid + self.f
        eta_c = self.sh.transform(eta_grid)

        # # Compute advection: -J(ψ, η)
        jacobian_grid = self.so.jacobian_pseudospectral(psi_c, eta_c)
        # jacobian_grid = self.so.jacobian_pseudospectral(psi_c, eta_c, sin_theta=self.grid.sincolat)
        advection_c = self.sh.transform(-jacobian_grid)

        # Compute diffusion: ν∇²ζ
        diffusion_c = self.nu * self.laplacian_eig[:, None] * zeta_c

        # Total tendency
        dzeta_dt = advection_c + diffusion_c + forcing_coeffs

        # Enforce mass conservation: global mean vorticity tendency must be zero
        # This prevents drift in Total Circulation Γ due to aliasing
        dzeta_dt[0, :] = 0.0

        return dzeta_dt


    def step_rk4(self, zeta_coeffs, dt, forcing_coeffs=None):
        """
        Fourth-order Runge-Kutta time step.

        Parameters
        ----------
        zeta_coeffs : cp.ndarray
            Current vorticity coefficients
        dt : float
            Time step [s]
        forcing_coeffs : cp.ndarray, optional
            Forcing in spectral space

        Returns
        -------
        zeta_new_coeffs : cp.ndarray
            Updated vorticity coefficients
        """
        if forcing_coeffs is None:
            forcing_coeffs = cp.zeros_like(zeta_coeffs)

        # RK4 stages
        k1 = self.tendency(zeta_coeffs, forcing_coeffs)
        k2 = self.tendency(zeta_coeffs + 0.5 * dt * k1, forcing_coeffs)
        k3 = self.tendency(zeta_coeffs + 0.5 * dt * k2, forcing_coeffs)
        k4 = self.tendency(zeta_coeffs + dt * k3, forcing_coeffs)

        zeta_new = zeta_coeffs + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return zeta_new

    def step_leapfrog(self, zeta_prev, zeta_curr, dt, forcing_coeffs=None):
        """
        Leapfrog time step (second-order, time-reversible).
        Requires two previous states.
        """
        if forcing_coeffs is None:
            forcing_coeffs = cp.zeros_like(zeta_curr)

        # Get tendency at current time
        psi_c = self.vorticity_to_streamfunction(zeta_curr)
        zeta_grid = self.sh.inv_transform(zeta_curr)
        eta_grid = zeta_grid + self.f
        eta_c = self.sh.transform(eta_grid)

        J = self.so.jacobian_pseudospectral(psi_c, eta_c)
        advection_c = self.sh.transform(-J)
        diffusion_c = self.nu * self.laplacian_eig[:, None] * zeta_curr

        dzeta_dt = advection_c + diffusion_c + forcing_coeffs

        # Enforce mass conservation
        dzeta_dt[0, :] = 0.0

        # Leapfrog step
        zeta_next = zeta_prev + 2 * dt * dzeta_dt

        return zeta_next
    