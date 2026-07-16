"""Rotating shallow-water dynamical core on the sphere.

Solves the flat-bottom, inviscid rotating shallow-water equations in
vorticity-divergence (vector-invariant) form, with the prognostic variables

    zeta   relative vorticity          (s^-1)
    delta  horizontal divergence       (s^-1)
    phi    PERTURBATION geopotential   (m^2 s^-2)

where the total geopotential is ``Phi = Phi0 + phi`` with the constant
resting geopotential ``Phi0 = g * H`` (g = gravity, H = mean fluid depth).
The global mean of Phi lives entirely in Phi0: the perturbation monopole
``phi_00`` is pinned to zero, exactly like the BVE pins circulation.

Governing equations (Williamson et al. 1992; Hack & Jakob 1992), with
``eta = zeta + f``, ``f = 2*Omega*sin(lat)``, ``K = |u|^2 / 2``:

    d(zeta)/dt = -div(eta * u)
    d(delta)/dt =  k . curl(eta * u) - laplacian(K + phi)
    d(phi)/dt  = -Phi0 * delta - div(phi * u)

Velocity is reconstructed from the Helmholtz decomposition
``u = k x grad(psi) + grad(chi)`` with ``laplacian(psi) = zeta`` and
``laplacian(chi) = delta``; in spectral space
``psi_lm = -a^2/(l(l+1)) zeta_lm`` and ``chi_lm = -a^2/(l(l+1)) delta_lm``
(the repository's inverse-Laplacian convention, l=0 modes zeroed).

Discretization of the nonlinear terms
-------------------------------------
The flux divergence/curl terms are evaluated pseudo-spectrally on the
backend's *product space* (the same fine-grid product machinery the BVE
Jacobian uses), via the pointwise-expanded identities

    div(q u)      = u . grad(q) + q * delta
    k . curl(q u) = q * zeta + (grad(q) x u) . k

which avoid differentiating grid products (no second transform round trip)
and use exactly the metric handling of ``jacobian_pseudospectral``:
with the derivative fields  q_lam = (1/a) dq/dlambda  and
q_snt = (1/a) sin(theta) dq/dtheta = -(cos(lat)/a) dq/dlat,

    u . grad(q)        = (u * q_lam - v * q_snt) / cos(lat)
    (grad(q) x u) . k  = (v * q_lam + u * q_snt) / cos(lat)

In the pure-rotational limit (delta = 0, phi = 0) the zeta tendency reduces
pointwise to the exact expression of ``jacobian_pseudospectral(psi, eta)``,
so the shallow-water core degenerates to the BVE core identically (a tested
invariant). Each nonlinear product (eta*u, phi*u, K) is analyzed once on the
product grid and truncated once with the 2/3 rule; the linear terms
(-Phi0*delta, -laplacian(phi), the Laplacian of the analyzed K) are exact
diagonal spectral operations.

Every tendency has its l=0 row hard-zeroed, so the zeta, delta, and phi
monopoles (circulation, integrated divergence, mean perturbation
geopotential == layer mass anomaly) are conserved to round-off. The optional
scale-selective ``nabla^4`` hyperdiffusion has eigenvalue
``-nu4 * (l(l+1)/a^2)^2``, which is exactly zero at l = 0, so it can never
modify the conserved monopoles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cupy as cp

from planetary_sandbox.planet import Planet

#: Standard gravity used by the Williamson et al. (1992) test suite (m/s^2).
WILLIAMSON_GRAVITY = 9.80616

#: Stack indices of the prognostic variables in ShallowWaterState.coeffs.
ZETA, DELTA, PHI = 0, 1, 2
PROGNOSTICS = ("zeta", "delta", "phi")


class ShallowWaterStateError(ValueError):
    """A shallow-water state violated a hard physical/numerical constraint."""


@dataclass
class ShallowWaterState:
    """Spectral shallow-water state: one (3, l_max+1, l_max+1) complex stack.

    Index 0 = zeta, 1 = delta, 2 = phi (PERTURBATION geopotential). Keeping
    the three prognostics in a single array makes the RK4 stage arithmetic a
    plain array expression (see ``run.engine.rk4_step_array``).
    """

    coeffs: cp.ndarray

    @classmethod
    def from_fields(cls, zeta_lm: cp.ndarray, delta_lm: cp.ndarray,
                    phi_lm: cp.ndarray) -> "ShallowWaterState":
        return cls(cp.stack([
            cp.asarray(zeta_lm, dtype=cp.complex128),
            cp.asarray(delta_lm, dtype=cp.complex128),
            cp.asarray(phi_lm, dtype=cp.complex128),
        ]))

    @classmethod
    def zeros(cls, l_max: int) -> "ShallowWaterState":
        n = l_max + 1
        return cls(cp.zeros((3, n, n), dtype=cp.complex128))

    @property
    def zeta(self) -> cp.ndarray:
        return self.coeffs[ZETA]

    @property
    def delta(self) -> cp.ndarray:
        return self.coeffs[DELTA]

    @property
    def phi(self) -> cp.ndarray:
        return self.coeffs[PHI]


class ShallowWaterModel:
    """Rotating shallow-water tendencies, velocity reconstruction, and checks.

    Parameters
    ----------
    planet : Planet
        Grid + transform + spectral-operator bundle (owns the product space).
    gravity : float
        Surface gravity g (m/s^2), > 0.
    mean_depth : float
        Mean (resting) fluid depth H (m), > 0. ``Phi0 = gravity * mean_depth``.
    hyperdiffusion_nu4 : float
        Optional scale-selective ``-nu4 * nabla^4`` damping coefficient
        (m^4/s), >= 0, applied to all three prognostics. Its spectral
        eigenvalue is exactly zero at l = 0, so conserved monopoles are
        untouched. Default 0 (the inviscid equations).
    """

    def __init__(self, planet: Planet, *,
                 gravity: float = WILLIAMSON_GRAVITY,
                 mean_depth: float,
                 hyperdiffusion_nu4: float = 0.0):
        if not (math.isfinite(gravity) and gravity > 0):
            raise ValueError(f"gravity must be finite and > 0, got {gravity}")
        if not (math.isfinite(mean_depth) and mean_depth > 0):
            raise ValueError(
                f"mean_depth must be finite and > 0, got {mean_depth}")
        if not (math.isfinite(hyperdiffusion_nu4) and hyperdiffusion_nu4 >= 0):
            raise ValueError(
                f"hyperdiffusion_nu4 must be finite and >= 0, "
                f"got {hyperdiffusion_nu4}")

        self.planet = planet
        self.sh = planet.sh
        self.so = planet.so
        self.grid = planet.grid
        self.R = float(planet.params.radius)
        self.Omega = float(planet.params.angular_velocity)
        self.gravity = float(gravity)
        self.mean_depth = float(mean_depth)
        self.phi0 = self.gravity * self.mean_depth
        self.nu4 = float(hyperdiffusion_nu4)
        self.l_max = self.sh.l_max

        # Exact spectral planetary vorticity: f = 2*Omega*sin(lat) is the pure
        # (1,0) mode with coefficient 2*Omega*sqrt(4*pi/3) (same construction
        # as the BVE core; avoids the lossy grid round trip, R-5).
        self.f_lm = cp.zeros((self.l_max + 1, self.l_max + 1),
                             dtype=cp.complex128)
        self.f_lm[1, 0] = 2.0 * self.Omega * math.sqrt(4.0 * math.pi / 3.0)

        # Laplacian eigenvalues -l(l+1)/R^2 (l=0 entry exactly 0) and the
        # inverse used by the Helmholtz solves (l=0 result pinned to zero,
        # matching BarotropicVorticity.vorticity_to_streamfunction).
        l = cp.arange(self.l_max + 1, dtype=cp.float64)
        self.lap_eigs = -l * (l + 1.0) / self.R**2

        # Product space: same object the BVE Jacobian uses (fine co-grid /
        # 3/2-rule Gauss grid / state grid, per backend + configuration).
        self._ps = self.so.backend.product_space(self.so.product_quadrature)

        # State-grid cos(lat), clamped like the backend product spaces, for
        # velocity reconstruction on the state sampling (diagnostics / CFL).
        coslat = getattr(self.grid, "coslat", None)
        if coslat is None:
            coslat = cp.cos(cp.asarray(self.grid.point_latitudes))
        self._state_coslat = cp.maximum(cp.asarray(coslat, cp.float64), 1e-8)

        self._trunc_cut = (2 * self.l_max) // 3

    # ------------------------------------------------------------------
    # Helmholtz decomposition / velocity reconstruction
    # ------------------------------------------------------------------

    def _inv_laplacian(self, coeffs: cp.ndarray) -> cp.ndarray:
        """Solve laplacian(x) = coeffs; the undetermined l=0 mode is zeroed."""
        out = cp.zeros_like(coeffs)
        out[1:, :] = coeffs[1:, :] / self.lap_eigs[1:, None]
        return out

    def helmholtz(self, state: ShallowWaterState
                  ) -> tuple[cp.ndarray, cp.ndarray]:
        """Return (psi_lm, chi_lm): streamfunction and velocity potential.

        psi_lm = -R^2/(l(l+1)) * zeta_lm,  chi_lm = -R^2/(l(l+1)) * delta_lm.
        """
        return (self._inv_laplacian(state.coeffs[ZETA]),
                self._inv_laplacian(state.coeffs[DELTA]))

    def _deriv_fields(self, sh, coeffs: cp.ndarray
                      ) -> tuple[cp.ndarray, cp.ndarray]:
        """Synthesize ((1/R) d/dlambda, (1/R) sin(theta) d/dtheta) of a field.

        ``sh`` selects the sampling (state transform or product-space
        transform); both derivative operators act in spectral space, so the
        fields are evaluated exactly at the requested points.
        """
        lam = sh.inv_transform(self.so.d_lambda_coeffs(coeffs)).real
        snt = sh.inv_transform(
            self.so.sin_theta_d_theta_coeffs(coeffs)).real / self.R
        return lam, snt

    def _wind_from(self, sh, coslat: cp.ndarray, psi_lm: cp.ndarray,
                   chi_lm: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """u = k x grad(psi) + grad(chi) on the sampling owned by ``sh``.

        With psi_snt = -(cos(lat)/R) dpsi/dlat and chi_lam = (1/R) dchi/dlambda:
            u (east)  = -dpsi/dlat / R  + dchi/dlambda / (R cos(lat))
                      = (psi_snt + chi_lam) / cos(lat)
            v (north) =  dpsi/dlambda / (R cos(lat)) + dchi/dlat / R
                      = (psi_lam - chi_snt) / cos(lat)
        """
        psi_lam, psi_snt = self._deriv_fields(sh, psi_lm)
        chi_lam, chi_snt = self._deriv_fields(sh, chi_lm)
        u = (psi_snt + chi_lam) / coslat
        v = (psi_lam - chi_snt) / coslat
        return u, v

    def wind_on_state_grid(self, state: ShallowWaterState
                           ) -> tuple[cp.ndarray, cp.ndarray]:
        """Eastward/northward velocity (m/s) on the state sampling."""
        psi_lm, chi_lm = self.helmholtz(state)
        return self._wind_from(self.sh, self._state_coslat, psi_lm, chi_lm)

    # ------------------------------------------------------------------
    # Tendencies
    # ------------------------------------------------------------------

    def _truncate(self, coeffs: cp.ndarray) -> cp.ndarray:
        """2/3-rule spectral truncation of an analyzed product (in place)."""
        cut = self._trunc_cut
        coeffs[cut + 1:, :] = 0.0
        coeffs[:, cut + 1:] = 0.0
        return coeffs

    def tendency(self, coeffs: cp.ndarray) -> cp.ndarray:
        """d/dt of the (3, l_max+1, l_max+1) prognostic stack.

        Takes and returns the raw coefficient stack (not the dataclass) so it
        plugs directly into ``run.engine.rk4_step_array``.
        """
        zeta_c = coeffs[ZETA]
        delta_c = coeffs[DELTA]
        phi_c = coeffs[PHI]

        psi_c = self._inv_laplacian(zeta_c)
        chi_c = self._inv_laplacian(delta_c)
        eta_c = zeta_c + self.f_lm

        ps = self._ps
        sh_p = ps.sh
        coslat = ps.coslat

        # Velocity and scalar fields on the product sampling (direct basis
        # evaluation at the product points; no interpolation).
        u, v = self._wind_from(sh_p, coslat, psi_c, chi_c)
        eta_lam, eta_snt = self._deriv_fields(sh_p, eta_c)
        phi_lam, phi_snt = self._deriv_fields(sh_p, phi_c)
        eta_g = sh_p.inv_transform(eta_c).real
        zeta_g = sh_p.inv_transform(zeta_c).real
        delta_g = sh_p.inv_transform(delta_c).real
        phi_g = sh_p.inv_transform(phi_c).real

        # div(eta u) = u.grad(eta) + eta*delta ; u.grad(q) and (grad(q) x u).k
        # use the jacobian_pseudospectral metric convention (module docstring).
        adv_eta = (u * eta_lam - v * eta_snt) / coslat
        zeta_dot_g = -(adv_eta + eta_g * delta_g)

        # k.curl(eta u) = eta*zeta + (grad(eta) x u).k
        curl_g = eta_g * zeta_g + (v * eta_lam + u * eta_snt) / coslat

        kinetic_g = 0.5 * (u * u + v * v)

        adv_phi = (u * phi_lam - v * phi_snt) / coslat
        phi_dot_g = -(adv_phi + phi_g * delta_g)

        # Analyze each nonlinear product once; truncate once (2/3 rule).
        zeta_dot = self._truncate(sh_p.transform(zeta_dot_g))
        curl_c = self._truncate(sh_p.transform(curl_g))
        kinetic_c = self._truncate(sh_p.transform(kinetic_g))
        phi_dot = self._truncate(sh_p.transform(phi_dot_g)) \
            - self.phi0 * delta_c

        # Linear pressure/divergence terms are exact spectral operations.
        delta_dot = curl_c - self.lap_eigs[:, None] * (kinetic_c + phi_c)

        out = cp.stack([zeta_dot, delta_dot, phi_dot])

        if self.nu4 > 0.0:
            # -nu4 * nabla^4: eigenvalue -nu4*(l(l+1)/R^2)^2, exactly 0 at l=0.
            out -= self.nu4 * (self.lap_eigs[None, :, None] ** 2) * coeffs

        # Pin the l=0 rows: conserves circulation, integrated divergence, and
        # the perturbation-geopotential monopole (i.e. total mass) exactly.
        out[:, 0, :] = 0.0
        return out

    def tendency_state(self, state: ShallowWaterState) -> ShallowWaterState:
        """Dataclass-in, dataclass-out convenience wrapper over tendency()."""
        return ShallowWaterState(self.tendency(state.coeffs))

    # ------------------------------------------------------------------
    # CFL characteristic speed
    # ------------------------------------------------------------------

    def characteristic_fields(self, state: ShallowWaterState) -> dict:
        """State-grid fields shared by the CFL estimate and diagnostics.

        Returns u, v (m/s), wind speed, total geopotential Phi0 + phi
        (m^2/s^2), and the characteristic speed |u| + sqrt(Phi0 + phi) used
        by the advective+gravity-wave CFL condition. The square root is
        evaluated on the clamped total geopotential purely to keep the
        estimate NaN-free; a genuinely non-positive total geopotential is a
        hard failure raised by validate_state().
        """
        u, v = self.wind_on_state_grid(state)
        wind = cp.sqrt(u * u + v * v)
        phi_total = self.phi0 + self.sh.inv_transform(state.coeffs[PHI]).real
        char_speed = wind + cp.sqrt(cp.maximum(phi_total, 0.0))
        return {"u": u, "v": v, "wind_speed": wind,
                "phi_total": phi_total, "char_speed": char_speed}

    def max_characteristic_speed(self, state: ShallowWaterState) -> float:
        """max(|u| + sqrt(Phi0 + phi)) over the state grid (m/s).

        This — not sqrt(phi) of the perturbation — is the model's
        characteristic-speed estimate handed to the model-independent
        adaptive-timestep controller (run.engine.advective_cfl_timestep).
        """
        return float(cp.max(self.characteristic_fields(state)["char_speed"]))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    #: Monopole tolerance relative to the field's spectral norm (the
    #: tendencies pin l=0 exactly; anything above round-off is a defect).
    _MONOPOLE_RTOL = 1e-10

    def validate_state(self, state: ShallowWaterState, *,
                       context: str = "") -> None:
        """Raise ShallowWaterStateError on any hard constraint violation.

        Checks, in order: finiteness of every coefficient (NaN/Inf), the
        three conserved monopoles (zeta, delta, phi must have zero global
        mean), and strict positivity of the total geopotential
        ``Phi0 + phi`` (equivalently, positive fluid depth). Floating-point
        time stagnation is detected independently by the integration engine
        (FloatingPointError from the scheduler).
        """
        where = f" {context}" if context else ""
        coeffs = state.coeffs
        if not bool(cp.isfinite(coeffs).all()):
            raise ShallowWaterStateError(
                f"shallow-water state contains NaN/Inf coefficients{where}")

        for idx, name in enumerate(PROGNOSTICS):
            field = coeffs[idx]
            monopole = float(cp.abs(field[0, 0]))
            scale = float(cp.linalg.norm(field))
            if monopole > self._MONOPOLE_RTOL * max(scale, 1e-30):
                raise ShallowWaterStateError(
                    f"{name} monopole is nonzero{where}: |a00| = {monopole:g} "
                    f"(field norm {scale:g}); the global mean of {name} must "
                    "stay exactly zero"
                    + (" (the mean geopotential is represented by Phi0, not "
                       "the prognostic phi)" if name == "phi" else ""))

        phi_total_min = self.phi0 + float(
            cp.min(self.sh.inv_transform(coeffs[PHI]).real))
        if not (phi_total_min > 0.0):
            raise ShallowWaterStateError(
                f"total geopotential is not strictly positive{where}: "
                f"min(Phi0 + phi) = {phi_total_min:g} m^2/s^2 "
                f"(Phi0 = {self.phi0:g}); the fluid depth has collapsed")
