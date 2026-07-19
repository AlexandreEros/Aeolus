"""Dry hydrostatic primitive-equation core on the sphere — FOUNDATION ONLY.

Scope (docs/PRIMITIVE_EQUATIONS_DESIGN.md): the spectral state
representation, hard state validation, hydrostatic geopotential
reconstruction, the discrete column continuity diagnostics (surface-
pressure tendency, interface sigma-velocity, layer mass closure), and the
characteristic-speed estimate for the future CFL controller.

**There is deliberately no ``tendency()`` method.** The prognostic
tendencies are a separate milestone, gated on this foundation's tests; no
placeholder physics that silently returns zero exists here. What IS
implemented is real, tested machinery the tendency will be built from.

Formulation summary (full derivation and sign conventions in the design
doc): sigma = p/p_s vertical coordinate, Lorenz staggering, prognostic
relative vorticity zeta_k, divergence delta_k, temperature T_k at the K
full levels and ln(p_s) at the surface; diagnostic geopotential from the
Simmons–Burridge hydrostatic recursion; diagnostic sigma_dot from discrete
column continuity with structural top/bottom impermeability.

State layout: ONE complex (3K+1, l_max+1, l_max+1) coefficient stack,
rows ``[zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, ln p_s]`` top to
bottom, so RK4 stage arithmetic will be the plain array expression
``run.engine.rk4_step_array`` already implements. Coefficients follow the
repository convention: complex orthonormal spherical harmonics, axis 0 =
degree l, axis 1 = order m >= 0; a constant field c has a_{00} =
c * sqrt(4*pi).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cupy as cp

from planetary_sandbox.planet import Planet
from .sigma_coordinate import (SigmaGrid, column_mass_tendency,
                               hydrostatic_geopotential,
                               interface_sigma_dot, layer_mass_residual)

#: Dry-air constants (design doc Section 1; fixed so tests cannot drift).
R_DRY = 287.04          # J kg^-1 K^-1
CP_DRY = 1004.64        # J kg^-1 K^-1
KAPPA_DRY = R_DRY / CP_DRY
GAMMA_DRY = CP_DRY / (CP_DRY - R_DRY)

PROGNOSTICS = ("zeta", "delta", "temperature", "ln_ps")


class PrimitiveEquationsStateError(ValueError):
    """A primitive-equation state violated a hard physical/numerical rule."""


@dataclass
class PrimitiveEquationsState:
    """Spectral PE state: one (3*nlev+1, l_max+1, l_max+1) complex stack.

    Rows 0..K-1 = zeta (top to bottom), K..2K-1 = delta, 2K..3K-1 = T,
    row 3K = ln(p_s). ``nlev`` is inferred from the leading axis.
    """

    coeffs: cp.ndarray

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 3 or (self.coeffs.shape[0] - 1) % 3 != 0 \
                or self.coeffs.shape[0] < 4 \
                or self.coeffs.shape[1] != self.coeffs.shape[2]:
            raise PrimitiveEquationsStateError(
                "primitive-equation coefficients must have shape "
                f"(3*nlev+1, l_max+1, l_max+1) with nlev >= 1, got "
                f"{self.coeffs.shape}")

    @property
    def nlev(self) -> int:
        return (self.coeffs.shape[0] - 1) // 3

    @property
    def zeta(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) relative-vorticity coefficients (s^-1)."""
        return self.coeffs[0:self.nlev]

    @property
    def delta(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) divergence coefficients (s^-1)."""
        return self.coeffs[self.nlev:2 * self.nlev]

    @property
    def temperature(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) full-temperature coefficients (K)."""
        return self.coeffs[2 * self.nlev:3 * self.nlev]

    @property
    def ln_ps(self) -> cp.ndarray:
        """(l_max+1, l_max+1) ln(p_s) coefficients (p_s in Pa)."""
        return self.coeffs[3 * self.nlev]

    @classmethod
    def from_fields(cls, zeta_lm: cp.ndarray, delta_lm: cp.ndarray,
                    temperature_lm: cp.ndarray,
                    ln_ps_lm: cp.ndarray) -> "PrimitiveEquationsState":
        """Assemble a state from per-variable coefficient arrays.

        ``zeta_lm``, ``delta_lm``, ``temperature_lm`` have shape
        (nlev, l_max+1, l_max+1); ``ln_ps_lm`` has shape
        (l_max+1, l_max+1).
        """
        parts = [cp.asarray(a, dtype=cp.complex128)
                 for a in (zeta_lm, delta_lm, temperature_lm)]
        lnps = cp.asarray(ln_ps_lm, dtype=cp.complex128)
        return cls(cp.concatenate(parts + [lnps[None]], axis=0))

    @classmethod
    def zeros(cls, l_max: int, nlev: int) -> "PrimitiveEquationsState":
        n = l_max + 1
        return cls(cp.zeros((3 * nlev + 1, n, n), dtype=cp.complex128))


def isothermal_rest_state(l_max: int, nlev: int, *,
                          temperature: float,
                          surface_pressure: float
                          ) -> PrimitiveEquationsState:
    """Exactly resting, horizontally uniform isothermal state.

    zeta = delta = 0 everywhere; T_k = ``temperature`` at every level;
    p_s = ``surface_pressure`` (Pa) everywhere. Constant fields are the
    pure (0,0) mode with coefficient value * sqrt(4*pi) (repository
    orthonormal-SH convention).
    """
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(
            f"temperature must be finite and > 0, got {temperature}")
    if not (math.isfinite(surface_pressure) and surface_pressure > 0):
        raise ValueError(
            f"surface_pressure must be finite and > 0, got "
            f"{surface_pressure}")
    state = PrimitiveEquationsState.zeros(l_max, nlev)
    monopole = math.sqrt(4.0 * math.pi)
    state.temperature[:, 0, 0] = temperature * monopole
    state.ln_ps[0, 0] = math.log(surface_pressure) * monopole
    return state


class PrimitiveEquationsModel:
    """Foundation operators for the dry hydrostatic PE core.

    Owns the horizontal machinery (per-level Helmholtz velocity
    reconstruction and spectral derivatives, on the same backend seams the
    BVE/SWE use) and the vertical machinery (a validated
    :class:`~planetary_sandbox.physics.sigma_coordinate.SigmaGrid` plus the
    hydrostatic and continuity column operators).

    The Helmholtz/derivative helpers intentionally mirror
    ``physics.shallow_water.ShallowWaterModel`` (same conventions, same
    metric identities) rather than importing them: the SWE core is frozen
    by its A/B guarantees and must not grow shared-code coupling in this
    milestone.
    """

    def __init__(self, planet: Planet, sigma: SigmaGrid, *,
                 r_dry: float = R_DRY, cp_dry: float = CP_DRY,
                 surface_geopotential_lm: cp.ndarray | None = None):
        if not (math.isfinite(r_dry) and r_dry > 0):
            raise ValueError(f"r_dry must be finite and > 0, got {r_dry}")
        if not (math.isfinite(cp_dry) and cp_dry > r_dry):
            raise ValueError(
                f"cp_dry must be finite and > r_dry, got {cp_dry}")

        self.planet = planet
        self.sh = planet.sh
        self.so = planet.so
        self.grid = planet.grid
        self.R = float(planet.params.radius)
        self.Omega = float(planet.params.angular_velocity)
        self.sigma = sigma
        self.nlev = sigma.nlev
        self.r_dry = float(r_dry)
        self.cp_dry = float(cp_dry)
        self.kappa = self.r_dry / self.cp_dry
        self.gamma = self.cp_dry / (self.cp_dry - self.r_dry)
        self.l_max = self.sh.l_max

        # Surface geopotential: fixed spectral field, representable from
        # day one (no topography yet -> zeros by default).
        n = self.l_max + 1
        if surface_geopotential_lm is None:
            self.phi_surface_lm = cp.zeros((n, n), dtype=cp.complex128)
        else:
            phi_s = cp.asarray(surface_geopotential_lm, dtype=cp.complex128)
            if phi_s.shape != (n, n):
                raise ValueError(
                    f"surface_geopotential_lm must have shape {(n, n)}, "
                    f"got {phi_s.shape}")
            if not bool(cp.isfinite(phi_s).all()):
                raise ValueError(
                    "surface_geopotential_lm contains NaN/Inf")
            self.phi_surface_lm = phi_s

        # Laplacian eigenvalues -l(l+1)/R^2 (l=0 exactly 0), as in BVE/SWE.
        l = cp.arange(self.l_max + 1, dtype=cp.float64)
        self.lap_eigs = -l * (l + 1.0) / self.R**2

        # Product space: same object the BVE Jacobian and SWE tendency use.
        self._ps = self.so.backend.product_space(self.so.product_quadrature)

        # State-grid cos(lat), clamped like the backend product spaces.
        coslat = getattr(self.grid, "coslat", None)
        if coslat is None:
            coslat = cp.cos(cp.asarray(self.grid.point_latitudes))
        self._state_coslat = cp.maximum(cp.asarray(coslat, cp.float64), 1e-8)

    # ------------------------------------------------------------------
    # Per-level horizontal helpers (SWE conventions, level-looped)
    # ------------------------------------------------------------------

    def _inv_laplacian(self, coeffs: cp.ndarray) -> cp.ndarray:
        """Solve laplacian(x) = coeffs per 2-D slice; l=0 mode zeroed."""
        out = cp.zeros_like(coeffs)
        out[..., 1:, :] = coeffs[..., 1:, :] / self.lap_eigs[1:, None]
        return out

    def _deriv_fields(self, sh, coeffs: cp.ndarray
                      ) -> tuple[cp.ndarray, cp.ndarray]:
        """((1/R) d/dlambda, (1/R) sin(theta) d/dtheta) of one 2-D field."""
        lam = sh.inv_transform(self.so.d_lambda_coeffs(coeffs)).real
        snt = sh.inv_transform(
            self.so.sin_theta_d_theta_coeffs(coeffs)).real / self.R
        return lam, snt

    def _wind_from(self, sh, coslat: cp.ndarray, psi_lm: cp.ndarray,
                   chi_lm: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """u = k x grad(psi) + grad(chi) on the sampling owned by ``sh``."""
        psi_lam, psi_snt = self._deriv_fields(sh, psi_lm)
        chi_lam, chi_snt = self._deriv_fields(sh, chi_lm)
        u = (psi_snt + chi_lam) / coslat
        v = (psi_lam - chi_snt) / coslat
        return u, v

    def wind_on_state_grid(self, state: PrimitiveEquationsState
                           ) -> tuple[cp.ndarray, cp.ndarray]:
        """Per-level eastward/northward velocity on the state sampling.

        Returns (u, v), each shape (nlev,) + grid-field shape.
        """
        us, vs = [], []
        for k in range(self.nlev):
            psi_lm = self._inv_laplacian(state.zeta[k])
            chi_lm = self._inv_laplacian(state.delta[k])
            u, v = self._wind_from(self.sh, self._state_coslat, psi_lm,
                                   chi_lm)
            us.append(u)
            vs.append(v)
        return cp.stack(us), cp.stack(vs)

    # ------------------------------------------------------------------
    # Diagnostic reconstructions (design doc Sections 4–6)
    # ------------------------------------------------------------------

    def temperature_on_state_grid(self, state: PrimitiveEquationsState
                                  ) -> cp.ndarray:
        """(nlev,) + grid-shape full-level temperature fields (K)."""
        return cp.stack([self.sh.inv_transform(state.temperature[k]).real
                         for k in range(self.nlev)])

    def geopotential_fields(self, state: PrimitiveEquationsState
                            ) -> dict:
        """Hydrostatic geopotential on the state grid (Simmons–Burridge).

        Returns ``phi_full`` (nlev, ...) full-level geopotential,
        ``phi_below`` (nlev, ...) the interface-below-layer geopotential
        (``phi_below[-1]`` is exactly the surface geopotential field), and
        ``phi_surface`` itself. The sigma = 0 interface value is never
        computed (design doc Section 4).
        """
        T = self.temperature_on_state_grid(state)
        phi_s = self.sh.inv_transform(self.phi_surface_lm).real
        phi_full, phi_below = hydrostatic_geopotential(
            self.sigma, T, phi_s, self.r_dry)
        return {"phi_full": phi_full, "phi_below": phi_below,
                "phi_surface": phi_s}

    def continuity_diagnostics(self, state: PrimitiveEquationsState
                               ) -> dict:
        """Discrete column continuity on the state grid.

        Computes G_k = delta_k + V_k . grad(ln p_s) pointwise per level
        (the same metric identity the SWE advection uses:
        u.grad(q) = (u q_lam - v q_snt)/cos(lat)), then the column
        operators of ``physics.sigma_coordinate``:

        * ``dlnps_dt``       d(ln p_s)/dt grid field (s^-1)
        * ``sigma_dot``      (nlev+1, ...) interface sigma-velocity, top
                             and bottom rows structurally zero
        * ``layer_residual`` (nlev, ...) discrete layer mass-budget
                             residual (round-off by construction)
        * ``g_full``         (nlev, ...) the integrand G_k
        * ``max_abs_layer_residual`` scalar float, the closure diagnostic

        This is a *diagnostic* operator (state grid, no product-grid
        dealiasing); the future tendency will evaluate its nonlinear
        products on the backend product grid like the SWE does.
        """
        lnps_lam, lnps_snt = self._deriv_fields(self.sh, state.ln_ps)
        u, v = self.wind_on_state_grid(state)
        gs = []
        for k in range(self.nlev):
            adv = (u[k] * lnps_lam - v[k] * lnps_snt) / self._state_coslat
            delta_g = self.sh.inv_transform(state.delta[k]).real
            gs.append(delta_g + adv)
        g_full = cp.stack(gs)

        dlnps_dt = column_mass_tendency(self.sigma, g_full)
        sigma_dot = interface_sigma_dot(self.sigma, g_full)
        residual = layer_mass_residual(self.sigma, g_full)
        return {
            "g_full": g_full,
            "dlnps_dt": dlnps_dt,
            "sigma_dot": sigma_dot,
            "layer_residual": residual,
            "max_abs_layer_residual": float(cp.abs(residual).max()),
        }

    def surface_pressure_on_state_grid(self, state: PrimitiveEquationsState
                                       ) -> cp.ndarray:
        """p_s = exp(ln p_s) on the state grid (Pa); positive when finite."""
        return cp.exp(self.sh.inv_transform(state.ln_ps).real)

    # ------------------------------------------------------------------
    # Characteristic speed (design doc Section 10)
    # ------------------------------------------------------------------

    def temperature_extrema(self, state: PrimitiveEquationsState
                            ) -> tuple[float, float]:
        """(min, max) of T over EVERY sampling the model evaluates on.

        Scans every full level on the state transform and the product-space
        transform (SWE positivity-envelope precedent: a field positive at
        every state point can still dip negative between them).
        """
        samplings = [self.sh]
        if self._ps.sh is not self.sh:
            samplings.append(self._ps.sh)
        lo = math.inf
        hi = -math.inf
        for sh in samplings:
            for k in range(self.nlev):
                g = sh.inv_transform(state.temperature[k]).real
                lo = min(lo, float(g.min()))
                hi = max(hi, float(g.max()))
        return lo, hi

    def max_characteristic_speed(self, state: PrimitiveEquationsState
                                 ) -> float:
        """max_k max|V_k| + sqrt(gamma_d * R_d * T_max) (m/s).

        The gravity-wave term bounds the Lamb/external-mode speed with the
        temperature maximum over every model sampling; the sum-of-maxima
        form is deliberately conservative (design doc Section 10). The
        maximum is clamped at zero only to keep the estimate NaN-free; a
        non-positive temperature is a hard failure of validate_state().
        """
        u, v = self.wind_on_state_grid(state)
        wind_max = float(cp.sqrt(u * u + v * v).max())
        _, t_max = self.temperature_extrema(state)
        return wind_max + math.sqrt(self.gamma * self.r_dry
                                    * max(t_max, 0.0))

    # ------------------------------------------------------------------
    # Validation (design doc Section 9)
    # ------------------------------------------------------------------

    #: Monopole tolerance relative to the per-level spectral norm (same
    #: value as the SWE core).
    _MONOPOLE_RTOL = 1e-10

    def validate_state(self, state: PrimitiveEquationsState, *,
                       context: str = "") -> None:
        """Raise PrimitiveEquationsStateError on any hard violation.

        Checks, in order: coefficient-array shape against this model's
        l_max and nlev; finiteness of every coefficient; per-level zeta and
        delta monopoles (the global circulation and integrated divergence
        of a single-valued velocity field are identically zero); strict
        temperature positivity over every model sampling; and finiteness of
        the synthesized p_s = exp(ln p_s) (positivity is automatic for the
        prognostic ln p_s — overflow to Inf is the failure mode).
        """
        where = f" {context}" if context else ""
        n = self.l_max + 1
        expected = (3 * self.nlev + 1, n, n)
        if state.coeffs.shape != expected:
            raise PrimitiveEquationsStateError(
                f"state shape {state.coeffs.shape} does not match the "
                f"model's expected {expected}{where}")
        if not bool(cp.isfinite(state.coeffs).all()):
            raise PrimitiveEquationsStateError(
                f"primitive-equation state contains NaN/Inf "
                f"coefficients{where}")

        for name, stack in (("zeta", state.zeta), ("delta", state.delta)):
            for k in range(self.nlev):
                field = stack[k]
                monopole = float(cp.abs(field[0, 0]))
                scale = float(cp.linalg.norm(field))
                if monopole > self._MONOPOLE_RTOL * max(scale, 1e-30):
                    raise PrimitiveEquationsStateError(
                        f"{name} monopole is nonzero at level {k + 1}"
                        f"{where}: |a00| = {monopole:g} (field norm "
                        f"{scale:g}); the global mean of {name} must stay "
                        "exactly zero on every level")

        t_min, _ = self.temperature_extrema(state)
        if not (t_min > 0.0):
            raise PrimitiveEquationsStateError(
                f"temperature is not strictly positive{where}: min(T) = "
                f"{t_min:g} K over the state/product samplings")

        ps = self.surface_pressure_on_state_grid(state)
        if not bool(cp.isfinite(ps).all()):
            raise PrimitiveEquationsStateError(
                f"surface pressure exp(ln p_s) overflowed to a non-finite "
                f"value on the state grid{where}")
