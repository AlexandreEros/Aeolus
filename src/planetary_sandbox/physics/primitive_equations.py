"""Dry hydrostatic primitive-equation core on the sphere.

Scope (docs/PRIMITIVE_EQUATIONS_DESIGN.md): the spectral state
representation, hard state validation, hydrostatic geopotential
reconstruction, the discrete column continuity diagnostics (surface-
pressure tendency, interface sigma-velocity, layer mass closure), the
characteristic-speed estimate for the CFL controller, and — the tendency
milestone — the first true nonlinear explicit tendency: product-grid
reconstruction of every primitive-equation field, thermodynamic and
surface-pressure tendencies, the vector-invariant momentum (zeta/delta)
tendencies using the weak-form vector curl/divergence analysis (design
doc Section 8a), and the public :meth:`PrimitiveEquationsModel.tendency`
gated behind the Phase-6 verification battery (exact rest, BVE
degeneracy, monopole/continuity invariants, RK4 stage-validated
stability). Deliberately NOT here (still deferred): runner/CLI/config,
semi-implicit terms, T_ref split, hyperdiffusion, forcing, topography
experiments, long integrations.

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
from .sigma_coordinate import (SigmaGrid, column_energy_conversion,
                               column_mass_tendency, column_pressure_work,
                               hydrostatic_geopotential,
                               interface_sigma_dot, layer_mass_residual,
                               omega_over_p, vertical_advection,
                               vertical_sbp)

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

        # Exact spectral planetary vorticity: f = 2*Omega*sin(lat) is the
        # pure (1,0) mode (same construction as BVE/SWE; avoids the lossy
        # grid round trip, R-5).
        n_f = self.l_max + 1
        self.f_lm = cp.zeros((n_f, n_f), dtype=cp.complex128)
        self.f_lm[1, 0] = 2.0 * self.Omega * math.sqrt(4.0 * math.pi / 3.0)

        # Product space: same object the BVE Jacobian and SWE tendency use.
        self._ps = self.so.backend.product_space(self.so.product_quadrature)

        # State-grid cos(lat), clamped like the backend product spaces.
        coslat = getattr(self.grid, "coslat", None)
        if coslat is None:
            coslat = cp.cos(cp.asarray(self.grid.point_latitudes))
        self._state_coslat = cp.maximum(cp.asarray(coslat, cp.float64), 1e-8)

        # 2/3-rule truncation cut for analyzed nonlinear products (the
        # SWE policy, applied once per combined quantity).
        self._trunc_cut = (2 * self.l_max) // 3

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
        advs, gs = [], []
        for k in range(self.nlev):
            adv = (u[k] * lnps_lam - v[k] * lnps_snt) / self._state_coslat
            delta_g = self.sh.inv_transform(state.delta[k]).real
            advs.append(adv)
            gs.append(delta_g + adv)
        v_grad_lnps = cp.stack(advs)
        g_full = cp.stack(gs)

        dlnps_dt = column_mass_tendency(self.sigma, g_full)
        sigma_dot = interface_sigma_dot(self.sigma, g_full)
        residual = layer_mass_residual(self.sigma, g_full)
        return {
            "g_full": g_full,
            "v_grad_lnps": v_grad_lnps,
            "dlnps_dt": dlnps_dt,
            "sigma_dot": sigma_dot,
            "layer_residual": residual,
            "max_abs_layer_residual": float(cp.abs(residual).max()),
        }

    def energy_exchange_diagnostics(self, state: PrimitiveEquationsState,
                                    continuity: dict | None = None) -> dict:
        """Simmons–Burridge energy conversion and the exchange identity.

        Evaluates, on the state grid, the Section-7b operators against this
        state's discrete hydrostatic geopotential:

        * ``omega_over_p``       (nlev, ...) the energy-conserving
                                 (omega/p)_k (s^-1)
        * ``heating``            (nlev, ...) the thermodynamic-equation term
                                 (kappa T omega/p)_k (K/s)
        * ``conversion``         column integral sum Dsigma R_d T (omega/p)
                                 (m^2 s^-3 per unit column mass fraction)
        * ``work``               column-local pressure work
                                 sum Dsigma [R_d T A - (Phi - Phi_s) G]
        * ``energy_residual``    conversion - work (round-off by
                                 construction; the identity E_d)
        * ``max_abs_energy_residual``  scalar closure diagnostic

        ``continuity`` may pass a precomputed :meth:`continuity_diagnostics`
        result to avoid recomputing winds; it must belong to the same state.
        """
        diag = (self.continuity_diagnostics(state) if continuity is None
                else continuity)
        g_full = diag["g_full"]
        v_grad_lnps = diag["v_grad_lnps"]
        T = self.temperature_on_state_grid(state)
        phi = self.geopotential_fields(state)

        wp = omega_over_p(self.sigma, g_full, v_grad_lnps)
        heating = self.kappa * T * wp
        conversion = column_energy_conversion(
            self.sigma, T, g_full, v_grad_lnps, self.r_dry)
        work = column_pressure_work(
            self.sigma, T, phi["phi_full"], phi["phi_surface"], g_full,
            v_grad_lnps, self.r_dry)
        residual = conversion - work
        return {
            "omega_over_p": wp,
            "heating": heating,
            "conversion": conversion,
            "work": work,
            "energy_residual": residual,
            "max_abs_energy_residual": float(cp.abs(residual).max()),
        }

    def vertical_transport_diagnostics(self, state: PrimitiveEquationsState,
                                       continuity: dict | None = None
                                       ) -> dict:
        """Lorenz-grid vertical transport of u, v, T on the state grid.

        Evaluates the Section-7a centered advective operator against this
        state's continuity-consistent ``sigma_dot`` (the future tendency
        subtracts these fields):

        * ``sigma_dot_dU``, ``sigma_dot_dV``  (nlev, ...) component-wise
          (sigma_dot dV/dsigma)_k of the reconstructed grid winds (m/s^2)
        * ``sigma_dot_dT``                    (nlev, ...) temperature
          transport (K/s)
        * ``ke_exchange_lhs`` / ``ke_exchange_rhs`` / ``ke_exchange_residual``
          the kinetic-energy exchange relation (SBP diagonal summed over
          u and v): 2<u, V_adv(u)> + 2<v, V_adv(v)> against
          sum Dsigma (u^2 + v^2)(G + d ln p_s/dt), per column
        * ``max_abs_ke_exchange_residual`` scalar closure diagnostic

        A resting atmosphere returns exactly zero everywhere; a constant
        temperature yields bitwise-zero ``sigma_dot_dT`` even in flow.
        ``continuity`` may pass a precomputed :meth:`continuity_diagnostics`
        result belonging to the same state.
        """
        diag = (self.continuity_diagnostics(state) if continuity is None
                else continuity)
        g_full = diag["g_full"]
        sigma_dot = diag["sigma_dot"]
        u, v = self.wind_on_state_grid(state)
        T = self.temperature_on_state_grid(state)

        du = vertical_advection(self.sigma, sigma_dot, u)
        dv = vertical_advection(self.sigma, sigma_dot, v)
        dT = vertical_advection(self.sigma, sigma_dot, T)

        sbp_u = vertical_sbp(self.sigma, g_full, u, u)
        sbp_v = vertical_sbp(self.sigma, g_full, v, v)
        lhs = sbp_u["lhs"] + sbp_v["lhs"]
        rhs = sbp_u["rhs"] + sbp_v["rhs"]
        residual = lhs - rhs
        return {
            "sigma_dot_dU": du,
            "sigma_dot_dV": dv,
            "sigma_dot_dT": dT,
            "ke_exchange_lhs": lhs,
            "ke_exchange_rhs": rhs,
            "ke_exchange_residual": residual,
            "max_abs_ke_exchange_residual": float(cp.abs(residual).max()),
        }

    def surface_pressure_on_state_grid(self, state: PrimitiveEquationsState
                                       ) -> cp.ndarray:
        """p_s = exp(ln p_s) on the state grid (Pa); positive when finite."""
        return cp.exp(self.sh.inv_transform(state.ln_ps).real)

    # ------------------------------------------------------------------
    # Tendency-path product-grid reconstruction (tendency milestone,
    # Phase 1 — docs/PRIMITIVE_EQUATIONS_TENDENCY_HANDOFF.md)
    # ------------------------------------------------------------------

    def _tendency_product_fields(self, coeffs: cp.ndarray) -> dict:
        """Every primitive-equation field on the backend PRODUCT sampling.

        This is the tendency path's own reconstruction: nothing here reuses
        the state-grid diagnostic fields (`continuity_diagnostics` etc.),
        because the nonlinear tendency terms must be formed where they are
        analyzed — on the backend product grid — exactly like the SWE. The
        state-grid diagnostics and this path must agree in the band-limited
        exact Gauss case (tested), but are deliberately separate samplings.

        Takes the raw ``(3K+1, l_max+1, l_max+1)`` coefficient stack (the
        RK4-stage object). Every quantity participating in the continuity
        and Simmons–Burridge exchange identities is derived from the SAME
        product-grid ``g_full`` and ``sigma_dot`` so the discrete identities
        close on this sampling to round-off.

        Returns a dict of product-sampling fields, level-stacked where
        applicable (shape ``(nlev, n_product)`` / ``(nlev+1, ...)`` for
        ``sigma_dot``):

        ``u, v``                 per-level wind (m/s)
        ``zeta, delta``          per-level relative vorticity / divergence
        ``temperature``          per-level full temperature (K)
        ``lnps``                 ln(p_s) grid field
        ``lnps_lam, lnps_snt``   (1/R) d(lnps)/dlambda, (1/R) sin(theta)
                                 d(lnps)/dtheta derivative fields
        ``grad_lnps_u/v``        eastward/northward components of
                                 grad(ln p_s) (1/m)
        ``v_grad_lnps``          A_k = V_k . grad(ln p_s)
        ``g_full``               G_k = delta_k + A_k
        ``dlnps_dt``             -sum_k G_k Dsigma_k
        ``sigma_dot``            continuity-consistent interface velocity
        ``phi_full``             Simmons–Burridge geopotential from
                                 product-grid T and Phi_s
        ``phi_surface``          Phi_s on the product sampling
        ``omega_over_p``         energy-conserving (omega/p)_k from the
                                 same G and A
        ``sigma_dot_dU/dV/dT``   centered vertical advection of u, v, T
                                 against the same sigma_dot
        ``coslat``               product-sampling cos(lat) (clamped)
        """
        K = self.nlev
        zeta_c = coeffs[0:K]
        delta_c = coeffs[K:2 * K]
        temp_c = coeffs[2 * K:3 * K]
        lnps_c = coeffs[3 * K]

        ps = self._ps
        sh_p = ps.sh
        coslat = ps.coslat

        lnps_lam, lnps_snt = self._deriv_fields(sh_p, lnps_c)
        us, vs, zetas, deltas, temps, advs, gs = [], [], [], [], [], [], []
        for k in range(K):
            psi_lm = self._inv_laplacian(zeta_c[k])
            chi_lm = self._inv_laplacian(delta_c[k])
            u, v = self._wind_from(sh_p, coslat, psi_lm, chi_lm)
            zeta_g = sh_p.inv_transform(zeta_c[k]).real
            delta_g = sh_p.inv_transform(delta_c[k]).real
            t_g = sh_p.inv_transform(temp_c[k]).real
            adv = (u * lnps_lam - v * lnps_snt) / coslat
            us.append(u)
            vs.append(v)
            zetas.append(zeta_g)
            deltas.append(delta_g)
            temps.append(t_g)
            advs.append(adv)
            gs.append(delta_g + adv)
        u = cp.stack(us)
        v = cp.stack(vs)
        temperature = cp.stack(temps)
        v_grad_lnps = cp.stack(advs)
        g_full = cp.stack(gs)

        dlnps_dt = column_mass_tendency(self.sigma, g_full)
        sigma_dot = interface_sigma_dot(self.sigma, g_full)
        phi_surface = sh_p.inv_transform(self.phi_surface_lm).real
        phi_full, _ = hydrostatic_geopotential(
            self.sigma, temperature, phi_surface, self.r_dry)
        wp = omega_over_p(self.sigma, g_full, v_grad_lnps)

        fields = {
            "u": u,
            "v": v,
            "zeta": cp.stack(zetas),
            "delta": cp.stack(deltas),
            "temperature": temperature,
            "lnps": sh_p.inv_transform(lnps_c).real,
            "lnps_lam": lnps_lam,
            "lnps_snt": lnps_snt,
            "grad_lnps_u": lnps_lam / coslat,
            "grad_lnps_v": -lnps_snt / coslat,
            "v_grad_lnps": v_grad_lnps,
            "g_full": g_full,
            "dlnps_dt": dlnps_dt,
            "sigma_dot": sigma_dot,
            "phi_full": phi_full,
            "phi_surface": phi_surface,
            "omega_over_p": wp,
            "sigma_dot_dU": vertical_advection(self.sigma, sigma_dot, u),
            "sigma_dot_dV": vertical_advection(self.sigma, sigma_dot, v),
            "sigma_dot_dT": vertical_advection(self.sigma, sigma_dot,
                                               temperature),
            "coslat": coslat,
        }
        return fields

    def _truncate(self, coeffs: cp.ndarray) -> cp.ndarray:
        """2/3-rule spectral truncation of an analyzed product (in place)."""
        cut = self._trunc_cut
        coeffs[cut + 1:, :] = 0.0
        coeffs[:, cut + 1:] = 0.0
        return coeffs

    def _thermo_mass_tendencies(self, coeffs: cp.ndarray, fields: dict
                                ) -> tuple[cp.ndarray, cp.ndarray]:
        """Spectral thermodynamic and surface-pressure tendencies.

        Thermodynamic equation, per level on the product grid (design doc
        Section 1; fully explicit, full T — no T_ref split):

            dT/dt = -V . grad(T) - V_adv(T) + kappa T (omega/p)

        All three terms use the mutually consistent product-grid fields of
        ``fields`` (one reconstruction); their sum is analyzed ONCE per
        level and truncated ONCE (2/3 rule). Surface pressure:

            d(ln p_s)/dt = -sum_k G_k Dsigma_k

        analyzed once, truncated once. NEITHER monopole is zeroed: the
        global-mean temperature evolves through the conversion term and
        the global-mean ln p_s through the mass divergence (its drift is
        the monitored diagnostic, design doc Section 9).

        Returns ``(t_dot_lm, lnps_dot_lm)`` with shapes
        ``(nlev, l_max+1, l_max+1)`` and ``(l_max+1, l_max+1)``.
        """
        K = self.nlev
        temp_c = coeffs[2 * K:3 * K]
        sh_p = self._ps.sh
        coslat = fields["coslat"]

        t_dots = []
        for k in range(K):
            t_lam, t_snt = self._deriv_fields(sh_p, temp_c[k])
            adv = (fields["u"][k] * t_lam
                   - fields["v"][k] * t_snt) / coslat
            t_dot_g = (-adv - fields["sigma_dot_dT"][k]
                       + self.kappa * fields["temperature"][k]
                       * fields["omega_over_p"][k])
            t_dots.append(self._truncate(sh_p.transform(t_dot_g)))

        lnps_dot = self._truncate(sh_p.transform(fields["dlnps_dt"]))
        return cp.stack(t_dots), lnps_dot

    def _momentum_tendencies(self, coeffs: cp.ndarray, fields: dict
                             ) -> tuple[cp.ndarray, cp.ndarray]:
        """Spectral zeta and delta tendencies (vector-invariant form).

        Design doc Section 1, fully explicit (full T, no T_ref split).
        With eta = zeta + f and the nonlinear residual vector

            Z_k = (sigma_dot dV/dsigma)_k + R_d T_k grad(ln p_s)

        the per-level tendencies are

            d(zeta_k)/dt  = -div(eta_k V_k) - k . curl(Z_k)
            d(delta_k)/dt =  k . curl(eta_k V_k) - div(Z_k)
                             - lap(Phi_k + E_k)

        Term treatment:

        * ``eta V`` uses the SWE's tested pointwise expansions
          (div(eta V) = u.grad(eta) + eta*delta, k.curl(eta V) =
          eta*zeta + (grad(eta) x V).k), which preserve the exact
          BVE-degeneracy property;
        * ``Z`` (no pointwise expansion exists) goes through the
          production weak-form vector analysis
          (SpectralOperators.vector_curl_div_spectral, design doc
          Section 8a); a bitwise-zero Z analyzes to bitwise zero, so the
          BVE degeneracy survives this pathway exactly;
        * ``-lap(Phi + E)`` is an exact diagonal spectral operation:
          Phi_lm comes from the hydrostatic recursion applied directly to
          the spectral T (linear, hence exact — tested against the grid
          path), E = |V|^2/2 is analyzed once and truncated once;
        * one 2/3 truncation per assembled nonlinear quantity; the l = 0
          rows of BOTH outputs are hard-zeroed (per-level circulation and
          integrated divergence are conserved exactly). T and ln p_s
          monopoles are handled in _thermo_mass_tendencies (NOT zeroed).
        """
        K = self.nlev
        zeta_c = coeffs[0:K]
        temp_c = coeffs[2 * K:3 * K]
        sh_p = self._ps.sh
        coslat = fields["coslat"]

        # Exact spectral hydrostatic Phi from spectral T (linear).
        phi_full_lm, _ = hydrostatic_geopotential(
            self.sigma, temp_c, self.phi_surface_lm, self.r_dry)

        zeta_dots, delta_dots = [], []
        for k in range(K):
            eta_c = zeta_c[k] + self.f_lm
            eta_lam, eta_snt = self._deriv_fields(sh_p, eta_c)
            eta_g = sh_p.inv_transform(eta_c).real
            u = fields["u"][k]
            v = fields["v"][k]

            # div(eta V) and k.curl(eta V), SWE pointwise expansions.
            adv_eta = (u * eta_lam - v * eta_snt) / coslat
            div_eta_v = adv_eta + eta_g * fields["delta"][k]
            curl_eta_v = eta_g * fields["zeta"][k] \
                + (v * eta_lam + u * eta_snt) / coslat

            # Z: vertical momentum transport + full R_d T grad(ln p_s).
            rt = self.r_dry * fields["temperature"][k]
            z_east = fields["sigma_dot_dU"][k] + rt * fields["grad_lnps_u"]
            z_north = fields["sigma_dot_dV"][k] + rt * fields["grad_lnps_v"]
            curl_z, div_z = self.so.vector_curl_div_spectral(
                z_east, z_north, truncate=False)

            # Kinetic energy: analyzed once, truncated once.
            kinetic_c = self._truncate(
                sh_p.transform(0.5 * (u * u + v * v)))

            zeta_dot = self._truncate(
                sh_p.transform(-div_eta_v) - curl_z)
            delta_dot = self._truncate(
                sh_p.transform(curl_eta_v) - div_z) \
                - self.lap_eigs[:, None] * (kinetic_c + phi_full_lm[k])
            zeta_dots.append(zeta_dot)
            delta_dots.append(delta_dot)

        zeta_dot = cp.stack(zeta_dots)
        delta_dot = cp.stack(delta_dots)
        # Conserve per-level circulation and integrated divergence exactly.
        zeta_dot[:, 0, :] = 0.0
        delta_dot[:, 0, :] = 0.0
        return zeta_dot, delta_dot

    # ------------------------------------------------------------------
    # Public tendency (the first true nonlinear dry hydrostatic PE
    # tendency; exposed only behind the Phase-6 verification battery)
    # ------------------------------------------------------------------

    def tendency(self, coeffs: cp.ndarray) -> cp.ndarray:
        """d/dt of the (3K+1, l_max+1, l_max+1) prognostic stack.

        Fully explicit, unsplit dry hydrostatic primitive equations in
        vorticity–divergence form (design doc Section 1): no T_ref split,
        no semi-implicit terms, full T in R_d T grad(ln p_s), no
        hyperdiffusion. Row layout matches the state:
        ``[zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, ln p_s]``.

        Takes and returns raw coefficient stacks so it plugs directly
        into ``run.engine.rk4_step_array`` (use ``validate_state`` wrapped
        as the ``stage_validator``, as ``run/swe/runner.py`` does).

        All nonlinear terms are evaluated on the backend product sampling
        via one shared reconstruction (:meth:`_tendency_product_fields`);
        every quantity in the continuity and energy-exchange identities
        derives from the same product-grid G and sigma_dot. Exact
        properties (tested): an isothermal resting atmosphere returns
        exactly zero; the BVE-degenerate state reproduces the barotropic
        tendency per level; zeta/delta monopole rows are bitwise zero;
        T and ln p_s monopoles evolve freely.
        """
        fields = self._tendency_product_fields(coeffs)
        t_dot, lnps_dot = self._thermo_mass_tendencies(coeffs, fields)
        zeta_dot, delta_dot = self._momentum_tendencies(coeffs, fields)
        return cp.concatenate(
            [zeta_dot, delta_dot, t_dot, lnps_dot[None]], axis=0)

    def tendency_state(self, state: PrimitiveEquationsState
                       ) -> PrimitiveEquationsState:
        """Dataclass-in, dataclass-out convenience wrapper over tendency()."""
        return PrimitiveEquationsState(self.tendency(state.coeffs))

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
