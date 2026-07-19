"""Sigma-coordinate vertical grid and column operators (backend-agnostic).

This module owns the *vertical* half of the dry primitive-equation
foundation (docs/PRIMITIVE_EQUATIONS_DESIGN.md):

* :class:`SigmaGrid` — immutable, validated Lorenz-staggered vertical-grid
  metadata (interfaces, full levels, thicknesses, Simmons–Burridge
  hydrostatic coefficients);
* :func:`hydrostatic_geopotential` — Simmons & Burridge (1981) hydrostatic
  reconstruction (design doc Section 4);
* :func:`column_mass_tendency`, :func:`interface_sigma_dot` — the discrete
  column continuity equation (Sections 5–6), with *structural* top/bottom
  impermeability;
* :func:`layer_mass_residual` — the per-layer discrete mass-budget residual
  used to test column mass closure (Section 6).

Array-backend independence: the column operators touch their level-stacked
inputs only through arithmetic, indexing/slicing, and the ``cumsum`` method,
all of which NumPy and CuPy arrays implement identically, and combine levels
with Python-float coefficients from the (NumPy/CPU) grid metadata. The same
code therefore runs unchanged on ``numpy.ndarray`` and ``cupy.ndarray``
inputs — a tested contract, not an accident. CuPy is never imported here.

Index conventions (design doc Section 3): ``K = nlev`` full levels are
numbered top to bottom; 0-based Python index ``k`` is layer ``k+1`` of the
doc. ``interfaces[i]`` is the doc's ``sigma_{i+1/2}``, so ``interfaces[0] =
0.0`` (model top) and ``interfaces[K] = 1.0`` (surface), exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np


class SigmaGridError(ValueError):
    """The vertical-grid metadata violated a hard structural constraint."""


@dataclass(frozen=True)
class SigmaGrid:
    """Immutable Lorenz-staggered sigma-coordinate vertical grid.

    ``interfaces`` are the K+1 interface coordinates ``sigma_{k+1/2}``,
    top to bottom. They must be finite, strictly increasing, and pin the
    boundaries EXACTLY: ``interfaces[0] == 0.0`` and ``interfaces[-1] ==
    1.0`` (no tolerance — the structural impermeability and hydrostatic
    telescoping proofs in the design doc rely on exact endpoint values).
    """

    interfaces: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            cleaned = tuple(float(s) for s in self.interfaces)
        except (TypeError, ValueError):
            raise SigmaGridError(
                f"sigma interfaces must be real numbers, got "
                f"{self.interfaces!r}")
        if len(cleaned) < 2:
            raise SigmaGridError(
                f"need at least 2 interfaces (1 layer), got {len(cleaned)}")
        for i, s in enumerate(cleaned):
            if not math.isfinite(s):
                raise SigmaGridError(f"interfaces[{i}] = {s} is not finite")
        if cleaned[0] != 0.0:
            raise SigmaGridError(
                f"top interface must be exactly 0.0, got {cleaned[0]!r}")
        if cleaned[-1] != 1.0:
            raise SigmaGridError(
                f"bottom interface must be exactly 1.0, got {cleaned[-1]!r}")
        for i in range(len(cleaned) - 1):
            if not cleaned[i] < cleaned[i + 1]:
                raise SigmaGridError(
                    "sigma interfaces must be strictly increasing; got "
                    f"interfaces[{i}] = {cleaned[i]} >= interfaces[{i+1}] "
                    f"= {cleaned[i+1]}")
        object.__setattr__(self, "interfaces", cleaned)

    @classmethod
    def uniform(cls, nlev: int) -> "SigmaGrid":
        """Equal-thickness grid: interfaces k/nlev, k = 0..nlev."""
        if not isinstance(nlev, int) or isinstance(nlev, bool) or nlev < 1:
            raise SigmaGridError(f"nlev must be a positive int, got {nlev!r}")
        # Endpoints are exact (0/n and n/n); interior values are the usual
        # correctly rounded quotients.
        return cls(tuple(k / nlev for k in range(nlev + 1)))

    # ------------------------------------------------------------------
    # Derived metadata (all CPU/NumPy; Python floats feed the column ops)
    # ------------------------------------------------------------------

    @property
    def nlev(self) -> int:
        return len(self.interfaces) - 1

    @cached_property
    def full_levels(self) -> tuple[float, ...]:
        """Full-level sigma_k = (sigma_{k-1/2} + sigma_{k+1/2}) / 2."""
        s = self.interfaces
        return tuple(0.5 * (s[k] + s[k + 1]) for k in range(self.nlev))

    @cached_property
    def thickness(self) -> tuple[float, ...]:
        """Layer thickness Dsigma_k = sigma_{k+1/2} - sigma_{k-1/2} (> 0)."""
        s = self.interfaces
        return tuple(s[k + 1] - s[k] for k in range(self.nlev))

    @cached_property
    def interface_log_ratios(self) -> tuple[float, ...]:
        """ln(sigma_{k+1/2} / sigma_{k-1/2}) per layer; +inf for the top layer.

        The top-layer entry is mathematically +inf (sigma_{1/2} = 0). It is
        stored as ``math.inf`` and MUST never be multiplied into a result:
        the hydrostatic recursion stops below the top layer (the sigma = 0
        interface geopotential is deliberately never computed, design doc
        Section 4) and alpha_1 = ln 2 replaces it at the top full level.
        """
        s = self.interfaces
        out = [math.inf]
        for k in range(1, self.nlev):
            out.append(math.log(s[k + 1] / s[k]))
        return tuple(out)

    @cached_property
    def alpha(self) -> tuple[float, ...]:
        """Simmons–Burridge full-level coefficients alpha_k.

        alpha_1 = ln 2 (top layer, sigma_{1/2} = 0);
        alpha_k = 1 - (sigma_{k-1/2}/Dsigma_k) ln(sigma_{k+1/2}/sigma_{k-1/2}).
        """
        s = self.interfaces
        out = [math.log(2.0)]
        for k in range(1, self.nlev):
            dsig = self.thickness[k]
            out.append(1.0 - (s[k] / dsig) * self.interface_log_ratios[k])
        return tuple(out)

    def interfaces_array(self) -> np.ndarray:
        return np.asarray(self.interfaces, dtype=np.float64)

    def full_levels_array(self) -> np.ndarray:
        return np.asarray(self.full_levels, dtype=np.float64)

    def thickness_array(self) -> np.ndarray:
        return np.asarray(self.thickness, dtype=np.float64)


def _require_level_stack(grid: SigmaGrid, arr, name: str):
    """Check that ``arr`` has the level axis first; return it unchanged."""
    if getattr(arr, "ndim", 0) < 1 or arr.shape[0] != grid.nlev:
        raise ValueError(
            f"{name} must have shape (nlev={grid.nlev}, ...), got "
            f"{getattr(arr, 'shape', None)}")
    return arr


# ---------------------------------------------------------------------------
# Hydrostatic geopotential (design doc Section 4)
# ---------------------------------------------------------------------------

def hydrostatic_geopotential(grid: SigmaGrid, temperature, phi_surface,
                             r_dry: float):
    """Simmons–Burridge hydrostatic reconstruction of the geopotential.

    Parameters
    ----------
    temperature : array, shape (nlev, ...)
        Full-level temperature, top to bottom. Any trailing dimensions
        (grid points, or spectral-coefficient axes — the operator is linear
        in T, so it is valid in either space).
    phi_surface : array or scalar broadcastable to ``temperature[0]``
        Surface geopotential Phi_s (the sigma = 1 boundary condition).
    r_dry : float
        Dry-air gas constant (J kg^-1 K^-1).

    Returns
    -------
    (phi_full, phi_below) : arrays, each shape (nlev, ...)
        ``phi_below[k]`` is the geopotential of the interface BELOW layer k
        (the doc's Phi_{k+1/2}); ``phi_below[nlev-1]`` equals ``phi_surface``
        exactly. ``phi_full[k]`` is the full-level Phi_k. The sigma = 0
        interface geopotential is never computed (it is infinite for any
        atmosphere with a nonzero top-layer temperature).
    """
    T = _require_level_stack(grid, temperature, "temperature")
    K = grid.nlev
    r = float(r_dry)
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"r_dry must be finite and > 0, got {r_dry}")

    phi_below = T * 0.0
    phi_full = T * 0.0

    # Upward interface recursion from the surface boundary condition.
    phi_below[K - 1] = phi_below[K - 1] + phi_surface
    for k in range(K - 1, 0, -1):
        phi_below[k - 1] = (phi_below[k]
                            + (r * grid.interface_log_ratios[k]) * T[k])
    # Full levels: Phi_k = Phi_{k+1/2} + alpha_k * R_d * T_k.
    for k in range(K):
        phi_full[k] = phi_below[k] + (r * grid.alpha[k]) * T[k]
    return phi_full, phi_below


# ---------------------------------------------------------------------------
# Discrete column continuity (design doc Sections 5–6)
# ---------------------------------------------------------------------------

def _thickness_weighted(grid: SigmaGrid, g_full):
    """weighted[k] = G_k * Dsigma_k, allocated in the input's array family."""
    G = _require_level_stack(grid, g_full, "g_full")
    weighted = G * 0.0
    for k in range(grid.nlev):
        weighted[k] = grid.thickness[k] * G[k]
    return weighted


def column_mass_tendency(grid: SigmaGrid, g_full):
    """d(ln p_s)/dt = -sum_k G_k Dsigma_k  (shape = trailing dims of G)."""
    return -_thickness_weighted(grid, g_full).sum(axis=0)


def interface_sigma_dot(grid: SigmaGrid, g_full):
    """Interface sigma-velocity, shape (nlev+1, ...), top to bottom.

    sigma_dot_{k+1/2} = sigma_{k+1/2} * sum_{j=1..K} G_j Dsigma_j
                        - sum_{j=1..k} G_j Dsigma_j

    Impermeability is structural: the returned top row is an exact zero
    array (both terms are empty), and the bottom row is the elementwise
    difference ``total - total`` of the SAME cumulative-sum row — bitwise
    zero, not a small residual (design doc Section 6).
    """
    weighted = _thickness_weighted(grid, g_full)
    partial = weighted.cumsum(axis=0)         # partial[k] = sum_{j<=k}
    total = partial[grid.nlev - 1]

    # Allocate (K+1, ...) zeros in the input's array family (``repeat`` is
    # a NumPy-and-CuPy array method; no backend import needed). Row 0 stays
    # the exact structural zero for the sigma = 0 interface.
    sigma_dot = (weighted[:1] * 0.0).repeat(grid.nlev + 1, axis=0)
    for k in range(1, grid.nlev):
        sigma_dot[k] = grid.interfaces[k] * total - partial[k - 1]
    sigma_dot[grid.nlev] = total - total      # bottom: bitwise cancellation
    return sigma_dot


# ---------------------------------------------------------------------------
# Simmons–Burridge energy-conversion term and exchange identity (Section 7b)
# ---------------------------------------------------------------------------

def omega_over_p(grid: SigmaGrid, g_full, v_grad_lnps):
    """Energy-conserving discrete (omega/p)_k (design doc Section 7b, eq. W).

        (omega/p)_k = A_k - (beta_k / Dsigma_k) * P_{k-1} - alpha_k * G_k

    with ``A_k = V_k . grad(ln p_s)`` (``v_grad_lnps``, shape (nlev, ...)),
    ``P_k`` the cumulative thickness-weighted sum of ``G`` and ``beta_k``
    the interface log ratio. The k = 1 beta-term is structurally absent
    (P_0 = 0), so the infinite top-layer ``beta_1`` is never touched.

    This is the UNIQUE choice for which the discrete column energy-exchange
    identity (E_d) holds exactly against the Simmons–Burridge geopotential
    of :func:`hydrostatic_geopotential` — see :func:`energy_exchange`.
    """
    weighted = _thickness_weighted(grid, g_full)
    A = _require_level_stack(grid, v_grad_lnps, "v_grad_lnps")
    partial = weighted.cumsum(axis=0)

    out = weighted * 0.0
    out[0] = A[0] - grid.alpha[0] * g_full[0]
    for k in range(1, grid.nlev):
        out[k] = (A[k]
                  - (grid.interface_log_ratios[k] / grid.thickness[k])
                  * partial[k - 1]
                  - grid.alpha[k] * g_full[k])
    return out


def column_energy_conversion(grid: SigmaGrid, temperature, g_full,
                             v_grad_lnps, r_dry: float):
    """Left side of (E_d): sum_k Dsigma_k R_d T_k (omega/p)_k.

    The column heating input to internal energy (per unit p_s/g column
    mass), using the Section-7b ``(omega/p)_k``. Shape: trailing dims.
    """
    T = _require_level_stack(grid, temperature, "temperature")
    wp = omega_over_p(grid, g_full, v_grad_lnps)
    out = (T[0] * wp[0]) * (float(r_dry) * grid.thickness[0])
    for k in range(1, grid.nlev):
        out = out + (T[k] * wp[k]) * (float(r_dry) * grid.thickness[k])
    return out


def column_pressure_work(grid: SigmaGrid, temperature, phi_full,
                         phi_surface, g_full, v_grad_lnps, r_dry: float):
    """Right side of (E_d): sum_k Dsigma_k [R_d T_k A_k - (Phi_k - Phi_s) G_k].

    The column-local part of the work the pressure-gradient force extracts
    from kinetic energy (the remainder is the horizontal flux divergence
    div(Phi V) and the Phi_s surface term, which vanish under global
    mass-weighted integration — design doc Section 7b). ``phi_full`` must
    be the Simmons–Burridge geopotential of the SAME temperature and
    ``phi_surface``; otherwise the identity has no reason to close.
    """
    T = _require_level_stack(grid, temperature, "temperature")
    phi = _require_level_stack(grid, phi_full, "phi_full")
    G = _require_level_stack(grid, g_full, "g_full")
    A = _require_level_stack(grid, v_grad_lnps, "v_grad_lnps")
    r = float(r_dry)
    out = grid.thickness[0] * (r * T[0] * A[0]
                               - (phi[0] - phi_surface) * G[0])
    for k in range(1, grid.nlev):
        out = out + grid.thickness[k] * (r * T[k] * A[k]
                                         - (phi[k] - phi_surface) * G[k])
    return out


def energy_exchange(grid: SigmaGrid, temperature, phi_surface, g_full,
                    v_grad_lnps, r_dry: float) -> dict:
    """Both sides of the discrete energy-exchange identity and its residual.

    Self-contained: recomputes the Simmons–Burridge geopotential from
    ``temperature`` and ``phi_surface`` so conversion and work are
    guaranteed to be evaluated against the consistent Phi. Returns
    ``conversion`` (E_d left side), ``work`` (E_d right side), and
    ``residual = conversion - work`` — round-off by construction; any
    future change that breaks the alpha/beta consistency between the
    hydrostatic and omega/p operators is caught by this diagnostic.
    """
    phi_full, _ = hydrostatic_geopotential(grid, temperature, phi_surface,
                                           r_dry)
    conversion = column_energy_conversion(grid, temperature, g_full,
                                          v_grad_lnps, r_dry)
    work = column_pressure_work(grid, temperature, phi_full, phi_surface,
                                g_full, v_grad_lnps, r_dry)
    return {"conversion": conversion, "work": work,
            "residual": conversion - work}


def layer_mass_residual(grid: SigmaGrid, g_full):
    """Per-layer discrete mass-budget residual (should be round-off).

    residual_k = Dsigma_k * d(ln p_s)/dt + G_k * Dsigma_k
                 + (sigma_dot_{k+1/2} - sigma_dot_{k-1/2})

    This identity holds to round-off by construction of the continuity
    operators; the diagnostic exists so any future change that breaks the
    telescoping is caught by tests immediately (design doc Section 9).
    """
    weighted = _thickness_weighted(grid, g_full)
    dlnps_dt = column_mass_tendency(grid, g_full)
    sigma_dot = interface_sigma_dot(grid, g_full)
    residual = weighted * 0.0
    for k in range(grid.nlev):
        residual[k] = (grid.thickness[k] * dlnps_dt + weighted[k]
                       + (sigma_dot[k + 1] - sigma_dot[k]))
    return residual
