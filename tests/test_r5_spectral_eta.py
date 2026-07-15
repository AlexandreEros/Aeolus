"""R-5 regression tests: absolute vorticity must be built in spectral space.

Locks the fix for docs/KNOWN_RISKS.md R-5: the production tendency used to compute
eta = transform(inv_transform(zeta) + f_grid), round-tripping the state through
the inexact transform every evaluation. Because f is a single huge (1,0) mode
(~48x ||zeta|| for Earth-like rotation at the two-vortices amplitude), the
transform's ~0.85% leakage of f injected coefficient errors of ~12% of ||zeta||
into eta on every call (see tests/audit_r5_mechanism.py). Measured effect at
res4/l21, rotating two-vortices: 0.5-day energy drift +1.1e-2 (old) vs +3.5e-5
(fixed) — ~300x. Long-horizon drift is only partly due to R-5 (12-day: -7.3%
old vs -4.6% fixed); the remainder tracks the resolved cascade hitting the
truncation (R-3), not the eta construction.

On the parent commit (0b6c135) the marked tests fail; on this branch they pass.
"""
import numpy as np
import pytest

try:
    import cupy as cp

    _HAS_CUDA = cp.is_available()
except Exception:  # pragma: no cover - import guard
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/CuPy not available")

if _HAS_CUDA:
    from planetary_sandbox.numerics import SpectralOperators
    from planetary_sandbox.planet import Planet, PlanetaryParameters
    from planetary_sandbox.run.bve.barotropic_vorticity import (
        BarotropicState,
        BarotropicVorticity,
    )
    from planetary_sandbox.run.bve.initial_conditions import make_ic
    from planetary_sandbox.run.bve.runner import rk4_step
    from planetary_sandbox.run.bve.diagnostics import spectral_diagnostics

RES, L_MAX = 4, 21


def _planet(day_hours):
    params = PlanetaryParameters.from_earth_like(day_hours=day_hours)
    return Planet.generate(params=params, grid_resolution=RES, l_max=L_MAX)


@pytest.fixture(scope="module")
def rotating():
    return _planet(24.0)


@pytest.fixture(scope="module")
def nonrotating():
    return _planet(np.inf)


# ---------------------------------------------------------------------------
# f representation
# ---------------------------------------------------------------------------

def test_analytic_f_matches_transform_within_envelope(rotating):
    """transform(f_grid) agrees with the analytic (1,0)-only representation;
    the leakage outside (1,0) is explicitly measured and bounded."""
    planet = rotating
    model = BarotropicVorticity(planet, viscosity=0.0)
    omega = planet.params.angular_velocity
    a10 = 2.0 * omega * np.sqrt(4.0 * np.pi / 3.0)

    # analytic construction used by the model
    assert abs(complex(model.f_lm[1, 0]) - a10) < 1e-18 * max(a10, 1.0)
    off = model.f_lm.copy(); off[1, 0] = 0.0
    assert float(cp.abs(off).max()) == 0.0

    # transformed construction
    f_grid = 2.0 * omega * cp.sin(cp.asarray(planet.grid.point_latitudes))
    f_lm_t = planet.sh.transform(f_grid)
    assert abs(complex(f_lm_t[1, 0]) / a10 - 1.0) < 1e-12  # a10 itself is exact

    leak = f_lm_t.copy(); leak[1, 0] = 0.0
    l_idx, m_idx = cp.indices(leak.shape)
    mult = cp.where(m_idx == 0, 1.0, 2.0) * (m_idx <= l_idx)
    leak_norm = float(cp.sqrt((mult * cp.abs(leak) ** 2).sum()))
    # measured 8.5e-3 relative at res4/l21 (voronoi envelope); bound with headroom
    assert leak_norm / a10 < 2e-2, f"f leakage {leak_norm/a10:.2e} above envelope"


# ---------------------------------------------------------------------------
# Structure: no state round trip in the tendency
# ---------------------------------------------------------------------------

def test_tendency_does_not_roundtrip_state(rotating):
    """Structural lock on the tendency's transform usage.

    History: the pre-R5 code performed 3 forward transforms on the state
    grid per tendency call (eta round trip + jacobian truncation round trip
    + advection). R-5 removed the eta round trip (2 calls). The R-3 fix
    ("overresolved product quadrature", spectral return) moves the single
    remaining product analysis onto the fine product grid and eliminates the
    truncation synthesis/re-analysis round trip entirely:

        state-grid forward transforms per tendency:   0
        product-grid forward transforms per tendency: 1

    FAILS ON EVERY EARLIER REVISION (parent of R-5 counts 3 on the state
    grid; post-R5/pre-R3 counts 2)."""
    planet = rotating
    model = BarotropicVorticity(planet, viscosity=0.0)
    zeta_lm = planet.sh.transform(make_ic("two_vortices", planet))

    assert planet.so.product_quadrature == "fine", \
        "production Planet.generate should default to the fine product grid"

    calls = {"coarse": 0, "fine": 0}
    orig_coarse = planet.sh.transform
    orig_fine = planet.so.product_sh.transform

    def counting_coarse(values):
        calls["coarse"] += 1
        return orig_coarse(values)

    def counting_fine(values):
        calls["fine"] += 1
        return orig_fine(values)

    planet.sh.transform = counting_coarse
    planet.so.product_sh.transform = counting_fine
    try:
        model.tendency(BarotropicState(zeta_lm), None)
    finally:
        planet.sh.transform = orig_coarse
        planet.so.product_sh.transform = orig_fine

    assert calls["coarse"] == 0, (
        f"tendency performed {calls['coarse']} state-grid analyses; expected 0 "
        "(a nonzero count indicates a synthesis/re-analysis round trip)."
    )
    assert calls["fine"] == 1, (
        f"tendency performed {calls['fine']} product-grid analyses; expected "
        "exactly 1 (the single analysis of the pointwise product)."
    )


# ---------------------------------------------------------------------------
# Physics agreement
# ---------------------------------------------------------------------------

def test_matches_old_construction_in_nonrotating_limit(nonrotating):
    """At Omega=0, eta==zeta, so the only difference from the old code is the
    removed zeta round trip; agreement is bounded by the transform envelope.

    Design note: the IC must have a genuinely nonzero tendency. The
    two_vortices scenario is unusable here — each Gaussian vortex is
    axisymmetric (zero self-advection) and the pair is ~120 deg apart, so at
    Omega=0 its tendency is ~1e-13 (pure round-off) and any old/new comparison
    is noise divided by noise. A seeded random low-degree state gives an O(1)
    advective tendency to compare against.
    """
    planet = nonrotating
    model = BarotropicVorticity(planet, viscosity=0.0)
    sh = planet.sh

    rng = np.random.default_rng(11)
    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    for l in range(1, 11):
        for m in range(0, l + 1):
            amp = 1e-5 * (l + 1.0) ** -1.5
            imag = rng.standard_normal() if m > 0 else 0.0
            zeta_lm[l, m] = amp * (rng.standard_normal() + 1j * imag)

    new = model.tendency(BarotropicState(zeta_lm), None)

    # replicate the OLD construction inline (round-tripped eta)
    psi_c = model.vorticity_to_streamfunction(BarotropicState(zeta_lm))
    eta_old = sh.transform(sh.inv_transform(zeta_lm) + model.f)
    J = planet.so.jacobian_pseudospectral(psi_c, eta_old)
    old = sh.transform(-J)
    old[0, :] = 0.0

    rel = float(cp.linalg.norm(new - old) / cp.linalg.norm(new))
    # zeta round-trip error is ~1% of ||zeta|| at this envelope; the bilinear
    # Jacobian keeps the tendency perturbation at the same order.
    assert rel < 5e-2, f"nonrotating old/new tendency relative L2 diff {rel:.2e}"


@pytest.mark.parametrize("l, m", [(4, 2), (6, 3)])
def test_rossby_mode_direction_amplitude_phase(rotating, l, m):
    """Single-harmonic Rossby propagation: for zeta = a*Y_l^m on a rotating
    sphere (no mean flow), dzeta/dt = +i * 2*Omega*m/(l(l+1)) * zeta
    (westward phase speed c = -2*Omega/(l(l+1))). Checks sign (direction),
    amplitude, and phase in one complex ratio."""
    planet = rotating
    model = BarotropicVorticity(planet, viscosity=0.0)
    omega = planet.params.angular_velocity

    zeta_lm = cp.zeros((L_MAX + 1, L_MAX + 1), dtype=cp.complex128)
    amp = 1e-5
    zeta_lm[l, m] = amp

    dz = model.tendency(BarotropicState(zeta_lm), None)
    expected = 1j * 2.0 * omega * m / (l * (l + 1.0)) * amp
    ratio = complex(dz[l, m] / expected)
    assert abs(ratio - 1.0) < 2e-2, (
        f"(l={l}, m={m}) Rossby tendency ratio {ratio}; "
        "sign flip would give ratio near -1"
    )


def test_rotating_short_integration_conserves_energy(rotating):
    """Integrated |E(t_i) - E(0)| / E(0) over the first 5 RK4 steps.

    Design note (replaces a single-endpoint check on a review recommendation):
    a single step count is a poor R-5 discriminator because the old code's
    energy trajectory oscillates through zero — initial +1.13e-2 spike at
    step 5, back through zero near step 9, then negative growth. Picking any
    step near a crossing gives spurious agreement (2x separation at step 20).
    Integrated absolute drift is sign-invariant and captures the transient
    regardless of where it happens to land.

    Measured at res4/l21 (day_hours=24, two_vortices, nu=0):
        sum_{i=1..5} |E_i - E_0| / E_0
             parent (round-tripped eta):  3.40e-2
             branch (spectral eta):       1.73e-3
        separation: 19.7x

    Threshold 5e-3 sits 2.9x above the measured branch value (margin against
    hardware jitter) and 6.8x below the measured parent value (regression
    discrimination). FAILS ON PARENT.

    Isolation note (added when the R-3 fine product landed): this test targets
    R-5 (the eta construction), so it pins the nonlinear-product path to the
    historical "coarse" quadrature. Otherwise the R-3 fine product would also
    enter: on the two_vortices IC, nonlinear transfer interacts with the l=14
    tendency cutoff while the state retains modes through l=21, so the scheme
    is not an invariant-conserving Galerkin truncation (neither of the l<=14
    nor of the l<=21 system) and carries no conservation guarantee for this
    energy exchange; the fine quadrature represents that transfer more
    accurately, which raises integ5 to 6.7e-3. This is an IC-specific
    truncation-mismatch effect with no relation to the eta round trip this
    test guards. Holding product_quadrature='coarse' keeps exactly one
    variable (eta construction) in play.
    """
    planet = rotating
    model = BarotropicVorticity(planet, viscosity=0.0)
    model.so = SpectralOperators(planet.sh, planet.params.radius, planet.grid,
                                 product_quadrature="coarse")
    sh, so, grid = planet.sh, model.so, planet.grid
    R, omega = planet.params.radius, planet.params.angular_velocity

    zeta_lm = sh.transform(make_ic("two_vortices", planet))
    psi0 = so.inv_laplacian(zeta_lm)
    u0, v0 = so.velocity_from_streamfunction(psi0)
    dt = 0.5 * grid.min_edge_length / float(cp.max(cp.sqrt(u0**2 + v0**2)))

    E0 = spectral_diagnostics(zeta_lm, R, omega)["energy"]
    state = BarotropicState(cp.copy(zeta_lm))
    integrated = 0.0
    for _ in range(5):
        state = rk4_step(model, state, 0.0, dt)
        E = spectral_diagnostics(state.coeffs, R, omega)["energy"]
        integrated += abs(E - E0) / E0

    assert integrated < 5e-3, (
        f"integrated |dE/E| over 5 rotating steps = {integrated:.2e} "
        "(threshold 5e-3, parent 3.40e-2, branch 1.73e-3)"
    )
