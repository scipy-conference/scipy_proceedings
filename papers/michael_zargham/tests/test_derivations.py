"""Tests for `derivations.py` — symbolic invariants of the HC differential game.

Adapted from `hc-marimo/test_phase1.py` (6 tests). Each test asserts on the
output of the corresponding function in `derivations.py`; the tests do NOT
re-derive the math — they encode the invariant the module must satisfy.

Math credited to:
  - Isaacs (1965), pp. 297-320: reduced body-frame coordinates, Hamiltonian,
    optimal-control structure
  - Merz (1971): saddle-point separability
  - Pachter & Coates (2019): modern restatement

These tests fail with ImportError until Phase C implements derivations.py.
That's the TDD intent.
"""
import math
import random

import sympy as sp
from sympy import atan2, cos, diff, expand, simplify, sin, sqrt


# ---------- R1: reduced body-frame dynamics ----------

def test_R1_reduced_dynamics():
    """Reduced dynamics match Isaacs (1965) pp. 297-300 canonical form.

      f1 = -phi*x2 + w*sin(psi)
      f2 =  phi*x1 + w*cos(psi) - 1
    """
    from derivations import reduced_dynamics, x1, x2, phi, psi, w  # noqa: PLC0415

    f1, f2 = reduced_dynamics()

    expected_f1 = -phi * x2 + w * sin(psi)
    expected_f2 = phi * x1 + w * cos(psi) - 1

    assert simplify(f1 - expected_f1) == 0, f"R1 FAIL: f1 = {f1}"
    assert simplify(f2 - expected_f2) == 0, f"R1 FAIL: f2 = {f2}"

    # Numerical sanity at one point
    subs = {phi: 0.5, psi: math.pi / 3, w: 0.3, x1: 2.0, x2: 1.0}
    f1_num = -0.5 * 1.0 + 0.3 * math.sin(math.pi / 3)
    f2_num = 0.5 * 2.0 + 0.3 * math.cos(math.pi / 3) - 1.0
    assert abs(float(f1.subs(subs)) - f1_num) < 1e-12
    assert abs(float(f2.subs(subs)) - f2_num) < 1e-12


# ---------- R2: Hamiltonian is linear in phi (bang-bang structure) ----------

def test_R2_hamiltonian_linearity():
    """H must be degree 1 in phi — the bang-bang precondition."""
    from derivations import hamiltonian, reduced_dynamics, p1, p2, phi  # noqa: PLC0415

    f1, f2 = reduced_dynamics()
    H = expand(hamiltonian(f1, f2, p1, p2))

    assert H.coeff(phi, 2) == 0, f"R2 FAIL: H has phi^2 term, coeff = {H.coeff(phi, 2)}"
    assert H.coeff(phi, 1) != 0, "R2 FAIL: H has no phi term — bang-bang structure absent"


# ---------- R3: switching function sigma = p2*x1 - p1*x2 ----------

def test_R3_switching_function():
    """The coefficient of phi in H is the switching function sigma."""
    from derivations import (  # noqa: PLC0415
        hamiltonian, reduced_dynamics, switching_function,
        p1, p2, x1, x2, phi,
    )

    f1, f2 = reduced_dynamics()
    H = hamiltonian(f1, f2, p1, p2)

    sigma = switching_function(H, phi)
    expected = p2 * x1 - p1 * x2

    assert simplify(sigma - expected) == 0, f"R3 FAIL: sigma = {sigma}"


# ---------- R4: optimal evader heading psi* = atan2(p1, p2) ----------

def test_R4_optimal_psi():
    """At psi* = atan2(p1, p2), the evader's contribution to H equals ||p||."""
    from derivations import (  # noqa: PLC0415
        hamiltonian, optimal_controls, reduced_dynamics,
        p1, p2, phi, psi,
    )

    f1, f2 = reduced_dynamics()
    H = hamiltonian(f1, f2, p1, p2)
    phi_star, psi_star = optimal_controls(H, phi, psi, p1, p2)

    # Verify psi_star matches atan2(p1, p2) up to branch choice — check by
    # substituting both into the original control law and confirming the
    # evader's contribution = ||p|| at random numerical points.
    contribution = p1 * sin(psi_star) + p2 * cos(psi_star)
    random.seed(42)
    for _ in range(20):
        p1v = random.uniform(-5, 5)
        p2v = random.uniform(-5, 5)
        if abs(p2v) < 0.01:
            continue
        val = float(contribution.subs({p1: p1v, p2: p2v}))
        norm = math.sqrt(p1v ** 2 + p2v ** 2)
        assert abs(val - norm) < 1e-10, (
            f"R4 FAIL: at p=({p1v},{p2v}), contribution={val}, expected ||p||={norm}"
        )


# ---------- R5: ||p||^2 conserved along the adjoint flow ----------

def test_R5_costate_conservation():
    """d/dt(p1^2 + p2^2) = 0 along the costate ODE."""
    from derivations import (  # noqa: PLC0415
        costate_ode, hamiltonian, reduced_dynamics,
        p1, p2, x1, x2,
    )

    f1, f2 = reduced_dynamics()
    H = hamiltonian(f1, f2, p1, p2)
    p1_dot, p2_dot = costate_ode(H, x1, x2)

    d_norm_sq = simplify(2 * (p1 * p1_dot + p2 * p2_dot))

    assert d_norm_sq == 0, f"R5 FAIL: d/dt(||p||^2) = {d_norm_sq}, expected 0"


# ---------- R6: saddle-point separability (Isaacs condition) ----------

def test_R6_separability():
    """No phi-psi cross-terms in H — the Isaacs saddle-point precondition."""
    from derivations import hamiltonian, reduced_dynamics, p1, p2, phi, psi  # noqa: PLC0415

    f1, f2 = reduced_dynamics()
    H = expand(hamiltonian(f1, f2, p1, p2))

    # phi-bearing terms and the rest should be separable
    H_phi_terms = H.coeff(phi) * phi
    H_remaining = expand(H - H_phi_terms)

    assert H_remaining.coeff(phi) == 0, "R6 FAIL: residual phi dependence"
    assert H_phi_terms.coeff(sin(psi)) == 0, "R6 FAIL: phi*sin(psi) cross-term"
    assert H_phi_terms.coeff(cos(psi)) == 0, "R6 FAIL: phi*cos(psi) cross-term"
