"""Symbolic derivations for the homicidal chauffeur differential game.

The mathematics is canonical and credited to its original sources:

* Reduced body-frame dynamics — Isaacs (1965) §6.4, pp. 297-300; Merz (1971)
  Chapter 2. The substitution that takes the 5-DOF absolute system
  (positions of pursuer/evader and pursuer heading) into the 2-DOF reduced
  system (x_1, x_2 measured in the pursuer's body frame, with the pursuer
  speed and turning radius normalised to 1).
* Hamiltonian for time-minimal / time-maximal differential games —
  Isaacs (1965) §1.9, with the +1 from minimising the capture time T.
* Optimal pursuer control phi* = -sign(sigma) where sigma = p_2 x_1 - p_1 x_2
  is the *switching function* — Merz (1971) §3.2.
* Optimal evader control psi* = atan2(p_1, p_2) — Isaacs (1965)
  §6.4; Pachter & Coates (2019) eq. (2.5).
* Costate (adjoint) dynamics dp/dt = -dH/dx and the conservation of
  ||p||^2 along the adjoint flow — standard Pontryagin maximum principle
  applied to the autonomous Hamiltonian above.

What we *claim* here is the *implementation*: a SymPy module that returns
each of these objects as inspectable symbolic expressions. The notebook
`paper.ipynb` narrates the derivation step by step and imports from this
module; the live `myst start` view shows the reader the same SymPy objects
that the tests below verify.
"""
from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

# Reduced body-frame state — Isaacs (1965) eq. (6.4.3)
x1, x2 = sp.symbols("x_1 x_2", real=True)

# Costate (adjoint) — Pontryagin's "co-state" variables
p1, p2 = sp.symbols("p_1 p_2", real=True)

# Controls
#   phi — pursuer turn rate, |phi| <= 1 (normalised turn rate)
#   psi — evader heading in the body frame (saddle-point control)
phi, psi = sp.symbols("phi psi", real=True)

# Parameters
#   w   — dimensionless speed ratio v_E / v_P, in (0, 1)
#   ell — dimensionless capture radius ell * omega_max / v_P
w = sp.symbols("w", positive=True)
ell = sp.symbols("ell", positive=True)

# Time
t = sp.symbols("t", real=True)

# ---- Lab-frame symbols (used in absolute_kinematics + reduction step) -----
# Pursuer / evader positions and lab-frame headings — only needed by the
# §5 derivation chain that shows the lab→body reduction explicitly.
x_P_sym, y_P_sym = sp.symbols("x_P y_P", real=True)
x_E_sym, y_E_sym = sp.symbols("x_E y_E", real=True)
theta_sym = sp.symbols(r"theta", real=True)        # pursuer lab heading
# Literal LaTeX in the symbol name so it renders as ψ_{lab} in serif
# subscript (the bare 'psi_lab' string would render as 'psi_{lab}'
# unstyled; with curly braces around 'lab' but no \mathrm, the SymPy
# latex printer also can't recognize psi as Greek — hence the
# explicit \psi escape).
psi_lab_sym = sp.Symbol(r"\psi_{\mathrm{lab}}", real=True)  # evader lab heading


# ---------------------------------------------------------------------------
# Public API — each function corresponds to one step in the derivation chain
# ---------------------------------------------------------------------------


def absolute_kinematics() -> sp.Matrix:
    """The 5-DOF lab-frame kinematics, as a SymPy column vector.

    Source: Isaacs (1965), §5.1 (lab-frame setup); the dimensionless
    normalisation v_P = 1, R_min = 1 is standard in the modern HC
    literature (Merz 1971, Pachter & Coates 2019).

    Returns ``Matrix([dx_P/dt, dy_P/dt, dtheta/dt, dx_E/dt, dy_E/dt])``
    with the dimensionless normalisation v_P = 1, R_min = 1:

        dx_P/dt = cos(theta)
        dy_P/dt = sin(theta)
        dtheta/dt = phi              (|phi| ≤ 1 turn-rate bound)
        dx_E/dt = w cos(psi_lab)
        dy_E/dt = w sin(psi_lab)

    Used in §5 as the starting point for the body-frame reduction:
    the notebook displays this Matrix, then shows the substitution
    that collapses it from 5 DOF to the 2-DOF (x_1, x_2) system in
    :func:`reduced_dynamics`.
    """
    return sp.Matrix([
        sp.cos(theta_sym),
        sp.sin(theta_sym),
        phi,
        w * sp.cos(psi_lab_sym),
        w * sp.sin(psi_lab_sym),
    ])


def body_frame_substitution() -> sp.Matrix:
    """The lab → body coordinate substitution as a SymPy column vector.

    Source: Isaacs (1965), pp. 297-300 (the original body-frame
    reduction of the homicidal-chauffeur state); Merz (1971), §2.

    Given lab-frame relative displacements Δx = x_E − x_P, Δy = y_E − y_P
    and the pursuer's lab heading θ, the body-frame relative position
    is the rotation by −θ:

        x_1 = -Δx sin(theta) + Δy cos(theta)   (perpendicular to heading)
        x_2 =  Δx cos(theta) + Δy sin(theta)   (along heading)

    plus the evader-heading relabelling psi = psi_lab - theta.

    Returns the 2-vector ``Matrix([x_1_expr, x_2_expr])`` in terms of
    x_P, y_P, x_E, y_E, theta — what §5 displays as the substitution
    that produces the body-frame coordinates.
    """
    dx = x_E_sym - x_P_sym
    dy = y_E_sym - y_P_sym
    x1_expr = -dx * sp.sin(theta_sym) + dy * sp.cos(theta_sym)
    x2_expr =  dx * sp.cos(theta_sym) + dy * sp.sin(theta_sym)
    return sp.Matrix([x1_expr, x2_expr])


def reduced_dynamics() -> tuple[sp.Expr, sp.Expr]:
    """The body-frame dynamics of the relative state, normalised so v_P = 1 and
    R_min = 1 (so phi in [-1, 1] is the bang-bang pursuer control).

    Source: Isaacs (1965) pp. 297-300; Merz (1971) eq. (2.X).

    Returns ``(f_1, f_2)`` such that

        d x_1 / dt = f_1(x_1, x_2, phi, psi, w)
        d x_2 / dt = f_2(x_1, x_2, phi, psi, w)
    """
    f1 = -phi * x2 + w * sp.sin(psi)
    f2 = phi * x1 + w * sp.cos(psi) - 1
    return f1, f2


def hamiltonian_saddle() -> sp.Expr:
    """The Hamiltonian after substituting the saddle-point optimal controls.

    Closed form (Isaacs 1965, pp. 297-320):

        H*(x, p) = w · ‖p‖  −  |σ|  −  p_2  +  1

    where ``‖p‖ = sqrt(p_1² + p_2²)`` and ``σ = p_2 x_1 − p_1 x_2``.

    SymPy can verify equivalence to the literal substitution
    ``H.subs([(phi, -sign(sigma)), (sin(psi), p_1/‖p‖), (cos(psi), p_2/‖p‖)])``
    via ``sp.simplify(H_subbed - hamiltonian_saddle()) == 0``. The
    literal substitution produces a ``Piecewise`` SymPy doesn't
    auto-reduce to the |σ| form because |·| isn't recovered from
    ``-sign(σ)·σ`` by default; that recognition step is what the §9
    cell shows the reader.

    Used in §9 to display the clean axiomatic-formal answer the
    saddle-point game converges to.
    """
    norm_p = sp.sqrt(p1 ** 2 + p2 ** 2)
    sigma = p2 * x1 - p1 * x2
    return w * norm_p - sp.Abs(sigma) - p2 + 1


def costate_norm_squared(p1_: sp.Symbol = p1, p2_: sp.Symbol = p2) -> sp.Expr:
    """The conserved quantity along optimal characteristics: ‖p‖² = p_1² + p_2².

    Useful for the §8 conservation-law live derivation: differentiating
    this along the costate ODE and applying sp.simplify should yield 0.
    """
    return p1_ ** 2 + p2_ ** 2


def costate_norm_drift() -> sp.Expr:
    """Time derivative of ‖p‖² along the costate ODE — symbolically zero.

    Computes d/dt[p_1² + p_2²] = 2 p_1 (dp_1/dt) + 2 p_2 (dp_2/dt)
    using the costate equations dp_i/dt = -∂H/∂x_i evaluated at the
    Hamiltonian of the saddle-point game. After ``simplify()`` this
    returns exactly 0 — proving the conservation law symbolically.

    The §8 cell ``sym-conservation-check`` calls this and shows the
    result as a one-line ``sp.simplify(...)`` returning the symbol 0.
    """
    f1, f2 = reduced_dynamics()
    H = hamiltonian(f1, f2, p1, p2)
    p1_dot, p2_dot = costate_ode(H, x1, x2)
    drift = 2 * p1 * p1_dot + 2 * p2 * p2_dot
    return sp.simplify(drift)


def hamiltonian(f1: sp.Expr, f2: sp.Expr, p1_: sp.Symbol, p2_: sp.Symbol) -> sp.Expr:
    """The Hamiltonian for the time-optimal differential game.

    H = p_1 f_1 + p_2 f_2 + 1

    The +1 comes from minimising the running cost L = 1 (capture time).
    """
    return p1_ * f1 + p2_ * f2 + 1


def switching_function(H: sp.Expr, phi_: sp.Symbol) -> sp.Expr:
    """The switching function sigma — the coefficient of phi in H.

    Source: Merz (1971) §3.2. Because H is *linear* in phi (test R2), the
    sign of sigma determines the optimal phi via the bang-bang law
    phi* = -sign(sigma). On the locus sigma = 0 (the *dispersal surface*)
    the optimal control is undetermined, which generates singular surfaces.
    """
    return sp.expand(H).coeff(phi_)


def optimal_controls(
    H: sp.Expr,
    phi_: sp.Symbol,
    psi_: sp.Symbol,
    p1_: sp.Symbol,
    p2_: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr]:
    """The Pontryagin-saddle-point optimal controls.

    phi* = -sign(sigma)              minimises the pursuer's contribution
    psi* = atan2(p_1, p_2)            maximises the evader's contribution

    With these substituted, p_1 sin(psi*) + p_2 cos(psi*) = ||p||.
    """
    sigma = switching_function(H, phi_)
    phi_star = -sp.sign(sigma)
    psi_star = sp.atan2(p1_, p2_)
    return phi_star, psi_star


def costate_ode(H: sp.Expr, x1_: sp.Symbol, x2_: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """The adjoint dynamics dp_i/dt = -dH/dx_i.

    With H = p_1 (-phi x_2 + w sin psi) + p_2 (phi x_1 + w cos psi - 1) + 1,
    this evaluates to (-p_2 phi, p_1 phi); the symbolic test R5 confirms
    ||p||^2 is constant along this flow.
    """
    p1_dot = -sp.diff(H, x1_)
    p2_dot = -sp.diff(H, x2_)
    return p1_dot, p2_dot


def characteristic_system() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """The 4D characteristic ODE system in (x_1, x_2, p_1, p_2), with the
    optimal controls phi* and psi* already substituted.

    This is what gets ``sympy.lambdify``-ed in ``numerics.lambdify_rhs`` —
    the glass-box → black-box encapsulation moment that NORM-02 highlights.

    Returns ``(dx1/dt, dx2/dt, dp1/dt, dp2/dt)`` as sympy expressions in
    (x_1, x_2, p_1, p_2, w).
    """
    f1, f2 = reduced_dynamics()
    H = hamiltonian(f1, f2, p1, p2)
    phi_star, psi_star = optimal_controls(H, phi, psi, p1, p2)
    p1_dot, p2_dot = costate_ode(H, x1, x2)

    subs = [(phi, phi_star), (psi, psi_star)]
    x1_dot = f1.subs(subs)
    x2_dot = f2.subs(subs)
    p1_dot_substituted = p1_dot.subs(subs)
    p2_dot_substituted = p2_dot.subs(subs)
    return x1_dot, x2_dot, p1_dot_substituted, p2_dot_substituted


__all__ = [
    # body-frame symbols
    "x1", "x2", "p1", "p2", "phi", "psi", "w", "ell", "t",
    # lab-frame symbols (for the §5 reduction-chain display)
    "x_P_sym", "y_P_sym", "x_E_sym", "y_E_sym",
    "theta_sym", "psi_lab_sym",
    # derivation chain
    "absolute_kinematics",
    "body_frame_substitution",
    "reduced_dynamics",
    "hamiltonian",
    "switching_function",
    "optimal_controls",
    "costate_ode",
    "characteristic_system",
    # conservation-law display
    "costate_norm_squared",
    "costate_norm_drift",
    # saddle-point H* (§9)
    "hamiltonian_saddle",
]
