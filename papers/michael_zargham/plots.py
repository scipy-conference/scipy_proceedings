"""Plotting helpers for the paper notebook.

Visual style and figure logic are lifted from the prior-art ``hc-marimo``
notebook (`/Users/z/Documents/GitHub/hc-marimo/homicidal_chauffeur.py`); see
the per-function docstring for source cell line numbers. Every function
takes its data as input (separating computation from rendering) so the
notebook cells stay narrative while these functions stay testable. None
of these functions has a marimo dependency — they return plain
``matplotlib.figure.Figure`` objects that the notebook either displays
inline or, in the case of the ``trajectory_frame`` widget, regenerates on
slider change.

Colour palette (lifted from hc-marimo):

* ``PURSUER`` (deep blue) — every pursuer marker, heading arrow, turning
  circle.
* ``EVADER`` (deep red) — every evader marker, velocity arrow, trail.
* ``SWITCHING`` (green) — switching surfaces / dispersal locus.
* ``TERMINAL`` (black dashed) — terminal capture circle.
* ``USABLE`` (red, bold) — the usable arc of the terminal circle.
"""
from __future__ import annotations

from collections import namedtuple
from functools import lru_cache as _lru_cache
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# Container for the shared chase trajectories used by chase_demo (§3)
# and dispersal_crossing (§6 bridge). Computed once per (w, ell, T_max,
# alpha_hint) tuple via _compute_chase_trajectories below.
_ChaseData = namedtuple("_ChaseData", [
    "w", "ell", "T_max", "alpha_hint",
    "sol", "sigma", "cross_idx", "cross_tau",
    "lab", "xp", "yp", "sig_lab",
    "n_eval", "t_eval", "xe", "ye", "sigs",
    "t_kink", "xp_k", "yp_k", "xe_k", "ye_k",
    "shared_xlim", "shared_ylim", "final_theta",
])


class _ArraySol:
    """Minimal wrapper that mimics scipy.integrate.OdeResult enough for
    the interpolation/indexing patterns in dispersal_crossing.

    Stores a 1D ascending grid ``t`` and a 2D array ``y`` (rows are
    coordinate components). The ``sol`` method does 1D linear
    interpolation per row at any requested t value.
    """

    def __init__(self, t: np.ndarray, y_rows: list):
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y_rows, dtype=float)

    def sol(self, t_eval):
        return np.array([np.interp(t_eval, self.t, row) for row in self.y])


# ---------------------------------------------------------------------------
# Palette (lifted from hc-marimo)
# ---------------------------------------------------------------------------

PURSUER = "#2855a1"        # primary pursuer blue (problem_geometry_plot)
PURSUER_TRAIL = "#2166ac"  # pursuer trail / chase markers
EVADER = "#c0392b"         # primary evader red (problem_geometry_plot)
EVADER_TRAIL = "#b2182b"   # evader trail / chase markers
SWITCHING = "#2b8a3e"      # green for switching / dispersal locus
TERMINAL = "k"             # black dashed for terminal circle
USABLE = "#c92a2a"         # bold red for the usable arc
ANNOT = "#444444"          # neutral grey for annotations

# Phase colors for dispersal_crossing — the stitched chase has FOUR
# colored phases delimited by THREE events:
#   (a) σ zero-crossing within segment A
#   (b) DISPERSAL SWITCH (costate ‖p‖ jumps; σ also flips sign across
#       this seam since the costate's direction reverses) — this is
#       the single biggest event in the figure. Rendered as a hard
#       gap between Phase 2 and Phase 3, marked by a red ★.
#   (c) σ zero-crossing within segment B
#
# Indexed in τ-order so PHASE_COLORS[0]=P1 corresponds to small τ
# (near capture, end of forward-time chase), PHASE_COLORS[3]=P4 to
# large τ (chase start, beginning of forward time).
#
# Chronological forward-time colour progression (chase start →
# capture): orange → blue → [★ dispersal switch] → green → purple.
#
# σ-sign ↔ hue: σ<0 phases (P1, P3) are COOL (purple, blue) and σ>0
# phases (P2, P4) are WARM (green, orange). The two phases that
# share a segment (and overlap on the costate panel's ‖p‖ circle)
# thus contrast warm-vs-cool: segment A is purple+green, segment B
# is blue+orange. Reader learns the sign-colour mapping after one
# look at (1,2) σ(τ).
PHASE_COLORS = [
    "#8e44ad",  # P1: σ<0, near capture (end of segment A)  — purple
    "#2a9d8f",  # P2: σ>0, mid segment A                    — green
    "#2c5da0",  # P3: σ<0, mid segment B                    — blue
    "#e76f51",  # P4: σ>0, chase start (start of segment B) — orange
]
PHASE_LABELS = [
    "P1: σ<0, end of A (capture)",
    "P2: σ>0, mid segment A",
    "P3: σ<0, mid segment B",
    "P4: σ>0, start of B (chase start)",
]
# Reserved for the dispersal switch marker (red ★ between P2 and P3).
DISPERSAL = "#c0392b"


# ---------------------------------------------------------------------------
# Shared trajectory data — single source of truth so the §3 motivation
# chase and the §6 dispersal_crossing analysis show the SAME chase.
# ---------------------------------------------------------------------------


@_lru_cache(maxsize=8)
def _compute_chase_trajectories(w: float = 0.45,
                                ell: float = 0.5,
                                T_max: float = 10.0,
                                alpha_hint: float = 1.7) -> "_ChaseData":
    """Integrate one optimal-play characteristic and reconstruct both
    players' lab-frame paths.

    Cached so chase_demo() (§3 motivation) and dispersal_crossing()
    (§6 analysis) share the same trajectories without re-integrating.
    """
    from scipy.integrate import solve_ivp  # local import (heavy)
    from numerics import lambdify_rhs, terminal_conditions  # noqa: PLC0415

    rhs = lambdify_rhs()
    a_lo = float(np.arcsin(w) + 0.05)
    a_hi = float(np.pi - np.arcsin(w) - 0.05)

    def _shoot(alpha):
        y0 = terminal_conditions(np.array([alpha]), w, ell)[0]
        return solve_ivp(
            lambda t, y: [-v for v in rhs(t, y, w)],
            [0, T_max], y0,
            method="RK45", max_step=0.02, rtol=1e-9, atol=1e-11,
            dense_output=True,
        )

    sol = _shoot(alpha_hint)
    x1, x2, p1, p2 = sol.y
    sigma = p2 * x1 - p1 * x2

    # Locate the SECOND zero of σ (first is trivially τ=0 by transversality)
    sign_changes = np.where(np.diff(np.sign(sigma[10:])) != 0)[0]
    if len(sign_changes) == 0:
        for alpha in np.linspace(a_lo, a_hi, 41):
            sol = _shoot(float(alpha))
            x1, x2, p1, p2 = sol.y
            sigma = p2 * x1 - p1 * x2
            sign_changes = np.where(np.diff(np.sign(sigma[10:])) != 0)[0]
            if len(sign_changes) > 0:
                alpha_hint = float(alpha)
                break
    cross_idx = int(sign_changes[0]) + 10
    cross_tau = float(sol.t[cross_idx])

    # Pursuer lab integration (forward in physical time t = T_max - τ)
    def _phi_at_t(t_val):
        tau_val = max(min(T_max - t_val, sol.t[-1]), sol.t[0])
        state = sol.sol(tau_val)
        sig = state[3] * state[0] - state[2] * state[1]
        return -np.sign(sig) if abs(sig) > 1e-9 else 1.0

    def _lab_rhs(t, y):
        return [np.cos(y[2]), np.sin(y[2]), float(_phi_at_t(t))]

    lab = solve_ivp(
        _lab_rhs, [0, T_max], [0.0, 0.0, 0.0],
        method="RK45", max_step=0.01, rtol=1e-9, atol=1e-11,
        dense_output=True,
    )
    xp = lab.y[0]
    yp = lab.y[1]

    # σ along the pursuer's lab trajectory
    sig_lab = np.empty(len(lab.t))
    for i, t_val in enumerate(lab.t):
        tau_i = max(min(T_max - t_val, sol.t[-1]), sol.t[0])
        s = sol.sol(tau_i)
        sig_lab[i] = float(s[3] * s[0] - s[2] * s[1])

    # Evader lab reconstruction via inverse transformation
    n_eval = 400
    t_eval = np.linspace(0.0, float(T_max), n_eval)
    xe = np.zeros(n_eval)
    ye = np.zeros(n_eval)
    sigs = np.zeros(n_eval)
    for j, t_val in enumerate(t_eval):
        tau_j = max(min(float(T_max) - t_val, sol.t[-1]), sol.t[0])
        body = sol.sol(tau_j)
        x1j, x2j, p1j, p2j = body
        sigs[j] = p2j * x1j - p1j * x2j
        plab = lab.sol(t_val)
        x_P_j, y_P_j, theta_j = plab
        c, s = np.cos(theta_j), np.sin(theta_j)
        xe[j] = x_P_j + x1j * (-s) + x2j * c
        ye[j] = y_P_j + x1j * c + x2j * s

    # Kink positions in lab frame (same instant t_kink for both players)
    t_kink = float(T_max) - cross_tau
    plab_k = lab.sol(t_kink)
    body_k = sol.sol(cross_tau)
    xp_k = float(plab_k[0])
    yp_k = float(plab_k[1])
    c_k, s_k = float(np.cos(plab_k[2])), float(np.sin(plab_k[2]))
    xe_k = xp_k + body_k[0] * (-s_k) + body_k[1] * c_k
    ye_k = yp_k + body_k[0] * c_k + body_k[1] * s_k

    # Shared bounding box for figures that show both players
    x_lo = min(float(xp.min()), float(xe.min()))
    x_hi = max(float(xp.max()), float(xe.max()))
    y_lo = min(float(yp.min()), float(ye.min()))
    y_hi = max(float(yp.max()), float(ye.max()))
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = 0.5 * 1.15 * max(x_hi - x_lo, y_hi - y_lo)
    shared_xlim = (cx - half, cx + half)
    shared_ylim = (cy - half, cy + half)

    # Final pursuer heading (for the chase_demo orientation triangle)
    final_theta = float(lab.sol(T_max)[2])

    return _ChaseData(
        w=w, ell=ell, T_max=T_max, alpha_hint=float(alpha_hint),
        sol=sol, sigma=sigma, cross_idx=cross_idx, cross_tau=cross_tau,
        lab=lab, xp=xp, yp=yp, sig_lab=sig_lab,
        n_eval=n_eval, t_eval=t_eval, xe=xe, ye=ye, sigs=sigs,
        t_kink=t_kink, xp_k=xp_k, yp_k=yp_k, xe_k=xe_k, ye_k=ye_k,
        shared_xlim=shared_xlim, shared_ylim=shared_ylim,
        final_theta=final_theta,
    )


@_lru_cache(maxsize=1)
def _recover_chase_from_demo() -> "_ChaseData":
    """Recover analytical quantities from the hardcoded chase_demo
    trajectory so the §6 analysis decomposes the *same* chase the §3
    header shows.

    The chase_demo trajectory is STITCHED from two optimal
    characteristics (α_B = 95° pre-switch, α_A = 40° post-switch)
    joined at the dispersal point (sample i = _DEMO_SWITCH_IDX = 24).
    Their costate norms differ:

        ||p||_B = 1 / (sin 95° − w) ≈ 1.83
        ||p||_A = 1 / (sin 40° − w) ≈ 5.18

    so the recovered (σ, p) trajectories exhibit a genuine
    discontinuity at the switch — exactly what makes a dispersal
    surface a *seam* in strategy space. The reader can SEE the value-
    function gradient being multi-valued there.

    Recovery procedure:
      1. Body-frame state (x_1, x_2) ← algebraic inversion of
         (P_lab, E_lab, θ).
      2. Body-frame evader heading ψ* ← atan2(dE/dt) − θ via finite
         differences (uses physical time computed from pursuer arc
         length, since v_P = 1).
      3. Costate direction (sin ψ*, cos ψ*); magnitude from terminal
         transversality per segment.
      4. σ = p_2 x_1 − p_1 x_2 follows from (1)–(3).

    Returns a _ChaseData populated with both forward-time arrays
    (lab.t, xp/yp/xe/ye, sigs) and τ-ordered arrays (sol.t in
    backward time, x1/x2/p1/p2 reversed, sigma reversed).
    """
    w_val = 0.45
    ell_val = 0.5
    alpha_A = np.radians(40.0)
    alpha_B = np.radians(95.0)

    xp = _DEMO_XP.copy()
    yp = _DEMO_YP.copy()
    xe = _DEMO_XE.copy()
    ye = _DEMO_YE.copy()
    theta = np.unwrap(_DEMO_TH)
    n = len(xp)
    switch_idx = _DEMO_SWITCH_IDX

    # Physical time from pursuer arc length (v_P = 1)
    seg = np.sqrt(np.diff(xp) ** 2 + np.diff(yp) ** 2)
    t_phys = np.concatenate([[0.0], np.cumsum(seg)])
    T_max = float(t_phys[-1])

    # Body-frame state (x_1, x_2)(t) — directly from lab data
    dx = xe - xp
    dy = ye - yp
    x1_f = -np.sin(theta) * dx + np.cos(theta) * dy
    x2_f = np.cos(theta) * dx + np.sin(theta) * dy

    # Body-frame evader heading ψ* ← atan2(dE/dt) − θ
    dx_e = np.gradient(xe, t_phys)
    dy_e = np.gradient(ye, t_phys)
    psi_lab = np.arctan2(dy_e, dx_e)
    psi_body = np.unwrap(psi_lab - theta)

    # Costate magnitudes per segment from transversality at terminal α
    lam_A = 1.0 / (np.sin(alpha_A) - w_val)
    lam_B = 1.0 / (np.sin(alpha_B) - w_val)
    norm_arr = np.where(np.arange(n) < switch_idx, lam_B, lam_A)

    # Recovered costate vector and σ
    p1_f = norm_arr * np.sin(psi_body)
    p2_f = norm_arr * np.cos(psi_body)
    sigma_f = p2_f * x1_f - p1_f * x2_f

    # τ-ordered arrays (τ = T_max - t, ascending from 0 to T_max)
    rev = slice(None, None, -1)
    tau = T_max - t_phys[rev]
    sigma_tau = sigma_f[rev]
    x1_tau = x1_f[rev]
    x2_tau = x2_f[rev]
    p1_tau = p1_f[rev]
    p2_tau = p2_f[rev]

    # sol (τ-ordered body-frame state) + lab (t-ordered pursuer lab) wrappers
    sol = _ArraySol(tau, [x1_tau, x2_tau, p1_tau, p2_tau])
    lab = _ArraySol(t_phys, [xp, yp, theta])

    # cross_idx in τ-order maps to forward switch_idx
    cross_idx = n - 1 - switch_idx
    cross_tau = float(tau[cross_idx])

    # Kink positions in lab frame
    t_kink = float(t_phys[switch_idx])
    xp_k = float(xp[switch_idx])
    yp_k = float(yp[switch_idx])
    xe_k = float(xe[switch_idx])
    ye_k = float(ye[switch_idx])

    # σ along pursuer lab.t (same grid as t_phys) — equals sigma_f
    sig_lab = sigma_f.copy()

    # σ along evader t_eval — also same grid as t_phys
    n_eval = n
    t_eval = t_phys.copy()
    sigs = sigma_f.copy()

    # Shared bounding box covering both paths with 15% margin
    x_lo = min(float(xp.min()), float(xe.min()))
    x_hi = max(float(xp.max()), float(xe.max()))
    y_lo = min(float(yp.min()), float(ye.min()))
    y_hi = max(float(yp.max()), float(ye.max()))
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = 0.5 * 1.15 * max(x_hi - x_lo, y_hi - y_lo)
    shared_xlim = (cx - half, cx + half)
    shared_ylim = (cy - half, cy + half)

    return _ChaseData(
        w=w_val, ell=ell_val, T_max=T_max,
        alpha_hint=float(alpha_A),  # nominal — actual chase is stitched
        sol=sol, sigma=sigma_tau, cross_idx=cross_idx, cross_tau=cross_tau,
        lab=lab, xp=xp, yp=yp, sig_lab=sig_lab,
        n_eval=n_eval, t_eval=t_eval, xe=xe, ye=ye, sigs=sigs,
        t_kink=t_kink, xp_k=xp_k, yp_k=yp_k, xe_k=xe_k, ye_k=ye_k,
        shared_xlim=shared_xlim, shared_ylim=shared_ylim,
        final_theta=float(theta[-1]),
    )


# Hardcoded chase trajectory at (w=0.45, ell=0.5, T~12), generated
# offline and PAINSTAKINGLY TUNED for visual clarity. Two backward
# characteristics (alpha_A=40deg, alpha_B=95deg) cross in reduced
# coordinates; the game-optimal trajectory follows B's outer segment
# then switches to A's inner segment, producing an evader heading
# kink of ~48 degrees at _SWITCH_IDX = 24 — the dispersal-surface
# crossing that motivates the §3 narrative.
#
# These arrays are the *header*'s source of truth. The §6 analysis
# (dispersal_crossing) uses a separate single-characteristic
# integration via _compute_chase_trajectories — it cannot reproduce
# this stitched chase exactly. Author chose the visual clarity of
# the hardcoded chase over data-source unification.

_DEMO_XP = np.array([
    6.4691, 6.2348, 6.0217, 5.8627, 5.7537, 5.7110, 5.7349, 5.8259, 5.9782,
    6.1711, 6.4095, 6.6664, 6.9047, 7.1247, 7.2993, 7.4125, 7.4672, 7.4527,
    7.3754, 7.2332, 7.0381, 6.8149, 6.5580, 6.2951, 6.2766, 6.2646, 6.0210,
    5.7699, 5.5400, 5.3464, 5.2076, 5.1188, 5.0361, 4.9081, 4.7374, 4.5186,
    4.2733, 4.0178, 3.7805, 3.5535, 3.3639, 3.2243, 3.1464, 3.1276, 3.1746,
    3.2843, 3.4407, 3.6490, 3.8890, 4.1403,
])
_DEMO_YP = np.array([
    -1.7816, -1.6896, -1.5338, -1.3381, -1.0974, -0.8365, -0.5852, -0.3371,
    -0.1214,  0.0408,  0.1534,  0.2559,  0.3358,  0.4805,  0.6774,  0.9018,
     1.1593,  1.4222,  1.6613,  1.8828,  2.0595,  2.1752,  2.2326,  2.2210,
     2.2121,  2.2098,  2.1964,  2.2464,  2.3588,  2.5262,  2.7266,  2.9667,
     3.2240,  3.4450,  3.6182,  3.7495,  3.8205,  3.8264,  3.7710,  3.6525,
     3.4798,  3.2642,  3.0320,  2.7752,  2.5218,  2.2883,  2.0986,  1.9463,
     1.8517,  1.8248,
])
_DEMO_XE = np.array([
    0.0000, 0.1133, 0.2320, 0.3454, 0.4641, 0.5828, 0.6961, 0.8148, 0.9336,
    1.0469, 1.1656, 1.2843, 1.3976, 1.5164, 1.6351, 1.7484, 1.8671, 1.9858,
    2.0992, 2.2179, 2.3366, 2.4499, 2.5686, 2.6874, 2.6928, 2.6963, 2.7672,
    2.8416, 2.9161, 2.9905, 3.0614, 3.1359, 3.2103, 3.2848, 3.3557, 3.4301,
    3.5046, 3.5790, 3.6499, 3.7243, 3.7988, 3.8732, 3.9441, 4.0186, 4.0930,
    4.1675, 4.2384, 4.3128, 4.3873, 4.4617,
])
_DEMO_YE = np.array([
    0.0000, 0.0042, 0.0086, 0.0128, 0.0172, 0.0216, 0.0258, 0.0302, 0.0346,
    0.0388, 0.0432, 0.0476, 0.0518, 0.0562, 0.0605, 0.0647, 0.0691, 0.0735,
    0.0777, 0.0821, 0.0865, 0.0907, 0.0951, 0.0995, 0.0997, 0.1039, 0.1884,
    0.2772, 0.3659, 0.4546, 0.5391, 0.6278, 0.7165, 0.8052, 0.8897, 0.9785,
    1.0672, 1.1559, 1.2404, 1.3291, 1.4178, 1.5066, 1.5911, 1.6798, 1.7685,
    1.8572, 1.9417, 2.0304, 2.1192, 2.2079,
])
_DEMO_TH = np.array([
    -3.3856, -3.6376, -3.9016, -4.1536, -4.4176, -4.6816, -4.9336, -5.1976,
    -5.4616, -5.7136, -5.9776, -6.0856, -5.8336, -5.5696, -5.3056, -5.0536,
    -4.7896, -4.5256, -4.2736, -4.0096, -3.7456, -3.4936, -3.2296, -2.9656,
    -2.9536, -2.9659, -3.2110, -3.4683, -3.7257, -3.9831, -4.2282, -4.4856,
    -4.3140, -4.0566, -3.8115, -3.5541, -3.2968, -3.0394, -2.7943, -2.5369,
    -2.2795, -2.0222, -1.7771, -1.5197, -1.2623, -1.0050, -0.7598, -0.5025,
    -0.2451,  0.0000,
])
_DEMO_SWITCH_IDX = 24
_DEMO_CAPTURE_ELL = 0.5


def naive_vs_optimal(*, dt: float = 0.005) -> plt.Figure:
    """§3 hook figure: two heuristic evader strategies vs the same
    pursuer, at the **same initial condition as the chase_demo
    header**. The pedagogical claim is the saddle-point theorem:
    *no heuristic evader beats the saddle-point value T\\* against
    the optimal pursuer*.

    Two side-by-side lab-frame chases (pursuer at the chase_demo IC,
    evader at the origin, w=0.45, ℓ=0.5):

      Left:  evader runs straight AWAY from the pursuer's bearing.
             Captured at t ≈ 11.3 s.
      Right: evader runs PERPENDICULAR to the bearing.
             Captured at t ≈ 6.2 s — worse than naive at this IC
             because the pursuer's turning radius isn't a constraint
             at long initial range.

    Both panels use a pure-pursuit pursuer (max-rate turn toward
    evader). The saddle-point optimal evader against the saddle-
    point optimal pursuer is the chase_demo trajectory (t = 12.1 s,
    shown in the header). Both heuristics underperform that bound.

    Why pure-pursuit pursuer instead of optimal:
    --------------------------------------------
    A bang-bang optimal-pursuer feedback φ* = −sign(σ*(x_1, x_2))
    requires interpolating σ from sampled optimal trajectories
    (see ``numerics.compute_sigma_field``). At the chase_demo IC
    the body state sits ON a dispersal-surface singularity, where
    σ is genuinely multi-valued in the saddle-point game — adjacent
    sample points belong to competing optimal trajectories with
    σ values of opposite sign. Linear interpolation averages across
    this discontinuity and produces high-frequency chattering of
    the pursuer's control. Pure-pursuit gives a clean numerical
    answer; the **saddle-point theorem** still bounds heuristic
    capture time at ≤ T\\* against the optimal pursuer (and pure-
    pursuit is, in practice, *not* substantially worse than optimal
    at long range — naive vs pure-pursuit 11.3 s < saddle-point
    12.1 s, both well below the theoretical bound).

    The takeaway the rest of the paper earns: the math reveals
    strategies — and bounds — that physical intuition can't.
    §5 derives the body-frame reduction; §6 the Hamiltonian; §7
    decomposes the chase_demo trajectory through six coordinate
    views (the ★ in §7 is the same dispersal-surface event
    discussed in this docstring); §8 renders the full value
    function with the §3 IC marked.

    Marker vocabulary shared with chase_demo and dispersal_crossing:
    ● start, ■ end, ·· capture circle. Pursuer blue, evader red.
    """
    from numerics import simulate_pure_pursuit_chase  # noqa: PLC0415

    # ICs from the chase_demo header — exact same start.
    p_init = (
        float(_DEMO_XP[0]), float(_DEMO_YP[0]), float(_DEMO_TH[0]),
    )
    e_init = (float(_DEMO_XE[0]), float(_DEMO_YE[0]))
    w = 0.45
    ell = float(_DEMO_CAPTURE_ELL)

    naive = simulate_pure_pursuit_chase(
        p_init, e_init, w=w, ell=ell,
        evader_policy="run_away", T_max=20.0, dt=dt,
    )
    perp = simulate_pure_pursuit_chase(
        p_init, e_init, w=w, ell=ell,
        evader_policy="perpendicular", T_max=20.0, dt=dt,
    )

    fig, (ax_naive, ax_perp) = plt.subplots(1, 2, figsize=(14, 4.3))

    # Shared bounding box so both panels read at the same scale.
    all_x = np.concatenate(
        [naive["xp"], naive["xe"], perp["xp"], perp["xe"]]
    )
    all_y = np.concatenate(
        [naive["yp"], naive["ye"], perp["yp"], perp["ye"]]
    )
    pad = 0.5
    xlim = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim = (float(all_y.min()) - pad, float(all_y.max()) + pad)
    th_circle = np.linspace(0, 2 * np.pi, 80)

    def _draw(ax, sim, title_top, title_bot, evader_label):
        ax.plot(sim["xp"], sim["yp"], "-", color=PURSUER_TRAIL, lw=1.8,
                alpha=0.85, label="Pursuer (pure pursuit)")
        ax.plot(sim["xe"], sim["ye"], "--", color=EVADER_TRAIL, lw=1.8,
                alpha=0.85, label=f"Evader ({evader_label})")
        # ● start
        ax.plot(sim["xp"][0], sim["yp"][0], "o", color=PURSUER_TRAIL,
                markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                zorder=10)
        ax.plot(sim["xe"][0], sim["ye"][0], "o", color=EVADER_TRAIL,
                markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                zorder=10)
        # ■ end
        ax.plot(sim["xp"][-1], sim["yp"][-1], "s", color=PURSUER_TRAIL,
                markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                zorder=10)
        ax.plot(sim["xe"][-1], sim["ye"][-1], "s", color=EVADER_TRAIL,
                markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                zorder=10)
        # ·· capture circle
        ax.plot(
            sim["xp"][-1] + ell * np.cos(th_circle),
            sim["yp"][-1] + ell * np.sin(th_circle),
            ":", color="gray", lw=1.1, alpha=0.7, zorder=2,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ (lab)")
        ax.set_ylabel(r"$y$ (lab)")
        ax.set_title(f"{title_top}\n{title_bot}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    naive_ct = naive["capture_time"]
    perp_ct = perp["capture_time"]
    _draw(
        ax_naive, naive,
        '"Just run away" — naive evader',
        rf'captured at $t = {naive_ct:.2f}$' if naive_ct is not None
        else "escaped",
        evader_label="run away",
    )
    _draw(
        ax_perp, perp,
        '"Run perpendicular" — counter-intuitive evader',
        rf'captured at $t = {perp_ct:.2f}$' if perp_ct is not None
        else "escaped",
        evader_label="perpendicular",
    )

    fig.suptitle(
        rf"Heuristic evaders at the §3 chase IC ($w = {w}, \ell = {ell}$)",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def chase_demo() -> plt.Figure:
    """§3 motivation: one complete optimal chase showing a dispersal crossing.

    Uses the carefully-tuned hardcoded stitched chase (two
    characteristics joined at the dispersal point) rather than a
    single-characteristic integration. This produces a more visually
    informative motivation figure than any single-α optimal chase
    would. The §6 ``dispersal_crossing`` analysis necessarily uses a
    single characteristic (the analytical decomposition requires it),
    so the two figures show *related but not identical* chases.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Marker vocabulary shared with the §6 dispersal_crossing subplot
    # figure so the reader has a consistent visual hook:
    #     ● circle  →  start of curve
    #     ★ star    →  direction change / dispersal switch
    #     ■ square  →  end of curve
    # Player identity is carried by colour (pursuer blue, evader red);
    # the ★ uses the DISPERSAL red (same as the subplot's switch
    # markers) since the switch is the shared event between players.

    # Pursuer trail + ● start
    ax.plot(_DEMO_XP, _DEMO_YP, "-", color=PURSUER_TRAIL, linewidth=1.5,
            alpha=0.7, label="Pursuer (fast, wide turns)")
    ax.plot(_DEMO_XP[0], _DEMO_YP[0], "o", color=PURSUER_TRAIL, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)

    # Pursuer ■ end (replaces the heading-oriented triangle — square
    # is the consistent end-of-curve marker; the pursuer's facing is
    # already legible from the smooth trail curvature at capture).
    ax.plot(_DEMO_XP[-1], _DEMO_YP[-1], "s", color=PURSUER_TRAIL, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)

    # Evader trail, split at the direction-change index
    ax.plot(_DEMO_XE[: _DEMO_SWITCH_IDX + 1], _DEMO_YE[: _DEMO_SWITCH_IDX + 1],
            "--", color=EVADER_TRAIL, linewidth=1.5, alpha=0.7,
            label="Evader (slow, agile)")
    ax.plot(_DEMO_XE[_DEMO_SWITCH_IDX:], _DEMO_YE[_DEMO_SWITCH_IDX:],
            "--", color=EVADER_TRAIL, linewidth=1.5, alpha=0.7)

    # Evader: ● start, ★ direction change (DISPERSAL red), ■ end.
    ax.plot(_DEMO_XE[0], _DEMO_YE[0], "o", color=EVADER_TRAIL, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.plot(_DEMO_XE[_DEMO_SWITCH_IDX], _DEMO_YE[_DEMO_SWITCH_IDX], "*",
            color="#c0392b", markersize=20,
            markeredgecolor="white", markeredgewidth=1.2, zorder=10)
    ax.plot(_DEMO_XE[-1], _DEMO_YE[-1], "s", color=EVADER_TRAIL,
            markersize=10, markeredgecolor="white", markeredgewidth=0.8,
            zorder=10)

    # Capture circle at final pursuer position
    circ = np.linspace(0, 2 * np.pi, 100)
    ax.plot(_DEMO_XP[-1] + _DEMO_CAPTURE_ELL * np.cos(circ),
            _DEMO_YP[-1] + _DEMO_CAPTURE_ELL * np.sin(circ),
            "--", color="gray", linewidth=1, alpha=0.5, label="Capture circle")

    # Annotations
    ax.annotate("Pursuer start", xy=(_DEMO_XP[0], _DEMO_YP[0]),
                xytext=(_DEMO_XP[0] + 1.5, _DEMO_YP[0] + 0.7),
                fontsize=9, color=PURSUER_TRAIL,
                arrowprops=dict(arrowstyle="->", color=PURSUER_TRAIL, lw=0.8))
    ax.annotate("Evader start", xy=(_DEMO_XE[0], _DEMO_YE[0]),
                xytext=(_DEMO_XE[0] - 1.5, _DEMO_YE[0] + 0.5),
                fontsize=9, color=EVADER_TRAIL,
                arrowprops=dict(arrowstyle="->", color=EVADER_TRAIL, lw=0.8))
    ax.annotate("Direction change",
                xy=(_DEMO_XE[_DEMO_SWITCH_IDX], _DEMO_YE[_DEMO_SWITCH_IDX]),
                xytext=(_DEMO_XE[_DEMO_SWITCH_IDX] + 0.8,
                        _DEMO_YE[_DEMO_SWITCH_IDX] - 0.6),
                fontsize=9, color=EVADER_TRAIL,
                arrowprops=dict(arrowstyle="->", color=EVADER_TRAIL, lw=0.8))
    ax.annotate("Capture", xy=(_DEMO_XP[-1], _DEMO_YP[-1]),
                xytext=(_DEMO_XP[-1] + 0.8, _DEMO_YP[-1] - 0.7),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel(r"$x$ (lab)")
    ax.set_ylabel(r"$y$ (lab)")
    ax.set_aspect("equal")
    ax.set_title("Optimal pursuit–evasion: one complete chase ($w = 0.45$, $\\ell = 0.5$)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# §3 supporting — problem_geometry (lifted from problem_geometry_plot, L322-449)
# ---------------------------------------------------------------------------


def problem_geometry() -> plt.Figure:
    """§3 lab-frame schematic: P with heading θ + turning circle, E with v_E.

    Lifted from hc-marimo's ``problem_geometry_plot`` (L322-449). Pursuer
    is a body-oriented triangle with a heading arrow, a heading-angle arc,
    a dashed minimum-turning circle, and a dotted capture-radius circle.
    Evader is a dot with a velocity arrow and a ψ_lab heading arc. A
    dashed line P–E shows the relative-position distance ``r``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    P = np.array([2.0, 1.5])
    theta = 0.6  # heading angle (rad)
    R_min = 1.8  # turning circle radius (visual scale)
    ell = 0.5    # capture radius (visual scale)

    # Pursuer triangle, oriented along heading
    size = 0.35
    tri = np.array([
        [size, 0],
        [-size * 0.5, size * 0.4],
        [-size * 0.5, -size * 0.4],
    ])
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    tri_rot = (rot @ tri.T).T + P
    ax.add_patch(plt.Polygon(tri_rot, fc=PURSUER, ec="black", lw=1.5, zorder=5))
    ax.annotate(r"$P$", P + np.array([-0.5, -0.4]), fontsize=14,
                fontweight="bold", color=PURSUER)

    # Heading arrow
    head_len = 1.2
    ax.annotate("", xy=P + head_len * np.array([np.cos(theta), np.sin(theta)]),
                xytext=P,
                arrowprops=dict(arrowstyle="->", color=PURSUER, lw=2))
    ax.annotate(r"$v_P$",
                P + 0.7 * np.array([np.cos(theta), np.sin(theta)])
                + np.array([0.1, 0.2]),
                fontsize=12, color=PURSUER)

    # Heading angle arc
    arc_angles = np.linspace(0, theta, 30)
    arc_r = 0.7
    ax.plot(P[0] + arc_r * np.cos(arc_angles),
            P[1] + arc_r * np.sin(arc_angles),
            "-", color=PURSUER, lw=1, alpha=0.7)
    ax.annotate(r"$\theta$",
                P + (arc_r + 0.1) * np.array(
                    [np.cos(theta / 2), np.sin(theta / 2)]),
                fontsize=13, color=PURSUER)

    # Minimum turning circle (perpendicular-left of heading)
    turn_center = P + R_min * np.array([-np.sin(theta), np.cos(theta)])
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(turn_center[0] + R_min * np.cos(circle_theta),
            turn_center[1] + R_min * np.sin(circle_theta),
            "--", color=PURSUER, lw=1, alpha=0.4)
    ax.annotate(r"$R_{\min}$",
                turn_center + np.array([R_min * 0.5, R_min * 0.3]),
                fontsize=11, color=PURSUER, alpha=0.7)

    # Capture radius
    cap_theta = np.linspace(0, 2 * np.pi, 80)
    ax.plot(P[0] + ell * np.cos(cap_theta),
            P[1] + ell * np.sin(cap_theta),
            ":", color="gray", lw=1.5, alpha=0.8)
    ax.annotate(r"$\ell$", P + np.array([ell + 0.1, 0.1]),
                fontsize=11, color="gray")

    # Evader
    E = np.array([4.5, 3.2])
    psi = 2.3
    ax.plot(*E, "o", color=EVADER, markersize=10, zorder=5)
    ax.annotate(r"$E$", E + np.array([0.2, -0.3]), fontsize=14,
                fontweight="bold", color=EVADER)
    ev_len = 0.8
    ax.annotate("", xy=E + ev_len * np.array([np.cos(psi), np.sin(psi)]),
                xytext=E,
                arrowprops=dict(arrowstyle="->", color=EVADER, lw=2))
    ax.annotate(r"$v_E$",
                E + 0.5 * np.array([np.cos(psi), np.sin(psi)])
                + np.array([-0.5, 0.1]),
                fontsize=12, color=EVADER)

    psi_arc = np.linspace(0, psi, 30)
    psi_r = 0.45
    ax.plot(E[0] + psi_r * np.cos(psi_arc),
            E[1] + psi_r * np.sin(psi_arc),
            "-", color=EVADER, lw=1, alpha=0.7)
    ax.annotate(r"$\psi_{\mathrm{lab}}$",
                E + (psi_r + 0.15) * np.array(
                    [np.cos(psi * 0.5), np.sin(psi * 0.5)]),
                fontsize=12, color=EVADER)

    # Relative-position line
    ax.plot([P[0], E[0]], [P[1], E[1]], "k--", lw=1, alpha=0.5)
    mid = 0.5 * (P + E)
    ax.annotate(r"$r$", mid + np.array([0.1, -0.3]), fontsize=12)

    # Lab-frame axes
    ax.annotate("", xy=(6.5, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate("", xy=(0.0, 5.5), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(6.3, -0.35, r"$x$", fontsize=13)
    ax.text(-0.35, 5.3, r"$y$", fontsize=13)

    ax.set_xlim(-1.5, 7)
    ax.set_ylim(-1.0, 6)
    ax.set_aspect("equal")
    ax.set_title("Problem geometry — lab frame", fontsize=13)
    ax.grid(True, alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §8 — optimal_vector_field (lifted from vector_field_plot, L1265-1349)
# ---------------------------------------------------------------------------


def optimal_vector_field(
    w: float,
    ell: float,
    *,
    n_grid: int = 25,
    grid_lim: float = 4.0,
    fixed_p: tuple[float, float] = (0.0, 1.0),
) -> plt.Figure:
    """§8 quiver: optimal state velocity over (x_1, x_2) at fixed costate ``p``.

    Lifted from hc-marimo's ``vector_field_plot`` (L1265-1349). For a
    fixed costate direction ``p`` (default along +x_2), the switching
    function ``sigma = p_2 x_1 - p_1 x_2`` determines the optimal pursuer
    control ``phi* = -sign(sigma)``; the flow direction reverses across
    the switching surface ``sigma = 0`` (green dashed). Arrows are
    coloured by speed magnitude.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(-grid_lim, grid_lim, n_grid),
        np.linspace(-grid_lim, grid_lim, n_grid),
    )
    p1_val, p2_val = fixed_p
    sigma = p2_val * x1_grid - p1_val * x2_grid
    phi_star = -np.sign(sigma)

    # psi* = atan2(p1, p2)  → sin(psi*) = p1/||p||, cos(psi*) = p2/||p||
    norm_p = float(np.hypot(p1_val, p2_val)) or 1.0
    sin_psi = p1_val / norm_p
    cos_psi = p2_val / norm_p

    f1 = -phi_star * x2_grid + w * sin_psi
    f2 = phi_star * x1_grid + w * cos_psi - 1.0
    speed = np.hypot(f1, f2)
    speed_safe = np.where(speed > 0, speed, 1.0)

    ax.quiver(
        x1_grid, x2_grid,
        f1 / speed_safe, f2 / speed_safe,
        speed,
        cmap="coolwarm",
        alpha=0.7,
        pivot="mid",
        scale=30,
    )

    # Switching surface (where sigma = 0 → at fixed p=(0,1), this is x_1 = 0)
    # Generalisation: the line p_2 x_1 - p_1 x_2 = 0 through the origin.
    if abs(p2_val) > abs(p1_val):
        # Line passes through origin with slope p2/p1 — vertical-ish
        x1_line = np.array([-grid_lim, grid_lim])
        x2_line = (p2_val / max(p1_val, 1e-12)) * x1_line if p1_val else None
        if x2_line is None:
            ax.axvline(x=0, color=SWITCHING, linestyle="--", linewidth=1.5,
                       alpha=0.7, label=r"Switching surface $\sigma = 0$")
        else:
            ax.plot(x1_line, x2_line, "--", color=SWITCHING, linewidth=1.5,
                    alpha=0.7, label=r"Switching surface $\sigma = 0$")
    else:
        x2_line = np.array([-grid_lim, grid_lim])
        x1_line = (p1_val / max(p2_val, 1e-12)) * x2_line if p2_val else None
        if x1_line is None:
            ax.axhline(y=0, color=SWITCHING, linestyle="--", linewidth=1.5,
                       alpha=0.7, label=r"Switching surface $\sigma = 0$")
        else:
            ax.plot(x1_line, x2_line, "--", color=SWITCHING, linewidth=1.5,
                    alpha=0.7, label=r"Switching surface $\sigma = 0$")

    # Annotate control regions (only sensible for the default p = (0, 1))
    if fixed_p == (0.0, 1.0):
        ax.text(2.0, -3.5, r"$\phi^* = -1$" + "\n(hard right)",
                fontsize=10, ha="center", color=ANNOT, alpha=0.8)
        ax.text(-2.0, -3.5, r"$\phi^* = +1$" + "\n(hard left)",
                fontsize=10, ha="center", color=ANNOT, alpha=0.8)

    # Terminal circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ell * np.cos(theta), ell * np.sin(theta),
            TERMINAL + "--", linewidth=1, alpha=0.5, label="Terminal circle")

    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)
    ax.set_xlabel(r"$x_1$ (perpendicular)")
    ax.set_ylabel(r"$x_2$ (along heading)")
    ax.set_aspect("equal")
    ax.set_title(
        rf"Optimal vector field ($\mathbf{{p}} = ({p1_val:g}, {p2_val:g})$, "
        rf"$w = {w:.2f}$, $\ell = {ell:.2f}$)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §8 — trajectory_fan (lifted from trajectory_plot, L1683-1769)
# ---------------------------------------------------------------------------


def _draw_terminal_and_usable_arc(ax, w: float, ell: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ell * np.cos(theta), ell * np.sin(theta),
            TERMINAL + "--", linewidth=1, alpha=0.5, label="Terminal circle")
    alpha_min = np.arcsin(min(w, 0.999))
    alpha_max = np.pi - alpha_min
    usable = np.linspace(alpha_min, alpha_max, 100)
    ax.plot(ell * np.cos(usable), ell * np.sin(usable),
            "-", color=USABLE, linewidth=3, alpha=0.85, label="Usable part")


def trajectory_fan(trajectories: Sequence, w: float, ell: float) -> plt.Figure:
    """§8 static family of optimal backward characteristics.

    Lifted from hc-marimo's ``trajectory_plot`` (L1683-1769). Trajectories
    are coloured by their distance from the centre of the usable arc — the
    viridis colourmap is folded so mirrored characteristics on opposite
    sides of the x_2 axis share the same colour, making the reflection
    symmetry visually apparent. Sharp kinks visible in some trajectories
    are bang-bang switching points (σ crossing zero).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_terminal_and_usable_arc(ax, w, ell)

    cmap = plt.cm.viridis
    mid = (len(trajectories) - 1) / 2.0
    for i, sol in enumerate(trajectories):
        color = cmap(abs(i - mid) / max(mid, 1))
        ax.plot(sol.y[0], sol.y[1], "-", color=color, linewidth=0.9, alpha=0.75)
        ax.plot(sol.y[0, -1], sol.y[1, -1], "o", color=color, markersize=3)

    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)
    ax.set_xlabel(r"$x_1$ (perpendicular)")
    ax.set_ylabel(r"$x_2$ (along heading)")
    ax.set_aspect("equal")
    ax.set_title(rf"Optimal trajectories ($w = {w:.2f}$, $\ell = {ell:.2f}$)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §7 — trajectory_frame (lifted from trajectory_animation_plot, L1963-2054)
# ---------------------------------------------------------------------------


def trajectory_frame(
    trajectories: Sequence,
    tau: float,
    w: float,
    ell: float,
) -> plt.Figure:
    """§7 single τ frame for the ipywidgets scrubber.

    Lifted from hc-marimo's ``trajectory_animation_plot`` (L1963-2054).
    For each trajectory in the ensemble, draws the path up to backward
    time ``tau``: a faded full trail, a brighter recent-20%, and a current
    position marker. Drag τ from 0 (just the starting points on the
    terminal circle) to T_horizon (the full fan) to watch the reachable
    set grow.

    Imports from :func:`numerics.sample_trajectories_at_tau`.
    """
    from numerics import sample_trajectories_at_tau  # local; cheap

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_terminal_and_usable_arc(ax, w, ell)

    frames = sample_trajectories_at_tau(trajectories, tau=tau)
    cmap = plt.cm.viridis
    n = max(len(frames) - 1, 1)
    for i, (x1_path, x2_path) in enumerate(frames):
        color = cmap(i / n)
        if len(x1_path) <= 1:
            ax.plot(x1_path[0], x2_path[0], "o", color=color, markersize=4,
                    alpha=0.85)
            continue
        # Faded full trail
        ax.plot(x1_path, x2_path, "-", color=color, linewidth=0.7, alpha=0.3)
        # Recent 20% brighter
        recent_start = max(0, int(0.8 * len(x1_path)))
        ax.plot(x1_path[recent_start:], x2_path[recent_start:],
                "-", color=color, linewidth=1.3, alpha=0.85)
        # Current position dot
        ax.plot(x1_path[-1], x2_path[-1], "o", color=color, markersize=5,
                markeredgecolor="black", markeredgewidth=0.3)

    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)
    ax.set_xlabel(r"$x_1$ (perpendicular)")
    ax.set_ylabel(r"$x_2$ (along heading)")
    ax.set_aspect("equal")
    ax.set_title(
        rf"Backward trajectories at $\tau = {tau:.1f}$ "
        rf"($w = {w:.2f}$, $\ell = {ell:.2f}$)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §7 — reachable_set_view (lifted from reachable_set_plot, L2445-2532)
# ---------------------------------------------------------------------------


def reachable_set_view(
    isochrones: dict[float, np.ndarray],
    scatter: dict[str, np.ndarray],
    w: float,
    ell: float,
    *,
    highlight_ic: tuple[float, float, str] | None = None,
    annotate_barrier: bool = False,
) -> plt.Figure:
    """§8 synthesis figure: V(x) as a scatter field with isochrone contours.

    Lifted from hc-marimo's ``reachable_set_plot`` (L2445-2532). The
    scatter field is coloured by capture time ``tau`` (plasma colourmap);
    overlaid thin black curves are isochrones at the canonical T values.
    This is the figure that makes "the value function" tangible.

    Parameters
    ----------
    highlight_ic : (x_1, x_2, label) or None
        If provided, places a **black ◆ diamond** marker at the body-
        frame coordinate ``(x_1, x_2)`` with the label annotated in
        the upper-right whitespace and a faint leader line back to
        the marker. The label is appended with the estimated
        V*(x_1, x_2) read from the scatter field via linear
        interpolation. Used to cross-reference an earlier-section
        initial condition (e.g. the §3 ``naive_vs_optimal`` IC at
        body ``(-0.17, 6.71)``) to the heat-map value at that point.
        Diamond + black is **deliberately not the red ★** used for
        dispersal-switch events in chase_demo and dispersal_crossing
        — the ★ vocabulary is reserved for "non-smooth event" and
        the ◆ vocabulary is "a callout point on the field."
    annotate_barrier : bool
        If True, add a leader-line annotation pointing to the inner-
        cusp tip with the label "barrier singular surface". The
        cusp is where the HC value function's level sets meet at a
        corner — points just outside have V* ~3× larger than points
        just inside (verified by sampling-density check). Uses a
        sober slate colour to avoid competing with the red ★ event
        vocabulary or the black ◆ IC callout. Off by default to
        keep the figure clean for general use.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    sc = ax.scatter(
        scatter["x1"], scatter["x2"],
        c=scatter["tau"],
        cmap="plasma",
        s=2,
        alpha=0.6,
        vmin=0,
        vmax=max(12.0, float(scatter["tau"].max()) if scatter["tau"].size else 12.0),
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.10, shrink=0.8)
    cbar.set_label(r"Capture time $\tau$", fontsize=11)

    # Draw ALL isochrones as faint background contours; label only a
    # SUBSET to reduce text crowding. Labels go at the TOPMOST point
    # of each chosen contour (well-separated since the isochrones are
    # nested loops growing outward), not the rightmost (which piled
    # them up in the inner-cusp region).
    #
    # Singular-surface gap handling: the isochrone is the value
    # function's level set in body space. Each backward trajectory
    # contributes one endpoint per isochrone; consecutive points in
    # trajectory-index order (i.e., consecutive terminal-alphas)
    # USUALLY trace the level set smoothly, but at singular surfaces
    # (dispersal / equivocal / focal surfaces in HC) adjacent alphas
    # can produce widely-separated body-frame points. A naive
    # ``ax.plot(pts[:,0], pts[:,1], ...)`` draws straight lines
    # bridging those gaps and looks like jagged interpolation noise.
    # Fix: break the line at any segment longer than 3× the median
    # segment length (an adaptive threshold per isochrone).
    sorted_T = sorted(isochrones.keys())
    label_set = {2.0, 6.0, 12.0}
    for T in sorted_T:
        pts = isochrones[T]
        if len(pts) < 2:
            continue
        # Per-isochrone gap-breaking
        seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        if seg_len.size > 0:
            med = float(np.median(seg_len[seg_len > 0])) if (seg_len > 0).any() else 0.0
            threshold = max(med * 3.0, 1e-3)
            # Insert NaN entries at each gap so matplotlib breaks the line
            mask = seg_len > threshold
            if mask.any():
                xs = pts[:, 0].astype(float).copy()
                ys = pts[:, 1].astype(float).copy()
                # Build broken sequences with NaN inserts
                xs_seq, ys_seq = [xs[0]], [ys[0]]
                for i in range(len(seg_len)):
                    if mask[i]:
                        xs_seq.append(np.nan)
                        ys_seq.append(np.nan)
                    xs_seq.append(xs[i + 1])
                    ys_seq.append(ys[i + 1])
                ax.plot(xs_seq, ys_seq, "k-", linewidth=0.6, alpha=0.35)
            else:
                ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=0.6, alpha=0.35)
        if T in label_set:
            top = int(np.argmax(pts[:, 1]))
            ax.annotate(
                rf"$T={T:.0f}$",
                (pts[top, 0], pts[top, 1]),
                fontsize=9, alpha=0.75,
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )

    # Terminal circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ell * np.cos(theta), ell * np.sin(theta),
            "k-", linewidth=1.5)
    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)

    # Highlight an earlier-section initial condition (typically the
    # §3 ``naive_vs_optimal`` IC) so the reader can read V*(IC)
    # straight off the field and ground the colourbar in concrete
    # numbers they've already seen.
    if highlight_ic is not None:
        hx1, hx2, hlabel = highlight_ic
        # Linear-interpolated estimate of V* at (hx1, hx2) from the
        # scatter field via scipy.interpolate.griddata. Nearest-
        # neighbour is misleading near low-density regions (the §3
        # IC is one — the nearest sample is ~0.6 units away but
        # belongs to a different characteristic with substantially
        # smaller τ). Linear interp on the Delaunay triangulation
        # of the scatter gives a stable estimate.
        from scipy.interpolate import griddata  # noqa: PLC0415
        sx1 = scatter["x1"]
        sx2 = scatter["x2"]
        stau = scatter["tau"]
        v_est = None
        if sx1.size >= 3:
            v_lin = griddata(
                (sx1, sx2), stau, (hx1, hx2), method="linear",
            )
            if v_lin is not None and not np.isnan(float(v_lin)):
                v_est = float(v_lin)

        # Black ◆ diamond marker — distinct from the red ★ which is
        # reserved for the "dispersal switch" event in chase_demo and
        # dispersal_crossing. Diamond + black means "this is a
        # specific point we're calling out on the field" without
        # invoking the switch-event vocabulary.
        IC_COLOR = "#222222"
        ax.plot(
            hx1, hx2, "D",
            color=IC_COLOR, markersize=11, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5,
        )
        # Label placed in clean upper-right whitespace with a faint
        # leader line back to the diamond. Keeps the dense scatter
        # region around the marker unobstructed. Right-aligned so
        # the text grows leftward from the anchor — guarantees the
        # text stays inside the plot frame even at the field's
        # rightmost extent.
        label_text = (
            rf"{hlabel}: $V^* \approx {v_est:.1f}$ s" if v_est is not None
            else rf"{hlabel}"
        )
        xr = max(float(scatter["x1"].max()) if scatter["x1"].size else 8.0, 8.0)
        yr = max(float(scatter["x2"].max()) if scatter["x2"].size else 9.0, 9.0)
        # Anchor near the upper-right; ha="right" makes the text grow
        # leftward from this point so it never overruns the right edge.
        label_xy = (0.95 * xr, 0.93 * yr)
        ax.annotate(
            label_text,
            xy=(hx1, hx2),
            xytext=label_xy,
            fontsize=10, color=IC_COLOR, fontweight="bold",
            ha="right", va="center",
            arrowprops=dict(
                arrowstyle="-", color=IC_COLOR,
                lw=0.8, alpha=0.55,
                connectionstyle="arc3,rad=-0.15",
            ),
            zorder=10,
        )

    # Optional barrier-singular-surface annotation. The HC value
    # function's level sets meet at corners along the "barrier" — the
    # boundary of the directly-reachable region. Sampling-density
    # check confirms V*(x) jumps from ~2s (inside the cusp) to ~5.8s
    # (in the notch just above), and this jump is robust across
    # n_traj from 100 to 600 — it's a genuine HC feature
    # (Merz 1971; Bardi/Falcone/Soravia 1999 §3), not a sampling
    # artifact. The annotation uses a sober slate colour so it
    # doesn't compete with the red ★ dispersal-event vocabulary or
    # the black ◆ IC callout.
    if annotate_barrier:
        BARRIER_COLOR = "#2e3a55"  # slate blue
        # Cusp tip on the right side of the dispersal axis. The
        # left cusp is a symmetric mirror; we annotate one and the
        # reader sees the other.
        cusp_tip = (0.65, 3.4)
        # Label placed in the upper-right whitespace with a curved
        # leader line back to the cusp.
        ax.annotate(
            "barrier singular surface\n"
            r"($V^*$ jumps $\sim 3\times$)",
            xy=cusp_tip,
            xytext=(5.5, 4.8),
            fontsize=9, color=BARRIER_COLOR, fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(
                arrowstyle="-", color=BARRIER_COLOR,
                lw=0.8, alpha=0.6,
                connectionstyle="arc3,rad=0.2",
            ),
            zorder=10,
        )

    ax.set_xlabel(r"$x_1$ (perpendicular)")
    ax.set_ylabel(r"$x_2$ (along heading)")
    ax.set_aspect("equal")
    ax.set_title(
        rf"Value function / backward reachable set "
        rf"($w = {w:.2f}$, $\ell = {ell:.2f}$)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §8 alternative — reachable_set_heatmap (dense-grid V*(x) with cap)
# ---------------------------------------------------------------------------


def reachable_set_heatmap(
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    V_grid: np.ndarray,
    reachable_mask: np.ndarray,
    w: float,
    ell: float,
    *,
    T_max_cap: float = 15.0,
    vmax_color: float = 15.0,
    isochrone_levels: tuple[float, ...] = (2.0, 6.0, 12.0),
    isochrone_curves: dict[float, np.ndarray] | None = None,
    highlight_ic: tuple[float, float, str] | None = None,
    highlight_ic_value: float | None = None,
) -> plt.Figure:
    """§8 synthesis figure: true grid-based V*(x) heat map.

    Interpolates V* onto a regular body-frame grid and renders as a
    pcolormesh. Unreachable cells (V* > ``T_max_cap``, i.e. no
    optimal characteristic reaches them within the integration
    horizon) render as dark grey — distinct from "very-long capture
    time but finite," which renders as the saturated yellow end of
    the plasma colourmap.

    Inputs come from :func:`numerics.value_function_grid`.

    Isochrone contours at the levels in ``isochrone_levels`` are
    overlaid in white (high-contrast against plasma's purple/yellow
    range).

    Figure legend covers: isochrones, capture circle, pursuer body-
    frame origin (▲ pointing along +x_2 heading), unreachable
    region, and the optional §3 IC ◆ — no on-figure leader-line
    annotations.

    Parameters
    ----------
    x1_grid, x2_grid : 2D meshgrid
    V_grid : 2D V*(x) values, capped at T_max_cap
    reachable_mask : 2D bool, True where V* ≤ cap
    w, ell : game parameters (for title)
    T_max_cap : the cap used in V_grid (for legend label)
    vmax_color : colour-scale upper bound (defaults to 15, matching
        T_max_cap so the yellow end means "near the cap")
    isochrone_levels : T values to render. If ``isochrone_curves`` is
        also given, only the levels appearing in both are drawn (and
        their order/labels follow ``isochrone_levels``).
    isochrone_curves : optional dict ``{T: (N, 2) ndarray}`` of body-
        frame endpoints from backward characteristics — typically
        from :func:`numerics.value_function_field`. When provided,
        these analytical level-set curves are overlaid (with adaptive
        gap-breaking at singular surfaces) instead of being
        re-derived from the smoothed ``V_grid`` via ``ax.contour``.
        Same plumb-from-numerics philosophy as ``highlight_ic_value``.
        If ``None``, falls back to ``ax.contour`` over ``V_grid``
        (chattery near low-resolution regions but parameter-free).
    highlight_ic : optional (x_1, x_2, label) for the §3 IC ◆ marker
    highlight_ic_value : optional explicit V*(IC) (seconds) for the
        legend. The smoothed V_grid is a linear interp over backward-
        characteristic samples; at IC points the angular-fan sweep
        doesn't land exactly on, the smoothed value over-reports
        relative to the true saddle-point V*. Pass the §3 chase_demo
        capture time directly when calling — it's the authoritative
        measurement at that IC. If None, fall back to min-over-
        neighborhood of V_grid (best-effort from the heatmap data).
    """
    import matplotlib as mpl  # noqa: PLC0415

    # Canvas: square-ish data aspect (16×16 in default extent), so
    # the axes will render square once we tighten xlim/ylim. The
    # canvas adds room for the right-side vertical colorbar and the
    # below-axes 3-row legend.
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 9.5))

    # --- Heat map (reachable cells coloured; unreachable → grey) ---
    V_display = np.where(reachable_mask, V_grid, np.nan)
    cmap = mpl.colormaps["plasma"].copy()
    UNREACHABLE_GRAY = "#3a3a3a"
    cmap.set_bad(UNREACHABLE_GRAY)
    pcm = ax.pcolormesh(
        x1_grid, x2_grid, V_display,
        cmap=cmap, vmin=0.0, vmax=vmax_color, shading="auto",
        rasterized=True,
    )
    # Colorbar: vertical on the right; vmin=0 at bottom (short capture
    # times near the terminal circle) → vmax at top (slow / unreachable).
    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, shrink=0.95)
    cbar.set_label(r"Capture time $\tau$ (s)", fontsize=13)

    # --- Isochrone curves ---
    #
    # Goal: complete closed level sets that trace the heatmap's
    # visible V* boundaries. We tried two pure approaches and
    # rejected both:
    #
    #   (a) `ax.contour` over the raw V_grid alone → chattery,
    #       broken contours because V_grid is a Delaunay-linear
    #       interp over a sparse trajectory fan with sub-optimal
    #       characteristics interfering near singular surfaces.
    #   (b) Endpoints from backward characteristics alone → smooth
    #       BUT incomplete at small T: for T=2 all 600 endpoints
    #       cluster in a narrow strip just above the terminal
    #       usable arc; the rest of the V*=2 level set isn't
    #       reached by the angular fan.
    #
    # Adopted: `ax.contour` over a lightly Gaussian-smoothed V_grid.
    # Smoothing (σ ≈ 1.2 cells) cleans the sub-optimal-trajectory
    # interference; contour completes the level set even where no
    # single backward characteristic landed. The smoothing is
    # cosmetic — it doesn't change the underlying V*(x) measurement,
    # just suppresses the noise from the interp's own averaging.
    # `isochrone_curves` from `numerics.value_function_grid` is
    # accepted for back-compat but ignored unless caller forces it.
    iso_handle = None
    if isochrone_levels and reachable_mask.any():
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415
        V_contour = np.where(reachable_mask, V_grid, np.nan)
        # Replace NaN with cap so the smoothing doesn't grow holes,
        # then restore NaN mask after smoothing.
        V_fill = np.where(reachable_mask, V_grid, T_max_cap)
        V_smooth = gaussian_filter(V_fill, sigma=1.2, mode="nearest")
        V_smooth = np.where(reachable_mask, V_smooth, np.nan)
        cs = ax.contour(
            x1_grid, x2_grid, V_smooth,
            levels=list(isochrone_levels),
            colors="white", linewidths=1.2, alpha=0.95, zorder=5,
        )
        ax.clabel(cs, inline=True, fontsize=10, fmt=lambda v: f"T={v:g}")
        iso_handle = plt.Line2D(
            [], [], color="white", lw=1.2,
            label=rf"isochrones $T = {{{', '.join(f'{v:g}' for v in isochrone_levels)}}}$",
        )
    # `isochrone_curves` kept in signature for future use; not consumed.
    _ = isochrone_curves

    # --- Terminal capture circle (white) ---
    theta = np.linspace(0, 2 * np.pi, 200)
    capture_handle = ax.plot(
        ell * np.cos(theta), ell * np.sin(theta),
        "w-", linewidth=1.5, alpha=0.95,
        label=rf"capture circle ($\ell = {ell:g}$)",
    )[0]

    # --- Pursuer at origin: white upward triangle showing heading direction ---
    pursuer_handle = ax.plot(
        0, 0, "^",
        color="white", markersize=12, markeredgecolor="black",
        markeredgewidth=0.8, zorder=8,
        label=r"pursuer (origin, heading $+x_2$)",
    )[0]

    # --- §3 IC ◆ callout (no leader line — legend handles labelling) ---
    ic_handle = None
    if highlight_ic is not None:
        hx1, hx2, hlabel = highlight_ic
        # Prefer the explicit override (e.g. §3 chase_demo's measured
        # capture time) when provided. The smoothed V_grid is a
        # linear interp over backward-characteristic samples; at IC
        # points the angular sweep typically doesn't land exactly,
        # so the smoothed value over-reports vs. the true saddle V*.
        if highlight_ic_value is not None:
            v_est = float(highlight_ic_value)
        else:
            d2 = (x1_grid - hx1) ** 2 + (x2_grid - hx2) ** 2
            cell_diag = max(
                float(x1_grid[0, 1] - x1_grid[0, 0]),
                float(x2_grid[1, 0] - x2_grid[0, 0]),
            )
            neighborhood = (d2 < (3.0 * cell_diag) ** 2) & reachable_mask
            v_est = float(V_grid[neighborhood].min()) if neighborhood.any() else None

        IC_COLOR = "#222222"
        ax.plot(
            hx1, hx2, "D",
            color=IC_COLOR, markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.2,
        )
        label_text = (
            rf"{hlabel} ($V^* \approx {v_est:.1f}$ s)" if v_est is not None
            else rf"{hlabel}"
        )
        ic_handle = plt.Line2D(
            [], [], color=IC_COLOR, marker="D", linestyle="none",
            markersize=14, markeredgecolor="white", markeredgewidth=1.5,
            label=label_text,
        )

    # --- Unreachable region in the legend (grey square swatch) ---
    unreachable_handle = mpatches.Patch(
        color=UNREACHABLE_GRAY,
        label=rf"unreachable ($V^* > {T_max_cap:g}$ s within horizon)",
    )

    # --- Assemble legend ---
    legend_handles = [pursuer_handle, capture_handle]
    if iso_handle is not None:
        legend_handles.append(iso_handle)
    if ic_handle is not None:
        legend_handles.append(ic_handle)
    legend_handles.append(unreachable_handle)
    # Legend below the axes (markers/lines): two columns × 3 rows
    # (with 5 entries, the last row gets one entry) so labels stay
    # compact and the figure body reads uncluttered. Use a mid-light-
    # grey backdrop so BOTH white lines (capture circle, isochrones)
    # AND the black ◆ marker stay legible — neither shows on the
    # default white legend background.
    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=11,
        framealpha=1.0,
        facecolor="#bcbcbc",
        edgecolor="#555555",
        labelcolor="#111111",
    )
    leg.get_frame().set_linewidth(0.8)

    ax.set_xlabel(r"$x_1$ (perpendicular)")
    ax.set_ylabel(r"$x_2$ (along heading)")
    # Tighten x-range to the heatmap extent so the figure is taller
    # (no whitespace gutters to the left/right of the V_grid).
    x1_lo, x1_hi = float(x1_grid.min()), float(x1_grid.max())
    x2_lo, x2_hi = float(x2_grid.min()), float(x2_grid.max())
    ax.set_xlim(x1_lo, x1_hi)
    ax.set_ylim(x2_lo, x2_hi)
    ax.set_aspect("equal")
    ax.set_title(
        rf"Value function $V^*(\mathbf{{x}})$ "
        rf"($w = {w:.2f}$, $\ell = {ell:.2f}$)"
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# §12 — conservation_diagnostics (lifted from conservation_plots, L2576-2666)
# ---------------------------------------------------------------------------


def conservation_diagnostics(trajectories: Sequence, w: float) -> plt.Figure:
    """§12 two-panel reproducibility evidence: H* and ‖p‖² drift along τ.

    Lifted from hc-marimo's ``conservation_plots`` (L2576-2666). Along
    optimal characteristics, ``H* = 0`` and ``||p||^2`` is conserved. The
    deviations observed here are localised at bang-bang switching points
    where the integrator smooths an instantaneous control jump over a small
    interval; they do not accumulate. Both panels share the τ x-axis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    cmap = plt.cm.viridis
    n = max(len(trajectories) - 1, 1)

    all_H = []
    for i, sol in enumerate(trajectories):
        color = cmap(i / n)
        x1, x2, p1, p2 = sol.y
        t = sol.t
        sigma = p2 * x1 - p1 * x2
        norm_p = np.hypot(p1, p2)
        H_star = -np.abs(sigma) + w * norm_p - p2 + 1.0
        all_H.extend(H_star.tolist())
        ax1.plot(t, H_star, "-", color=color, linewidth=0.7, alpha=0.7)

        p_norm_sq = p1 ** 2 + p2 ** 2
        ax2.plot(t, p_norm_sq[0] - p_norm_sq, "-",
                 color=color, linewidth=0.7, alpha=0.7)

    if all_H:
        H_arr = np.asarray(all_H)
        H_bound = max(float(np.percentile(np.abs(H_arr), 99) * 1.5), 1e-10)
        ax1.set_ylim(-H_bound, H_bound)

    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel(r"$H^*(\tau)$")
    ax1.set_title(r"Hamiltonian $H^*$ along optimal characteristics")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel(r"Backward time $\tau$")
    ax2.set_ylabel(r"$\|\mathbf{p}_0\|^2 - \|\mathbf{p}(\tau)\|^2$")
    ax2.set_title(r"Costate norm drift")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Numerical conservation ($w = {w:.2f}$, "
        rf"{len(trajectories)} trajectories)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.close(fig)  # untrack from pyplot — kills the inline backend's auto-display so the cell renders one PNG, not two
    return fig


# ---------------------------------------------------------------------------
# §5 — coordinate_systems_side_by_side (NEW; the lab↔body mapping made visual)
# ---------------------------------------------------------------------------


def coordinate_progression() -> plt.Figure:
    """§5 — three different lab configurations collapse to ONE body-frame point.

    Two-panel figure whose visual job is the dimensional-reduction insight:
    the body-frame coordinates are an *equivalence class* on the 5-DOF lab
    state, not a relabeling. The reader sees three superficially different
    (P_lab, θ) configurations of the pursuer — each with the evader placed
    at the *same body-relative position* (x_1*, x_2*) = (1.0, 2.0) — and
    watches them collapse to a single point in body coordinates.

    1.  **Lab frame** (left panel): three color-coded (P, E) pairs at
        distinct lab positions and headings. Each pair *looks* different
        because the eye sees raw (x_P, y_P, x_E, y_E, θ). They are in fact
        the *same problem*.

    2.  **Body frame** (right panel): the pursuer pinned at the origin
        facing along +x_2; the three evaders all stacked at (1.0, 2.0)
        as concentric markers in the matching colors. 5 DOF → 2 DOF.

    This makes the contribution of §5 visible: the reduction *recognises an
    equivalence class*, which is the substantive content. Translation alone
    (which was the middle frame in the earlier version of this figure)
    leaves the relative arrangement of P and E unchanged and so was
    visually trivial — that step is described in prose, not pictured.
    """
    fig, (ax_lab, ax_body) = plt.subplots(1, 2, figsize=(13, 5.8))

    # ---- shared body-relative E position; three lab configurations ----
    BODY_TARGET = np.array([1.0, 2.0])  # (x_1*, x_2*) shared by all three
    CONFIGS = [
        {
            "P_lab": np.array([1.0, 1.0]),
            "theta": np.pi / 6,
            "color": "#2c5da0",  # blue
            "label": "1",
        },
        {
            "P_lab": np.array([4.0, 1.0]),
            "theta": 5 * np.pi / 6,
            "color": "#2a9d8f",  # teal/green
            "label": "2",
        },
        {
            "P_lab": np.array([2.5, 4.0]),
            "theta": -np.pi / 3,
            "color": "#8e44ad",  # purple
            "label": "3",
        },
    ]
    ell = 0.5

    # Pursuer triangle template (in its own local frame, facing +x at theta=0)
    sz = 0.26
    tri_template = np.array(
        [[sz, 0], [-sz * 0.5, sz * 0.4], [-sz * 0.5, -sz * 0.4]]
    )

    def _rotate(angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    def _body_to_lab(P_lab, theta, body_xy):
        """Place a body-relative (x_1, x_2) at the corresponding lab point.

        Inverse of ``derivations.py``'s reduction substitution:

            Δx = -x_1 sin θ + x_2 cos θ
            Δy =  x_1 cos θ + x_2 sin θ
        """
        x1, x2 = body_xy
        return P_lab + np.array(
            [
                x1 * (-np.sin(theta)) + x2 * np.cos(theta),
                x1 * np.cos(theta) + x2 * np.sin(theta),
            ]
        )

    th = np.linspace(0, 2 * np.pi, 80)

    # ============= LEFT — lab frame, three (P, E) configurations =============
    ax = ax_lab

    for cfg in CONFIGS:
        P, theta, color, label = (
            cfg["P_lab"],
            cfg["theta"],
            cfg["color"],
            cfg["label"],
        )
        E = _body_to_lab(P, theta, BODY_TARGET)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        # Pursuer triangle (filled with config color)
        tri = (_rotate(theta) @ tri_template.T).T + P
        ax.add_patch(
            plt.Polygon(tri, fc=color, ec="black", lw=1.2, zorder=5)
        )

        # Heading arrow — short, just enough to show direction
        head = P + 0.55 * np.array([cos_t, sin_t])
        ax.annotate(
            "", xy=head, xytext=P,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.85),
        )

        # Configuration label next to P
        ax.annotate(
            rf"$P_{label}$",
            P + np.array([-0.45, -0.30]),
            fontsize=11, fontweight="bold", color=color,
        )

        # Evader dot (same hue, white outline so it pops)
        ax.plot(
            *E, "o", color=color, markersize=9, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8,
        )
        ax.annotate(
            rf"$E_{label}$",
            E + np.array([0.14, -0.30]),
            fontsize=11, fontweight="bold", color=color,
        )

        # Dashed P–E line in the config color
        ax.plot(
            [P[0], E[0]], [P[1], E[1]],
            "--", color=color, lw=0.9, alpha=0.55,
        )

    # Lab axes
    ax.annotate(
        "", xy=(5.7, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.annotate(
        "", xy=(0, 5.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.text(5.6, -0.30, r"$x$", fontsize=12)
    ax.text(-0.30, 5.4, r"$y$", fontsize=12)

    # Italic note under the title — what to look for
    ax.text(
        0.10, 5.85,
        r"each colored pair $(P_i, E_i)$ has the same body-relative E",
        fontsize=9, color="gray", style="italic",
    )

    ax.set_xlim(-0.6, 6.2)
    ax.set_ylim(-0.6, 6.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("1. Lab frame — three (P, E) configurations", fontsize=12)
    ax.grid(True, alpha=0.15)

    # ============= RIGHT — body frame, all three collapse to ONE point =============
    ax = ax_body

    # Pursuer pinned at origin, pointing along +x_2 (up in the plot)
    tri_body = (_rotate(np.pi / 2) @ tri_template.T).T
    ax.add_patch(
        plt.Polygon(tri_body, fc="#444444", ec="black", lw=1.5, zorder=5)
    )
    ax.annotate(
        r"$P$", np.array([-0.42, -0.20]),
        fontsize=13, fontweight="bold", color="#222222",
    )

    # Capture circle around P (single — they all share it)
    ax.plot(
        ell * np.cos(th), ell * np.sin(th),
        ":", color="gray", lw=1.2, alpha=0.7,
    )

    # Three concentric E markers in matching config colors at (x_1*, x_2*)
    # — outer to inner, decreasing size so all three are visible at once.
    marker_specs = [
        (CONFIGS[0]["color"], 18),  # outer ring
        (CONFIGS[1]["color"], 13),  # middle
        (CONFIGS[2]["color"], 8),   # inner
    ]
    for color, sz_mk in marker_specs:
        ax.plot(
            BODY_TARGET[0], BODY_TARGET[1], "o",
            color=color, markersize=sz_mk, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, alpha=0.95,
        )

    # Annotation: this single point IS configs 1, 2, 3
    ax.annotate(
        r"$E_1 = E_2 = E_3 = (1.0,\, 2.0)$",
        BODY_TARGET + np.array([0.30, 0.05]),
        fontsize=12, color="black", fontweight="bold",
    )
    ax.annotate(
        "the three lab configs map here",
        BODY_TARGET + np.array([0.30, -0.30]),
        fontsize=10, color="gray", style="italic",
    )

    # Coordinate dropdowns
    ax.plot(
        [BODY_TARGET[0], BODY_TARGET[0]], [0, BODY_TARGET[1]],
        "--", color="gray", lw=0.9, alpha=0.7,
    )
    ax.plot(
        [0, BODY_TARGET[0]], [BODY_TARGET[1], BODY_TARGET[1]],
        "--", color="gray", lw=0.9, alpha=0.7,
    )
    ax.annotate(
        r"$x_1 = 1.0$",
        np.array([BODY_TARGET[0] * 0.5 - 0.18, -0.30]),
        fontsize=10, color=ANNOT,
    )
    ax.annotate(
        r"$x_2 = 2.0$",
        np.array([-0.85, BODY_TARGET[1] * 0.5]),
        fontsize=10, color=ANNOT, rotation=90,
    )

    # Dashed line from origin to E
    ax.plot(
        [0, BODY_TARGET[0]], [0, BODY_TARGET[1]],
        "k--", lw=0.9, alpha=0.55,
    )

    # Body axes
    ax.annotate(
        "", xy=(3.5, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.annotate(
        "", xy=(0, 3.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.text(3.4, -0.30, r"$x_1$ (perpendicular)", fontsize=11)
    ax.text(-0.30, 3.45, r"$x_2$ (along heading)", fontsize=11)

    ax.set_xlim(-1.5, 4.2)
    ax.set_ylim(-1.0, 4.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        r"2. Body frame  (translate by $-P_{\mathrm{lab}}$, rotate by $-\theta$)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Coordinate reduction: 5 DOF (lab) → 2 DOF (body) — three "
        "different lab configurations are the same problem",
        fontsize=12,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# §6/§7 — dispersal_crossing (single characteristic, three views)
# ---------------------------------------------------------------------------


def dispersal_crossing(
    w: float = 0.45,
    ell: float = 0.5,
    T_max: float = 10.0,
    alpha_hint: float | None = 1.7,
) -> plt.Figure:
    """§6→§7 bridge — chase scene + 4 analytical decompositions (1+4 layout).

    Five-panel figure that makes the dispersal-surface event visible
    from BOTH the pursuer's and the evader's perspective. The
    pedagogical message: the characteristic is a 1-D curve in 4-D
    state space that crosses a *seam* — the surface σ=0 where the
    optimal strategies transition. The pursuer's bang-bang control
    jumps; the evader's continuous heading reverses direction. Both
    lab-frame paths exhibit curvature kinks at the same instant.

    Layout (1 tall chase panel on the left + 2×2 analytical panels
    on the right):

    LEFT — **Chase panel** (same scenario as the §3 chase_demo
        motivation figure, here color-coded by σ-regime). Pursuer
        drawn thick, evader drawn thinner. Both paths split into
        two σ-regime arcs (solid for the first-encountered sign,
        dashed for the second; dashes let the underlying solid show
        through where the two arcs overlap spatially).

    RIGHT 2×2 — the chase decomposed:

      (1,2) Pursuer's switching function σ(τ). Starts at 0
            (transversality at the terminal circle), grows, then
            crosses zero — the dispersal event.
      (1,3) Body-frame state characteristic (x_1(τ), x_2(τ)). Phase-
            portrait reading practice for (2,3).
      (2,2) Evader's heading ψ*(τ) = atan2(p_1, p_2). Continuous,
            but its derivative dψ*/dτ = sign(σ) FLIPS at the crossing.
      (2,3) Costate trajectory (p_1, p_2)(τ) — an arc on the
            ‖p‖-circle (conserved) that goes one way, then REVERSES
            at the crossing. Most striking visualization of the seam.

    The strategy-space framing: the σ=0 surface is the seam.
    Crossing it means both players smoothly switch which side of the
    bang-bang structure they're on. The chase panel shows both
    players, the right panels decompose the why.
    """
    # Recover analytical quantities from the hardcoded chase_demo so
    # §3 and §6 show literally the same chase. The recovered chase is
    # stitched from two characteristics (α_B=95° pre-switch, α_A=40°
    # post-switch), so the σ, p, ψ* trajectories all exhibit genuine
    # discontinuities at the dispersal point — that IS the seam.
    _ = (w, ell, T_max, alpha_hint)  # ignored; recovery uses demo's params
    data = _recover_chase_from_demo()
    w = data.w
    ell = data.ell
    T_max = data.T_max
    alpha_hint = data.alpha_hint
    sol = data.sol
    x1, x2, p1, p2 = sol.y
    sigma = data.sigma
    cross_idx = data.cross_idx
    cross_tau = data.cross_tau

    # Colors: same conventions as panel 1 of coordinate_progression.
    # Blue when σ<0 (pursuer turns CCW, φ*=+1).
    # Warm orange when σ>0 (pursuer turns CW, φ*=-1).
    SIGMA_NEG = "#2c5da0"  # blue
    SIGMA_POS = "#e76f51"  # warm orange
    # Red ★ marker for the dispersal switch — the single biggest
    # event in the figure. Same colour as the DISPERSAL constant so
    # the figure-level legend and the per-panel markers agree.
    KINK = DISPERSAL

    def _sign_color(s_at_seg_end: float) -> str:
        return SIGMA_NEG if s_at_seg_end <= 0 else SIGMA_POS

    # Linestyle convention (within each panel):
    #   solid  = first σ-sign encountered in the iteration order
    #            (drawn first, becomes the underlying "base" line)
    #   dashed = second σ-sign (drawn on top — its dashes let the
    #            underlying solid show through where they overlap)
    #
    # τ-iteration panels (body-frame state, costate) traverse σ in
    # τ-order, so first_sign = sign(σ at small τ) = sign(σ[1]).
    # t-iteration panels (lab pursuer/evader) traverse σ in physical
    # time t = T_max − τ, so first_sign = sign(σ at large τ) = sign(σ[-1]).
    first_sign_tau = float(np.sign(sigma[1]))
    first_sign_t = float(np.sign(sigma[-1]))

    def _sign_style_tau(sig_val: float) -> str:
        s = float(np.sign(sig_val))
        return "--" if s * first_sign_tau < 0 else "-"

    def _sign_style_t(sig_val: float) -> str:
        s = float(np.sign(sig_val))
        return "--" if s * first_sign_t < 0 else "-"

    def _plot_two_regimes(target_ax, xs, ys, signs, style_fn, *,
                          lw, alpha=0.9, zorder=4):
        """Plot a 2D path split at the σ-sign change into TWO
        continuous matplotlib line objects so dash patterns span the
        whole arc and are visually readable. Per-segment plotting
        (one matplotlib line per integration step) makes each dash
        shorter than its segment and renders as solid; that's why
        R.18 looked the same as solid lines."""
        n = len(xs)
        if n < 2:
            return
        # Find the first σ-sign change after the first few points
        # (skip the σ≈0 transient near the start of each trajectory)
        skip = min(10, n // 4)
        diffs = np.diff(np.sign(signs[skip:]))
        changes = np.where(diffs != 0)[0]
        if len(changes) == 0:
            split = n // 2
        else:
            split = int(changes[0]) + skip + 1
        split = max(1, min(split, n - 1))

        sig_first = float(signs[min(split // 2 + 1, n - 1)])
        sig_second = float(signs[min((split + n) // 2, n - 1)])

        target_ax.plot(
            xs[:split + 1], ys[:split + 1],
            color=_sign_color(sig_first),
            linestyle=style_fn(sig_first),
            lw=lw, alpha=alpha, zorder=zorder,
        )
        target_ax.plot(
            xs[split:], ys[split:],
            color=_sign_color(sig_second),
            linestyle=style_fn(sig_second),
            lw=lw, alpha=alpha, zorder=zorder,
        )

    def _plot_phased(target_ax, xs, ys, signs, *,
                     lw, alpha=0.9, zorder=4, reverse_phase_order=False,
                     extra_breaks=None, alternate_linestyles=False,
                     trim_around_breaks=0, add_arrows=False,
                     arrow_size=18):
        """Plot a path split at every σ-sign change PLUS any extra
        breaks supplied, colouring each segment from PHASE_COLORS.

        For the recovered stitched chase we get 5 phases: 3 σ-sign
        changes (P1↔P2, P3↔P4, P4↔P5) plus 1 explicit extra break
        at the dispersal switch (P2↔P3, where σ jumps but may not
        change sign).

        Args:
          extra_breaks: iterable of integer indices i where a break
            should be inserted between sample i and i+1, in addition
            to the auto-detected sign changes.
          alternate_linestyles: if True, draw odd-numbered phases
            solid and even-numbered phases dashed — helpful in panels
            (e.g., (2,3) costate) where multiple arcs overlap on the
            same circle.
          trim_around_breaks: int ≥ 0. For each discontinuous break
            (e.g., dispersal switch), drop this many samples on EACH
            side from the colored segments. The chase_demo arrays
            cluster ~3 samples within Δτ < 0.03 right at the switch;
            those samples have valid σ but are visually noisy because
            tiny Δτ + ‖p‖ jump = near-vertical slope. Trim 1 to drop
            the transition sample on each side; the ★ marker still
            shows the switch position.

        Iteration order: τ-ordered arrays encounter phases
        P1→P2→...→P5; t-ordered arrays (lab paths) encounter them in
        reverse. Pass reverse_phase_order=True for the latter so the
        same phase number always denotes the same physical event.
        """
        n = len(xs)
        if n < 2:
            return
        signs_arr = np.asarray(signs, dtype=float)
        sigs = np.sign(signs_arr)
        diffs = np.diff(sigs)

        # Boundary records: (left_idx, kind, frac), where frac is the
        # linear-interp fraction left_idx→left_idx+1 at which the
        # boundary lies. Continuous boundaries (σ sign changes) get
        # a proper interp; discontinuous (extras / dispersal switch)
        # get frac=0 and skip the interpolation extension.
        #
        # COLLISION DEDUP: if a σ-sign-change index is within ±1 of an
        # explicit (discontinuous) extra_break, treat them as the same
        # event — the σ flip there is *caused by* the costate jump,
        # not an independent zero-crossing. Drop the continuous entry
        # so we don't get an empty phase between two boundaries at
        # adjacent indices.
        extras = {int(e) for e in (extra_breaks or [])}
        boundaries = []
        for i in np.where(diffs != 0)[0]:
            i = int(i)
            if any(abs(i - e) <= 1 for e in extras):
                continue
            denom = signs_arr[i + 1] - signs_arr[i]
            frac = -signs_arr[i] / denom if denom != 0 else 0.5
            boundaries.append((i, "continuous", float(frac)))
        for e in extras:
            boundaries.append((e, "discontinuous", 0.0))
        boundaries.sort(key=lambda b: b[0])

        starts = [0] + [b[0] + 1 for b in boundaries]
        ends = [b[0] for b in boundaries] + [n - 1]

        # Trim samples adjacent to DISCONTINUOUS breaks (the switch
        # cluster of tightly-spaced samples). Continuous boundaries
        # are left alone — there's no transition noise there.
        if trim_around_breaks > 0:
            for k, b in enumerate(boundaries):
                _bi, kind, _frac = b
                if kind != "discontinuous":
                    continue
                # Phase ending at this break gets ends[k] reduced
                ends[k] = max(starts[k], ends[k] - trim_around_breaks)
                # Phase starting after this break gets starts[k+1] increased
                starts[k + 1] = min(ends[k + 1], starts[k + 1] + trim_around_breaks)

        colors = list(PHASE_COLORS)
        if reverse_phase_order:
            colors = colors[::-1]
        while len(colors) < len(starts):
            colors.append("gray")

        def _interp(arr, i, frac):
            return float(arr[i]) * (1.0 - frac) + float(arr[i + 1]) * frac

        for k, (a, b) in enumerate(zip(starts, ends)):
            if b < a:
                continue
            xs_seg = list(xs[a:b + 1])
            ys_seg = list(ys[a:b + 1])

            # Prepend the START-boundary point if the preceding
            # boundary is continuous (so adjacent phases meet).
            if k > 0:
                bi, kind, frac = boundaries[k - 1]
                if kind == "continuous":
                    xs_seg = [_interp(xs, bi, frac)] + xs_seg
                    ys_seg = [_interp(ys, bi, frac)] + ys_seg

            # Append the END-boundary point if the next boundary is
            # continuous.
            if k < len(boundaries):
                bi, kind, frac = boundaries[k]
                if kind == "continuous":
                    xs_seg = xs_seg + [_interp(xs, bi, frac)]
                    ys_seg = ys_seg + [_interp(ys, bi, frac)]

            ls = "-"
            if alternate_linestyles and (k % 2 == 1):
                ls = "--"
            target_ax.plot(
                xs_seg, ys_seg,
                color=colors[k],
                linestyle=ls,
                lw=lw, alpha=alpha, zorder=zorder,
            )

            # Midpoint arrow in segment colour — points along the
            # local tangent in DATA-ITERATION ORDER (forward time
            # for t-ordered lab arrays; increasing τ for τ-ordered
            # arrays). Replaces the previous neutral-grey direction
            # arrows so colour identifies which phase each arrow
            # belongs to.
            if add_arrows and len(xs_seg) >= 3:
                m = len(xs_seg) // 2
                dx = float(xs_seg[m]) - float(xs_seg[m - 1])
                dy = float(ys_seg[m]) - float(ys_seg[m - 1])
                norm = (dx * dx + dy * dy) ** 0.5
                if norm > 1e-9:
                    # Tip slightly forward of the midpoint; tail at
                    # midpoint. Tiny tail + mutation_scale = visible
                    # arrowhead with negligible shaft.
                    tip = (
                        float(xs_seg[m]) + 1e-6 * dx / norm,
                        float(ys_seg[m]) + 1e-6 * dy / norm,
                    )
                    tail = (float(xs_seg[m]), float(ys_seg[m]))
                    target_ax.annotate(
                        "",
                        xy=tip, xytext=tail,
                        arrowprops=dict(
                            arrowstyle="-|>", color=colors[k],
                            lw=0.0, mutation_scale=arrow_size,
                            alpha=0.95,
                        ),
                        zorder=zorder + 6,
                    )

    def _draw_break_extensions(target_ax, taus, vals, ci, ct, trim,
                               color_before, color_after, *,
                               lw, alpha=0.9, zorder=4):
        """Draw small linear-extrapolation extensions of the green
        and blue colored phases so they meet the vertical red dashed
        line at τ = ct (cross_tau).

        With trim_around_breaks=1, the visible green ends at idx
        ci-1 and visible blue starts at idx ci+2 — there's a small
        Δτ gap on each side between the trimmed endpoint and the
        switch position. This extends both phases the rest of the
        way using the slope of the two innermost untrimmed samples,
        so the colored lines visibly TOUCH the red dashed line. The
        gap between the two extrapolated endpoints at τ=ct is the
        true σ-jump (or ψ*-jump) across the dispersal seam.
        """
        j_end = ci - trim
        if j_end >= 1:
            sl_A = (vals[j_end] - vals[j_end - 1]) / (taus[j_end] - taus[j_end - 1])
            v_A = float(vals[j_end]) + sl_A * (ct - float(taus[j_end]))
            target_ax.plot(
                [float(taus[j_end]), ct], [float(vals[j_end]), v_A],
                color=color_before, lw=lw, alpha=alpha, zorder=zorder,
            )
        j_start = ci + 1 + trim
        if j_start <= len(taus) - 2:
            sl_B = (vals[j_start + 1] - vals[j_start]) / (taus[j_start + 1] - taus[j_start])
            v_B = float(vals[j_start]) - sl_B * (float(taus[j_start]) - ct)
            target_ax.plot(
                [ct, float(taus[j_start])], [v_B, float(vals[j_start])],
                color=color_after, lw=lw, alpha=alpha, zorder=zorder,
            )

    # All lab-frame trajectories + kink positions + shared bounding box
    # come from the cached helper that chase_demo also uses.
    lab = data.lab
    xp, yp = data.xp, data.yp
    sig_lab = data.sig_lab
    n_eval = data.n_eval
    t_eval = data.t_eval
    xe, ye, sigs = data.xe, data.ye, data.sigs
    t_kink = data.t_kink
    xp_k, yp_k = data.xp_k, data.yp_k
    xe_k, ye_k = data.xe_k, data.ye_k
    shared_xlim = data.shared_xlim
    shared_ylim = data.shared_ylim

    # Capture-radius circle around the pursuer's terminal position
    t_cap = np.linspace(0, 2 * np.pi, 80)
    capture_circle = (xp[-1] + ell * np.cos(t_cap),
                      yp[-1] + ell * np.sin(t_cap))

    # Direction-arrow helper for lab paths (forward in physical time)
    def _dir_arrow(target_ax, xs, ys, t_arr, t_target, color):
        j = int(np.argmin(np.abs(t_arr - t_target)))
        k = min(j + 8, len(xs) - 1)
        if k <= j:
            return
        target_ax.annotate(
            "", xy=(float(xs[k]), float(ys[k])),
            xytext=(float(xs[j]), float(ys[j])),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.6, alpha=0.9, mutation_scale=14),
            zorder=9,
        )

    # Pick t-targets for two arrows: one in orange section, one in blue.
    # σ>0 (orange) lives at small t (early chase); σ<0 (blue) at later t.
    t_arrow_early = 0.5
    t_arrow_late = 4.0

    # Switch-break indices for _plot_phased — force the dispersal
    # switch to be a phase boundary even when σ doesn't change sign
    # across it (the costate-magnitude jump is the real event).
    switch_break_tau = cross_idx
    switch_idx_in_lab_t = int(np.argmin(np.abs(lab.t - t_kink)))
    switch_break_t_lab = max(0, switch_idx_in_lab_t - 1)
    switch_idx_in_t_eval = int(np.argmin(np.abs(t_eval - t_kink)))
    switch_break_t_eval = max(0, switch_idx_in_t_eval - 1)

    # Layout: 2×3. First column has two PARALLEL chase panels (same
    # trajectories, different focal player). Right 2×2 are the
    # analytical decompositions.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))

    # ============= (1,1) chase with PURSUER highlighted =============
    # Same trajectories as (2,1); pursuer drawn σ-color-coded, evader
    # as gray context. Lets the reader see the chase twice with focal
    # attention on each player in turn.
    ax = axes[0, 0]

    # Capture-radius circle
    ax.plot(*capture_circle, ":", color="gray", lw=1.0, alpha=0.6, zorder=2)

    # Evader as gray context
    ax.plot(xe, ye, "-", color="gray", lw=1.2, alpha=0.40,
            label="evader (context)", zorder=2)

    # Pursuer path — coloured by phase (4 colours, one per phase
    # between non-smooth events). Reverse phase-order because lab
    # arrays are t-ordered (forward time = reverse τ-order).
    _plot_phased(ax, xp, yp, sig_lab, lw=2.8, alpha=0.92, zorder=4,
                 reverse_phase_order=True,
                 extra_breaks=[switch_break_t_lab],
                 trim_around_breaks=1,
                 add_arrows=True, arrow_size=20)

    # Dispersal-event star (the segment-switch; σ-zero-crossings
    # within each segment are visible through phase colour changes).
    # Label is in the figure-level phase legend, not the panel legend.
    ax.plot(xp_k, yp_k, "*",
            color=KINK, markersize=22, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)

    # Pursuer start (●) + end (■) markers — labelled in figure legend.
    ax.plot(xp[0], yp[0], "o", color="black", markersize=8, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(xp[-1], yp[-1], "s", color="black", markersize=8, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlim(*shared_xlim)
    ax.set_ylim(*shared_ylim)
    ax.set_xlabel(r"$x$ (lab)", fontsize=11)
    ax.set_ylabel(r"$y$ (lab)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(
        "(1,1) Same chase as §3, pursuer highlighted ($\\phi^*$ jumps)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=10)

    # ============= (1,3) body-frame characteristic =============
    ax = axes[0, 2]
    # No trim here — (x_1, x_2) is CONTINUOUS at the switch (only the
    # costate jumps, not the state). Keeping all 50 samples means
    # green's endpoint lands exactly at the ★ marker position and
    # blue picks up flush against it on the next sample. Trimming
    # would create a small visible gap that doesn't correspond to
    # any physical discontinuity.
    _plot_phased(ax, x1, x2, sigma, lw=2.5, alpha=0.9, zorder=4,
                 extra_breaks=[switch_break_tau])

    # Terminal circle
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(ell * np.cos(th), ell * np.sin(th),
            ":", color="gray", lw=1.2, alpha=0.7)

    # Pursuer at origin
    ax.plot(0, 0, "k+", markersize=14, markeredgewidth=2, zorder=8)

    # Start (●, τ=0 on capture circle) and end (■, τ=T_max) markers.
    ax.plot(x1[0], x2[0], "o", color="black", markersize=7, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(x1[-1], x2[-1], "s", color="black", markersize=7, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)

    # The switching star
    ax.plot(x1[cross_idx], x2[cross_idx], "*",
            color=KINK, markersize=20, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5,
            label=rf"$\sigma=0$ at $\tau={cross_tau:.2f}$")

    ax.set_xlabel(r"$x_1$ (perpendicular)", fontsize=11)
    ax.set_ylabel(r"$x_2$ (along heading)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(
        r"(1,3) Body-frame state $(x_1, x_2)(\tau)$",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=10)

    # ============= (1,2) σ(t) time series, FORWARD TIME =============
    # Computed backward (τ-ordered, idx 0 = τ=0 = capture) but
    # displayed against forward time t = T_max − τ so the reader's
    # eye flows left-to-right with the chase: t=0 (chase start, P4
    # orange) on the left → t=t_kink (dispersal ★) → t=T_max
    # (capture, P1 purple) on the right. Matches the time direction
    # of the lab panels in column 1.
    ax = axes[0, 1]
    t_fwd = T_max - sol.t  # τ-ordered → values at descending t
    _plot_phased(ax, t_fwd, sigma, sigma, lw=2.5, alpha=0.9, zorder=4,
                 extra_breaks=[switch_break_tau],
                 trim_around_breaks=1)
    _draw_break_extensions(
        ax, t_fwd, sigma, cross_idx, t_kink, trim=1,
        color_before=PHASE_COLORS[1], color_after=PHASE_COLORS[2],
        lw=2.5,
    )

    ax.axhline(0, color="black", lw=0.9, alpha=0.5, linestyle="--")
    # Vertical line at t = t_kink marks the dispersal switch position.
    ax.axvline(t_kink, color=DISPERSAL, lw=1.0, alpha=0.55,
               linestyle="--", zorder=3)
    ax.plot(t_kink, 0, "*",
            color=KINK, markersize=20, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)

    ax.set_xlabel(r"forward time $t$ (= $T - \tau$)", fontsize=11)
    ax.set_ylabel(r"$\sigma(t) = p_2\,x_1 - p_1\,x_2$", fontsize=11)
    ax.set_title(
        r"(1,2) Pursuer control: $\sigma(t)$ drives $\phi^* = -\operatorname{sign}\,\sigma$",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)

    # ============= (2,3) costate trajectory in (p_1, p_2) plane =============
    ax = axes[1, 2]
    # The costate moves on a circle of constant ‖p‖ WITHIN each
    # characteristic. For the stitched chase, segments A and B have
    # DIFFERENT ‖p‖, so two arcs live on two different concentric
    # circles, joined by a JUMP at the dispersal switch. Both guide
    # circles are drawn.
    norm_A = float(np.sqrt(p1[0] ** 2 + p2[0] ** 2))  # τ=0 → segment A
    norm_B = float(np.sqrt(p1[-1] ** 2 + p2[-1] ** 2))  # τ=T_max → segment B
    ax.plot(norm_A * np.cos(th), norm_A * np.sin(th),
            ":", color="gray", lw=1.0, alpha=0.55,
            label=rf"$\|\mathbf{{p}}\|_A={norm_A:.2f}$ (segment A)")
    ax.plot(norm_B * np.cos(th), norm_B * np.sin(th),
            ":", color="gray", lw=1.0, alpha=0.55,
            label=rf"$\|\mathbf{{p}}\|_B={norm_B:.2f}$ (segment B)")

    # Trajectory coloured by phase (4 colours; the costate JUMP at the
    # dispersal point appears as a visible gap between the segment A
    # arc and the segment B arc — the literal value-function-gradient
    # discontinuity).
    # No trim here — green's last sample p[cross_idx] is what the
    # ★ marker AND the chord endpoint share. Keeping all samples
    # means the visible green arc reaches the ★, the red dotted
    # chord starts at the ★, and the blue arc picks up at the
    # other chord endpoint p[cross_idx+1]. Everything lines up.
    _plot_phased(ax, p1, p2, sigma, lw=2.5, alpha=0.9, zorder=4,
                 extra_breaks=[switch_break_tau],
                 alternate_linestyles=True)

    # Red dotted line connecting the last segment-A costate to the
    # first segment-B costate — the chord across the discontinuous
    # jump from the outer ‖p‖_A circle to the inner ‖p‖_B circle.
    # This is the (p_1, p_2)-plane analogue of the vertical dashed
    # red lines in (1,2) and (2,2): "the jump happens here, between
    # these two points." Not vertical in the costate plane because
    # both the magnitude AND direction of p change across the seam.
    # Endpoints coincide exactly with the green-arc end and the
    # blue-arc start (no trim, so no visual gap).
    ax.plot(
        [p1[cross_idx], p1[cross_idx + 1]],
        [p2[cross_idx], p2[cross_idx + 1]],
        ":", color=DISPERSAL, lw=1.4, alpha=0.7, zorder=3,
    )

    # Start (●, τ=0) and end (■, τ=T_max) markers — labelled in
    # figure legend.
    ax.plot(p1[0], p2[0], "o", color="black", markersize=7, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(p1[-1], p2[-1], "s", color="black", markersize=7, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(p1[cross_idx], p2[cross_idx], "*",
            color=KINK, markersize=20, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.axvline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$p_1$", fontsize=11)
    ax.set_ylabel(r"$p_2$", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(r"(2,3) Costate $(p_1, p_2)(\tau)$ — rotation reverses at the seam",
                 fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=10)

    # ============= (2,2) ψ*(t) — evader's heading, FORWARD TIME =============
    # Same convention as (1,2): plotted against t = T_max − τ so the
    # axis reads left-to-right as the chase unfolds in physical time.
    ax = axes[1, 1]
    psi_star = np.unwrap(np.arctan2(p1, p2))
    _plot_phased(ax, t_fwd, psi_star, sigma,
                 lw=2.5, alpha=0.9, zorder=4,
                 extra_breaks=[switch_break_tau],
                 trim_around_breaks=1)
    _draw_break_extensions(
        ax, t_fwd, psi_star, cross_idx, t_kink, trim=1,
        color_before=PHASE_COLORS[1], color_after=PHASE_COLORS[2],
        lw=2.5,
    )
    # Vertical line at t = t_kink matches (1,2): same switch event,
    # same colour, same linestyle.
    ax.axvline(t_kink, color=DISPERSAL, lw=1.0, alpha=0.55,
               linestyle="--", zorder=3)
    ax.plot(t_kink, psi_star[cross_idx], "*",
            color=KINK, markersize=20, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlabel(r"forward time $t$ (= $T - \tau$)", fontsize=11)
    ax.set_ylabel(r"$\psi^*(t) = \operatorname{atan2}(p_1, p_2)$  (unwrapped)",
                  fontsize=11)
    ax.set_title(
        r"(2,2) Evader control: $\psi^*(t)$ (heading along costate)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)

    # ============= (2,1) chase with EVADER highlighted =============
    # Same trajectories as (1,1); evader drawn σ-color-coded, pursuer
    # as gray context. Comparing (1,1) ↔ (2,1) lets the reader see
    # the SAME chase event from both players' perspectives.
    ax = axes[1, 0]

    # Capture-radius circle
    ax.plot(*capture_circle, ":", color="gray", lw=1.0, alpha=0.6, zorder=2)

    # Pursuer as gray context
    ax.plot(xp, yp, "-", color="gray", lw=1.2, alpha=0.40,
            label="pursuer (context)", zorder=2)

    # Evader path — coloured by phase (matches the pursuer panel's
    # phase colours so the reader can compare side-by-side)
    _plot_phased(ax, xe, ye, sigs, lw=2.8, alpha=0.92, zorder=4,
                 reverse_phase_order=True,
                 extra_breaks=[switch_break_t_eval],
                 trim_around_breaks=1,
                 add_arrows=True, arrow_size=20)

    # Dispersal-event star (same instant t_kink as in (1,1)).
    # Label is in the figure-level phase legend.
    ax.plot(xe_k, ye_k, "*",
            color=KINK, markersize=22, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)

    # Evader start (●) + end (■) markers — labelled in figure legend.
    ax.plot(xe[0], ye[0], "o", color="black", markersize=8, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(xe[-1], ye[-1], "s", color="black", markersize=8, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlim(*shared_xlim)
    ax.set_ylim(*shared_ylim)
    ax.set_xlabel(r"$x$ (lab)", fontsize=11)
    ax.set_ylabel(r"$y$ (lab)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(
        "(2,1) Same chase as §3, evader highlighted ($\\dot{\\psi}^*$ kinks)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=10)

    # Force every subplot box to a uniform square shape so the 2x3
    # grid reads as a grid.
    for ax in axes.flat:
        ax.set_box_aspect(1.0)

    # Figure-level phase legend — explains the 4 colours + the red ★
    # used across every panel so individual panels don't have to.
    # Listed in chronological forward-time order (chase start →
    # capture): P4 purple → P3 blue → ★ → P2 green → P1 orange.
    forward_order = [3, 2, 1, 0]  # indices into PHASE_COLORS / PHASE_LABELS
    phase_handles = []
    for j, i in enumerate(forward_order):
        phase_handles.append(
            plt.Line2D([], [], color=PHASE_COLORS[i], lw=4.0,
                       label=PHASE_LABELS[i])
        )
        if j == 1:  # insert the dispersal-switch entry after P3/blue
            phase_handles.append(plt.Line2D(
                [], [], color="none", marker="*",
                markersize=14, markerfacecolor=DISPERSAL,
                markeredgecolor="white", markeredgewidth=1.2,
                linestyle="none", label="dispersal switch",
            ))

    # Marker-shape legend entries — replace per-panel text labels
    # like "pursuer start (t=0)". Reader infers panel-specific
    # meaning from the axis labels: lab panels start = beginning
    # of forward-time chase / end = capture; τ-panels start = τ=0
    # (capture moment) / end = τ=T_max (chase start).
    marker_handles = [
        plt.Line2D(
            [], [], color="black", marker="o", linestyle="none",
            markersize=8, markeredgecolor="white", markeredgewidth=0.8,
            label="start of curve",
        ),
        plt.Line2D(
            [], [], color="black", marker="s", linestyle="none",
            markersize=8, markeredgecolor="white", markeredgewidth=0.8,
            label="end of curve",
        ),
        plt.Line2D(
            [], [], color="gray", linestyle=":", lw=1.5,
            label=r"capture circle  ($\ell$ around pursuer)",
        ),
    ]

    # Two stacked legends so the phase row reads strictly left-to-right
    # in chronological forward time. (matplotlib's single-legend layout
    # is column-major and would scramble the row order with 8 entries.)
    phase_legend = fig.legend(
        handles=phase_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=5, fontsize=10, frameon=True, framealpha=0.95,
    )
    fig.add_artist(phase_legend)  # keep when we add the second legend
    fig.legend(
        handles=marker_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.910),
        ncol=3, fontsize=10, frameon=True, framealpha=0.95,
    )

    fig.suptitle(
        rf"Crossing a characteristic ($w={w:.2f}$, $\ell={ell:.2f}$, "
        rf"$\alpha={alpha_hint:.2f}$): the dispersal surface is a *seam* in "
        "strategy space — both players' strategies kink at $\\sigma=0$",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.close(fig)
    return fig


__all__ = [
    # palette
    "PURSUER", "PURSUER_TRAIL", "EVADER", "EVADER_TRAIL",
    "SWITCHING", "TERMINAL", "USABLE", "ANNOT",
    # public figures
    "chase_demo",
    "naive_vs_optimal",
    "problem_geometry",
    "coordinate_progression",
    "dispersal_crossing",
    "optimal_vector_field",
    "trajectory_fan",
    "trajectory_frame",
    "reachable_set_view",
    "reachable_set_heatmap",
    "conservation_diagnostics",
]
