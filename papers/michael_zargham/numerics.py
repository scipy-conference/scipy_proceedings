"""Numerical machinery for backward characteristic shooting on the HC game.

This module is the black-box phase of Buchberger's recursion: the symbolic
derivations in `derivations.py` have done the algebra; here we turn the
4D characteristic ODE into something `scipy.integrate.solve_ivp` can run,
and we provide the terminal-circle parametrization that seeds the backward
shoot.

Mathematics credited to:

* Method of characteristics for HJI on the HC value function — Isaacs (1965)
  §6.4, Bardi/Falcone/Soravia (1999) §3.
* Terminal-condition transversality (p parallel to the outward normal at the
  capture circle, with H = 0) — standard PMP applied at a free-terminal-time
  game with terminal cost zero. The closed-form lambda below is the unique
  multiplier that makes H = 0 on the terminal circle.
* Usable-part criterion sin(alpha) > w — Merz (1971) §4.

Notation:

  alpha — angle parameter on the terminal circle, measured so that the
          parametrization is (x_1, x_2) = (ell cos(alpha), ell sin(alpha)).
          alpha = pi/2 is the "front of the pursuer" (top of the circle);
          this matches the convention in `tests/test_numerics.py::T6`.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from derivations import (
    characteristic_system,
    p1 as _p1_sym,
    p2 as _p2_sym,
    w as _w_sym,
    x1 as _x1_sym,
    x2 as _x2_sym,
)


# ---------------------------------------------------------------------------
# The glass-box → black-box encapsulation moment
# ---------------------------------------------------------------------------


def lambdify_rhs():
    """Return a numpy-callable RHS of the 4D characteristic ODE.

    Signature: ``rhs(t, state, w_val) -> list[float]`` returning
    ``[dx1/dt, dx2/dt, dp1/dt, dp2/dt]``. The system is autonomous so ``t``
    is ignored, kept for ``solve_ivp`` compatibility.

    Backward integration (the actual physics: time-to-go decreases as the
    pursuer closes in) is achieved by negating the RHS at the call site, as
    in ``solve_ivp(lambda t, y: [-v for v in rhs(t, y, w)], ...)``.

    This is the encapsulation step: a symbolic expression for the optimal
    closed-loop dynamics becomes an opaque numeric oracle. We make it
    inspectable by *constructing* it from `derivations.characteristic_system`
    in one line — that's the point.
    """
    x1d, x2d, p1d, p2d = characteristic_system()
    args = (_x1_sym, _x2_sym, _p1_sym, _p2_sym, _w_sym)

    f_x1 = sp.lambdify(args, x1d, modules="numpy")
    f_x2 = sp.lambdify(args, x2d, modules="numpy")
    f_p1 = sp.lambdify(args, p1d, modules="numpy")
    f_p2 = sp.lambdify(args, p2d, modules="numpy")

    def rhs(t, state, w_val):
        x1v, x2v, p1v, p2v = state
        return [
            float(f_x1(x1v, x2v, p1v, p2v, w_val)),
            float(f_x2(x1v, x2v, p1v, p2v, w_val)),
            float(f_p1(x1v, x2v, p1v, p2v, w_val)),
            float(f_p2(x1v, x2v, p1v, p2v, w_val)),
        ]

    return rhs


# ---------------------------------------------------------------------------
# Terminal-circle parametrization
# ---------------------------------------------------------------------------


def usable_part_lambda(alpha: float, w: float, ell: float) -> float:
    """Transversality multiplier lambda at terminal alpha.

    Derived by enforcing H = 0 on the capture circle with phi* = 0 (the
    boundary of the dispersal locus, where sigma = 0 by construction) and
    psi* = atan2(p1, p2). The unique lambda is

        lambda(alpha, w) = 1 / (sin(alpha) - w)

    which is positive iff sin(alpha) > w — exactly Merz's usable-part
    criterion. Outside the usable part (lambda <= 0) the terminal point is
    not reachable as an optimal-play capture.

    The ``ell`` argument is accepted for API symmetry with
    `terminal_conditions` but does not enter the formula in normalised
    coordinates.
    """
    _ = ell  # unused in normalised units, kept for symmetry
    return 1.0 / (np.sin(alpha) - w)


def terminal_conditions(alphas: np.ndarray, w: float, ell: float) -> np.ndarray:
    """Terminal-circle states (x_1, x_2, p_1, p_2) for an array of alphas.

    Parametrization (matches `tests/test_numerics.py::T4` and ::T6):

        (x_1, x_2) = (ell cos(alpha), ell sin(alpha))

    The costate at the terminal points outward and is normalised by the
    transversality lambda so the Hamiltonian vanishes:

        (p_1, p_2) = lambda(alpha, w) * (cos(alpha), sin(alpha))

    Returns an array of shape ``(len(alphas), 4)``.
    """
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    out = np.empty((alphas.size, 4), dtype=float)
    for i, alpha in enumerate(alphas):
        lam = usable_part_lambda(float(alpha), w, ell)
        out[i, 0] = ell * np.cos(alpha)
        out[i, 1] = ell * np.sin(alpha)
        out[i, 2] = lam * np.cos(alpha)
        out[i, 3] = lam * np.sin(alpha)
    return out


# ---------------------------------------------------------------------------
# Reachable-set computation (level curves of the value function)
# ---------------------------------------------------------------------------


def reachable_set_curve(
    T: float,
    w: float,
    ell: float,
    n_alphas: int = 60,
    alpha_margin: float = 0.02,
) -> np.ndarray:
    """Backward-reachable boundary at time T: ``V(x) = T``.

    Shoots backward characteristics from a dense sampling of the usable
    arc of the terminal circle. Returns an array of shape ``(n_alphas, 2)``
    with the ``(x_1, x_2)`` endpoints; consumers can call
    ``matplotlib.pyplot.plot(curve[:, 0], curve[:, 1])`` to draw the
    isochrone V = T.

    ``alpha_margin`` keeps the shoot away from the singularity sin(alpha) = w
    where lambda blows up.
    """
    from scipy.integrate import solve_ivp  # local import; heavy

    a_lo = np.arcsin(w) + alpha_margin
    a_hi = np.pi - np.arcsin(w) - alpha_margin
    alphas = np.linspace(a_lo, a_hi, n_alphas)
    tc = terminal_conditions(alphas, w, ell)

    rhs = lambdify_rhs()
    curve = np.empty((n_alphas, 2), dtype=float)
    for i, y0 in enumerate(tc):
        sol = solve_ivp(
            lambda t, y: [-v for v in rhs(t, y, w)],
            [0.0, T], y0,
            method="RK45", max_step=0.02, rtol=1e-9, atol=1e-11,
        )
        curve[i, 0] = float(sol.y[0, -1])
        curve[i, 1] = float(sol.y[1, -1])
    return curve


# ---------------------------------------------------------------------------
# Trajectory ensembles + value-function field
# (lifted from hc-marimo: backward_trajectories L1640-1679, reachable_set
#  L2376-2441, trajectory_animation_plot L1996-2023)
# ---------------------------------------------------------------------------


def compute_optimal_trajectories(
    n_traj: int,
    T_horizon: float,
    w: float,
    ell: float,
    *,
    eps: float = 1e-3,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.05,
):
    """Integrate ``n_traj`` optimal backward characteristics from the usable arc.

    Returns a list of ``scipy.integrate.OdeResult`` with ``dense_output=True``
    so callers (the embodied-view τ-scrubber, the trajectory-fan figure, the
    conservation-diagnostics figure) can evaluate each trajectory at arbitrary
    backward-time samples without re-integrating.

    Mirrors hc-marimo's ``backward_trajectories`` cell with the (w, ell)
    sliders folded into explicit arguments.
    """
    from scipy.integrate import solve_ivp  # local import; heavy

    alpha_min = np.arcsin(min(w, 0.999))
    alpha_max = np.pi - alpha_min
    alphas = np.linspace(alpha_min + eps, alpha_max - eps, n_traj)
    terminal = terminal_conditions(alphas, w, ell)

    rhs = lambdify_rhs()
    trajectories = []
    for i in range(n_traj):
        sol = solve_ivp(
            lambda t, y, w_val=w: [-v for v in rhs(t, y, w_val)],
            [0.0, T_horizon],
            terminal[i],
            method="RK45",
            max_step=max_step,
            dense_output=True,
            rtol=rtol,
            atol=atol,
        )
        trajectories.append(sol)
    return trajectories


def value_function_field(
    w: float,
    ell: float,
    *,
    T_max: float = 12.0,
    n_traj: int = 80,
    n_time_samples: int = 200,
    time_horizons: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0),
):
    """One integration pass → isochrone curves + scatter field for V(x).

    Mirrors hc-marimo's ``reachable_set`` cell (L2376-2441): integrates a
    dense bundle of backward characteristics once and reuses the dense output
    to extract both:

      * ``isochrones``: dict mapping ``T -> ndarray of (x_1, x_2) endpoints``.
        Each row is one trajectory's position at backward time ``T``.
      * ``scatter``: dict ``{"x1": ndarray, "x2": ndarray, "tau": ndarray}``
        with parallel arrays sampling every trajectory at
        ``n_time_samples`` points. Suitable for a single ``plt.scatter`` call
        coloured by ``tau`` — the embodied-view payoff figure.

    Doing this with a single integration pass (vs. one integration per
    isochrone) is roughly N× cheaper, where N = len(time_horizons).
    """
    trajs = compute_optimal_trajectories(
        n_traj=n_traj, T_horizon=T_max, w=w, ell=ell,
        rtol=1e-8, atol=1e-10, max_step=0.1,  # match hc-marimo's tolerances
    )

    isochrones: dict[float, np.ndarray] = {}
    for T in time_horizons:
        endpoints = []
        for sol in trajs:
            if sol.success and sol.t[-1] >= T:
                state = sol.sol(T)
                endpoints.append([float(state[0]), float(state[1])])
        isochrones[float(T)] = (
            np.array(endpoints) if endpoints else np.empty((0, 2))
        )

    x1_chunks, x2_chunks, tau_chunks = [], [], []
    for sol in trajs:
        if not sol.success:
            continue
        taus = np.linspace(0.0, float(sol.t[-1]), n_time_samples)
        states = sol.sol(taus)
        x1_chunks.append(states[0])
        x2_chunks.append(states[1])
        tau_chunks.append(taus)

    scatter = {
        "x1": np.concatenate(x1_chunks) if x1_chunks else np.array([]),
        "x2": np.concatenate(x2_chunks) if x2_chunks else np.array([]),
        "tau": np.concatenate(tau_chunks) if tau_chunks else np.array([]),
    }
    return isochrones, scatter


# Bump if the value_function_grid output format / semantics change
# in a way the cache should NOT be reused across.
_VFG_CACHE_VERSION = 2


def value_function_grid(
    w: float,
    ell: float,
    *,
    T_max_cap: float = 15.0,
    grid_extent: tuple[float, float, float, float] = (-10.0, 10.0, -8.0, 12.0),
    n_grid: int = 200,
    n_traj: int = 600,
    n_time_samples: int = 500,
    isochrone_levels: tuple[float, ...] = (2.0, 6.0, 12.0),
    cache_dir: str | None = ".cache",
):
    """Compute V*(x) on a regular 2D body-frame grid by interpolating
    capture-time samples from dense backward characteristics.

    Where a grid cell falls outside the interpolation support (no
    optimal characteristic reaches it within ``T_max_cap``) the cell
    is flagged unreachable. The cap lets the visualisation distinguish
    "very-long capture" from "no capture in finite time" cleanly.

    Process:
      1. Run ``compute_optimal_trajectories(n_traj, T_horizon=T_max_cap)``
         backward from the terminal usable arc.
      2. Sample each trajectory at ``n_time_samples`` τ points and
         concatenate into a (x_1, x_2, τ) scatter.
      3. Build a regular ``n_grid × n_grid`` mesh over ``grid_extent``.
      4. Interpolate τ at each mesh point via
         ``scipy.interpolate.griddata(method="linear")``.
      5. Mark cells where interpolation returned NaN OR where the
         interpolated τ exceeds ``T_max_cap`` as unreachable.

    Returns
    -------
    x1_grid, x2_grid : 2D arrays
        meshgrid coordinates of the grid.
    V_grid : 2D array
        V*(x) at each grid cell, clipped to ``T_max_cap``. Cells
        flagged unreachable retain the cap value (use
        ``reachable_mask`` to distinguish).
    reachable_mask : 2D bool array
        True where interpolation succeeded and τ ≤ ``T_max_cap``.
    isochrones : dict[float, ndarray]
        ``{T: (N, 2) ndarray of (x_1, x_2)}`` body-frame endpoints of
        every backward characteristic at time ``T``. Empty for any
        T > T_max_cap. Provides smooth analytical level-set curves
        for overlay on the heat map (avoids the chattery
        ``ax.contour`` artifacts of interpolating from V_grid).

    Notes
    -----
    Computationally heavier than ``value_function_field``: defaults
    integrate 600 backward characteristics out to τ=15 (vs. 80
    out to τ=12), then interpolate onto a 200×200 grid. Expect
    15-30 seconds wall time for default parameters.

    Caching: when ``cache_dir`` is set (default ``".cache"`` in the
    paper folder), results are persisted to a ``.npz`` keyed by a
    SHA-256 hash of all parameters. Subsequent calls with identical
    parameters load from disk instead of recomputing — a notebook
    re-execute drops from ~30s → <1s. Pass ``cache_dir=None`` to
    skip caching (e.g. for tests). Bump ``_VFG_CACHE_VERSION`` at
    module top to invalidate all caches if the output semantics
    change.
    """
    import hashlib  # noqa: PLC0415
    import json as _json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from scipy.interpolate import griddata  # noqa: PLC0415

    # ---- Cache lookup ----
    cache_file = None
    if cache_dir is not None:
        params = {
            "v": _VFG_CACHE_VERSION,
            "w": float(w), "ell": float(ell),
            "T_max_cap": float(T_max_cap),
            "grid_extent": [float(g) for g in grid_extent],
            "n_grid": int(n_grid),
            "n_traj": int(n_traj),
            "n_time_samples": int(n_time_samples),
            "isochrone_levels": [float(T) for T in isochrone_levels],
        }
        key = hashlib.sha256(
            _json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir_path / f"value_function_grid_{key}.npz"
        if cache_file.exists():
            d = np.load(cache_file, allow_pickle=False)
            iso = {}
            for T in isochrone_levels:
                k = f"iso_{float(T):g}"
                if k in d.files:
                    iso[float(T)] = d[k]
                else:
                    iso[float(T)] = np.empty((0, 2))
            return (
                d["x1_grid"], d["x2_grid"],
                d["V_grid"], d["reachable_mask"], iso,
            )

    # ---- Compute from scratch ----
    trajs = compute_optimal_trajectories(
        n_traj=n_traj, T_horizon=T_max_cap, w=w, ell=ell,
    )

    x1_chunks, x2_chunks, tau_chunks = [], [], []
    for sol in trajs:
        if not sol.success:
            continue
        taus = np.linspace(0.0, float(sol.t[-1]), n_time_samples)
        states = sol.sol(taus)
        x1_chunks.append(states[0])
        x2_chunks.append(states[1])
        tau_chunks.append(taus)

    x1_arr = np.concatenate(x1_chunks) if x1_chunks else np.array([])
    x2_arr = np.concatenate(x2_chunks) if x2_chunks else np.array([])
    tau_arr = np.concatenate(tau_chunks) if tau_chunks else np.array([])

    # ---- Smooth isochrone curves (level sets of V*) ----
    # For each requested T, evaluate every backward characteristic at
    # τ=T (when the trajectory was integrated that far). Endpoints
    # are ordered by trajectory index (= terminal-α order along the
    # usable arc), so consecutive points usually trace the level set
    # smoothly. The plot layer breaks the line at any jump (singular
    # surface) longer than 3× the median segment length.
    isochrones: dict[float, np.ndarray] = {}
    for T in isochrone_levels:
        endpoints = []
        for sol in trajs:
            if sol.success and float(sol.t[-1]) >= float(T):
                state = sol.sol(float(T))
                endpoints.append([float(state[0]), float(state[1])])
        isochrones[float(T)] = (
            np.array(endpoints) if endpoints else np.empty((0, 2))
        )

    x1_lo, x1_hi, x2_lo, x2_hi = grid_extent
    x1_lin = np.linspace(x1_lo, x1_hi, n_grid)
    x2_lin = np.linspace(x2_lo, x2_hi, n_grid)
    x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)

    V_grid = griddata(
        (x1_arr, x2_arr), tau_arr,
        (x1_grid, x2_grid),
        method="linear",
    )

    reachable_mask = np.isfinite(V_grid) & (V_grid <= T_max_cap)
    V_grid_capped = np.where(reachable_mask, V_grid, T_max_cap)

    if cache_file is not None:
        save_kwargs = {
            "x1_grid": x1_grid, "x2_grid": x2_grid,
            "V_grid": V_grid_capped, "reachable_mask": reachable_mask,
        }
        for T, pts in isochrones.items():
            save_kwargs[f"iso_{T:g}"] = pts
        np.savez(cache_file, **save_kwargs)

    return x1_grid, x2_grid, V_grid_capped, reachable_mask, isochrones


def sample_trajectories_at_tau(
    trajectories,
    tau: float,
    *,
    samples_per_unit_time: int = 30,
    min_samples: int = 10,
):
    """For each trajectory, return ``(x1_path, x2_path)`` sampled up to ``tau``.

    Mirrors the slicing logic in hc-marimo's ``trajectory_animation_plot``
    cell (L1996-2023): evaluates each ``OdeResult.sol`` on a uniform
    backward-time grid from 0 to min(τ, traj end). Used by the §7 τ scrubber
    to render the build-up of optimal trajectories from the terminal circle.

    Parameters
    ----------
    trajectories
        List of ``scipy.integrate.OdeResult`` with ``dense_output=True``,
        as produced by :func:`compute_optimal_trajectories`.
    tau
        Backward-time horizon to sample up to. Clipped per trajectory to
        ``min(tau, traj.t[-1])``; ``tau <= 0`` returns the single starting
        point.
    samples_per_unit_time, min_samples
        Density controls. The number of evaluation points per trajectory is
        ``max(int(τ_clipped * samples_per_unit_time), min_samples)`` for
        ``τ_clipped > 0`` and 1 for ``τ_clipped <= 0``.
    """
    frames = []
    for sol in trajectories:
        t_max = min(float(tau), float(sol.t[-1]))
        if t_max <= 0:
            x1 = np.array([float(sol.y[0, 0])])
            x2 = np.array([float(sol.y[1, 0])])
            frames.append((x1, x2))
            continue
        n_pts = max(int(t_max * samples_per_unit_time), min_samples)
        t_eval = np.linspace(0.0, t_max, n_pts)
        states = sol.sol(t_eval)
        frames.append((states[0], states[1]))
    return frames


# ---------------------------------------------------------------------------
# Forward-time lab-frame simulator for the §3 naive-vs-optimal hook
# ---------------------------------------------------------------------------


def compute_sigma_field(
    *,
    w: float = 0.45,
    ell: float = 0.5,
    n_traj: int = 200,
    T_horizon: float = 14.0,
):
    """Build a scatter of (x_1, x_2, σ) by sampling along many optimal
    backward characteristics.

    The σ value at each scatter point is the value-function-derived
    switching function σ = p_2 x_1 − p_1 x_2, computed at (x_1, x_2)
    along the characteristic that passes through it. This scatter
    is the raw material for the optimal-pursuer feedback policy
    φ*(x_1, x_2) = −sign(σ*(x_1, x_2)) used in the §3 hook figure
    to chase heuristic evaders with a saddle-point-correct pursuer.

    Returns a dict with parallel ndarrays ``x1``, ``x2``, ``sigma``.
    Used together with :func:`scipy.interpolate.LinearNDInterpolator`
    or :func:`scipy.interpolate.griddata` to query σ at arbitrary
    body-frame coordinates.
    """
    trajs = compute_optimal_trajectories(
        n_traj=n_traj, T_horizon=T_horizon, w=w, ell=ell,
    )
    x1_chunks, x2_chunks, sigma_chunks = [], [], []
    for sol in trajs:
        x1, x2, p1, p2 = sol.y
        sigma = p2 * x1 - p1 * x2
        x1_chunks.append(x1)
        x2_chunks.append(x2)
        sigma_chunks.append(sigma)
    return {
        "x1": np.concatenate(x1_chunks),
        "x2": np.concatenate(x2_chunks),
        "sigma": np.concatenate(sigma_chunks),
    }


def simulate_chase(
    pursuer_init: tuple[float, float, float],
    evader_init: tuple[float, float],
    *,
    w: float = 0.45,
    ell: float = 0.5,
    pursuer_policy: str = "pure_pursuit",
    evader_policy: str = "run_away",
    sigma_field: dict | None = None,
    T_max: float = 30.0,
    dt: float = 0.005,
) -> dict:
    """Forward-time lab-frame chase under a chosen pursuer × evader
    policy pair.

    Pursuer policies:
      "pure_pursuit": max-rate turn toward the evader's current
          position (heuristic).
      "optimal":     bang-bang φ* = −sign(σ*(x_1, x_2)) where σ* is
          read from ``sigma_field`` via linear interpolation. This
          IS the saddle-point pursuer feedback in the limit of dense
          sigma_field sampling. Requires ``sigma_field`` (built via
          :func:`compute_sigma_field`). Outside the rendered field
          (interpolation returns NaN), falls back to pure pursuit.

    Evader policies (heading in the lab frame, evader speed w):
      "run_away":      ψ = bearing from pursuer to evader (straight away)
      "perpendicular": ψ = bearing + π/2 (cross the bearing line)
      "toward":        ψ = bearing − π (straight toward pursuer)

    Dynamics (v_P = 1, R_min = 1):
      ẋ_P = cos θ_P,  ẏ_P = sin θ_P,  θ̇_P = φ (clipped to [-1, 1])
      ẋ_E = w cos ψ,  ẏ_E = w sin ψ
    Stops when ‖P − E‖ ≤ ell (capture) or t > T_max.

    Returns dict with keys: t, xp, yp, theta_p, xe, ye,
    capture_time (float or None), capture_idx (int or None),
    pursuer_policy, evader_policy, w, ell.
    """
    if pursuer_policy not in ("pure_pursuit", "optimal"):
        raise ValueError(
            f"pursuer_policy must be 'pure_pursuit' or 'optimal', got "
            f"{pursuer_policy!r}"
        )
    if evader_policy not in ("run_away", "perpendicular", "toward"):
        raise ValueError(
            f"evader_policy must be 'run_away'/'perpendicular'/'toward', "
            f"got {evader_policy!r}"
        )
    if pursuer_policy == "optimal" and sigma_field is None:
        raise ValueError(
            "pursuer_policy='optimal' requires sigma_field (call "
            "compute_sigma_field first)"
        )

    # Build the linear interpolator once if optimal
    sigma_interp = None
    if pursuer_policy == "optimal":
        from scipy.interpolate import LinearNDInterpolator  # noqa: PLC0415
        sigma_interp = LinearNDInterpolator(
            np.column_stack([sigma_field["x1"], sigma_field["x2"]]),
            sigma_field["sigma"],
        )

    xp, yp, th = pursuer_init
    xe, ye = evader_init
    n_steps = int(T_max / dt)
    xp_a = np.empty(n_steps + 1)
    yp_a = np.empty(n_steps + 1)
    th_a = np.empty(n_steps + 1)
    xe_a = np.empty(n_steps + 1)
    ye_a = np.empty(n_steps + 1)
    t_a = np.empty(n_steps + 1)
    xp_a[0], yp_a[0], th_a[0] = xp, yp, th
    xe_a[0], ye_a[0] = xe, ye
    t_a[0] = 0.0

    capture_idx = None
    last = n_steps

    for k in range(n_steps):
        # Bearing from pursuer to evader (lab frame)
        bearing = float(np.arctan2(ye - yp, xe - xp))

        # Pursuer
        if pursuer_policy == "pure_pursuit":
            heading_err = (bearing - th + np.pi) % (2.0 * np.pi) - np.pi
            phi = float(np.clip(heading_err / dt, -1.0, 1.0))
        else:  # optimal
            # Body-frame (x_1, x_2) of the evader relative to the pursuer
            dx, dy = xe - xp, ye - yp
            cos_t, sin_t = float(np.cos(th)), float(np.sin(th))
            x_1 = -sin_t * dx + cos_t * dy
            x_2 = cos_t * dx + sin_t * dy
            sigma_val = float(sigma_interp(x_1, x_2))
            if np.isnan(sigma_val):
                # Fall back to pure pursuit if outside the interpolated field
                heading_err = (bearing - th + np.pi) % (2.0 * np.pi) - np.pi
                phi = float(np.clip(heading_err / dt, -1.0, 1.0))
            else:
                # Bang-bang: ±1, clipped tight (the |φ| ≤ 1 cap is the
                # minimum-turning-radius constraint).
                phi = float(np.clip(-np.sign(sigma_val), -1.0, 1.0))

        # Evader
        if evader_policy == "run_away":
            psi = bearing
        elif evader_policy == "perpendicular":
            psi = bearing + 0.5 * np.pi
        else:  # "toward"
            psi = bearing + np.pi

        # Step
        xp = xp + np.cos(th) * dt
        yp = yp + np.sin(th) * dt
        th = th + phi * dt
        xe = xe + w * np.cos(psi) * dt
        ye = ye + w * np.sin(psi) * dt

        xp_a[k + 1] = xp
        yp_a[k + 1] = yp
        th_a[k + 1] = th
        xe_a[k + 1] = xe
        ye_a[k + 1] = ye
        t_a[k + 1] = (k + 1) * dt

        if (xe - xp) ** 2 + (ye - yp) ** 2 <= ell * ell:
            capture_idx = k + 1
            last = k + 1
            break

    s = slice(0, last + 1)
    return {
        "t": t_a[s],
        "xp": xp_a[s], "yp": yp_a[s], "theta_p": th_a[s],
        "xe": xe_a[s], "ye": ye_a[s],
        "capture_time": (None if capture_idx is None else float(t_a[capture_idx])),
        "capture_idx": capture_idx,
        "pursuer_policy": pursuer_policy,
        "evader_policy": evader_policy,
        "w": w, "ell": ell,
    }


def simulate_pure_pursuit_chase(
    pursuer_init: tuple[float, float, float],
    evader_init: tuple[float, float],
    *,
    w: float = 0.45,
    ell: float = 0.5,
    evader_policy: str = "run_away",
    T_max: float = 30.0,
    dt: float = 0.01,
) -> dict:
    """Forward-time lab-frame chase under PURE-PURSUIT pursuer.

    Used to support the §3 ``plots.naive_vs_optimal`` hook figure: at
    the same initial conditions, the same pure-pursuit pursuer captures
    a run-away evader much faster than an evader using one of the
    counter-intuitive maneuvering policies. The pedagogical question
    the figure poses — *"how did anyone figure out that running across,
    or even toward, the pursuer survives longer?"* — is what the rest
    of the paper answers via the saddle-point game and the optimal
    feedback ψ* = atan2(p_1, p_2).

    Dynamics (normalized to v_P = 1, R_min = 1):

        Pursuer (pure pursuit):
            ẋ_P = cos(θ_P),  ẏ_P = sin(θ_P)
            φ   = clip((bearing − θ_P + π) mod 2π − π) / dt  ∈ [-1, 1]
            θ̇_P = φ
          (Turn at maximum rate toward the current evader position;
          the |φ| ≤ 1 cap is the minimum-turning-radius constraint.)

        Evader (one of three reference policies — `evader_policy`):
            ẋ_E = w cos(ψ),  ẏ_E = w sin(ψ)
            "run_away":      ψ = bearing            (naive — point straight away)
            "perpendicular": ψ = bearing + π/2      (cross the pursuer's path)
            "toward":        ψ = bearing − π        (point straight toward pursuer)

    Stops when ‖P − E‖ ≤ ell (capture) or t > T_max (escaped).

    Parameters
    ----------
    pursuer_init : (x_P0, y_P0, θ_P0)
    evader_init  : (x_E0, y_E0)
    w            : evader speed (in units of pursuer speed; should be < 1)
    ell          : capture radius
    evader_policy: "run_away" | "perpendicular" | "toward"
    T_max        : maximum simulation time
    dt           : integration step

    Returns
    -------
    dict with keys:
        t (1D), xp/yp/theta_p (1D, pursuer), xe/ye (1D, evader),
        capture_time (float or None), capture_idx (int or None).
    """
    xp, yp, th = pursuer_init
    xe, ye = evader_init

    if evader_policy not in ("run_away", "perpendicular", "toward"):
        raise ValueError(
            f"evader_policy must be one of "
            f"'run_away'/'perpendicular'/'toward', got {evader_policy!r}"
        )

    n_steps = int(T_max / dt)
    xp_arr = np.empty(n_steps + 1)
    yp_arr = np.empty(n_steps + 1)
    th_arr = np.empty(n_steps + 1)
    xe_arr = np.empty(n_steps + 1)
    ye_arr = np.empty(n_steps + 1)
    t_arr = np.empty(n_steps + 1)

    xp_arr[0], yp_arr[0], th_arr[0] = xp, yp, th
    xe_arr[0], ye_arr[0] = xe, ye
    t_arr[0] = 0.0

    capture_idx = None
    last = n_steps

    for k in range(n_steps):
        # Bearing from pursuer to evader (lab frame)
        bearing = float(np.arctan2(ye - yp, xe - xp))

        # Pursuer pure-pursuit: maximum-rate turn toward evader
        heading_err = (bearing - th + np.pi) % (2.0 * np.pi) - np.pi
        # |φ| ≤ 1: the minimum-turning-radius constraint
        phi = float(np.clip(heading_err / dt, -1.0, 1.0))

        # Evader policy → lab-frame heading ψ
        if evader_policy == "run_away":
            psi = bearing
        elif evader_policy == "perpendicular":
            psi = bearing + 0.5 * np.pi
        else:  # "toward"
            psi = bearing + np.pi

        # Forward-Euler step (dt small enough for visual continuity)
        xp = xp + np.cos(th) * dt
        yp = yp + np.sin(th) * dt
        th = th + phi * dt
        xe = xe + w * np.cos(psi) * dt
        ye = ye + w * np.sin(psi) * dt

        xp_arr[k + 1] = xp
        yp_arr[k + 1] = yp
        th_arr[k + 1] = th
        xe_arr[k + 1] = xe
        ye_arr[k + 1] = ye
        t_arr[k + 1] = (k + 1) * dt

        if (xe - xp) ** 2 + (ye - yp) ** 2 <= ell * ell:
            capture_idx = k + 1
            last = k + 1
            break

    out_slice = slice(0, last + 1)
    return {
        "t": t_arr[out_slice],
        "xp": xp_arr[out_slice],
        "yp": yp_arr[out_slice],
        "theta_p": th_arr[out_slice],
        "xe": xe_arr[out_slice],
        "ye": ye_arr[out_slice],
        "capture_time": (None if capture_idx is None else float(t_arr[capture_idx])),
        "capture_idx": capture_idx,
        "policy": evader_policy,
        "w": w,
        "ell": ell,
    }


__all__ = [
    "lambdify_rhs",
    "usable_part_lambda",
    "terminal_conditions",
    "reachable_set_curve",
    "compute_optimal_trajectories",
    "value_function_field",
    "value_function_grid",
    "sample_trajectories_at_tau",
    "compute_sigma_field",
    "simulate_chase",
    "simulate_pure_pursuit_chase",
]
