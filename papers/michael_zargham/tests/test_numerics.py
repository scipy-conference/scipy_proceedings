"""Tests for `numerics.py` — numerical integration invariants of the HC system.

Adapted from `hc-marimo/test_phase2.py` (6 tests selected from the 16-test
corpus). The selected subset covers the invariants that the paper's narrative
relies on:

  T1: lambdified RHS agrees with a hand-coded reference at random points
  T2: H* ≈ 0 conserved along integrated backward trajectories
  T3: ||p||^2 conserved along integrated trajectories
  T4: terminal states lie on the capture circle
  T6: stationary evader (w=0) collapses to straight-line capture
  T7: usable part of the terminal circle = arc where sin(alpha) > w

These tests fail with ImportError until Phase C implements numerics.py.
That's the TDD intent.
"""
import numpy as np
from scipy.integrate import solve_ivp


# ---------- Reference (hand-coded) RHS for the consistency check ----------

def _rhs_forward_reference(t, state, w_val):
    """Hand-coded RHS of the 4D characteristic ODE — the ground truth
    against which numerics.rhs_forward() is checked in T1."""
    x1, x2, p1, p2 = state
    norm_p = np.sqrt(p1 ** 2 + p2 ** 2)
    if norm_p < 1e-15:
        return [0.0, 0.0, 0.0, 0.0]
    sigma = p2 * x1 - p1 * x2
    phi_star = -np.sign(sigma)
    x1d = -phi_star * x2 + w_val * p1 / norm_p
    x2d = phi_star * x1 + w_val * p2 / norm_p - 1.0
    p1d = -phi_star * p2
    p2d = phi_star * p1
    return [x1d, x2d, p1d, p2d]


# ---------- T1: lambdified RHS matches the reference ----------

def test_T1_lambdify_consistency():
    """numerics.lambdify_rhs() output must match the hand-coded reference."""
    from numerics import lambdify_rhs  # noqa: PLC0415

    rng = np.random.default_rng(42)
    rhs = lambdify_rhs()

    for _ in range(50):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)
        norm_p = rng.uniform(0.1, 3.0)
        p1 = norm_p * np.cos(angle)
        p2 = norm_p * np.sin(angle)
        w_val = rng.uniform(0.05, 0.5)
        state = [x1, x2, p1, p2]

        ref = _rhs_forward_reference(0, state, w_val)
        got = rhs(0, state, w_val)

        for j in range(4):
            assert abs(ref[j] - got[j]) < 1e-10, (
                f"T1 FAIL: component {j} mismatch at state={state}, w={w_val}: "
                f"ref={ref[j]}, got={got[j]}"
            )


# ---------- T2: Hamiltonian conservation along integrated trajectories ----------

def test_T2_hamiltonian_conservation():
    """H* ≈ 0 conserved along backward integration from the terminal circle."""
    from numerics import lambdify_rhs, terminal_conditions  # noqa: PLC0415

    w_val = 0.25
    ell_val = 0.5
    alpha = np.pi / 2  # sin(pi/2) = 1 > w
    y0 = terminal_conditions(np.array([alpha]), w_val, ell_val)[0]

    rhs = lambdify_rhs()
    sol = solve_ivp(
        lambda t, y: [-v for v in rhs(t, y, w_val)],
        [0, 10], y0,
        method="RK45", max_step=0.02, rtol=1e-10, atol=1e-12,
    )
    assert sol.success, f"T2 FAIL: integration failed: {sol.message}"

    x1, x2, p1, p2 = sol.y
    norm_p = np.sqrt(p1 ** 2 + p2 ** 2)
    sigma = p2 * x1 - p1 * x2
    H_star = -np.abs(sigma) + w_val * norm_p - p2 + 1.0

    max_drift = float(np.max(np.abs(H_star)))
    assert max_drift < 1e-6, f"T2 FAIL: max |H*| = {max_drift}"


# ---------- T3: ||p||^2 conservation ----------

def test_T3_costate_norm_conservation():
    """||p||^2 must be conserved along trajectories (symbolic R5, numerical here)."""
    from numerics import lambdify_rhs, terminal_conditions  # noqa: PLC0415

    w_val = 0.3
    ell_val = 0.4
    alpha = 1.2  # sin(1.2) ≈ 0.93 > w
    y0 = terminal_conditions(np.array([alpha]), w_val, ell_val)[0]

    rhs = lambdify_rhs()
    sol = solve_ivp(
        lambda t, y: [-v for v in rhs(t, y, w_val)],
        [0, 10], y0,
        method="RK45", max_step=0.02, rtol=1e-10, atol=1e-12,
    )
    assert sol.success, "T3 FAIL: integration failed"

    p_norm_sq = sol.y[2] ** 2 + sol.y[3] ** 2
    drift = float(np.max(np.abs(p_norm_sq - p_norm_sq[0])))
    assert drift < 1e-8, f"T3 FAIL: ||p||^2 drift = {drift}"


# ---------- T4: terminal-circle membership ----------

def test_T4_capture_condition():
    """numerics.terminal_conditions() must produce states on the capture circle."""
    from numerics import terminal_conditions  # noqa: PLC0415

    w_val = 0.25
    ell_val = 0.5
    alphas = np.linspace(
        np.arcsin(w_val) + 0.01,
        np.pi - np.arcsin(w_val) - 0.01,
        10,
    )
    tc = terminal_conditions(alphas, w_val, ell_val)

    for i, y0 in enumerate(tc):
        r_sq = float(y0[0] ** 2 + y0[1] ** 2)
        assert abs(r_sq - ell_val ** 2) < 1e-12, (
            f"T4 FAIL: r^2 = {r_sq}, expected {ell_val ** 2} at alpha={alphas[i]}"
        )


# ---------- T6: stationary-evader degenerate limit ----------

def test_T6_stationary_evader():
    """w=0 limit: straight-line pursuit from (0, d) captures at tau = d - ell."""
    from numerics import lambdify_rhs  # noqa: PLC0415

    w_val = 0.0
    ell_val = 0.5
    d = 5.0
    # Exact terminal state at alpha = pi/2 (top of circle), avoids trig noise
    lam = 1.0 / (ell_val * 1.0)
    y0 = [0.0, ell_val, 0.0, lam * ell_val]

    T_expected = d - ell_val
    rhs = lambdify_rhs()
    sol = solve_ivp(
        lambda t, y: [-v for v in rhs(t, y, w_val)],
        [0, T_expected + 1], y0,
        method="RK45", max_step=0.02, rtol=1e-10, atol=1e-12,
        dense_output=True,
    )
    assert sol.success, "T6 FAIL: integration failed"

    x1_end = float(sol.sol(T_expected)[0])
    x2_end = float(sol.sol(T_expected)[1])

    assert abs(x1_end) < 1e-6, f"T6 FAIL: x1 = {x1_end}, expected ~0"
    assert abs(x2_end - d) < 0.05, f"T6 FAIL: x2 = {x2_end}, expected ~{d}"


# ---------- T7: usable part of the terminal circle ----------

def test_T7_usable_part():
    """numerics.usable_part_lambda(alpha, w, ell) > 0 iff sin(alpha) > w."""
    from numerics import usable_part_lambda  # noqa: PLC0415

    for w_val in [0.1, 0.25, 0.4]:
        ell_val = 0.5
        alpha_inside = (np.arcsin(w_val) + np.pi / 2) / 2 + np.pi / 4
        lam_inside = usable_part_lambda(alpha_inside, w_val, ell_val)
        assert lam_inside > 0, (
            f"T7 FAIL: lambda={lam_inside} at alpha={alpha_inside}, w={w_val}"
        )

        alpha_outside = np.arcsin(w_val) / 2
        lam_outside = usable_part_lambda(alpha_outside, w_val, ell_val)
        assert lam_outside < 0, (
            f"T7 FAIL: lambda={lam_outside} at alpha={alpha_outside}, w={w_val}"
        )


# ---------- T8: compute_optimal_trajectories returns dense-output OdeResults ----------

def test_T8_trajectory_dense_output():
    """Each returned OdeResult has dense_output enabled and reaches T_horizon."""
    from numerics import compute_optimal_trajectories  # noqa: PLC0415

    w_val, ell_val, T_horizon = 0.3, 0.4, 6.0
    trajs = compute_optimal_trajectories(
        n_traj=8, T_horizon=T_horizon, w=w_val, ell=ell_val,
    )
    assert len(trajs) == 8, f"T8 FAIL: expected 8 trajectories, got {len(trajs)}"
    for i, sol in enumerate(trajs):
        assert sol.success, f"T8 FAIL: integration {i} failed: {sol.message}"
        assert sol.sol is not None, f"T8 FAIL: trajectory {i} missing dense_output"
        # Final time should be at the horizon (within max_step tolerance)
        assert sol.t[-1] >= T_horizon - 0.5, (
            f"T8 FAIL: trajectory {i} stopped at {sol.t[-1]}, expected ~{T_horizon}"
        )


# ---------- T9: value_function_field isochrone alignment ----------

def test_T9_value_function_field_isochrone_alignment():
    """Isochrone points are reachable within the labeled time T (with tolerance)."""
    from numerics import value_function_field  # noqa: PLC0415

    w_val, ell_val = 0.25, 0.5
    isochrones, scatter = value_function_field(
        w=w_val, ell=ell_val, T_max=8.0, n_traj=40,
    )
    # The field should produce at least the standard time horizons
    for T in [1.0, 2.0, 4.0]:
        assert T in isochrones, f"T9 FAIL: T={T} missing from isochrones dict"
        pts = isochrones[T]
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"T9 FAIL: T={T} isochrone shape {pts.shape} not (n, 2)"
        )
        assert len(pts) > 0, f"T9 FAIL: T={T} produced empty isochrone"

    # Scatter field invariants
    for key in ("x1", "x2", "tau"):
        assert key in scatter, f"T9 FAIL: scatter missing key {key!r}"
        assert isinstance(scatter[key], np.ndarray), (
            f"T9 FAIL: scatter[{key!r}] is {type(scatter[key]).__name__}, want ndarray"
        )
    n_pts = scatter["x1"].size
    assert scatter["x2"].size == n_pts and scatter["tau"].size == n_pts, (
        "T9 FAIL: scatter arrays have mismatched lengths"
    )
    assert (scatter["tau"] >= 0).all(), "T9 FAIL: scatter contains negative tau"


# ---------- T10: sample_trajectories_at_tau truncation behaviour ----------

def test_T10_sample_trajectories_truncation():
    """sample_trajectories_at_tau returns positions up to tau, never past trajectory end."""
    from numerics import (  # noqa: PLC0415
        compute_optimal_trajectories,
        sample_trajectories_at_tau,
    )

    w_val, ell_val, T_horizon = 0.3, 0.4, 4.0
    trajs = compute_optimal_trajectories(
        n_traj=5, T_horizon=T_horizon, w=w_val, ell=ell_val,
    )

    # tau = 0 should give starting positions on (or near) the terminal circle
    frames_zero = sample_trajectories_at_tau(trajs, tau=0.0)
    assert len(frames_zero) == 5
    for x1_path, x2_path in frames_zero:
        # At tau=0 the path is just the starting point (1 sample)
        assert len(x1_path) >= 1
        r0 = float(np.hypot(x1_path[0], x2_path[0]))
        assert abs(r0 - ell_val) < 1e-3, (
            f"T10 FAIL: tau=0 endpoint not on terminal circle: r={r0}"
        )

    # tau in the middle should give intermediate paths
    frames_mid = sample_trajectories_at_tau(trajs, tau=2.0)
    for x1_path, x2_path in frames_mid:
        assert len(x1_path) == len(x2_path)
        assert len(x1_path) > 1, "T10 FAIL: mid-tau sample produced single point"

    # tau beyond T_horizon should clip to T_horizon, not crash
    frames_over = sample_trajectories_at_tau(trajs, tau=T_horizon + 5.0)
    assert len(frames_over) == 5


# ---------- T11/T12: simulate_pure_pursuit_chase capture detection ----------


def test_T11_pure_pursuit_capture_detected():
    """Naive evader (run away) at the §3 hook IC: pursuer captures
    in finite time, capture detection fires when ||P-E|| <= ell."""
    from numerics import simulate_pure_pursuit_chase  # noqa: PLC0415

    r = simulate_pure_pursuit_chase(
        (0.0, 0.0, 0.0), (0.5, 1.5),
        w=0.45, ell=0.5, evader_policy="run_away",
        T_max=20.0, dt=0.005,
    )
    assert r["capture_time"] is not None, "T11 FAIL: naive evader should be captured"
    assert 1.0 < r["capture_time"] < 10.0, (
        f"T11 FAIL: naive capture time {r['capture_time']} out of plausible range"
    )
    # And the final separation is at most the capture radius (within one step)
    sep = float(np.hypot(r["xe"][-1] - r["xp"][-1], r["ye"][-1] - r["yp"][-1]))
    assert sep <= 0.5 + 1e-2, f"T11 FAIL: final separation {sep} > ell"


def test_T12_pure_pursuit_perpendicular_outlasts_naive():
    """The §3 pedagogical hook: at the IC (P=(0,0,0), E=(0.5, 1.5)),
    the perpendicular evader survives longer than the naive runaway
    (in fact, escapes for the rendered horizon). This codifies the
    visual claim of `plots.naive_vs_optimal` so it can't silently
    regress."""
    from numerics import simulate_pure_pursuit_chase  # noqa: PLC0415

    naive = simulate_pure_pursuit_chase(
        (0.0, 0.0, 0.0), (0.5, 1.5),
        w=0.45, ell=0.5, evader_policy="run_away",
        T_max=20.0, dt=0.005,
    )
    perp = simulate_pure_pursuit_chase(
        (0.0, 0.0, 0.0), (0.5, 1.5),
        w=0.45, ell=0.5, evader_policy="perpendicular",
        T_max=15.0, dt=0.005,
    )
    naive_t = naive["capture_time"]
    perp_t = perp["capture_time"]
    assert naive_t is not None, "T12 setup: naive must be captured"
    # Either perpendicular escapes (None) OR it lasts longer than naive
    if perp_t is None:
        assert perp["t"][-1] > naive_t, (
            "T12 FAIL: perpendicular sim ran shorter than naive capture time"
        )
    else:
        assert perp_t > naive_t, (
            f"T12 FAIL: perpendicular caught at {perp_t} <= naive at {naive_t}"
        )


# ---------- T13: value_function_grid smoke + cap behaviour ----------

def test_T13_value_function_grid_basic(tmp_path):
    """value_function_grid returns the expected 5-tuple shape, cap is
    honoured, reachable_mask is consistent with V_grid <= T_max_cap, and
    isochrones for the requested levels are nonempty. Use a tiny grid +
    sparse fan to keep the test fast (<3 s)."""
    from numerics import value_function_grid  # noqa: PLC0415

    x1, x2, V, mask, iso = value_function_grid(
        w=0.45, ell=0.5,
        T_max_cap=10.0,
        grid_extent=(-3.0, 3.0, -2.0, 4.0),
        n_grid=30, n_traj=60, n_time_samples=60,
        isochrone_levels=(2.0, 6.0),
        cache_dir=str(tmp_path),  # isolated cache per-test
    )

    # Shape
    assert x1.shape == x2.shape == V.shape == mask.shape == (30, 30)
    # Cap behaviour: every V value is at most T_max_cap
    assert float(V.max()) <= 10.0 + 1e-9
    # reachable_mask is consistent with cap
    assert (V[mask] <= 10.0).all()
    # Isochrones: requested levels present, both nonempty
    assert set(iso.keys()) == {2.0, 6.0}
    assert iso[2.0].shape[1] == 2  # (N, 2) endpoints
    assert iso[6.0].shape[1] == 2


def test_T14_value_function_grid_cache_roundtrip(tmp_path):
    """Two calls with identical params produce identical arrays and the
    second call is reading from disk (verified by checking the cache file
    actually exists between calls)."""
    from numerics import value_function_grid  # noqa: PLC0415
    import os  # noqa: PLC0415

    kwargs = dict(
        w=0.45, ell=0.5, T_max_cap=8.0,
        grid_extent=(-2.0, 2.0, -1.0, 3.0),
        n_grid=20, n_traj=40, n_time_samples=40,
        isochrone_levels=(2.0,),
        cache_dir=str(tmp_path),
    )
    # Cold
    assert not any(p.endswith(".npz") for p in os.listdir(tmp_path))
    a = value_function_grid(**kwargs)
    # Cache file now present
    files = [p for p in os.listdir(tmp_path) if p.endswith(".npz")]
    assert len(files) == 1
    # Warm
    b = value_function_grid(**kwargs)
    # Identical arrays
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])
    # Isochrone dict equal element-wise
    assert set(a[4].keys()) == set(b[4].keys())
    for k in a[4]:
        np.testing.assert_array_equal(a[4][k], b[4][k])


# ---------- T15: barrier singular-surface invariant under sampling ----------

def test_T15_barrier_persists_across_sampling_densities(tmp_path):
    """The §8 prose claims the barrier singular surface (the sharp inner
    boundary between V*~2 s and V*~6 s in the heat map) is a real HC
    feature, not a sampling artefact. Pin this by asserting the kidney-
    shaped low-V* region (V* <= 3 s, inside the barrier) has roughly
    the same area across n_traj ∈ {100, 300, 600}.

    "Roughly the same" = within 25% relative across the three counts.
    The barrier is the OUTER boundary of this region; if the boundary
    moved with sampling density, the area would jump.

    Fast: ~10 s total for three small-grid sweeps."""
    from numerics import value_function_grid  # noqa: PLC0415

    common = dict(
        w=0.45, ell=0.5,
        T_max_cap=10.0,
        grid_extent=(-4.0, 4.0, -3.0, 6.0),
        n_grid=60, n_time_samples=120,
        isochrone_levels=(2.0,),
        cache_dir=str(tmp_path),  # isolated cache; one .npz per n_traj
    )
    THRESHOLD = 3.0  # capture-time cap defining "inside the barrier"

    areas = {}
    for n_traj in (100, 300, 600):
        _, _, V_grid, reachable_mask, _ = value_function_grid(
            **common, n_traj=n_traj,
        )
        inside_barrier = reachable_mask & (V_grid <= THRESHOLD)
        areas[n_traj] = int(inside_barrier.sum())

    # All three counts should agree to within 25%
    a_min, a_max = min(areas.values()), max(areas.values())
    rel_spread = (a_max - a_min) / max(a_min, 1)
    assert rel_spread < 0.25, (
        f"T15 FAIL: inner-V* area varies more than 25% across n_traj. "
        f"Areas {areas}; spread {rel_spread:.2%}"
    )
    # Sanity: area should be > 0 (the region exists)
    assert min(areas.values()) > 0, (
        f"T15 SANITY: inner-V* region empty at some n_traj: {areas}"
    )
