"""Smoke tests for `plots.py`.

Each public plot function must:
  - return a ``matplotlib.figure.Figure`` instance
  - not raise on its documented call signature
  - produce the expected subplot count where applicable

These are intentionally lightweight — we are not pixel-comparing rendered
figures (brittle). Visual review happens in Phase V.

The trajectory-dependent figures share a session-scoped pytest fixture so we
integrate the characteristic ODEs once per test session, not once per test.
"""
import matplotlib

matplotlib.use("Agg")  # headless backend; required for CI without a display

import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402


# Reference parameters reused across the figures (match plots.py defaults).
W_REF, ELL_REF = 0.3, 0.4


@pytest.fixture(scope="session")
def trajectories():
    """A small trajectory ensemble shared across the trajectory-dependent tests."""
    from numerics import compute_optimal_trajectories  # noqa: PLC0415
    return compute_optimal_trajectories(
        n_traj=6, T_horizon=4.0, w=W_REF, ell=ELL_REF,
    )


@pytest.fixture(scope="session")
def value_field():
    """Cached (isochrones, scatter) for the reachable-set-view test."""
    from numerics import value_function_field  # noqa: PLC0415
    return value_function_field(w=W_REF, ell=ELL_REF, T_max=6.0, n_traj=20)


# ---------- §3: chase_demo (hardcoded demo data, no inputs) ----------

def test_chase_demo_returns_figure():
    """Lifted-from-hc-marimo §3 motivation: real chase with dispersal crossing."""
    from plots import chase_demo  # noqa: PLC0415

    fig = chase_demo()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


# ---------- §3 supporting: problem_geometry (lab-frame schematic) ----------

def test_problem_geometry_returns_figure():
    """Lab-frame schematic with P, E, heading angles, turning circle."""
    from plots import problem_geometry  # noqa: PLC0415

    fig = problem_geometry()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


# ---------- §5: coordinate_progression (Lab multi-config → Body collapsed, 1×2) ----------

def test_coordinate_progression_two_panels():
    """Coordinate-progression figure must have exactly 2 panels (Lab, Body).

    R.9 redesign: three lab configurations all collapse to one body-frame
    point. Translation alone (the previous middle panel) was visually
    trivial — relative arrangement doesn't change under translation —
    and is now described in prose, not pictured.
    """
    from plots import coordinate_progression  # noqa: PLC0415

    fig = coordinate_progression()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2, (
        f"coordinate_progression produced {len(fig.axes)} axes, expected 2"
    )


# ---------- §6/§7 bridge: dispersal_crossing (2×3 = 6 panels) ----------

def test_dispersal_crossing_six_panels():
    """2×3 layout: (1,1)+(2,1) parallel chase panels + 2×2 decomposition."""
    from plots import dispersal_crossing  # noqa: PLC0415

    fig = dispersal_crossing()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 6, (
        f"dispersal_crossing produced {len(fig.axes)} axes, expected 6"
    )


def _single_point_markers(ax, shape):
    """All single-vertex Line2D markers of the given shape drawn on ``ax``."""
    out = []
    for ln in ax.lines:
        if ln.get_marker() == shape and len(ln.get_xdata()) == 1:
            out.append((float(ln.get_xdata()[0]), float(ln.get_ydata()[0])))
    return out


def _panel(fig, needle):
    for ax in fig.axes:
        if needle in ax.get_title():
            return ax
    raise AssertionError(f"no panel titled like {needle!r}")


def test_dispersal_crossing_start_end_are_forward_time():
    """Start/end markers are forward-time facts, even in backward-computed panels.

    The body-frame-state (1,3) and costate (2,3) panels integrate the
    characteristic backwards, so their arrays are τ-ordered: index 0 = τ=0 =
    capture, index -1 = τ=T_max = chase start. The shared figure legend marks
    ``●`` "start of curve" and ``■`` "end of curve" with the forward-time
    meaning the lab panels set (start = chase start, end = capture). So in these
    τ panels ``●`` must sit at array index -1 (chase start) and ``■`` at index 0
    (capture) — NOT the raw array endpoints.
    """
    import numpy as np  # noqa: PLC0415
    from plots import (  # noqa: PLC0415
        dispersal_crossing,
        _recover_chase_from_demo,
    )

    data = _recover_chase_from_demo()
    x1, x2, p1, p2 = data.sol.y  # τ-ordered: [0]=capture, [-1]=chase start
    fig = dispersal_crossing()

    for needle, coords in (
        ("Body-frame state", (x1, x2)),
        ("Costate", (p1, p2)),
    ):
        ax = _panel(fig, needle)
        start = _single_point_markers(ax, "o")
        end = _single_point_markers(ax, "s")
        assert len(start) == 1, f"{needle}: expected one ● marker, got {len(start)}"
        assert len(end) == 1, f"{needle}: expected one ■ marker, got {len(end)}"
        a, b = coords
        np.testing.assert_allclose(
            start[0], (a[-1], b[-1]), atol=1e-9,
            err_msg=f"{needle}: ● start must be the forward-time chase start (τ=T_max)",
        )
        np.testing.assert_allclose(
            end[0], (a[0], b[0]), atol=1e-9,
            err_msg=f"{needle}: ■ end must be capture (τ=0)",
        )


# ---------- §8: optimal_vector_field (quiver with switching surface) ----------

def test_optimal_vector_field_returns_figure():
    """Quiver plot of optimal state velocity over a grid."""
    from plots import optimal_vector_field  # noqa: PLC0415

    fig = optimal_vector_field(W_REF, ELL_REF, n_grid=12)  # small grid for speed
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


# ---------- §8: trajectory_fan (static family of optimal characteristics) ----------

def test_trajectory_fan_returns_figure(trajectories):
    """Static figure of all optimal backward trajectories."""
    from plots import trajectory_fan  # noqa: PLC0415

    fig = trajectory_fan(trajectories, W_REF, ELL_REF)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


# ---------- §7: trajectory_frame (single τ frame for the scrubber widget) ----------

def test_trajectory_frame_returns_figure(trajectories):
    """One frame of the τ scrubber; pure function of tau."""
    from plots import trajectory_frame  # noqa: PLC0415

    fig = trajectory_frame(trajectories, tau=2.0, w=W_REF, ell=ELL_REF)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


# ---------- §7: reachable_set_view (the embodied-view payoff figure) ----------

def test_reachable_set_view_returns_figure(value_field):
    """Scatter field coloured by capture time + overlaid isochrones."""
    from plots import reachable_set_view  # noqa: PLC0415

    isochrones, scatter = value_field
    fig = reachable_set_view(isochrones, scatter, W_REF, ELL_REF)
    assert isinstance(fig, matplotlib.figure.Figure)
    # reachable_set_view has both the main axes AND a horizontal colorbar
    assert len(fig.axes) >= 1


# ---------- §12: conservation_diagnostics (H* and ‖p‖² drift evidence) ----------

def test_conservation_diagnostics_returns_two_axes(trajectories):
    """Two-panel figure: H* vs τ on top, ‖p‖² drift on bottom."""
    from plots import conservation_diagnostics  # noqa: PLC0415

    fig = conservation_diagnostics(trajectories, W_REF)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2, (
        f"conservation_diagnostics produced {len(fig.axes)} axes, expected 2"
    )


# ---------- §3 hook: naive_vs_optimal (counter-intuitive payoff) ----------

def test_naive_vs_optimal_returns_figure():
    """Side-by-side comparison: naive evader (captured) vs perpendicular
    evader (escapes the pursuer's turning radius)."""
    from plots import naive_vs_optimal  # noqa: PLC0415

    fig = naive_vs_optimal()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2, (
        f"naive_vs_optimal produced {len(fig.axes)} axes, expected 2"
    )


def test_reachable_set_heatmap_returns_figure(tmp_path):
    """§8 synthesis figure: pcolormesh + smoothed-contour isochrones +
    vertical colorbar + 2-col 3-row legend with grey backdrop +
    ◆ IC callout. Single axis for the heatmap; the colorbar lives on
    its own axis (matplotlib appends it), so we expect >= 2 axes."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    from numerics import value_function_grid  # noqa: PLC0415
    from plots import reachable_set_heatmap  # noqa: PLC0415

    x1, x2, V, mask, iso = value_function_grid(
        w=0.45, ell=0.5, T_max_cap=8.0,
        grid_extent=(-3.0, 3.0, -2.0, 4.0),
        n_grid=30, n_traj=60, n_time_samples=60,
        isochrone_levels=(2.0, 4.0),
        cache_dir=str(tmp_path),
    )
    fig = reachable_set_heatmap(
        x1, x2, V, mask, w=0.45, ell=0.5,
        T_max_cap=8.0, vmax_color=8.0,
        isochrone_levels=(2.0, 4.0),
        highlight_ic=(0.0, 1.0, "test IC"),
        highlight_ic_value=2.5,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    # Heatmap axis + colorbar axis
    assert len(fig.axes) >= 2, (
        f"reachable_set_heatmap produced {len(fig.axes)} axes, "
        f"expected >= 2 (heatmap + colorbar)"
    )
