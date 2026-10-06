import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Set writable cache directory for Matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12,
})
from scipy.optimize import curve_fit
from nodefit.neural_ode import NeuralODE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def cubic(t, a, b, c, d):
    """Misspecified template: flexible enough to fit training, no steady state."""
    return a + b * t + c * t**2 + d * t**3


class AutonomousDrift(nn.Module):
    """Drift depends on state only; appropriate when kinetics are time-invariant."""

    def __init__(self, hidden_dim=20, state_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x):
        return self.net(x[1:])


def fit_curve_fit_baseline(t_train, data_train, t_eval):
    """Fit each channel independently with a wrong functional form."""
    fits = []
    for channel in range(data_train.shape[1]):
        y = data_train[:, channel]
        popt, _ = curve_fit(
            cubic,
            t_train,
            data_train[:, channel],
            p0=[1.0, 0.5, 0.0, 0.0],
            maxfev=10000,
        )
        fits.append(cubic(t_eval, *popt))
    return np.stack(fits, axis=1)


def run_ode_example():
    print("Running Neural ODE example with curve_fit baseline...")
    # Generate smooth synthetic time-series data (2D)
    t_train = np.linspace(0, 5, 50)
    y1 = 1.0 + 2.2 * (1 - np.exp(-0.5 * t_train))
    y2 = 1.0 + 0.6 * (1 - np.exp(-0.5 * t_train))
    data = np.stack([y1, y2], axis=1)

    # Baseline: scipy.optimize.curve_fit with a misspecified template
    t_plot = np.linspace(0, 10, 200)
    baseline = fit_curve_fit_baseline(t_train, data, t_plot)

    ode = NeuralODE(AutonomousDrift().double(), t_train, data)
    ode.train(num_epochs=1000, print_every=200)
    extrapolated = ode.extrapolate(tf=10, npts=150)

    t_np = ode.t.cpu().numpy()
    data_np = ode.data.cpu().numpy()
    nn_fit = ode.nn_data.detach().cpu().numpy()
    t_ex = extrapolated["time"]
    nn_ex = extrapolated["values"].detach().cpu().numpy()
    t_ode = np.concatenate([t_np, t_ex[1:]])
    nn_traj = np.concatenate([nn_fit, nn_ex[1:]])
    train_end = t_train[-1]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, ax in enumerate(axes):
        ax.plot(
            t_plot,
            baseline[:, i],
            color=colors[i],
            linestyle="--",
            linewidth=2,
            label="curve_fit (cubic)",
        )
        ax.plot(
            t_ode,
            nn_traj[:, i],
            color=colors[i],
            linewidth=2,
            label="Neural ODE",
        )
        ax.plot(
            t_np,
            data_np[:, i],
            "o",
            color=colors[i],
            markersize=4,
            label="Training data",
        )
        ax.axvline(train_end, color="0.5", linestyle="-.", linewidth=1)
        ax.set_title(f"State {i + 1}", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        if i == 0:
            ax.set_ylabel("Value", fontsize=12)
        ax.set_xlim(0, 10)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, loc="upper right" if i == 1 else "lower right")

    axes[1].text(
        train_end + 0.05,
        axes[1].get_ylim()[0],
        "train end",
        fontsize=12,
        color="0.4",
        va="bottom",
    )
    fig.suptitle(
        "Saturating kinetics: cubic curve_fit inflects beyond training; Neural ODE extrapolates",
        fontsize=12,
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ode_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved as {out_path}")


if __name__ == "__main__":
    run_ode_example()
