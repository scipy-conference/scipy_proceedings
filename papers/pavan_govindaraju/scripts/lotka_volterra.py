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
from scipy.integrate import solve_ivp
from nodefit.neural_ode import NeuralODE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

torch.manual_seed(1)
np.random.seed(1)


def lotka_volterra(t, state, alpha=1.5, beta=1.0, gamma=3.0, delta=1.0):
    prey, predator = state
    return [
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator,
    ]


class LotkaVolterraDrift(nn.Module):
    """A structured neural ODE whose coefficients are learned from the data."""

    def __init__(self):
        super().__init__()
        initial = torch.tensor([1.0, 0.8, 2.5, 0.8]).log()
        self.log_rates = nn.Parameter(initial)

    def forward(self, x):
        prey, predator = x[1:]
        alpha, beta, gamma, delta = self.log_rates.exp()
        return torch.stack(
            (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )
        )


def run_lotka_volterra_example():
    t_full = np.linspace(0, 12, 241)
    solution = solve_ivp(
        lotka_volterra,
        (t_full[0], t_full[-1]),
        [1.5, 1.0],
        t_eval=t_full,
        rtol=1e-9,
        atol=1e-11,
    )
    true_values = solution.y.T

    train_mask = t_full <= 6
    t_train = t_full[train_mask][::2]
    data_train = true_values[train_mask][::2]

    model = NeuralODE(
        LotkaVolterraDrift().double(),
        t_train,
        data_train,
    )
    model.train(num_epochs=1000, print_every=200)
    extrapolated = model.extrapolate(tf=12, npts=241)

    t_fit = model.t.detach().cpu().numpy()
    fit_values = model.nn_data.detach().cpu().numpy()
    t_extra = extrapolated["time"]
    extra_values = extrapolated["values"].detach().cpu().numpy()
    t_model = np.concatenate([t_fit, t_extra[1:]])
    model_values = np.concatenate([fit_values, extra_values[1:]])
    learned_rates = model.neural_net.log_rates.exp().detach().cpu().numpy()
    print(f"Learned rates alpha, beta, gamma, delta: {learned_rates}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Prey", "Predator"]
    colors = ["#1f77b4", "#d62728"]
    for index, label in enumerate(labels):
        axes[0].plot(
            t_full,
            true_values[:, index],
            color=colors[index],
            linestyle=":",
            label=f"True {label}",
        )
        axes[0].plot(
            t_model,
            model_values[:, index],
            color=colors[index],
            linewidth=2,
            label=f"Neural ODE {label}",
        )
        axes[0].plot(
            t_train,
            data_train[:, index],
            "o",
            color=colors[index],
            markersize=3,
            label=f"Observed {label}" if index == 0 else None,
        )

    axes[0].axvline(6, color="0.5", linestyle="-.", linewidth=1)
    axes[0].set_xlabel("Time", fontsize=12)
    axes[0].set_ylabel("Population", fontsize=12)
    axes[0].set_title("Time evolution", fontsize=12)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(fontsize=12)

    axes[1].plot(
        true_values[:, 0],
        true_values[:, 1],
        "k:",
        label="True phase portrait",
    )
    axes[1].plot(
        model_values[:, 0],
        model_values[:, 1],
        color="#2ca02c",
        linewidth=2,
        label="Neural ODE",
    )
    axes[1].plot(
        data_train[:, 0],
        data_train[:, 1],
        "o",
        color="#9467bd",
        markersize=3,
        label="Observed",
    )
    axes[1].set_xlabel("Prey population", fontsize=12)
    axes[1].set_ylabel("Predator population", fontsize=12)
    axes[1].set_title("Phase portrait", fontsize=12)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(fontsize=12)

    fig.suptitle(
        "Lotka–Volterra reconstruction from the first half of the trajectory",
        fontsize=12,
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "lotka_volterra_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_lotka_volterra_example()
