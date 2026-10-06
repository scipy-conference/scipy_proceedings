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

from nodefit.neural_sde import NeuralSDE, SDE
from nodefit.constants import DEVICE
from torchsde import sdeint
from tqdm import tqdm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

class AutonomousFastSDE(SDE):
    def __init__(self, drift_nn, diffusion_nn, noise_type="diagonal", sde_type="ito", numerical_method="euler"):
        super().__init__(drift_nn, diffusion_nn, noise_type, sde_type, numerical_method)

    def f(self, t, y):
        # Autonomous drift: depends on state y only
        return self.drift_nn(y)

    def g(self, t, y):
        # Autonomous diffusion: depends on state y only
        return self.diffusion_nn(y)

class FastNeuralSDE(NeuralSDE):
    def __init__(self, drift_nn: nn.Module, diffusion_nn: nn.Module, t, data, batch_size=2, dt=0.1):
        # Get dtype from the neural network
        dtype = next(drift_nn.parameters()).dtype

        # Initialize with standard NeuralSDE
        super().__init__(drift_nn, diffusion_nn, t, data, batch_size)

        # Override with AutonomousFastSDE and correct dtypes
        self.sde = AutonomousFastSDE(drift_nn, diffusion_nn)
        self.dt = dt

        # Ensure all tensors match the NN dtype
        self.t = self.t.to(dtype=dtype, device=DEVICE)
        self.data = self.data.to(dtype=dtype, device=DEVICE)
        self.y0 = self.y0.to(dtype=dtype, device=DEVICE)

    def loss(self):
        if self.data is None:
            raise Exception('Load the data before training')

        # Simulate trajectories
        # self.nn_data shape: (time, batch_size, data_dim)
        self.nn_data = sdeint(self.sde, self.y0, self.t,
                              method=self.sde.numerical_method,
                              dt=self.dt).to(DEVICE)

        # Calculate mean and std across the trajectories (batch dimension)
        mu = self.nn_data.mean(dim=1)
        std = self.nn_data.std(dim=1) + 1e-4  # Add epsilon for stability

        # Gaussian Negative Log-Likelihood:
        # We want to maximize the likelihood of the observed 'data'
        # under the distribution N(mu, std^2) produced by the SDE.
        diff = (self.data - mu)
        nll = 0.5 * (torch.pow(diff / std, 2) + 2 * torch.log(std)).mean()

        return nll

    def train(self, num_epochs, print_every=100, lr=0.01):
        self.sde.drift_opt = torch.optim.Adam(self.sde.drift_nn.parameters(), lr=lr)
        self.sde.diffusion_opt = torch.optim.Adam(self.sde.diffusion_nn.parameters(), lr=lr)
        for i in tqdm(range(num_epochs)):
            self.sde.drift_opt.zero_grad()
            self.sde.diffusion_opt.zero_grad()

            # Call loss() only once per iteration
            current_loss = self.loss()
            current_loss.backward()

            self.sde.drift_opt.step()
            self.sde.diffusion_opt.step()

            if i % print_every == 0:
                print(f'Epoch {i}/{num_epochs}, Loss: {current_loss.item()}')

def run_ode_trajectory_sde_example():
    print("Running Optimized Neural SDE on ODE trajectory (with noise)...")
    # Generate smooth synthetic time-series data (2D) - same as ODE example but with noise
    t = np.linspace(0, 5, 50)

    # Underlying theoretical trajectory
    y1_theory = 1.0 + 2.2 * (1 - np.exp(-0.5 * t))
    y2_theory = 1.0 + 0.6 * (1 - np.exp(-0.5 * t))

    # Noisy observations
    y1 = y1_theory + np.random.normal(0, 0.1, 50)
    y2 = y2_theory + np.random.normal(0, 0.1, 50)
    data = np.stack([y1, y2], axis=1)

    # Use float32 for speed
    # Autonomous formulation: Input dim = 2 (data dim y only)
    drift_nn = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 2)
    )

    diffusion_nn = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 2)
    )

    # Initialize and train the optimized SDE model
    # Using batch_size=20 and dt=0.1 for speed and stability
    sde = FastNeuralSDE(drift_nn, diffusion_nn, t, data, batch_size=20, dt=0.1)
    sde.train(num_epochs=500, print_every=50, lr=0.01)

    # Extrapolate
    tf_extra = 8
    extrapolated = sde.extrapolate(tf=tf_extra, npts=40)

    # Theoretical extrapolation
    t_extra = np.linspace(0, tf_extra, 100)
    y1_theory_extra = 1.0 + 2.2 * (1 - np.exp(-0.5 * t_extra))
    y2_theory_extra = 1.0 + 0.6 * (1 - np.exp(-0.5 * t_extra))

    # Plotting
    plt.figure(figsize=(12, 8))
    t_np = sde.t.cpu().numpy()
    data_np = sde.data.cpu().numpy()
    nn_data_np = sde.nn_data.detach().cpu().numpy()
    extra_t = extrapolated['time']
    extra_v = extrapolated['values'].detach().cpu().numpy()

    theory_extra = [y1_theory_extra, y2_theory_extra]

    for i in range(data_np.shape[1]):
        # Plot trained data
        line, = plt.plot(t_np, data_np[:, i], 'o', label=f'Observed Data {i+1}', markersize=4)
        color = line.get_color()

        # Plot theoretical trajectory
        plt.plot(t_extra, theory_extra[i], ':', color='black', alpha=0.6, label=f'Theoretical Mean {i+1}' if i == 0 else "")

        # Plot NN solution mean and std
        # nn_data_np shape: (time, batch_size, data_dim)
        mean = np.mean(nn_data_np[:, :, i], axis=1)
        std = np.std(nn_data_np[:, :, i], axis=1)
        plt.plot(t_np, mean, '-', color=color, label=f'Predicted Mean {i+1}')
        plt.fill_between(t_np, mean - std, mean + std, color=color, alpha=0.2)

        # Plot extrapolation mean and std
        ex_mean = np.mean(extra_v[:, :, i], axis=1)
        ex_std = np.std(extra_v[:, :, i], axis=1)
        plt.plot(extra_t, ex_mean, '--', color=color, label=f'Extrapolated Mean {i+1}')
        plt.fill_between(extra_t, ex_mean - ex_std, ex_mean + ex_std, color=color, alpha=0.1)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Neural SDE: Theoretical vs Predicted Mean on Complex Dynamics')
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "trajectory_sde_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Neural SDE on ODE trajectory finished. Plot saved as {out_path}")

if __name__ == "__main__":
    run_ode_trajectory_sde_example()
