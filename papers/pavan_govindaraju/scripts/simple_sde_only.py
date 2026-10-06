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

class FastSDE(SDE):
    def __init__(self, drift_nn, diffusion_nn, noise_type="diagonal", sde_type="ito", numerical_method="euler"):
        super().__init__(drift_nn, diffusion_nn, noise_type, sde_type, numerical_method)

    def f(self, t, y):
        # Faster way to concatenate t: use y.new_full
        t_tensor = y.new_full((y.shape[0], 1), t)
        combined = torch.cat((t_tensor, y), dim=1)
        return self.drift_nn(combined)

    def g(self, t, y):
        t_tensor = y.new_full((y.shape[0], 1), t)
        combined = torch.cat((t_tensor, y), dim=1)
        return self.diffusion_nn(combined)

class FastNeuralSDE(NeuralSDE):
    def __init__(self, drift_nn: nn.Module, diffusion_nn: nn.Module, t, data, batch_size=2, dt=0.1):
        # Get dtype from the neural network
        dtype = next(drift_nn.parameters()).dtype

        # Initialize with standard NeuralSDE
        super().__init__(drift_nn, diffusion_nn, t, data, batch_size)

        # Override with FastSDE and correct dtypes
        self.sde = FastSDE(drift_nn, diffusion_nn)
        self.dt = dt

        # Ensure all tensors match the NN dtype
        self.t = self.t.to(dtype=dtype)
        self.data = self.data.to(dtype=dtype)
        self.y0 = self.y0.to(dtype=dtype)

        # Pre-calculate repeated data to save time in loss()
        self.repeated_data = self.data.unsqueeze(1).repeat(1, self.batch_size, 1).to(DEVICE)

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
        # Loss = 0.5 * [ ((data - mu) / std)^2 + 2 * log(std) ]
        diff = (self.data - mu)
        nll = 0.5 * (torch.pow(diff / std, 2) + 2 * torch.log(std)).mean()

        return nll

    def train(self, num_epochs, print_every=100):
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

def run_simple_sde_example():
    print("Running Optimized Simple Neural SDE example (Linear drift + noise)...")
    # Generate simple linear data with noise
    t = np.linspace(0, 5, 30)
    y = 0.5 * t + 1.0 + np.random.normal(0, 0.2, 30)
    data = y.reshape(-1, 1) # 1D data

    # Use float32 for speed
    # input_dim = 1 (time) + 1 (data dim) = 2
    drift_nn = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )

    diffusion_nn = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )

    # Initialize and train the optimized SDE model
    # dt=0.1 is set here outside the core module
    sde = FastNeuralSDE(drift_nn, diffusion_nn, t, data, batch_size=20, dt=0.1)
    sde.train(num_epochs=1000, print_every=100)

    # Extrapolate
    extrapolated = sde.extrapolate(tf=8, npts=40)

    # Plotting
    plt.figure(figsize=(10, 6))
    t_np = sde.t.cpu().numpy()
    data_np = sde.data.cpu().numpy()
    nn_data_np = sde.nn_data.detach().cpu().numpy()
    extra_t = extrapolated['time']
    extra_v = extrapolated['values'].detach().cpu().numpy()

    # Plot trained data
    plt.plot(t_np, data_np[:, 0], 'o', label='Trained Data', markersize=4)

    # Plot NN solution mean and std (across the batch dimension)
    # nn_data_np shape: (time, batch_size, data_dim)
    mean = np.mean(nn_data_np[:, :, 0], axis=1)
    std = np.std(nn_data_np[:, :, 0], axis=1)
    plt.plot(t_np, mean, '-', label='NN Mean')
    plt.fill_between(t_np, mean - std, mean + std, alpha=0.3, label='NN Std Dev')

    # Plot extrapolation mean and std
    ex_mean = np.mean(extra_v[:, :, 0], axis=1)
    ex_std = np.std(extra_v[:, :, 0], axis=1)
    plt.plot(extra_t, ex_mean, '--', label='Extrapolated Mean')
    plt.fill_between(extra_t, ex_mean - ex_std, ex_mean + ex_std, alpha=0.2, label='Extrapolated Std Dev')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Optimized Simple Neural SDE: Linear Drift + Noise')
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "simple_sde_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Simple Neural SDE example finished. Plot saved as {out_path}")

if __name__ == "__main__":
    run_simple_sde_example()
