"""
Benchmark comparing the baseline NeuralSDE with the optimized FastNeuralSDE.

This script measures the empirical training speedup achieved by:
1. Using y.new_full for state-time concatenation (avoids repeated tensor creation/transfers).
2. Using float32 instead of float64.
3. Pre-allocating repeated batch data and setting explicit solver step size (dt=0.1).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchsde import sdeint
from nodefit.neural_sde import NeuralSDE, SDE
from nodefit.constants import DEVICE

# Set MPLCONFIGDIR to avoid font cache rebuild overhead if matplotlib is imported
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))


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
        self.nn_data = sdeint(self.sde, self.y0, self.t,
                              method=self.sde.numerical_method,
                              dt=self.dt).to(DEVICE)

        # Calculate mean and std across the trajectories (batch dimension)
        mu = self.nn_data.mean(dim=1)
        std = self.nn_data.std(dim=1) + 1e-4  # Add epsilon for stability

        # Gaussian Negative Log-Likelihood
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


def run_benchmark(num_epochs=20, batch_size=20):
    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate synthetic time-series data
    t = np.linspace(0, 5, 30)
    y = 0.5 * t + 1.0 + np.random.normal(0, 0.2, 30)
    data = y.reshape(-1, 1)

    print(f"--- Benchmarking Neural SDE ({num_epochs} Epochs, batch_size={batch_size}) ---")

    # 1. Baseline Model (double precision, default NeuralSDE)
    drift_nn_base = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1)).double()
    diff_nn_base = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1)).double()
    sde_base = NeuralSDE(drift_nn_base, diff_nn_base, t, data, batch_size=batch_size)

    # Warmup
    sde_base.train(2, print_every=100)

    print("Running baseline NeuralSDE training...")
    t0 = time.time()
    sde_base.train(num_epochs, print_every=num_epochs + 1)
    base_time = time.time() - t0
    base_per_epoch = base_time / num_epochs

    # 2. Optimized Model (FastNeuralSDE with float32 and dt=0.1)
    drift_nn_fast = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))
    diff_nn_fast = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))
    sde_fast = FastNeuralSDE(drift_nn_fast, diff_nn_fast, t, data, batch_size=batch_size, dt=0.1)

    # Warmup
    sde_fast.train(2, print_every=100)

    print("Running optimized FastNeuralSDE training...")
    t0 = time.time()
    sde_fast.train(num_epochs, print_every=num_epochs + 1)
    fast_time = time.time() - t0
    fast_per_epoch = fast_time / num_epochs

    speedup = base_time / fast_time

    print("\n--- Results ---")
    print(f"Baseline:  {base_time:.2f} s total ({base_per_epoch:.4f} s/epoch)")
    print(f"Optimized: {fast_time:.2f} s total ({fast_per_epoch:.4f} s/epoch)")
    print(f"Speedup:   {speedup:.1f}x faster")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_benchmark(num_epochs=epochs)
