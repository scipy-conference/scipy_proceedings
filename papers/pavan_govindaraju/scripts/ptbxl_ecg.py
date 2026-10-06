"""Fit a Neural SDE to a dense continuous heart rate trajectory from a PTB-XL ECG recording."""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Set writable cache directory for Matplotlib before importing it
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

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
import torch
import torch.nn as nn
from nodefit.constants import DEVICE
from nodefit.neural_sde import NeuralSDE, SDE
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from torchsde import sdeint

PAPER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_DIR / "data"
RESULTS_DIR = PAPER_DIR / "results"
RECORD = DATA_DIR / "16834_hr"
TRAIN_END_S = 6.0
RECORD_END_S = 10.0
N_EPOCHS = 600
BATCH_SIZE = 35
DT = 0.05
W0 = 2.0 * np.pi * 0.35  # fundamental frequency ~ 0.35 Hz (respiratory period ~ 2.85s)
K_HARMONICS = 3

LEAD_FIGURES = {
    "II": ("ptbxl_ecg_raw.png", "ptbxl_hrv_sde.png"),
    "V5": ("ptbxl_ecg_raw_v5.png", "ptbxl_hrv_sde_v5.png"),
}
COMBINED_RAW = "ptbxl_ecg_raw_combined.png"
COMBINED_SDE = "ptbxl_hrv_sde_combined.png"


def read_wfdb_lead(record_path, lead_name):
    """Read a specific lead from a WFDB format pair (.hea / .dat)."""
    header_path = Path(str(record_path) + ".hea")
    data_path = Path(str(record_path) + ".dat")
    lines = header_path.read_text().splitlines()
    _name, n_signals, fs, n_samples = lines[0].split()[:4]
    n_signals, fs, n_samples = int(n_signals), float(fs), int(n_samples)

    leads, gains, baselines = [], [], []
    for line in lines[1 : 1 + n_signals]:
        parts = line.split()
        gain_token = parts[2]
        gain = float(gain_token.split("(")[0].split("/")[0])
        baseline = 0.0
        if "(" in gain_token:
            baseline = float(gain_token.split("(")[1].split(")")[0])
        leads.append(parts[-1])
        gains.append(gain)
        baselines.append(baseline)

    raw = np.fromfile(data_path, dtype="<i2").reshape(n_samples, n_signals)
    index = leads.index(lead_name)
    physical = (raw[:, index].astype(np.float64) - baselines[index]) / gains[index]
    time_arr = np.arange(n_samples, dtype=np.float64) / fs
    return time_arr, physical, fs


def cubic(t, a, b, c, d):
    return a + b * t + c * t**2 + d * t**3


class FastSDE(SDE):
    def f(self, t, y):
        t_tensor = y.new_full((y.shape[0], 1), t)
        return self.drift_nn(torch.cat((t_tensor, y), dim=1))

    def g(self, t, y):
        t_tensor = y.new_full((y.shape[0], 1), t)
        return self.diffusion_nn(torch.cat((t_tensor, y), dim=1))


class PeriodicStationarySDE(SDE):
    def __init__(self, drift_nn, diffusion_nn, k_harmonics=3, w0=W0):
        super().__init__(drift_nn, diffusion_nn)
        self.k_harmonics = k_harmonics
        self.w0 = w0

    def _features(self, t, y):
        t_col = y.new_full((y.shape[0], 1), t)
        feats = []
        for k in range(1, self.k_harmonics + 1):
            feats.append(torch.sin((k * self.w0) * t_col))
            feats.append(torch.cos((k * self.w0) * t_col))
        feats.append(y)
        return torch.cat(feats, dim=1)

    def f(self, t, y):
        return self.drift_nn(self._features(t, y))

    def g(self, t, y):
        raw = self.diffusion_nn(self._features(t, y))
        return 0.18 + 0.12 * torch.sigmoid(raw)


class FastNeuralSDE(NeuralSDE):
    def __init__(self, drift_nn, diffusion_nn, t, data, batch_size=35, dt=0.05):
        super().__init__(drift_nn, diffusion_nn, t, data, batch_size)
        self.sde = FastSDE(drift_nn, diffusion_nn)
        self.dt = dt
        self.t = self.t.to(dtype=torch.float32, device=DEVICE)
        self.data = self.data.to(dtype=torch.float32, device=DEVICE)
        self.y0 = self.y0.to(dtype=torch.float32, device=DEVICE)
        self.drift_nn = drift_nn.to(dtype=torch.float32, device=DEVICE)
        self.diffusion_nn = diffusion_nn.to(dtype=torch.float32, device=DEVICE)

    def loss(self):
        self.nn_data = sdeint(
            self.sde, self.y0, self.t, method=self.sde.numerical_method, dt=self.dt
        )
        mu = self.nn_data.mean(dim=1)
        diff = self.data - mu
        return torch.mean(diff**2)

    def train(self, num_epochs=600):
        for i in range(num_epochs):
            self.sde.drift_opt.zero_grad()
            self.sde.diffusion_opt.zero_grad()
            current_loss = self.loss()
            current_loss.backward()
            self.sde.drift_opt.step()
            self.sde.diffusion_opt.step()

    def extrapolate_dense(self, t_dense, n_samples=150):
        y0_eval = self.y0[0:1].repeat(n_samples, 1)
        return sdeint(
            self.sde,
            y0_eval,
            torch.tensor(t_dense, dtype=torch.float32, device=DEVICE),
            method=self.sde.numerical_method,
            dt=self.dt,
        )


def extract_hr(t_raw, voltage, fs):
    peaks, _ = find_peaks(voltage, prominence=0.3, distance=int(0.35 * fs))
    peak_times = t_raw[peaks]
    rr_intervals = np.diff(peak_times)
    rr_midpoints = (peak_times[:-1] + peak_times[1:]) / 2.0
    hr_inst = 60.0 / rr_intervals
    pchip = PchipInterpolator(rr_midpoints, hr_inst)
    t_dense_all = np.linspace(rr_midpoints[0], rr_midpoints[-1], 90)
    hr_dense_all = pchip(t_dense_all)
    return peaks, peak_times, rr_midpoints, hr_inst, t_dense_all, hr_dense_all


def interpolate_at(t_query, t_grid, y_grid):
    return np.interp(t_query, t_grid, y_grid)


def coverage_stats(t_obs, y_obs, t_grid, mean_grid, std_grid):
    mean_obs = interpolate_at(t_obs, t_grid, mean_grid)
    std_obs = interpolate_at(t_obs, t_grid, std_grid)
    err = np.abs(y_obs - mean_obs)
    within_1 = np.mean(err <= std_obs)
    within_2 = np.mean(err <= 2.0 * std_obs)
    rmse = float(np.sqrt(np.mean((y_obs - mean_obs) ** 2)))
    mean_1s = float(np.mean(std_obs))
    mean_2s = float(np.mean(2.0 * std_obs))
    return within_1, within_2, rmse, mean_1s, mean_2s


def fit_lead(lead="II"):
    torch.manual_seed(42)
    np.random.seed(42)

    t_raw, voltage, fs = read_wfdb_lead(RECORD, lead)
    peaks, peak_times, rr_midpoints, hr_inst, t_dense_all, hr_dense_all = extract_hr(
        t_raw, voltage, fs
    )

    train_mask = t_dense_all <= TRAIN_END_S
    t_train = t_dense_all[train_mask]
    y_train = (hr_dense_all[train_mask] / 100.0).reshape(-1, 1)
    test_mask = t_dense_all > TRAIN_END_S
    t_test = t_dense_all[test_mask]
    y_test = (hr_dense_all[test_mask] / 100.0).reshape(-1, 1)
    n_train, n_test = len(t_train), len(t_test)
    print(
        f"Lead {lead}: {len(peaks)} R-peaks, {n_train} train points, {n_test} test points"
    )

    in_dim = 2 * K_HARMONICS + 1
    drift_nn = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )
    diffusion_nn = nn.Sequential(
        nn.Linear(in_dim, 24),
        nn.Tanh(),
        nn.Linear(24, 1),
    )

    started = time.perf_counter()
    sde = FastNeuralSDE(
        drift_nn, diffusion_nn, t_train, y_train, batch_size=BATCH_SIZE, dt=DT
    )
    sde.sde = PeriodicStationarySDE(drift_nn, diffusion_nn, k_harmonics=K_HARMONICS, w0=W0)
    sde.sde.drift_opt = torch.optim.Adam(drift_nn.parameters(), lr=0.008)
    sde.sde.diffusion_opt = torch.optim.Adam(diffusion_nn.parameters(), lr=0.008)
    sde.train(num_epochs=N_EPOCHS)

    t_eval = np.linspace(t_train[0], t_dense_all[-1], 250)
    full_sim = sde.extrapolate_dense(t_eval, n_samples=150).detach().cpu().numpy()
    elapsed = time.perf_counter() - started
    print(f"Lead {lead} wall-clock time (train + extrapolate): {elapsed:.2f} s")

    mean_dense = full_sim[:, :, 0].mean(axis=1) * 100.0
    std_dense = full_sim[:, :, 0].std(axis=1) * 100.0

    popt, _ = curve_fit(cubic, t_train, y_train[:, 0] * 100.0, p0=[100.0, 0.0, 0.0, 0.0])
    base_eval = cubic(t_eval, *popt)
    base_test = cubic(t_test, *popt)
    y_test_bpm = y_test[:, 0] * 100.0
    y_train_bpm = y_train[:, 0] * 100.0

    within_1, within_2, rmse_sde, mean_1s, mean_2s = coverage_stats(
        t_test, y_test_bpm, t_eval, mean_dense, std_dense
    )
    rmse_base = float(np.sqrt(np.mean((y_test_bpm - base_test) ** 2)))
    print(
        f"Lead {lead} held-out: SDE RMSE={rmse_sde:.1f} BPM, "
        f"curve_fit RMSE={rmse_base:.1f} BPM, "
        f"within 1σ={100 * within_1:.0f}%, within 2σ={100 * within_2:.0f}%, "
        f"mean ±1σ={mean_1s:.1f} BPM, mean ±2σ={mean_2s:.1f} BPM"
    )

    return {
        "lead": lead,
        "t_raw": t_raw,
        "voltage": voltage,
        "peaks": peaks,
        "peak_times": peak_times,
        "rr_midpoints": rr_midpoints,
        "hr_inst": hr_inst,
        "t_train": t_train,
        "y_train_bpm": y_train_bpm,
        "t_test": t_test,
        "y_test_bpm": y_test_bpm,
        "t_eval": t_eval,
        "mean_dense": mean_dense,
        "std_dense": std_dense,
        "base_eval": base_eval,
        "n_train": n_train,
        "n_test": n_test,
        "rmse_sde": rmse_sde,
        "rmse_base": rmse_base,
        "within_1": within_1,
        "within_2": within_2,
        "mean_1s": mean_1s,
        "mean_2s": mean_2s,
    }


def _panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", pad=6)


def plot_raw_panel(ax, result, ylabel=True, xlabel=True, legend=False):
    lead = result["lead"]
    t_raw = result["t_raw"]
    volt = result["voltage"]
    ax.plot(t_raw, volt, color="#4a5568", lw=0.75, label="ECG waveform", zorder=2)
    ax.plot(result["peak_times"], volt[result["peaks"]], "r^", markersize=4.0, label="Detected R-peaks", zorder=4)
    ax.axvline(TRAIN_END_S, color="0.45", linestyle="-.", lw=1.1, label=r"Forecast horizon ($t=6$s)", zorder=3)

    # Highlight a representative QRS complex interval (~1.45s to 1.60s) in distinct color
    qrs_mask = (t_raw >= 1.45) & (t_raw <= 1.60)
    ax.plot(t_raw[qrs_mask], volt[qrs_mask], color="#3182ce", lw=1.8, label="QRS complex (sample)", zorder=3)

    # Annotation arrow pointing to the R-peak of this QRS complex (placed cleanly inside plot area without title clash)
    qrs_t = result["peak_times"][2]  # ~ 1.53s
    qrs_v = volt[result["peaks"][2]]
    ax.annotate(
        f"QRS deflection\n({qrs_v:.2f} mV)",
        xy=(qrs_t, qrs_v),
        xytext=(qrs_t + 1.1, 0.75 if lead == "V5" else 0.65),
        arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.3),
        fontsize=8.5,
        fontweight="bold",
        color="#2c5282",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ebf8ff", edgecolor="#90cdf4", alpha=0.95),
        zorder=5,
    )

    if ylabel:
        ax.set_ylabel("Voltage (mV)", fontsize=12)
    if xlabel:
        ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xlim(0, RECORD_END_S)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="upper right", fontsize=11)


def plot_sde_panel(ax, result, xlabel=True, legend=False):
    n_train, n_test = result["n_train"], result["n_test"]
    ax.fill_between(
        result["t_eval"],
        result["mean_dense"] - 2 * result["std_dense"],
        result["mean_dense"] + 2 * result["std_dense"],
        color="#bee3f8",
        alpha=0.25,
        label=r"Neural SDE ($\pm 2\sigma$)",
        zorder=1,
    )
    ax.fill_between(
        result["t_eval"],
        result["mean_dense"] - result["std_dense"],
        result["mean_dense"] + result["std_dense"],
        color="#63b3ed",
        alpha=0.35,
        label=r"Neural SDE ($\pm 1\sigma$)",
        zorder=2,
    )
    ax.plot(result["t_eval"], result["mean_dense"], color="#2b6cb0", lw=2.0, label="Neural SDE mean", zorder=3)
    ax.plot(result["t_eval"], result["base_eval"], "r--", lw=1.7, label="curve_fit (cubic)", zorder=3)
    ax.plot(
        result["t_train"],
        result["y_train_bpm"],
        "k.",
        markersize=4.0,
        label=rf"Training ($t\leq 6$s, $N={n_train}$)",
        zorder=4,
    )
    ax.plot(
        result["t_test"],
        result["y_test_bpm"],
        "s",
        color="#2b6cb0",
        markersize=3.2,
        label=rf"Held-out ($t>6$s, $N={n_test}$)",
        zorder=4,
    )
    ax.plot(
        result["rr_midpoints"],
        result["hr_inst"],
        "o",
        color="#e53e3e",
        markersize=4.5,
        fillstyle="none",
        markeredgewidth=1.1,
        label="Discrete beats",
        zorder=5,
    )
    ax.axvline(TRAIN_END_S, color="0.45", linestyle="-.", lw=1.1)
    ax.set_ylabel("Heart rate (BPM)", fontsize=12)
    if xlabel:
        ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xlim(0, RECORD_END_S)
    ax.set_ylim(50, 180)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="upper left", fontsize=12, ncol=3, framealpha=0.92)


def save_individual_figures(result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_name, sde_name = LEAD_FIGURES[result["lead"]]

    fig1, ax1 = plt.subplots(figsize=(9.0, 2.9))
    plot_raw_panel(ax1, result, legend=True)
    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / raw_name, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9.0, 4.2))
    plot_sde_panel(ax2, result, legend=True)
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / sde_name, dpi=200, bbox_inches="tight")
    plt.close(fig2)


def save_combined_figures(results):
    """1x2 raw ECG (II, V5) and two stacked rectangular HR / SDE panels."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    by_lead = {r["lead"]: r for r in results}

    fig_raw, axes_raw = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, lead, tag in zip(axes_raw, ("II", "V5"), ("a", "b")):
        plot_raw_panel(ax, by_lead[lead], ylabel=(lead == "II"), legend=False)
        _panel_label(ax, f"({tag}) Lead {lead}")
    vmin = min(by_lead[lead]["voltage"].min() for lead in ("II", "V5"))
    vmax = max(by_lead[lead]["voltage"].max() for lead in ("II", "V5"))
    axes_raw[0].set_ylim(vmin - 0.08, vmax + 0.35)
    handles, labels = axes_raw[0].get_legend_handles_labels()
    fig_raw.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=10.0,
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig_raw.tight_layout(rect=(0, 0, 1, 0.90))
    out_raw = RESULTS_DIR / COMBINED_RAW
    fig_raw.savefig(out_raw, dpi=200, bbox_inches="tight")
    plt.close(fig_raw)

    fig_sde, axes_sde = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    for ax, lead, tag in zip(axes_sde, ("II", "V5"), ("a", "b")):
        plot_sde_panel(ax, by_lead[lead], xlabel=(lead == "V5"), legend=False)
        _panel_label(ax, f"({tag}) Lead {lead}")
    handles, labels = axes_sde[0].get_legend_handles_labels()
    fig_sde.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=12,
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    fig_sde.tight_layout(rect=(0, 0, 1, 0.93))
    out_sde = RESULTS_DIR / COMBINED_SDE
    fig_sde.savefig(out_sde, dpi=200, bbox_inches="tight")
    plt.close(fig_sde)

    print(f"Generated combined figures:\n  {out_raw}\n  {out_sde}")
    return out_raw, out_sde


def run_ptbxl_example(lead="II", save_individual=False):
    result = fit_lead(lead)
    if save_individual:
        save_individual_figures(result)
        raw_name, sde_name = LEAD_FIGURES[lead]
        print(f"Generated figures:\n  {RESULTS_DIR / raw_name}\n  {RESULTS_DIR / sde_name}")
    return result


CACHE_FILE = RESULTS_DIR / "ptbxl_ecg_results.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "leads",
        nargs="*",
        default=["II", "V5"],
        help="ECG leads to analyze (default: II and V5)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also write per-lead figure files",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate plots from cached results if available",
    )
    args = parser.parse_args()

    if args.plot_only and CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded cached results from {CACHE_FILE}")
    else:
        results = [run_ptbxl_example(lead=lead, save_individual=args.individual) for lead in args.leads]
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(results, f)

    if {"II", "V5"}.issubset({r["lead"] for r in results}):
        save_combined_figures(results)
