"""
Batch evaluation of NODEFit Neural SDE on a multi-patient PTB-XL cohort.

This script:
1. Downloads PTB-XL records for N unique patients (e.g. 2,000) from PhysioNet.
2. Runs continuous heart rate trajectory fitting on a specified number of samples (variable N).
3. Evaluates both Lead II and Lead V5 separately.
4. Compares Neural SDE against curve_fit deterministic baselines (cubic polynomial, harmonic).
5. Tracks empirical 1-SD and 2-SD coverage as a function of forecast distance from t=6s.
6. Exports structured summary tables and publication-ready summary plots.
"""

import argparse
import os
import signal
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configure headless matplotlib environment
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from torchsde import sdeint

from nodefit.constants import DEVICE
from nodefit.neural_sde import NeuralSDE, SDE

PAPER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_DIR / "data"
RESULTS_DIR = PAPER_DIR / "results"
RECORDS_DIR = DATA_DIR / "ptbxl_records"
DATABASE_CSV = DATA_DIR / "ptbxl_database.csv"
BASE_PHYSIONET_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

TRAIN_END_S = 6.0
RECORD_END_S = 10.0
W0 = 2.0 * np.pi * 0.35  # fundamental autonomic frequency ~ 0.35 Hz
K_HARMONICS = 3


# ---------------------------------------------------------------------------
# 1. Dataset Downloading & Management
# ---------------------------------------------------------------------------

def ensure_database_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATABASE_CSV.exists() or DATABASE_CSV.stat().st_size == 0:
        print("Downloading PTB-XL database metadata index (ptbxl_database.csv)...")
        url = f"{BASE_PHYSIONET_URL}ptbxl_database.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(DATABASE_CSV, "wb") as f:
            f.write(r.read())
        print("Downloaded database index.")
    return pd.read_csv(DATABASE_CSV)


def get_unique_patient_records(n_patients=2000, seed=42):
    df = ensure_database_csv()
    # Filter for unique patient IDs with valid filenames
    unique_df = df.dropna(subset=["patient_id", "filename_hr", "filename_lr"])
    unique_df = unique_df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    if n_patients is not None and n_patients > 0:
        unique_df = unique_df.head(n_patients)
    return unique_df


def download_single_record(row, records_root=RECORDS_DIR, hr=True):
    col = "filename_hr" if hr else "filename_lr"
    rel_path = row[col]
    record_name = Path(rel_path).name
    subfolder = Path(rel_path).parent.name
    target_dir = records_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for ext in [".hea", ".dat"]:
        out_file = target_dir / f"{record_name}{ext}"
        if out_file.exists() and out_file.stat().st_size > 0:
            continue
        url = f"{BASE_PHYSIONET_URL}{rel_path}{ext}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r, open(out_file, "wb") as f:
                f.write(r.read())
        except Exception:
            if out_file.exists():
                out_file.unlink()
            success = False
    return success


def download_cohort_records(n_patients=2000, max_workers=20, hr=True):
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    patient_df = get_unique_patient_records(n_patients)
    print(f"Checking/downloading records for {len(patient_df)} unique patients (sampling_rate={'500Hz' if hr else '100Hz'})...")

    t0 = time.time()
    downloaded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_record, row, RECORDS_DIR, hr): row for _, row in patient_df.iterrows()}
        pbar = tqdm(as_completed(futures), total=len(patient_df), desc="Syncing Records", unit="rec")
        for future in pbar:
            if future.result():
                downloaded += 1
            else:
                failed += 1
            pbar.set_postfix({"ready": downloaded, "failed": failed})

    print(f"Cohort data sync complete: {downloaded} records verified in {time.time()-t0:.1f}s.")
    return patient_df


# ---------------------------------------------------------------------------
# 2. WFDB Lead Reader & Signal Utilities
# ---------------------------------------------------------------------------

def get_local_record_path(row, hr=True):
    col = "filename_hr" if hr else "filename_lr"
    rel_path = row[col]
    subfolder = Path(rel_path).parent.name
    record_name = Path(rel_path).name
    return RECORDS_DIR / subfolder / record_name


def read_wfdb_lead_from_path(record_path, lead_name):
    header_path = Path(str(record_path) + ".hea")
    data_path = Path(str(record_path) + ".dat")
    if not header_path.exists() or not data_path.exists():
        return None, None, None

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

    if lead_name not in leads:
        return None, None, None

    raw = np.fromfile(data_path, dtype="<i2").reshape(n_samples, n_signals)
    index = leads.index(lead_name)
    physical = (raw[:, index].astype(np.float64) - baselines[index]) / gains[index]
    time_arr = np.arange(n_samples, dtype=np.float64) / fs
    return time_arr, physical, fs


def extract_heart_rate_trajectory(time_arr, voltage, fs):
    """Robust R-peak detection and continuous HR trajectory derivation."""
    v_ptp = np.ptp(voltage)
    if v_ptp < 0.1 or np.isnan(v_ptp):
        return None

    # Multi-threshold peak search for diverse patient morphologies
    peaks = None
    for prom_ratio in [0.35, 0.25, 0.18]:
        p, _ = find_peaks(voltage, prominence=prom_ratio * v_ptp, distance=int(0.30 * fs))
        if len(p) >= 5:
            peaks = p
            break

    if peaks is None or len(peaks) < 5:
        return None

    peak_times = time_arr[peaks]
    rr_intervals = np.diff(peak_times)

    # Filter non-physiological artifacts
    if np.any(rr_intervals < 0.25) or np.any(rr_intervals > 2.5):
        valid = (rr_intervals >= 0.25) & (rr_intervals <= 2.5)
        if np.sum(valid) < 4:
            return None
        rr_intervals = rr_intervals[valid]
        peak_times_mid = (peak_times[:-1] + peak_times[1:]) / 2.0
        rr_midpoints = peak_times_mid[valid]
    else:
        rr_midpoints = (peak_times[:-1] + peak_times[1:]) / 2.0

    hr_inst = 60.0 / rr_intervals

    # Check sufficient train and test beats
    if np.sum(rr_midpoints <= TRAIN_END_S) < 3 or np.sum(rr_midpoints > TRAIN_END_S) < 2:
        return None

    pchip = PchipInterpolator(rr_midpoints, hr_inst)
    t_dense = np.linspace(rr_midpoints[0], rr_midpoints[-1], 90)
    hr_dense = pchip(t_dense)

    train_mask = t_dense <= TRAIN_END_S
    test_mask = t_dense > TRAIN_END_S

    if np.sum(train_mask) < 20 or np.sum(test_mask) < 10:
        return None

    return {
        "peaks": peaks,
        "peak_times": peak_times,
        "rr_midpoints": rr_midpoints,
        "hr_inst": hr_inst,
        "t_dense": t_dense,
        "hr_dense": hr_dense,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


# ---------------------------------------------------------------------------
# 3. Model Definitions & Baseline Fits
# ---------------------------------------------------------------------------

def cubic_model(t, a, b, c, d):
    return a + b * t + c * (t**2) + d * (t**3)


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


def fit_and_evaluate_patient_lead(row, lead, hr_traj, epochs=200, batch_size=16, dt=0.08):
    t_train = hr_traj["t_dense"][hr_traj["train_mask"]]
    y_train = (hr_traj["hr_dense"][hr_traj["train_mask"]] / 100.0).reshape(-1, 1)

    t_test = hr_traj["t_dense"][hr_traj["test_mask"]]
    y_test_bpm = hr_traj["hr_dense"][hr_traj["test_mask"]]

    in_dim = 2 * K_HARMONICS + 1
    drift_nn = nn.Sequential(
        nn.Linear(in_dim, 24),
        nn.Tanh(),
        nn.Linear(24, 24),
        nn.Tanh(),
        nn.Linear(24, 1),
    ).to(DEVICE)
    diffusion_nn = nn.Sequential(
        nn.Linear(in_dim, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    ).to(DEVICE)

    t_t = torch.tensor(t_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    y0 = y_t[0:1].repeat(batch_size, 1)

    sde = PeriodicStationarySDE(drift_nn, diffusion_nn, k_harmonics=K_HARMONICS, w0=W0)
    drift_opt = torch.optim.Adam(drift_nn.parameters(), lr=0.01)
    diff_opt = torch.optim.Adam(diffusion_nn.parameters(), lr=0.01)

    # Train Neural SDE
    for _ in range(epochs):
        drift_opt.zero_grad()
        diff_opt.zero_grad()
        sim = sdeint(sde, y0, t_t, method="euler", dt=dt)
        mu = sim.mean(dim=1)
        loss = torch.mean((y_t - mu) ** 2)
        loss.backward()
        drift_opt.step()
        diff_opt.step()

    # Extrapolate to evaluation grid covering test window
    t_eval = np.linspace(t_train[0], hr_traj["t_dense"][-1], 200)
    y0_eval = y_t[0:1].repeat(50, 1)
    with torch.no_grad():
        sim_eval = sdeint(
            sde,
            y0_eval,
            torch.tensor(t_eval, dtype=torch.float32, device=DEVICE),
            method="euler",
            dt=dt,
        ).detach().cpu().numpy()

    sde_mean_grid = sim_eval[:, :, 0].mean(axis=1) * 100.0
    sde_std_grid = sim_eval[:, :, 0].std(axis=1) * 100.0

    # Evaluate Neural SDE at test points
    sde_pred_test = np.interp(t_test, t_eval, sde_mean_grid)
    sde_std_test = np.interp(t_test, t_eval, sde_std_grid)
    abs_err_sde = np.abs(y_test_bpm - sde_pred_test)

    in_1sd = (abs_err_sde <= sde_std_test).astype(float)
    in_2sd = (abs_err_sde <= 2.0 * sde_std_test).astype(float)
    rmse_sde = float(np.sqrt(np.mean((y_test_bpm - sde_pred_test) ** 2)))

    # Baseline: curve_fit cubic polynomial
    y_train_bpm = y_train[:, 0] * 100.0
    try:
        popt_cubic, _ = curve_fit(cubic_model, t_train, y_train_bpm, p0=[y_train_bpm[0], 0.0, 0.0, 0.0], maxfev=2000)
        cubic_pred_test = cubic_model(t_test, *popt_cubic)
        rmse_cubic = float(np.sqrt(np.mean((y_test_bpm - cubic_pred_test) ** 2)))
    except Exception:
        cubic_pred_test = np.full_like(y_test_bpm, np.mean(y_train_bpm))
        rmse_cubic = float(np.sqrt(np.mean((y_test_bpm - cubic_pred_test) ** 2)))

    # Point-by-point time distance from t=6s
    delta_t = t_test - TRAIN_END_S

    return {
        "ecg_id": row["ecg_id"],
        "patient_id": row["patient_id"],
        "lead": lead,
        "is_norm": bool("NORM" in str(row.get("scp_codes", ""))),
        "n_train": len(t_train),
        "n_test": len(t_test),
        "rmse_sde": rmse_sde,
        "rmse_cubic": rmse_cubic,
        "cov_1sd_pct": float(np.mean(in_1sd) * 100.0),
        "cov_2sd_pct": float(np.mean(in_2sd) * 100.0),
        "mean_sigma": float(np.mean(sde_std_test)),
        "delta_t": delta_t.tolist(),
        "in_1sd_pts": in_1sd.tolist(),
        "in_2sd_pts": in_2sd.tolist(),
        "sde_err_pts": abs_err_sde.tolist(),
        "cubic_err_pts": np.abs(y_test_bpm - cubic_pred_test).tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Cohort Execution Engine
# ---------------------------------------------------------------------------

def run_cohort_benchmark(
    num_samples=100,
    download_samples=2000,
    leads=("II", "V5"),
    epochs=200,
    sampling_rate=500,
    checkpoint_csv=None,
    resume=True,
    save_every=5,
):
    hr = (sampling_rate == 500)
    ensure_database_csv()

    # Step 1: Ensure dataset is available
    patient_df = get_unique_patient_records(download_samples)
    print(f"\n=======================================================")
    print(f"PTB-XL Cohort Benchmark Pipeline (Headless & Checkpointed)")
    print(f"Total Unique Patient Records Target: {download_samples}")
    print(f"Active Evaluation Batch Size (N): {num_samples} patients")
    print(f"Leads: {leads} | SDE Epochs per Record: {epochs} | Fs: {sampling_rate}Hz")
    print(f"Resume Existing Checkpoint: {resume} | Auto-save frequency: every {save_every} patients")
    print(f"=======================================================\n")

    download_cohort_records(n_patients=download_samples, max_workers=25, hr=hr)

    # Step 2: Select sub-cohort of size num_samples
    eval_df = patient_df.head(num_samples)
    print(f"\nStarting benchmark on {len(eval_df)} patient records...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if checkpoint_csv is None:
        checkpoint_csv = RESULTS_DIR / "ptbxl_cohort_metrics.csv"
    else:
        checkpoint_csv = Path(checkpoint_csv)

    pts_csv = RESULTS_DIR / "ptbxl_cohort_pointwise.csv"
    summary_plot_path = RESULTS_DIR / "ptbxl_cohort_distance_summary.png"

    results_records = []
    point_distance_records = []
    completed_pairs = set()

    # Resume from existing checkpoints if requested
    if resume and checkpoint_csv.exists() and pts_csv.exists():
        try:
            existing_metrics = pd.read_csv(checkpoint_csv)
            existing_pts = pd.read_csv(pts_csv)
            results_records = existing_metrics.to_dict("records")
            point_distance_records = existing_pts.to_dict("records")
            for r in results_records:
                completed_pairs.add((float(r["patient_id"]), str(r["lead"])))
            print(f"Loaded existing checkpoint: {len(results_records)} lead fits already completed across {len(set(r['patient_id'] for r in results_records))} patients.")
        except Exception as e:
            print(f"Warning: Could not load existing checkpoint ({e}). Starting fresh.")
            results_records = []
            point_distance_records = []
            completed_pairs = set()

    def persist_progress():
        if not results_records:
            return
        m_df = pd.DataFrame(results_records)
        m_df.to_csv(checkpoint_csv, index=False)
        p_df = pd.DataFrame(point_distance_records)
        p_df.to_csv(pts_csv, index=False)
        if len(p_df) > 0 and len(m_df) > 0:
            try:
                generate_distance_summary_plots(p_df, m_df, out_path=summary_plot_path)
            except Exception:
                pass

    # Signal handling for safe exit
    def handle_interrupt(signum, frame):
        print("\nProcess interrupted! Persisting progress before exit...")
        persist_progress()
        print(f"Checkpoint safely saved ({len(results_records)} fits recorded).")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    t_start = time.time()
    pbar = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Benchmarking Patients", unit="patient")
    patients_since_save = 0

    for idx, (_, row) in enumerate(pbar, 1):
        rec_path = get_local_record_path(row, hr=hr)
        if not (Path(str(rec_path) + ".hea").exists() and Path(str(rec_path) + ".dat").exists()):
            continue

        patient_id = float(row["patient_id"])
        needed_leads = [lead for lead in leads if (patient_id, lead) not in completed_pairs]

        if not needed_leads:
            continue

        for lead in needed_leads:
            t_raw, voltage, fs = read_wfdb_lead_from_path(rec_path, lead)
            if t_raw is None or voltage is None:
                continue

            hr_traj = extract_heart_rate_trajectory(t_raw, voltage, fs)
            if hr_traj is None:
                continue

            try:
                res = fit_and_evaluate_patient_lead(row, lead, hr_traj, epochs=epochs)
            except Exception as e:
                continue

            results_records.append({
                "ecg_id": res["ecg_id"],
                "patient_id": res["patient_id"],
                "lead": res["lead"],
                "is_norm": res["is_norm"],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "rmse_sde": res["rmse_sde"],
                "rmse_cubic": res["rmse_cubic"],
                "cov_1sd_pct": res["cov_1sd_pct"],
                "cov_2sd_pct": res["cov_2sd_pct"],
                "mean_sigma": res["mean_sigma"],
            })
            completed_pairs.add((patient_id, lead))

            # Pointwise distance points
            for dt_val, in1, in2, err_sde, err_cub in zip(
                res["delta_t"], res["in_1sd_pts"], res["in_2sd_pts"],
                res["sde_err_pts"], res["cubic_err_pts"]
            ):
                point_distance_records.append({
                    "ecg_id": res["ecg_id"],
                    "lead": res["lead"],
                    "is_norm": res["is_norm"],
                    "delta_t": round(dt_val, 2),
                    "in_1sd": in1,
                    "in_2sd": in2,
                    "err_sde": err_sde,
                    "err_cubic": err_cub,
                })

        patients_since_save += 1
        if patients_since_save >= save_every:
            persist_progress()
            patients_since_save = 0

        # Update tqdm live metrics
        if results_records:
            latest_sde = np.mean([r["rmse_sde"] for r in results_records[-4:]])
            pbar.set_postfix({
                "fits": len(results_records),
                "SDE_RMSE": f"{latest_sde:.1f}BPM"
            })

    persist_progress()
    print(f"\nSaved final cohort metrics to {checkpoint_csv} and {pts_csv}")

    metrics_df = pd.DataFrame(results_records)
    pts_df = pd.DataFrame(point_distance_records)
    return metrics_df, pts_df


# ---------------------------------------------------------------------------
# 5. Summary Tables and Publication Plots
# ---------------------------------------------------------------------------

def compute_summary_table(metrics_df):
    """Generate structured summary tables split by Lead II and Lead V5."""
    tables = {}
    for lead in metrics_df["lead"].unique():
        lead_df = metrics_df[metrics_df["lead"] == lead]
        norm_df = lead_df[lead_df["is_norm"] == True]
        abnorm_df = lead_df[lead_df["is_norm"] == False]

        def get_stats(sub_df):
            return {
                "N": len(sub_df),
                "SDE_RMSE": f"{sub_df['rmse_sde'].mean():.2f} ± {sub_df['rmse_sde'].std():.2f}",
                "Cubic_RMSE": f"{sub_df['rmse_cubic'].mean():.2f} ± {sub_df['rmse_cubic'].std():.2f}",
                "Cov_1SD": f"{sub_df['cov_1sd_pct'].mean():.1f}%",
                "Cov_2SD": f"{sub_df['cov_2sd_pct'].mean():.1f}%",
                "Sigma": f"{sub_df['mean_sigma'].mean():.1f} BPM",
            }

        tables[lead] = {
            "All": get_stats(lead_df),
            "Normal": get_stats(norm_df),
            "Abnormal / Arrhythmia": get_stats(abnorm_df),
        }
    return tables


def format_markdown_table(tables):
    lines = []
    lines.append("| Lead | Cohort Stratum | N Patients | Neural SDE RMSE (BPM) | Cubic curve_fit RMSE (BPM) | Within 1σ Coverage (%) | Within 2σ Coverage (%) | Mean Diffusion ±1σ (BPM) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for lead, strata in tables.items():
        for stratum_name, stats in strata.items():
            lines.append(
                f"| **Lead {lead}** | {stratum_name} | {stats['N']} | "
                f"{stats['SDE_RMSE']} | {stats['Cubic_RMSE']} | "
                f"{stats['Cov_1SD']} | {stats['Cov_2SD']} | {stats['Sigma']} |"
            )
    return "\n".join(lines)


def generate_distance_summary_plots(pts_df, metrics_df, out_path=None):
    """
    Generate summary plots showing:
    1. How many samples fell within 1sd and 2sd at distance x from t=6s.
    2. RMSE comparison between Neural SDE and cubic curve_fit baseline vs distance from t=6s.
    Split cleanly into Lead II and Lead V5 rows/panels.
    """
    if out_path is None:
        out_path = RESULTS_DIR / "ptbxl_cohort_distance_summary.png"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)

    leads = [l for l in ["II", "V5"] if l in pts_df["lead"].unique()]
    if not leads:
        leads = list(pts_df["lead"].unique())[:2]

    # Time bins from 0 to 4.0 seconds past t=6s
    bin_edges = np.linspace(0.0, 3.8, 16)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for row_idx, lead in enumerate(leads):
        lead_pts = pts_df[pts_df["lead"] == lead].copy()
        lead_pts["bin"] = pd.cut(lead_pts["delta_t"], bins=bin_edges)

        binned = lead_pts.groupby("bin", observed=True).agg({
            "in_1sd": "mean",
            "in_2sd": "mean",
            "err_sde": lambda x: np.sqrt(np.mean(x**2)),
            "err_cubic": lambda x: np.sqrt(np.mean(x**2)),
            "delta_t": "mean",
        }).dropna()

        # Left column: Empirical Coverage vs Distance from t=6s
        ax_cov = axes[row_idx, 0]
        ax_cov.axhline(68.3, color="#718096", linestyle=":", lw=1.2, label=r"Theoretical $1\sigma$ (68.3%)")
        ax_cov.axhline(95.4, color="#a0aec0", linestyle="--", lw=1.2, label=r"Theoretical $2\sigma$ (95.4%)")
        ax_cov.plot(binned["delta_t"], binned["in_1sd"] * 100.0, "o-", color="#2b6cb0", lw=2.0, markersize=5, label=r"Empirical $\pm 1\sigma$ Coverage")
        ax_cov.plot(binned["delta_t"], binned["in_2sd"] * 100.0, "s-", color="#319795", lw=2.0, markersize=5, label=r"Empirical $\pm 2\sigma$ Coverage")
        ax_cov.set_ylabel(f"Lead {lead}\nSamples in Band (%)", fontsize=12, fontweight="bold")
        ax_cov.set_ylim(20, 105)
        ax_cov.tick_params(labelsize=12)
        ax_cov.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_cov.set_title(r"(a) Neural SDE Coverage vs Forecast Horizon ($\Delta t = t - 6\text{s}$)", fontsize=12, pad=6)
            ax_cov.legend(loc="lower left", fontsize=12)

        # Right column: Error Comparison (RMSE) vs Distance from t=6s
        ax_err = axes[row_idx, 1]
        ax_err.plot(binned["delta_t"], binned["err_sde"], "o-", color="#2b6cb0", lw=2.2, markersize=5, label="Neural SDE (Ours)")
        ax_err.plot(binned["delta_t"], binned["err_cubic"], "x--", color="#e53e3e", lw=1.8, markersize=4.5, label="Cubic curve_fit")
        ax_err.set_ylabel(f"Lead {lead}\nForecast RMSE (BPM)", fontsize=12, fontweight="bold")
        ax_err.tick_params(labelsize=12)
        ax_err.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_err.set_title(r"(b) Model Error vs Forecast Horizon ($\Delta t = t - 6\text{s}$)", fontsize=12, pad=6)
            ax_err.legend(loc="upper left", fontsize=12)

    for c in range(2):
        axes[1, c].set_xlabel(r"Forecast Distance from Horizon ($\Delta t = t - 6.0\text{s}$)", fontsize=12)
        axes[1, c].set_xlim(0, 3.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated distance summary plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 6. CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTB-XL Multi-Patient Cohort Neural SDE Benchmark")
    parser.add_argument("--download_samples", type=int, default=2000, help="Number of unique patient records to download (default: 2000)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of patient records to evaluate (variable N, default: 50)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per patient record (default: 200)")
    parser.add_argument("--leads", nargs="+", default=["II", "V5"], help="ECG leads to evaluate (default: II V5)")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500], help="Sampling rate in Hz (default: 500)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path for CSV checkpoint")
    parser.add_argument("--save_every", type=int, default=5, help="Autosave checkpoint frequency in patients (default: 5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint and start from patient 1")
    parser.add_argument("--plot_only", action="store_true", help="Generate summary plots from existing CSV results without re-running benchmark")

    args = parser.parse_args()

    if args.plot_only:
        pts_path = RESULTS_DIR / "ptbxl_cohort_pointwise.csv"
        metrics_path = RESULTS_DIR / "ptbxl_cohort_metrics.csv"
        if not pts_path.exists() or not metrics_path.exists():
            print(f"Error: Required CSV files not found in {RESULTS_DIR}")
            sys.exit(1)
        pts_df = pd.read_csv(pts_path)
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df, pts_df = run_cohort_benchmark(
            num_samples=args.num_samples,
            download_samples=args.download_samples,
            leads=args.leads,
            epochs=args.epochs,
            sampling_rate=args.sampling_rate,
            checkpoint_csv=args.checkpoint,
            resume=not args.overwrite,
            save_every=args.save_every,
        )

    summary_tables = compute_summary_table(metrics_df)
    md_table = format_markdown_table(summary_tables)
    print("\n" + "=" * 60)
    print("COHORT BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    print(md_table)
    print("=" * 60)

    plot_path = RESULTS_DIR / "ptbxl_cohort_distance_summary.png"
    generate_distance_summary_plots(pts_df, metrics_df, out_path=plot_path)
