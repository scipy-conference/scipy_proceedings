# -*- Encoding: UTF-8 -*-

from os.path import isdir, join, isfile

import numpy as np 
import pandas as pd

import datetime

import matplotlib.pyplot as plt 
import seaborn as sns

from matplotlib.ticker import MultipleLocator, NullLocator


dir_run1 = "../data/tests_performance/N128_OMP16_32nodes_2node"
dir_run2 = "../data/tests_performance/N128_OMP16_32nodes_bp4"
assert(isdir(dir_run1))
assert(isdir(dir_run2))

logfile_run1 = join(dir_run1, "delta.log")
logfile_run2 = join(dir_run2, "delta.log")
assert(isfile(logfile_run1))
assert(isfile(logfile_run2))

golden_ratio = 1.618
columns = ["time_log", "tidx", "time_fft", "mode"]
dframe = pd.DataFrame(columns=columns)

with open(logfile_run1, "r") as df:
    l0 = df.readline()
    splits = l0.split()
    # There is a , that needs to be removed
    fix_mus = splits[2].replace(",", ".", 1)
    fix_mus = fix_mus.replace(",", "")
    toff_run1 = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 

    for line in df:
        if "FFT took" in line:
            splits = line.split()
            fix_mus = splits[2].replace(",", ".", 1)
            fix_mus = fix_mus.replace(",", "")
            time_log = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 
            tidx = int(splits[10][:-1])
            time_fft = float(splits[13][:-1])
            
            
            new_row = {"time_log": time_log - toff_run1, "tidx":int(splits[10][:-1]), 
                       "time_fft": datetime.timedelta(seconds=float(splits[13][:-1])),
                       "mode": "2node"}
            dframe = dframe.append(new_row, ignore_index=True)

with open(logfile_run2, "r") as df:
    l0 = df.readline()
    splits = l0.split()
    # There is a , that needs to be removed
    fix_mus = splits[2].replace(",", ".", 1)
    fix_mus = fix_mus.replace(",", "")
    toff_run2 = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 

    for line in df:
        if "FFT took" in line:
            splits = line.split()
            fix_mus = splits[2].replace(",", ".", 1)
            fix_mus = fix_mus.replace(",", "")
            time_log = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 
            tidx = int(splits[10][:-1])
            time_fft = float(splits[13][:-1])
            
            
            new_row = {"time_log": time_log - toff_run2, "tidx":int(splits[10][:-1]), 
                       "time_fft": datetime.timedelta(seconds=float(splits[13][:-1])),
                       "mode": "bp4"}
            dframe = dframe.append(new_row, ignore_index=True)

fig = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

for index, row in dframe.iterrows():
    toff = row.time_log.seconds + row.time_log.microseconds * 1e-6
    tdelta = row.time_fft.seconds + row.time_fft.microseconds * 1e-6

    fc = None
    if row["mode"] == "bp4":
        fc = "C0"
    elif row["mode"] == "2node":
        fc = "C1"
    
    ax.broken_barh([(toff, tdelta)], (row.tidx, 1), facecolors=fc)

ax.xaxis.set_major_locator(MultipleLocator(50.0))
ax.xaxis.set_minor_locator(MultipleLocator(10.0))

ax.yaxis.set_major_locator(MultipleLocator(100.0))
ax.yaxis.set_minor_locator(MultipleLocator(20.0))

ax.set_xlabel("Walltime / s")
ax.set_ylabel(r"$n_\mathrm{ch}$")
fig.savefig("performance_fft.png", dpi=300)



dframe["time_fft_secs"] = dframe["time_fft"].apply(lambda x: x.seconds + x.microseconds * 1e-6)

fig2 = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
ax2 = fig2.add_axes([0.2, 0.2, 0.75, 0.75])
sns.violinplot(x="mode", y="time_fft_secs", data=dframe, ax=ax2)
ax2.set(ylabel=r"$t_{\mathrm{FFT}} / \mathrm{s}$", xlabel="")
ax2.set_xticklabels(["2-node", "file"])

ax2.yaxis.set_major_locator(MultipleLocator(2.5))
ax2.yaxis.set_minor_locator(MultipleLocator(0.5))

fig2.savefig("performance_fft_violin.png", dpi=300)


plt.show()


# End of file mkplot_performance_fft.py