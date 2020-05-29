# -*- Encoding: UTF-8 -*-

from os.path import isdir, join, isfile

import numpy as np 
import pandas as pd

import datetime

import matplotlib.pyplot as plt 
import seaborn as sns

from matplotlib.ticker import MultipleLocator, NullLocator

golden_ratio = 1.618


dir_list = ["../data/tests_performance/N128_OMP16_32nodes_file_v10",
            "../data/tests_performance/N128_OMP16_32nodes_2node_v10_fast_sleep01",
            "../data/tests_performance/N128_OMP16_32nodes_3node_v10"]

            
logfile_list = []
for d in dir_list:
    print(d)
    logfile_list.append(join(d, "delta.log"))
    assert(isfile(logfile_list[-1]))
    assert(isdir(d))

columns = ["time_log", "tidx", "time_fft", "runnr"]
dframe = pd.DataFrame(columns=columns)
for run, logfile in enumerate(logfile_list):
    runnr = f"run{run:1d}"
    with open(logfile, "r") as df:
        l0 = df.readline()
        splits = l0.split()
        # There is a , that needs to be removed
        fix_mus = splits[2].replace(",", ".", 1)
        fix_mus = fix_mus.replace(",", "")
        toff_run = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 

        print(f"run: {run}, toff_run = {toff_run}")        
        
        for line in df:
            if "FFT took" in line:
                splits = line.split()
                fix_mus = splits[2].replace(",", ".", 1)
                fix_mus = fix_mus.replace(",", "")
                time_log = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 
                tidx = int(splits[10][:-1])
                time_fft = float(splits[13][:-1])


                new_row = {"time_log": time_log - toff_run, "tidx":int(splits[10][:-1]), 
                           "time_fft": datetime.timedelta(seconds=float(splits[13][:-1])),
                           "runnr": runnr}
                dframe = dframe.append(new_row, ignore_index=True)

fig = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

for index, row in dframe.iterrows():
    toff = row.time_log.seconds + row.time_log.microseconds * 1e-6
    tdelta = row.time_fft.seconds + row.time_fft.microseconds * 1e-6

    fc = "C" + row["runnr"][-1]
    ax.broken_barh([(toff, tdelta)], (row.tidx, 1), facecolors=fc)
    

ax.plot([0.0], [0.0], 'C0.', label="File")
ax.plot([0.0], [0.0], 'C1.', label="2-node")
ax.plot([0.0], [0.0], 'C2.', label="3-node")
ax.set_xlim((10.0, 175.0))
ax.set_ylim((0.0, 505))
ax.legend(loc="lower right")

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
sns.violinplot(x="runnr", y="time_fft_secs", data=dframe, ax=ax2)
ax2.set(ylabel=r"$t_{\mathrm{FFT}} / \mathrm{s}$", xlabel="")
ax2.set_xticklabels(["file", "2-node", "3-node"])

ax2.yaxis.set_major_locator(MultipleLocator(2.5))
ax2.yaxis.set_minor_locator(MultipleLocator(0.5))

fig2.savefig("performance_fft_violin.png", dpi=300)


plt.show()


# End of file mkplot_performance_fft.py