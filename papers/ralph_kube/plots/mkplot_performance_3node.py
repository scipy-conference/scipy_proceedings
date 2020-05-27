# -*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import datetime
import pandas as pd

from matplotlib.ticker import MultipleLocator, NullLocator

golden_ratio = 1.618
data_dir = "../data/tests_performance/N128_OMP16_32nodes_3node_v1/"

file_list_raw = [join(data_dir, f"outfile_{N:03d}.txt") for N in range(128)]
file_list = [f for f in file_list_raw if isfile(f)]
file_list_raw = None

columns = ["Name", "tidx", "node", "Start", "End"]
dframe = pd.DataFrame(columns=columns)

for foo in file_list:
    # Grab node from file name
    node = int(foo[-7:-4])
    with open(foo, "r") as df:
        for line in df.readlines():
            splits = line.split()
            tidx = int(splits[2][5:])
            name = splits[3]
            tstart = datetime.datetime.strptime(splits[5] + " " + splits[6], "%Y-%m-%d %H:%M:%S.%f")
            tend = datetime.datetime.strptime(splits[8] + " " + splits[9], "%Y-%m-%d %H:%M:%S.%f")
            
            new_row = {"Name": name, "tidx":tidx, "node": node, "Start": tstart, "End": tend}
            dframe = dframe.append(new_row, ignore_index=True)


with open(join(data_dir, "delta.log"), "r") as lf:
    l0 = lf.readline()
    splits = l0.split()
    # There is a , that needs to be removed
    bad_idx = splits[2].find(",")
    toff = datetime.datetime.strptime(splits[1] + " " + splits[2][:-8], "%Y-%m-%d %H:%M:%S") 


fig = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

for rank in range(128):
    delta_start = dframe[dframe["node"] == rank]["Start"] - toff
    delta_end = dframe[dframe["node"] == rank]["End"] - toff

    tic = np.array([r.seconds for r in delta_start])
    toc = np.array([r.seconds for r in delta_end])
    
    ax.barh(width=toc-tic, height=1.0, y=[rank] * len(tic), left=tic)
    
ax.set_ylabel("MPI rank")
ax.set_xlabel(r"Walltime / s")

ax.xaxis.set_major_locator(MultipleLocator(50.0))
ax.xaxis.set_minor_locator(MultipleLocator(10.0))

ax.yaxis.set_major_locator(MultipleLocator(32.0))
ax.yaxis.set_minor_locator(MultipleLocator(4.0))


fig.savefig("nodes_walltime_3node.png", dpi=300)

plt.show()
# End of file mkplot_performance_3node.py