# -*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import datetime
import pandas as pd

from matplotlib.ticker import MultipleLocator, NullLocator

golden_ratio = 1.618

dir_list = ["../data/tests_performance/N128_OMP16_32nodes_file_v10",
            "../data/tests_performance/N128_OMP16_32nodes_2node_v10_fast_sleep01",
            "../data/tests_performance/N128_OMP16_32nodes_3node_v10"]


NMPI = 128

for rundir, title in zip(dir_list, ["file", "2node", "3node"]):

    file_list_raw = [join(rundir, f"outfile_{N:03d}.txt") for N in range(NMPI)]
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
                
    with open(join(rundir, "delta.log"), "r") as lf:
        l0 = lf.readline()
        splits = l0.split()
        print(splits)
        # There is a , that needs to be removed
        fix_mus = splits[2].replace(",", ".", 1)
        fix_mus = fix_mus.replace(",", "")
        toff = datetime.datetime.strptime(splits[1] + " " + fix_mus, "%Y-%m-%d %H:%M:%S.%f") 
        print(toff)
    
    fig = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    for rank in range(1, 128):
        dframe_node = dframe.loc[dframe["node"] == rank]

        for name, C in zip(["cross_phase", "cross_power", "cross_correlation", "coherence"],
                        ["C0", "C1", "C2", "C3"]):
            dframe_sub = dframe_node.loc[dframe["Name"] == name]
            
    #         print(name, rank)
    #         print(dframe_sub)
            
            delta_start = dframe_sub["Start"] - toff
            delta_end = dframe_sub["End"] - toff

            tic = np.array([r.seconds for r in delta_start])
            toc = np.array([r.seconds for r in delta_end]) + 0.1

            ax.barh(width=toc-tic, height=1.0, y=[rank] * len(tic), left=tic, color=C)

        ax.set_ylabel("MPI rank")
        ax.set_xlabel(r"Walltime / s")

        ax.xaxis.set_major_locator(MultipleLocator(50.0))
        ax.xaxis.set_minor_locator(MultipleLocator(10.0))
        ax.yaxis.set_major_locator(MultipleLocator(32.0))
        ax.yaxis.set_minor_locator(MultipleLocator(4.0))

        ax.set_xlim((0.0, 375.0))

    fig.savefig(f"mpirank_utilization_{title}.png")


# End of file mkplot_mpi_ranks.py