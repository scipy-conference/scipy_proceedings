# Encoding: UTF-8 -*-

from os.path import join
import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

mpl.use("AGG")


data_dir = "../data"
df_fname = "kstar_dtn.csv"
save = True
fig_fname = "kstar_dtn_xfer.png" 

# 3.46inch is half column

data = np.genfromtxt(join(data_dir, df_fname), delimiter=",", skip_header=1)

fig = plt.figure(figsize=(3.46, 3.46 / 1.618))
ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

ax.plot(data[:, 0], label="1 proc.")
ax.plot(data[:, 1], label="2 proc.")
ax.plot(data[:, 2], label="4 proc.")
ax.plot(data[:, 3], label="8 proc.")
ax.legend(loc="lower right", fancybox=True, framealpha=0.5, ncol=2)

ax.set_ylabel("Transfer rate / MByte/s")
ax.set_xlabel("time/s")

plt.show()

if save:
    print("Saving to ", fig_fname)
    fig.savefig(fig_fname, dpi=300)


# End of file mkplot_kstar_dtn_xfer.py