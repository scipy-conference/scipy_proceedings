# Encoding: UTF-8 -*-

from os.path import join
import numpy as np 

from matplotlib import rc
rc("font",**{'family':'sans-serif','sans-serif':['Helvetica']})
rc("text", usetex=True)
import matplotlib.pyplot as plt

data_dir = "../data"
df_fname = "kstar_dtn.csv"
save = True
fig_fname = "kstar_dtn_xfer.png" 

# 3.46inch is half column

data = np.genfromtxt(join(data_dir, df_fname), delimiter=",", skip_header=1)

fig = plt.figure(figsize=(3.46, 3.46))
ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

ax.plot(data[:, 0], label="1 proc.")
ax.plot(data[:, 1] / 2, label="2 proc.")
ax.plot(data[:, 2] / 4, label="4 proc.")
ax.plot(data[:, 3] / 8, label="8 proc.")
ax.legend(loc="lower right", fancybox=True, framealpha=0.5, ncol=2)

ax.set_ylabel(r"Transfer rate / MByte/s")
ax.set_xlabel(r"time/s")

if save:
    print("Saving to ", fig_fname)
    fig.savefig(fig_fname)



# End of file mkplot_kstar_dtn_xfer.py