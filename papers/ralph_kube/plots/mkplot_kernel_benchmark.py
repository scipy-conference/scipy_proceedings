# -*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

golden_ratio = 1.618

#mpl.use("AGG")

runtime_coherence_no = [0.2886, 0.2740, 0.2561, 0.2690]
runtime_coherency_cy = [0.1805, 0.0900, 0.0430, 0.0239]

runtime_crosspower_no = [0.0341, 0.0300, 0.0311, 0.0325]
runtime_crosspower_cy = [0.0310, 0.0142, 0.0060, 0.0049]

runtime_crossphase_no = [0.0357, 0.0319, 0.0339, 0.0321]
runtime_crossphase_cy = [0.0336, 0.0195, 0.0070, 0.0054]

num_threads = [1, 2, 4, 8]

labels = num_threads

x = np.arange(len(num_threads))  # the label locations
width = 0.35  # the width of the bars

# fig = plt.figure(figsize=(3.46, 3.46 / golden_ratio))
# ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
# rects1 = ax.bar(x - width/2, runtime_coherence_no, width, label='numpy')
# rects2 = ax.bar(x + width/2, runtime_coherency_cy, width, label='Cython')

# ax.set_xlabel(r"Threads")
# ax.set_ylabel("Walltime / s")

# ax.xaxis.set_ticklabels(["", "1", "2", "4", "8", ""])
# ax.legend()
# ax.set_title(r"$C$")
# fig.savefig("performance_coherence.png", dpi=300) 

# fig = plt.figure(figsize=(3.46, 3.46))
# ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
# rects1 = ax.bar(x - width/2, runtime_crosspower_no, width, label='numpy')
# rects2 = ax.bar(x + width/2, runtime_crosspower_cy, width, label='Cython')

# ax.set_xlabel(r"Threads")
# ax.set_ylabel("Walltime / s")

# ax.xaxis.set_ticklabels(["", "1", "2", "4", "8", ""])
# ax.legend()
# ax.set_title(r"$S$")
# fig.savefig("performance_crosspower.png", dpi=300)


# fig = plt.figure(figsize=(3.46, 3.46))
# ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
# rects1 = ax.bar(x - width/2, runtime_crossphase_no, width, label='numpy')
# rects2 = ax.bar(x + width/2, runtime_crossphase_cy, width, label='Cython')

# ax.set_xlabel(r"Threads")
# ax.set_ylabel(r"Walltime / s")

# ax.xaxis.set_ticklabels(["", "1", "2", "4", "8", ""])
# ax.legend()
# ax.set_title(r"$P$")
# fig.savefig("performance_crossphase.png", dpi=300)


fig2 = plt.figure(figsize=(3.46, 3.46 / golden_ratio))

ax_c = fig2.add_axes([0.1, 0.2, 0.25, 0.65])
ax_c.set_title(r"C")
ax_s = fig2.add_axes([0.36, 0.2, 0.25, 0.65])
ax_s.set_title(r"S")
ax_p = fig2.add_axes([0.6, 0.2, 0.25, 0.65])
ax_p.set_title(r"P")

rects1 = ax_c.bar(x - width/2, runtime_coherence_no, width, label='numpy')
rects2 = ax_c.bar(x + width/2, runtime_coherency_cy, width, label='Cython')

rects1 = ax_s.bar(x - width/2, runtime_crosspower_no, width)
rects2 = ax_s.bar(x + width/2, runtime_crosspower_cy, width)

rects1 = ax_p.bar(x - width/2, runtime_crossphase_no, width)
rects2 = ax_p.bar(x + width/2, runtime_crossphase_cy, width)


ax_c.set_ylim((0.0, 0.3))
ax_s.set_ylim((0.0, 0.04))
ax_p.set_ylim((0.0, 0.04))

ax_c.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax_c.yaxis.set_tick_params(direction="in", left=True, right=True)

ax_s.yaxis.set_ticks_position("right")
ax_s.yaxis.set_tick_params(direction="in", left=True, right=True)
ax_s.yaxis.set_ticklabels([])
ax_p.yaxis.set_ticks_position("right")
ax_p.yaxis.set_tick_params(direction="in", left=True, right=True)

ax_s.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax_p.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

fig2.text(0.5, 0.01, "Threads", ha="center", va="bottom")
fig2.text(0.975, 0.5, "Walltime / s", ha="right",va="center", rotation=270)


for ax in [ax_c, ax_s, ax_p]:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_ticklabels(["", "1", "2", "4", "8", ""])

plt.show()

fig2.savefig("kernel_performance.png", dpi=300)

# End of file mkplot_kernel_benchmark.py