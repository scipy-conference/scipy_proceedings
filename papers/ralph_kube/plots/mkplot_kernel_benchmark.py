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

# Runtime of the kernels, tested on 2020-05-22 on cori measured tests_performance/performance_coherence.py
# Average of 10 kernel-calls
# OMP_NUM_THREADS = 1, 2, 4, 8

# Runtime of analysis.kernels_spectral.kernel_coherence
runtime_coherence_no = [20.52, 10.14]
# Runtime of analysis.kernels_spectral_cy.kernel_coherence_64_cy
runtime_coherency_cy = [18.86, 9.46, 4.78, 2.63, 1.54, 1.01]# 64:, 0.75]

#runtime_crosspower_no = [5.32, 5.24]
runtime_crosspower_cy = [3.48, 1.71, 0.87, 0.50, 0.30, 0.18]# 64:, 0.21]

#runtime_crossphase_no = [5.45, 5.36]
runtime_crossphase_cy = [3.99, 1.96, 1.0, 0.57, 0.33, 0.21]# 64:, 0.23]

num_threads = [1, 2, 4, 8, 16, 32]

labels = num_threads

x = np.arange(len(num_threads))  # the label locations
width = 0.35  # the width of the bars

fig2 = plt.figure(figsize=(3.46, 3.46 / golden_ratio))

ax_c = fig2.add_axes([0.1, 0.2, 0.375, 0.65])
ax_c.set_title(r"C")
ax_sp = fig2.add_axes([0.5, 0.2, 0.375, 0.65])
ax_sp.set_title(r"S, P")
#ax_p = fig2.add_axes([0.6, 0.2, 0.25, 0.65])
#ax_p.set_title(r"P")

#rects1 = ax_c.bar(x - width/2, runtime_coherence_no, width, label='numpy')
rects2 = ax_c.bar(x + width/2, runtime_coherency_cy, width)#, label='Cython')

#rects1 = ax_s.bar(x - width/2, runtime_crosspower_no, width)
rects2 = ax_sp.bar(x - width/2, runtime_crosspower_cy, width, label="S")

#rects1 = ax_p.bar(x - width/2, runtime_crossphase_no, width)
rects2 = ax_sp.bar(x + width/2, runtime_crossphase_cy, width, label="P")

ax_sp.legend(loc="upper right")


ax_c.set_ylim((0.0, 21.5))
ax_sp.set_ylim((0.0, 4.6))
#ax_p.set_ylim((0.0, 4.6))

ax_c.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
ax_c.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
ax_c.yaxis.set_tick_params(direction="in", which="both", left=True, right=True)

ax_sp.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax_sp.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
#ax_p.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
#ax_p.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

ax_sp.yaxis.set_ticks_position("right")
ax_sp.yaxis.set_tick_params(direction="in", which="both", left=True, right=True)
#ax_sp.yaxis.set_ticklabels([])
#ax_p.yaxis.set_ticks_position("right")
#ax_p.yaxis.set_tick_params(direction="in", which="both", left=True, right=True)


fig2.text(0.5, 0.01, "Threads", ha="center", va="bottom")
fig2.text(0.975, 0.5, "Walltime / s", ha="right",va="center", rotation=270)


for ax in [ax_c, ax_sp]:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_ticklabels(["", "1", "2", "4", "8", "16", "32"])


plt.show()

fig2.savefig("kernel_performance.png", dpi=300)

# End of file mkplot_kernel_benchmark.py