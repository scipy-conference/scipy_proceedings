import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

############################################################
# Example 1: Photon Counts
print("Example #1")

# Generate the data
np.random.seed(1)  # for repeatability
F_true = 1000  # true flux, say number of photons measured in 1 second
N = 50 # number of measurements
F = stats.poisson(F_true).rvs(N)  # N measurements of the flux
e = np.sqrt(F)  # errors on Poisson counts estimated via square root

# Visualize the data
fig, ax = plt.subplots()
ax.errorbar(F, np.arange(N), xerr=e, fmt='ok', ecolor='gray', alpha=0.5)
ax.vlines([F_true], 0, N, linewidth=5, alpha=0.2)
ax.set_xlabel("Flux");ax.set_ylabel("measurement number")
ax.set_xlim(850, 1150)
ax.set_ylim(0, 50)
fig.savefig("figure1.png")
print("  Saving figure1.png")

# Frequentist Result
w = 1. / e ** 2
print("""
      F_true = {0}
      F_est  = {1:.0f} +/- {2:.0f} (based on {3} measurements)
      """.format(F_true, (w * F).sum() / w.sum(), w.sum() ** -0.5, N))
