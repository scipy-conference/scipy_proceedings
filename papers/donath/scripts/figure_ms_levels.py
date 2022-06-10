import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

fig = plt.figure(figsize=(3.8, 3.8))

ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_axis_off()


sub_pixel = Rectangle(
    (0, 0.75), width=0.25, height=0.25, color="tab:blue", alpha=0.5
)
ax.add_patch(sub_pixel)

pixel = Rectangle(
    (0, 0.5), width=0.5, height=0.5, color="tab:blue", alpha=0.2
)
ax.add_patch(pixel)

x = np.linspace(0, 1, 5)
ax.vlines(x=x, ymin=0, ymax=1, color="black", lw=0.5, alpha=0.4)

y = np.linspace(0, 1, 5)
ax.hlines(y=y, xmin=0, xmax=1, color="black", lw=0.5, alpha=0.4)

x = np.linspace(0, 1, 3)
ax.vlines(x=x, ymin=0, ymax=1, color="black", lw=1)

y = np.linspace(0, 1, 3)
ax.hlines(y=y, xmin=0, xmax=1, color="black", lw=1)


ax.text(0.25, 0.75, "$Q_1$", va="center", ha="center", size=26)
ax.text(0.75, 0.75, "$Q_2$", va="center", ha="center", size=26)
ax.text(0.25, 0.25, "$Q_3$", va="center", ha="center", size=26)
ax.text(0.75, 0.25, "$Q_4$", va="center", ha="center", size=26)
ax.text(0.125, 0.875, "$\Lambda_{11}$", va="center", ha="center", size=26)

plt.savefig("../images/ms-levels.png", dpi=300, facecolor="white")
