import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import richardson_lucy
from pylira.data import lincoln
from astropy.visualization import simple_norm

random_state = np.random.RandomState(936)

dataset = lincoln(random_state=random_state)


gridspec_kw = {
    "wspace": 0.1,
    "left": 0.01,
    "right": 0.99,
    "bottom": 0.01,
}

fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(3.8, 3), gridspec_kw=gridspec_kw
)

data = dataset["counts"]
norm = simple_norm(data, min_cut=0, max_cut=20, stretch="asinh", asinh_a=0.1)
axes[0, 0].imshow(data, norm=norm, origin="lower")
axes[0, 0].set_axis_off()
axes[0, 0].set_title("Data")

flux = dataset["flux"]
axes[0, 1].imshow(flux, norm=norm, origin="lower")
axes[0, 1].set_axis_off()
axes[0, 1].set_title("Ground Truth")

n_iters = [10, 30, 100, 300]

for n_iter, ax in zip(n_iters, axes.flat[2:]):
    result = richardson_lucy(
        image=dataset["counts"],
        psf=dataset["psf"],
        clip=False,
        num_iter=n_iter
    )
    ax.imshow(result, norm=norm, origin="lower")
    ax.set_axis_off()
    ax.set_title(f"RL $N_{{iter}} = {n_iter}$")

plt.savefig("../images/richardson-lucy.png", dpi=300, facecolor="w")
