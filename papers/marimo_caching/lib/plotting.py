"""Figure-saving helper and shared matplotlib styling."""
import pathlib

import matplotlib.pyplot as plt


FIGS = pathlib.Path("figs")
FIGS.mkdir(exist_ok=True)


plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi":     150,
})


def save_fig(fig, name: str):
    """Save the figure as both `.svg` and `.png`. The caller is
    expected to call `fig.tight_layout()` or manage gridspec spacing
    itself before invoking — multi-panel figures using gridspec
    incompatible with `tight_layout` were previously emitting a
    matplotlib warning on every figure save.
    """
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    return fig
