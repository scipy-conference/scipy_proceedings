# --- Figure helpers (schematic diagrams) ------------------------------------
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)

# A color-blind-aware qualitative palette (after Paul Tol's "muted" scheme).
TOL = ["#332288", "#117733", "#CC6677", "#88CCEE", "#DDCC77",
       "#AA4499", "#44AA99", "#882255"]
INK, MUTED = "#1b1b1b", "#6b6b6b"


def _save(fig, name):
    fig.savefig(f"figures/{name}.png")
    fig.savefig(f"figures/{name}.pdf")   # vector copy for typesetting

def fig_timeline():
    """Figure 1: a staggered timeline of interactive scientific computing."""
    eras = [(1983, 2000.5, "Foundations\n(pre-Jupyter)", "#e9edf2"),
            (2000.5, 2013.5, "IPython & the\nscientific stack", "#dde9f1"),
            (2013.5, 2020.5, "Project Jupyter", "#e2efe5"),
            (2020.5, 2027, "IDE-native\n& AI", "#f3ecdc")]
    events = [(1984, "Knuth: Literate Programming", +1), (1984, "MATLAB / MathWorks", -1),
              (1991, "Python released", +2), (1995, "Numeric; WaveLab", -1),
              (2001, "IPython (F. Perez)", +1), (2003, "Matplotlib", -1),
              (2006, "NumPy 1.0", +2), (2008, "pandas", -1),
              (2011, "IPython Notebook", +1), (2014, "Project Jupyter", -1),
              (2015, '"Big Split"; Dask', +2), (2016, "JupyterLab; FAIR", -2),
              (2017, "ACM Software System Award", +1), (2018, "Binder 2.0", -3),
              (2020, "NumPy & SciPy in Nature", +3), (2021, "LLM code models", -1),
              (2024, "IDE-native notebooks", +2)]
    fig, ax = plt.subplots(figsize=(13.2, 6.0)); step = 0.255
    for a, b, label, color in eras:
        ax.axvspan(a, b, color=color, zorder=0)
        ax.text((a + b) / 2, 0.985, label, ha="center", va="top", fontsize=10.5,
                color="#3a3a3a", fontstyle="italic", zorder=2,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=INK, lw=2.4, zorder=3)
    for yr, label, lvl in events:
        y = step * lvl
        ax.plot([yr, yr], [0, y], color=MUTED, lw=1.0, zorder=3)
        ax.scatter([yr], [0], s=46, color="white", edgecolor=INK, zorder=5, linewidth=1.5)
        up = lvl > 0
        txt = f"{label}\n{yr}" if up else f"{yr}\n{label}"
        ax.annotate(txt, xy=(yr, y + (0.035 if up else -0.035)), ha="center",
                    va="bottom" if up else "top", fontsize=8.7, color=INK,
                    linespacing=1.05, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cfcfcf",
                              lw=0.6, alpha=0.92))
    ax.set_xlim(1982.5, 2027.5); ax.set_ylim(-1.02, 1.02); ax.set_yticks([])
    ax.set_xticks(np.arange(1985, 2026, 5))
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False); ax.tick_params(axis="x", length=0, pad=8, labelsize=10.5)
    ax.set_title("Twenty-five years of interactive scientific computing", fontsize=14, pad=14)
    fig.tight_layout(); _save(fig, "fig01_timeline")

def _box(ax, x, y, w, h, title, lines=None, fc="#eef2f7", ec="#3a4a5a",
         title_size=11, body_size=9.2, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h - 0.10, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color="#22303c", zorder=4)
    if lines:
        ax.text(x + w / 2, y + h - 0.34, "\n".join(lines), ha="center", va="top",
                fontsize=body_size, color="#33424e", linespacing=1.35, zorder=4)

def fig_architecture():
    """Figure 2: the Jupyter client/kernel architecture and messaging channels."""
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.3, 1.4, 3.2, 3.2, "Frontends (clients)",
         ["Jupyter Notebook", "JupyterLab", "Qt / terminal console",
          "nbconvert / nbclient", "IDE notebook editors"], fc="#e7eef6")
    _box(ax, 8.5, 1.4, 3.2, 3.2, "Kernels (language processes)",
         ["IPython / ipykernel  (Python)", "IRkernel  (R)", "IJulia  (Julia)",
          "xeus  (C++, ...)", "many community kernels"], fc="#e6f1e8")
    channels = [("Shell  -  execute_request / reply", "#2b6cb0", 4),
                ("IOPub  -  stdout, display_data, results", "#2f855a", 3),
                ("stdin  -  input_request", "#975a16", 2),
                ("Control  -  interrupt, shutdown", "#702459", 1),
                ("Heartbeat  -  liveness", "#822727", 0)]
    y_top, y_bot, n = 4.35, 1.65, 5
    for label, color, i in channels:
        y = y_bot + (y_top - y_bot) * (i / (n - 1))
        ax.annotate("", xy=(8.45, y), xytext=(3.55, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.7,
                                    shrinkA=0, shrinkB=0), zorder=2)
        ax.text(6.0, y + 0.085, label, ha="center", va="bottom", fontsize=8.6,
                color=color, zorder=5)
    ax.text(6.0, 4.95, "ZeroMQ messaging protocol  (JSON over five sockets)",
            ha="center", va="center", fontsize=10.2, fontstyle="italic", color="#2d3748")
    _box(ax, 4.1, 0.2, 3.8, 0.95, "Notebook document  -  nbformat (.ipynb, JSON)",
         ["cells  -  outputs  -  metadata  -  kernelspec"], fc="#f4eee0",
         ec="#7a6a3a", title_size=9.6, body_size=8.4, lw=1.2)
    ax.annotate("", xy=(3.4, 1.4), xytext=(4.6, 1.15),
                arrowprops=dict(arrowstyle="-", color="#7a6a3a", lw=1.0, ls=(0, (3, 2))))
    ax.text(3.95, 1.30, "read / write", fontsize=7.6, color="#7a6a3a", rotation=18)
    ax.set_title("The Jupyter client-kernel architecture", fontsize=13.5, pad=6)
    fig.tight_layout(); _save(fig, "fig02_architecture")

def fig_ecosystem():
    """Figure 4: the array-centric scientific Python stack under an interactive layer."""
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(1.55, 7.35); ax.axis("off")
    layers = [(6.2, 0.9, "Language runtime",
               "CPython  -  C / C++ / Fortran / Cython extensions", "#eceff4"),
              (5.1, 0.9, "Array core",
               "NumPy  (n-dimensional arrays, ufuncs, broadcasting)", "#dbe7f3"),
              (4.0, 0.9, "Core scientific libraries",
               "SciPy   -   pandas   -   Matplotlib   -   SymPy", "#d6e8da"),
              (2.9, 0.9, "Domain & analysis libraries",
               "scikit-learn - xarray - statsmodels - scikit-image - NetworkX", "#f1e7d2"),
              (1.8, 0.9, "Scale-out & acceleration",
               "Dask   -   array protocols / GPU back ends", "#efdfe4")]
    x0, w = 1.0, 10.0
    for y, h, title, items, color in layers:
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=color, ec="#5a6573", lw=1.3, zorder=3))
        ax.text(x0 + 0.25, y + h / 2, title, ha="left", va="center", fontsize=10.2,
                fontweight="bold", color="#283038", zorder=4)
        ax.text(x0 + w - 0.25, y + h / 2, items, ha="right", va="center",
                fontsize=9.4, color="#33424e", zorder=4)
    ax.add_patch(FancyBboxPatch((0.18, 1.8), 0.5, 5.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#3a4a5a", ec="#3a4a5a", zorder=2))
    ax.text(0.43, 4.3, "Interactive layer  -  IPython / Jupyter", ha="center", va="center",
            fontsize=10.4, color="white", rotation=90, fontweight="bold", zorder=5)
    for y in [3.05, 4.15, 5.25, 6.35]:
        ax.annotate("", xy=(6.0, y), xytext=(6.0, y - 0.18),
                    arrowprops=dict(arrowstyle="-|>", color="#5a6573", lw=1.2))
    ax.set_title("An interactive interface to an array-centric ecosystem", fontsize=13.5, pad=8)
    fig.tight_layout(); _save(fig, "fig03_ecosystem")

def fig_repro_bars():
    """Figure 7: the reproducibility funnel for public notebooks (Pimentel et al., 2019)."""
    stats = [("Attempted\n(valid Python,\nordered)", 100.0, "#88a0b8"),
             ("Finished execution\nwithout errors", 24.1, "#4a7aa7"),
             ("Reproduced the\nstored results", 4.0, "#2b5d86")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    vals = [s[1] for s in stats]
    bars = ax.bar(range(len(stats)), vals, color=[s[2] for s in stats], width=0.62,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold", color="#22303c")
    ax.set_xticks(range(len(stats))); ax.set_xticklabels([s[0] for s in stats], fontsize=9.4)
    ax.set_ylim(0, 108); ax.set_ylabel("Share of analysed notebooks (%)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_title("Reproducibility of public Jupyter notebooks", fontsize=12.5, pad=8)
    fig.tight_layout(); _save(fig, "fig04_repro")

def fig_capabilities():
    """Figure 8: software-engineering capabilities across four generations of tools."""
    gens = ["REPL /\nshell", "IPython\nshell", "Jupyter\nNotebook", "IDE-native\nnotebooks"]
    caps = ["Interactive exploration", "Rich inline output", "Literate narrative",
            "Language independence", "Reproducible environments", "Interactive debugging",
            "Automated testing", "Refactoring / language server",
            "First-class version control", "Real-time collaboration"]
    M = np.array([[2,2,2,2],[0,1,2,2],[0,0,2,2],[0,0,2,2],[0,1,1,2],
                  [1,1,1,2],[1,1,1,2],[0,0,1,2],[1,1,1,2],[0,0,1,2]])
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    cmap = mpl.colors.ListedColormap(["#f4f1ec", "#cfe0cf", "#3f7d56"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(gens))); ax.set_xticklabels(gens, fontsize=10)
    ax.set_yticks(range(len(caps))); ax.set_yticklabels(caps, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(gens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(caps), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    sym = {0: "-", 1: "partial", 2: "native"}
    for i in range(len(caps)):
        for j in range(len(gens)):
            v = M[i, j]
            ax.text(j, i, sym[v], ha="center", va="center", fontsize=8.6,
                    color="white" if v == 2 else "#3a3a3a",
                    fontweight="bold" if v == 2 else "normal")
    legend = [Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=12, label=l)
              for c, l in zip(["#f4f1ec", "#cfe0cf", "#3f7d56"],
                              ["absent", "partial / via add-ons", "native"])]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=9.5, title="capability", title_fontsize=9.5)
    ax.set_title("Software-engineering capabilities across generations", fontsize=12.5, pad=10)
    fig.tight_layout(); _save(fig, "fig05_capabilities")
