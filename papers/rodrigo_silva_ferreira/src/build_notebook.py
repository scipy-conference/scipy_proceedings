#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble the executable SciPy-2026 paper notebook from prose + code cells.

Run:  ./.venv/bin/python build_notebook.py
Produces: interactive_scientific_computing.ipynb  (un-executed)
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []
def md(text):
    CELLS.append(new_markdown_cell(text.strip("\n")))
def code(src):
    CELLS.append(new_code_cell(src.strip("\n")))

# ===========================================================================
#  SETUP CELL  (imports, publication style, figure helpers)
# ===========================================================================
SETUP = r'''
# --- Reproducible setup -----------------------------------------------------
# This single cell configures the environment, fixes the random seed, defines a
# restrained publication style, and provides the helper functions that draw the
# schematic figures used throughout the paper. Scientific demonstrations are
# kept inline, at the point in the narrative where they are discussed.
import os, sys, platform, hashlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

%matplotlib inline

os.makedirs("figures", exist_ok=True)
SEED = 20260619
rng = np.random.default_rng(SEED)        # one generator, threaded through the paper

# A color-blind-aware qualitative palette (after Paul Tol's "muted" scheme).
TOL = ["#332288", "#117733", "#CC6677", "#88CCEE", "#DDCC77",
       "#AA4499", "#44AA99", "#882255"]
INK, MUTED = "#1b1b1b", "#6b6b6b"

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#d8d8d8", "grid.linewidth": 0.7,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
set_style()

def _save(fig, name):
    fig.savefig(f"figures/{name}.png")
    fig.savefig(f"figures/{name}.pdf")   # vector copy for typesetting

print(f"Python {platform.python_version()} | NumPy {np.__version__} | "
      f"Matplotlib {mpl.__version__} | seed {SEED}")
'''

FIGFUNCS = r'''
# --- Figure helpers (schematic diagrams) ------------------------------------
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
    """Figure 3: the array-centric scientific Python stack under an interactive layer."""
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
    """Figure 5: the reproducibility funnel for public notebooks (Pimentel et al., 2019)."""
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
    """Figure 6: software-engineering capabilities across four generations of tools."""
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

print("figure helpers defined:", [n for n in dir() if n.startswith('fig_')])
'''

# ===========================================================================
#  TITLE & ABSTRACT
# ===========================================================================
md(r"""
# 25 Years of Interactive Scientific Computing: From IPython and Jupyter to Open Source IDE-Native Notebooks

*[Author Name]*<sup>1</sup>

<sup>1</sup>*[Affiliation, City, Country]* &nbsp;&middot;&nbsp; <sup>*</sup>*author and affiliation are placeholders to be completed for submission*

---

**Abstract.** Interactive computing has reshaped how scientists explore data, develop methods, and communicate results. This paper traces twenty-five years of that transformation, from the 2001 release of IPython through the formation of Project Jupyter to the contemporary movement that embeds notebooks inside full software-development environments. We argue that this history is best understood not as the story of a file format or a user interface, but as the steady reduction of friction between four activities that were once pursued with separate tools: exploratory analysis, software engineering, collaboration, and reproducible publication. Read this way, each generation inherited the strengths of the previous one while absorbing a capability that had previously lived elsewhere. We further argue that *openness*&mdash;of source code, of standards such as the Jupyter messaging protocol and the `nbformat` document schema, of community governance, and of the executable publication itself&mdash;has been the decisive enabler at every stage. After surveying the pre-Jupyter landscape and the scientific Python ecosystem that grew alongside these tools, we examine the well-documented limitations of traditional notebooks, the engineering practices that IDE-native notebooks restore, and the opportunities and risks introduced by large language models. The paper is itself an executable Jupyter notebook: every figure and quantitative claim is produced by the code it contains.

*Keywords:* interactive computing, Jupyter, IPython, reproducible research, scientific Python, literate programming, open science.
""")

md(r"""
> **About this document.** This is an *executable paper*. It runs top to bottom from a fresh kernel with no manual intervention. Schematic figures are drawn programmatically; scientific claims are demonstrated with short, self-contained computations using NumPy, SciPy, pandas, Matplotlib, SymPy, scikit-learn, and ipywidgets. The complete, source-verified bibliography is emitted to `references.bib` by the final cell. Setup instructions are given in the closing *Reproducibility* appendix.
""")

code(SETUP)
code(FIGFUNCS)

# ===========================================================================
#  1. INTRODUCTION
# ===========================================================================
md(r"""
## 1. Introduction

Computation has become a third mode of scientific inquiry, standing beside theory and experiment rather than merely serving them. Climate projections, genome assemblies, gravitational-wave detections, and the calibration of language models are all, at bottom, programs run over data. As the centre of gravity of research has shifted toward software, the tools used to write that software have stopped being incidental laboratory equipment and have become part of the scientific method itself. How a result is computed&mdash;and whether that computation can be inspected, rerun, and trusted by others&mdash;is now inseparable from whether the result is believed.

This shift has placed *reproducibility* at the heart of methodological debate. Donoho (2010) framed reproducible computational research as the discipline of publishing, alongside each figure and table, the complete software environment that generated it, reviving a tradition that Claerbout and Karrenbach (1992) and Buckheit and Donoho (1995) had pioneered in geophysics and signal processing. A decade later, an interdisciplinary consensus statement in *Science* called for code and data to be treated as first-class research artifacts (Stodden et al., 2016), and community guides distilled the accumulated experience into actionable practice (Wilson et al., 2014, 2017). The recurring lesson is that scientific software is rarely written the way production software is. It grows through *exploratory programming*: a tight loop of hypothesis, quick implementation, visualization, and revision, in which the program and the scientist's understanding co-evolve.

Interactivity changed research because it changed the unit of iteration. When a computation must be edited, recompiled, and resubmitted as a batch job, the cost of a single question is measured in minutes or hours, and the scientist learns to ask few, large questions. When the same computation answers in the time it takes to press a key, the cost of a question falls toward zero, and a different epistemic style becomes possible: a hypothesis is posed, tested against data, visualized, and revised dozens of times in an afternoon. The interactive loop does not merely speed up analysis; it widens the space of analyses a researcher will attempt, surfacing patterns and mistakes that a batch workflow would never have prompted anyone to look for. This is why cheap interactivity, far from being a mere convenience, reorganized the practice of computational science around exploration&mdash;and why the tools that host that exploration came to matter so much.

Exploration and communication, however, pull in opposite directions. Exploration rewards speed, mutable state, and disposable code; communication rewards clarity, structure, and permanence. Knuth's (1984) vision of *literate programming*&mdash;a program written as a human narrative, with source code woven into prose&mdash;was an early attempt to reconcile the two, prioritizing explanation to people over instruction to machines. Interactive computing environments inherited this tension and, over twenty-five years, progressively dissolved it. The thesis of this paper is that the history of interactive scientific computing is the history of *reducing friction* among four activities&mdash;exploration, engineering, collaboration, and reproducible publication&mdash;and that *openness* in its several forms has been the mechanism by which that friction was removed. Figure 1 previews the milestones along the way; the sections that follow trace them in turn.
""")

code(r"""
fig_timeline()
""")

md(r"""
**Figure 1.** A timeline of interactive scientific computing, organized into four overlapping eras. Each era inherits the capabilities of the previous one while integrating a concern&mdash;rich output, language independence, reproducible environments, software engineering&mdash;that had previously required separate tools. Dates are drawn from primary sources cited throughout the paper.
""")

# ===========================================================================
#  2. BEFORE JUPYTER
# ===========================================================================
md(r"""
## 2. Before Jupyter: REPLs, computer algebra, and literate programming

Interactivity in computing is older than the personal computer. Its lineage runs back to Lisp, whose read&ndash;eval&ndash;print loop (REPL) let a programmer type an expression and see its value immediately; McCarthy's language of 1958&ndash;1960 and the interactive Lisp systems built on it established a style of conversation with a running process that every later environment would imitate. For decades, however, the dominant mode of scientific computing was the opposite of conversational. Programs were compiled, submitted as batch jobs, and inspected only after they finished&mdash;a workflow well matched to expensive, shared mainframes but hostile to the rapid, speculative iteration that data analysis demands.

The first tools to make scientific computing genuinely interactive were, tellingly, commercial. MATLAB began in the late 1970s when Cleve Moler wrote a Fortran "matrix laboratory" so that students could call the LINPACK and EISPACK linear-algebra libraries without writing Fortran themselves; he used it during a 1979 sabbatical at Stanford, and in 1984 he and Jack Little founded MathWorks to commercialize it (Moler and Little, 2020). Computer-algebra systems followed a parallel path: Maple was conceived at the University of Waterloo in late 1980 as a lower-cost alternative to running Macsyma, and Mathematica was launched by Wolfram Research on 23 June 1988, introducing a *notebook* interface in which cells of evaluatable input and formatted output were interleaved on a single page. The notebook metaphor that Jupyter would later make ubiquitous was thus already two decades old. Statistical computing, meanwhile, was transformed by the S language developed at Bell Labs by Becker, Chambers, and colleagues (Becker and Chambers, 1984), whose open-source reimplementation as R by Ihaka and Gentleman (1996) would become the lingua franca of statistics.

These systems were powerful, but each was, in a precise sense, *closed*. Most were proprietary; each was wedded to a single language and a single vendor; and the documents they produced&mdash;where documents existed at all&mdash;were rarely separable from the application that created them. In parallel, a quieter intellectual movement was preparing the ground for something more open. Knuth (1984, 1992) argued in *literate programming* that a program should be written for human readers, as an essay whose narrative order need not match the compiler's, with code extracted ("tangled") and documentation typeset ("woven") from one source; his WEB system, built to document the TeX typesetting program, was the proof of concept. At nearly the same moment, Claerbout and Karrenbach (1992) coined the term *reproducible research* for the practice of bundling, with a publication, the electronic documents and code needed to regenerate every figure, and Buckheit and Donoho (1995) popularized the idea in the WaveLab toolbox, distributing the exact code that produced each plot in their wavelet papers. Literate programming supplied the *form*&mdash;narrative and code as one document&mdash;and reproducible research supplied the *purpose*. Interactive scientific computing in the Python era would inherit both, and add what the earlier systems lacked: an open-source implementation, an open document standard, and a community free to extend them.
""")

# ===========================================================================
#  3. IPYTHON
# ===========================================================================
md(r"""
## 3. IPython: an open, interactive shell for scientific Python

In 2001, Fernando P&eacute;rez was a graduate student in physics at the University of Colorado, Boulder, who wanted for Python the conveniences he admired in Mathematica and Maple: numbered input and output prompts, easy access to previous results, on-the-spot help and introspection, and a configurable, comfortable interactive session. What he wrote, as he later described it, was a 259-line "thesis procrastination project"&mdash;a script loaded at startup that gave the standard Python prompt a memory and a personality. Released as IPython 0.0.1 in December 2001, it soon merged ideas and code from two kindred efforts, Janko Hauser's IPP and Nathaniel Gray's LazyPython, combining a well-organized architecture with terse interactive syntax and rich, colored tracebacks (IPython development team; P&eacute;rez and Granger, 2007).

The design choices that followed proved durable. IPython introduced *magic commands*&mdash;line magics such as `%timeit` and `%matplotlib`, and cell magics prefixed with `%%`&mdash;that extend the language of the session without polluting the Python namespace. It made *introspection* immediate: appending `?` to any object prints its documentation and signature, and `??` shows its source. It cached inputs and outputs so that prior results remained addressable, and it offered tab completion and a configuration system that let users shape their environment. Crucially, and unlike the commercial systems that inspired it, IPython was open source from its first day, developed in the open by a widening circle of contributors and eventually sustained by foundation grants&mdash;an Alfred P. Sloan Foundation award in 2013, followed in 2015 by roughly six million dollars from the Helmsley, Moore, and Sloan foundations to expand the project (P&eacute;rez and Granger, 2007; Perkel, 2018). The small convenience of a better prompt had become community infrastructure.

The demonstration below uses one of those magics. The kernel executing this very notebook is IPython, so `%timeit` is available to us as we read; it quantifies the single most important habit in scientific Python&mdash;replacing explicit Python loops with vectorized array operations&mdash;a point we return to in Section 5.
""")

code(r"""
import numpy as np

x = rng.standard_normal(1_000_000)

# %timeit is an IPython "magic": it runs a statement many times and reports the best.
t_loop = %timeit -o -n 3 -r 3 [v * v for v in x]      # pure-Python loop
t_vec  = %timeit -o -n 50 -r 5 x * x                   # vectorized (NumPy)

print(f"\nVectorization speed-up on this machine: "
      f"{t_loop.best / t_vec.best:,.0f}x")
""")

md(r"""
The two lines compute the same thing&mdash;the element-wise square of a million numbers&mdash;but the vectorized form runs hundreds of times faster. Being able to *measure* that gap in a single keystroke, mid-thought, is the essence of what IPython added: not a new language, but a new tightness in the loop between writing code and understanding its behaviour.
""")

# ===========================================================================
#  4. PROJECT JUPYTER
# ===========================================================================
md(r"""
## 4. Project Jupyter: from a shell to an open protocol

IPython's most consequential idea was architectural rather than cosmetic. To support a graphical Qt console (2011) and then a browser-based notebook (first shipped in IPython 0.12 in December 2011), the project separated the *frontend* that a user interacts with from the *kernel* that executes code, with the two communicating over a well-specified network protocol. Once execution lived behind a protocol rather than inside the shell, the language of the kernel became an implementation detail. A frontend that could talk to a Python kernel could talk to any kernel.

This realization motivated "the Big Split." IPython 3.0, released in February 2015, was the last release to bundle the language-agnostic notebook with the Python-specific shell; IPython 4.0, on 11 August 2015, moved the frontend, the notebook document format, and the protocol into a new, language-neutral umbrella: Project Jupyter, whose name honours its three founding languages&mdash;Julia, Python, and R&mdash;and alludes to Galileo's notebooks recording the moons of Jupiter. The split was not a rebranding but a deliberate act of *opening*: it declared that the value of the notebook lay in a set of open standards that no single language community owned.

Figure 2 sketches the resulting architecture. A frontend and a kernel communicate over a ZeroMQ-based messaging protocol carried on five sockets&mdash;Shell, IOPub, stdin, Control, and Heartbeat&mdash;each with a defined role (Jupyter development team, *Messaging*). The Shell channel carries `execute_request` messages and their replies; the IOPub channel broadcasts side effects&mdash;standard output, rich `display_data`, and `execute_result`&mdash;to every connected client; stdin handles input prompts; Control carries high-priority interrupt and shutdown messages; and Heartbeat confirms liveness. Every message is signed with an HMAC digest and serialized, by default, as JSON, so that any process implementing the protocol can serve as a kernel.
""")

code(r"""
fig_architecture()
""")

md(r"""
**Figure 2.** The Jupyter client&ndash;kernel architecture. Decoupling the user-facing frontend from the language kernel behind an open, signed messaging protocol is what makes Jupyter language-agnostic: a *kernelspec* registers any conforming interpreter, and dozens of community kernels now exist. The notebook document (`nbformat`) is an independent JSON artifact that frontends read and write.
""")

md(r"""
The flow of a single execution is worth making concrete, because it explains both the power and the hazards of notebooks discussed later. Figure 3 shows the message exchange when a user runs one cell.
""")

md(r"""
```mermaid
sequenceDiagram
    autonumber
    participant F as Frontend
    participant S as Shell channel
    participant K as Kernel
    participant IO as IOPub channel
    F->>S: execute_request (source, msg_id)
    S->>K: deliver request
    K-->>IO: status = busy
    K-->>IO: execute_input (echo source, count = n)
    K-->>IO: stream / display_data / execute_result
    K-->>S: execute_reply (status = ok, count = n)
    K-->>IO: status = idle
```

**Figure 3.** The messaging sequence for executing one cell. The kernel maintains an execution counter `n` and a persistent namespace; the frontend records each cell's outputs and that counter in the notebook document. Because the counter reflects the *order of execution*, not the order of cells on the page, a saved notebook can encode a history that no top-to-bottom run would reproduce&mdash;the seed of the reproducibility problems examined in Section 7.
""")

md(r"""
Two open standards complete the picture. The *kernelspec* mechanism&mdash;a small `kernel.json` file in a known location&mdash;lets any language that speaks the protocol be installed as a kernel, which is why Jupyter today drives Python, R, Julia, C++, and many other languages from one interface. The `nbformat` schema defines the notebook itself as a JSON document with top-level `cells`, `metadata`, and version fields, where each cell is `code`, `markdown`, or `raw` and code cells carry their outputs inline (Jupyter development team, *nbformat*; Kluyver et al., 2016). Because the format is open and text-based, notebooks can be generated, transformed, executed, and rendered by an entire ecosystem of tools rather than a single application. On this foundation grew JupyterLab&mdash;unveiled in alpha at SciPy 2016 and released as 1.0 in June 2019&mdash;and JupyterHub for multi-user deployments, work recognized by the 2017 ACM Software System Award, presented in 2018 (Granger and P&eacute;rez, 2021). The recurring pattern is openness compounding: an open protocol enabled an open document format, which enabled an open ecosystem.
""")

# ===========================================================================
#  5. SCIENTIFIC PYTHON
# ===========================================================================
md(r"""
## 5. The scientific Python ecosystem, accelerated

Jupyter did not create the scientific Python ecosystem, but it gave that ecosystem an interactive interface of unprecedented reach, and the two grew together. The stack is layered (Figure 4). At its base sits CPython with compiled extensions; above that, NumPy provides the n-dimensional array, its universal functions, and the broadcasting rules that let array expressions stand in for explicit loops (Harris et al., 2020). NumPy traces its lineage to Numeric (Jim Hugunin, 1995) and numarray, which Travis Oliphant unified into NumPy, released as 1.0 in 2006. On the array core rest the libraries most analyses touch directly: SciPy for numerical algorithms (Virtanen et al., 2020), pandas for labeled, heterogeneous, tabular and time-series data (McKinney, 2010), Matplotlib for publication-quality graphics (Hunter, 2007), and SymPy for symbolic mathematics (Meurer et al., 2017). Domain libraries&mdash;scikit-learn for machine learning (Pedregosa et al., 2011), xarray for labeled N-dimensional data (Hoyer and Hamman, 2017), and others&mdash;build on these, while Dask extends the same array and dataframe interfaces to larger-than-memory and parallel computation (Rocklin, 2015).
""")

code(r"""
fig_ecosystem()
""")

md(r"""
**Figure 4.** The array-centric scientific Python stack. A shared array substrate (NumPy) lets independently developed libraries interoperate, and a common interactive layer (IPython/Jupyter) lets a researcher move fluidly among them. The ecosystem's coherence is a direct dividend of open, shared interfaces.
""")

md(r"""
The remainder of this section is a short tour of that stack in action, with each computation chosen to make a point rather than to decorate the page. We begin with the idiom that defines numerical Python&mdash;vectorization&mdash;already previewed in Section 3, now measured across problem sizes.
""")

code(r"""
import timeit
import numpy as np
import matplotlib.pyplot as plt

sizes = [10_000, 100_000, 1_000_000]
loop_t, vec_t = [], []
for n in sizes:
    a = rng.standard_normal(n)
    loop_t.append(timeit.timeit(lambda: [v * v for v in a], number=3) / 3)
    vec_t.append(timeit.timeit(lambda: a * a, number=50) / 50)

speedup = [l / v for l, v in zip(loop_t, vec_t)]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
idx = np.arange(len(sizes))
ax.bar(idx - 0.19, loop_t, 0.36, color=TOL[2], label="Python loop")
ax.bar(idx + 0.19, vec_t, 0.36, color=TOL[0], label="NumPy vectorized")
ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels([f"{n:,}" for n in sizes])
ax.set_xlabel("array length"); ax.set_ylabel("seconds per call (log scale)"); ax.legend()
ax.set_title(f"Vectorization: up to {max(speedup):.0f}x faster on this machine")
_save(fig, "demo_numpy")
print("speed-ups by size:", [f"{s:.0f}x" for s in speedup])
""")

md(r"""
**Figure 5.** Element-wise squaring via an explicit Python loop versus a NumPy array expression, timed across three problem sizes (note the logarithmic axis). The gap widens with size: pushing the loop into compiled, vectorized code is the foundation on which every higher library is built.

Real data is rarely an anonymous array; it carries labels, units, and time. pandas exists to keep those labels attached through every operation. The next cell synthesizes an hourly temperature record with a diurnal cycle and noise, then uses a `DatetimeIndex` to resample to daily means and to compute a rolling trend&mdash;operations that would be error-prone with bare arrays but are one line each with labeled data.
""")

code(r"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

hours = pd.date_range("2026-01-01", periods=24 * 30, freq="h")
t = np.arange(len(hours))
signal = (12 + 6 * np.sin(2 * np.pi * t / 24)           # diurnal cycle
          + 0.01 * t                                     # slow warming trend
          + rng.normal(0, 1.5, len(hours)))              # measurement noise
ts = pd.Series(signal, index=hours, name="temperature_C")

daily = ts.resample("D").mean()                          # label-aware aggregation
trend = ts.rolling("3D").mean()                          # 3-day rolling mean

fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.plot(ts.index, ts.values, color="#c7d2dc", lw=0.6, label="hourly readings")
ax.plot(trend.index, trend.values, color=TOL[0], lw=1.8, label="3-day rolling mean")
ax.scatter(daily.index, daily.values, color=TOL[2], s=14, zorder=5, label="daily mean")
ax.set_xlabel("date"); ax.set_ylabel("temperature (degC)"); ax.legend(loc="upper left")
ax.set_title("pandas: label- and time-aware aggregation")
_save(fig, "demo_pandas")
print(f"hourly points: {len(ts)} | daily means: {len(daily)} | "
      f"overall mean: {ts.mean():.2f} degC")
""")

md(r"""
**Figure 6.** A synthetic hourly temperature series aggregated to daily means and a three-day rolling average. Because the index is time-aware, `resample("D")` and `rolling("3D")` express scientific intent directly; the labels, not the programmer, track which observations belong together.

SciPy turns arrays into algorithms. A canonical example is recovering structure hidden by noise. Below, a signal composed of 7&nbsp;Hz and 23&nbsp;Hz tones is buried under noise of comparable amplitude; in the time domain it looks like little more than static, yet a discrete Fourier transform exposes both frequencies cleanly.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1.0, 1000, endpoint=False)
clean = np.sin(2 * np.pi * 7 * t) + 0.5 * np.sin(2 * np.pi * 23 * t)
noisy = clean + 0.8 * rng.standard_normal(t.size)

freqs = np.fft.rfftfreq(t.size, d=t[1] - t[0])
amp = np.abs(np.fft.rfft(noisy)) * 2 / t.size

fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
ax[0].plot(t, noisy, color="#b8c4d0", lw=0.8, label="measured (noisy)")
ax[0].plot(t, clean, color=TOL[0], lw=1.6, label="true signal")
ax[0].set(xlabel="time t (s)", ylabel="amplitude", xlim=(0, 0.5))
ax[0].legend(loc="upper right"); ax[0].set_title("(a) A noisy time series", fontsize=11)
ax[1].plot(freqs, amp, color=TOL[1], lw=1.3)
for f0 in (7, 23):
    ax[1].axvline(f0, color="#999", ls=(0, (3, 2)), lw=0.9)
    ax[1].text(f0 + 0.6, amp.max() * 0.92, f"{f0} Hz", fontsize=9, color="#555")
ax[1].set(xlabel="frequency (Hz)", ylabel="spectral amplitude", xlim=(0, 40))
ax[1].set_title("(b) ...with structure the eye cannot see", fontsize=11)
_save(fig, "demo_scipy")
peaks = np.sort(freqs[np.argsort(amp)[-2:]])
print(f"recovered dominant frequencies: {peaks[0]:.1f} Hz and {peaks[1]:.1f} Hz")
""")

md(r"""
**Figure 7.** A noisy two-tone signal (left) and its amplitude spectrum (right). SciPy's transform recovers the underlying 7&nbsp;Hz and 23&nbsp;Hz components from data in which they are visually imperceptible&mdash;the kind of result that is most persuasive when the reader can rerun it.

Not all mathematics is numerical. SymPy manipulates expressions exactly, and&mdash;importantly for the exploration-to-engineering arc of this paper&mdash;it can hand off an exact result to fast numerical code. Here we derive the Taylor series of $\sin(x)$ symbolically, then compile the exact polynomial with `lambdify` and compare it to the true function.
""")

code(r"""
import numpy as np, sympy as sp
import matplotlib.pyplot as plt

from IPython.display import display

x = sp.symbols("x")
series = sp.series(sp.sin(x), x, 0, 10).removeO()      # exact 9th-order Taylor poly
print("Taylor series of sin(x) to 9th order:")
display(series)                                         # rendered as typeset math

approx = sp.lambdify(x, series, "numpy")               # exact math -> fast NumPy fn
xs = np.linspace(-np.pi, np.pi, 400)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(xs, np.sin(xs), color=TOL[0], lw=2.0, label=r"$\sin(x)$")
ax.plot(xs, approx(xs), color=TOL[2], lw=1.6, ls="--",
        label="9th-order Taylor (SymPy)")
ax.set_xlabel("x"); ax.set_ylabel("value"); ax.legend(loc="upper center")
ax.set_title("SymPy: an exact derivation, compiled to numerical code")
_save(fig, "demo_sympy")
""")

md(r"""
**Figure 8.** A Taylor polynomial derived symbolically with SymPy and evaluated numerically through `lambdify`. The bridge from exact derivation to compiled numerical evaluation, inside one document, is exactly the kind of seam that interactive notebooks make seamless.

Finally, machine learning. scikit-learn provides a consistent estimator interface across hundreds of algorithms. The cell below standardizes the classic Iris measurements and projects them onto their first two principal components, separating the three species with two synthetic axes that together capture most of the variance.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = StandardScaler().fit_transform(data.data)
pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)
ev = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for k, name in enumerate(data.target_names):
    m = data.target == k
    ax.scatter(Z[m, 0], Z[m, 1], s=28, color=TOL[k], edgecolor="white",
               linewidth=0.5, label=name, alpha=0.9)
ax.set_xlabel(f"PC 1 ({ev[0]:.0f}% variance)")
ax.set_ylabel(f"PC 2 ({ev[1]:.0f}% variance)")
ax.legend(title="species", loc="lower right")
ax.set_title("scikit-learn: principal components of the Iris dataset")
_save(fig, "demo_sklearn")
print(f"variance captured by first two components: {ev[:2].sum():.0f}%")
""")

md(r"""
**Figure 9.** A principal-component projection of the standardized Iris measurements. Two components capture the great majority of the variance and nearly separate the species&mdash;a compact illustration of scikit-learn's uniform `fit`/`transform` interface.

The same array interface scales outward and labels upward. xarray attaches named dimensions and coordinates to N-dimensional arrays, so that operations read as scientific statements rather than axis arithmetic, while Dask preserves the NumPy interface for arrays too large for memory, building a task graph that is only evaluated on demand. The next cell shows both: a labeled `DataArray` reduced along a named dimension, and a lazy Dask array whose result matches NumPy exactly once computed.
""")

code(r"""
import numpy as np, xarray as xr
import dask.array as da

# xarray: an array that knows what its axes mean
temps = xr.DataArray(
    rng.normal(15, 5, size=(3, 4)),
    dims=("city", "month"),
    coords={"city": ["Oslo", "Nairobi", "Lima"],
            "month": ["Mar", "Apr", "May", "Jun"]},
    name="temperature_C",
)
print("Per-city mean temperature (named-dimension reduction):")
print(temps.mean(dim="month").to_series().round(2).to_string())

# Dask: the NumPy interface, evaluated lazily and in chunks
big = da.random.random((20_000, 20_000), chunks=(2_000, 2_000))   # ~3 GB if realized
lazy_mean = big.mean()                       # nothing computed yet -> a task graph
print(f"\nDask array: {big.shape} in {big.npartitions} chunks; "
      f"graph builds instantly, mean computes to {float(lazy_mean.compute()):.4f}")
""")

md(r"""
Taken together, these libraries share two properties that recur throughout this paper: they are open source, and they agree on interfaces. The array protocol lets independently authored packages compose; the estimator and labeled-array conventions let a researcher carry intuition from one library to the next. Jupyter's contribution was to place a single, interactive, narrative surface over all of them. The interactivity is not merely a convenience for the author. Because a notebook can embed live controls, a reader can manipulate a computation directly&mdash;turning a static figure into an instrument. The widget below lets one vary a sample size and a distribution and watch an empirical histogram converge toward its theoretical density.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ipywidgets import interact, IntSlider, Dropdown

DISTS = {"normal": stats.norm(0, 1),
         "exponential": stats.expon(scale=1.0),
         "uniform": stats.uniform(-2, 4)}

def explore(n=400, dist="normal"):
    d = DISTS[dist]
    samples = d.rvs(size=n, random_state=np.random.default_rng(SEED))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(samples, bins=40, density=True, color="#cdd9e3",
            edgecolor="white", label=f"{n} samples")
    grid = np.linspace(samples.min(), samples.max(), 300)
    ax.plot(grid, d.pdf(grid), color=TOL[2], lw=2.2, label="true density")
    ax.set_title(f"Empirical vs. theoretical density  (n = {n}, {dist})")
    ax.set_xlabel("value"); ax.set_ylabel("density"); ax.legend(loc="upper right")
    plt.show()

interact(explore,
         n=IntSlider(min=20, max=5000, step=20, value=400, description="samples"),
         dist=Dropdown(options=list(DISTS), value="normal", description="distribution"));
""")

md(r"""
**Figure 10 (interactive).** A live control over sample size and distribution. In a running kernel the sliders update the figure continuously; in a static rendering, the initial state is shown. Interactivity of this kind turns a reader into an experimenter, which is precisely why notebooks became the medium of choice for teaching, exploration, and&mdash;increasingly&mdash;publication (Perkel, 2018).
""")

# ===========================================================================
#  6. OPEN SCIENCE AND REPRODUCIBILITY
# ===========================================================================
md(r"""
## 6. Open science and reproducibility

If interactivity made notebooks the natural home for exploration, openness made them a candidate vehicle for reproducible publication. The two agendas converged in the mid-2010s. The FAIR Guiding Principles (Wilkinson et al., 2016) argued that scholarly outputs should be Findable, Accessible, Interoperable, and Reusable&mdash;principles later extended explicitly to research software (Barker et al., 2022). Practical guidance accumulated in parallel: the "ten simple rules" for reproducible computational research (Sandve et al., 2013) and the "best practices" and "good enough practices" series (Wilson et al., 2014, 2017) translated the aspiration into habits&mdash;version everything, automate the pipeline, record the environment.

Recording the environment proved to be the crux. A notebook captures code and narrative, but its results depend on a web of library versions and system libraries that the document itself does not pin. Two technologies addressed this gap. Containerization, popularized by Docker (Merkel, 2014), let an entire software environment be specified declaratively and rebuilt identically elsewhere. Binder (Project Jupyter et al., 2018) connected that capability to notebooks: given only a public Git repository with its dependencies declared, the `repo2docker` tool builds a container image and JupyterHub serves a live, executable session in the browser, so that a reader can run a paper's notebooks without installing anything. For the first time, "click here to reproduce my analysis" became a realistic offer rather than a rhetorical one.

The honest assessment is that these tools made reproducibility *achievable* but not *automatic*, and each has limits worth stating plainly. Public Binder sessions are ephemeral and resource-limited, unsuited to long or data-heavy computations. Container images pin software but not always the data, the hardware, or sources of nondeterminism such as random seeds, thread counts, and floating-point reduction order. Dependency specifications drift: an environment that omits exact versions may resolve differently months later, and upstream packages disappear. Reproducibility, in short, is a property a researcher must actively engineer&mdash;by pinning versions, fixing seeds (as this notebook does in its setup cell), separating data acquisition from analysis, and testing that the pipeline runs end to end. Openness lowers the cost of doing so dramatically, because open formats and open infrastructure let the community build the supporting tools; but it does not remove the responsibility. The next section quantifies how often, in practice, that responsibility goes unmet.
""")

# ===========================================================================
#  7. CHALLENGES OF TRADITIONAL NOTEBOOKS
# ===========================================================================
md(r"""
## 7. The challenges of traditional notebooks

The very features that make notebooks excellent for exploration create difficulties when the same documents are asked to serve as durable, shareable software. The root cause is *hidden state*. A kernel holds a persistent namespace, and cells may be executed in any order and any number of times. The visible document records the cells and their last outputs, but not the sequence of executions that produced them. A notebook can therefore look correct on screen while depending on a variable defined in a cell that has since been deleted, or on a definition that no longer exists in the order shown&mdash;a discrepancy invisible until someone restarts the kernel and runs from the top.

The empirical literature documents how widespread the consequences are. In the largest study of its kind, Pimentel et al. (2019) collected about 1.4 million Jupyter notebooks from roughly 264,000 GitHub repositories and attempted to execute the 863,878 that were valid Python with an unambiguous cell order. Only 24.1% ran to completion without raising an error, and only 4.0% reproduced the outputs stored in the document; failures were dominated by unmet dependencies (`ImportError`/`ModuleNotFoundError` accounted for about 29% of them). Symptoms of hidden state were pervasive: 36.4% of notebooks had been executed out of order, and 76.9% had at least one gap in their execution counters, indicating cells run and then removed. A follow-up that re-ran the analysis on a deduplicated corpus reached broadly similar conclusions (Pimentel et al., 2021). Figure 11 summarizes the funnel.
""")

code(r"""
fig_repro_bars()
""")

md(r"""
**Figure 11.** The reproducibility funnel from Pimentel et al. (2019), as a share of the 863,878 valid notebooks they attempted to execute. The collapse from "runs without error" to "reproduces the stored results" measures the gap between a notebook that looks finished and one that actually is.

The problems extend beyond execution. Because `.ipynb` files are JSON documents that embed outputs&mdash;including rendered images as base64 text&mdash;they interact badly with line-oriented version control: diffs are noisy and merges conflict over machine-generated metadata, discouraging the branching-and-review workflow that underpins collaborative software. Testing and debugging are awkward when the unit of code is a cell rather than an importable function, and a long notebook tends to accumulate globals and implicit ordering that resist refactoring as the analysis grows. Qualitative studies trace these frictions to a deeper tension. Rule et al. (2018), analyzing over a million notebooks alongside interviews, found a persistent conflict between *exploration* and *explanation*: notebooks accrete exploratory mess and are seldom cleaned up for sharing. Interview and survey studies catalogue the resulting pain points across the whole workflow&mdash;setup, exploration, refactoring, deployment, and collaboration (Chattopadhyay et al., 2020)&mdash;and characterize the informal versioning and narrative practices analysts improvise in their absence (Kery et al., 2018). A line of tools has tried to help from within: "code gathering" to extract the minimal cells behind a result (Head et al., 2019), and methods to restore a reproducible execution order automatically (Wang et al., 2020). These are valuable, but they treat symptoms. The structural response&mdash;bringing the notebook the engineering machinery that scripts and IDEs have long enjoyed&mdash;is the subject of the next section.
""")

# ===========================================================================
#  8. OPEN SOURCE IDE-NATIVE NOTEBOOKS
# ===========================================================================
md(r"""
## 8. Open source IDE-native notebooks

The most important recent development in interactive scientific computing is not a new notebook application but a change in where notebooks live. Increasingly, the notebook is a first-class citizen inside a full software-development environment, sitting beside source files, a debugger, a test runner, a version-control panel, and a terminal, and benefiting from the same language intelligence as ordinary code. This is less a product category than a *movement*&mdash;a convergence of the exploratory notebook and the engineering workbench&mdash;and, like the developments before it, it was made possible by open standards.

Two protocols are central. The Language Server Protocol (LSP), opened by Microsoft in 2016 in collaboration with Red Hat and Codenvy, is a JSON-RPC standard by which an editor obtains language intelligence&mdash;completion, hover documentation, go-to-definition, diagnostics, and refactorings&mdash;from a separate *language server* (Microsoft, *LSP*). Its significance is combinatorial: instead of every one of *M* editors implementing support for every one of *N* languages, each editor and each language implement the protocol once, turning an *M*&times;*N* problem into an *M*+*N* one. The Debug Adapter Protocol (DAP) does the same for debuggers (Microsoft, *DAP*). Jupyter adopted both ideas. The Jupyter debugger protocol (JEP&nbsp;47) wraps DAP messages inside Jupyter messages&mdash;`debug_request` and `debug_reply` on the Control channel, `debug_event` on IOPub&mdash;so that a frontend can drive breakpoints and variable inspection in a running kernel; the kernel advertises support in its `kernel_info_reply`, and implementations arrived through `debugpy` in `ipykernel` and through `xeus`-based kernels (Corlay and Mabille, 2020). Because these protocols are open, a notebook hosted in any conforming environment can offer the same completion, refactoring, linting, and step-through debugging that script authors take for granted.

Open standards also reconciled notebooks with version control. Jupytext (Wouts, 2018) represents a notebook as a plain-text script or Markdown file and keeps the two paired, so that the text form&mdash;free of outputs and volatile metadata&mdash;produces clean line diffs and reviewable pull requests, while the `.ipynb` retains rich outputs for reading. nbdime supplies content-aware diff and merge that understand notebook structure rather than raw JSON (Project Jupyter, *nbdime*). For execution and testing, papermill parameterizes and runs notebooks as pipeline stages (nteract, *papermill*); nbval turns a notebook into a regression test by comparing recomputed outputs against saved ones, and testbook unit-tests functions defined in notebooks directly (Fangohr et al., 2020); and jupyter-cache re-executes only the cells that changed (Executable Books Project). Perhaps the fullest expression of the convergence is nbdev (Howard and Husain, 2019), which treats notebooks as the source of truth for a Python package&mdash;generating modules, tests, and documentation from them&mdash;and thereby realizes Knuth's literate-programming ideal four decades on, with the fast.ai library itself built this way (Howard and Gugger, 2020).

Surrounding these notebook-aware tools is the ordinary furniture of a development environment, now within reach of the cell a scientist is editing. An integrated terminal makes shell commands&mdash;installing a package, inspecting a data file, launching a job&mdash;part of the same session rather than a switch to another window. Package and environment management, whether through virtual environments, conda, or lockfile-based resolvers, can be driven from the same interface that runs the notebook, so that the environment a result depends on is created and recorded where the result is produced. Linters and type checkers flag mistakes as code is typed, and the plain-text representations that jupytext maintains drop directly into continuous-integration pipelines, where nbval or papermill can execute a notebook on every commit and fail the build if a figure or a number changes unexpectedly. None of these capabilities is new to software engineering; what is new is that the scientist need not leave the exploratory document to use any of them. Figure 12 places these capabilities against earlier generations.
""")

code(r"""
fig_capabilities()
""")

md(r"""
**Figure 12.** A qualitative assessment by the authors of where ten software-engineering capabilities sit across four generations of interactive tools. The pattern, not any single cell, is the point: capabilities that classic notebooks could reach only through add-ons become native when the notebook is hosted inside a development environment&mdash;without sacrificing the interactive exploration of the first row.

The decisive question is whether this engineering rigor comes at the expense of the exploratory freedom that made notebooks valuable. We argue that it need not, because the integration is *additive*. A scientist still opens a cell, types an idea, and runs it; what changes is that the same idea can, without leaving the environment, be linted as it is written, extracted into a tested function, stepped through in a debugger when it misbehaves, committed with a readable diff, and packaged for reuse. The exploratory loop is preserved; the cliff that once separated a promising notebook from maintainable software is graded into a ramp. The demonstration below enacts the smallest version of this transition. We write a function to a source module&mdash;so that notebook and script literally share the same code&mdash;then import it back and test it against an analytic solution, all without leaving the document.
""")

code(r'''
%%writefile cooling.py
# A small module written from the notebook -- source and notebook now coexist.
import numpy as np

def newton_cooling(T0, T_env, k, t):
    """Analytic solution of Newton's law of cooling: T(t)=T_env+(T0-T_env)e^{-kt}."""
    return T_env + (T0 - T_env) * np.exp(-k * np.asarray(t, dtype=float))

def euler_step(T, T_env, k, dt):
    """One explicit-Euler step of dT/dt = -k (T - T_env)."""
    return T + dt * (-k * (T - T_env))
''')

code(r"""
import numpy as np
from cooling import newton_cooling, euler_step      # the module we just wrote

# A lightweight in-notebook test: numerical integration must approach the
# analytic solution as the step size shrinks. This is the habit that nbval,
# testbook, and CI build upon -- testable code, defined and checked in place.
def integrate(T0, T_env, k, t_end, dt):
    T, ts, Ts = T0, [0.0], [T0]
    while ts[-1] < t_end - 1e-9:
        T = euler_step(T, T_env, k, dt); ts.append(ts[-1] + dt); Ts.append(T)
    return np.array(ts), np.array(Ts)

t_end = 5.0
errors = {}
for dt in (0.5, 0.1, 0.01):
    ts, Ts = integrate(90.0, 20.0, 0.7, t_end, dt)
    errors[dt] = np.max(np.abs(Ts - newton_cooling(90.0, 20.0, 0.7, ts)))

assert errors[0.5] > errors[0.1] > errors[0.01], "error must shrink with dt"
assert errors[0.01] < 0.5, "fine step should track the analytic solution"
print("test passed -- max error by step size:",
      {dt: round(e, 3) for dt, e in errors.items()})
""")

md(r"""
That cell is mundane on purpose. A function lives in a real `.py` module *and* in the running notebook; a test asserts a mathematical property and would fail loudly if the implementation regressed; and the whole exchange happened inside the exploratory document. Multiply this by language-server completion, a visual debugger, and a version-control panel, and the notebook stops being a place from which code must eventually escape to "become serious," and becomes a place where serious software can be written from the start.
""")

# ===========================================================================
#  9. ARTIFICIAL INTELLIGENCE
# ===========================================================================
md(r"""
## 9. Artificial intelligence in the notebook

The newest force acting on interactive computing is the large language model. Code-generating models&mdash;beginning with systems such as Codex, a GPT model fine-tuned on public source code (Chen et al., 2021)&mdash;now power in-editor assistants that complete lines, draft whole cells, and increasingly converse about and edit a notebook in natural language. The appeal is the same friction reduction that has driven this entire history: the distance from intention to running code shrinks again. Empirical studies temper the enthusiasm with nuance. A controlled study of an LLM-based completion tool found that it did not always reduce task time and that programmers spent significant effort understanding and verifying suggestions (Vaithilingam et al., 2022). Observational work identified two distinct modes of use&mdash;*acceleration*, where the model fills in code the programmer already has in mind, and *exploration*, where the programmer uses the model to discover an approach&mdash;each with different needs and failure modes (Barke et al., 2023). Work specific to notebooks has begun to ask how assistants should be designed for a medium built around cells, outputs, and narrative (McNutt et al., 2023).

For science, the promise must be weighed against a sharpened version of the reproducibility and trust concerns of Sections 6 and 7. A suggestion that is fluent is not therefore correct: models hallucinate plausible APIs and subtly wrong numerics, and a cell accepted without scrutiny can introduce an error that no traceback reveals. More subtly, AI assistance complicates *provenance*. If part of an analysis was generated by a model from a prompt, the scientific record arguably should capture what was asked, what was produced, and what the human verified&mdash;information that today's tools rarely preserve. The notebook is unusually well placed to carry such a record, because it already interleaves intent, code, and result; and the research tradition of automatic provenance capture in notebooks (Pimentel et al., 2015, 2017) points to how generation events might be logged as data rather than lost. The constructive stance is that AI changes the *balance of effort* from writing toward verification, and that openness and transparency are the safeguards: open models and open logs make scrutiny possible, and the discipline of recording a verifiable provenance trail&mdash;of the kind the next cell sketches&mdash;keeps a human accountable for what the machine proposes.
""")

code(r'''
import hashlib, json, platform
import numpy as np

def provenance_stamp(inputs: dict, result, libs=("numpy", "scipy", "pandas")):
    """A minimal, deterministic provenance record: hash the inputs and the
    result together with the environment, so a claim can be re-verified later."""
    import importlib.metadata as m
    payload = {"inputs": inputs,
               "result_sha256": hashlib.sha256(np.asarray(result).tobytes()).hexdigest()[:16],
               "python": platform.python_version(),
               "libraries": {k: m.version(k) for k in libs},
               "seed": SEED}
    payload["record_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload

# Anchor an actual computation from this paper with a provenance stamp.
demo = (rng.standard_normal(10_000) ** 2).mean()
stamp = provenance_stamp({"operation": "mean_of_squares", "n": 10_000}, demo)
print(json.dumps(stamp, indent=2))
''')

md(r"""
The stamp is deliberately small, but it captures the idea: a result, the inputs that produced it, the exact library versions, and the random seed, bound together into a content-addressed record that another researcher&mdash;or a future version of oneself&mdash;can recompute and compare. In a world where some code is written by people and some by models, a verifiable trail of *what was computed, with what, and by whom* is the form of trust that openness uniquely enables. Human oversight remains the load-bearing element; the tooling exists to make that oversight cheap and auditable rather than to replace it.
""")

# ===========================================================================
#  10. FUTURE DIRECTIONS
# ===========================================================================
md(r"""
## 10. Future directions

If the last twenty-five years removed friction between exploration, engineering, collaboration, and publication, the next decade is likely to remove what remains at the boundaries between *places* of computation. The *executable publication*&mdash;of which this notebook is a modest instance&mdash;is maturing from novelty toward expectation, as journals and conferences increasingly accept, and occasionally require, artifacts that a reader can run. Real-time collaborative editing, already familiar from shared documents, is arriving in notebooks, promising the synchronous, multi-author analysis that science as a team sport demands. And the substrate itself is diversifying: WebAssembly runtimes now execute Python entirely in the browser, so that a notebook can run with zero installation on a reader's own machine, while cloud platforms offer the opposite trade-off&mdash;elastic compute and managed data next to the notebook. The likely future is not a winner among browser, local, and cloud execution but a continuum across them, with the same open document moving freely between contexts.

What will determine whether that future is healthy is, once more, openness. The portability just described depends on the notebook document, the messaging protocol, and the kernel interface remaining open standards that no single vendor controls, and on the governance of those standards remaining genuinely communal&mdash;the role that Project Jupyter's open processes have played, and that the broader ecosystem must continue to play as commercial interest intensifies. The same applies to the AI layer now settling over interactive computing: open models, open evaluation, and transparent provenance are what will let science treat machine-generated code with the same scrutiny it applies to any other instrument. The encouraging lesson of this history is that the field has repeatedly chosen openness at exactly the moments it mattered&mdash;P&eacute;rez releasing a procrastination script as free software, the deliberate "Big Split" that made the notebook language-neutral, the publication of `nbformat` and the messaging protocol as specifications anyone may implement. Interactive scientific computing became what it is not because any one tool was best, but because the community kept the standards open and the door open behind it. Sustaining that disposition&mdash;open source, open standards, open governance, and reproducible, executable scholarship&mdash;is the surest way to ensure that the next twenty-five years compound the gains of the last.
""")

# ===========================================================================
#  ACKNOWLEDGEMENTS + REFERENCES
# ===========================================================================
md(r"""
## Acknowledgements

This paper stands on the work of the IPython and Project Jupyter contributors, the maintainers of the scientific Python ecosystem, and the researchers whose empirical studies of notebooks are cited throughout. The factual claims herein were checked against primary sources during preparation; any remaining errors are the author's own.
""")

md(r"""
## References

The complete, source-verified bibliography is emitted to `references.bib` by the cell at the end of this notebook (BibTeX). The works cited above, in author&ndash;year form, are listed below.

Barker, M., Chue Hong, N. P., Katz, D. S., et al. (2022). Introducing the FAIR Principles for research software. *Scientific Data*, 9, 622.

Becker, R. A., & Chambers, J. M. (1984). *S: An Interactive Environment for Data Analysis and Graphics*. Wadsworth.

Buckheit, J. B., & Donoho, D. L. (1995). WaveLab and Reproducible Research. In *Wavelets and Statistics*, Lecture Notes in Statistics 103 (pp. 55&ndash;81). Springer.

Barke, S., James, M. B., & Polikarpova, N. (2023). Grounded Copilot: How Programmers Interact with Code-Generating Models. *Proc. ACM Program. Lang.*, 7(OOPSLA1), 85&ndash;111.

Chattopadhyay, S., Prasad, I., Henley, A. Z., Sarma, A., & Barik, T. (2020). What's Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities. *CHI 2020*.

Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.

Claerbout, J. F., & Karrenbach, M. (1992). Electronic Documents Give Reproducible Research a New Meaning. *SEG Technical Program Expanded Abstracts 1992* (pp. 601&ndash;604).

Corlay, S., & Mabille, J. (2020). *Jupyter Enhancement Proposal 47: Jupyter Debugger Protocol*.

Donoho, D. L. (2010). An invitation to reproducible computational research. *Biostatistics*, 11(3), 385&ndash;388.

Fangohr, H., Fauske, V., Kluyver, T., et al. (2020). Testing with Jupyter Notebooks: NoteBook VALidation (nbval) Plug-in for pytest. arXiv:2001.04808.

Granger, B. E., & P&eacute;rez, F. (2021). Jupyter: Thinking and Storytelling With Code and Data. *Computing in Science & Engineering*, 23(2), 7&ndash;14.

Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357&ndash;362.

Head, A., Hohman, F., Barik, T., Drucker, S. M., & DeLine, R. (2019). Managing Messes in Computational Notebooks. *CHI 2019*.

Howard, J., & Gugger, S. (2020). fastai: A Layered API for Deep Learning. *Information*, 11(2), 108.

Howard, J., & Husain, H. (2019). *nbdev: Create Delightful Software with Jupyter Notebooks*.

Hoyer, S., & Hamman, J. (2017). xarray: N-D Labeled Arrays and Datasets in Python. *Journal of Open Research Software*, 5(1), 10.

Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90&ndash;95.

Ihaka, R., & Gentleman, R. (1996). R: A Language for Data Analysis and Graphics. *Journal of Computational and Graphical Statistics*, 5(3), 299&ndash;314.

IPython Development Team. *IPython History* (project documentation).

Jupyter Development Team. *Messaging in Jupyter*; *The Notebook File Format (nbformat)* (project documentation).

Kery, M. B., Radensky, M., Arya, M., John, B. E., & Myers, B. A. (2018). The Story in the Notebook: Exploratory Data Science using a Literate Programming Tool. *CHI 2018*.

Kluyver, T., Ragan-Kelley, B., P&eacute;rez, F., et al. (2016). Jupyter Notebooks&mdash;a publishing format for reproducible computational workflows. *ELPUB 2016* (pp. 87&ndash;90). IOS Press.

Knuth, D. E. (1984). Literate Programming. *The Computer Journal*, 27(2), 97&ndash;111.

Knuth, D. E. (1992). *Literate Programming*. CSLI Lecture Notes 27. Center for the Study of Language and Information.

McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proc. 9th Python in Science Conf.* (pp. 56&ndash;61).

McNutt, A. M., Wang, C., DeLine, R. A., & Drucker, S. M. (2023). On the Design of AI-powered Code Assistants for Notebooks. *CHI 2023*.

Merkel, D. (2014). Docker: Lightweight Linux Containers for Consistent Development and Deployment. *Linux Journal*, 2014(239).

Meurer, A., Smith, C. P., Paprocki, M., et al. (2017). SymPy: Symbolic Computing in Python. *PeerJ Computer Science*, 3, e103.

Microsoft. *Language Server Protocol Specification* (2016); *Debug Adapter Protocol*.

Millman, K. J., & Aivazis, M. (2011). Python for Scientists and Engineers. *Computing in Science & Engineering*, 13(2), 9&ndash;12.

Moler, C., & Little, J. (2020). A History of MATLAB. *Proc. ACM Program. Lang.*, 4(HOPL), Article 81.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825&ndash;2830.

P&eacute;rez, F., & Granger, B. E. (2007). IPython: A System for Interactive Scientific Computing. *Computing in Science & Engineering*, 9(3), 21&ndash;29.

P&eacute;rez, F., Granger, B. E., & Hunter, J. D. (2011). Python: An Ecosystem for Scientific Computing. *Computing in Science & Engineering*, 13(2), 13&ndash;21.

Perkel, J. M. (2018). Why Jupyter is data scientists' computational notebook of choice. *Nature*, 563(7729), 145&ndash;146.

Pimentel, J. F., Braganholo, V., Murta, L., & Freire, J. (2015). Collecting and Analyzing Provenance on Interactive Notebooks: When IPython Meets noWorkflow. *TaPP 2015* (USENIX).

Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2017). noWorkflow: a Tool for Collecting, Analyzing, and Managing Provenance from Python Scripts. *Proc. VLDB Endowment*, 10(12), 1841&ndash;1844.

Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2019). A Large-Scale Study about Quality and Reproducibility of Jupyter Notebooks. *MSR 2019* (pp. 507&ndash;517).

Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2021). Understanding and Improving the Quality and Reproducibility of Jupyter Notebooks. *Empirical Software Engineering*, 26(4), Article 65.

Project Jupyter, Bussonnier, M., Forde, J., et al. (2018). Binder 2.0&mdash;Reproducible, Interactive, Sharable Environments for Science at Scale. *Proc. 17th Python in Science Conf.* (pp. 113&ndash;120).

Project Jupyter. *nbdime: Diffing and Merging of Jupyter Notebooks* (project documentation).

Rocklin, M. (2015). Dask: Parallel Computation with Blocked Algorithms and Task Scheduling. *Proc. 14th Python in Science Conf.* (pp. 126&ndash;132).

Rule, A., Tabard, A., & Hollan, J. D. (2018). Exploration and Explanation in Computational Notebooks. *CHI 2018*.

Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten Simple Rules for Reproducible Computational Research. *PLoS Computational Biology*, 9(10), e1003285.

Stodden, V., McNutt, M., Bailey, D. H., et al. (2016). Enhancing reproducibility for computational methods. *Science*, 354(6317), 1240&ndash;1241.

Vaithilingam, P., Zhang, T., & Glassman, E. L. (2022). Expectation vs. Experience: Evaluating the Usability of Code Generation Tools Powered by Large Language Models. *CHI EA 2022*.

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261&ndash;272.

Wang, J., Li, T.-Y., Li, L., & Zeller, A. (2020). Restoring Reproducibility of Jupyter Notebooks. *ICSE 2020 Companion* (pp. 288&ndash;289).

Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018.

Wilson, G., Aruliah, D. A., Brown, C. T., et al. (2014). Best Practices for Scientific Computing. *PLoS Biology*, 12(1), e1001745.

Wilson, G., Bryan, J., Cranston, K., et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510.

Wouts, M. (2018). *Jupytext: Jupyter Notebooks as Markdown Documents, Julia, Python or R Scripts*.
""")

# ===========================================================================
#  REPRODUCIBILITY APPENDIX
# ===========================================================================
md(r"""
## Appendix A. Reproducibility

This notebook was designed to run unmodified from a fresh kernel. To reproduce it locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib sympy scikit-learn \
            ipywidgets xarray dask jupyterlab nbconvert
python -m ipykernel install --user --name scipy2026
jupyter lab   # then: Run > Run All Cells   (or: jupyter nbconvert --to notebook --execute)
```

The exact versions used to execute this document are captured below, directly from the running interpreter, so that the environment is part of the published record (cf. Sections 6 and 9).
""")

code(r"""
import platform
import importlib.metadata as m
import pandas as pd

pkgs = ["numpy", "scipy", "pandas", "matplotlib", "sympy",
        "scikit-learn", "ipywidgets", "xarray", "dask",
        "ipykernel", "nbformat", "nbconvert"]
rows = [{"package": p, "version": m.version(p)} for p in pkgs]
env = pd.DataFrame(rows)
print(f"Python {platform.python_version()} on {platform.system()} "
      f"({platform.machine()})\n")
print(env.to_string(index=False))
""")

md(r"""
## Appendix B. The verified bibliography

The next cell writes the complete BibTeX bibliography to `references.bib`. Every entry was verified against a primary source (publisher page, DOI resolver, or official project documentation) during preparation of this paper.
""")

# embed the verified .bib generated alongside this script (avoid f-string brace clashes)
with open("references.bib", encoding="utf-8") as _f:
    _BIB = _f.read()
assert "'''" not in _BIB, "bib content would break the embedded raw string"
BIBCELL = ("bibtex = r'''\n" + _BIB + "\n'''\n"
           "with open('references.bib', 'w', encoding='utf-8') as f:\n"
           "    f.write(bibtex)\n"
           "n = bibtex.count('@')\n"
           "print(f'wrote references.bib with {n} verified entries')\n")
code(BIBCELL)

md(r"""
---

*This executable paper was prepared as a single Jupyter notebook. Every figure is generated programmatically, every quantitative claim is computed from code in the document, and the bibliography is verified against primary sources. It is intended as a demonstration, in form as well as content, of the argument it makes: that open, interactive, reproducible computing is now a natural medium for scientific communication.*
""")

# ===========================================================================
#  WRITE NOTEBOOK
# ===========================================================================
if __name__ == "__main__":
    nb = new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (scipy2026)",
                       "language": "python", "name": "scipy2026"},
        "language_info": {"name": "python"},
        "title": "25 Years of Interactive Scientific Computing",
        "authors": [{"name": "[Author Name]"}],
    }
    out = "interactive_scientific_computing.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    n_md = sum(c["cell_type"] == "markdown" for c in CELLS)
    n_code = sum(c["cell_type"] == "code" for c in CELLS)
    print(f"wrote {out}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
