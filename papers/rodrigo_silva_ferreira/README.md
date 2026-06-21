# 25 Years of Interactive Scientific Computing — executable paper

An executable Jupyter notebook prepared for the **SciPy 2026 Proceedings**:

> **25 Years of Interactive Scientific Computing: From IPython and Jupyter to Open Source IDE-Native Notebooks**

The paper argues that the history of interactive scientific computing is the steady
reduction of friction among four activities — exploration, software engineering,
collaboration, and reproducible publication — and that *openness* (open source, open
standards, open governance, and executable publication) has been the decisive enabler
at every stage. It is itself an executable notebook: every figure and quantitative
claim is produced by the code it contains.

## Contents

| File | Description |
|------|-------------|
| `interactive_scientific_computing.ipynb` | The paper (executed, with figures and outputs embedded). |
| `build_notebook.py` | Generator that reproduces the notebook from source prose + code. |
| `references.bib` | Complete bibliography (61 entries), each verified against a primary source. |
| `figures/` | Publication-quality figures (`.png` and vector `.pdf`) written during execution. |
| `cooling.py` | A small module written *by* the notebook (Section 8 demonstration). |

## Reproducing it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib sympy scikit-learn \
            ipywidgets xarray dask jupyterlab nbconvert ipykernel
python -m ipykernel install --user --name scipy2026 --display-name "Python (scipy2026)"

# Option A — interactive:
jupyter lab interactive_scientific_computing.ipynb     # then: Run > Run All Cells

# Option B — headless, end to end:
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=scipy2026 \
    interactive_scientific_computing.ipynb

# Regenerate the notebook from source instead:
python build_notebook.py
```

The notebook runs top to bottom from a fresh kernel with no manual intervention.
It fixes a random seed in its setup cell, captures the exact package versions it ran
with (Appendix A), and emits the verified bibliography to `references.bib` (Appendix B).

## Verified against

Python 3.14 · NumPy 2.4 · SciPy 1.17 · pandas 3.0 · Matplotlib 3.11 ·
SymPy 1.14 · scikit-learn 1.9 · ipywidgets 8.1 · xarray · Dask.