"""Library code for the paper notebook.

Exposes two names to the figure cells in main.md:

- ``compute``: benchmark primitives, plotting helpers, and the
  ``claim(condition, label, **values)`` assertion helper.
- ``save_fig``: writes a matplotlib figure to ``figs/<name>.{svg,png}``.
"""
from . import compute
from .plotting import save_fig

__all__ = ["compute", "save_fig"]
