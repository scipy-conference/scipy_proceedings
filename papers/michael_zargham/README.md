# Derivations, Not Just Simulations

Source folder for the SciPy 2026 Proceedings submission *"Derivations, Not Just Simulations: Teaching Applied Mathematics with Scientific Python"* by Michael Zargham (Dynamical Systems Group).

## Architectural choices

**The article is a Jupyter notebook** (`paper.ipynb`) configured as MyST's primary export source — the same notebook-primary architecture as Kadie's accepted 2025 SciPy paper on solvable and unsolvable equations [@kadie2025solvable]. We chose this because the paper's argument *is* the executable derivation chain: SymPy cells, lambdified RHS, matplotlib figures, and the prose that walks the reader through them are inseparable. The notebook IS the article rather than a supplement.

**The paper sits in a mathematics-education lineage.** Specifically: David Tall's three-worlds account of mathematical learning [@tall2008transition; @tall2013] — *conceptual-embodied* (figures, intuition), *operational-symbolic* (procedures, computation), *axiomatic-formal* (symbol-as-object manipulation) — together with Bruno Buchberger's white/black-box principle for teaching with computer algebra [@buchberger1990] and Drijvers/Artigue's *glass-box* refinement [@artigue2002; @tall1991amtcomputer]. SymPy is the SciPy-ecosystem tool that makes that lineage *executable* rather than only citable; the article makes that case with one worked exemplar.

**The exemplar is a classic differential game, not novel mathematics.** We picked Isaacs's 1951 *homicidal chauffeur* — a canonical pursuit-evasion problem worked out across the calculus-of-variations and optimal-control literature [@isaacs1951; @isaacs1965; @merz1971; @bardi1999pursuit; @pachter2019hc] — precisely because the underlying mathematics is settled and well-cited. We do not claim a new theorem; we claim a new *executable presentation* of a seventy-year-old derivation chain. Pursuit-evasion is also pedagogically accessible: it is essentially the children's game of *tag*, so every reader has embodied experience to anchor the formalism on.

## What's in this folder

- **`paper.ipynb`** — the article (Jupyter notebook; primary published artifact)
- **`supporting_material.ipynb`** — supporting material, not part of the counted article: the full notation reference (Appendix A) and the verification audit (Appendix B — test catalog, live in-notebook pytest run, symbolic-vs-numerical conservation cross-check)
- **`derivations.py`** — SymPy: body-frame reduction, Hamiltonian, optimal controls, costate ODE
- **`numerics.py`** — SciPy: `lambdify` wrappers, backward characteristic integration, value-function grid with on-disk cache
- **`plots.py`** — matplotlib: motivation chase, lab-frame geometry, coordinate-progression, 6-panel `dispersal_crossing`, V*(x) heat map, and supporting figures kept in module + tested
- **`utils.py`** — citation hygiene helpers (`validate_citations`, `bib_to_yaml`)
- **`tests/`** — pytest assertions: 6 derivations + 15 numerics + 10 plot smoke + 4 utils (35 total; runs in ~25 s — T15 sweeps three backward-characteristic counts to pin the §8 barrier-singular-surface invariance)
- **`mybib.bib`** — bibliography (29 entries; the human-validatable source of truth — MyST consumes this directly)
- **`myst.yml`** — MyST/Curvenote project metadata
- **`pyproject.toml`** + **`uv.lock`** — reproducible Python environment (uv-managed)
- **`banner.png`** + **`thumbnail.png`** — venue-required imagery

## Reproducing this paper

This folder ships a [uv](https://docs.astral.sh/uv/)-managed Python environment. uv (from Astral) is the modern Python project manager that resolves dependencies once and pins the lockfile; `uv sync` reproduces the same `.venv` the author tested with on any machine. With `uv` installed:

```bash
cd papers/michael_zargham
uv sync                          # creates .venv with pinned versions from uv.lock
uv run pytest tests/             # runs 35 tests (derivations + numerics + plots + utils)
uv run jupyter lab paper.ipynb   # opens the live notebook
```

`uv.lock` makes the dependency resolution byte-level reproducible.

**The verification audit (Appendix B) in `supporting_material.ipynb` runs the full test suite live in-notebook** and reports the symbolic-vs-numerical cross-checks for the conservation invariants the body relies on (`d/dt‖p‖² = 0` proven by `sp.simplify` in §8 is verified numerically along every integrated characteristic; visual inspection of the §8 heat map is the third Tall-world audit). The reviewer who runs `uv run pytest tests/` and then opens `supporting_material.ipynb` gets the same checks executed in two forms.

## Notes for reviewers

- The mathematics is credited to its original sources (Isaacs 1965; Merz 1971; Bardi–Falcone–Soravia 1999; Pachter–Coates 2019). What we claim is the **implementation** — the SymPy derivations and SciPy numerics in the modules above, made inspectable in `paper.ipynb`.
- The earlier (longer, broader) notebook this paper draws on is public at <https://github.com/mzargham/hc-marimo> and runs live at <https://mzargham.github.io/hc-marimo/>. The SciPy paper here focuses on one concept (the value function / reachable set); the live demo treats nine other concepts in the same application domain.
- All citations are MyST `[@key]` form against `mybib.bib`. `utils.validate_citations()` checks `paper.ipynb` ↔ `mybib.bib` consistency.
