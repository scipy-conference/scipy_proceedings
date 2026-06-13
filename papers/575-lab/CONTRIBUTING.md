# Contributing to the 575-Lab SciPy 2026 Paper

This guide is for collaborators on **"Opening the Black Box: Mechanistic
Interpretability for AI Agent Tool Selection Using Sparse Autoencoders"**
(`papers/575-lab/` in the `scipy_proceedings` repo). It covers setting up a
local environment and rendering the paper for review.

The paper is written and reviewed in **HTML** (via `myst start`). The PDF is
only built on acceptance, so day-to-day you only need the live HTML preview.

---

## 1. Prerequisites

Install these once:

- **Node.js** (v18+). MyST is a Node CLI under the hood — without it `myst`
  cannot render.
- **Python 3.10+**.
- **[uv](https://docs.astral.sh/uv/)** — fast Python env/dependency manager
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Plain `pip` works too if
  you prefer; commands for both are shown below.
- **git**.

---

## 2. Get the repo

We work on the **`2026`** branch.

```bash
git clone https://github.com/scipy-conference/scipy_proceedings.git
cd scipy_proceedings
git checkout 2026          # or: git checkout -b 2026 --track origin/2026
```

> Only edit files inside `papers/575-lab/`. Per SciPy proceedings policy, do
> not commit changes elsewhere in the repo.

---

## 3. Set up the environment

MyST ships as the `mystmd` Python package. From the **repo root**:

**With uv (recommended):**

```bash
uv venv                          # creates .venv/
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install mystmd pre-commit
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install mystmd pre-commit
```

Verify the install:

```bash
myst --version
```

### Enable the git pre-commit hooks

The repo uses pre-commit to keep YAML/JSON/Markdown tidy and to normalize the
BibTeX file (`bibtex-tidy`). Install the hooks once:

```bash
pre-commit install
```

They run automatically on `git commit`. To run them across the paper manually:

```bash
pre-commit run --all-files
```

---

## 4. Render / preview the paper

> [!IMPORTANT]
> **Run `myst start` from inside `papers/575-lab/`, NOT from the repo root.**
>
> The `site:` configuration lives in `papers/papers.yml` and our `myst.yml`
> inherits it via `extends: ../papers.yml` — that chain only resolves from the
> paper folder. Running from the repo root fails with:
>
> ```
> ⛔️ Cannot (re)load site config. No configuration file found with "site" property.
> Do you need to run myst init?
> ```
>
> If you see this, you are in the wrong directory. **Do not run `myst init`** —
> the config already exists.

Start the live preview:

```bash
cd papers/575-lab
myst start          # or: uv run myst start
```

Open the local URL it prints (usually <http://localhost:3000>). The preview
hot-reloads as you edit `main.md`, `myst.yml`, or the figures.

### Build the PDF (only needed on request / acceptance)

```bash
cd papers/575-lab
myst build --pdf
```

This uses the Typst `scipy` template and writes `full_text.pdf`. It downloads
the template on first run and takes ~a minute.

---

## 5. Project layout

| Path                | Purpose                                                      |
| ------------------- | ----------------------------------------------------------- |
| `main.md`           | The manuscript (MyST Markdown). This is what you edit.       |
| `myst.yml`          | Paper metadata: title, authors, abstract, keywords, exports.|
| `references.bib`    | Bibliography. Cite with `[@citekey]` in `main.md`.          |
| `images/`           | Figures referenced from `main.md` (e.g. `images/sae_architecture.png`). |
| `example/`          | Supporting/example material.                                |

---

## 6. Authoring tips

- **Figures:** put image files in `images/` and reference them with a
  `{figure}` directive plus a `:label:` so you can cross-reference via
  `[](#fig:yourlabel)`.
- **Citations:** add entries to `references.bib` and cite with `[@key]`. The
  `bibtex-tidy` pre-commit hook will reformat the file — commit the result.
- **Math:** MyST renders math with **KaTeX**. Some LaTeX macros are
  unsupported — e.g. use `\mathbb{1}` rather than `\mathbbm{1}` for the
  indicator function.
- **Metadata** (authors, affiliations, ORCIDs, abstract, keywords) lives in
  `myst.yml`, not in `main.md`. Keep the title in both in sync.

---

## 7. Submitting changes

1. Make focused commits inside `papers/575-lab/`.
2. Push to the shared branch; the paper is auto-built by GitHub Actions, which
   posts a preview link as a PR comment.
3. Confirm the online build matches your local `myst start` preview.

---

## Troubleshooting

- **"No configuration file found with 'site' property"** — you ran `myst` from
  the repo root. `cd papers/575-lab` first. See §4.
- **`myst: command not found`** — your virtualenv isn't active
  (`source .venv/bin/activate`) or `mystmd` isn't installed (§3).
- **Figures don't appear** — check the path is relative to the paper folder
  (`images/...`) and the file exists.
- **Warning about `banner.png` / `thumbnail.png`** — these are optional
  per-paper assets declared in `papers/papers.yml`; missing them only produces
  a harmless warning during preview.
- **Stray `_build/` at the repo root** — left over from accidentally running
  `myst` at the root. Safe to delete: `rm -rf _build` (the real build output
  lives in `papers/575-lab/_build/`).
