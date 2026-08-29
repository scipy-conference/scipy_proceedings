"""Tests for `utils.py` — citation hygiene + bib→yaml round-trip.

  - validate_citations(): orphan + missing detection
  - bib_to_yaml(): round-trip equality by key

These tests fail with ImportError until Phase C implements utils.py.
"""
import textwrap
from pathlib import Path

import pytest


# ---------- helpers: synthetic fixtures written to tmp_path ----------

GOOD_BIB = textwrap.dedent(r"""
@article{foo2024,
  author = {Foo, Alice},
  title = {On foo},
  journal = {J. Foo},
  year = {2024},
}

@book{bar2023,
  author = {Bar, Bob},
  title = {The bar book},
  publisher = {Bar Press},
  year = {2023},
}
""").strip()


def _write_notebook(path: Path, cells_text: list[str]) -> None:
    """Write a minimal Jupyter notebook with the given markdown cells."""
    import json
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": text,
            }
            for text in cells_text
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


# ---------- validate_citations ----------

def test_validate_citations_pass(tmp_path):
    """Known-good (bib, notebook) pair returns no missing, no orphan."""
    from utils import validate_citations  # noqa: PLC0415

    bib = tmp_path / "good.bib"
    bib.write_text(GOOD_BIB)
    nb = tmp_path / "good.ipynb"
    _write_notebook(
        nb,
        ["Foo was shown by [@foo2024]. Bar was earlier [@bar2023]."],
    )

    result = validate_citations(str(bib), str(nb))
    assert result.get("missing", []) == [], f"unexpected missing: {result}"
    assert result.get("orphan", []) == [], f"unexpected orphan: {result}"


def test_validate_citations_orphan(tmp_path):
    """A bib entry with no in-prose citation is flagged as orphan."""
    from utils import validate_citations  # noqa: PLC0415

    bib = tmp_path / "orphan.bib"
    bib.write_text(GOOD_BIB)  # has foo2024 + bar2023
    nb = tmp_path / "orphan.ipynb"
    _write_notebook(nb, ["Foo only [@foo2024]."])  # bar2023 unused

    result = validate_citations(str(bib), str(nb))
    assert "bar2023" in result.get("orphan", []), (
        f"expected bar2023 in orphan list, got {result}"
    )


def test_validate_citations_missing(tmp_path):
    """A notebook citation with no matching bib entry is flagged as missing."""
    from utils import validate_citations  # noqa: PLC0415

    bib = tmp_path / "missing.bib"
    bib.write_text(GOOD_BIB)
    nb = tmp_path / "missing.ipynb"
    _write_notebook(
        nb,
        ["Citing [@foo2024] and a typo [@bxr2023]."],
    )

    result = validate_citations(str(bib), str(nb))
    assert "bxr2023" in result.get("missing", []), (
        f"expected bxr2023 in missing list, got {result}"
    )


# ---------- bib_to_yaml ----------

def test_bib_to_yaml_roundtrip(tmp_path):
    """bib → yaml → re-parse: entries match by key, with key fields preserved."""
    import yaml

    from utils import bib_to_yaml  # noqa: PLC0415

    bib = tmp_path / "rt.bib"
    bib.write_text(GOOD_BIB)

    yaml_text = bib_to_yaml(str(bib))
    parsed = yaml.safe_load(yaml_text)

    # Normalise to a dict keyed by entry id, whether parsed is a list or dict.
    if isinstance(parsed, list):
        by_key = {entry["id"]: entry for entry in parsed}
    else:
        by_key = parsed

    assert "foo2024" in by_key, "foo2024 missing from yaml view"
    assert "bar2023" in by_key, "bar2023 missing from yaml view"

    foo = by_key["foo2024"]
    assert foo.get("title", "").lower().startswith("on foo")
    # Author may be a string or a list-of-author-dicts; either is fine.
    author = foo.get("author")
    assert author, "foo2024 lost its author field"
