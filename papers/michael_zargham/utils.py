"""Author-side utilities. Keep narrow; only add what the paper actually needs.

Currently:

* ``validate_citations(bib_path, nb_path)`` — flag orphan bib entries
  (defined but never cited) and missing citations (cited but no bib entry)
  by walking the markdown cells of a Jupyter notebook.

* ``bib_to_yaml(bib_path)`` — render a ``.bib`` file as a YAML view that
  MyST / CSL-JSON consumers can read. The ``.bib`` remains the source of
  truth (it's what the author reads when validating citations); the YAML
  is generated on demand for tooling.

The bib parser is intentionally small — a regex walk over balanced
braces — so we don't take on a heavyweight dependency for two functions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# BibTeX → structured entries
# ---------------------------------------------------------------------------


_ENTRY_HEADER_RE = re.compile(
    r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,",
    re.IGNORECASE,
)


def _split_balanced_brace_block(text: str, start: int) -> tuple[str, int]:
    """Return (block_body, end_index_inclusive_of_closing_brace).

    Assumes ``text[start]`` is the opening '{' of the entry's body.
    Walks character by character honouring brace depth so values like
    ``title = {A {nested} title}`` parse correctly.
    """
    assert text[start] == "{", f"expected '{{' at position {start}"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
    raise ValueError("unbalanced braces in bib entry")


_FIELD_RE = re.compile(
    # field = {value}   or   field = "value"   or   field = 1234
    r"(?P<name>\w+)\s*=\s*(?P<value>\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\"[^\"]*\"|[^,\n}]+)",
)


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in _FIELD_RE.finditer(body):
        name = m.group("name").lower()
        raw = m.group("value").strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        elif raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        fields[name] = raw.strip()
    return fields


class BibEntry(TypedDict, total=False):
    id: str
    type: str
    # plus whatever fields the entry had (title, author, year, ...)


def parse_bib(bib_path: str | Path) -> list[BibEntry]:
    """Parse a ``.bib`` file into a list of dicts.

    Each dict has ``id`` (the cite-key), ``type`` (article/book/...), and
    any other fields the entry declared (title, author, year, doi, ...).
    Lines starting with ``%`` and BibTeX-style comments are ignored.
    """
    text = Path(bib_path).read_text(encoding="utf-8")
    # Strip whole-line comments
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )

    entries: list[BibEntry] = []
    pos = 0
    while True:
        m = _ENTRY_HEADER_RE.search(text, pos)
        if m is None:
            break
        brace_start = text.find("{", m.end() - 1)  # just past the comma
        # Re-locate the entry-body opening brace (the one right after the key,
        # i.e. the one we already passed). The regex consumed `@type{key,`,
        # so the body continues without a fresh '{'. We treat what follows
        # the comma as field-list text up to a matching closing brace.
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise ValueError(f"unbalanced braces near cite-key {m.group('key')}")
        body = text[m.end() : i]
        fields = _parse_fields(body)
        entry: BibEntry = {"id": m.group("key"), "type": m.group("type").lower()}
        entry.update(fields)  # type: ignore[typeddict-item]
        entries.append(entry)
        pos = i + 1
        _ = brace_start  # silence linter; not used
    return entries


# ---------------------------------------------------------------------------
# Citation extraction from Jupyter notebooks
# ---------------------------------------------------------------------------


# MyST / pandoc style: `[@key]`, `[@key; @other]`, `[@key, p. 7]`.
# Keys allow letters, digits, underscores, colons, hyphens, dots, slashes
# (the last few support `doi:10.xxxx/yyy` style keys).
_CITE_RE = re.compile(r"\[@([A-Za-z][A-Za-z0-9_:./\-]*)(?:[,;\s][^\]]*)?\]")
_CITE_MULTI_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:./\-]*)")


def _cited_keys_in_text(text: str) -> set[str]:
    """Return all cite-keys mentioned in bracketed `[@...]` form.

    Multi-cite brackets like ``[@a; @b; @c]`` yield {a, b, c}. Bare
    ``@key`` outside of brackets is intentionally ignored to keep
    e-mail addresses, @-mentions, and notebook narrative pronouns
    from causing false positives.
    """
    keys: set[str] = set()
    for m in re.finditer(r"\[(@[^\]]+)\]", text):
        inner = m.group(1)
        for k in _CITE_MULTI_RE.finditer(inner):
            keys.add(k.group(1))
    return keys


def _markdown_cells_text(nb_path: str | Path) -> str:
    """Concatenate the markdown source of every cell in a Jupyter notebook."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    out: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            out.append("".join(src))
        else:
            out.append(src)
    return "\n".join(out)


class CitationReport(TypedDict):
    missing: list[str]
    orphan: list[str]


def validate_citations(bib_path: str | Path, nb_path: str | Path) -> CitationReport:
    """Return missing + orphan cite-keys for a ``(bib, notebook)`` pair.

    * ``missing``: cite-keys used in the notebook with no matching bib entry.
    * ``orphan``: bib entries that no notebook prose actually cites.

    Both lists are sorted. An empty ``CitationReport`` means the notebook
    and bib are in sync.
    """
    entries = parse_bib(bib_path)
    bib_keys = {e["id"] for e in entries}
    cited = _cited_keys_in_text(_markdown_cells_text(nb_path))

    missing = sorted(cited - bib_keys)
    orphan = sorted(bib_keys - cited)
    return {"missing": missing, "orphan": orphan}


# ---------------------------------------------------------------------------
# .bib → YAML view
# ---------------------------------------------------------------------------


def _yaml_quote(value: str) -> str:
    """Quote a YAML scalar conservatively.

    Most bib values are safe inline, but a few characters (``:``, ``#``,
    leading/trailing whitespace) demand quoting. We always emit
    double-quoted strings with backslash-escaped quotes and backslashes —
    overkill for the common case but unconditionally correct.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def bib_to_yaml(bib_path: str | Path) -> str:
    """Render a ``.bib`` file as a YAML list of entries.

    Format::

        - id: foo2024
          type: article
          title: "On foo"
          author: "Foo, Alice"
          year: "2024"

    This is the structure ``yaml.safe_load`` will return as a list of
    dicts. Callers that prefer a key-indexed dict can do
    ``{e['id']: e for e in yaml.safe_load(text)}``.
    """
    entries = parse_bib(bib_path)
    lines: list[str] = []
    for e in entries:
        # First line of each entry uses `-` to mark a new list item.
        lines.append(f"- id: {_yaml_quote(e['id'])}")
        # Reorder so type comes second, then the rest of the fields
        # alphabetically — deterministic output is easier to diff.
        type_value = e.get("type", "misc")
        lines.append(f"  type: {_yaml_quote(type_value)}")
        for name in sorted(k for k in e.keys() if k not in {"id", "type"}):
            lines.append(f"  {name}: {_yaml_quote(e[name])}")  # type: ignore[literal-required]
    return "\n".join(lines) + "\n"


__all__ = [
    "parse_bib",
    "validate_citations",
    "bib_to_yaml",
    "BibEntry",
    "CitationReport",
]
