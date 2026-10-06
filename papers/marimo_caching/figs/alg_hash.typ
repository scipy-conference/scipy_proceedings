// Standalone render of the key-derivation listing, rasterized to PNG
// and composed into figs/fig1_dispatch.png (see compose_fig1.py).
// Mirrors the styling the paper previously applied via @preview/algo.
#import "@preview/algo:0.3.6": code

#set page(width: auto, height: auto, margin: 2pt, fill: none)

#code(
  fill: rgb(99%, 99%, 99%),
  stroke: 0.6pt + rgb(70%, 70%, 70%),
  radius: 2pt,
  inset: 7pt,
  row-gutter: 6.5pt,
  main-text-styles: (size: 8.5pt),
  line-number-styles: (size: 7.5pt, fill: rgb(45%, 45%, 45%)),
)[#raw(lang: "py", block: false, "def derive_key(cell, graph, scope) -> HashKey:\n    refs = ScopedVisitor(cell.module).refs         # static analysis\n    refs = normalize_scope(refs, scope)            # UI/state -> values\n    h, kind = sha256(), \"Pure\"                     # no refs: pure cell\n\n    if refs:                                       # content-address first\n        kind = \"ContentAddressed\"\n        refs, serial = content_serialize(refs, scope)\n        for ref in sorted(serial):\n            h.update(serial[ref])                  # buffer-protocol view\n\n    if refs:                                       # substitute producers\n        kind = \"ExecutionPath\"\n        for parent in producing_cells(refs, graph):\n            h.update(derive_key(parent, graph, scope).digest)\n\n    if refs - scoped_refs:                         # same-cell context\n        kind = \"ContextExecutionPath\"\n        h.update(hash_module(cell.context))\n\n    h.update(hash_module(compile(cell.module)))    # bytecode, not source\n    return HashKey(h.digest(), kind)")]
