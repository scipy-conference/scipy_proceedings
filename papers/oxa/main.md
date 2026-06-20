---
# Ensure that this title is the same as the one in `myst.yml`
title: 'OXA: An Open Exchange Architecture for Modular and Composable Scientific Content'
abstract: |
  Scientific communication remains trapped in a paper-shaped container. The
  dominant unit of exchange — a static, narrative PDF backed by Journal Article
  Tag Suite (JATS) XML — was designed to describe finished, print-shaped
  documents. It is poorly suited to modern research, in which the data, code,
  protocols, computational notebooks, and interactive results that constitute
  the actual evidence are relegated to hard-to-access supplementary files. This
  paper introduces the Open Exchange Architecture (OXA), an emerging,
  community-driven specification that represents scientific documents and their
  components as structured, typed, and uniquely addressable JSON objects. OXA is
  designed to enable exchange, interoperability, and long-term preservation
  while remaining compatible with modern web and data standards.
  We describe the technical architecture of OXA — a typed
  node model with `children` arrays inspired by unified.js and the Pandoc
  abstract syntax tree (AST) — and trace its provenance through JATS, the Stencila
  schema for executable documents, Curvenote's connected publishing model, and the
  document pipelines of Pandoc, MyST Markdown, and Quarto.
  We report on the first large-scale implementation of OXA: a
  Curvenote–openRxiv partnership that translated the bioRxiv JATS archive into
  an early version of OXA to power the openRxiv Labs "Curvenote Reader"
  experience. OXA is stewarded by the Continuous Science Foundation and governed
  through an open Request for Comments (RFC) process.
---
