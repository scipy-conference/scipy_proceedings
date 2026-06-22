# Docling for Multimodal Retrieval

SciPy 2026 Conference Paper

## Overview

This paper presents a multimodal document parsing framework built on Docling that preserves document structure and multimodal elements in RAG systems through specialized chunking strategies for text, images, and tables.

## Authors

- Mahima Arora (Red Hat India Pvt. Ltd.)
- Aarti Jha (Red Hat India Pvt. Ltd.)

## Building the Paper

This paper uses MyST Markdown format for the SciPy 2026 proceedings.

### Requirements

- MyST-MD (`myst` CLI tool)

### Local Preview

```bash
myst start
```

This will open an interactive HTML view of the paper.

### Build PDF

```bash
myst build --pdf
```

## File Structure

- `main.md` - Main paper content in MyST Markdown
- `myst.yml` - Configuration and metadata
- `mybib.bib` - Bibliography
- `*.png`, `*.jpg` - Figures and images

## License

TBD
