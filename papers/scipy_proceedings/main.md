---
title: 'SciPy Proceedings: An Exemplar for Publishing Computational Open Science'
abstract: |
  The SciPy Proceedings have long served as a cornerstone of scholarly communication within the scientific Python ecosystem. In 2024, the publication process underwent a significant transformation, adopting a new open-source infrastructure built on MyST Markdown and Curvenote. This transition enabled a web-first, interactive publishing workflow that improves reproducibility, readability, and metadata quality. Authors now submit articles through GitHub, receive structured feedback via continuous integration (CI), and benefit from live previews, automated PDF generation with Typst, and archival JATS XML export. This paper details the infrastructure, authoring workflow, and technical tooling that powers the new SciPy Proceedings. We demonstrate how these advances support executable content, interactive figures, and modern metadata standards—offering a model for computational publishing in other scientific communities.
---

# Introduction

Since its inception, the SciPy Proceedings have offered a high-quality venue for publishing scientific and technical contributions built on the Python ecosystem. With over 350 peer-reviewed articles published over 17 years, the proceedings have become an essential resource for the community. Yet, the traditional PDF-based publishing model—common across many academic fields—has remained largely unchanged, limiting interactivity, reproducibility and discoverability.

In 2024, the SciPy Proceedings committee reimagined the entire publishing workflow by adopting modern, open-source community-built infrastructure built on[ ](https://mystmd.org)[MyST Markdown](https://mystmd.org) and[ ](https://curvenote.com)[Curvenote](https://curvenote.com). This enabled the SciPy Proceedings to be part of a larger movement and ecosystem of open source publishing tools and workflows. The result was a new model of computational publishing that supports executable content, high-quality structured metadata, and a web-first reading experience with interactive features. This shift not only enhances the publishing experience for authors and reviewers but also significantly improves the way readers interact with scientific content.

# Interactive Reading & Publishing Experience

The most noticeable improvement in the 2024 Proceedings is the transition from static PDFs to web-native articles rendered directly from MyST Markdown. Each article is hosted as an interactive web page, preserving semantic structure and supporting accessibility, deep linking, and citation preview.

Articles can include:

- Executed **Jupyter Notebooks** with full output.
- **3D interactive visualizations** using tools like k3d, pythreejs, or vtk.js.
- Embedded figures, equations, and citations.
- Linkable sections, figures, and code blocks.

This structure enhances readability and facilitates reuse. Readers can link directly to sections or figures, view outputs in-line, and copy or preview code embedded from GitHub repositories. Unlike previous workflows, the articles are rendered using the same MyST engine throughout authoring, preview, and publication—what authors see locally is exactly what readers experience on the web.

:::{figure #listing} https://lh7-rt.googleusercontent.com/docsz/AD_4nXfiL0EM5HyvulsbxndhDr5lNF3ccv6B4TDACudp6121gPLf17FbpGDwqNmudrOF0OKRUQiI_ZfJc5FS7nAZY0UYCZh68uyFCGf9cX3GMeRUCkDoxQCe93M3CEYbEJCCWW7DapcA7w?key=vSig9yoi3-QTUzhbwsUJyw

Side-by-side comparison of a 2023 Proceedings listing page and the 2024 listing in the new format. All previous years have been transferred to the new format, and the 2023 have been converted to full text as well; previous years have not yet been fully converted and only show abstracts.
:::

:::{figure #web-rendering} https://lh7-rt.googleusercontent.com/docsz/AD_4nXdCBI5bg362tMHeJqmnaGVA65ShiYWLQIuTRXdCM7zfXEsVV6TjF6mDHiBzCkyIlTkzVjz_9I1bNwcUW57qOHaR_5RFAFsqjbBqAyi6QAl0GI4pj8U2X7KGyhEYUBBJtoL-j8y7aw?key=vSig9yoi3-QTUzhbwsUJyw

We have focused on web-native rendering, hover previews and including interactive figures.
:::

# Authoring & Submission Workflow

Authors submit their articles using a standardized MyST template scaffolded by the mystmd CLI. Each submission consists of:

- A myst.yml file capturing authorship, affiliations, ORCIDs, RORs, keywords, and licensing.
- Markdown or Jupyter Notebook source files using MyST syntax.
- Static assets such as figures, datasets.

After previewing the article locally with myst start, authors open a Pull Request on GitHub. This submission triggers a **GitHub Actions CI pipeline** that performs:

- **Structural validation** of the document.
- **Metadata schema validation** (e.g., required ORCID, DOIs for references).
- Rendering to HTML for preview.
- PDF generation via Typst.
- JATS XML export for archival.

Authors use the CI feedback to identify and resolve issues, such as missing identifiers, improper headings, or unreferenced figures. This structured feedback improves metadata quality and consistency across the proceedings—analogous to "linting" in software development.

:::{figure #pull-request} https://lh7-rt.googleusercontent.com/docsz/AD_4nXcg7QwsU0j_gXffNQ7QDXQv_lx-gQeOIqiEIuyDTTMIdPInsKNMbflnnzS5slqltjy6f2BdesOBHR3i9ehXWzc1E1U9TdkQkx7DN3c3xR4zcN_4irLZxOJTOj3zYrkgSOIOC9-04Q?key=vSig9yoi3-QTUzhbwsUJyw

Screenshot of a GitHub Pull Request showing CI failure due to missing ORCID metadata, with the corrected version in a follow-up commit ([Cervantes-Sanchez, 2025](https://github.com/scipy-conference/scipy_proceedings/pull/1090))
:::

As with previous years, open peer review occurs directly within GitHub, with reviewers leaving line-level comments and discussing changes collaboratively. Once accepted, the article is merged and included in the final build.

# Build System and Infrastructure

The SciPy Proceedings infrastructure is built on GitHub Actions and the Curvenote CLI. Each submission is automatically validated and rendered across several formats using a reproducible build pipeline:

- **Web (HTML)**: Rendered using MyST Markdown and published to the Proceedings site.
- **PDF**: Generated using[ ](https://typst.app)[Typst](https://typst.app), replacing LaTeX for greater speed, reliability, and customizability.
- **JATS XML**: Introduced in 2024, this machine-readable format supports long-term preservation and interoperability with scholarly indexing systems.
- **Previews**: GitHub Pages hosts live previews per Pull Request, allowing reviewers and authors to inspect the exact output.

:::{figure #pdf-template} https://lh7-rt.googleusercontent.com/docsz/AD_4nXcLNMaAGtbrNFh_8tZTM67NzMKAXP44SxIFeL_GK0w0KlnztfNj4e0L3ptp2EbQX4MFKMUhaLMc-b8oneOb3XMrOQ3eyBOi1ax_zhqzKDgXkvV9ZulZFUDYXkZsM6hhptb2NU9HcQ?key=vSig9yoi3-QTUzhbwsUJyw

Comparison between 2023 and 2024 PDF templates, 2024 improved the branding and template to be more modern; the template now uses Typst over LaTeX, which also improves the build speed.
:::

The transition to Typst allowed us to build a fully customized SciPy Proceedings PDF template, ensuring consistency with the new web style while supporting traditional academic formatting and citation standards.

# Metadata, Reuse, and Interoperability

The move to structured MyST Markdown and schema-validated metadata unlocks new possibilities for reuse, attribution, and discoverability. Metadata is required at the submission stage and validated throughout the review process, including:

- Author names and ORCIDs
- Institutional affiliations (ROR)
- Keywords, license, and abstract
- DOI for associated datasets or software

Each component of an article (e.g., figures, equations, or code blocks) is preserved in a structured way, improving reusability. For example, figures can be referenced externally, cited independently, or embedded into institutional repositories without modification.

JATS XML export supports interoperability with discovery engines and indexing services, while also laying the foundation for cross-journal content federation and machine-readable citation networks.

# Broader Implications and Future Work

The 2024 SciPy Proceedings represent a working example of open, executable scientific publishing. Unlike most journals, this infrastructure is entirely built on open tools:

- **MyST Markdown**: An open standard with community-led development.
- **Curvenote**: Open-source tooling for authoring and submission.
- **Typst**: A modern, open-source typesetting engine.
- **GitHub Actions**: Transparent, reproducible CI workflows.

Looking ahead we plan to improve the Jupyter-based live execution support in future years and integration with computational notebooks.

# Conclusion

The SciPy Proceedings have embraced a new model for computational publishing—one that is open-source, interactive, and built for reproducibility. By combining MyST Markdown, Curvenote, GitHub-native workflows, and Typst, we provide authors and readers with a more dynamic, structured, and trustworthy experience. This infrastructure is not specific to SciPy. It offers a replicable model for other conferences, journals, and scientific communities seeking to modernize how research is written, reviewed, and shared.

# Acknowledgements

We gratefully acknowledge the contributions of Chris Calloway, Meghann Agarwal, Stefan van der Walt, and Jim Weis for their support, coordination, and feedback throughout the transition to the new SciPy Proceedings infrastructure. Their efforts were instrumental in shaping the submission process, reviewing tooling, and ensuring a smooth experience for authors and reviewers alike.

We thank the current SciPy 2024 Proceedings Committee — Amey Ambade, Ana Comesana, Sean Freeman, Sanhita Joshi, Charles Lindsey, and Hongsup Shin — for their dedication and hard work in supporting authors, managing reviews, and ensuring the quality and integrity of this year’s volume.

We also thank the many members of past SciPy Proceedings committees and infrastructure teams whose work laid the foundation for this transformation. The continuity, care, and community leadership built over the last 17 years have made this evolution possible.
