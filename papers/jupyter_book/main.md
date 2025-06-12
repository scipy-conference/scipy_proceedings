---
title: Jupyter Book 2 and the MyST Document Stack
subtitle: A composable, extensible, machine-readable, next-generation stack for authoring and sharing computational content
abstract: |
  Jupyter Book allows researchers and educators to create books and knowledge bases that are reusable, reproducible, and interactive. Over the past three years, Jupyter Book has been entirely rebuilt atop the MyST Document Engine (mystmd). The `mystmd` engine prioritizes machine readability, extensible and flexible deployment, modular and composable architecture, and treats computation as a first-class citizen in authoring and reading. This new foundation introduces a scalable way to publish interactive computational content, support structured metadata, and enable content reuse across contexts. Jupyter Book 2 (JB2) is now a Jupyter subproject, powering open scientific resources including [The Turing Way](https://book.the-turing-way.org/), [QuantEcon](https://quantecon.org/), [Project Pythia](https://projectpythia.org/), and the [QIIME 2](https://qiime2.org/) Framework (Q2F) documentation ecosystem. This paper introduces the principles, architecture, and capabilities of JB2, shares real-world adoption stories, and explores how these tools are shaping the future of open computational publishing.
---

## Introduction

Jupyter Book has become a cornerstone tool in computational science education and open publishing, powering over 14,000 publicly available books, lectures, tutorials, and knowledge repositories. It bridges the gap between exploratory computation and readable narrative, making it possible to document, share, and publish computational workflows in accessible formats.

With Jupyter Book 2, the project has undergone a major transformation. The entire system has been rewritten around the MyST Document Engine (mystmd), a next-generation engine for authoring computational narratives. The project has also formalized the MyST Document Standard as a structured and semantically-meaningful way of representing computational content in a reusable way, with MyST Markdown as a primary authoring syntax. The goal of this overhaul is to make structured scientific content more interoperable, composable, extensible, and machine-readable. This enables researchers to publish high-quality, interactive content with support for standards-based reuse, full-text APIs, executable notebooks, and output formats like Typst, LaTeX, Microsoft Word, JATS XML, and more.

In this article, we: (a) document the guiding principles and design thinking behind this major release; (b) present high-level architecture of Jupyter Book and the MyST Markdown ecosystem; (c) compare to other tools in the ecosystem; and (d) highlight case studies of notable projects.

### Bridging Exploration and Publication

The mission of Jupyter Book and MyST is to build composable tools and standards for people that communicate technical and computational narratives. This communication spans the full lifecycle of discovery—from informal lab notes and collaborative tutorials to formal research articles and textbooks. Our goal is to support authors, educators, and researchers as they share work within local communities and publish across the global scientific commons.

We are focused on two overlapping but distinct communities:

Data-driven communicators
: Researchers, analysts, and educators who need to share working, reproducible content with collaborators, students, or the public. These users rely on Jupyter, Python, and other computational tools in their daily workflows, but often lack well-integrated systems to communicate what they’ve done or how to reproduce it.

Scholarly Authors and Publishers
: Scientists and academics preparing content for peer review, journals, repositories, or formal documentation. These users need structured metadata, citation systems, and outputs compatible with academic and institutional publishing pipelines.

Both of these groups are underserved by the current ecosystem. Today’s scientific workflows rely on computational tools like Jupyter Notebooks, yet the systems used to communicate and publish these workflows are fragmented and poorly integrated. Interactive computing environments are designed for exploration; publishing tools are optimized for presentation. Content is regularly duplicated, reformatted, and stripped of context in order to meet the constraints of output-specific pipelines (e.g. the PDF!).

Jupyter Book and MyST address this gap by adopting several guiding principles:

Structured Documents and Machine Readability
: Produce a machine-readable abstract syntax tree (AST) for every page. This serves as the canonical representation of the content, allowing it to be rendered into multiple outputs—without losing structure, metadata, or intent. We give specific effort to support the open-science ecosystem, especially around structured, scholarly metadata that help discoverability (e.g. DOIs, ORCIDs, RORs).

Modular and Composable Design
: Content is built from small, reusable components. This enables partial reuse, easy cross-referencing, and support for diverse outputs—whether you're building a textbook, a course module, or an academic article.

Computation as a First Class Citizen
: Code, data, and outputs are treated as integral to the narrative—not hidden, summarized, or stripped away. This enables readers to explore, verify, reproduce and reuse computational logic directly.

Community Governance and Development
: The tools we develop are governed by an open community in order to ensure they continue to represent the interests of its community members and are not driven by a single stakeholder or organization.

These guiding principles define how Jupyter Book and mystmd have been implemented in practice. In the sections that follow, we outline the specific strategies and design decisions that underpin Jupyter Book 2. Each addresses major challenges in modern scientific communication—from ensuring computation is never divorced from narrative, to building a publishing system flexible enough to support both informal collaboration and formal scholarly outputs.

## Design Principles

The design of Jupyter Book 2 is grounded in practical responses to the needs of researchers, educators, and publishers who must communicate with and about data. The guiding principles introduced earlier—structured documents, composability, computation as a first-class element, and community-led development—translate directly into architectural and interface-level decisions in JB2 and the mystmd engine.

In this section, we describe how these principles are operationalized in Jupyter Book 2, outlining the technical and conceptual strategies that distinguish it from previous versions and from other publishing tools.

### Structured and Machine Readable Content

Underpinning every Jupyter Book 2 site is a structured, machine-readable abstract syntax tree (AST). This internal representation captures not just the text of a document, but also its semantic components: citations, headings, roles, directives, code blocks, figures, equations, and embedded metadata like ORCIDs, DOIs, or author affiliations.

Pages can expose machine-readable metadata endpoints (page.json), enabling full-text search, granular cross-referencing, and live content embedding across projects. By aligning with community standards (e.g., JATS, DOIs, ORCID, ROR), the system supports today’s scholarly publishing workflows, remains grounded in lightweight, markdown-based authoring, and opens up new possibilities for more composable workflows in the future for scientific authoring.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdK1XxQZhyEImdqfWjMh8sAY2av-1A5SFm-uqbXCsfDeW53EVq4P6cQ7oMeWIGqJ8uYqTJK-RPra18BdJvJls0JX50jhIENjU4LJe5fn3n7ix_maR8vkl8f8FU6lt4bXZ10-zb8SQ?key=lS7P7yOBhfjw9T7nfZGN0w)

### Modular and Composable Design

The MyST ecosystem is designed for composition. Documents are structured into small, addressable components—sections, figures, equations, terms, figures, outputs—that can be reused across books, projects, and publications. This modularity supports new patterns of collaboration and publication: a figure generated in a course textbook can be cited and embedded in a different tutorial or a journal article without duplication or loss of provenance.

Each page or resource in a Jupyter Book project is assigned a unique slug and corresponding AST. These can be referenced across builds or even embedded with hover previews. Updates to shared content propagate automatically to referencing projects, enabling collaborative knowledge bases and modular curricula. This is based on both the MyST Specification ([mystmd.org/spec](http://mystmd.org/spec)) and a cross-reference manifest myst.xref.json. Similar to intersphinx references the myst.xref.json gives stable links to any aspect of the content, however, the structured JSON APIs enable hover-previews across projects. This feature allows for projects to add a reference to the project, and then use xref: links to deeply link or embed the content; these links are either done at build-time or render time and allow for dynamic linking between projects with a simple user interface.

This composability is designed to make it easier to scaffold different kinds of outputs from the same underlying materials. For example, a long-form research article could be built from shared methods sections, tutorials can be composed of reusable datasets and figures, and institutional knowledge can be aggregated across lab groups.

### First-Class Computation

Many researchers begin their communication journey in a computational notebook—exploring data, testing hypotheses, or sharing early-stage insights. Bringing that work into a readable, shareable, and publishable format traditionally can require significant rework: copying plots into a document, cleaning outputs, manually formatting results. These extra steps not only introduce friction, but also weaken reproducibility and create a disconnect between the computation and its communication.

Our goal with Jupyter Book 2 is to integrate computational content at every level. Code, figures, outputs, and interactive elements are embedded directly in the narrative and treated as primary content—not sidebars or supplemental material. Executable content can be powered by Binder, JupyterHub, or JupyterLite, allowing readers to rerun computations on demand, right from the page without leaving the context of the narrative. This is powered by [thebe](https://thebe.readthedocs.io/en/stable/), which enables static documents to become interactive by linking code cells with a computational kernel. It is designed to be flexible and forward looking, supporting the full range of kernels available from in-browser execution through to large server based environments that could include specialist hardware such as GPUs and TPUs.

A major motivation for rebuilding Jupyter Book on top of the new mystmd engine was to enable a truly **web-native execution model**—something that was difficult or impossible within the Sphinx-based architecture. mystmd is built in JavaScript and leverages the unist syntax tree ecosystem, making it directly compatible with modern browser environments and frontend frameworks like React, as well as a host of other plugins and transformations (e.g. unified-latex). This shift has also allowed us to develop jupyterlab-myst, a plugin that brings MyST rendering capabilities directly into JupyterLab.

With jupyterlab-myst, authors can preview how their markdown content will appear in the final rendered book—including directives, equations, citations, and even code outputs—without leaving the notebook environment. This tight feedback loop reduces friction during authoring, while preserving a single source of truth between exploratory notebooks and published outputs.

The plugin is a key part of our vision to unify **authoring, execution, and publishing** in one environment. Rather than writing in one tool and rendering in another, as this tool evolves—alongside the real-time collaboration capabilities in JupyterLab—authors will be able to use JupyterLab as a full-featured writing and publishing environment. This lowers the barrier to creating high-quality computational narratives, especially for teams already working inside Jupyter for their day-to-day research and teaching.

### Community-led

Jupyter Book and MyST are community-led projects, developed in the open and governed collaboratively. \[add team compass links etc.] The tools are maintained by contributors from open infrastructure organizations, universities, independent developers, and end-users who rely on them in practice. This ensures that the ecosystem is not steered by a single stakeholder or locked into one publishing or business model.

The system is also designed to be extensible and reactive to the community needs. Themes, plugins, and content directives can be customized or replaced entirely, allowing different communities to shape the tooling to fit their own needs. Journals can define their own publishing templates and lab groups can maintain shared standards across projects. There is still much work to be done to fully realize this vision, however, many of the lab-groups, independent publishers, and individuals have taken the tools as is and adapted them to their own workflows and requirements.

## Architecture of Jupyter Book 2

Jupyter Book 2 uses a structured publishing engine built on top of the [mystmd](https://mystmd.org) document system, moving away from a Sphinx-based static site generator. This new architecture is designed to enable dynamic rendering, multi-format output, and composable content workflows—backed by a structured document model that makes metadata, structure, and computation portable and machine-readable. This section describes the major architectural decisions and components that drive the Jupyter Book 2 stack.

### A Foundational Structured Document Specification

At the heart of Jupyter Book 2 is the **MyST Abstract Syntax Tree (AST)**, a machine-readable representation of each document’s content and structure. When a user builds a Jupyter Book, the system parses all inputs—including markdown files, MyST directives, Jupyter notebooks, LaTeX documents, and frontmatter—into this unified AST. This process preserves not only the narrative content, but also computational blocks, Jupyter Outputs, citations, cross-references, headings, figures, and semantic metadata such as author affiliations.

### Content First, Format Second

The architecture emphasizes a **“content-first” model**, where authoring is separated from output format. The AST serves as the single source of truth for rendering content into multiple target formats: web-native HTML, JATS XML, PDF via Typst or LaTeX, Microsoft Word, and more. Each of these renderers consumes the AST and applies format-specific styling, layout, and transformations—without requiring changes to the source content. Our investment is currently focused on the React-renderers, Typst PDF templates (a new PDF renderer that is significantly faster than LaTeX), and JATS XML used in scholarly publishing. For example, the SciPy Proceedings for 2024 and 2025 use both the React renderers, the typst renderer for PDF creation, and JATS XML for all articles.

This AST is the “source of truth” for rendering content to different formats—HTML, PDF (via LaTeX or Typst), DOCX, JATS XML, or custom layouts. The goal of an intermediate representation means that authors can write once and publish anywhere without manually adapting content for each target format.

### API-Accessible and Federated by Design

One of the most transformative aspects of the JB2 architecture is that every page has an associated .json API endpoint that exposes its AST and metadata. This means content can be programmatically indexed, referenced, embedded, transcluded, or transformed without needing to scrape or recompile the site. Readers and developers can query structure, citations, and content across books, enabling new tools like federated search, content previews, and live citation graphs.

This approach lays the groundwork for a **federated publishing network**—where content from different lab groups, institutions, or journals can be referenced, reused, and remixed while maintaining proper attribution and structure.

Just as the Jupyter Notebook format (.ipynb) has enabled interoperability across dozens of tools by serving as a stable, well-defined format, the MyST AST aims to serve a similar role for scientific communication. The AST specification is versioned and documented, and efforts are underway to standardize it further via the myst-spec initiative. We are also investing in forward-and-backwards interoperability in this spec through the reference implementation of mystmd. This allows developers, publishers, and downstream platforms to build confidently on top of Jupyter Book 2 without fear of breaking changes.

### A Unified Toolchain

Jupyter Book 2 is built around a cohesive, modular toolchain that enables fast builds, partial updates, and web-optimized outputs. This toolchain includes:

- **jupyter book build** — the core CLI that parses source files into the AST and compiles them into static assets or structured outputs. Built on and mirroring the myst CLI.
- **\@myst-theme** — a modern frontend theme and rendering engine for web output, based on React and optimized for interactivity and performance.
- **jupyterlab-myst** — a JupyterLab extension that renders MyST documents natively inside the notebook interface, creating a tight feedback loop between authoring and final presentation.
- **Plugin system** — supporting directives, roles, and custom transforms of the AST.

The system also supports per-page and per-project metadata configuration using frontmatter.yml, enabling authors to define publishing metadata, contributor lists, keywords, license info, and custom identifiers in a structured, version-controlled format.

### The Relationship Between Jupyter Book and the MyST Engine

Jupyter Book is a distribution of the mystmd engine. Every Jupyter Book is also a MyST project, and building a Jupyter Book means building a MyST project. Over time, we expect mystmd to be more geared towards power users and jupyter-book to have more built-in, automatic configuration and opinions for building textbooks, community documentation, and course materials, etc. This is similar to the relationship between jupyter book and Sphinx in JB1, and includes escape-hatches for power users to use the underlying modular ecosystem of MyST tools and extensions directly.

## Comparison to Other Tools

Jupyter Book 2 sits among a number of tools for authoring and publishing computational and scientific content. These include static site generators, document converters, and academic publishing frameworks. Many of these systems are well-suited to particular workflows—whether technical documentation, reproducible research, or educational materials—but often focus on specific output formats, assume linear document structures, or lack integration with browser-based and computational environments. JB2 is designed to address these gaps, particularly for Python and Jupyter-focused communities working in open science.

Jupyter Book 1 was built on [Sphinx](https://www.sphinx-doc.org/), a Python documentation generator that enabled integration with reStructuredText, MyST extensions, and a range of HTML output themes. While powerful, Sphinx is not optimized for browser-based authoring or rendering, and customizing output across multiple formats (e.g., PDF, JATS XML) required deep knowledge of its internal build system.

[Quarto](https://quarto.org/), developed by Posit, is the most comparable tool to Jupyter Book 2 and the MyST ecosystem. Quarto is a flexible publishing system supporting Jupyter notebooks, markdown, and multiple output formats including HTML, PDF, DOCX, and JATS. It is built on top of [Pandoc](https://pandoc.org/) and is tightly integrated with R, RStudio, and Positron ecosystems. At this time, Quarto has better customization support, better attention to details around LaTeX PDF rendering (e.g. sub-figures, tables), and native presentation support. The web-builds are static HTML sites and do not, at this time, support similar ideas of content APIs, which are core to Jupyter Book—and are pointed towards a different, complimentary vision of federated and interoperable content. Jupyter Book’s governance is community driven, as opposed to a single company, and uses a permissive [MIT license](https://github.com/executablebooks/mystmd/blob/main/LICENSE), which may be preferable for downstream projects with more open licensing requirements.

Other systems like [Docusaurus](https://docusaurus.io/), [MkDocs](https://www.mkdocs.org/), and [Hugo](https://gohugo.io/) are popular choices for technical documentation and developer blogs. These tools offer good performance and theming options but are not designed for scientific publishing or computational narratives. They lack features such as executable content, math, and citation handling, semantic metadata, or persistent identifiers—all of which are built into the Jupyter Book 2 ecosystem.

While these tools each serve their respective audiences well, Jupyter Book 2 is focused specifically on the needs of computational and scientific communities. It supports both narrative and executable content, prioritizes open infrastructure and interoperability, and is designed to integrate with existing Jupyter workflows and the wider open-science ecosystems.

## Case Studies

Jupyter Book 2 is already being used across a broad range of scientific and educational projects. From national training initiatives to domain-specific research platforms, these case studies demonstrate how JB2 supports reproducible publishing, modular content reuse, and scalable collaboration.

**Project Pythia** is a community-driven effort to advance geoscience education and training through open-source learning materials. Pythia’s Jupyter Book sites draw from multiple repositories and contributors, making the structured content model and cross-repository referencing features of JB2 particularly valuable. With JB2, Pythia authors can assemble lesson content from shared modules, customize rendering and metadata, and publish federated resources that support both learners and instructors. There were two standout features to the community: (1) the ability to compose and share configuration between sites, including things like footers, shared references, navigation links; and (2) the ability to cross-reference content between un-related projects. Both of these build on the composability and modularity designs that are at the core of Jupyter Book 2.

**The QIIME 2 Framework (Q2F)** is a platform for biological data science tools, originally developed for the QIIME 2 microbiome bioinformatics toolkit, but now supporting tools across diverse subdomains of biology including multiplexed serology and pathogen genomics. Q2F has adopted JB2 to support its documentation ecosystem which includes documentation that is general to users of all Q2F-based tools (e.g., creating and using a Q2F data artifact cache) and documentation that is specific to subdomains (e.g., an end-to-end tutorial for microbiome data analysis). JB2 enables embedding of content across documentation projects, such that general content can be embedded in domain-specific content where relevant, making it easier to hide the software stack complexity from end users while avoiding duplication of source content.

**QuantEcon** provides open educational resources in economics and quantitative methods, built using Python, Julia, and Jupyter. The transition to JB2 included investment in a custom theme for JupyterBook, which allowed the project to modernize its publishing infrastructure while maintaining interactive notebooks and supporting export to multiple formats including html, pdf, and ipynb. The experience for authors remained mostly constant, however, The upgraded theme allowed for adoption of new features like hover cross-reference between exercises and solutions using the proof and exercise extensions.

These case studies demonstrate the flexibility of JB2 to meet the needs of both education and research, large and small teams, and varied technical audiences. Each project benefits from the same core strengths: modular content, structured metadata, and tight integration with Jupyter.

### A note on migrating from Jupyter Book 1

A key goal of Jupyter Book 2 was to leverage the design and standards from Jupyter Book 1 and the Sphinx stack in order to facilitate the upgrade process. As a result JB2 leverages the same MyST Markdown syntax as JB1 with minimal disruption. It also aims to expose a key subset of the extension points that were available in JB1 and Sphinx (for example, roles, directives, custom transforms, etc). While there is a subset of functionality that is still unique to Sphinx, the JB2 team is focusing their efforts on developing key missing functionality to narrow this gap. This effort has been driven by upgrade collaborations with heavy community users of JB1, described above as case studies.

## Conclusion

Jupyter Book 2 is an open-source publishing system for computational and scientific content. It enables authors to combine markdown, notebooks, structured metadata, and executable code into documents that can be rendered as websites, PDFs, JATS XML, Word documents, and more. By rebuilding the system around a structured document model and a web-native engine, mystmd, JB2 supports workflows that are modular, reproducible, and extensible; qualities essential for modern research and education. It bridges the gap between exploratory computation and formal, published communication, enabling authors to write once and publish to many formats.

Unlike general-purpose documentation systems, static website builders, or single-format converters, Jupyter Book 2 is purpose-built for the needs of researchers, educators, and institutions working with code, data, and narrative. Each page produced with JB2 has a structured, machine-readable representation that can be queried, embedded, or reused via an API. This enables a federated publishing model, where content from different projects, lab groups, or institutions can be referenced, previewed, and composed together—without duplication or loss of structure. The result is a more connected and discoverable ecosystem of scientific knowledge, where modular content can be maintained in place but surfaced in multiple contexts, with proper attribution and rich metadata preserved.

The Jupyter Book project is openly governed under Project Jupyter, actively maintained, and already powering a diverse set of high-impact initiatives across domains. From reproducible data science guides like The Turing Way, to educational curricula like QuantEcon and Project Pythia, to domain-specific platforms like the QIIME 2 Framework, Jupyter Book 2 is helping authors publish richer, more reusable, and more transparent scientific outputs.

As scientific publishing continues to evolve, Jupyter Book offers a model for infrastructure that is interoperable, open by design, and responsive to the needs of its users. It is not just a tool for building books—it is a foundation for the next generation of computational publishing.

## Acknowledgements

Jupyter Book has been developed over the course of nearly a decade, with contributions from dozens of individuals and organizations. We are grateful to the many collaborators, maintainers, and community members who have helped shape the project’s evolution—from its early days as a Jekyll-based tool for compiling notebooks into a textbook, through the Sphinx-based Jupyter Book 1 release, and now with the MyST Markdown and mystmd-based 2 architecture. This article did not intend to capture those wide-ranging contributions or fully document the history of the project — for those who we didn’t reference specifically, the spirit of the first author of the Jupyter Project is to capture your contributions.

This project has benefited from key financial support from multiple organizations. The development of MyST Markdown, the Executable Books ecosystem, and Jupyter Book v1 was supported by a grant from the **Alfred P. Sloan Foundation** \[Grant #9231]. The development of mystmd and the rearchitecture of Jupyter Book 2 were made possible in part by funding from **Alberta Innovates, The Stanford Doerr School of Sustainability**, as well as in-kind support from **Curvenote**, **2i2c**, and the broader open-source community. The refactoring of the QIIME 2 Framework documentation ecosystem was supported in part by the NIH National Cancer Institute \[Grant 1U24CA248454-01].

We especially acknowledge **Chris Sewell**, whose foundational work on the MyST Markdown specification, the Sphinx parser (myst-parser), and the wider Jupyter Book v1 ecosystem in Sphinx. Chris’s attention to these technical underpinnings laid the groundwork for many of the capabilities now realized in Jupyter Book 2.

We also acknowledge the work of the **Executable Books team**, the contributors to **JupyterLab, Thebe**, and the maintainers of MyST-related tooling and plugins. The ongoing development of Jupyter Book 2 is supported and stewarded by the **Jupyter Community**, and is now an official Jupyter Subproject.
