---
title: Transformative Approaches in Scientific Publishing
abstract: |
  Difficult problems that are globally relevant and urgent — like climate change, pandemics, and water security — require us to radically rethink how we publish and work towards scientific breakthroughs.
  Advances in technology for data workflows have increased the speed and scope of scientific discovery.
  However, scientific dialogue still uses outdated technology for communicating and sharing knowledge, relying on static PDFs that poorly represent of the complexities of science.
  This gap impedes the speed of research dissemination, reuse, and uptake: we require new mediums to compose ideas and ways to share research findings as early and as often as possible; this gap is even more evident in the computational sciences.
  In this paper we discuss two tools for scientific dialogue, MyST Markdown and Curvenote, and illustrate examples of improving metadata, reimagining the reading experience, and transforming publishing practices for individuals and societies.
---

## Introduction and Motivation

In the face of mounting global challenges such as climate change, pandemics, and water security, the imperative for rapid, effective scientific discovery and dissemination has never been more acute.
The pace at which these problems evolve and impact societies worldwide demands an equally dynamic and innovative approach to how scientific research is published.
Despite significant advancements in technologies that enhance data collection, analysis, and workflow efficiency, the mechanisms through which scientific knowledge is shared have remained largely unchanged for decades [@10.4230/DagMan.1.1.41].
The widespread reliance on static PDF formats for research papers starkly contrasts with the complex, data-driven and increasingly computational nature of modern science, creating bottlenecks in knowledge dissemination and uptake.

This paper documents some of the design decisions made to address challenges in science communication and publishing in two tools: (1) MyST Markdown (Markedly Structured Text, <https://mystmd.org>), a community-run open-source Jupyter sub-project[^myst-jep], which is a text-based authoring framework that integrates computational content (e.g. Jupyter Notebooks); and (2) Curvenote (<https://curvenote.com>), which is a set of open-source utilities, command-line tools, actions and services[^curvenote] aimed to improve scientific publishing by journals, societies, lab-groups, and individuals.
In this article we provide background, motivation and perspective for our efforts in developing new open-source tools for science communication, with examples ranging from individual authors to journal administrators.
Though we present an overview of MyST Markdown, it should be emphasized that MyST Markdown is a community-run project and the authors of this article do not speak for all project participants; the community has varied goals for the project (including API documentation, community guidelines, educational tutorials).
Our focus in this article is to give our perspectives on scientific writing and publishing and how it intersects with these open-community projects in addition to the open-source efforts that Curvenote is undertaking around scientific publishing.

[^myst-jep]: MyST Markdown became a Jupyter Project on June 28, 2024 [](https://github.com/jupyter/enhancement-proposals/pull/123), and was previously hosted by Executable Books (<https://executablebooks.org>).
[^curvenote]:
    Curvenote is a company that provides many different tools for authoring and publishing content, including a collaborative WYSIWYG online editor that can export to MyST Markdown.
    In this article we discuss Curvenote's open-source tools and highlight ideas from working with Curvenote's partners when they pertain to improving scientific publishing.

In developing these integrated tools and workflows, our goal is to lower the barriers to continuously releasing and iterating on scientific ideas in the open and address the related challenges of _authoring_ and _publishing_ in the context of computational, open-science documents.
Introducing authoring tools that can understand and express structured, interactive, and computational content has the potential to fundamentally change the way scientific writing is checked, shared, and published — enabling faster iterations and direct ties to reproducible, interactive content.

(sec:structured-science)=

### Authoring Structured Content

There are currently many challenges for individuals or groups to author research information that can be shared in a structured and rigorous way.
By this we mean the things that _structurally_ set a scientific article apart from, for example, a blog post: structured content, cross-references, valid citations with persistent identifiers (PIDs), and standardized metadata for licensing, funding information, authors, and affiliations.
These structured content and metadata, as well as the standards behind them, are what define the "scientific record" and enable archiving, discoverability, accessibility, interoperability and the ability to reuse or cite content [@10.1038/sdata.2016.18].
One metric for measuring the difficulty of satisfying these scientific standards is to look at the direct costs that are spent on transforming author submissions (e.g. a PDF or a Word Document) into something that conforms to these standards and is ultimately archived.
In scientific publishing, about 15% of Article Processing Charges (APCs) go to direct publication costs[^pub-costs] [@10.12688/f1000research.27468.2].
When applied to the global publishing industry[^pub-revenue], this figure suggests that approximately USD\$2 billion dollars is spent on transforming author submissions (e.g. a word-document, LaTeX, or a PDF) into a copyedited, well-formatted, typeset document that can be archived with appropriate metadata [@10.4045/tidsskr.20.0118].
This estimate does not include the approximately USD\$230 million spent on reformatting articles by scientists _before_ publication [@10.1186/s12916-023-02882-y].
Many of these processes are hidden from authors[^hidden-processes] as well as actionable access to many of the benefits of structured data beyond citation graphs.

[^pub-revenue]: Global revenue in scientific publishing is around USD$19 billion, with over 50% of the market controlled by Elsevier, Wiley, Taylor & Francis, Springer Nature and SAGE [@10.4045/tidsskr.20.0118].
[^pub-costs]: Direct publication costs include: checking of manuscript, copyediting, typesetting, formatting figures/graphs/tables, XML and metadata preparation, and handling corrections [@10.12688/f1000research.27468.2].
[^hidden-processes]:
    Much of the production publication processes are hidden from scientific authors, with typesetting focused on cross-references, linking citations, ensuring citations have appropriate IDs (e.g. DOIs) as well as conversion to JATS XML (a NISO standard for archiving scientific articles), metadata preparation to [CrossRef](https://crossref.org), and archiving services like LOCKSS (<https://lockss.org>) and CLOCKSS (<https://clockss.org>).
    Additionally, the many proprietary services and tools to create both online and PDF outputs of the authors work that are nicely typeset for reading on the web or online.

One goal of the MyST Markdown project is to _dramatically_ reduce these direct-publication costs[^zero-cost] and directly provide direct unfettered access to structured data as an output of authoring.
The availability of this structured data directly enables exported content in a variety of formats including HTML, PDF and JATS-XML (a NISO standard for archiving scientific articles).
In this article, we will demonstrate that having structured data throughout authoring can lead to a number of novel reading and authoring experiences [e.g. @eg:hover], and can provide new opportunities for reuse and quality checks when publishing [e.g. @eg:checks].
Furthermore, these transformation processes can be run _continuously_, opening the possibilities for faster feedback [See @sec:continuous-science], iterative drafts, small tweaks and versioned improvements that otherwise would not be worth the time and cost.

[^zero-cost]:
    The cost of transforming author submissions to produce structured content and metadata should approach zero, at least for a subset of users.
    For example, _technical_ users who can use open-source command-line tools like MyST Markdown and GitHub.
    This has been shown to be the case for the Journal of Open Source Software (JOSS), for example, which has very low direct-publication costs per article [@10.7717/peerj-cs.147].

(sec:computational-articles)=

### Computational Articles

A compounding challenge to scientific publishing that we are exploring through MyST Markdown and Curvenote is how to deeply integrate computational workflows and content into science communication to promote interactive explorations and reproducible practices.
There are a host of challenges from user-interface design, to maintenance, to archiving computational content.
Many other tools have worked on aspects of integrating computation into scientific articles, notably R-Markdown [@10.1201/9781138359444] and it's successor Quarto (<https://quarto.org>); both of these projects have similar aims to MyST Markdown.
From a user-experience goals perspective, we are interested in questions such as:

- how to make a change in a notebook figure and have that immediately show up in a document;
- how to ensure computed values are inserted directly, rather than through copy-and-paste;
- how to expose interactivity and exploration that a researcher often has when analyzing a data-set;
- how to provide and launch archived interactive computing environments.

These questions require authoring tools to be able to execute content [c.f. @binder], to integrate and display computational/interactive outputs directly in reading experiences, as well as scientific publishing systems that can understand and archive computational content (e.g. Docker containers).
This deep integration can open up possibilities of embedding interactive visualizations and computational notebooks directly into scientific documents [e.g. @eg:interactivity], transforming articles from static texts into rich, interactive, reproducible narratives.
In 2023, the authors helped to lead several working groups related to these challenges as part of _Notebooks Now!_, a Sloan Foundation funded project led by the American Geophysical Union.
Those working groups found that integrating computational documents, via Jupyter Notebooks, into scholarly publishing system requires a re-imagination of the publishing processes (from submission to peer-review to production to reading) and that many existing processes and platforms are ill-equipped to handle computational articles [c.f. @10.1029/2023EA003458].
The "executable research articles" project out of eLife [@elife-era] has similar aims to _Notebooks Now!_, with some differences in how notebooks and articles are separated which we will discuss in @sec:computational-content.

The ability to deeply link computational content into how we communicate science can improve reproducible practices, and surface more interlinked content about methods and algorithms in use.
If used to their full extent, these can also fully integrate live computational environments into scientific articles, which provides many exciting possibilities for interrogating and extending scientific data and methods.

(sec:continuous-science)=

### Continuous Science Practices

The manual effort involved in article production [@sec:structured-science] coupled with the inability to integrate computational work [@sec:computational-articles] negatively impacts the number of iterations/versions and the immediacy of feedback to authors[^feedback].
In other disciplines, such as software development, these metrics of iteration and rapid feedback are often highly encouraged, measured and constantly improved.
For example, software organizations often measure and improve: the release cadence of a software product (e.g. continuous delivery); how confident you are in that release (e.g. based on continuous integration tests); how you get early feedback and confidence from linters and tests (e.g. the speed of your unit tests and integrated linters into development environments); and how fast you can obtain feedback from real usage and users on in-progress work (e.g. observability, analytics, customer interviews, design prototypes).
Continuous delivery practices of software development are also extremely well studied, with large-scale surveys of organizational performance, design, robustness, and speed (e.g. [DORA Reports](https://www.devops-research.com/research.html)).
One study compared elite teams with low-performing teams for software delivery, and found elite teams were **46x faster** to release to production (i.e. on-demand and multiple times per day vs monthly or bi-annually) and had **7x fewer errors** (due in part to better continuous deployment and testing infrastructure as well as smaller changesets) ([2018 DORA Report](https://services.google.com/fh/files/misc/state-of-devops-2018.pdf))[^additional-benefits].

[^feedback]:
    There are two types of feedback that we mean: (1) technical feedback as you are authoring, for example, "is this formatted correctly?" or "is this DOI correct?"; and (2) more substantial feedback from reviewers and readers who can only give you feedback when you have published.
    In the current system, technical feedback of an article-proof can take weeks and should be measured in milliseconds.
    Improving the immediacy of feedback from readers and peer-reviewers is a harder problem that involves how our existing sociotechnical system incentivizes article publishing rather than research communication and sharing findings as early and as often as possible.

[^additional-benefits]:
    In addition to increasing speed and robustness, continuous delivery practices also demonstrated that the elite teams spent 20% more time on new work, had 5-20% less manual work, and were 1.8x more likely to recommend their team as a great place to work ([2018 DORA Report](https://services.google.com/fh/files/misc/state-of-devops-2018.pdf)).
    These numbers are intriguing when contrasted to researchers, where (a) scientists already work 53.96 hours a week on average, and only about 36% of their time is actually spent on research (8% on grants, 32% on teaching, and 24% on service) [@10.1016/j.econedurev.2007.04.002]; and (b) graduate students are six times more likely to experience depression than the general population [@10.1038/nbt.4089].

The analogy between scientific publishing and software releases is imperfect and non-prescriptive (i.e. scientific research is very different than developing a product).
However, the analogy is illustrative in areas where there is a focus on iterations, smaller changesets and releasing in-progress work as soon as possible to get feedback from peers (i.e. scientific peer-review) or users (in the case of software products).
The speed of scientific progress depends _on part_ on the speed of iteration and feedback.
The time it takes for the peer-review process is over three months, in high profile journals like Nature that time has almost doubled over the past decade [@Powell2016Does].
Rejections are anywhere from 50-90% [@10.12688/f1000research.27468.2] with valuable reviews and expertise coming months or even years after the work is completed[^time-to-feedback].
There are wide spread efforts in scientific publishing that focus both on sharing smaller components of research (e.g. FigShare [@10.7771/2380-176X.7269], Octopus [@10.17863/CAM.96819], MicroPublications [@10.1186/2041-1480-5-28], NanoPublications [@10.3233/ISU-2010-0613], Protocols [@10.1371/journal.pbio.1002538], PreRegistrations [@10.1073/pnas.1708274114]) as well as sharing research sooner in the life cycle especially through preprints, for example, ASAPbio's mission to support "rapid open sharing, discussion, and collaboration around research" and promotion of "Preprints in Progress" [@10.1371/journal.pgen.1008565].

[^time-to-feedback]:
    In @10.1371/journal.pgen.1008565, they describe the importance of adopting preprints, as the "overall peer review process can take years".
    For the programmers reading this, it is worth a mental comparison to the pain of a pull-request being open for years.
    Having relevant feedback that is close to the time of implementation or writing is invaluable.

We refer to these related concepts as "_continuous science_", adopting language and concepts from "continuous integration and deployment".
The mechanisms to support continuous processes are through automation, rapid feedback on errors, and focusing on small, rapid changesets to accelerate feedback from peers.
This gives us a technical lens to assess, for example:

- How can you assess and test that the structure of a document applies to editorial rules?
- How long does it take to get feedback if your metadata or DOI is incorrect?
- When a computational figures or data output changes, how long does it take to integrate that into your document?

In science there is a highly manual, absurdly expensive and disconnected process between authoring and publishing.
By moving to continuous practices and investing in the appropriate infrastructure to support _continuous science_ we believe there is an opportunity to accelerate scientific discovery by multiple orders of magnitude while simultaneously increasing reproducibility and robustness of the underlying science.

% Current research incentives are focused on article publishing rather than research communication and sharing findings as early and as often as possible.

### Article Outline

For research-communication to be transformative on a similar scale as open-source software, researchers require modern tools for authoring and publishing.
There are two inter-related capabilities that are necessary for this transition:

1. authoring mediums that support data, computation and structured content without the need for expensive typesetting; and
2. publishing that is open and accessible to researchers at a variety of scales – individual publishing, lab-groups, societies and institutions.

Through the lens of MyST Markdown and Curvenote, this paper will explore how these tools can help to address critical gaps in current scientific publishing practices.
Our motivation is to enhance the _speed_ and _impact_ of research dissemination, fostering a scientific ecosystem that is more collaborative, reproducible, and equipped to tackle the urgent global challenges of our time.

(sec:authoring)=

## Authoring Tools

MyST Markdown (Markedly Structured Text, <https://mystmd.org>) is a community-driven markup language that is a superset of [CommonMark](https://commonmark.org/) (a standard form of Markdown) with special syntax for citations, cross-references, and block and inline extension points called "directives" and "roles".
The block-level content provides multi-line containers surrounded by either backticks or colons; examples include [callout panels](https://mystmd.org/guide/admonitions), [figures](https://mystmd.org/guide/figures), [equations](https://mystmd.org/guide/math) and [tables](https://mystmd.org/guide/tables) (see [documentation](https://mystmd.org)).
There is also specialized support for the types of metadata that are important to collect for scientific articles (funding, ORCIDs, CRediT Roles, etc.).
In 2022, the Executable Books project (<https://executablebooks.org>, which hosts Jupyter Book and MyST) started work on the `mystmd` command line interface (CLI), which was initially developed as the [Curvenote CLI](https://github.com/curvenote/curvenote), and later transferred to the ExecutableBooks project.
In June 2024, MyST Markdown officially became part of Project Jupyter (See [enhancement proposal](https://jupyter.org/enhancement-proposals/122-jupyter-book-incorporation/jupyter-book-incorporation.html)).
This tool allows authors writing in MyST Markdown to easily build websites and documents and supports the [JupyterLab MyST plugin](https://github.com/jupyter-book/jupyterlab-myst).
MyST is influenced by [reStructuredText (RST)](https://en.wikipedia.org/wiki/reStructuredText) and [Sphinx](https://www.sphinx-doc.org) – pulling on the nomenclature and introducing additional standards where appropriate.
There are also many intentional similarities of MyST Markdown to R-Markdown [@10.1201/9781138359444], Pandoc, and Quarto (<https://quarto.org>), especially in citation syntax and frontmatter structure.
The initial use case driving the development and design of MyST Markdown has been [JupyterBook](https://jupyterbook.org), which can create educational online textbooks and tutorials with Jupyter Notebooks and narrative content written in MyST.

This article will not attempt to describe the markup syntax directly, for that we suggest browsing the documentation at <https://mystmd.org>, instead we will focus our attention on the use cases for scientific publishing that we are trying to make as easy as possible.
Specifically, the ability to add persistent identifiers (PIDs) and links to other structured content; hover previews to show details on demand; integrating live computational content and interactive figures; and exporting to many different formats including those used by scholarly publishing.

### Utility of Links and Identifiers

In MyST Markdown, citations can be added inline using `@cite-key`, following Pandoc-style syntax and referencing a BibTeX file in a project.
It is also possible to directly link to DOIs using `@10.5281/zenodo.6476040`, which will create a hover reference [@10.5281/zenodo.6476040] as well as a references section at the bottom of the page.

This enhanced-links concept can be extended to [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia), RRIDs, RORs, GitHub issues or code, and other scientific databases to augment writing.
For example, the link `<rrid:SCR_008394>` becomes <rrid:SCR_008394>, with rich-metadata and citations created.
Wikipedia links come with previews, for example, `<wiki:gravitational_waves>` becomes <wiki:gravitational_waves>.
GitHub links to pull-requests also give hover information (for example, the following link [#87](https://github.com/jupyter-book/myst-theme/pull/87) shows a hover preview of the GitHub pull request in the online-version).

The use of DOIs and other structured scientific metadata can be reused in multiple different formats such as JATS XML and CrossRef deposits to create a DOI.
Our goal with these integrations is to make the use of persistent identifiers (PIDs) both easy and rewarding to the author as they are writing; this metadata is generally only added at publication time.

### Hover Previews for Content and Metadata

In @doi:10.1145/3411764.3445648 the authors show a speed up in comprehension of an article by 26% when showing information in context (i.e. "details on demand" [@10.1109/VL.1996.545307]), rather than requiring researchers to scroll back and forth to find figures and equations.
MyST supports these concepts natively for cross-referencing equations, figures, and tables using hover-previews [@eg:hover].
This enhances the reading experience of scientific documents and retrieval of information.

::::{prf:example} Hover and Dive Deeper
:label: eg:hover

Any figure, table, or equation can be referenced in MyST and in addition to automated numbering the cross-references have hover-references.
This design feature is important for two reasons: (1) it improves reading comprehension; and (2) it focuses on structured data which can be accessible between papers, creating an open-ecosystem of machine-actionable, reusable content.
The referenced content can also be interactive or computational.

```{figure} ./images/links.*
:label: fig:deep-dive
Instantly accessible information can deep-dive link all the way to interactive figures.
These practices help with reading comprehension by around 26% by providing information when the reader needs it [@doi:10.1145/3411764.3445648].
```

::::

In MyST Markdown we have also extended this "details on demand" concept to abbreviations to make it trivial to disambiguate the meaning of acronyms.
In an analysis of over 18 million articles in @10.7554/eLife.60080, the authors found that the vast majority of abbreviations (79%) appeared fewer than 10 times and many abbreviations had conflicting meanings even in the same discipline.
In MyST Markdown, there is a trivial way to document abbreviations in YAML frontmatter or the project configuration [See @prg:abbr], these are then applied to all instances of that abbreviation in the article or notebooks giving a hover-preview and accessible HTML (e.g. try hovering over these in the online version: JATS, XML, VoR).

```{code-block} yaml
:caption: YAML metadata used in MyST frontmatter to give accessible hovers to all abbreviations with minimal effort for the author. In this case, the acronym `UA` has over 18 distinct meanings in medicine [@10.20316/ESE.2019.45.18018] not to mention other disciplines.
:label: prg:abbr
abbreviations:
  UA: ulnar artery
```

(sec:computational-content)=

### Integrating Computational Content

Beyond structured typography, integrated metadata and hover-previews, MyST Markdown understands computational content and has been integrated with Jupyter [@jupyter].
The goal of this is two fold: (1) allowing for updates to computational content, figures, tables, and calculations to directly update documents; and (2) to bring interactive figures and integrated computation directly into articles.
In the composition of a scientific narrative, scientists often use individual notebooks to create various components of their research (e.g. the preparation of a single figure, table or calculation).
The outputs of these notebooks (a figure, table, or calculation) can then be used in a main, narrative-driven document — a scientific article or presentation.

In some cases, it is possible to collapse all computational information into a single "Computational Article", and visually hide the code to focus on the narrative or presentational flow [c.f. @elife-era].
This approach is appropriate for tutorials or the reproduction of visualizations rather than reproduction of a distinct detailed methodology that requires its own explanation and/or lengthy computation.
However, given the possibility of publishing a Computational Article, even in these cases authors can rethink how to communicate their work and prepare specific visualizations and compact results datasets to take advantage of the format.
Another approach is to include supplemental notebooks that can capture those individual steps [c.f. @10.1029/2023EA003458], and [transclude](wiki:transclude) content [@fig:reuse].
Both approaches are appropriate in different circumstances, and depends on the goal of the communication, nature of the research, speed of execution, and if individual steps require dedicated narrative explanation.

:::{figure} ./images/reuse-jupyter-outputs.png
:label: fig:reuse
A schematic of embedding content from Jupyter Notebooks into an article.
The purple and orange components, interactive figure or other computational outputs, are created in a computational notebook and subsequently used in other narrative or presentation-focused scientific article.
:::

In @fig:reuse, we show an example of reusing computational outputs, such as figures or tables directly in a single computational research article.
By _embedding_ these rather than using a screenshot or copy-paste, any changes to the computational content can be immediately applied on a re-render.
The embedding is completed through a simple MyST Markdown syntax that references a labeled cell in a Jupyter Notebook in, for example, a figure or table [@prg:embed].

```{code-block} markdown
:caption: Embedding a cell from a supplementary notebook directly into a computational document by referencing the label/ID of the cell and adding a caption. The cell in the notebook must be labeled with a `#| label: embedded-cell` as the first line of the content.
:label: prg:embed
:::{figure} #embedded-cell
Additional caption.
:::
```

Similar to the hover-references in @eg:hover, this approach improves the metadata around the notebooks and exposes individual outputs or code-snippets to be imported into other documents or projects.
Additionally, we can attach either static-interactivity (e.g. Plotly, Altair) or dynamic computation (e.g. BinderHub, JupyterHub or JupyterLite) to these figures to run live computations directly in the article [@eg:interactivity].
Here we are aiming at a much richer, structured information commons that moves beyond just tracking scientific metadata towards easy-to-use tools that reuse scientific content.

::::{prf:example} Computational Reproducibility and Interactivity
:label: eg:interactivity
MyST allows for the full reproducible environment to be specified (via REES) and reproduced through tools like MyBinder.
Figures can be integrated directly into articles, pressing a button to launch live and interactive figures that build upon the Jupyter ecosystem.
These tools build on the Jupyter protocols and reuse components from the JupyterLab ecosystem, extending that into various pages using a package called [`thebe`](https://github.com/jupyter-book/thebe).

```{figure} ./images/thebe.*
:label: fig:thebe
Embedded notebook cells with live computation directly in an articles with computation backed by Jupyter.
These can be running on BinderHub or directly in your browser through JupyterLite.
```

::::

### Single Source to Many Outputs

These capabilities of cross-references, typography and embedding visualizations and data-frames are complemented by an single-source export system that supports professional article templates and JATS XML [@fig:export].
This is referred to as single-source publishing [c.f. @10.1109/MITP.2003.1176491], however, many implementations in scientific publishing focus first on manual translation to XML (e.g. from Word or LaTeX), rather than on an author-facing implementation.

:::{figure} ./images/myst-build.png
:label: fig:export
Export to PDF using LaTeX or Typst is supported for hundreds of different journal templates in addition to Microsoft Word or JATS XML, which is used throughout scientific publishing (showing content from @10.1190/tle35080703.1; CC-BY-SA-4.0).
:::

With single-source publishing, we can rely on rich transformations of the source content that can create professional PDFs, interactive HTML, and structured metadata such as JATS XML, which is the current standard for scientific archiving and text-mining.

(sec:publishing)=

## Publishing

There are many ways to publish a MyST Markdown site.
The native way that MyST is represented is through content-as-data, a collection of JSON files that represent the full documents and all associated metadata (if you are reading this online, you can add a `.json` to the URL to access the content).
This MyST content may be served alongside an independent website theme, which dynamically creates HTML based on these JSON files (very similar to XML-based single source publishing [@10.1109/MITP.2003.1176491], but using modern web-tooling).
This server-side approach has a number of advantages, in that it allows you to maintain a journal/site theme, without having to upgrade or rebuild all content as would be required by static tooling.
It also puts content addressability as a first class concern (via the `.json`), enabling global cross referencing and opening up opportunities for varied publishing models in future.
This is the approach that we take with Curvenote journals, and provide managed services to maintain journal sites, manage and curate content, as well as provide the editorial management tools and workflows.

You can also easily self-host a MyST Markdown site using GitHub Pages.
To create a GitHub pages output you can run `myst init --gh-pages`, which will walk you through the steps of creating a GitHub action to publish your content as a static website.
In this scenario, a static HTML site is built from your content which can be hosted as any other static website, while some of the advantages of dynamic hosting are lost, it is an easy and accessible way for individuals to self-publish.

In 2024, Curvenote was asked to improve our integrations to GitHub to support the SciPy Proceedings and re-imagine a MyST based publishing approach that uses GitHub for open-peer-review, implementing a submission, editorial and peer review process with GitHub issues, PRs and actions as a fabric.
The process previously used technology shared with the Journal of Open Source Software (JOSS), which popularized this approach [@10.7717/peerj-cs.147].
The workflow was updated to use MyST Markdown for the authoring process[^also-latex] (previously RST and LaTeX were supported, and the build process was in Sphinx), and the submission process now uses the [Curvenote CLI](https://github.com/curvenote/curvenote) in combination with dedicated Curvenote GitHub Actions to build, check, and preview the content of each commit, using GitHub workflows to automate the process, providing immediate feedback for authors and the conference editorial team.

[^also-latex]: LaTeX is also supported by parsing and rendering directly with the `mystmd` CLI, this is completed through [`@unified-latex`](https://github.com/siefkenj/unified-latex).

(sec:checks)=

### Structural Checks

The open-source Curvenote CLI (https://github.com/curvenote/curvenote) provides checks for the structure of a document to ensure it meets automated quality controls for things like references, valid links, author identifiers (e.g. ORCID), or funding information.
Executing `curvenote check` in a MyST project will build the project and use the structured data to assess metadata (e.g. do authors provide ORCIDs), structural checks (e.g. does the article have an abstract?; is the article below a word count?), and check that references have valid DOIs.

We have designed these checks in a similar pattern to linting and/or unit tests in scientific software, which continually give feedback on the structural health of a codebase with near immediate feedback to authors [@eg:checks].
Authors can take action to improve their metadata directly, by including DOIs, CRediT Roles, ORCIDs, and structural checks such as word count or missing sections.
For example, in the SciPy Proceedings in 2024, which used Curvenote checks for their submission system, required and optional metadata were improved by authors reacting to these automated checks without any intervention from the proceedings editorial team (e.g. [](https://github.com/scipy-conference/scipy_proceedings/pull/915) added CRediT roles, ORCIDs abbreviations, and DOIs to get a passing check [See @fig:actions]).
This is a low-friction way of improving metadata at the time of authoring or submission to elevate content to the standards that are required for a certain type of publication (e.g. proceedings vs. blog post vs. peer-reviewed journal).

::::{prf:example} Structural Checks and Metadata Checks
:label: eg:checks

Specific checks can be setup for any kind of content being submitted with MyST, for example, a "Short Report" might have different length requirements or metadata standards than a "Research Article" or "Editorial".
Using Curvenote, these checks can be configured for a specific venue or kind of article, for example, to check specifically for SciPy Proceedings articles, `curvenote check scipy --kind Article --collection 2024` can be run, where the list of checks is configured remotely by journal administrators.

```{figure} ./images/checks.*
:label: fig:checks
Running `curvenote check` on a SciPy Proceedings article to check for missing DOIs, author ORCIDs, word-count and specific sections (e.g. abstract).
These can be run in less than a few seconds both locally and through GitHub actions [@fig:actions], which provides a user interface on the checks.
```

::::

(sec:automation)=

### Automated Actions

Upon submission of there are a number of checks that are run and a submission or draft is deployed to Curvenote's platform, which stores the structured MyST project.
We utilized GitHub Actions to automate initial checks and generate previews of submissions (via https://github.com/curvenote/actions).
The actions automatically generate a preview of the manuscript exactly as it would appear in publication using MyST Markdown, and these are linked within the pull request comments for easy access by reviewers [@fig:actions].

```{figure} ./images/gh-commits-and-checks.png
:label: fig:actions
:class: framed
An example of a comment by a GitHub action, which shows the checks and preview of the article directly.
The checks in this example have promoted the author to improve metadata, see [](https://github.com/scipy-conference/scipy_proceedings/pull/911).
```

### Enabling Researchers

Although SciPy submissions are made in a central repository, the Curvenote checks can be setup on any repository.
As MyST Markdown brings a researcher's writing back into a simple text-based format, and if they are working with Jupyter notebooks, it integrates directly with their computational analysis and results.
A researcher's manuscript can be easily created and maintained as an integral part of their research code base and updated in lock step with it.
Imagine writing your next paper for submission to SciPy in a way where your draft was automatically rebuilt with the latest figures and data tables on every push to `main`, and any issues that you would hit on submission are flagged as they happen and so are more easily fixed, in context.
This is just one of the ways that continuous science practices we spoke about in @sec:continuous-science can lead to radically improved efficiencies at scale.

Curvenote's tools build on this theme of enabling researchers to do more and to have more visibility into how their work will appear when submitted and published.
The previews generated during the journal submission process, or by the actions on a researchers own repo can also be generated by an author at any time.
By running `curvenote submit <journal> --draft` researchers can get a set of check results and a preview in the style of that venue, and where submissions of `Computational Articles` are permitted, authors and reviewers will even get feedback on reproducible environment check that any interactive figures and notebooks execute as expected.

### Use Cases

The rapid feedback in authoring [@sec:authoring] coupled with structural checks [@sec:checks], automation [@sec:automation], and archiving of the content opens workflows for varied sizes of teams to adopt these _continuous science_ practices.
For example, individuals publishing static MyST Markdown sites, lab-groups creating a better archive to highlight the research contributions of their team (e.g. [Applied Geophysics](https://appliedgeophysics.org)), conference proceedings (e.g. [SciPy Proceedings](https://proceedings.scipy.org)), or more formalized society journals (e.g. [Physiome](https://journal.physiomeproject.org), [American Geophysical Union](https://agu.curve.space), [Elemental Microscopy](https://elementalmicroscopy.com/)) [@fig:use-cases].

:::{figure} images/use-cases.png
:label: fig:use-cases
:width: 90%
MyST Markdown and Curvenote tools can help lower the barrier to entry for scientific publishing workflows for journals, research institutes, conferences, private consortiums, universities, and lab groups.
:::

## Conclusions

Scientific publishing infrastructure is disconnected from the day-to-day workflows of scientists, and at scale this slows the progress of science.
These misalignments are particularly pronounced in computational disciplines, where rapid evolution of methodologies, software, and data demands equally dynamic and interconnected platforms.
This gap — between the authoring/doing of research and the communicating/publishing of the research — slows the speed of research dissemination, reuse, and uptake and completely impedes "networked knowledge" and importing/reusing work in a structured way.
For example, "importing" visualizations, equations or any other deeply-linked content – including provenance information – into new research articles, documentation or educational sites is completely impossible in today’s research ecosystem.
As a metaphor, compare open-access science to open-source programming: it would be a world without package managers to share, version, reuse, and rapidly build upon other peoples work in a structured way.
The open-source ecosystem would not exist without this infrastructure.

Open infrastructure for communicating science also has to be easy to integrate into existing tools, support computational, interactive components, be archivable for the long term, and be adopted by our existing sociotechnical system of societies, journals, and institutions.
There are two interconnected problems that need to be solved: (1) upgrade existing scientific authoring tools, ensuring these are integrated into both scientific and data-science ecosystems; and (2) develop radically better ways to share content as individuals, small groups, preprints, and formalized, traditional journals with existing societies and institutions.
The two problems are connected, in that the authoring tools should be able to deeply integrate with publishing mediums (e.g. referencing a figure from a publication should be able to show you that figure directly as you are authoring, including all interactivity and computation).

In this article we have presented some of the goals behind features of MyST Markdown and Curvenote, to support new open-science infrastructure including authoring tools as well as publishing workflows that support checks and automation.
To support every next-generation research tool on top of open access knowledge, we need access to **high-quality, structured content**, data and software — not just the scholarly metadata and citation graph.
Our goal of contributing to MyST Markdown is to make the processes behind creating this structured data more accessible and affordable.
These tools in the hands of researchers can also enable process changes and **continuous science** practices: where checking and automation can support rapid iterations and feedback.
The analogies between continuous delivery of software and continuous science give us an opportunity to peek ahead a decade to an analogous future and draw on many learnings on how to organize and focus infrastructure to get the best out of our scientific community.

There is, of course, an _enormity_ of work ahead of these tools to transform science publishing at scale.
We are grateful to our society partners who are changing community practices around publishing to support HTML-first publishing, experimenting with computational articles, and implementing new peer-review workflows.
Tools on their own do not make change, but can help to enable it.
Improvements to scientific publishing require many diverse community efforts to improve the quality and speed of how we communicate knowledge, and ultimately to accelerate scientific progress.

## Acknowledgements

Portions of this manuscript have been previously shared in the Curvenote and MyST Markdown documentation.
The authors would like to thank Lindsey Heagy, Arfon Smith, Chris Holdgraf, Greg Caporaso, Jim Colliander, Fernando Perez, J.J. Allaire, and Kristen Ratan who have provided input on versions of these ideas over the last few years.
We would also like to thank the reviews Nate Jacobs, Angus Hollands, Lindsey Heagy, Stefan van der Walt, Andy Terrel, and Hongsup Shin who helped improved the manuscript.
Funding for portions of this work have come from the Sloan Foundation (for MyST Markdown and _Notebooks Now!_) and Alberta Innovates (for the initial version of the Curvenote CLI, which became `mystmd`).
Thank you to the growing list of MyST Markdown contributors who continue to make MyST a fantastic community project.
