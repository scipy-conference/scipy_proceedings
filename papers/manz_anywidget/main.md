---
# Ensure that this title is the same as the one in `myst.yml`
title: 'Anywidget: easily author and share reusable interactive widgets for computational notebooks'
abstract: |
  Computational notebooks have become the programming environment of choice for data scientists. The open-source Jupyter Project has fostered a robust ecosystem around notebook-based computing, which has also led to a proliferation of Jupyter-compatible platforms. In Jupyter, widgets serve as a powerful component system to enable custom views and controls within notebooks that interact with data and notebook artifacts bidirectionally. However, they are a less modular component of Jupyter’s architecture. Due to the coupling of widgets to platform-specific internals and the portability challenges this brings, widget authorship and distribution have evolved into complex processes, to the frustration of developers and end users alike. Anywidget addresses these challenges by introducing a standard for widget front-end code that is based on the web browser’s native module system and is decoupled from notebook runtime dependencies. Anywidget serves as an adapter to ensure cross-platform compatibility, loading front-end modules from the notebook kernel, just like complementary Python code, rather than from independent sources. This design simplifies the authorship and sharing of widgets, consolidates widget publishing, enables rapid prototyping, and removes unnecessary installation steps, thereby enhancing overall developer and user experiences. Anywidget also lowers the barrier to entry for widget authorship, leading to a more diverse and robust widget ecosystem. The adoption of anywidget has already sparked a widget renaissance and improved widget reusability and interoperability, making interactive computing more accessible and efficient.
---

## Introduction

The Jupyter Notebook interface is the _de facto_ standard for interactive
computing, combining live code, equations, prose, visualizations, and other
media within a single environment [@Perez2007-im; @Kluyver2016-xa;
@Granger2021-jb]. Key to Jupyter's widespread adoption are its modular
architecture and standardization of interacting components, which have fostered
an extensive ecosystem of tools that reuse these elements. For example, the
programs responsible for executing code written in notebooks, called
**kernels**, can be implemented by following the Jupyter Messaging Protocol
[@jupmsgprotocol]. This design allows users to install kernels for various
different languages and types of computation. Similarly, Jupyter's
open-standard notebook format (`.ipynb`) ensures that notebooks can be shared
and interpreted across different platforms [@jupnbformat].

Jupyter’s modular architecture has also supported innovation in **notebook
front ends** — the user interfaces (UIs) for editing and executing code, as
well as inspecting kernel outputs. The success of the classic Jupyter Notebook
web-based UI, offering easy installation and the ability to connect to both
local and remote kernels, spurred the development of several similar
Jupyter-compatible platforms (JCPs), such as JupyterLab, Google Colab, and
Visual Studio Code. These platforms provide unique UIs and editing features
while reusing Jupyter's other standardized components. This interoperability
allows users to choose the platform that best suits their needs, while
retaining a familiar interactive computing experience and the ability to share
notebooks. The separation of computation from UI has largely benefited end
users by offering a wider selection of both front ends and kernels. However,
the proliferation of JCPs has led to significant challenges for one particular
component of Jupyter: Jupyter Widgets.

Jupyter Widgets extend notebook outputs with interactive views and controls for
objects residing in the kernel. For instance,
[ipywidgets](https://github.com/jupyter-widgets/ipywidgets) provides basic form
elements like buttons, sliders, and dropdowns to adjust individual variables.
Other community projects offer interactive visualizations for domain-specific
needs, such as 3D volume rendering
([ipyvolume](https://github.com/widgetti/ipyvolume)), genome browsing
([higlass-python](https://github.com/higlass/higlass-python)), and mapping
([ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet),
[lonboard](https://developmentseed.org/lonboard/)), which users can update by
executing other code cells or interact with in the UI to update properties in
the kernel. Widgets are unique among Jupyter components in that they consist of
two separate programs — kernel-side code and front-end code — that communicate
directly via custom messages [@fig:overview], rather than through a mediating
Jupyter process. With widgets, communication is bidirectional: a kernel action
(e.g. the execution of a notebook cell) can update the UI, such as causing a
slider to move, while a user interaction (e.g. dragging a slider), can drive
changes in the kernel, like updating a variable. This two-way communication
distinguishes widgets from other interactive elements in notebook outputs, such
as HTML displays, which cannot communicate back and forth with the kernel.

:::{figure} widget-lifecycle.png
:label: fig:overview
Jupyter Widget conceptual architecture.
:::

Widgets are designed to be pluggable components, similar to kernels. However,
only the protocol for communication between kernel and front-end widget code,
known as the [Jupyter Widgets Message
Protocol](https://github.com/jupyter-widgets/ipywidgets/blob/main/packages/schema/messages.md),
is standardized. Critical components, such as the distribution format for
front-end modules and methods for discovering, loading, and executing these
modules, remain unspecified. As a result, JCPs have adopted diverse third-party
module formats, installation procedures, and execution models. These
inconsistencies place the onus on widget authors to ensure cross-platform
compatibility.

JCPs load front-end widget code by searching in various external sources, such
as local file systems or Content Distribution Networks (CDNs) while kernel-side
(Python) code loads and runs in the kernel [@fig:before-afm]. This splits the
distribution of custom widgets between Python and JavaScript package
registries, complicating releases and requiring widget authors to understand
both packaging ecosystems. Moreover, this divided system is incompatible with
shared, multi-user environments like [JupyterHub](https://jupyter.org/hub),
where users can only install kernel-side widget code in their custom kernels.
Since front-end widget code must be installed separately by an administrator
into the shared environment, widgets often fail to work when the complementary
front-end code has not been installed or does not match with the user’s
kernel-side widget code.

:::{figure} dev-before-afm-01.png
:label: fig:before-afm
Without anywidget, widget authors must transform their widget JavaScript code for each JCP to ensure compatibility, and distribute and install front-end code separately from kernel-side Python code.
:::

These limitations make widget development complex and time-consuming, demanding
expertise in multiple domains. They make user experiences across JCPs
frustrating and unreliable. The high barrier to entry discourages new
developers and domain scientists from contributing to widgets, limiting growth
and diversity in the ecosystem. This leaves a small group of authors
responsible for adapting their code for cross-platform compatibility, hindering
widget reliability and maintainability.

## Methodology

Anywidget simplifies the authoring, sharing, and distribution of Jupyter
Widgets by (i) introducing a standard for widget front-end code based on the
web browser’s native module system, (ii) loading these modules from the kernel,
and (iii) providing the necessary "glue code" to adapt existing JCPs to load
and execute these components [@fig:after-afm]. This separation of concerns
allows widget authors to write portable code that runs consistently across JCPs
without manual installation steps.

:::{figure} dev-after-afm-02.png
:label: fig:after-afm
Anywidget simplifies widget authorship and sharing and ensures cross-platform
compatibility. With anywidget, developers author a single, standard portable ES
module (AFM), which is loaded from the kernel and executed using the browser's
native module system. For existing JCPs, anywidget provides a front-end adapter
to load and execute these standardized modules, while new platforms can add
native AFM support directly. Widget kernel-side code and AFM can be run
directly from within notebooks, from source files, or distributed as single
Python packages.
:::

Packaging custom Jupyter Widgets is complex due to the need to adapt JavaScript
source code for various module systems used by JCPs. Initially, JavaScript
lacked a built-in module system, leading to diverse third-party solutions
adopted by JCPs. Without a standardized widget front-end format, authors
compile their code for each JCP. In the context of Jupyter Notebook and
JupyterLab, this problem is described in the Jupyter Widgets documentation
[@doc_widgets] as follows:

> Because the API of any given widget must exist in the kernel, the kernel is
> the natural place for widgets to be installed. However, kernels, as of now,
> don’t host static assets. Instead, static assets are hosted by the webserver,
> which is the entity that sits between the kernel and the front-end. This is a
> problem because it means widgets have components that need to be installed
> both in the webserver and the kernel. The kernel components are easy to
> install, because you can rely on the language’s built-in tools. The static
> assets for the webserver complicate things, because an extra step is required
> to let the webserver know where the assets are.

ECMAScript (ES) modules, introduced in 2015, are an official standard for
packaging JavaScript code for reuse [@rojas2021modules]. While most JCPs
predate its standardization, ES modules are universally supported by browsers
today. By adopting ES modules, anywidget is able to use the browser's native
import mechanism to load and execute widget front-end code from the Jupyter
kernel, thereby bypassing JCP import systems and eliminating third-party
dependencies. This approach not only overcomes many development challenges, it
also eliminates installation procedures for front-end code. Consequently,
developers can prototype and share widgets directly within notebooks, making
them more reliable and easier to use across JCPs.

An anywidget front-end module (AFM) is an ES module with a `default` export
defining widget behavior. This export includes lifecycle methods, or "hooks,"
for managing a widget's lifecycle stages: initialization, rendering, and
destruction [@fig:afm]. AFM lifecycle methods receive the interfaces required
for kernel communication and notebook output modifications as arguments, rather
than creating them internally or relying on global variables provided by the
JCP. This practice, known as dependency injection, improves AFM portability by
making integration interfaces explicit. New runtimes can support AFMs by
implementing the required APIs, and existing JCPs can refactor their internals
without breaking existing (any)widgets.

:::{figure}
:label: fig:afm
```{code-block}javascript
export default {
  initialize({ model }) {
    // Set up shared state or event handlers.
    return () => {
      // Optional: Called when the widget is destroyed.
    } 
  },
  render({ model, el }) {
    // Render the widget's view into the el HTMLElement.
    return () => {
      // Optional: Called when the view is destroyed.
    } 
  }
}
```
Stub of an anywidget front-end module (AFM) with initialization and rendering
lifecycle methods.
:::

## Features

Adhering to predictable standards benefits both developers and end users in
many other ways beyond cross-platform interoperability.

**Web Over Libraries.** Front-end libraries change rapidly and often introduce
breaking changes, whereas the web platform remains more backward-compatible.
Traditional Jupyter Widgets require extensions from UI libraries provided by
JCPs, coupling widget implementations to particular third-party frameworks. In
contrast, AFM defines a minimal set of essential interfaces focused on (1)
communicating with the kernel and (2) modifying notebook output cells, without
dictating state or UI models. This approach allows widgets to be defined
without dependencies, reducing boilerplate and preventing lock-in. While
authors are free to incorporate third-party JavaScript tooling or frameworks to
enhance their own widgets or boost their productivity, importantly, no such
tools are needed for JCP compatibility, user installation, or publishing.

**Rapid Iteration.** The web ecosystem's adoption of ES modules has led to new
technologies that enhance developer experience and enable rapid prototyping.
One such innovation is hot module replacement (HMR), a method that uses the
browser’s module graph to dynamically update applications without reloading the
page or losing state. Since traditional Jupyter Widgets rely on legacy module
systems, they cannot benefit from HMR and instead require full page clearance,
reload, and re-execution to see changes during development. By contrast,
anywidget is able to provide opt-in HMR, implemented through the Jupyter
messaging protocol, in order to support live development of custom widgets
without any front-end tooling. For example, developers can adjust a widget's
appearance, like a chart's color scheme, without losing its data or needing a
page refresh.

**Progressive Development.** Anywidget makes it possible to prototype widgets
directly within a notebook since all widget code is loaded from the kernel.
Custom widgets can start as a few code cells and transition to separate files,
gradually evolving into standalone scripts or packages – just like kernel-side
programs [@fig:solution b]. In contrast, developing traditional Jupyter Widgets
is a cumbersome process limited to the Jupyter Notebook and JupyterLab
platforms. It involves using a project generator [@js_cookiecutter;
@ts_cookiecutter] to bootstrap a project with over 50 files, creating and
installing a local Python package with custom-built extensions, compiling
JavaScript code, and manually linking build assets to install extensions. By
removing these barriers, anywidget accelerates development, and allows
prototypes to grow into robust tools over time.

**Simplified Publishing.** Serving AFMs and other static assets from the kernel
removes the need to publish widget kernel-side and front-end code separately
and coordinate their releases. For example, many JCPs retrieve traditional
widget Javascript code from the npm registry, misusing the registry for
distributing specialized programs rather than reusable JavaScript modules.
Instead, with anywidget, developers can publish a widget (kernel-side module,
AFM, and stylesheets) as a unified package to the distribution channels
relevant to the kernel language, such as the Python Package Index (PyPI).
Consolidating the distribution process this way greatly simplifies publishing
and discovery.

## Impact and Outlook

Anywidget fills in the specification gaps for Jupyter Widgets by embracing open
standards and carefully separating developer concerns. It defines an API for
authoring portable and reusable widget components that decouples widget
authorship from JCP runtimes, resulting in multiple downstream benefits. First,
anywidget—not widget authors—ensures compatibility and interoperability across
existing JCPs, and authors can focus on important features rather than wrestle
with build configuration and tooling. Second, by circumventing bespoke JCP
import systems and loading web-standard ES modules from the kernel, anywidget
does away with front-end installation steps and delivers a superior developer
experience to widget authorship. Third, anywidget unifies and simplifies widget
distribution. Widgets can be prototyped and shared as notebooks, or mature into
pip-installable packages and distributed just like other tools in the Python
data science ecosystem. End users benefit from standardization because widgets
are easy to install and behave consistently across different platforms.

Since its release, anywidget has led to a proliferation of widgets and a more
diverse widget ecosystem [@fig:widgetstats]. New widgets range from educational
tools for experimenting with toy datasets (e.g.,
[DrawData](https://drawdata.xyz/)) to high-performance data visualization
(e.g., [Lonboard](https://developmentseed.org/lonboard/latest/),
[Mosaic](https://uwdata.github.io/mosaic/jupyter/),
[Jupyter-Scatter](https://jupyter-scatter.dev/)). These visualization tools
leverage anywidget’s support for binary data transport, enabling efficient
interactive visualization with minimal serialization overhead. Existing widget
projects have also migrated to anywidget
([higlass-python](https://github.com/higlass/higlass-python),
[ipyaladin](https://github.com/cds-astro/ipyaladin))  and other libraries have
introduced or refactored existing widget functionality to use anywidget
([Altair](https://altair-viz.github.io/)) due to the simplified distribution
and authoring capabilities.

:::{figure} widgetstats.png
:label: fig:widgetstats
Custom Jupyter Widgets per year as of May 30, 2024. Date for each project is
the initial commit or the date of the commit when a widget was added to the
repository project. Projects are tracked at
https://github.com/manzt/anywidget-usage.
:::

The portable widget standard also opens avenues for other emerging
notebook-inspired platforms to make use of the widget ecosystem. For example,
Marimo, a new reactive Python notebook that automatically runs dependent cells,
supports AFM without needing extensions, allowing anywidgets to run natively
without additional "glue code." Additionally, Panel is exploring a version of
authoring widgets based on AFM to unify standards for data app component
frameworks.

Anywidget’s approach allows widgets to serve a wider range of stakeholders. Most
end users do not need to understand kernel-web communication details when using
widgets. For example, data scientists can install interactive visualizations,
link multiple widget instances together, and create custom views and controls
for their analysis by writing kernel code and executing cells. However,
anywidget also makes tinkering, learning, and exploring the front-end and
kernel-web communication layers more approachable for potential widget authors,
especially non-web developers. By focusing on web standards, the skills learned
are transferable to general front-end development. Similarly, anywidget-powered
notebooks provide an ideal environment for evaluating data visualization designs
and fostering collaboration between data science and visualization teams.
Instead of investing time and resources into complex standalone web
applications, anywidget makes it easy to embed high-performance visualizations
directly into analytical environments, thereby integrating into existing
workflows and reusing data structures.

One of the stated goals of the Jupyter Notebook is to minimize the “distance”
between user and data, and widgets play a key role by allowing users to
customize the way they view and manipulate data in the kernel through a UI.
Anywidget advances this goal by removing the primary sources of friction
associated with widget development and sharing. By making widget authorship
practical and accessible, anywidget also helps narrow the distance between data
practitioner and developer, and between machine learning and visualization
experts.

## Acknowledgements

We thank ...

## Funding

TM, NG, and NA acknowledge funding from the NIH Common Fund 4D Nucleome Program
(UM1 HG011536).


<!--
## Introduction

Twelve hundred years ago — in a galaxy just across the hill...

This document should be rendered with MyST Markdown [mystmd.org](https://mystmd.org),
which is a markdown variant inspired by reStructuredText. This uses the `mystmd`
CLI for scientific writing which can be [downloaded here](https://mystmd.org/guide/quickstart).
When you have installed `mystmd`, run `myst start` in this folder and
follow the link for a live preview, any changes to this file will be
reflected immediately.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
justo vitae odio tincidunt tempus at vitae tortor.

## Bibliographies, citations and block quotes

Bibliography files and DOIs are automatically included and picked up by `mystmd`.
These can be added using pandoc-style citations `[@doi:10.1109/MCSE.2007.55]`
which fetches the citation information automatically and creates: [@doi:10.1109/MCSE.2007.55].
Additionally, you can use any key in the BibTeX file using `[@citation-key]`,
as in [@hume48] (which literally is `[@hume48]` in accordance with
the `hume48` cite-key in the associated `mybib.bib` file).
Read more about [citations in the MyST documentation](https://mystmd.org/guide/citations).

If you wish to have a block quote, you can just indent the text, as in:

> When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication.
>
> -- @hume48

Other typography information can be found in the [MyST documentation](https://mystmd.org/guide/typography).

### DOIs in bibliographies

In order to include a DOI in your bibliography, add the DOI to your bibliography
entry as a string. For example:

```{code-block} bibtex
:emphasize-lines: 7
:linenos:
@book{hume48,
  author    =  "David Hume",
  year      = {1748},
  title     = "An enquiry concerning human understanding",
  address   = "Indianapolis, IN",
  publisher = "Hackett",
  doi       = "10.1017/CBO9780511808432",
}
```

### Citing software and websites

Any paper relying on open-source software would surely want to include citations.
Often you can find a citation in BibTeX format via a web search.
Authors of software packages may even publish guidelines on how to cite their work.

For convenience, citations to common packages such as
Jupyter [@jupyter],
Matplotlib [@matplotlib],
NumPy [@numpy],
pandas [@pandas1; @pandas2],
scikit-learn [@sklearn1; @sklearn2], and
SciPy [@scipy]
are included in this paper's `.bib` file.

In this paper we not only terraform a desert using the package terradesert [@terradesert], we also catch a sandworm with it.
To cite a website, the following BibTeX format plus any additional tags necessary for specifying the referenced content is recommended.
If you are citing a team, ensure that the author name is wrapped in additional braces `{Team Name}`, so it is not treated as an author's first and last names.

```{code-block} bibtex
:emphasize-lines: 2
:linenos:
@misc{terradesert,
  author = {{TerraDesert Team}},
  title  = {Code for terraforming a desert},
  year   = {2000},
  url    = {https://terradesert.com/code/},
  note   = {Accessed 1 Jan. 2000}
}
```

## Source code examples

No paper would be complete without some source code.
Code highlighting is completed if the name is given:

```python
def sum(a, b):
    """Sum two numbers."""

    return a + b
```

Use the `{code-block}` directive if you are getting fancy with line numbers or emphasis. For example, line-numbers in `C` looks like:

```{code-block} c
:linenos: true

int main() {
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
    return 0;
}
```

Or a snippet from the above code, starting at the correct line number, and emphasizing a line:

```{code-block} c
:linenos: true
:lineno-start: 2
:emphasize-lines: 3
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
```

You can read more about code formatting in the [MyST documentation](https://mystmd.org/guide/code).

## Figures, Equations and Tables

It is well known that Spice grows on the planet Dune [@Atr03].
Test some maths, for example $e^{\pi i} + 3 \delta$.
Or maybe an equation on a separate line:

```{math}
g(x) = \int_0^\infty f(x) dx
```

or on multiple, aligned lines:

```{math}
\begin{aligned}
g(x) &= \int_0^\infty f(x) dx \\
     &= \ldots
\end{aligned}
```

The area of a circle and volume of a sphere are given as

```{math}
:label: circarea

A(r) = \pi r^2.
```

```{math}
:label: spherevol

V(r) = \frac{4}{3} \pi r^3
```

We can then refer back to Equation {ref}`circarea` or
{ref}`spherevol` later.
The `{ref}` role is another way to cross-reference in your document, which may be familiar to users of Sphinx.
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas[^footnote-1] diam turpis, placerat[^footnote-2] at adipiscing ac,
pulvinar id metus.

[^footnote-1]: On the one hand, a footnote.
[^footnote-2]: On the other hand, another footnote.

:::{figure} figure1.png
:label: fig:stream
This is the caption, sandworm vorticity based on storm location in a pleasing stream plot. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/arrays/streamplot.html).
:::

:::{figure} figure2.png
:label: fig:em
This is the caption, electromagnetic signature of the sandworm based on remote sensing techniques. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/stats/hist2d.html).
:::

As you can see in @fig:stream and @fig:em, this is how you reference auto-numbered figures.
To refer to a sub figure use the syntax `@label [a]` in text or `[@label a]` for a parenhetical citation (i.e. @fig:stream [a] vs [@fig:stream a]).
For even more control, you can simply link to figures using `[Figure %s](#label)`, the `%s` will get filled in with the number, for example [Figure %s](#fig:stream).
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

```{list-table} This is the caption for the materials table.
:label: tbl:materials
:header-rows: 1
* - Material
  - Units
* - Stone
  - 3
* - Water
  - 12
* - Cement
  - {math}`\alpha`
```

We show the different quantities of materials required in
@tbl:materials.

Unfortunately, markdown can be difficult for defining tables, so if your table is more complex you can try embedding HTML:

:::{table} Area Comparisons (written in html)
:label: tbl:areas-html

<table>
<tr><th rowspan="2">Projection</th><th colspan="3" align="center">Area in square miles</th></tr>
<tr><th align="right">Large Horizontal Area</th><th align="right">Large Vertical Area</th><th align="right">Smaller Square Area<th></tr>
<tr><td>Albers Equal Area   </td><td align="right"> 7,498.7   </td><td align="right"> 10,847.3  </td><td align="right">35.8</td></tr>
<tr><td>Web Mercator        </td><td align="right"> 13,410.0  </td><td align="right"> 18,271.4  </td><td align="right">63.0</td></tr>
<tr><td>Difference          </td><td align="right"> 5,911.3   </td><td align="right"> 7,424.1   </td><td align="right">27.2</td></tr>
<tr><td>Percent Difference  </td><td align="right"> 44%       </td><td align="right"> 41%       </td><td align="right">43%</td></tr>
</table>
:::

or if you prefer LaTeX you can try `tabular` or `longtable` environments:

```{raw} latex
\begin{table*}
  \begin{longtable*}{|l|r|r|r|}
  \hline
  \multirow{2}{*}{\bf Projection} & \multicolumn{3}{c|}{\bf Area in square miles} \\
  \cline{2-4}
   & \textbf{Large Horizontal Area} & \textbf{Large Vertical Area} & \textbf{Smaller Square Area} \\
  \hline
  Albers Equal Area   & 7,498.7   & 10,847.3  & 35.8  \\
  Web Mercator        & 13,410.0  & 18,271.4  & 63.0  \\
  Difference          & 5,911.3   & 7,424.1   & 27.2  \\
  Percent Difference  & 44\%      & 41\%      & 43\%  \\
  \hline
  \end{longtable*}

  \caption{Area Comparisons (written in LaTeX) \label{tbl:areas-tex}}
\end{table*}
```

Perhaps we want to end off with a quote by Lao Tse[^footnote-3]:

> Muddy water, let stand, becomes clear.

[^footnote-3]: $\mathrm{e^{-i\pi}}$ -->
