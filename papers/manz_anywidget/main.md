---
# Ensure that this title is the same as the one in `myst.yml`
title: 'Anywidget: easily author and share reusable interactive widgets for computational notebooks'
abstract: |
  Computational notebooks have become the programming environment of choice for data scientists. The open-source Jupyter Project has fostered a robust ecosystem around notebook-based computing, which has also led to a proliferation of Jupyter-compatible platforms. In Jupyter, widgets serve as a powerful component system to enable custom views and controls within notebooks that interact with data and notebook artifacts bidirectionally. However, they are a less modular component of Jupyter’s architecture. Due to the coupling of widgets to platform-specific internals and the portability challenges this brings, widget authorship and distribution have evolved into complex processes, to the frustration of developers and end users alike. Anywidget addresses these challenges by introducing a standard for widget front-end code that is based on the web browser’s native module system and is decoupled from notebook runtime dependencies. Anywidget serves as an adapter to ensure cross-platform compatibility, loading front-end modules from the notebook kernel, just like complementary Python code, rather than from independent sources. This design simplifies the authorship and sharing of widgets, consolidates widget publishing, enables rapid prototyping, and removes unnecessary installation steps, thereby enhancing overall developer and user experiences. Anywidget also lowers the barrier to entry for widget authorship, leading to a more diverse and robust widget ecosystem. The adoption of anywidget has already sparked a widget renaissance and improved widget reusability and interoperability, making interactive computing more accessible and efficient.
---

## Introduction

The Jupyter Notebook interface is the _de facto_ standard for interactive
computing, combining live code, equations, prose, visualizations, and other
media within a single environment [@Perez2007-im; @Kluyver2016-xa; @Granger2021-jb]. Key to
Jupyter's widespread adoption are its modular architecture and standardization
of interacting components, which have fostered an extensive ecosystem of tools
that reuse these elements. For example, the programs responsible for executing
code written in notebooks, called **kernels**, can be implemented by following
the Jupyter Messaging Protocol [@doc_jupmsgprotocol]. This
design allows users to install kernels for various different languages and types
of computation. Similarly, Jupyter's open-standard notebook format (`.ipynb`)
ensures that notebooks can be shared and interpreted across different platforms [@doc_jupnbformat].

Jupyter’s modular architecture has also supported innovation in **notebook front
ends** — the user interfaces (UIs) for editing and executing code, as well as
inspecting kernel outputs. The success of the classic Jupyter Notebook web-based
UI, offering easy installation and the ability to connect to both local and
remote kernels, spurred the development of several similar Jupyter-compatible
platforms (JCPs), such as JupyterLab, Google Colab, and Visual Studio Code.
These platforms provide unique UIs and editing features while reusing Jupyter's
other standardized components. This interoperability allows users to choose the
platform that best suits their needs, while retaining a familiar interactive
computing experience and the ability to share notebooks. The separation of
computation from UI has largely benefited end users by offering a wider
selection of both front ends and kernels. However, the proliferation of JCPs has
led to significant challenges for one particular component of Jupyter: Jupyter
Widgets.

Jupyter Widgets extend notebook outputs with interactive views and controls for
objects residing in the kernel [@doc_juparch]. For instance, [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) provides basic form
elements like buttons, sliders, and dropdowns to adjust individual variables.
Other community projects offer interactive visualizations for domain-specific
needs, such as 3D volume rendering ([ipyvolume](https://github.com/widgetti/ipyvolume)), genome browsing
([higlass-python](https://github.com/higlass/higlass-python)), and mapping ([ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet), [lonboard](https://developmentseed.org/lonboard/)), which users can update by
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

Widgets are intended to be pluggable components, similar to kernels. However,
only the protocol for communication between kernel and front-end widget code,
known as the [Jupyter Widgets Message
Protocol](https://github.com/jupyter-widgets/ipywidgets/blob/main/packages/schema/messages.md),
is standardized. Critical components, such as the distribution format for
front-end modules and methods for discovering, loading, and executing these
modules, remain unspecified. As a result, JCPs have adopted diverse third-party
module formats, installation procedures, and execution models to support widgets. These
inconsistencies place the onus on widget authors to ensure cross-platform
compatibility.

JCPs load front-end widget code by searching in various external sources, such
as local file systems or Content Distribution Networks (CDNs) while kernel-side
(Python) code loads and runs in the kernel [@fig:before-afm]. This splits the distribution of
custom widgets between Python and JavaScript package registries, complicating
releases and requiring widget authors to understand both packaging ecosystems.
Moreover, this divided system is incompatible with shared, multi-user
environments like [JupyterHub](https://jupyter.org/hub), where users can only install kernel-side widget
code in their custom kernels. Since front-end widget code must be installed
separately by an administrator into the shared environment, widgets often fail
to work when the complementary front-end code has not been installed or does not
match with the user’s kernel-side widget code.

:::{figure} dev-before-afm-01.png
:label: fig:before-afm
Without anywidget, widget authors must transform their widget JavaScript code for each JCP to ensure compatibility, and distribute and install front-end code separately from kernel-side Python code.
:::

These limitations make widget development complex and time-consuming, demanding
expertise in multiple domains. They make user experiences across JCPs
frustrating and unreliable. The high barrier to entry discourages new developers
and domain scientists from contributing to widgets, limiting growth and
diversity in the ecosystem. This leaves a small group of authors responsible for
adapting their code for cross-platform compatibility, hindering widget
reliability and maintainability.

## Methodology

Anywidget simplifies the authoring, sharing, and distribution of Jupyter Widgets
by (i) introducing a standard for widget front-end code based on the web
browser’s native module system, (ii) loading these modules from the kernel, and
(iii) providing the necessary "glue code" to adapt existing JCPs to load and
execute these components [@fig:after-afm]. This separation of concerns allows
widget authors to write portable code that runs consistently across JCPs without
manual installation steps.

:::{figure} dev-after-afm-02.png
:label: fig:after-afm
Anywidget simplifies widget authorship and sharing and ensures cross-platform compatibility. With anywidget, developers author a single, standard portable ES module (AFM), which is loaded from the kernel and executed using the browser's native module system. For existing JCPs, anywidget provides a front-end adapter to load and execute these standardized modules, while new platforms can add native AFM support directly. Widget kernel-side code and AFM can be run directly from within notebooks, from source files, or distributed as single Python packages.
:::

Packaging custom Jupyter Widgets is complex due to the need to adapt JavaScript
source code for various module systems used by JCPs. Initially, JavaScript
lacked a built-in module system, leading to diverse third-party solutions
adopted by JCPs. Without a standardized widget front-end format, authors compile
their code for each JCP. In the context of Jupyter Notebook and JupyterLab, this
problem is described in the Jupyter Widgets documentation [@doc_widgets] as follows:

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
packaging JavaScript code for reuse [@rojas2021modules]. While most JCPs predate its
standardization, ES modules are universally supported by browsers today. By
adopting ES modules, anywidget is able to use the browser's native import
mechanism to load and execute widget front-end code from the Jupyter kernel,
thereby bypassing JCP import systems and eliminating third-party dependencies.
This approach not only overcomes many development challenges, it also eliminates
installation procedures for front-end code. Consequently, developers can
prototype and share widgets directly within notebooks, making them more reliable
and easier to use across JCPs.

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
Stub of an anywidget front-end module (AFM) with initialization and rendering lifecycle methods.
:::

## Features

Adhering to predictable standards benefits both developers and end users in 
many other ways beyond cross-platform interoperability.

### Web Over Libraries

Front-end libraries change rapidly and often introduce
breaking changes, whereas the web platform remains more backward-compatible.
Traditional Jupyter Widgets require extensions from UI libraries provided by
JCPs, coupling widget implementations to particular third-party frameworks. In
contrast, AFM defines a minimal set of essential interfaces focused on (1)
communicating with the kernel and (2) modifying notebook output cells, without
dictating state or UI models. This approach allows widgets to be defined without
dependencies, reducing boilerplate and preventing lock-in. While authors are
free to incorporate third-party JavaScript tooling or frameworks to enhance
their own widgets or boost their productivity, importantly, no such tools are
needed for JCP compatibility, user installation, or publishing.

### Rapid Iteration

The web ecosystem's adoption of ES modules has led to new
technologies that enhance developer experience and enable rapid prototyping. One
such innovation is hot module replacement (HMR), a method that uses the
browser’s module graph to dynamically update applications without reloading the
page or losing state. Since traditional Jupyter Widgets rely on legacy module
systems, they cannot benefit from HMR and instead require full page clearance,
reload, and re-execution to see changes during development. By contrast,
anywidget is able to provide opt-in HMR, implemented through the Jupyter
messaging protocol, in order to support live development of custom widgets
without any front-end tooling. For example, developers can adjust a widget's
appearance, like a chart's color scheme, without losing its data or needing a
page refresh.

### Progressive Development 

Anywidget makes it possible to prototype widgets
directly within a notebook since all widget code is loaded from the kernel.
Custom widgets can start as a few code cells and transition to separate files,
gradually evolving into standalone scripts or packages – just like kernel-side
programs [@fig:after-afm]. In contrast, developing traditional Jupyter Widgets
is a cumbersome process limited to the Jupyter Notebook and JupyterLab
platforms. It involves using a project generator [@js_cookiecutter; @ts_cookiecutter] to bootstrap a project with
over 50 files, creating and installing a local Python package with custom-built
extensions, compiling JavaScript code, and manually linking build assets to
install extensions. By removing these barriers, anywidget accelerates
development, and allows prototypes to grow into robust tools over time.

### Simplified Publishing

Serving AFMs and other static assets from the kernel
removes the need to publish widget kernel-side and front-end code separately and
coordinate their releases. For example, many JCPs retrieve traditional widget
Javascript code from the npm registry, misusing the registry for distributing
specialized programs rather than reusable JavaScript modules. Instead, with
anywidget, developers can publish a widget (kernel-side module, AFM, and
stylesheets) as a unified package to the distribution channels relevant to the
kernel language, such as the Python Package Index (PyPI). Consolidating the
distribution process this way greatly simplifies publishing and discovery.

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
diverse widget ecosystem [@fig:widgetstats]. New widgets range from educational tools
for experimenting with toy datasets (e.g., [DrawData](https://drawdata.xyz/)) to high-performance data
visualization (e.g., [Lonboard](https://developmentseed.org/lonboard/latest/), [Mosaic](https://uwdata.github.io/mosaic/jupyter/), [Jupyter-Scatter](https://jupyter-scatter.dev/)). These visualization
tools leverage anywidget’s support for binary data transport, enabling efficient
interactive visualization with minimal serialization overhead. Existing widget
projects have also migrated to anywidget ([higlass-python](https://github.com/higlass/higlass-python), [ipyaladin](https://github.com/cds-astro/ipyaladin))  and other
libraries have introduced or refactored existing widget functionality to use
anywidget ([Altair](https://altair-viz.github.io/), ) due to the simplified distribution and authoring
capabilities.

:::{figure} widgetstats.png
:label: fig:widgetstats
Custom Jupyter Widgets per year as of May 30, 2024. Date for each project is the initial commit or the date of the commit when a widget was added to the repository project. Projects are tracked at https://github.com/manzt/anywidget-usage.
:::

The portable widget standard also opens avenues for other emerging
notebook-inspired platforms to leverage the widget ecosystem. For example,
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
