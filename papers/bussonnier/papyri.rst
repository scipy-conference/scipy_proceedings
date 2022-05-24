:author: Matthias Bussonnier
:email: bussonniermatthias@gmail.com
:institution: QuanSight, Inc
:institution: Digital Ours Lab, SARL.
:orcid: 0000-0002-7636-8632
:corresponding:
:author: Camille Carvalho
:email: 
:institution: University of California Merced, Merced, CA, USA
:institution: Univ Lyon, INSA Lyon, UJM, UCBL, ECL, CNRS UMR 5208, ICJ, F-69621, France
:orcid: 0000-0002-8426-549X

====================================================================
Papyri: Better documentation for the Scientific Ecosystem in Jupyter
====================================================================

.. class:: abstract

   We present here the idea behind Papyri, a framework we are developing to
   provide a better documentation experience for the scientific ecosystem. In
   particular, we wish to provide a documentation browser from within Jupyter or
   other IDEs and Python editors that gives a unified experience, cross library
   navigation search and indexing. By decoupling documentation generation from
   rendering we hope this can help address some of the documentation
   accessibility concerns, and allow customisation based on users' preferences. 
   

.. class:: keywords

   Document, Jupyter, ecosystem, accessibility

Introduction
============

Over the past decades, the Python ecosystem has grown rapidly, and one of the
last bastion where some of the proprietary competition tools shine is integrated
documentation. Indeed, open-source libraries are usually developed in
distributed settings that can make it hard to develop coherent and integrated
systems. 

While a number of tools and documentations exists (and improvements are made
everyday), most efforts attempt to build documentation in an isolated way,
inherently creating a heterogeneous framework. The consequences are twofolds:
(i) it becomes difficult for newcomers to grasp the tools properly, (ii) there
is a lack of cohesion and unified framework as library authors make their proper
choices and must maintain build scripts or services.

Many users, colleagues, and members of the community have been frustrated with
the documentation experience in the Python ecosystem. Given a library, who
hasn't struggle to find the "official" website for the documentation ? Often,
users stumble across an old documentation version that is better ranked in their
favorite search engine, and this impacts less experienced users' learning
greatly.

The experience on users' local machine is affected by limited documentation
rendering. Indeed, while the inspector in many Integrated Development
Environments (IDEs) provides some documentation, users do not get access to
the narrative, or full documentation gallery. For Command Line Interface (CLI)
users, documentation is often displayed as raw source where no navigation is
possible. On the maintainers' side, the final documentation rendering is less a
priority. Rather, maintainers should aim at making users gain from improvement
in the rendering without having to rebuild all the docs.

Efforts such as conda-forge [CFRG]_ have shown that concerted efforts can
give a much better experience to end-users, and in today's world where sharing
libraries source on code platforms, continuous integration, and many other tools
is ubiquitous, we believe a better documentation framework for many of the
libraries of the scientific Python should be available.

Thus, against all advice we received and based on our own experience, we have decided to
rebuild an *opinionated* documentation framework, from scratch, and with minimal
dependencies: *Papyri*. Papyri focuses on building an intermediate
documentation representation format, that lets us decouple building, and
rendering the docs. This highly simplifies many operations and gives us access
to many desired features that where not available up to now.

In what follows we provide the framework in which Papyri has been created and
present its objectives (Context and goals), we describe the Papyri features
(format, installation, and usage), then present its current implementation. We
end this paper with comments on current challenges and future work.


1) Context and objectives
=========================

Through out the paper, we will draw several comparisons between documentation
building and compiled languages. Also, we will borrow and adapt commonly used
terminology. In particular, similarities with "ahead-of-time" (AOT) [AOT]_,
"just-in-time"" (JIT) [JIT]_, intermediate representation (IR) [IR]_, link-time
optimization (LTO) [LTO]_, static vs dynamic linking will be highlighted. This
allows to clarify the presentation of the underlying architecture, however there
is no requirement to be familiar with the above to understand the concepts
underneath Papyri. In that context, we wish to discuss documentation building as
a process from a source-code meant for a machine to a final output targeting the
flesh and blood machine between the keyboard and the chair. 

1) Current tools and limitations
--------------------------------

In the scientific Python ecosystem, it is well known that Docutils [docutils]_
and Sphinx [sphinx]_ are major cornerstones for publishing html documentation
for Python, and are used by all the libraries in this ecosystem. While a few
alternatives exist, most tools and services have some internal knowledge of
Sphinx. For instance, `Read the Docs` [RTD]_ provides a specific Sphinx theme
[RTD-theme]_ users can opt-in to, `Jupyter-book` is built on top of Sphinx, and
MyST parser [MYST]_ (which is made to allow markdown in documentation) 
targets Sphinx as a backend, to name a few. All of the above provide an
"ahead-of-time" documentation compilation and rendering, which is slow and
computationally intensive. When a project needs its specific plugins, extensions
and configurations to properly build (which is almost always the case), it is
relatively difficult to build documentation for a single object (like a single
function, module or class). This makes AOT tools difficult to use for
interactive exploration. One can then consider a JIT approach, as done
for `Docrepr` (integrated both in `Jupyter` and `Spyder`). However in that case,
interactive documentation lacks inline plots, crosslinks, indexing, search and
many custom directives.

Some of the above limitations are inherent to the design of documentation build
tools that were intended for a separate documentation construction. While Sphinx does
provide features like `intersphinx`, link resolutions are done at documentation
build time. Thus, this is inherently unidirectional, and can easily get broken.
To illustrate this, we consider `NumPy` and `SciPy`, two extremely close
libraries. In order to obtain proper cross-linked documentation, one is required to perform at least five
steps:

- build NumPy documentation

- publish NumPy ``object.inv`` file. 

- build SciPy documentation using NumPy ``obj.inv`` file.

- publish SciPy ``object.inv`` file
  
- rebuild NumPy docs to make use of SciPy's ``obj.inv``

Only then can both SciPy's and NumPy's documentation refer to each other. As one can expect, crosslinks break everytime a new version of a library is published [#]_. Pre-produced html in IDEs and other tools are then prone to error and difficult to maintain. This also raises security issues: some institutions become reluctant to use tools like `Docrepr` or viewing pre-produced html. 

.. [#] `ipython/ipython#12210 <https://github.com/ipython/ipython/pull/12210>`_, `numpy/numpy#21016 <https://github.com/numpy/numpy/pull/21016>`_, `& #29073 <https://github.com/numpy/numpy/pull/20973>`_


2) Docstrings format
--------------------

The `numpydoc` format is ubiquitous among the scientific ecosystem [NPDOC]_. It
is loosely based on RST syntax, and despite supporting full rst syntax,
docstrings rarely contain full-featured directive. Maintainers confronted to the following dilemma:

- keep the docstrings simple. This means mostly text-based docstrings with few directive for efficient readability. The end-user may be exposed to raw docstring, there is no on-the-fly directive interpretation. This is the case for tools such as IPython and Jupyter. 

- write an extensive docstring. This includes references, and directive that
  potentially creates graphics, tables and more, allowing an enriched end-user experience. However this may be computationally intensive, and executing code to view docs could be a security risk.

Other factors impact this choice: (i) users, (ii) format, (iii) runtime. IDE users or non-Terminal users motivate to push for extensive docstrings, and tools like `Docrepr` can mitigate this problem. However, users are often exposed to raw docstrings (see for example the discussion `SymPy
<https://github.com/sympy/sympy/issues/14964>`_ on how should equations be
represented in docstrings), and :ref:`Fig1`. In terms of format, markdown is appealing, however inconsistencies in the rendering will be created between libraries. Finally, some libraries can dynamically modify their docstring at runtime. While this avoids using directives, it ends up being more expensive (runtime costs, complex maintenance, and contribution costs).

..   :align: center
..   :figclass: w
.. figure:: scipy-dpss-old-new.png

   The following screenshot shows current help for ``scipy.signal.dpss`` as
   currently accessible on the left, as shown by Papyri for Jupyterlab
   extension on the right. :label:`Fig1`


3) Objectives of the project
----------------------------

We now layout the objectives of the Papyri documentation framework. 
Let us emphasize that the project is in no way intended to replace or cover many features included in well established documentation tools such as Sphinx or Jupyter-book.
Those projects are extremely flexible and fit the need of their users. The Papyri project addresses specific documentation challenges (mentioned above), we present below what is (and what is not) the scope of work.

a) A generic (little customisable) website builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When authors want or need complete control of the output and wide
personalisation options, or branding, then Papyri is likely not the project to look
at. That is to say single-project websites where appearance, layout, domain need to be
controlled by the author is not part of the objectives.

b) A uniform documentation structure and syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Papyri project prescribes stricter requirements in terms of format and structure compared to other tools such as Docutils and Sphinx. When possible, the documentation follows the Diátaxis Framework [DT]_. This provides a uniform documentation setup and syntax, simplifying contributions to the project and easing error catching at compile time. 
Such strict environment is qualitatively supported by number of documentation fixes done upstream during the development stage of the project **ADD REFERENCES,
HERE to many fixes to numpy/scipy**.
Since Papyri is not fully-customisable, users who are already using documentation tools such as Sphinx, `mkdocs` **Not cited before in the context section, why not ?** and others should expect their project to require minor modifications to work with Papyri. 


c) Accessibility and user proficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessibility is a top priority of the project. To that aim, items are associated to semantic meaning as much as possible, and documentation rendering is separated from documentation building phase. That way, accessibility features such as high contract themes (for better speech-to-text raw data reading), early example highlights (for newcomers) and type annotation (for advanced users) can be quickly available. With the uniform documentation structure, this provides a coherent experience where users become more comfortable to find information (and in a single location) (see Figure 1).

d) Simplicity, speed, and independence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One objective of the project is to make documentation installation and rendering relatively straightforward and fast. To that aim, the project includes relative independence of documentation building across libraries, allowing bidirectional crosslinks (i.e. both forward and backward links between pages) to be maintained more easily. In other words, a single library can be built without the need to access documentation from another. Also, the project should include straightforward lookup documentation for an object from the
interactive REPL. Finally, efforts are put to limit the installation speed (to avoid polynomial growth when installing packages on large distributed systems).

**TO MB: should IRD be introduced in this section then ??**

2) The Papyri solution
======================

In this section we describe in more details how Papyri has been implemented to address the objectives mentioned above. 


1) Making documentation a multi-step process
--------------------------------------------

.. When building documentation, one can either customise the ``.. code-block:`` directive to execute/reformat entries, or create a ``:rc:`` role to link to configure parameters, several custom directives and plug-ins to simplify the rendering (including creating references, auto-genering documentation)
.. and sync with libraries source code. 


When using current documentation tools, customisation made by maintainers usually
falls into the following two categories:

- simpler input convenience,
- modification of final rendering.

This first category often requires arbitrary code execution and must import the
library currently being built. For example implicit imports of ``..
code-block:``, or custom ``:rc:`` directive). The second one offers a more user
friendly environment. For example,
`sphinx-copybutton` **Add ref** adds a button to easily copy code snippets in a single
click, and `pydata-sphinx-theme` or `sphinx-rtd-dark-mode` **add REF** provide a different
appearance. As a consequence, developers must make choices on behalf of their
end-users: this may concern syntax highlights, type annotations display,
light/dark theme. 

Being able to modify extensions and re-render the documentation without the
rebuilding and executing stage is quite appealing. Thus, the building phase in
Papyri (collecting documentation information) is separated from the rendering
phase (Objective (c)): at this step, Papyri has no knowledge and no
configuration options that permit to modify the appearance of the final
documentation. Additionally, the optional rendering process has no knowledge of
the building step, and can be run without accessing the libraries involved.

This technique is commonly used in the field of compilers **add ref**, but to
our knowledge, it has not been implemented for documentation in the Python
ecosystem. As mentioned before, this separation is key to achieve many features
proposed in Objectives (c), (d).

2) Intermediate Representation for Documentation (IRD)
------------------------------------------------------

IRD format
~~~~~~~~~~
.. We borrow the name IR again from compilers.

Papyri relies on standard interchangeable "Intermediate Representation for
Documentation format" (IRD). This allows to reduce operation complexity of the
documentation build. For example, given M documentation producers and N
renderers, the documentation build would be O(MN). If each producer only cares
about producing IRD, and if each renderer only consumes it, then one can reduce
to O(M+N). Additionally, one can take IRD from multiple producers at once, and
render them all to a single target, breaking the silos between libraries.

At the moment, IRD files are currently separated into four main categories:**provide a sketch or example with all ??**

- API files describe the documentation for a single object, expressed as a
  Json object. When possible, the information is encoded semantically (Objective (c)).
  Files are organized based on the fully-qualified name of the Python object
  they reference, and contain either absolute reference to another object
  (library, version and identifier), or delayed references to objects that may
  exist in another library. Some extra per-object meta information like
  file/line number of definitions can be stored as well. 
- Narrative files are similar to API files, except that they do not
  represent a given object, but possesses a previous/next page, and are organised
  in an ordered tree related to the table of content. 
- Examples files are a non-ordered collection of files.
- Assets files are untouched binary blobs **find better word** that can be referenced by any of the above
  three ones. They are the only ones that contain backward references, and no forward references.

In addition to the four categories above, metadata about the current package is
stored: this includes library name, current version, PyPi name, GitHub slug **real term?**, maintainers' names,
logo, issue tracker and others. In particular, metadata allows us to auto generate
links to issue trackers, and to source files when rendering. 
In order to properly resolve some references and normalize links convention, we also store a mapping from fully qualified names to canonical ones.

IRD files must be standardized in order to achieve a uniform syntax structure (Objective (b)), In this paper, we do not discuss the IRD files distribution. The final specification IRD files is still in progress. We thus invite contributors to
consult the current state on the GitHub repository **add ref to the repo**.

IRD bundles
~~~~~~~~~~~
**Why is it important to talk about IRD bundles ? to clarify ? Having a difficult time to edit this subsection**

Unlike packages installation IRD bundles do not have the notion of dependencies,
thus a full-fledge package manager is not necessary, and installing can be
limited to downloading corresponding files and unpacking them.

IRD bundles for multiple versions of the same library (or conflicting libraries) is not inherently problematic, and can be shared across
multiple multiple environments.

From a security standpoint, installing IRD bundles does not require the
execution of arbitrary code. This is critical for adoption in deployments.

There is an opportunity at IRD installation time to provide localized variant,
but we have not explored much the opportunity of IRD bundle translations.


IRD and high level usage 
~~~~~~~~~~~~~~~~~~~~~~~~

Papyri-based documentation involves three broad categories of stakeholders (library maintainers, end-users, IDE developers), and processes. This leads to certain requirements on IRD files and bundles.

On the maintainers' side, the goal is to ensure that Papyri can build IRD files, and publish IRD bundles.

Creation of IRD files and bundles is the most computational intensive step. It may
require complex dependencies, or specific plugins. Thus, this can be a multi-step process, or one can use external tooling (not related to Papyri nor
uses Python) to create them. Visual appearance and rendering of documentation is not taken into account in this process.

.. comment:
   maybe move next paragrah somewehre else ?
   
End-users are responsible from installing desired IRD bundles. In most cases, it will consist of IRD bundles from
already installed libraries.  While Papyri is not currently integrated with
packages manager or IDEs, one could imagine this process being automatic, or on demand.

Finally, IDEs developers want to make sure
IRD files can be properly rendered and browsed by their users when requested. This may
potentially take into account users' preferences, and may provide added
values such as indexing, searching, bookmarks, etc.), as seen in rustsdocs, devdocs.io. 


Overall, building cost depends on the three stakeholders. For example, building SciPy & NumPy
documentation IRD files on a 2021 Macbook Pro M1 (base model), including executing
examples in most docstrings and type inferring most examples (with most variables semanticly inferred) can take several minutes. **Is that good ? Comparison with other docs building ?Can we make a comment on this ?**


Current implementation
======================

In this section we'll describe a few of the choices we've make for a our current
implementation. 


IRD file Generation
-------------------

While the core idea around papyri resides in the IRD files and bundles, we 
we made with current implementation. As a wide majority of the core Scientific python stack
uses sphinx, RST and Numpydoc, the current implementation only support those. 
We do hope to extend it with MyST later, or provide it as a plugin.

We use Tree-Sitter, and tree-sitter-rst to parse RST syntax, in particular
tree-sitter allow us to easily "unparse" an AST node when necessary as the ast
nodes contains bytes offset to the original buffer. This was relatively
convenient to handle custom directive and number of edge cases where project
relied on loose definition of the rst syntax. For example rst directive are of
the form::

  .. directive:: arguments
      
      body

While technically there is no space before the ``::``, docutils and sphinx allow
this, but it fails in tree-sitter with an error node. We check for error nodes,
un-parse, and add heuristics to restore a proper syntax and parse again  to
obtain the new node.

Alternatively a number of directive like ``warnings``, ``notes``
``admonitions`` still contain valid RST. Instead of storing the directive with
the raw text, we parse the full document (potentially finding invalid syntax),
and unparse to the raw text only if the directive requires it.


Serialisation of data structure into IRD files are currently using a custom
serialiser that we hope to swap for msgspec **ADD REF**. The AST objects are completely
typed but contains a number of Unions and Sequences of Unions. We found out that
many frameworks like ``pydantic`` do not support sequences of Unions where each
item in the Union may be of a different type.

We currently try to type-infer all code examples with Jedi, and pre-syntax
highlight using pygments when possible.

IRD File Installation
---------------------

Download and Installation of IRD files is done concurrently using ``httpx``,
with ``trio`` as an async framework. This let us download files concurrently.

As the current implementation of Papyri is targeted at Python documentation and
written in Python, we can query the existing version of Python libraries
installed, and infer the right version of the requested documentation. Our
implementation currently attempt to guess relevant libraries version when the
exact version number is missing from for the install command. 


The IRD files are post-processed into a local custom format. Object informations are
store in 3 different places: A local SQLite database, CBOR representation of
each document, and raw storage on disk for assets and binary blobs. 

SQlite allows us to easily query graph informations at run time, just before
rendering, and is mostly optimised for infrequent read access. While we still
mostly resolve some SQLite information at runtime, we are planning to move some
of this processing to installation time. For example, determining whether inter
libraries links exists.

CBOR object for post-processed IRD files has been chosen to provide a more
compact representation than JSON which keys are often is highly redundant, while
still avoiding to use compression for fast access.

Access to these resources is providing via an internal ``GraphStore`` API which
is agnostic of the backend, and ensure the consistency of operations like
adding/removing/replacing documents.

Documentation Rendering
-----------------------

The current papyri implementation contains Wea number of rendering engines, each
of them mostly consist of fetching a single page, it's metadata, and
walking the IRD AST tree, and rendering each nodes with user preferences. 

- An ASCII terminal render using Jinja2. This can be useful to pipe
  documentation to other tools like grep, less, cat. 
  This also helps us to work in a highly restricted environment, and make sure
  reading the documentation is sensible; for example as a proxy to using a
  screen reader.

- A Textual User Interface browser using urwid. This lets you navigate in the
  terminal, reflow long line on window resize, and can even open images files in
  external editors. We encountered several bugs in urwid and are considering
  rewriting it using Rich/Textual. Our project is for this renderer to replace
  CLI IPython ``?`` interface which currently only shows raw docstrings.

- A "Just-in-Time" rendering engine using Jinja2/quart/trio ; Quart being an async
  version of flask. This version is the one with the most features, and is the
  principal one we use for development. This environment let us iterate rapidly
  over the rendering engine.

- A static "Ahead of time", rendering of all the existing pages that can be
  rendered ahead of time, using the same class as the Just-in-time rendering
  that basically loops through all entries in the SQLite database and render
  each independently. We use this renderer mostly for exhaustive testing, and
  measure performance. 

  With this renderer we can render most of the API documentation of IPython,
  astropy, dask, distributed, matplotlib, networkx, numpy, pandas, papyri, scipy,
  scikit-image. This represent ~28000 pages in ~60 seconds, so about 450 pages/sec on
  a recent macbook pro M1.

For all of the above renderer, our profiling shows that documentation rendering is
mostly limited by object de-serialisation from disk as well a Jinja2
templating engine. We've played with writing a static html renderer in a
compiled language (Rust, using compiled, and typed checked templates), and
managed to get about a factor 10 speedup, but this implementation is now out of
sync with the main papyri code base. 


Finally we've started implementing a JupyterLab extension that present itself as
a side-panel and is capable of basic browsing and rendering. Is uses typescript,
react and native JupyterLab component. Future plan is to replace and complement
JupyterLab's ``?`` and ``?`` operator as well as JupyterLab Inspector when
possible. A screen shot of current development version of the JupyterLab
extension can be seen in :ref:`Fig1` and :ref:`Fig2`.


.. figure:: jupyterlab-prototype.png
   :scale: 80%


   Zoomed out view of the papyri for jupyterlab extension, we can see that the
   code examples include plots. Most token in each examples are link to the
   corresponding page. Early navigatin bar visible at the top. :label:`Fig2`


.. figure:: local-graph.png

   (screenshot). We played with the possibility of using D3.js to a local graph
   of connection among the most important node arround ``numpy.ndarray``. Nodes
   are sized with respectd to the number of incomming links, and colored with
   respect to their library.




Challenges
==========

In order to be able to link to object documentation without having access
the build IRD bundles from all the library we need to come up with a schema that
uniquely identify each object. For this we decided to use the fully qualified
names of an object. That is to say the concatenation of the module in which it
is defined, with its local name. We encountered multiple edge cases with that. 

- To mirror python syntax is it easy to use ``.`` to concatenate both parts. 
  Unfortunately that leads to ambiguity when modules re-export functions of
  the same name. 

  .. code-block:: python

      # module mylib/__init__.py

      from .mything import mything

  ``mylib.mything`` is ambiguous with respect to the ``mything`` submodule and
  the object reexported. In future version we'll  use ``:`` as a module/name
  separator.

- Decorated functions or other dynamic approaches to expose function to users
  end up having ``<local>>`` in their fully qualified names, which is invalid. 

- Many builtins functions (``np.sin``, ``np.cos``, ...) do not have a fully
  qualified name that can be extracted by object introspection. We believe it 
  should be possible to identify those via other means (e.g. docstring hash) but
  haven't explored those possibilities yet.

- Fully qualified names are often not canonical names (the name that are
  typically use for import), and finding the canonical name automatically is not
  always straightforward. 

- There are also challenges with case sensitivity, in particular of
  MacOS file systems, and a couple of object ends up referring to the same IRD file
  on disk if proper care is not taken. We currently append a case-sensitive hash
  at end of the filename to disambiguate.

- Many libraries have syntax that _looks_ right once rendered to html, but does
  not follow proper syntax, or relies on peculiarities of docutils and sphinx
  rendering and parsing.

- Many custom directive plugins cannot be reused from sphinx, and need to be
  reimplemented.



Future possibilities
====================

Beyond what has been presented in this paper, there is a number of opportunities
to improved and extend on what papyri can allow for the Scientific Python
ecosystem. 

One of the area we have not talked about is the ability to build IRD bundle on
Continuous Integration platform. Services like GitHub action, Azure pipeline and
many other are already setup to test packages. We hope to leverage this
infrastructure to build IRD file and make them available to users. 

Hosting of intermediate IRD file has also not been covered, while we currently
have a prototype of http index using GitHub pages, it is likely not a
sustainable hosting platform as disk space is limited. IRD being in our
experience smaller than HTML documentation, we hope that other platform like
readthedoc can be leveraged. A platform like readthedocs could also provide a
single domain that renders the documentation for multiple libraries, thus
avoiding having many sub domains for each library and giving a more unified
experience to users. 

It should be possible for projects to avoid using many dynamic docstrings
interpolation that are use to documents ``*args`` and ``**kwargs``. This would
make sources easier to read, and potentially speedup some library import time. 

Once a given library is confident enough of its users use an IDE that support
papyri for documentation, docstring syntax could be exchanged for markdown.


As IRD files are structured, it should be feasible to provide cross-version
information in documentation. For example, if one installs multiple version of
IRD bundle for a library. Assuming the user does not use the latest version,
the renderer could inspect IRD file from previous/future versions to indicate
the range of version for which the documentation has not changed.
With a bit more work, it should be possible  to infer *when* a parameter was
removed, or will be removed, or simply allow to display the difference between
two versions.





- post deprecation
- translation
  - automatic gallery.

Misc
----

Is is common for compiler to use IR (MIRI, LLVM IR)
Not a novel idea, allow to mix compilation from multiple targets, LTO.
Diataxis
rustdocs.
https://markdoc.io/
USE CI to build documentatino



.. comment: 
    In this talk we will demo and discuss the work that is being done on Papyri, a
    new framework to provide rich documentation in Jupyter and Terminal IPython
    with plots, crosslink, equations. We will describe how libraries can opt-in to
    this new framework while still in beta to provide feedback, what are the trade-off of using it, the current
    capabilities and the one planed with current funding, as well as where this
    could go in the future.

    This talk discusses a solution to a widely encountered problem of documentation while using Jupyter and Terminal IPython. This will be an impactful talk to the community of all scientific groups.



    ## Summary

    This submission is very interesting! I would have liked if the authors gave
    more detail on the difference between user perspectives (that is, library
    users navigating documentation with this tool), and developer perspectives
    (developers of libraries that may want to integrate this documentation
    framework into their projects). I also hope that the authors comment on
    documentation accessibilty for users of different skill levels and if / how
    this framework addresses it.

    ## Is the abstract compelling?

    Absolutely! This sounds like a fantastic tool that would be of interest to package developers and users in the SciPy community.

    ## How relevant, immediately useful, and novel is the topic?

    The topic is both relevant and useful to the community.





References
----------

.. [docutils] https://docutils.sourceforge.io/
.. [sphinx] https://www.sphinx-doc.org/en/master/
.. [RTD] https://readthedocs.org/
.. [RTD-theme] https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. [AOT] https://en.wikipedia.org/wiki/Ahead-of-time_compilation
.. [JIT] https://en.wikipedia.org/wiki/Just-in-time_compilation
.. [IR] https://en.wikipedia.org/wiki/Intermediate_representation
.. [LTO] https://en.wikipedia.org/wiki/Interprocedural_optimization
.. [DT] https://diataxis.fr/
.. [CFRG] https://conda-forge.org/
.. [MYST] https://myst-parser.readthedocs.io/en/latest/
.. [NPDOC] https://numpydoc.readthedocs.io/en/latest/format.html
