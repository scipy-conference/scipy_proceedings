:author: Matthias Bussonnier
:email: bussonniermatthias@gmail.com
:institution: QuanSight, Inc
:institution: Digital Ours Lab, SARL.
:orcid: 0000-0002-7636-8632
:corresponding:

--------------------------------------------------------------------
Papyri: Better documentation for the Scientific Ecosystem in Jupyter
--------------------------------------------------------------------

.. class:: abstract

   We present here the idea behind Papyri, a framework we are developing to
   provide a better documentation experience for the scientific ecosystem.
   In particular, we wish to provide a documentation browser from within Jupyter
   or other IDE**s?** and Python editors that gives a unified experience, cross
   library navigation search and indexing. 
 %% I don't understand the last part of the sentence (let's chat).  
   By decoupling documentation generation
   from rendering we hope this can help adress some of the documentation accessibility
   concerns, and allow customisation based on users' preferences. 
   
   To be continued.




.. class:: keywords

   Document, Jupyter, ecosystem, accessibility

Introduction
------------

The Python ecosystem has grown rapidly over the past two decades, one of the
last bastion where some of the proprietary competition tools shine is integrated
documentation. Open-source libraries are also developed in distributed setting
that can make it hard to develop coherent and integrated systems. 

While a number of tools and documentation exists, and improvements are made
every-day, most efforts attempt to build documentation in an isolated way. This
inherently leads to a heterogamous aspect of documentation that can be hard to
grasp for the newcomers. This also means that each library authors much make
choices and maintain build script or services.

Efforts such as conda-forge have shown that concerted efforts can give a much
better experience to end-users, and in todays world where sharing libraries
source on code platforms, Continuous Integration, and many other tools is
ubiquitous, we believe a better documentation framework for many of the
libraries of the scientific Python is possible.


Motivation
----------

We've been frustrated by the documentation experience in the Python ecosystem, 
and have found the many of our colleagues are as well. To often have we seen
less experienced users struggle to find the correct "official" website for the
documentation of a given libraries, or stumble across an old version that is
better ranked in your favorite search engine. 

While access to inspector in many IDE gives access to some documentation, it
does not get access to narrative, or galleries. CLI users are in a  even worse
place as raw source is often displayed and no navigation is possible.

As libraries, maintainers we do not want to have to think about final rendering.
Though we'd like our users to gain from improvement in the rendering without
having to rebuild all our docs.

Against all advices we received and our own experience we decided to rebuild a
documentation framework from scratch, and with minimal dependencies.



Parallel with to Compiled languages
-----------------------------------

We'll draw several comparison between documentation building and compiled
languages, plus borrow an adapt a couple of terms from the domain.
Needed, what is building the documentation bur going from a source-code meant for
a machine to a final output targeting the flesh and blood machine between the
keyboard and the chair.

In particular we'll draw similarities with "ahead-of-time" [AOT]_,
"just-in-time" [JIT]_, "intermediate representation (IR)" [IR]_, link-time
optimization (LTO) [LTO]_, static vs dynamic linking.

If you are familiar with these concept that might be a good parallel to keep in
mind in order to follow the reasoning and architecture, but is not necessary to
understand the concepts behind papyri.

Current Tools and their limitations
-----------------------------------

It is difficult to speak about the scientific python ecosystem documentation
without speaking about docutils [docutils]_ and sphinx [sphinx]_ which are
virtually use by all the libraries in the scientific Python ecosystem. Both hese
libraries are the cornerstone of publishing html documentation for Python. While
few alternative exists, most tools and services have some internal knowledge of
sphinx. read the docs [RTD]_ provide a specific sphinx theme [RTD-theme]_ user
can opt-in to, `Jupyter-book` is  built on top of sphinx, and `MyST` parser made
to allow markdown in documentation targets sphinx as a backend. 

All the above tools provides an "ahead of time" documentation compilation and
rendering, a step which is slow and computationally intensive. Each project
needs its specific plugins, extensions and configurations to properly build. It
is also often relatively difficult to build documentation for a single object (a
single function, module or class), making use of those tools for interactive
exploration difficult.  While this "just-in-time" approach is attempted by
projects like `docrepr` that is integrated both in `Jupyter` and `Spyder`, the
above limitation means interactive documentation lacks inline plots, crosslinks,
indexing, search and many custom directives.


Some of the above limitation are inherent to the design of documentation build
tools that were designed to build documentation in isolation. While sphinx
does provide features like `intersphinx`, link resolutions are done at
documentation build time and are thus inherently unidirectional. Even
considering `numpy` and `scipy` that are two extremely close libraries, having
proper cross-linked of documentation requires at least three 5 steps:

- build numpy documentation

- publish numpy ``object.inv`` file. 

- build scipy documentation using numpy ``obj.inv`` file.

- publish scipy ``object.inv`` file
  
- rebuild numpy docs to make use of scipy's ``obj.inv``

Any of the created links being potentially invalidated on the publication of a
new version of any of those libraries. 

RPy2 moved : https://github.com/ipython/ipython/pull/12210


This make using pre-produced html in IDEs and other tools difficult and error
prone. This has also raised security issue where some institution are reluctant
to use either tools like `docrepr` or viewing pre-produced html. 

Editing docstring between a rock and a hard place
-------------------------------------------------

The numpydoc format is ubiquitous among the scientific ecosystem
https://numpydoc.readthedocs.io/en/latest/format.html, It is loosely based on
RST syntax, and despite supporting full rst syntax, docstrings often rarely
contain full-featured directive.  As many tools show raw docstrings and are
incapable of interpreting directive on the fly maintainers are often pull in two
opposite directions. 

- keeping the docstrings simple, mostly text based with few directive in order
  to have readability to the end user that might be exposed to the docstring
  when using tools like IPython and Jupyter. 

- Write an extensive docstring, with references, and directive that
  potentially create graphics, tables and more, but impede readability. 

While tools like `docrepr` mitigate this problem, this is true only for IDE
users and not Terminal users that will still be exposed to raw docstrings. This
leads to long discussions, for example in `sympy
<https://github.com/sympy/sympy/issues/14964>` on how should equations be
represented in docstrings. 


Some libraries would also prefer to use markdown in their docstrings, but this
would create inconsistencies for the end user with respect to rendering. 

Thus we have a



Making documentation multi-step
-------------------------------

We first recognised that many of the customisation made by user when building
documentation with sphinx fall in two categories:

- simpler input convenience. 
- modification of final rendering. 


Wether you customise the ``.. code-block:`` directive to execute or reformat your
entry, or create a ``:rc:`` role to link to configuration parameters, a large
number of custom directive and plug-in make it easier to create references, or
make sure the content is auto generated to avoid documentation becoming out of
sync with libraries source code. This first category often require arbitrary
code execution and must import the library you are currently building the
documentation for. 


The second category of plugins attempt to improve the rendering in order to be
more user friendly. For example `sphinx-copybutton` add a button to easily copy
code snippets in a single click, `pydata-sphinx-theme` provide a different light
theme. We'll note that this second category many of the improvement can fall
into user preferences (`sphinx-rtd-dark-mode`), and developers end up making
choices on behalf of their end users: 

- which syntax highlight to use ?
- should I show type annotations ?
- do I provide a light or dark theme ? 


We have often wished to modify the second category of extension and rebuild
documentation without having to go through the long and slow process of
rebuilding everything. 


Non Goals
---------

Many of the existing projects to build online documentation are well
established, extremely flexible and fits the need or their users. We are in no
way trying to cover many of the use case covered by projects like sphinx, or
Jupyter Book. When authors want or need complete control of the output and wide
personalisation options, or branding papyri is likely not the project to look
at. That is to say single-project websites where appearance, layout, domain is
controlled by the author is an explicit non-goal.

For user who are already using sphinx, mkdocs or other projects and are
interested in using Papyri, we also not targeting 100% compatibilities. You
should expect your project to requires minor modifications to work with papyri. 
We in particular are stricter on many of the rst directive than docutils and
sphinx are, and we believe that a stricter requirements leads to more uniform
documentation setup and syntax, which is simpler for contributors and allow to
catch more errors at compile time. This is qualitatively supported by number of
documentation fixes we did upstream during the developments ADD REFERENCES,
HERE`.


Standadarzing IRD format
------------------------


High level Architecture 
-----------------------

The papyri lifecycle for documentation can roughly be decomposed into 3 broad
categories of stakeholders, and processes. 

The first stakeholders are library maintainers. Those should ensure that papyri
can build Intermediate Representation Documentation (IRD) files. And publish
and IRD bundle.

Creation of IRD files and bundles is a computation intensive step, that may
requires complex dependencies, or specific plugins. Creation of these files may
be a multi-step process or use external tooling that is not related to papyri or
does not use Python. Note that these steps do not requires the libraries
maintainer to worry about visual appearance and rendering of documentation.


The second category of stakeholder are end-users. Those users are responsible
from installing IRD bundles from the libraries the wish to use on their
machines. Note that IRD from libraries that are not in use are installable as
well, and that IRD bundle not attached to a particular library could also be
installed, providing for example domain specific tutorials or examples. 


The third category of stakeholder are IDE developers, who want to make sure
IRD files can be properly rendered and browsed by their users; potentially
taking into account user preferences, and providing added values with for
example indexing, searching, bookmarks. Such a category of stakeholder could
also be opinionated web hosting in a similar fashion to rustsdocs, devdocs.io


Future possibilities
--------------------

- Removal of dynamic docstrings, 
- Markdown
- Static website,
- post deprecation
- translation


Challenges
----------

Fully qualified names vs canonical names. 
case sensitivity.



Current implementation
----------------------


IRD file Generation
~~~~~~~~~~~~~~~~~~~

While the core idea around papyri resides in the IRD files and bundles, we can
come back on some of the decision we made with current implementation.

The current implementation only support parsing RST and Numpydoc in docstrings. 
While we hope to extend it with MyST later, or provide it as a plugin, this is
our main focus as a wide majority of the core Scientific python stack.
We use Tree-Sitter, and tree-sitter-rst to parse RST syntax, in particular
tree-sitter allow us to easily "unparse" an AST node when necessary as the ast
nodes contains bytes offset to the original buffer. This was relatively
convenient to handle custom directive a number of edge cases where project
relied on loose definition of the rst syntax. For example rst directive are of
the form::

  .. directive:: arguments
      
      body

While technically there is no space before the ``::``, docutils and sphinx allow
this, but it fails in tree-sitter with an error node. We can check error nodes,
un-parse, add heuristics to restore a proper syntax and parse the new node.

Alternatively a number of directive like ``warnings``, ``notes``
``admonitions`` still contain valid RST. Instead of storing the directive with
the raw text, we parse the full document (potentially finding invalid syntax),
and unparse to the raw text only if the directive requires it.


Serialisation of datastructure into IRD files are currently using a custom
serialiser that we hope to swap for msgspec. The AST objects are completely
typed but contains a number of Unions and Sequences of Unions. We found out that
many frameworks like ``pydantic`` do not support sequences of Unions where each
item in the Union may be of a different type.


We currently try to type-infer all code examples with Jedi, and pre-syntax
highlight using pygments when possible.

IRD File Installation
~~~~~~~~~~~~~~~~~~~~~

Download and Installation of IRD files is done concurrently using ``httpx``,
with ``trio`` as an async framework. 

The IRD files post-processed into a local custom format. Object informations are
store in 3 different places: A local SQLite database, CBOR representation of
each document, and raw storage on disk for assets and binary blobs. 

SQlite allows us to easily query graph informations at run time, just before
rendering, and is mostly optimised for infrequent read access.

CBOR object for post-processed IRD files has been chosen to provide a more
compact representation than JSON which is highly redundant, while still
avoiding to use compression for fast access.


Access to these resources is providing via an internal ``GraphStore`` API which
is agnostic of the backend, and ensure the consistency of operation like
adding/removing/replacing documents.

Documentation Rendering
~~~~~~~~~~~~~~~~~~~~~~~

We've prototypes a number of rendering engines, each of them basically consist
of fetching a single page and it's metadata, and walking the IRD AST tree, and
rendering each nodes with user preferences. 

- An ASCII terminal render using Jinja2. This can be useful to pipe
  documentation to other tools like grep, less, cat.

- A TUI browser using urwid. This lets you navigate in the terminal, reflow long
  line on window resize, and can even open images files in external editors. We
  encountered several bugs in urwid and are considering rewriting it using
  Rich/Textual.

- A Just-in-Time rendering engine using Jinja2/quart/trio ; Quart being an async
  version of flask. This version is the one with the most features.

- A static "Ahead of time", rendering of all the existing pages that can be
  rendered ahead of time, using the same class as the Just-in-time rendering
  that basically loops through all entries in the SQLite database and render
  each.


Our profile show that documentation rendering is limited by object serialisation
and de serialisation from disk as well a Jinja2 templating engine. 
We've played with writing a static html renderer in a compiled language (Rust,
using compiled, and typed checked templates), and managed to get about a factor
10 speedup, but this implementation is now out of syn with the main papyri
code base. 


Finally we've started implementing a JupyterLab extension that is capable of
basic IRD file browsing and rendering, using react and typescript. It has
limited capabilities, like ability to browse to previous pages.


Misc
----

Is is common for compiler to use IR (MIRI, LLVM IR)
Not a novel idea, allow to mix compilation from multiple targets, LTO.
Diataxis
rustdocs.



.. comment: 
    In this talk we will demo and discuss the work that is being done on Papyri, a
    new framework to provide rich documentation in Jupyter and Terminal IPython
    with plots, crosslink, equations. We will describe how libraries can opt-in to
    this new framework while still in beta to provide feedback, what are the trade-off of using it, the current
    capabilities and the one planed with current funding, as well as where this
    could go in the future.

    This talk discusses a solution to a widely encountered problem of documentation while using Jupyter and Terminal IPython. This will be an impactful talk to the community of all scientific groups.



    ## Summary

    This submission is very interesting! I would have liked if the authors gave more detail on the difference between user perspectives (that is, library users navigating documentation with this tool), and developer perspectives (developers of libraries that may want to integrate this documentation framework into their projects). I also hope that the authors comment on documentation accessibilty for users of different skill levels and if / how this framework addresses it.

    ## Is the abstract compelling?

    Absolutely! This sounds like a fantastic tool that would be of interest to package developers and users in the SciPy community.

    ## How relevant, immediately useful, and novel is the topic?

    The topic is both relevant and useful to the community.






It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

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
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. .. figure:: figure1.png
.. 
..    This is the caption.:code:`chunk of code` inside of it. :label:`egfig` 
.. 
.. .. figure:: figure1.png
..    :align: center
..    :figclass: w
.. 
..    This is a wide figure, specified by adding "w" to the figclass.  It is also
..    center aligned, by setting the align keyword (can be left, right or center).
..    This caption also has :code:`chunk of code`.
.. 
.. .. figure:: figure1.png
..    :scale: 20%
..    :figclass: bht
.. 
..    This is the caption on a smaller figure that will be placed by default at the
..    bottom of the page, and failing that it will be placed inline or at the top.
..    Note that for now, scale is relative to a completely arbitrary original
..    reference size which might be the original size of your image - you probably
..    have to play with it.  :label:`egfig2`
.. 
.. As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
.. figures.
.. 
.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.
.. [docutils] https://docutils.sourceforge.io/
.. [sphinx] https://www.sphinx-doc.org/en/master/
.. [RTD] https://readthedocs.org/
.. [RTD-theme] https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. [AOT] https://en.wikipedia.org/wiki/Ahead-of-time_compilation
.. [JIT] https://en.wikipedia.org/wiki/Just-in-time_compilation
.. [IR] https://en.wikipedia.org/wiki/Intermediate_representation
.. [LTO] https://en.wikipedia.org/wiki/Interprocedural_optimization
