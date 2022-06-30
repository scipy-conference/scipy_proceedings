:author: Henry Schreiner
:email: henryfs@princeton.edu
:institution: Princeton University

:author: Jim Pivarski
:email: pivarski@princeton.edu
:institution: Princeton University

:author: Eduardo Rodrigues
:email: eduardo.rodrigues@cern.ch
:institution: University of Liverpool

:bibliography: mybib

--------------------------------------
Awkward Packaging: building Scikit-HEP
--------------------------------------

.. class:: abstract

   Scikit-HEP has grown rapidly over the last few years, not just to serve the
   needs of the High Energy Physics (HEP) community, but in many ways, the
   Python ecosystem at large. AwkwardArray, boost-histogram/hist, and iminuit
   are examples of libraries that are used beyond the original HEP focus. In
   these proceedings, we will look key packages in the ecosystem, at how the
   collection of 30+ packages was developed and maintained, and the software
   ecosystem contributions made to packages like cibuildwheel, pybind11, nox,
   scikit-build, build, and pipx that support this effort, and the Scikit-HEP
   developer pages, and initial WebAssembly support.

.. class:: keywords

   packaging, ecosystem, high energy physics, community project

Introduction
------------

..
   High Energy Physics needs. Info about dataset size, etc. Historical ROOT info.

High Energy Physics (HEP) has always had intense computing needs due to the
size and scale of the data collected. The World Wide Web was invented at the CERN Physics laboratory in Switzerland in
1989 when scientists in the EU were trying to communicate results with
scientist in the US, and vice-versa :cite:`leiner2009brief`.  Today, HEP has the largest
scientific machine in the world, at CERN: the Large Hadron Collider (LHC), 27 km in
circumference :cite:`evans2008lhc`, with multiple experiments with thousands of
collaborators processing over a petabyte of raw data every day, with 100
petabytes being stored per year at CERN. This is one of the largest scientific
datasets in the world of exabyte scale :cite:`peters2011exabyte`, which is roughly
comparable in order of magnitude to all of astronomy or YouTube :cite:`stephens2015big`.

HEP scientists have been interested in Python since 1993
:cite:`templon_jeffrey_2022_6353621`. With the release of Python 1.0.0 in 1994,
a few physicists started using Python. A year later, the ROOT project
for an analysis toolkit (and framework) was released, making C++ the main
language for HEP. The ROOT project also needed an interpreted language to
driving analysis code. Python was rejected for this role due to being "exotic"
at the time, and because it was considered too much to ask physicists to code
in two languages. Instead, ROOT provided a C++ interpreter, called CINT, which
later was replaced with Cling, which is the basis for the clang-repl project in
LLVM today :cite:`ifrim2022gpu`.

Python would start showing up in the late 90's in experiment frameworks as a
configuration language. These frameworks were primarily written in C++, but
were made of many configurable parts :cite:`LAMMEL199832`; the glueing together
of the system was done in Python - a model still popular today, though some
experiments are now using Python + Numba as an alternative model,
such as for example the Xenon1T experiment :cite:`remenska2017giving,rousselle2021fast`.

In the early 2000s, the use of Python HEP exploded, heavily driven by
experiments like LHCb developing frameworks and user tools for
scripting. ROOT started providing Python bindings in 2004
:cite:`generowicz2004reflection` that were not considered Pythonic
:cite:`galli2020new`, and still required a complex multi-hour build of ROOT to
use [#]_. Analyses still consisted largely of ROOT, with Python sometimes
showing up.

.. [#] Almost 20 years later ROOT's Python bindings have been rewritten for
   easier Pythonizations, and installing ROOT in Conda is now much easier,
   thanks in large part to efforts from Scikit-HEP developers.

By the mid 2010's, a marked change had occurred. Many students were coming in
with little or no C++ experience, but had existing knowledge of Python and the
growing Python data science ecosystem, like NumPy and Pandas. Several HEP experiment analyses
were performed driven by Python, with ROOT only being used for things that were not
available in the Python ecosystem. Some of these were HEP specific: ROOT is
also a data format, so users needed to be able to read data from ROOT files.
Others were less specific: HEP users have intense histogram requirements due to
the data sizes, large portions of HEP data is "jagged" rather than rectangular,
vectors manipulation was important (especially Lorenz Vectors, a four
dimensional relativistic vector with a non-Euclidean metric), and data fitting
was important, especially with complex models and accurate error estimation.


.. figure:: shells-hep.pdf

   The Scikit-HEP ecosystem and affiliated packages.
   :label:`fig-shells`

Beginnings of a scikit
----------------------

..
   About how it was planned and built.

In 2016, the ecosystem for Python in HEP was rather fragmented. There were
a handful of popular packages that were useful in HEP spread around among
different authors. The rootpy project had several packages that made the
ROOT-Python bridge a little easier, such as the ``root-numpy`` and related
``root-pandas`` packages. The C++ MINUIT fitting library was integrated into
ROOT, but the ``iminuit`` package :cite:`iminuit` provided an easy to install
standalone Python package with an extracted copy of MINUIT. Several other
specialized standalone C++ packages had bindings as well. Many of the initial
authors were transitioning to a less-code centric role or leaving for industry,
leaving projects like rootpy and iminuit without maintainers.

Eduardo Rodrigues, a scientist working on the LHCb experiment for the
University of Cincinnati, started working on a package called ``scikit-hep``
that would provide a set to tools useful for physicists working on HEP analysis.
The initial version of the ``scikit-hep`` package had a simple vector library,
HEP related units and conversions, several useful statistical tools, and provenance recording
functionality,

He also placed the GitHub repository into a GitHub organization of the same
name, and asked several of the other HEP related packages to join. The ROOTPy
project was ending, with the primary author moving on, and so several of the
then-popular packages [#]_ that were included in the rootpy organization were
happily transferred to Scikit-HEP. Several other existing HEP libraries,
primarily interfacing to existing C++ simulation and tracking frameworks, also
joined, like ``IMinuit``, ``Probfit``, ``PyJet``, and ``NumPythia``. Some of these libraries
have been retired or replaced today, but were an important part of Scikit-HEP's
initial growth.

.. [#] The primary package rootpy was not transferred, but instead had a final
   release and then died. It was an inspiration for the new PyROOT bindings,
   and influenced later Scikit-HEP packages like ``mplhep``. The transferred
   libraries have since been replaced by integrated ROOT functionality. All
   three packages required ROOT, which is not on PyPI, so were not suited for a
   Python-centric ecosystem.

First initial success
---------------------

In 2016, the largest barrier to using Python in HEP in a Pythonic way was ROOT.
It was challenging to compile, had many non-Python dependencies, was huge
compared to most Python libraries, not very Pythonic, and didn't play well with
packaging. Many Python analyses started with a "convert data" step using PyROOT
to read ROOT files and convert them to a Python friendly format like HDF5. Then
the bulk of the analysis would use reproducible Python virtual environments or
Conda environments.

This changed when Jim Pivarski introduced the ``Uproot`` package (originally envisioned as
"µroot"). This was a pure-Python implementation of a ROOT file reader (and
later writer) that could remove the initial conversion environment by simply
pip installing a package. It also had a simple, Pythonic interface and produced
outputs Python users could immediately use, like NumPy arrays, instead of
PyROOT's wrapped C++ pointers.

Uproot needed to do more than just be file format reader/writer; it needed to
provide a way to represent the special structure and common objects that ROOT
files could contain. This lead to the development of two related packages that
would support uproot. One, uproot-methods, included Pythonic access to
functionality provided by ROOT for its core classes, like spacial and Lorentz
vectors. The other was AwkwardArray, which would grow to become one of the most
important and most general packages in Scikit-HEP. This package allows
NumPy-like idioms for array-at-a-time manipulation on jagged data structures. A
jagged array is a (possibly structured) array with a variable length dimension.
These are very common and relevant in HEP; events have a variable number of
tracks, tracks have a variable number of hits in the detector, etc.  Many other
fields also have jagged data structures. While there are formats to store such
structures, computations on jagged structures have usually been closer to SQL
queries on multiple tables than direct object manipulation. Pandas handles this
through multiple indexing and a lot of duplication.

Uproot was a huge hit with incoming HEP students; suddenly they could access
HEP data using a library installed with pip or conda and no external compiler
or library requirements, use tools they already knew like Pandas and the
rapidly growing machine learning frameworks. There were still some gaps and
pain points in the ecosystem, but an analysis without C++ or compiling ROOT was
finally possible. Scikit-HEP did not and does not intend to replace ROOT, but
it provides alternative solutions that work natively in the Python "Big Data"
ecosystem.

Several other useful HEP libraries were also written. Particle was written for
accessing the Particle Data Group (PDG) particle data in a simple and Pythonic
way. DecayLanguage originally provided tooling for decay definitions, but was
quickly expanded to include tools to read and validate "DEC" decay files, an
existing text format used to configure simulations in HEP.

Building better
---------------

.. figure:: github-histogram-libraries.pdf
   :figclass: w
   :scale: 65%

   The landscape of different libraries for Histograms in HEP.
   From the HEP Software Foundation Analysis Ecosystems II Workshop, May 2022.
   :label:`fig-github-histogram`

In 2018, HEP physicist and programmer Hans Dembinski proposed a histogram
library to the Boost libraries, the most respected C++ library collection. It
provided a histogram-as-an-object concept from HEP, but rethought histograms in
C++14, using composable axes and storage types. It originally had an initial
Python binding, written in Boost::Python. Henry Schreiner proposed the creation
of a standalone binding to be written with pybind11 in Scikit-HEP. The original
bindings were removed, Boost::Histogram was accepted into the Boost libraries,
and work began on ``boost-histogram``. The IRIS-HEP grant had just started,
which was providing funding for several developers to work on Scikit-HEP
project packages such as this one.

There were already a variety of attempts at histogram libraries, but none of
them filled the requirements of HEP physicists fully, and most of them were not easy
to install or use. Any new attempt here would have to be clearly better than
the existing collection of diverse attempts (see Fig
:ref:`fig-github-histogram`). The development of a library with compiled
components intended to be usable everywhere required good support for building
libraries.  Advancements in the packaging ecosystem, such as the wheel format
and the manylinux specification and docker image had made redistributable
Python wheels possible, but there still were many challenges to making a new
library that could be used anywhere.

The boost-histogram library only depended on header-only components of the
Boost libraries, and the header only pybind11 package, and all needed files
were packed into the SDist, and everything was possible using only setuptools,
making build-from-source simple on any system supporting C++14 (which did not
include RHEL 7 or manylinux1).

The first stand-alone development was ``azure-wheel-helpers``, a set of files
that helped produce wheels on the new Azure Pipelines platform. Building
redistributable wheels requires a variety of techniques, even without shared
libraries, that vary dramatically between platforms and were/are poorly
documented. This worked well, and was quickly adapted for the other packages in
Scikit-HEP that included non-ROOT binary components. Work here would eventually
be merged into ``cibuildwheel``, which would become the build tool for all
non-ROOT binary packages in Scikit-HEP.

The second major development was the upstreaming of CI and build system
developments to pybind11. Pybind11 provided significant benefits to our
packages over (mis)-using Cython for bindings: reduced maintenance, simpler
builds, no need to pin NumPy when building, and a cross-package API. The
``iMinuit`` package was later moved to pybind11 as well, and pybind11 became
the Scikit-HEP recommended binding tool. Scikit-HEP developers, notably Henry Schreiner, contributed a
variety of fixes and features to pybind11, including positional and keyword
arguments, prepending to the overload chain, type access and manipulation,
completely redesigned CMake integration, a new pure-Setuptools helpers, and a
complete CI redesign based on GitHub Actions, with over 70 jobs, and expanded
compiler support. We also helped improve all the example projects.

This example of a project with binary components being usable everywhere then
encouraged the development of Awkward 1.0, a rewrite of AwkwardArray replacing
the Python-only code with compiled code, fixing some long-standing limitations
and enabling further developments in backends :cite:`pivarski2020awkward`.

Scikit-HEP had become a reasonably popular "toolset" for HEP analysis in Python, a collection of packages that worked together,
instead of a "toolkit" like ROOT, which is one monopackage that tries to
provide everything :cite:`Rodrigues:2020syo`.  A toolset is more natural in the
Python ecosystem, where we have good packaging tools and many existing
libraries. Scikit-HEP only needed to fill existing gaps, instead of covering
every possible aspect of an analysis like ROOT (from 1994) did. The
``scikit-hep`` package started to be pulled out into separate packages, and
instead simply was becoming a metapackage that would install a useful subset of
libraries for a physicist starting a new analysis.


Broader ecosystem
-----------------

Scikit-HEP was quickly becoming the center of Python focused analysis in HEP
(see Fig. :ref:`fig-shells`).  Several other projects or packages joined Scikit-HEP, like
iMinuit, a popular HEP and astrophysics fitting library, probably the most
popular single package to have joined. PyHF and cabinetry also joined; these were
larger frameworks built on Scikit-HEP packages.

Other packages, like Coffea and zFit, were not added, but were built on
Scikit-HEP packages and had developers working closely with Scikit-HEP
maintainers. Scikit-HEP introduced an "affiliated" classification, which
allowed an external package to be listed on the Scikit-HEP website. Currently
all affiliated packages have at least one Scikit-HEP developer as a maintainer,
though that is currently not a requirement.  An affiliated package fills a
particular need for the community. Scikit-HEP doesn't have to, or need to, attempt to develop a package that others are providing, but rather tries to ensure that the externally provided package
works well with the broader HEP ecosystem.

Scikit-HEP continues to grow with new packages: for example, vector manipulation,
which had been part of the original scikit-hep "package", and had been
rewritten (as the unreleased HEPVector and also in uproot-methods) was finally
put together into a package "Vector", and include Awkward and Numba backends.
Mplhep added important matplotlib plot types and style for HEP usage.

Histogramming was designed to be a collection of specialized packages
(see Fig. :ref:`fig-histogram`); boost-histogram for manipulation and filling,
Hist for a user-friendly interface and simple plotting tools, histoprint for
displaying histograms, and the existing mplhep and uproot packages also needed
to be able to work with histograms. This ecosystem was build and is held
together with UHI, which is a formal specification, backed by a statically
typed Protocol, for a PlottableHistogram object. Producers of histograms, like
boost-histogram/hist and uproot provide objects that follow this specification,
and users of histograms, such as mplhep and histoprint take any object that
follows this specification. UHI is not required at runtime, though it does
provide a few simple utilities to help a library also accept ROOT histograms,
which do not (currently) follow the Protocol.

One example of a package pulling together many components is
``uproot-browser``, a tool that combines uproot, Hist, and Python libraries
like textual and plotext to provide a terminal browser for ROOT files.

.. figure:: histogram-convergence.pdf

   The collection of histogram packages and related packages in Scikit-HEP.
   :label:`fig-histogram`

Scikit-HEP's external contributions continued to grow. One of the most notable
ones was our work on cibuildwheel. This was a Python package that supported
building redistributable wheels on multiple CI systems. Unlike our own
``azure-wheel-helpers`` or the competing multibuild package, it was written in
Python, so good practices in package design could apply, and it was easy to
remain independent of the underlying CI system. Building wheels on Linux
requires a docker image, macOS requires the python.org Python, and Windows can
use any copy of Python - cibuildwheel uses this to supply Python in all cases,
which keeps it from depending on the CI's support for a particular Python
version. We merged our improvments to cibuildwheel, dropped
azure-wheel-helpers, and eventually joined the cibuildwheel project.
``cibuildwheel`` would go on to join the PyPA, and is now in use in over 600
packages, including ``numpy``, ``matplotlib``, ``mypy``, ``scikit-learn``, and
more.

Our continued contributions to cibuildwheel included a new TOML-based
configuration system for cibuildwheel 2.0, an override system to make
supporting multiple manylinux and musllinux targets easier, build directly from
SDists, option to use ``build`` instead of ``pip``, automatic detection of
python version requirements, better globbing support, and more. We also helped
fully statically type the codebase, apply various checks and style controls,
automate CI processes, improve support for special platforms like CPython 3.8
on macOS Apple Silicon, and much more.

We also have helped with ``build``, ``nox``, ``pyodide``, and many other packages.

The Scikit-HEP Developer Pages
------------------------------

A variety of packaging best practices were coming out of the boost-histogram
work, supporting both ease of installation for users as well as various static
checks and styling to keep the package easy to maintain and reduce bugs. These
techniques would also be useful apply to Scikit-HEP's nearly thirty other
packages, but applying them one-by-one was not scalable. The development and
adoption of ``azure-wheel-helpers`` included a series of blog posts that
covered the Azure Pipelines platform and wheel building details. This ended up
serving as the inspiration for a new set of pages on the Scikit-HEP website for
developers interested in making Python packages. Unlike blog posts, these would
be continuously maintained and extended over the years, serving as a template
and guide for updating and adding packages to Scikit-HEP, and educating new
developers.

These pages grew to describe the best practices for developing and maintaining
a package, covering recommended configuration, style checking, testing,
continuous integration setup, task runners, and more. Shortly after the
introduction of the developer pages, Scikit-HEP developers started asking for a
template to quickly produce new packages following the guidelines. This
was eventually produced; the "cookiecutter" based template is kept in sync with
the developer pages; any new addition to one is also added to the other. The
developer pages are also kept up to date using a CI job that bumps any GitHub
Actions or pre-commit versions to the most recent versions weekly. Some portions
of the developer pages have been contributed to packaging.python.org, as well.

The cookie cutter was developed to be able to support multiple build backends;
the original design was to target both pure Python and Pybind11 based binary
builds.  This has expanded to include 11 different backends by mid 2022,
including Rust extensions, many PEP 621 based backends, and a Scikit-Build
based backend for pybind11 in addition to the classic Setuptools one.  This has
helped work out bugs and influence the design of several PEP 621 packages,
including helping with the addition of PEP 621 to Setuptools.

The most recent addition to the pages was based on a new ``repo-review`` package
which evaluates and existing repository to see what parts of the guidelines are
being followed. This was helpful for monitoring adoption of the developer
pages, especially newer additions, across the Scikit-HEP packages. This package
was then implemented directly into the Scikit-HEP pages, using Pyodide to run
Python in WebAssembly directly inside a user's browser. Now anyone visiting the
page can enter their repository and branch, and see the adoption report in a
couple of seconds.


Working toward the future
-------------------------

Scikit-HEP is looking toward the future in several different areas. We have
been working with the Pyodide developers to support WebAssembly;
boost-histogram is compiled into Pyodide 0.20, and Pyodide's support for
pybind11 packages is significantly better due to that work, including adding
support for C++ exception handling. PyHF's documentation includes a live
Pyodide kernel, and a try-pyhf site (based on the repo-review tool) lets users
run a model without installing anything - it can even be saved as a webapp on
mobile devices.

We have also been working with Scikit-Build to try to provide a modern build
experience in Python using CMake. This project is just starting, but we expect
over the next year or two that the usage of CMake as a first class build tool
for binaries in Python will be possible using modern developments and avoiding
distutils/setuptools hacks.

Summary
-------


The Scikit-HEP project started in Autumn 2016 and has grown to be a core component in
many HEP analyses. It has also provided packages that are growing in usage
outside of HEP, like AwkwardArray, boost-histogram/Hist, and iMinuit. The
tooling developed and improved by Scikit-HEP has helped Scikit-HEP developers
as well as the broader Python ecosystem. 



.. 
    In this talk attendees will learn about the origins and key features of the
    Scikit-HEP effort. Emphasis will be placed on the underlying infrastructure and
    developments that are not specific to High Energy Physics (HEP), but will learn
    about the methodology of developing highly compatible scientific packages and
    learn key useful outcomes from Scikit-HEP that are general. Attendees will take
    away knowledge about a variety of useful tools both inside and supporting the
    Scikit-HEP ecosystem. 

    Scikit-HEP started in in 2016 in response to a need to fill in gaps in the
    scientific Python stack and to consolidate the existing high energy projects.
    The first major success was uproot, a pure Python interpretation of the
    HEP-specific ROOT analysis framework.This enabled easy access to files that
    previously look a complex, multi-hour compile to access. ROOT also, however,
    had something special: a tree/branch structure that held “jagged” data. The
    library AwkwardArray was created as a response to pythonizing this data, and
    has since grown to be useful to a wide variety of disciplines. It has numba
    support, integration with our Vector package, and is gaining GPU and Dask
    support. 

    The next major success of Scikit-HEP was the boost-histogram family, which
    brought fast bindings for the C++ Boost libraries. One of the key advancements
    has been UHI, a library providing a statically typed protocol that different
    libraries in the ecosystem can conform to; this allows the histogram
    production/reading tool and plotting tools to avoid having any
    interdependencies; histoprint can display an uproot histogram without adding a
    dependency on boost-histogram or hist to either library. The development of
    boost-histogram has prompted a variety of tooling improvements affecting the
    whole Python binary packaging ecosystem. Pybind11 gained much better CMake and
    setuptools support. Cibuildwheel received improvements for supporting static,
    overridable configuration and local builds. 

    Possibly the most general tool in Scikit-HEP is the developer pages, which
    helps guide the design and packaging of the family of libraries for our
    different developers, as well as has influenced the python.packaging.org
    webpages. We will look at the process of making a new package using
    scikit-hep/cookie, which supports 9 build backends including binary builds with
    C++ and Rust and dozens of useful correctness and style checking additions, all
    explicitly explained and kept in sync with the developer pages. This has
    enabled consistency across the package ecosystem. 

    We will finish with a few of the cutting edge ventures of the Scikit-HEP
    project, including pyodide WebAssembly support, plans for integration with
    Scikit-Build, and more. 
