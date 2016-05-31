:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:


----------------------------------------------------------------------------------------
PySPH: a reproducible and high-performance framework for smoothed particle hydrodynamics
----------------------------------------------------------------------------------------

.. class:: abstract

    Smoothed Particle Hydrodynamics (SPH) is a general purpose technique to
    numerically compute the solutions to partial differential equations.  The
    method is grid-free and uses particles to discretize the various
    properties of interest.  The method is Lagrangian and particles are moved
    with the local velocity.

    PySPH is an open source framework for Smoothed Particle Hydrodynamics.  It
    is implemented in a mix of Python and Cython.  It is designed to be easy
    to use on multiple platforms, high-performance and support parallel
    execution.  Users write pure-Python code and HPC code is generated on the
    fly, compiled, and executed.  PySPH supports OpenMP and MPI for
    distributed computing.  This is transparent to the user.  PySPH is also
    designed to make it easy to perform reproducible research.  In this paper
    we discuss the design and implementation of PySPH.


Background and Introduction
----------------------------

SPH (Smoothed Particle Hydrodynamics) is a general purpose technique to
numerically compute the solutions to partial differential equations.  The
method is grid-free and uses particles to discretize the various properties of
interest.  The method is Lagrangian and particles are moved with the local
velocity.  The method was originally developed for astrophysical problems
(compressible gas-dynamics) but has since been extended to simulate
incompressible fluids, solid mechanics, free-surface problems and a variety of
other problems.

The SPH method is relatively easy to implement.  This has resulted in a large
number of schemes and implementations proposed by various researchers.  It is
often difficult to reproduce published results due to the variety of
implementations.  While a few standard packages like (SPHysics, DualSPHysics,
JOSEPHINE etc.) exist, they are usually tailor-made for particular
applications and are not general purpose.

Our group has been developing PySPH (http://pysph.bitbucket.org) over 5 years.
PySPH is open source, and distributed under the new BSD license.  Our initial
implementation was based on Cython (http://cython.org) and also featured some
parallelization using MPI.  Unfortunately, this proved difficult to use as
users were forced to implement most of their code in Cython.  It was felt that
we might as well have implemented it all in C++ and exposed a Python interface
to that.

In early 2013, we redesigned PySPH so that users were able to implement an
entire simulation using pure Python.  This was done by auto-generating HPC
code from the pure Python code that users provided.  This version ended up
being faster than our original Cython implementation.  Since we were
auto-generating code, with a bit of additional effort it was possible to
support OpenMP.  PySPH has thus matured into an easy to use, yet
high-performance framework where users can develop their schemes in pure
Python and yet obtain performance close to that of a lower-level language
implementation.  PySPH also supports running on a cluster of machines via MPI.
This is seamless and a serial script using PySPH can be run with almost no
changes using MPI.

PySPH features a reasonable test-suite and we use continuous integration
servers to test it on Linux and Windows.  Our documentation is hosted on
http://pysph.readthedocs.org.  The framework supports several of the standard
SPH algorithms.  A suite of about 30 examples are provided and are shipped as
part of the sources and installed when a user does a pip install for example.
The examples are written in a way that makes it easy to extend and also
perform comparisons between schemes.  These features make PySPH well suited
for reproducible numerical work.  In fact one of our recent papers was written
such that every figure in the paper is automatically generated using PySPH.

In this paper we discuss the use, design, and implementation of PySPH.  In the
next section we provide a high-level overview of the SPH method.

Smoothed Particle Hydrodynamics
-------------------------------

- SPH method introduction.


Numerical implementation
-------------------------

- General numerical approach used for SPH

- Outline of algorithm.

The PySPH framework
-------------------

- Introduction
- Features


High-level overview
~~~~~~~~~~~~~~~~~~~

- Installation
- Getting started
- Visualizing and post-processing
- Using multiple cores.
- Distributed computing.
- Extending PySPH


Essential software engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Git/bitbucket/PRs
- Unit and functional tests
- Documentation
- Continuous integration on multiple platforms


Design overview
~~~~~~~~~~~~~~~~

- General approach and high level objects.

High performance
~~~~~~~~~~~~~~~~~

- How HPC code is generated
- The (ab)use of Mako templates

Parallel processing
~~~~~~~~~~~~~~~~~~~

- Cython/OpenMP related tricks and issues
- Parallel implementation details


Reproducibility
~~~~~~~~~~~~~~~~

- Extensible and OO API
- Shipping examples with the sources
- Reusable examples
- Automation support


Future plans
-------------

- GPU
- Cleanup

Conclusions
-----------


Acknowledgements
-----------------

Thanks to all the major PySPH developers: Kunal, Chandrashekhar, Pankaj, and
others.
