:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:
:bibliography: references

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
:cite:`lucy77`, :cite:`monaghan77` (compressible gas-dynamics) but has since
been extended to simulate incompressible fluids
:cite:`sph:fsf:monaghan-jcp94`, solid mechanics :cite:`sph-elastic:gray:2001`,
free-surface problems :cite:`sph:fsf:monaghan-jcp94` and a variety of other
problems.  Monaghan :cite:`monaghan-review:2005`, provides a good review of
the method.

The SPH method is relatively easy to implement.  This has resulted in a large
number of schemes and implementations proposed by various researchers.  It is
often difficult to reproduce published results due to the variety of
implementations.  While a few standard packages like (SPHysics
:cite:`sphysics`, DualSPHysics :cite:`dualsphysics`, JOSEPHINE
:cite:`josephine-sph:cpc:2015`, GADGET-2 :cite:`gadget2-springel:mnras:2005`
etc.)  exist, they are usually tailor-made for particular applications and are
not general purpose.  They are all implemented in FORTRAN (77 or 90) or C.
They do not have a convenient Python interface.

Our group has been developing PySPH (http://pysph.bitbucket.org) over the last
5 years.  PySPH is open source, and distributed under the new BSD license.
Our initial implementation was based on Cython :cite:`behnel2010cython` and
also featured some parallelization using MPI.  This was presented at SciPy
2010 :cite:`PR-pysph-scipy-2010`.  Unfortunately, this previous version of
PySPH, proved difficult to use as users were forced to implement most of their
code in Cython.  This was not a matter of simply writing a few high
performance functions in Cython.  Our library is object oriented and
supporting a new SPH formulation would require subclassing one or more classes
and this would need to be done with Cython.  Our type system also ended up
becoming more rigid because if we desired performance as well as a general
purpose code, all the types needed to be pre-defined.  In addition, writing
all this in Cython meant that users had to manage compilation and linking the
Cython code during development.  This was not pleasant.

It was felt that we might as well have implemented it all in C++ and exposed a
Python interface to that.  A traditional compiled language has more developer
tooling around it.  For example debugging, performance tuning, profiling would
all be easier if everything were written in C or C++.  Unfortunately, such a
mixed code-base would not be as easy to use, extend or maintain as a largely
pure Python library.  In our experience, a pure Python library is a lot easier
for say an undergraduate student to grasp and use over a C/C++ code.  Others
are also finding this to be true :cite:`py:nature:2015`.  Many of the top US
universities are teaching Python as their first language
:cite:`py:teaching-us`. This means that a Python library would also be easier
for relatively inexperienced programmers.  It is also true that a Python
library would be easier and shorter to write for the other
non-high-performance aspects (which is often a significant amount of code).
So it seemed that our need for performance was going against our desire for an
easy to use Python library that could be used by programmers who were not
C/C++ developers.

In early 2013, we redesigned PySPH so that users were able to implement an
entire simulation using pure Python.  This was done by auto-generating HPC
code from the pure Python code that users provided.  This version ended up
being faster than our original Cython implementation!  Since we were
auto-generating code, with a bit of additional effort it was possible to
support OpenMP as well.  The external user API did not change so users did not
have to modify their code at all to benefit from this development.  PySPH has
thus matured into an easy to use, yet high-performance framework where users
can develop their schemes in pure Python and yet obtain performance close to
that of a lower-level language implementation.  PySPH has always supported
running on a cluster of machines via MPI.  This is seamless and a serial
script using PySPH can be run with almost no changes using MPI.

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

The SPH method works by approximating the identity:

.. math::

   f(x) = \int f(x') \delta (x-x') dx',

where, :math:`\delta` is the Dirac Delta distribution.  This identity is
approximated using:

.. math::
   :label: eq:delta-approx

   f(x) \approx \int f(x') W (x-x', h) dx',

where :math:`W` is a smooth and compact function and is called the kernel.  It
is an approximate Dirac delta distribution that is parametrized on the
parameter :math:`h` and :math:`W \rightarrow \delta` as :math:`h\rightarrow
0`.  :math:`h` is called the smoothing length or smoothing radius of the
kernel.  The kernel typically will need to satisfy a few properties notably
its area should be unity and if it is symmetric, it can be shown that the
approximation is at least second order in :math:`h`.  The above equation can
be discretized as,

.. math::
   :label: eq:sph-discr

   f(x) \approx \langle f(x) \rangle = \sum_{j \in \mathcal{N}(x)} W(x-x_j, h) f(x_j) \Delta x_j,

where :math:`x_j` is the position of the particle :math:`j`, :math:`\Delta
x_j` is the volume associated with this particle.  :math:`\mathcal{N}(x)` is the
set of particle indices that are in the neighborhood of :math:`x`.  In SPH
each particle carries a mass :math:`m` and associated density :math:`\rho`
with it and the particle volume is typically chosen as
:math:`\Delta x_j = m_j/\rho_j`.  This results in the following SPH
approximation for a function,

.. math::
   :label: eq:sph-approx

   <f(x)> = \sum_{j \in \mathcal{N}(x)} \frac{m_j}{\rho_j} W(x-x_j, h) f(x_j).

Derivatives of functions at a location :math:`x_i` are readily approximated by
taking the derivative of the smooth kernel.  This results in,

.. math::
   :label: eq:deriv-sph-approx

   \frac{\partial f_i}{\partial x_i} = \sum_{j \in \mathcal{N}(x)}
        \frac{m_j}{\rho_j} (f_j - f_i) \frac{\partial W_{ij}}{\partial x_i}.

Here :math:`W_{ij} = W(x_i - x_j)`.  Similar discretizations exist for the
divergence and curl operators.  Given that derivatives can be approximated one
can solve differential equations fairly easily.  For example the conservation
of mass equation for a fluid can be written as,

.. math::
   :label: eq:cons-mass

   \frac{d \rho}{dt} = - \rho \nabla \cdot \vec{v},

where :math:`v` is the velocity of the fluid and the LHS is the material or
total derivative of the density.  The equation :ref:`eq:cons-mass` is in a
Lagrangian form, in that it represents the rate of change of density as one is
moving locally with the fluid.  If an SPH discretization of this equation were
performed we would get,

.. math::
   :label: eq:sph-continuity

   \frac{d \rho_i}{d t} =  -\rho_i \sum_{j \in \mathcal{N}(x)}
   \frac{m_j}{\rho_j} \vec{v}_{ji} \cdot \nabla_i W_{ij},

where :math:`\vec{v}_{ji} = \vec{v}_j - \vec{v}_i`.  This equation is typical
of most SPH discretizations.  SPH can therefore be used to discretize any
differential equation.  This works particularly well for a variety of
continuum mechanics problems.  Consider the momentum equation for an inviscid
fluid,

.. math::
   :label: eq:momentum

   \frac{d \vec{u}}{dt} = - \frac{1}{\rho} \nabla p

A typical SPH discretization of this could be written as,

.. math::
   :label: eq:sph-momentum

   \frac{d \vec{u_i} }{dt} = -\sum_j m_j \left ( \frac{p_j}{\rho_j^2} +
   \frac{p_i}{\rho_i^2} \right) \nabla W_{ij}


More details of these and various other equations can be seen in the review by
Monaghan :cite:`monaghan-review:2005`.  It is easy to see that equations
:ref:`eq:sph-continuity` and :ref:`eq:sph-momentum` are ordinary differential
equations that govern the rate of change of the density and velocity of a
fluid particle.  In principle one can integrate these ODEs to obtain the flow
solution given a suitable initial condition and appropriate boundary
conditions.


Numerical implementation
-------------------------

As discussed in the previous section, in an SPH scheme, the field properties
are first discretized into particles carrying them.  Partial differential
equations are reduced to a system of coupled ordinary differential equations
and discretized using an SPH approximation.  This results in a system of ODEs
for each particle.  These ODEs need to be integrated in time along with
suitable boundary and initial conditions in order to solve a particular
problem.  To summarize, a typical SPH computation proceeds as follows,

- Given an initial condition, the field variables are discretized into
  particles carrying the various properties.
- Depending on the scheme used to integrate the ODEs, the RHS of the ODEs
  needs to be computed (see equations :ref:`eq:sph-continuity` and
  :ref:`eq:sph-momentum`).  These RHS terms are called "accelerations" or
  "acceleration terms".
- Once the RHS is computed the ODE can be integrated using a suitable scheme
  and the fluid properties are found at the next timestep.

The RHS is typically computed as follows:

- Initalize the particle accelerations (i.e. the RHS terms).
- For each particle in the flow, compute the neighbors of the particle which
  will influence the particle.
- For each neighbor compute the acceleration due to that particle and
  increment the acceleration.

Given the total accelerations, the ODEs can be readily integrated with a
variety of schemes.  Any general purpose abstraction of the SPH method must
hence provide functionality to:

1. Easily discretize properties into particles.  This is easily done with
   ``numpy`` arrays representing the property values in Python.
2. Given a particle, compute the neighbors that influence the particle.  This
   is typically called Nearest Neighbor Particle Search (NNPS) in the
   literature.
3. Define the interactions between the particles, i.e. an easy way to specify
   the inter particle accelerations.
4. A way to specify how to integrate the ODEs.

Of the above, the NNPS algorithm is usually a well-known algorithm.  For
incompressible flows where the smoothing radius of the particles, :math:`h`,
is constant, a simple bin-based linked list implementation is standard.  For
cases where :math:`h` varies, a tree-based algorithm is typically used.  Users
usually do not need to experiment or modify these algorithms.


The PySPH framework
-------------------

PySPH allows a user to specify the inter-particle interactions as well as the
ODE integration in pure Python with a rather simple and low-level syntax.
This is described in greater detail further below.  As discussed in the
introduction, with older versions of PySPH as discussed in
:cite:`PR-pysph-scipy-2010`, these interactions would all need to be written
in Cython.  This was not very easy or convenient.  It was also rather
limiting.

The current version of PySPH supports the following:

- Define a complete SPH simulation entirely in Python.
- High-performance code is generated from this high-level Python code
  automatically and called.  The performance of this code is comparable to
  hand-written FORTRAN solvers.
- PySPH can use OpenMP seamlessly.  Users do not need to modify their code at
  all to use this.  This works on Linux, OSX, and Windows and
  produces good scale-up.
- PySPH also works with MPI and once again this is transparent to the user in
  that the user does not have to change code to use multiple machines.  This
  feature requires mpi4py_ and Zoltan_ to be installed.
- PySPH provides a built-in 3D viewer for the particle data generated.  The
  viewer requires Mayavi_ :cite:`it:mayavi:cise:gael2011` to be installed.
- PySPH is also free and currently hosted at http://pysph.bitbucket.org


In the following subsection we provide a high-level overview of PySPH and how
it can be used by a user.  Subsequent subsections discuss the design and
implementation in greater detail.

.. _mpi4py: http://mpi4py.scipy.org
.. _Zoltan: http://www.cs.sandia.gov/zoltan/
.. _Mayavi: http://code.enthought.com/projects/mayavi



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
