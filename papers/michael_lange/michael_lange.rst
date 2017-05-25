:author: Michael Lange
:email: michael.lange@imperial.ac.uk
:institution: Imperial College London
:corresponding:

:author: Navjot Kukreja
:email: n.kukreja@imperial.ac.uk
:institution: Imperial College London

:author: Fabio Luporini
:email: f.luporini12@imperial.ac.uk
:institution: Imperial College London

:author: Mathias Louboutin
:email: mloubout@eos.ubc.ca
:institution: The University of British Columbia

:author: Gerard J. Gorman
:email: g.gorman@imperial.ac.uk
:institution: Imperial College London

---------------------------------------------------------------
Optimised finite difference computation from symbolic equations
---------------------------------------------------------------

.. class:: abstract

Domain-specific high-productivity environments are playing an
increasingly important role in scientific computing due to the
levels of abstraction and automation they provide. In this
paper we introduce Devito, an open-source domain-specific framework for
solving partial differential equations from symbolic problem
definitions by the finite difference method. We highlight the
generation and automated execution of highly optimized stencil code
from only a few lines of high-level symbolic Python for a set of
operators used in seismic inversion problems, before exploring the use
of Devito for a range of scientific equations.

.. class:: keywords

    Finite difference, domain-specific languages, symbolic Python

Introduction
------------

Domain-specific high-productivity environments are playing an
increasingly important role in scientific computing. The level
of abstraction and automation provided by such frameworks not only
increases productivity and accelerates innovation, but also allows the
combination of expertise from different specialised disciplines. This
synergy is necessary when creating the complex software stack needed
to solve leading edge scientific problems, since domain specialists as
well as high performance computing experts are required to fully
leverage modern computing architectures. Based on this philosophy we
introduce Devito, an open-source domain-specific framework for solving
partial differential equations from symbolic problem definitions by
the finite difference method.

Symbolic computation, where optimized numerical code is automatically
derived from a high-level problem definition, is a powerful technique
that allows domain scientists to focus on algorithmic development
rather than implementation details. For this reason Devito exposes an
API based on Python (SymPy) [Meuer17]_ that allow users to express
equations symbolically, from which it generates and executes optimized
stencil code via just-in-time (JIT) compilation. Using latest advances
in stencil compiler research, Devito thus provides domain scientists
with the ability to quickly and efficiently generate high-performance
kernels from only a few lines of Python code, making Devito composable
with existing open-source software.

While Devito was originally developed for seismic imaging workflows,
the automated generation and optimisation of stencil codes can be
utilised for a much broader set of computational problems. In this
paper we will give a brief overview of the design concepts and
Devito's key features before demonstrating the Devito API on a set of
classic examples from computational fluid dynamics (CFD). Based on
this, we will then highlight the use of Devito in an example of a
complex seismic inversion algorithms to demonstrate its use in
practical scientific applications and to showcase the performance
achieved by the auto-generated and optimised code.

Background
----------

*<Embedded DSLs in Python, including Fenics/UFL and Firedrake, leading
to the power of symbolic adjoint computation.>*

*<Use of DSLs for FD and stencil optimisation frameworks.>*

Design and API
--------------

The primary objective of Devito is to enable the quick and effective
creation of highly optimised finite difference operators for use in a
realistic scientific application context. As such, its design is
centred around composability with the existing Python software stack
to provide users with the tools to dynamically generate optimised
stencil computation kernels, but also to enable access to the full
scientific software stack in the Python ecosystem. In addition, to
accommodate the needs of "real life" scientific applications, a
secondary API is provided that enables users to inject custom
expressions, such as boundary conditions or sparse point
interpolation, into the generate kernels.

Moreover, the use of SymPy as the driver for the symbolic generation
of stencil expressions and the subsequent code-generation are at the
heart of the Devito philosophy. While SymPy is fully capable of
auto-generating low-level C code for pre-compiled execution from
high-level symbolic expressions, Devito is designed to combine theses
capabilities with the latest advances in stencil compiler technology.
The result is a framework that is capable of automatically generating
and optimising complex stencil code from high-level symbolic
definitions.

The Devito API is based around two key concepts that allow users to
express finite difference problems in a concise symbolic notation:

* **Symbolic data objects:** Devito's high-level symbolic objects
  behave like :code:`sympy.Function` objects and provide a set of
  shorthand notations for generating derivative expressions, while
  also managing user data. The rationale for this duality is that many
  stencil optimisation algorithms rely on data layout changes,
  mandating that Devito needs to be in control of data allocation and
  access.

* **Operator:** An :code:`Operator` creates, compiles and executes a
  single executable kernel from a set of SymPy expressions. The code
  generation and optimisation process involves various stages and
  accepts a mixture of high-level and low-level expressions to allow
  the injection of customised code.

Fluid Dynamics Examples
-----------------------

In the following section we are going to demonstrate the use of the
Devito API to implement two examples from classical fluid dynamics,
before demonstrating Devito operators in a seismic inversion context.
Both CFD examples are based in part on tutorials from the introductory
blog "CFD Python: 12 steps to Navier-Stokes"[#]_ by the
Lorena A. Barba group.

.. [#] http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

Linear Convection
~~~~~~~~~~~~~~~~~

We will demonstrate a basic Devito operator definition based on a
linear two-dimensional convection flow (step 5 in the original
tutorials). The governing equation we are implementing here is:

.. math::
   :label: 2dconvection

   \frac{\partial u}{\partial t}+c\frac{\partial u}{\partial x}
           + c\frac{\partial u}{\partial y} = 0

A discretised version of this equation, using a forward difference
scheme in time and a backward difference scheme in space might be written
as

.. math::
   :label: 2dconvdiscr

   u_{i,j}^{n+1} = u_{i,j}^n-c \frac{\Delta t}{\Delta x}(u_{i,j}^n-u_{i-1,j}^n)
   - c \frac{\Delta t}{\Delta y}(u_{i,j}^n-u_{i,j-1}^n)

where the subscripts :math:`i` and :math:`j` denote indices in the
space dimensions and the superscript :math:`n` denotes the index in
time, while :math:`\Delta t`, :math:`\Delta x`, :math:`\Delta y`
denote the spacing in time and space dimensions respectively.

The first thing we need is a function object that we can take build
a timestepping scheme with. For this purpose Devito provides so-called
:code:`TimeData` objects that encapsulate functions that one may take space
and time derivatives of.

.. code-block:: python

    from devito import *

    c = 1.
    u = TimeData(name='u', shape=(nx, ny))

    u_dx = first_derivative(u, dim=x, side=left)
    u_dy = first_derivative(u, dim=y, side=left)

    eq = Eq(u.dt + c * u_dx + c * u_dy)

    [In] print eq
    [Out] Eq(-u(t, x, y)/s + u(t + s, x, y)/s
            + 2.0*u(t, x, y)/h - 1.0*u(t, x, y - h)/h
            - 1.0*u(t, x - h, y)/h, 0)

The above expression results in a :code:`sympy.Equation` object that
contains the fully discretised form of Eq. :ref:`2dconvection`,
including placeholder symbols for spacing in space (:code:`h`) and
time (:code:`s`). These spacing symbols will be resolved during the
code generation process, as described in **FORWARD-REF**. It is also
important to note here that the explicit generation of the space
derivatives :code:`u_dx` and :code:`u_dy` is due to the use of a
backward derivative in space to align with the original example. A
similar notation to the forward derivative in time (:code:`u.dt`) will
soon be provided.

In order to create a functional :code:`Operator` object, the
expression :code:`eq` needs to be re-arranged so that we may solve for
the unknown :math:`u_{i,j}^{n+1}`. This is easily achieved by using
SymPy's :code:`solve` utility and the Devito shorthand
:code:`u.forward` which denotes the furthest forward stencil point in
a time derivative (:math:`u_{i,j}^{n+1}`).

.. code-block:: python

   from sympy import solve

   stencil = solve(eq, u.forward)[0]

   [In] print(stencil)
   [Out] (h*u(t, x, y) - 2.0*s*u(t, x, y)
        + s*u(t, x, y - h) + s*u(t, x - h, y))/h

The above variable :code:`stencil` now represents the RHS of
Eq. :ref:`2dconvdiscr`, allowing us to construct a SymPy expression
that updates :math:`u_{i,j}^{n+1}` and build a :code:`devito.Operator`
from it. When creating this operator we also supply concnrete values
for the spacing terms :code:`h` and :code:`s` via an additional
substitution map argument :code:`subs`.

.. code-block:: python

   op = Operator(Eq(u.forward, stencil),
                 subs={h: dx, s:dt})

   # Set initial condition as a "hat function"
   u.data[:] = 1.
   u.data[int(.5 / dx):int(1 / dx + 1),
          int(.5 / dy):int(1 / dy + 1)] = 2.

   op(u=u, time=100)  # Apply for 100 timesteps

Using this operator we can now re-create the example from the original
tutorial by initialising the data associated with the symbolic function
:math:`u`, :code:`u.data`,  with a "hat function" according to

.. math::
   :type: eqnarray

   2\ \text{for}\ 0.5 \leq x, y \leq 1 \\
   1\ \text{everywhere else}

.. figure:: 2dconv_init.png
   :scale: 42%
   :figclass: hbt

   Initial condition of :code:`u.data` in the 2D convection
   example. :label:`fig2dconv`

.. figure:: 2dconv_final.png
   :scale: 42%
   :figclass: hbt

   State of :code:`u.data` after 100 timesteps in convection
   example. :label:`fig2dconvfinal`

The initial condition and the final result after executing the operator
for 100 timesteps are depicted in Figures :ref:`fig2dconv` and
:ref:`fig2dconvfinal` respectively.


Laplace equation
~~~~~~~~~~~~~~~~

The above example showed how Devito can be used to create finite
difference stencil oprators from only a few lines of high-level
symbolic code. For more complex examples, boundary conditions are
required though, which are not currently provided through the symbolic
high-level API. However, for exactly this reason, Devito provides a
low-level, or "indexed" API, where custom SymPy expressions can be
created with explicitly resolved grid accesses to manually inject
custom code into the auto-generation toolchain.

To demonstrate this, we will use the Laplace example from the original
CFD tutorials (step 9), which implements the steady-state heat equation
with Dirichlet and Neuman boundary conditions. The governing equation
for this problem is

.. math::
   :label: 2dlaplace

   \frac{\partial ^2 p}{\partial x^2} + \frac{\partial ^2 p}{\partial y^2} = 0

The rearranged discretised form is

.. math::
   :label: 2dlaplace_discr

   p_{i,j}^n = \frac{\Delta y^2(p_{i+1,j}^n+p_{i-1,j}^n)
           +\Delta x^2(p_{i,j+1}^n + p_{i,j-1}^n)}
           {2(\Delta x^2 + \Delta y^2)}

Using a similar approach to the previous example, we can construct
the SymPy expression to update the state of a field :math:`p`. For
demonstration purposes we will use two separate function objects
of type :code:`DenseData` in this example, since the Laplace equation
does not contain a time-dependence.

*<Some words on the BC types and the definition of the prescibed BC>*

.. code-block:: python

   # Create two separate symbols with space dimensions
   p = DenseData(name='p', shape=(nx, ny),
                 space_order=2)
   pn = DenseData(name='pn', shape=(nx, ny),
                  space_order=2)

   # Create an additional symbol for our prescibed BC
   bc_right = DenseData(name='bc_right', shape=(nx, ),
                        dimensions=(x, ))
   bc_right.data[:] = np.linspace(0, 1, nx)

   # Define equation and sovle for the central point
   eq = Eq(a * pn.laplace)
   stencil = solve(eq, pn)[0]
   # The update expression to populate buffer `p`
   eq_stencil = Eq(p, stencil)

   # Create explicit boundary condition expressions
   bc = [Eq(p.indexed[x, 0], 0.)]
   bc += [Eq(p.indexed[x, ny-1], bc_right.indexed[x])]
   bc += [Eq(p.indexed[0, y], p.indexed[1, y])]
   bc += [Eq(p.indexed[nx-1, y], p.indexed[nx-2, y])]

   # Build operator with update and BC expressions
   op = Operator(expressions=[eq_stencil] + bc,
                 subs={h: dx, a: 1.})

After buildign the operator, we can now use it in a time-independent
conversion loop, but we do need to make sure we switch between buffers.

.. code-block:: python

   l1norm = 1
   counter = 0
   while l1norm > 1.e-4:
       # Determine buffer order
       if counter % 2 == 0:
           _p, _pn = p, pn
       else:
           _p, _pn = pn, p

       # Apply operator
       op(p=_p, pn=_pn)

       # Compute L1 norm
       l1norm = (np.sum(np.abs(_p.data[:])
                 - np.abs(_pn.data[:]))
                 / np.sum(np.abs(_pn.data[:])))
       counter += 1


.. figure:: 2dlaplace_init.png
   :scale: 42%

   Initial condition of :code:`pn.data` in the 2D Laplace example.

.. figure:: 2dlaplace_final.png
   :scale: 42%

   State of :code:`p.data` after convergence in Laplace example.

Seismic Inversion Example
-------------------------

The primary motivating application behind the design of Devito are
seismic exploration problems that require highly optimized wave
propagation operators for forward modelling and adjoint-based
inversion. Obviously, the speed and accuracy of the generated kernels
is of vital importance. Moreover, the ability to efficiently define rigorous
forward modelling and adjoint operators from high-level symbolic
definitions also implies that domain scientists are able to quickly
adjust the numerical method and discretisation to the individual problem
and hardware architecture **[CITE]**. In the following example we will
demonstrate the generation of forward and adjoint operators for the
acoustic wave equation to implement the so-called adjoint test. The
governing equation is defined as

.. math::
    m \frac{\partial^2 u}{\partial t^2}
    + \eta \frac{\partial u}{\partial t} - \nabla^2 u = q

where :math:`u` denotes the pressure wave field, :math:`m` is the
square slowness, :math:`q` is the source term and :math:`\eta` denotes
the spatially varying dampening factor used to implement an absorbing
boundary condition.

On top of fast stencil operators, seismic inversion kernels also rely
on sparse point interpolation to inject the modelled wave as a point
source (:math:`q`) and to infer the recorded value at individual point
locations. To accomodate this, Devito provides another symbolic data
type :code:`PointData`, which allows the generation of sparse-point
interpolation expressions using the "indexed" low-level API. These
symbolic objects provide utility routines
:code:`pt.interpolate(expression)` and :code:`pt.inject(field,
expression)` to create symbolic expressions that perform linear
interpolation between the sparse points and the cartesian grid for
insertion into :code:`Operator` kernels. A separate set of explicit
coordinate values are associated with the sparse point objects for
this purpose in addition to the function values stored in the
:code:`data` property.

Adjoint Test
~~~~~~~~~~~~

The first step for implementing the adjoint test is to build a forward
operator that models the wave propagating through an anisotropic
medium, where the square slowness of the wave is denoted as :math:`m`.
Since :code:`m`, as well as the boundary dampening function
:code:`eta`, are re-used between forward and adjoint runs the only
symbolic data object we need to create here is the wavefield :code:`u`
in order to implement amd re-arrange our discretised equation
:code:`eqn` to form the update expression for :code:`u`.  It is
important to note here that the spatial discretisation of the
:code:`u.laplace` term is provided by the user.
      
In addition to the state update of :code:`u`, we are also inserting
two additional terms into the forward modelling operator:

* :code:`src_term` injects the modelled wave at a point location
  according to a prescribed time series stored in :code:`src.data`
  that is accessible in symbolic form via the symbol :code:`src` in
  the term :code:`src * dt**2 / m`. **[ADD LINE NUMBERS]**

* :code:`rec_term` adds the expression to interpolate the wavefield
  :code:`u` for a set of "receiver" hydrophones that measure the
  propagated wave at a varying distances from the source. The
  resulting interpolated point data will be stored in
  :code:`rec.data` and is accessible to the user as a NumPy array.

.. code-block:: python

   def forward(model, m, eta, src, rec, order=2):
       # Create the wavefeld function
       u = TimeData(name='u', shape=model.shape,
                    time_order=2, space_order=order)

       # Derive stencil from symbolic equation
       eqn = m * u.dt2 - u.laplace + eta * u.dt
       stencil = solve(eqn, u.forward)[0]
       update_u = [Eq(u.forward, stencil)]

       # Add source injection and receiver interpolation
       src_term = src.inject(field=u,
                             expr=src * dt**2 / m)
       rec_term = rec.interpolate(expr=u)

       # Create operator with source and receiver terms
       return Operator(update_u + src_term + rec_term,
                       subs={s: dt, h: model.spacing})

After buildign a forward operator, we can now implement the adjoint
operator in a similar fashion. Using the provided symbols :code:`m`
and :code:`eta`, we can again define the adjoint wavefield :code:`v`
and implement its update expression from the discretised
equation. However, since the adjoint operator needs to operato
backwards in time there are two notable differences:

* The update expression now updates the backward stencil point in the
  time derivative :math:`v_{i,j}^{n-1}`, denoted as
  :code:`v.backward`.  In addition to that, the :code:`Operator` is
  forced to invert its internal time loop by providing the argument
  :code:`time_axis=Backward`
* Since the acoustic wave equation is self-adjoint, the only change
  required in the governing equation is to inverte the dampening term
  :code:`eta * u.dt`.

Morevover, the role of the sparse point objects has now switched:
Instead of injecting the source term, we are now injecting the
previously recorded receiver values into the adjoint wavefield, while
we are interpolating the resulting wave at the original source
location. The difference between the measured readings in the adjoint
run and the original time series is the core error measure of the
adjoint test.

.. code-block:: python

   def adjoint(model, m, eta, srca, rec, order=2):
       # Create the adjoint wavefeld function
       v = TimeData(name='v', shape=model.shape,
                    time_order=2, space_order=order)

       # Derive stencil from symbolic equation
       # Note the inversion of the dampening term
       eqn = m * v.dt2 - v.laplace - eta * v.dt
       stencil = solve(eqn, u.forward)[0]
       update_v = [Eq(v.backward, stencil)]

       # Inject the previous receiver readings
       rec_term = rec.inject(field=v,
                             expr=rec * dt**2 / m)

       # Interpolate the adjoint-source
       srca_term = srca.interpolate(expr=v)

       # Create operator with source and receiver terms
       return Operator(update_v + rec_term + srca_term,
                       subs={s: dt, h: model.spacing},
                       time_axis=Backward)

Having established how to build the required operators we can now
define the workflow for our adjoint example.  For illustration
purposes we are using a utility object :code:`Model` that provides the
core information for seismic inversion runs, such as the values for
:code:`m` and the dampening term :code:`eta`, as well as the
coordinates of the point source and receiver hydrophones. It is worth
noting that the spatial discretisation and thus the stencil size of
the oeprators is still fully parameterisable.

.. code-block:: python

   # Create the seismic model of the domain
   model = Model(...)

   # Create source with Ricker wavelet
   src = PointData(name='src', ntime=ntime,
                   ndim=2, npoint=1)
   src.data[0, :] = ricker_wavelet(ntime)
   src.coordinates.data[:] = source_coords

   # Create empty set of receivers
   rec = PointData(name='rec', ntime=ntime,
                   ndim=2, npoint=101)
   rec.coordinates.data[:] = receiver_coords

   # Create empty adjoint source symbol
   srca = PointData(name='srca', ntime=ntime,
                    ndim=2, npoint=1)
   srca.coordinates.data[:] = source_coords

   # Create symbol for square slowness
   m = DenseData(name='m', shape=model.shape,
                 space_order=order)
   m.data[:] = model  # Set m from model data

   # Create dampening term from model
   eta = DenseData(name='eta', shape=shape,
                   space_order=order)
   eta.data[:] = model.dampening

   # Execute foward and adjoint runs
   fwd = forward(model, m, eta, src, rec)
   fwd(time=ntime)
   adj = adjoint(model, m, eta, srca, rec)
   adj(time=ntime)

   # Test prescribed against adjoint source
   adjoint_test(src.data, srca.data)


.. figure:: shot_record.png
   :scale: 50%

   *<Shot record of the measured point values in
   :code:`rec.data`.>* :label:`figshotrecord`

The above test can be used to verify the accuracy of the forward
propagation and adjoint operators and has been shown to agree for
2D and 3D implementations **[CITE]**. The shot record of the data
measured at the receiver locations after the forward run is shown
in :ref:`figshotrecord`.


Automated code generation
-------------------------

The role of the :code:`Operator` in the previous examples is to
generate semantically equivalent C code to the provided SymPy
expressions, complete with loop constructs and annotations for
performance optimisation, such as OpenMP pragmas. Unlike many other
DSL-based frameworks, Devito employs actual compiler technology during
the code generation and optimisation process. The symbolic
specification is progressively lowered to C code through a series of
passes manipulating abstract syntax trees (AST), rather than working
with rigid templates. This software engineering choice has an
invaluable impact on maintainability, extensibility and composability.

The code generation process consists of a sequence of compiler passes,
which progressively lower the symbolic representation to C. Following
the initial resolution of explicit grid indices into the low-level
format, Devito is able to apply several types of automated performance
optimisation throughout the code generation pipeline, which are grouped
into two distinct sub-modules:

* **DSE - Devito Symbolic Engine:** The first set of optimisation
  passes consists of manipulating SymPy equations with the aim to
  decrease the number of floating-point operations performed when
  evaluating a single grid point. This initial optimisation is
  performed following an initial analysis of the provided expressions
  and consists of sub-passes such as common sub-expressions
  elimination, detection and promotion of time-invariants, and
  factorization of common finite-difference weights. These
  transformations not only optimize the operation count, but they also
  improve the symbolic processing and low-level compilation times of
  later processing stages.

* **DLE - Devito Loop Engine:** After the initial symbolic processing
  Devito schedules the optimised expressions in a set of loops by
  creating an Abstract Syntax Tree (AST). The loop engine (DLE) is now
  able to perform typical loop-level optimisations in multiple passes
  by manipulating this AST, including data alignment through array
  annotations and padding, SIMD vectorization through OpenMP pragmas
  and thread parallelism through OpenMP pragmas. On top of that, loop
  blocking is used to fully exploit the memory bandwidth of a target
  architecture by increasing data locality and thus cache
  utilization. Since the effectiveness of the blocking technique is
  highly architecture-dependent, Devito can determine optimal block
  size through runtime auto-tuning.

Performance Benchmark
~~~~~~~~~~~~~~~~~~~~~

<*Hardware spec for (Endeavour?) Broadwell nodes.>*

.. figure:: acoustic_dle.pdf
   :scale: 60%

   *<Performance optimisation results for DLE with limited DSE.>*

<*Results for DLE and auto-tuned thread-parallel runs.>*

.. figure:: acoustic_maxperf.pdf
   :scale: 60%

   Performance benchmarks with

*<Full-throttle performance benchmarks. It is worth noting that peak
performance may drop with DSE, but that is expected as less "empty
flops" are performed.>*


Integration with YASK
~~~~~~~~~~~~~~~~~~~~~

*<YASK, and why it is so great.>* **[CITE]**

*<Ongoing integration effort as an alternative backend. Also
highlighting that this underpins the generality idea of the backend
engines.>*

Discussion
----------

References
----------
.. [Meuer17] Meurer A, Smith CP, Paprocki M, Čertík O, Kirpichev SB,
             Rocklin M, Kumar A, Ivanov S, Moore JK, Singh S,
             Rathnayake T, Vig S, Granger BE, Muller RP, Bonazzi F,
             Gupta H, Vats S, Johansson F, Pedregosa F, Curry MJ,
             Terrel AR, Roučka Š, Saboo A, Fernando I, Kulal S,
             Cimrman R, Scopatz A. (2017) SymPy: symbolic computing in
             Python. PeerJ Computer Science 3:e103
             https://doi.org/10.7717/peerj-cs.103
