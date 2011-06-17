:author: Shoaib Kamil
:email: skamil@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

:author: Derrick Coetzee
:email: dcoetzee@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

:author: Armando Fox
:email: fox@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

------------------------------------------------------------------------------------------------------------
Bringing Parallel Performance to Python  with Domain-Specific Selective Embedded Just-in-Time Specialization
------------------------------------------------------------------------------------------------------------


..    Due to physical limits, processor clock scaling is no longer the path
    to better performance.  Instead, hardware designers are using Moore's law
    scaling to increase the available hardware parallelism on modern processors.
    At the same time, domain scientists are increasingly using modern scripting
    languages such as Python, augmented with C libraries, for productive,
    exploratory science. However, due to Python's limited support for parallelism, these programmers
    have not been able to take advantage of increasingly powerful hardware; in
    addition, many domain scientists do not have the expertise to directly write
    parallel codes for many different kinds of hardware, each with specific
    idiosyncrasies.
    Instead, we propose SEJITS [Cat09]_, a methodology that uses high-level abstractions and the
    capabilities of powerful scripting languages to bridge this
    performance-productivity gap.  SEJITS, or Selective Embedded Just-In-Time Specialization,
    takes code written to use domain-specific abstractions and selectively generates efficient, parallel,
    low-level C++ code, compiles it and runs it, all invisibly to the user.  Efficiency programmers, who 
    know how to obtain the highest performance from a parallel machine, encapsulate their knowledge into 
    domain-specific "specializers", which translate abstractions into
    parallel code.
    We have been implementing Asp, A SEJITS implementation for Python,
    to bring the SEJITS methodology to Python programmers.  Although
    Asp is still under development, the current version shows
    promising results and provides insights and ideas into the
    viability of the SEJITS approach.

.. class:: abstract
    Due to physical limits, processor clock scaling is no longer the path
    to better performance.  Instead, hardware designers are using Moore's law
    scaling to increase the available hardware parallelism on modern processors,
    which are increasingly complex and difficult to program efficiently. "Efficiency"
    languages such as CUDA, OpenCL, and C++ allow programmers to exploit this parallel
    hardware, while more "productive" languages such as Python lack the hardware
    transparency required.  At the same time, Python and other high-level productivity
    languages are easier to program and maintain due to their ability to express
    more functionality in less code.

    As a result, programmers must choose whether to write high-level, maintainable
    code or code that exploits low-level hardware features.  This "performance-productivity
    gap" is addressed by the SEJITS (Selective Embedded Just-in-Time Specialization)
    methodology, which allows programmers to have performance-portable, highly-productive
    code by embedding domain-specific DSLs within high-level languages that are then
    specialized into high-performance low-level code
    by leveraging metaprogramming and introspection features of productivity languages,
    all occurring invisibly to the programmer.

    We present Asp (Asp is a SEJITS implementation for Python) and initial results from
    several domains. Using domain-specific specializers, highly-productive Python code
    can obtain high parallel performance without sacrificing maintainability or portability.

    

   

.. class:: keywords

   parallel programming, specialization

Introduction
------------
Clock speed scaling is no longer a viable way to increase delivered
performance due to physical limitations, and thus, hardware architects
have instead begun focusing on increasing parallelism using multiple
cores to maximize performance capabilities of modern hardware.
Software performance improvements now rely on exploiting the available
parallel hardware, but writing parallel high-performance is
non-trivial and requires understanding intricacies of the particular
architecture.

As this revolution in hardware proceeds, programmers doing exploratory
science have moved away from low-level languages due to their limited
expressiveness when compared to productive high-level languages such
as Python and Matlab, which allow scientists to write code that
closely matches the mathematical expression of their algorithms.
Libraries such as SciPy [SciPy]_ have made the task of scientific
programmers easier by providing constructs and libraries that make
expressing computations simpler.  Performance-critical portions of 
such high-level language packages call code written in C or C++,
either as part of the package (i.e. written using Python's C API) or
by calling out to third-party libraries.  Due to the lack of support
for parallelism within these high-level languages, any parallel code
in computation packages is contained within external low-level libraries.

However, some computations cannot be easily expressed as calls to
statically compiled external libraries.  For example, stencil
computations, which iteratively update points in a structured grid
with a function of a point's neighbors, requires calling a function at
each point; creating a library to do these calculations would require
a function call since the function cannot be inlined.  The dynamic and
high-level nature of modern productivity languages points to an
alternative methodology for producing fast code which does not rely on
interpreter support for parallelism.


High-level productivity or scripting languages have evolved to include
sophisticated introspection and FFI (foreign function interface)
capabilities.  SEJITS [Cat09]_, or Selective Embedded JIT
Specialization, refers to the technique of using these capabilities to
build kernel- and machine-specific *specializers* that transform
user-written code in a high-level language in various ways to expose
parallelism, and then generate the code for a a specific machine.
Then, the code is compiled, linked, and executed.  This entire process
occurs without user knowledge; to the user, it appears that a
interpreted function is being called.


Vision: SEJITS and Asp
----------------------
.. include:: papers/shoaib_kamil/vision.include

Approach/Mechanics of Asp
-------------------------
.. 2 pages including the next 2 sections.  Need to make sure we differentiate between the host language and the transformation language.

Asp brings the SEJITS approach to Python, using Python both as the host language (i.e. 
application programmers write their code in Python) and as the transformation system
(code generation and transformation are also performed in Python). Specializers are
encapsulated through classes and inheritance; other SEJITS implementations could use 
mechanisms such as decorators.

One of Asp's primary purposes is to promote a separation of concerns that separates
application and algorithmic logic from making the application run fast.  Application
writers need only program with high-level class-based constructs provided by 
specializer writers.  It is the task of these specializer writers to ensure the constructs
can be specialized into fast versions using infrastructure provided by the Asp team
as well as third-party libraries.  An overview of this separation is shown in Figure
:ref:`separation`.

.. figure:: separation.pdf
   :figclass: bt

   Separation of concerns in Asp.  App authors write code that is transformed by specializers,
   using Asp infrastructure and third-party libraries. :label:`separation`

In the rest of this section, we outline Asp from the point of view of application writers and
specializer writers, and outline the mechanisms the Asp infrastructure provides.

Application Writers
...................
From the point of view of application writers, using a specializer means installing it and using
the domain-specific classes defined by the specializer, while following the conventions outlined
in the specializer documentation.  As a concrete example of a non-trivial specializer, we have
implemented a specializer for structured grid (stencil) calculations, which provides a ``StencilKernel``
class and a ``StencilGrid`` class (the latter is for the grid over which the stencil operates; it
uses NumPy internally). An application writer merely needs to subclass the ``StencilKernel`` class
and within this subclass, define a function ``kernel()`` which operates on ``StencilGrid`` instances.
As long as the defined kernel function is restricted to the class of stencils outlined in the
documentation, it will be specialized; otherwise the program will still run in pure Python.

An example using our stencil specializer's constructs is shown in Figure :ref:`exampleapp`.

Specializer Writers
...................
Specializer writers use Asp infrastructure to build their domain-specific translators.  In Asp, we
provide two ways to generate low-level code: templates (using Mako [Mako]_) and abstract syntax tree
(AST) transformation. For many kinds of computations, using templates is sufficient to translate from
Python to C++, but for others, phased AST transformation allows application programmers to express
arbitrary computations to specialize.

In the structured grid specializer, the user-defined stencil kernel is first translated into a 
Python AST, and analyzed to see if the specializer can produce correct code. If the application
writer provided a kernel function that adheres to the restrictions of the specializer, the code
is then processed through a series of AST transformations (more details are in the Example Walkthrough
section below). Specializer writers subclass Asp infrastructure classes that implement a visitor
patter on these ASTs (similar to Python's ``ast.NodeTransformer``) to implement their specialization
phases. The last phase transforms the AST into a C++ AST, implemented using CodePy [CodePy_].

Specializer writers can then use the Asp infrastructure to automatically compile, link, and execute
the code in the final AST.  In many cases, the programmer may supply several code variants, represented
by several ASTs, to the Asp infrastructure.  The different variants are run for subsequent calls to the
specialized function until the fastest variant is determined, which is then always called by Asp. Performance
data as well as cached compiled code is captured and stored to disk to be used even across
interpreter startups.

For specializer writers, the bulk of the work consists of exposing an understandable abstraction
for specializer users, ensuring programs execute whether specialized or not, writing test functions
to determine specializability (and giving the user meaningful feedback if not), and 
expressing their translations as phased transforms.

Currently, specializers do have several limitations.  The most important current limitation is
that specialized code cannot call into the Python interpreter due to the interpreter not being
thread safe.  We are implementing functionality to allow serialized calls back into the interpreter
from specialized code.

In the next section, we show an end-to-end walkthrough of an example using our stencil
specializer.

Example Walkthrough
-------------------
In this section we will walk through a complete example of a SEJITS
translation and execution on a simple stencil example. We begin with
the application source shown in Figure :ref:`exampleapp`. This simple
two-dimensional stencil walks over the interior points of a grid and
for each point computes the sum of the four surrounding points.

.. figure:: exampleapp.pdf
   :scale: 80 %
   :align: center

   Example stencil application. Colored source lines match up to nodes of same color in Figure :ref:`pythonast`. :label:`exampleapp`

This code is executable Python and can be run and debugged using
standard Python tools, but is slow. By merely modifying ExampleKernel
to inherit from the StencilKernel base class, we activate the stencil
specializer. Now, the first time the kernel() function is called, the
call is redirected to the stencil specializer, which will translate it
to low-level C++ code, compile it, and then dynamically bind the
machine code to the Python environment and invoke it.

The translation performed by any specializer consists of five main phases, as shown in Figure :ref:`pipeline`:

#. Front end: Translate the application source into a domain-specific intermediate representation (DSIR).
#. Perform platform-independent optimizations on the DSIR using domain knowledge.
#. Select a platform and translate the DSIR into a platform-specific intermediate representation (PSIR).
#. Perform platform-specific optimizations using platform knowledge.
#. Back end: Generate low-level source code, compile, and dynamically bind to make available from the host language.

.. figure:: pipeline.pdf
   :scale: 80 %
   :align: center

   Pipeline architecture of a specializer. :label:`pipeline`

As with any pipeline architecture, each phase's component is reusable
and can be easily replaced with another component, and each component
can be tested independently. This supports porting to other
application languages and other hardware platforms, and helps divide
labor between domain experts and platform performance experts. These
phases are similar to the phases of a typical optimizing compiler, but
are dramatically less complex due to the domain-specific focus and the
Asp framework, which provides utilities to support many common tasks.

In the stencil example, we begin by invoking the Python runtime to
parse the kernel() function and produce the abstract syntax tree shown
in Figure :ref:`pythonast`. The front end walks over this tree and
matches certain patterns of nodes, replacing them with other
nodes. For example, a call to the function interior_points() is
replaced by a domain-specific StencilInterior node. If the walk
encounters any pattern of Python nodes that it doesn't handle, for
example a function call, the translation fails and produces an error
message, and the application falls back on running the kernel()
function as pure Python. In this case, the walk succeeds, resulting in
the DSIR shown in Figure :ref:`dsir`. Asp provides utilities to
facilitate visiting the nodes of a tree and tree pattern matching.

.. figure:: pythonast.pdf
   :scale: 90 %
   :align: center

   Initial Python abstract syntax tree. :label:`pythonast`

.. figure:: dsir.pdf
   :scale: 90 %
   :align: center

   Domain-specific intermediate representation. :label:`dsir`

The second phase uses our knowledge of the stencil domain to perform
platform-independent optimizations. For example, we know that a point
in a two-dimensional grid has four neighbors with known relative
locations, allowing us to unroll the innermost loop, an optimization
that makes sense on all platforms.

The third phase selects a platform and translates to a
platform-specific intermediate representation. In general, the
platform selected will depend on available hardware, performance
characteristics of the machine, and properties of the input (such as
grid size). In this example we will target a multicore platform using
the OpenMP framework. At this point the loop over the interior points
is mapped down to nested parallel for loops, as shown in Figure
:ref:`asir`. The Asp framework provides general utilities for
transforming arithmetic expressions and simple assignments from the
high-level representation used in DSIRs to the low-level
platform-specific representation, which handles the body of the loop.

.. figure:: asir.pdf
   :scale: 70 %
   :align: center

   Application-specific intermediate representation. :label:`asir`

Because the specializer was invoked from the first call of the
kernel() function, the arguments passed to that call are available. In
particular, we know the dimensions of the input grid. By hardcoding
these dimensions into the intermediate representation, we enable a
wider variety of optimizations during all phases, particularly phases
4 and 5. For example, on a small grid such as the 8x8 blocks
encountered in JPEG encoding, the loop over interior points may be
fully unrolled.

The fourth phase performs platform-specific optimizations. For
example, we may partially unroll the inner loop to reduce branch
penalties. This phase is the best place to include autotuning, which
times several variants with different optimization parameters and
selects the best one.

Finally, the fifth phase, the backend, is performed entirely by
third-party components in the Asp framework and CodePy library. The
PSIR is transformed into source code, compiled, and dynamically bound
to the Python environment, which then invokes it and returns the
result to the application. Interoperation between Python and C++ uses
the Boost.Python library, which handles marshalling and conversion of
types.

The compiled kernel() function is cached so that if the function is
called again later, it can be re-invoked directly without the overhead
of specialization and compilation. If the input grid dimensions were
used during optimization, the input dimensions must match on
subsequent calls to reuse the cached version.


Results
-------
Results.

.. figure:: stencilresults.pdf
   :figclass: bt
   :align: center

   Performance as fraction of memory bandwidth peak for two specialized stencil kernels.
   All tests compiled using the Intel C++ compiler 12.0 on a Core i7-840.

Other Specializers
------------------
Aside from the stencil specializer, a number of other specializers are currently under development.
We present limited results from two of these: a Gaussian Mixture Model training specializer and
a specializer for the matrix powers computational kernel.

Gaussian Mixture Modeling
.........................
Gaussian Mixture Models (GMMs) are a class of statistical models used in a
wide variety of applications, including image segmentation, speech recognition,
document classification, and many other areas. Training such models is done
using the Expectation Maximization (EM) algorithm, which is
iterative and highly data parallel, making it amenable to execution on GPUs as
well as modern multicore processors. However, writing high performance GMM training
algorithms are difficult due to the fact that different code variants will perform
better for different problem characteristics. This makes the problem of producing
a library for high performance GMM training amenable to the SEJITS approach.

A specializer using the Asp infrastructure has been built by Cook and Gonina [Co10]_
that targets both CUDA-capable GPUs and Intel multicore processors (with Cilk+).
The specializer implements four different parallelization strategies for the algorithm;
depending on the sizes of the data structures used in GMM training, different strategies
perform better.  Figure :ref:`gmmperf` shows performance for different strategies for
GMM training on an Nvidia Fermi GPU as one of the GMM parameters are varied.  The specializer
uses the best-performing variant (by using the different variants to do one iteration each,
and selecting the best-performing one) for the majority of iterations.  As a result, even
if specialization overhead (code generation, compilation/linking, etc.) is included, the 
specialized GMM training algorithm outperforms the original, hand-tuned CUDA implementation
on some classes of problems, as shown in Figure :ref:`gmmperfoverall`.

.. figure:: gmmperf.pdf
   :figclass: bt
   :align: center

   Runtimes of GMM variants as the D parameter is varied on an Nvidia Fermi GPU (lower is better).  The 
   specializer picks the best-performing variant to run. :label:`gmmperf`

.. figure:: gmmperfoverall.pdf
   :figclass: bt
   :align: center

   Overall performance of specialized GMM training versus original optimized CUDA algorithm.
   Even including specializer overhead, the specialized EM training outperforms the original
   CUDA implementation. :label:`gmmperfoverall`

Matrix Powers
.............
Sentence about CA algorithms. Matrix powers, which computes :math:`\{x, Ax, A^2x, ...,A^kx\}`
for a sparse matrix :math:`A` and vector :math:`x`, is an important building block
for communication-avoiding sparse Krylov solvers. A specializer, currently under development
by Jeffrey Morlan, enables efficient parallel computation of this set of vectors on
multicore processors.

.. figure:: akxnaive.pdf
   :figclass: bt
   :scale: 95%
   :align: center

   Naive :math:`A^kx` computation.  Communication required at each level. :label:`akxnaive`

.. figure:: akxpa1.pdf
   :figclass: bt
   :scale: 95%
   :align: center

   Algorithm PA1 for communication-avoiding matrix powers.  Communication occurs only
   after k levels of computation, at the cost of redundant computation. :label:`akxpa1`

The specializer generates parallel communication avoiding code using the pthread library 
that implements the PA1 [Ho09]_ kernel to compute the vectors more efficiently than
just repeatedly doing the multiplication :math:`A \times x`. The naive
algorithm, shown in Figure :ref:`akxnaive`, requires communication at each level. However, for
many matrices, we can restructure the computation such that communication only occurs
every :math:`k` steps, and before every superstep of :math:`k` steps, all communication
required is completed. At the cost of redundant computation, this reduces the number
of communications required.  Figure :ref:`akxpa1` shows the restructured algorithm.

The specializer implementation further optimizes the PA1 algorithm using traditional
matrix optimization techniques such as cache and register blocking.  Further optimization
using vectorization is in progress.

.. figure:: akxresults.pdf
   :scale: 115%
   :figclass: bht

   Results comparing communication-avoiding CG with our matrix powers specializer and
   SciPy's default solver. FIXME: MACHINE?:label:`akxresults`

To see what kinds of performance improvements are possible using the specialized
communication-avoiding matrix powers kernel, Morlan implemented a conjugate gradient (CG)
solver in Python that uses the specializer. Figure :ref:`akxresults` shows the results for three test
matrices and compares performance against ``scipy.linalg.solve`` which calls the LAPACK
``dgesv`` routine.  Even with just the matrix powers kernel specialized, the CA CG
already outperforms the native solver routine used by SciPy.


Status and Future Plans
------------------------
0.5 page.  AspDB, platform detection.


Related Work
------------
0.5 page.  Auto-tuning, Pochoir, Python stuff.

Allowing domain scientists to program in higher-level languages is the
goal of a number of projects in Python, including SciPy [SciPy]_ which
brings Matlab-like functionality for numeric computations into
Python. In addition, domain-specific projects such as Biopython [Biopy]_
and the Python Imaging Library (PIL) [PIL]_ also attempt to hide complex
operations and data structures behind Python infrastructure, 
making programming simpler for users.  

Another approach, used by the
Weave subpackage of SciPy, allows users to express C++ code
that uses the Python C API as strings, inline with other Python code,
that is then compiled and run.  Cython [Cython]_ is an effort to write
a compiler for a subset of Python, while also allowing users to write
extension code in C.

Paragraph about Copperhead.

The idea of using multiple code variants, with different optimizations 
applied to each variant, is a cornerstone of auto-tuning.  Auto-tuning
was first applied to dense matrix computations in the PHiPAC (Portable
High Performance ANSI C) library [PHiPAC]_. Using parametrized code
generation scripts written in Perl, PHiPAC generated variants of
generalized matrix multiply (GEMM) with loop unrolling, cache
blocking, and a number of other optimizations, plus a search engine,
to, at install time, determine the best GEMM routine for the particular machine.
After PHiPAC, auto-tuning has been applied to a number of domains
including sparse matrix-vector multiplication (SpMV) [OSKI]_, Fast
Fourier Transforms (FFTs) [SPIRAL]_, and multicore versions of 
stencils [KaDa09]_, [Kam10]_, [Poich]_, showing large improvements 
in performance over simple implementations of these kernels.

Acknowledgements
----------------
We would like to acknowledge Henry Cook, Ekaterina Gonina, and Jeffrey Morlan
for their work implementing specializers.  
Research supported by DARPA as well as Microsoft (Award #024263) and Intel (Award #024894) funding 
and by matching funding by U.C. Discovery (Award #DIG07-10227). Additional support 
from Par Lab affiliates National Instruments, NEC, Nokia, Nvidia, Oracle, and Samsung.



References
----------
.. [SciPy] Scientific Tools for Python. http://www.scipy.org.

.. [Biopy] Biopython.  http://biopython.org.

.. [PIL] Python Imaging Library. http://pythonware.com/products/pil.

.. [Cython] R. Bradshaw, S. Behnel, D. S. Seljebotn, G. Ewing, et al., The Cython compiler, http://cython.org.

.. [Mako] Mako Templates for Python. http://www.makotemplates.org

.. [CodePy] CodePy Homepage. http://mathema.tician.de/software/codepy

.. [PHiPAC] J. Bilmes, K. Asanovic, J. Demmel, D. Lam, and
   C.W. Chin. PHiPAC: A Portable, High-Performance, ANSI C Coding
   Methodology and its Application to Matrix Multiply. LAPACK Working Note 111.

.. [KaDa09] K. Datta. Auto-tuning Stencil Codes for Cache-Based
   Multicore Platforms. PhD thesis, EECS Department, University of
   California, Berkeley, Dec 2009.

.. [Kam10] S. Kamil, C. Chan, L. Oliker, J. Shalf, and S. Williams. An
   Auto-Tuning Framework for Parallel Multicore Stencil Computations.
   International Parallel and Distributed Processing Symposium, 2010.

.. [Poich] Y.Tang, R. A. Chowdhury, B. C. Kuszmaul, C.-K. Luk, and
   C. E. Leiserson. The Pochoir Stencil Compiler. 23rd ACM Symposium 
   on Parallelism in Algorithms and Architectures, 2011.

.. [OSKI] OSKI: Optimized Sparse Kernel Interface.  http://bebop.cs.berkeley.edu/oski.

.. [SPIRAL] M. Püschel, J. M. F. Moura, J. Johnson, D. Padua,
    M. Veloso, B. Singer, J. Xiong, F. Franchetti, A. Gacic,
    Y. Voronenko, K. Chen, R. W. Johnson,  N. Rizzolo. 
    SPIRAL: Code generation for DSP transforms. Proceedings of the
    IEEE special issue on "Program Generation, Optimization, and Adaptation".

.. [Cat09] B. Catanzaro, S. Kamil, Y. Lee, K. Asanovic, J. Demmel,
   K. Keutzer, J. Shalf, K. Yelick, A. Fox. SEJITS: Getting
   Productivity and Performance with Selective Embedded Just-in-Time
   Specialization. Workshop on Programming Models for Emerging Architectures (PMEA), 2009

.. [Co10] H. Cook, E. Gonina, S. Kamil, G. Friedland†, D. Patterson, A. Fox.
   CUDA-level Performance with Python-level Productivity for Gaussian Mixture Model Applications.
   3rd USENIX Workshop on Hot Topics in Parallelism (HotPar) 2011.

.. [Ho09] M. Hoemmen. Communication-Avoiding Krylov Subspace Methods.  PhD thesis, EECS Department,
   University of California, Berkeley, May 2010.
