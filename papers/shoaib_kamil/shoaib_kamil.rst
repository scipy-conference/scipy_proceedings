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



.. class:: abstract

    Due to physical limits, processor clock scaling is no longer the path
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
1 page.

Approach/Mechanics of Asp
-------------------------
2 pages including the next 2 sections.  Need to make sure we differentiate between the host language and the transformation language.




Walkthru Example
----------------
Here is where the walkthru of the stencil example goes.


Results
-------
Results.


Other Specializers
------------------
Aside from the stencil specializer, a number of other specializers are currently under development.
We present limited results from two of these: a Gaussian Mixture Model training specializer and
a specializer for the matrix powers computational kernel.

Gaussian Mixture Modeling
.........................
Gaussian Mixture Models (GMMs) are a class of statistical models used in a
wide variety of applications, including image segmentation, speech recognition,
document classification, and many other areas. Training such models is
iterative and highly data parallel, making it amenable to execution on GPUs as
well as modern multicore processors. However, writing high performance GMM training
algorithms are difficult due to the fact that different code variants will perform
better for different problem characteristics. This makes the problem of producing
a library for high performance GMM training amenable to the SEJITS approach.

A specializer using the Asp infrastructure has been built by Cook and Gonina [Co10]_
that targets both CUDA-capable GPUs and Intel multicore processors (with Cilk+).
(Insert figure here from them?)


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

   Naive :math:`A^kx` computation.  Communication required at each level. :label:`akxnaive`

.. figure:: akxpa1.pdf
   :figclass: bt
   :scale: 95%

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
   SciPy's default solver. :label:`akxresults`

To see what kinds of performance improvements are possible using the specialized
communication-avoiding matrix powers kernel, Morlan implemented a conjugate gradient (CG)
solver in Python that uses the specializer. Figure :ref:`akxresults` shows the results for three test
matrices and compares performance against ``scipy.linalg.solve`` which calls the LAPACK
``dgesv`` routine.  Even with just the matrix powers kernel specialized, the CA CG
already outperforms the native C routine used by SciPy.


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



References
----------
.. [SciPy] Scientific Tools for Python. http://www.scipy.org.

.. [Biopy] Biopython.  http://biopython.org.

.. [PIL] Python Imaging Library. http://pythonware.com/products/pil.

.. [Cython] R. Bradshaw, S. Behnel, D. S. Seljebotn, G. Ewing, et al., The Cython compiler, http://cython.org.

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
