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
1 page.  Talk about GMM and Akx.


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

.. [SPIRAL] M. Püschel, J. M. F. Moura, J. Johnson, D. Padua,
    M. Veloso, B. Singer, J. Xiong, F. Franchetti, A. Gacic,
    Y. Voronenko, K. Chen, R. W. Johnson,  N. Rizzolo. 
    SPIRAL: Code generation for DSP transforms. Proceedings of the
    IEEE special issue on "Program Generation, Optimization, and Adaptation".

.. [Cat09] B. Catanzaro, S. Kamil, Y. Lee, K. Asanovic, J. Demmel,
   K. Keutzer, J. Shalf, K. Yelick, A. Fox. SEJITS: Getting
   Productivity and Performance with Selective Embedded Just-in-Time
   Specialization. Workshop on Programming Models for Emerging Architectures (PMEA), 2009
