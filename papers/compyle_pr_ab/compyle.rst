:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:


:author: Aditya Bhosale
:email: adityapb1546@gmail.com
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India

.. :bibliography: references


------------------------------------
Compyle: Python once, HPC anywhere!
------------------------------------


.. class:: abstract


   Compyle allows users to execute a restricted subset of Python (somewhat
   similar to C) on a variety of HPC platforms. It currently supports
   multi-core execution using Cython, and OpenCL and CUDA for GPU devices.
   Users write low-level code in Python that is automatically transpiled to
   high-performance C. Compyle also provides a few very general purpose and
   useful parallel algorithms that allow users to write code once and have
   them run on a variety of HPC platforms.


.. class:: keywords

   High-performance computing, multi-core CPUs, GPGPU accelerators, parallel
   algorithms, transpilation


Introduction
------------

** FIXME: need to work on this**

In this talk we provide an overview of compyle (https://compyle.rtfd.io).
Compyle is a BSD licensed, Python tool that allows users to write the code
once in Python and have it execute transparently on both multi-core CPUs or
GPGPUs via CUDA or OpenCL. Compyle is available on PyPI and hosted on github
at https://github.com/pypr/compyle

Users often write their code in one language (sometimes a high-performance
language), only to find out later that the platform has changed and that they
can no longer extract best performance on newer hardware. For example, many
scientists still do not make use of GPGPU hardware despite their excellent
performance and availability. One of the problems is that it is often hard to
reuse code developed in one language and expect it to work in all of these
platforms.

Compyle does not provide a greater abstraction of the hardware but allows a
user to write code in pure Python and have that same code execute on multiple
different platforms. We currently support multi-core execution using OpenMP
and Cython, and also transparently support OpenCL and CUDA so the same could
could potentially be reused on a GPGPU. Compyle makes this possible by
providing three important parallel algorithms, an elementwise operation (a
parallel map), a parallel scan (also known as a prefix sum), and a parallel
reduction. The Cython backend provides a native implementation whereas the
OpenCL and CUDA backend simply wrap up the implementation provided by PyOpenCL
and PyCUDA. These three algorithms make it possible to write a variety of
non-trivial parallel algorithms for high performance computing. Compyle also
provides the ability to write custom kernels with support for local/shared
memory specifically for OpenCL and CUDA backends. Compyle provides simple
facilities to annotate arguments and types and can optionally make use of
Python 3's type annotation feature as well. Compyle also features JIT
compilation if desired.

Compyle is quite different from Numba. One major difference is that it does
not rely on LLVM at all and instead performs source-to-source transpilation.
Under the covers, compyle produces simple and readable C or Cython code which
looks similar to the user's original code. Compyle does not provide support
for any high level Python and only works with a highly restricted Python
syntax. While some may argue that this is not user-friendly, we find that in
practice this is vitally important as it ensures that the code users write
will seamlessly execute on both a CPU and a GPU with minimum modifications.
Furthermore compyle provides the basic parallelization algorithms that users
can use to extract good performance from their hardware.

In addition, compyle allows users to generate code using mako templates in
order to maximize code reuse. Since compyle performs source transpilation, it
is also possible to use compyle as a code-generation engine and put together
code from pure Python to build fairly sophisticated computational engines.

Compyle is not a toy and is actively used by a non-trivial, open source, SPH
framework called PySPH (https://pysph.rtfd.io). Compyle makes it possible for
users to write their SPH codes in high-level Python and have it executed on
multi-core and GPU accelerators with negligible changes to their code.

In the future, we would like to improve the package by adding support for
"objects" that would allow users to compose their libraries in a more object
oriented manner. This would also open up the possibility of implementing more
high-level data structures in an easy way.


Motivation and background
--------------------------


High-level overview
--------------------

Parallel algorithms
--------------------

Elementwise
~~~~~~~~~~~

Reduction
~~~~~~~~~

Trivial examples
~~~~~~~~~~~~~~~~~

Brute-force N-body simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Laplace equation?
~~~~~~~~~~~~~~~~~~~~~~~~


Scans
~~~~~

Basic ideas, theory, and assumptions.

A parallel "where"
------------------

Simple nearest neighbors
------------------------


Simple n-body treecode
-----------------------


Limitations
------------


Future work
-------------


Conclusions
-----------
