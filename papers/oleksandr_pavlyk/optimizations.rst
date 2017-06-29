:author: Oleksandr Pavlyk
:email: Oleksandr.Pavlyk@intel.com
:institution: Intel Corporation
:corresponding:

:author: Denis Nagorny
:email: denis.nagorny@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Andres Guzman-Ballen
:email: andres.guzman-ballen@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Anton Malakhov
:email: Anton.Malakhov@intel.com
:institution: Intel Corporation
:equal-contributor:
:supervising-contributor:

:author: Hai Liu
:email: hai.liu@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Ehsan Totoni
:email: ehsan.totoni@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Todd A. Anderson
:email: todd.a.anderson@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Sergey Maidanov
:email: sergey.maidanov@intel.com
:institution: Intel Corporation
:equal-contributor:

:year: 2017

:video: http://www.youtube.com/watch?v=

-------------------------------------------------------
Accelerating Scientific Python with Intel Optimizations
-------------------------------------------------------

.. class:: abstract

    It is well-known that the performance difference between Python and basic C code can be up 200x,
    but for numerically intensive code another speed-up factor of 240x or even greater is possible.
    The performance comes from software's ability to take advantage of your CPU's multiple cores,
    single instruction multiple data (SIMD) instructions, and high performance caches.
    This talk is for Python programmers who want to get the most out of their hardware
    but do not have time or expertise to re-code their applications using techniques such as native 
    extensions or Cython.

.. class:: keywords

   numpy,scipy,scikit-learn,numba,simd,parallel,optimization,performance

Introduction
------------

Scientific software is usually algorthmically rich and compute intensive.
The expressiveness of Python language as well as abundance of quality packages 
offering implementations of advanced algorithms allow scientists and engineers 
alike to code their software in Python. The ability of this software to solve
realistic problems in a reasonable time is often hampered by inefficient use of 
hardware resources. Intel Distribution for Python [IDP]_ attempts to enable 
scientific Python community with optimized computational packages,
such as NumPy*, SciPy*, Scikit-learn*, Numba* and PyDAAL across a range of Intel |R| processors,
from Intel |R| Core |TM| CPUs to Intel |R| Xeon |R| and Intel |R| Xeon Phi |TM| processors.
This paper offers a detailed report about optimization that went into the 
Intel |R| Distribution for Python*, which might be interesting for developers of SciPy tools.


Fast Fourier Transforms
-----------------------

Intel |R| Distribution for Python* offers a thin layered interface for the
Intel |R| Math Kernel Library (Intel |R| MKL) that allows efficient
access to native FFT optimizations from a range of NumPy and SciPy
functions. The optimizations are provided for real and complex data
types in both single and double precision. Update 2 improves
performance of both one-dimensional and multi-dimensional transforms,
for in-place and out-of-place modes of operation. As a result,
Python performance may improve up to 60x over Update 1 and is now close to
performance of native C/Intel MKL.


.. include:: papers/oleksandr_pavlyk/fft/fft.rst

Arithmetic and transcendental expressions
-----------------------------------------

.. include:: papers/oleksandr_pavlyk/umath/umath.rst


Memory management optimizations
-------------------------------

Update 2 introduces extensive optimizations in NumPy memory
management operations. As a dynamic language, Python manages memory
for the user. Memory operations, such as allocation, de-allocation,
copy, and move, affect performance of essentially all Python programs.
Specifically, Update 2 ensures NumPy allocates arrays that are
properly aligned in memory (their address is a multiple of a specific factor, usually 64) on Linux, 
so that NumPy and SciPy compute functions can benefit from respective aligned versions of SIMD memory
access instructions. This is especially relevant for Intel |R| Xeon
Phi |TM| processors.  The most significant improvements in memory
optimizations in Update 2 comes from replacing original memory copy
and move operations with optimized implementations from Intel |R| MKL. The
result: improved performance because these Intel |R| MKL routines are
optimized for both a range of Intel |R| CPUs and multiple CPU cores.

Faster Machine Learning with Scikit-learn
-----------------------------------------

.. include:: papers/oleksandr_pavlyk/sklearn/sklearn.rst

Numba vectorization
-------------------

.. include:: papers/oleksandr_pavlyk/numba/svml.rst

Auto-parallelization for Numba
------------------------------

.. include:: papers/oleksandr_pavlyk/parallel/parallel.rst


Summary
-------

The Intel |R| Distribution for Python is powered by Anaconda* and conda
build infrastructures that give all Python users the benefit of
interoperability within these two environments and access to the
optimized packages through a simple ``conda install`` command.
Intel |R| Distribution for Python* delivers significant
performance optimizations for many core algorithms and Python
packages, while maintaining the ease of downloading and installation.


References
----------


.. |C| unicode:: 0xA9 .. copyright sign
   :ltrim:
.. |R| unicode:: 0xAE .. registered sign
   :ltrim:
.. |TM| unicode:: 0x2122 .. trade mark sign
   :ltrim:

.. [IDP] `Intel |R| Distribution for Python* <http://software.intel.com/en-us/distribution-for-python>`_
