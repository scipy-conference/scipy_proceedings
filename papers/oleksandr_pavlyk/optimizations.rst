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

:year: 2017

:video: http://www.youtube.com/watch?v=

-------------------------------------------------------
Accelerating Scientific Python with Intel Optimizations
-------------------------------------------------------

.. class:: abstract

    It is well known that the performance difference between python and basic C code can be 200x.
    Did you know that for numerically intensive code there is another 240x or more speedup possible?
    The performance comes from software’s ability to take advantage of your CPU's multiple cores,
    SIMD instructions, and high performance caches.
    This talk is for python programmers who want to get the most out of their hardware
    but do not have time or expertise to re-code their applications using native extensions,
    Cython, or Just-In-Time compilers that generate native code.

.. class:: keywords

   numpy,scipy,sk-learn,numba,simd,parall,optimization,performance

Introduction
------------

Intel released new version of its Distribution.. [TBC]
We suggest detailed report about optimization work we accomplished for Intel Distribution for Python,
which might be interesting for developers of SciPy tools.
It already offered great performance improvements for NumPy*, SciPy*, and Scikit-learn*
that you can see across a range of Intel processors,
from Intel |R| Core |TM| CPUs to Intel |R| Xeon |R| and Intel |R| Xeon Phi |TM| processors.
Here is the list of what we did for update 2:

Fast Fourier Transforms
-----------------------
In addition to initial Fast Fourier Transforms (FFT) optimizations offered in previous releases, Update 2 brings widespread optimizations for NumPy and SciPy FFT. It offers a layered interface for the Intel |R| Math Kernel Library (Intel |R| MKL) that allows efficient access to native FFT optimizations from a range of NumPy and SciPy functions. The optimizations include real and complex data types, both single and double precision. Update 2 covers both one-dimensional and multi-dimensional data, as well as in place and out of place modes of operation. As a result, performance may improve up to 60x over Update 1 and is now close to performance of native C/Intel MKL.

.. include:: papers/oleksandr_pavlyk/fft/fft.rst

Arithmetic and transcendental expressions
-----------------------------------------

.. include:: papers/oleksandr_pavlyk/umath/umath.rst


Memory management optimizations
-------------------------------
Update 2 introduces widespread optimizations in NumPy memory management operations. As a dynamic language, Python manages memory for the user. Memory operations, such as allocation, de-allocation, copy, and move, affect performance of essentially all Python programs.
Specifically, Update 2 ensures NumPy allocates arrays that are properly aligned in memory on Linux, so that NumPy and SciPy compute functions can benefit from respective aligned versions of SIMD memory access instructions. This is especially relevant for Intel |R| Xeon Phi |TM| processors.
The most significant improvements in memory optimizations in Update 2 comes from replacing original memory copy and move operations with optimized implementations from Intel MKL. The result: improved performance because these Intel MKL routines are optimized for both a range of Intel CPUs and multiple CPU cores.

Faster Machine Learning with Scikit-learn
-----------------------------------------

.. include:: papers/oleksandr_pavlyk/sklearn/sklearn.rst

Numba vectorization
-------------------
We worked with Continuum Analytics to make Numba to vectorize math code with transcedential functions using Intel SVML library.


Auto-parallelization for Numba
------------------------------

.. include:: papers/oleksandr_pavlyk/parallel/parallel.rst


Summary
-------
The Intel Distribution for Python is powered by Anaconda* and conda build infrastructures that give all Python users the benefit of interoperability within these two environments and access to the optimized packages through a simple conda install command.
Intel Distribution for Python 2017 Update 2 delivers significant performance optimizations for many core algorithms and Python packages, while maintaining the ease of download and install.


References
----------


.. |C| unicode:: 0xA9 .. copyright sign
   :ltrim:
.. |R| unicode:: 0xAE .. registered sign
   :ltrim:
.. |TM| unicode:: 0x2122 .. trade mark sign
   :ltrim:
