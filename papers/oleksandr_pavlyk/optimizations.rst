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

:year: 2017

:video: http://www.youtube.com/watch?v=

-------------------------------------------------------
Accelerating Scientific Python with Intel Optimizations
-------------------------------------------------------

.. class:: abstract

    It is well known that the performance difference between Python and basic C code can be 200x.
    Did you know that for numerically intensive code there is another 240x or more speedup possible?
    The performance comes from software's ability to take advantage of your CPU's multiple cores,
    single instruction multiple data (SIMD) instructions, and high performance caches.
    This talk is for Python programmers who want to get the most out of their hardware
    but do not have time or expertise to re-code their applications using techniques such as native 
    extensions or Cython.

.. class:: keywords

   numpy,scipy,sk-learn,numba,simd,parallel,optimization,performance

Introduction
------------

Intel released new version of its Distribution [IDP-Release]_. We suggest
detailed report about optimization work we accomplished for Intel |R|
Distribution for Python*, which might be interesting for developers of
SciPy tools. It already offered great performance improvements for
NumPy*, SciPy*, and Scikit-learn* that you can see across a range of
Intel processors, from Intel |R| Core |TM| CPUs to Intel |R| Xeon |R|
and Intel |R| Xeon Phi |TM| processors.


Fast Fourier Transforms
-----------------------

In addition to initial optimizations of Fast Fourier Transform (FFT)
offered in previous releases, Update 2 brings widespread optimizations
for NumPy and SciPy FFT. It offers a think layered interface for the
Intel |R| Math Kernel Library (Intel |R| MKL) that allows efficient
access to native FFT optimizations from a range of NumPy and SciPy
functions. The optimizations are provided for real and complex data
types in both single and double precision. Update 2 improves
performance of both one-dimensional and multi-dimensional transforms,
for in-place and out-of-place modes of operation. As a result,
performance may improve up to 60x over Update 1 and is now close to
performance of native C/Intel MKL.


.. include:: papers/oleksandr_pavlyk/fft/fft.rst

Arithmetic and transcendental expressions
-----------------------------------------

.. include:: papers/oleksandr_pavlyk/umath/umath.rst


Memory management optimizations
-------------------------------

Update 2 introduces widespread optimizations in NumPy memory
management operations. As a dynamic language, Python manages memory
for the user. Memory operations, such as allocation, de-allocation,
copy, and move, affect performance of essentially all Python programs.
Specifically, Update 2 ensures NumPy allocates arrays that are
properly aligned in memory on Linux, so that NumPy and SciPy compute
functions can benefit from respective aligned versions of SIMD memory
access instructions. This is especially relevant for Intel |R| Xeon
Phi |TM| processors.  The most significant improvements in memory
optimizations in Update 2 comes from replacing original memory copy
and move operations with optimized implementations from Intel MKL. The
result: improved performance because these Intel MKL routines are
optimized for both a range of Intel CPUs and multiple CPU cores.

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

The Intel Distribution for Python is powered by Anaconda* and conda
build infrastructures that give all Python users the benefit of
interoperability within these two environments and access to the
optimized packages through a simple conda install command.
Intel Distribution for Python 2017 Update 2 delivers significant
performance optimizations for many core algorithms and Python
packages, while maintaining the ease of download and install.


References
----------


.. |C| unicode:: 0xA9 .. copyright sign
   :ltrim:
.. |R| unicode:: 0xAE .. registered sign
   :ltrim:
.. |TM| unicode:: 0x2122 .. trade mark sign
   :ltrim:

.. [IDP-Release] `Intel Distribution for Python 2017 Update 3 README <https://software.intel.com/en-us/articles/intel-distribution-for-python-2017-update-3-readme>`_
