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

The elementwise operator operates on each element of an input array and maps it to
an output array. 

Reduction
~~~~~~~~~

The reduction operator reduces an array to a single value. Given an input array
:math:`(a_0, a_1, a_2, \cdots, a_{n-1})` and an associative binary operator 
:math:`\oplus`, the reduction operation returns the 
value :math:`a_0 \oplus a_1 \oplus \cdots \oplus a_{n-1}`.

Compyle also allows users to give a map expression to map the
input before applying the reduction operator. Following is a simple
example.

.. code-block:: python

    from math import cos, sin
    x = np.linspace(0, 1, 1000)/1000
    y = x.copy()
    x, y = wrap(x, y, backend=backend)

    @annotate
    def map(i=0, x=[0.0], y=[0.0]):
        return cos(x[i])*sin(y[i])

    r = Reduction('a+b', map_func=map, backend=backend)
    result = r(x, y)

Trivial examples
~~~~~~~~~~~~~~~~~

Brute-force N-body simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Laplace equation?
~~~~~~~~~~~~~~~~~~~~~~~~


Scans
~~~~~
Scans are generalizations of prefix sums / cumulative sums and can be used as 
building blocks to construct a number of parallel algorithms. These include but 
not are limited to sorting, polynomial evaluation, and tree operations.

Given an input array
:math:`a = (a_0, a_1, a_2, \cdots, a_{n-1})` and an associative binary operator 
:math:`\oplus`, a prefix sum operation returns the following array

.. math::
   y = \left(a_0, (a_0 \oplus a_1), \cdots, (a_0 \oplus a_1 \oplus \cdots \oplus a_{n-1}) \right)

The scan semantics in compyle are similar to those of the :code:`GenericScanKernel` in PyOpenCL.
This allows us to construct generic scans by having an input expression, an output expression
and a scan operator. The input function takes the input array and the array
index as arguments.
Assuming an input function :math:`f`, the generic scan will return the following array,

.. math::
   y_i = \bigoplus_{k=0}^{i} f(a, k) 

Note that using an input expression :math:`f(a, k) = a_k` gives the same result as a
prefix sum.

The output expression can then be used to map and write the scan result as
required. The output function also operates on the input array and an
index but also has the scan result, the previous item and the last item 
in the scan result available as arguments.

Following is a simple example of a cumulative sum over all elements of an
array.

.. code-block:: python

    ary = np.arange(10000, dtype=np.int32)
    ary = wrap(ary, backend=backend)

    @annotate
    def input_expr(i, ary):
        return ary[i]

    @annotate
    def output_expr(i, item, ary):
        ary[i] = item

    scan = Scan(input_expr, output_expr, 'a+b',
                dtype=np.int32, backend=backend)
    scan(ary=ary)
    ary.pull()

    # Result = ary.data

A more complex example is given in section ??.

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
