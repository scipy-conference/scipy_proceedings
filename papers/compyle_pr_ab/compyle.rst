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
   high-performance Cython or C. Compyle also provides a few very general
   purpose and useful parallel algorithms that allow users to write code once
   and have them run on a variety of HPC platforms.

   In this article, we show how to implement a simple molecular dynamics
   simulation package in pure Python using Compyle. The result is a fully
   parallel program that is quite easy to implement and solves a non-trivial
   problem. The code transparently executes on multi-core CPUs and GPGPUs.
   This is not possible with any of the other tools that are available
   currently in the Python ecosystem.


.. class:: keywords

   High-performance computing, multi-core CPUs, GPGPU accelerators, parallel
   algorithms, transpilation


Motivation and background
--------------------------

In this brief article we provide an overview of compyle
(https://compyle.rtfd.io). Compyle is a BSD licensed, Python tool that allows
users to write code once in pure Python and have it execute transparently on
both multi-core CPUs or GPGPUs via CUDA or OpenCL. Compyle is available on
PyPI and hosted on github at https://github.com/pypr/compyle

Users often write their code in one language (sometimes a high-performance
language), only to find out later that the platform has changed and that they
can no longer extract best performance on newer hardware. For example, many
scientists still do not make use of GPGPU hardware despite their excellent
performance and availability. One of the problems is that it is often hard to
reuse code developed in one language and expect it to work in all of these
platforms.

There are many powerful tools available in the Python ecosystem today that
facilitate high-performance computing. PyPy_ is a Python implementation in
Python that features a JIT that allows one to execute pure Python code at
close to C-speeds. Numba_ uses the LLVM_ compiler infrastructure to generate
machine code that can rival native C code. Numba also supports execution on
GPUs. There are also libraries like Pythran_ that transpile a subset of Python
to C++. Of these, Numba has matured a great deal and is very versatile and
powerful. pybind11_ now makes it a breeze to integrate with C++ and tools like
cppimport_ make it very convenient to use. In addition, there are also
powerful interfaces to GPUs via low-level tools like PyOpenCL_ or PyCUDA_.

Given this context, one may wonder why Compyle exists at all. One answer would
be that compyle grew out of a project that pre-dates numba but the real reason
is that compyle solves a different problem. Understanding this requires a bit
of a context. As a prototypical example, we look at a simple molecular
dynamics simulation where we create :math:`N` particles and these particles
interact with each other via a Lennard-Jones potential as discussed in XXX
(Cite the Schroeder paper).

The typical workflow for a Python programmer would be to prototype this in
pure Python and get a working proof of concept. One would then try to optimize
this code to run fast so as to run larger problems in a smaller amount of
time. Very often this would mean changing some data structures, writing
vectorized code using NumPy arrays, and then resorting to some tools like
numba to extract even more performance. Numba abstracts away a lot of things
and almost works magically well. In fact, for some problems it will even do a
good job of parallelizing your code to run on multiple cores. Now you have
code that runs fast on a CPU. However, running this code on a GPU is entirely
a different ball game. While Numba offers some help here with the CUDA and
ROCm support, you still have to change quite a lot of your code to have it
work on these architectures.

The second major issue is that running code on GPUs requires quite a
significant re-think of the algorithms used. While trivial problems are easy,
most useful computational codes involves non-trivial algorithms. The problem
is that most of us are trained to think of serial algorithms. GPUs are
inherently massively parallel and if you need an algorithm to scale on a GPU
it has to be **fully** parallel.

What compyle attempts to do is to allow you to write your code **once** in a
highly restrictive subset of pure Python and have this run in parallel on both
CPUs and GPUs. This is a significant difference from all the tools that we
have mentioned above.

The difficulty in doing this is that it does require a change in mindset and
also a loss of the typical convenience with high-level Python.

Compyle provides the most important parallel programming algorithms that you
need to write massively parallel codes. These are element wise operations,
reductions, and most important, parallel prefix scans. This allows us to write
algorithms once and have them run on both multi-core CPUs and GPUs with
minimal or no changes whatsoever to the code.

This is currently not possible with any of the other tools. In addition,
Compyle has the following features:

- Generates either Cython or ANSI C code depending on the backend and this
  code is quite readable (to a user familiar with Cython or C). This makes it
  much easier to understand and debug.
- Designed to be relatively easy to use as a code generator.
- Support for templated code generation to minimize repetitive code.
- Highly restrictive language that facilitates cross-platform execution.

We use the prototypical example given above of writing a simple N-particle
molecular dynamics system that is described and discussed in the article by
[XXX Schroeder]. Our goal is to implement this system in pure Python using
Compyle. Through this we demonstrate the ease of use and power of Compyle. We
write a single code that executes efficiently in parallel on CPUs and GPUs. We
use this example to also discuss the three important parallel algorithms and
show how they nicely allow us to solve non-trivial problems.

Compyle is not a toy and is actively used by a non-trivial, open source, SPH
framework called PySPH (https://pysph.rtfd.io). Compyle makes it possible for
users to write their SPH codes in high-level Python and have it executed on
multi-core and GPU accelerators with negligible changes to their code.



.. _PyPy: https://pypy.prg
.. _Numba: http://numba.pydata.org/
.. _Pythran: https://pythran.readthedocs.io/
.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _PyCUDA: https://documen.tician.de/pycoda
.. _LLVM: https://llvm.org/
.. _pybind11: https://pybind11.readthedocs.io/
.. _cppimport: https://github.com/tbenthompson/cppimport


High-level overview
--------------------

We now provide a high-level overview of Compyle and its basic approach. This
is helpful when using ocmpyle.

It is important to keep in mind that Compyle does **not** provide a greater
abstraction of the hardware but allows a user to write code in pure Python and
have that same code execute on multiple different platforms. We currently
support multi-core execution using OpenMP and Cython, and also transparently
support OpenCL and CUDA so the same could could potentially be reused on a
GPGPU. Compyle makes this possible by providing three important parallel
algorithms, an elementwise operation (a parallel map), a parallel scan (also
known as a prefix sum), and a parallel reduction. The Cython backend provides
a native implementation whereas the OpenCL and CUDA backend simply wrap up the
implementation provided by PyOpenCL and PyCUDA. These three algorithms make it
possible to write a variety of non-trivial parallel algorithms for high
performance computing. Compyle also provides the ability to write custom
kernels with support for local/shared memory specifically for OpenCL and CUDA
backends. Compyle provides simple facilities to annotate arguments and types
and can optionally make use of Python 3's type annotation feature as well.
Compyle also features JIT compilation if desired.

Compyle is quite different from Numba. One major difference is that it does
not rely on LLVM at all and instead performs source-to-source transpilation.
Under the covers, compyle produces simple and readable C or Cython code which
looks similar to the user's original code. Compyle does not provide support
for any high level Python and only works with a highly restricted Python
syntax. While this is not very user-friendly, we find that in practice this is
vitally important as it ensures that the code users write will run efficiently
and seamlessly execute on both a CPU and a GPU with minimum or ideally no
modifications. Furthermore compyle provides the basic parallelization
algorithms that users can use to extract good performance from their hardware.

In addition, compyle allows users to generate code using mako templates in
order to maximize code reuse. Since compyle performs source transpilation, it
is also possible to use compyle as a code-generation engine and put together
code from pure Python to build fairly sophisticated computational engines.


The functionality that Compyle provides falls broadly in two categories,

* Common parallel algorithms that will work across backends. This includes,
  elementwise operations, reductions, and prefix-sums/scans.
* Specific support to run code on a particular backend. This is for code that
  will only work on one backend by definition. This is necessary in order to
  best use different hardware and also use differences in the particular
  backend implementations. For example, the notion of local (or shared) memory
  only has meaning on a GPGPU. In this category we provide support to compile
  and execute Cython code, and also create and execute a GPU kernel. This is
  not discussed in too much detail in this article.

In what follows we provide a high-level introduction to the basic parallel
algorithms in the context of the prototypical molecular dynamics problem. By
the end of the article we show how easy it is to write the code with Compyle
and have it execute on multi-core CPUs and GPGPUs. We provide a convenient
notebook on google colab where users can run the simple examples on a GPU as
well.

Installation
-------------

Discuss these in brief.

- pip
- github
- Dependencies and requirements.


Parallel algorithms
--------------------

Elementwise
~~~~~~~~~~~

The elementwise operator operates on each element of an input array and maps it to
an output array. Below is a simple example of calculating :math:`y = a\sin{x} + b`.

.. code-block:: python

    import numpy as np
    from compyle.api import annotate, Elementwise, \
        get_config

    @annotate
    def axpb(i, x, y, a, b):
        y[i] = a[i]*sin(x[i]) + b[i]

    # Setup the input data
    n = 1000000
    x = np.linspace(0, 1, n)
    y = np.zeros_like(x)
    a = np.random.random(n)
    b = np.random.random(n)

    # Use OpenMP
    get_config().use_openmp = True

    # Now run this in parallel with Cython.
    backend = 'cython'
    e = Elementwise(axpb, backend=backend)
    e(x, y, a, b)

This will call the axpb function in parallel using OpenMP. To call this
function on the GPU, the arrays need to be sent to the device. This can
be acheived by using the :code:`Array` wrapper as follows,

.. code-block:: python

    from compyle.api import wrap

    backend = 'opencl'
    x, y, a, b = wrap(x, y, a, b, backend=backend)

This wraps the arrays and sends the data to the device. :code:`x.pull()` gets
the data from device to host and :code:`x.push()` sends the data from
host to device.

Here is an example of using elementwise for implementing the step
function for solving laplace equation.

.. code-block:: python

    @annotate
    def laplace_step(i, u, res, err, nx, ny, dx2, dy2,
                     dnr_inv):
        xid = i % nx
        yid = i / nx

        if xid == 0 or xid == nx - 1 or yid == 0 or \
            yid == ny - 1:
            return

        res[i] = ((u[i - 1] + u[i + 1]) * dx2 +
                  (u[i - nx] + u[i + nx]) * dy2) * \
                  dnr_inv

        diff = res[i] - u[i]

        err[i] = diff * diff

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

Below is a more complex example of implementing a parallel "where".
This returns elements of an array where a given condition is satisfied.
The following example returns elements of the array that are smaller
than 50.

.. code-block:: python

    ary = np.random.randint(0, 100, 1000, dtype=np.int32)
    result = np.zeros(len(ary.data), dtype=np.int32)
    result = wrap(result, backend=backend)
    result_count = np.zeros(1, dtype=np.int32)
    result_count = wrap(result_count, backend=backend)
    ary = wrap(ary, backend=backend)

    @annotate
    def input_expr(i, ary):
        return 1 if ary[i] < 50 else 0

    @annotate
    def output_expr(i, prev_item, item, N, ary, result,
                    result_count):
        if item != prev_item:
            result[item - 1] = ary[i]
        if i == N - 1:
            result_count[0] = item

    scan = Scan(input_expr, output_expr, 'a+b',
                dtype=np.int32, backend=backend)
    scan(ary=ary, result=result,
         result_count=result_count)
    result.pull()
    result_count.pull()
    result_count = result_count.data[0]
    result = result.data[:result_count]

    # Result = result

The :code:`input_expr` could also be used as the map function
for reduction and the required size of result could be found
before running the scan and the result array can be allocated
accordingly.

Example using local memory
--------------------------

Simple nearest neighbors
------------------------


Simple n-body treecode
-----------------------

Performance comparison with numba
---------------------------------

Limitations
------------


Future work
-------------

In the future, we would like to improve the package by adding support for
"objects" that would allow users to compose their libraries in a more object
oriented manner. This would also open up the possibility of implementing more
high-level data structures in an easy way.



Conclusions
-----------
