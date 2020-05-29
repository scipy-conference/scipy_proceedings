:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:


:author: Aditya Bhosale
:email: adityapb1546@gmail.com
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:bibliography: references


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
   problem. The code transparently executes on multi-core CPUs and GPGPUs. As
   far as we are aware, this is not possible with any of the other tools that
   are available currently in the Python ecosystem.


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
interact with each other via a Lennard-Jones potential as discussed in
:cite:`schroeder2015`.


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
:cite:`schroeder2015`. Our goal is to implement this system in pure Python
using Compyle. Through this we demonstrate the ease of use and power of
Compyle. We write a single code that executes efficiently in parallel on CPUs
and GPUs. We use this example to also discuss the three important parallel
algorithms and show how they nicely allow us to solve non-trivial problems.

Compyle is not a toy and is actively used by a non-trivial, open source, SPH
framework called PySPH (https://pysph.rtfd.io) and discussed in some detail in
:cite:`pysph2019` and :cite:`pysph16`. Compyle makes it possible for users to
write their SPH codes in high-level Python and have it executed on multi-core
and GPU accelerators with negligible changes to their code.



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
is helpful when using compyle.

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

Installation of Compyle is by itself straightforward and this can be done with
pip_ using::

  pip install compyle

For execution on a CPU, Compyle depends on Cython and a C++ compiler on the
local machine. Detailed instructions for installation are available at the
`compyle installation documentation
<https://compyle.readthedocs.io/en/latest/installation.html>`_. For execution
on a GPU compyle requires that either PyOpenCL_ or PyCUDA_ be installed. It is
possible to install the required dependencies using the extras argument as
follows::

  pip install compyle[opencl]

Compyle is still under heavy development and one can also easily install the
package using a git checkout from the repository on github at
https://github.com/pypr/compyle


.. _pip: https://pip.pypa.io/

Parallel algorithms
--------------------

We will work through a molecular dynamics simulation of N particles (in two
dimensions) using the Lennard Jones potential energy for interaction. Each
particle interacts with every other particle and together the system of
particles evolves in time. The Lennard-Jones potential energy is given by,

.. math::
    u(r) = 4\epsilon \left( \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right)

Each particle introduces an energy potential and if another particle is at a
distance of :math:`r` from it, then the potential experienced by the particle
is given by the above equation. The gradient of this potential energy function
produces the force on the particle. Therefore if we are given two particles at
positions, :math:`\vec{r}_i` and :math:`\vec{r}_j` respectively then the force
on the particle :math:`j` is dependent on the value of :math:`|\vec{r_j} -
\vec{r_i}|` and the gradient is:

.. math::
   \vec{F}_{i \leftarrow j} = \frac{24 \epsilon}{r_{ij}^2} \left( 2\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6 \right) \vec{r}_{ij}

Where :math:`r_{ij} = |\vec{r}_{ij}|` and :math:`\vec{r}_{ij} = \vec{r}_i -
\vec{r}_j`. The left hand side is the force on particle :math:`i` due to
particle at :math:`j`. Here, we use :math:`\sigma = \epsilon = m = 1` for our
implementation. We use the velocity Verlet algorithm in order to integrate the
system in time. We use a timestep of :math:`\Delta t` and as outlined in
:cite:`schroeder2015`, the position and velocity of the particles are updated
in the following sequence:

1. Positions of all particles are updated using the current velocities as
   :math:`x_i = x_i + v_i \Delta t + \frac{1}{2} a_i \Delta t`. The velocities
   are then updated by half a step as :math:`v_i = v_i + \frac{1}{2} a_i
   \Delta t`.

2. The new acceleration of all particles are calculated using the
   updated positions.

3. The velocities are then updated by another half a step.

In the simplest implementation of this, all particles influence all other
particles. This can be implemented very easily in Python. We first look at how
to implement this using Compyle. Our implementation will be parallel from the
get-go and will work on both CPUs and GPUs.

Once we complete the simple implementation we consider a very important
performance improvement where particles that are beyond 3 natural units, i.e.
:math:`r_{ij} > 3` do not influence each other (beyond this distance the force
is negligible). This can be used to reduce the complexity of the computation
of the mutual forces from an :math:`O(N^2)` to an :math:`O(N)` computation.
However, implementing this easily in parallel is not so straightforward.

Due to the simplicity of the initial implementation, all of these steps can be
implemented using what are called "elementwise" operations. This is the
simplest building block for parallel computing and is also known as the
"parallel map" operation.

Elementwise
~~~~~~~~~~~

An elementwise operation can be thought of as a parallel for loop. It can be
used to map every element of an input array to a corresponding output. Here is
a simple elementwise function implemented using compyle to execute step 1 of
the above algorithm.

.. code-block:: python

    @annotate(float='m, dt',
              gfloatp='x, y, vx, vy, fx, fy')
    def integrate_step1(i, m, dt, x, y, vx, vy, fx, fy):
        axi, ayi = declare('float', 2)
        axi = fx[i] / m
        ayi = fy[i] / m
        x[i] += vx[i] * dt + 0.5 * axi * dt * dt
        y[i] += vy[i] * dt + 0.5 * ayi * dt * dt
        vx[i] += 0.5 * axi * dt
        vy[i] += 0.5 * ayi * dt

The annotate decorator is used to specify types of arguments and
the declare function is used to specify types of variables
declared in the function. This can be avoided by using the JIT
compilation feature which infers the types of arguments and
variables based on the types of arguments passed to the function
at runtime. Following is the implementation of steps 2 and 3
without the type declarations.

.. code-block:: python

    @annotate
    def calculate_force(i, x, y, fx, fy, pe,
                        num_particles):
        force_cutoff = 3.
        force_cutoff2 = force_cutoff * force_cutoff
        for j in range(num_particles):
            if i == j:
                continue
            xij = x[i] - x[j]
            yij = y[i] - y[j]
            rij2 = xij * xij + yij * yij
            if rij2 > force_cutoff2:
                continue
            irij2 = 1.0 / rij2
            irij6 = irij2 * irij2 * irij2
            irij12 = irij6 * irij6
            pe[i] += (4 * (irij12 - irij6))
            f_base = 24 * irij2 * (2 * irij12 - irij6)

            fx[i] += f_base * xij
            fy[i] += f_base * yij

    @annotate
    def integrate_step2(i, m, dt, x, y, vx, vy, fx, fy):
        vx[i] += 0.5 * fx[i] * dt / m
        vy[i] += 0.5 * fy[i] * dt / m

Finally, these components can be brought together to write
the step functions for our simulation,

.. code-block:: python

    @annotate
    def step_method1(i, x, y, vx, vy, fx, fy, pe, xmin,
                     xmax, ymin, ymax, m, dt,
                     num_particles):
        integrate_step1(i, m, dt, x, y, vx, vy, fx, fy)


    @annotate
    def step_method2(i, x, y, vx, vy, fx, fy, pe, xmin,
                     xmax, ymin, ymax, m, dt,
                     num_particles):
        calculate_force(i, x, y, fx, fy, pe,
                        num_particles)
        integrate_step2(i, m, dt, x, y, vx, vy, fx, fy)

These can then be wrapped using the :code:`Elementwise`
class and called as normal python functions.

.. code-block:: python

        step1 = Elementwise(step_method1,
                            backend=self.backend)
        step2 = Elementwise(step_method2,
                            backend=self.backend)


- Elaborate a little bit about the annotation decorator. Mention that Python3
  type annotation also works.
- Mention the ``@elementwise`` decorator.


- Create the arrays at this point and complete the program showing how to
  run the code -- pseudo code should do.


Reduction
~~~~~~~~~

To check the accuracy of the simulation, the total energy of the
system can be monitored.
The total energy for each particle can be calculated as the sum of
its potential and kinetic energy. The total energy of the system
can then be calculated by summing the total energy over all
particles.

The reduction operator reduces an array to a single value. Given an input array
:math:`(a_0, a_1, a_2, \cdots, a_{n-1})` and an associative binary operator
:math:`\oplus`, the reduction operation returns the
value :math:`a_0 \oplus a_1 \oplus \cdots \oplus a_{n-1}`.

Compyle also allows users to give a map expression to map the
input before applying the reduction operator.
The total energy of our system can thus be found as follows using
reduction operator in compyle.

.. code-block:: python

    @annotate
    def calculate_energy(i, vx, vy, pe, num_particles):
        ke = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])
        return pe[i] + ke

    energy_calc = Reduction('a+b',
                            map_func=calculate_energy,
                            backend=backend)
    total_energy = energy_calc(vx, vy, pe, num_particles)



Initial Results
~~~~~~~~~~~~~~~~~

Show the results of the above work, along with simple comparisons on CPU and
GPU.


Scans
~~~~~

- Provide background and motivation of why we need scans in the context of the
  LJ problem.

- Show a simpler example first to illustrate the ideas.


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

Below is an example of implementing a parallel "where".
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

Performance comparison
----------------------

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
