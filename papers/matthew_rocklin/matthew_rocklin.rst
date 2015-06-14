:author: Matthew Rocklin
:email: mrocklin@gmail.com
:institution: Continuum Analytics

----------------------------------------------------------------------
Dask: Parallel Computation with Blocked algorithms and Task Scheduling
----------------------------------------------------------------------

.. class:: abstract

    Dask parallel variants of foundational Python data structures through blocked algorithms and dynamic task scheduling.

.. class:: keywords

   parallelism, numpy, scheduling

Introduction
------------



Background
----------

Task Scheduling
~~~~~~~~~~~~~~~

Parallel Arrays
~~~~~~~~~~~~~~~


Dask Graphs
-----------

Normally humans write programs and then compilers/interpreters interpret them
(e.g.  ``python``, ``javac``, ``clang``).  Sometimes humans disagree with how
these compilers/interpreters choose to interpret and execute their programs.
In these cases humans often bring the analysis, optimization, and execution of
code into the code itself.

Commonly a desire for parallel execution causes this shift of responsibility
from compiler to human developer.  In these cases we often represent the
structure of our program explicitly as data within the program itself.


Dask is a specification that encodes task schedules with minimal incidental
complexity using terms common to all Python projects, namely dicts, tuples,
and callables.  Ideally this minimum solution is easy to adopt and understand
by a broad community.

We define a dask graph as a Python dictionary mapping keys to tasks or values.
A key is any python hashable, a value is any Python object that is not a task,
and a task is a Python tuple with a callable first element.

Example
~~~~~~~

.. figure:: dask-simple.png
   :scale: 40%

   A simple dask dictionary

Consider the following simple program

.. code-block:: python

   def inc(i):
       return i + 1

   def add(a, b):
       return a + b

   x = 1
   y = inc(x)
   z = add(y, 10)

We encode this as a dictionary in the following way

.. code-block:: python

   d = {'x': 1,
        'y': (inc, 'x'),
        'z': (add, 'y', 10)}

While less pleasant than our original code this representation can be analyzed
and executed by other Python code, not just the CPython interpreter.  We don't
recommend that users write code in this way, but rather that it is an
appropriate target for automated systems.  Also, in non-toy examples the
execution times are likely much larger than for ``inc`` and ``add``, warranting
the extra complexity.

Specification
~~~~~~~~~~~~~

We represent a computation as a directed acyclic graph of tasks with data
dependencies.  Dask is a specification to encode such a graph using ordinary
Python data structures, namely dicts, tuples, functions, and arbitrary Python
values.

A **dask graph** is a dictionary mapping data-keys to values or tasks.

.. code-block:: python

   {'x': 1,
    'y': 2,
    'z': (add, 'x', 'y'),
    'w': (sum, ['x', 'y', 'z'])}

A **key** can be any hashable value that is not a task.

.. code-block:: python

   'x'
   ('x', 2, 3)

A **task** is a tuple with a callable first element.  Tasks represent atomic
units of work meant to be run by a single worker.

.. code-block:: python

   (add, 'x', 'y')

We represent a task as a tuple such that the *first element is a callable
function* (like ``add``), and the succeeding elements are *arguments* for that
function.

An **argument** may be one of the following:

1.  Any key present in the dask like ``'x'``
2.  Any other value like ``1``, to be interpreted literally
3.  Other tasks like ``(inc, 'x')``
4.  List of arguments, like ``[1, 'x', (inc, 'x')]``

So all of the following are valid tasks

.. code-block:: python

   (add, 1, 2)
   (add, 'x', 2)
   (add, (inc, 'x'), 2)
   (sum, [1, 2])
   (sum, ['x', (inc, 'x')])
   (np.dot, np.array([...]), np.array([...]))

To encode keyword arguments we recommend the use of ``functools.partial`` or
``toolz.curry``.



Dask Arrays
-----------

The ``dask.array`` submodule uses dask graphs to create a numpy clone that uses
all of your cores and operates on datasets that do not fit in memory.  It does
this by building up a dask graph of blocked array algorithms.

Blocked Array Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~

Blocked algorithms compute a large result like "take the sum of these trillion
numbers" with many small computations like "break up the trillion numbers into
one million chunks of size one million, sum each chunk, then sum all of the
intermediate sums."  Through tricks like this we can evaluate one large problem
by solving very many small problems.

Example: ``arange``
~~~~~~~~~~~~~~~~~~~

Dask array functions produce ``Array`` objects that hold on to dask graphs.
These dask graphs use several ``numpy`` functions to achieve the full result.
In the following example one call to ``da.arange`` creates a graph with three
calls to ``np.arange``

.. code-block:: python

   >>> import dask.array as da
   >>> x = da.arange(15, chunks=(5,))
   >>> x       # Array object metadata
   dask.array<arange-1, shape=(15,), chunks=((5, 5, 5)), dtype=int64>
   >>> x.dask  # Every dask array holds a dask graph
   {('x', 0): (np.arange, 0, 5),
    ('x', 1): (np.arange, 5, 10),
    ('x', 2): (np.arange, 10, 15)}

Further operations on ``x`` create more complex graphs

.. code-block:: python

   >>> z = (x + 100).sum()
   >>> z.dask
   {('x', 0): (np.arange, 0, 5),
    ('x', 1): (np.arange, 5, 10),
    ('x', 2): (np.arange, 10, 15),
    ('y', 0): (add, ('x', 0), 100),
    ('y', 1): (add, ('x', 1), 100),
    ('y', 2): (add, ('x', 2), 100),
    ('z', 0): (np.sum, ('y', 0)),
    ('z', 1): (np.sum, ('y', 1)),
    ('z', 2): (np.sum, ('y', 2)),
    ('z',): (sum, [('z', 0), ('z', 1), ('z', 2)])}

Dask.array also holds convenience functions to execute this graph, completing
the illusion of a numpy clone

.. code-block:: python

   >>> z.compute()
   1605


Array metadata
~~~~~~~~~~~~~~

In the example above ``x`` and ``z`` are both ``dask.array.Array`` objects.
These objects contain the following data

1.  A dask graph, ``.dask``
2.  Information about shape and chunk shape, called ``.chunks``
3.  A name identifying which keys in the graph correspond to the result,
    ``.name``
4.  A dtype

The second item here, ``chunks``, deserves further explanation.  A normal numpy
array knows its ``shape``, a dask array must know its shape and the shape of
all of the internal numpy blocks that make up the larger array.  These shapes
can be concisely described by a tuple of tuples of integers, where each
internal tuple corresponds to the lengths along a single dimension.

.. figure:: array.png
   :scale: 40%

   A dask array

In the example above we have a 20 by 24 array cut into uniform blocks of size 5
by 8.  The ``chunks`` attribute describing this array is the following:

.. code-block:: python

   chunks = ((5, 5, 5, 5), (8, 8, 8))

Where the four fives correspond to the heights of the blocks along the first
dimension and the three eights correspond to the widths of the blocks along the
second dimension.  This particular example has uniform sizes along each
dimension but this need not be the case.  Consider the chunks of the following example
operations

.. code-block:: python

   >>> x[::2].chunks
   ((3, 2, 3, 2), (8, 8, 8))

   >>> x[::2].T.chunks
   ((8, 8, 8), (3, 2, 3, 2))

Every ``dask.array`` operation, like ``add``, slicing, or ``transpose`` must
take the graph and all metadata, add new tasks into the graph and determine new
values for each piece of metadata.


Capabilities and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding subgraphs and managing metadata for most of numpy is difficult but
straightforward.  At present ``dask.array`` is around 5000 lines of code
(including about half comments and docstrings).  It encompasses most commonly
used operations including the following:

*  Arithmetic and scalar mathematics, ``+, *, exp, log, ...``
*  Reductions along axes, ``sum(), mean(), std(), sum(axis=0), ...``
*  Tensor contractions / dot products / matrix multiply, ``tensordot``
*  Axis reordering / transpose, ``transpose``
*  Slicing, ``x[:100, 500:100:-2]``
*  Fancy indexing along single axes with lists or numpy arrays, ``x[:, [10, 1, 5]]``
*  A variety of utility functions, ``bincount, where, ...``

However dask.array is unable to handle any operation whose shape can not be
determined ahead of time.  Consider for example the following common numpy
operation

.. code-block:: python

   x[x > 0]

The shape of this array depends on the number of positive elements in ``x``.
This shape is not known given only metadata; it requires knowledge of the
values underlying ``x``, which are not available at graph creation time.


Dynamic Task Scheduling
-----------------------

We now execute task graphs.  How we execute these graphs strongly impacts
performance.  Fortunately we can tackle this problem with a variety of
approaches without touching the graph creation problem discussed above.  Graph
creation and graph execution are separable problems.  The dask library contains
schedulers for single-threaded, multi-threaded, multi-process, and distributed
execution.

Current dask schedulers all operate *dynamically*, meaning that execution order
is determined during execution rather than ahead of time through static
analysis.  This is good when runtimes are not known ahead of time or when the
execution environment contains uncertainty.  However dynamic scheduling does
preclude certain clever optimizations.

The logic behind dask schedulers reduces to the following situation:  A worker
reports that it has completed a task and that it is ready for another.  We
update runtime state to record the finished task, mark which new tasks can be
run, which data can be released, etc..  We then choose a task to give to this
worker from among the set of ready-to-run tasks.  This small choice governs the
macro-scale performance of the scheduler.

Traditional task scheduling with data dependencies scheduling literature usually focues on policies to expose parallelism or chip away at the critical path.  We find that for bulk data analytics these are not very relevant as parallelism is abundant and critical paths are comparatively short relative to the depth of the graph.

Instead for out-of-core coputation we find value in choosing tasks that allow
us to release intermediate results and keep a small memory footprint.
This lets us avoid spilling intermediate values to disk which hampers
performance significantly.

After several other policies we find that the policy of *last in, first out* is
surprisingly effective.  We select tasks that were most recently made
available.  We implement this with a simple stack, which can operate in
constant time.  This is very important.

We endeavor to keep scheduling overhead low at around 1ms per task.  Updating
executino state and deciding which task to run must be made very quickly.  To
do this we maintain a great deal of state about the currently executing
computation.  The set of ready-to-run tasks is commonly quite large, in the
tens or hundreds of thousands in common workloads and so in practice we must
maintain enough state so that we can choose the right task in constant time (or
at least far sub-linear time).


Example: Matrix Multiply
~~~~~~~~~~~~~~~~~~~~~~~~

We benchmark dask's blocked matrix multiply on an out-of-core dataset.  This
demonstrates the following:

1.  How to interact with on-disk data
2.  The blocked algorithms in dask.array achieve similar performance to modern
    BLAS implementations on compute-bound tasks

We set up a trivial input dataset

.. code-block:: python

   import h5py
   f = h5py.File('myfile.hdf5')
   A = f.create_dataset(name='A', shape=(200000, 4000), dtype='f8',
                                  chunks=(250, 250), fillvalue=1.0)
   B = f.create_dataset(name='B', shape=(4000, 4000), dtype='f8',
                                  chunks=(250, 250), fillvalue=1.0)
   out = f.create_dataset(name='out', shape=(4000, 4000), dtype='f8',
                          chunks=(250, 250))

The Dask convenience method, ``da.from_array``, creates a graph that can pull
data from any object that implements numpy slicing syntax.  The ``da.store``
function can then store a large result in any object that implements numpy
setitem syntax.

.. code-block:: python

   import dask.array as da
   a = da.from_array(A, chunks=(1000, 1000))
   b = da.from_array(B, chunks=(1000, 1000))

   c = a.dot(b)  # another dask Array, not yet computed
   c.store(out)  # Store result into output space

**Results**: We do this same operation in different settings.

We use either use NumPy or dask.array

1.  Use NumPy on a big-memory machine
2.  Use dask.array in a small amount of memory, pulling data from disk, using
    four threads

We different BLAS implementations

1.  Reference BLAS, single threaded, unblocked
2.  OpenBLAS, single threaded
3.  OpenBLAS, multi-threaded

For each configuration we compute the number of floating point operations per
second.

.. table:: Matrix Multiply GFLOPS

   +-----------------------+--------+--------------+
   | Performance (GFLOPS)  | NumPy  |  Dask.array  |
   +=======================+========+==============+
   | Reference BLAS        | 6      |  18          |
   +-----------------------+--------+--------------+
   | OpenBLAS one thread   | 11     |  23          |
   +-----------------------+--------+--------------+
   | OpenBLAS four threads | 22     |  11          |
   +-----------------------+--------+--------------+

We note the following

1.  Compute-bound tasks are computationally bound by memory; we don't
    experience a slowdown
2.  Dask.array can effectively parallize and block reference BLAS for matrix
    multiplies
3.  Dask.array doesn't significantly improve when using an optimized BLAS,
    presumably this is because we've already reaped most of the benefits of
    blocking and multi-core
4.  One should not mix multiple forms of multi-threading.  Four dask.array
    threads each spawning multi-threaded OpenBLAS DGEMM calls results in worse
    performance.


Example: Meteorology
~~~~~~~~~~~~~~~~~~~~

Performance is secondary to capability.  In this example we use dask.array to
manipulate climate datasets that are larger than memory.  This example shows
the following:

1.  Use ``concatenate`` and ``stack`` to manage large piles of HDF5 files (a
    common case)
2.  Using reductions and slicing to manipulate stacks of arrays
3.  Interacting with other libraries in the ecosystem using the ``__array__``
    protocol.

We start with a typical setup, a large pile of NetCDF files.::

   $ ls
   2014-01-01.nc3  2014-03-18.nc3  2014-06-02.nc3  2014-08-17.nc3  2014-11-01.nc3
   2014-01-02.nc3  2014-03-19.nc3  2014-06-03.nc3  2014-08-18.nc3  2014-11-02.nc3
   2014-01-03.nc3  2014-03-20.nc3  2014-06-04.nc3  2014-08-19.nc3  2014-11-03.nc3
   2014-01-04.nc3  2014-03-21.nc3  2014-06-05.nc3  2014-08-20.nc3  2014-11-04.nc3
   ...             ...             ...             ...             ...

Each of these files contains the temperature at two meters above ground over
the earth at quarter degree resolution, every six hours.

.. code-block:: python

   >>> import netCDF4
   >>> t = netCDF4.Dataset('2014-01-01.nc3').variables['t2m']
   >>> t.shape
   (4, 721, 1440)

We can collect many of these files together using ``da.concatenate``, resulting
in a single large array.

.. code-block:: python

   >>> from glob import glob
   >>> filenames = sorted(glob('2014-*.nc3'))
   >>> temps = [netCDF4.Dataset(fn).variables['t2m'] for fn in filenames]

   >>> import dask.array as da
   >>> arrays = [da.from_array(t, blockshape=(4, 200, 200)) for t in temps]
   >>> x = da.concatenate(arrays, axis=0)

   >>> x.shape
   (1464, 721, 1440)

We can now play with this array as though it were a numpy array.  Because
dask.arrays implement the ``__array__`` protocol we can dump them directly into
functions of other libraries.  These libraries will trigger computation when
they call ``np.array(...)`` on their input.

.. code-block:: python

>>> from matplotlib import imshow
>>> imshow(x[::4].mean(axis=0) - x[2::4].mean(axis=0), cmap='RdBu_r')

.. figure:: day-vs-night.png

   We use typical numpy slicing and reductions on a large volume of data to
   show the average temperature difference between noon and midnight for year
   2014

This computation took about a minute on an old notebook computer.  The computation seemed to be bound by disk access.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.
