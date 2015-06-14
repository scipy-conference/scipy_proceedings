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

.. image:: dask-simple.png
   :alt: A simple dask dictionary

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

.. image:: array.png
   :alt: A dask array

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

Example: Matrix Multiply
~~~~~~~~~~~~~~~~~~~~~~~~

Example: Meteorology
~~~~~~~~~~~~~~~~~~~~


Dynamic Task Scheduling
-----------------------

We now execute task graphs.  How
we execute these graphs strongly impacts performance.  Fortunately we can
tackle this problem with a variety of approaches without touching the graph
creation problem discussed above.  Graph creation and graph execution are
separable problems.  The dask library contains schedulers for single-threaded,
multi-threaded, multi-process, and distributed execution.

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

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


