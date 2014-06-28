:author: Mark Wiebe
:email: mwiebe@continuum.io
:institution: Continuum Analytics

:author: Matthew Rocklin
:email: mrocklin@continuum.io
:institution: Continuum Analytics

:author: TJ Alumbaugh
:email: tj.alumbaugh@continuum.io
:institution: Continuum Analytics

:author: Andy Terrel
:email: aterrel@continuum.io
:institution: Continuum Analytics

-------------------------------------------------------------------
Blaze: Building A Foundation for Array-Oriented Computing in Python
-------------------------------------------------------------------

.. class:: abstract

Python’s scientific computing and data analysis ecosystem, built around NumPy, SciPy, Matplotlib, Pandas, and a host of other libraries, is a tremendous success. NumPy provides an array object, the array-oriented ufunc primitive, and standard practices for exposing and writing numerical libraries to Python all of which have assisted in making it a solid foundation for the community. Over time, however, it has become clear that there are some limitations of NumPy that are difficult to address via evolution from within. Notably, the way NumPy arrays are restricted to data with regularly strided memory structure on a single machine is not easy to change.

Blaze is a project being built with the goal of addressing these limitations, and becoming a foundation to grow Python’s success in array-oriented computing long into the future. It consists of a small collection of libraries being built to generalize NumPy’s notions of array, dtype, and ufuncs to be more extensible, and to represent data and computation that is distributed or does not fit in main memory.

Blaze provides abstractions to connect users familiar with NumPy and Pandas to
other data analytics libraries both within and without the standard numeric
Python ecosystem.  Blaze specifically targets backends that support streaming
out-of-core, or distributed storage and computation.

We give an overview of the Blaze architecture and then demonstrate its use on a
typical problem.  We use the abstract nature of Blaze to quickly benchmark and
compare the performance of a variety of backends on a standard problem.

.. class:: keywords

   array programming, big data, numpy, scipy, pandas

Introduction
------------

Standard Interfaces
~~~~~~~~~~~~~~~~~~~

The data analytics ecosystem grows rapidly.  Today we see a growth both in
computational systems such as Postgres, Pandas and Hadoop and also an
increasing breadth of users ranging from physical scientists with a strong
tradition of computation to social scientists and policy makers with less
rigorous training.  While these upward trends are encouraging, they also place
significant strain on the programming ecosystem.  Keeping novice users adapted
to quickly changing programming paradigms and operational systems is
challenging.

Standard interfaces facilitate interactions between layers of complex and
changing systems.  For example NumPy fancy indexing syntax has become a
standard among array programming systems within the Python ecosystem.  Projects
with very different backends (e.g. NumPy, SciPy.sparse, Theano, SciDB) all
provide the same indexing interface despite operating very differently.
This uniformity facilitates smoother adoption by existing user communities.

Standard interfaces help users to adapt to new technologies.  Standard
interfaces allow library developers to evolve rapidly without the drag of a
hostage user community.

Interactive Arrays and Tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Projects like NumPy and Pandas have demonstrated the value of interactive array
and table objects.  These projects connect a broad base of users to efficient
low-level code through a high-level interface.  This approach has given rise to
large and productive software ecosystems within numeric Python (e.g. SciPy,
Scikits, ....)  However both NumPy and Pandas are largely restricted to an in-memory
computational model, limiting problem sizes to a certain scale.  They also both
use somewhat custom in memory storage, requiring explicit conversions when
dealing data from foreign systems.

Concurrently developed foreign data analytic ecosystems like R and Julia
provide similar styles of functionality with different application foci.
The Hadoop File System (HDFS) has accrued a menagerie of powerful distributed
computing systems like Hadoop, Spark, and Impala.  The broader scientific
computing community has produced projects like Elemental and SciDB for
distributed array computing in various contexts.  Finally traditional SQL
databases like MySQL and Postgres remain both popular and very powerful.

As problem sizes increase (growth of big data) and applications become more
interdisciplinary (e.g. the increased use of machine learning), analysts
increasingly require interaction with projects outside of the NumPy/Pandas
ecosystem.  Unfortunately these foreign projects rarely feel as comfortable or
as usable as the Pandas DataFrame.

What is Blaze
~~~~~~~~~~~~~

Blaze provides a familiar DataFrame interface around computation on other
systems.  It provides extensible mechanisms to connect this interface to
diverse computational backends.  The Blaze project explicitly provides hooks to
Streaming Python, Pandas, SQLAlchemy, and Spark.

This abstract connection to a variety of projects has the following virtues:

*   Novice users gain access to relatively exotic technologies
*   Users can trivially shift computational backends within a single workflow
*   Projects can trivially shift backends as technologies change
*   New technologies are provided with a stable interface and a trained set of
    users

Blaze doesn't do any computation itself.  Instead it depends heavily on
existing projects to perform that computation.  Blaze orchestrates other
projects to perform Table-like computations.  We intend to extend this to array
and more general models in the future.


Blaze Architecture
------------------

Blaze separates data analytics into three isolated components:

* Data access: efficient access across different storage systems,

  e.g. ``CSV``, ``HDF5``, ``HDFS``, ....

* Symbolic Expression: symbolic reasoning about the desired result,

  e.g. ``Join``, ``Sum``, ``Split-Apply-Combine``, ....

* Backend Computation: how to perform computations on a variety of backends,

  e.g. ``SQL``, ``Pandas``, ``Spark``, ....

We isolate these elements to enble experts to create well crafted solutions in
each domain without needing to understand the others, e.g. a Pandas expert can
contribute without knowing Spark and vice versa.  We provide abstraction layers
between these components to enable them to work together cleanly.

This process results in a multi-format, multi-backend computational engine
capable of common data analytics operations.


Blaze Data
~~~~~~~~~~

Blaze Data Descriptors provide uniform access to a variety of common data
formats.  They provide standard iteration, insertion, and numpy-like fancy
indexing over on-disk files in common formats like csv, json, and hdf5 in
memory data strutures like core Python data structures and DyND arrays as well
as more sophisticated data stores like SQL databases.  The data descriptor
interface is analogous to the Python buffer interface described in PEP 3118,
but with some more flexibility.

Over the course of this document we'll refer to the following simple
``accounts.csv`` file:

::

   id, name, balance
   1, Alice, 100
   2, Bob, -200
   3, Charlie, 300
   4, Denis, 400
   5, Edith, -500

.. code-block:: python

   >>> csv = CSV('accounts.csv')

Iteration
`````````

Data descriptors expose the ``__iter__`` method, which iterates over the
outermost dimension of the data.  This iterator yields vanilla Python objects
by default.

.. code-block:: python

   >>> list(csv)
   [(1L, u'Alice', 100L),
    (2L, u'Bob', 200L),
    (3L, u'Charlie', 300L),
    (4L, u'Denis', 400L),
    (5L, u'Edith', 500L)]


Data descriptors also expose a ``chunks`` method, which also iterates over the
outermost dimension but instead of yielding single rows of Python objects
instead yields larger chunks of compactly stored data.  These chunks emerge as
DyND arrays which are more efficient for bulk processing and data transfer.
DyND arrays support the ``__array__`` interface and so can be easily converted
to NumPy arrays.

.. code-block:: python

   >>> next(csv.chunks())
   nd.array([[1, "Alice", 100],
             [2, "Bob", 200],
             [3, "Charlie", 300],
             [4, "Denis", 400],
             [5, "Edith", 500]],
    type="5 * {id : int64, name : string, balance : int64}")

Insertion
`````````

Analagously to ``__iter__`` and ``chunks`` the methods ``extend`` and
``extend_chunks`` allow for insertion of data into the data descriptor.  These
methods take iterators of Python objects and DyND arrays respectively.  The
data is coerced into whatever form is native for the storage medium e.g. text
for CSV or ``INSERT`` statements for SQL.


.. code-block:: python

   >>> csv = CSV('accounts.csv', mode='a')
   >>> csv.extend([(6, 'Frank', 600),
   ...             (7, 'Georgina', 700)])


Migration
`````````

The combination of uniform iteration and insertion enables trivial data
migration between storage systems.

.. code-block:: python

   >>> sql = SQL('postgresql://user:pass@host/',
                 'accounts', schema=csv.schema)
   >>> sql.extend(iter(csv))  # Migrate csv file to DB


Indexing
````````

Data descriptors also support fancy indexing.  As with iteration this supports
either Python objects or DyND arrays with the ``.py[...]`` and ``.dynd[...]``
interfaces.

.. code-block:: python

   >>> list(csv.py[::2, ['name', 'balance']])
   [(u'Alice', 100L),
    (u'Charlie', 300L),
    (u'Edith', 500L),
    (u'Georgina', 700L)]

   >>> csv.dynd[::2, ['name', 'balance']]
   nd.array([["Alice", 100],
             ["Charlie", 300],
             ["Edith", 500],
             ["Georgina", 700]],
        type="var * {name : string, balance : int64}")

Performance of this approach varies depending on the underlying storage system.
For file-based storage systems like CSV and JSON we must seek through the file
to find the right line (see [iopro]_), but don't incur deserialization costs.
Some storage systems, like HDF5, support random access natively.


Cohesion
````````

Different storage techniques manage data differently.  Cohesion between these
disparate systems is accomplished with the two projects ``datashape``, which
specifies the intended meaning of the data, and DyND, which manages efficient
type coercions and serves as an efficient intermediate representation.


Blaze Expr
~~~~~~~~~~

To be able to run analytics on a wide variety of computational
back ends, it's important to have a way to represent them independent of any
particular back end. Blaze uses abstract expression trees for this,
including convenient syntax for creating them and a pluggable multiple
dispatch mechanism for lowering them to a computation back end. Once an
analytics computation is represented in this form, there is an opportunity
to do analysis and transformation on it prior to handing it off to a back end,
both for optimization purposes and to give heuristic feedback to the user
about the expected performance.

To illustrate how blaze expression trees work, we will build up an expression
on a table from the bottom , showing the structure of the trees along the way.
Let's start with a single table, for which we'll create an expression node

.. code-block:: python

    >>> accts = TableSymbol('accounts',
    ...       '{id: int, name: string, balance: int}')

to represent a abstract table of accounts. By defining operations on expression
nodes which construct new abstract expression trees, we can provide a familiar
interface closely matching that of NumPy and of Pandas. For example, in
structured arrays and dataframes you can access fields as ``accts['name']``.

Extracting fields from the table gives us ``Column`` objects, to which we can
now apply operations. For example, we can select all accounts with a negative
balance.

.. code-block:: python

    >>> deadbeats = accts[accts['balance'] < 0]['name']

or apply the split-apply-combine pattern to get the highest grade in
each class

.. code-block:: python

    >>> By(accts, accts['name'], accts['balance'].sum())

In each of these cases we get an abstract expression tree representing
the analytics operation we have performed, in a form independent of any
particular back end.

::

                   -----By-----------
                 /       |            \
              accts   Column         Sum
                      /     \           |
                  accts    'name'     Column
                                     /      \
                                accts    'balance'

Blaze Compute
~~~~~~~~~~~~~

Once an analytics expression is represented as a Blaze expression tree,
it needs to be mapped onto a back end. This is done by walking the tree
using the multiple dispatch ``compute`` function, which defines for how
an abstract Blaze operation maps to an operation in the target back end.

To see how this works, let's consider how to map the ``By`` node from the
previous section into a Pandas back end. The code which handles this is
an overload of ``compute`` which takes a ``By`` node and a
``DataFrame`` object. First, each of the child nodes must be computed,
so ``compute`` gets called on the three child nodes. This validates the
provided dataframe against the ``accts`` schema, and extracts the
'name' and 'balance' columns from it. Then, the pandas ``groupby``
call is used to group the 'balance' column according to the 'name'
column, and apply the ``sum`` operation.

Each back end can map the common analytics patterns supported by Blaze
to its way of dealing with it, either by computing it on the fly as the
Pandas back end does, or by building up an expression in the target system
such as an SQL statement or an RDD map and groupByKey in Spark.

Multiple dispatch provides a pluggable mechanism to connect new back
ends, and handle interactions between different back ends.

Example
~~~~~~~

We demonstrate the pieces of Blaze in a small toy example.

Recall our accounts dataset

.. code-block:: python

   >>> L = [(1, 'Alice', 100),
            (2, 'Bob', -200),
            (3, 'Charlie', 300),
            (4, 'Denis', 400),
            (5, 'Edith', -500)]

And our computation for names of account holders with negative balances

.. code-block:: python

   >>> deadbeats = accts[accts['balance'] < 0]['name']

We compose the abstract expression, ``deadbeats`` with the data ``L`` using the
function ``compute``.

.. code-block:: python

   >>> list(compute(deadbeats, L))
   ['Bob', 'Edith']

We observe that the correct answer was returned as a list.

If we now store our same data ``L`` into a Pandas DataFrame and then run the
exact same ``deadbeats`` computation against it we find the same semantic
answer.

.. code-block:: python

   >>> df = DataFrame(L, columns=['id', 'name', 'balance'])
   1      Bob
   4    Edith
   Name: name, dtype: object

Similarly against Spark

.. code-block:: python

   >>> sc = pyspark.SparkContext('local', 'Spark-app')
   >>> rdd = sc.parallelize(L) # Distributed DataStructure

   >>> compute(deadbeats, rdd)
   PythonRDD[1] at RDD at PythonRDD.scala:37

   >>> _.collect()
   ['Bob', 'Edith']

In each case of running ``compute(deadbeats, ...)`` against a different data source a Blaze orchestrated the right computational backend to execute the desired query.  The result was given in the form recieved and computation was done either with streaming Python, in memory Pandas, or distributed memory Spark.  The user experience was much the same.


Blaze Interface
~~~~~~~~~~~~~~~

The separation of expressions and backend computation provides a powerful
multi-backend experience.  Unfortunately this separation may also be confusing
for a novice programmer.  To this end we provide an interactive object that
feels much like a Pandas DataFrame, but in fact can be driving any of our
backends.

.. code-block:: python

   >>> sql = SQL('postgresql://postgres@localhost',
   ...           'accounts')
   >>> t = Table(sql)
   >>> t
      id     name  balance
   0   1    Alice      100
   1   2      Bob     -200
   2   3  Charlie      300
   3   4    Denis      400
   4   5    Edith     -500

   >>> t[t['balance'] < 0]['name']
       name
   0    Bob
   1  Edith

The astute reader will note the use of Pandas like user experience and output.
Note however that these outputs are the result of computations on a Postgres
database.


Experiment
----------

To demonstrate the capabilities and motivation for Blaze we execute a simple
split-apply-combine computation against a few backends.  We do this for a range
of problem sizes and so compare scalability across backends across scales.


Bitcoin
~~~~~~~

We consider financial transactions using the Bitcoin digital currency.  In
particular we consider transactions between de-anonymized identities as
computed by the process laid out in [Reid]_.  Each transaction consists of a
transaction ID, sender, recipient, timestamp, and a number of bitcoins sent.
Some example data

::

   # Transaction, Sender, Recipient, Timestamp, Value
   4,39337,39337,20120617120202,0.31081764
   4,39337,3,20120617120202,69.1
   5,2071196,2070358,20130304143805,61.60235182
   5,2071196,5,20130304143805,100.0

Expression
~~~~~~~~~~

We load in this data using `blaze.data`

.. code-block:: python

   >>> from blaze.data.csv import CSV
   >>> csv = CSV('user_edges.txt',
   ...           columns=['transaction', 'sender',
   ...              'recipient', 'timestamp', 'value'],
   ...           typehints={'timestamp': 'datetime'})

We then build an abstract table with this same schema

.. code-block:: python

   >>> t = TableSymbol('t', csv.schema)

And describe a simple computation, finding the ten senders that have sent the most bitcoins

.. code-block:: python

   >>> big_spenders = (By(t,
                          t['sender'],
                          t['value'].sum())
   ...                  .sort('value', ascending=False)
   ...                  .head(10))


Benchmark
~~~~~~~~~

We run this computation using streaming Python, Pandas, SQLite, Postgres, and Spark.  First we migrate the data to a variety of different data stores

.. code-block:: python

   >>> sqlite = into(SQL('sqlite:///btc.db', 'user_edges',
                         schema=csv.schema), csv)
   >>> postgres = into(SQL('postgresql:///user:pass',
                           'user_edges', schema=csv.schema),
                       csv)
   >>> hdf5 = into(HDF5('btc.hdf5', 'user_edges',
                        schema=csv.schema), csv)
   >>> df = into(DataFrame, csv)
   >>> rdd = into(SparkContext, csv)
   >>> py = into([], csv)

We then run our computation for a variety of sizes on the variety of backends

.. code-block:: python

   >>> from numpy import logspace
   >>> sizes = list(map(int, logspace(1, 8, 16)))

   >>> times = [[measure(lambda: compute(big_spenders.subs({t: t.head(size)}),
   ...                                   dataset))
   ...              for size in sizes]
   ...              for dataset in [py, df, rdd, sqlite, postgres, hdf5]]

TODO: Plot results

We see roughly what we expect, that Pandas performs about an order of magnitude
better than the others while in memory, but fails outside.  We get a good
comparison of technologies like SQLite, Postgres, and Streaming Python.  We see
that these technologies are able to span outside of single machine main memory.

For variety we benchmark a slightly different computation.

.. code-block:: python

   >>> popular_senders = (By(t, t['sender'], t['recipient'].nunique())
   ...                     .sort('value', ascending=False)
   ...                     .head(10))

TODO: Plot results

Here we see surprising results.  Pandas does not perform as well as expected
(though more performant alternatives to ``Series.nunique`` exist) and so we may
wish to choose one of the other backends as we scale out.

A quick survey of StackOverflow shows that ``df.nunique()`` is
significantly slower than ``len(df.unique())``.  We alter the
implementation for the ``nunique`` operation on a ``DataFrame``.

.. code-block:: python

   @dispatch(nunique, DataFrame)
   def compute(expr, df):
       parent = compute(expr.parent, df)  # Recurse up the tree
       # return parent.nunique()
       return len(parent.unique())

With this quick change we can redefine how Blaze interprets abstract
``nunique`` operations on Pandas DataFrames.  This change (if committed) can
accelerate all future Blaze/Pandas computations.


Discussion
~~~~~~~~~~

Blaze provides both rapid ability to migrate data between data formats and the
ability to rapidly prototype common computations against a wide variety of
backends.  It allows us to easily compare our options and choose the best for
our particular setting.  As that setting changes (i.e. if our data grows
considerably) our implementation can transition easily.

References
----------

.. [iopro] http://docs.continuum.io/iopro/index.html
.. [Reid] Reid, Fergal, and Martin Harrigan. "An analysis of anonymity in the
          bitcoin system." Security and Privacy in Social Networks. Springer New York,
          2013. 197-223.
