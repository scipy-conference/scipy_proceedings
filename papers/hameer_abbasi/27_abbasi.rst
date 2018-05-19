:author: Hameer Abbasi
:email: hameerabbasi@yahoo.com
:institution: TU Darmstadt
:corresponding:
:bibliography: references

:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------
Sparse: A more modern sparse array library
------------------------------------------

.. class:: abstract

   This paper is about sparse multi-dimensional arrays in Python. We discuss
   their applications, layouts, and current implementations in the SciPy
   ecosystem along with strengths and weaknesses. We then introduce a new
   package for sparse arrays that builds on the legacy of the scipy.sparse
   implementation, but supports more modern interfaces, dimensions greater
   than two, and improved integration with newer array packages, like XArray
   and Dask. We end with performance benchmarks and notes on future
   work.
   Additionally, this work provides a concrete implementation of the recent
   NumPy array protocols to build generic array interfaces for improved
   interoperability, and so may be useful for broader community discussion.

.. class:: keywords

   sparse, sparse arrays, sparse matrices, scipy.sparse, ndarray, ndarray interface

Introduction
------------

Sparse arrays are important in many situations and offer both speed and memory benefits
over regular arrays when solving a broad spectrum of problems. For example, they can be
used in solving systems of equations :cite:`liu1989limited`, solving partial differential
equations :cite:`mu1991organization`, machine learning problems involving Bayesian models
:cite:`tipping2001sparse` and natural language processing :cite:`nickel2011three`.

Traditionally, within the SciPy ecosystem, sparse arrays have been provided within SciPy
:cite:`scipy` in the submodule :code:`scipy.sparse`, which is arguably the most
feature-complete implementation of sparse matrices within the ecosystem, providing support
for basic arithmetic, linear algebra and graph theoretic algorithms.

However, it lacks certain features which prevent it from working nicely with other packages
in the ecosystem which consume NumPy's :cite:`numpy` :code:`ndarray` interface:

* It doesn't follow the :code:`ndarray` interface (rather, it follows NumPy's deprecated
  :code:`matrix` interface)
* It is limited to two dimensions only (even one-dimensional structures aren't supported)

This is important for a number of other packages that are quite innovative, but cannot take
advantage of :code:`scipy.sparse` for these reasons, because they expect objects following
the :code:`ndarray` interface. These include packages like Dask :cite:`dask` (which is
useful for parallel computing, even across clusters, for both NumPy arrays and Pandas
dataframes) and XArray :cite:`xarray` (which extends Pandas dataframes to multiple
dimensions).

Both of these frameworks could benefit tremendously from sparse structures. In the case of
Dask, it could be used in combination with sparse structures to scale up computational tasks
that need sparse structures. In the case of XArray, datasets with large amounts of missing
data could be represented efficiently.

In this paper, we present Sparse :cite:`sparse`, a sparse array library that supports
arbitrary dimension sparse arrays and supports most common parts of the :code:`ndarray`
interface. It supports basic arithmetic, application of :code:`ufunc` s directly to sparse
arrays (including with broadcasting), most common reductions, indexing, concatenation, stacking,
transpose, reshape and a number of other features. The primary format in this library is based on
the coordinate format, which stores indices where the array is nonzero, and the corresponding data.

Since a full explanation of usage would be a repeat of the NumPy user manual and the package
documentation, we only provide brief demos of certain features of interest. Then, we move on
to some of the design decisions that went into making this package, some optimizations,
applications and possible future work.

Usage Overview
--------------

Sparse can be installed both via :code:`pip` and with :code:`conda` from conda-forge. [#]_ The main
sparse storage structure in the library at this time is based on the COO format. It stores the
coordinates of nonzero entries; as well as the data corresponding to those entries. All coordinates
are sorted in the order they would appear in a C-contiguous array, and there are no duplicates allowed.

.. [#] All features shown here are tested against the :code:`master` branch on GitHub at the time of
   writing.

Arrays can be constructed from SciPy sparse matrices or NumPy arrays as follows:

.. code-block:: pycon

   >>> x1 = sparse.COO.from_scipy_sparse(sps_matrix)
   >>> x2 = sparse.COO.from_numpy(np_array)

It can also be constructed from DOK arrays:

.. code-block:: pycon

   >>> d = sparse.DOK((3, 4, 5), dtype=np.int64)
   >>> d[1:3, 2:3] = -23
   >>> x3 = d.asformat('coo')

Two COO arrays can have arithmetic performed on them, just like regular NumPy arrays. Arithmetic supports
broadcasting.

.. code-block:: pycon

   >>> x_sum = x1 + x2
   >>> x_prod = x1 * x2

However, at the time of writing, the first line below will not work, and the user must explicitly densify,
i.e. convert to a regular NumPy array:

.. code-block:: pycon

   >>> # Can't divide because it will densify
   >>> x_div = x1 / x2
   >>> # The user can explicitly convert to dense
   >>> x_div = x1.todense() / x2.todense()

Applying a :code:`ufunc` will also work, provided the :code:`ufunc` preserves zeros.

.. code-block:: pycon

   >>> x_sin = np.sin(x)

Basic reductions like :code:`sum`, :code:`prod`, :code:`min` and :code:`max` work.

.. code-block:: pycon

   >>> x_sum = np.sum(x)
   >>> x_min = x.min(axis=0)

Indexing a sparse COO array also works for most common cases.

.. code-block:: pycon

   >>> x_subarray = x[233]
   >>> x_slice = x[34:56]

Some features don't directly work via NumPy functions, but work through library-provided alternatives.
Among these are the following examples:

.. code-block:: pycon

   >>> # np.where doesn't work
   >>> x_where = sparse.where(condition, x, y)
   >>> # np.transpose doesn't work
   >>> x_tr = x.transpose((2, 0, 1))
   >>> # np.dot doesn't work
   >>> x_dot = sparse.dot(x1, x2)

For details on usage and the latest supported features, see the sparse package documentation.
:cite:`sparse`.

Design Considerations
---------------------

Storage Format
^^^^^^^^^^^^^^

Although not the most efficient in any respect, for simplicity of operations, we chose the COO
format for its storage simplicity. In this format, two dense arrays are required to store the
sparse array's data. The first is a coordinates array, which stores the coordinates where the
array is nonzero. This array has a shape :code:`(ndim, nnz)`. The second is a data array, which
stores the data corresponding to each coordinate, and thus it has the shape :code:`(nnz,)`. Here,
:code:`ndim` represents the number of dimensions of the array and :code:`nnz` represents the number
of nonzero entries in the array.

For simplicity of operations in many cases, the coordinates are always stored in C-contiguous order.
Table :ref:`tab:coo-vis` shows a visual representation of COO.

To save on memory, we always choose the smallest possible data type for the coordinates array.

.. table:: A visual representation of the COO format. :label:`tab:coo-vis`

   ==== ==== ==== === ====
   dim1 dim2 dim3 ... data
   ==== ==== ==== === ====
      0    0    0 ...   10
      0    0    3 ...   13
      0    2    2 ...    9
    ...  ...  ... ...  ...
     3    1     4 ...   21
   ==== ==== ==== === ====

Element-wise Operations
^^^^^^^^^^^^^^^^^^^^^^^

There was a challenge around designing a function that could perform any kind of element-wise
operation on any number of variables. The design we decided to adopt at the end was slow compared
to :code:`scipy.sparse`, but rather fast compared to dense arrays.

This was chosen in order to avoid writing different code for different operations. It also allows
element-wise operations like :code:`ufunc` s, :code:`astype` and the three-argument version of
:code:`where` to be handled under a single umbrella. We used the :code:`__array_ufunc__` protocol
to allow application of :code:`ufunc` s to COO arrays. Here is some simplified psuedocode for the
algorithm that we use::

   all_coords = []
   all_data = []

   for each combination of inputs where some are zero
       and some nonzero:
       if all inputs are zero:
           continue

       coords = find coordinates common to
                nonzero inputs
       coords = filter out coordinates that are
                in zero inputs
       data = apply function to data corresponding
              to these coordinates

       all_coords.append(coords)
       all_data.append(data)

   concatenate all_coords and all_data

This gets a lot more complex when dealing with broadcasting. Below, we show some simplified
psuedocode in the broadcasting case. New psuedocode is in parentheses::

   all_coords = []
   all_data = []

   for each combination of inputs where some are zero
       and some nonzero:
       if all inputs are zero:
           continue

       coords = find coordinates common to
                nonzero inputs
                (for dimensions that are not being
                broadcast in both, with repetition
                similar to an SQL outer join)
       coords = filter out coordinates that are
                in zero inputs
                (for non-broadcast dimensions)
       data = apply function to data corresponding
              to these coordinates

       filter out zeros in data and corresponding
           coordinates

       broadcast coordinates and data to output shape

       all_coords.append(coords)
       all_data.append(data)

   concatenate all_coords and all_data



Reductions
^^^^^^^^^^

Only some reductions are possible with this library at the moment, but most common ones are supported.
Supported reductions must have a few properties:

* They must be implemented in the form of :code:`ufunc.reduce`
* The :code:`ufunc` must be reorderable
* Reducing by multiple zeros shouldn't change the result
* An all-zero reduction must produce a zero.

Although these restrictions may seem crippling, in practice most reductions such as :code:`sum`,
:code:`prod`, :code:`min`, :code:`max`, :code:`any` and :code:`all` actually fall within the class
of supported reductions. We used :code:`__array_ufunc__` protocol to allow application of :code:`ufunc`
reductions to COO arrays.

Notable exceptions are :code:`argmin` and :code:`argmax`. The following is some simplified psuedocode that
we use for reductions::

   x = x.transpose((selected_axes, non_selected_axes))
   x = x.reshape((selected_axes_size,
                  non_selected_axes_size))

   y, counts = perform a reduce on x
               grouped by the first coordinate
               using ufunc.reduceat
   where counts < non_selected_axes_size, reduce
       an extra time by zero

   y = y.reshape(non_selected_axes_shape)

Indexing
^^^^^^^^

For indexing, we realize that to construct the new coordinates and data, we can perform two kinds of
filtering as to which coordinates will be in the new array and which ones won't.

* We can work directly with the coordinates and filter out unwanted coordinates and data. This turns
  out to be :math:`O(\text{ndim} \times \text{nnz})`.
* We can realize that for a fixed value of :code:`coords[:n]`, where :code:`n` is some non-negative
  integer, the sorting order implies that the sub-coords :code:`coords[n:]` will also be sorted.
  Getting a single item in this case is :math:`O(\text{ndim} \times \log \text{nnz})`, as we can use
  a binary search.

We realized that we can get successively smaller slices of the original COO array and append them to
the required coordinates for indexing, using the second method listed above. However, this presents
issues when calling code like :code:`x[:500, :500, :500]` as we will have to do a large amount of
binary searches.

So we used a hybrid approach where the second method is used until there are a sufficiently low
number of coordinates, then we fall back to simple filtering.

After getting the required coordinates and corresponding data, we apply some simple transformations
to it to get the output coordinates and data.

Transposing and Reshaping
^^^^^^^^^^^^^^^^^^^^^^^^^

Transposing corresponds to a simple reordering of the dimensions in the coordinates, along with a re-sorting
of the coordinates and data to make the coordinates C-contiguous again.

Reshaping corresponds to linearizing the coordinates and then doing the reverse for the new shape, similar to
:code:`np.ravel_multi_index` and :code:`np.unravel_index`. However, we write our own custom implementation for
this, in order to save on memory.

:code:`dot` and :code:`tensordot`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :code:`tensordot`, we currently just use the NumPy implementation, replacing :code:`np.dot` with
:code:`scipy.sparse.csr_matrix.dot`.

For :code:`dot`, we simply dispatch to :code:`tensordot`, providing the appropriate axes.

Optimizations
-------------

In our element-wise algorithm, we perform quite a few optimizations. The first is for operations like
multiplication, which is (barring some edge cases) zero if either of the operands are zero. We detect
this when calculating the output data, and neglect to append this data at all.

Benchmarks
----------


