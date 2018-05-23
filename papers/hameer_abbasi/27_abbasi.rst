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

In addition, :code:`scipy.sparse` is depended on by many downstream projects, which makes
removing NumPy's :code:`matrix` interface that much more difficult, and limits usage of
both :code:`ndarray` style duck arrays and :code:`scipy.sparse` arrays within the same
codebase.

This is important for a number of other packages that are quite innovative, but cannot take
advantage of :code:`scipy.sparse` for these reasons, because they expect objects following
the :code:`ndarray` interface. These include packages like Dask :cite:`dask` (which is
useful for parallel computing, even across clusters, for both NumPy arrays and Pandas
dataframes) and XArray :cite:`xarray` (which extends Pandas dataframes to multiple
dimensions).

Both of these frameworks could benefit tremendously from sparse structures. In the case of
Dask, it could be used in combination with sparse structures to scale up computational tasks
that need sparse structures. In the case of XArray, datasets with large amounts of missing
data could be represented efficiently, as well as other benefits such as broadcasting by
axis name rather than by rather opaque axis positions.

In this paper, we present Sparse :cite:`sparse`, a sparse array library that supports
arbitrary dimension sparse arrays and supports most common parts of the :code:`ndarray`
interface. It supports basic arithmetic, application of :code:`ufunc` s directly to sparse
arrays (including with broadcasting), most common reductions, indexing, concatenation, stacking,
transpose, reshape and a number of other features. The primary format in this library is based on
the coordinate format, which stores indices where the array is nonzero, and the corresponding data.

Since a full explanation of usage would be a repeat of the NumPy user manual and the package
documentation, we move on to some of the design decisions that went into making this package,
including some challenges we had to face some optimizations, applications and possible future work.

Algorithms and Challenges
-------------------------

Choice of storage format
........................

We chose the COO format for its simplicity while storing and accessing elements, even though it
isn't the most efficient storage format. In this format, two dense arrays are required to store the
sparse array's data. The first is a coordinates array, which stores the coordinates where the
array is nonzero. This array has a shape :code:`(ndim, nnz)`. The second is a data array, which
stores the data corresponding to each coordinate, and thus it has the shape :code:`(nnz,)`. Here,
:code:`ndim` represents the number of dimensions of the array and :code:`nnz` represents the number
of nonzero entries in the array.

For simplicity of operations in many cases, the coordinates are always stored in C-contiguous order.
Table :ref:`tab:coo-vis` shows a visual representation of how data is stored in the COO format.

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

To save on memory, we always choose the smallest possible data type for the coordinates array.

Element-wise operations
.......................

Element-wise operations are an important and common part of any array interface. For example,
arithmetic, casting an array, and all NumPy :code:`ufunc` s are common examples of element-wise
operations.

These turn out to be simple for NumPy arrays, but are surprisingly complex for sparse arrays.
The first problem to overcome was the lack of dependency on Numba :cite:`numba`/Cython
:cite:`cython`/C++. At the time, I wished to solve the problem in NumPy, therefore looping over
all possible nonzero coordinates was not an option, and we had to process the coordinates and
data in batches. The batches that made sense at the time were something like the following:

1. Coordinates in the first array but not in the second.
2. Coordinates in the second array but not in the first.
3. Coordinates in both arrays simultaneously.

This algorithm (when applied to multiple inputs instead of just two) looks like the following::

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

The addition of broadcasting makes this problem even more complex to solve, as it turns out
that for sparse arrays, simply broadcasting all arrays to a common shape and then performing
element-wise operations is not the most efficient way to perform such an operation.

Consider two arrays, one shaped :code:`(n,)` and another shaped :code:`(m, n)`, both with only
one nonzero entry. If all we wanted to do was multiply them, the result would have just one
nonzero entry, yet broadcasting the first array would result in an array with :code:`m` nonzero
entries (which clearly isn't the most optimal way to do things). For this reason, we chose to
handle broadcasting within the algorithm itself, instead of broadcasting all inputs upfront.

Effectively, this resulted in the following algorithm, which doesn't have the limitation mentioned
above. This is because any zeros are filtered out before any broadcasting is done::

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
       data = apply function to data corresponding
              to these coordinates

       coords, data = filter out zeros from coords/data

       coords, data = filter out coordinates/data that
                      are in zero inputs
                      (again, for non-broadcast dimensions)

       broadcast coordinates and data to output shape

       all_coords.append(coords)
       all_data.append(data)

   concatenate all_coords and all_data

While this algorithm is effective at applying all sorts of element-wise operations for
any amount of inputs, it does have a few drawbacks:

* It's slower than :code:`scipy.sparse`, because

  * It loops over all possible combinations of zero/nonzero
    coordinates, which makes it :math:`O \left( \left(2^\text{nin} - 1 \right) \times \text{nnz} \right)`
    in the worst case.
  * It's in COO format rather than CSR/CSC.
  * :code:`scipy.sparse` uses specialized code paths for each operation that greatly
    reduce the strain on the CPU whereas we keep everything generic.

* In the current implementation, sorting of coordinates is sometimes done unnecessarily.

This can be improved in the future in the following ways:

* Looping over possibly nonzero coordinates with something like Numba or Cython.

  * This approach will solve most of the speed issues.
  * Sorting will be rendered unnecessary.
  * Specialized code paths introduce a large maintenance burden, but can be implemented.

* Introducing multidimensional CSR/CSC.

You can see the current performance of the code in table :ref:`tab:bench`.

Reductions
..........

We implemented reductions by the simple concept of a "grouped reduce". The idea is to first group the
coordinates by the non-selected axes, and then reduce along the selected axes. This is simple to
implement in practice, and also works quite well. Here is some psuedocode that we use for reductions::

   x = x.transpose((selected_axes, non_selected_axes))
   x = x.reshape((selected_axes_size,
                  non_selected_axes_size))

   y, counts = perform a reduce on x
               grouped by the first coordinate
               using ufunc.reduceat
   where counts < non_selected_axes_size, reduce
       an extra time by zero

   y = y.reshape(non_selected_axes_shape)

Only some reductions are possible with this algorithm at the moment, but most common ones are supported.
Supported reductions must have a few properties:

* They must be implemented in the form of :code:`ufunc.reduce`
* The :code:`ufunc` must be reorderable
* Reducing by multiple zeros shouldn't change the result
* An all-zero reduction must produce a zero.

Although these restrictions may seem crippling, in practice most reductions such as :code:`sum`,
:code:`prod`, :code:`min`, :code:`max`, :code:`any` and :code:`all` actually fall within the class
of supported reductions. We used :code:`__array_ufunc__` protocol to allow application of :code:`ufunc`
reductions to COO arrays. Notable unsupported reductions are :code:`argmin` and :code:`argmax`.

This is nearly as fast as the reductions in :code:`scipy.sparse` when reducing along C-contiguous axes,
but is slow otherwise. Performance results can be seen in table :ref:`tab:bench`. Profiling reveals
that most of the time in the slow case is taken up by sorting, as :code:`ufunc.reduceat` expects all
"groups" to be right next to each other. This can be improved in the following ways:

* Implement a radix argsort, which will significantly speed up the sorting.
* Perform a "grouped reduce" by other methods, such as how Pandas does it, perhaps
  by using a :code:`dict` to maintain the results.

Indexing
........

For indexing, we realize that to construct the new coordinates and data, we can perform two kinds of
filtering as to which coordinates will be in the new array and which ones won't.

* We can work directly with the coordinates and filter out unwanted coordinates and data. This turns
  out to be :math:`O(\text{ndim} \times \text{nnz})` in total.
* We can realize that for a fixed value of :code:`coords[:n]`, where :code:`n` is some non-negative
  integer, the sorting order implies that the sub-coords :code:`coords[n:]` will also be sorted.
  Getting a single item or an integer slice in this case is
  :math:`O(\text{ndim} \times \log \text{nnz})`, as we can use a binary search.

We realized that we can get successively smaller slices of the original COO array and append them to
the required coordinates for indexing, using the second method listed above. However, this presents
issues when calling code like :code:`x[:500, :500, :500]` as we will have to do a large amount of
binary searches (:math:`500^3` in this case).

So we used a hybrid approach where the second method is used until there are a sufficiently low
number of coordinates left for filtering, then we fall back to simple filtering. Where we do the
switch is determined by a heuristic: will the expected number of binary searches be faster in a
specific case, or directly filtering the number of left-over coordinates? The overall algorithm
is implemented in Numba.

After getting the required coordinates and corresponding data, we apply some simple transformations
to it to get the output coordinates and data.

Transposing and Reshaping
.........................

Transposing corresponds to a simple reordering of the dimensions in the coordinates, along with a re-sorting
of the coordinates and data to make the coordinates C-contiguous again.

Reshaping corresponds to linearizing the coordinates and then doing the reverse for the new shape, similar to
:code:`np.ravel_multi_index` and :code:`np.unravel_index`. However, we write our own custom implementation for
this, in order to save on memory.

:code:`dot` and :code:`tensordot`
.................................

For :code:`tensordot`, we currently just use the NumPy implementation, replacing :code:`np.dot` with
:code:`scipy.sparse.csr_matrix.dot`. This is mainly just transposing and reshaping the matrix into
2-D, using :code:`np.dot` (or :code:`scipy.sparse.csr_matrix.dot` in our case), and performing the
reshape and transpose operations in reverse.

For :code:`sparse.dot`, we simply dispatch to :code:`tensordot`, providing the appropriate axes.

Benchmarks
----------

Because of our desire for clean and generic code as well as using mainly pure Python as opposed to
Cython/C/C++ in most places, our code is not as fast as :code:`scipy.sparse.csr_matrix`. It, however,
does beat :code:`numpy.ndarray`, provided the sparsity of the array is small enough. The benchmarks
were performed on a laptop with a Core i7-3537U processor and 16 GB of memory. Any arrays used had a
shape of :code:`(10000, 10000)` with a density of :code:`0.001`. The results are tabulated in table
:ref:`tab:bench`.

The NumPy results are given only for comparison, and for the purposes of illustrating that using sparse
arrays does, indeed, have benefits over using dense arrays when the density of the sparse array is
sufficiently low.

.. table:: Benchmarks :label:`tab:bench`

   +----------------+-------------------+-------------------+-------------------+
   | Benchmark      | Sparse            | SciPy Sparse      | NumPy             |
   +================+===================+===================+===================+
   | Addition       | 50.8 ms ± 3.45 ms | 2.49 ms ± 211 µs  | 507 ms ± 6.43 ms  |
   +----------------+-------------------+-------------------+-------------------+
   | Multiplication | 10.7 ms ± 526 µs  | 14.9 ms ± 1.68 ms | 529 ms ± 13.5 ms  |
   +----------------+-------------------+-------------------+-------------------+
   | Sum, Axis=0    | 12 ms ± 116 µs    | 545 µs ± 49.8 µs  | 97.8 ms ± 4.19 ms |
   +----------------+-------------------+-------------------+-------------------+
   | Sum, Axis=1    | 959 µs ± 23.7 µs  | 641 µs ± 83.9 µs  | 62.7 ms ± 4.86 ms |
   +----------------+-------------------+-------------------+-------------------+

Outlook and Future Work
-----------------------

We discussed the current leading solution for sparse arrays in the ecosystem, :code:`scipy.sparse`,
along with its shortcomings and limitations. We then introduced a new package for N-dimensional
sparse arrays, and how it has the potential to address these shortcomings. We discuss its current
implementation, including the algorithms used in some of the different operations and the limitations
and drawbacks of each algorithm. We also discuss future improvements that could be made to improve
these algorithms.

There are a number of areas we would like to focus on in the future. These include, in very broad terms:

* Better performance
* Better integration with community packages, such as scikit-learn, Dask and XArray
* Support for more of the :code:`ndarray` interface (particularly through protocols)
* Implementation of more linear algebra routines, such as :code:`eig`, :code:`svd`, and :code:`solve`
* Implementation of more sparse storage formats, such as a generalization of CSR/CSC
