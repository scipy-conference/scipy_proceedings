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
in the submodule :code:`scipy.sparse` :cite:`sps110`, which is arguably the most
feature-complete implementation of sparse matrices within the ecosystem, providing support
for basic arithmetic, linear algebra and graph theoretic algorithms.

However, it lacks certain features which prevent it from working nicely with other packages
in the ecosystem which consume NumPy's :code:`ndarray` interface:

* It doesn't follow the :code:`ndarray` interface (rather, it follows NumPy's deprecated
  :code:`matrix` interface)
* It is limited to two dimensions only (even one-dimensional structures aren't supported)

This is important for a number of other packages that are quite innovative, but cannot take
advantage of :code:`scipy.sparse` for these reasons, because they expect objects following
the :code:`ndarray` interface. These include packages like Dask :cite:`dask` and XArray
:cite:`xarray`.

Both of these frameworks could benefit tremendously from sparse structures. In the case of
Dask, it could be used in combination with sparse structures to scale up computational tasks
that need sparse structures. In the case of XArray, datasets with large amounts of missing
data could be represented efficiently.

In this paper, we present Sparse, a sparse array library that supports arbitrary dimension
sparse arrays and supports most common parts of the :code:`ndarray` interface. It supports
basic arithmetic, application of :code:`ufunc` s directly to sparse arrays (including with
broadcasting), most common reductions, indexing, concatenation, stacking, transpose, reshape
and a number of other features. The primary format in this library is based on the coordinate
format, which stores indices where the array is nonzero, and the corresponding data.

Since a full explanation of usage would be a repeat of the NumPy user manual and the package
documentation, we only provide brief demos of certain features of interest. Then, we move on
to some of the design decisions that went into making this package, some optimizations,
applications and possible future work.
