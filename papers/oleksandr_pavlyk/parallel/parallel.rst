In this section, we introduce a new feature in Numba that automatically parallelizes NumPy programs.
Achieving high performance with Python on modern multi-core CPUs is challenging since Python implementations are generally interpreted and prohibit parallelism.
To speed up sequential execution, Python functions can be compiled to fast native code using Numba, which in turns uses the LLVM JIT compiler. 
All a programmer has to do to use Numba is to annotate their functions with Numba's `@jit` decorator.
However, the Numba JIT will not parallelize NumPy functions, even though the majority of them are known to have parallel semantics, and thus cannot make use of multiple cores.
Furthermore, even if individual NumPy functions were parallelized, a program containing many such functions would likely have lackluster performance due to poor cache behavior. 
Numba's existing solution is to allow users to write scalar kernels in OpenCL style, which can be executed in parallel. 
However, this approach requires significant programming effort to rewrite existing array code into explicit parallelizable scalar kernels and therefore hurts productivity
and is beyond the capabilities of some programmers. 
To achieve both high performance and high programmer productivity, 
we have implemented an automatic parallelization feature as part of the Numba JIT compiler. 
With auto-parallelization turned on, Numba attempts to identify operations with parallel semantics and to fuse adjacent ones together to form kernels that are automatically run in parallel, all fully automated without manual effort from the user.

Our implementation supports the following parallel operations:

1. Common arithmetic functions between NumPy arrays, and between arrays and scalars, as well as NumPy ufuncs. 
   They are often called `element-wise` or `point-wise` array operations:

    * unary operators: ``+`` ``-`` ``~``
    * binary operators: ``+`` ``-`` ``*`` ``/`` ``/?`` ``%`` ``|`` ``>>`` ``^`` ``<<`` ``&`` ``**`` ``//``
    * comparison operators: ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``
    * NumPy ufuncs that are supported in Numba's nopython mode.
    * User defined `DUFunc` through `numba.vectorize`.

2. NumPy reduction functions ``sum`` and ``prod``. Note that they have to be
   written as ``numpy.sum(a)`` instead of ``a.sum()``.

3. NumPy ``dot`` function between a matrix and a vector, or two vectors.
   In all other cases, Numba's default implementation is used.

4. Multi-dimensional arrays are also supported for the above operations when operands have matching dimension and size. 
   The full semantics of NumPy broadcast between arrays with mixed dimensionality or size is not supported, nor is the reduction across a selected dimension.


As an example, a Logistic Regression function is given below:

.. code-block:: python

    @jit(parallel=True)
    def logistic_regression(Y, X, w, iterations):
        for i in range(iterations):
            w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
        return w

We will not discuss details of the algorithm, but instead focus on how this program behaves with auto-parallelization:

1. Input ``Y`` is a vector of size ``N``, ``X`` is an ``N x D`` matrix, and ``w`` is a vector of size ``D``.

2. The function body is an iterative loop that updates variable ``w``.
   The loop body consists of a sequence of vector and matrix operations.

3. The inner ``dot`` operation produces a vector of size ``N``, followed by a sequence of arithmetic operations either between a scalar and vector of size ``N``, or two vectors both of size ``N``.

4. The outer ``dot`` produces a vector of size ``D``, followed by an inplace array subtraction on variable ``w``.

5. With auto-parallelization, all operations that produce array of size ``N`` are fused together to become a single parallel kernel. 
   This includes the inner ``dot`` operation and all point-wise array operations following it.

6. The outer ``dot`` operation produces a result array of different dimension, and is not fused with the above kernel.

Here, the only thing required to take advantage of parallel hardware is to set the `parallel=True` option for `@jit`, with no modifications to the ``logistic_regression`` function itself.  
If we were to give an equivalent parallel implementation using Numba's `@guvectorize` decorator, it would require a pervasive change that rewrites the code to extract kernel computation that can be parallelized, which is both tedious and challenging.

We measure the performance of automatic parallelization over three workloads, comparing auto-parallelization with Numba's sequential JIT, in relative speed to the default Python implementation.

.. figure:: parallel/parallel_benchmarks.jpg

(Some discussion on benchmarks)

(Some discussion on implementation)

