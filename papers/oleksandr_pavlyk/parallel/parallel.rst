Numba speeds up array-oriented Python programs by compiling them to native code using the LLVM JIT compiler. 
This is often achieved by adding a simple annotation `@jit` without user having to make signification changes to their existing programs written using the NumPy libraries.
However, Numpy functions are still compiled to sequential code and cannot make use of multi-core CPUs.
Numba's existing solution is to allow user to write scalar kernels in OpenCL style, which can be executed in parallel. 
This, however, requires significant programming effort to rewrite existing array code into explicit parallelizable scalar kernels, and hurts productivity. 
As we all know, the majority of NumPy array functions, such as adding s scalar value to an array and so on, are known to have parallel semantics. 
Therefore, it would makes perfect sense to give them a parallel implementation. 
Futhermore, a user program may contain many such operations and while each operation could be parallelized individually, such an approach often has lackluster performance due to poor cache behavior. 
To achieve this goal, we have implemented an automatic parallelization feature as part of the Numba JIT compiler. 
With auto-parallelization turned on, Numba attempts to identify such operations in a user program, and fuse adjacent ones together, to form one or more kernels that are automatically run in parallel, all fully automated without manual effort from the user.

Our implementation supports the following parallel operations:

1. Common arithmetic functions between Numpy arrays, and between arrays and scalars, as well as Numpy ufuncs. 
   They are often called `element-wise` or `point-wise` array operations:

    * unary operators: ``+`` ``-`` ``~``
    * binary operators: ``+`` ``-`` ``*`` ``/`` ``/?`` ``%`` ``|`` ``>>`` ``^`` ``<<`` ``&`` ``**`` ``//``
    * comparison operators: ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``
    * Numpy ufuncs that are supported in Numba's nopython mode.
    * User defined `DUFunc` through `numba.vectorize`.

2. Numpy reduction functions ``sum`` and ``prod``. Note that they have to be
   written as ``numpy.sum(a)`` instead of ``a.sum()``.

3. Numpy ``dot`` function between a matrix and a vector, or two vectors.
   In all other cases, Numba's default implementation is used.

4. Multi-dimensional arrays are also supported for the above operations when operands have matching dimension and size. 
   The full semantics of Numpy broadcast between arrays with mixed dimensionality or size is not supported, nor is the reduction across a selected dimension.


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

Here, the only thing required to take advantage of parallel hardware is to set the `parallel=True` option for `@jit`, with no modifications to the ``logistic_regression`` function itself.  If we were to give an equivalence parallel implementation using Numba's `@guvectorize` decorator, it would require a pervasive change that rewrites the code to extract kernel computation that can be parallelized, which was both tedious and challenging.

We measure the performance of automatic parallelization over three workloads, comparing auto-parallelization with Numba's sequential JIT, in relative speed to the default Python implementation.

.. figure:: parallel/parallel_benchmarks.jpg

(Some discussion on benchmarks)

(Some discussion on implementation)

