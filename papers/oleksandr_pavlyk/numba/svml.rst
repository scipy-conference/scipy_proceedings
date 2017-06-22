Wikipedia defines SIMD as:
    
    Single instruction, multiple data (SIMD), is a class of parallel computers in Flynn's taxonomy. 
    It describes computers with multiple processing elements that perform the same operation on multiple data points simultaneously.
    Thus, such machines exploit data level parallelism, but not concurrency: there are simultaneous (parallel) computations,
    but only a single process (instruction) at a given moment.
    Most modern CPU designs include SIMD instructions in order to improve the performance of multimedia use.

To utilize power of CPU's SIMD instructions compilers need to implement special optimization passes so-called - code vectorization.
Modern optimizing compilers implements automatic vectorization - a special case of automatic parallelization, 
where a computer program is converted from a scalar implementation, which processes a single pair of operands at a time,
to a vector implementation, which processes one operation on multiple pairs of operands at once.

According Numba's project page Numba is an Open Source NumPy-aware optimizing compiler for Python. 
It uses the remarkable LLVM compiler infrastructure to compile Python syntax to machine code. And it quite expected that Numba tries
to use all this features to improve performance especially for scientific applications. 


LLVM implemented autovectorization for simple cases several years ago but there was quite sigificant problem with math functions.
For proper vectorization it's needed to have special vectorized implementation of math fucntions as sin, cos, exp... 

The Intel |R| C++ Compiler provides short vector math library (SVML) intrinsics to compute vector math functions.
These intrinsics are available for IA-32 and Intel® 64 architectures running on supported operating systems.

The SVML intrinsics are vector variants of corresponding scalar math operations using ``__m128``, ``__m128d``, ``__m256``, ``__m256d``, and ``__m256i`` data types.
They take packed vector arguments, perform the operation on each element of the packed vector argument, and return a packed vector result.

For example, the argument to the ``_mm_sin_ps`` intrinsic is a packed 128-bit vector of four 32-bit precision floating point numbers. The intrinsic computes the sine of each of these four numbers and returns the four results in a packed 128-bit vector.

Using SVML intrinsics is faster than repeatedly calling the scalar math functions. However, the intrinsics differ from the scalar functions in accuracy.

Beginning from 4.0 version of LLVM there is available (I'd say experimentally) model of autovectorization using SVML library,
so now we can say that full stack of technologies should be ready to exploit in-core parallelization of python code.

Let's see how it works:

.. code-block:: python
    
    # small example

.. code-block:: LLVM
    
    # IR generated

.. code-block:: Asm
    
    # generated code


TBD: fast and slow variants performance measurements.


