Wikipedia defines SIMD as:
    
    Single instruction, multiple data (SIMD), is a class of parallel computers in Flynn's taxonomy. 
    It describes computers with multiple processing elements that perform the same operation on multiple data points simultaneously.
    Thus, such machines exploit data level parallelism, but not concurrency: there are simultaneous (parallel) computations,
    but only a single process (instruction) at a given moment.
    Most modern CPU designs include SIMD instructions in order to improve the performance of multimedia use.

To utilize power of CPU's SIMD instructions compilers need to implement special optimization passes, so-called code vectorization.
Modern optimizing compilers implement automatic vectorization - a special case of automatic parallelization, 
where a computer program is converted from a scalar implementation, which processes a single pair of operands at a time,
to a vector implementation, which processes a single operation on multiple pairs of operands at once.

According Numba's project page Numba is an Open Source NumPy-aware optimizing compiler for Python. 
It uses the remarkable LLVM compiler infrastructure to compile Python syntax to machine code. And it is quite expected that Numba tries
to use all these features to improve performance especially for scientific applications. 


LLVM implemented auto-vectorization for simple cases several years ago but there remain sigificant problems with vectorization of elementary transcendental math functions.
To enable proper vectorization support a special vectorized implementation of math functions such as ``sin``, ``cos``, ``exp`` is needed.

The Intel |R| C++ Compiler provides short vector math library (SVML) intrinsics implementing vectorized mathematical functions.
These intrinsics are available for IA-32 and Intel |R| 64 architectures running on supported operating systems.

The SVML intrinsics are vector variants of corresponding scalar math operations using ``__m128``, ``__m128d``, ``__m256``, ``__m256d``, and ``__m256i`` data types.
They take packed vector arguments, simultaneously perform the operation on each element of the packed vector argument, and return a packed vector result. Due to low overhead
of the packing for aligned contiguously laid out data, vector operations may offer speed-ups over scalar operations which are proportional to the width of the vector register.

For example, the argument to the ``_mm_sin_ps`` intrinsic is a packed 128-bit vector of four 32-bit precision floating point numbers. The intrinsic simultaneously computes values of the sine function for each of these four numbers and returns the four results in a packed 128-bit vector, all within about the time of scalar evaluation of only one argument. 

Using SVML intrinsics is faster than repeatedly calling the scalar math functions. However, the intrinsics may differ from the corresponding scalar functions in accuracy of their results.

Besides intrinsics available with Intel |R| compiler there is opportunity to call vectorized implementations directly from svml library by their names.

Beginning with version 4.0 LLVM features (experimental) model of autovectorization using SVML library, so a full stack of technologies is now available to exploit in-core parallelization of python code.

Let's see how it works with a small example:

.. code-block:: python

    import math
    import numpy as np
    from numba import njit

    def foo(x,y):
        for i in range(x.size):
            y[i] = math.sin(x[i])
    foo_compiled = njit(foo)

Inspite of the fact that numba generates call for usual ``sin`` function, as seen in the following excerpt from the generated LLVM code:

.. code-block:: text
    
    label 16:
        $16.2 = iternext(value=$phi16.1)    ['$16.2',
                                             '$phi16.1']
        $16.3 = pair_first(value=$16.2)     ['$16.2', 
                                             '$16.3']
        $16.4 = pair_second(value=$16.2)    ['$16.2', 
                                             '$16.4']
        del $16.2                           []
        $phi19.1 = $16.3                    ['$16.3', 
                                             '$phi19.1']
        del $16.3                           []
        branch $16.4, 19, 48                ['$16.4']
    label 19:
        del $16.4                           []
        i = $phi19.1                        ['$phi19.1', 
                                             'i']
        del $phi19.1                        []
        $19.2 = global(math: <module 'math'\
         from '/path_stripped/lib-dynload/\
         math.cpython-35m-x86_64-...,.so'>) ['$ 19.2']
        $19.3 = getattr(attr=sin, 
                        value=$19.2)        ['$19.2',
                                             '$19.3']
        del $19.2                           []
        $19.6 = getitem(index=i, value=x)   ['$19.6',
                                             'i', 'x']
        $19.7 = call $19.3($19.6)           ['$19.3',
                                             '$19.6',
                                             '$19.7']
        del $19.6                           []
        del $19.3                           []
        y[i] = $19.7                        ['$19.7',
                                             'i', 'y']
        del i                               []
        del $19.7                           []
        jump 16                             []

    
We can see direct use of the SVML-provided vector implementation of sine function:

.. code-block:: Asm

            movq    %rdi, 8(%rsp)
            movq    %r13, 16(%rsp)
            movq    %r15, 24(%rsp)
            subq    %rbx, %r12
            leaq    96(%rdx), %r14
            leaq    96(%rsi), %r15
            movabsq $__svml_sin4_ha, %rbp
            movq    %rbx, %r13
            .p2align        4, 0x90
    .LBB0_13:
            vmovups -96(%r14), %ymm0
            vmovups -64(%r14), %ymm1
            vmovups %ymm1, 32(%rsp)
            vmovups -32(%r14), %ymm1
            vmovups %ymm1, 64(%rsp)
            vmovups (%r14), %ymm1
            vmovups %ymm1, 128(%rsp)
            callq   *%rbp
            vmovups %ymm0, 96(%rsp)
            vmovups 32(%rsp), %ymm0
            callq   *%rbp
            vmovups %ymm0, 32(%rsp)
            vmovups 64(%rsp), %ymm0
            callq   *%rbp
            vmovups %ymm0, 64(%rsp)
            vmovupd 128(%rsp), %ymm0
            callq   *%rbp
            vmovups 96(%rsp), %ymm1
            vmovups %ymm1, -96(%r15)
            vmovups 32(%rsp), %ymm1
            vmovups %ymm1, -64(%r15)
            vmovups 64(%rsp), %ymm1
            vmovups %ymm1, -32(%r15)
            vmovupd %ymm0, (%r15)
            subq    $-128, %r14
            subq    $-128, %r15
            addq    $-16, %r13
            jne     .LBB0_13

Thanks to enabled support of high accuracy SVML functions in LLVM this jitted code sees more than 4x increase in performance.

svml enabled:

.. code-block:: python

    %timeit foo_compiled(x,y)
    1000 loops, best of 3: 403 us per loop

svml disabled:

.. code-block:: python

    %timeit foo_compiled(x,y)
    1000 loops, best of 3: 1.72 ms per loop

