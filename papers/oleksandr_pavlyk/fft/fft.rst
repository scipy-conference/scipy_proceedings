Thin wrapper exposing Intel |R| MKL's FFT functionality directly on NumPy arrays was used to optimize FFT support in both NumPy and SciPy.
It allows FFT-based Python programs to approach performance of equivalent C implementations.

.. provide charts of Python code performance in terms of percent of native performance [ reuse charts for Haswell from release notes ]
.. figure:: fft/FFT_perf_percent_native_u2.png

Thanks to Intel |R| MKL's flexibility in its supports for arbitrarily strided input and output arrays [1]_ both one-dimensional and
multi-dimensional complex Fast Fourier Transforms along distinct axes can be performed directly, without the need to copy the input
into a contiguous array first. Input strides can be arbitrary, including negative or zero, as long strides remain an integer multiple
of array's item size.

.. provide charts of computing FFT along axis, FFT of transposed array, FFT of stack of images, etc.

.. table:: Table of times of repeted executions of FFT computations over an array of complex doubles in different Python distributions on Intel (R) Xeon (R) E5-2698 v3 @ 2.30GHz with 64GB of RAM.

   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | computation  | `np.fft.fft(arg)` | `np.fft.fft(arg, axis=0)` | `np.fft.fft2(arg)` | `np.fft.fftn(arg)`         |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | arg.shape    | `(3000000,)`      |  `(1860, 1420)`           |  `(275, 274, 273)` | `(275, 274, 273)`          |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | arg.strides  | `(10*16,)`        |  C-contiguous             |  F-contiguous      | `(16, 274*275*16, 275*16)` |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | repetitions  |  16               |  16                       |  8                 | 8                          |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | IDP 2017.0.3 | 0.162 |+-| 0.01   |  0.113 |+-| 0.01          |  8.869 |+-| 0.085  | 0.863 |+-| 0.011           |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | IDP 2017.0.1 | 0.187 |+-| 0.06   |  1.046 |+-| 0.03          |  10.301 |+-| 0.098 | 12.382 |+-| 0.027          |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+
   | pip numpy    | 2.333 |+-| 0.01   |  1.769 |+-| 0.02          |  29.941 |+-| 0.029 | 34.455 |+-| 0.007          |
   +--------------+-------------------+---------------------------+--------------------+----------------------------+


The wrapper supports both in-place and out-of-place modes, enabling it to efficiently power both `numpy.fft` and `scipy.fftpack` submodules.
In-place operations are only performed where possible.

.. provide charts comparing timings of in-place and out-of-place FFT computations
.. provide charts comparing timings of in-place operations in update 2|3 vs. update 1

.. table:: Table of times of repeated execution of `scipy.fftpack` functions with `overwrite_x=True` (in-place) and `overwrite_x=False` (out-of-place) on a C-contiguous arrays of complex double and complex singles.

   +----------------+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   | computation    | `fft(arg)`      | `fft(arg)`      | `fft2(arg)`     |  `fft2(arg)`    |  `fftn(arg)`      |     `fftn(arg)`  |
   +----------------+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   | `overwrite_x`  | False           |  True           |  False          |  True           |  False            |      True        |
   +----------------+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   | `arg.shape`    | `(3*10**6,)`    | `(3*10**6,)`    | `(1860, 1420)`  | `(1860, 1420)`  | `(273, 274, 275)` | `(273, 274, 275)`|
   +-------------+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cd| 1.40 |+-| 0.02  | 0.885 |+-| 0.005| 0.090 |+-| 0.001| 0.067 |+-| 0.001| 0.868 |+-| 0.007  | 0.761 |+-| 0.001 |
   | IDP 2017.0.3+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cs| 0.734 |+-| 0.004| 0.450 |+-| 0.002| 0.056 |+-| 0.001|0.041 |+-| 0.0002| 0.326 |+-| 0.003  | 0.285 |+-| 0.002 |
   +-------------+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cd| 1.77 |+-| 0.02  | 1.760 |+-| 0.012| 2.208 |+-| 0.004| 2.219 |+-| 0.002| 22.77 |+-| 0.38   | 22.7  |+-| 0.5   |
   | IDP 2017.0.1+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cs| 5.79 |+-| 0.14  | 5.75 |+-| 0.02  | 1.996 |+-| 0.1  | 2.258 |+-| 0.001| 27.12 |+-| 0.05   | 26.8  |+-| 0.25  |
   +-------------+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cd| 26.06 |+-| 0.01 | 23.51 |+-| 0.01 | 4.786 |+-| 0.002| 3.800 |+-| 0.003| 67.69 |+-| 0.12   | 81.46 |+-| 0.01  |
   | pip numpy   +--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+
   |             |cs| 28.4 |+-| 0.1   | 11.9 |+-| 0.05  | 5.010 |+-| 0.003| 3.77  |+-| 0.02 | 69.49 |+-| 0.02   | 80.54 |+-| 0.07  |
   +-------------+--+-----------------+-----------------+-----------------+-----------------+-------------------+------------------+


Direct support for multivariate transforms along distinct array axis. Even when multivariate transform ends up being computed as iterations
of one-dimensional transforms, all subsequent iterations are performed in place for efficiency.

The update also provides dedicated support for complex FFTs on real inputs, such as `np.fft.fft(real_array)`, by leveraging corresponding
functionality in MKL [2]_.

.. Illustrate the point that this became faster

Dedicated support for specialized real FFTs, which only store independent complex harmonics. Both `numpy.fft.rfft` and `scipy.fftpack.rfft`
storage  modes are natively supported via Intel MKL.

.. show rfft is faster in update 2 relative to update 1


.. |+-| unicode:: 0x00B1 .. plus-minus sign
   :ltrim:

.. [1] https://software.intel.com/en-us/mkl-developer-reference-c-dfti-input-strides-dfti-output-strides#10859C1F-7C96-4034-8E66-B671CE789AD6
.. [2] https://software.intel.com/en-us/mkl-developer-reference-c-dfti-complex-storage-dfti-real-storage-dfti-conjugate-even-storage#CONJUGATE_EVEN_STORAGE
.. [fft_bench] http://github.com/intelpython/fft_benchmark
