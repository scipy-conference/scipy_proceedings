One of the great benefits found in our Intel |R| Distribution for Python* is the performance boost
gained from leveraging SIMD and multithreading in (select) NumPy's UMath arithmetic and
transcendental operations, on a range of Intel |R| CPUs, from Intel |R| Core |TM| to Intel |R| Xeon
|TM| & Intel |R| Xeon Phi |TM|. With stock python as our baseline, we demonstrate the scalability of 
Intel |R| Distribution for Python* by using functions that are intensively used in financial math 
applications and machine learning:

.. figure:: umath/speedup1.png

.. figure:: umath/speedup2.png

.. figure:: umath/speedup3.png


One can see that stock Python (pip-installed NumPy from PyPI) on Intel |R| Core |TM| i5 performs
basic operations such as addition, subtraction, and multiplication just as well as Intel |R| Python,
but not on Intel |R| Xeon |TM| and Intel |R| Xeon Phi |TM|, where Intel |R| Distribution for Python* provides over 10x speedup. 
This can be explained by the fact that basic arithmetic operations in stock
NumPy are hard-coded AVX intrinsics (and thus already leverage SIMD, but do not scale to other instruction 
set architectures (ISA), e.g. AVX-512). These operations in stock Python also do not leverage multiple cores (i.e. no
multi-threading of loops under the hood of NumPy exist with such operations). Intel Python's
implementation allows for this scalability by utilizing both respective Intel |R| MKL VML
CPU-dispatched and multi-threaded primitives under the hood, and Intel |R| SVML intrinsics - a compiler-provided short vector
math library that vectorizes math functions for both IA-32 and Intel |R| 64-bit architectures on
supported operating systems. Depending on the problem size, NumPy will choose one of the two
approaches. On small array sizes, Intel |R| SVML outperforms VML due to high library call overhead, but for 
larger problem sizes, VML's ability to both vectorize math functions and multi-thread loops offsets the overhead.


Specifically, on Intel |R| Core |TMP| i5 processor the Intel |R| Distribution for Python delivers greater performance in 
numerical evaluation of transcendental functions (log, exp, erf, etc.) due to utilization of both SIMD and multi-threading. 
We do not see any visible benefit of multi-threading basic operations (as shown on the graph) unless NumPy arrays are very
large (not shown on the graph). On Intel |R| Xeon |TM| processor, the 10x-1000x boost is explained by leveraging both
(a) AVX2 instructions to evaluate transcendentals and (b) multiple cores (32 in our setup). Even greater
scalability of Intel |R| Xeon Phi |TM| relative to Intel |R| Xeon |TM| is explained by larger number of cores (64 in our
setup) and wider vector registers.


The following charts provide another view of Intel |R| Distribution for Python performance versus stock Python on
arithmetic and transcendental vector operations in NumPy by measuring how close UMath performance
is to the respective native MKL call:

.. figure:: umath/native_c_comp1.png

.. figure:: umath/native_c_comp2.png

  
Again on Intel |R| Core |TM| i5 the stock Python performs well on basic operations (due to
hard-coded AVX intrinsics and because multi-threading from Intel |R| Distribution for Python does not add much 
on basic operations) but does not scale on transcendentals (loops with transcendentals are not vectorized in
stock Python). Intel |R| Distribution for Python delivers performance close to native speeds (90% of MKL) on relatively
big problem sizes.


To demonstrate the benefits of vectorization and multi-threading in a real-world application, we
chose to use the Black Scholes model, used to estimate the price of financial derivatives, 
specifically European vanilla stock options. A Python implementation of the Black Scholes formula 
gives an idea of how NumPy UMath optimizations can be noticed at the application level:

.. figure:: umath/black_scholes1.png

.. figure:: umath/black_scholes2.png

.. figure:: umath/black_scholes3.png


One can see that on Intel |R| Core |TM| i5 the Black Scholes Formula scales nicely with Intel
Python on small problem sizes but does not perform well on bigger problem sizes, which is explained
by small cache sizes. Stock Python does marginally scale due to leveraging AVX instructions on
basic arithmetic operations, but it is a whole different story on Intel |R| Xeon |TM| and Intel
|R| Xeon Phi |TM|. Using Intel |R| Distribution for Python to execute the same Python code on 
server processors, much greater scalability on much greater problem sizes is observed. 
Intel |R| Xeon Phi |TM| scales better due to bigger number of cores and as expected, while the 
stock Python does not scale on server processors due to the lack of AVX2/AVX-512 support for 
transcendentals and no utilization of multiple cores.
