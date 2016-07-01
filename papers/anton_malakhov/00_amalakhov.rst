:author: Anton Malakhov
:email: Anton.Malakhov@intel.com
:institution: Intel Corporation

-----------------------------------------------
Composable Multi-Threading for Python Libraries
-----------------------------------------------

.. class:: abstract

   Python is popular among numeric communities that value it for easy to use number crunching modules like Numpy/Scipy, Dask, Numba, and many others.
   These modules often use multi-threading for efficient parallelism (on a node) in order to utilize all the available CPU cores.
   But being used together in one application, their threads can interfere with each other leading to overheads and inefficiency.
   The lost performance can still be recovered if all the multi-threaded parties are coordinated.
   This paper describes usage of Intel |R| Threading Building Blocks (Intel |R| TBB), an open-source cross-platform library for multi-core parallelism, as coordination layer for Python libraries which helps to extract additional performance for numeric applications on multi-core systems.

.. class:: keywords
   Multi-threading, GIL, Over-subscription, Parallel Computations, Parallelism, Multi-core, Dask, Joblib, Numpy, Scipy

Introduction to Over-subscription
---------------------------------
Multi-processing parallelism in Python is prone to inefficiency due to memory-related overheads. On the other hand, multi-threaded parallelism is know to be more efficient but with Python, it suffers from the global interpreter lock (GIL) :cite:`gil` which prevents scaling of Python programs. However, when it comes to numeric computations, most of the time is spent in native codes where the GIL can easily be released and programs can scale. This is why Python libraries such as Dask and Numba can use multi-threading to greatly speed up the computations. But when used together, e.g. when a Dask task calls Numba's threaded ufunc, it leads to the situation where there are more active software threads than available hardware resources. This situation is called over-subscription and it can lead to sub-optimal execution due to frequent context switches, thread migration, broken cache-locality, and finally to a load imbalance when some threads finished their work but others are stuck along with the overall progress.

Another example are Numpy/Scipy libraries. For example in  Intel |R| Distribution for Python, they are accelerated using Intel |R| Math Kernel Library (Intel |R| MKL). Intel |R| MKL is threaded by default using OpenMP which is known for its inherent restrictions. For instance, OpenMP keeps the threads active so they can be reused in subsequent parallel regions. Usually, this is useful approach to reduce work distribution overhead. But with another active thread pool in the application, it plays against better performance because while OpenMP worker threads keep consuming CPU time in busy-waiting loops, the other parallel work (like Numba's ufunc with :code:`target=parallel`) cannot start until OpenMP threads stop spinning or are pre-empted by the OS.

Though overheads from linear over-subscription (e.g. 2x) are not always visible on the application level (especially for small systems) and can be tolerated in many cases when the work for parallel region is big enough. But the worst case is when a program starts multiple parallel tasks and each of these tasks ends up executing an OpenMP parallel region. This results by default in quadratic over-subscription which ruins multi-threaded performance on systems with significant number of threads (roughly, tens and more). In some big systems, sometimes, it is not even possible to create as many software threads as the number of hardware threads multiplied by itself, it just eats up all the available resources.


Threading Composability
-----------------------
.. figure:: components.png

   Intel |R| Threading Building Blocks is used as a common coordinating runtime for different Python modules. :label:`components`

Our approach to solve these co-existence problems is to share single thread pool among all the program modules and native libraries so that one user-level task scheduler will take care of composability between them. Intel |R| Threading Building Blocks (Intel |R| TBB) library works as such a task scheduler in our solution, see Figure :ref:`components`. Intel |R| TBB is an open-source, cross-platform, recognized C++ library for enabling multi-core parallelism. It was designed for composability and nested parallelism support from its foundation so that preventing of over-subscription is a specialization of this library.

In the Intel |R| Distribution for Python* 2017 Beta and later as part of Intel |R| TBB release 4.4 Update 4, I introduce an experimental module which unlocks opportunities for additional performance for multi-threaded Python programs by enabling threading composability between two or more thread-enabled libraries. Threading composability can accelerate programs by avoiding inefficient threads allocation as discussed above.

The TBB module implements :code:`Pool` class with the standard Python interface using Intel |R| TBB which can be used to replace Python's *ThreadPool*. Thanks to the monkey-patching technique implemented in class :code:`Monkey`, no source code change is needed in order to enable single thread pool across different Python modules. It also enables TBB-based threading layer for Intel |R| MKL which automatically enables composable parallelism for Numpy and Scipy calls.


Usage example
-------------
For our first experiment, we need Intel |R| Distribution for Python* :cite:`intelpython` to be installed along with Dask :cite:`dask` library which simplifies parallelism with Python.

.. code-block:: sh

    # install Intel(R) Distribution for Python*
    <path to installer of the Distribution>/install.sh
    # setup environment
    source <path to the Distribution>/bin/pythonvars.sh
    # install Dask
    conda install dask

Now, let's write a simple program using Numpy that validates QR decomposition by multiplying resulted components and comparing result agianst original input:

.. code-block:: python
    :linenos:

    import time, numpy as np
    x = np.random.random((100000, 2000))
    t0 = time.time()
    q, r = np.linalg.qr(x)
    test = np.allclose(x, q.dot(r))
    assert(test)
    print(time.time() - t0)

And same program using Dask:

.. code-block:: python
    :linenos:

    import time, dask, dask.array as da
    x = da.random.random((100000, 2000),
                   chunks=(10000, 2000))
    t0 = time.time()
    q, r = da.linalg.qr(x)
    test = da.all(da.isclose(x, q.dot(r)))
    assert(test.compute()) # threaded
    print(time.time() - t0)

Here, Dask splits the array into 10 chunks and processes them in parallel using multiple threads. But each Dask task executes the same Numpy matrix operations which are accelerated using Intel |R| MKL under the hood and thus multi-threaded by default. This combination results in nested parallelism, i.e. when one parallel component calls another component which is also threaded.

Let's run it in 3 different modes:

.. code-block:: sh
    :linenos:

    python bench.py                   # Default MKL
    OMP_NUM_THREADS=1 python bench.py # Serial MKL
    python -m TBB bench.py            # Intel TBB mode

.. figure:: dask_qr_bench.png
   
   Execution times for QR validation example. :label:`qrpic`

Figure :ref:`qrpic` shows times (lower is better) acquired on 32-core (no HT) machine with 64GB RAM. By default, Dask version runs worse than Numpy version because 10 outermost tasks end up calling 10 OpenMP-based parallel regions which creates 10 times more threads than available hardware resourses.

The second command runs this benchmark with innormost OpenMP parallelism disabled. It results in the worst performance for Numpy version since everything is now serialized. And Dask version is not able to close the gap completely since it has only 10 tasks which can run in parallel while Numpy with parallel MKL is able to utilize the whole machine with 32 threads. The reason why only 10 tasks were selected for this demonstration is the following. If top-level parallelism can load all the available cores on the machine, there is no much sense in the nested parallelism and Intel |R| TBB shows no speedup over serial MKL version. In such cases, TBB can help by load-balancing at the end of the work, but this example is quite balanced so that there is also no visible difference.

The last command demostrates how Intel TBB can be enabled as orchestrator of multi-threaded modules. TBB module runs the benchmark in context of :code:`with TBB.Monkey():` which replaces standard Python *ThreadPool* class used by Dask and also switches MKL into TBB mode. Numpy with TBB shows more than double time comparing to default Numpy run. This happens because TBB-based threading in MKL is new and not as optimized as OpenMP-based MKL threading implementation. But despite that fact, Dask in TBB mode shows the best performance for this benchamark, more than 50% improvement comparing to default Numpy. This happens because the Dask version exposes more parallelism to the system without oversubscription overheads, hiding latencies of serial regions and fork-join synchronization in MKL functions.

.. [#] For more complete information about compiler optimizations, see our Optimization Notice


Case study
----------

Previous example was intentionaly selected to be small enough to fit into this paper with all the sources. Another case study :cite:`codefest` is closer to real-world applications. It implements recomendation system similar to the ones used on popular web-sites for generating seggestions for the next application to download or the next movie to watch. Though, the core of the algorithm is still quite simple and spends most of the time in matrix multiplication. Figure :ref:`casestudy` shows results collected on an older machine with bigger number of cores.

.. figure:: case_study.png

    Case study results: Generation of User Recommendations. :label:`casestudy`

The leftmost result was acquired on pure, non-accelerated Python which comes by default on Fedora 23. It is the base. Running the same application without modifications with Intel |R| Distribution for Python* results in 17x times speedup. One reason for this performance increase is that Intel |R| MKL runs computations in parallel. Thus for sake of experiment, outermost parallelism was implemented on the application level processing different user requests in parallel. For the same system-default python the new version helped to close the gap with MKL-based version though not completely: with x15 times faster than the base. However, running same parallel application with Intel Distribution resulted in worse performance (11x). This is explained by overheads induced by oversubscription.

In order to remove overheads, previous experiment was executed with TBB module on the command line. It results in the best performance for the application - x27 times speedup against the base.

.. [#] For more complete information about compiler optimizations, see our Optimization Notice

   
Numba
-----
Numpy and Scipy provide rich but fixed set of mathematical instruments accelerated with C extensions. However sometimes, one might need non-standard math to be as fast as C extensions. That's where Numba :cite:`numba` can be efficiently used. Numba is a JIT compiler based on LLVM :cite:`llvm`. It aims to close the gap in performance between Python and statically typed, compiled languages like C/C++ which also have popular implementation based on LLVM.

Numba implements the notion of ufunc defined in Scipy and extends it to a computation kernel which can be not only mapped onto arrays but also spread across multiple cores. The original Numba version implements it using pool of native threads and simple work-sharing scheduler which coordinates work distribution between them. If used in a parallel numeric Python application, it adds the third thread pool to the threading mess. Thus our strategy was to put it on top of common Intel |R| TBB runtime as well.

Original Numba's multi-threading runtime was replaced with very basic and naive implementation based on TBB tasks. It resulted in improved performance even without nested parallelism and advanced features of TBB partitioning algorithms. 

.. figure:: numba_tbb.png

    Black Scholes benchmark implemented with @numba.guvectorize and target=parallel. :label:`numbatbb`

The Figure :ref:`numbatbb` shows how original and Intel |R| TBB-based runtimes perform with Black Scholes benchmark implemented with Numba similar to the following code:

.. code-block:: python
    :linenos:

    @nb.guvectorize('(f4[:],f4[:],f4[:],f4[:],f4[:]'\
                    ',f4[:])', '(),(),(),(),(),()',
                    nopython=True, target='parallel')
    def BlackScholes(S, X, T, V, C, P):
        q = V[0] * sqrt(T[0])
        d1 = (log(S[0]/X[0])+(R+.5*V[0]*V[0])*T[0])/q
        d2 = d1 - q
        n1 = cnd_numba(d1)
        n2 = cnd_numba(d2)
        e = exp(-R[0] * T[0])
        C[0] = (S[0] * n1 - X[0] * e * n2)
        P[0] = (X[0] * e * (1.-n2) - S[0] * (1.-n1))


Limitations and Future Work
---------------------------
Intel |R| TBB does not work well for blocking I/O operations because it limits number of active threads. It is applicable only for tasks which do not block in the operating system. Python module for Intel |R| TBB is in an experimental stage and might be not sufficiently optimized and verified with different use-cases. In particular, it does not yet use master thread efficiently as a regular TBB program is supposed to do. As was shown before, Intel |R| MKL does not optimize TBB-based threading layer and there are huge gaps in stand-alone performance with default MKL threading. In particular, TBB-based MKL is not yet efficient on Intel |R| Xeon |R| Phi processors. But all these problems can go away as more users will be interested in solving theirs composability issues and Intel |R| MKL and the TBB module are further developed.

Another limitation is that Intel |R| TBB coordinates threads only inside single process while the most popular approach to parallelism in Python is multi-processing. Though, Intel |R| TBB survives better than OpenMP in oversubscribed environment because it does not rely on particular the number of threads at any moment participating in parallel computation thus the threads preemted by the OS are not affecting overall progress. Nevertheless, it is possible to implement cross-process coordination mechanism that prevents creation and consumption of excessive threads system-wise.

On the other hand, slow adaption of Intel |R| TBB by Intel |R| MKL suggests to find and evaluate alternative ways such as implementation of restricted subset of OpenMP on top of TBB threads or vice-versa, OpenMP threads used as Intel |R| TBB workers. In both cases, we have prototypes with initial experemental data. Anoter approach is suggested by the observation that a moderate oversubscription, such as from two fully subscribed thread pools, does not significantly affect performance of the most workloads. In this case, solving quadratic oversubscription from running multiple OpenMP regions at the same time should be practical alternative. And the solution for that can be as simple as "Global OpenMP Lock" (GOL) or more eleborated inter-process semaphore which coordinates OpenMP threads. 


Conclusion
----------
This paper described the issues of over-subscription and threading composability which affects performance of Python libraries and frameworks such as Numpy, Scipy, and Numba. Suggested solution is to use a common threading runtime such as Intel |R| TBB which limits number of threads in order to prevent oversubscription and coordinates parallel execution of independent program modules. Python module for Intel |R| TBB was implemented to substitute Python's ThreadPool implementation and switch Intel |R| MKL into TBB-based mode. Few examples show promising results where thanks to nested parallelism and TBB mode, the best performance was achieved. Intel |R| TBB along with the Python module are avaiable in open-source :cite:`opentbb` for different platforms and architectures while Intel |R| Distribution for Python* accelerated with Intel |R| MKL is available for free as stand-alone package :cite:`intelpy` and on anaconda.org/intel chanel. So, everyone are welcome to try it out and provide feedback, bug reports, and feature requests.

References
----------
.. [ParUniv] Vipin Kumar E.K. *A Tale of Two High-Performance Libraries*,
             The Parallel Universe Magazine, Special Edition, 2016.
             https://software.intel.com/en-us/intel-parallel-universe-magazine

.. figure:: opt-notice-en_080411.png
   :figclass: b
.. |C| unicode:: 0xA9 .. copyright sign
   :ltrim:
.. |R| unicode:: 0xAE .. registered sign
   :ltrim:
