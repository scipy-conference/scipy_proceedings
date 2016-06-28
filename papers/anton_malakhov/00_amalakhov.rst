:author: Anton Malakhov
:email: Anton.Malakhov@intel.com
:institution: Intel Corporation

-----------------------------------------------
COMPOSABLE MULTI-THREADING FOR PYTHON LIBRARIES
-----------------------------------------------

.. class:: abstract

   Python is popular among numeric communities that value it for easy to use number crunching modules like Numpy/Scipy, Dask, Numba, and many others.
   These modules often use multi-threading for efficient parallelism (on a node) in order to utilize all the available CPU cores.
   But being used together in one application, their threads can interfere with each other leading to overheads and inefficiency.
   The lost performance can still be recovered if all the multi-threaded parties are coordinated.
   This paper describes usage of Intel(R) Threading Building Blocks (Intel(R) TBB), an open-source cross-platform library for multi-core parallelism, as coordination layer for Python libraries which helps to extract additional performance for numeric applications on multi-core system.

.. class:: keywords
   Multi-threading, GIL, Over-subscription, Parallel Computations, Parallelism, Threading, Dask, Joblib, Numpy, Scipy

Over-subscription
-----------------
Multi-processing parallelism in Python is prone to inefficiency due to memory-related overheads. On the other hand, multi-threaded parallelism is know to be more efficient but with Python, it suffers from the global interpreter lock (GIL) [TODO: Ref?] which prevents scaling of Python programs. However, when it comes to numeric computations, most of the time is spent in native codes where the GIL can easily be released and programs can scale. This is why Python libraries such as Dask and Numba can use multi-threading to greatly speed up the computations. But when used together, e.g. when a Dask task calls Numba's threaded ufunc, it leads to the situation where there are more active software threads than available hardware resources. This situation is called over-subscription and it can lead to sub-optimal execution due to frequent context switches, thread migration, broken cache-locality, and finally to a load imbalance when some threads finished their work but others are stuck along with the overall progress.

Another example are Numpy/Scipy libraries. When they are accelerated using Intel(R) Math Kernel Library (Intel(R) MKL) like the ones shipped as part of Intel(R) Distribution for Python. Intel MKL is by default is threaded using OpenMP which is known for its inherent restrictions. In particular, OpenMP threads keep busy-waiting after the work is done - which is usually useful to reduce work distribution overhead for the next possible parallel region; but with another active thread pool in application, it plays against performance because while OpenMP worker threads keep consuming CPU time in busy-waiting, the other parallel work like Numba's ufunc (with target=parallel) cannot start until OpenMP threads stop spinning or are pre-empted by the OS.

Though overheads from linear over-subscription (e.g. 2x) are not always visible on the application level for small systems and can be tolerable in many cases when the work for parallel region is big enough. But the worst case is when a program starts multiple parallel tasks and each of these tasks ends up executing an OpenMP parallel region. This results by default in quadratic over-subscription which ruins multi-threaded performance on systems with significant number of threads (roughly, tens and more).

Especially, over-subscription is bad for Many Integrated Core (MIC) systems such as Intel(R) Xeon Phi which support more than 2 hundred hardware threads. In such big systems, sometimes it is not even possible to create as many software threads as the number of hardware threads multiplied by itself. It will just eat up all the available RAM.


Threading Composability
-----------------------
Our approach to solve these co-existence problems is to share one thread pool among all the program modules and native libraries so that one task scheduler will take care of composability between them. Intel(R) Threading Building Blocks (Intel TBB) library works as such a task scheduler in our solution. Intel TBB is a wide-spread and recognized C++ library for enabling multi-core parallelism. It was designed for composability and nested parallelism support from its foundation so that preventing of over-subscription is a specialization of this library.

In the Intel(R) Distribution for Python* 2017 Beta, I introduce an experimental module which unlocks opportunities for additional performance for multi-threaded Python programs by enabling threading composability between two or more thread-enabled libraries. Threading composability can accelerate programs by avoiding inefficient threads allocation discussed above.

The module implements Pool class with the standard Python interface using Intel(R) TBB which can be used to replace Python's ThreadPool. Thanks to the monkey-patching technique implemented in class Monkey, no source code change is needed in order to enable single thread pool across different Python modules. It also enables TBB-based threading layer for Intel(R) MKL which automatically enables composable parallelism for Numpy and Scipy calls.

Using TBB module
----------------
For our first experiment, we need Intel(R) Distribution for Python to be installed along with Dask library:

.. code-block:: sh

    source <path to Intel(R) Distribution for Python*>/bin/pythonvars.sh
    conda install dask

Now, let's write a simple program in bench.py which exploits nested parallelism and prints time spent for the computation, like the following:

.. code-block:: python

    import dask, time
    import dask.array as da
     
    t0 = time.time()
     
    x = da.random.random((10000, 10000), chunks=(4096, 4096))
    x.dot(x.T).sum().compute()
     
    print(time.time() - t0)

Here, Dask splits the array into chunks and processes them in parallel using multiple threads. But each Dask task executes expensive matrix multiplication (`dot`) which is accelerated using Intel(R) MKL under the hood and thus multi-threaded by itself. It results in nested parallelism which is handled best with Intel(R) TBB.

To run it as is (baseline)::

    python bench.py

And to unlock additional performance::

    python -m TBB bench.py

[TODO: Replace with real data] That's it! Depending on machine configuration, you can get about 20%-50% reduction of the compute time for this particular example or even more if there is a background activity on the machine.

Numba
-----
Another area where we applied Intel TBB is Numba. I replaced multi-threading runtime used by original Numba with implementation based on TBB tasks. It improved performance even without nested parallelism:
[Diagram here]

[TODO: add another example with nested parallelism based on Numba and the performance data]

Multi-processing
----------------
[TODO: I can show that TBB helps even with multiprocessing parallelism and discuss ways how it can be further improved]

Disclaimers
-----------
TBB module does not work well for blocking I/O operations, it is applicable only for tasks which do not block in the operating system. This version of TBB module is experimental and might be not sufficiently optimized and verified with different use-cases. In particular, it does not yet use master thread efficiently as regular TBB program is supposed to do. But all these problems well go away as more users will be interested in solving theirs composability issues and the TBB module is further developed.
