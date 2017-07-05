:author: Anton Malakhov
:email: Anton.Malakhov@intel.com
:institution: Intel Corporation
:corresponding:

:author: Anton Gorshkov
:email: Anton.V.Gorshkov@intel.com
:institution: Intel Corporation
:equal-contributor:

:author: Terry Wilmarth
:email: Terry.L.Wilmarth@intel.com
:institution: Intel Corporation

:year: 2017
:video: Unknown yet

---------------------------------------------------------------------
Composable Multi-Threading and Multi-Processing for Numeric Libraries
---------------------------------------------------------------------

.. class:: abstract

   Python is popular among scientific communities that value its simplicity and power, which comes with number crunching modules like [NumPy]_, [SciPy]_, [Dask]_, [Numba]_, and many others.
   These modules often use multi-threading for efficient multi-core parallelism in order to utilize all the available CPU cores.
   Nevertheless, their threads can interfere with each other leading to overhead and inefficiency if used together in one application on machines with large number of cores.
   The loss of performance can be prevented if all the multi-threaded parties are coordinated.
   This paper continues the work started in [AMala16]_ by introducing more approaches to such coordination for both multi-threading and multi-processing cases:
   using static settings, limiting the number of simultaneously active [OpenMP]_ parallel regions, and optional parallelism with Intel |R| Threading Building Blocks (Intel |R| [TBB]_).
   These approaches help to unlock additional performance for numeric applications on multi-core systems.

.. class:: keywords

   Multi-threading, Multi-processing, Oversubscription, Parallel Computations, Nested Parallelism, Multi-core, Python, GIL, Dask, Joblib, NumPy, SciPy, TBB, OpenMP

.. [AMala16] Anton Malakhov, "Composable Multi-Threading for Python Libraries", Proc. of the 15th Python in Science Conf. (SCIPY 2016), July 11-17, 2016.
.. [NumPy] NumPy, http://www.numpy.org/
.. [SciPy] SciPy, https://www.scipy.org/
.. [Dask]  Dask, http://dask.pydata.org/
.. [Numba] Numba, http://numba.pydata.org/
.. [TBB]   Intel(R) TBB open-source site, https://www.threadingbuildingblocks.org/
.. [OpenMP] The OpenMP(R) API specification for parallel programming, http://openmp.org/


1. Motivation
-------------
The fundamental shift toward parallelism was loudly declared more than 11 years ago [HSutter]_ and multi-core processors have become ubiquitous nowadays [WTichy]_.
However, the adoption of multi-core parallelism in the software world has been slow and Python along with its computing ecosystem is not an exception.
Python suffers from several issues which make it suboptimal for parallel processing.

.. [HSutter] Herb Sutter, "The Free Lunch Is Over", Dr. Dobb's Journal, 30(3), March 2005.
             http://www.gotw.ca/publications/concurrency-ddj.htm
.. [WTichy]  Walter Tichy, "The Multicore Transformation", Ubiquity, Volume 2014 Issue May, May 2014. DOI: 10.1145/2618393.
             http://ubiquity.acm.org/article.cfm?id=2618393

In particular, Python infamous global interpreter lock [GIL]_ makes it hard to scale an interpreter-dependent code
using multiple threads, effectively serializing them.
Thus the parallelism with multiple isolated processes is popular and widely used in Python
since it allows to avoid the issues with GIL but it is prone to inefficiency due to memory-related overhead.
However, when it comes to numeric computations with libraries like Numpy,
most of the time is spent in C extensions whithout accessing Python data structures.
It allows to release the GIL during computations and which enables scaling of compute-intensive applications.
Thus, both multi-processing and multi-threading approaches are valuable for Python users and have its own areas of applicability.

.. [GIL] David Beazley, "Understanding the Python GIL", PyCON Python Conference, Atlanta, Georgia, 2010.
         http://www.dabeaz.com/python/UnderstandingGIL.pdf

Indeed, scaling parallel programs is not an easy thing.
There are two fundamental laws which mathematically describe and predict scalability of a program:
Amdahl's Law and Gustafson-Barsis' Law [AGlaws]_.
According to Amdahl's Law, speedup is limited by the serial portion of the work,
which effectively puts a limit on scalability of parallel processing for a fixed-size job.
Python is especially vulnerable to this because it makes the serial part of the same code much slower
compared to implementations in some other languages due to its deeply dynamic and interpretative nature.
Moreover, the GIL makes things serial often where they potentially can be parallel, further adding to the serial portion of a program.

.. [AGlaws] Michael McCool, Arch Robison, James Reinders, "Amdahl's Law vs. Gustafson-Barsis' Law", Dr. Dobb's Parallel, October 22, 2013.
            http://www.drdobbs.com/parallel/amdahls-law-vs-gustafson-barsis-law/240162980

Gustafson-Barsis' law offers some hope stating that if the problem-size grows along with the number of parallel processors,
while the serial portion grows slowly or remains fixed, speedup grows as processors are added.
This might relax the concerns regarding Python as a language for parallel computing
since the serial portion is mostly fixed in Python when all the data-processing is hidden behind libraries like NumPy and SciPy.
Nevertheless, a larger problem-size demands more operational memory to be used for processing it, but memory is a limited resource.
Thus, even if problem-size is nearly unlimited, like for "Big Data", it has to be processed by chunks that fit into memory.
Overall, the limitted growth of the problem-size on a single node leaves us with the scalability defined by Amdahl's Law anyway.
As a result, the best strategy to efficiently load a multi-core system is still to avoid serial regions and synchronization.


1.1. Nested Parallelism
-----------------------
One way to avoid serial regions is to expose parallelism on all the possible levels of an application, for example,
by making outermost loops parallel or exploring functional, flow graph, or pipeline types of parallelism on the application level.
Python libraries that help to achieve this are Dask, Joblib, and the built-in :code:`multiprocessing` and :code:`concurrent.futures` modules.
On the innermost level, data-parallelism can be delivered by Python modules like [NumPy]_ and [SciPy]_.
These modules can be accelerated with an optimized math libraries like Intel |R| Math Kernel Library (Intel |R| [MKL]_),
which is multi-threaded internally using OpenMP (with default settings).

.. [MKL]    Intel(R) MKL, https://software.intel.com/intel-mkl
.. [Joblib] Joblib, http://pythonhosted.org/joblib/

When everything is combined together,
it results in a construction where code from one parallel region calls a function with another parallel region inside.
This is called *nested parallelism*.
It is an efficient way for hiding serial regions which are an inevitable part of regular NumPy/SciPy programs.


1.2. Issues of Oversubscription
-------------------------------
Nevertheless, the libraries named above do not coordinate the creation or pooling of threads, which may lead to *oversubscription*,
where there are much more active software threads than available hardware resources.
For sufficiently big machines, it can lead to sub-optimal execution due to frequent context switches, thread migration, broken cache-locality,
and finally to a load imbalance when some threads have finished their work but others are stuck, thus halting the overall progress.

For example, Intel OpenMP [*]_ runtime library (used by NumPy/SciPy)
may keep its threads active for some time to start subsequent parallel regions quickly.
Usually, this is a useful approach to reduce work distribution overhead.
However, with another active thread pool in the application,
it impairs performance because while OpenMP worker threads keep consuming CPU time in busy-waiting loops,
the other parallel work cannot start until OpenMP threads stop spinning or are preempted by the OS.

.. [*] Other names and brands may be claimed as the property of others.

Because overhead from linear oversubscription (e.g. 2x) is not always visible on the application level (especially for small systems),
it can be tolerated in many cases when the work for parallel regions is big enough.
However, in the worst case a program starts multiple parallel tasks and each of these tasks ends up executing an OpenMP parallel region.
This results in quadratic oversubscription (with default settings) which ruins multi-threaded performance on systems with a significant number of threads.
Within some big systems like Intel |R| Xeon Phi |TM|, it may not be even possible to create as many software threads as the number of hardware threads multiplied by itself due to insufficient resources.


1.3. Threading Composability
----------------------------
Altogether, the co-existing issues of multi-threaded components define *threading composability* of a program module or a component.
A perfectly composable component should be able to function efficiently among other such components without affecting their efficiency.
The first aspect of building a composable threading system is to avoid creation of an excessive number of software threads, preventing oversubscription.
That effectively means that a component and especially a parallel region cannot dictate how many threads it needs for execution (*mandatory parallelism*).
Instead, it should expose available parallelism to a run-time library, which provides contol over the number of threads or
which automatically coordinates tasks between components and parallel regions and map them onto available software threads (*optional parallelism*).


1.4. OMP_NUM_THREADS=1
----------------------
The most common way in the industry to solve the issues of oversubscription is to disable the nested level of parallelism
or carefully adjust it according to the number of application threads,
which is usually accomplished through setting environment variables controlling OpenMP run-time library
(example: :code:`OMP_NUM_THREADS=1`).
We are not discouraging from using this approach as it might be good enough to solve the problems in majority of use cases.
However, it has few deficiencies, which one might want to keep in mind on the way for better performance:

#. There might be not enough parallelism on the application level thus blindly disabling data parallelism can result in underutilization and so in slower execution.
#. Global settings provided once and for all cannot take into account different parts or stages of the application, which can have opposite requirements for better performance.
#. Setting right values might require from regular users deep enough understanding of the issue, architecture of the application, and the system it uses.
#. There are more settinggs to take into account like :code:`KMP_BLOCKTIME` and especially various thread affinity settings.
#. It is not limited solely to OpenMP. Many Python packages like Numba, PyDAAL, OpenCV, and Intel's optimized SciKit-Learn are based on Intel |R| TBB or custom threading runtime.


2. New approaches
-----------------
Our goal is to provide alternative solutions for composing multiple levels of parallelism across multiple threading libraries
with better or at least the same performance comparing to usual approaches
while simplifying interface and requiring less knowledge and decisions from end-users.
We prepared and evaluted few approaches which we now discuss in this paper.


2.1. Static Settings
--------------------
One of the common ways of making parallel code in Python is to employ process or threads *pools* (or *executors*)
provided through standard library.
These pools are also used by other Python libraries implementing parallel computations like Dask and Joblib.
We suggest to fix them in such a way that each pool worker being used to call nested parallel computation
can use only some particular number of processor cores.

For example, if we have an eight core CPU and want to create a pool of two workers,
we can limit the number of threads per pool worker to four.
When using a process pool, the best way to do so is to set thread affinity mask accordingly for each worker process
thus limitting any threads created within this process to operte only on specified processor cores.
In our example, the first process will use cores 0 through 3 and the second process will use cores 4 through 7.
Furthermore, since both OpenMP and Intel |R| TBB respect the incoming affinty mask during initialization,
they limit the number of threads per each process to four.
As a result, we have a simple way of sharing threads between pool workers without any oversubscription issues.

In case of a multi-threading pool being used for application-level parallelism, the idea is the same,
just instead of setting process affinity masks, we limit number of threads per each pool worker using threading runtime API.
For example, we use :code:`omp_set_num_threads()` function for specifying number of threads for OpenMP.
This approach is pretty much the same as when :code:`OMP_NUM_THREADS` environment variable is specified for entire application.
The difference is that we use knowledge of how many outermost workers are requested by application and
how much hardware parallelism is available on the machine,
then making the necessary calculation automatically and applying them for specific instance of pool.
It is more flexible approach for applications which might use pools of different sizes within the same run.

To implement this approach we have created Python module called *smp* (coming from static or symmetric multi-processing).
It works with both thread and process pools from :code:`multiprocessing` and :code:`concurrent.futures` modules
using *monkey patching* technique that allows to use this solution without any code modifications in user applications.
To run it, one should use one of the following commands:

.. code-block:: sh

    python -m smp script.py
    python -m smp -f <oversubscription_factor> script.py

Optional argument :code:`-f <oversubscription_factor>` sets oversubscription factor that will be used
to compute number of threads per pool worker.
By default it equals to 2, which means that in our example, 8 threads will be used per process.
Allowing this limited degree of oversubscription by default, we hope that for most applications benefits from load balancing
will overwheight the overheads incurred by it, as discussed in details in p3.5.
Though, for particular examples we show in this paper, the best performance is achieved with :code:`-f 1` specified on the command line.


2.2. Limiting Simultaneous OpenMP Parallel Regions
--------------------------------------------------
The second approach relies on modifications to the OpenMP runtime.
The basic idea, is to prevent oversubscription by not allowing concurrent parallel regions to collide,
which resembles in a sense "Global OpenMP Lock" as was suggested in [AMala16]_.
The actual implementation suggests two modes for scheduling parallel regions: *exclusive* and *counting*.
Exclusive mode implements exclusive lock that is acquired before running a parallel region and releases it after the work is done.
Counting mode implements mechanism equivalent to semaphore, which allows multiple parallel regions with small number of threads as long
as the total number of threads does not cross the limit.
When the limit is exceeded, it blocks in a similar way as the lock in exclusive mode until requested resources become available.
This idea is easily extended to the case of multiple processes using Inter Process Coordination (IPC) mechanisms such as
system-wide semaphore.

The exclusive mode approach is implemented in the Intel |R| OpenMP* runtime library being released
as part of Intel |R| Distrubution for Python 2018 [#]_ as an experimental preview feature.
To enable this mode, :code:`KMP_COMPOSABILITY` environment variable should be set, for example:

.. [#] It was also introduced on Anaconda cloud along with the version 2017.0.3 in limited, undocumented form.
.. code-block:: sh

    env KMP_COMPOSABILITY=mode=exclusive python app.py

This enables each OpenMP parallel region to run exclusively, eliminating most of oversubscription issues.

With the composability mode on, the multi-processing coordination is enabled automatically on the first usage.
In the case, each process will have its own pool of OpenMP worker threads.
While these threads will be coordinated across the processes preventing oversubscription,
the many co-existing threads may still cause resource exhaustion issue.


2.3. Coordinated Task Scheduler with Intel |R| TBB
--------------------------------------------------
The last approach has been initially introduced in our previous paper [AMala16]_.
It is based upon using Intel |R| TBB as a sigle task scheduler for coordinating parallelism for all the Python pools and modules.
Its work stealing task scheduler is used to map tasks onto the set of TBB worker threads
while monkey-patching technique is used to redirect Python's :code:`ThreadPool` on top of TBB tasks.
That allows to dynamically balance the load across multiple tasks from multiple modules but has been limited to multi-threading only.

In this paper we extended this approach by introducing the multi-processing coordination layer for Intel |R| TBB.
As shown in figure :ref:`components`, different modules, that can be used in an applicaion,
work on top of the shared Intel |R| TBB pool, which is coordinated across multiple processes.

.. figure:: components.png

   Intel |R| TBB provides a common runtime for Python modules and coordinates threads across processes. :label:`components`

This works as in the following way.
We create a number of processes not to exceed the number of hardware threads.
In each separate process, there is a thread pool.
Before starting any thread in any pool, one should acquire a system-wide semaphore with maximum value equal to the number of CPU hardware threads.
To acquire the semaphore, a greedy algorithm is used that may lead to a situation when some processes do not have pool workers.
However, each process uses at least one master thread to perform computations.
Thus, the total number of working threads for all running processes doesn't exceed twice the number of CPU hardware threads in the worst case
(instead of the quadratic oversubscripton case one could face with).
To make this solution truly dynamic, an additional worker thread is added to each Intel |R| TBB thread pool,
which allows processes to acquire threads that become free on other processes thereby eliminating CPU underutilization.

However, from the point of view of simultaneously existing threads, we still may have resource exhaustion issues.
Since we can't just move a thread from one process to another, it may happen that there are too many threads alive at the same time.
To eliminate such issues, we have implemented an algorithm that disposes of unused threads when a shortage or resources is detected.

This solution is different from the approach that uses an OpenMP runtime with global lock,
it allows the processing of several parallel regions simultaneously and provides the ability to do work balancing on the fly.
Even a more flexible locking mechanism in OpenMP would need to wait for all the requested threads to become available while Intel |R| TBB allows threads joining when the work is ongoing.

3. Evaluation
-------------
For our experiments, we need Intel |R| Distribution for Python [IntelPy]_ to be installed along with the [Dask]_ library which simplifies parallelism with Python.

.. [IntelPy] Intel(R) Distribution for Python, https://software.intel.com/python-distribution

.. code-block:: sh

    # install Intel(R) Distribution for Python
    <path to installer of the Distribution>/install.sh
    # activate in environment
    source <path to the Distribution>/bin/activate.sh
    # install Dask
    conda install dask


3.1. Balanced QR Decomposition with Dask
----------------------------------------
The code below is a simple program using Dask that validates QR decomposition by multiplying computed components and comparing the result against the original input.

.. code-block:: python
    :linenos:

    import time, dask, dask.array as da
    x = da.random.random((440000, 1000),
                         chunks=(10000, 1000))
    for i in range(3):
        t0 = time.time()
        q, r = da.linalg.qr(x)
        test = da.all(da.isclose(x, q.dot(r)))
        test.compute()
        print(time.time() - t0)

Dask splits the array into 44 chunks and processes them in parallel using multiple threads.
However, each Dask task executes the same NumPy matrix operations which are accelerated using Intel |R| MKL under the hood and thus multi-threaded by default.
This combination results in nested parallelism, i.e. when one parallel component calls another component, which is also threaded.
For this example, we will talk mostly about the multi-threading case, but according to our investigations,
all conclusions that will be shown are applicable for the multi-processing case as well.

Here is an example of running the benchmark program in five different modes:

.. code-block:: sh
    :linenos:

    python bench.py             # Default OpenMP mode
    KMP_BLOCKTIME=0 OMP_NUM_THREADS=1 \
        python bench.py         # Tunned OpenMP mode
    python -m SMP -f 1 bench.py # OpenMP + SMP mode
    KMP_COMPOSABILITY=mode=exclusive \
        python bench.py         # Composable OpenMP mode
    python -m TBB bench.py      # Composable TBB mode

.. figure:: dask_static.png

   Execution times for balanced QR decomposition workload. :label:`sdask`

Figure :ref:`sdask` shows performance results acquired on a 44-core (88-thread) machine with 128 GB memory.
The results presented here were acquired with cpython v3.5.2; however, there is no significant performance difference with cpython v2.7.12.
By default, Dask will process a chunk in a separate thread so there will be 44 threads on the top level
(note that by default Dask will create a thread pool with 88 workers but only half of them will be really used since there are only 44 chunks).
Each chunk will be computed in parallel with 44 OpenMP workers.
Thus, there will be 1936 threads vying for 44 cores, resulting in oversubscripton and poor performance.

An simple way to improve performance is to tune the OpenMP runtime using environment variables.
First, we need to limit total number of threads.
We will set 1x oversubscription instead of quadratic as our target.
Since we work on an 88-thread machine, we should set number of threads per parallel region to 1 ((88 CPU threads / 88 workers in thread pool) * 1x over-subscription).
We also noticed that reducing period of time after which Intel OpenMP worker threads will go to sleep, helps to improve performance in such workloads with oversubscription
(this works best for the multi-processing case but helps for multi-threading as well).
We achieve this by setting KMP_BLOCKTIME to zero.
These simple optimizations allows reduce the computational time by more than 3x.

The third mode with *SMP.py* module in fact does the same optimizations but automatically, and shows the same level of performance as the second one.
Moreover, it is more flexible and allows to work carefully with several thread/process pools in the application scope even if they have different sizes.
Thus, it is a good alternative to manual OpenMP tunning.

The fourth and fifth modes represents our dynamic OpenMP- and Intel |R| TBB-based approaches.
Both modes improve the default result, but exclusive execution with OpenMP gave us the fastest results.
As described above, the OpenMP-based solution allows processes chunks one by one without any oversubscription, since each separate chunk can utilize the whole CPU.
In contrast, the work stealing task scheduler from Intel |R| TBB is truly dynamic and tries to use a single thread pool to process all given tasks simultaneoulsy.
As a result, it has worse cache utilization, and higher overhead for work balancing.

.. [#] For more complete information about compiler optimizations, see our Optimization Notice [OptNote]_


3.2. Balanced Eignevalues Search with NumPy
-------------------------------------------
The code below performs an algorithm of eigenvalues and right eigenvectors search in a square matrix using Numpy:

.. code-block:: python
    :linenos:

    import time, numpy as np
    from multiprocessing.pool import ThreadPool
    x = np.random.random((256, 256))
    p = ThreadPool(88)
    for j in range(3):
        t0 = time.time()
        p.map(np.linalg.eig, [x for i in range(1024)])
        print(time.time() - t0)

In this example we process several matricies from an array in parallel using :code:`ThreadPool`
while each separate matrix is computed using OpenMP parallel regions from Intel |R| MKL.
As a result, simillary to QR decomposition benchmark we've faced with quadratic oversubscription here.
But this code has a distinctive feature, in spite of parallel execution of eigenvalues search algorithm,
it cannot fully utilize all available CPU cores.
That is why an additional level of parallelizm we used here may significantly improve overall benchmark performance.

.. figure:: numpy_static.png

   Execution time for balanced eignevalues search workload. :label:`snumpy`

Figure :ref:`snumpy` shows benchmark execution time in the same five modes as we used for QR decomposition.
As previously the best choice here is to limit number of threads statically either using manual settings or *smp* module.
Such approach allows to obtain more than 7x speed-up.
But this time Intel |R| TBB based approach looks much better than serialization of OpenMP parallel regions.
And the reason is low CPU utilization in each separate chunk.
In fact exclusive OpenMP mode leads to serial matrix processing, one by one, so significant part of the CPU stays unsed.
As a result, execution time in this case becomes even larger than by default.

3.3. Unbalanced QR Decomposition with Dask
------------------------------------------
In previous sections we looked into balanced workloads where amount of work per thread on top level is near the same.
It's rather expected that for such cases the best solution is static one.
But what if one need to deal with dynamic workloads where amount of work per thread or process may vary?
To investigate such cases we've developed unbalanced versions of our static benchmarks.
An idea we used is the following. There is a single thread pool with 44 workers.
But this time we will perform computations in three stages.
The first stage will use only one thread from the pool which is able to fully utilize the whole CPU.
During the second stage half of top level threads will be used (22 in our examples).
And on the third stage the whole pool will be employed (44 threads).

The code above demonstrates unbalanced version of QR decomposition workload:

.. code-block:: python
    :linenos:

    import time, dask, dask.array as da
    def qr(x):
        t0 = time.time()
        q, r = da.linalg.qr(x)
        test = da.all(da.isclose(x, q.dot(r)))
        test.compute(num_workers=44)
        print(time.time() - t0)
    x01 = da.random.random((440000, 1000),
                           chunks=(440000, 1000))
    x22 = da.random.random((440000, 1000),
                           chunks=(20000, 1000))
    x44 = da.random.random((440000, 1000),
                           chunks=(10000, 1000))
    qr(x01)
    qr(x22)
    qr(x44)

To run this benchmark, we used the four modes: default, OpenMP with *SMP.py*, composable OpenMP and composable Intel |R| TBB.
We don't show results for OpenMP with manual optimizations since they are very close to the results for "OMP + SMP" mode.

.. figure:: dask_dynamic.png

   Execution times for unbalanced QR decomposition workload. :label:`ddask`

Figure :ref:`ddask` demonstrates execution time for all four modes.
The first observation here is that static *SMP.py* approach doesn't achieve good performance with imbalanced workloads.
Since we have a single thread pool with a fixed number of workers and we don't know which of these workers will be used or how intensively,
it is difficult to set an appropriate number of threads statically.
Thus, we limit the number of threads per parallel region based on the size of the pool only.
As a result, in the first stage just a few threads are really used which leads to performance degradation.
On the other hand, the second and third stages work well.
However, overall we have a mediocre result.

The work stealing scheduler from Intel |R| TBB works better than the default version,
but due to redundant work balancing in this particular case it has significant overhead and not the best performance result.

The best execution time is obtained using exclusive OpenMP mode.
Since there is sufficient work to do in each parallel region,
allowing ech chunk to be calculated one after the other avoids oversubscription and gets the best performance - nearly a 34% speed-up.


3.4. Unbalanced Eigenvalues Search with NumPy
---------------------------------------------
The second dynamic exapmle we'd like to discuss is based on eigenvalues search algorithm from NumPy:

.. code-block:: python
    :linenos:

    import time, numpy as np
    from multiprocessing.pool import ThreadPool
    from functools import partial

    x = np.random.random((256, 256))
    y = np.random.random((8192, 8192))
    p = ThreadPool

    t0 = time.time()
    mmul = partial(np.matmul, y)
    p.map(mmul, [y for i in range(6)], 6)
    print(time.time() - t0)

    t0 = time.time()
    p.map(np.linalg.eig, [x for i in range(1408)], 64)
    print(time.time() - t0)

    t0 = time.time()
    p.map(np.linalg.eig, [x for i in range(1408)], 32)
    print(time.time() - t0)

In this workload we have same three stages. The second and the third stage computes eignevalues and the first one performs matrix multiplication.
The reason of why we don't use eignevalues search for the first stage as well is that it cannot fully load CPU as we planned.

.. figure:: numpy_dynamic.png

   Execution time for unbalanced eignevalues search workload. :label:`dnumpy`

From figure :ref:`dnumpy` one can see that the best solution for this workload is work stealing scheduler from Intel |R| TBB which allows to reduce execution time on 35%.
*SMP.py* module works even slower than default version due to the same issues as described for unbalanced QR decomposition example.
And as for the mode with serialization of OpenMP parallel regions, it works significantly slower than default version since there is no enough work for each parallel region that leads to CPU underutilization.


3.5. Acceptable Level of Oversubscription
-----------------------------------------
We did some experiments to determine what level of oversubscription has acceptable performance.
We started with various sizes for the top level thread or process pool,
and ran our balanced eigenvalues search workload with different pool sizes from 1 to 88 (since our machine has 88 threads).

.. figure:: scalability_multithreading.png

   Multi-threading scalability of eigenvalues seach workload. :label:`smt`

Figure :ref:`smt` shows the scalability results for the multi-threading case.
Two modes are compared: default and OpenMP with *SMP.py* as the best approach for this benchmark.
As one can see, the difference in execution time between these two methods starts from 8 threads in top level pool and becomes larger as the pool size increases.

.. figure:: scalability_multiprocessing.png

   Multi-processing scalability of eigenvalues seach workload. :label:`smp`

The multi-processing scalability results are shown in figure :ref:`smp`.
They can be obtained from the same eigenvalues search workload by replacing :code:`ThreadPool` to :code:`Pool`.
The results are very similar to the multi-threading case: oversubscription effects become visible starting from 8 processes at the top level of parallelization.


4. Solutions Applicability
--------------------------
In summary, all three suggested approaches to avoid oversubscription are valuable and can obtain significant performance increases for both multi-threading and multi-processing cases.
Moreover, the approaches complement each other and have their own fields of applicability.

.. figure:: recommendation_table.png

   How to choose the best approach to deal with oversubscription issues. :label:`rtable`

The *SMP.py* module works perfectly for balanced workloads where each pool's workers have the same load.
Compared with manual tunning of OpenMP options, it is more stable,
since it can work with pools of different sizes within the scope of a single application without performance degradation.
It also works with Intel |R| TBB.

The exclusive mode for the OpenMP runtime works best with unbalanced benchmarks for the cases where there is enough work for each innermost parallel region.

The dynamic work stealing scheduler from Intel |R| TBB obtains the best performance
when innermost parallel regions cannot fully utilize the whole CPU and have varying amounts of work to do.

To summarize our conclusions, we've prepared a table to help choose which approach will work best for which case (see figure :ref:`rtable`).


5. Limitations and Future Work
------------------------------
*smp* module currently works only based on the pool size and does not take into account its real usage.
We think it can be improved in future to trace task scheduling pool events and so to become more flexible.
The *smp* module works only for Linux currently.

The OpenMP global lock solution works fine with parallel regions with high CPU utilization,
but has significant performance gap in other cases, so can be improved.
For example, in our ongoing work, we use a semaphore instead of a mutex to allow multiple parallel regions to run at the same time and thus impove overall CPU utilization.

Intel |R| TBB does not work well for blocking I/O operations because it limits the number of active threads.
It is applicable only for tasks, which do not block in the operating system.
If your program uses blocking I/O, please consider using asynchronous I/O that blocks only one thread for the event loop and so prevents other threads from being blocked.

The Python module for Intel |R| TBB is in an experimental stage and might be insufficiently optimized and verified with different use cases.
In particular, it does not yet use the master thread efficiently as a regular TBB program is supposed to do.
This reduces performance for small workloads and on systems with small numbers of hardware threads.

The TBB-based implementation of Intel |R| MKL threading layer is yet in its infancy and is therefore suboptimal.
However, all these problems can be eliminated as more users will become interested in solving their composability issues and Intel |R| MKL and the TBB module are further developed.

.. [OptNote] https://software.intel.com/en-us/articles/optimization-notice
.. [#] For more complete information about compiler optimizations, see our Optimization Notice [OptNote]_


6. Conclusion
-------------
This paper starts by substantiating the necessity of broader usage of nested parallelism for multi-core systems.
Then, it defines threading composability and discusses the issues of Python programs and libraries which use nested parallelism with multi-core systems, such as GIL and oversubscription.
These issues affect the performance of Python programs that use libraries like NumPy, SciPy, Dask, and Numba.

Three approaches are described as potential solutions.
The first one is to statically limit the number of threads created inside each worker pool.
The second one is limiting simultaneous OpenMP parallel regions.
The third one is to use a common threading runtime library such as Intel |R| TBB,
which limits the number of threads in order to prevent oversubscription and coordinates parallel execution of independent program modules.

The examples referred to in the paper show promising results of achieving the best performance using nested parallelism and threading composability.
In particular, balanced QR decomposition and eigenvalues search examples are 2.8x and 7x faster compared to the baseline implementations.
Imbalanced versions of these benchmarks are 34-35% faster than the baseline.

These improvements were achieved with all different approaches, demonstrating that the three solutions are valuable and complement each other.
We've compared suggested approaches and provided recommendations of when it makes sense to employ each of them.

All described solutions are available as open source software,
and the Intel |R| Distribution for Python accelerated with Intel |R| MKL is available for free as a stand-alone package [IntelPy]_ and on anaconda.org/intel channel.


7. References
-------------

.. figure:: opt-notice-en_080411.png
   :figclass: b
.. |C| unicode:: 0xA9 .. copyright sign
   :ltrim:
.. |R| unicode:: 0xAE .. registered sign
   :ltrim:
.. |TM| unicode:: 0x2122 .. trade mark sign
   :ltrim:
