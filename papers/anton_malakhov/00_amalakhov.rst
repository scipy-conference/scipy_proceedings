:author: Anton Malakhov
:email: Anton.Malakhov@intel.com
:institution: Intel Corporation

-----------------------------------------------
COMPOSABLE MULTI-THREADING FOR PYTHON LIBRARIES
-----------------------------------------------

.. class:: abstract

   Python is popular among numeric communities that value it for easy to use
   number crunching modules like Numpy/Scipy, Dask, Numba, and many others.
   These modules often use multi-threading for efficient parallelism (on a node).
   But being used together in one application, their thread pools can interfere
   with each other leading to overhead and inefficiency. We'll show how to fix this problem.

.. class:: keywords
   Multithreading, Oversubscription, Parallel Computations, Carallelism, Threading, Dask, Joblib, Numpy, Scipy
   
Blog
----
In the Beta release of Intel® Distribution for Python* I am proud to introduce something new and unusual for the Python world. It is an experimental module which unlocks additional performance for multi-threaded Python programs by enabling threading composability between two or more thread-enabled libraries.

Threading composability can accelerate programs by avoiding inefficient threads allocation (called oversubscription) when there are more software threads than available hardware resources.

The biggest improvement is achieved when a task pool like the ThreadPool from standard library or libraries like Dask or Joblib (used in multi-threading mode) execute tasks calling compute-intensive functions of Numpy/Scipy/PyDAAL which in turn are parallelized using Intel® MKL or/and Intel® Threading Building Blocks (Intel® TBB).

The module implements Pool class with the standard interface using Intel® TBB which can be used to replace Python’s ThreadPool. Thanks to the monkey-patching technique implemented in class Monkey, no source code change is needed in order to unlock additional speedups.

Let’s try it!

Assuming you have installed Intel® Distribution for Python, we need to install Dask library which makes parallelism very simple for Python:

source <path to Intel® Distribution for Python*>/bin/pythonvars.sh
conda install dask

Now, let’s write a simple program in bench.py which exploits nested parallelism and prints time spent for the computation, like the following:

.. code-block:: python

    import dask, time
    import dask.array as da
     
    t0 = time.time()
     
    x = da.random.random((10000, 10000), chunks=(4096, 4096))
    x.dot(x.T).sum().compute()
     
    print(time.time() - t0)

Here, Dask splits the array into chunks and processes them in parallel using multiple threads. But each Dask task executes expensive matrix multiplication (`dot’) which is accelerated using Intel® MKL under the hood and thus multi-threaded by itself. It results in nested parallelism which is handled best with Intel® TBB.

To run it as is (baseline)::

    python bench.py

And to unlock additional performance::

    python -m TBB bench.py

That's it! Depending on machine configuration, you can get about 20%-50% reduction of the compute time for this particular example or even more if there is a background activity on the machine.

Disclaimers: TBB module does not work well for blocking I/O operations, it is applicable only for tasks which do not block in the operating system. This version of TBB module is experimental and might be not sufficiently optimized and verified with different use-cases.

For additional details on how to use the TBB module, please refer to built-in documentation, e.g. run `pydoc TBB`.

This module is available in sources as preview feature of Intel TBB 4.4 Update 5 release.

We’ll greatly appreciate your feedback! Please get back to me, especially if you are interested enough to use it in your production/every-day environment. And if you are a lucky attendee of PyCon 2016 in Portland, Oregon, I'll be glad to meet you at my poster presentation "Composable Multi-threading for Python Libraries".

