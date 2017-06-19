Scikit-learn is well know library that provides a log of algorithms for many areas of machine learning.
Having limited developer resources this project prefers universal solutions and proven algorithms.
For performance improvement scikit-learn uses Cython and underling (through SciPy and Numpy) BLAS/LAPACK libraries.
OpenBLAS and MKL uses threaded based parallelism to utilize multicores of modern CPUs.
Unfortunately  BLAS/LAPACK’s functions are too low level primitives and their usage is often not very efficient comparing to possible high-level parallelism.
For high-level parallelism scikit-learn uses multiprocessing approach that is not very efficient from technical point of view.
On the other hand Intel provides Intel |R| Data Analytics Acceleration Library (Intel(R) DAAL) that helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes of computation.
It is originally written in C++ and provides Java and Python bindings.
In spite of the fact that DAAL is heavily optimized for all Intel Architectures including Xeon Phi it’s very questionable how to use DAAL binding from Python.
DAAL bindings for python are generated automatically and reflects original C++ API very close that makes its usage quite complicated because of not native for python programmers idioms and chary documentation.

To mix the power of well optimized for HW native code with familiar ML API Intel Python Distribuition started efforts on scikit-learn optimization.
Therefore, beginning from 2017 U2 IDP includes scikit-learn with daal4sklearn module.
Specifically, Update 2 optimizes Principal Component Analysis (PCA), Linear and Ridge Regressions, Correlation and Cosine Distances, and K-Means. Speedups may range from 1.5x to 160x.

There are no exact matching between sklearn and DAAL API and they aren’t fully compatible for all inputs so for the cases when daal4sklearn detects incompatibility it fallbacks to original sklearn’s implementation.

Daal4sklearn is enabled by default but provides simple API that allows disabling its functionality:

.. code-block:: python

        from sklearn.daal4sklearn import dispatcher
        dispatcher.disable()
        dispatcher.enable()

We prepared several benchmarks to demonstrate performance that can be achieved with DAAL.

.. code-block:: python

        from __future__ import print_function
        import numpy as np
        import timeit
        from numpy.random import rand
        from sklearn.cluster import KMeans

        import argparse
        argParser = argparse.ArgumentParser(prog="pairwise_distances.py",
                                            description="sklearn pairwise_distances benchmark",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        argParser.add_argument('-i', '--iteration', default='10', help="iteration", type=int)
        argParser.add_argument('-p', '--proc', default=-1, help="n_jobs for algorithm", type=int)
        args = argParser.parse_args()

        REP = args.iteration 

        try:
            from daal.services import Environment
            nThreadsInit = Environment.getInstance().getNumberOfThreads()
            Environment.getInstance().setNumberOfThreads(args.proc)
        except:
            pass

        def st_time(func):
            def st_func(*args, **keyArgs):
                times = []
                for n in range(REP):
                    t1 = timeit.default_timer()
                    r = func(*args, **keyArgs)
                    t2 = timeit.default_timer()
                    times.append(t2-t1)
                print (min(times), end='')
                return r
            return st_func

        problem_sizes = [
                (10000,   2),
                (10000,   25),
                (10000,   50),
                (50000,   2),
                (50000,   25),
                (50000,   50),
                (100000,  2),
                (100000,  25),
                (100000,  50)]

        X={}
        for p, n in problem_sizes:
            X[(p,n)] = rand(p,n)


        kmeans = KMeans(n_clusters=10, n_jobs=args.proc)
        @st_time
        def train(X):
            kmeans.fit(X)

        for p, n in problem_sizes:
            print (p,n, end=' ')
            X_local = X[(p,n)]
            train(X_local)
            print('')

Using all 32 cores of Xeon E5-2698v3 IDP’s KMeans can be faster more than 50 times comparing with python available on Ubuntu 14.04.
P below means amount of CPU cores used.

.. table:: 

   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | rows   | cols | IDP,s P=1 | IDP,s P=32 | System,s P=1 | System,s P=32 | Vs System,P=1 | Vs System,P=32 | 
   +========+======+===========+============+==============+===============+===============+================+
   | 10000  | 2    | 0.01      | 0.01       | 0.38         | 0.27          | 28.55         | 36.52          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 10000  | 25   | 0.05      | 0.01       | 1.46         | 0.57          | 27.59         | 48.22          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 10000  | 50   | 0.09      | 0.02       | 2.21         | 0.87          | 23.83         | 40.76          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 50000  | 2    | 0.08      | 0.01       | 1.62         | 0.57          | 20.57         | 47.43          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 50000  | 25   | 0.67      | 0.07       | 14.43        | 2.79          | 21.47         | 38.69          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 50000  | 50   | 1.05      | 0.10       | 24.04        | 4.00          | 22.89         | 38.52          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 100000 | 2    | 0.15      | 0.02       | 3.33         | 0.87          | 22.30         | 56.72          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 100000 | 25   | 1.34      | 0.11       | 33.27        | 5.53          | 24.75         | 49.07          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+
   | 100000 | 50   | 2.21      | 0.17       | 63.30        | 8.36          | 28.65         | 47.95          | 
   +--------+------+-----------+------------+--------------+---------------+---------------+----------------+

We compared the similar runs for other algorithms and normalized results by results obtained with DAAL in C++ without python to estimate overhead from python wrapping.


.. figure:: sklearn/sklearn_perf.jpg


You can find some benchmarks [sklearn_benches]_


.. [sklearn_benches] https://github.com/dvnagorny/sklearn_benchs

