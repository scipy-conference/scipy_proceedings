:author: Andrew Cron
:email: ajc40@stat.duke.edu
:institution: Duke University

:author: Wes McKinney
:email: wesmckinn@gmail.com
:institution: Duke University

---------------------------------------------------------
gpustats: GPU Library for Statistical Computing in Python
---------------------------------------------------------

.. class:: abstract

   In this talk we will discuss **gpustats**, a new Python library for assisting
   in "big data" statistical computing applications, particularly Monte
   Carlo-based inference algorithms. The library provides a general code
   generation / metaprogramming framework for easily implementing discrete and
   continuous probability density functions and random variable samplers. These
   functions can be utilized to achieve more than 100x speedup over their CPU
   equivalents. We will demonstrate their use in an Bayesian MCMC application
   and discuss avenues for future work.

.. class:: keywords

   GPU, statistical inference, computation, statistics, Monte Carlo


Introduction
------------

Due to the high theoretical computational power and low cost of graphical
processing units (GPUs), researchers and scientists in a wide variety of fields
have become interested in applying them within their problem domains. However,
an major catalyst for making GPUs widely accessible was the development of the
general purpose GPU computing frameworks, [CUDA]_ and [OpenCL]_, which enable
the user to implement general numerical algorithms in a simple extension of the
C language to run on the GPU. In this paper, we will restrict our technical
discussion to the CUDA / NVIDIA architecture, while later commenting on CUDA
versus OpenCL.

.. The basic process for GPU computing involves
.. copying data to the (GPU) device memory, performing some computation (written
.. with the CUDA or OpenCL APIs), then copying results back to the main (CPU)
.. memory space.

In python, [PyCUDA]_ and [PyOpenCL]_ provide a nearly one-to-one wrapper to
their C equivalents. For this paper, we will assume we are in the [PyCUDA]_
case. However, the discussion pertains to [PyOpenCL]_ as well.  The kernels are
still written in C, and are passed to the libraries as strings to be compiled on
the fly. Most of our kernels have the same format: (1) copy some data from
global memory to local memory, (2) do some computation, (3) write results to
main memory. Since the first and third steps are usually identical in our
applications, we use meta-programming to generate kernel strings for
computations that have the same memory structure and only the computation
changes.

GPUs are having a huge impact in many areas of statistics including : examples :
our goal is to give a fundamental python library that will allow researchers to
do statistical analysis on large data while knowing minimal GPU programming.

Development Challenges in GPU Computing
---------------------------------------

While a CPU may have 4 or 8 cores, a latest generation GPU may have 256, 512, or
even more computational cores. However, the GPU memory architecture is highly
specialized to so-called single instruction multiple data (SIMD) problems. This
generally limits the usefulness of GPUs to highly parallelizable data processing
applications. The developer writes a small function, known as a *kernel*, to
process a unit of data. The kernel function is then executed once for each unit
of data, coordinated by hundreds of threads across the GPU.

The GPU has a large single *global* memory store (typically 512MB to 4GB) from
which data sets can be written and read by the CPU. However, each group, or
*block*, of threads are assigned a small piece (typically 16K to 64K) of ultra
low-latency *shared* cache memory which is orders of magnitude faster than the
global memory. Therefore, the main challenge for the developer, outside of
writing the kernel function, is structuring the computation to optimally utilize
each thread block's shared memory and minimizing reads from the global
memory. Careful coordination of the threads is required to transfer memory
efficiently from global to shared. We will not get into the low-level details of
this process but instead refer the interested reader to the CUDA API guide
([NvidiaGuide]_).

As a larger computation is divided up into a *grid* of thread blocks, a typical
CUDA kernel will therefore take the following structure:

* Coordinate threads within a block to transfer relevant data for block from
  global to shared memory
* Perform computation using (fast) shared memory
* Coordinate threads to transfer results back into global memory

.. figure:: diag1.png

   Mock-up of GPU computing architecture :label:`egfig`

.. This allows for extremely low memory latency making GPU programming extremely
.. attractive in large data contexts ([NvidiaGuide]_).

Computational Challenges in Likelihood-based Statistical Inference
------------------------------------------------------------------

In most standard and Bayesian inference, a distribution is assumed for each
realization of the data given some parameters to estimate. The central
consequence of this assumption is the creation of a log likelihood (the log of
the joint probability distribution of the data). Most numerical algorithms for
fitting these likelihood-based models involves evaluating this function at every
iteration. Therefore, as the size of the observed data grows, computational
expense grows *as least* linearly in the number of data points. The linear case
is when the data is assumed to be observed independently in light of the
parameters you wish to estimate. In this case, the likelihood becomes a product
and the log likelihood becomes a sum with each term consisting of a function of
one data point.  This case is very naturally decomposed into a fine grain
parallel problem with a high data throughput twist. Eg. the number of
computations is fairly small compared to the amount of memory transactions. This
becomes a very natural setting for GPUs, and it is quite easy for GPUs to
perform even better than large compute clusters because of the very low memory
latency in GPUs. Suchard et al. studied these advantages in the Bayesian mixture
model setting and found very promising results (100x speedup) on graphics cards
that are now 2 years old ([JCGS]_).

Challenges of GPU Computing in Statistical Inference
----------------------------------------------------

In the **gpustats** package, we have three primary goals that we will address
here. First, we hide the ubiquitous boilerplate code common to all GPU
programs. We naturally want to achieve respectable performance by taking full
advantage of the GPU's execution and memory architecture. Finally, we minimize
the effort in developing new **gpustats** functions by using meta-programming to
handle the 90% identical code across tasks.

In any GPU based application, there are some necessary functions that must be
called. The most prominent are initialization routines, and memory transfers
between GPU memory and main memory before and after the parallel GPU
code. [PyCUDA]_ handles this very nicely with the `gpuarray` object which can
take a [NumPy]_ array and handle the memory transfers behind the scenes. As for
initialization, there are several tuning parameters that need to be considered
before launching a kernel. Again, [PyCUDA]_ is capable of querying the GPU and
kernel for all the necessary information to perform optimization. We have
implemented optimization routines for our general cases which further ease the
development of new GPU code in the genre.

To maximize the performance of our code, we need to fully utilize and appeal to
the memory structure on the GPU. It is a hierarchical structure with three
levels: global, shared, and local. Global memory is the standard RAM on the card
and is usually a few gigabytes. Threads do not read from this memory, but rather
the multiprocessor makes transactions for all the threads together. To take
advantage of this structure, threads must read from memory in a coalesced
manner. Threads usually read data from global memory to pplocal memory. Groups
of threads cooperatively have access to the same pool of global memory which is
usually 16 KB. Since this is small, the general kernel structure tends to be to
read a little piece of data from global memory, do computation, write the
results, and repeat.  Furthermore, each thread has a very small local memory
which is just used for storing current values in computations.

In **gpustats** we are usually doing one of two computations: evaluating many
points on a distribution or generating many values from a distribution.  In the
both cases, the structure of the input data doesn't change across
distributions. In the first case, the input is a large pile of data and a set of
parameters. In the second, the input is a large set of uniform random numbers
and a set of parameters. Therefore, not only is most of the boilerplate code the
same, but most of the kernel is the same. In fact, the only part of the kernel
that changes is the actually computation. This implies a straightforward
meta-programming approach.


References
----------

.. [CUDA] NVIDIA Corporation. CUDA GPU computing framework
      http://www.nvidia.com/object/cuda_home_new.html

.. [OpenCL] Kronos Group. OpenCL parallel programming framework
      http://www.khronos.org/opencl/

.. [JCGS] M. Suchard, Q. Wang, C. Chan, J. Frelinger, A. Cron and M. West.
   	  *Understanding GPU programming for statistical computation: Studies
	  in massively parallel massive mixtures.* Journal of Computational
	  and Graphical Statistics 19 (2010): 419-438
	  http://pubs.amstat.org/doi/abs/10.1198/jcgs.2010.10016

.. [NvidiaGuide] NVIDIA Corporation. *Nvidia CUDA: Programming Guide.* (2010),
   		 http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf

.. [PyMC] C. Fonnesbeck, A. Patil, D. Huard,
          *PyMC: Markov Chain Monte Carlo for Python*,
          http://code.google.com/p/pymc/

.. [NumPy] T. Oliphant,
           http://numpy.scipy.org

.. [SciPy] E. Jones, T. Oliphant, P. Peterson,
           http://scipy.org

.. [PyCUDA] A. Klockner,
   	    http://mathema.tician.de/software/pycuda

.. [PyOpenCL] A. Klockner,
   	      http://mathema.tician.de/software/pyopencl
