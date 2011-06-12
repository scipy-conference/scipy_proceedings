:author: Andrew Cron
:email: ajc40@stat.duke.edu
:institution: Duke University

-----------------------------------------------
gpustats: GPU Library for Statistical Computing
-----------------------------------------------

.. class:: abstract 
   In this talk we will discuss `gpustats`, a new
   PyCUDA-based library for assisting in "big data" statistical
   computing applications, particularly Monte Carlo-based inference
   algorithms. The library provides a general code generation /
   metaprogramming framework for easily implementing discrete and
   continuous probability density functions and samplers. These
   functions can be utilized to achieve more than 100x speedup over
   their CPU equivalents. We will demonstrate their use in an Bayesian
   MCMC application and discuss avenues for future work.

.. class:: keywords

   GPU, statistics, computation, statistics


Introduction
------------

Due to the high theoretical computational power and low cost of
graphical processing units (GPUs), researchers and computer scientists
in a wide variety of fields are becoming more interested in GPUs. The
catalyst was the development of a simple programming framework as an
extension of C. There are two primary languages for GPU programming:
CUDA and openCL.  The basic programming tasks are to copy memory to
the device, execute some parallel program(s), copy the results to the
main memory.

The GPU programming framework is single instruction multiple data
(SIMD). The developer writes a small kernel function that is called by
hundreds of threads in parallel. For optimal performance, the only
variant in the thread execution should be the input and output. Memory
transfers are handled by separate hardware on the GPU, and the threads
are executed by the processor in groups of 16 or 32 in lock step. The
groups of threads may run in any order and will stop and start based
on memory transfers. The threads do have limited local memory which is
extremely fast, so most kernels copy data from the global memory to
local memory then do computation. This allows for extremely low memory
latency making GPU programming extremely attractive in large data
contexts ([NvidiaGuide]_).

In python, [PyCUDA]_ and [PyOpenCL]_ provide a nearly one-to-one
wrapper to their C equivalents. For this paper, we will assume we are
in the [PyCUDA]_ case. However, the discussion pertains to [PyOpenCL]_
as well.  The kernels are still written in C, and are passed to the
libraries as strings to be compiled on the fly. Most of our kernels
have the same format: (1) copy some data from global memory to local
memory, (2) do some computation, (3) write results to main
memory. Since the first and third steps are usually identical in our
applications, we use meta-programming to generate kernel strings for
computations that have the same memory structure and only the
computation changes.

GPUs are having a huge impact in many areas of statistics including :
examples : our goal is to give a fundamental python library that will
allow researchers to do statistical analysis on large data while
knowing minimal GPU programming.

Computational Challenges in Likelihood-based Statistical Inference
------------------------------------------------------------------

In most standard and Bayesian inference, a distribution is assumed for
each realization of the data given some parameters to estimate. The
central consequence of this assumption is the creation of a log
likelihood (the log of the joint probability distribution of the
data). Most numerical algorithms for fitting these likelihood-based
models involves evaluating this function at every iteration. Therefore,
as the size of the observed data grows, computational expense grows
*as least* linearly in the number of data points. The linear case is
when the data is assumed to be observed independently in light of the
parameters you wish to estimate. In this case, the likelihood becomes
a product and the log likelihood becomes a sum with each term
consisting of a function of one data point.  This case is very
naturally decomposed into a fine grain parallel problem with a high
data throughput twist. Eg. the number of computations is fairly small
compared to the amount of memory transactions. This becomes a very
natural setting for GPUs, and it is quite easy for GPUs to perform
even better than large compute clusters because of the very low memory
latency in GPUs. Suchard et al. studied these advantages in the
Bayesian mixture model setting and found very promising results (100x
speedup) on graphics cards that are now 2 years old ([JCGS]_).

Challenges of GPU Computing in Statistical Inference
----------------------------------------------------

In the gpustats package, we have three primary goals that we will
address here. First, we hide the ubiquitous boilerplate code common to
all GPU programs. We naturally want to achieve respectable performance
by taking full advantage of the GPU's execution and memory
architecture. Finally, we minimize the effort in developing new
gpustats functions by using meta-programming to handle the 90%
identical code across tasks.

In any GPU based application, there are some necessary functions that
must be called. The most prominent are initialization routines, and
memory transfers between GPU memory and main memory before and after
the parallel GPU code. [PyCUDA]_ handles this very nicely with the
`gpuarray` object which can take a [NumPy]_ array and handle the
memory transfers behind the scenes. As for initialization, there are
several tuning parameters that need to be considered before launching
a kernel. Again, [PyCUDA]_ is capable of querying the GPU and kernel
for all the necessary information to perform optimization. We have
implemented optimization routines for our general cases which further
ease the development of new GPU code in the genre. 

To maximize the performance of our code, we need to fully utilize and
appeal to the memory structure on the GPU. It is a hierarchical
structure with three levels: global, shared, and local. Global memory
is the standard RAM on the card and is usually a few
gigabytes. Threads do not read from this memory, but rather the
multiprocessor makes transactions for all the threads together. To
take advantage of this structure, threads must read from memory in a
coalesced manner. Threads usually read data from global memory to
local memory. Groups of threads cooperatively have access to the same
pool of global memory which is usually 16 KB. Since this is small, the
general kernel structure tends to be to read a little piece of data
from global memory, do computation, write the results, and repeat.
Furthermore, each thread has a very small local memory which is just
used for storing current values in computations.

In `gpustats` we are usually doing one of two computations: evaluating
many points on a distribution or generating many values from a
distribution.  In the both cases, the structure of the input data
doesn't change across distributions. In the first case, the input is a
large pile of data and a set of parameters. In the second, the input
is a large set of uniform random numbers and a set of
parameters. Therefore, not only is most of the boilerplate code the
same, but most of the kernel is the same. In fact, the only part of
the kernel that changes is the actually computation. This implies a
straightforward meta-programming approach.


 


References ----------

.. [JCGS] M. Suchard, Q. Wang, C. Chan, J. Frelinger, A. Cron and M. West.
   	  *Understanding GPU programming for statistical computation: Studies
	  in massively parallel massive mixtures.* Journal of Computational 
	  and Graphical Statistics 19 (2010): 419-438
	  http://pubs.amstat.org/doi/abs/10.1198/jcgs.2010.10016

.. [NvidiaGuide] Nvidia, Inc. *Nvidia CUDA: Programming Guide.* (2010),
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
