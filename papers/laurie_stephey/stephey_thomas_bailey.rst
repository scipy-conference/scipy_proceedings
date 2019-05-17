:author: Laurie A. Stephey
:email: lastephey@lbl.gov
:institution: NERSC
:corresponding:

:author: Rollin C. Thomas
:email: rcthomas@lbl.gov
:institution: NERSC

:author: Stephen J. Bailey
:email: stephenbailey@lbl.gov
:institution: LBL
:bibliography: scipybib

-----------------------------------------------------------------------------
Optimizing Python-Based Spectroscopic Data Processing on NERSC Supercomputers
-----------------------------------------------------------------------------

.. class:: abstract

   We present a case study of optimizing a python-based cosmology data processing
   pipeline designed to run in parallel on thousands of cores using supercomputers
   at the National Energy Research Scientific Computing Center (NERSC).

   The goal of the Dark Energy Spectroscopic Instrument (DESI) experiment is to
   better understand dark energy by making the most detailed 3D map of the
   universe to date. Over a five-year period starting this year (2019), around 
   1000 CCD frames per night (30 per exposure) will be read out from the 
   instrument and transferred to NERSC for processing analysis on the Cori and 
   Perlmutter supercomputers in near-real time. This fast turnaround helps DESI 
   monitor survey progress and update the next night's observing schedule.

   The DESI spectroscopic pipeline for processing these data is written almost
   exclusively in Python. Using Python allows the DESI scientists to write
   very readable and maintainable scientific code in a relatively short amount of 
   time, which is important due to limited DESI developer resources. However, the 
   drawback is that Python can be substantially slower than more traditional high 
   performance computing languages like C, C++, and Fortran.

   The goal of this work is to increase the efficiency of the DESI
   spectroscopic data processing at NERSC while satisfying their requirement that
   the software remain in Python. Within this space we have obtained throughput
   improvements of over 5x and 6x on the Cori Haswell and Knights Landing partitions,
   respectively. Several profiling techniques were used to determine potential
   areas for improvement including Python's cProfile and line_profiler packages, 
   and other tools like Intel Vtune and Tau. Once we identified expensive kernels, 
   we used the following techniques: 1) JIT-compiling hotspots using Numba
   and 2) re-structuring the code to compute and store 
   key data structures rather than repeatedly calling expensive functions. We have 
   considered Dask as a more flexible and robust alternative to MPI for parallelism 
   in the DESI extraction code, but have found that once a code has been designed 
   with MPI in mind, it is non-trivial to transition to another kind of parallelism. 
   We will also explore the requirements for transitioning DESI spectroscopic 
   extraction to GPUs (coming in the next NERSC system, Perlmutter, in 2020).

.. class:: keywords

   NumPy, SciPy, Numba, JIT compile, spectroscopy, HPC, MPI, Dask

Introduction
------------

DESI is the Dark Energy Spectroscopic Instrument :cite:`noauthor_dark_nodate`.
Though dark energy is estimated to comprise over 70 percent of our universe, it
is not currently well-understood
:cite:`peebles_cosmological_2003,mortonson_dark_2013`.  Many experiments,
including DESI, are seeking to uncover more information about the nature of
dark energy. The goal of the DESI experiment is, over 5 years, to map 30
million galaxies and use spectroscopically obtained redshift data to obtain
their distances. This information will allow the most detailed 3D map of the
universe to be constructed, which will help better understand the role of dark
energy throughout the history of the universe. An image of the Mayall
telescope, on Kitt Peak, Arizona, where the DESI instrument is installed, is
shown in Figure :ref:`kittpeak`.

.. figure:: figures/desi_kitt_peak.png

   A photograph of the Mayall telescope (large dome in the center of the
   image), where the DESI instrument has been installed, in Kitt Peak, Arizona.
   :label:`kittpeak`

In a typical night of observing, DESI expects to obtain roughly 1000 CCD
images. In late 2019, DESI will begin sending these images to NERSC for nightly
data processing and will continue sending image data for five years. The sooner
these images can be processed to determine if the data quality was acceptable
(perhaps the weather was poor or the exposure time was not long enough), the
better. The DESI team will examine these results before they can plan their
next night of observing, so in a very real sense every minute counts. 

The DESI instrument is comprised of 5000 robotically controlled optical fibers.
In each patch of sky, every fiber will be positioned to image a target like a
galaxy or quasar. The light from each fiber will be passed into one of 10
spectrographs with three channels: one in the infrared, one in the red, and one
in the blue, which results in a total of 30 frames obtained per exposure. The
5000 individual spectra are stacked on the CCD in a pattern like that shown in
Figure :ref:`ccdpsfexample` (a), where the y-axis is wavelength and
the x-axis is each individual fiber.

.. figure:: figures/ccd_psf_example.png

   (a) This is an example of a DESI CCD image, where the x-axis are the individual fibers
   and the y-axis is wavelength. (b) Since the point spread functions are 2D 
   in nature (note the difference in shapes between the red and green profiles),
   a full 2D fitting is required to accurately capture their
   shape. Image courtesy of S. J. Bailey. :label:`ccdpsfexample`

The data from each fiber spectra have some 2D extent: these 2D elipses are
called point-spread functions (PSFs). One of these PSFs is shown in Figure
:ref:`ccdpsfexample` (b). This figure demonstrates that because the PSF is
elliptical rather than perfectly circular, a 2D fitting is necessary to capture
all of the PSF information. The DESI spectral extraction is performed in two
dimensions using a technique known as "spectroperfectionism"
:cite:`bolton_spectro-perfectionism:_2010`, which is only computationally
feasible due to a divide-and-conquer approach developed by S. J. Bailey and
collaborators. The DESI spectral extraction code performs a variety of eigenvalue
decomposition, evaluating special functions, and all the necessary bookkeeping
required to manage the spectral data in each exposure (about 6GB).

The overarching goal of this work is to speed up the DESI experiment's Python
spectroscopic data processing on the NERSC Cori KNL partition. NERSC is the
National Energy Research Scientific Computing center
:cite:`noauthor_national_nodate`. It is the largest Department of Energy
computing facility in terms of number of users (7000) and scientific output
:cite:`noauthor_publications_nodate`. Cori is NERSC's current flagship supercomputer,
a Cray XC40 with a theoretical peak of 28 PF, comprised of approximately 20
percent Intel Haswell nodes and 80 percent manycore Intel Knights Landing (KNL)
nodes.  Achieving good performance with the manycore KNL nodes has proven
difficult for many science teams; for this reason NERSC established a program
called NESAP (NERSC Exascale Science Applications Program,
:cite:`noauthor_nesap_nodate`). NESAP provides technical expertise from NERSC
staff and vendors like Intel and Cray to a select set of science teams to
improve the performance of their application on the Cori KNL partition.
Achieving optimal Python performance on KNL is especially challenging due a
slower clock speed and difficulty taking advantage of the KNL AVX-512 vector
units (which is not possible in native Python). A more detailed discussion of
the difficulties of extracting Python performance on KNL can be found in
:cite:`ronaghi_python_2017`. Despite these difficulties, DESI requested
that their code should not be re-written in another language like C due
to their own limited developer resources.

In what follows we will present a case study that describes how a Python image
processing pipeline was optimized *without re-writing the code in another language like C*
for increased throughput of 5-7x on a
high-performance system. We will describe our workflow of using profiling tools
to find candidate kernels for optimization, we will describe how we used just
in time compiling to speed up these kernels. We will also describe our efforts
to restructure the code to minimize the impact of calling expensive kernels. We
will compare parallelization strategies using MPI and Dask, and we will discuss
preliminary considerations for moving the code to GPUs.

Profiling the code
------------------

Our first step in this study was to use profiling tools to determine places in
the DESI code where it was worthwhile to target our optimization efforts. We
made heavy use of tools designed especially for Python. In general our process
was to start with the simplest tools and then, when we knew what we were
looking for, use the more complex tools.

We should note that we profiled the DESI code on both Cori Haswell and KNL
nodes. There were some minor differences in the relative time spent in each
kernel between the two architectures, but overall the same patterns were
present on both Haswell and KNL.

cProfile
~~~~~~~~

.. figure:: figures/cpu_2.png
   :align: center
   :scale: 20%
   :figclass: wt

   This is an example image created from data collected using cProfile and
   visualized using gprof2dot :cite:`fonseca_converts_2019`.
   This profile was obtained from an early stage in
   the DESI python optimization effort. :label:`gprof2dot`

Python's built-in cProfile :cite:`noauthor_26.3._nodate` was the first tool we
used for collecting profiling data. We found cProfile simple and quick to use
because it didn't require any additions or changes to the DESI code. cProfile
can write data to a human-readable file, but we found that using either
Snakeviz :cite:`noauthor_snakeviz_nodate` or gprof2dot
:cite:`fonseca_converts_2019` to visualize the profiling data was substantially
more clear and useful.

An example of data collected using cProfile and visualized with gprof2dot is
shown in Figure :ref:`gprof2dot`. We prefer gprof2dot to Snakeviz
visualizations because they are static images instead of browser-based. This
makes them easier to store, share, quickly view, and embed in papers and talks.
If you prefer accessing the cProfile data interactively, and clicking on a
function to see all of its children, for example, Snakeviz can provide this
functionality. However, we found the several extra steps required to use
Snakeviz, and the difficulty storing and sharing the visualizations, made it
less appealing than gprof2dot. 

Examining the visualized cProfile data allowed us to identify expensive kernels
in the DESI calculation. In Figure :ref:`gprof2dot`, the functions are
color-coded according to how much total time is spent in each of them. In this
example, the function "traceset" accounts for approximately 37 percent of the
total runtime and was a good candidate for optimization efforts.

Information like that shown in Figure :ref:`gprof2dot` is nevertheless
incomplete in that it can only provide detail at the function level. From
these data alone it was difficult to know what specifically in the function
"traceset" was so time-consuming. Once we had a list of expensive kernels from
our cProfile/gprof2dot analysis, we started using the line_profiler tool.


line_profiler
~~~~~~~~~~~~~

line_profiler :cite:`kern_line-by-line_2019` is an extremely useful tool which
provides line-by-line profiling information for a Python function. However,
this more detailed information comes at a cost: the user must manually decorate
functions that he or she wishes to profile. For a small code this exercise
might be trivial, but for the many thousand line DESI code 1) hand-decorating
every function would have been both extremely time-consuming and 2) searching
through the line_profiler output data to find expensive functions would have
also been cumbersome and potentially error-prone. For this reason we recommend
starting with cProfile and then moving to line_profiler once the user has
identified a few key functions of interest.

Once decorated, line_profiler provides a great deal of information
for each line of the function, including how many times each line was invoked
and the total amount of time spent on each line. An example of line_profiler
output for the function "xypix" is shown in Figure :ref:`lineprofiler`. This
information was vital to our optimization efforts because it could point to
functions that were particularly expensive, such as numpy's legval or scipy's
erf. Once we had this information, we could make decisions about how to try to
reduce the time spent in these functions, either by speeding up the functions
themselves by JIT-compiling, or by restructuring the code to avoid calling
these expensive functions as often. We will describe both approaches in the
sections that follow.

Together, cProfile and line_profiler were sufficient for almost all of the
performance optimization work in this case study. However,
because the DESI extraction code is an MPI code, these profiling tools do have
some limitations.  Both of these tools can be used to collect data for each MPI
rank, but visualizing and using the information in a meaningful way is
challenging, especially when there are 68 outputs from a KNL node, for example.

.. figure:: figures/line_profiler_xypix.png

   Here is a sample output window from line_profiler
   :cite:`kern_line-by-line_2019` for the function "xypix". The clear,
   human-readable output files produced by line_profiler
   are a very nice feature.
   :label:`lineprofiler`

Vtune and Tau
~~~~~~~~~~~~~

Once we reached the point where we wanted to investigate 1) each individual MPI
rank and 2) whether all ranks were appropriately load-balanced, we needed more
powerful profiling tools like Intel Vtune :cite:`admin_python*_nodate` and Tau
:cite:`noauthor_tau_nodate`. We started with Vtune but ultimately found this
was an unsatisfying tool for several reasons. We found that it was difficult to
get the information we wanted in a clear, understandable format. For example,
Vtune would often display extremely low-level information that obfuscated the
higher-level Python calls we were trying to investigate. It also offered almost
no helpful visualizations. We ultimately found the Tau profiler more useful and
well-suited for our application, although we should note that we required the
help of the Tau developers to build it. (Tau works best when it is built for
the type of application you will profile. In our case it was a Python MPI code
running on a Cray system, all of which are configurations that Tau supports.)
Though building a profiling tool from scratch was non-trivial, it was also very
possible with the help of the Tau team. Once built, Tau provided clear
information about how each MPI rank was occupied and how each rank compared to
the others.  A sample Tau output window is shown in Figure :ref:`tau`. These
profiling data were obtained before the DESI frame was parallelized over
subbundles, leaving 12 of the 32 Haswell ranks unoccupied. It is clear from
this Tau visualization that we were not making good use of processor resources.

.. figure:: figures/tau_main.png

   A sample Tau :cite:`noauthor_tau_nodate` output for the DESI spectral
   extraction code on a
   Haswell processor (which has 32 ranks). It is clear from this output that only
   20 of the ranks are being utilized. This motivated the re-structure to allow
   parallelization of subbundles, rather than bundles, which could more flexibly
   utilize the whole processor's resources. :label:`tau`

Just-in-time (JIT) compilation with Numba
-----------------------------------------

The first major approach to achieve speedups in this work has been to focus on
making expensive functions run more quickly. To achieve this, we have used
Numba :cite:`lam_numba:_2015` is a just-in-time compiler for Python.

We used Numba for three functions that, through profiling, we identified as
expensive. These functions were 1) numpy.polynomial.legendre.legval
:cite:`noauthor_numpy.polynomial.legendre.legval_nodate`, 2) scipy.special.erf
:cite:`noauthor_scipy.special.erf_nodate`, and 3) scipy.special.hermitenorm
:cite:`noauthor_scipy.special.hermitenorm_nodate`. Henceforth we will refer to
these functions as legval, erf, and hermitenorm.

legval was perhaps the most straightforward of these three to JIT compile.
Unlike Python, Numba will allow all variables and arrays to have only a single type.
The types and sizes of
all variables must be known prior to compile time. This required several small
changes to the legval algorithm to put it in the form required by Numba.
Several other lines of the function that performed type checking were removed.
This placed the onus on the developer to make sure the correct types are
supplied, which was acceptable for us. The original and modified legval
functions are shown in Figure :ref:`legval`.

.. figure:: figures/legval_old_vs_new.png
   :align: center
   :scale: 50%
   :figclass: wt

   (A) The official numpy.polynomial.legendre.legval function. Profiling data
   indicated that this was an expensive function. To conserve space the docstring
   has been removed. (B) Our modified legval function that was much faster than
   its original numpy counterpart. Note the removal of the type checking and the
   addition of the np.ones array to instruct Numba about the sizes of each array
   (and prevent them from changing during every iteration.) :label:`legval`

The two scipy functions were also somewhat challenging to implement in Numba.
At the time of this writing, Numba does not yet support directly compiling
scipy functions. This meant that we needed to extract the core part of these
scipy functions and mold them into a form that Numba will accept. For scipy
erf, this meant translating the Fortran source code into Python. For scipy
hermitenorm which was fortunately already in Python, algorithmic changes
similar to those we made in legval were necessary to ensure all variables
were a constant type and size.

We should note that we tried to cache the compiled Numba functions with the cache=True
option to save time, but with larger numbers of MPI ranks, we found that this sometimes caused
a data race between the Numba caches written by each rank. To avoid this problem we
considered using ahead of time (AOT) instead of JIT compiling but since this change
was somewhat cumbersome, for now we removed the cache=True setting and will consider
using AOT in the future.

Restructuring the code
----------------------

The second major optimization strategy we used was to intelligently
restructure the code. This meant that we 1) tried to call expensive functions
fewer times, which often meant that we 2) tried to call expensive functions
with vectors rather than scalars, and 3) had to add machinery to store these
results and reuse them as necessary.

Implement subbundles
~~~~~~~~~~~~~~~~~~~~

One recommendation from an Intel Dungeon session (a collaborative hack session
between NESAP teams and Intel engineers) was to reduce the number of fibers
processed from bundles (25 fibers at a time) into subbundles of approximately 6
fibers at a time. (We confirmed later that for 2-10 fibers at a time, the
performance was relatively unchanged on both Haswell and KNL). These smaller
matrix sizes resulted in faster matrix operations such as multiplication and
eigenvalue decomposition. Presumably this speedup is because these smaller
matrices fit better into lower level cache on both Haswell and KNL, although we
did not verify that this was in fact what was happening. Perhaps this is a
lesson to the reader: profile your code early and often to understand the
impact of the changes you have recently made and re-evaluate your current
optimization plan.

Add cached legval values
~~~~~~~~~~~~~~~~~~~~~~~~

Another outcome from the Intel Dungeon session was the recommendation to
re-structure the code to avoid calling legval. The problem with legval wasn't
just that it was an expensive function; rather, it was also contributing to a
large fraction of the total runtime because it was called millions of times for
each CCD image in the DESI spectral extraction calculation. Worse, legval was
called with scalar values even though it was able to handle vector inputs.

This restructuring required us to modify several major functions and redefine
some of the bookkeeping that keeps track of which data corresponds to which
part of the image on the CCD. Prior to the restructure, profiling data indicated
that legval was called approximately 7 million times per frame with scalar values.

The code was restructured so that legval was now called 800,000 times per
frame. Of course this is still a large number, but it is almost an order of
magnitude fewer times than the original implementation. The calculated values
were stored as key-value pairs in a dictionary. We then modified the part of
the code that previously calculated legval to instead look up the required
values stored in the dictionary.

Parallelize over subbundles instead of bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current DESI MPI framework is to split COMM_WORLD into n
bundle communicators where n is the number of processors per chip. This is
inefficient on a single processor because 20 bundles only use a fraction of the
available processors on either a Haswell or KNL. To process additional frames
(and additional multiples of 20 bundles), a specific number of nodes must be
carefully chosen to fill the processors as much as possible. This means to
process a full exposure of 30 frames (600 bundles), 19 Haswell nodes and 9 KNL
nodes are required to efficiently use the processors.

In this case, the goal was to restructure the code to divide the spectral
extraction into smaller, more flexible pieces. This would relax the previous
requirement that each frame be divided into 20 bundles, which is an awkward
number for NERSC hardware (Haswell has 32 processors and KNL has 68
processors). Furthermore, it meant that only certain numbers of nodes could be
chosen to efficiently process an exposure (30 frames). For example, on Haswell,
this number is 19 (ceil 600/32), and on KNL, this number is 9 (ceil 600/68).

Dividing the workload into subbundles (smaller bundles) means that about 500
spectra are now more evenly doled out to 32 processors (about 16 spectra each)
or 68 processors (about 7 spectra each). The COMM_WORLD communicator
orchestrates all 30 frames within a single exposure, and the frame level
communicator orchestrates the subbundle processing within the frame.
Implementing this change was nontrivial but the speedup and flexibility gains
made it worthwhile to the DESI team. Using all processors more efficiently
resulted in a per-frame speedup for both Haswell and KNL...

Optimization results
--------------------

How effective were all these different optimization efforts we just described?
The most straightforward benchmark is one in which raw runtime (and hopefully
speedup) is measured. In this case, we measured the time to complete the
processing of a single DESI frame on a single Edison, Cori Haswell, and Cori
KNL node. In Figure :ref:`singlenode` we show how each optimization affected
the single frame runtime. The optimizations are plotted chronologically against
the overall runtime of the frame on each architecture.

Figure :ref:`singlenode` shows that the first few changes we made had the
largest overall impact: the later optimizations exhibited some diminishing
returns as we continued to implement them. Over the course of this work the
runtime for a single frame was decreased from 4000 s to 525 s for KNL, from 862
to 130 seconds for Haswell, and from 1146 s to 116 s for Ivy Bridge. The
overall increases in raw speed varied between 7-10x for each architecture. One
major goal of the NESAP program was to reduce the DESI runtime on KNL to below
the original Edison Ivy Bridge benchmark, which is indicated by the red dotted
line. Once we implemented our legval cache fix, we achieved this goal.

.. figure:: figures/single_node_benchmark.png

   The single-node speedup achieved on Intel Ivy Bridge, Haswell, and KNL architectures
   throughout the course of this study. :label:`singlenode`

A more meaningful benchmark for DESI is the number of frames that can be
processed during a given amount of time using a given number of nodes. We call
this throughput metric "frames per node hour". We performed these frames per node hour
benchmarks with a full exposure (30 frames), instead of a single frame, on
either 19 or 9 nodes for Haswell and KNL, respectively. Though a single
exposure is still a relatively small test because DESI expects to collect 50 or
more exposures per night, it much more closely approaches the real DESI
workload than the single frame benchmark. One feature encoded in this benchmark
which is not captured in the speed benchmark is the increasingly important role
that MPI overhead begins to play in multi-node jobs, which is a real factor the
DESI will have to contend with during its large processing runs. The frames per
node hour results are plotted in Figure :ref:`framespernodehour`. While the
increases in throughput we have obtained are more modest than the raw speedup,
these values are a more accurate representation of the actual improvements in
DESI's processing capability. For this reason we emphasize that we were able to
achieve a 5-7x throughput increase instead of the (more exciting but less
meaningful) 7-10x in raw processing speed.

.. figure:: figures/frames_per_node_hour.png

   This figure shows the improvement over the course of this study in the DESI
   spectral extraction throughput. :label:`framespernodehour`

Finally, in Table 1 we summarize the incremental speedups we obtained
throughout this study on Edison Ivy Bridge, Cori Haswell, and Cori KNL
according to their type. Perhaps these results are the most generally
instructive. First, they demonstrate the restructuring-based optimizations were
more valuable the JIT-based optimizations. For example, the overall speedup of
adding the legval cached values was approximately 1.7x, although this was also
the most difficult of all the optimizations in this study. In contrast, our
relatively painless JIT compiled optimizations were not as effective in terms
of speedup, averaging between a factor of 1.1-1.5x improvement. The takeaway
from these results might be that if a developer has enough time, the larger,
more complex restructuring optimizations may be extremely worthwhile. The flip
side is that if the developer has limited time, small fixes like JIT compiling
can still provide reasonable gains without a major time investment.

.. raw:: latex

   \begin{table*}

     \begin{longtable}{|c|c|c|c|c|c|}
     \hline
     \textbf{Optimization}  & \textbf{Type} & \textbf{Mean Speedup} & Ivy Bridge Speedup & Haswell Speedup & KNL Speedup \tabularnewline
     \hline
     Add subbundles & Restructure & 1.55106 & 1.62882 & 1.73696 & 1.28741 \tabularnewline
     \hline
     Fix legval & JIT compile & 1.11607 & 1.16106 & 1.06005 & 1.12709 \tabularnewline
     \hline
     Add caching & Restructure & 1.70416 & 1.72505 & 1.70197 & 1.68546 \tabularnewline
     \hline
     Fix pgh & JIT compile & 1.28906 & 1.33125 & 1.15036 & 1.38556 \tabularnewline
     \hline
     Fix xypix & JIT compile & 1.49806 & 1.51875 & 1.31501 & 1.66042 \tabularnewline
     \hline
     \end{longtable}

     \caption{Types of optimization efforts performed in this study and their
        resulting speedups on Intel Ivy Bridge, Haswell, and Knights Landing architectures.
        The geometric mean speedup achieved on all three architectures is displayed in
        the third column. The order of these optimizations is displayed chronologically.}

   \end{table*}


What about using Dask instead of MPI?
-------------------------------------

A few problems with the current MPI implementation of the DESI spectral
extraction code prompted us to take a step back and consider if perhaps Dask
:cite:`noauthor_dask:_nodate` would be a better solution for parallelization
within DESI. The first was the relative inflexibility of the division of work
between bundles (although this has been addressed now in the subbundle
division). The second was the issue of resiliency: if a node goes down, it will
take the entire MPI job with it. (This is not an issue in Dask, in which dead
workers can be seamlessly revived while the calculation continues.) An additional feature
we liked about Dask is the ability to monitor Dask jobs in real time with their
Bokeh status page. We thought Dask seemed promising enough that it was worth
taking a careful look at what it would mean to replace the DESI MPI with Dask.

Dask is a task-based parallelization system for Python. It is comprised of a
scheduler and some number of workers which communicate with each other via a
client. Dask is more flexible than traditional MPI because it can start workers
and collect their results via a concurrent futures API. (It should be noted
that this is also possible in MPI with dynamic process management, but since
Cray does not officially support this due to problems with SLURM functionality,
we haven't been able to try this API.)

During this process, we discovered that is that it is non-trivial to convert a
code already written in MPI to Dask, and it would likely be difficult to
convert from Dask to MPI as well. (It would likely be easier to convert from
dynamic process management MPI to Dask, but the DESI spectral extraction code
is not written with this API.)

One major difference between MPI and Dask is the point at which the decision of
how to divide the problem occurs. In MPI since all ranks are generally passing
over the code, dividing the data and performing some operation on it in
parallel can be done on the fly. In Dask, however, the scheduler needs to know
in advance which work to assign to workers. This means that the work must
already be divided in sensible way. Collecting the information required for
Dask-style parallelism in advance would have required a substantial
re-structuring on the order of what was performed for legval, if not more
ambitious. At this point we decided that if the DESI code had been written from
the start with Dask-type parallelism in mind using Dask would have been a good
choice, but converting existing MPI code into Dask was unfortunately not a
reasonable solution for us.

Does it make sense to run DESI on GPUs?
---------------------------------------

Because HPC systems are becoming increasingly heterogeneous, it is important to
consider how the DESI code will run on future architectures. The next NERSC
system Perlmutter :cite:`noauthor_perlmutter_nodate` will include a CPU and GPU
partition that will provide a large fraction of the system's overall FLOPS, so
it is pertinent to examine if and how the DESI code could take advantage of
these accelerated nodes.

Since GPUs are fundamentally different than CPUs, it may be necessary to
rethink much of the way in which the DESI spectral extraction is performed. At
the moment, each CCD frame is divided into 20 bundles, and each bundle is
divided into 60 patches, and each of those 60 patches is further divided into 6
smaller subbundles. Though this division of a larger frame into smaller pieces
makes sense for CPU architectures, it doesn't make sense for GPU architectures.
In fact for GPUs often the opposite is true: the programmer should give the GPU
as much work as possible to keep it occupied and make the relatively expensive
transfer of data between the host and device worthwhile. This means that to
help the DESI extraction code run efficiently on GPUs it will likely require a
major restructuring to better adapt the problem for the capabilities of the
hardware.

Preliminary testing is underway to give some indication of what we might expect
from a major overhaul. From profiling information we expect that the
scipy.linalg.eigh function will constitute a major part of the workload as
matrix sizes increase. We have measured the runtime of scipy.lialg.eigh and
cupy.linalg.eigh :cite:`noauthor_cupy.linalg.eigh_nodate` on Edison Ivy Bridge
and Cori Haswell, KNL, and the new Cori Volta GPUs. Figure :ref:`eigh` shows
the eigh runtime for various sizes of positive definite input matrices. These
results show that at low matrix sizes, perhaps unsurprisingly, the Volta
performs poorly, but at larger matrix sizes (above 1000) the Volta performance
dominates by an order of magnitude. This demonstrates, at least for scipy eigh,
that breaking the DESI frame into fewer, larger pieces for a GPU could result
in substantial performance gains. Of course the question is 1) is this large
restructuring worthwhile and 2) if so, what is the best approach? As we have
detailed above, we have had reasonably good success with Numba, which also
supports GPU offloading. Other options are CuPy :cite:`noauthor_cupy_nodate`,
which aims to be a drop-in replacement for NumPy, pyCUDA
:cite:`noauthor_pycuda_nodate`, and pyOpenCL :cite:`noauthor_pyopencl_nodate`.
How best to support GPU offloading without having to fill the DESI code with
distinct CPU and GPU blocks, and additionally avoid being tied to a particular
vendor, is still an open question for us.

.. figure:: figures/eigh.png

   Data from performing an eigh matrix decomposition of various sizes on Edison
   Ivy Bridge, Cori Haswell, Cori KNL, and Cori Volta. :label:`eigh`



Conclusions and Future Work
---------------------------

Over the course of this work, we have achieved our goal of speeding up the
throughput of the DESI spectral extraction code on NERSC Cori Haswell and KNL
processors by a factor of 5-7x. Our strategy was as follows: we employed
profiling tools, starting with the most simple tools (cProfile + gprof2dot) and
progressing as necessary to more complex tools (line_profiler and Tau), to get
an idea of which kernels are most expensive and what types of structural
changes could help improve runtime and flexibility. We used Numba to JIT
compile several expensive functions. This was a relatively quick way to obtain
some speedup without changing many lines of code. We also made larger
structural changes to avoid calling expensive functions and also to increase
the flexibility and efficiency of the parallelism. In general these larger
structural changes were more complex to implement, as well as more time
consuming, but also resulted in the biggest payoff in terms of speedup. We
considered changing the parallelism strategy from MPI to Dask, but ultimately
found that changing an existing code is non-trivial due to the fundamentally
different strategies of dividing the workload, and decided to continue using
MPI. Finally, we are now investigating how the DESI code could run effectively
on GPUs by since the next NERSC system Perlmutter will include a large CPU and
GPU partition. Exploratory studies for how the DESI code can be optimized are
being performed using scipy.linalg.eigh and cupy.linlg.eigh as a test case now
and will continue as future work.

Acknowledgments
---------------

The authors thank their partners at Intel, the Intel Python Team, Intel tools
developers, performance engineers, and their management. The authors also would
like to thank the Tau Performance System team at the University of Oregon for
their help in building Tau for our application. This work used resources of the
National Energy Research Scientific Computing Center, a DOE Office of Science
User Facility supported by the Office of Science of the U.S.  Department of
Energy under Contract No. DE-AC02-05CH11231. Additionally, this research is
supported by the Director, Office of Science, Office of High Energy Physics of
the U.S.  Department of Energy under Contract No.  DE–AC02–05CH1123, and by the
National Energy Research Scientific Computing Center, a DOE Office of Science
User Facility under the same contract; additional support for DESI is provided
by the U.S. National Science Foundation, Division of Astronomical Sciences
under Contract No.  AST-0950945 to the National Optical Astronomy Observatory;
the Science and Technologies Facilities Council of the United Kingdom; the
Gordon and Betty Moore Foundation; the Heising-Simons Foundation; the National
Council of Science and Technology of Mexico, and by the DESI Member
Institutions.  The authors are honored to be permitted to conduct astronomical
research on Iolkam Du’ag (Kitt Peak), a mountain with particular significance
to the Tohono O’odham Nation.




