:author: Ralph Kube
:email: rkube@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:author: R Michael Churchill
:email: rchurchi@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:author: Jong Youl Choi
:email: choij@ornl.gov
:institution: Oak Ridge National Laboratory

:author: Ruonan Wang
:email: wangr1@ornl.gov
:institution: Oak Ridge National Laboratory

:author: Scott Klasky
:email: klasky@ornl.gov
:institution: Oak Ridge National Laboratory

:author: CS Chang
:email: cschang@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:video: http://www.youtube.com/watch?v=AXG3ma_f-iM:

----------------------------------------------------------------------
Leading magnetic fusion energy science into the big-and-fast data lane
----------------------------------------------------------------------

.. class:: abstract

We present ``Delta``, a Python framework for efficient wide-area network transfer of high-velocity
high-dimensional data streams from remote scientific experiments, to HPC resources for parallelized
processing of typical scientific analysis workflows. Targeting magnetic fusion research, we use DELTA 
to adapt an existing Python code-base that performs spectral analysis of imaging data on single-core 
architectures to a code that processes the data stream on a modern HPC architecture. ``Delta`` facilitates
data transfers using the ADIOS2 I/O middleware and data analysis tasks are dispatched using PoolExecutors.
For the magnetic fusion data case ``Delta`` reduces the wall-time to run the entire suite of 
data analysis routines from 12 hours to less than 10 minutes.


.. class:: keywords

   streaming analysis, mpi4py, queue, adios, HPC


Magnetic Fusion Energy research and its data analysis needs
-----------------------------------------------------------

Research on magnetic fusion energy (MFE) combines theoretical physics, experimental physics, engineering,
and even economics to achieve the goal of developing an unlimited, clean energy source. Python has
established itself in the fusion community with projects like plasmapy [PPY]_ or OMFIT [OMF]_. With
a plethora of scientific disciplines at its footing, scientific workflows in MFE 
vary significantly by the amount of data and computational time they require. Here we present a
``Delta``, a new workflow tool we hope will aid research on magnetic fusion energy.

But before we begin, what is the goal of magnetic fusion energy research?
If you could harvest the energy from controlled nuclear fusion reactions you would have 
a potentially unlimited, environmentally friendly energy source. Nuclear fusion reactions
are the opposite to nuclear fusion reactions, which power todays nuclear power plants.
In a fusion reaction two light atomic nuclei merge into a heavier one, while converting a 
fraction of the reactants binding energy into kinetic energy of the products. As a nuclear reaction,
the amount of energy released is larger by orders of magnitude than for chemical reactions
such as oxidization of carbon when burning coal. At the same time nuclear fusion reactions
are inherently safe. In order to bring positively charged atomic nuclei close enough together 
so that they fuse requires a temperature upwards of 100 million degrees. Such a requirement unfortunately
excludes any material container to
confine a fusion fuel. The most promising approach to confine a fusion fuel is in the 
state of a plasma - a hot gas where the atoms are stripped of their electrons. Such a 
plasma can be confined in a strong magnetic field, shaped like a donut-shaped. Since the energy yield 
of a fusion reaction is so large, only little fusion plasma needs to be confined
to power a fusion reactor. To produce 1 GW of fusion power, enough to power about 700,000 homes, 
just 2 kg of fusion plasma would need to be burned per day [Ent18]_. Thus, a catastrophic event
such as total loss of plasma confinement can cause no more than local damage to the plasma vessel. 
The physical principles of Fusion Energy also forbid uncontrolled chain reactions. Under operation, 
plasma facing components of the reactor will be activated. These materials will be safe to handle after
about 10 years. Fuels for fusion reactions are readily
extracted from sea water, which is available in near-infinite quantities. 

The plasma confinement devices with the best performance, called tokamaks, are donut shaped.
Medium-sized tokamaks, such as DIII-D [D3D]_, NSTX-U [NSTX]_, ASDEX Upgrade [AUG]_,
MAST [MAST]_, TCV [TCV]_ or KSTAR [KSTAR], have a major radius R=1-1.5m and a minor radius a=0.2-0.7m.
In experiments at these facilities, researchers configure parameters such as the plasma density or
the shaping and strength of the magnetic field and study the behaviour of the plasma in this setup.
During a typical experimental workflow, about 10-30 ``shots`` are performed on a given day where 
each shot lasts for a couple of seconds up to minutes. In this time numerous measurements of the plasma
and the mechanical components of the tokamak are performed. After a cool-down phase - the tokamaks
contain cryogenic components - the device is ready for the next shot. 

A common diagnostic in magnetic fusion experiments are so-called Electron Cyclotron Emission Imaging (ECEI) 
systems [Cos74]_. They measure emission intensity by free electrons in the plasma,
which allows to infer their temperature as a function of radius. Modern ECEI systems, as the 
one installed in the KSTAR tokamak [Yun14]_ have hundreds of spatial channels and sample data on a
microsecond time-scale, producing data streams upwards of 500 MB/sec. 

The aim of the ``Delta`` framework is to integrate measurement-based decision making in the experimental workflow.
This use-case falls in between two other common data analysis workflows in fusion energy research, listed in 
table :ref:`timescale`. Real-time control systems for plasma engineering components require data on a millisecond
time scale. The amounts of data provided by these algorithms is constrained by the processing time. Post-shot
batch analysis of measurements on the other hand serve scientific discovery. The data and the analysis methods
are selected on a per-case basis. By providing timely analysis results of plasma measurements to the experimentalists,
they can make more informed decisions about the next plasma shot. Such a workflow has been used in experiments at
TAE, where the machine-learning based optometrist algorithm was leveraged to significantly increase fusion yield [Bal17]_. 


.. table:: Time-scales on which analysis results of fusion data is required for different tasks.  :label:`timescale`

    +-----------------------------+--------------------+
    |    Task                     | Time-scale         |
    +=============================+====================+
    | real-time control           | millisecond        |
    +-----------------------------+--------------------+
    | live/inter-shot analysis    | seconds, minutes   |
    +-----------------------------+--------------------+
    | scientific discovery        | hours, days, weeks |
    +-----------------------------+--------------------+


Designing the Delta framework
-----------------------------


We are designing the ``Delta`` framework in a bottom-up approach, tailoring it to the needs of the
ECEi analysis workflow and a specific deployment platform in mind. While plasma diagnostics 
operated at fusion experiments produce a heterogeneous set of data streams, the ECEi analysis  
is still representative for a large set of data streams produced by other diagnostics. HPC environments
are also rather heterogeneous. There are significant differences in local area network topologies, such 
as the speed network links between data-transfer nodes to compute node and even compute node interconnects,
network security policies, and granted allocation of compute time for research projects that make it unpractical
to start with a top-down approach that generalizes will to arbitrary HPC targets. In the remainder of this section
we describe the data analysis asks for ECEI data, the targeted network and deployment architecture and 
give an overview of how ``Delta`` connects them with one another.

Electron Cyclotron Emission Imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Electron Cyclotron Emission Imgaging diagnostic installed in the KSTAR tokamak 
measures the electron temperature :math:`T_e` on a 0.15m by 0.5m grid, resolved using 8 horizontal
and 24 vertical channels [Yun10]_ [Yun14]_. Each individual channel produces an intensity time series
:math:`I_{h, v}(t_i)` where h and v index the horizontal and vertical channel number and
:math:`t_i = i * \Delta_t` denotes the time where the intensity is sampled with 
:math:`\Delta_t \approx 1 \mu s` being the sampling time. Digitizing the samples with a 16-bit 
digitizer results in a data stream of 2 byte * 192 channels * 1,000,000 samples / sec = 384 MByte/sec.
The spatial view of this diagnostic covers a significant area of the plasma cross-section which allows it to directly visualize the large-scale 
structures of the plasma. Besides analyzing the normalized intensity, several quantities calculated 
off the Fourier Transformed intensity :math:`X(\omega)`, here :math:`\omega` denotes the angular frequency, are used
to study the plasma dynamics. The cross-power S, the coherence C, the cross-phase P and 
the cross-coherence R are respectively defined for two Fourier Transformed intensity signals X and Y as


.. math:: 
   S_{xy}(\omega) = E[F_x(\omega) F_y^{\dagger}(\omega)],
   :label: eq-S
   
   
.. math::
   C_{xy}(\omega) = |S_{xy}(\omega)| / \sqrt{S_{xx}(\omega)} / \sqrt{S_{yy}(\omega)},
   :label: eq-C


.. math::
   P_{xy}(\omega) = arctan(Im(S_{xy}(\omega)) / Re(S_{xy}(\omega)),
   :label: eq-P
   

and

.. math::
   R_{xy}(t) = IFFT(S_{xy}(\omega)).
   :label: eq-R
   

Here E denotes an ensemble average, :math:`^{\dagger}` denotes complex conjugation, :math:`Re` and
:math:`Im` denote the real and imaginary part of a complex number and :math:`IFFT` denotes the
inverse Fourier Transform. Spectral quantities calculated off local :math:`T_e` fluctuations, such
as the cross coherence or the cross phases, are used to identify macro-scale structures in the
plasma, so called magnetic islands [Cho17]_. Detection of magnetic islands is an important task as
they can disrupt plasma confinement.

Commonly, ECEI measurements are analyzed manually batch-wise hours, days, or weeks after a given plasma shot.
In a typical workflow, the raw data files and maybe a copy of common analysis routines are copied to a workstation
or researchers they write their own analysis codes. Then the channel pairs for which spectral quantities
Eq.(:ref:`eq-S`) are to be computed are specified by hand. Output and visualization are stored in another file. 

Abundant high performance computing resources make it possible to design a streaming workflow for
this task . Modern high-performance computing (HPC) resources provide ample computing power
to perform calculations of all relevant spectral quantities, for any given channel pair in near
real-time. Furthermore, the calculated quantities can be stored indefinitely for future access
together with descriptibe meta-data for later access and re-analysis. 

The ``Delta`` framework implements analysis routines for Eqs.(:ref:`eq-S`) - (ref:`eq-R`) as separate kernels 
The performance of the current implementation is compared to a commonly used pure python implemenation
in the next section.


Targeted HPC architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Designed with a specific application in mind, we implement ``Delta`` for streaming data from KSTAR to the 
National Energy Research Scientific Computing Centre (NERSC). NERSC operates Cori, a Cray XC-40 supercomputer
that is comprised of 2,388 Intel Xeon "Haswell" processor nodes, 9,688 Intel Xeon Phi "Knight's Landing" (KNL)
nodes and ranks 13 on the Top500 list. Figure :ref:`fig-topo` illustrates the network topology which ``Delta``
targets. Cori is placed in a separate network at NERSC. To transfer data to Cori at high speeds the traffic 
needs to be routed through a specialized Data Transfer Node (DTN). DTNs are servers dedicated to performing
data transfers. As such, they feature large-bandwidth network interfaces, both for internal and external
connections. Table :ref:`tab1` lists the hardware of the DTNs and Cori's compute nodes. 100Gbit/s links
connect both DTNs via the internet. At NERSC, the DTN is connected to Cori with dual 10 Gbit/s NICs.
In Cori, the individual compute nodes are connected with a Cray Aries interconnect, peaking at > 45 TB/s
[cori]_.

.. figure:: plots/delta_arch.png
   :align: center
   :scale: 40%
   :figclass: w

   The network topology for which the ``Delta`` framework is designed. Data is streamed in the
   direction indicated by the orange arrow. At KSTAR, measurement data is staged from its DTN to
   the NERSC DTN. Internally at NERSC, the data is forwarded to compute nodes at the Cori supercomputer
   and analyzed. Orange arrows mark sections of the network where a custom high-performance streaming
   solution is. Black arrows denote standard TCP/IP connections. The analysis results are stored in a
   database backend and can be ingested by visualizers. :label:`fig-topo`
   

.. table:: Hardware and network interconnections of the data transfer nodes (DTNs) and compute nodes :label:`tab1`
 
    +---------------+--------------------+----------+--------------------------+
    | Where         |   CPU              |    RAM   |  NIC                     |
    +===============+====================+==========+==========================+
    | | KSTAR DTN   | | Xeon E5-2640 v4  | | 128GB  | | 100 Gbit (ext)         |
    +---------------+--------------------+----------+--------------------------+
    | |  NERSC DTN  | | Xeon E5-2680 v2  | | 128GB  | | 2 * 100 Gbit  (ext)    |
    |               |                    |          | | 2 * 10 Gbit  (int)     |
    +---------------+--------------------+----------+--------------------------+
    | | Cori compute| | Xeon E5-2698 v3  |  | 128GB | | Cray Aries             | 
    |               | |  32 threads      |          |                          |
    +---------------+--------------------+----------+--------------------------+




Connection science experiments to HPC resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to connect KSTAR to Cori, ``Delta`` uses three separate software components. A **generator**
running on the KSTAR DTN, a **middle_man** running on the NERSC DTN, and a **processor** running on 
Cori. The generator ingests data from an experiment and sends it through the Internet to NERSC where
the middle_man is running. The middle_man forwards the received data and forwards it to the processor.
The processor receives the data, executes the appropriate analysis kernels and stores the analysis resuls.
To facilitate high bandwidth streaming, ``Delta`` uses ADIOS2 [adios2]_ on the paths marked with orange
arrows in :ref:`fig-topo`. ADIOS2 is a unified input/output system that transports and transforms groups 
of self-describing data variables across different media with performance as a main goal. Its transport 
interface is step-based, which resembles the generation of scientific data. ADIOS2 implements multiple transport
mechanisms as engines, such as DataMan or a Sustainable Staging Transport, which take advantage of underlying
network communication mechanisms to provide optimal performance.
For the topology at hand, ``Delta`` uses the DataMan engine for trans-oceanic data transfer from KSTAR to NERSC.
For intra-datacenter transfer ``Delta`` uses the SST engine.




Gritty details
--------------

After providing an overview of the ``Delta`` framework and introducing its component in the previous section
we continue by describing the implementation details and present performance analysis of the components. 


Performance of the WAN connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


To measured the practically available bandwidth between the KSTAR and NERSC DTNs using iperf3
[iperf]_.
Multiple data streams are often necessary to exhaust high-bandwidth networks. Varying the 
number of senders from 1 to 8, we measure data transfer rates from 500 MByte/sec using 1 
process up to a peak rate of 1500 MByte/sec using 8 processes, shown in Figure :ref:`kstar-dtn-xfer`.
Using 1 thread we find that the data transfer rate is approximately 500 MByte/sec with little 
variation throughout the benchmakr. Running the 2 and 4 process benchmark we see initial transfer
rates of more than 1000 MByte/sec. After about 5 to 8 seconds, TCP observes network congestion and
falls back to fast recovery mode where the transfer rates increase to the approximately the 
initial transfer rates until the end of the benchmark run. The 8 process benchmark shows a
qualitatively similar behaviour but the congestion avoidance starts at approximately 15 seconds
where the transfer enters a fast recovery phase.

.. figure:: plots/kstar_dtn_xfer.png
   :scale: 100%
   :figclass: h

   Data transfer rates between the KSTAR and NERSC DTNs measured using iperf3
   using 1, 2, 4, and 8 processes :label:`kstar-dtn-xfer`





Components of the ``Delta`` framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As shown in Fig. :ref:`fig-topo`, the architecture of ``Delta`` consists of three 
components. At the data staging site a **generator** ingests data from a local source, for example the
diagnostic digitizer, and sends it to the processing facility. At NERSC, the  **middle man**
runs on the DTN, receives the data stream from the WAN and forwards it to Cori. On cori the **processor**
runs as an MPI program, receives the data stream, performs data analysis and stores the results in a backend,
such as a database. Once stored, the analyzed can readily be ingested by visualizers, such as a dashboard. Figure 
:ref:`fig-sw-arch` visualizes the architecture, but leaves out the middle man for simplicity.


.. figure:: plots/delta-sw-arch.png
   :align: center
   :figclass: w
   :scale: 40%

   Schematic of the ``Delta`` framework. The **generator** runs at the data staging site and
   transmits time chunks to the **processor** via the ADIOS2 channels SSSSS_ECEI_NN. Here SSSSS 
   denotes the shot number and NN enumerates the ADIOS2 channels.  :label:`fig-sw-arch`.


The generator is implemented as a single-threaded application. Data is sourced using a loader
class, that handles all diagnostic specific data transformations. For the
ECEI diagnostic this includes for example calculating a channel-dependent normalization and 
the aggregation of data into time chunks. A time chunk are :math:`n_{ch}` consecutive voltage samples.
Data is transferred by a writer class which handles all calls to ADIOS2. 
Pseudo-code for the generator looks like this:

.. code:: python
   :linenos:

   loader = loader_ecei(cfg["ECEI"])
   writer = writer_gen(cfg["transport_tx"])
   writer.Open()

   batch_gen = loader.batch_generator()
   for batch in batch_gen:
       writer.BeginStep()
       writer.put_data(batch)
       writer.EndStep()


Here, cfg is a framwork-wide json configuration file. Diagnostic-specific parameters, such as :math:`n_{ch}`
and details on how to calculate data normalization, are stored in the ``ECEI`` section. ADIOS2 parameters
for the writer, such as to use the DataMan IO engine and connection details, are stored in the ``transport_tx`` section.
Moving all diagnostic-dependent transformations into the loader class, the generator code appears 
diagnostic-agnostic. We note however that in the current version, the number of generated data
batches, which is specific to the ECEi diagnostic, defines the number of steps. Furthermore, the
pseudo-code  example above demonstrates the step-centered design of the ADIOS2 library. It encapsulates 
each time chunk in a single time step.

The middle-man runs on the NERSC DTN. It's task is to read data from the generator and pass it along 
to the processor. Using the classes available in ``Delta``, the pseudo-code looks similar to the
generator. But instead of a loader, a reader object is instantiated that consumes the generators
writer stream. This stream is passed to a writer object that sends the stream to the processor.

The processor is run on Cori. It receives the incoming data stream and dispatches specified analysis
tasks. In pseudo-code the processor looks like this

.. code:: python
   :linenos:

   def consume(Q, task_list):
     while True:
        try:
          msg = Q.get(timeout=5.0)
        except queue.Empty:
          break
        task_list.submit(msg)
      Q.task_done()


   def main():
      executor_fft = MPIPoolExecutor(max_workers=NF, 
                                     mpi_info={"host": 
                                               "root_node"})
      executor_anl = MPIPoolExecutor(max_workers=NA,
                                     mpi_info={"hostfile": 
                                               "worker_nodes"})
      a2_reader = reader(cfg["transport_rx"])
      reader.Open()
      task_list = task_list_spectral(executor_anl, 
                                     executor_fft, cfg)

      dq = Queue.Queue()
      workers = []
      for _ in range(n_thr):
         w = threading.Thread(target=consume, 
                              args=(dq, task_list))
         w.start()
         workers.append(w)


      while True:
        stepStatus = reader.BeginStep()
        if stepStatus:
          stream_data = a2_reader.Get(varname)
          dq.put_nowait((stream_data, 
                         reader.CurrentStep()))
          reader.EndStep()
        else:
          break
      
      worker.join()
      dq.join()


To access the many cores available, ``processor`` needs to be run as an MPI program under control of
``mpi4py.futures``: ``run -n NP -m mpi4py.futures processor.py``.
The number of MPI ranks should be equal to the workers requested in the PoolExecutors,
``NP == NF + NA``. Lines 12 - 29 show the setup. Two ``MPIPoolExecutors`` are instantiated, ``executor_fft`` defines an 
execution space for Fourier Transformations and ``executor_anl`` defines an execution space for the
analysis kernels. Dispatching Fourier Transformations and data analysis tasks is handled by ``task_list_spectral``.
Then ``a2_reader`` is instantiated with a configuration mirroring the one of the writer. After defining a 
Queue for Inter-process communication a series of worker threads is started. In the main loop ``a2_reader``
consumes the data stream and the data packets are inserted in the queue. The array of worker tasks 
subsequently read data from the queue and dispatch it to the data analysis code.

The actual data analysis code is done in cython kernels which are described in a later subsection.
While the low-level implementation of Eqs. (:ref:`eq-S`) - (:ref:`eq-R`) is in cython, ``Delta`` abstracts
them through the ``task`` class. Sans initialization the relevant class structure looks like this:

.. code:: python
   :linenos:

   class task():
   ...
   def calc_and_store(self, data, **kwargs):
     try:
       result = self.kernel(data, **kwargs)
       self.storage_backend.store(data, tidx)
      
   def submit(self, executor, data, tidx):
     ...
     _ = [executor.submit(self.calc_and_store, data, ch_it, tidx) for ch_it in (self.get_dispatch_sequence())]


The actual call to the analysis kernel happens in ``calc_and_store``. This member function also handles 
storage to the data backend. Above we only pass ``tidx`` as metadata to ``store``, for real-world application additional
meta-data is passed. Once ``calc_and_store`` returns the data has been analyzed and stored. The member 
function ``submit`` launches ``calc_and_store``. It allows for additionaly granularity by specifying 
``ch_it``, an iterateable over channel pairs :math:`X` and :math:`Y`. 

Adding another level of abstraction, the different tasks are grouped together by a ``task_list`` class:

.. code:: python
   :linenos:

   from scipy.signal import stft

   class task_list():

   def submit(self, data, tidx):
     fft_future = self.executor_fft.submit(stft, data, **kwargs)

     for task in self.task_list:
       task.submit(self.executor_anl, fft_future.result(), tidx)


This bare-bones example demonstrates the utilization of the two PoolExecutors. All FFTs, implemented 
by ``scipy.signal.stft``, are executed on ``executor_fft``. Assuming a single-threaded implementation,
the number of queue worker processes should correspond to the number of processes used to instantiate
this executor, ``n_thr == NF``.  Our experiments show that reserving CPU resources for the Fourier 
Transformation through a separate PoolExecutor significantly decreases the total processing time.
After a data chunk has been Fourier Transformed, it is distributed to the analysis routines.


``Delta`` utilizes the ``futures`` interface defined in PEP 3148 Since however both Cori and ADIOS2 are
designed for MPI applications we use the ``mpi4py`` [mpi4py]_ implementation. Being a standard interface,
other implemenations like ``concurrent.futures`` can readily be used. Note that the reason why calls to
``executor.submit`` are enacpsulated in classes is to pass kernel-dependent keyword arguments. The 
Python Standard Library defines the inerface as :code:`executor.submit(fn, *args **kwargs)`. We are passing 
an executor to the ``submit`` wrapper call and class-specific information is passed to ``kwargs``.


Explored alternative architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides ``mpi4py`` we also explored executing ``task.calc_and_store`` calls on a ``Dask`` [dask]_ cluster.
Exposing ``concurrent.futures``-compatible interface, both libraries can be interchanged with little
work. Running on a single node we found little difference in execution speed. However once the
dask-distributed cluster was deployed on multiple nodes we observed a significant slowdown due to
network traffic overhead. We did not investigate this problem any further.

As an alternative to using a queue with threads, we also explored using asynchronous I/O. In this
scenario, the main task would define a coroutine receiving the data time chunks and a second one
dispatching them to an executor. In our tested implementation, the coroutines would run in a main loop
and communicate via a queue. Our experiments showed no measurable difference against a threaded
implementation. On the other hand, the threaded implementation fits more naturally in the multi-processing
design approach.



Using data analysis codes  ``Delta``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the most general case, data analysis can be formulated as applying a transformation :math:`F` 
to some data :math:`d`,

.. math::
   y = F(d; \lambda_1, \ldots, \lambda_n),
   :label: eq-transf


given some parameters :math:`\lambda_1 \ldots \lambda_n`. Translating the relation between the 
function and the data into an object-oriented setting is not always ambiguous. The approach taken by
packages such as scipy or scikit-learn is to implement a transformatio :math:`F` as a class
and interface to data through its member functions. Taking Principal Component Analysis in 
scikit-learn as an example, the default way of working with it is

.. code:: python

   from sklearn.decomposition import PCA 
   X = np.array([...])
   pca = PCA(n_components=2)
   pca.fit(X)

This approach has proven itself useful and is the common way of organizing libraries. ``Delta``
deviates slightly from this approach and calls transformations in the ``calc_and_store`` member
function of the ``task_ecei`` class. The specific kernel to be called is set in the constructor:

.. code:: python
   
   from kernels import kernel_crossphase, kernel_crosspower, ...

   class task():
      def __init__(self, cfg):
         ...
      if (cfg["analysis"] == "cross-phase"):
         self.kernel = kernel_crossphase
      elif (cfg["analysis"] == cross-power"):
         self.kernel = kernel.crosspower

      ...

     def calc_and_store(self, data, ...):
        ...
        result = self.kernel(data, ...)


At the time of writing, ``Delta`` only implements a workflow for ECEi data and this design choice 
minimizes the number of classes present in the framework. Grouping the data analysis methods by 
diagnostic also allows to collectively execute diagnostic-specific pre-transformations that are best
performed after transfer to the processing site. One may wish for example to distribute calculations of
the 18336 channel pairs among multiple instances of ``task_ecei``. This approach lets us seamlessly
do that.



Performance analysis
--------------------

A goal of the ``Delta`` framework is to enable near real-time intra-shot analysis for tasks that are
commonly performed post-shot in a batch-wise manner. Based on the architecture sketched in Figure :ref:`fig-sw-arch`,
the different knobs that can be turned in the different components are

generator
  ``get_batch`` Can be improved to maximize load-speed when loading pre-recorded data. In a streaming setting
  this should be as fast as the diagnostic producing the data

ADIOS2 communication
  Data needs to be transferred as fast as possible from the data staging site to the HPC facility

processor
  Data analysis routines should be optimized for the HPC environment. Overhead can also be expected when
  storing large amounts of data from multiple sources simultaneously in the backend.
  
For development purposes, the generator stages pre-recorded data for transport. Without resorting to
parallel data-access patterns and multiple writers data loading and staging is limited by the I/O 
resources of the KSTAR DTN. As such, this is not a candidate for performance optimization. Next we present
a performance analysis of the data transfer methods and of the data analysis kernels implemented in
``Delta``.


Data Transfer Methods
^^^^^^^^^^^^^^^^^^^^^

We need to test dataman transfers.


Entering NERSC, data is moved from the DTN to Cori using ADIOS2 SST. Internally, SST can take advantage 
of RDMA. Data is transferred with approximately 35 MByte/sec. As soon as data is received on Cori, the
analysis starts. To transfer all time chunks from the DTN to Cori it takes about 200secs. After that,
the time to perform the FFT is reduced to about 0.8secs from about 3 secs.




Data Analysis Kernels 
^^^^^^^^^^^^^^^^^^^^^

Foreshadowed in the code-example above, ``Delta`` implements data analysis routines as computational
kernels. To fully utilize the multi-threading capabilities of Cori, all currently used kernels 
are implemented using cython. The coherence, Eq. (:ref:`eq-C`), is implemented as


.. code:: python

  @cython.boundscheck(False)
  @cython.wraparound(False)
  @cython.cdivision(True)
  def kernel_coherence_64_cy(cnp.ndarray[cnp.complex128_t, 
                                         ndim=3] data, 
                                         ch_it, 
                                         fft_config):
      cdef size_t num_idx = len(ch_it)      # Length of index array
      cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
      cdef size_t num_bins = data.shape[2]  # Number of ffts
      cdef size_t ch1_idx, ch2_idx
      cdef size_t idx, nn, bb # Loop variables
      cdef double complex Sxx, Syy, _tmp
      
      cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr =
         np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], 
                  dtype=np.uint64)
      cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = 
         np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], 
                  dtype=np.uint64)
      cdef cnp.ndarray[cnp.float64_t, ndim=2] result = 
         np.zeros([num_idx, num_fft], dtype=np.float64)

      with nogil: 
          for idx in prange(num_idx, schedule=static):
              ch1_idx = ch1_idx_arr[idx]
              ch2_idx = ch2_idx_arr[idx]
  
              for nn in range(num_fft):
                  _tmp = 0.0
                  for bb in range(num_bins):
                      Sxx = data[ch1_idx, nn, bb] * 
                        conj(data[ch1_idx, nn, bb])
                      Syy = data[ch2_idx, nn, bb] * 
                        conj(data[ch2_idx, nn, bb])
                      _tmp +=  data[ch1_idx, nn, bb] * 
                               conj(data[ch2_idx, nn, bb]) / 
                               csqrt(Sxx * Syy)
  
                  result[idx, nn] = creal(cabs(_tmp)) 
                                   / num_bins
      return(result) 

The arguments passed to the kernel are the three-dimensional array of Fourier Coefficients,
``ch_it`` - an iterator over the channel lists, and ``fft_config`` - a dictionary of parameters used 
for the Fourier Transformation. While the data stream produced by the ECEi diagnostic is only 
two-dimensional, ``fft_data`` is three-dimensional as we use a sliding-window Fourier Transformation.
The second argument ``ch_it`` is an iterator over a list of channel pairs, defining the pairs for which
to calculate :math:`C`. After defining the output array and temporary data, the kernel defines a 
section where it discards the global interpreter lock. This is crucial to enable the compiler to 
generate multi-threaded code for the section. 

The ranges of the three for loops within these section decrease by order of magnitude. 
For a full dataset, each kernel iterators over 18336 distinct channel pairs, 512 to 1024 Fourier 
Coefficients and 19 to 38 sliding window bins. Data caching occurs after each for-loop header.
Furthermore are the channel-pairs a tuple-like data structure and sorted by the first item,
``ch1_idx`` in the case above. This sorting allows to better utilize the CPU cache. The preferred
compiler on Cori is the cray compilier, which is a wrapper for the Intel compiler. Since this
compiler stack is incompatible with mpi4py on Cori, we choose to use the Gnu compiler for the 
diagnostic kernels as well.


.. figure:: plots/kernel_performance.png
   :scale: 100%

   Runtime of the multi-threaded kernels for coherence :math:`C`, cross-power :math:`S` and cross-phase :math:`P` compared against numpy implementations. :label:`kernel-perf`

Figure :ref:`kernel-perf` shows the performance gained by using multi-threaded kernels 
over kernels implemented in numpy. Running with a single-thread, the coherence kernel written in cython



Conclusions and future work
---------------------------


Next generation HPC facilities such as Perlmutter will be equipped with Nvidia Ampere GPUs. 

RMCs ECEi disruption detection paper

Making delta adaptiv:
 * Allow other diagnostic data to be transferred
 * Real-time detection of interesting features, coupled to compression
 * ECEi has large view, maybe we need fewer channels


Acknowledgements
----------------
The authors would like to acknowledge the excellent technical support from engineers and developers
at the National Energy Research Scientific Computing Center in developing delta. This work used
resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. DOE Office of
Science User Facility operated under Contract No. DE-AC02-05CH11231.

References
----------

.. [PPY] https://www.plasmapy.org

.. [OMF] O. Meneghini, S.P. Smith, L.L. Lao et al. *Integrated modeling applications for tokamak experiments with OMFIT*
         Nucl. Fusion **55** 083008 (2015)

.. [Ent18] S. Entler, J. Horacek, T. Dlouhy and V. Dostal *Approximation of the economy of fusion energy*
           Energy 152 p. 489 (2018)

.. [D3D] DIII-D http://www.ga.com/diii-d

.. [NSTX] NSTX https://www.pppl.gov/nstx

.. [KSTAR] KSTAR Tokamak https://www.nfri.re.kr/kor/index

.. [AUG] ASDEX Upgrade https://www.ipp.mpg.de/16195/asdex

.. [MAST] Mega Amp Spherical Tokamak https://ccfe.ukaea.uk/research/mast-upgrade/

.. [TCV] https://www.epfl.ch/research/domains/swiss-plasma-center/research/tcv/research_tcv_tokamak/

.. [Cos74] A.E Costley, R.J. Hastie, J.W.M. Paul, and J. Chamberlain *Electron Cyclotron Emission from a Tokamak Plasma: Experiment and Theory*
           Phys. Rev. Lett. 33 p. 758 (1974).

.. [Yun14] G.S. Yun, W. Lee, M.J. Choi et al. *Quasi 3D ECE imaging system for study of MHD instabilities in KSTAR*
           Rev. Sci. Instr. 85 11D820 (2014)
           http://dx.doi.org/10.1063/1.4890401

.. [Bal17] E.A. Baltz, E. Trask, M. Binderbauer et al. *Achievement of Sustained Net Plasma Heating in a Fusion Experiment with the Optometrist Algorithm*
           Sci. Reports 6425 (2017)
           https://doi.org/10.1038/s41598-017-06645-7

.. [Bel18] V. A. Belyakov and A. A. *Kavin Fundamentals of Magnetic Thermonuclear Reactor Design*
           Chapter 8 Woodhead Publishing Series in Energy

.. [Yun10] G. S. Yun, W. Lee, M. J. Choi et al. *Development of KSTAR ECE imaging system for measurement of temperature fluctuations and edge density fluctuations*
           Rev. Sci. Instr. 81 10D930 (2010)
           https://dx.doi.org/10.1063/1.3483209

.. [Cho17] M. J. Choi, J. Kim, J.-M. Kwon et al. *Multiscale interaction between a large scale magnetic island and small scale turbulence*
           Nucl. Fusion **57** 126058 (2017)
           https://doi.org/10.1088/1741-4326/aa86fe

.. [cori] https://docs.nersc.gov/systems/cori/

.. [nerscdtn] https://docs.nersc.gov/systems/dtn/

.. [iperf] https://iperf.fr

.. [adios2] https://adios2.readthedocs.io/en/latest/index.html

.. [PEP3148] https://www.python.org/dev/peps/pep-3148/

.. [mpi4py] https://mpi4py.readthedocs.io/en/stable/

.. [dask] https://dask.org

.. [FFT] G. Heinzel, A. Rüdiger, R. Schilling, *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows*
         Max Planck Institute für Gravitationsphysik (Albert-Einstein-Institut) Feb. 2002