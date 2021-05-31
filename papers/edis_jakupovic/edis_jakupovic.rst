:author: Edis Jakupovic
:email: ejakupov@asu.edu
:institution: Arizona State University

:author: Oliver Beckstein
:email: obeckste@asu.edu
:institution: Arizona State University
:corresponding:

:bibliography: references

.. definitions (like \newcommand)

.. |Calpha| replace:: :math:`\mathrm{C}_\alpha`
.. |tinit_top| replace:: :math:`t^\text{init\_top}`
.. |tinit_traj| replace:: :math:`t^\text{init\_traj}`
.. |tcomp| replace:: :math:`t^{\text{compute}}`
.. |tIO| replace:: :math:`t^\text{I/O}`
.. |tcomm| replace:: :math:`t^\text{comm\_gather}`
.. |twait| replace:: :math:`t^\text{wait}`
.. |ttotal| replace:: :math:`t^\text{total}`
.. |Ncores| replace:: :math:`M`
.. |r(t)| replace:: :math:`\mathbf{r}(t)`


---------------------------------------------------------------------------------------------------------
MPI-parallel Molecular Dynamics Trajectory Analysis with the H5MD Format in the MDAnalysis Python Package
---------------------------------------------------------------------------------------------------------

.. class:: abstract

   MDAnalysis is a widely used Python library that can read and write over 20 popular MD file formats. With the increasing size of molecular dynamics (MD) simulation files, parallel analysis is becoming a necessity for the efficient use of time and HPC resources. For any task-based approach to parallel analysis, file I/O typically becomes the limiting factor of overall analysis speed. Our previous feasibility study suggested that parallel reading via MPI-IO and HDF5 can lead to good scaling, so we have implemented a parallel MPI-IO capable HDF5-based file format trajectory reader into MDAnalysis, H5MDReader, that adheres to H5MD (HDF5 for Molecular Dynamics) specifications. We benchmarked its parallel file reading capabilities on three HPC clusters: ASU Agave, SDSC Comet, and PSC Bridges. The benchmark consisted of a simple split-apply-combine scheme of an IO bound task that split a 90k frame (113GB) trajectory into N chunks for N processes, where each process performed an RMSD calculation on their chunk of data, and then gathered the results back to the root process. We found maximum IO speedups at 2 full nodes, with Agave showing 20x, while Bridges and Comet capped out at 4-5x speedup. On the other hand, the computation time of the RMSD calculation scaled well on all three HPC resources, with a maximum speedup on Comet of 373x on 384 cores. Therefore, for a compute bound task, our implementation would likely scale very well, however the file I/O still seems to impose a bottleneck on the total scaling for the I/O bound task. To investigate MPI rank competition, we increased the stripe count on Bridge’s and Comet’s Lustre filesystem up to 96. We found marginal IO scaling improvements of 1.2x on up to 4 full nodes. To investigate how the amount of data read at each frame affects I/O performance, implemented a feature into the H5MDReader where only the necessary coordinates for the computation task are read from the file, which resulted in a fixed improvement of approximately 5x on all three HPCs. Furthermore, we investigated how frontloading all the file IO by loading the trajectory into memory prior to the computation, rather than iterating through each timestep, affected the IO performance. This resulted in an improvement on Agave of up to 10x on up to 112 cores. With respect to baseline performance, this resuleted in a 91x speedup on 112 cores on Agave. Upon further investigation, however, we found similar improvements in performance with respect to baseline results by altering the HDF5 chunk layout of the file. This suggests that the boost in performance in the previous case was due to I/O issues of intial inefficient chunk layout that reading the trajectory into memory in one go was able to avoid. We also found that applying HDF5's built in gzip compression does not affect parallel I/O performance at higher core counts while providing ~33% smaller files.

.. class:: keywords

   Molecular Dynamics Simulations, High Performance Computing, Python, MDAnalysis, HDF5, H5MD, MPI I/O





Introduction
============

As HPC resources continue to increase, the size of molecular dynamics (MD) simulation files are now commonly terabytes in size, making serial analysis of these trajectory files impractical. Parallel analysis is a necessity for the efficient use of both HPC resources and a scientist’s time. MDAnalysis is a widely used Python library that can read and write over 20 popular MD file formats while providing the same user-friendly interface :cite:`Michaud-Agrawal:2011,Gowers:2016`. Previous work that focused on developing a task-based approach to parallel analysis found that an IO bound task only scaled to 12 cores due to a file IO bottleneck :cite:`Fan:2019`. Our previous feasibility study suggested that parallel reading via MPI-IO and HDF5 can lead to good scaling although it only used a reduced size custom HDF5 trajectory and did not provide a usable implementation of a true MD trajectory reader :cite:`Khoshlessan:2020`.

H5MD, or "HDF5 for molecular data", is an HDF5-based file format that is used to store MD simulation data, such as particle coordinates, box dimensions, and thermodynamic observables :cite:`Buyl:2014`. HDF5 is a structured, binary file format that organizes data into 2 objects: groups and datasets, which follows a hierarchical, tree-like structure, where groups represent nodes of the tree, and datasets represent the leaves :cite:`Collette:2014`. The HDF5 library can be built on top of a message passing interface (MPI) implementation so that a file can be accessed in parallel on a parallel filesystem such as Lustre or BeeGFS. We implemented a parallel MPI-IO capable HDF5-based file format trajectory reader into MDAnalysis, H5MDReader, that adheres to H5MD specifications. H5MDReader interfaces with h5py, a high level Python package that provides a Pythonic interface to the HDF5 format such that accessing a file in parallel is as easy as passing a keyword argument into ``h5py.File``, and all of parallel disk access occurs under the hood.

We benchmarked H5MDReader's parallel reading capabilities with MDAnalysis on three HPC clusters: ASU Agave, SDSC Comet, and PSC Bridges. The benchmark consisted of a simple split-apply-combine scheme of an IO-bound task that split a 90k frame (113GB) trajectory into n chunks for n processes, where each process a task on their chunk of data, and then gathered the results back to the root process. For the computational task, we computed the time series root mean squared distance (RMSD) of the positions of the alpha carbons in the protein to their initial coordinates at the first frame of the trajectory. The RMSD calculation is not only a very common task performed to analyze the dynamics of the structure of a protein, but more importantly is a very fast computation that is heavily bounded by how quickly data can be read from the file. Therefore it provided an excellent analysis candidate to test the I/O capabilities of H5MDReader.

Across the three HPC clusters tested, the benchmarks were done on both a BeeGFS and Lustre parallel filesystem which is highly suited for multi-node MPI parallelization. We tested various algorithmic optimizations for our benchmark, including altering the stripe count, loading only necessary coordinate information with ``numpy.Masked\_arrays``, and front loading all I/O by loading the entire trajectory into memory prior to the RMSD calculation.

We also tested the effects of HDF5 file chunking and file compression on I/O performance. An HDF5 file's datasets can be stored either contiguously on disk, or scattered accross the disk in different locations in *chunks*. These chunks must be defined on intialization of the dataset, and for any element to be read from a chunk, the entire chunk must be read. In general we found that altering the stripe count and loading only necessary coordniates via masked arrays provided little improvement in benchmark times. Loading the entire trajectory into memory in one pass instead of iterating through, frame by frame, showed the greatest improvement in performance. This was compounded by our results with HDF5 chunking. Our baseline test file was auto-chunked with ``h5py``s auto-chunking algorithm. When we recast the file into a contiguous form and a custom, optimized chunk layout, we saw improvements in serial I/O on the order of 10x. Additionally, our results from applying gzip compression to the file show no loss in performance at high processor counts, indicating H5MD files can be compressed without fear of losing performance in parallel analysis tasks.



Methods
=======

We implemented a simple split-apply-combine parallelization algorithm that divides the number of frames in the trajectory evenly among all available processes. Each process receives a unique start and stop for which to iterate through their section of the trajectory and compute the RMSD at each frame. The data files used in our benchmark included a topology file ``YiiP_system.pdb`` with 111,815 atoms, and a trajectory file ``YiiP_system_9ns_center100x.h5md`` with 90100 frames. The trajetory data file was converted on the fly with MDAnalysis with different HDF5 chunking arrangements and compression settings. Table 1 gives all of the H5MD files benchmarked with how they are identified in this paper as well as their corresponding file size.

.. raw:: latex

   \begin{table}
   \begin{tabular}{||c | c | c ||}
    \hline
    \textbf{name} & \textbf{format} & \textbf{file size (GB)} \\ [0.5ex]
    \hline\hline
    H5MD-default     & H5MD       & 113    \\
    \hline
    H5MD-chunked     & H5MD       & 113    \\
    \hline
    H5MD-contiguous  & H5MD       & 113    \\
    \hline
    H5MD-gzipx1      & H5MD       & 77     \\
    \hline
    H5MD-gzipx9      & H5MD       & 75     \\ [0.75ex]
    \hline
   \end{tabular}
   \caption{Data files benchmarked on all three HPCS. \textbf{name} is the name that is used to identify the file in this paper. \textbf{format} is the format of the file, and \textbf{file size} gives the size of the file in gigabytes.
      \textbf{H5MD-default} original data file written with pyh5md which uses h5py's auto-chunking algorithm. \textbf{H5MD-chunked} is the same file but written with chunk size (1, n atoms, 3) and \textbf{H5MD-contiguous} is the
      same file but written with no HDF5 chunking. \textbf{H5MD-gzipx1} and \textbf{H5MD-gzipx9} have the same chunk arrangement as \textbf{H5MD-chunked} but are written with gzip compression where 1 is the lowest level of compression
      and 9 is the highest level.}
   \end{table}

In order to obtain detailed timing information we instrumented code as follows:

.. code-block:: python
   :linenos:

   class timeit(object):
       def __enter__(self):
           self._start_time = time.time()
           return self

       def __exit__(self, exc_type, exc_val, exc_tb):
           end_time = time.time()
           self.elapsed = end_time - self._start_time
           # always propagate exceptions forward
           return False

The ``timeit`` class was used as a context manager to record how long our benchmark spent on particular lines of code. Below, we give example code of how each benchmark was performed:

.. code-block:: python
   :linenos:

   import MDAnalysis as mda
   from MDAnalysis.analysis.rms import rmsd
   from mpi4py import MPI
   import numpy as np

   def benchmark(topology, trajectory):
       with timeit() as init_top:
           u = mda.Universe(topology)
       with timeit() as init_traj:
           u.load_new(trajectory,
                      driver="mpio",
                      comm=MPI.COMM_WORLD)
       t_init_top = init_top.elapsed
       t_init_traj = init_traj.elapsed
       CA = u.select_atoms("protein and name CA")
       x_ref = CA.positions.copy()

       total_io = 0
       total_rmsd = 0
       rmsd_array = np.empty(bsize, dtype=float)
       for i, frame in enumerate(range(start, stop)):
           with timeit() as io:
               ts = u.trajectory[frame]
           total_io += io.elapsed
           with timeit() as rms:
               rmsd_array[i] = rmsd(CA.positions,
                                    x_ref,
                                    superposition=True)
           total_rmsd += rms.elapsed

       with timeit() as wait_time:
           comm.Barrier()
       t_wait = wait_time.elapsed

       with timeit() as comm_gather:
           rmsd_buffer = None
           if rank == 0:
               rmsd_buffer = np.empty(n_frames,
                                      dtype=float)
           comm.Gatherv(sendbuf=rmsd_array,
                        recvbuf=(rmsd_buffer,
                                 sendcounts),
                        root=0)
       t_comm_gather = comm_gather.elapsed

The time |tinit_top| records the time it takes to load a ``universe`` from the topology file. |tinit_traj| records the time it takes to open the trajectory file. The HDF5 file is opened with the ``mpio`` driver and the ``MPI.COMM_WORLD`` communicator to ensure the file is accessed in parallel via MPI I/O. It's important to separate the topology and trajectory initialization times, as the topology file is not opened in parallel and represents a fixed cost each process must pay to open the file.  |tIO| represents the time it takes to read the data for each frame into the corresponding ``MDAnalysis.Universe.trajectory.ts`` attribute. MDAnalysis reads data from MD trajectory files one frame, or "snapshot" at a time. Each time the ``u.trajectory[frame]`` is iterated through, MDAnalysis reads the file and fills in numpy arrays corresponding to that timestep. Each MPI process runs an identical copy of the script, but receives a unique ``start`` and ``stop`` variable such that the entire file is read in parallel. |tcomp| gives the total RMSD computation time. |twait| records how long each process waits before the results are gathered with ``comm.Gather()``. Gathering the results is done collectively by MPI, which means all processes must finish their iteration blocks before the results can be returned. Therefore, it's important to measure |twait| as it represents the existence of "straggling" processes. If one process takes substantially longer than the others to finish its iteration block, all processes are slowed down. |tcomm| measures the time MPI spends communicating the results from each process back to the root process.

We applied this benchmark scheme to H5MD test files on Agave, Bridges, and Comet. We also tested 3 algorithmic optimizations: Lustre file striping, loading the entire trajectory into memory, and using ``Masked Arrays`` to only load the alpha carbon coordinates required for the RMSD calculation. For striping, we ran the benchmark on Bridges and Comet with a file stripe count of 48 and 96. For the into memory optimization, we used ``MDAnalysis.Universe.transfer_to_memory()`` to read the entire file in one go and pass all file I/O to the HDF5 library. For the masked array optimization, we allowed ``u.load_new()`` to take a list or array of atom indices as an argument, ``sub``, so that the ``MDAnalysis.Universe.trajectory.ts`` arrays are instead initialized as ``ma.masked_array``'s and only the indices corresponding to ``sub`` are read from the file.

Performance was quantified by measuring the I/O timing returned from the benchmarks, and strong scaling was assessed by calculating the speedup :math:`S(N) = t_{1}/t_{N}` and the efficiency :math:`E(N) = S(N)/N`.


Results and Discussion
======================

Baseline Benchmarks
-------------------

We first ran benchmarks with the simplest parallelization scheme of splitting the frames of the trajectory evenly among all participating processes. The H5MD file involved in the benchmarks was written with ``pyh5md``, a python library that can easily read and write H5MD files :cite:`Buyl:2014`. The datasets in the data file were chunked automatically by the auto-chunking algorithm in ``h5py``. File I/O remains the largest contributor to the total benchmark time, as shown by Figure :ref:`fig:components-vanilla` (A). Figure :ref:`fig:components-vanilla` (B, D-F) also show that the initialization, computation, and MPI communication times are negligible with regards to the overall analysis time. |twait|, however, becomes increasingly relevant as the number of processes increases (Figure `fig:components-vanilla` C), indicating a growing variance in the iteration block time across all processes. Although the total benchmark time continues to decrease as the number of processes increases to over 100, the maximum total speedup observed is only 15x (Figure `fig:scaling-vanilla` A,B).

.. figure:: figs/components-vanilla.pdf

   Benchmark timings breakdown for the ASU Agave, PSC Bridges, and SDSC Comet HPC clusters. The benchmark was run on up to 4 full nodes on each HPC, where N\_processes was 1, 28, 56, and 112 for Agave and Bridges, and 1, 24, 48, and 96 on Comet. The ``H5MD-default`` file was used in the benchmark, where the trajectory was split in N chunks for each corresponding N process benchmark. Points represent the mean over three repeats with the standard deviation shown as error bars.
   :label:`fig:components-vanilla`

.. figure:: figs/scaling-vanilla.pdf

   Strong scaling I/O performance of the RMSD analysis task of the ``H5MD-default`` data file on Agave, Bridges, and Comet. N Processes ranged from 1 core, to 4 full nodes on each HPC, and the number of trajectory blocks was equal to the number of processes involved.
   :label:`fig:scaling-vanilla`

Effects of Algorithmic Optimizations on File I/O
------------------------------------------------
We tested three optimizations aimed at shortening file I/O time for the same data file. To investigate MPI rank competition, we increased the stripe count on Bridge’s and Comet’s Lustre filesystem up to 96. We found marginal IO scaling improvements of 1.2x on up to 4 full nodes (not shown). For any analysis task, not all coordinates in the trajectory may be necessary for the computation. For example, in our analysis test case, the RMSD was calculated for only the alpha carbons of the protein backbone, therefore the coordinates of all other atoms read from the file is essentially wasted I/O. To circumvent this issue, we impelemented the use of NumPy ``ma.masked_array``s, where the arrays of coordinate data are instead initialized as masked arrays that only fill data from selected coordinate indices. We found that Bridges showed the best scaling with the masked array implementation, with a total scaling of 23x at 4 full nodes as seen in Figure :ref:`scaling-masked` B. Agave showed a maximum scaling of 11x at 2 full nodes, while Comet showed 5x scaling at 4 full nodes (Figure :ref:`scaling-masked` B). In terms of absolute timings, Agave showed longer I/O time with masked arrays, with I/O time increasing when using masked arrays. For Bridges and Comet, we observed an approximate 5x speedup in I/O time for the masked array case when compared to the baseline benchmark.

.. figure:: figs/components-masked.pdf

   Benchmark timings breakdown for the ASU Agave, PSC Bridges, and SDSC Comet HPC clusters for the ``masked_array`` optimization technique. The benchmark was run on up to 4 full nodes on each HPC, where N processes was 1, 28, 56, and 112 for Agave and Bridges, and 1, 24, 48, and 96 on Comet. The ``H5MD-default`` file was used in the benchmark, where the trajectory was split in N chunks for each corresponding N process benchmark. Points represent the mean over three repeats with the standard deviation shown as error bars.
   :label:`fig:components-masked`

.. figure:: figs/scaling-masked.pdf

   Strong scaling performance of the RMSD analysis task with the ``masked_array`` optimization technique. The benchmark used the ``H5MD-default`` data file on Agave, Bridges, and Comet. N Processes ranged from 1 core, to 4 full nodes on each HPC, and the number of trajectory blocks was equal to the number of processes involved.
   :label:`fig:scaling-masked`

With an MPI implementation, processes participating in parallel I/O communicate with one another. It is commonly understood that repeated, small file reads performs worse than a large, contiguous read of data. With this in mind, we tested this concept in our benchmark by loading the entire trajectory into memory prior to the RMSD task. Modern super computers make this possible as they contain hundreds of GB of memory per node. Figure :ref:`components-mem` shows that file I/O remains the largest contributor to the benchmark time. Interestingly, we found that the |twait| does not increase as the number of processes increases as in the other benchmark cases (Figure :ref:`components-mem` C). This indicates that there are no straggling processes, and all processes take approximately the same time to load their section of data. Comet showed the worst improvment in I/O, with a speedup of 2x with respect to the baseline benchmarks. In terms of absolute time, Agave showed the most substantial increase in performance, where in the single process case the baseline benchmark time was 4648s (Figure :ref:`scaling-vanilla` A) and 911s in the single process into-memory benchmark (Figure :ref:`scaling-mem` A). In the 4 full node case, Agave showed a 91x speedup with respect to the baseline benchmark performance (4658s to 73s at 112 cores). This gives strong evidence that the default access pattern of iterating through each frame was inefficient as opposed to loading the entire trajectory into memory in one go.

.. figure:: figs/components-mem.pdf

   Benchmark timings breakdown for the ASU Agave, PSC Bridges, and SDSC Comet HPC clusters for the loading-into-memory optimization technique. The benchmark was run on up to 4 full nodes on each HPC, where N processes was 1, 28, 56, and 112 for Agave and Bridges, and 1, 24, 48, and 96 on Comet. The ``H5MD-default`` file was used in the benchmark, where the trajectory was split in N chunks for each corresponding N process benchmark. Points represent the mean over three repeats with the standard deviation shown as error bars.
   :label:`fig:components-mem`

.. figure:: figs/scaling-mem.pdf

   Strong scaling I/O performance of the RMSD analysis task with the loading-into-memory optimization technique. The benchmark used the ``H5MD-default`` data file on Agave, Bridges, and Comet. N Processes ranged from 1 core, to 4 full nodes on each HPC, and the number of trajectory blocks was equal to the number of processes involved.
   :label:`fig:scaling-mem`


Effects of HDF5 Chunking on File I/O
------------------------------------
The speed at which a file can be read from disk depends not only on access pattern, but also the file's layout on disk. We rewrote the H5MD-default test file on the fly with MDAnalysis and tested two cases: one in which the file is written with no chunking applied (H5MD-contiguous), and one in which we applied a custom chunk layout to match the access pattern on the file (H5MD-chunked). Our benchmark follows a common MD trajecotry analysis scheme in that it iterates through the trajectory one frame at a time. Therefore, we applied a chunk shape of ``(1, n atoms, 3)`` which matched exactly the shape of data to be read at each iteration step. First, we tested how the chunk layout affects baseline serial I/O performance for the file. We found I/O performance strongly depends on the layout of the file on disk. The auto-chunked H5MD-default file I/O time was 4101s, while our custom chunk layout resulted in an I/O time of 460s (Figure :ref:`serial-IO`). So, we effectively saw a 10x speedup from optimizing the chunk layout alone. To see what effect the chunk layout had on parallel I/O performance, we repeated our benchmarks on Agave but with the H5MD-chunked and H5MD-contiguous data files. For the serial one process case, we found a similar result in that the I/O time was dramatically increased with an approximate 10x speedup for both the contiguous and chunked file, with respect to the baseline benchmark (Figure :ref:`components-chunk` A). The rest of the timings remained unaffected (Figure :ref:`components-chunk` B-F). Although the absolute total benchmark time is much improved (Figure :ref:`scaling-chunk` A), the scaling remains challenging, with a maximum observed speedup of 12x for the contiguous file (Figure :ref:`scaling-chunk` B).

.. figure:: figs/serial-IO.pdf

   Serial I/O time for H5MD-default, H5MD-contiguous, and H5MD-chunked data files. Each file contained the same data (113GB, 90100 frames) but was written with a different HDF5 chunk arrangement, as outlined in Table :ref:`tab:files`. Each bar represents the mean of 5 repeat benchmark runs, with the standard deviation shown as error bars.
   :label:`fig:serial-IO`

.. figure:: figs/components-chunk.pdf

   Benchmark timings breakdown on ASU Agave for the three chunk arrangements tested. The benchmark was run on up to 4 full nodes on each HPC, where N processes was 1, 28, 56, and 112. \textbf{H5MD-default} was auto-chunked by ``h5py``. \textbf{H5MD-contiguous} was written with no chunking applied, and \textbf{H5MD-chunked} was written with a chunk shape of ``(1, n atoms, 3)``. The trajectory was split in N chunks for each corresponding N process benchmark. Points represent the mean over three repeats with the standard deviation shown as error bars.
   :label:`fig:components-chunk`

.. figure:: figs/scaling-chunk.pdf

   Strong scaling I/O performance of the RMSD analysis task with various chunk layouts tested on ASU Agave. N Processes ranged from 1 core, to 4 full nodes, and the number of trajectory blocks was equal to the number of processes involved.
   :label:`fig:scaling-chunk`


Effects of HDF5 GZIP Compression on File I/O
--------------------------------------------
HDF5 files offer the ability to compress the files. To see how compression affected parallel I/O, we tested HDF5's gzip compression with a minimum setting of 1 and a maximum setting of 9. In the serial 1 process case, we found that I/O performance is slightly hampered, with I/O times approximately 4x longer with compression applied (Figure :ref:`scaling-gzip` A), however at increasing number of processes this difference disappears (Figure :ref:`scaling-gzip` A and Figure :ref:`components-gzip` A). This shows a clear benefit of applying gzip compression to a chunked HDF5 file for parallel analysis tasks, as the compressed file is ~2/3 the size of the original. Additionaly we found speedups of up to 36x on 2 full nodes for the compressed data file benchmarks (Figure :ref:`scaling-gzip` B).

.. figure:: figs/components-gzip.pdf

   Benchmark timings breakdown on ASU Agave for the minimum gzip compression 1 and maximum gzip compression 9. The benchmark was run on up to 4 full nodes on each HPC, where N processes was 1, 28, 56, and 112. The trajectory was split in N chunks for each corresponding N process benchmark. Points represent the mean over three repeats with the standard deviation shown as error bars.
   :label:`fig:components-gzip`

.. figure:: figs/scaling-gzip.pdf

   Strong scaling I/O performance of the RMSD analysis task with minimum and maximum gzip compression applied. N Processes ranged from 1 core, to 4 full nodes, and the number of trajectory blocks was equal to the number of processes involved.
   :label:`fig:scaling-gzip`

Conclusions
===========

MDAnalysis is a Python library for the analysis of molecular dynamics simulations that provides a uniform user interface for many different MD file formats. The growing size of trajectory files demands parallelization of trajectory analysis, however file I/O has become a bottleneck in the workflow of analyzing simulation trajectories. Our implemententaion an HDF5-based file format trajectory reader into MDAnalysis can perform parallel MPI I/O, and our benchmarks on various national HPC environments show that speed-ups on the order of 20x for 48 cores are attainable. Scaling up to achieve higher parallel data ingestion rates remains challenging, so we developed several algorithmic optimizations in our analysis workflows that lead to improvements in IO times of up to 91x on 112 cores when compared to the baseline benchmark results, however this speedup is likely caused by the inefficient chunk layout of the original file. With a custom, optimized chunk layout and gzip compression, we found maximum scaling of 36x on 2 full nodes on Agave. To garner futher improvements in parallel I/O performance, a more sophisticated I/O pattern may be required. The addition of the HDF5 reader provides a foundation for the development of parallel trajectory analysis with MPI and the MDAnalysis package.



Acknowledgments
===============
Funding was provided by the National Science Foundation for a REU supplement to award ACI1443054.
The SDSC Comet computer at the San Diego Supercomputer Center and PSC Bridges computer at the Pittsburgh Supercomputing Center was used under allocation TG-MCB130177.
The authors acknowledge Research Computing at Arizona State University for providing HPC resources that have contributed to the research results reported within this paper.
We would like to acknowledge Gil Speyer and Jason Yalim from the Research Computing Core Facilities at Arizona State University for advice and consultation.



References
----------
