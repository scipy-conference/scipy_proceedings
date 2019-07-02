.. -*- mode: rst; mode: visual-line; fill-column: 9999; coding: utf-8 -*-

   
:author: Shujie Fan
:email: sfan19@asu.edu
:institution: Arizona State University
:equal-contributor:	      

:author: Max Linke
:email: max.linke88@gmail.com
:institution: Max Planck Institute of Biophysics
:equal-contributor:
	      
:author: Ioannis Paraskevakos
:email: i.paraskev@rutgers.edu
:institution: Rutgers University

:author: Richard J. Gowers
:email: richardjgowers@gmail.com
:institution: University of New Hampshire
:institution: present address: NextMove Software Ltd. 

:author: Michael Gecht
:email: michael.gecht@biophys.mpg.de
:institution: Max Planck Institute of Biophysics

:author: Oliver Beckstein
:email: obeckste@asu.edu 
:institution: Arizona State University 
:corresponding:

:bibliography: pmda

.. Standard reST tables do not properly build and the first header column is lost.
.. We therefore use raw LaTeX tables. However, booktabs is not automatically included
.. unless rest2latex sees a table so we have to add it here manually.
.. latex::
   :usepackage: booktabs
	       

.. STYLE GUIDE
.. ===========
.. .
.. Writing
..  - use past tense to report results
..  - use present tense for intro/general conclusions
.. .
.. Formatting
..  - restructured text
..  - hard line breaks after complete sentences (after period)
..  - paragraphs: empty line (two hard line breaks)
.. .
.. Workflow
..  - use PRs (keep them small and manageable)
..  - build the paper locally from the top level
..       rm -r output/shujie_fan      # sometimes needed to recover from errors
..       make_paper.sh papers/shujie_fan/
..       open  output/shujie_fan/paper.pdf
..   
   
.. definitions (like \newcommand)

.. |Calpha| replace:: :math:`\mathrm{C}_\alpha`
.. |tprepare| replace:: :math:`t^\text{prepare}`		      
.. |tcomp| replace:: :math:`t^{\text{compute}}_{k,t}`
.. |tIO| replace:: :math:`t^\text{I/O}_{k,t}`
.. |tconclude| replace:: :math:`t^\text{conclude}_{k}`
.. |tuniverse| replace:: :math:`t^\text{Universe}_{k}`
.. |twait| replace:: :math:`t^\text{wait}_{k}`
.. |ttotal| replace:: :math:`t^\text{total}`		 			 		     
.. |avg_tcomp| replace:: :math:`\langle t_\text{compute} \rangle`
.. |avg_tIO| replace:: :math:`\langle t_\text{I/O} \rangle`
.. |Ncores| replace:: :math:`M`
.. |r(t)| replace:: :math:`\mathbf{r}(t)`

	  
-------------------------------------------		      
PMDA - Parallel Molecular Dynamics Analysis
-------------------------------------------

.. class:: abstract

   MDAnalysis_ is an object-oriented Python library to analyze trajectories from molecular dynamics (MD) simulations in many popular formats.
   With the development of highly optimized MD software packages on high performance computing (HPC) resources, the size of simulation trajectories is growing up to many terabytes in size.
   However efficient usage of multicore architecture is a challenge for MDAnalysis, which does not yet provide a standard interface for parallel analysis.
   To address the challenge, we developed PMDA_, a Python library that builds upon MDAnalysis to provide parallel analysis algorithms.
   PMDA parallelizes common analysis algorithms in MDAnalysis through a task-based approach with the Dask_ library.
   We implement a simple split-apply-combine scheme for parallel trajectory analysis.
   The trajectory is split into blocks, analysis is performed separately and in parallel on each block ("apply"),
   then results from each block are gathered and combined.
   PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks.
   In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules.
   PMDA supports all schedulers in Dask, and one can run analysis in a distributed fashion on HPC machines, ad-hoc clusters, a single multi-core workstation or a laptop.
   We tested the performance of PMDA on single node and multiple nodes on a national supercomputer.
   The results show that parallelization improves the performance of trajectory analysis and, depending on the analysis task, can cut down time to solution from hours to minutes.
   Although still in alpha stage, it is already used on resources ranging from multi-core laptops to XSEDE supercomputers to speed up analysis of molecular dynamics trajectories.
   PMDA is available as open source under the GNU General Public License, version 2 and can be easily installed via the ``pip`` and ``conda`` package managers.

.. class:: keywords

   Molecular Dynamics Simulations, High Performance Computing, Python, Dask, MDAnalysis





Introduction
============

Classical molecular dynamics (MD) simulations have become an invaluable tool to understand the function of biomolecules :cite:`Karplus:2002ly, Dror:2012cr, Seyler:2014il, Orozco:2014dq, Bottaro:2018aa, Huggins:2019aa` (often with a view towards drug discovery :cite:`Borhani:2012mi`) and diverse problems in materials science :cite:`Rottler:2009aa, Li:2015aa, Varela:2015aa, Lau:2018aa, Kupgan:2018aa, Frederix:2018aa`.
Systems are modeled as particles (for example, atoms) whose interactions are approximated with a classical potential energy function :cite:`FrenkelSmit02, Braun:2018ab`.
Forces on the particles are derived from the potential and Newton's equations of motion for the particles are solved with an integrator algorithm, typically using highly optimized MD codes that run on high performance computing (HPC) resources or workstations (often equipped with GPU accelerators).
The resulting trajectories, the time series of particle positions :math:`\mathbf{r}(t)` (and possibly velocities), are analyzed with statistical mechanics approaches :cite:`Tuckerman:2010cr, Braun:2018ab` to obtain predictions or to compare to experimentally measured quantities.
Currently simulated systems may contain millions of atoms and the trajectories can consist of hundreds of thousands to millions of individual time frames, thus resulting in file sizes ranging from tens of gigabytes to tens of terabytes.
Processing and analyzing these trajectories is increasingly becoming a rate limiting step in computational workflows :cite:`Cheatham:2015qf, Beckstein:2018aa`.
Modern MD packages are highly optimized to perform well on current HPC clusters with hundreds of cores such as the XSEDE supercomputers :cite:`XSEDE` but current general purpose trajectory analysis packages :cite:`Giorgino:2019aa` were not designed with HPC in mind.

In order to scale up trajectory analysis from workstations to HPC clusters with the MDAnalysis_ Python library :cite:`Michaud-Agrawal:2011fu,Gowers:2016aa` we leveraged Dask_ :cite:`Rocklin:2015aa, Dask:2016aa`, a task-graph parallel framework, together with Dask's various schedulers (in particular *distributed*), and created the *Parallel MDAnalysis* (PMDA_) library.
By default, PMDA follows a simple split-apply-combine :cite:`Wickham:2011aa` approach for trajectory analysis, whereby each task analyzes a single trajectory segment and reports back the individual results that are then combined into the final result :cite:`Khoshlessan:2017ab`.
Our previous work established that Dask worked well with MDAnalysis :cite:`Khoshlessan:2017ab` and that this approach was competitive with other task-parallel approaches :cite:`Paraskevakos:2018aa`.
However, we did not provide a general purpose framework to write parallel analysis tools with MDAnalysis.
Here we show how the split-apply-combine approach lends itself to a generalizable Python implementation that makes it straightforward for users to implement their own parallel analysis tools.
At the heart of PMDA is the idea that the user only needs to provide a function that analyzes a single trajectory frame.
PMDA provides the remaining framework via the :code:`ParallelAnalysisBase` class to split the trajectory, apply the user's function to trajectory frames, run the analysis in parallel via Dask/*distributed*, and combines the data.
It also contains a growing library of ready-to-use analysis classes, thus enabling users to immediately accelerate analysis that they previously performed in serial with the standard MDAnalysis analysis classes :cite:`Gowers:2016aa`.





Methods
=======

At the core of PMDA is the idea that a common interface makes it easy to create code that can be easily parallelized, especially if the analysis can be split into independent work over multiple trajectory slices and a final step, in which all data from the trajectory slices are combined.
We first describe typical steps in analyzing MD trajectories and then outline the approach taken in PMDA.


Trajectory analysis
-------------------

A trajectory with :math:`T` saved time steps consists of a sequence of coordinates :math:`\big\{\big(\mathbf{r}_1(t), \mathbf{r}_2(t), \dots \mathbf{r}_N(t)\big)\big\}_{1\le t \le T}` where :math:`\mathbf{r}_i(t)` are the Cartesian coordinates of particle :math:`i` at time step :math:`t` with :math:`N` particles in the simulated system, i.e., :math:`T \times N \times 3` floating point numbers in total.
To simplify notation, we consider :math:`t` as an integer that indexes the trajectory frames; each frame index corresponds to a physical time in the trajectory that we could obtain if needed.
In general, the coordinates are passed to a function :math:`\mathcal{A}(\{\mathbf{r}_i(t)\})` to compute a time-dependent quantity

.. math::
   :label: eq:timeseries

   A(t) = \mathcal{A}(\{\mathbf{r}_i(t)\}).
   
This quantity does not have to be a simple scalar; it may be a vector or a function of another parameter.
In many cases, the *time series* :math:`A(t)` is the desired result.
It is, however, also common to perform some form of *reduction* on the data, which can be as simple as a time average to compute a thermodynamic average :math:`\langle A\rangle \equiv \bar{A} = T^{-1} \sum_{t=1}^{T} A(t)`.
Such an average can be easily calculated in a post-analysis step after the time series has been obtained.
An example of a more complicated reduction is the calculation of a histogram such as a radial distribution function (RDF) :cite:`FrenkelSmit02, Tuckerman:2010cr` between two types of particles with numbers :math:`N_a` and :math:`N_b`,

.. math::
   :label: eq:rdf

   g(r) = \left\langle \frac{1}{N_a N_b} \sum_{i=1}^{N_a} \sum_{j=1}^{N_b} \delta(|\mathbf{r}_{a,i} - \mathbf{r}_{b,j}| - r) \right\rangle

where the Dirac delta function counts the occurrences of particles :math:`i` and :math:`j` at distance :math:`r`.
To compute a RDF, we could generate a time series of histograms along the spatial coordinate :math:`r`, i.e., :math:`A(t; r)` for each frame, and then perform the average in post-analysis.
However, storage of such histograms becomes problematic, especially if instead of 1-dimensional RDFs, densities on 3-dimensional grids are being calculated.
It is therefore better to reformulate the algorithm to perform a partial average (or reduction) during the analysis on a per-frame basis.
For histograms, this could mean building a partial histogram and updating counts in the bins after every frame.
PMDA supports the simple time series data collection and the per-frame reduction.

Split-apply-combine
-------------------

The *split-apply-combine* strategy can be thought of as a simplified map-reduce :cite:`Wickham:2011aa` that provides a conceptually simple approach to operate on data in parallel.
It is based on the fundamental assumption that the data can be partitioned into blocks that can be analyzed independently.
The trajectory is split along the time axis into :math:`M` blocks of approximately equal size, :math:`\tau = T/M`.
One trajectory block can be viewed as a slice of a trajectory, e.g., for block :math:`k`, :math:`\big\{\big(\mathbf{r}_1(t), \mathbf{r}_2(t), \dots \mathbf{r}_N(t)\big)\big\}_{t_k \le t < t_k + \tau_k}` with :math:`\tau_k` frames in the block.
Each block :math:`k` is analyzed in parallel by applying the function :math:`\mathcal{A}` to the frames in each block.
Finally, the results from all blocks are gathered and combined.

The advantage of this approach is its simplicity.
Many typical analysis tasks are based on calculations of time series from single trajectory frames as in Eq. :ref:`eq:timeseries` and it is this calculation that varies from task to task while the book-keeping and trajectory slicing is the same.
Given a function :math:`\mathcal{A}` that performs the *single frame calculation*, PMDA provides code to perform the other necessary steps (Fig. :ref:`fig:schema`).

.. figure:: figs/pmda-schema.pdf
	    
   High-level view of the split-apply-combine algorithm in PMDA.
   Steps are labeled with the methods in :code:`pmda.parallel.ParallelAnalysisBase` that perform the corresponding function.
   Methods in red (:code:`_single_frame()` and :code:`_conclude()`) must be implemented for every analysis function because they are not general.
   The blue method :code:`_reduce()` must be implemented unless a simple time series is being calculated.
   The :code:`_prepare()` method is optional and provides a hook to initialize custom data structures.
   :label:`fig:schema`

As explained in more detail later, a class derived from :code:`pmda.parallel.ParallelAnalysisBase` encapsulates one trajectory analysis calculation.
Individual methods correspond to different steps and in the following (and in Fig. :ref:`fig:schema`) we will mention the names of the relevant methods to make clear how PMDA abstracts parallel analysis.
The calculation with :math:`M` parallel workers is *prepared* by setting up data structures to hold the final result (method :code:`_prepare()`).
The indices for the :math:`M` trajectory slices are created in such a way that the number of frames :math:`\tau_k` are balanced and do not differ by more than 1.
For each slice or block :math:`k`, the *single frame* analysis function :math:`\mathcal{A}` (:code:`_single_frame()`) is sequentially applied to all frames in the slice.
The result, :math:`A(t)`, is *reduced*, i.e., added to the results for this block.
For time series, :math:`A(t)` is simply appended to a list to form a partial time series for the block.
More complicated reductions (method :code:`_reduce()`) can be implemented, for  example, the data may be histogrammed and added to a partial histogram for the block (as necessary for the implementation of the parallel RDF Eq. :ref:`eq:rdf`).




Implementation
--------------

PMDA is written in Python and, through MDAnalysis :cite:`Gowers:2016aa`, reads trajectory data from the file system into NumPy arrays :cite:`Oliphant:2007aa, Van-Der-Walt:2011aa`.
Dask's :code:`delayed()` function is used to build a task graph that is then executed using any of the schedulers available to Dask :cite:`Dask:2016aa`.

MDAnalysis combines a trajectory file (frames of coordinates that change with time) and a topology file (list of particles, their names, charges, bonds — all information that does not change with time) into a :code:`Universe(topology, trajectory)` object.
Arbitrary selections of particles (often atoms) are made available as an :code:`AtomGroup` and the common approach in MDAnalysis is to work with these objects :cite:`Gowers:2016aa`; for instance, all coordinates of an :code:`AtomGroup` with :math:`N` atoms named :code:`protein` are accessed as the :math:`N \times 3` NumPy array :code:`protein.positions`.

:code:`pmda.parallel.ParallelAnalysisBase` is the base class for defining a split-apply-combine parallel multi frame analysis in PMDA.
It requires a :code:`Universe` to operate on and any :code:`AtomGroup` instances that will be used.
A parallel analysis class must be derived from :code:`ParallelAnalysisBase` and at a minimum, must implement the :code:`_single_frame(ts, agroups)` and :code:`_conclude()` methods.
The arguments of :code:`_single_frame(ts, agroups)` are a MDAnalysis :code:`Timestep` instance and a tuple of :code:`AtomGroup` instances so that the following code could be run (the code is a simplified version of the current implementation):

.. code-block:: python
   :linenos:		

   @delayed
   def analyze_block(blockslice):
       result = []		
       for ts in u.trajectory[blockslice]:		
	   A = self._single_frame(ts, agroups)
	   result.append(A)
       return result

The task graph is constructed by wrapping the above code into :code:`delayed()` and appending a delayed instance for each trajectory slice to a (delayed) list:

.. code-block:: python
   :linenos:
   :linenostart: 7      

   blocks = delayed([analyze_block(blockslice)
                     for blockslice in slices])
   results = blocks.compute(**scheduler_kwargs)

Calling the :code:`compute()` method of the delayed list object hands the task graph over to the scheduler, which then executes the graph on the available Dask workers.
For example, the *multiprocessing* scheduler can be used  to parallelize task graph execution on a single multiprocessor machine while the *distributed* scheduler is used to run on multiple nodes of a HPC cluster.
After all workers have finished, the variable :code:`results` contains a list of results from the individual blocks.
PMDA actually stores these raw results as :code:`ParallelAnalysisBase._results` and leaves it to the :code:`_conclude()` method to process the results; this can be as simple as :code:`numpy.hstack(self._results)` to generate a time series by concatenating the individual time series from each block.
		        
The default :code:`_reduce()` method appends the results and is equivalent to line 6.
In general, line 6 reads

.. code-block:: python
   :linenos:
   :linenostart: 6  

           result = self._reduce(result, A)

where variable :code:`result` should have been properly initialized in :code:`_prepare()`.
In order to be parallelizable, the :code:`_reduce()` method must be a static method that does not access any class variables but returns its modified first argument.
For example, the default "append" reduction is

.. code-block:: python

        @staticmethod
        def _reduce(res, result_single_frame):
            res.append(result_single_frame)
            return res


In general, the :code:`ParallelAnalysisBase` controls access to instance attributes via a context manager :code:`ParallelAnalysisBase.readonly_attributes()`.
It sets them to "read-only" for all parallel parts to prevent the common mistake to set an instance attribute in a parallel task, which breaks under parallelization as the value of an attribute in an instance in a parallel process is never communicated back to the calling process.

	    

Using PMDA
==========

PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks. In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules. Here, we will introduce some basic usages of PMDA.

Pre-defined Analysis
--------------------
PMDA contains a growing number of pre-defined analysis classes that are modeled after functionality in :code:`MDAnalysis.analysis` and that can be used right away.
Current examples are :code:`pmda.rms` for  RMSD analysis, :code:`pmda.contacts` for native contacts analysis, :code:`pmda.rdf` for radial distribution functions, and :code:`pmda.leaflet` for the LeafletFinder analysis tool :cite:`Michaud-Agrawal:2011fu, Paraskevakos:2018aa` for the topological analysis of lipid membranes.
While the first three modules are based on :code:`pmda.parallel.ParallelAnalysisBase` as described above and follow the strict split-apply-combine approach, :code:`pmda.leaflet` is an example of a more complicated task-based algorithm that can also easily be implemented with MDAnalysis and Dask :cite:`Paraskevakos:2018aa`.
All PMDA classes can be used in a similar manner to classes in :code:`MDAnalysis.analysis`, which makes it easy for users of MDAnalysis to switch to parallelized versions of the algorithms.
One example is the calculation of the root mean square distance (RMSD) of |Calpha| atoms of the protein with :code:`pmda.rms.RMSD`.
An analysis class object is instantiated with the necessary input data such as the :code:`AtomGroup` containing the |Calpha| atoms and a reference structure.
To perform the analysis, the :code:`run()` method is called. 


.. code-block:: python

    import MDAnalysis as mda
    from pmda import rms
    # Create a Universe based on simulation topology
    # and trajectory
    u = mda.Universe(top, trj)

    # Select all the C alpha atoms
    ca = u.select_atoms('name CA')

    # Take the initial frame as the reference
    u.trajectory[0]
    ref = u.select_atoms('name CA')

    # Build the parallel rms object, and run 
    # the analysis with 4 workers and 4 blocks.
    rmsd = rms.RMSD(ca, ref)
    rmsd.run(n_jobs=4, n_blocks=4)

    # The results can be accessed in rmsd.rmsd.
    print(rmsd.rmsd)

Here the only difference between using the serial version and the parallel version is that the :code:`run()` method takes additional arguments :code:`n_jobs` and :code:`n_blocks`, which determine the level of parallelization.
When using the *multiprocessing* scheduler (the default),  :code:`n_jobs` is the number of processes to start and typically the number of blocks  :code:`n_blocks` is set to the number of available CPU cores.
When the *distributed* scheduler is used, Dask will automatically learn the number of available Dask worker processes and :code:`n_jobs` is meaningless; instead it makes more sense to set the number of trajectory blocks that are then spread across all available workers. 



User-defined Analysis
---------------------

PMDA makes it easy to create analysis classes such as the ones discussed above.
If the per-frame analysis can be expressed as a simple function, then an analysis class can be created with a factory function.
Otherwise, a class has to be derived from :code:`pmda.parallel.ParallelAnalysisBase`.
Both approaches are described below.


:code:`pmda.custom.AnalysisFromFunction()`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PMDA provides helper functions in :code:`pmda.custom` to rapidly build a parallel class for users who already have a *single frame* function that 
1. takes one or more ``AtomGroup`` instances as input,
2. analyzes one frame in a trajectory and returns the result for this frame.
For example, if we already have a function to calculate the radius of gyration :cite:`Mura:2014kx` of a protein given in :code:`AtomGroup` ``ag``, namely ``ag.radius_of_gyration()`` (as available in MDAnalysis), then we can write a simple function ``rgyr()`` that returns for each trajectory frame a tuple containing the time at the current time step and the value of the radius of gyration:

.. code-block:: python

    import MDAnalysis as mda
    u = mda.Universe(top, traj)
    protein = u.select_atoms('protein')

    def rgyr(ag):
        return (ag.universe.trajectory.time,
	        ag.radius_of_gyration())

	
We can wrap :code:`rgyr()` in the :code:`pmda.custom.AnalysisFromFunction()` class instance factory function to build a parallel version of :code:`rgyr()`:

.. code-block:: python
     
    import pmda.custom
    parallel_rgyr = pmda.custom.AnalysisFromFunction(
                    rgyr, u, protein)

This new parallel analysis class can be run just as the existing ones:

.. code-block:: python

    parallel_rgyr.run(n_jobs=4, n_blocks=4)
    print(parallel_rgyr.results)

The time series of the results is stored in the attribute :code:`parallel_rgyr.results`; for our example where each per-frame result is a tuple ``(time, Rgyr)``, the time series is stored as a :math:`T \times 2` array that can be plotted with

.. code-block:: python
		
    import matplotlib.pyplot as plt
    data = parallel_rgyr.results
    plt.plot(data[:, 0] , data[:, 1])




:code:`pmda.parallel.ParallelAnalysisBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more general cases, one can write the parallel class with the help of :code:`pmda.parallel.ParallelAnalysisBase`, following the schema in Fig. :ref:`fig:schema`.
To build a new analysis class, one should derive a class from :code:`pmda.parallel.ParallelAnalysisBase` that implements

1. the single frame analysis method :code:`_single_frame()` (*required*),
2. the final results conclusion method :code:`_conclude()` (*required*),
3. the additional preparation method :code:`_prepare()` (*optional*),
4. the reduce method for frames within the same block :code:`_reduce()` (*optional* for time series, *required* for anything else).

As an example, we show how one can build a class to calculate the radius of gyration of a protein given in :code:`AtomGroup` ``protein``; of course, in this case the simple approach with :code:`pmda.custom.AnalysisFromFunction()` would be easier.

.. code-block:: python

    import numpy as np
    from pmda.parallel import ParallelAnalysisBase

    class RGYR(ParallelAnalysisBase):
        def __init__(self, protein):
            universe = protein.universe
            super(RGYR, self).__init__(universe,
	                               (protein,))
        def _prepare(self):
            self.rgyr = None
        def _conclude(self):
            self.rgyr = np.vstack(self._results)

The :code:`_conclude()` method reshapes the attribute :code:`self._results`, which always holds the results from all blocks, into a time series.  	    
The call signature for method :code:`_single_frame()` is fixed and ``ts`` must contain the current MDAnalysis :code:`Timestep` and ``agroups`` must be a tuple of :code:`AtomGroup` instances.
The current frame number, time and radius of gyration are returned as the single frame results:

.. code-block:: python

        def _single_frame(self, ts, atomgroups):
            protein = atomgroups[0]            
            return (ts.frame, ts.time,
                    protein.radius_of_gyration())

Because we want to return a time series, it is not necessary to define a :code:`_reduce()` method.		    
This class can be used in the same way as the class that we defined with :code:`pmda.custom.AnalysisFromFunction`:  

.. code-block:: python

    parallel_rgyr = RGYR(protein)
    parallel_rgyr.run(n_jobs=4, n_blocks=4)
    print(parallel_rgyr.results)



Performance Evaluation
======================

In order to characterize the performance of PMDA on a typical HPC machine we performed computational experiments for two different analysis tasks, the RMSD calculation after optimum superposition (*RMSD*) and the water oxygen radial distribution function (*RDF*).

For the *RMSD* task we computed the time series of root mean square distance after optimum superposition (RMSD) of all 564 |Calpha| atoms of a protein with the initial coordinates at the first frame as reference, as implemented in class :code:`pmda.rms.RMSD`.
The RMSD calculation with optimum superposition was performed with the fast QCPROT algorithm :cite:`Theobald:2005vn` as implemented in MDAnalysis :cite:`Michaud-Agrawal:2011fu`.

As a second test case we computed the water oxygen-oxygen radial distribution function (*RDF*, Eq. :ref:`eq:rdf`) in 75 bins up to a cut-off of 5 Å for all 24,239 oxygen atoms in the water molecules in our test system, using the class :code:`pmda.rdf.InterRDF`.
The RDF calculation is compute-intensive due to the necessity to calculate and histogram a large number (:math:`\mathcal{O}(N)` because of the use of a cut-off) of distances for each time step; it additionally exemplifies a non-trivial reduction.

These two common computational tasks differ in their computational cost and represent two different requirements for data reduction and thus allow us to investigate two distinct use cases.
We investigated a long (9000 frames) and a short trajectory (900 frames) to assess to which degree parallelization remained practical. 
The computational experiments were performed in different scenarios to assess the influence of different Dask schedulers (*multiprocessing* and *distributed*) and the role of the file storage system (shared Lustre parallel file system and local SSD), as described below and summarized in Table :ref:`tab:configurations`.



Test system, benchmarking environment, and data files
-----------------------------------------------------

We tested PMDA 0.2.1, MDAnalysis 0.20.0 (development version), Dask 1.2.0, and NumPy 1.15.4 under Python 3.6.
All packages except PMDA and MDAnalysis were installed with the `conda`_ package manager from the `conda-forge`_ channel.
PMDA and MDAnalysis development versions were installed from source in a conda environment  with ``pip install``.

Benchmarks were run on the CPU nodes of XSEDE's :cite:`XSEDE` *SDSC Comet* supercomputer, a 2 PFlop/s cluster with 1,944 Intel Haswell Standard Compute Nodes in total.
Each node contains two Intel Xeon CPUs (E5-2680v3, 12 cores, 2.5 GHz) with 24 CPU cores per node, 128 GB DDR4 DRAM main memory, and a non-blocking fat-tree InfiniBand FDR 56 Gbps node interconnect.
All nodes share a Lustre parallel file system and have access to node-local 320 GB SSD scratch space.
Jobs are run through the SLURM batch queuing system.
Our SLURM submission shell scripts  and Python benchmark scripts for *SDSC Comet* are available in the repository https://github.com/Becksteinlab/scipy2019-pmda-data and are archived under DOI `10.5281/zenodo.3228422`_. 

The test data files consist of a topology file ``YiiP_system.pdb`` (with :math:`N = 111,815` atoms) and two trajectory files ``YiiP_system_9ns_center.xtc`` (Gromacs XTC format, :math:`T = 900` frames) and ``YiiP_system_90ns_center.xtc`` (Gromacs XTC format, :math:`T = 9000` frames) of the membrane protein YiiP in a lipid bilayer together with water and ions.
The test trajectories are made available on figshare at DOI `10.6084/m9.figshare.8202149`_.

.. raw:: latex

   \begin{table}
   \begin{longtable*}[c]{p{0.3\tablewidth}p{0.1\tablewidth}lp{0.07\tablewidth}p{0.07\tablewidth}}
    \toprule
    \textbf{configuration label} & \textbf{file storage} & \textbf{scheduler} & \textbf{max nodes} & \textbf{max processes} \tabularnewline
    \midrule
    \endfirsthead
    Lustre-distributed-3nodes & Lustre       & \textit{distributed}       &  3        & 72         \tabularnewline
    Lustre-distributed-6nodes & Lustre       & \textit{distributed}       &  6        & 72         \tabularnewline
    Lustre-multiprocessing    & Lustre       & \textit{multiprocessing}   &  1        & 24         \tabularnewline
    SSD-distributed           & SSD          & \textit{distributed}       &  1        & 24         \tabularnewline
    SSD-multiprocessing       & SSD          & \textit{multiprocessing}   &  1        & 24         \tabularnewline
    \bottomrule
    \end{longtable*}
    \caption{Testing configurations on \textit{SDSC Comet}.
	   \textbf{max nodes} is the maximum number of nodes that were tested; the \textit{multiprocessing} scheduler is limited to a single node.
	   \textbf{max processes} is the maximum number of processes or Dask workers that were employed.
	   \DUrole{label}{tab:configurations}
	   }
   \end{table}

We tested different combinations of Dask schedulers (*distributed*, *multiprocessing*) with different means to read the trajectory data (either from the shared Lustre parallel file system or from local SSD) as shown in Table :ref:`tab:configurations`.
Using either the *multiprocessing* scheduler or the SSD restrict runs to a single node (maximum 24 CPU cores).
With *distributed* (and Lustre) we tested fully utilizing all cores on a node and also only occupying half the available cores, while doubling the total number of nodes.
In all cases the trajectory were split in as many blocks as there were available processes or Dask workers.
We performed five independent repeat runs for all scenarios in Table :ref:`tab:configurations` and plotted the mean of the reported timing quantity together with the standard deviation from the mean to indicate the variance of the runs.




Measured timing quantities
--------------------------

The :code:`ParallelAnalysisBase` class collects detailed timing information for all blocks and all frames and makes these data available in the attribute :code:`ParallelAnalysisBase.timing`:
We measured the time |tprepare| for :code:`_prepare()`, the time |twait| that each task :math:`k` waits until it is executed by the scheduler, the time |tuniverse| to create a new :code:`Universe` for each Dask task (which includes opening the shared trajectory and topology files and loading the topology into memory), the time |tIO| to read each frame :math:`t` in each block :math:`k` from disk into memory, the time |tcomp| to perform the computation in :code:`_single_frame()` and reduction in :code:`_reduce()`, the time |tconclude| to perform the final processing of all data in :code:`_conclude()`, and the total wall time to solution |ttotal|.

We analyzed the total time to completion as a function of the number of CPU cores, which was equal to the number of trajectory blocks, so that each block could be processed in parallel.
We quantified the strong scaling behavior by calculating the *speed-up* for running on :math:`M` CPU cores with :math:`M` parallel Dask tasks as :math:`S(M) = t^\text{total}(1)/t^\text{total}(M)`, where :math:`t^\text{total}(1)` is the performance of the PMDA code using the serial scheduler.
The *efficiency* was calculated as :math:`E(M) = S(M)/M`. The errors of these quantities were derived by the standard error propagation.

To gain better insight into the performance-limiting steps in our algorithm (Fig. :ref:`fig:schema`) we plotted the *maximum* times over all ranks because the overall time to completion cannot be faster than the slowest parallel process.
For example, for the read I/O time we calculated the total read I/O time for each rank :math:`k` as :math:`t^\text{I/O}_k = \sum_{t=t_k}^{t_k + \tau_k} t^\text{I/O}_{k, t}` and then reported :math:`\max_k t^\text{I/O}_k`.




RMSD analysis task
------------------

.. figure:: figs/Total_Eff_SU_rms.pdf

   Strong scaling performance of the RMSD analysis task with short (900 frames) and long (9000) frames trajectories on *SDSC Comet*, where a single node contains 24 cores.
   The total time to completion |ttotal| was measured for different testing configurations (Table :ref:`tab:configurations`).
   **A** and **D**: |ttotal| as a function of processes or Dask workers, i.e., the number of CPU cores that were actually used.
   The number of trajectory blocks was the same as the number of CPU cores.
   **B** and **E**: efficiency :math:`E`. The ideal case is :math:`E = 1`.
   **C** and **F**: speed-up :math:`S`. The dashed line represents ideal strong scaling :math:`S(M) = M`.
   Points represent the mean over five repeats with the standard deviation shown as error bars.
   :label:`fig:rmsd`


The parallelized RMSD analysis in :code:`pmda.rms.RMSD` scaled well only to about half a node (12 cores), as shown in Fig. :ref:`fig:rmsd` A, D, regardless of the length of the trajectory.
The efficiency dropped below 0.8 (Fig. :ref:`fig:rmsd` B, E) and the maximum achievable speed-up remained below 10 for the short trajectory (Fig. :ref:`fig:rmsd` C) and below 20 for the long one (Fig. :ref:`fig:rmsd` F).
Overall, using the *multiprocessing* scheduler and either Lustre or SSD gave the best performance and shortest time to solution.
The *distributed* scheduler with SSD gave widely variable results as seen by large standard deviations over multiple repeats.
It still performed better than when the Lustre file system was used but overall, for a single node, the *multiprocessing* scheduler always gave better performance with less variation in run time.
These results were consistent with findings in our earlier pilot study where we had looked at the RMSD task with Dask and had found that *multiprocessing* with both SSD and Lustre had given good single node performance but, using *distributed*, had not scaled well beyond a single *SDSC Comet* node :cite:`Khoshlessan:2017ab`.

.. figure:: figs/wait_compute_io_rms.pdf

   Detailed per-task timing analysis for parallel components of RMSD analysis task.
   Individual times per task were measured for different testing configurations (Table :ref:`tab:configurations`).
   **A** and **D**: Maximum waiting time for the task to be executed by the Dask scheduler.
   **B** and **E**: Maximum total compute time per task.
   **C** and **F**: Maximum total read I/O time per task.
   Points represent the mean over five repeats with the standard deviation shown as error bars.
   :label:`fig:rms-wait-comp-io`

A detailed look at the maximum times (Fig. :ref:`fig:rms-wait-comp-io`) that the Dask worker processes spent on waiting to be executed, performing the RMSD calculation with data in memory, and reading the trajectory frame data from the file into memory showed that the waiting time (Fig. :ref:`fig:rms-wait-comp-io` A, D) either increased from about 0.02 s to 0.1 s for *multiprocessing* or was roughly a constant 1 s for *distributed* (on Lustre).
For reasons that were not clear, the *distributed* scheduler with SSD had on average the largest wait times, with large fluctuations, ranging from 0.1 s to 10 s (red lines in Fig. :ref:`fig:rms-wait-comp-io` A, D).
The computation itself scaled very well (Fig. :ref:`fig:rms-wait-comp-io` B, E) with only small variations, indicating that split-apply-combine is a robust approach to parallelization, once the data are in memory.
The reading time scaled fairly well but exhibited some variation beyond a single node (24 cores) and an unexplained decline in performance for the longer trajectory, as seen in Fig. :ref:`fig:rms-wait-comp-io` C, F.
The read I/O results indicated that both Lustre and SSD can perform equally well.
Beyond 12 cores, the waiting time started approaching the time for read I/O (compute was an order of magnitude less than I/O) and hence parallel speed-up was limited by the wait time.

.. figure:: figs/pre_con_uni_rms.pdf
	    
   Detailed timing analysis for other components of the RMSD analysis task.
   Individual times per task were measured for different testing configurations (Table :ref:`tab:configurations`).
   **A** and **D**: Maximum time for a task to load the :code:`Universe`.
   **B** and **E**: Time |tprepare| to execute :code:`_prepare()`. 
   **C** and **F**: Time |tconclude| to execute :code:`_conclude()`.
   Points represent the mean over five repeats with the standard deviation shown as error bars.
   :label:`fig:rms-pre-con-uni`

The second major component that limited scaling performance was the time to create the :code:`Universe` data structure (Fig. :ref:`fig:rms-pre-con-uni` A, D).
The time to read the topology and open the trajectory file on the shared file system typically increased from 1 s to about 2 s and thus, for the given total trajectory lengths, also became comparable to the time for read I/O.
The other components (prepare and conclude, see Fig. :ref:`fig:rms-pre-con-uni`) remained negligible with times below :math:`10^{-3}` s.

.. figure:: figs/percentage_stack_rms.pdf

   Fraction of the total run time taken by individual steps in the parallel *RMSD* calculation for *distributed* on up to three nodes (Lustre-distributed-3nodes).
   Compute (green) and read I/O (red) represent the parallelizable fraction of the program; all other components are effectively serial.
   **A** Trajectory with 900 frames.
   **B** Trajectory with 9000 frames.   
   :label:`fig:timefraction-rms`

The parallelizable fraction of the workload consisted of the compute and read I/O steps.
Because this fraction was relatively small and was dominated by the wait time from the Dask scheduler and the time to initialize the ``Universe`` data structure (Fig. :ref:`fig:timefraction-rms`), the overall performance gain by parallelization remained modest, as explained by Amdahl's law :cite:`Amdahl:1967aa`.
Thus, for a highly optimized and fast computation such as the RMSD calculation, the best performance (speed-up on the order of 10 fold) could already be achieved on the equivalent of a modern workstation.
The *multiprocessing* scheduler seemed to be the more consistent and better performing choice in this scenario; therefore PMDA defaults to *multiprocessing*.
Performance would likely improve with longer trajectories because the "fixed" serial costs (waiting, :code:`Universe` creation) would decrease in relevance to the time spent on computation and data ingestion (see Fig. :ref:`fig:timefraction-rms` B), which benefit from parallelization :cite:`Gustafson:1988aa`.
However, all things considered, a single node seemed sufficient to accelerate RMSD analysis.

	  

RDF analysis task
-----------------

Unlike the RMSD analysis task, the parallelized RDF analysis in :code:`pmda.rdf.InterRDF` showed decreasing total time to solution up to the highest number of CPU cores tested (see Fig. :ref:`fig:rdf` A, D).
The efficiency on a single node remained above 0.6 for almost all cases (Fig. :ref:`fig:rdf` B, E) and remained above 0.6 for the best case (*distributed* on Lustre and half-filling of nodes for the long trajectory), up to 3 nodes (72 cores, Fig. :ref:`fig:rdf` E).
Even when filling complete nodes, the efficiency for the long trajectory remained above 0.5 (Fig. :ref:`fig:rdf` E).
Consequently, a sizable speed-up could be maintained that approached 40 fold in the best case (Fig. :ref:`fig:rdf` F), which cut down the time to solution from about 40 min to under 1 min.
On a single node, all approaches performed similarly well, with the *distributed* scheduler now having a slight edge over *multiprocessing* (Fig. :ref:`fig:rdf`), with the exception of the combination of *distributed* with the SSD, which for unknown reasons performed much worse than everything else (similar to the situation observed for the *RMSD* case).


.. figure:: figs/Total_Eff_SU_rdf.pdf

   Strong scaling performance of the RDF analysis task.
   The total time to completion |ttotal| was measured for different testing configurations (Table :ref:`tab:configurations`).
   **A** and **D**: |ttotal| as a function of processes or Dask workers, i.e., the number of CPU cores that were actually used.
   The number of trajectory blocks was the same as the number of CPU cores.
   **B** and **E**: efficiency :math:`E`. The ideal case is :math:`E = 1`.
   **C** and **F**: speed-up :math:`S`. The dashed line represents ideal strong scaling :math:`S(M) = M`.
   Points represent the mean over five repeats with the standard deviation shown as error bars.   
   :label:`fig:rdf`

.. figure:: figs/wait_compute_io_rdf.pdf

   Detailed per-task timing analysis for parallel components of the RDF analysis task.
   Individual times per task were measured for different testing configurations (Table :ref:`tab:configurations`).
   **A** and **D**: Maximum waiting time for the task to be executed by the Dask scheduler.
   **B** and **E**: Maximum total compute time per task.
   **C** and **F**: Maximum total read I/O time per task.
   Points represent the mean over five repeats with the standard deviation shown as error bars.   
   :label:`fig:rdf-wait-comp-io`

The detailed analysis of the individual components in Fig. :ref:`fig:rdf-wait-comp-io` clearly showed that the RDF analysis task required much more computational effort than the RMSD task and that it was dominated by the compute component (Fig. :ref:`fig:timefraction-rdf`), which scaled very well to the highest core numbers (Fig. :ref:`fig:rdf-wait-comp-io` B, E).
However, *multiprocessing* and especially *distributed* with *SSD* took longer for the computational part at :math:`\ge` 8 cores (one third of a single node), indicating that in these cases some sort of competition  reduced performance.
For comparison, serial computation required about 250 s while read I/O required less than 10 s, and this ratio was approximately maintained as the read I/O also scaled reasonably well (Fig. :ref:`fig:rdf-wait-comp-io` C, F)
Although the variance increased markedly when multiple nodes were included such as when using six half-filled nodes, this effect did not strongly impact overall performance because |tcomp| :math:`\gg` |tIO|.
The differences between using all cores on a node compared to only using half the cores on each node were small but only using half a node was consistently better, especially in the compute time, and hence the overall performance of the latter approach was better. 
For the shorter trajectory, the wait time was a factor in reducing performance at higher core numbers (Fig. :ref:`fig:rdf-wait-comp-io` A).
The other components (|tuniverse| :math:`< 2` s, |tprepare| :math:`< 3 \times 10^{-5}` s , |tconclude| :math:`< 4 \times 10^{-4}` s) were similar or better (i.e., shorter) than the ones shown for the RMSD task in Fig. :ref:`fig:rms-pre-con-uni` and are not shown; only the time to set up the :code:`Universe` played a role in reducing the scaling performance in the *Lustre-distributed-3nodes* scenario at 60 or more CPU cores.

.. figure:: figs/percentage_stack_rdf.pdf
	    
   Fraction of the total run time taken by individual steps in the parallel *RDF* calculation for *distributed* on up to three nodes (Lustre-distributed-3nodes).
   Compute (green) and read I/O (red) represent the parallelizable fraction of the program; all other components are effectively serial.
   **A** Trajectory with 900 frames.
   **B** Trajectory with 9000 frames.   
   :label:`fig:timefraction-rdf`
   
In summary, the performance increase for a compute-intensive task such as RDF was sizable and, although not extremely efficient, was large enough (about 30-40) to justify the use of about 100 cores on a HPC supercomputer.
Because scaling seemed mostly limited by constant costs such as the scheduling wait time (see Fig. :ref:`fig:timefraction-rdf`), processing longer trajectories, for which more work has to be done in the parallelizable compute and read I/O steps, should improve the scaling behavior :cite:`Gustafson:1988aa`.



Conclusions
===========

The PMDA_ Python package provides a framework to parallelize analysis of MD trajectories with a simple *split-apply-combine* approach by combining Dask_ with MDAnalysis_.
Although still in early development, it provides useful functionality for users to speed up analysis, ranging from a growing library of included tools to different approaches for users to write their own parallel analysis.
In simple cases, just wrapping a user supplied function is enough to immediately use PMDA but the package also provides a documented API to derive from the :code:`pmda.parallel.ParallelAnalysisBase` class.
We showed that performance depends on the type of analysis that is being performed.
Compute-intensive tasks such as the RDF calculation can show good strong scaling up to about a hundred cores on a typical supercomputer and speeding up the time to solution from hours in serial to minutes in parallel should make this an attractive solution for many users.
For other analysis tasks such as the RMSD calculation and other similar ones (e.g., simple distance calculations), a single multi-core workstation seems sufficient to achieve speed-ups on the order of 10 and HPC resources would not be useful.
But thanks to the design of Dask, running a PMDA analysis on a laptop, workstation, or supercomputer requires absolutely no changes in the code and users are free to immediately choose the computational resource that best fits their purpose.


Code availability and development process
-----------------------------------------

PMDA_ is available in source form under the GNU General Public License v2 from the GitHub repository `MDAnalysis/pmda`_, and as a `PyPi package`_ and `conda package`_  (via the `conda-forge`_ channel).
Python 2.7 and Python :math:`\ge` 3.5 are fully supported and tested.
The package uses `semantic versioning`_ to make it easy for users to judge the impact of upgrading.
The development process uses continuous integration (`Travis CI`_): extensive tests are run on all commits and pull requests via pytest_, resulting in a current code coverage of 97\% and documentation_ is automatically generated by `Sphinx`_ and published as GitHub pages.
Users are supported through the `community mailing list`_ (Google group) and the GitHub `issue tracker`_.


Acknowledgments
===============

We would like to thank reviewer Cyrus Harrison for the idea to plot the fractional time spent on different stages of the program (Figs. :ref:`fig:timefraction-rms` and :ref:`fig:timefraction-rdf`).
This work was supported by the National Science Foundation under grant numbers ACI-1443054 and used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562.
The SDSC Comet computer at the San Diego Supercomputer Center was used under allocation TG-MCB130177. Max Linke was supported by NumFOCUS under a small development grant.



References
==========

.. We use a bibtex file ``pmda.bib`` and use
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations


.. _PMDA: https://www.mdanalysis.org/pmda/
.. _MDAnalysis: https://www.mdanalysis.org
.. _Dask: https://dask.org
.. _`MDAnalysis/pmda`: https://github.com/MDAnalysis/pmda
.. _`PyPi package`: https://pypi.org/project/pmda/
.. _`conda package`: https://anaconda.org/conda-forge/pmda
.. _`semantic versioning`: https://semver.org/
.. _documentation: https://www.mdanalysis.org/pmda/
.. _pytest: https://pytest.org
.. _Sphinx: https://www.sphinx-doc.org/
.. _`Travis CI`: https://travis-ci.com/
.. _`community mailing list`: https://groups.google.com/forum/#!forum/mdnalysis-discussion
.. _`issue tracker`: https://github.com/MDAnalysis/pmda/issues
.. _`10.6084/m9.figshare.8202149`: https://doi.org/10.6084/m9.figshare.8202149
.. _`10.5281/zenodo.3228422`: https://doi.org/10.5281/zenodo.3228422
.. _`conda`: https://docs.conda.io
.. _`conda-forge`: https://anaconda.org/conda-forge/
