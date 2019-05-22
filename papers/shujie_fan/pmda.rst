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

:author: Michael Gecht
:email: michael.gecht@biophys.mpg.de
:institution: Max Planck Institute of Biophysics

:author: Oliver Beckstein
:email: obeckste@asu.edu 
:institution: Arizona State University 
:corresponding:

:bibliography: pmda


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
   With the development of highly optimized molecular dynamics software (MD) packages on HPC resources, the size of simulation trajectories is growing to many  terabytes in size.
   Thus efficient analysis of MD simulations becomes a challenge for MDAnalysis, which does not yet provide a standard interface for parallel analysis.
   To address the challenge, we developed PMDA_, a Python library that provides parallel analysis algorithms based on MDAnalysis.
   PMDA parallelizes common analysis algorithms in MDAnalysis through a task-based approach with the Dask_ library.
   We implement a simple split-apply-combine scheme for parallel trajectory analysis.
   The trajectory is split into blocks and analysis is performed separately and in parallel on each block ("apply").
   The results from each block are gathered and combined.
   PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks.
   In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules.
   PMDA supports all schedulers in Dask, and one can run analysis in a distributed fashion on HPC or ad-hoc clusters or on a single machine.
   We tested the performance of PMDA on single node and multiple nodes on local supercomputing resources and workstations.
   The results show that parallelization improves the performance of trajectory analysis.
   Although still in alpha stage, it is already used on resources ranging from multi-core laptops to XSEDE supercomputers to speed up analysis of molecular dynamics trajectories.
   PMDA is available under the GNU General Public License, version 2.

.. class:: keywords

   Molecular Dynamics Simulations, High Performance Computing, Python, Dask, MDAnalysis





Introduction
============

Classical molecular dynamics (MD) simulations have become an invaluable tool to understand the function of biomolecules :cite:`Karplus:2002ly, Dror:2012cr, Seyler:2014il, Orozco:2014dq, Bottaro:2018aa, Huggins:2019aa` (often with a view towards drug discovery :cite:`Borhani:2012mi`) and diverse problems in materials science :cite:`Rottler:2009aa, Li:2015aa, Varela:2015aa, Lau:2018aa, Kupgan:2018aa, Frederix:2018aa`.
Systems are modelled as particles (for example, atoms) whose interactions are approximated with a classical potential energy function :cite:`FrenkelSmit02, Braun:2018ab`.
Forces on the particles are derived from the potential and Newton's equations of motion for the particles are solved with an integrator algorithm, typically using highly optimized MD codes that run on high performance computing (HPC) resources or workstations (often equipped with GPU accelerators).
The resulting trajectories, the time series of particle positions :math:`\mathbf{r}(t)` (and possibly velocities), are analyzed with statistical mechanics approaches :cite:`Tuckerman:2010cr, Braun:2018ab` to obtain predictions or to compare to experimentally measured quantities.
Currently simulated systems may contain millions of atoms and the trajectories can consist of hundreds of thousands to millions of individual time frames, thus resulting in file sizes ranging from tens of gigabytes to tens of terabytes.
Processing and analyzing these trajectories is increasingly becoming a rate limiting step in computational workflows :cite:`Cheatham:2015qf, Beckstein:2018aa`.
Modern MD packages are highly optimized to perform well on current HPC clusters with hundreds of cores such as the XSEDE supercomputers :cite:`XSEDE` but current general purpose trajectory analysis packages :cite:`Giorgino:2019aa` were not designed with HPC in mind.

In order to scale up trajectory analysis from workstations to HPC clusters with the MDAnalysis_ Python library :cite:`Michaud-Agrawal:2011fu,Gowers:2016aa` we leveraged Dask_ :cite:`Rocklin:2015aa, Dask:2016aa`, a task-graph parallel framework, together with Dask's distributed scheduler, and created the *Parallel MDAnalysis* (PMDA_) library.
By default, PMDA follows a simple split-apply-combine :cite:`Wickham:2011aa` approach for trajectory analysis, whereby each task analyzes a single trajectory segment and reports back the individual results that are then combined into the final result :cite:`Khoshlessan:2017ab`.
Our previous work established that Dask worked well with MDAnalysis :cite:`Khoshlessan:2017ab` and that this approach was competitive with other task-parallel approaches :cite:`Paraskevakos:2018aa`.
However, we did not provide a general purpose framework to write parallel analysis tools with MDAnalysis.
Here we show how the split-apply-combine approach lends itself to a generalizable Python implementation that makes it straightforward for users to implement their own parallel analysis tools.
At the heart of PMDA is the idea that the user only needs to provide a function that analyzes a single trajectory frame.
PMDA provides the remaining framework via the :code:`ParallelAnalysisBase` class to split the trajectory, apply the user's function to trajectory frames, run the analysis in parallel via Dask/distributed, and and combines the data.
It also contains a growing library of ready-to-use analysis classes, thus enabling users to immediately accelerate analysis that they previously performed in serial with the standard MDAnalysis analysis classes :cite:`Gowers:2016aa`.





Methods
=======

At the core of PMDA is the idea that a common interface makes it easy to create code that can be easily parallelized, especially if the analysis can be split into independent work over multiple trajectory slices and a final step in which all data from the trajectory slices is combined.
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

   g(r) = \left\langle \frac{1}{N_a N_b} \sum_{i=1}^{N_a} \sum_{j=1}^{N_b} \delta(|\mathbf{r}_i - \mathbf{r}_j| - r) \right\rangle

where the Dirac delta function counts the occurences of particles :math:`i` and :math:`j` at distance :math:`r`.
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
	    
   Schema of the split-apply-combine approach in PMDA.
   Steps are labeled with the methods in :code:`pmda.parallel.ParallelAnalysisBase` that perform the corresponding function.
   Methods in red (:code:`_single_frame()` and :code:`_conclude()`) must be implemented for every analysis function because they are not general.
   The blue method :code:`_reduce()` must be implemented unless a simple time series is being calculated.
   The :code:`_prepare()` method is optional an provides a hook to initialize custom data structures.
   :label:`fig:schema`

As explained in more detail later, a class derived from :code:`pmda.parallel.ParallelAnalysisBase` encapsulates one trajectory analysis calculation.
Individual methods correspond to different steps and in the following (and in Fig. :ref:`fig:schema`) we will mention the names of the relevant methods to make clear how PMDA abstracts parallel analysis.
The calculation with :math:`M` parallel workers is *prepared* by setting up data structures to hold the final result (method :code:`_prepare()`).
The indices for the :math:`M` trajectory slices are created in such a way that the number of frames :math:`\tau_k` are balanced and do not differ by more than 1.
For each slice or block :math:`k`, the *single frame* analysis function :math:`\mathcal{A}` (:code:`_single_frame()`) is sequentially applied to all frames in the slice.
The result, :math:`A(t)`, is *reduced*, i.e., added to the results for this block.
For time series, :math:`A(t)` is simply appended to a list to form a partial time series for the block.
More complicated reductions (method :code:`_reduce()`) can be implemented, for  example, the date may be histogrammed and added to a partial histogram for the block (as necessary for the implementation of the paralle RDF Eq. :ref:`eq:rdf`).




Implementation
--------------

PMDA is written in Python and, through MDAnalysis :cite:`Gowers:2016aa`, reads trajectory data from the file system into NumPy arrays :cite:`Oliphant:2007aa, Van-Der-Walt:2011aa`.
Dask's :code:`delayed()` function is used to build a task graph that is then executed using any of the schedulers available to Dask :cite:`Dask:2016aa`.
We tested MDAnalysis 0.20.0 (development version), Dask 1.1.1, NumPy 1.15.4.

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

Calling the :code:`compute()` method of the delayed list object hands the task graph over to the scheduler, which then executes the graph on the available dask workers.
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


Performance evaluation
----------------------

To evaluate the performance of the parallelization, two common computational tasks were tested that differ in their computational cost and represent two different requirements for data reduction.
We computed the time series of root mean square distance after optimum superposition (RMSD) of all |Calpha| atoms of a protein with the initial coordinates at the first frame as reference, as implemented in class :code:`pmda.rms.RMSD`.
The RMSD calculation with optimum superposition was performed with the fast QCPROT algorithm :cite:`Theobald:2005vn` as implemented in MDAnalysis :cite:`Michaud-Agrawal:2011fu`.
As a second test case we computed the water-water radial distribution function (RDF, Eq. :ref:`eq:rdf`) for all water molecules in our test system, using the class :code:`pmda.rdf.InterRDF`.
The RDF calculation is compute-intensive due to the necessity to calculate and histogram a large number (:math:`\mathcal{O}(N^2)`) of distances for each time step; it additionally exemplifies a non-trivial reduction.

The test data files consist of a topology file ``YiiP_system.pdb`` (with :math:`N = 111815` atoms) and two trajectory files ``YiiP_system_9ns_center.xtc`` (Gromacs XTC format, :math:`T = 900` frames) and ``YiiP_system_90ns_center.xtc`` (Gromacs XTC format, :math:`T = 9000` frames) of the membrane protein YiiP in a lipid bilayer together with water and ions.

The :code:`ParallelAnalysisBase` class collects detailed timing information for all blocks and all frames and makes these data available in the attribute :code:`ParallelAnalysisBase.timing`:
We measured the time |tprepare| for :code:`_prepare()`, the time |twait| that each task :math:`k` waits until it is executed by the scheduler, the time |tuniverse| to create a new :code:`Universe` for each Dask task (which includes opening the shared trajectory and topology files and loading the topology into memory), the time |tIO| to read each frame :math:`t` in each block :math:`k` from disk into memory, the time |tcomp| to perform the computation in :code:`_single_frame()` and reduction in :code:`_reduce()`, the time |tconclude| to perform the final processing of all data in :code:`_conclude()`, and the total wall time to solution |ttotal|.

We quantified the strong scaling behavior by calculating the speed-up for running on :math:`M` CPU cores with :math:`M` parallel Dask tasks as :math:`S(M) = t^\text{total}(M)/t^\text{total}(1)`, where :math:`t^\text{total}(1)` is the performance of the PMDA code using the serial scheduler.
The efficiency was calculated as :math:`E(M) = S(M)/M`.

	    

Using PMDA
==========

PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks. In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules. Here, we will introduce some basic usages of PMDA.

Pre-defined Analysis
--------------------
PMDA contains a number of pre-defined analysis classes that are modelled after functionality in ``MDAnalysis.analysis`` and that can be used right away. PMDA currently has four predefined analysis tasks to use:

``pmda.rms``: RMSD analysis tools

``pmda.comtacts``: Native contacts analysis tools

``pmda.rdf``: Radial distribution function tools

``pmda.leaflet``: LeafletFinder analysis tool

While the first 3 classes are developed based on ``pmda.parallel.ParallelAnalysisBase`` which separates the trajectory into work blocks containing multiple frames, ``pmda.leaflet`` partitions the system based on a 2-dimensional partitioning. 
The usage of these tools is similar to ``MDAnalysis.analysis``. One example is calculating root mean square distance(RMSD) of |Calpha| atoms of the protein with ``pmda.rms``.

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


User-defined Analysis
---------------------

With pmda.custom.AnalysisFromFunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PMDA provides helper functions in ``pmda.custom`` to rapidly build a parallel class for users who already have a function:
1. takes one or more AtomGroup instances as input,
2. analyzes one frame in a trajectory and returns the result for this frame.
For example, we already have a function to calculate the radius of gyration of a protein given in ``AtomGroup`` ``ag``:

.. code-block:: python

    import MDAnalsys as mda
    u = mda.Universe(top, traj)
    protein = u.select_atoms('protein')

    def rgyr(ag):
        return(ag.radius_of_gyration)

We can wrap rgyr() in ``pmda.custom.AnalysisFromFunction`` to build a paralleled version of ``rgyr()``:

.. code-block:: python
     
    import pmda.custom
    parallel_rgyr = pmda.custom.AnalysisFromFucntion(
                    rgyr, u, protein)

Run the analysis on 4 cores and show the timeseries of the results stored in ``parallel_rgyr.results``:

.. code-block:: python

    parallel_rgyr.run(n_jobs=4, n_blocks=4)
    print(parallel_rgyr.results)

With pmda.parallel.ParallelAnalysisBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In more common cases, one can write the parallel class with the help of ``pmda.parallel.ParallelAnalysisBase``. To build a new analysis class, one should 

1. (Required) Define the single frame analysis function ``_single_frame``,

2. (Required) Define the final results conclusion function ``_conclue``,

3. (Not Required) Define the additional preparation function ``_prepare``,

4. (Not Required) Define the accumulation function for frames within the same block ``_reduce``, if the result is not time-series data,

5. Derive a class from ``pmda.parallel.ParallelAnalysisBase`` that uses these functions. 

As an example, we show how one can build a class to calculate the radius of gyration of a protein givin in ``AtomGroup`` ``protein``. The class needs to be initialized with ``pmda.parallel.ParallelAnalysisBase`` subclassed. The conclusion function reshapes the ``self._results`` which stores the results from all blocks.  


.. code-block:: python

    import numpy as np
    from pmda.parallel import ParallelAnalysisBase

    class RGYR(ParallelAnalysisBase):
        def __init__(self, protein):
            universe = protein.universe
            super(RMSD, self).__init__(universe, (protein, ))

        def _prepare(self):
            self.rgyr = None
        def _conclude(self):
            self.rgyr = np.vstack(self._results)

The inputs for ``_single_frame`` are fixed. ``ts`` contains the current time step and ``agroups`` is a tuple of atomgroups that are updated to the current frame. The current frame number, time and radius of gyration are returned as the single frame results. Here we don't need to define a new ``_reduce``.

.. code-block:: python

        def _single_frame(self, ts, atomgroups):
            protein = atomgroups[0]
            
            return (ts.frame, ts.time,
                    protein.radius_of_gyration))

The usage of this class is the same as the function we defined with ``pmda.custom.AnalysisFromFunction``.  

.. code-block:: python

    import MDAnalsys as mda
    u = mda.Universe(top, traj)
    protein = u.select_atoms('protein')
    
    parallel_rgyr = RGYR(protein)
    parallel_rgyr.run(n_jobs=4, n_blocks=4)
    print(parallel_rgyr.results)



Results and Discussion
======================
	    

Conclusions
===========



Code availability and development process
-----------------------------------------

PMDA is available in source form under the GNU General Public License v2 from the GitHub repository `MDAnalysis/pmda`_, and as a `PyPi package`_ and `conda package`_  (via the conda-forge channel).
Python 2.7 and Python :math:`\ge` 3.5 are fully supported and tested.
The package uses `semantic versioning`_ to make it easy for users to judge the impact of upgrading.
The development process uses continuous integration (`Travis CI`_): extensive tests are run on all commits and pull requests via pytest_, resulting in a current code coverage of 97\% and documentation_ is automatically generated by `Sphinx`_ and published as GitHub pages.
Users are supported through the `community mailinglist`_ (Google group) and the GitHub `issue tracker`_.


Acknowledgments
===============

This work was supported by the National Science Foundation under grant numbers ACI-1443054 and used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562.
The SDSC Comet computer at the San Diego Supercomputer Center was used under allocation TG-MCB130177.



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
.. _`community mailinglist`: https://groups.google.com/forum/#!forum/mdnalysis-discussion
.. _`issue tracker`: https://github.com/MDAnalysis/pmda/issues
