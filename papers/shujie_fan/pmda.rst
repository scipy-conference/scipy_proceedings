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
.. |tN| replace:: :math:`t_N`
.. |tcomp| replace:: :math:`t_\text{comp}`
.. |tIO| replace:: :math:`t_\text{I/O}`
.. |tcomptIO| replace:: :math:`t_\text{comp}+t_\text{I/O}`
.. |avg_tcomp| replace:: :math:`\langle t_\text{compute} \rangle`
.. |avg_tIO| replace:: :math:`\langle t_\text{I/O} \rangle`
.. |Ncores| replace:: :math:`N`

---------------------------------------------
 PMDA - Parallel Molecular Dynamics Analysis
---------------------------------------------

.. class:: abstract

   MDAnalysis_ is an object-oriented Python library to analyze trajectories from molecular dynamics (MD) simulations in many popular formats.
   With the development of highly optimized molecular dynamics software (MD) packages on HPC resources, the size of simulation trajectories is growing to terabyte size.
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

   MDAnalysis, High Performance Computing, Dask



============
Introduction
============

Molecular dynamics (MD) simulations have become an invaluable tool to understand the function of biomolecules :cite:`Karplus:2002ly, Dror:2010fk, Seyler:2014il, Orozco:2014dq, Bottaro:2018aa, Huggins:2019aa` (often with a view towards drug discovery :cite:`Borhani:2012mi`) and diverse problems in materials science :cite:`Rottler:2009aa, Li:2015aa, Varela:2015aa, Lau:2018aa, Kupgan:2018aa, Frederix:2018aa`.




MDAnalysis  :cite:`Michaud-Agrawal:2011fu,Gowers:2016aa`

Dask :cite:`Rocklin:2015aa`

Other packages :cite:`Giorgino:2019aa`

split-apply-combine :cite:`Wickham:2011aa` for trajectory analysis :cite:`Khoshlessan:2017ab,Paraskevakos:2018aa`




=======
Methods
=======

``pmda.parallel.ParallelAnalysisBase`` is the base class for defining a split-apply-combine parallel multi frame analysis in PMDA. This class will automatically take care of setting up the trajectory reader for iterating in parallel. The class is based on the following libraries: MDAnalysis 0.20.0, Dask 1.1.1, NumPy 1.15.4.

.. code-block:: python

    import MDAnalysis as mda
    from dask.delayed import delayed
    import dask
    import dask.distributed
    import numpy as np

The parallel analysis algorithms are performed on ``Universe`` and tuple of ``AtomGroups``. The topology, trajectory filenames and the list of AtomGroup indices are passed as attributes to make them accessiable to each block. 

.. code-block:: python

    class ParallelAnalysisBase(object):
	def __init__(self, universe, atomgroups):
	    self._trajectory = universe.trajectory 
	    self._top = universe.filename
	    self._traj = universe.trajectory.filename
	    self._indices = [ag.indices 
                             for ag in atomgroups]

``run()`` performs the split-apply-combine parallel analysis. The trajectory is split into n_blocks blocks by :code:`make_balanced_slices` with first frame start, final frame stop and step length step (corresponding to the split step).  :code:`make_balanced_slices` is a function defined in pmda.util. It generates blocks in such a way that they contain equal numbers of frames when possible, but there are also no empty blocks. The final start and stop frames for each block are restored in a list slices. ``n_jobs`` is the number of jobs to start, this argument will be ignored when the distributed scheduler used. After the additional preparation defined in :code:`_prepare`, the analysis jobs (the apply step, defined in :code:`_dask_helper()`)  on each block are delayed with the :code:`delayed()` function in dask. The results from all blocks are moved and reshaped into a sensible new variable ``self.results`` (may have other name) with the :code:`_conclude()` function.

.. code-block:: python

        def run(self, start=None, stop=None, step=None,
            n_jobs=1, n_blocks=None):
            n_frames = len(range(start, stop, step))
            slices = make_balanced_slices(n_frames, 
                                  n_blocks, start=start,
                                  stop=stop, step=step)
            self._prepare()
                blocks = []
                for bslice in slices:
                    task = delayed(self._dask_helper, 
                             pure=False)(bslice,
                                 self._indices,
                                 self._top,
                                 self._traj, )
                    blocks.append(task)
                    blocks = delayed(blocks)
                    res = blocks.compute(**scheduler_kwargs)
                    self._results = np.asarray(
                                      [el[0] for el in res])
                    self._conclude()
            return self

:code:`_dask_helper()` is the single block analysis function. It first reconstructs the Universe and the tuple of AtomGroups. Then the single-frame analysis :code:`_single_frame()` is performed on each trajectory frame by iterating  over ``u.trajectory[bslice.start:bslice.stop]``. 

.. code-block:: python

        def _dask_helper(self, bslice, indices, top, traj):
            u = mda.Universe(top, traj)
            agroups = [u.atoms[idx] for idx in indices]
            res = []
            for i in range(bslice.start, 
                           bslice.stop, bslice.step):
                ts = u.trajectory[i]
                res = self._reduce(res, 
                      self._single_frame(ts, agroups))
            return np.asarray(res)

Accumulation of frames within a block happens in the :code:`_reduce` function. It is called for every frame. ``res`` contains all the results before current time step, and ``result_single_frame`` is the result of ``_single_frame`` for the current time step. The return value is the updated ``res``. The default is to append results to a python list. This approach is sufficient for time-series data, such as the root mean square distance(RMSD) of the |Calpha| atoms of a protein. 

.. code-block:: python

        @staticmethod
        def _reduce(res, result_single_frame):
            # 'append' action for a time series
            res.append(result_single_frame)
            return res


===========
Basic Usage 
===========

PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks. In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules. Here, we will introduce some basic usages of PMDA.

Pre-defined Analysis
====================
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
=====================

With pmda.custom.AnalysisFromFunction
+++++++++++++++++++++++++++++++++++++
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
+++++++++++++++++++++++++++++++++++++++

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


=========
Benchmark
=========





Method
======

``timeit`` is a context manager defined in pmda.util (to be used with the ``with`` statement) that records the execution time for the enclosed context block ``elapsed``. Here, we record the time for `prepare`, `compute`, `I/O`, `conclude`, `universe`, `wait` and `total`. These timing results are finally stored in the attributes of the class ``pmda.parallel.Timing``. 


Results and Discussion
======================


Conclusions
===========




Acknowledgments
===============

SF and IP were supported by grant ACI-1443054 from the National Science Foundation.
OB was supported in part by grant ACI-1443054 from the National Science Foundation.
Computational resources were in provided the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1053575 (allocation MCB130177 to OB.


References
==========

.. We use a bibtex file ``pmda.bib`` and use
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations


.. _PMDA: https://www.mdanalysis.org/pmda/
.. _MDAnalysis: https://www.mdanalysis.org
.. _Dask: https://dask.org
