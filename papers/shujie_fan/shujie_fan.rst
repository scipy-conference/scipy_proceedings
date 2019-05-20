:author: Shujie Fan
:email: sfan19@asu.edu
:institution: Arizona State University

:author: Max Linke
:email: max.linke88@gmail.com
:institution: Max Planck Institute of Biophysics

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
:bibliography: mybib



------------------------------------------------
PMDA - Parallel Molecular Dynamics Analysis
------------------------------------------------

.. class:: abstract

PMDA is a Python library that provides parallel analysis algorithms based on MDAnalysis. With a simple split-apply-combine scheme from Dask library, PMDA provides pre-defined analysis tasks and a common interface to create user-defined analysis taks. Although still in alpha stage, it is already used on resources ranging from multi-core laptops to XSEDE supercomputers to speed up analysis of molecular dynamics trajectories.

.. class:: keywords

   PMDA, MDAnalysis, High Performance Computing, Dask, Map-Reduce

Introduction
------------




Method
--------------------

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

``run()`` performs the split-apply-combine parallel analysis. The trajectory is split into n_blocks blocks by :code:`make_balanced_slices` with first frame start, final frame stop and step length step (corresponding to the split step).  :code:`make_balanced_slices` is a function defined in pmda.util. It generates blocks in such a way that they contain equal numbers of frames when possible, but there are also no empty blocks. The final start and stop frames for each block are restored in a list slices. ``n_jobs`` is the number of jobs to start, this argument will be ignored when the distributed scheduler used. After the additional preparation defined in :code:`_prepare`, the analysis jobs (the apply step, defined in :code:`_dask_helper()`)  on each block are delayed with the :code:`delayed()` function in dask. Finally, the results from all blocks are gathered and combined in the :code:`_conclude()` function.

``timeit`` is a context manager defined in pmda.util (to be used with the ``with`` statement) that records the execution time for the enclosed context block ``elapsed``. Here, we record the time for `prepare`, `compute`, `I/O`, `conclude`, `universe`, `wait` and `total`. These timing results are finally stored in the attributes of the class ``pmda.parallel.Timing``. 

.. code-block:: python

        def run(self, start=None, stop=None, step=None,
            n_jobs=1, n_blocks=None):
	
            # Get the indices of the start, stop 
            # and step frames.
            start, stop, step = 
                   self._trajectory.check_slice_indices(
                       start, stop, step)
            n_frames = len(range(start, stop, step))
            slices = make_balanced_slices(n_frames, 
                                  n_blocks, start=start,
                                  stop=stop, step=step)
            with timeit() as total:
                with timeit() as prepare:
                    self._prepare()
                time_prepare = prepare.elapsed
                blocks = []
                with self.readonly_attributes():
                    for bslice in slices:
                        task = delayed(
                             self._dask_helper, 
                             pure=False)(
                                 bslice,
                                 self._indices,
                                 self._top,
                                 self._traj, )
                        blocks.append(task)
                    blocks = delayed(blocks)
                    # record the time when scheduler
                    # starts working
                    wait_start = time.time()
                    res = blocks.compute(**scheduler_kwargs)
                with timeit() as conclude:
                    self._results = np.asarray(
                                      [el[0] for el in res])
                    self._conclude()
            self.timing = Timing(
                np.hstack([el[1] for el in res]),
                np.hstack([el[2] for el in res]), 
                total.elapsed,
                np.array([el[3] for el in res]), 
                time_prepare,
                conclude.elapsed,
                # waiting time = wait_end - wait_start
                np.array([el[4]-wait_start for el in res]))
            return self

:code:`_dask_helper()` is the single block analysis function. It first reconstructs the Universe and the tuple of AtomGroups. Then the single-frame analysis :code:`_single_frame()` is performed on each trajectory frame by iterating  over ``u.trajectory[bslice.start:bslice.stop]``. 

.. code-block:: python

        def _dask_helper(self, bslice, indices, top, traj):
            # wait_end needs to be first line 
            # for accurate timing
            wait_end = time.time()
            with timeit() as b_universe:
                u = mda.Universe(top, traj)
                agroups = [u.atoms[idx] for idx in indices]
            res = []
            times_io = []
            times_compute = []
            for i in range(bslice.start, 
                           bslice.stop, bslice.step):
                with timeit() as b_io:
                ts = u.trajectory[i]
                with timeit() as b_compute:
                    res = self._reduce(res, 
                       self._single_frame(ts, agroups))
                times_io.append(b_io.elapsed)
                times_compute.append(b_compute.elapsed)
            return np.asarray(res), np.asarray(times_io),
                np.asarray(times_compute), 
                b_universe.elapsed, wait_end

Accumulation of frames within a block happens in the :code:`_reduce` function. It is called for every frame. ``res`` contains all the results before current time step, and ``result_single_frame`` is the result of ``_single_frame`` for the current time step. The return value is the updated ``res``. The default is to append results to a python list. This approach is sufficient for time-series data, such as the root mean square distance(RMSD) of the :math:`C_{\alpha}` atoms of a protein. 

.. code-block:: python

        @staticmethod
        def _reduce(res, result_single_frame):
            # 'append' action for a time series
            res.append(result_single_frame)
            return res



Basic Usage 
--------------

PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks. In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules. Here, we will introduce some basic usages of PMDA.

Pre-defined Analysis
++++++++++++++++++++++
PMDA contains a number of pre-defined analysis classes that are modelled after functionality in ``MDAnalysis.analysis`` and that can be used right away. PMDA currently has four predefined analysis tasks to use:

``pmda.rms``: RMSD analysis tools

``pmda.comtacts``: Native contacts analysis tools

``pmda.rdf``: Radial distribution function tools

``pmda.leaflet``: LeafletFinder analysis tool

The usage of these tools is similar to ``MDAnalysis.analysis``. The simplest example is calculating root mean square distance(RMSD) of :math:`C_{\alpha}` atoms of the protein with ``pmda.rms``.

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
++++++++++++++++++++++



References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


