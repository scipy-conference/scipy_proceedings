:author: Bradley D. Dice
:email: bdice@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor

:author: Vyas Ramasubramani
:email: vramasub@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Eric S. Harper
:email: harperic@umich.edu
:institution: Department of Materials Science & Engineering, University of Michigan, Ann Arbor

:author: Matthew P. Spellings
:email: mspells@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Joshua A. Anderson
:email: joaander@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Sharon C. Glotzer
:email: sglotzer@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: Department of Materials Science and Engineering, University of Michigan, Ann Arbor
:institution: Biointerfaces Institute, University of Michigan, Ann Arbor

:bibliography: paper

-------------------------------------------------------------------------------------
Analyzing Particle Systems for Machine Learning and Data Visualization with ``freud``
-------------------------------------------------------------------------------------

.. class:: abstract

The ``freud`` Python library analyzes particle data output from molecular dynamics simulations.
The library utilizes Cython and Intel Threading Building Blocks to offer high-performance analysis routines via multithreaded C++ code.
Here, we present practical applications of ``freud`` to analyze nano-scale particle systems, with methods for coupling traditional simulational analyses to machine learning libraries and examples of how to visualize particle quantities computed by ``freud``.
We demonstrate that among Python packages used in the computational molecular sciences, ``freud`` offers a unique set of analysis methods with efficient computations and seamless coupling into powerful data analysis pipelines.

.. class:: keywords

   molecular dynamics, analysis, particle simulation, particle system, computational physics, computational chemistry

Introduction
------------

.. figure:: freud_scales.pdf
   :align: center
   :scale: 80 %
   :figclass: w

   Common Python tools for simulation analysis at varying length scales.
   The freud library is designed for nanoscale systems, such as colloidal crystals and nanoparticle assemblies.
   In such systems, interactions are described by coarse-grained models where particles' atomic constituents are often irrelevant and particle anisotropy (non-spherical shape) is common, thus requiring a generalized concept of particle "types" and orientation-sensitive analyses.
   These features contrast the assumptions of most analysis tools designed for biomolecular simulations and materials science.
   :label:`fig:scales`

The availability of "off-the-shelf" molecular dynamics engines has made simulating complex systems possible across many scientific fields.
Simulations of systems ranging from large biomolecules to colloids are now common, allowing researchers to ask new questions about reconfigurable materials and develop coarse-graining approaches to access increasing timescales.
Various tools have arisen to facilitate the analysis of these simulations, many of which are immediately interoperable with the most popular simulation tools.
The ``freud`` library is one such analysis package that differentiates itself from others through its focus on colloidal and nano-scale systems.

Due to their diversity and adaptability, colloidal materials are a powerful model system for exploring soft matter physics.
Such materials are also a viable platform for harnessing photonic :cite:`Cersonsky2018a`, plasmonic :cite:`Tan2011BuildingDNA`, and other useful structurally-derived properties.
In colloidal systems, features like particle anisotropy play an important role in creating complex crystal structures, some of which have no atomic analogues :cite:`Damasceno2012`.
Design spaces encompassing wide ranges of particle morphology :cite:`Damasceno2012` and interparticle interactions :cite:`Adorf2018` have been studied, yielding phase diagrams filled with complex behavior.

The ``freud`` Python package targets these systems by avoiding trajectory management and the analysis of chemically bonded structures, the province of most other analysis platforms like MDAnalysis and MDTraj :cite:`Michaud-Agrawal2011,McGibbon2015`, and instead provides a unique feature set that is tailored to capturing the important properties that characterize colloidal systems.
In particular, ``freud`` excels at performing analyses based on characterizing local particle environments, which makes it a powerful tool for tasks such as calculating order parameters to track crystallization or finding prenucleation clusters.
Among the unique methods present in ``freud`` are the potential of mean force and torque, which allows users to understand the effects of particle anisotropy on entropic self-assembly :cite:`VanAnders2014c,VanAnders2014d,Karas2016,Harper2015,Anderson2017`, and various tools for identifying and clustering particles by their local crystal environments :cite:`Teich2019`.
All such tasks are accelerated by ``freud``'s extremely fast neighbor finding routines and are automatically parallelized, making it an ideal tool for researchers performing peta- or exascale simulations of particle systems.
The ``freud`` library's scalability is exemplified by its use in computing correlation functions on systems of over a million particles, calculations that were used to elucidate the elusive hexatic phase transition in two-dimensional systems of hard polygons :cite:`Anderson2017`.
More details on the use of ``freud`` can be found in **CITE ARXIV PAPER**.

The ``freud`` package is especially useful because it can be organically integrated into a data pipeline.
Many research tasks in computational molecular sciences can be expressed in terms of data pipelines; in molecular simulations, such a pipeline typically involves:

1. **Generating** an input file that defines a simulation.
2. **Simulating** the system of interest, saving its trajectory to a file.
3. **Analyzing** the resulting data by computing and storing various quantities.
4. **Visualizing** the trajectory, using colors or styles determined from previous analyses.

However, in modern workflows the lines between these stages is typically blurred, particularly with respect to analysis.
While direct visualization of simulation trajectories can provide insights into the behavior of a system, integrating higher-order analyses is often necessary to provide real-time interpretable visualizations in that allow researchers to identify meaningful features like defects and ordered domains of self-assembled structures.
Studies of complex systems are also often aided or accelerated by a real-time coupling of simulations with on-the-fly analysis.
This simultaneous usage of simulation and analysis is especially relevant because modern machine learning techniques frequently involve wrapping this pipeline entirely into a higher-level optimization problem, since the descriptors it computes can be used to, for instance, construct objective functions targeting a specific materials design problem.

The ``freud`` package is uniquely well-suited to such applications because its interfaces are designed entirely around direct usage of numerical arrays of data.
This access pattern contrasts with most existing existing analysis libraries, like MDAnalysis :cite:`Michaud-Agrawal2011`, which are tightly coupled to the reading of files output by simulation engines and the system representations embedded in these files.
By decoupling itself from file parsing and specific trajectory representations, ``freud`` can be efficiently integrated into simulations, machine learning applications, and visualization toolkits with no I/O overhead and limited additional code complexity.

In this paper, we will show how this UNIX-like philosophy allows ``freud`` to be easily integrated with various tools for data visualization and machine learning.
We will provide demonstrations of how ``freud`` can be integrated with using popular tools in the scientific Python ecosystem like TensorFlow, ``scikit-learn``, ``scipy``, or ``matplotlib``.
We will discuss how the analyses in ``freud`` can reduce the 6N-dimensional space of particle positions and orientations into a tractable set of features that can be fed into machine learning algorithms.
We will further show that ``freud`` can be used for visualizations even outside of scripting contexts, enabling a wide range of forward-thinking applications including Jupyter notebook integrations, versatile 3D renderings, and integration with various standard tools for visualizing simulation trajectories.
These topics are aimed at computational molecular scientists and data scientists alike, with discussions of real-world usage as well as theoretical motivation and conceptual exploration.
The full source code of all examples in this paper can be found online [#]_.

.. [#] https://github.com/glotzerlab/freud-examples

Integrating ``freud`` with the Scientific Python Ecosystem
----------------------------------------------------------

NumPy arrays are used for all inputs and outputs in ``freud`` :cite:`Oliphant2006a`.
Because of the wide range of trajectory formats used by different simulation engines, ``freud`` does not provide a tool for parsing data directly from trajectory output files.
A number of libraries (such as MDAnalysis and mdtraj, as well as format-specific tools like ``gsd`` [#]_ for the HOOMD-blue simulation engine) can parse trajectory files and provide their data as NumPy arrays for analysis with ``freud`` :cite:`Michaud-Agrawal2011,McGibbon2015`.

.. [#] https://github.com/glotzerlab/gsd

In addition to ``freud``'s simple NumPy inputs and outputs, the library integrates other important tools from the Scientific Python ecosystem.
The ``scipy`` package is one such example, where ``freud`` wraps ``scipy``'s behavior to compute Voronoi diagrams in periodic systems.
Enforcing periodicity with triclinic boxes where the sides are tilted (and thus not orthogonal to one another) can be tricky, necessitating ``freud``'s implementation for determining Voronoi tesselations in both 2D and 3D periodic systems.

Similarly, the mean-squared displacement module (``freud.msd``) utilizes fast Fourier transforms from ``numpy`` or ``scipy`` to accelerate its computations.
The resulting MSD data help to identify how particles' dynamics change over time, e.g. from ballistic to diffusive.

Machine Learning
----------------

A common challenge in molecular sciences is identifying crystal structures.
Recently, several approaches have been developed that use machine learning for detecting ordered phases :cite:`Schoenholz2015,Spellings2018,Fulford2019,Steinhardt1983,Lechner2008`.
The Steinhardt order parameters are often used as a structural fingerprint, and are derived from rotationally invariant combinations of spherical harmonics.
In the example below, we create face-centered cubic (fcc), body-centered cubic (bcc), and simple cubic (sc) crystals with added Gaussian noise, and use Steinhardt order parameters with a support vector machine to train a simple crystal structure identifier.
Steinhardt order parameters characterize the spherical arrangement of neighbors around a central particle, and combining values of
:math:`Q_l` for a range of :math:`l` often gives a unique signature for simple crystal structures.
This example demonstrates a simple case of how ``freud`` can be used to help solve the problem of structural identification, which often requires a sophisticated approach for complex crystals.

.. figure:: noisy_structures_q6.pdf
   :align: center
   :scale: 100 %

   Histogram of the Steinhardt :math:`Q_6` order parameter for 4000 particles in simple cubic, body-centered cubic, and face-centered cubic structures with added Gaussian noise.
   :label:`fig:noisystructuresq6`

In figure :ref:`fig:noisystructuresq6`, we show the distribution of :math:`Q_6` values for sample structures with 4000 particles.
Below, we demonstrate how to compute the Steinhardt :math:`Q_6`, using neighbors found via a periodic Voronoi diagram.
Neighbors with small facets in the Voronoi polytope are filtered out to reduce noise.

.. code-block:: python

   import freud
   import numpy as np

   def get_features(box, positions, structure):
       voro = freud.voronoi.Voronoi(
           box, buff=max(box.L)/2)
       voro.computeNeighbors(positions)
       nlist = voro.nlist
       nlist.filter(nlist.weights > 0.1)
       features = {}
       for l in [4, 6, 8, 10, 12]:
           ql = freud.order.LocalQl(
               box, rmax=max(box.L)/2, l=l)
           ql.compute(positions, nlist)
           features['q{}'.format(l)] = ql.Ql.copy()

       return features

   structures = {}
   structures['fcc'] = get_features(
       fcc_box, fcc_positions, 'fcc')
   # ... repeat for all structures

Then, using ``pandas`` and ``scikit-learn``, we can train a support vector machine to identify these structures:

.. code-block:: python

   structure_dfs = {}
   for i, struct in enumerate(structures):
       df = pd.DataFrame.from_dict(structures[struct])
       df['class'] = i
       structure_dfs[struct] = df

   df = pd.concat(structure_dfs.values())
   df = df.reset_index(drop=True)

   from sklearn.preprocessing import normalize
   from sklearn.model_selection import train_test_split
   from sklearn.svm import SVC

   X = df.drop('class', axis=1).values
   X = normalize(X)
   y = df['class'].values
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.33, random_state=42)

   svm = SVC()
   svm.fit(X_train, y_train)
   print('Score:', svm.score(X_test, y_test))
   # The model is ~98% accurate.

To interpret crystal identification models like this, it can be helpful to use a dimensionality reduction tool such as UMAP (Uniform Manifold Approximation and Projection), as shown in figure :ref:`fig:steinhardtumap` :cite:`McInnes2018`.
The low-dimensional UMAP projection shown is generated directly from our ``pandas`` ``DataFrame``:

.. code-block:: python

    from umap import UMAP
    umap = UMAP()
    data = umap.fit_transform(df)
    plt.plot(data[:, 0], data[:, 1])

.. figure:: steinhardt_umap.pdf
   :align: center
   :scale: 80 %

   UMAP of particle descriptors computed for simple cubic, body-centered cubic, and face-centered cubic structures of 4000 particles with added Gaussian noise.
   The particle descriptors include :math:`Q_l` for :math:`l \in \{4, 6, 8, 10, 12\}`.
   Some noisy configurations of bcc can be confused as fcc and vice versa, which accounts for the small number of errors in the support vector machine's test classification.
   :label:`fig:steinhardtumap`

Extending Crystal Descriptors
=============================

Computing a different set of descriptors tuned for a particular system of interest (e.g. by using more values of :math:`Q_l`, the higher-order Steinhardt :math:`W_l` parameters, or other order parameters provided by ``freud``) is possible with just a few more lines of code.
For example, this paper :cite:`Harper2019` used ``freud`` to compute the cubatic order parameter and generate high-dimensional descriptors of structural motifs, which were visualized with t-SNE dimensionality reduction :cite:`vanDerMaaten2008`.
The open-source ``pythia`` [#]_ library offers a number of descriptor sets, all of which leverage ``freud`` for their fast computations.
These descriptors have been used with TensorFlow for supervised and unsupervised learning of crystal structures in complex phase diagrams :cite:`Spellings2018,TensorFlow2015`.

.. [#] https://github.com/glotzerlab/pythia

Another useful module for machine learning with ``freud`` is ``freud.cluster``, which uses a distance-based cutoff to locate clusters of particles while accounting for 2D or 3D periodicity.
Locating clusters in this way can identify crystalline grains, helpful for building a training set for machine learning models.

Visualization
-------------

Many analyses performed by the ``freud`` library can be readily plotted.
Some analyses like the radial distribution function or correlation functions return data that is binned as a one-dimensional histogram -- these are usually best visualized with a line graph via ``matplotlib.pyplot.plot``, with the bin locations and bin counts given by properties of the compute object.
Other classes provide multi-dimensional histograms, like the Gaussian density or Potential of Mean Force and Torque, which can be plotted with ``matplotlib.pyplot.imshow``.

The most complex case for visualization is that of per-particle properties, which also comprises some of the most useful features in ``freud``.
Quantities that are computed on a per-particle level can be continuous (e.g. Steinhardt order parameters) or discrete (e.g. clustering, where the integer value corresponds to a unique cluster ID).
Continuous quantities can be plotted as a histogram, but typically the most helpful visualizations use these quantities with a color map assigned to particles in a two- or three-dimensional view of the system itself.
For such particle visualizations, several open-source tools exist that interoperate well with ``freud``.
Below are examples of how one can integrate ``freud`` with ``plato`` [#]_, ``fresnel`` [#]_, and OVITO :cite:`Stukowski2010`.

.. [#] https://github.com/glotzerlab/plato
.. [#] https://github.com/glotzerlab/fresnel

plato
=====

.. figure:: plato_pythreejs.png
   :align: center
   :scale: 20 %

   Interactive visualization of a Lennard-Jones particle system, rendered in a Jupyter notebook using ``plato`` with the ``pythreejs`` backend.
   :label:`fig:platopythreejs`

``plato`` is an open-source graphics package that expresses a common interface for defining two- or three-dimensional scenes which can be rendered as an interactive Jupyter widget or saved to a high-resolution image using one of several backends (``pythreejs``, ``matplotlib``, ``fresnel``, POVray [#]_, and Blender [#]_, among others).
Below is an example of how to render particles from a HOOMD-blue snapshot, colored by the density of their local environment :cite:`Anderson2008,Glaser2015`.
The result is shown in figure :ref:`fig:platopythreejs`.

.. [#] https://www.povray.org/
.. [#] https://www.blender.org/

.. code-block:: python

   import plato
   import plato.draw.pythreejs as draw
   import numpy as np
   import matplotlib.cm
   import freud
   from sklearn.preprocessing import minmax_scale

   # snap comes from a previous HOOMD-blue simulation
   positions = snap.particles.position
   ld = freud.density.LocalDensity(
       r_cut=3.0, volume=1.0, diameter=1.0)
   box = freud.box.Box.from_box(snap.box)
   ld.compute(box, positions)
   radii = 0.5 * np.ones(len(positions))
   colors = matplotlib.cm.viridis(
       minmax_scale(ld.density))
   spheres_primitive = draw.Spheres(
       positions=positions,
       radii=radii,
       colors=colors)
   scene = draw.Scene(spheres_primitive, zoom=2)
   scene.show()  # Interactive view in Jupyter

fresnel
=======

``fresnel`` [#]_ is a GPU-accelerated ray tracer designed for particle simulations, with customizable material types and scene lighting, as well as support for a set of common anisotropic shapes.
Its feature set is especially well suited for publication-quality graphics.
Its use of ray tracing also means that an image's rendering time scales with the image size, instead of the number of particles -- a desirable feature for extremely large simulations.
An example of ``fresnel`` integration is available online.

.. [#] https://github.com/glotzerlab/fresnel

OVITO
=====

.. figure:: ovito_selection.png
   :align: center
   :scale: 20 %

   A crystalline grain identified using ``freud``'s ``LocalDensity`` module and cut out for display using OVITO.
   :label:`fig:ovitoselection`


OVITO is a GUI application with features for particle selection, making movies, and support for many trajectory formats :cite:`Stukowski2010`.
OVITO has several built-in analysis functions (e.g. Polyhedral Template Matching), which complement the methods in ``freud``.
The Python scripting functionality built into OVITO enables the use of  ``freud`` modules, demonstrated in the code below and shown in figure :ref:`fig:ovitoselection`.

.. code-block:: python

   import freud

   def modify(frame, input, output):

       if input.particles != None:
           box = freud.box.Box.from_matrix(
               input.cell.matrix)
           ld = freud.density.LocalDensity(
               r_cut=3, volume=1, diameter=0.05)
           ld.compute(box, input.particles.position)
           output.create_user_particle_property(
               name='LocalDensity',
               data_type=float,
               data=ld.density.copy())

Benchmarking ``freud``
----------------------

.. figure:: comparison_rcut_1.pdf
   :align: center
   :scale: 60 %

   Comparison of runtime for neighbor finding algorithms in ``freud`` and ``scipy`` for varied system sizes. See text for details.
   :label:`fig:scipycomparison`

In figure :ref:`fig:scipycomparison`, a comparison is shown between the neighbor finding algorithms in ``freud`` and ``scipy`` :cite:`Jones2001`.
For each system size, :math:`N` particles are uniformly distributed in a 3D periodic cube of side length :math:`L = 10`.
Neighbors are found for each particle by searching within a cutoff distance :math:`r_{cut} = 1`.
The methods compared are ``scipy.spatial.cKDTree``'s ``query_ball_tree``, ``freud.locality.AABBQuery``'s ``queryBall``, and ``freud.locality.LinkCell``'s ``compute``.
The benchmarks were performed on a 3.6 GHz Intel Core i3 processor with 16 GB 2667 MHz DDR4 RAM.
The parallel C++ backend implemented with Cython and Intel Threading Building Blocks makes ``freud`` perform quickly for large periodic systems :cite:`Behnel2011,Intel2018`.
Furthermore, ``freud`` supports the triclinic boxes found in many simulations (which can be sheared, as opposed to ``scipy`` which supports only cubic boxes).

Conclusions
-----------

The ``freud`` library offers a unique set of high-performance algorithms designed to accelerate the study of nanoscale and colloidal systems.
We have demonstrated several ways in which these tools for particle analysis can be used in conjunction with other packages for machine learning and data visualization.
We hope these examples are of use to the computational molecular science community and spark new ideas for analysis and scientific exploration.

Getting ``freud``
-----------------

The ``freud`` library is tested for Python 2.7 and 3.5+ and is compatible with Linux, macOS, and Windows.
To install ``freud``, execute

.. code-block:: bash

    conda install -c conda-forge freud

or

.. code-block:: bash

    pip install freud-analysis

Its source code is available on GitHub [#]_ and its documentation is available via ReadTheDocs [#]_.

.. [#] https://github.com/glotzerlab/freud
.. [#] https://freud.readthedocs.io/

Acknowledgments
---------------

Thanks to Jin Soo Ihm for benchmarking the neighbor finding features of ``freud`` against ``scipy``.
Support for the design and development of ``freud`` has evolved over time and programmatic research directions.
Conceptualization and early implementations were supported in part by the DOD/ASD(R&E) under Award No. N00244-09-1-0062 and also by the National Science Foundation, Integrative Graduate Education and Research Traineeship, Award # DGE 0903629 (to E.S.H. and M.P.S.).
A majority of the code development including all public code releases was supported by the National Science Foundation, Division of Materials Research under a Computational and Data-Enabled Science & Engineering Award # DMR 1409620.
M.P.S. also acknowledges support from the University of Michigan Rackham Predoctoral Fellowship program.
B.D. is supported by a National Science Foundation Graduate Research Fellowship Grant DGE 1256260.
Computational resources and services supported in part by Advanced Research Computing at the University of Michigan, Ann Arbor.
