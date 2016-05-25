.. -*- mode: rst; fill-column: 9999; coding: utf-8 -*-

:author: Richard J. Gowers
:email: richardjgowers@gmail.com
:institution: University of Manchester, Manchester, UK
:institution: University of Edinburgh, Edinburgh, UK
:equal-contributor:

:author: Max Linke
:email: max.linke@biophys.mpg.de
:institution: Max Planck Institut für Biophysik, Frankfurt, Germany
:equal-contributor:

:author: Jonathan Barnoud
:email: j.barnoud@rug.nl
:institution: University of Groningen, Groningen, The Netherlands
:equal-contributor:

:author: Tyler J. E. Reddy
:email: tyler.reddy@bioch.ox.ac.uk
:institution: University of Oxford, Oxford, UK

:author: Manuel N. Melo
:email: m.n.melo@rug.nl
:institution: University of Groningen, Groningen, The Netherlands

:author: Sean L. Seyler
:email: slseyler@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA

:author: Jan Domanski
:email: jan.domanski@bioch.ox.ac.uk
:institution: University of Oxford, Oxford, UK

:author: David L. Dotson
:email: dldotson@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA

:author: Sébastien Buchoux
:email: sebastien.buchoux@u-picardie.fr
:institution: Université de Picardie Jules Verne, Amiens, France

:author: Ian M. Kenney
:email: Ian.Kenney@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA


:author: Oliver Beckstein
:email: oliver.beckstein@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA
:corresponding:

:bibliography: ``mdanalysis``


-------------------------------------------------------------------------------------
MDAnalysis: A Python Package for the Rapid Analysis of Molecular Dynamics Simulations
-------------------------------------------------------------------------------------

.. class:: abstract

MDAnalysis (http://mdanalysis.org) is an library for structural and temporal analysis of molecular dynamics (MD) simulation trajectories and individual protein structures.
MD simulations of biological molecules have become an important tool to elucidate the relationship between molecular structure and physiological function.
Simulations are performed with highly optimized software packages on HPC resources but most codes generate output trajectories in their own formats so that the development of new trajectory analysis algorithms is confined to specific user communities and widespread adoption and further development is delayed.
The MDAnalysis library addresses this problem by abstracting access to the raw simulation data and presenting a uniform object-oriented Python interface to the user.
It thus enables users to rapidly write code that is portable and immediately usable in virtually all biomolecular simulation communities.
The user interface and modular design work equally well in complex scripted workflows, as foundations for other packages, and for interactive and rapid prototyping work in IPython_ :cite:`Perez2007` / Jupyter_ notebooks, especially together with molecular visualization provided by nglview_ and time series analysis with pandas_ :cite:`McKinney2010`.
MDAnalysis is written in Python and Cython and uses NumPy_ :cite:`Vanderwalt2011` arrays for easy interoperability with the wider scientific Python ecosystem.
It is widely used and forms the foundation for more specialized biomolecular simulation tools.
MDAnalysis is available under the GNU General Public License v2.

.. _IPython: http://ipython.org/
.. _Jupyter: http://jupyter.org/
.. _nglview: https://github.com/arose/nglview
.. _pandas: http://pandas.pydata.org/
.. _NumPy: http://www.numpy.org

.. class:: keywords

   molecular dynamics simulations, science, chemistry, physics, biology


.. For example file, see ../00_vanderwalt/00_vanderwalt.rst
.. Shows how to do figures, maths, raw latex, tables, citations


Introduction
------------

.. initial copy and paste


Molecular dynamics (MD) simulations of biological molecules have become an important tool to elucidate the relationship between molecular structure and physiological function.
Simulations are performed with highly optimized software packages on HPC resources but most codes generate output trajectories in their own formats so that the development of new trajectory analysis algorithms is confined to specific user communities and widespread adoption and further development is delayed.
Typical trajectory sizes range from gigabytes to terabytes so it is typically not feasible to convert trajectories into a range of different formats just to use a tool that requires this specific form.
Instead, a framework is required that provides a common interface to raw simulation data.

Results
-------

The MDAnalysis library :cite:`Michaud-Agrawal:2011fu` addresses this problem by abstracting access to the raw simulation data and presenting a uniform object-oriented Python interface to the user. MDAnalysis is written in Python and Cython and uses NumPy arrays for easy interoperability with the wider scientific Python ecosystem.
It currently supports more than 25 different file formats and covers the vast majority of data formats that are used in the biomolecular simulation community, including the formats required and produced by the most popular packages NAMD, Amber, Gromacs, CHARMM, LAMMPS, DL_POLY, HOOMD.
The user interface provides "physics-based" abstractions (e.g. "atoms", "bonds", "molecules") of the data that can be easily manipulated by the user.
It hides the complexity of accessing data and frees the user from having to implement the details of different trajectory and topology file formats (which by themselves are often only poorly documented and just adhere to certain "community expectations" that can be difficult to understand for outsiders).

The user interface and modular design work equally well in complex scripted workflows, as foundations for other packages like ENCORE_ :cite:`Tiberti:2015fk` and ProtoMD_ `Somogyi:2016aa`, and for interactive and rapid prototyping work in IPython/Jupyter notebooks, especially together with molecular visualization provided by nglview_ and time series analysis with pandas_.
Since the original publication :cite:`Michaud-Agrawal:2011fu`, improvements in speed and data structures make it now possible to work with terabyte-sized trajectories containing up to ~10 million particles.
MDAnalysis also comes with specialized analysis classes in the MDAnalysis.analysis module that are unique to MDAnalysis such as the LeafletFinder graph-based algorithm for the analysis of lipid bilayers :cite:`Michaud-Agrawal:2011fu` or the Path Similarity Analysis for the quantitative comparison of macromolecular conformational changes :cite:`Seyler:2015fk`.

MDAnalysis is available in source form under the GNU General Public License v2 from GitHub https://github.com/MDAnalysis/mdanalysis, PyPi_ and as conda_ packages.
The documentation is extensive http://docs.mdanalysis.org including an introductory tutorial http://www.mdanalysis.org/MDAnalysisTutorial/.
Our develoment community is very active with over 5 active core developers and lots of community contributions every release.
We use modern software development practices with continous integration and an extensive testsuite, >3500 tests and >92% for our core modules.
If you like to use MDAnalysis for your project please join our community_ board.

.. _PyPi: https://pypi.python.org/pypi/MDAnalysis
.. _conda: https://anaconda.org/mdanalysis/dashboard
.. _community: https://groups.google.com/forum/#!forum/mdnalysis-discussion
.. _ENCORE: https://github.com/encore-similarity/encore
.. _ProtoMD: https://github.com/CTCNano/proto_md

Analysis Module
---------------

In the MDAnalysis.analysis module we provide a large variety of standard analysis algorithms, like RMSD, alignment, native contacts, as well as unique algorithms, like the LeaftleftFinder and Path Similarty Analysis.
We have recently started to unify the interface to the different algorithms with an `AnalysisBase` class.
Currently PersistenceLength, InterRDF, LinearDensity and Contacts analysis have been ported.
If applicable we also strive to make the API's to the algorithms generic.
Most other tools hand the user analysis algorithms as black boxes, we want to avoid that and give the users all he needs to adapt an analysis to his/her needs.

The new Contacts class is a good example a generic API that allows easy adaptations of algorithms while still offering an easy setup for standard analysis types.
For contact analysis it is possible to have different metrics based on coordinates alone :cite:`Best2013,Franklin2007`.
We have designed the API to choose between common metrics and pass user defined functions to develop new metrics.
This generic interface allowed us to implement a q1q2 analysis ontop of the Contacts class.
Below is incomplete code example that shows how to implement a q1q2 analysis, a more detailed explanatain can be found in the docs. **how to generate a link/should I generate a link**

.. code-block:: python

   def radius_cut_q(r, r0, radius):
       y = r <= radius
       return y.sum() / r.size

   contacts = Contacts(u, selection,
                       (first_frame_refs, last_frame_refs),
                       radius=radius, method=radius_cut_q,
                       start=start, stop=stop, step=step,
                       kwargs={'radius': radius})

This type of flexible analysis algorithms paired with a collection of base classes allow quick and easy analysis of simulations as well as development of new algorithms.


Conclusions
-----------

MDAnalysis provides a uniform interface to simulation data, which comes in a bewildering array of formats.
It enables users to rapidly write code that is portable and immediately usable in virtually all biomolecular simulation communities.
It has a very active international developer community with researchers that are expert developers and users of a wide range of simulation codes.
MDAnalysis is widely used (the original paper :cite:`Michaud-Agrawal:2011fu` has been cited more than 180 times) and forms the foundation for more specialized biomolecular simulation tools.
Ongoing and future developments will improve performance further, introduce transparent parallelisation schemes to utilize multi-core systems efficiently, and interface with the `SPIDAL library`_ for high performance data analytics algorithms.



References
----------
.. We use a bibtex file ``mdanalysis.bib`` and use
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations

.. _`SPIDAL library`: http://spidal.org
