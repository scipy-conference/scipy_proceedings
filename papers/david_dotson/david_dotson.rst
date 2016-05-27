.. -*- mode: rst; fill-column: 9999; coding: utf-8 -*-

:author: David L. Dotson
:email: dldotson@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA
:equal-contributor:
:corresponding:

:author: Sean L. Seyler
:email: slseyler@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA

:author: Max Linke
:email: max.linke@biophys.mpg.de
:institution: Max Planck Institut für Biophysik, Frankfurt, Germany

:author: Richard J. Gowers
:email: richardjgowers@gmail.com
:institution: University of Manchester, Manchester, UK
:institution: University of Edinburgh, Edinburgh, UK

:author: Oliver Beckstein
:email: oliver.beckstein@asu.edu
:institution: Arizona State University, Tempe, Arizona, USA

:bibliography: ``datreant``

-----------------------------------------------------------
datreant: persistent, Pythonic trees for heterogeneous data
-----------------------------------------------------------

.. class:: abstract

In science the filesystem often serves as a *de facto* database, with directory trees being the zeroth-order scientific data structure.
But it can be tedious and error prone to work directly with the filesystem to retrieve and store heterogeneous datasets.
**datreant** makes working with directory structures and files Pythonic with **Treants**: specially marked directories with distinguishing characteristics that can be discovered, queried, and filtered.
Treants can be manipulated individually and in aggregate, with mechanisms for granular access to the directories and files in their trees.
Disparate datasets stored in any format (CSV, HDF5, NetCDF, Feater, etc.) scattered throughout a filesystem can thus be manipulated as meta-datasets of Treants.
**datreant** is modular and extensible by design to allow specialized applications to be built on top of it, with **MDSynthesis** as an example for working with molecular dynamics simulation data. http://datreant.org/


.. class:: keywords

   data management, science 

.. For example file, see ../00_vanderwalt/00_vanderwalt.rst
.. Shows how to do figures, maths, raw latex, tables, citations


Introduction
------------
.. must motivate datreant, and make a good sell as to why it is a useful and general-purpose tool






.. copied from talk submission; probably can be less hurried in its language with a bit more detail on context
In many fields of science, especially those analyzing experimental or simulation data, there is an existing ecosystem of specialized tools and file formats which new tools must work around.
Often this makes the filesystem serve as a *de facto* database, with directory trees the zeroth-order data structure for scientific data.
But it can be tedious and error prone to work with these directory trees to retrieve and store heterogeneous datasets, especially over projects spanning years with no strict organizational scheme.

To address this pain point, we present `**datreant** <http://datreant.org/>`_.
At the core of datreant are **Treants**: specially marked directories with distinguishing characteristics that can be discovered, queried, and filtered.
Treants map the filesystem as it is into a Pythonic interface, making heterogeneous data easier to leverage while enhancing scientific reproducibility.




Treants as filesystem manipulators
----------------------------------


.. note somehow that it is not necessary to use Treants to manipulate the filesystem, but they serve as flagposts for places of interest

Aggregation and splitting on Treant metadata
--------------------------------------------


Treant modularity with attachable Limbs
---------------------------------------


Using Treants as the basis for dataset access and manipulation with the PyData stack
------------------------------------------------------------------------------------
.. should emphasize that we don't need specific limbs per se to work with different datasets or to use other libraries
.. since Treants are filesystem manipulators, can use them as the access points for things like blaze, dask, distributed, etc.

.. would love to give Fireworks a shout-out here, since building workflows that operate on Treants works *really* well

Building domain-specific applications on datreant
-------------------------------------------------
.. not only can applications *use* Treants, they can define their own Treant subclasses that work in special ways





References
----------

.. We use a bibtex file ``mdanalysis.bib`` and use 
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations


