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

   data management, science, filesystems

.. For example file, see ../00_vanderwalt/00_vanderwalt.rst
.. Shows how to do figures, maths, raw latex, tables, citations


Introduction
------------
.. must motivate datreant, and make a good sell as to why it is a useful and general-purpose tool

In many fields of science, especially those analyzing experimental or simulation data, there is an existing ecosystem of specialized tools and file formats which new tools must work around.
Often this makes the filesystem serve as a *de facto* database, with directory trees the zeroth-order data structure for scientific data.
This is particularly the case for fields centered around simulation: simulation systems can vary widely in size, composition, rules, parameters, and starting conditions.
And with increases in computational power, it is often necessary to store intermediate results from large amounts of simulation data so it can be accessed and explored interactively.

These problems make data management difficult, and serve as a barrier to answering scientific questions.
To make things easier, ``datreant`` is a collection of packages that provide a Pythonic interface to the filesystem and the data that lives within it.
It solves a boring problem, so we can focus on interesting ones.


Treants as filesystem manipulators
----------------------------------
The central object of ``datreant`` is the ``Treant``.
A ``Treant`` is a directory in the filesystem that has been specially marked with **state file**.
A ``Treant`` is also a Python object.
We can create a ``Treant`` with:

.. code-block:: python

   >>> import datreant.core as dtr
   >>> t = dtr.Treant('sprout')
   >>> t
   <Treant: 'sprout'>

This creates a directory ``sprout/`` in the filesystem if it didn't already exist, and places a special file inside which both stores the ``Treant``'s state.
This file also serves as a flagpost indicating that this is not just a directory, but a ``Treant``::

    > ls sprout
    Treant.1dcbb3b1-c396-4bc6-975d-3ae1e4c2983a.json

The name of this file includes the type of ``Treant`` it corresponds to, as well as the ``uuid`` of the ``Treant``, which is its unique identifier.
This is the state file containing all the information needed to regenerate an identical instance of this ``Treant``.
We can start a separate Python session and use this ``Treant`` immediately there:

.. code-block:: python

   # python session 2
   >>> import datreant.core as dtr
   >>> t = dtr.Treant('sprout')
   >>> t
   <Treant: 'sprout'>

Making a modification to the ``Treant`` in one session is immediately reflected by the same ``Treant`` in any other session.
For example, a ``Treant`` can store any number of descriptive tags that are useful for differentiating it from others:

.. code-block:: python

   # python session 1
   >>> t.tags.add('green', 'cork')
   >>> t.tags
   <Tags(['cork', 'green'])>

And in the other Python session, we see the same ``Treant``:

.. code-block:: python

   # python session 2
   >>> t.tags
   <Tags(['cork', 'green'])>

Internally, advisory locking is done to avoid race conditions, making a ``Treant`` multiprocessing safe.

Introspecting a Treant's Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

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


