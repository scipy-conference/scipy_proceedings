.. -*- mode: rst; mode: visual-line; fill-column: 9999; coding: utf-8 -*-

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
A Treant is a directory in the filesystem that has been specially marked with **state file**.
A ``Treant`` is also a Python object.
We can create a ``Treant`` with:

.. code-block:: python

   >>> import datreant.core as dtr
   >>> t = dtr.Treant('sprout')
   >>> t
   <Treant: 'sprout'>

This creates a directory ``sprout/`` in the filesystem if it didn't already exist, and places a special file inside which stores the Treant's state.
This file also serves as a flagpost indicating that this is more than just a directory::

    > ls sprout
    Treant.1dcbb3b1-c396-4bc6-975d-3ae1e4c2983a.json

The name of this file includes the type of Treant it corresponds to, as well as the ``uuid`` of the Treant, its unique identifier.
This is the state file containing all the information needed to generate an identical instance of this ``Treant``.
We can start a separate Python session and use this Treant immediately there:

.. code-block:: python

   # python session 2
   >>> import datreant.core as dtr
   >>> s = dtr.Treant('sprout')
   >>> s
   <Treant: 'sprout'>

Making a modification to the ``Treant`` in one session is immediately reflected by the same ``Treant`` in any other session.
For example, a ``Treant`` can store any number of descriptive tags that are useful for differentiating it from others:

.. code-block:: python

   # python session 1
   >>> s.tags.add('green', 'cork')
   >>> s.tags
   <Tags(['cork', 'green'])>

And in the other Python session, we see the same ``Treant``:

.. code-block:: python

   # python session 2
   >>> s.tags
   <Tags(['cork', 'green'])>

Internally, advisory locking is done to avoid race conditions, making a ``Treant`` multiprocessing safe.


Introspecting a Treant's Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A ``Treant`` can be used to introspect and manipulate its filesystem tree.
We can, for example, create directory structures rather easily:

.. code-block:: python

   >>> s['a/place/for/data/'].makedirs()
   <Tree: 'sprout/a/place/for/data/'>
   >>> s['a/place/for/text/'].makedirs()
   <Tree: 'sprout/a/place/for/text/'>

and so we now have::

   >>> s.draw()
   sprout/
    +-- Treant.1dcbb3b1-c396-4bc6-975d-3ae1e4c2983a.json
    +-- a/
        +-- place/
            +-- for/
                +-- data/
                +-- text/

Accessing paths in this way returns ``Tree`` and ``Leaf`` objects, referring to directories and files, respectively.
These paths may not point to directories or files that actually exist, but they can be used to create and work with these elements.

We can, for example, easily store a ``pandas`` DataFrame somewhere in the tree for reference later:

.. todo: change to an example where we store a dataframe with arboreal data;
.. more fun, less space, fits theme

.. code-block:: python

   >>> import pandas as pd
   >>> df = pd.DataFrame(pd.np.random.randn(3, 2),
                         columns=['A', 'B'])
   >>> data = s['a/place/for/data/']
   >>> data
   <Tree: 'sprout/a/place/for/data/'>
   >>> df.to_csv(data['random_dataframe.csv'].abspath)
   >>> data.draw()
   data/
   +-- random_dataframe.csv

and we can introspect the file directly:

.. code-block:: python

   >>> csv = data['random_dataframe.csv']
   >>> csv
   <Leaf: 'sprout/a/place/for/data/random_dataframe.csv'>
   >>> print(csv.read())
   ,A,B
   0,-0.573730932177663,-0.08857033924376226
   1,0.03157276797041359,-0.10977921690694506
   2,-0.2080757315892524,0.6825003213837373
    
Using ``Treant``, ``Tree``, and ``Leaf`` objects, we can work with the filesystem Pythonically without giving much attention to *where* these objects live within that filesystem.
This becomes especially powerful when we have many directories/files we want to work with, possibly in many different places.


Aggregation and splitting on Treant metadata
--------------------------------------------
What makes a ``Treant`` distinct from a ``Tree`` is its **state file**.
This file stores metadata that can be used to filter and split ``Treant`` objects when treated in aggregate.
It also serves as a flagpost, making Treant directories discoverable.

If we have many Treants, perhaps scattered about the filesystem:

.. code-block:: python

   >>> for path in ('an/elm/', 'the/oldest/oak'):
   ...     dtr.Treant(path)

we can gather them up with ``datreant.core.discover``:

.. code-block:: python

   >>> b = dtr.discover('.')
   >>> b
   <Bundle([<Treant: 'oak'>, <Treant: 'sprout'>,
            <Treant: 'elm'>])>

A ``Bundle`` is an ordered set of ``Treant`` objects.
This collection gives convenient mechanisms for working with Treants as a single logical unit.
For example, it exposes a few basic properties for directly accessing its member data:

.. code-block:: python

   >>> b.relpaths
   ['the/oldest/oak/', 'sprout/', 'an/elm/']
   >>> b.names
   ['oak', 'sprout', 'elm']

A ``Bundle`` can be constructed in a variety of ways, most commonly using existing ``Treant`` instances or paths to Treants in the filesystem.

We can use a ``Bundle`` to subselect Treants in typical ways, including integer indexing and slicing, fancy indexing, boolean indexing, and indexing by name.
But in addition to these, we can use metadata features such as **tags** and **categories** to filter and group Treants as desired.

Filtering Treants with tags
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tags are individual strings that describe a Treant.
Adding some tags to each of our Treants separately:

.. code-block:: python

   >>> b['sprout'].tags.add('plant')
   >>> b['oak'].tags.add('for building', 'plant', 'building')
   >>> b['elm'].tags.add('firewood', 'shady', 'paper', 'plant', 'building')

we can now work with these tags in aggregate:

.. code-block:: python

   # will only show tags present in *all* members
   >>> b.tags
   <AggTags(['plant'])>

   # will show tags present among *any* member
   >>> b.tags.any
   {'building',
    'cork',
    'firewood',
    'for building',
    'green',
    'paper',
    'plant',
    'shady'}

and we can filter on them. For example, getting all Treants that are good
for construction work:

.. code-block:: python

   # gives a boolean index for members with this tag
   >>> b.tags['building']
   [True, False, True]

   # we can use this to index the Bundle itself
   >>> b[b.tags['building']]
   <Bundle([<Treant: 'oak'>, <Treant: 'elm'>])>

or getting back Treants that are both good for construction *and* used for
making paper by giving tags as a list:

.. code-block:: python

   # a list of tags serves as an *and* query
   >>> b[b.tags[['building', 'paper']]]
   <Bundle([<Treant: 'elm'>])>

And other tag expressions can be constructed using tuples (for *or* operations)
and sets (for *not and*):

.. code-block:: python

   # we can get *or* by using a tuple instead of a list
   >>> b[b.tags['firewood', 'green']]
   <Bundle([<Treant: 'sprout'>, <Treant: 'elm'>])>

   # and we can get *not and* by using a set
   >>> b[b.tags[{'building', 'paper'}]]
   <Bundle([<Treant: 'oak'>, <Treant: 'sprout'>])>

Using tag expressions, we can filter to Treants of interest from a ``Bundle`` counting many, perhaps hundreds, of Treants as members.
A common workflow is to use ``datreant.core.discover`` to gather up many Treants from a section of the filesystem, then use tags to extract only those Treants one actually needs.

Splitting Treants on categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

Making molecular dynamics great again with MDSynthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Acknowledgements
----------------

This work was in part supported by grant ACI-1443054 from the National Science Foundation.



References
----------

.. We use a bibtex file ``mdanalysis.bib`` and use 
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations
