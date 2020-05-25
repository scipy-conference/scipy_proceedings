:author: Jim Pivarski
:email: pivarski@princeton.edu
:institution: Princeton University

:author: Ianna Osborne
:email: Ianna.Osborne@cern.ch
:institution: Princeton University

:author: Pratyush Das
:email: reikdas@gmail.com
:institution: Institute of Engineering and Management

:author: Anish Biswas
:email: anishbiswas271@gmail.com
:institution: Manipal Institute of Technology

:author: Peter Elmer
:email: Peter.Elmer@cern.ch
:institution: Princeton University

------------------------------------------------
Awkward Array: JSON-like data, NumPy-like idioms
------------------------------------------------

.. class:: abstract

    NumPy simplifies and accelerates mathematical calculations in Python, but only for rectilinear arrays of numbers. Awkward Array provides a similar interface for JSON-like data: slicing, masking, broadcasting, and performing vectorized math on the attributes of objects, unequal-length nested lists (i.e. ragged/jagged arrays), and heterogeneous data types.

    Awkward Arrays are columnar data structures, like (and convertible to/from) Apache Arrow, with a focus on manipulation, rather than serialization/transport. These arrays can be passed between C++ and Python, and they can be used in functions that are JIT-compiled by Numba.

    Development of a GPU backend is in progress, which would allow data analyses written in array-programming style to run on GPUs without modification.

.. class:: keywords

   NumPy, Numba, C++, Apache Arrow, Columnar data, AOS-to-SOA, Ragged array, Jagged array, JSON

Introduction
------------

NumPy is a powerful tool for data processing, at the center of a large ecosystem of scientific software. Its built-in functions are general enough for many scientific domains, particularly those that analyze time series, images, or voxel grids. However, it is difficult to apply NumPy to tasks that require data structures beyond N-dimensional arrays of numbers.

More general data structures can be expressed as JSON and processed in pure Python, but at an expense of performance and often conciseness. NumPy is faster and more memory efficient than pure Python because its routines are precompiled and its arrays of numbers are packed contiguously in memory. Some calculations can be expressed more succinctly in NumPy's "vectorized" notation, which describe actions to perform on whole arrays, rather than scalar values.

In this paper, we will describe Awkward Array, a generalization of NumPy's core functions to the nested records, variable-length lists, missing values, and heterogeneity of JSON-like data. In doing so, we'll focus on the internal representation of data structures as columnar arrays, very similar to (and compatible with) Apache Arrow. The key feature of the Awkward Array library, however, is that calculations, including those that restructure data, are performed on the columnar arrays themselves, rather than instantiating objects. Since the logical data structure is implicit in the arrangement of columnar arrays and most operations only rearrange integer indexes, these functions can be precompiled for specialized data types, like NumPy.

Motivation from particle physics
--------------------------------

Awkward Array was created to make it easier to analyze particle physics data using scientific Python tools. Particle physics datasets are big and intrinsically structured: trillions of particle collisions result in thousands of particles each, grouped by type and clustered as jets. The operations to be performed depend on this structure.

The most basic operation reconstructs particles that decayed before they could be detected, using kinematic constraints on their visible decay products. For example, if a neutral kaon (:math:`K_S`) decays into two charged pions (:math:`\pi^+`, :math:`\pi^-`), the energy (:math:`E`) and momenta (:math:`\vec{p}`) of the observed pions can be combined to reconstruct the unobserved kaon's mass (:math:`m`):

.. math::

   m_{K_S} = \sqrt{(E_{\pi^+} + E_{\pi^-})^2 - \left|\vec{p}_{\pi^+} + \vec{p}_{\pi^-}\right|^2}

Since kaons have a well-defined mass, the :math:`m_{K_S}` values that corresponding to real kaons form a peak in a histogram: see Figure :ref:`physics-example`. Not knowing where to look, every pair of pions in a collision event must be searched.

.. figure:: figures/physics-example.pdf
   :align: center
   :scale: 13%
   
   Example of a particle physics problem requiring combinatorial search: all pairs of pions in a collision event must be tested for compatibility with decay from a kaon. :label:`physics-example`

Physicists employ many other techniques, but most of them involve combinatorial searches like this one. Since the numbers of particles of each type differ from event to event and they carry dozens of attributes that must be referenced in the combinatorial pairs, this would be a hard analysis to do in NumPy.

With Awkward Array, however, it would be

.. code-block:: python

   kaon_masses = ak.combinations(pions[good], 2).mass

where :code:`ak.combinations` is a built-in function, :code:`pions[good]` preselects good pion objects in all events using an array of variable-length lists of booleans, and :code:`.mass` is a user-defined property that computes :math:`m_{K_S}` using NumPy universal functions.

Demonstration with GeoJSON bike routes
--------------------------------------

However, nested data structures are not unique to particle physics, so we present a more complete example using GeoJSON map data. Suppose we want to analyze the following Chicago bike routes, a dataset with two nested levels of latitude, longitude polylines, string-valued street names, and metadata as a JSON file.

.. code-block:: python

    import urllib.request
    import json

    url = "https://raw.githubusercontent.com/Chicago/" \
          "osd-bike-routes/master/data/Bikeroutes.geojson"
    bikeroutes_json = urllib.request.urlopen(url).read()
    bikeroutes_pyobj = json.loads(bikeroutes_json)

Importing this JSON object into Awkward Array splits its record-oriented structure into a contiguous buffer for each field, making it ready for columnar operations. Heterogeneous data are split by type, such that each buffer in memory has one numerical type.

.. code-block:: python

    import awkward1 as ak
    bikeroutes = ak.Record(bikeroutes_pyobj)

Longitude and latitude are in the first two components of fields named :code:`"coordinates"` of fields named :code:`"geometry"` of fields named :code:`"features"`. They can be accessed with NumPy-like slices, including ellipsis, :code:`np.newaxis`, masks, etc.

.. code-block:: python
    
    longitude = bikeroutes["features", "geometry",
                           "coordinates", ..., 0]
    latitude  = bikeroutes["features", "geometry",
                           "coordinates", ..., 1]

The :code:`longitude` and :code:`latitude` arrays both have type :code:`1061 * var * var * float64` (expressed as a Datashape): 1061 routes with a variable number of variable-length polylines.

To compute distances, we can use NumPy universal functions (such as :code:`np.sqrt`) and reducers (such as :code:`np.sum`), which are overridden by Awkward-aware functions using NumPy's NEP-13 and NPE-18 protocols. Distances between points can be computed with :code:`a[:, :, 1:] - a[:, :, :-1]` even though each inner list :code:`a[:, :]` has a different length.

.. code-block:: python

    km_east = longitude * 82.7
    km_north = latitude * 111.1

    segment_length = np.sqrt(
        (km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
        (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

    route_length = np.sum(segment_length, axis=-1)
    total_length = np.sum(route_length, axis=-1)

The same could be performed with the following pure Python, though the vectorized form is more succinct and 8 times faster; see Figure :ref:`bikeroutes-scaling`.

.. code-block:: python

    total_length = []
    for route in bikeroutes_pyobj["features"]:
        route_length = []
        for polyline in route["geometry"]["coordinates"]:
            segment_length = []
            last = None
            for lng, lat in polyline:
                km_east = lng * 82.7
                km_north = lat * 111.1
                if last is not None:
                    dx2 = (km_east - last[0])**2
                    dy2 = (km_north - last[1])**2
                    segment_length.append(
                        np.sqrt(dx2 + dy2))
                last = (km_east, km_north)

            route_length.append(sum(segment_length))
        total_length.append(sum(route_length))

.. figure:: figures/bikeroutes-scaling.pdf
   :align: center
   :scale: 45%

   This is the caption. :label:`bikeroutes-scaling`

Data types and common operations
--------------------------------

types: lists, records, missing data, heterogeneous, virtual, partitioned

operations: slice/mask with variable-width/missing data extensions, broadcasting, universal functions, reducers, num, zip/unzip, flatten, pad_none/fill_none, cartesian, combinations

Columnar representation, columnar implementation
------------------------------------------------

asdf (talk about Arrow in this section, maybe also Pandas)

Numba for just-in-time compilation
----------------------------------

asdf

ArrayBuilder: creating columnar data in-place
---------------------------------------------

asdf

High-level behaviors
--------------------

asdf

GPU backend
-----------

asdf

Conclusions
-----------

asdf

Acknowledgements
----------------

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

References
----------
