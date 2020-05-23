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

Motivation from particle physics
--------------------------------

asdf

Tentatively labeling each as :math:`\pi^+` and :math:`\pi^-`

.. math::

   \sqrt{(E_{\pi^+} + E_{\pi^-})^2 - \left|\vec{p}_{\pi^+} + \vec{p}_{\pi^-}\right|^2}

.. code-block:: python

   kaon_masses = ak.combinations(pions[good], 2).mass

   plt.hist(ak.flatten(kaon_masses))

.. figure:: figures/physics-example.pdf
   :align: center
   :scale: 13%

   This is the caption. :label:`physics-example`


Demonstration with GeoJSON bike routes
--------------------------------------

asdf

.. code-block:: python

    import urllib.request
    import json

    url = "https://raw.githubusercontent.com/Chicago/" \
          "osd-bike-routes/master/data/Bikeroutes.geojson"
    bikeroutes_json = urllib.request.urlopen(url).read()
    bikeroutes_pyobj = json.loads(bikeroutes_json)

.. code-block:: python

    import awkward1 as ak
    bikeroutes = ak.Record(bikeroutes_pyobj)
    
    geometry = bikeroutes["features", "geometry"]
    km_east = geometry.coordinates[..., 0] * 82.7
    km_north = geometry.coordinates[..., 1] * 111.1

    segment_length = np.sqrt(
        (km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
        (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

    route_length = np.sum(segment_length, axis=-1)
    total_length = np.sum(route_length, axis=-1)

The ``km_east`` and ``km_north`` have type ``1061 * var * var * float64``, for 1061 bike routes, variable number of polylines, variable number of numerical positions in each.

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
