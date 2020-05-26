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

   NumPy, Numba, Pandas, C++, Apache Arrow, Columnar data, AOS-to-SOA, Ragged array, Jagged array, JSON

Introduction
------------

NumPy is a powerful tool for data processing, at the center of a large ecosystem of scientific software. Its built-in functions are general enough for many scientific domains, particularly those that analyze time series, images, or voxel grids. However, it is difficult to apply NumPy to tasks that require data structures beyond N-dimensional arrays of numbers.

More general data structures can be expressed as JSON and processed in pure Python, but at an expense of performance and often conciseness. NumPy is faster and more memory efficient than pure Python because its routines are precompiled and its arrays of numbers are packed contiguously in memory. Some expressions can be more concise in NumPy's "vectorized" notation, which describe actions to perform on whole arrays, rather than scalar values.

In this paper, we will describe Awkward Array, a generalization of NumPy's core functions to the nested records, variable-length lists, missing values, and heterogeneity of JSON-like data. In doing so, we'll focus on the internal representation of data structures as columnar arrays, very similar to (and compatible with) Apache Arrow. The key feature of the Awkward Array library, however, is that calculations, including those that restructure data, are performed on the columnar arrays themselves, rather than instantiating objects. Since the logical data structure is implicit in the arrangement of columnar arrays and most operations only rearrange integer indexes, these functions can be precompiled for specialized data types, like NumPy.

Motivation from particle physics
--------------------------------

Awkward Array was created to make it easier to analyze particle physics data using scientific Python tools. Particle physics datasets are big and intrinsically structured: trillions of particle collisions result in thousands of particles each, grouped by type and clustered as jets. The operations to be performed depend on this structure.

The most basic operation reconstructs particles that decayed before they could be detected, using kinematic constraints on their visible decay products. For example, if a neutral kaon (:math:`K_S`) decays into two charged pions (:math:`\pi^+`, :math:`\pi^-`), the energy (:math:`E`) and momenta (:math:`\vec{p}`) of the observed pions can be combined to reconstruct the unobserved kaon's mass (:math:`m`):

.. math::

   m_{K_S} = \sqrt{(E_{\pi^+} + E_{\pi^-})^2 - \left|\vec{p}_{\pi^+} + \vec{p}_{\pi^-}\right|^2}

Since kaons have a well-defined mass, the :math:`m_{K_S}` values that correspond to real kaons form a peak in a histogram: see Figure :ref:`physics-example`. Not knowing which pions are due to kaon decays, every pair of pions in a collision event must be searched, in principle.

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

Importing this JSON object as an Awkward Array splits its record-oriented structure into a contiguous buffer for each field, making it ready for columnar operations. Heterogeneous data are split by type, such that each buffer in memory has a single numerical type.

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

To compute lengths of each route, we can use NumPy universal functions (like :code:`np.sqrt`) and reducers (like :code:`np.sum`), which are overridden by Awkward-aware functions using NumPy's NEP-13 and NPE-18 protocols. Distances between points can be computed with :code:`a[:, :, 1:] - a[:, :, :-1]` even though each inner list :code:`a[:, :]` has a different length.

.. code-block:: python

    km_east = longitude * 82.7
    km_north = latitude * 111.1

    segment_length = np.sqrt(
        (km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
        (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

    route_length = np.sum(segment_length, axis=-1)
    total_length = np.sum(route_length, axis=-1)

The same could be performed with the following, though the vectorized form is shorter and 8 times faster; see Figure :ref:`bikeroutes-scaling`.

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

   Scaling of Awkward Arrays and pure Python loops for the bike routes calculation shown in the text. :label:`bikeroutes-scaling`

Scope: data types and common operations
---------------------------------------

Awkward Array supports the same suite of abstract data types and features as "typed JSON" serialization formats—Apache Arrow, Parquet, Protobuf, Thrift, Avro, etc.

Namely, there are

* primitive types: numbers and booleans,
* variable-length lists,
* regular-length lists as a separate type (i.e. tensors),
* records/structs/objects (named, typed fields),
* fixed-width tuples (unnamed, typed fields),
* missing/nullable data,
* mixed, yet specified, types (i.e. union/sum types),
* virtual arrays (functions generate arrays on demand),
* partitioned arrays (for off-core and parallel analysis).

Like Apache Arrow and Parquet, arrays with these features are laid out as columns in memory (more on that below).

Like NumPy, the Awkward Array library contains a primary Python class, :code:`ak.Array`, and a collection of generic operations. Most of these operations change the structure of the data in the array, since NumPy, SciPy, and others already provide numerical math as universal functions (ufuncs). In each case where an Awkward function generalizes a NumPy function, it is provided with the same interface (corresponds exactly for rectilinear grids).

Awkward functions include

* basic and advanced slices (:code:`__getitem__`) including variable-length and missing data as advanced slices,
* masking, an alternative to slices that maintains length but introduces missing values instead of dropping elements,
* broadcasting of universal functions into structures,
* reducers of and across variable-length lists,
* zip/unzip/project free arrays into and out of records,
* flattening and padding to make rectilinear data,
* Cartesian products (cross join) and combinations (self join) at :code:`axis >= 1` (per element of one or more arrays).

Conversions to other formats, such as Arrow, access in third-party libraries, such as Numba and Pandas, as well as methods of building data structures and customizing high-level behavior are also in the library's scope.

Columnar representation, columnar implementation
------------------------------------------------

Like Arrow, Awkward data structures are not localized in memory. Instead of concentrating all data for one array element in nearby memory (as an "array of structs"), all data for a given field are contiguous, and all data for another field are elsewhere contiguous (as a "struct of arrays"). This favors a pattern of data access in which only a few fields are needed at a time.

Additionally, Awkward operations are performed on columnar data without returning to the record-oriented format. To illustrate, consider a list of variable-length lists, such as longitude points along a bike route,

.. code-block:: python

    [[1.1, 2.2, 3.3], [4.4], [5.5, 6.6], [7.7, 8.8, 9.9]]

Instead of creating objects to represent the four lists, we flatten the :code:`content` and introduce :code:`starts` and :code:`stops` buffers to indicate where each sublist starts and stops.

.. code-block:: python

    starts:  0, 3, 4, 6
    stops:   3, 4, 6, 9
    content: 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9

A list of lists of lists would use these three buffers as the :code:`content` of another node with its own :code:`starts` and :code:`stops`. In general, a hierarchy of columnar array nodes mirrors the hierarchy of the nested data, except that the number of such nodes scales with the complexity of the data type, not the number of elements in the array. Particle physics use-cases require thousands of nodes to describe complex collision events, but billions of events in memory at once. Figure :ref:`example-hierarchy` shows a small example.

.. figure:: figures/example-hierarchy.pdf
   :align: center
   :scale: 60%
   :figclass: w

   Hierarchy for an example data structure: an array of lists of records, in which field :code:`"x"` of the records are numbers and field :code:`"y"` of the records are lists of numbers. This might, for example, represent :code:`[[], [{"x": 1, "y": [1]}, {"x": 2, "y": [2, 2]}]]`, but it also might represent an array with billions of elements (of the same type). The number of nodes scales with complexity, not data volume. :label:`example-hierarchy`

To compute distances in each bike route, we needed :code:`a[:, 1:] - a[:, :-1]`. To compute :code:`a[:, 1:]`, we only have to add one to the :code:`starts`:

.. code-block:: python

    starts:  1, 4, 5, 7
    stops:   3, 4, 6, 9
    content: 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9

Now this represents

.. code-block:: python

    [[     2.2, 3.3], [   ], [     6.6], [     8.8, 9.9]]

but we didn't have to change the :code:`content` to make it. Since the :code:`content` is untouched, this can be a precompiled routine on :code:`starts` that treats the :code:`content` as an opaque pointer. It could contain other lists or records, or the example in Figure :ref:`example-hierarchy`. Similarly, :code:`a[:, :-1]` is computed by subtracting one from the original :code:`stops`, and the :code:`-` operation has to align the :code:`content` of both arguments before applying the NumPy ufunc.

The first widely used version of Awkward Array was released in September 2018 as a Python module, in which all of the columnar operations were implemented using NumPy. This library was successful, but limited, since some data transformations are difficult or impossible to write without explicit loops. In August 2019, we began a half-year project to rewrite the library in C++, isolating all of the array manipulations in a "CPU kernels" library that can be swapped for "GPU kernels." This project is complete, though transitioning the userbase from "Awkward 0.x" to "Awkward 1.x" is not.

Figure :ref:`awkward-1-0-layers` shows how the new library is organized:

* the high-level interface is in Python,
* the array nodes (managing ownership and lifetimes of :code:`start` and :code:`stop` buffers) in C++ through pybind11,
* an alternate implementation of array navigation exists for use in Numba,
* array manipulation algorithms (without memory management) independently in "CPU kernels" and "GPU kernels" plugins. The kernels' interface is pure C, allowing for reuse in other languages.

.. figure:: figures/awkward-1-0-layers.pdf
   :align: center
   :scale: 45%

   Components of Awkward Array, as described in the text. :label:`awkward-1-0-layers`

Most array operations are shallow, affecting only one or a few levels of the hierarchy, as in our :code:`a[:, 1:]` example above. Thus, the recursive walk over array nodes to perform operations such as these only steps over a few nodes in C++, at most a thousand for particle physics use-cases. Iterations over billions of array elements are restricted to CPU kernels and GPU kernels, so performance optimization can be exclusively focused on this layer.

The C++ layer is primarily motivated by interoperability with C++ codebases, which are common in particle physics.

Numba for just-in-time compilation
----------------------------------

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.


ArrayBuilder: creating columnar data in-place
---------------------------------------------

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.


High-level behaviors
--------------------

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

GPU backend
-----------

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Conclusions
-----------

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

Acknowledgements
----------------

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

References
----------
